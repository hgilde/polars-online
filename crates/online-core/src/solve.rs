//! Shared dense solves: Cholesky via `faer` with a jittered-diagonal fallback
//! (docs/PLAN.md §7). Never NaN silently; callers count `solve_failures`.

use std::cmp::Ordering;
use std::sync::OnceLock;

use faer::Conj;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;

/// The diagonal jitters tried in turn, as multiples of `trace/k`: `A` as
/// given first, then escalating.
const JITTER: [f64; 5] = [0.0, 1e-12, 1e-9, 1e-6, 1e-3];

/// The jitter ladder, once for every solve here: factorize `A` (row-major
/// `k*k`) as given, then with `eps · trace/k` added to the diagonal for each
/// `eps` in [`JITTER`], returning the lower factor `L` (`L Lᵀ` the matrix
/// factorized, zeros above the diagonal), the attempts it took and the
/// jitter, or `None` if every rung fails. It was written out twice, in
/// [`solve_spd`] and [`SpdFactor::of`], identical and free to drift apart
/// (review 2026-09-12, D2).
///
/// Each rung is `faer`'s `Mat::llt(Side::Lower)` step for step -- the lower
/// triangle of the matrix copied over zeros, factorized in place at the
/// global parallelism, the strict upper triangle zeroed -- so the factor is
/// that one's to the bit; it is held as a matrix of its own so that
/// [`SpdFactor::updated`] can move it in place, which `faer`'s `Llt` does
/// not allow.
fn factorize(a: &[f64], k: usize) -> Option<(Mat<f64>, u32, f64)> {
    debug_assert_eq!(a.len(), k * k);
    let trace: f64 = (0..k).map(|i| a[i * k + i]).sum();
    let base = if trace > 0.0 { trace / k as f64 } else { 1.0 };
    let par = faer::get_global_parallelism();
    for (attempts, &eps) in JITTER.iter().enumerate() {
        let jitter = base * eps;
        let mut l = Mat::from_fn(k, k, |i, j| match i.cmp(&j) {
            Ordering::Less => 0.0,
            _ => a[i * k + j] + if i == j { jitter } else { 0.0 },
        });
        let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
            k,
            par,
            Default::default(),
        ));
        let factored = llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        if factored.is_ok() {
            for j in 1..k {
                for i in 0..j {
                    l[(i, j)] = 0.0;
                }
            }
            return Some((l, attempts as u32, jitter));
        }
    }
    None
}

/// `X ← A⁻¹ X` from the lower factor `L` of `A`: the two triangular solves
/// `faer`'s `Llt::solve_in_place` makes, at the same parallelism.
fn solve_in_place(l: &Mat<f64>, rhs: MatMut<'_, f64>) {
    let par = faer::get_global_parallelism();
    let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(
        l.nrows(),
        rhs.ncols(),
        par,
    ));
    llt::solve::solve_in_place_with_conj(l.as_ref(), Conj::No, rhs, par, MemStack::new(&mut mem));
}

/// Solve `A x = B` for symmetric positive definite `A` (row-major `k*k`),
/// `B` column-major `k x m`. On factorization failure retries with jitter
/// `eps * trace/k` added to the diagonal (eps escalating), returning
/// `(solution, jitter_attempts)`. Returns `None` if even the largest jitter fails.
pub fn solve_spd(a: &[f64], b: &[f64], k: usize, m: usize) -> Option<(Vec<f64>, u32)> {
    debug_assert_eq!(b.len(), k * m);
    let (l, attempts, _) = factorize(a, k)?;
    let mut x = Mat::from_fn(k, m, |i, j| b[j * k + i]);
    solve_in_place(&l, x.as_mut());
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
/// same to the bit (docs/PERFORMANCE.md §13). A caller whose matrix moves by
/// a scale and a rank-one step, or by a rescaling of its rows and columns,
/// can move the factor with it in `O(k²)` ([`Self::updated`],
/// [`Self::congruent`]); a factor so moved holds the moved matrix to
/// rounding, not to the bit of a fresh factorization of it.
#[derive(Clone, Debug)]
pub struct SpdFactor {
    /// `L`, lower triangular with a positive diagonal and zeros above it:
    /// `L Lᵀ` is the matrix factorized, jitter included.
    l: Mat<f64>,
    /// `ln det`, taken on the first read: `ewridge`'s solves never read it,
    /// and at one solve a row its `k` logarithms were 1% of the row.
    log_det: OnceLock<f64>,
    attempts: u32,
    jitter: f64,
    /// In-place moves since the factorization ([`SpdFactor::MAX_MOVES`]).
    moves: u32,
}

impl SpdFactor {
    /// In-place moves a factor takes ([`Self::updated`], [`Self::congruent`])
    /// before it refuses the next, and its owner factorizes afresh: how the
    /// rounding the moves leave is bounded (task 170). Each move is backward
    /// stable -- the factor it leaves is that of the moved matrix to a few
    /// units of rounding in that matrix's size -- and the errors add, so a
    /// factor moved `n` times carries `n` of them where a fresh one carries
    /// one. Under a mean form each update's scale `c < 1` also shrinks the
    /// older ones, but without decay `c` tends to 1 and they stay. A count
    /// bounds that drift without reading the matrix, which a check would do
    /// at `O(k²)` a move, the cost of the move itself, and it is a function
    /// of the moves alone, so where the refactorizations fall does not depend
    /// on how a stream is chunked. Measured over positive definite matrices
    /// conditioned up to `1e12` (`a_factor_moved_to_its_cap_still_holds_
    /// its_matrix_and_moves_no_more`), the error grows about as the square
    /// root of the moves: after 64 to 127 of them `L Lᵀ` was off its matrix
    /// by at most 17 units of rounding in the matrix's size, after 256 to 511
    /// by at most 35, where a fresh factorization was off by at most 2. At
    /// this cap, within an order of magnitude of a fresh factorization, the
    /// refactorization it forces costs `O(k³)/64` a move, a fraction of the
    /// move.
    pub const MAX_MOVES: u32 = 64;

    /// Factorize `A` (row-major `k*k`), or `None` if every jitter fails.
    pub fn of(a: &[f64], k: usize) -> Option<Self> {
        let (l, attempts, jitter) = factorize(a, k)?;
        Some(Self {
            l,
            log_det: OnceLock::new(),
            attempts,
            jitter,
            moves: 0,
        })
    }

    /// `ln det` of the matrix factorized (jitter included).
    pub fn log_det(&self) -> f64 {
        *self.log_det.get_or_init(|| {
            let l = &self.l;
            2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>()
        })
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

    /// In-place moves since the factorization: 0 for a factor fresh from
    /// [`Self::of`], whose numbers are a one-shot factorization's to the bit.
    pub fn moves(&self) -> u32 {
        self.moves
    }

    /// Whether [`Self::updated`] and [`Self::congruent`] would take a step:
    /// the factorization needed no jitter and the factor has room under
    /// [`Self::MAX_MOVES`]. A caller with work to do to form a move's arguments
    /// asks this first.
    pub fn can_move(&self) -> bool {
        self.attempts == 0 && self.moves < Self::MAX_MOVES
    }

    /// The factor of `A' = c·A + v vᵀ`, made from this one in `O(k²)` rather
    /// than by factorizing `A'` in `O(k³)`. `√c·L` factors `c·A`, and `v vᵀ`
    /// is added by plane rotations, column by column (LINPACK's `dchud`):
    /// with `d` the pivot and `p` the step's entry there, `r = √(d² + p²)`
    /// replaces the pivot, and each later entry of the column and of the step
    /// is rotated by `cos = d/r`, `sin = p/r`. `[√c·Lᵀ; vᵀ]` is rotated to
    /// `[L'ᵀ; 0]`, so `L' L'ᵀ = c·A + v vᵀ`, and both factors are at most 1,
    /// so a step far larger than a pivot costs no accuracy: the update is
    /// backward stable. `faer`'s `rank_r_update_clobber` forms the same
    /// column as a difference of two terms of size `r/d`, and lost three
    /// digits where the step dominated a pivot (a 2x2 at condition `1e6`:
    /// `L Lᵀ` off the matrix by 1415 units of rounding, where a fresh
    /// factorization was off by 1.9; task 170). `r` is taken over the
    /// larger of `d` and `p`, so neither overflows nor underflows in the
    /// square. `v` is the update's working space and is left holding nothing
    /// of use.
    ///
    /// Allowed only where the step is the factor's own matrix moving exactly:
    /// `c` finite and `> 0`, `v` of the factor's size and finite, and the
    /// factor one that needed no jitter. A jittered factor holds `A + δI`,
    /// and `c·(A + δI) + v vᵀ` is not the factor of `A'` nor of `A'` with the
    /// jitter a fresh factorization of it would choose (likely none), so its
    /// owner factorizes `A'` afresh. So it does after [`Self::MAX_MOVES`] moves,
    /// which bound the rounding the moves accumulate. `None` refuses the step,
    /// or reports a factor the step left with a pivot or an entry that is not
    /// a positive finite number: either way the factor is spent, and its
    /// owner factorizes `A'`.
    pub fn updated(mut self, c: f64, v: &mut [f64]) -> Option<Self> {
        let k = self.l.nrows();
        if !(self.can_move() && c > 0.0 && c.is_finite())
            || v.len() != k
            || !v.iter().all(|x| x.is_finite())
        {
            return None;
        }
        let root = c.sqrt();
        for j in 0..k {
            let col = &mut self.l.col_mut(j).try_as_col_major_mut()?.as_slice_mut()[j..];
            let (d, p) = (root * col[0], v[j]);
            // Over the larger of the two, so that neither square overflows
            // nor underflows. A pivot and a step both 0 -- the pivot
            // underflowed under `√c` -- give `0/0`, a pivot that is not a
            // number, which spends the factor (`Self::moved`).
            let big = d.abs().max(p.abs());
            let (dd, pp) = (d / big, p / big);
            let r = big * (dd * dd + pp * pp).sqrt();
            let (cos, sin) = (d / r, p / r);
            col[0] = r;
            for (li, xi) in col[1..].iter_mut().zip(&mut v[j + 1..]) {
                let l = root * *li;
                *li = cos * l + sin * *xi;
                *xi = cos * *xi - sin * l;
            }
        }
        self.moved()
    }

    /// The factor of `E·A·E` with `E = diag(e)`, every `e_i` finite and
    /// `> 0` -- `A`'s rows and columns rescaled, as a correlation matrix
    /// moves when the scales it is taken in do -- made from this one in
    /// `O(k²)`: `E·L` is lower triangular with a positive diagonal and
    /// `(E·L)(E·L)ᵀ = E·A·E`, so each entry of `L` is multiplied by its row's
    /// `e_i`, one rounding each. Allowed, and `None`, as [`Self::updated`].
    pub fn congruent(mut self, e: &[f64]) -> Option<Self> {
        let k = self.l.nrows();
        if !self.can_move() || e.len() != k || !e.iter().all(|x| *x > 0.0 && x.is_finite()) {
            return None;
        }
        for j in 0..k {
            let col = self.l.col_mut(j).try_as_col_major_mut()?.as_slice_mut();
            for (x, &ei) in col[j..].iter_mut().zip(&e[j..]) {
                *x *= ei;
            }
        }
        self.moved()
    }

    /// The bookkeeping of a move: every pivot and every entry below it a
    /// finite number and the pivots positive, or the factor is spent; the
    /// move counted; `ln det` taken again on its next read.
    fn moved(mut self) -> Option<Self> {
        let k = self.l.nrows();
        for j in 0..k {
            let col = self.l.col(j).try_as_col_major()?.as_slice();
            if !(col[j] > 0.0 && col[j..].iter().all(|x| x.is_finite())) {
                return None;
            }
        }
        self.moves += 1;
        self.log_det = OnceLock::new();
        Some(self)
    }

    /// Solve `A X = B` from the kept factor, `B` column-major `k x m`, the
    /// answer laid out the same way: the numbers [`solve_spd`] gives, to the
    /// bit, since both are this factor's two triangular solves.
    pub fn solve(&self, b: &[f64], k: usize, m: usize) -> Vec<f64> {
        debug_assert_eq!(b.len(), k * m);
        // faer's `solve` is a zeroed copy of its right-hand side solved in
        // place; this is that, on the right-hand side's own matrix, with one
        // allocation and one copy fewer.
        let mut x = Mat::from_fn(k, m, |i, j| b[j * k + i]);
        solve_in_place(&self.l, x.as_mut());
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
        let mut x = Mat::<f64>::identity(k, k);
        solve_in_place(&self.l, x.as_mut());
        (0..k).map(|i| x[(i, i)]).collect()
    }

    /// Whether [`Self::inverse_diagonal`] is sure to come out as numbers,
    /// told in `O(k²)` without taking it (docs/PLAN.md task 140). A factor
    /// faer accepts can still overflow its inverse -- a Gram near `1e-309`
    /// has pivots near `1e-155` -- and past an infinity the two triangular
    /// solves can form `∞ − ∞`. With `M(L)` the comparison matrix, `|l_jj|`
    /// on the diagonal and `−|l_ij|` off it, `|L⁻¹| ≤ M(L)⁻¹` entrywise
    /// (`L = D(I − N)` with `N` nilpotent, so `L⁻¹ = Σ Nᵖ D⁻¹`), and
    /// `M(L)⁻¹ ≥ 0`, so each of its entries is at most its row's sum,
    /// `z = M(L)⁻¹ 1`: one substitution whose terms are all nonnegative, so
    /// nothing cancels. With `b = max z` bounding every entry of `L⁻¹` and
    /// `L⁻ᵀ`, and `l` every entry of `L`, each quantity the two solves form
    /// is at most `k² · max(l, 1) · max(b, 1)²`; held to `1e300`, eight
    /// orders below overflow, nothing overflows, and past positive pivots
    /// only an infinity makes a NaN. `false` claims nothing: the diagonal
    /// may still be finite.
    ///
    /// The substitution runs by columns, over contiguous slices: once `z_i`
    /// is known, column `i` below the diagonal adds `|l_ji| · z_i` to each
    /// later row's sum. An entry of `L` that is not finite needs no check of
    /// its own: `z_i > 0`, so it makes its row's `z` infinite or NaN, which
    /// the check on each `z` refuses.
    pub fn inverse_is_finite(&self) -> bool {
        let l = self.l.as_ref();
        let k = l.nrows();
        let mut z = vec![1.0f64; k];
        let mut l_max = 1.0f64;
        for i in 0..k {
            let Some(col) = l.col(i).try_as_col_major() else {
                return false;
            };
            let col = col.as_slice();
            let d = col[i];
            if !(d > 0.0 && d.is_finite()) {
                return false;
            }
            let zi = z[i] / d;
            if zi.is_nan() || zi > 1e150 {
                return false;
            }
            z[i] = zi;
            l_max = l_max.max(d);
            for (s, &v) in z[i + 1..].iter_mut().zip(&col[i + 1..]) {
                let v = v.abs();
                l_max = l_max.max(v);
                *s += v * zi;
            }
        }
        let b = z.iter().fold(1.0f64, |a, &v| a.max(v));
        let k = k as f64;
        k * k * l_max * b * b <= 1e300
    }

    /// Quadratic forms `d_jᵀ A⁻¹ d_j` for the `m` column vectors of `d`
    /// (column-major `k x m`), each clamped at zero: it is a squared norm,
    /// and rounding in the solve must not hand the caller a negative one.
    pub fn quad_forms(&self, d: &[f64], k: usize, m: usize) -> Vec<f64> {
        debug_assert_eq!(d.len(), k * m);
        let mut x = Mat::from_fn(k, m, |i, j| d[j * k + i]);
        solve_in_place(&self.l, x.as_mut());
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
    Some((f.quad_forms(d, k, m), f.log_det(), f.attempts))
}

/// `beta · [1, x]` when `fit_intercept`, else `beta · x`, summed left to
/// right from zero -- the order every model's `step` uses on its augmented
/// row buffer, so a `predict` built on this is bit-for-bit the step's own
/// prediction.
pub(crate) fn dot_aug(beta: &[f64], x: &[f64], fit_intercept: bool) -> f64 {
    let (mut acc, slopes) = if fit_intercept {
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

    /// A factor fresh from [`SpdFactor::of`] -- never moved -- gives the
    /// one-shot numbers to the bit. A moved one does not: it holds the moved
    /// matrix to rounding, as `a_moved_factor_is_a_fresh_factorization_of_
    /// the_moved_matrix` measures (task 170).
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
        assert_eq!(f.moves(), 0, "a fresh factor");
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

    /// `inverse_is_finite` is sound: over matrices at every scale from the
    /// subnormal to near overflow, well and badly conditioned, whenever it
    /// says so the diagonal of `A⁻¹` is finite; and diagonals that are not
    /// finite are among them, for it to catch (docs/PLAN.md task 140).
    #[test]
    fn inverse_is_finite_never_passes_a_diagonal_that_is_not() {
        let mut s = 1u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let (mut certified, mut not_finite) = (0, 0);
        for exp in [
            -320, -312, -310, -309, -308, -305, -300, -200, -100, 0, 100, 150, 154,
        ] {
            let scale: f64 = format!("1e{exp}").parse().unwrap();
            for k in 1..=6 {
                for shift in [0.0, 1e-8, 1.0] {
                    for _ in 0..20 {
                        // `scale · (B Bᵀ + shift · I)`.
                        let b: Vec<f64> = (0..k * k).map(|_| next()).collect();
                        let mut a = vec![0.0; k * k];
                        for i in 0..k {
                            for j in 0..k {
                                let v: f64 = (0..k).map(|l| b[i * k + l] * b[j * k + l]).sum();
                                a[i * k + j] = scale * (v + if i == j { shift } else { 0.0 });
                            }
                        }
                        let Some(f) = SpdFactor::of(&a, k) else {
                            continue;
                        };
                        let finite = f.inverse_diagonal(k).iter().all(|v| v.is_finite());
                        if f.inverse_is_finite() {
                            certified += 1;
                            assert!(finite, "scale 1e{exp}, k {k}: {a:?}");
                        }
                        not_finite += usize::from(!finite);
                    }
                }
            }
        }
        assert!(
            certified > 0 && not_finite > 0,
            "{certified} certified, {not_finite} not finite: both must occur"
        );
    }

    /// The jitter is `eps · trace/k` (`trace/k` = 2.5 here, a singular
    /// matrix whose first row sums to 6), and 1 stands in for `trace/k`
    /// when the trace is not positive: the zero matrix is rescued at the
    /// first rung with exactly `1e-12`, a negative definite one fails every
    /// rung (task 158).
    #[test]
    fn the_jitter_is_a_rung_times_trace_over_k() {
        let a = [4.0, 2.0, 2.0, 1.0];
        let f = SpdFactor::of(&a, 2).unwrap();
        assert!(f.attempts() > 0, "the singular matrix needed a rung");
        let eps = [0.0, 1e-12, 1e-9, 1e-6, 1e-3][f.attempts() as usize];
        assert_eq!(f.jitter(), 2.5 * eps);
        // The 3x3 trace is 9, `trace/k` 3.
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0];
        let g = SpdFactor::of(&b, 3).unwrap();
        let eps = [0.0, 1e-12, 1e-9, 1e-6, 1e-3][g.attempts() as usize];
        assert!(g.attempts() > 0);
        assert_eq!(g.jitter(), 3.0 * eps);
        let zero = SpdFactor::of(&[0.0; 4], 2).expect("a zero Gram is rescued");
        assert_eq!((zero.attempts(), zero.jitter()), (1, 1e-12));
        assert!(SpdFactor::of(&[-1.0, 0.0, 0.0, -1.0], 2).is_none());
    }

    /// `inverse_is_finite`'s certificate from its definition, by faer
    /// rather than the hand-written substitution: `z = M(L)⁻¹ 1` with
    /// `M(L)` the comparison matrix (`|l_jj|` on the diagonal, `−|l_ij|`
    /// below it), `b = max(max z, 1)`, `l = max(max |l_ij|, 1)`, and the
    /// quantity `k² · l · b²` it holds to `1e300`.
    fn certificate(f: &SpdFactor) -> f64 {
        let l = f.l.as_ref();
        let k = l.nrows();
        let m = Mat::from_fn(k, k, |i, j| match i.cmp(&j) {
            std::cmp::Ordering::Equal => l[(i, i)].abs(),
            std::cmp::Ordering::Greater => -l[(i, j)].abs(),
            std::cmp::Ordering::Less => 0.0,
        });
        let mut z = Mat::from_fn(k, 1, |_, _| 1.0);
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            m.as_ref(),
            z.as_mut(),
            Par::Seq,
        );
        let b = (0..k).fold(1.0f64, |a, i| a.max(z[(i, 0)]));
        let mut lmax = 1.0f64;
        for i in 0..k {
            for j in 0..=i {
                lmax = lmax.max(l[(i, j)].abs());
            }
        }
        (k * k) as f64 * lmax * b * b
    }

    /// Matrices built to land the certificate a few per cent either side of
    /// its `1e300` limit, so every term of it decides one of them: `k` from
    /// 1 to 4, rows coupled strongly (so `z` is mostly the substitution's
    /// sums), and a first row at `1e6` so `l > 1`. `inverse_is_finite` is
    /// the faer-computed certificate's verdict on each (task 158).
    #[test]
    fn inverse_is_finite_is_its_certificate_on_either_side_of_the_limit() {
        // `L0`, lower triangular with strong, uneven coupling below the diagonal.
        let l0 = |i: usize, j: usize| -> f64 {
            match i.cmp(&j) {
                std::cmp::Ordering::Equal => 1.0 + 0.25 * i as f64,
                std::cmp::Ordering::Greater => -(1.5 + 0.5 * (i + 2 * j) as f64),
                std::cmp::Ordering::Less => 0.0,
            }
        };
        // `A = (D L0)(D L0)ᵀ`, `D = diag(√s, ..., √s)` with its first entry
        // `top` instead when that is past 1; the factor is `D L0` up to
        // rounding.
        let build = |k: usize, top: f64, s: f64| -> Vec<f64> {
            let d = |i: usize| if i == 0 && top > 1.0 { top } else { s.sqrt() };
            let lt = |i: usize, j: usize| d(i) * l0(i, j);
            let mut a = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    a[i * k + j] = (0..k).map(|r| lt(i, r) * lt(j, r)).sum();
                }
            }
            a
        };
        let (mut certified, mut refused) = (0, 0);
        for k in 1..=4usize {
            for top in [1.0, 1e6] {
                if k == 1 && top > 1.0 {
                    continue;
                }
                let s0 = 1e-280;
                let q0 = certificate(&SpdFactor::of(&build(k, top, s0), k).unwrap());
                for r in [0.5, 0.95, 1.05, 2.0] {
                    // The certificate goes as `1/s` once `z` is past 1.
                    let s = s0 * q0 / (r * 1e300);
                    let f = SpdFactor::of(&build(k, top, s), k).unwrap();
                    assert_eq!(f.attempts(), 0, "k {k}, top {top}, r {r}");
                    let q = certificate(&f);
                    let ratio = q / 1e300;
                    assert!(
                        (ratio - r).abs() < 0.01 * r,
                        "k {k}, top {top}: certificate {q:e} for target {r}"
                    );
                    assert_eq!(f.inverse_is_finite(), q <= 1e300, "k {k}, top {top}, r {r}");
                    if top > 1.0 {
                        assert!(f.l[(0, 0)] > 1e5, "l > 1 on this case");
                    }
                    certified += usize::from(q <= 1e300);
                    refused += usize::from(q > 1e300);
                }
            }
        }
        assert_eq!((certified, refused), (14, 14));
    }

    /// `z_0 = 1/l_00` exactly at the substitution's `1e150` cut is not
    /// refused there: `1e150²` rounds to just under `1e300`, so the 1x1
    /// matrix `[1e-300]`, whose inverse is `1e300`, is certified (task 158).
    #[test]
    fn a_pivot_exactly_at_the_cut_is_certified() {
        let f = SpdFactor::of(&[1e-300], 1).unwrap();
        assert_eq!(1.0 / f.l[(0, 0)], 1e150, "z_0 lands on the cut");
        assert!(certificate(&f) <= 1e300);
        assert!(f.inverse_is_finite());
        let inv = f.inverse_diagonal(1)[0];
        assert!((inv / 1e300 - 1.0).abs() < 1e-15, "{inv:e}");
    }

    // ---- task 170: the moves, against faer ------------------------------

    /// A uniform draw in `[-1, 1)` from a seeded LCG, the suite's generator.
    fn draw(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// `A = Q diag(λ) Qᵀ`, row-major, with `Q` the orthogonal factor of a
    /// random matrix (faer's QR) and `λ` geometric from 1 to `cond`, exactly
    /// symmetric: a positive definite matrix of condition number `cond`.
    fn spd(k: usize, cond: f64, s: &mut u64) -> Vec<f64> {
        let g = Mat::from_fn(k, k, |_, _| draw(s));
        let q = g.qr().compute_Q();
        let lam = |i: usize| {
            if k == 1 {
                1.0
            } else {
                cond.powf(i as f64 / (k - 1) as f64)
            }
        };
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|r| q[(i, r)] * lam(r) * q[(j, r)]).sum();
            }
        }
        for i in 0..k {
            for j in 0..i {
                let v = 0.5 * (a[i * k + j] + a[j * k + i]);
                a[i * k + j] = v;
                a[j * k + i] = v;
            }
        }
        a
    }

    /// faer's own factor of `A` (row-major), from scratch: the oracle.
    fn faer_factor(a: &[f64], k: usize) -> Mat<f64> {
        Mat::from_fn(k, k, |i, j| a[i * k + j])
            .llt(faer::Side::Lower)
            .expect("the oracle factorizes")
            .L()
            .to_owned()
    }

    /// `‖X‖_F` of a row-major matrix.
    fn frob(a: &[f64]) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// `‖L Lᵀ − A‖_F`, the product by faer.
    fn residual(l: MatRef<'_, f64>, a: &[f64]) -> f64 {
        let k = l.nrows();
        let llt = l * l.transpose();
        let d: Vec<f64> = (0..k * k).map(|ij| llt[(ij / k, ij % k)] - a[ij]).collect();
        frob(&d)
    }

    /// `κ₂(A)` from faer's eigenvalues.
    fn kappa(a: &[f64], k: usize) -> f64 {
        let ev = Mat::from_fn(k, k, |i, j| a[i * k + j])
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .unwrap();
        ev[k - 1] / ev[0]
    }

    /// `uᵀA⁻¹u` and `ln det A` by faer from scratch: the oracle's numbers.
    fn faer_quad_and_log_det(a: &[f64], k: usize, u: &[f64]) -> (f64, f64) {
        let llt = Mat::from_fn(k, k, |i, j| a[i * k + j])
            .llt(faer::Side::Lower)
            .expect("the oracle factorizes");
        let x = llt.solve(Mat::from_fn(k, 1, |i, _| u[i]));
        let q = (0..k).map(|i| u[i] * x[(i, 0)]).sum();
        let l = llt.L();
        (q, 2.0 * (0..k).map(|i| l[(i, i)].ln()).sum::<f64>())
    }

    /// `c·A + v vᵀ`, row-major: the matrix an update moves to.
    fn stepped(a: &[f64], k: usize, c: f64, v: &[f64]) -> Vec<f64> {
        (0..k * k)
            .map(|ij| c * a[ij] + v[ij / k] * v[ij % k])
            .collect()
    }

    /// `E·A·E` with `E = diag(e)`, row-major: the matrix a congruence moves to.
    fn rescaled(a: &[f64], k: usize, e: &[f64]) -> Vec<f64> {
        (0..k * k).map(|ij| e[ij / k] * a[ij] * e[ij % k]).collect()
    }

    /// The factor is `faer`'s own `Mat::llt(Side::Lower)` to the bit, held as
    /// a matrix of its own so that it can be moved, and its solve is faer's
    /// `Llt::solve` to the bit (task 170): over matrices conditioned from 1
    /// to `1e12`, and over a singular one, whose rung the factor took is
    /// faer's factor of the matrix with that jitter on its diagonal.
    #[test]
    fn the_factor_is_faers_own_to_the_bit() {
        let mut s = 3u64;
        let mut cases: Vec<(Vec<f64>, usize)> = Vec::new();
        for cond in [1.0, 1e4, 1e8, 1e12] {
            for k in [1usize, 2, 5, 10, 17] {
                cases.push((spd(k, cond, &mut s), k));
            }
        }
        cases.push((vec![4.0, 2.0, 2.0, 1.0], 2));
        for (a, k) in cases {
            let f = SpdFactor::of(&a, k).unwrap();
            let jittered: Vec<f64> = (0..k * k)
                .map(|ij| a[ij] + if ij / k == ij % k { f.jitter() } else { 0.0 })
                .collect();
            let llt = Mat::from_fn(k, k, |i, j| jittered[i * k + j])
                .llt(faer::Side::Lower)
                .unwrap();
            for i in 0..k {
                for j in 0..k {
                    assert_eq!(f.l[(i, j)].to_bits(), llt.L()[(i, j)].to_bits(), "k {k}");
                }
            }
            let b: Vec<f64> = (0..2 * k).map(|_| draw(&mut s)).collect();
            let want = llt.solve(Mat::from_fn(k, 2, |i, j| b[j * k + i]));
            let got = f.solve(&b, k, 2);
            for j in 0..2 {
                for i in 0..k {
                    assert_eq!(got[j * k + i].to_bits(), want[(i, j)].to_bits(), "k {k}");
                }
            }
        }
    }

    /// A factor moved by [`SpdFactor::updated`] or [`SpdFactor::congruent`]
    /// is a factorization of the moved matrix, held to faer's factorization
    /// of that matrix from scratch, the oracle (task 170). Over matrices
    /// conditioned from 1 to `1e12`, `k` from 1 to 10, scales `c` from
    /// `1e-3` to 1 and steps at the matrix's own scale, and rescalings
    /// spread over up to six orders of magnitude (as far as keeps the
    /// rescaled matrix's condition under `1e13`), each move's reconstruction is
    /// within `8 ε` of the matrix's size (measured: 2.6), its factor within
    /// `4 ε κ` of faer's (1.2), the quadratic form it serves within `8 ε κ`
    /// (3.4) and `ln det` within `8 k ε κ`: the bounds a backward stable
    /// factorization of the moved matrix meets, which a fresh one meets at
    /// 1.9, 1 and 1. `ln det` is read before the move, so it is taken again
    /// after it.
    #[test]
    fn a_moved_factor_is_a_fresh_factorization_of_the_moved_matrix() {
        let eps = f64::EPSILON;
        let mut s = 11u64;
        let (mut updates, mut congruences) = (0, 0);
        for cond in [1.0, 1e4, 1e8, 1e12] {
            for k in [1usize, 2, 5, 10] {
                for trial in 0..12 {
                    let a = spd(k, cond, &mut s);
                    let f = SpdFactor::of(&a, k).unwrap();
                    assert_eq!(f.attempts(), 0, "cond {cond:e}, k {k}");
                    let _ = f.log_det();
                    let (moved, matrix) = if trial % 3 < 2 {
                        let c = match trial {
                            0 => 1.0,
                            1 => 1e-3,
                            _ => 0.5 * (draw(&mut s) + 1.0) + 1e-3,
                        };
                        let scale = frob(&a).sqrt() / (k as f64).sqrt();
                        let v: Vec<f64> = (0..k).map(|_| scale * draw(&mut s)).collect();
                        let mut work = v.clone();
                        updates += 1;
                        (f.updated(c, &mut work).unwrap(), stepped(&a, k, c, &v))
                    } else {
                        // `κ(E·A·E)` is up to `κ(A)` times the square of the
                        // spread of `e`: kept under `1e13`, which `f64` holds.
                        let spread = ((13.0 - cond.log10()) / 4.0).min(3.0);
                        let e: Vec<f64> =
                            (0..k).map(|_| 10f64.powf(spread * draw(&mut s))).collect();
                        congruences += 1;
                        (f.congruent(&e).unwrap(), rescaled(&a, k, &e))
                    };
                    let case = format!("cond {cond:e}, k {k}, trial {trial}");
                    assert_eq!(moved.moves(), 1, "{case}");
                    let size = frob(&matrix);
                    let back = residual(moved.l.as_ref(), &matrix);
                    assert!(
                        back <= 8.0 * eps * size,
                        "{case}: {:.2} eps",
                        back / (eps * size)
                    );
                    let kap = kappa(&matrix, k);
                    let lf = faer_factor(&matrix, k);
                    let dl: Vec<f64> = (0..k * k)
                        .map(|ij| moved.l[(ij / k, ij % k)] - lf[(ij / k, ij % k)])
                        .collect();
                    let lf_size: Vec<f64> = (0..k * k).map(|ij| lf[(ij / k, ij % k)]).collect();
                    assert!(
                        frob(&dl) <= 4.0 * eps * kap * frob(&lf_size),
                        "{case}: factor off by {:.2} eps kappa",
                        frob(&dl) / (eps * kap * frob(&lf_size))
                    );
                    let u: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
                    let (q, ld) = faer_quad_and_log_det(&matrix, k, &u);
                    let got = moved.quad_forms(&u, k, 1)[0];
                    assert!(
                        (got - q).abs() <= 8.0 * eps * kap * q,
                        "{case}: quadratic form {got:e} against {q:e}"
                    );
                    assert!(
                        (moved.log_det() - ld).abs()
                            <= 8.0 * k as f64 * eps * kap * (1.0 + ld.abs()),
                        "{case}: ln det {} against {ld}",
                        moved.log_det()
                    );
                }
            }
        }
        assert!(
            updates > 100 && congruences > 50,
            "{updates} updates, {congruences} congruences"
        );
    }

    /// A factor moved to its cap, [`SpdFactor::MAX_MOVES`], still holds its matrix,
    /// and moves no more: the next update or congruence is refused, and its
    /// owner factorizes afresh (task 170). Pairs of moves as `robust`'s
    /// standardized band rows make them -- a mean-form update `c = W/(W+1)`
    /// with or without decay, then a rescaling by up to 5% -- over matrices
    /// conditioned from 1 to `1e12`: the reconstruction stays within `64 ε`
    /// of the matrix's size, the matrix itself tracked by the same
    /// recursion in `f64` (measured: 17 ε at up to 127 moves, 35 ε at up to
    /// 511, against a fresh factorization's 2 ε; the error grows about as
    /// the square root of the moves).
    #[test]
    fn a_factor_moved_to_its_cap_still_holds_its_matrix_and_moves_no_more() {
        let eps = f64::EPSILON;
        let mut s = 29u64;
        for cond in [1.0, 1e4, 1e8, 1e12] {
            for k in [2usize, 10] {
                for decay in [0.999, 1.0] {
                    let case = format!("cond {cond:e}, k {k}, decay {decay}");
                    let mut a = spd(k, cond, &mut s);
                    let mut f = SpdFactor::of(&a, k).unwrap();
                    assert!(f.can_move() && f.moves() == 0, "{case}");
                    let mut w_sum = 100.0f64;
                    while f.moves() < SpdFactor::MAX_MOVES {
                        let c = decay * w_sum / (decay * w_sum + 1.0);
                        w_sum = decay * w_sum + 1.0;
                        let scale = frob(&a).sqrt() / (k as f64).sqrt();
                        let v: Vec<f64> = (0..k)
                            .map(|_| (c / w_sum).sqrt() * scale * draw(&mut s))
                            .collect();
                        a = stepped(&a, k, c, &v);
                        let mut work = v.clone();
                        f = f.updated(c, &mut work).unwrap();
                        let e: Vec<f64> = (0..k).map(|_| 1.0 + 0.05 * draw(&mut s)).collect();
                        a = rescaled(&a, k, &e);
                        f = f.congruent(&e).unwrap();
                    }
                    assert_eq!(f.moves(), SpdFactor::MAX_MOVES, "{case}");
                    let back = residual(f.l.as_ref(), &a);
                    assert!(
                        back <= 64.0 * eps * frob(&a),
                        "{case}: {:.1} eps after {} moves",
                        back / (eps * frob(&a)),
                        SpdFactor::MAX_MOVES
                    );
                    assert!(!f.can_move(), "{case}");
                    let mut v = vec![0.1; k];
                    assert!(f.clone().updated(0.9, &mut v).is_none(), "{case}");
                    assert!(f.congruent(&vec![1.0; k]).is_none(), "{case}");
                }
            }
        }
    }

    /// A factor that needed jitter holds `A + δI`, which no move of `A`
    /// keeps: it is never moved (task 170).
    #[test]
    fn a_factor_that_needed_jitter_is_never_moved() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let f = SpdFactor::of(&a, 2).unwrap();
        assert!(f.attempts() > 0 && !f.can_move());
        assert!(f.clone().updated(0.5, &mut [1.0, -1.0]).is_none());
        assert!(f.congruent(&[2.0, 0.5]).is_none());
    }

    /// A move that leaves a pivot of 0, or an entry that is not finite,
    /// spends the factor: `L Lᵀ` would not be positive definite, or not a
    /// number (task 170). A rescaling by `1e-300` takes a pivot of `1e-30`
    /// to 0 where one by `1e-200` keeps it; one by `1e300` takes an entry of
    /// `1e10` past the largest double where one by `1e290` does not; and an
    /// update whose pivot, `1e154` scaled by `√c` to `1.3e308`, meets a step
    /// of `1.3e308` rotates to `r = 1.84e308`, past it.
    #[test]
    fn a_move_that_leaves_a_zero_pivot_or_an_overflow_spends_the_factor() {
        let small = SpdFactor::of(&[1e-60, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(small.l[(0, 0)], 1e-30);
        assert!(small.clone().congruent(&[1e-300, 1.0]).is_none());
        assert!(small.congruent(&[1e-200, 1.0]).is_some());
        let big = SpdFactor::of(&[1e20, 0.0, 0.0, 1.0], 2).unwrap();
        assert!(big.clone().congruent(&[1e300, 1.0]).is_none());
        assert!(big.congruent(&[1e290, 1.0]).is_some());
        let huge = SpdFactor::of(&[1e308, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(huge.l[(0, 0)], 1e154);
        assert!(
            huge.clone()
                .updated(1.69e308, &mut [1.3e308, 0.0])
                .is_none()
        );
        assert!(huge.updated(1.69e308, &mut [1e307, 0.0]).is_some());
    }

    /// Each argument a move cannot take is refused on its own: a scale that
    /// is not a positive number, a step of the wrong size or with an entry
    /// that is not finite, a rescaling of the wrong size or with an entry
    /// that is not a positive finite number. Beside each, the same move with
    /// good arguments is taken.
    #[test]
    fn a_move_with_an_argument_it_cannot_take_is_refused() {
        let a = [4.0, 1.0, 1.0, 3.0];
        let f = SpdFactor::of(&a, 2).unwrap();
        assert!(f.clone().updated(0.5, &mut [1.0, -1.0]).is_some());
        for c in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            assert!(f.clone().updated(c, &mut [1.0, -1.0]).is_none(), "c {c}");
        }
        for v in [
            vec![1.0],
            vec![1.0, 2.0, 3.0],
            vec![f64::NAN, 1.0],
            vec![1.0, f64::INFINITY],
        ] {
            assert!(f.clone().updated(0.5, &mut v.clone()).is_none(), "{v:?}");
        }
        assert!(f.clone().congruent(&[2.0, 0.5]).is_some());
        for e in [
            vec![2.0],
            vec![2.0, 0.5, 1.0],
            vec![0.0, 1.0],
            vec![1.0, -1.0],
            vec![f64::NAN, 1.0],
            vec![1.0, f64::INFINITY],
        ] {
            assert!(f.clone().congruent(&e).is_none(), "{e:?}");
        }
    }
}
