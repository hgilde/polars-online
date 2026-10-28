//! The condition estimate and the limit (docs/PLAN.md task 278, D4),
//! against `crate::oracle`: faer's partial-pivot LU inverse, which shares
//! no arithmetic with the Cholesky factor the estimate reads.

use super::*;
use crate::oracle;

/// The unit roundoff, `2⁻⁵³`, written out here rather than taken from the
/// module, so that the limit is checked against the number it is derived
/// from.
const U: f64 = 1.0 / 9_007_199_254_740_992.0;

/// A uniform draw in `[-1, 1)` from a seeded LCG, the suite's generator.
fn draw(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A centred, column-scaled Gram of `n` random rows: `x₁ = x₀ + eps·b`,
/// the rest independent, each column multiplied by `scale[j]`. Row-major.
fn gram(k: usize, n: usize, eps: f64, scale: &[f64], s: &mut u64) -> Vec<f64> {
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let mut x: Vec<f64> = (0..k).map(|_| draw(s)).collect();
            if k > 1 {
                x[1] = x[0] + eps * x[1];
            }
            x.iter().zip(scale).map(|(v, c)| v * c).collect()
        })
        .collect();
    let mean: Vec<f64> = (0..k)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = rows
                .iter()
                .map(|r| (r[i] - mean[i]) * (r[j] - mean[j]))
                .sum::<f64>()
                / n as f64;
        }
    }
    a
}

/// `‖H⁻¹‖₁` to the oracle's rounding: `H = D⁻¹ A D⁻¹` inverted by faer's
/// LU.
fn exact(a: &[f64], k: usize) -> f64 {
    let d: Vec<f64> = (0..k).map(|i| a[i * k + i].sqrt()).collect();
    let h: Vec<f64> = (0..k * k).map(|p| a[p] / (d[p / k] * d[p % k])).collect();
    let inv = oracle::inverse(&h);
    (0..k)
        .map(|j| (0..k).map(|i| inv[i * k + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Where the oracle and the estimate may part by rounding alone: each
/// inverts `H` to about `κ u` of itself (`κ ≤ k ‖H⁻¹‖₁`, `H`'s entries at
/// most 1), with a few `k` roundings to spare.
fn rounding(want: f64, k: usize) -> f64 {
    1e-12 + 64.0 * (k * k) as f64 * want * U
}

/// The estimate never exceeds `‖H⁻¹‖₁` (each candidate is a product's norm
/// over its vector's). On a nearly collinear design, the case the limit is
/// for, it reaches it, to the rounding of inverting `H`, with `x₁ = x₀ +
/// eps·b` for `eps` from `1e-2` to `1e-6` at orders from 1 to 40: the inverse is then
/// dominated by one direction, which the first sign vector finds. On a
/// clean design, far from any limit, it read from 0.51 of it (`k = 10`) to
/// all of it, measured 2026-10-25, within the factor of 3 Higham reports of
/// the method in practice (§15.3).
#[test]
fn the_estimate_is_the_inverse_norm_from_below() {
    let mut s = 278u64;
    let mut checked = 0;
    for k in [1usize, 2, 3, 5, 10, 40] {
        for eps in [1.0, 1e-2, 1e-4, 1e-6] {
            let a = gram(k, 4 * k + 20, eps, &vec![1.0; k], &mut s);
            let f = SpdFactor::of(&a, k).expect("positive definite");
            assert_eq!(f.attempts(), 0);
            let est = f.equilibrated_inverse_norm(&a);
            let want = exact(&a, k);
            let tol = rounding(want, k);
            assert!(
                est <= want * (1.0 + tol),
                "k {k} eps {eps}: {est:e} above {want:e}"
            );
            assert!(est >= want / 3.0, "k {k} eps {eps}: {est:e} of {want:e}");
            if eps < 1.0 {
                assert!(
                    est >= want * (1.0 - tol),
                    "k {k} eps {eps}: {est:e} of {want:e}"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 24);
}

/// The estimate reads the equilibrated matrix, so a column's units do not
/// move it (van der Sluis): features from `1e-10` to `1e10`, a raw Gram
/// whose own condition is past `1e40`, read as the unit-scale design does,
/// to rounding, and are not refused.
#[test]
fn the_estimate_is_free_of_the_columns_units() {
    let k = 3;
    let unit = gram(k, 200, 1e-2, &[1.0, 1.0, 1.0], &mut 7u64);
    let scaled = gram(k, 200, 1e-2, &[1e-10, 1.0, 1e10], &mut 7u64);
    let f = SpdFactor::of(&unit, k).unwrap();
    let g = SpdFactor::of(&scaled, k).unwrap();
    let (a, b) = (
        f.equilibrated_inverse_norm(&unit),
        g.equilibrated_inverse_norm(&scaled),
    );
    assert!((a / b - 1.0).abs() < 1e-10, "{a:e} {b:e}");
    assert!(!g.carries_no_digit(&scaled));
}

/// The limit is `3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1`, `1/(42 u) = 2.1e14` at `k = 2`:
/// a matrix whose estimate sits at half of it carries a digit and one at
/// twice it does not. `H = [[1, c], [c, 1]]` with `c` near 1, whose
/// inverse's 1-norm is `(1 + c)/(1 − c²) = 1/(1 − c)`, so the side of the
/// limit is set by `c`; the method reaches it exactly here (its unit
/// vector `e₀` is the largest column).
#[test]
fn the_limit_is_where_the_forward_error_bound_reaches_one() {
    let k = 2;
    let m = (3 * k + 1) as f64 * U;
    let limit = 1.0 / (3.0 * k as f64 * m / (1.0 - m));
    assert!((limit * 42.0 * U - 1.0).abs() < 1e-12, "{limit:e}");
    for (gap, refused) in [(2.0 / limit, false), (0.5 / limit, true)] {
        let c = 1.0 - gap;
        let a = [1.0, c, c, 1.0];
        let f = SpdFactor::of(&a, k).expect("positive definite");
        assert_eq!(f.attempts(), 0);
        let est = f.equilibrated_inverse_norm(&a);
        let want = 1.0 / (1.0 - c);
        // `1 − c²` keeps the rounding of `c²`, `u` against `2 (1 − c)`, under
        // 1% at these gaps, and the pivot its square root.
        assert!(
            (est / want - 1.0).abs() < 0.01,
            "gap {gap:e}: {est:e} of {want:e}"
        );
        assert_eq!(f.carries_no_digit(&a), refused, "gap {gap:e}: {est:e}");
    }
}

/// A matrix that fools the iteration: `H = [[1, c, 0], [c, 1, 0], [0, 0,
/// 1]]`. The first sign vector's product is largest in the third entry, the
/// iteration reads the third column, whose norm is 1, and stops there; the
/// alternating vector `[1, −1.5, 2]` then reads `2/(3k)·5/(2(1 − c)) =
/// 0.556/(1 − c)` of the truth `1/(1 − c)`, which is taken (Higham, §15.3,
/// on why the last vector is there). Within the factor of 3 the method
/// promises in practice, and far past the limit for a `c` that is.
#[test]
fn the_alternating_vector_catches_a_matrix_that_fools_the_iteration() {
    let gap = 1e-15;
    let c = 1.0 - gap;
    let a = [1.0, c, 0.0, c, 1.0, 0.0, 0.0, 0.0, 1.0];
    let f = SpdFactor::of(&a, 3).expect("positive definite");
    let est = f.equilibrated_inverse_norm(&a);
    let want = 1.0 / (1.0 - c);
    assert!((est / want - 5.0 / 9.0).abs() < 0.01, "{est:e} of {want:e}");
    assert!(f.carries_no_digit(&a));
}

/// A factor that needed jitter is never judged: it factors `A + δI`, and
/// its solve is counted already. One that factorized as given and is past
/// the limit is refused, and so is one whose estimate is not a number.
#[test]
fn a_jittered_factor_is_not_judged_and_a_singular_one_is() {
    let singular = [1.0, 1.0, 1.0, 1.0];
    let f = SpdFactor::of(&singular, 2).expect("a jitter rescues it");
    assert!(f.attempts() > 0);
    assert!(!f.carries_no_digit(&singular));
    // A correlation of `1 − 2⁻⁵⁰`: positive definite in floating point,
    // with `‖H⁻¹‖₁` near `2⁵⁰`, past the limit at `k = 2` (`1/(42u)`).
    let c = 1.0 - 2f64.powi(-50);
    let near = [1.0, c, c, 1.0];
    let g = SpdFactor::of(&near, 2).expect("positive definite");
    assert_eq!(g.attempts(), 0);
    assert!(g.carries_no_digit(&near));
    let mut nan = near;
    nan[0] = f64::NAN;
    assert!(g.carries_no_digit(&nan), "a NaN estimate is refused");
}

/// Random positive definite matrices, their eigenvectors random and their
/// conditions from 1 to `1e12`, at orders from 2 to 32: the estimate is a
/// lower bound of the oracle's `‖H⁻¹‖₁` and within a factor of 3 of it on
/// every one, as Higham reports of the method in practice (§15.3).
#[test]
fn the_estimate_is_within_three_of_the_truth_on_random_matrices() {
    let mut s = 20261025u64;
    let mut worst = 1.0f64;
    for k in [2usize, 4, 8, 16, 32] {
        for trial in 0..20 {
            let cond = 10f64.powi(trial % 13);
            let g = faer::Mat::from_fn(k, k, |_, _| draw(&mut s));
            let q = g.qr().compute_Q();
            let lam = |i: usize| cond.powf(i as f64 / (k - 1) as f64);
            let mut a = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let v: f64 = (0..k).map(|r| q[(i, r)] * lam(r) * q[(j, r)]).sum();
                    a[i * k + j] = v;
                    a[j * k + i] = v;
                }
            }
            let f = SpdFactor::of(&a, k).expect("positive definite");
            let est = f.equilibrated_inverse_norm(&a);
            let want = exact(&a, k);
            assert!(
                est <= want * (1.0 + rounding(want, k)),
                "k {k}: {est:e} above {want:e}"
            );
            worst = worst.min(est / want);
        }
    }
    assert!(worst >= 1.0 / 3.0, "worst ratio {worst}");
}

/// The screen's bound is an upper bound on `‖H⁻¹‖₁` (against the oracle),
/// so where it is inside the limit the system is, and the estimate it
/// skips could not have reached the limit: on every matrix here the
/// decision with the screen is the decision of the estimate alone. On a
/// clean design the bound is inside the limit, and the estimate is not run;
/// on the numerically singular one it is not, and the estimate decides.
#[test]
fn the_screen_bounds_the_inverse_norm_from_above_and_never_moves_a_decision() {
    let mut s = 2027u64;
    let k_limit = digit_limit;
    let mut screened = 0;
    for k in [1usize, 2, 3, 5, 10, 40] {
        for eps in [1.0, 1e-2, 1e-4, 1e-6, 1e-11] {
            let a = gram(k, 4 * k + 20, eps, &vec![1.0; k], &mut s);
            let Some(f) = SpdFactor::of(&a, k) else {
                continue;
            };
            if f.attempts() != 0 {
                continue;
            }
            let bound = f.with_work(&a, |f, d, work| {
                let (z, w) = work.split_at_mut(k);
                f.comparison_bound(d, z, w)
            });
            let est = f.equilibrated_inverse_norm(&a);
            let decided_alone = est.is_nan() || est >= k_limit(k);
            assert_eq!(f.carries_no_digit(&a), decided_alone, "k {k} eps {eps}");
            if eps > 1e-11 {
                let want = exact(&a, k);
                assert!(
                    bound >= want * (1.0 - rounding(want, k)),
                    "k {k} eps {eps}: {bound:e} below {want:e}"
                );
            }
            screened += usize::from(bound < k_limit(k));
            if eps == 1.0 {
                assert!(
                    bound < k_limit(k),
                    "k {k}: a clean design was not screened, {bound:e}"
                );
            }
        }
    }
    assert!(screened >= 6, "{screened}");
}

/// The limit, `1 / (3 k γ_{3k+1}) = (1 − m) / (3 k m)` with `m = (3k + 1)
/// u`, against the formula written out with this file's own `u`, to the bit,
/// at several orders: at `k = 2`, `(1 − 7u) / (42 u)`.
#[test]
fn the_limit_is_the_formula() {
    for k in [1usize, 2, 3, 10, 50, 1000] {
        let m = (3 * k + 1) as f64 * U;
        let want = (1.0 - m) / (3.0 * k as f64 * m);
        let got = digit_limit(k);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "k {k}: {got:e} against {want:e}"
        );
    }
    assert!((digit_limit(2) * 42.0 * U / (1.0 - 7.0 * U) - 1.0).abs() < 1e-15);
    assert!((digit_limit(3) / 1.0e14 - 1.0).abs() < 0.01);
    assert!((digit_limit(10) / 9.7e12 - 1.0).abs() < 0.01);
    assert!((digit_limit(50) / 4.0e11 - 1.0).abs() < 0.01);
}

/// The screen's bound is its definition: `max(M(R)⁻¹ 1) · max(M(R)⁻ᵀ 1)`
/// with `R = D⁻¹ L` and `M(R)` its comparison matrix, here formed entry by
/// entry from the factor and inverted by the oracle's LU, on designs whose
/// columns are scaled from `1e-3` to `1e3`, so that `D` and the substitutions'
/// intermediate sums are far from 1.
#[test]
fn the_screens_bound_is_the_comparison_matrix_bound() {
    let mut s = 31u64;
    for k in [2usize, 3, 6, 12] {
        let scale: Vec<f64> = (0..k)
            .map(|i| 10f64.powf(-3.0 + 6.0 * i as f64 / (k - 1) as f64))
            .collect();
        for eps in [1.0, 1e-2, 1e-4] {
            let a = gram(k, 4 * k + 20, eps, &scale, &mut s);
            let f = SpdFactor::of(&a, k).expect("positive definite");
            let got = f.with_work(&a, |f, d, work| {
                let (z, w) = work.split_at_mut(k);
                f.comparison_bound(d, z, w)
            });
            let mut m = vec![0.0; k * k];
            for i in 0..k {
                let di = a[i * k + i].sqrt();
                for j in 0..=i {
                    let r = f.l[(i, j)] / di;
                    m[i * k + j] = if i == j { r.abs() } else { -r.abs() };
                }
            }
            let inv = oracle::inverse(&m);
            let rows = (0..k)
                .map(|i| (0..k).map(|j| inv[i * k + j]).sum::<f64>())
                .fold(0.0, f64::max);
            let cols = (0..k)
                .map(|j| (0..k).map(|i| inv[i * k + j]).sum::<f64>())
                .fold(0.0, f64::max);
            let want = rows * cols;
            assert!(
                (got / want - 1.0).abs() < 1e-9 + rounding(want, k),
                "k {k} eps {eps}: {got:e} against {want:e}"
            );
        }
    }
}

/// The screen settles a system only where its bound is below the limit:
/// strictly, so a limit at the bound itself, to the bit, hands the decision
/// to the estimate, and a limit above it does not. On a clean design the
/// bound and the estimate differ, so which one was read shows.
#[test]
fn the_screen_settles_only_below_the_limit() {
    let k = 6;
    let a = gram(k, 60, 1.0, &[1.0; 6], &mut 5u64);
    let f = SpdFactor::of(&a, k).unwrap();
    let bound = f.with_work(&a, |f, d, work| {
        let (z, w) = work.split_at_mut(k);
        f.comparison_bound(d, z, w)
    });
    let est = f.equilibrated_inverse_norm(&a);
    assert!(est < bound, "{est:e} {bound:e}");
    assert_eq!(f.judged_norm(&a, bound).to_bits(), est.to_bits());
    assert_eq!(f.judged_norm(&a, 2.0 * bound).to_bits(), bound.to_bits());
    assert!(!f.carries_no_digit(&a));
}

/// How a run of the method ended, for the evidence that each way was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ended {
    SignsRepeated,
    StoppedGrowing,
    SameColumn,
    Iterations,
}

/// Hager's method as Higham's Algorithm 15.4 and LAPACK's `dlacn2` state it
/// (for a symmetric `B`), with the estimate the largest candidate, written
/// out from the algorithm on a dense `B` the oracle inverted -- so its
/// products share no arithmetic with the triangular solves the estimate
/// makes -- and reporting how the iteration ended and whether the
/// alternating vector's candidate was the one taken.
fn reference(b: &[f64], k: usize) -> (f64, Ended, bool) {
    let times = |v: &[f64]| -> Vec<f64> {
        (0..k)
            .map(|i| (0..k).map(|j| b[i * k + j] * v[j]).sum())
            .collect()
    };
    let one = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>();
    let sgn = |v: &[f64]| {
        v.iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
            .collect::<Vec<f64>>()
    };
    // The first index of the largest absolute entry.
    let top = |v: &[f64]| (0..k).fold(0, |j, i| if v[i].abs() > v[j].abs() { i } else { j });
    let mut est = one(&times(&vec![1.0 / k as f64; k]));
    let mut xi = sgn(&times(&vec![1.0 / k as f64; k]));
    let mut j = top(&times(&xi));
    let mut ended = Ended::Iterations;
    for _ in 2..=5 {
        let mut e = vec![0.0; k];
        e[j] = 1.0;
        let y = times(&e);
        let (last, now) = (est, one(&y));
        est = est.max(now);
        if sgn(&y) == xi {
            ended = Ended::SignsRepeated;
            break;
        }
        if now <= last {
            ended = Ended::StoppedGrowing;
            break;
        }
        xi = sgn(&y);
        let z = times(&xi);
        let previous = j;
        j = top(&z);
        if z[previous].abs() == z[j].abs() {
            ended = Ended::SameColumn;
            break;
        }
    }
    let alt: Vec<f64> = (0..k)
        .map(|i| {
            let size = 1.0 + i as f64 / (k - 1) as f64;
            if i % 2 == 0 { size } else { -size }
        })
        .collect();
    let candidate = 2.0 * one(&times(&alt)) / (3 * k) as f64;
    (est.max(candidate), ended, candidate > est)
}

/// The estimate is Hager's method step for step: against the method
/// written out on the oracle's dense inverse, on random positive definite
/// matrices of orders 2 to 12 and conditions to `1e5`, it gives the same
/// number to rounding. Two ways of ending the iteration were taken, the
/// signs repeating and the largest entry staying put, and the alternating
/// vector's candidate was taken. The other two are not reached on the
/// inverse of a positive definite matrix: in 60,000 random ones of orders 2
/// to 12 (numpy, 2026-10-25) the estimate stopped growing only at order 2,
/// where `B e_0`'s norm is `|p| + |q|` against `|p + q|` and only rounding
/// puts it below, and no run took five iterations.
#[test]
fn the_estimate_is_hagers_method_step_for_step() {
    let mut s = 1529u64;
    let mut seen = std::collections::HashSet::new();
    let mut alternating = 0;
    for trial in 0..600 {
        let k = 2 + trial % 11;
        let cond = 10f64.powf(5.0 * draw(&mut s).abs());
        let g = faer::Mat::from_fn(k, k, |_, _| draw(&mut s));
        let q = g.qr().compute_Q();
        let lam = |i: usize| cond.powf(i as f64 / (k - 1) as f64);
        // A random diagonal scaling too, which the estimate takes off.
        let scale: Vec<f64> = (0..k).map(|_| 10f64.powf(2.0 * draw(&mut s))).collect();
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                let v: f64 = (0..k).map(|r| q[(i, r)] * lam(r) * q[(j, r)]).sum();
                a[i * k + j] = v * scale[i] * scale[j];
                a[j * k + i] = a[i * k + j];
            }
        }
        let Some(f) = SpdFactor::of(&a, k) else {
            continue;
        };
        let d: Vec<f64> = (0..k).map(|i| a[i * k + i].sqrt()).collect();
        let h: Vec<f64> = (0..k * k).map(|p| a[p] / (d[p / k] * d[p % k])).collect();
        let (want, ended, alt) = reference(&oracle::inverse(&h), k);
        let got = f.equilibrated_inverse_norm(&a);
        assert!(
            (got / want - 1.0).abs() < 1e-8,
            "trial {trial}, k {k}: {got:e} against {want:e} ({ended:?})"
        );
        seen.insert(ended);
        alternating += usize::from(alt);
    }
    assert!(
        seen.contains(&Ended::SignsRepeated) && seen.contains(&Ended::SameColumn),
        "{seen:?}"
    );
    assert!(alternating > 0);
}

/// The three small pieces of the method, as `dlacn2` defines them: a sign
/// of `+1` for 0 and `−1` for a NaN, the first of equal largest entries,
/// and the alternating vector's entries.
#[test]
fn the_methods_pieces_are_dlacn2s() {
    assert_eq!(sign(0.0), 1.0);
    assert_eq!(sign(2.5), 1.0);
    assert_eq!(sign(-1e-300), -1.0);
    assert_eq!(sign(f64::NAN), -1.0);
    assert_eq!(first_largest(&[1.0, -3.0, 3.0, 2.0]), 1);
    assert_eq!(first_largest(&[-2.0, 2.0]), 0);
    assert_eq!(first_largest(&[0.5, 0.25, 0.75]), 2);
    let mut x = [0.0; 5];
    alternating(&mut x);
    assert_eq!(x, [1.0, -1.25, 1.5, -1.75, 2.0]);
}

/// Task 281 (mutants): the iteration stops where the estimate stops
/// growing, as `dlacn2` does, even with the signs not repeating and the
/// estimate short of `‖B‖₁` -- the method's estimate is a lower bound, not
/// the norm. On this symmetric (indefinite) `B` of small integers, every
/// product exact on any machine, the first two products read 3, the column
/// they pick reads 3 again, and the method ends at 3, where its column 1
/// holds `‖B‖₁ = 7` and going on would have found it.
#[test]
fn the_iteration_ends_where_the_estimate_stops_growing() {
    let k = 3;
    let b = [-2.0, 0.0, 1.0, 0.0, -5.0, -2.0, 1.0, -2.0, 2.0];
    let (want, ended, alt) = reference(&b, k);
    assert_eq!((want, ended, alt), (3.0, Ended::StoppedGrowing, false));
    let mut work = vec![0.0; 2 * k];
    let got = norm1_estimate(k, &mut work, |x| {
        let y: Vec<f64> = (0..k)
            .map(|i| (0..k).map(|j| b[i * k + j] * x[j]).sum())
            .collect();
        x.copy_from_slice(&y);
    });
    assert_eq!(got, want);
}
