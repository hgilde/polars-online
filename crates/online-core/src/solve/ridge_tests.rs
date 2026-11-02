//! The ridge's shortcut (docs/PLAN.md task 306; the module docs, *The
//! ridge's shortcut*): its premise, that a live Gram's co-moments are
//! semidefinite to within `γ_{2n+1}` of their trace, `n` the rows of
//! positive weight; its bound on `‖H⁻¹‖₁` against `crate::oracle`'s LU
//! inverse and on Hager's estimate; and its decision at the edges, against
//! the long way's.

use super::*;
use crate::EwCov;
use crate::oracle;

/// The unit roundoff, `2⁻⁵³`, written out.
const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;

fn draw(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `C`'s lower triangle as a whole symmetric matrix.
fn comoments(cov: &EwCov) -> Vec<f64> {
    cov.comoments().to_full()
}

fn trace(a: &[f64], k: usize) -> f64 {
    (0..k).map(|i| a[i * k + i]).sum()
}

fn top(a: &[f64], k: usize) -> f64 {
    (0..k).map(|i| a[i * k + i]).fold(0.0, f64::max)
}

/// The premise's bound, item 1 with the constants of item 2: `−λ_min(C) ≤
/// 1.042 (2n + 5) u tr C`.
fn premise(n: u64, tr: f64) -> f64 {
    1.042 * (2 * n + 5) as f64 * UNIT * tr
}

/// Two rows, then `tail` rows at the running mean with weight `1e-12` and
/// no decay: `C` starts rank one, and every later row rounds `a·c_ij`
/// (`a = W/(W + 1e-12)` is not 1) while adding nothing to it, so the
/// entries' relative errors walk apart, and `W` stays at 2.
fn walked(k: usize, tail: usize, seed: u64) -> EwCov {
    let mut s = seed;
    let mut cov = EwCov::new(k);
    for _ in 0..2 {
        let x: Vec<f64> = (0..k).map(|_| 1.0 + draw(&mut s)).collect();
        cov.update(&x, 1.0, 1.0);
    }
    for _ in 0..tail {
        let x = cov.means().to_vec();
        cov.update(&x, 1.0, 1e-12);
    }
    cov
}

/// Item 1: the co-moments' indefiniteness grows with the rows, not with
/// the weight. On the walked stream the oracle's smallest eigenvalue of
/// `C` is negative, past any slack proportional to the weight `W` (`8 k W u
/// max c_ii`, the prototype's), and within the premise's `(2n + 5) u tr C`.
#[test]
fn the_rows_not_the_weight_bound_the_rounding() {
    for (k, tail, seed) in [
        (3usize, 200_000usize, 1u64),
        (3, 200_000, 2),
        (6, 100_000, 3),
    ] {
        let cov = walked(k, tail, seed);
        let c = comoments(&cov);
        let (values, _) = oracle::sym_eigen(&c);
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let (tr, w, n) = (trace(&c, k), cov.n_eff(), cov.rows_learned());
        assert_eq!(n, 2 + tail as u64);
        let by_weight = 8.0 * k as f64 * w * UNIT * top(&c, k);
        eprintln!(
            "k {k} seed {seed}: -lmin/(u tr) {:.1}, weight slack/(u tr) {:.1}, premise/(u tr) {:.1}",
            -low / (UNIT * tr),
            by_weight / (UNIT * tr),
            premise(n, tr) / (UNIT * tr)
        );
        assert!(-low > by_weight, "k {k}: {low:e} within {by_weight:e}");
        assert!(
            -low <= premise(n, tr),
            "k {k}: {low:e} past {:e}",
            premise(n, tr)
        );
    }
}

/// What a slack proportional to the weight would have decided, the
/// prototype's rule: `λ = ρ + δ − 8 k W u t`, and the system settled where
/// `2 √k t / λ` is below the limit.
fn by_weight_settles(a: &[f64], k: usize, ridge: f64, w: f64, f: &SpdFactor) -> bool {
    let t = top(a, k) + f.jitter();
    let lam = ridge + f.jitter() - 8.0 * k as f64 * w.max(1.0) * UNIT * t;
    lam > 0.0 && 2.0 * (k as f64).sqrt() * t / lam < digit_limit(k)
}

/// Why the slack counts rows: on the walked stream, a ridge just past
/// `C`'s most negative eigenvalue leaves `A = C + ρI` positive definite by
/// a few units in its last place of `tr C`. Its factorization succeeds as
/// given, and the estimate refuses it, past the limit; a slack
/// proportional to the weight would have settled it inside the limit and
/// served the fit, while the shipped rule, counting the rows, leaves it to
/// the estimate, and decides as the long way does.
#[test]
fn a_ridge_over_walked_moments_is_left_to_the_estimate() {
    let mut checked = 0;
    for (k, tail, seed) in [
        (3usize, 200_000usize, 1u64),
        (3, 200_000, 2),
        (6, 100_000, 3),
    ] {
        let cov = walked(k, tail, seed);
        let c = comoments(&cov);
        let (values, _) = oracle::sym_eigen(&c);
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let tr = trace(&c, k);
        let ridge = -low + 4.0 * UNIT * tr;
        let mut a = c.clone();
        for i in 0..k {
            a[i * k + i] += ridge;
        }
        let f = SpdFactor::of(&a, k).expect("the ladder");
        assert_eq!(f.attempts(), 0, "k {k} seed {seed}: factorized as given");
        assert!(
            f.carries_no_digit(&a),
            "k {k} seed {seed}: the estimate refuses"
        );
        assert!(
            by_weight_settles(&a, k, ridge, cov.n_eff(), &f),
            "k {k} seed {seed}: a weight's slack would have served it"
        );
        let over = OverMoments {
            ridge,
            rows: cov.rows_learned(),
        };
        assert!(!f.ridge_settles(&a, over), "k {k} seed {seed}");
        assert!(f.carries_no_digit_over(&a, Some(over)), "k {k} seed {seed}");
        checked += 1;
    }
    assert_eq!(checked, 3);
}

/// The design of a test stream's features.
#[derive(Clone, Copy, Debug)]
enum Design {
    /// Independent, uniform on `[−1, 1)` around 3.
    Free,
    /// `x₁ = x₀ + 1e-11 b`: the two equal to within `1e-11`.
    Near,
    /// Fewer rows than features: `C` singular.
    Few,
    /// A few rows, then rows at the running mean of small weight (`walked`).
    Walked,
}

/// A stream's co-moments under `design`, `lam` a row and the weights
/// `weights` in turn.
fn stream(k: usize, design: Design, lam: f64, weights: &[f64], seed: u64) -> EwCov {
    let mut s = seed;
    let mut cov = EwCov::new(k);
    let rows = match design {
        Design::Few => k / 2 + 1,
        _ => 4 * k + 40,
    };
    for r in 0..rows {
        let mut x: Vec<f64> = (0..k).map(|_| 3.0 + draw(&mut s)).collect();
        if matches!(design, Design::Near) && k > 1 {
            x[1] = x[0] + 1e-11 * draw(&mut s);
        }
        let w = weights[r % weights.len()];
        if matches!(design, Design::Walked) && r > k {
            let m = cov.means().to_vec();
            cov.update(&m, lam, 1e-9);
        } else {
            cov.update(&x, lam, w);
        }
    }
    cov
}

/// `A` as the centred solve forms it: `C`'s entries, or `c_ij/(s_i s_j)`,
/// with the ridge added to the diagonal. `None` where standardizing would
/// drop a column (a variance that is not positive).
fn system(cov: &EwCov, k: usize, ridge: f64, standardize: bool) -> Option<Vec<f64>> {
    let s: Vec<f64> = (0..k).map(|i| cov.cov(i, i).max(0.0).sqrt()).collect();
    if standardize && !(0..k).all(|i| cov.cov(i, i) > 0.0) {
        return None;
    }
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = if standardize {
                cov.cov(i, j) / (s[i] * s[j])
            } else {
                cov.cov(i, j)
            };
        }
        a[i * k + i] += ridge;
    }
    Some(a)
}

/// `‖H⁻¹‖₁` by the oracle's LU, `H = D⁻¹ (A + δI) D⁻¹`, `D = diag(√(a_ii +
/// δ))`: the matrix the estimate judges.
fn exact(a: &[f64], k: usize, delta: f64) -> f64 {
    let d: Vec<f64> = (0..k).map(|i| (a[i * k + i] + delta).sqrt()).collect();
    let h: Vec<f64> = (0..k * k)
        .map(|p| {
            let (i, j) = (p / k, p % k);
            let v = if i == j { a[p] + delta } else { a[p] };
            v / (d[i] * d[j])
        })
        .collect();
    let inv = oracle::inverse(&h);
    (0..k)
        .map(|j| (0..k).map(|i| inv[i * k + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The shortcut's slack test, item 5 written out: `1.25 ĝ ≤ (ρ + δ)/2`.
fn slack_holds(a: &[f64], k: usize, ridge: f64, rows: u64, delta: f64) -> bool {
    let (t, tr, kf, n) = (top(a, k), trace(a, k), k as f64, rows as f64);
    let g = (2.0 * n + 5.0) * UNIT * tr
        + (3.0 * kf + 1.0) * UNIT * (tr + kf * delta)
        + UNIT * (2.0 * t + delta);
    1.25 * g <= 0.5 * (ridge + delta)
}

/// Items 1 to 5 against the oracle. Over orders from 2 to 60, ridges from
/// `1e-12` to `100`, unit, mixed and zero weights, no decay and a short
/// one, standardized or not, and designs free, nearly collinear (features
/// equal to within `1e-11`), rank-deficient and walked: every `C` is
/// semidefinite to within the premise; wherever the slack test holds, the
/// oracle's `‖H⁻¹‖₁` and the estimate are both under `0.89` of the bound
/// `2 √k (t + δ)/(ρ + δ)` the shortcut compares with the limit (items 4 and
/// 5 put them under `1.78/2`); and wherever the shortcut fires, the long
/// way would not refuse. Both outcomes are reached: systems the shortcut
/// settles and systems it leaves to the estimate, some of them refused.
#[test]
fn the_ridge_bounds_the_inverse_norm_from_above() {
    let (mut settled, mut left, mut refused, mut slack) = (0, 0, 0, 0);
    let mut closest = 0.0f64;
    let mut seed = 306u64;
    for k in [2usize, 5, 20, 60] {
        for design in [Design::Free, Design::Near, Design::Few, Design::Walked] {
            for (lam, weights) in [
                (1.0, &[1.0][..]),
                (0.99, &[1.0][..]),
                (1.0, &[0.0, 0.1, 1.0, 5.0, 40.0][..]),
                (0.9, &[1.0, 0.0, 1e-6][..]),
            ] {
                seed += 1;
                let cov = stream(k, design, lam, weights, seed);
                let c = comoments(&cov);
                let n = cov.rows_learned();
                let (values, _) = oracle::sym_eigen(&c);
                let low = values.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(
                    -low <= premise(n, trace(&c, k)),
                    "k {k} {design:?} lam {lam}: {low:e}"
                );
                for ridge in [1e-12, 1e-9, 1e-6, 1e-3, 1.0, 100.0] {
                    for standardize in [false, true] {
                        let Some(a) = system(&cov, k, ridge, standardize) else {
                            continue;
                        };
                        let what =
                            format!("k {k} {design:?} lam {lam} ridge {ridge:e} std {standardize}");
                        let f = SpdFactor::of(&a, k).expect("a ridge factorizes");
                        let delta = f.jitter();
                        let over = OverMoments { ridge, rows: n };
                        let long = f.carries_no_digit(&a);
                        refused += usize::from(long);
                        if slack_holds(&a, k, ridge, n, delta) {
                            slack += 1;
                            let bound =
                                2.0 * (k as f64).sqrt() * (top(&a, k) + delta) / (ridge + delta);
                            let want = exact(&a, k, delta);
                            let est = f.equilibrated_inverse_norm(&a);
                            assert!(
                                want <= 0.89 * bound * (1.0 + 1e-6),
                                "{what}: {want:e} {bound:e}"
                            );
                            assert!(est <= 0.89 * bound, "{what}: {est:e} {bound:e}");
                            closest = closest.max(want / bound);
                        }
                        if f.ridge_settles(&a, over) {
                            settled += 1;
                            assert!(
                                !long,
                                "{what}: the shortcut settled what the long way refuses"
                            );
                            assert!(exact(&a, k, delta) < digit_limit(k), "{what}");
                        } else {
                            left += 1;
                        }
                        assert_eq!(f.carries_no_digit_over(&a, Some(over)), long, "{what}");
                    }
                }
            }
        }
    }
    eprintln!(
        "settled {settled}, left {left} (refused {refused}), slack held {slack}, \
         the oracle's norm at most {closest:.3} of the bound"
    );
    assert!(
        settled > 300 && left > 50 && refused > 0,
        "{settled} {left} {refused}"
    );
}

/// The two sides of a decision: whether the shortcut settles `a` under
/// `over`, and whether the long way refuses it; and `carries_no_digit_over`
/// must be the long way's either way.
fn sides(a: &[f64], k: usize, over: OverMoments) -> (bool, bool, f64) {
    let f = SpdFactor::of(a, k).expect("factorizes");
    assert_eq!(f.attempts(), 0, "{over:?}: no jitter");
    let long = f.carries_no_digit(a);
    assert_eq!(f.carries_no_digit_over(a, Some(over)), long, "{over:?}");
    (
        f.ridge_settles(a, over),
        long,
        f.equilibrated_inverse_norm(a),
    )
}

/// The ridge at which the bound `2 √k (t + δ)/(ρ + δ)` crosses the limit,
/// to the last bit: the smallest `ρ` it settles, by bisection over the
/// doubles' bits (the bound falls as the ridge grows).
fn edge_ridge(c: &[f64], k: usize, rows: u64, standardize: bool) -> f64 {
    let settles = |rho: f64| {
        let mut a = c.to_vec();
        if standardize {
            let s: Vec<f64> = (0..k).map(|i| c[i * k + i].sqrt()).collect();
            for p in 0..k * k {
                a[p] = c[p] / (s[p / k] * s[p % k]);
            }
        }
        for i in 0..k {
            a[i * k + i] += rho;
        }
        let f = SpdFactor::of(&a, k).expect("factorizes");
        f.ridge_settles(&a, OverMoments { ridge: rho, rows })
    };
    // Around where it crosses, `2 √k t_C / limit` (`t_C` the largest of
    // `C`'s diagonal, 1 standardized): far enough above `C`'s rounding that
    // the factor needs no jitter, which would carry a ridge of its own.
    let t_c = if standardize { 1.0 } else { top(c, k) };
    let near = 2.0 * (k as f64).sqrt() * t_c / digit_limit(k);
    let (mut lo, mut hi) = ((0.25 * near).to_bits(), (4.0 * near).to_bits());
    assert!(!settles(f64::from_bits(lo)) && settles(f64::from_bits(hi)));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if settles(f64::from_bits(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    f64::from_bits(hi)
}

/// At the edge. On a rank-one `C` (two rows of three features), whose
/// `‖H⁻¹‖₁` the ridge alone holds, so that the estimate sits closest to the
/// bound: at the ridge where the bound first falls under the limit the
/// shortcut settles the system and the long way does not refuse it, and
/// one bit of ridge below, where the bound is at or past the limit, the
/// shortcut leaves it to the estimate, which still does not refuse it
/// (it read under half the limit on both sides: the margin of items 4 and
/// 5). The slack's own edge, the rows at which `1.25 ĝ` reaches `(ρ +
/// δ)/2`, likewise: settled at the last count it allows, left one row
/// after. Counting more rows than built the moments only widens the slack,
/// so the proof holds at every count tried. Unscaled and standardized.
#[test]
fn at_the_edge_the_decision_is_the_estimates() {
    let k = 3;
    let mut s = 61u64;
    let mut cov = EwCov::new(k);
    for _ in 0..2 {
        let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
        cov.update(&x, 1.0, 1.0);
    }
    let c = comoments(&cov);
    let rows = cov.rows_learned();
    for standardize in [false, true] {
        let rho = edge_ridge(&c, k, rows, standardize);
        let below = f64::from_bits(rho.to_bits() - 1);
        let limit = digit_limit(k);
        for (ridge, settles) in [(rho, true), (below, false)] {
            let a = system(&cov, k, ridge, standardize).unwrap();
            let (got, long, est) = sides(&a, k, OverMoments { ridge, rows });
            assert_eq!(got, settles, "std {standardize} ridge {ridge:e}");
            assert!(!long, "std {standardize} ridge {ridge:e}: {est:e}");
            assert!(
                est < 0.5 * limit && est > 0.05 * limit,
                "{est:e} of {limit:e}"
            );
            let bound = 2.0 * (k as f64).sqrt() * top(&a, k) / ridge;
            assert_eq!(bound < limit, settles, "{bound:e} {limit:e}");
        }
        // The slack's edge, at a ridge well inside the limit's.
        let ridge = 8.0 * rho;
        let a = system(&cov, k, ridge, standardize).unwrap();
        let holds = |n: u64| sides(&a, k, OverMoments { ridge, rows: n }).0;
        let (mut lo, mut hi) = (rows, 1u64 << 40);
        assert!(holds(lo) && !holds(hi));
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if holds(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!(slack_holds(&a, k, ridge, lo, 0.0) && !slack_holds(&a, k, ridge, hi, 0.0));
        for n in [lo, hi] {
            let (_, long, _) = sides(&a, k, OverMoments { ridge, rows: n });
            assert!(!long, "std {standardize} rows {n}");
        }
    }
}

/// A factor that needed jitter is judged with it, by the shortcut as by
/// the long way. A rank-one `C` with a ridge of `1e-20` of its trace fails
/// to factorize as given, and the first rung's `δ`, `1e-12` of the mean
/// diagonal, is then the effective ridge: at three features the bound
/// counts it and settles the system, as the long way does not refuse it; at
/// forty, past the limit there (a first-rung solve is refused from the
/// thirties, the module docs), the shortcut leaves it, and the long way
/// refuses it.
#[test]
fn a_jittered_factor_is_judged_with_its_jitter() {
    for (k, settles) in [(3usize, true), (40, false)] {
        let mut s = 7u64 + k as u64;
        let mut cov = EwCov::new(k);
        for _ in 0..2 {
            let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
            cov.update(&x, 1.0, 1.0);
        }
        let c = comoments(&cov);
        let ridge = 1e-20 * trace(&c, k);
        let a = system(&cov, k, ridge, false).unwrap();
        let f = SpdFactor::of(&a, k).unwrap();
        assert!(f.attempts() > 0 && f.jitter() > 0.0, "k {k}: needed jitter");
        let over = OverMoments {
            ridge,
            rows: cov.rows_learned(),
        };
        assert_eq!(f.ridge_settles(&a, over), settles, "k {k}");
        assert_eq!(f.carries_no_digit(&a), !settles, "k {k}");
        assert_eq!(f.carries_no_digit_over(&a, Some(over)), !settles, "k {k}");
    }
}

/// Activation at a thousand features, the width the shortcut is for: a
/// Gram of unit-variance features under a half-life of 1,000 rows, its
/// system's diagonal with the default ridge of `1e-6`, read at every 50th
/// of 2,000 rows. The ridge settles every one: the bound reads about
/// `2 √1000 · 1.2 / 1e-6 ≈ 8e7`, under the limit of `1e9`. With the
/// features' spread 100 (variances near `1e4`), it settles none past the
/// first row: `8e11`, past the limit, and the estimate decides, as before.
/// The diagonal alone, since a factorization of this order in a test build
/// takes minutes; the factor's part, its jitter and its columns, the other
/// tests hold.
#[test]
#[ignore = "extended: 2,000 rank-one steps at k = 1,000 in a test build"]
fn a_thousand_features_settle_by_the_ridge_and_spread_ones_do_not() {
    let k = 1000;
    let lam = 0.5f64.powf(1.0 / 1000.0);
    for (scale, settles) in [(1.0, true), (100.0, false)] {
        let mut s = 1000u64;
        let mut cov = EwCov::new(k);
        let mut read = 0;
        for r in 0..2_000 {
            let x: Vec<f64> = (0..k).map(|_| scale * 3f64.sqrt() * draw(&mut s)).collect();
            cov.update(&x, lam, 1.0);
            if r % 50 == 49 {
                let diag = (0..k).map(|i| cov.cov(i, i) + 1e-6);
                let over = OverMoments {
                    ridge: 1e-6,
                    rows: cov.rows_learned(),
                };
                assert_eq!(
                    ridge_holds(diag, k, over, 0.0),
                    settles,
                    "scale {scale} row {r}"
                );
                read += 1;
            }
        }
        assert_eq!(read, 40);
    }
}
