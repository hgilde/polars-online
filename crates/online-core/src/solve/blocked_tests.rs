//! The ridge's shortcut on a blocked Gram (docs/PLAN.md task 308; the
//! module docs, *A blocked Gram's bound*, items 6 to 9): the kept bound's
//! premise, that the merged co-moments are semidefinite to within it,
//! against `crate::oracle`'s eigensolver, on streams built to break a bound
//! read from the matrix; its decision against the oracle's LU inverse and
//! the long way's; and its edges.

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

fn trace(a: &[f64], k: usize) -> f64 {
    (0..k).map(|i| a[i * k + i]).sum()
}

fn top(a: &[f64], k: usize) -> f64 {
    (0..k).map(|i| a[i * k + i]).fold(0.0, f64::max)
}

fn lowest(a: &[f64]) -> f64 {
    let (values, _) = oracle::sym_eigen(a);
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// The design of a blocked test stream's features.
#[derive(Clone, Copy, Debug)]
enum Design {
    /// Independent, uniform on `[−1, 1)` around 3.
    Free,
    /// `x₁ = x₀ + 1e-11 b`: the two equal to within `1e-11`.
    Near,
    /// Fewer rows than features: `C` singular.
    Few,
    /// `Near`, and each block opened by a row of weight `1e-46` at `1e20`,
    /// its first two features equal so that it adds no spread where those
    /// two nearly agree: the row the block's mean is summed about, so the
    /// rounded mean is off by about `1e5` against a spread of 1 and the
    /// merge subtracts a residue of that size -- the stream that broke every
    /// bound read from the matrix (task 306).
    Shifted,
}

/// A blocked stream's accumulator: `design`, `lam` a row, the weights
/// `weights` in turn, blocks of `block` rows, merged at the end.
fn stream(k: usize, design: Design, lam: f64, weights: &[f64], block: usize, seed: u64) -> EwCov {
    let mut s = seed;
    let mut cov = EwCov::new(k);
    cov.set_block_rows(block);
    let rows = match design {
        Design::Few => k / 2 + 1,
        _ => 4 * k + 40 + 3 * block,
    };
    for r in 0..rows {
        let mut x: Vec<f64> = (0..k).map(|_| 3.0 + draw(&mut s)).collect();
        if matches!(design, Design::Near | Design::Shifted) && k > 1 {
            x[1] = x[0] + 1e-11 * draw(&mut s);
        }
        let mut w = weights[r % weights.len()];
        if matches!(design, Design::Shifted) && r % block == 0 {
            x.iter_mut()
                .for_each(|v| *v = 1e20 * (1.0 + 0.01 * draw(&mut s)));
            if k > 1 {
                x[1] = x[0];
            }
            w = 1e-46;
        }
        cov.update(&x, lam, w);
    }
    cov.flush();
    cov
}

/// `C` and its kept bound `e`, as the solve reads them.
fn moments(cov: &EwCov) -> (Vec<f64>, Vec<f64>) {
    let e = cov.merge_rounding().expect("a merged bound").to_vec();
    (cov.comoments().to_full(), e)
}

/// `S⁻¹ C S⁻¹`, `S = diag(√c_ii)`, and `Σ e_i / c_ii`; `None` where a
/// variance is not positive.
fn standardized(c: &[f64], e: &[f64], k: usize) -> Option<(Vec<f64>, f64)> {
    if !(0..k).all(|i| c[i * k + i] > 0.0) {
        return None;
    }
    let s: Vec<f64> = (0..k).map(|i| c[i * k + i].sqrt()).collect();
    let h = (0..k * k).map(|p| c[p] / (s[p / k] * s[p % k])).collect();
    Some((h, (0..k).map(|i| e[i] / c[i * k + i]).sum()))
}

/// Item 6 against the oracle, where a bound read from the matrix fails:
/// on the shifted streams the merged `C`'s smallest eigenvalue is negative
/// by far more than item 1's slack for the same rows, `(2n + 5) u tr C`
/// (measured 2026-11-02: by up to `1.3e-7` against `1e-13`, on 8 of 16), and
/// on every stream, unscaled and standardized, it is within the kept
/// bound: `λ_min ≥ −Σ e_i` and `λ_min(S⁻¹ C S⁻¹) ≥ −Σ e_i / c_ii`.
#[test]
fn the_kept_bound_holds_where_the_rows_do_not() {
    let mut beyond_rows = 0;
    for (cov, what) in streams() {
        let k = cov.k();
        let (c, e) = moments(&cov);
        let low = lowest(&c);
        let kept: f64 = e.iter().sum();
        let rows = (2 * cov.rows_learned() + 5) as f64 * UNIT * trace(&c, k);
        assert!(-low <= kept, "{what}: {low:e} past {kept:e}");
        beyond_rows += usize::from(-low > rows);
        if let Some((h, kept_s)) = standardized(&c, &e, k) {
            let low_s = lowest(&h);
            assert!(-low_s <= kept_s, "{what}: std {low_s:e} past {kept_s:e}");
        }
    }
    // Which merges round which way is the platform's arithmetic's (faer's
    // kernels), so the count is held loosely.
    assert!(beyond_rows >= 3, "{beyond_rows}");
}

/// The test streams: orders 2 to 20, every design, no decay and two,
/// unit, mixed and zero weights, blocks of 5 to 64 rows.
fn streams() -> Vec<(EwCov, String)> {
    let mut out = Vec::new();
    let mut seed = 308u64;
    for k in [2usize, 3, 8, 20] {
        for design in [Design::Free, Design::Near, Design::Few, Design::Shifted] {
            for (lam, weights, block) in [
                (1.0, &[1.0][..], 16usize),
                (0.99, &[1.0][..], 7),
                (1.0, &[0.0, 0.1, 1.0, 5.0, 40.0][..], 64),
                (0.9, &[1.0, 0.0, 1e-6][..], 5),
            ] {
                seed += 1;
                let cov = stream(k, design, lam, weights, block, seed);
                out.push((cov, format!("k {k} {design:?} lam {lam} block {block}")));
            }
        }
    }
    out
}

/// `A` as the centred solve forms it from `C`, with the ridge, and `Ê` as
/// it reads the kept bound: `Σ e_i`, or `Σ e_i / c_ii` standardized (item
/// 9). `None` where standardizing would drop a column.
fn system(
    c: &[f64],
    e: &[f64],
    k: usize,
    ridge: f64,
    standardize: bool,
) -> Option<(Vec<f64>, f64)> {
    let (mut a, e_hat) = if standardize {
        if !(0..k).all(|i| c[i * k + i] > 0.0) {
            return None;
        }
        let s: Vec<f64> = (0..k).map(|i| c[i * k + i].max(0.0).sqrt()).collect();
        let a = (0..k * k).map(|p| c[p] / (s[p / k] * s[p % k])).collect();
        let mut e_hat = 0.0;
        for i in 0..k {
            e_hat += e[i] / c[i * k + i];
        }
        (a, e_hat)
    } else {
        let mut e_hat = 0.0;
        for &v in e {
            e_hat += v;
        }
        (c.to_vec(), e_hat)
    };
    for i in 0..k {
        a[i * k + i] += ridge;
    }
    Some((a, e_hat))
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

/// The slack test with item 9's term, written out: `1.25 ĝ ≤ (ρ + δ)/2`.
fn slack_holds(a: &[f64], k: usize, ridge: f64, e_hat: f64, delta: f64) -> bool {
    let (t, tr, kf) = (top(a, k), trace(a, k), k as f64);
    let g = e_hat
        + 5.0 * UNIT * tr
        + (3.0 * kf + 1.0) * UNIT * (tr + kf * delta)
        + UNIT * (2.0 * t + delta);
    1.25 * g <= 0.5 * (ridge + delta)
}

/// Items 6 to 9 with 3 to 5, against the oracle. Over the test streams,
/// ridges from `1e-16` to `100`, standardized or not: wherever the slack
/// test holds, the oracle's `‖H⁻¹‖₁` and the estimate are both under `0.89`
/// of the bound `2 √k (t + δ)/(ρ + δ)` the shortcut compares with the
/// limit; wherever the shortcut fires, the long way would not refuse; and
/// `carries_no_digit_over` is the long way's on every system. Both
/// outcomes are reached, on the shifted streams too: systems the shortcut
/// settles and systems it leaves to the estimate, some of them refused.
#[test]
fn the_kept_bound_lets_the_ridge_decide_as_the_estimate_would() {
    let (mut settled, mut left, mut refused, mut slack) = (0, 0, 0, 0);
    let mut shifted_settled = 0;
    for (cov, what) in streams() {
        let k = cov.k();
        let (c, e) = moments(&cov);
        for ridge in [1e-16, 1e-14, 1e-12, 1e-9, 1e-6, 1e-3, 1.0, 100.0] {
            for standardize in [false, true] {
                let Some((a, e_hat)) = system(&c, &e, k, ridge, standardize) else {
                    continue;
                };
                let what = format!("{what} ridge {ridge:e} std {standardize}");
                let f = SpdFactor::of(&a, k).expect("a ridge factorizes");
                let delta = f.jitter();
                let over = OverMoments::merged(ridge, e_hat);
                let long = f.carries_no_digit(&a);
                refused += usize::from(long);
                if slack_holds(&a, k, ridge, e_hat, delta) {
                    slack += 1;
                    let bound = 2.0 * (k as f64).sqrt() * (top(&a, k) + delta) / (ridge + delta);
                    let want = exact(&a, k, delta);
                    let est = f.equilibrated_inverse_norm(&a);
                    assert!(
                        want <= 0.89 * bound * (1.0 + 1e-6),
                        "{what}: {want:e} {bound:e}"
                    );
                    assert!(est <= 0.89 * bound, "{what}: {est:e} {bound:e}");
                }
                if f.ridge_settles(&a, over) {
                    settled += 1;
                    shifted_settled += usize::from(what.contains("Shifted"));
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
    eprintln!(
        "settled {settled} (shifted {shifted_settled}), left {left} (refused {refused}), \
         slack held {slack}"
    );
    assert!(
        settled > 300 && shifted_settled > 20 && left > 50 && refused > 0,
        "{settled} {shifted_settled} {left} {refused}"
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

/// At the edges, on a blocked Gram of one merge of two rows over twenty
/// features -- rank one, so that the estimate sits closest to the bound,
/// and wide enough that the limit's ridge clears the merge's slack -- its
/// kept bound `Ê` as read: the ridge at which the bound `2 √k (t + δ)/(ρ +
/// δ)` first falls under the limit, found by bisection over the doubles'
/// bits, is settled, and one bit below it is left to the estimate; and at
/// a ridge eight times that, the largest `Ê` the slack allows is settled
/// and the next double is left. The long way refuses none of them, so the
/// answer is the estimate's on both sides of each edge. Unscaled and
/// standardized.
#[test]
fn at_the_edges_the_decision_is_the_estimates() {
    let k = 20;
    let mut s = 62u64;
    let mut cov = EwCov::new(k);
    cov.set_block_rows(2);
    for _ in 0..2 {
        let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
        cov.update(&x, 1.0, 1.0);
    }
    let (c, e) = moments(&cov);
    for standardize in [false, true] {
        let read = |ridge: f64| system(&c, &e, k, ridge, standardize).unwrap();
        let settles = |ridge: f64| {
            let (a, e_hat) = read(ridge);
            sides(&a, k, OverMoments::merged(ridge, e_hat)).0
        };
        let t_c = if standardize { 1.0 } else { top(&c, k) };
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
        let rho = f64::from_bits(hi);
        let limit = digit_limit(k);
        for (ridge, want) in [(rho, true), (f64::from_bits(hi - 1), false)] {
            let (a, e_hat) = read(ridge);
            let (got, long, est) = sides(&a, k, OverMoments::merged(ridge, e_hat));
            assert_eq!(got, want, "std {standardize} ridge {ridge:e}");
            assert!(!long, "std {standardize} ridge {ridge:e}: {est:e}");
            let bound = 2.0 * (k as f64).sqrt() * top(&a, k) / ridge;
            assert_eq!(bound < limit, want, "{bound:e} {limit:e}");
        }
        // The slack's edge, over `Ê` itself.
        let ridge = 8.0 * rho;
        let (a, e_hat) = read(ridge);
        let holds = |x: f64| sides(&a, k, OverMoments::merged(ridge, x)).0;
        assert!(holds(e_hat) && !holds(ridge));
        let (mut lo, mut hi) = (e_hat.to_bits(), ridge.to_bits());
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if holds(f64::from_bits(mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (last, past) = (f64::from_bits(lo), f64::from_bits(hi));
        assert!(slack_holds(&a, k, ridge, last, 0.0) && !slack_holds(&a, k, ridge, past, 0.0));
        for x in [last, past] {
            let (_, long, _) = sides(&a, k, OverMoments::merged(ridge, x));
            assert!(!long, "std {standardize} e_hat {x:e}");
        }
        // A bound that is not a number, or negative, is none.
        for x in [f64::NAN, f64::INFINITY, -1e-300] {
            assert!(!holds(x), "e_hat {x:e}");
        }
    }
}

/// Where the bound is known (items 7 and 8): a new accumulator's first
/// merge starts it at 0 and it is read once nothing is pending; a held row
/// hides it; a state file does not carry it, and a loaded accumulator's
/// next merge rebuilds it from the matrix, at least each row's largest
/// entry, so that the shortcut waits for the decay to take the loaded
/// history down; setting the moments or switching blocking forgets it.
#[test]
fn the_bound_is_rebuilt_where_it_is_not_known() {
    let k = 3;
    let mut s = 9u64;
    let mut cov = EwCov::new(k);
    cov.set_block_rows(4);
    assert!(cov.merge_rounding().is_none(), "nothing merged yet");
    let mut row = |cov: &mut EwCov| {
        let x: Vec<f64> = (0..k).map(|_| draw(&mut s)).collect();
        cov.update(&x, 0.99, 1.0);
    };
    for _ in 0..8 {
        row(&mut cov);
    }
    let fresh = cov.merge_rounding().expect("known after a merge").to_vec();
    // Eight rows: the bound is a few hundred units in the last place of `C`.
    assert!(fresh.iter().all(|&v| v > 0.0 && v < 1e-12), "{fresh:?}");
    row(&mut cov);
    assert!(cov.merge_rounding().is_none(), "a row pending");
    for _ in 0..3 {
        row(&mut cov);
    }
    assert!(cov.merge_rounding().is_some());

    let bytes = rmp_serde::to_vec_named(&cov).unwrap();
    let mut loaded: EwCov = rmp_serde::from_slice(&bytes).unwrap();
    loaded.set_block_rows(4);
    assert!(loaded.merge_rounding().is_none(), "not in the file");
    let c = loaded.comoments().to_full();
    // A merge's `σ = W_A/W'`, from the weights either side of its block.
    let block = |cov: &mut EwCov, row: &mut dyn FnMut(&mut EwCov)| {
        let w_a = 0.99f64.powi(4) * cov.n_eff();
        for _ in 0..4 {
            row(cov);
        }
        w_a / cov.n_eff()
    };
    let sigma = block(&mut loaded, &mut row);
    let rebuilt = loaded.merge_rounding().unwrap().to_vec();
    for i in 0..k {
        let largest = (0..k).map(|j| c[i * k + j].abs()).fold(0.0, f64::max);
        assert!(rebuilt[i] >= sigma * largest, "{i}: {rebuilt:?} {largest}");
    }
    // The history decays by `σ` a merge, and a merge of four rows adds a
    // few units in the last place of `C`.
    let before = rebuilt[0];
    let sigma = block(&mut loaded, &mut row);
    let after = loaded.merge_rounding().unwrap()[0];
    assert!(
        (after / (sigma * before) - 1.0).abs() < 1e-9,
        "{before} {after}"
    );

    loaded.set_moments(&[0.0; 3], &c, 1.0, None);
    assert!(loaded.merge_rounding().is_none(), "set moments");
    let mut off = cov.clone();
    off.set_block_rows(0);
    assert!(off.merge_rounding().is_none(), "blocking off");
    off.set_block_rows(4);
    assert!(off.merge_rounding().is_none(), "blocking switched");
}

/// Activation at W256's shape, the width a blocked fit is for: a thousand
/// unit-variance features in blocks of 256 under a half-life of 1,000 rows,
/// its system's diagonal with the default ridge of `1e-6` and the kept
/// bound read after every merge of 2,000 rows. The ridge settles every
/// one: the bound reads about `2 √1000 · 1.2 / 1e-6 ≈ 8e7`, under the limit
/// of `1e9`, and `Ê` under `4e-9` (it settles near `7e-9`), against the
/// slack's `4e-7`. With the features' spread 100 it settles none: `8e11`,
/// past the limit. The diagonal alone, as the live Gram's activation test reads it: a
/// factorization of this order in a test build takes minutes.
#[test]
#[ignore = "extended: 2,000 rows merged at k = 1,000 in a test build"]
fn a_thousand_blocked_features_settle_by_the_ridge_and_spread_ones_do_not() {
    let (k, block) = (1000, 256);
    let lam = 0.5f64.powf(1.0 / 1000.0);
    for (scale, settles) in [(1.0, true), (100.0, false)] {
        let mut s = 1000u64;
        let mut cov = EwCov::new(k);
        cov.set_block_rows(block);
        let mut read = 0;
        for r in 0..2_000 {
            let x: Vec<f64> = (0..k).map(|_| scale * 3f64.sqrt() * draw(&mut s)).collect();
            cov.update(&x, lam, 1.0);
            if r % block == block - 1 {
                let e_hat: f64 = cov.merge_rounding().unwrap().iter().sum();
                let diag = (0..k).map(|i| cov.cov(i, i) + 1e-6);
                eprintln!("scale {scale} row {r}: e_hat {e_hat:.3e}");
                assert_eq!(
                    ridge_holds(diag, k, OverMoments::merged(1e-6, e_hat), 0.0),
                    settles,
                    "scale {scale} row {r}"
                );
                read += 1;
            }
        }
        assert_eq!(read, 7);
    }
}
