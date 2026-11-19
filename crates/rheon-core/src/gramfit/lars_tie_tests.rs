//! LARS on exact ties (docs/PLAN.md task 236): every column whose step
//! ties the smallest, to within rounding, is settled at that one knot,
//! entering or leaving as the lasso admits.
//! Each case is held to its closed-form path or to faer's LU
//! (`crate::oracle`), which shares no arithmetic with the path's factor,
//! and to the lasso's conditions at every knot. scikit-learn is no second
//! opinion here: its `lars_path_gram` (1.9.1) ends the first case below at
//! `[2, 1]`, the least-squares fit being `[1, 1]`.

use super::tests::{
    assert_kkt, design, knot_error, mixed_design, path_with_a_drop, residual_bounds,
};
use super::*;
use crate::oracle;

fn identity(k: usize) -> Vec<f64> {
    (0..k * k)
        .map(|ij| if ij / k == ij % k { 1.0 } else { 0.0 })
        .collect()
}

/// `(1 − ρ) I + ρ 11ᵀ`, `k × k`.
fn equicorrelated(k: usize, rho: f64) -> Vec<f64> {
    (0..k * k)
        .map(|ij| if ij / k == ij % k { 1.0 } else { rho })
        .collect()
}

/// The square blocks on the diagonal, 0 elsewhere.
fn block_diagonal(blocks: &[&[f64]]) -> Vec<f64> {
    let sizes: Vec<usize> = blocks.iter().map(|b| b.len().isqrt()).collect();
    let k: usize = sizes.iter().sum();
    let mut r = vec![0.0; k * k];
    let mut at = 0;
    for (block, &m) in blocks.iter().zip(&sizes) {
        for i in 0..m {
            for j in 0..m {
                r[(at + i) * k + at + j] = block[i * m + j];
            }
        }
        at += m;
    }
    r
}

/// The lasso's conditions at every knot.
fn assert_kkt_everywhere(r: &[f64], d: &[f64], path: &LarsPath, tol: f64) {
    let k = d.len();
    for (n, &lambda) in path.penalties.iter().enumerate() {
        assert_kkt(r, d, &path.coefs[n * k..(n + 1) * k], lambda, tol);
    }
}

/// The path's last knot is least squares on the columns `keep` (faer's
/// LU), and 0 on the rest: within `tol` where a case gives one, and
/// otherwise within the residual the path's steps and ties leave, through
/// `R_kk⁻¹` ([`super::tests::knot_error`] of
/// [`super::tests::residual_bounds`], docs/PLAN.md task 274). That bound
/// allows every knot where several columns change the tie's 64 ulps of
/// `λ_max`; a case built on an exact tie, or whose offset from the tie is
/// worked out by hand, keeps the tighter tolerance it states.
fn assert_ends_at_least_squares(
    r: &[f64],
    d: &[f64],
    path: &LarsPath,
    keep: &[usize],
    tol: Option<f64>,
) {
    let k = d.len();
    let rk: Vec<f64> = keep
        .iter()
        .flat_map(|&i| keep.iter().map(move |&j| r[i * k + j]))
        .collect();
    let dk: Vec<f64> = keep.iter().map(|&i| d[i]).collect();
    let ls = oracle::solve(&rk, &dk);
    let last = &path.coefs[(path.len() - 1) * k..];
    assert_eq!(*path.penalties.last().unwrap(), 0.0);
    let mut active = path.active.last().unwrap().clone();
    active.sort_unstable();
    assert_eq!(active, keep, "the last knot's columns");
    let res = residual_bounds(r, d, path);
    let bound = knot_error(r, keep, res.last().unwrap(), &dk);
    for (j, &got) in last.iter().enumerate() {
        let at = keep.iter().position(|&i| i == j);
        let want = at.map_or(0.0, |p| ls[p]);
        let tol = match (tol, at) {
            (Some(t), _) => t,
            (None, Some(p)) => bound[p],
            (None, None) => 0.0,
        };
        oracle::assert_within(got, want, tol, format_args!("column {j}, least squares"));
    }
}

/// `R = I`: each coefficient is `d_j` soft-thresholded at `λ`,
/// `sign(d_j) max(|d_j| − λ, 0)`, with a knot at each distinct `|d_j|` --
/// every number here exact in binary. Columns tied in `|d_j|` enter at one
/// knot; before task 236 the second of a tie had a step of exactly 0, which
/// the entry test passed over for good, so `d = [1, 1]` ended at `[1, 0]`.
#[test]
fn an_orthogonal_design_is_soft_thresholding_with_tied_columns_at_one_knot() {
    let path = lars_lasso(&identity(2), &[1.0, 1.0], &[true; 2], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 0.0]);
    assert_eq!(path.coefs, vec![0.0, 0.0, 1.0, 1.0]);
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1]]);
    assert_eq!(path.stop, LarsStop::End);
    // Ties of both signs at λ = 3, and a tie at λ = 2 below the first knot.
    let d = [3.0, -3.0, 2.0, 2.0, -1.0, 0.5];
    let r = identity(6);
    let path = lars_lasso(&r, &d, &[true; 6], LarsLimits::default());
    assert_eq!(path.penalties, vec![3.0, 2.0, 1.0, 0.5, 0.0]);
    for (n, &lambda) in path.penalties.iter().enumerate() {
        let soft: Vec<f64> = d
            .iter()
            .map(|&v: &f64| v.signum() * (v.abs() - lambda).max(0.0))
            .collect();
        assert_eq!(path.coefs[n * 6..(n + 1) * 6], soft[..], "knot {n}");
    }
    assert_kkt_everywhere(&r, &d, &path, 0.0);
    let all: Vec<usize> = (0..6).collect();
    assert_eq!(
        path.active,
        vec![
            vec![0, 1],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3, 4],
            all.clone(),
            all
        ]
    );
    // A diagonal not of ones: b_j = soft(d_j, λ) / R_jj.
    let r = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0];
    let d = [2.0, -2.0, 1.0];
    let path = lars_lasso(&r, &d, &[true; 3], LarsLimits::default());
    assert_eq!(path.penalties, vec![2.0, 1.0, 0.0]);
    assert_eq!(
        path.coefs,
        vec![0.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.5, -0.5, 1.0]
    );
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1, 2], vec![0, 1, 2]]);
    assert_kkt_everywhere(&r, &d, &path, 0.0);
}

/// `R = (1 − ρ) I + ρ 11ᵀ` with every `|d_j|` equal: all columns tie at the
/// first knot. With one sign the path is `b(λ) = (c − λ) / (1 + (k − 1) ρ)`
/// on each, to least squares at 0. With mixed signs at `ρ = −0.4` the
/// direction with all three active, `R⁻¹ s = [3, 3, 1] / 1.4` for
/// `s = [1, 1, −1]`, would move column 2 from 0 against its sign: the
/// lasso leaves it out of the first knot, and its correlation runs from −1
/// through 0 to meet `λ` at 1/7 with the other sign. Then `b = [10, 10, 0]
/// / 7`, and least squares `[15, 15, 5] / 7` at the end.
#[test]
fn an_equicorrelated_design_with_equal_correlations_has_its_closed_form_path() {
    let (k, rho, c) = (5, 0.3, 0.6);
    let r = equicorrelated(k, rho);
    let d = vec![c; k];
    let path = lars_lasso(&r, &d, &vec![true; k], LarsLimits::default());
    assert_eq!(path.penalties, vec![c, 0.0]);
    let all: Vec<usize> = (0..k).collect();
    assert_eq!(path.active, vec![all.clone(), all.clone()]);
    assert!(path.coefs[..k].iter().all(|&v| v == 0.0));
    let want = c / (1.0 + (k - 1) as f64 * rho);
    for &got in &path.coefs[k..] {
        assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
    }
    assert_kkt_everywhere(&r, &d, &path, 1e-15);
    // Mixed signs at ρ = 0.3: R⁻¹ s keeps every sign, so all five enter.
    let d: Vec<f64> = [1.0, -1.0, 1.0, 1.0, -1.0].iter().map(|s| s * c).collect();
    let path = lars_lasso(&r, &d, &vec![true; k], LarsLimits::default());
    assert_eq!(path.penalties, vec![c, 0.0]);
    assert_eq!(path.active, vec![all.clone(), all.clone()]);
    // An exact tie: the end to 4.5 ulps of its largest coefficient, where
    // the derived bound (6.2e-14) carries the tie at its first knot.
    assert_ends_at_least_squares(&r, &d, &path, &all, Some(1e-15));
    assert_kkt_everywhere(&r, &d, &path, 1e-15);
    // Mixed signs at ρ = −0.4: one of the three tied columns stays out.
    let r = equicorrelated(3, -0.4);
    let d = [1.0, 1.0, -1.0];
    let path = lars_lasso(&r, &d, &[true; 3], LarsLimits::default());
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1, 2], vec![0, 1, 2]]);
    let pen = [1.0, 1.0 / 7.0, 0.0];
    let coefs = [
        0.0,
        0.0,
        0.0,
        10.0 / 7.0,
        10.0 / 7.0,
        0.0,
        15.0 / 7.0,
        15.0 / 7.0,
        5.0 / 7.0,
    ];
    assert_eq!(path.len(), 3);
    for (got, want) in path.penalties.iter().zip(pen) {
        assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
    }
    for (got, want) in path.coefs.iter().zip(coefs) {
        assert!((got - want).abs() <= 1e-14, "{got} vs {want}");
    }
    assert_eq!(path.coefs[5], 0.0, "column 2 is 0 at the knot it enters");
    assert_kkt_everywhere(&r, &d, &path, 1e-14);
}

/// Two copies of a design with a drop, side by side and uncorrelated: the
/// lasso decouples, so the path is the one block's, at its knots, with each
/// event made twice. Both copies of the dropping column reach 0 at one
/// step, and leave at one knot, exactly 0 there.
#[test]
fn two_coefficients_reaching_zero_at_one_step_leave_at_one_knot() {
    let (r1, d1, one) = path_with_a_drop();
    let k = 6;
    let r = block_diagonal(&[&r1, &r1]);
    let d: Vec<f64> = d1.iter().chain(&d1).copied().collect();
    let path = lars_lasso(&r, &d, &[true; 12], LarsLimits::default());
    assert_eq!(path.len(), one.len(), "{:?}", path.penalties);
    // The two copies' steps part in the last bits, as the factor reads them
    // in other orders, and each knot is at the smaller of a pair.
    let scale = one.penalties[0];
    for n in 0..one.len() {
        assert!(
            (path.penalties[n] - one.penalties[n]).abs() <= lars::TIE * scale,
            "knot {n}: {} vs {}",
            path.penalties[n],
            one.penalties[n]
        );
        let twice: Vec<usize> = one.active[n].iter().flat_map(|&j| [j, j + k]).collect();
        assert_eq!(path.active[n], twice, "knot {n}");
        for j in 0..k {
            let want = one.coefs[n * k + j];
            for copy in [j, j + k] {
                let got = path.coefs[n * 2 * k + copy];
                assert!(
                    (got - want).abs() <= 1e-13,
                    "knot {n}, column {copy}: {got} vs {want}"
                );
            }
        }
    }
    let drop = path
        .active
        .windows(2)
        .position(|w| w[1].len() + 2 == w[0].len())
        .expect("a knot where two columns leave")
        + 1;
    let gone: Vec<usize> = path.active[drop - 1]
        .iter()
        .copied()
        .filter(|j| !path.active[drop].contains(j))
        .collect();
    assert_eq!(gone.len(), 2);
    assert_eq!(gone[1], gone[0] + k, "the two copies of one column");
    for &j in &gone {
        assert_eq!(path.coefs[drop * 2 * k + j], 0.0, "it leaves at exactly 0");
    }
    assert_kkt_everywhere(&r, &d, &path, 1e-12);
    // The derived bound (task 274), 2.2e-12, where the case gave `1e-10`:
    // its error, 1.6e-15, is 7.0e-4 of it on aarch64 (macOS), the two
    // columns leaving at one knot allowed the whole tie.
    assert_ends_at_least_squares(&r, &d, &path, &(0..2 * k).collect::<Vec<_>>(), None);
}

/// A seventh column, uncorrelated with the six of a design with a drop,
/// whose correlation is the penalty at the drop: it enters at the knot
/// where the dropping column leaves, both at one knot. The six's path is
/// unchanged, and column 6's coefficient is `max(d_6 − λ, 0)`. Its step is
/// `λ − d_6` and the drop's `−b_j / δ_j`, two formulas that agree to
/// within rounding, not to the bit.
#[test]
fn an_entry_and_an_exit_at_one_step_are_one_knot() {
    let (r1, d1, one) = path_with_a_drop();
    let at = one
        .active
        .windows(2)
        .position(|w| w[1].len() < w[0].len())
        .unwrap()
        + 1;
    let lambda_drop = one.penalties[at];
    let r = block_diagonal(&[&r1, &[1.0]]);
    let d: Vec<f64> = d1.iter().copied().chain([lambda_drop]).collect();
    let path = lars_lasso(&r, &d, &[true; 7], LarsLimits::default());
    assert_eq!(path.len(), one.len(), "{:?}", path.penalties);
    let scale = one.penalties[0];
    for n in 0..one.len() {
        assert!(
            (path.penalties[n] - one.penalties[n]).abs() <= 1e-15 * scale,
            "knot {n}: {} vs {}",
            path.penalties[n],
            one.penalties[n]
        );
        for j in 0..6 {
            let (got, want) = (path.coefs[n * 7 + j], one.coefs[n * 6 + j]);
            assert!(
                (got - want).abs() <= 1e-13,
                "knot {n}, column {j}: {got} vs {want}"
            );
        }
        let want = (lambda_drop - path.penalties[n]).max(0.0);
        assert!(
            (path.coefs[n * 7 + 6] - want).abs() <= 1e-15 * scale,
            "knot {n}"
        );
    }
    let (before, after) = (&path.active[at - 1], &path.active[at]);
    assert!(
        after.contains(&6) && !before.contains(&6),
        "{before:?} -> {after:?}"
    );
    let gone: Vec<&usize> = before.iter().filter(|j| !after.contains(j)).collect();
    assert_eq!(gone.len(), 1, "{before:?} -> {after:?}");
    assert_eq!(path.coefs[at * 7 + gone[0]], 0.0);
    assert_kkt_everywhere(&r, &d, &path, 1e-12);
    // The derived bound (task 274), 6.5e-14, where the case gave `1e-10`:
    // its error is 0.017 of it on aarch64 (macOS).
    assert_ends_at_least_squares(&r, &d, &path, &(0..7).collect::<Vec<_>>(), None);
}

/// Columns 1 and 2 are symmetric about column 0 (`R_01 = R_02 = ½`,
/// `R_12 = 0`, `d_1 = d_2 = 0.3`), so their correlations meet `λ` together,
/// at `2/15` with sign −1, once column 0 has run alone from `λ = 1`. Then
/// `b = [13/15, 0, 0]`, and least squares `[1.4, −0.4, −0.4]` at the end.
/// Before task 236 column 1 entered alone, and column 2's step at the next
/// knot, 0 by the symmetry, came out `−2.8e-17`: `λ` is carried by
/// subtraction and `c` formed afresh, so they part in the last bit. A test
/// for a positive step passed it over, and the path ended at `[1.13, −0.27,
/// 0]`.
#[test]
fn a_tie_whose_step_rounds_below_zero_still_enters() {
    let r = [1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0];
    let d = [1.0, 0.3, 0.3];
    let path = lars_lasso(&r, &d, &[true; 3], LarsLimits::default());
    assert_eq!(path.active, vec![vec![0], vec![0, 1, 2], vec![0, 1, 2]]);
    let pen = [1.0, 2.0 / 15.0, 0.0];
    let coefs = [0.0, 0.0, 0.0, 13.0 / 15.0, 0.0, 0.0, 1.4, -0.4, -0.4];
    assert_eq!(path.len(), 3);
    for (got, want) in path.penalties.iter().zip(pen) {
        assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
    }
    for (got, want) in path.coefs.iter().zip(coefs) {
        assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
    }
    assert_kkt_everywhere(&r, &d, &path, 1e-15);
}

/// Columns 0 and 1 are one column twice, as are 3 and 4, and column 2 is
/// uncorrelated with all of them. Each pair ties; the first of it enters,
/// and the collinearity check sets the second aside, so the fit stays
/// determined: least squares on columns 0, 2 and 3, with 1 and 4 at 0.
/// Column 2 ties the first pair at `λ = 1`, which before task 236 it never
/// entered from.
#[test]
fn duplicate_columns_tie_and_the_collinearity_check_sets_one_aside() {
    #[rustfmt::skip]
    let r = [
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
    ];
    let d = [1.0, 1.0, 1.0, 0.5, 0.5];
    let path = lars_lasso(&r, &d, &[true; 5], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 0.5, 0.0]);
    assert_eq!(path.active, vec![vec![0, 2], vec![0, 2, 3], vec![0, 2, 3]]);
    #[rustfmt::skip]
    let coefs = vec![
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.5, 0.0, 0.5, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.5, 0.0,
    ];
    assert_eq!(path.coefs, coefs);
    assert_kkt_everywhere(&r, &d, &path, 0.0);
    assert_ends_at_least_squares(&r, &d, &path, &[0, 2, 3], Some(0.0));
}

/// `max_steps` counts knots, each at its own penalty, however many columns
/// change at one; `max_active` stops at the first knot whose active set
/// reaches it, which a tie can carry past it.
#[test]
fn max_steps_counts_knots_and_max_active_stops_where_a_tie_passes_it() {
    let d = [3.0, -3.0, 2.0, 2.0, -1.0, 0.5];
    let r = identity(6);
    let run = |max_steps, max_active| {
        lars_lasso(
            &r,
            &d,
            &[true; 6],
            LarsLimits {
                max_steps,
                max_active,
            },
        )
    };
    let p = run(None, Some(1));
    assert_eq!(p.stop, LarsStop::MaxActive);
    assert_eq!((p.penalties, p.active), (vec![3.0], vec![vec![0, 1]]));
    let p = run(None, Some(3));
    assert_eq!(p.stop, LarsStop::MaxActive);
    assert_eq!(p.penalties, vec![3.0, 2.0]);
    assert_eq!(p.active.last().unwrap().len(), 4);
    let p = run(Some(1), None);
    assert_eq!(p.stop, LarsStop::MaxSteps);
    assert_eq!(p.penalties, vec![3.0, 2.0]);
    let p = run(Some(3), None);
    assert_eq!(p.penalties, vec![3.0, 2.0, 1.0, 0.5]);
}

/// On designs with no tie each knot is one column entering or leaving, at
/// penalties that fall strictly: the tolerance that reads ties merges no
/// two of their events.
#[test]
fn random_designs_change_one_column_at_each_knot() {
    let mut knots = 0;
    for seed in 0..40 {
        let (r, d) = if seed % 2 == 0 {
            design(seed, 120, 12, 0.6, &[1.0, -0.8, 0.6, 0.0, 0.3])
        } else {
            mixed_design(seed, 150, 10)
        };
        let k = d.len();
        let path = lars_lasso(&r, &d, &vec![true; k], LarsLimits::default());
        assert_eq!(path.active[0].len(), 1, "seed {seed}");
        let last = path.len() - 1;
        for n in 1..last {
            let changed = path.active[n]
                .iter()
                .filter(|j| !path.active[n - 1].contains(j))
                .count()
                + path.active[n - 1]
                    .iter()
                    .filter(|j| !path.active[n].contains(j))
                    .count();
            assert_eq!(changed, 1, "seed {seed}, knot {n}");
            knots += 1;
        }
        assert!(
            path.penalties.windows(2).all(|w| w[1] < w[0]),
            "seed {seed}"
        );
    }
    assert!(knots > 400, "{knots}");
}

/// The tolerance, inclusive, on `R = I` with `λ_max = 1`, where every step
/// is a difference of `|d_j|` and exact: `TIE = 2⁻⁴⁶` and `0.5 − 2⁻⁴⁶` is a
/// double. A column `TIE` from `λ_max` joins the first knot, and a step
/// `TIE` past the smallest joins its knot; twice that is a knot of its own.
#[test]
fn a_step_within_tie_of_the_smallest_joins_its_knot_and_one_past_it_does_not() {
    assert_eq!(lars::TIE, 2f64.powi(-46));
    let r = identity(3);
    let run = |d: &[f64]| lars_lasso(&r, d, &[true; 3], LarsLimits::default());
    let p = run(&[1.0, 1.0 - lars::TIE, 0.25]);
    assert_eq!(p.active[0], vec![0, 1]);
    assert_eq!(p.penalties, vec![1.0, 0.25, 0.0]);
    let p = run(&[1.0, 1.0 - 2.0 * lars::TIE, 0.25]);
    assert_eq!(p.active[..2], [vec![0], vec![0, 1]]);
    assert_eq!(p.penalties, vec![1.0, 1.0 - 2.0 * lars::TIE, 0.25, 0.0]);
    let p = run(&[1.0, 0.5, 0.5 - lars::TIE]);
    assert_eq!(p.penalties, vec![1.0, 0.5, 0.0]);
    assert_eq!(p.active[1], vec![0, 1, 2]);
    let p = run(&[1.0, 0.5, 0.5 - 2.0 * lars::TIE]);
    assert_eq!(p.penalties, vec![1.0, 0.5, 0.5 - 2.0 * lars::TIE, 0.0]);
}

/// A step within `TIE` of 0 is the knot just made, which its column joins:
/// the knot is recorded again rather than a second one made a hair below
/// it. Column 1's correlation starts `1.25 TIE` short of `λ_max`, past the
/// first knot's tolerance, but it closes on `λ` at 1.5 times the speed `λ`
/// falls, `1 − a_1 = 1.5` with `R_01 = −½`, so its step is `0.83 TIE`.
#[test]
fn a_step_within_tie_of_zero_joins_the_knot_just_made() {
    let r = [1.0, -0.5, -0.5, 1.0];
    let d = [1.0, 1.0 - 1.25 * lars::TIE];
    let path = lars_lasso(&r, &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 0.0]);
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1]]);
    assert_eq!(path.coefs[..2], [0.0, 0.0]);
    // Column 1 entered `1.25 TIE` short of the active correlation and stays
    // that far off it, which moves the end from least squares by
    // `R⁻¹ (1.25 TIE)`: the size of the tie itself.
    assert_kkt_everywhere(&r, &d, &path, 2.0 * lars::TIE);
    // Worked by hand above: `R⁻¹(1.25 TIE)`, at most 1.67 TIE, where the
    // derived bound reads the joining step's offset as `1 + |R_jA||δ|`
    // times the tie, 1.5.
    assert_ends_at_least_squares(&r, &d, &path, &[0, 1], Some(2.0 * lars::TIE));
}

/// Every column ties at `λ_max` (`|d_j| = 1`, random signs) on correlation
/// matrices with correlations of both signs. The first knot's active set is
/// the one subset `P` the lasso admits, `s_j δ_j > 0` on `P` and
/// `s_j a_j > 1` off it for `δ = R_PP⁻¹ s_P`, found by trying every subset
/// with faer's LU. Most of these leave a tied column out, and some (seeds 1
/// and 10 among them) need a column put back after another has left: both
/// of Murty's flips. The conditions hold at every knot to the rounding of
/// `c`'s terms, which the larger least-squares fits here make up to `1e-9`.
#[test]
fn the_columns_a_tie_admits_are_the_one_subset_the_lasso_allows() {
    let mut partial = 0;
    for seed in 0..300u64 {
        let k = 3 + (seed % 4) as usize;
        let (r, _) = mixed_design(seed, 40, k);
        let r = &r[..];
        let mut rng = crate::SplitMix64::new(seed ^ 0x5157);
        let d: Vec<f64> = (0..k)
            .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        let admits = |p: &[usize]| {
            let rpp: Vec<f64> = p
                .iter()
                .flat_map(|&i| p.iter().map(move |&j| r[i * k + j]))
                .collect();
            let sp: Vec<f64> = p.iter().map(|&j| d[j]).collect();
            let delta = oracle::solve(&rpp, &sp);
            p.iter().zip(&delta).all(|(&j, &v)| d[j] * v > 0.0)
                && (0..k).filter(|j| !p.contains(j)).all(|j| {
                    let a: f64 = p.iter().zip(&delta).map(|(&i, &v)| r[j * k + i] * v).sum();
                    d[j] * a > 1.0
                })
        };
        let allowed: Vec<Vec<usize>> = (1u32..1 << k)
            .map(|mask| (0..k).filter(|&j| mask >> j & 1 == 1).collect::<Vec<_>>())
            .filter(|p| admits(p))
            .collect();
        assert_eq!(allowed.len(), 1, "seed {seed}: {allowed:?}");
        let path = lars_lasso(r, &d, &vec![true; k], LarsLimits::default());
        let mut first = path.active[0].clone();
        first.sort_unstable();
        assert_eq!(first, allowed[0], "seed {seed}");
        partial += usize::from(first.len() < k);
        for (n, &lambda) in path.penalties.iter().enumerate() {
            let b = &path.coefs[n * k..(n + 1) * k];
            let terms = (0..k)
                .map(|j| (0..k).map(|i| (r[j * k + i] * b[i]).abs()).sum::<f64>())
                .fold(1.0f64, f64::max);
            assert_kkt(r, &d, b, lambda, 64.0 * f64::EPSILON * terms);
        }
    }
    assert!(partial > 150, "{partial} of 300 leave a tied column out");
}

/// Two copies of a four-column block in disjoint rows, standardized from
/// the rows: column j and column j + 4 tie before rounding, and the Gram's
/// own sums, over the rows in other orders, part them by a few ulps. Near
/// the end of the path columns 2 and 6 leave at one knot, their steps 7.2
/// ulps of `λ_max` apart. Their coefficients fall 1.6e7 times as fast as
/// `λ`, so the later one, set to 0 at the earlier step, was moved by
/// `6.1e-8`. Solving the columns that stay afresh at the knot keeps the
/// conditions to `1.2e-14`; without it they would be off by that `6.1e-8`
/// from there (measured as `4.5e-8`, on steps 5.3 ulps apart, on the
/// stream before task 273).
///
/// The stream's `ln` and `cos` are the `libm` crate's, the same bits on
/// every platform (docs/PLAN.md task 273): the tie, the knot and the steps'
/// ulps all turn on the stream's last bits, and with the platform's `ln`
/// and `cos` moved by an ulp the Gram's sums left columns 0 and 4 tied.
#[test]
fn columns_merged_at_a_knot_leave_the_conditions_to_rounding() {
    let (n, m, seed) = (40, 4, 113);
    let mut rng = crate::SplitMix64::new(seed ^ 0x51ab);
    let mut normal = || {
        let u = 1.0 - rng.uniform();
        let v = rng.uniform();
        (-2.0 * libm::log(u)).sqrt() * libm::cos(std::f64::consts::TAU * v)
    };
    let mix: Vec<f64> = (0..m * m)
        .map(|ij| normal() + if ij / m == ij % m { 1.0 } else { 0.0 })
        .collect();
    let beta: Vec<f64> = (0..m).map(|_| normal()).collect();
    let k = 2 * m;
    let mut x = vec![0.0; 2 * n * k];
    let mut y = vec![0.0; 2 * n];
    for i in 0..n {
        let z: Vec<f64> = (0..m).map(|_| normal()).collect();
        let row: Vec<f64> = (0..m)
            .map(|j| (0..m).map(|q| z[q] * mix[q * m + j]).sum())
            .collect();
        let yi = (0..m).map(|j| beta[j] * row[j]).sum::<f64>() + normal();
        for j in 0..m {
            x[i * k + j] = row[j];
            x[(n + i) * k + m + j] = row[j];
        }
        y[i] = yi;
        y[n + i] = yi;
    }
    let (r, d) = super::tests::standardized(&x, &y, 2 * n, k);
    assert_ne!(d[0], d[m], "the Gram's rounding parts the tie");
    let path = lars_lasso(&r, &d, &vec![true; k], LarsLimits::default());
    let two_leave = path
        .active
        .windows(2)
        .any(|w| w[0].iter().filter(|j| !w[1].contains(j)).count() == 2);
    assert!(two_leave, "{:?}", path.active);
    assert_kkt_everywhere(&r, &d, &path, 1e-13);
    // The two steps 7.2 ulps apart leave through `|δ| = 1.6e7`: the derived
    // bound, which allows them the whole tie of 64, is 3.4e-5; the case's
    // own tolerance is the measured move, 1.0e-7, ten times over.
    assert_ends_at_least_squares(&r, &d, &path, &(0..k).collect::<Vec<_>>(), Some(1e-6));
}

/// Penalty weights tie columns in the scaled problem: on `R = I` with
/// `d = [2, 1]` and `w = [2, 1]`, `d_j / w_j` is 1 for both, so both enter
/// at `λ = 1`, and `b(λ) = [2, 1] (1 − λ)` below it, to least squares.
#[test]
fn penalty_weights_that_tie_columns_enter_them_at_one_knot() {
    let r = identity(2);
    let path = lars_lasso_weighted(
        &DenseRows { r: &r, k: 2 },
        &[2.0, 1.0],
        &[true; 2],
        &[2.0, 1.0],
        LarsLimits::default(),
    )
    .unwrap();
    assert_eq!(path.penalties, vec![1.0, 0.0]);
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1]]);
    assert_eq!(path.coefs, vec![0.0, 0.0, 2.0, 1.0]);
}

/// A coefficient the direction holds still neither moves nor leaves. On
/// `R = [[1, ½], [½, ½]]` with `d = [1, ¾]`, column 0 runs alone to
/// `λ = ½`, where column 1 enters. Then `δ = R⁻¹ [1, 1] = [0, 2]`, exactly 0
/// for column 0 in the factor's binary arithmetic, so `b_0` stays at ½ to
/// least squares `[½, 1]`. A coefficient moving toward 0 at a speed of 0
/// would reach it at no step.
#[test]
fn a_coefficient_the_direction_holds_still_stays() {
    let r = [1.0, 0.5, 0.5, 0.5];
    let d = [1.0, 0.75];
    let path = lars_lasso(&r, &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 0.5, 0.0]);
    assert_eq!(path.active, vec![vec![0], vec![0, 1], vec![0, 1]]);
    assert_eq!(path.coefs, vec![0.0, 0.0, 0.5, 0.0, 0.5, 1.0]);
    assert_kkt_everywhere(&r, &d, &path, 0.0);
}

/// A step within `TIE` of `λ` itself is the path's end: on `R = I` with
/// `d = [1, TIE / 2]`, column 1 would enter at `λ = TIE / 2`, which is 0
/// to within rounding, so the path ends at the least-squares fit on column
/// 0 alone, `TIE / 2` from the fit on both.
#[test]
fn a_step_within_tie_of_the_end_ends_the_path() {
    let d = [1.0, 0.5 * lars::TIE];
    let path = lars_lasso(&identity(2), &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 0.0]);
    assert_eq!(path.active, vec![vec![0], vec![0]]);
    assert_eq!(path.coefs, vec![0.0, 0.0, 1.0, 0.0]);
    let d = [1.0, 2.0 * lars::TIE];
    let path = lars_lasso(&identity(2), &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.penalties, vec![1.0, 2.0 * lars::TIE, 0.0]);
}

/// A column joining the knot just made is settled with every column of
/// that knot, not alone. Columns 0 and 1 tie at `λ_max = 1` and enter;
/// column 3, `1.5 TIE` short of it, closes on `λ` fast enough that its step
/// is under `TIE`, and joins the first knot. With it, the direction moves
/// column 1 against its sign, so the lasso leaves column 1 out until its
/// correlation meets `λ` again further down. `R` is in multiples of 1/256,
/// so no library's rounding moves a step across the tolerance.
#[test]
fn a_column_joining_a_knot_is_settled_with_the_columns_already_there() {
    #[rustfmt::skip]
    let r = [
        1.0, 0.16796875, -0.19140625, 0.8046875,
        0.16796875, 1.0, -0.47265625, -0.12109375,
        -0.19140625, -0.47265625, 1.0, 0.31640625,
        0.8046875, -0.12109375, 0.31640625, 1.0,
    ];
    let d = [-1.0, -1.0, 1.0 - 2.0 * lars::TIE, 1.0 - 1.5 * lars::TIE];
    let path = lars_lasso(&r, &d, &[true; 4], LarsLimits::default());
    assert_eq!(
        path.active,
        vec![
            vec![0, 3],
            vec![0, 3, 2],
            vec![0, 3, 2, 1],
            vec![0, 3, 2, 1]
        ]
    );
    assert_kkt_everywhere(&r, &d, &path, 1e-13);
    // Column 3 joined `1.5 TIE` short of the active correlation, which the
    // least-squares fit, of coefficients near 11, magnifies to `1.6e-13`.
    // The derived bound, which allows the joining column the whole tie, is
    // 1.8e-12; the case's own tolerance is five times its measured 1.8e-13.
    assert_ends_at_least_squares(&r, &d, &path, &[0, 1, 2, 3], Some(1e-12));
}

/// Two columns tie below the first knot, and the lasso admits one. On
/// `R = 0.6 I + (−0.4) 11ᵀ` with `d = [1, 0.3, −0.7]`, column 0 runs alone
/// to `λ = ½`, where column 1 meets it with sign +1 and column 2 with −1.
/// With both, the direction moves column 2 against its sign, so it stays
/// out until its correlation meets `λ` with sign +1 at 1/14. Then
/// `b = [17, 10, 0] / 14`, and least squares `[22, 15, 5] / 14` at the end.
#[test]
fn two_columns_tying_below_the_first_knot_can_admit_one() {
    let r = equicorrelated(3, -0.4);
    let d = [1.0, 0.3, -0.7];
    let path = lars_lasso(&r, &d, &[true; 3], LarsLimits::default());
    assert_eq!(
        path.active,
        vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2]]
    );
    let pen = [1.0, 0.5, 1.0 / 14.0, 0.0];
    #[rustfmt::skip]
    let coefs = [
        0.0, 0.0, 0.0,
        0.5, 0.0, 0.0,
        17.0 / 14.0, 10.0 / 14.0, 0.0,
        22.0 / 14.0, 15.0 / 14.0, 5.0 / 14.0,
    ];
    assert_eq!(path.len(), 4);
    for (got, want) in path.penalties.iter().zip(pen) {
        assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
    }
    for (got, want) in path.coefs.iter().zip(coefs) {
        assert!((got - want).abs() <= 1e-14, "{got} vs {want}");
    }
    assert_kkt_everywhere(&r, &d, &path, 1e-14);
}

/// A tied column the direction holds at 0 is active: on `R = [[1, ½],
/// [½, ½]]` with `d = [1, 1]` both tie at `λ = 1`, and `δ = [0, 2]`, so
/// `b_0` stays 0 to least squares `[0, 2]` while column 0 stays in the set.
/// One the direction moves against its sign by less than `TIE` of the
/// direction's size is active too: shrinking `R_11` by `2⁻⁵⁰` makes
/// `δ_0` about `−2⁻⁴⁸` beside `δ_1 = 2`. The whole problem is scaled by
/// `2⁻²⁰` there, which scales `δ` by `2²⁰`, so the test reads the
/// tolerance against the direction's size and not against 1.
#[test]
fn a_tied_column_the_direction_holds_at_zero_is_active() {
    let r = [1.0, 0.5, 0.5, 0.5];
    let d = [1.0, 1.0];
    let path = lars_lasso(&r, &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1]]);
    assert_eq!(path.coefs, vec![0.0, 0.0, 0.0, 2.0]);
    let c = 2f64.powi(-20);
    let r: Vec<f64> = [1.0, 0.5, 0.5, 0.5 - 2f64.powi(-50)]
        .iter()
        .map(|v| v * c)
        .collect();
    let d = [c, c];
    let path = lars_lasso(&r, &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.active, vec![vec![0, 1], vec![0, 1]]);
    assert_eq!(path.penalties, vec![c, 0.0]);
    assert!(path.coefs[2].abs() < 1e-14, "{}", path.coefs[2]);
}

/// A tied column the direction moves against its sign by more than `TIE`
/// of the direction's size is left out. On `R = [[¾, ½], [½, ½ − e]]` with
/// `e = 3 · 2⁻⁴⁹` and `d = [1, 1]`, `δ_0` is about `−3 TIE` beside
/// `δ_1 = 2`. Left out, column 0's correlation falls at `1 + 2e` times the
/// speed of `λ`, within `TIE` of it, which the test for staying out allows.
#[test]
fn a_tied_column_moved_against_its_sign_past_the_tolerance_stays_out() {
    let e = 3.0 * 2f64.powi(-49);
    let r = [0.75, 0.5, 0.5, 0.5 - e];
    let d = [1.0, 1.0];
    let path = lars_lasso(&r, &d, &[true; 2], LarsLimits::default());
    assert_eq!(path.active, vec![vec![1], vec![1]]);
    assert_eq!(path.penalties, vec![1.0, 0.0]);
    assert_eq!(path.coefs[2], 0.0);
    assert_kkt_everywhere(&r, &d, &path, 2.0 * lars::TIE);
}

/// A path without a tie is the one before task 236, to the bit: thirteen
/// knots and three columns leaving, on `R = L Lᵀ` for `L` in multiples of
/// 1/8 and `d` in multiples of 1/32, which are exact in binary, so the bits
/// depend on no library's rounding. The digest is FNV-1a over the
/// penalties' and coefficients' bits and the active sets, of the path that
/// `36f33db` walked.
#[test]
fn a_path_without_a_tie_is_the_one_before_ties_were_read() {
    let k = 6;
    #[rustfmt::skip]
    let l = [
        0.75, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.375, 0.625, 0.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, 0.75, 0.0, 0.0, 0.0,
        0.625, 1.0, -0.75, 0.5, 0.0, 0.0,
        0.625, 0.5, 0.625, 0.75, 0.625, 0.0,
        -0.5, -0.875, -0.125, -0.375, -0.25, 0.75,
    ];
    let mut r = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            r[i * k + j] = (0..k).map(|q| l[i * k + q] * l[j * k + q]).sum();
        }
    }
    let d = [-1.09375, -1.71875, -1.3125, 0.78125, -1.78125, -0.625];
    let path = lars_lasso(&r, &d, &[true; 6], LarsLimits::default());
    assert_eq!(path.len(), 13);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for v in path.penalties.iter().chain(&path.coefs) {
        eat(v.to_bits());
    }
    for a in &path.active {
        eat(a.len() as u64);
        for &j in a {
            eat(j as u64);
        }
    }
    assert_eq!(h, 0x9b5d_d963_7002_7a13, "{h:#018x}");
    assert_kkt_everywhere(&r, &d, &path, 1e-14);
}

/// A column whose correlation is not finite never enters, and the path on
/// the rest ends. An infinite one made `λ_max` infinite, so every step's
/// tolerance was NaN and no step ever ended the path.
#[test]
fn a_column_whose_correlation_is_not_finite_never_enters() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let path = lars_lasso(&identity(2), &[bad, 0.5], &[true; 2], LarsLimits::default());
        assert_eq!(path.penalties, vec![0.5, 0.0], "{bad}");
        assert_eq!(path.active, vec![vec![1], vec![1]], "{bad}");
        assert_eq!(path.coefs, vec![0.0, 0.0, 0.0, 0.5], "{bad}");
    }
}
