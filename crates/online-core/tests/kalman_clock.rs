//! `coef_half_life` is a half-life on the clock (docs/PLAN.md task 150): the
//! process noise a row adds is `sigma^2 (ln2 * d / h)^2` for a row `d` clock
//! units after the last, so EW-RLS's steady-state gain -- the promise the
//! parameter is defined by -- holds at any row spacing. Held here two ways: the
//! clock time a slope step takes to be learned agrees across spacings and with
//! `ewridge` at the same half-life, where the random walk's `q d` parted by a
//! factor of five (on this stream and criterion; the plan's 74/38/13 are
//! another's); and a gap row adds exactly `q d^2` to `P`. The Kalman's 1.4
//! over `ewridge` here is `R = sigma^2` inflating at the step before `P`
//! catches up through `q` (itself proportional to `sigma^2`): a property of
//! that choice of `R`, the same at every spacing, which is why the band is
//! loose.

use online_core::*;

const H: f64 = 50.0;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn kalman() -> Kalman {
    Kalman::new(KalmanCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(H),
        half_life: vec![H],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    })
    .unwrap()
}

fn ewridge() -> EwRidge {
    EwRidge::new(EwRidgeCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(H),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

/// The fitted slope, read off two predictions at the same clock (both
/// models report coefficients, but through different accessors).
fn slope(m: &impl OnlineModel) -> f64 {
    m.predict(&[1.0], 0.0).pred[0] - m.predict(&[0.0], 0.0).pred[0]
}

/// A stream whose slope steps from 1 to 3 at clock 400, rows `spacing`
/// apart: the clock time after the step until the fitted slope is within
/// a fifth of the way of the new one.
fn adaptation(spacing: f64, model: &mut impl OnlineModel) -> f64 {
    let mut s = 7u64;
    let mut t = 0.0;
    let mut rows = 0u64;
    loop {
        let x = [lcg(&mut s)];
        let truth = if t < 400.0 { 1.0 } else { 3.0 };
        let y = truth * x[0] + 0.05 * lcg(&mut s);
        model.step(&x, &[Some(y)], if rows == 0 { 0.0 } else { spacing }, 1.0);
        if t >= 400.0 && (slope(model) - 3.0).abs() < 0.4 {
            return t - 400.0;
        }
        t += spacing;
        rows += 1;
        assert!(rows < 200_000, "never adapted at spacing {spacing}");
    }
}

#[test]
fn a_slope_step_is_learned_in_the_same_clock_time_at_any_spacing() {
    let spacings = [1.0, 0.25, 0.04];
    let k: Vec<f64> = spacings
        .iter()
        .map(|&d| adaptation(d, &mut kalman()))
        .collect();
    let r: Vec<f64> = spacings
        .iter()
        .map(|&d| adaptation(d, &mut ewridge()))
        .collect();
    eprintln!("adaptation in clock units, kalman {k:?} ewridge {r:?}");
    let spread = |v: &[f64]| {
        v.iter().cloned().fold(0.0, f64::max) / v.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(
        spread(&k) < 1.35,
        "kalman adapts in {k:?} clock units across spacings"
    );
    assert!(
        spread(&r) < 1.35,
        "ewridge adapts in {r:?} clock units across spacings"
    );
    for (a, b) in k.iter().zip(&r) {
        assert!(a / b > 0.5 && a / b < 2.0, "kalman {a} against ewridge {b}");
    }
}

/// A gap of `D` clock units adds `q D^2` to each slot's variance, the
/// transition being the identity: the exact form, which `q D` would miss by
/// the factor `D`. Charged once, by the row that next observes the target,
/// for the whole gap however many rows it is cut into (docs/PLAN.md task
/// 211): rows of 10 and 5 with no target add `q 15² = 0.225` and not `q (10²
/// + 5²) = 0.125`. Read through the readiness statistic, `1 + zᵀ P⁻ z / R`
/// at `z = [1, 1]` with `R = 1`, whose `P⁻` is the prior the next
/// observation's update starts from; and through the prediction variance
/// `zᵀ P z + R`, whose `P` as it stands carries the gap's noise so far.
#[test]
fn a_gap_adds_q_times_the_square_of_the_clock_step() {
    let mut m = Kalman::new(KalmanCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(H),
        half_life: vec![H],
        q: Some(vec![0.0, 0.001]),
        obs_var: Some(1.0),
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    })
    .unwrap();
    let mut s = 3u64;
    for i in 0..40 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let prior = |m: &Kalman, d: f64| {
        let mut out = Vec::new();
        m.row_error_inflation_into(&[1.0], d, &mut out);
        out[0] * out[0] - 1.0
    };
    let (before, var) = (prior(&m, 0.0), m.pred_var(&[1.0])[0]);
    m.step(&[0.5], &[None], 10.0, 1.0);
    let after = prior(&m, 0.0);
    assert!(
        (after - before - 0.001 * 100.0).abs() <= 1e-12,
        "q D^2 = 0.1, got {}",
        after - before
    );
    m.step(&[0.5], &[None], 5.0, 1.0);
    let longer = prior(&m, 0.0);
    assert!(
        (longer - before - 0.001 * 225.0).abs() <= 1e-12,
        "q D^2 = 0.225 for the gap whole, got {}",
        longer - before
    );
    // `P` as it stands carries the gap's noise as the next observation will
    // charge it: the prediction variance is that prior plus `R = 1`.
    let now = m.pred_var(&[1.0])[0];
    assert!(
        (now - var - 0.001 * 225.0).abs() <= 1e-12,
        "pred_var moved by {}, not q D^2 = 0.225",
        now - var
    );
    assert!(
        (now - (longer + 1.0)).abs() <= 1e-12,
        "{now} against {}",
        longer + 1.0
    );
    // And the row asked about a unit after the last adds its own clock to
    // the gap: `q (15 + 1)²`.
    let ahead = prior(&m, 1.0);
    assert!(
        (ahead - before - 0.001 * 256.0).abs() <= 1e-12,
        "q (D + d)^2 = 0.256, got {}",
        ahead - before
    );
    // The observation takes the gap's noise and clears it: a unit after
    // it, the prior adds `q` itself, so the two forms agree only there.
    m.step(&[0.5], &[Some(1.0)], 0.0, 1.0);
    let step = prior(&m, 1.0) - prior(&m, 0.0);
    assert!(
        (step - 0.001).abs() <= 1e-12,
        "q d^2 at d = 1 is q, got {step}"
    );
}

// ---------------------------------------------------------------------------
// The process noise is charged per informative row (docs/PLAN.md task 211;
// the review of 2026-10-08, kalman item 1).
// ---------------------------------------------------------------------------

/// A stream with gaps: two features at levels and scales of their own, a
/// target on a slope that drifts, clock steps of 1 with a step of 9 before
/// every 17th row.
fn gappy(n: usize, seed: u64) -> Vec<([f64; 2], f64, f64)> {
    let mut s = seed;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = [3.0 + 2.0 * lcg(&mut s), -1.0 + 0.3 * lcg(&mut s)];
        let drift = i as f64 / n as f64;
        let y = 0.5 + (1.5 - drift) * x[0] + (2.0 * drift - 0.8) * x[1] + 0.2 * lcg(&mut s);
        let d = match i {
            0 => 0.0,
            _ if i % 17 == 0 => 9.0,
            _ => 1.0,
        };
        rows.push((x, y, d));
    }
    rows
}

fn kalman_with(standardize: bool, share_p: bool, revert: f64) -> Kalman {
    Kalman::new(KalmanCfg {
        n_features: 2,
        n_targets: if share_p { 2 } else { 1 },
        fit_intercept: true,
        decay: Decay::Halflife(H),
        half_life: vec![20.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY, revert, f64::INFINITY],
        standardize,
    })
    .unwrap()
}

/// What a real row reports: its predictions, and after it the coefficients
/// and the prediction variance at a fixed row.
fn report(m: &Kalman, step: &Step) -> Vec<f64> {
    let mut out = step.pred.clone();
    out.extend(m.coefficients().into_iter().flatten());
    out.extend(m.pred_var(&[2.5, -0.9]));
    out
}

/// A row's targets and weight.
type Inserted = (Vec<Option<f64>>, f64);

/// Runs `rows`, and the same rows with a row inserted half-way through every
/// gap after the first row (`insert` gives it its targets and weight from
/// the next real row's target), and returns the worst difference over every
/// real row's report, relative to `1 + |value|`, NaN for NaN.
fn worst_move(
    build: &dyn Fn() -> Kalman,
    rows: &[([f64; 2], f64, f64)],
    targets: usize,
    insert: &dyn Fn(f64) -> Inserted,
) -> f64 {
    let ys = |y: f64| -> Vec<Option<f64>> { (0..targets).map(|j| Some(y + j as f64)).collect() };
    let (mut plain, mut with) = (build(), build());
    let mut worst = 0.0f64;
    for (i, (x, y, d)) in rows.iter().enumerate() {
        let a = plain.step(x, &ys(*y), *d, 1.0);
        let b = if i > 0 {
            // Half-way through the gap, at the row's own features shifted:
            // a row the moments would learn if it had a weight.
            let (yi, wi) = insert(*y);
            let xi = [x[0] + 1.0, x[1] - 0.5];
            with.step(&xi, &yi, d / 2.0, wi);
            with.step(x, &ys(*y), d / 2.0, 1.0)
        } else {
            with.step(x, &ys(*y), *d, 1.0)
        };
        for (u, v) in report(&plain, &a).iter().zip(&report(&with, &b)) {
            assert_eq!(u.is_nan(), v.is_nan(), "row {i}: {u} against {v}");
            if u.is_finite() {
                worst = worst.max((u - v).abs() / (1.0 + u.abs()));
            }
        }
    }
    worst
}

/// Hard rule 9: a row of weight 0 advances the clock and learns nothing, so
/// one inserted half-way through every gap leaves every real row's
/// prediction, coefficients and prediction variance where they were, to
/// rounding. The process noise was `Q d²` per row, which is not additive over
/// a split gap (`d₁² + d₂² < (d₁ + d₂)²`): the inserted rows shrank each
/// gap's noise and moved every later prediction, by 0.35 at
/// `coef_half_life` 20 (the review's probe; the bank's `rls`, `ewridge` and
/// `sgd` moved by rounding). Standardized and not, a reverting slope, and
/// two targets on one covariance.
#[test]
fn a_zero_weight_row_inside_a_gap_moves_nothing() {
    let rows = gappy(300, 211);
    for (standardize, share_p, revert) in [
        (true, false, f64::INFINITY),
        (false, false, f64::INFINITY),
        (true, false, 40.0),
        (true, true, f64::INFINITY),
    ] {
        let targets = if share_p { 2 } else { 1 };
        let build = || kalman_with(standardize, share_p, revert);
        let worst = worst_move(&build, &rows, targets, &|y| {
            ((0..targets).map(|j| Some(y + j as f64)).collect(), 0.0)
        });
        eprintln!("standardize {standardize}, share_p {share_p}, revert {revert}: {worst:.2e}");
        assert!(
            worst <= 1e-12,
            "standardize {standardize}, share_p {share_p}, revert {revert}: a zero-weight \
             row inside a gap moved a real row's report by {worst:e}"
        );
    }
}

/// The same of a row whose target is null, unstandardized, so that the row's
/// features reach nothing the filter reads (standardized, they move the
/// moments, which is a move of the coordinates the process noise is defined
/// in). It moved every later prediction by 0.071 at `coef_half_life` 20
/// (the review's probe, with `obs_var`).
#[test]
fn a_null_target_inside_a_gap_moves_nothing_unstandardized() {
    let rows = gappy(300, 212);
    for share_p in [false, true] {
        let targets = if share_p { 2 } else { 1 };
        let build = || kalman_with(false, share_p, f64::INFINITY);
        let worst = worst_move(&build, &rows, targets, &|_| (vec![None; targets], 1.0));
        eprintln!("share_p {share_p}: {worst:.2e}");
        assert!(
            worst <= 1e-12,
            "share_p {share_p}: a null-target row inside a gap moved a real row's report by \
             {worst:e}"
        );
    }
}

/// The slope of a target seen on one row in `n`, its features on every row:
/// it steps from +1 to -1 at clock 4000, a row that observes it, and the
/// clock time from there to the first observation after which the fitted
/// slope is below 0 is its memory. The median over seeds.
fn crossing(n: usize, standardize: bool) -> f64 {
    let mut times = Vec::new();
    for seed in 0..15u64 {
        let mut m = Kalman::new(KalmanCfg {
            n_features: 1,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(500.0),
            half_life: vec![H],
            q: None,
            obs_var: if standardize { None } else { Some(0.09) },
            p0: 1.0,
            share_p: false,
            min_weight: 0.0,
            revert_half_life: vec![f64::INFINITY],
            standardize,
        })
        .unwrap();
        let mut s = 1000 + seed;
        let mut found = f64::NAN;
        for i in 0..8000usize {
            let x = [1.7 * lcg(&mut s)];
            let t = i as f64;
            let truth = if t < 4000.0 { 1.0 } else { -1.0 };
            let y = truth * x[0] + 0.5 * lcg(&mut s);
            let seen = i % n == 0;
            m.step(
                &x,
                &[seen.then_some(y)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            if seen && t >= 4000.0 && m.coefficients()[0][1] < 0.0 {
                found = t - 4000.0;
                break;
            }
        }
        times.push(found);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// EW-RLS's crossing for observations `d` clock units apart, the first at
/// the step: after its `j`-th observation since the step the new slope holds
/// `1 − 2^(−(j+1) d / h)` of the weight, so the fit crosses 0 at the first
/// `j` with `(j + 1) d ≥ h`, clock `(⌈h/d⌉ − 1) d`: 49 at `d` = 1, 48, 40
/// and 25 at 4, 10 and 25. Seen on the clock a sparse target crosses
/// earlier, because its first observation of the new slope comes at the
/// step itself with the whole of a gap's weight.
fn ew_rls_crossing(d: f64) -> f64 {
    ((H / d).ceil() - 1.0) * d
}

/// `coef_half_life` is a half-life on the clock for a target seen on one
/// row in `n` as for one seen on every row: the noise a row adds is charged
/// for the whole clock since its covariance last took an observation, `Q D²`,
/// which is what EW-RLS's forgetting `2^(-D/h)` over that clock matches (the
/// module doc). So the sparse target's crossing is the dense one's scaled
/// by EW-RLS's ratio at the same spacings ([`ew_rls_crossing`]), to within
/// one observation's spacing or a fifth, whichever is larger: both with
/// `obs_var`, where the dense crossing is EW-RLS's own 49, and with the EW
/// residual variance as the noise, whose inflation at the step slows the
/// filter by about 1.5 and slows it less where fewer residuals see the step.
/// Charged per row, the sparse target's covariance took `n` times `Q d²`
/// where `Q (n d)²` was due, and its memory grew as `h √n`: 166, 270 and 462
/// clock units at `n` = 4, 10 and 25 against 74.5 dense (the review's table
/// 3d).
#[test]
fn a_sparse_target_adapts_on_the_clock_as_a_dense_one_does() {
    let mut seen = Vec::new();
    for standardize in [true, false] {
        let dense = crossing(1, standardize);
        for n in [4usize, 10, 25] {
            let sparse = crossing(n, standardize);
            let want = dense * ew_rls_crossing(n as f64) / ew_rls_crossing(1.0);
            eprintln!(
                "standardize {standardize}: n = {n}: {sparse} against {dense} dense, \
                 {want:.1} by EW-RLS's ratio"
            );
            seen.push((standardize, n, sparse, dense, want));
        }
    }
    for (standardize, n, sparse, dense, want) in seen {
        assert!(
            (sparse - want).abs() <= (n as f64).max(0.2 * want),
            "standardize {standardize}: a target seen on one row in {n} crossed in {sparse} \
             clock units, a dense one in {dense}: {want} by EW-RLS's ratio"
        );
    }
}
