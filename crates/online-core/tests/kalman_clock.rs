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

fn ew_ridge() -> EwRidge {
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
        .map(|&d| adaptation(d, &mut ew_ridge()))
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

/// A gap row adds `q d^2` to each slot's variance, the transition being the
/// identity: the exact per-row form, which `q d` would miss by the factor
/// `d`. Read through the prediction variance `zᵀ P z + R` at `z = [1, 1]`
/// with a fixed `R`, which moves by exactly the slope slot's `q d^2`.
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
    let before = m.pred_var(&[1.0])[0];
    m.step(&[0.5], &[None], 10.0, 1.0);
    let after = m.pred_var(&[1.0])[0];
    assert!(
        (after - before - 0.001 * 100.0).abs() <= 1e-12,
        "q d^2 = 0.1, got {}",
        after - before
    );
    // And a unit row adds `q` itself, so the two forms agree only there.
    let mut unit = m.clone();
    unit.step(&[0.5], &[None], 1.0, 1.0);
    let step = unit.pred_var(&[1.0])[0] - after;
    assert!(
        (step - 0.001).abs() <= 1e-12,
        "q d^2 at d = 1 is q, got {step}"
    );
}
