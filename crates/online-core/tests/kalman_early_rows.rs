//! A standardizing `kalman`'s first rows (docs/PLAN.md task 211; the review
//! of 2026-10-08, kalman item 2).
//!
//! The filter reads each row against the moments of the rows before it, and
//! over the first rows those rest on nothing: on row 0 a feature is read raw,
//! against a mean of 0 and a scale of 1, and on row 1 against a variance of
//! 0. The prior was sized on row 0, from that row's squared innovation alone,
//! and a 22-row warm-up then read the state in the coordinates of whatever
//! the moments had become. Each coefficient's prior is now set on the first
//! row its feature's scale is usable, from the mean squared innovation over
//! at least three rows, and the state follows the moments from there.
//!
//! Every stream here decays by a literal factor at unit clock steps, so no
//! call into the platform's libm is on a path a number is pinned on.

use online_core::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn kalman(k: usize, decay: Decay, coef_half_life: f64, min_weight: f64) -> Kalman {
    Kalman::new(KalmanCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay,
        half_life: vec![coef_half_life],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    })
    .unwrap()
}

/// `2^(-1/20)`, a half-life of 20 rows, written as a literal.
const HALF_LIFE_20: f64 = 0.965_936_328_924_846;

/// Two uniform features and a target on them (the research behind task
/// 206's regime 3); the features at `level`.
fn regime3(level: f64) -> Vec<([f64; 2], f64)> {
    let mut s = 5u64;
    (0..3000)
        .map(|_| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
            ([x[0] + level, x[1] + level], y)
        })
        .collect()
}

/// A standardizing filter's predictions do not depend on the level its
/// features sit at, past the rows where the level's rounding is all the
/// moments have: from row 100 on, the fit at 1e8 predicts what the fit at 0
/// does to 1e-6. It parted by 0.87 at `coef_half_life` 100: row 0 read each
/// feature raw, 1e8 against a mean of 0, and sized the prior from that row's
/// innovation, so the first correction was made in coordinates a hundred
/// million times off and was kept (the review's table 4).
#[test]
fn a_level_of_1e8_in_the_features_moves_no_prediction_after_the_first_rows() {
    for coef_half_life in [100.0, f64::INFINITY] {
        let run = |level: f64| -> Vec<f64> {
            let mut m = kalman(2, Decay::Lam(HALF_LIFE_20), coef_half_life, 3.0);
            regime3(level)
                .iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
                        .pred[0]
                })
                .collect()
        };
        let (base, high) = (run(0.0), run(1e8));
        let worst = (100..base.len())
            .map(|i| (base[i] - high[i]).abs())
            .fold(0.0f64, f64::max);
        eprintln!("coef_half_life {coef_half_life}: worst {worst:.3e}");
        assert!(
            worst <= 1e-6,
            "coef_half_life {coef_half_life}: the fit at a level of 1e8 parted from the fit at \
             0 by {worst:e} after row 100"
        );
    }
}

/// With no process noise a coefficient's variance only shrinks, so a prior
/// sized too small pins the fit for good. Sized from one squared innovation
/// -- a chi-square with one degree of freedom, below 1% of its mean one time
/// in twelve -- a first row whose target happened to sit near the fit's
/// prediction of 0 left the slopes at a fraction of their truth after
/// thousands of rows (the review's seed 2: an innovation of 0.004 on a target
/// of spread 4.5, slopes 0.3 to 1.2 against 2 after 30,000 rows). Here the
/// first row's target is exactly 0.002 on a target of spread 2: the prior
/// rests on three rows' innovations, and the fit finds its slopes.
#[test]
fn a_tiny_first_innovation_does_not_pin_a_fit_with_no_process_noise() {
    let mut m = kalman(2, Decay::Halflife(f64::INFINITY), f64::INFINITY, 3.0);
    let mut s = 21u64;
    let (mut sq, mut n) = (0.0, 0.0);
    for i in 0..4000usize {
        let (x, noise) = if i == 0 {
            ([0.0004, 0.0006], 0.0)
        } else {
            ([1.7 * lcg(&mut s), 1.7 * lcg(&mut s)], 0.01 * lcg(&mut s))
        };
        let y = 2.0 * x[0] + 2.0 * x[1] + noise;
        let p = m
            .step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
            .pred[0];
        if i >= 1000 {
            sq += (y - p).powi(2);
            n += 1.0;
        }
    }
    let coef = m.coefficients()[0].clone();
    let mse = sq / n;
    // The noise's variance is 0.01² / 3.
    let noise = 1e-4 / 3.0;
    eprintln!("mse over the noise {:.3e}, coef {coef:?}", mse / noise);
    assert!(
        mse < 2.0 * noise,
        "the fit pinned by its first row's prior: error {mse:e} against a noise of {noise:e}, \
         coefficients {coef:?}"
    );
    assert!(
        coef[1..].iter().all(|c| (c - 2.0).abs() < 1e-2),
        "coefficients {coef:?} against [0, 2, 2]"
    );
}
