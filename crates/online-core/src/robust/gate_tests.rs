//! `min_weight` withholds the output alone (docs/PLAN.md task 278, D10).
//! Its job is to withhold predictions; a row it withholds was still
//! predicted by the fit, and the fit learns from that prediction: the
//! residual scale takes its residual, Huber weighs it, the quantile band is
//! drawn around it, and the first solve falls on the first row with weight.
//! So every coefficient, the scale and the refit are the same at any
//! `min_weight`, and the output differs only by the rows it withholds.
//! Under the gate that held learning back -- what a state written before
//! schema 64 goes on with (`RobustCfg::min_weight_holds_learning`) -- the
//! same streams part, which is the evidence each case reaches the path.

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(loss: RobustLoss, scale: HuberScale, refit_rows: u32) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        loss,
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: f64::INFINITY,
        max_rows_between_solves: 7,
        solve_share: None,
        quantile_eps: 0.2,
        huber_scale: scale,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows,
        min_weight_holds_learning: false,
    }
}

/// A row: the features, the targets, the clock step and the weight.
type Row = ([f64; 2], [Option<f64>; 2], f64, f64);

/// Two targets, one joining at row 30, one-sided outliers the cut binds
/// on, uneven weights with a zero every 11 rows, and a gap of 150 clock
/// units at row 200 that takes the weight under every `min_weight` here.
fn rows() -> Vec<Row> {
    let mut s = 23u64;
    (0..400)
        .map(|i| {
            let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
            let wild = if x[0] > 0.3 && lcg(&mut s) > 0.5 {
                20.0
            } else {
                0.0
            };
            let y0 = 4.0 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s) + wild;
            let y1 = (i >= 30).then(|| -1.0 + x[1] + 0.2 * lcg(&mut s) - wild);
            let d = match i {
                0 => 0.0,
                200 => 150.0,
                _ => 1.0,
            };
            let w = if i % 11 == 5 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            (x, [Some(y0), y1], d, w)
        })
        .collect()
}

/// Per row the coefficients' bits (`None` before the first solve), the
/// output, and the residual scale after it; and the refit's status at the
/// end.
type Run = (
    Vec<Option<Vec<u64>>>,
    Vec<Vec<f64>>,
    Vec<Vec<u64>>,
    Vec<crate::RefitStatus>,
);

fn run(c: &RobustCfg, min_weight: f64, own: Option<Vec<f64>>) -> Run {
    let mut m = Robust::new(RobustCfg {
        min_weight,
        ..c.clone()
    })
    .unwrap();
    if let Some(own) = own {
        m.set_target_min_weight(own).unwrap();
    }
    let (mut coef, mut out, mut scale) = (Vec::new(), Vec::new(), Vec::new());
    for (x, y, d, w) in rows() {
        out.push(m.step(&x, &y, d, w).pred);
        coef.push(
            m.coefficients()
                .map(|b| b.iter().flatten().map(|v| v.to_bits()).collect()),
        );
        scale.push(
            (0..2)
                .map(|j| m.residual_scale(j).unwrap_or(f64::NAN).to_bits())
                .collect(),
        );
    }
    (coef, out, scale, m.refit_status())
}

fn cases() -> Vec<(&'static str, RobustCfg)> {
    vec![
        (
            "huber, mad, refit",
            cfg(RobustLoss::Huber { delta: 1.345 }, HuberScale::Mad, 60),
        ),
        (
            "huber, rms",
            cfg(RobustLoss::Huber { delta: 1.345 }, HuberScale::Rms, 0),
        ),
        (
            "quantile, mad",
            cfg(RobustLoss::Quantile { tau: 0.7 }, HuberScale::Mad, 0),
        ),
        (
            "quantile, rms",
            cfg(RobustLoss::Quantile { tau: 0.7 }, HuberScale::Rms, 0),
        ),
    ]
}

/// The fit at `min_weight` 0, 3 (a row per coefficient), 25 and a per
/// target list is the same to the bit -- coefficients, scale and refit --
/// and the output is the same wherever both report one, a null only where
/// that run's gate withheld the row. Under the gate that held learning
/// back, 50 moves the coefficients.
#[test]
fn min_weight_withholds_the_output_alone() {
    for (name, c) in cases() {
        let base = run(&c, 0.0, None);
        let mut withheld_somewhere = false;
        for (mw, own) in [(3.0, None), (25.0, None), (2.0, Some(vec![2.0, 40.0]))] {
            let other = run(&c, mw, own.clone());
            let what = format!("{name}, min_weight {mw}, own {own:?}");
            assert_eq!(other.0, base.0, "{what}: the coefficients");
            assert_eq!(other.2, base.2, "{what}: the scale");
            assert_eq!(other.3, base.3, "{what}: the refit");
            let mut withheld = 0;
            for (i, (a, b)) in base.1.iter().zip(&other.1).enumerate() {
                for (p, q) in a.iter().zip(b) {
                    if q.is_nan() && p.is_finite() {
                        withheld += 1;
                    } else {
                        assert_eq!(p.to_bits(), q.to_bits(), "{what}, row {i}");
                    }
                }
            }
            println!("{what}: {withheld} outputs withheld");
            withheld_somewhere |= withheld > 0;
            if mw == 25.0 {
                // The first rows and those after the gap are withheld, the
                // rest reported (the weight's ceiling is near 43).
                let reported = other.1.iter().flatten().filter(|p| p.is_finite()).count();
                assert!(
                    withheld > 40 && reported > 400,
                    "{what}: {withheld}, {reported}"
                );
            }
        }
        assert!(withheld_somewhere);
        // The old gate on the same stream moves the fit.
        let held = RobustCfg {
            min_weight_holds_learning: true,
            ..c.clone()
        };
        assert_ne!(
            run(&held, 50.0, None).0,
            run(&held, 0.0, None).0,
            "{name}: the gate that held learning back moved the fit"
        );
    }
}

/// The refit fires on its own count of learned rows, whatever
/// `min_weight` withholds: at 60 rows per target, under `min_weight`
/// past the refit's row, the same refit, on the same row.
#[test]
fn the_refit_fires_on_its_own_count_whatever_min_weight() {
    let c = cfg(RobustLoss::Huber { delta: 1.345 }, HuberScale::Mad, 60);
    let at = |mw: f64| {
        let mut m = Robust::new(RobustCfg {
            min_weight: mw,
            ..c.clone()
        })
        .unwrap();
        let mut fired = vec![None; 2];
        for (i, (x, y, d, w)) in rows().into_iter().enumerate() {
            m.step(&x, &y, d, w);
            for (j, st) in m.refit_status().iter().enumerate() {
                if st.done && fired[j].is_none() {
                    fired[j] = Some(i);
                }
            }
        }
        fired
    };
    let free = at(0.0);
    assert!(free.iter().all(Option::is_some), "{free:?}");
    assert_eq!(at(150.0), free);
}

/// A target's history below the smallest normal double of the row's weight
/// is forgotten, as a decay that underflows forgets it (`gaps::decay_into`),
/// so the fit learns the same from it as from none; and the refit's buffer,
/// mid-fill, keeps the model's factor and saves a state that loads. Rows of
/// weight `2^-1074` (the least subnormal) then rows of weight 1.
#[test]
fn a_history_below_the_normal_range_of_a_row_is_forgotten() {
    let c = cfg(RobustLoss::Huber { delta: 1.345 }, HuberScale::Mad, 60);
    let tiny = f64::from_bits(1);
    let run = |history: bool| {
        let mut m = Robust::new(c.clone()).unwrap();
        let mut s = 3u64;
        let mut out = Vec::new();
        for i in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = [Some(1.0 + x[0] - x[1]), Some(2.0 * x[1])];
            let w = if i < 10 { tiny } else { 1.0 };
            if i >= 10 || history {
                out = m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, w).pred;
            }
        }
        let back = Robust::restore(&m.state()).expect("the state loads");
        assert_eq!(back.refit_status(), m.refit_status());
        (out, m.coefficients().map(<[Vec<f64>]>::to_vec))
    };
    let (with, b1) = run(true);
    let (without, b2) = run(false);
    assert_eq!(b1, b2, "the subnormal history is forgotten");
    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&with), bits(&without));
}

/// The floor's counts are state: a target's first rows, to a row per
/// coefficient (three here) and no further, saved and restored; a count
/// list of the wrong length, a count past the floor, and counts under the
/// gate that held learning back, which has no floor, are each refused.
#[test]
fn the_floors_counts_are_state_and_a_damaged_one_is_refused() {
    let c = cfg(RobustLoss::Quantile { tau: 0.5 }, HuberScale::Mad, 0);
    let mut m = Robust::new(c.clone()).unwrap();
    for (i, (x, y, d, w)) in rows().into_iter().take(40).enumerate() {
        m.step(&x, &y, d, w);
        if i == 1 {
            assert_eq!(m.seen, [2, 0], "row {i}");
        }
    }
    assert_eq!(m.seen, [3, 3]);
    assert_eq!(Robust::restore(&m.state()).unwrap(), m);
    let refused = |edit: &dyn Fn(&mut Robust)| {
        let mut bad = m.clone();
        edit(&mut bad);
        matches!(Robust::restore(&bad.state()), Err(StateError::Invalid(_)))
    };
    assert!(refused(&|b| b.seen.truncate(1)), "one count short");
    assert!(refused(&|b| b.seen[0] = 4), "past the floor");
    assert!(
        refused(&|b| b.cfg.min_weight_holds_learning = true),
        "under the old gate"
    );
    let mut held = Robust::new(RobustCfg {
        min_weight_holds_learning: true,
        ..c
    })
    .unwrap();
    held.step(&[0.1, 1.0], &[Some(1.0), None], 0.0, 1.0);
    assert!(held.seen.is_empty(), "the old gate keeps no floor");
}
