//! `huber`'s feature sets, sum-scale prior and emitted scale (docs/PLAN.md
//! task 272 (a), docs/HUBER.md §8 request 1): each held to a reference
//! built here from a model that already has the property -- `ewridge` on
//! the rows at the weights the full set's residuals give, and `rls` -- not
//! to this model's own arithmetic.

use super::*;
use crate::{EwRidge, EwRidgeCfg, OnlineModel};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn huber(k: usize, m: usize) -> RobustCfg {
    RobustCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(150.0),
        loss: RobustLoss::Huber { delta: 1.345 },
        ridge: 1e-4,
        standardize: false,
        min_weight: 4.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: Some(Decay::Halflife(60.0)),
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        min_weight_holds_learning: false,
    }
}

fn ridge_on(k: usize, decay: Decay, ridge: f64) -> EwRidge {
    EwRidge::new(EwRidgeCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay,
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: crate::TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

/// One row of a stream with a level, outliers the cut binds on, a null on
/// the second target every 13 rows and uneven weights.
fn row(i: usize, s: &mut u64) -> ([f64; 3], [Option<f64>; 2], f64, f64) {
    let x = [lcg(s), 2.0 + lcg(s), lcg(s)];
    let wild = if i % 19 == 5 { 30.0 } else { 0.0 };
    let y0 = 100.0 + 2.0 * x[0] - x[1] + 0.5 * x[2] + 0.3 * lcg(s) + wild;
    let y1 = (i % 13 != 4).then(|| -5.0 + x[1] + 0.2 * lcg(s) - wild);
    let d = if i == 0 { 0.0 } else { 0.5 + lcg(s).abs() };
    (x, [Some(y0), y1], d, 0.5 + lcg(s).abs())
}

/// Each set's fit is the ridge fit of its features on the target's rows,
/// each row at its weight times the Huber weight of the *largest* set's
/// residual, the one number per row and target the shared sums can hold.
/// The reference: a plain `huber` on every feature gives the residual and
/// the scale before each row (the largest set is every feature), the
/// Huber weight is formed from them here, and one `ewridge` per set and
/// target is fed the rows at that weight. The largest set's slot is the
/// plain model's, to the bit.
#[test]
fn each_set_is_weighed_by_the_largest_sets_residual() {
    let sets = vec![
        ("one".to_string(), vec![0]),
        ("all".to_string(), vec![0, 1, 2]),
        ("two".to_string(), vec![2, 1]),
    ];
    // Every prediction the fit learns from is reported, so the reference
    // reads the residuals the model weighed its rows by.
    let mut with_sets = Robust::new(RobustCfg {
        feature_sets: sets.clone(),
        min_weight: 0.0,
        ..huber(3, 2)
    })
    .unwrap();
    let mut plain = Robust::new(RobustCfg {
        min_weight: 0.0,
        ..huber(3, 2)
    })
    .unwrap();
    assert_eq!(with_sets.cfg().weighing_set(), 1);
    assert_eq!(with_sets.n_outputs(), 6);
    let decay = Decay::Halflife(150.0);
    // refs[j][c]: target j, set c, on that set's columns.
    let mut refs: Vec<Vec<EwRidge>> = (0..2)
        .map(|_| {
            sets.iter()
                .map(|(_, idx)| ridge_on(idx.len(), decay, 1e-4))
                .collect()
        })
        .collect();
    let mut s = 31u64;
    let (mut compared, mut cut, mut worst) = (0, 0, 0.0f64);
    for i in 0..1500 {
        let (x, y, d, w) = row(i, &mut s);
        // The scale and the prediction the row is weighed by, read before
        // the row; the decay here is the scale's, which moves no median.
        let scales: Vec<Option<f64>> = (0..2).map(|j| plain.residual_scale(j)).collect();
        let p = plain.step(&x, &y, d, w);
        let q = with_sets.step(&x, &y, d, w);
        for j in 0..2 {
            assert_eq!(
                q.pred[j * 3 + 1].to_bits(),
                p.pred[j].to_bits(),
                "row {i}, target {j}: the largest set is the plain fit"
            );
            let w_rob = match (scales[j], y[j]) {
                (Some(sc), Some(yj)) if p.pred[j].is_finite() => {
                    let a = (yj - p.pred[j]).abs();
                    let c = 1.345 * sc;
                    if a <= c { 1.0 } else { c / a }
                }
                _ => 1.0,
            };
            cut += usize::from(w_rob < 1.0);
            for (c, (_, idx)) in sets.iter().enumerate() {
                let xs: Vec<f64> = idx.iter().map(|&f| x[f]).collect();
                let r = refs[j][c].step(&xs, &[y[j]], d, w * w_rob);
                let got = q.pred[j * 3 + c];
                if got.is_finite() && r.pred[0].is_finite() {
                    compared += 1;
                    worst = worst.max((got - r.pred[0]).abs() / (1.0 + r.pred[0].abs()));
                }
            }
        }
    }
    assert!(cut > 100, "the cut must bind: {cut} rows down-weighted");
    assert!(compared > 8000, "{compared}");
    assert!(
        worst < 1e-9,
        "the sets parted from the reference by {worst:e}"
    );
}

/// A slot outside its set has a coefficient of 0 and a data share of NaN,
/// as `ewridge`'s sets have it; the intercept's share is NaN too.
#[test]
fn a_slot_outside_its_set_has_no_coefficient_and_no_share() {
    let mut m = Robust::new(RobustCfg {
        feature_sets: vec![("a".into(), vec![1]), ("all".into(), vec![0, 1, 2])],
        ..huber(3, 2)
    })
    .unwrap();
    let mut s = 5u64;
    for i in 0..200 {
        let (x, y, d, w) = row(i, &mut s);
        m.step(&x, &y, d, w);
    }
    let beta = m.coefficients().unwrap();
    let shares = m.support_coef().unwrap();
    assert_eq!((beta.len(), shares.len()), (4, 4));
    for j in 0..2 {
        let (a, all) = (j * 2, j * 2 + 1);
        assert_eq!((beta[a][1], beta[a][3]), (0.0, 0.0));
        assert!(beta[a][2] != 0.0 && beta[all].iter().all(|v| *v != 0.0));
        assert!(shares[a][0].is_nan() && shares[a][1].is_nan() && shares[a][3].is_nan());
        assert!((0.0..=1.0).contains(&shares[a][2]));
        assert!(shares[all][0].is_nan());
        assert!(shares[all][1..].iter().all(|v| (0.0..=1.0).contains(v)));
    }
}

/// The weighing set is the one with the most features, the first of them
/// where two tie; without sets it is the one fit.
#[test]
fn the_weighing_set_is_the_largest_and_the_first_of_a_tie() {
    let with = |sets: Vec<Vec<usize>>| RobustCfg {
        feature_sets: sets
            .into_iter()
            .enumerate()
            .map(|(i, v)| (format!("s{i}"), v))
            .collect(),
        ..huber(4, 1)
    };
    assert_eq!(huber(4, 1).weighing_set(), 0);
    assert_eq!(
        with(vec![vec![0], vec![1, 2], vec![3, 0]]).weighing_set(),
        1
    );
    assert_eq!(with(vec![vec![0, 1, 2, 3], vec![1]]).weighing_set(), 0);
    assert_eq!(with(vec![vec![2], vec![0, 1, 3]]).weighing_set(), 1);
}

/// A state with sets goes on to the bit, and one whose fits are not one
/// per target and set is refused.
#[test]
fn a_state_with_sets_restores_and_a_short_one_is_refused() {
    let cfg = RobustCfg {
        feature_sets: vec![("a".into(), vec![0]), ("all".into(), vec![0, 1, 2])],
        ..huber(3, 2)
    };
    let mut m = Robust::new(cfg).unwrap();
    let mut s = 8u64;
    for i in 0..120 {
        let (x, y, d, w) = row(i, &mut s);
        m.step(&x, &y, d, w);
    }
    let mut back = Robust::restore(&m.state()).unwrap();
    for i in 120..200 {
        let (x, y, d, w) = row(i, &mut s);
        let (a, b) = (m.step(&x, &y, d, w), back.step(&x, &y, d, w));
        let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.pred), bits(&b.pred), "row {i}");
    }
    let mut st = m.state();
    let crate::ModelState::Robust(inner) = &mut st.model else {
        unreachable!()
    };
    if let Some(b) = inner.beta.as_mut() {
        b.0.pop();
    }
    assert!(matches!(
        Robust::restore(&st),
        Err(crate::StateError::Invalid(e)) if e.contains("wrong shape")
    ));
}

#[test]
fn feature_sets_are_refused_where_they_mean_nothing() {
    let bad = |cfg: RobustCfg, want: &str| match Robust::new(cfg) {
        Err(e) => assert!(e.contains(want), "{e}"),
        Ok(_) => panic!("accepted; wanted {want}"),
    };
    let sets = |v: Vec<usize>| vec![("s".to_string(), v)];
    bad(
        RobustCfg {
            loss: RobustLoss::Quantile { tau: 0.5 },
            huber_scale: HuberScale::Rms,
            huber_scale_decay: None,
            feature_sets: sets(vec![0]),
            ..huber(3, 1)
        },
        "quantile: feature_sets are for huber",
    );
    bad(
        RobustCfg {
            feature_sets: sets(vec![]),
            ..huber(3, 1)
        },
        "is empty",
    );
    bad(
        RobustCfg {
            feature_sets: sets(vec![3]),
            ..huber(3, 1)
        },
        "out-of-range",
    );
    bad(
        RobustCfg {
            feature_sets: sets(vec![1, 0, 1]),
            ..huber(3, 1)
        },
        "more than once",
    );
}

/// `huber(ridge_scale)` at `huber_delta = inf`, solved on every row, is
/// `rls` with `delta = ridge` and `ewridge(ridge_scale)` under `own_rows`:
/// every row at its own weight, RLS's prior on the decaying sum scale, the
/// intercept penalized. The identity `rls.rs` holds `ewridge` to, held here
/// for the Huber form, on a stream whose second target is null on some
/// rows: each target's sums learn its own rows and age over the rest, as
/// `rls` does per target and `ewridge`'s `own_rows` Gram does.
#[test]
fn the_sum_scale_huber_at_an_infinite_cut_is_rls() {
    let (k, hl, ridge) = (3usize, 40.0, 0.7);
    let mut hub = Robust::new(RobustCfg {
        loss: RobustLoss::Huber {
            delta: f64::INFINITY,
        },
        decay: Decay::Halflife(hl),
        ridge,
        ridge_scale: true,
        min_weight: 0.0,
        ..huber(k, 2)
    })
    .unwrap();
    let mut rls: Vec<crate::Rls> = (0..2)
        .map(|_| {
            crate::Rls::new(crate::RlsCfg {
                n_features: k,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(hl),
                delta: ridge,
                coef_prior: None,
                min_weight: 0.0,
            })
            .unwrap()
        })
        .collect();
    let mut ew = EwRidge::new(EwRidgeCfg {
        n_targets: 2,
        ridge_scale: true,
        ..ridge_on(k, Decay::Halflife(hl), ridge).cfg().clone()
    })
    .unwrap();
    let mut s = 99u64;
    let (mut worst_rls, mut worst_ew, mut compared) = (0.0f64, 0.0f64, 0);
    for i in 0..400 {
        let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let y0 = 1.5 * x[0] - x[1] + 0.3 + 0.05 * lcg(&mut s);
        let y1 = (i % 7 != 3).then(|| 4.0 + 0.2 * x[2] + 0.01 * lcg(&mut s));
        let d = if i == 0 { 0.0 } else { 0.5 + lcg(&mut s).abs() };
        // Two leading rows of weight 0 age the prior and teach nothing.
        let w = if i < 2 { 0.0 } else { 0.5 + lcg(&mut s).abs() };
        let y = [Some(y0), y1];
        let h = hub.step(&x, &y, d, w);
        let e = ew.step(&x, &y, d, w);
        for j in 0..2 {
            let r = rls[j].step(&x, &[y[j]], d, w);
            if i > 8 && h.pred[j].is_finite() {
                compared += 1;
                worst_rls = worst_rls.max((h.pred[j] - r.pred[0]).abs());
                worst_ew = worst_ew.max((h.pred[j] - e.pred[j]).abs());
            }
        }
    }
    assert!(compared > 700, "{compared}");
    assert!(worst_rls < 1e-9, "huber against rls: {worst_rls:e}");
    assert!(worst_ew < 1e-9, "huber against ewridge: {worst_ew:e}");
}

/// Under the sum scale a cut that binds still weighs the rows, and the
/// prior's pull fades: a constant target's intercept, shrunk toward 0 at
/// first, reaches the target as the rows arrive, where the mean-form ridge
/// at the same value holds it short for good.
#[test]
fn the_sum_scale_prior_fades_and_the_mean_one_does_not() {
    let fit = |sum: bool| {
        let mut m = Robust::new(RobustCfg {
            decay: Decay::Halflife(50.0),
            ridge: 10.0,
            ridge_scale: sum,
            min_weight: 0.0,
            ..huber(1, 1)
        })
        .unwrap();
        let mut s = 3u64;
        let mut at = Vec::new();
        for i in 0..300 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(5.0)], f64::from(u8::from(i > 0)), 1.0);
            if i == 19 || i == 299 {
                at.push(m.coefficients().unwrap()[0][0]);
            }
        }
        at
    };
    let sum = fit(true);
    assert!(sum[0] < 4.5 && (sum[1] - 5.0).abs() < 0.02, "{sum:?}");
    let mean = fit(false);
    // The mean-form ridge leaves the intercept unpenalized: 5 throughout.
    assert!(
        (mean[0] - 5.0).abs() < 1e-9 && (mean[1] - 5.0).abs() < 1e-9,
        "{mean:?}"
    );
}

#[test]
fn ridge_scale_is_refused_where_ewridge_refuses_it() {
    let bad = |cfg: RobustCfg, want: &str| match Robust::new(cfg) {
        Err(e) => assert!(e.contains(want), "{e}"),
        Ok(_) => panic!("accepted; wanted {want}"),
    };
    bad(
        RobustCfg {
            ridge_scale: true,
            standardize: true,
            ..huber(3, 1)
        },
        "ridge_scale is incompatible with standardize",
    );
    bad(
        RobustCfg {
            ridge_scale: true,
            feature_sets: vec![("a".into(), vec![0]), ("b".into(), vec![1])],
            ..huber(3, 1)
        },
        "two or more feature sets",
    );
    bad(
        RobustCfg {
            ridge_scale: true,
            loss: RobustLoss::Quantile { tau: 0.5 },
            huber_scale: HuberScale::Rms,
            huber_scale_decay: None,
            ..huber(3, 1)
        },
        "quantile: ridge_scale is for huber",
    );
    // One set is no grid, as for `ewridge`.
    assert!(
        Robust::new(RobustCfg {
            ridge_scale: true,
            feature_sets: vec![("a".into(), vec![0, 2])],
            ..huber(3, 1)
        })
        .is_ok()
    );
}

fn scales(step: &Step) -> Vec<f64> {
    match &step.extra {
        Some(crate::Extra::Huber { scale }) => scale.clone(),
        other => panic!("no scale: {other:?}"),
    }
}

/// The emitted scale is the one the row's cut is drawn in: each target's
/// residual scale after the row's decay and before its residual, NaN
/// before there is one. It does not read the row's own target, so a wild
/// value on the row leaves it as it was, and the next row reads the wild
/// residual in it. A row scored and not learned reads it too.
#[test]
fn the_emitted_scale_is_the_cuts_and_reads_no_row_its_own() {
    for scale in [HuberScale::Mad, HuberScale::Rms] {
        let cfg = RobustCfg {
            emit_scale: true,
            refit_rows: 0,
            huber_scale: scale,
            ..huber(3, 2)
        };
        let mut m = Robust::new(cfg.clone()).unwrap();
        let mut quiet = Robust::new(RobustCfg {
            emit_scale: false,
            refit_rows: 0,
            ..cfg
        })
        .unwrap();
        let mut s = 12u64;
        let mut set = 0;
        for i in 0..600 {
            let (x, y, d, w) = row(i, &mut s);
            let before: Vec<f64> = (0..2)
                .map(|j| m.residual_scale(j).unwrap_or(f64::NAN))
                .collect();
            let scored = m.predict(&x, d);
            let got = scales(&m.step(&x, &y, d, w));
            let off = quiet.step(&x, &y, d, w);
            assert!(off.extra.is_none(), "{scale:?}: off is off");
            for j in 0..2 {
                // The scale's decay moves neither a median nor a mean.
                assert_eq!(got[j].to_bits(), before[j].to_bits(), "{scale:?} row {i}");
                assert_eq!(scales(&scored)[j].to_bits(), before[j].to_bits());
                set += usize::from(got[j].is_finite());
            }
            if i == 0 {
                assert!(got.iter().all(|v| v.is_nan()), "no scale before a residual");
            }
        }
        assert!(set > 1100, "{scale:?}: {set}");
        // A wild target on the row: its own scale is unmoved, the next
        // row's is.
        // Under the MAD the wild row moves the median by its weight, which
        // can stay inside the sketch's bucket (`tanh(1/128)` wide), so the
        // evidence it entered is the sketch itself.
        let (x, mut y, d, w) = row(600, &mut s);
        let mut wild = m.clone();
        let tame = scales(&m.step(&x, &y, d, w));
        y[0] = Some(1e6);
        assert_eq!(scales(&wild.step(&x, &y, d, w)), tame, "{scale:?}");
        if scale == HuberScale::Mad {
            assert_ne!(
                m.mad_sketch()[0],
                wild.mad_sketch()[0],
                "the wild residual enters the sketch"
            );
        }
        let (x, y, d, w) = row(601, &mut s);
        let (a, b) = (
            scales(&m.step(&x, &y, d, w)),
            scales(&wild.step(&x, &y, d, w)),
        );
        let moved = if scale == HuberScale::Mad {
            b[0] >= a[0]
        } else {
            b[0] > a[0]
        };
        assert!(
            moved,
            "{scale:?}: the wild residual enters the next row's scale: {a:?} {b:?}"
        );
    }
}

#[test]
fn the_emitted_scale_is_refused_on_a_quantile_fit() {
    let e = Robust::new(RobustCfg {
        emit_scale: true,
        refit_rows: 0,
        loss: RobustLoss::Quantile { tau: 0.5 },
        huber_scale: HuberScale::Rms,
        huber_scale_decay: None,
        ..huber(3, 1)
    })
    .unwrap_err();
    assert!(e.contains("emit_huber_scale is for huber"), "{e}");
}

/// Under the sum scale one feature set is a fit on its columns alone: at
/// `huber_delta = inf`, solved on every row, the set `[1]` of three
/// features is `rls` on the second feature, with the intercept, per
/// target; the columns outside the set have a coefficient of 0
/// (docs/PLAN.md task 272 (m)).
#[test]
fn a_feature_set_under_the_sum_scale_is_rls_on_its_columns() {
    let (k, hl, ridge) = (3usize, 40.0, 0.7);
    let mut hub = Robust::new(RobustCfg {
        loss: RobustLoss::Huber {
            delta: f64::INFINITY,
        },
        decay: Decay::Halflife(hl),
        ridge,
        ridge_scale: true,
        min_weight: 0.0,
        feature_sets: vec![("second".into(), vec![1])],
        ..huber(k, 2)
    })
    .unwrap();
    let mut rls: Vec<crate::Rls> = (0..2)
        .map(|_| {
            crate::Rls::new(crate::RlsCfg {
                n_features: 1,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(hl),
                delta: ridge,
                coef_prior: None,
                min_weight: 0.0,
            })
            .unwrap()
        })
        .collect();
    let mut s = 41u64;
    let (mut worst, mut compared) = (0.0f64, 0);
    for i in 0..300 {
        let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let y0 = 1.5 * x[0] - x[1] + 0.3 + 0.05 * lcg(&mut s);
        let y1 = (i % 7 != 3).then(|| 4.0 + 0.2 * x[2] - 3.0 * x[1] + 0.01 * lcg(&mut s));
        let d = if i == 0 { 0.0 } else { 0.5 + lcg(&mut s).abs() };
        let w = 0.5 + lcg(&mut s).abs();
        let y = [Some(y0), y1];
        let h = hub.step(&x, &y, d, w);
        for j in 0..2 {
            let r = rls[j].step(&[x[1]], &[y[j]], d, w);
            if i > 8 && h.pred[j].is_finite() {
                compared += 1;
                worst = worst.max((h.pred[j] - r.pred[0]).abs());
            }
        }
    }
    assert!(compared > 500, "{compared}");
    assert!(worst < 1e-9, "the set against rls on its column: {worst:e}");
    for beta in hub.coefficients().unwrap() {
        assert_eq!((beta[1], beta[3]), (0.0, 0.0), "{beta:?}");
        assert!(beta[0] != 0.0 && beta[2] != 0.0, "{beta:?}");
    }
}

/// A column that holds no data -- a feature that is 0 on every row -- has
/// none of the data's support: its share `1 − λ (A⁻¹)_jj` is 0, since with
/// no penalty its row of the sum-scale system is the jitter `ε` alone,
/// `(A⁻¹)_jj = 1/ε` and `λ = ε`. The system is singular without that
/// jitter, so every solve takes a rung of the ladder, and each is counted
/// in `solve_failures` (docs/PLAN.md task 272 (m)).
#[test]
fn a_column_with_no_data_has_no_share_under_the_sum_scale() {
    let mut m = Robust::new(RobustCfg {
        loss: RobustLoss::Huber {
            delta: f64::INFINITY,
        },
        ridge: 0.0,
        ridge_scale: true,
        min_weight: 0.0,
        ..huber(2, 1)
    })
    .unwrap();
    let mut s = 13u64;
    let mut solved = 0;
    for i in 0..60 {
        let x = [lcg(&mut s), 0.0];
        let before = m.solve_failures;
        m.step(
            &x,
            &[Some(2.0 - x[0] + 0.01 * lcg(&mut s))],
            f64::from(u8::from(i > 0)),
            1.0,
        );
        if m.coefficients().is_some() {
            solved += 1;
            assert_eq!(m.solve_failures, before + 1, "row {i}: one rung a solve");
        }
    }
    assert!(solved > 50, "{solved}");
    let shares = m.support_coef().unwrap();
    let share = &shares[0];
    assert!(share[0].is_nan(), "the intercept's is no share");
    assert!(share[1] > 0.99 && share[1] <= 1.0, "{share:?}");
    assert!(share[2] < 1e-9, "the empty column: {share:?}");
}
