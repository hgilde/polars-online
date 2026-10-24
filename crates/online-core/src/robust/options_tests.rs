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
    let mut with_sets = Robust::new(RobustCfg {
        feature_sets: sets.clone(),
        ..huber(3, 2)
    })
    .unwrap();
    let mut plain = Robust::new(huber(3, 2)).unwrap();
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
