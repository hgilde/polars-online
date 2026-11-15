//! `ewridge`'s solves form their lists in a kept scratch (task 415): the
//! combos, readers and feature indices, the system and its right-hand
//! sides and the solution, which the thread keeps, and the fit's storage
//! and the pending solves' `Arc`s, which the model does, each written over
//! by the next solve. Held here to a model whose scratch is emptied before
//! every row, which forms each list afresh as the solves did before: every prediction, every state
//! byte and every row's inflation read against the kept systems are the
//! same to the bit.

use super::*;

/// A seeded generator in `[-1, 1)`.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A configuration of `k` features and three targets, a solve on every row.
fn cfg(k: usize, shape: (bool, bool, bool), sets: bool, window: Option<f64>) -> EwRidgeCfg {
    let (fit_intercept, standardize, ridge_scale) = shape;
    EwRidgeCfg {
        n_features: k,
        n_targets: 3,
        fit_intercept,
        decay: Decay::Halflife(40.0),
        ridge: if sets { vec![1e-3, 1.0] } else { vec![1e-3] },
        feature_sets: if sets {
            vec![
                ("a".to_string(), vec![0, 1]),
                ("b".to_string(), (1..k).collect()),
            ]
        } else {
            vec![]
        },
        standardize,
        ridge_scale,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// What a model keeps besides its state: emptied, the solves form what they
/// work in afresh.
fn empty_the_scratch(m: &mut EwRidge) {
    drop(SOLVE_BUFS.take());
    m.spare_fit = crate::window::Kept::default();
    m.ready.spare = crate::window::Kept::default();
    m.factors.work = None;
}

#[test]
fn solves_in_a_kept_scratch_are_the_ones_in_a_fresh_scratch() {
    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
    for (k, rows) in [(3usize, 300), (70, 90)] {
        for shape in [
            (true, false, false),
            (true, true, false),
            (true, false, true),
            (false, false, false),
            (false, true, false),
        ] {
            for keep_factor in [false, true] {
                // The sum scale takes neither a window nor a grid (`validate`),
                // and feature sets and windows are tried where they are cheap.
                for (sets, window) in [(false, None), (true, None), (false, Some(30.0))] {
                    if (shape.2 && (sets || window.is_some()))
                        || (k > 3 && (sets || window.is_some()))
                    {
                        continue;
                    }
                    let case = format!(
                        "k {k}, shape {shape:?}, keep_factor {keep_factor}, sets {sets}, \
                         window {window:?}"
                    );
                    let mut kept = EwRidge::new(cfg(k, shape, sets, window)).unwrap();
                    kept.set_keep_factor(keep_factor);
                    let mut fresh = kept.clone();
                    let mut s = 41u64;
                    for i in 0..rows {
                        let x: Vec<f64> = (0..k)
                            .map(|j| {
                                // A column with no spread, which the standardiser drops.
                                if j == 2 && k == 3 {
                                    1.0
                                } else {
                                    2.0 * lcg(&mut s) + j as f64
                                }
                            })
                            .collect();
                        let y: Vec<Option<f64>> = (0..3)
                            .map(|t| {
                                let v = 1.0
                                    + x.iter()
                                        .enumerate()
                                        .map(|(j, xj)| (j + t + 1) as f64 * 0.3 * xj)
                                        .sum::<f64>()
                                    + 0.2 * lcg(&mut s);
                                (t == 0 || i % 3 != t).then_some(v)
                            })
                            .collect();
                        let w = 0.5 + (lcg(&mut s) + 1.0);
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        // The kept model reads the lists the fresh one's last
                        // solve left on the thread, another model's; the fresh
                        // one forms its own.
                        let a = kept.step(&x, &y, d, w);
                        empty_the_scratch(&mut fresh);
                        let b = fresh.step(&x, &y, d, w);
                        assert_eq!(bits(&a.pred), bits(&b.pred), "{case}, row {i}");
                        let (mut ra, mut rb) = (Vec::new(), Vec::new());
                        kept.row_error_inflation_into(&x, 1.0, &mut ra);
                        fresh.row_error_inflation_into(&x, 1.0, &mut rb);
                        assert_eq!(bits(&ra), bits(&rb), "{case}, row {i}: inflation");
                        if i % 53 == 52 {
                            kept.settle_readiness();
                            fresh.settle_readiness();
                            // The pending solves it kept are hollow: no factor.
                            assert!(
                                kept.ready
                                    .spare
                                    .get()
                                    .0
                                    .iter()
                                    .all(|p| p.factor.order() == 0),
                                "{case}: a spare holds a factor"
                            );
                        }
                    }
                    assert_eq!(
                        format!("{:?}", kept.state()),
                        format!("{:?}", fresh.state()),
                        "{case}"
                    );
                    assert_eq!(
                        rmp_serde::to_vec_named(&kept.state()).unwrap(),
                        rmp_serde::to_vec_named(&fresh.state()).unwrap(),
                        "{case}: the state's bytes"
                    );
                    // A copy and a debug print show none of the scratch.
                    assert!(!format!("{kept:?}").contains("SolveBufs"), "{case}");
                    assert!(kept.clone().ready.spare.get().0.is_empty(), "{case}");
                }
            }
        }
    }
}

/// What the solves keep is bounded: no more pending solves than
/// [`Spare::KEPT`], and no list past [`KEEP_ENTRIES`] -- a model of order
/// 70 forms its system afresh, as it did.
#[test]
fn the_scratch_is_bounded() {
    let mut m = EwRidge::new(cfg(70, (true, false, false), false, None)).unwrap();
    let mut s = 3u64;
    for i in 0..90 {
        let x: Vec<f64> = (0..70).map(|_| lcg(&mut s)).collect();
        let y = [Some(x[0]), Some(x[1]), Some(x[2])];
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m.settle_readiness();
    assert!(m.ready.spare.get().0.len() <= Spare::KEPT);
    let bufs = SOLVE_BUFS.take().expect("the thread keeps its lists");
    let parts = &bufs.parts;
    for (what, cap) in [("a", parts.a.capacity()), ("out", parts.out.capacity())] {
        assert!(cap <= KEEP_ENTRIES, "{what}: {cap} entries kept");
    }
}
