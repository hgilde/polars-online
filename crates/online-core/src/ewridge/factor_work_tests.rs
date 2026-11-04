//! `ewridge`'s solves factorize in a kept working space (task 301): faer's
//! scratch and the factors the solves have done with, written over by the
//! next factorization of their order. Held here to a model whose working
//! space is emptied before every row, which factorizes afresh as the
//! solves did before.

use super::*;

/// A seeded generator in `[-1, 1)`.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 3;

/// The factors `m`'s working space keeps to be written over.
fn spares(m: &EwRidge) -> usize {
    m.factors.work.as_ref().map_or(0, |w| w.kept())
}

/// Two targets over three features, the second absent on every third row,
/// so its own rows split it into a Gram of its own; a solve on every row.
fn cfg(
    fit_intercept: bool,
    standardize: bool,
    ridge_scale: bool,
    window: Option<f64>,
) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept,
        decay: Decay::Halflife(40.0),
        ridge: vec![1e-3],
        feature_sets: vec![],
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

/// The factors built in the kept working space are the ones a model that
/// keeps none builds, to the bit: every prediction, every row's inflation
/// read against the kept systems, every data share and the state's
/// `Debug`, which shows none of the working space -- centred, standardized,
/// on the sum scale and through the origin, with the systems kept for the
/// row inflation and without, with a window and without, across a settle
/// every 97 rows, which drops the working space as the bank's end of a
/// chunk does. The kept model writes over a factor a solve dropped on most
/// rows.
#[test]
fn factors_built_in_the_kept_working_space_are_the_fresh_ones() {
    for (fit_intercept, standardize, ridge_scale) in [
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (false, false, false),
        (false, true, false),
    ] {
        for keep_factor in [false, true] {
            // A window and the sum scale do not combine (`validate`).
            for window in [None, Some(30.0)]
                .into_iter()
                .filter(|w| !ridge_scale || w.is_none())
            {
                let case = format!(
                    "intercept {fit_intercept}, standardize {standardize}, \
                     ridge_scale {ridge_scale}, keep_factor {keep_factor}, window {window:?}"
                );
                let mut kept =
                    EwRidge::new(cfg(fit_intercept, standardize, ridge_scale, window)).unwrap();
                kept.set_keep_factor(keep_factor);
                let mut fresh = kept.clone();
                let (mut written_over, mut most) = (0, 0);
                let mut s = 29u64;
                let rows = 400;
                for i in 0..rows {
                    let x: Vec<f64> = (0..K).map(|j| 2.0 * lcg(&mut s) + j as f64).collect();
                    let y: Vec<Option<f64>> = (0..2)
                        .map(|t| {
                            let v = 1.0
                                + x.iter()
                                    .enumerate()
                                    .map(|(j, xj)| (j + t + 1) as f64 * 0.3 * xj)
                                    .sum::<f64>()
                                + 0.2 * lcg(&mut s);
                            (t == 0 || i % 3 != 1).then_some(v)
                        })
                        .collect();
                    let w = 0.5 + (lcg(&mut s) + 1.0);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    fresh.factors.work = None;
                    written_over += usize::from(spares(&kept) > 0);
                    most = most.max(spares(&kept));
                    let a = kept.step(&x, &y, d, w);
                    let b = fresh.step(&x, &y, d, w);
                    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&a.pred), bits(&b.pred), "{case}, row {i}");
                    let (mut ra, mut rb) = (Vec::new(), Vec::new());
                    kept.row_error_inflation_into(&x, 1.0, &mut ra);
                    fresh.row_error_inflation_into(&x, 1.0, &mut rb);
                    assert_eq!(bits(&ra), bits(&rb), "{case}, row {i}: inflation");
                    if i % 97 == 96 {
                        kept.settle_readiness();
                        fresh.settle_readiness();
                        assert_eq!(spares(&kept), 0, "{case}: a settle keeps none");
                    }
                }
                assert!(
                    written_over > rows * 3 / 4,
                    "{case}: {written_over} of {rows}"
                );
                // Two Grams, and under `keep_factor` a copy of each solve's
                // factor: the working space keeps no more than the solves
                // gave back.
                assert!(most <= 4, "{case}: {most} factors kept");
                assert_eq!(
                    format!("{:?}", kept.state()),
                    format!("{:?}", fresh.state()),
                    "{case}"
                );
                let shown = format!("{kept:?}");
                assert!(
                    shown.contains("factors: Factors(") && !shown.contains("FactorWork"),
                    "{case}"
                );
            }
        }
    }
}

/// A load rebuilds the kept factors in the working space, as a fresh
/// factorization of each system the state carries gives them: the row
/// inflation the loaded model reads is the saved model's, to the bit, and
/// a second rebuild writes over the factors of the first.
#[test]
fn a_load_rebuilds_the_kept_factors_in_the_working_space() {
    let mut m = EwRidge::new(cfg(true, true, false, None)).unwrap();
    m.set_keep_factor(true);
    let mut s = 5u64;
    for i in 0..60 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s)).collect();
        let y = [Some(x[0] - x[2] + 0.1 * lcg(&mut s)), Some(x[1])];
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let x = [0.3, -0.4, 0.8];
    let mut saved = Vec::new();
    m.row_error_inflation_into(&x, 1.0, &mut saved);
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let mut back = EwRidge::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    let mut loaded = Vec::new();
    back.row_error_inflation_into(&x, 1.0, &mut loaded);
    assert!(saved.iter().all(|v| v.is_finite()), "{saved:?}");
    assert_eq!(
        saved.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        loaded.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let held = back.factors.kept.iter().flatten().count();
    assert!(held > 0);
    back.refactor();
    assert_eq!(spares(&back), 0, "the first rebuild's factors written over");
    let mut again = Vec::new();
    back.row_error_inflation_into(&x, 1.0, &mut again);
    assert_eq!(loaded, again);
}

/// A settle with nothing pending does not look at the working space, so a
/// solve that leaves nothing pending must keep no spare itself: here the
/// solve past a gap longer than the window, whose empty window clears every
/// slot and gives back the factors they held.
#[test]
fn a_solve_that_leaves_nothing_pending_keeps_no_spare() {
    let mut m = EwRidge::new(cfg(true, false, false, Some(5.0))).unwrap();
    let mut s = 11u64;
    let mut row = |m: &mut EwRidge, d: f64, w: f64| {
        let x: Vec<f64> = (0..K).map(|j| 2.0 * lcg(&mut s) + j as f64).collect();
        m.step(&x, &[Some(x[0]), Some(x[1])], d, w);
    };
    for i in 0..20 {
        row(&mut m, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(
        m.pending_readiness() > 0,
        "the rows solve and nothing settled"
    );
    assert!(spares(&m) > 0, "the solves' replaced factors are kept");
    // A gap past the window with no weight learns nothing and solves on no
    // schedule, so the solve on the empty window is called here.
    row(&mut m, 100.0, 0.0);
    m.solve();
    assert_eq!(
        m.pending_readiness(),
        0,
        "the empty window clears every slot"
    );
    assert_eq!(spares(&m), 0);
    // And a settle after it finds nothing to do, and a later solve still runs.
    m.settle_readiness();
    for _ in 0..3 {
        row(&mut m, 1.0, 1.0);
    }
    assert!(m.pending_readiness() > 0);
}

/// What one system's solves keep between them (task 316, review 11's
/// S3-2): one spare factor, the pending solve's that the next solve
/// replaced, and under `keep_factor` a second, the kept factor the next
/// solve replaced too. One Gram here, both targets on every row, a solve
/// on every row: after the second solve the working space keeps one factor
/// without `keep_factor` and two with it, and no more on the rows after.
#[test]
fn a_system_keeps_one_spare_factor_and_two_under_keep_factor() {
    for (keep_factor, want) in [(false, 1), (true, 2)] {
        let mut m = EwRidge::new(cfg(true, true, false, None)).unwrap();
        m.set_keep_factor(keep_factor);
        let mut s = 7u64;
        for i in 0..12 {
            let x: Vec<f64> = (0..K).map(|j| 2.0 * lcg(&mut s) + j as f64).collect();
            let y = [Some(x[0] - x[2] + 0.1 * lcg(&mut s)), Some(x[1])];
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i >= 6 {
                assert_eq!(
                    spares(&m),
                    want,
                    "keep_factor {keep_factor}, row {i}: the spares kept"
                );
            }
        }
    }
}
