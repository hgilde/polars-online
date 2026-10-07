//! The rule for a value that is not usable, held against `ew_cov`'s lag
//! ring (`OnlineModel`, docs/PLAN.md task 183). A file of its own because
//! `ewcov.rs` stands at the source-size cap (`tests/test_repo_hygiene.py`).

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A row with a value that is not usable takes no slot in the lag ring, as a
/// row of weight 0 takes none: the rows either side of it are adjacent in
/// the ring, as they are when the plumbing skips it, and the lag matrices
/// age over it and learn nothing from it. It took a slot, its NaN paired
/// with every later row up to the deepest lag, and the matrices never left
/// the NaN. A feature that is not a number, one past the input bound, and a
/// weight that is not usable alike.
#[test]
fn a_refused_row_takes_no_slot_in_the_lag_ring() {
    let cfg = EwCovCfg {
        n_features: 2,
        decay: crate::Decay::Halflife(20.0),
        stats: vec![EwCovStat::Mean, EwCovStat::LagCorr],
        min_weight: 2.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: vec![1, 2],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    let ring =
        |m: &EwCovModel| serde_json::to_value(m.lag.as_ref().unwrap()).unwrap()["ring"].clone();
    for (bad, w) in [
        ([f64::NAN, 0.5], 1.0),
        ([0.5, 2.0 * crate::INPUT_BOUND], 1.0),
        ([0.5, 0.25], f64::INFINITY),
    ] {
        let case = format!("{bad:?} at weight {w}");
        let (mut with, mut without) = (
            EwCovModel::new(cfg.clone()).unwrap(),
            EwCovModel::new(cfg.clone()).unwrap(),
        );
        let mut s = 23u64;
        for i in 0..12 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut with, &x, &[], d, 1.0);
            crate::OnlineModel::step(&mut without, &x, &[], d, 1.0);
        }
        let before = ring(&with);
        let lagged = with.lag.as_ref().unwrap().comoments().to_vec();
        let out = crate::OnlineModel::step(&mut with, &bad, &[], 1.0, w);
        assert_eq!(ring(&with), before, "{case}: the ring took the row");
        let lam = with.cfg.decay.factor(1.0);
        assert_eq!(with.n_eff(), lam * without.n_eff(), "{case}");
        assert_eq!(with.lag.as_ref().unwrap().comoments(), lagged, "{case}");
        assert!(
            w.is_finite() || out.pred.iter().all(|v| v.is_finite()),
            "{case}"
        );
        let x = [0.3, -0.7];
        crate::OnlineModel::step(&mut with, &x, &[], 1.0, 1.0);
        crate::OnlineModel::step(&mut without, &x, &[], 2.0, 1.0);
        assert_eq!(ring(&with), ring(&without), "{case}: the rows either side");
        let (a, b) = (
            with.lag.as_ref().unwrap().comoments(),
            without.lag.as_ref().unwrap().comoments(),
        );
        for (u, v) in a.iter().zip(b) {
            assert!(
                u.is_finite() && (u - v).abs() <= 1e-12 * (1.0 + v.abs()),
                "{case}: {u} against {v}"
            );
        }
    }
}
