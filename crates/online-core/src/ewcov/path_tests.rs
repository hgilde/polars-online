//! Task 252: the accumulator's and the model's paths no other test
//! reached, each driven and held to what it produces. A file of its own:
//! `ewcov.rs` is near the repository's 250 KB cap for a source file
//! (`tests/test_repo_hygiene.py`).

use super::*;
use crate::{OnlineModel, Stamp};

fn cfg(k: usize, stats: Vec<EwCovStat>) -> EwCovCfg {
    EwCovCfg {
        bias: false,
        n_features: k,
        decay: crate::Decay::Halflife(10.0),
        stats,
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// A window beside the Mahalanobis quantiles, and a snapshot cadence with
/// no window, are each refused by name.
#[test]
fn a_window_beside_quantiles_or_a_cadence_without_a_window_is_refused() {
    let e = EwCovCfg {
        precision_prior: Some(1e-3),
        mahal_quantiles: vec![0.9],
        window: Some(10.0),
        ..cfg(2, vec![EwCovStat::Mahal])
    }
    .validate()
    .unwrap_err();
    assert!(
        e.contains("ew_cov: window_size and mahal_quantiles do not combine"),
        "{e}"
    );
    let e = EwCovCfg {
        window_every: Some(1.0),
        ..cfg(2, vec![EwCovStat::Mean])
    }
    .validate()
    .unwrap_err();
    assert_eq!(e, "ew_cov: snapshot_every needs `window_size`");
    assert!(
        EwCovModel::new(EwCovCfg {
            max_rows_between_snapshots: Some(3),
            ..cfg(2, vec![EwCovStat::Mean])
        })
        .is_err()
    );
}

/// A stamp handed to a windowed model keys its next learned row; a window
/// edge handed to a model with no window changes nothing.
#[test]
fn a_stamp_keys_a_windowed_row_and_an_edge_without_a_window_is_nothing() {
    let mut m = EwCovModel::new(EwCovCfg {
        window: Some(5.0),
        ..cfg(2, vec![EwCovStat::Mean])
    })
    .unwrap();
    m.step(&[0.5, 1.0], &[], 0.0, 1.0);
    let stamp = Stamp::Raw(42.0, 0.5);
    m.stamp_next(stamp);
    let win = m.win.as_ref().expect("windowed");
    assert_eq!(win.snaps.next_at(win.clock, 1.0).stamp, stamp);

    let mut plain = EwCovModel::new(cfg(2, vec![EwCovStat::Mean])).unwrap();
    plain.step(&[0.5, 1.0], &[], 0.0, 1.0);
    let before = rmp_serde::to_vec(&plain.state()).unwrap();
    plain.set_window_closed(crate::WindowClosed::Both);
    assert_eq!(rmp_serde::to_vec(&plain.state()).unwrap(), before);
}

/// An accumulator with no sum of squared weights -- moments set without
/// one, as a state written before that sum was kept -- ages its weight and
/// keeps reporting no Kish size.
#[test]
fn an_accumulator_without_a_kish_sum_ages_its_weight_alone() {
    let mut ew = EwCov::new(2);
    ew.set_moments(&[1.0, 2.0], &[1.0, 0.25, 0.25, 2.0], 4.0, None);
    ew.decay(0.5);
    assert_eq!(ew.n_eff(), 2.0);
    assert_eq!(ew.q_sum(), None);
    assert_eq!(ew.n_kish(), None);
    assert_eq!(ew.mean(0), 1.0, "a decay moves no mean");
}

/// A blocked accumulator handed a first row of no weight holds nothing:
/// the row would add no weight and no decay, so the block stays empty and
/// the weight 0 (hard rule 9).
#[test]
fn a_blocked_accumulator_holds_no_row_of_no_weight_at_the_start() {
    let mut ew = EwCov::new(2);
    ew.set_block_rows(4);
    ew.update(&[1.0, 2.0], 0.9, 0.0);
    assert!(!ew.has_pending());
    assert_eq!(ew.n_eff(), 0.0);
    ew.update(&[1.0, 2.0], 0.9, 1.0);
    assert!(ew.has_pending(), "a row of weight is held");
}

/// A held block a damaged file left -- rows of the wrong width, or no
/// weight to merge them into -- is dropped at the merge, leaving the moments
/// where they stood before the block: the rows cannot be trusted. Release
/// builds only: a debug build asserts the block's shape and weight first,
/// as a caller's error.
#[test]
fn a_damaged_held_block_is_dropped_at_the_merge() {
    if cfg!(debug_assertions) {
        return;
    }
    let mut ew = EwCov::new(2);
    ew.update(&[1.0, 2.0], 1.0, 1.0);
    ew.update(&[3.0, 1.0], 1.0, 1.0);
    let (m0, c0) = (ew.means().to_vec(), ew.comoments().to_vec());
    ew.set_block_rows(4);
    ew.update(&[5.0, -1.0], 1.0, 1.0);
    ew.update(&[7.0, 0.0], 1.0, 1.0);
    assert!(ew.has_pending());
    for (key, what) in [("x", "a row short"), ("w_sum", "no weight")] {
        let mut v = serde_json::to_value(&ew).unwrap();
        match key {
            "x" => {
                let x = v["pending"]["x"].as_array_mut().unwrap();
                x.pop();
            }
            _ => v["w_sum"] = serde_json::json!(0.0),
        }
        let mut damaged: EwCov = serde_json::from_value(v).unwrap();
        damaged.flush();
        assert!(!damaged.has_pending(), "{what}");
        assert_eq!(damaged.means(), m0.as_slice(), "{what}");
        assert_eq!(damaged.comoments(), c0.as_slice(), "{what}");
    }
}
