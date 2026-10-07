//! A damaged `ew_cov` state is refused at `restore`, never read until it
//! panics (review 2026-10-06, CB2 and CB8). A file of its own: `ewcov.rs`
//! is at the repository's 250 KB cap for a source file
//! (`tests/test_repo_hygiene.py`).

use super::*;
use crate::{OnlineModel, State, StateError};

fn cfg(k: usize, stats: Vec<EwCovStat>) -> EwCovCfg {
    EwCovCfg {
        n_features: k,
        decay: crate::Decay::Halflife(10.0),
        stats,
        min_weight: 2.0,
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

/// `n` rows of `k` columns that move, from a clock of 0.
fn fitted(c: EwCovCfg, n: usize) -> EwCovModel {
    let k = c.n_features;
    let mut m = EwCovModel::new(c).unwrap();
    for i in 0..n {
        let x: Vec<f64> = (0..k).map(|j| ((i * (j + 3)) % 7) as f64).collect();
        m.step(&x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m
}

/// The state with one JSON edit, read back.
fn edited(m: &EwCovModel, key: &str, edit: &mut dyn FnMut(&mut serde_json::Value)) -> State {
    let mut v = serde_json::to_value(m.state()).unwrap();
    crate::window::json_edit(&mut v, key, edit);
    serde_json::from_value(v).unwrap()
}

/// Refused as damaged, naming the part, or -- what the old build did --
/// loaded, in which case the next row is stepped to show what that read.
fn assert_refused(what: &str, state: &State, says: &str) {
    match EwCovModel::restore(state) {
        Err(StateError::Invalid(e)) => assert!(e.contains(says), "{what}: {e}"),
        Ok(mut back) => {
            let k = back.n_features();
            back.step(&vec![1.0; k], &[], 1.0, 1.0);
            let _ = back.predict(&vec![1.0; k], 1.0);
            panic!("{what}: loaded and was read");
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

/// The Mahalanobis scores' sketch: a level pointer outside the buckets
/// loaded and the next row's `add` indexed `buckets[k - lo]` far past them;
/// pointers or weights below them a level short, a bucket weight below 0,
/// a scale of 0 loaded too (CB2).
#[test]
fn a_damaged_sketch_is_refused() {
    let mut c = cfg(2, vec![EwCovStat::Mahal]);
    c.precision_prior = Some(1e-6);
    c.mahal_quantiles = vec![0.5, 0.9];
    let m = fitted(c, 30);
    assert!(
        EwCovModel::restore(&m.state()).is_ok(),
        "the state as saved"
    );
    type Damage<'a> = (&'a str, &'a str, &'a dyn Fn(&mut serde_json::Value));
    let damage: [Damage; 5] = [
        ("a pointer far below the buckets", "at", &|x| {
            *x = serde_json::json!([-60_000, -60_000]);
        }),
        ("a pointer a level short", "at", &|x| {
            x.as_array_mut().unwrap().pop();
        }),
        ("a weight below a level short", "below", &|x| {
            x.as_array_mut().unwrap().pop();
        }),
        ("a bucket weight below 0", "buckets", &|x| {
            x.as_array_mut().unwrap()[0] = serde_json::json!(-1.0);
        }),
        ("a scale of 0", "scale", &|x| *x = serde_json::json!(0.0)),
    ];
    for (what, key, f) in damage {
        let state = edited(&m, key, &mut |x| f(x));
        assert_refused(what, &state, "wrong shape");
    }
}

/// A window's snapshots: one short of a mean or a co-moment loaded and the
/// next `predict` read `old.m[2]` of a two-entry vector in
/// `crate::truncated`; one whose weight is not a number loaded too (CB2).
#[test]
fn a_window_snapshot_of_the_wrong_shape_is_refused() {
    let mut c = cfg(3, vec![EwCovStat::Mean, EwCovStat::Var]);
    c.window = Some(5.0);
    let m = fitted(c, 20);
    assert!(
        EwCovModel::restore(&m.state()).is_ok(),
        "the state as saved"
    );
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut crate::Moments));
    let damage: [Damage; 3] = [
        ("a mean short", &|s| {
            s.m.pop();
        }),
        ("a co-moment short", &|s| {
            s.c.pop();
        }),
        ("a weight that is not a number", &|s| s.w = f64::NAN),
    ];
    for (what, f) in damage {
        let mut state = m.state();
        let crate::ModelState::EwCovModel(inner) = &mut state.model else {
            unreachable!()
        };
        inner.win.as_mut().unwrap().snaps.iter_mut().for_each(f);
        assert_refused(what, &state, "wrong shape");
    }
}

/// The cfg's check, which a `restore` runs, reads a lag without reserving a
/// ring of it: built through `EwLagCov::new`, a lag of `usize::MAX` in a
/// damaged state asked for a ring that size and panicked on its capacity,
/// where the load had read the state (review 2026-10-06, CF4). Whatever a
/// ceiling on lags makes of it, a load answers.
#[test]
fn a_lag_past_any_ring_is_read_without_reserving_one() {
    let mut c = cfg(2, vec![EwCovStat::Corr, EwCovStat::LagCorr]);
    c.lags = vec![1, 3];
    let m = fitted(c, 20);
    let mut v = serde_json::to_value(m.state()).unwrap();
    let huge = serde_json::json!([1, usize::MAX]);
    crate::window::json_edit(&mut v, "lags", &mut |x| *x = huge.clone());
    let state: State = serde_json::from_value(v).unwrap();
    let loaded = std::panic::catch_unwind(|| EwCovModel::restore(&state).map(drop));
    assert!(loaded.is_ok(), "the load panicked");
}

/// The lag ring holds the last `max(lags)` learned rows: a deeper one
/// loaded and was carried for ever, `update` popping one row a push
/// (CB8).
#[test]
fn a_lag_ring_deeper_than_its_lags_is_refused() {
    let mut c = cfg(2, vec![EwCovStat::Corr, EwCovStat::LagCorr]);
    c.lags = vec![1, 3];
    let m = fitted(c, 20);
    assert!(
        EwCovModel::restore(&m.state()).is_ok(),
        "the state as saved"
    );
    let state = edited(&m, "ring", &mut |x| {
        let ring = x.as_array_mut().unwrap();
        assert_eq!(ring.len(), 3, "the ring is as deep as the deepest lag");
        ring.push(ring[2].clone());
    });
    match EwCovModel::restore(&state) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
        Ok(back) => panic!(
            "a ring of 4 rows under lags [1, 3] loaded: {}",
            back.lag.unwrap().depth()
        ),
        Err(e) => panic!("{e}"),
    }
}
