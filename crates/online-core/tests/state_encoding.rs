//! Every model must round-trip through **both** msgpack encodings, with each
//! of its optional parts present and absent.
//!
//! `rmp_serde` writes a struct two ways: `to_vec_named` as a map of field
//! name to value, and `to_vec` (compact) as a bare array, position only. The
//! bank's state files are named, so a new field with a `default` is
//! backward-compatible there. The compact form has no such slack: a
//! `skip_serializing_if` field simply disappears from the array and
//! everything after it slides down one. So **at most one field may skip, and
//! it must be last** -- otherwise a state with the skipped field absent and
//! a later field present decodes the later field into the earlier field's
//! slot, and the failure is a type error at a distance.
//!
//! This is not hypothetical: E66's lag moments were added with a skip in
//! front of the window's, and a marginal with a window but no lags stopped
//! round-tripping compactly. Nothing shipped broke, because bank files are
//! named -- which is exactly why a test is needed rather than a habit.
//!
//! The rule is crate-wide; `marginal` is where it is checked, because it is
//! the model with the most optional parts (lags, bins and a window) and so
//! the only one that can get the ordering wrong in more than one way.

use online_core::{Decay, MarginalCfg, OnlineModel};

/// Both encodings, or the reason one failed.
fn roundtrip<T>(m: &T, what: &str)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let compact = rmp_serde::to_vec(m).unwrap();
    let named = rmp_serde::to_vec_named(m).unwrap();
    let from_compact: T = rmp_serde::from_slice(&compact)
        .unwrap_or_else(|e| panic!("{what}: compact encoding does not round-trip: {e}"));
    let from_named: T = rmp_serde::from_slice(&named)
        .unwrap_or_else(|e| panic!("{what}: named encoding does not round-trip: {e}"));
    assert_eq!(
        &from_compact, m,
        "{what}: compact round-trip changed the state"
    );
    assert_eq!(&from_named, m, "{what}: named round-trip changed the state");
}

#[test]
fn marginal_round_trips_with_every_optional_part_present_or_absent() {
    for lags in [vec![], vec![1usize, 2]] {
        for window in [None, Some(50.0)] {
            for bins in [None, Some(4usize)] {
                // A window and bins are refused together (a snapshot of the
                // histogram would be bins times the size of one), so that
                // pairing has no state to encode.
                if window.is_some() && bins.is_some() {
                    continue;
                }
                let cfg = MarginalCfg {
                    n_features: 2,
                    n_targets: 1,
                    decay: Decay::Halflife(10.0),
                    min_periods: vec![0.0],
                    lags: lags.clone(),
                    serial_rule: None,
                    bins: bins.map(|n| {
                        Box::new(online_core::BinCfg {
                            n_bins: n,
                            edges: None,
                            rule: online_core::BinRule::Quantile,
                            warm_rows: 4,
                        })
                    }),
                    window,
                    window_every: window.map(|_| 1),
                };
                let mut m = online_core::Marginal::new(cfg).unwrap();
                // Bins have two states worth encoding: the warm-up hold with
                // rows in it (3 of the 4 it waits for), and the histogram
                // it becomes. Both must survive, or a bank saved during the
                // warm-up loses the rows it was holding.
                for i in 0..3 {
                    let v = i as f64;
                    OnlineModel::step(&mut m, &[v, -v], &[Some(v * 0.5)], v, 1.0);
                }
                roundtrip(
                    &m,
                    &format!("marginal lags={lags:?} window={window:?} bins={bins:?} (held)"),
                );
                for i in 3..8 {
                    let v = i as f64;
                    OnlineModel::step(&mut m, &[v, -v], &[Some(v * 0.5)], v, 1.0);
                }
                roundtrip(
                    &m,
                    &format!("marginal lags={lags:?} window={window:?} bins={bins:?}"),
                );
            }
        }
    }
}

/// A `humanfloat`-annotated field must be **byte-identical** to a bare `f64`
/// in msgpack, and tagged in JSON.
///
/// This is the whole safety argument for `crate::humanfloat`. The helpers
/// exist so a JSON export can carry `NaN` and `±inf`, which `serde_json`
/// otherwise writes as `null`; they key on `is_human_readable()`, which
/// msgpack reports as `false`. If that were ever untrue -- a serde change, a
/// different msgpack crate -- every state file ever written would stop
/// loading, silently, because the field would decode as a string. So the
/// bytes are pinned here rather than assumed.
#[test]
fn a_human_readable_float_does_not_move_the_msgpack() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Bare {
        a: f64,
        b: Vec<f64>,
        c: Option<f64>,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Tagged {
        #[serde(with = "online_core::humanfloat::f64_or_tag")]
        a: f64,
        #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
        b: Vec<f64>,
        #[serde(with = "online_core::humanfloat::opt_f64_or_tag")]
        c: Option<f64>,
    }

    for (a, b, c) in [
        (1.5, vec![2.0, 3.0], Some(4.0)),
        (
            f64::INFINITY,
            vec![f64::NEG_INFINITY, 1.0],
            Some(f64::INFINITY),
        ),
        (f64::NAN, vec![f64::NAN], None),
        (-0.0, vec![], Some(-0.0)),
    ] {
        let bare = Bare { a, b: b.clone(), c };
        let tagged = Tagged { a, b: b.clone(), c };

        for (what, bare_bytes, tag_bytes) in [
            (
                "named",
                rmp_serde::to_vec_named(&bare).unwrap(),
                rmp_serde::to_vec_named(&tagged).unwrap(),
            ),
            (
                "compact",
                rmp_serde::to_vec(&bare).unwrap(),
                rmp_serde::to_vec(&tagged).unwrap(),
            ),
        ] {
            assert_eq!(
                bare_bytes, tag_bytes,
                "{what} msgpack moved for a={a} b={b:?} c={c:?}"
            );
        }

        // and it reads back, through msgpack and through JSON
        let back: Tagged =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&tagged).unwrap()).unwrap();
        let json = serde_json::to_string(&tagged).unwrap();
        let from_json: Tagged = serde_json::from_str(&json).unwrap();
        for (label, got) in [("msgpack", &back), ("json", &from_json)] {
            assert_eq!(got.a.is_nan(), a.is_nan(), "{label}: NaN-ness of a");
            if !a.is_nan() {
                assert_eq!(got.a, a, "{label}: a");
            }
            assert_eq!(got.b.len(), b.len(), "{label}: b length");
            assert_eq!(got.c.is_some(), c.is_some(), "{label}: c presence");
        }

        // JSON says the non-finite ones out loud rather than writing null
        let json = serde_json::to_string(&tagged).unwrap();
        if !a.is_finite() {
            assert!(
                json.contains("\"inf\"") || json.contains("\"-inf\"") || json.contains("\"nan\""),
                "json did not tag a non-finite: {json}"
            );
            // `"c":null` is legal -- that is `Option::None`. `a` is a plain
            // `f64`, so a null there is the loss this module exists to stop.
            assert!(
                !json.contains("\"a\":null"),
                "json wrote a plain f64 as null: {json}"
            );
        }
    }
}
