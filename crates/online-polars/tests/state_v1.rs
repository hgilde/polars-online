//! Backward compatibility with version 1 bank state files (ENHANCEMENTS E10).
//!
//! CLAUDE.md hard rule 5 says a schema change must keep a loader for the
//! previous version. That was argued from the encoding rather than exercised,
//! because after the v2 change there is no v1 *writer* left to produce a
//! fixture. So the fixture is frozen here as bytes instead: a real v1 file,
//! generated once from the v1 layout, and never regenerated. (It is a hex
//! constant rather than a checked-in binary because of hard rule 1 — no data
//! files in the repo.)
//!
//! v1 differs from v2 in exactly two ways:
//!   * no `format_version` field (it defaults to 1);
//!   * group keys are plain strings, where v2 uses a nullable key — which is
//!     why v1 could not tell a null group from one literally named `"<null>"`.
//!
//! The `StreamState` fields added later (`resid_var`, `resid_w`, `drift`) are
//! absent too, and must default cleanly.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// A v1 bank state: `ew_ridge` over one feature, grouped, after 60 rows of a
/// deterministic stream. See the module docs for why this is frozen bytes.
const V1_STATE_HEX: &str = concat!(
    "85a56d61676963b2706f6c6172732d6f6e6c696e652d62616e6bae736368656d615f76657273",
    "696f6e01af7061636b6167655f76657273696f6ea5302e312e30a5737065637391de0019a46e",
    "616d65a16da56d6f64656c8aa474797065a865775f7269646765a57269646765c0ac66656174",
    "7572655f73657473c0ab7374616e64617264697a65c2ab72696467655f6465636179c2a5636f",
    "656630c0ae73657373696f6e5f736872696e6bc0ad6c6f6e675f68616c666c696665c0ab736f",
    "6c76655f6576657279c0b76d61785f726f77735f6265747765656e5f736f6c76657301a77461",
    "726765747391a179a8666561747572657391a27830ad6164645f696e74657263657074c3a563",
    "6c6f636bc0a868616c666c696665cb4049000000000000a36c616dc0aa6d61785f64636c6f63",
    "6bc0ae6f6e5f636c6f636b5f7265736574a36d6178a773657373696f6ec0ab73657373696f6e",
    "5f676170c0a6776569676874c0ab6d696e5f706572696f6473cb4008000000000000aa636f65",
    "665f657665727900aa656d69745f7369676d61c2ac656d69745f72657369645f7ac2aa656d69",
    "745f6472696674c2ab64726966745f64656c7461c0af64726966745f7468726573686f6c64c0",
    "ac64726966745f616374696f6ea4666c6167ad656d69745f6176657261676564c2ab61766572",
    "6167655f657461c0ad656d69745f73656c6563746564c2a567726f7570a167a6737461746573",
    "919292a16183a5636c6f636b84aa707265765f636c6f636bc0ac707265765f73657373696f6e",
    "c0a770656e64696e67cb0000000000000000a773746172746564c3a66d6f64656c739182ae73",
    "6368656d615f76657273696f6e01a56d6f64656c81a7457752696467658ba36366678eaa6e5f",
    "666561747572657301a96e5f7461726765747301ad6164645f696e74657263657074c3a56465",
    "63617981a848616c666c696665cb4049000000000000a5726964676591cb3eb0c6f7a0b5ed8d",
    "ac666561747572655f7365747390ab7374616e64617264697a65c2ab72696467655f64656361",
    "79c2a5636f656630c0ae73657373696f6e5f736872696e6bc0ad6c6f6e675f68616c666c6966",
    "65c0ab6d696e5f706572696f6473cb4008000000000000ab736f6c76655f6576657279cb3ff0",
    "000000000000b76d61785f726f77735f6265747765656e5f736f6c76657301a3636f7688a16b",
    "02a5775f73756dcb4038b6cdf4ef3ee9ab7072696f725f7363616c65cb3fe56826b94393e0a1",
    "6d92cb3ff0000000000000cb3fa6adb90c1a7a18a16394cb0000000000000000cb0000000000",
    "000000cb0000000000000000cb3fdeeb104660b5f0a3696e76c0a9696e765f7072696f72cb00",
    "00000000000000a9696e765f7363616c65cb3ff0000000000000a2776a91cb4038b6cdf4ef3e",
    "e9a1729192cbbfda5491bcf96175cb3fee55c7991b2cd4a47773696791cb4035fb4ed240e651",
    "a47369673291cb3da5369fb0b354efa4626574619192cbbfdfffff3b1a2988cb3ffffffba8b1",
    "1bc9a4736c6f77c0b1636c6f636b5f73696e63655f736f6c7665cb0000000000000000b0726f",
    "77735f73696e63655f736f6c766500ae736f6c76655f6661696c7572657300a9726f77735f73",
    "65656e1e92a16283a5636c6f636b84aa707265765f636c6f636bc0ac707265765f7365737369",
    "6f6ec0a770656e64696e67cb0000000000000000a773746172746564c3a66d6f64656c739182",
    "ae736368656d615f76657273696f6e01a56d6f64656c81a7457752696467658ba36366678eaa",
    "6e5f666561747572657301a96e5f7461726765747301ad6164645f696e74657263657074c3a5",
    "646563617981a848616c666c696665cb4049000000000000a5726964676591cb3eb0c6f7a0b5",
    "ed8dac666561747572655f7365747390ab7374616e64617264697a65c2ab72696467655f6465",
    "636179c2a5636f656630c0ae73657373696f6e5f736872696e6bc0ad6c6f6e675f68616c666c",
    "696665c0ab6d696e5f706572696f6473cb4008000000000000ab736f6c76655f6576657279cb",
    "3ff0000000000000b76d61785f726f77735f6265747765656e5f736f6c76657301a3636f7688",
    "a16b02a5775f73756dcb4038b6cdf4ef3ee9ab7072696f725f7363616c65cb3fe56826b94393",
    "e0a16d92cb3ff0000000000000cb3fa108d795de0cd1a16394cb0000000000000000cb000000",
    "0000000000cb0000000000000000cb3fdf71f88d1ff735a3696e76c0a9696e765f7072696f72",
    "cb0000000000000000a9696e765f7363616c65cb3ff0000000000000a2776a91cb4038b6cdf4",
    "ef3ee9a1729192cbbfdbbdca1a887ccbcb3feefbd49f726d59a47773696791cb4035fb4ed240",
    "e651a47369673291cb3da3c84598985da3a4626574619192cbbfdfffff6e954261cb3ffffffb",
    "bb50df91a4736c6f77c0b1636c6f636b5f73696e63655f736f6c7665cb0000000000000000b0",
    "726f77735f73696e63655f736f6c766500ae736f6c76655f6661696c7572657300a9726f7773",
    "5f7365656e1e",
);

/// The spec is recovered *from* the fixture rather than restated here: a v1
/// file carries its own specs, and duplicating them in the test would only
/// test that two hand-written copies agree.
fn spec_from_fixture() -> Spec {
    Bank::load_bytes(&bytes(), None)
        .expect("v1 fixture should load without spec expectations")
        .specs()[0]
        .clone()
}

/// The same stream the fixture was built from.
fn frame(start: usize, n: usize) -> DataFrame {
    let x: Vec<f64> = (start..start + n).map(|i| (i as f64 * 0.7).sin()).collect();
    let g: Vec<String> = (start..start + n)
        .map(|i| if i % 2 == 0 { "a" } else { "b" }.to_string())
        .collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v - 0.5).collect();
    df!("g" => g, "x0" => x, "y" => y).unwrap()
}

fn bytes() -> Vec<u8> {
    (0..V1_STATE_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&V1_STATE_HEX[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn a_v1_state_file_still_loads() {
    let bank = Bank::load_bytes(&bytes(), None);
    assert!(bank.is_ok(), "v1 state failed to load: {:?}", bank.err());
    // and the spec it carries round-trips through the match check
    let spec = spec_from_fixture();
    assert!(Bank::load_bytes(&bytes(), Some(&[spec])).is_ok());
    // Written before the last learned row travelled with the state
    // (docs/PLAN.md task 34): every group reports a row of nulls.
    let (keys, col) = bank.unwrap().last_row(0, None).unwrap();
    assert_eq!(keys.len(), 2);
    assert_eq!(
        col.null_count(),
        0,
        "a struct row of nulls is not a null struct"
    );
    for f in col.struct_().unwrap().fields_as_series() {
        assert_eq!(f.null_count(), 2, "{}", f.name());
    }
}

/// Written before the data summary travelled with the state (docs/PLAN.md
/// task 35): the file has no count of what its streams were fed, so the
/// summary is null except for what the stream always kept -- and it stays
/// null as the stream goes on, rather than starting a count partway that
/// would read as the whole history. The rows fed from here are counted in
/// the bank's own `rows_seen`, as before.
#[test]
fn a_v1_state_has_no_data_summary_and_never_grows_one() {
    let spec = spec_from_fixture();
    let mut bank = Bank::load_bytes(&bytes(), Some(std::slice::from_ref(&spec))).unwrap();
    let no_summary = |bank: &Bank, what: &str| {
        let s = bank.summary(0, None).unwrap();
        assert_eq!(s.height(), 2, "{what}: one row per group");
        for c in s.columns() {
            match c.name().as_str() {
                "group" | "rows_processed" => assert_eq!(c.null_count(), 0, "{what}: {}", c.name()),
                // The fixture is on a row-count clock: null with or without a summary.
                _ => assert_eq!(c.null_count(), 2, "{what}: {} should be null", c.name()),
            }
        }
        let d = bank.describe(0, None).unwrap();
        assert_eq!(d.height(), 2 * (spec.features.len() + spec.targets.len()));
        for c in d.columns() {
            match c.name().as_str() {
                "group" | "column" | "role" => assert_eq!(c.null_count(), 0),
                _ => assert_eq!(
                    c.null_count(),
                    d.height(),
                    "{what}: {} should be null",
                    c.name()
                ),
            }
        }
    };
    no_summary(&bank, "as loaded");
    bank.fit_predict(&frame(60, 20)).unwrap();
    no_summary(&bank, "after more rows");
    let s = bank.summary(0, None).unwrap();
    assert_eq!(
        s.column("rows_processed")
            .unwrap()
            .u64()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .sum::<u64>(),
        80,
        "rows processed is the stream's own count and keeps going"
    );
    // A re-save keeps it absent: not a count that began at the load.
    let again = Bank::load_bytes(
        &bank.save_bytes().unwrap(),
        Some(std::slice::from_ref(&spec)),
    )
    .unwrap();
    no_summary(&again, "after save/load");
}

#[test]
fn a_v1_state_continues_the_stream_correctly() {
    // Loading v1 and continuing must match a run that never left this version.
    let spec = spec_from_fixture();
    let mut fresh = Bank::new(vec![spec.clone()]).unwrap();
    fresh.fit_predict(&frame(0, 60)).unwrap();
    let next = frame(60, 20);
    let expected = fresh.fit_predict(&next).unwrap();

    let mut restored = Bank::load_bytes(&bytes(), Some(&[spec])).unwrap();
    let got = restored.fit_predict(&next).unwrap();

    let a = DataFrame::new(next.height(), expected).unwrap();
    let b = DataFrame::new(next.height(), got).unwrap();
    assert!(
        a.unnest(["m"], None)
            .unwrap()
            .equals_missing(&b.unnest(["m"], None).unwrap()),
        "a v1 state did not continue identically to a v2 one"
    );
}

#[test]
fn a_v1_state_round_trips_to_v2() {
    // Loading v1 and saving produces v2, which must then load as well.
    let spec = spec_from_fixture();
    let restored = Bank::load_bytes(&bytes(), Some(std::slice::from_ref(&spec))).unwrap();
    let upgraded = restored.save_bytes().unwrap();
    assert!(Bank::load_bytes(&upgraded, Some(&[spec])).is_ok());
    assert_ne!(upgraded, bytes(), "saving should write the current format");
}

#[test]
fn a_corrupt_state_is_refused() {
    // The other half of the contract: a file from a newer build must not be
    // silently misread.
    let raw = bytes();
    assert!(
        Bank::load_bytes(&raw[..raw.len() / 2], None).is_err(),
        "a truncated state should be refused"
    );
    assert!(
        Bank::load_bytes(&[], None).is_err(),
        "an empty state should be refused"
    );
}

#[test]
fn a_file_without_the_row_counter_reports_its_streams_sum() {
    // `rows_fed` (docs/IMPROVEMENTS.md U3) is an optional field of the map-
    // encoded file. A file from before it existed reports what its streams
    // processed -- 30 rows in each of the fixture's two groups, none skipped
    // -- and the counter is live from there on.
    let mut bank = Bank::load_bytes(&bytes(), None).unwrap();
    assert_eq!(bank.rows_seen(), 60);
    bank.fit_predict(&frame(60, 20)).unwrap();
    assert_eq!(bank.rows_seen(), 80);
    let saved = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
    assert_eq!(saved.rows_seen(), 80);
}

/// A file from a newer build is reported as such, even when its specs carry
/// a key this build's `Spec` refuses: the envelope is checked before the
/// body is parsed, so the refusal cannot masquerade as "not a bank file".
#[test]
fn a_newer_build_s_file_is_reported_as_newer_not_as_garbage() {
    let mut file: serde_json::Value = {
        // The fixture, re-encoded as JSON so it can be edited.
        let bank = Bank::load_bytes(&bytes(), None).unwrap();
        let mut spec = serde_json::to_value(&bank.specs()[0]).unwrap();
        spec["a_knob_this_build_has_not_got"] = serde_json::json!(1);
        serde_json::json!({
            "magic": "polars-online-bank",
            "format_version": 2,
            "schema_version": 1,
            "package_version": "0.1.0",
            "specs": [spec],
            "states": [[]],
        })
    };
    let refused = Bank::load_bytes(&rmp_serde::to_vec_named(&file).unwrap(), None)
        .err()
        .expect("an unknown spec key should be refused");
    assert!(
        refused.contains("not a polars-online bank state file")
            && refused.contains("a_knob_this_build_has_not_got"),
        "{refused}"
    );
    file["format_version"] = serde_json::json!(999);
    let newer = Bank::load_bytes(&rmp_serde::to_vec_named(&file).unwrap(), None)
        .err()
        .expect("a newer format version should be refused");
    assert!(newer.contains("format version 999 is newer"), "{newer}");
}
