//! Backward compatibility with schema-2 model states (0.2.0, `docs/PLAN.md`
//! task 33): `kalman` and `sgd` standardized with a full `EwCov` and now
//! keep an `EwDiag`, its diagonal.
//!
//! As with `state_v1.rs`, there is no schema-2 *writer* left once the layout
//! moves, so the fixture is frozen here as bytes: a real 0.2.0-era bank file
//! -- a `kalman` and an `sgd` with `scale_features`, grouped, after 60 rows
//! of a deterministic stream -- generated once from the schema-2 layout and
//! never regenerated. A hex constant rather than a checked-in binary because
//! of hard rule 1 (no data files in the repo).
//!
//! The bank envelope is unchanged (format 2); what moved is the model state
//! inside it, which each model converts in its own `Deserialize`. The tests
//! therefore ask for more than "it loads": the converted model must continue
//! the stream with the *same bits* as one that never left this build, since
//! the diagonal a schema-2 model read is exactly the accumulator a schema-3
//! one keeps.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// A schema-2 bank state: specs `k` (`kalman`, standardized, intercept) and
/// `s` (`sgd`, `scale_features`, `coef_min`), grouped by `g`, after
/// `frame(0, 60)`. See the module docs for why this is frozen bytes.
const SCHEMA2_STATE_HEX: &str = concat!(
    "87a56d61676963b2706f6c6172732d6f6e6c696e652d62616e6bae666f726d61745f76657273",
    "696f6e02ae736368656d615f76657273696f6e02af7061636b6167655f76657273696f6ea530",
    "2e322e30a5737065637392de001fa46e616d65a16ba56d6f64656c88a474797065a66b616c6d",
    "616ead636f65665f68616c666c696665cb4049000000000000a171c0a76f62735f766172c0a2",
    "7030c0a773686172655f70c2af7265766572745f68616c666c696665c0ab7374616e64617264",
    "697a65c3a77461726765747391a179a8666561747572657392a27830a27831ad6164645f696e",
    "74657263657074c3a5636c6f636bc0a868616c666c696665cb4034000000000000a36c616dc0",
    "aa6d61785f64636c6f636bc0ae6f6e5f636c6f636b5f7265736574a36d6178a773657373696f",
    "6ec0ab73657373696f6e5f676170c0a6776569676874c0ab6d696e5f706572696f6473cb4008",
    "000000000000aa636f65665f657665727900aa656d69745f7369676d61c2ac656d69745f7265",
    "7369645f7ac2ac656d69745f6d657472696373c2a9636f6e666f726d616cc0ae636f6e666f72",
    "6d616c5f72617465c0af72657369645f7175616e74696c6573c0ad656d69745f6175746f636f",
    "7272c2b272657369645f6175746f636f72725f6c6167c0aa656d69745f6472696674c2ab6472",
    "6966745f64656c7461c0af64726966745f7468726573686f6c64c0ac64726966745f61637469",
    "6f6ea4666c6167ad656d69745f6176657261676564c2ab617665726167655f657461c0ad656d",
    "69745f73656c6563746564c2a567726f7570a167de001fa46e616d65a173a56d6f64656c8ea4",
    "74797065a3736764a46c6f7373a773717561726564ab68756265725f64656c7461c0a8717561",
    "6e74696c65c0a3657073c0ad6c6561726e696e675f72617465cb3fa999999999999aa8736368",
    "6564756c65a8636f6e7374616e74a5706f776572c0a26c32c0ad636c69705f6772616469656e",
    "74c0ae7363616c655f6665617475726573c3a8636f65665f6d696ecbc014000000000000a863",
    "6f65665f6d6178c0a8636f65665f73756dc0a77461726765747391a179a86665617475726573",
    "92a27830a27831ad6164645f696e74657263657074c3a5636c6f636bc0a868616c666c696665",
    "cb4034000000000000a36c616dc0aa6d61785f64636c6f636bc0ae6f6e5f636c6f636b5f7265",
    "736574a36d6178a773657373696f6ec0ab73657373696f6e5f676170c0a6776569676874c0ab",
    "6d696e5f706572696f6473cb4008000000000000aa636f65665f657665727900aa656d69745f",
    "7369676d61c2ac656d69745f72657369645f7ac2ac656d69745f6d657472696373c2a9636f6e",
    "666f726d616cc0ae636f6e666f726d616c5f72617465c0af72657369645f7175616e74696c65",
    "73c0ad656d69745f6175746f636f7272c2b272657369645f6175746f636f72725f6c6167c0aa",
    "656d69745f6472696674c2ab64726966745f64656c7461c0af64726966745f7468726573686f",
    "6c64c0ac64726966745f616374696f6ea4666c6167ad656d69745f6176657261676564c2ab61",
    "7665726167655f657461c0ad656d69745f73656c6563746564c2a567726f7570a167a6737461",
    "746573929292a1618aa5636c6f636b84aa707265765f636c6f636bc0ac707265765f73657373",
    "696f6ec0a770656e64696e67cb0000000000000000a773746172746564c3a66d6f64656c7391",
    "82ae736368656d615f76657273696f6e02a56d6f64656c81a64b616c6d616e87a36366678caa",
    "6e5f666561747572657302a96e5f7461726765747301ad6164645f696e74657263657074c3a5",
    "646563617981a848616c666c696665cb4034000000000000a868616c666c69666591cb404900",
    "0000000000a171c0a76f62735f766172c0a27030cb3ff0000000000000a773686172655f70c2",
    "ab6d696e5f706572696f6473cb4008000000000000af7265766572745f68616c666c69666591",
    "cb7ff0000000000000ab7374616e64617264697a65c3a3636f7687a16b03a5775f73756dcb40",
    "32fa43c5d33776ab7072696f725f7363616c65cb3fd76ce51f6a0bc1a16d93cb3ff000000000",
    "0000cbbf8bfebea2d2fd92cb4026c5a2bbc5e554a16399cb0000000000000000cb0000000000",
    "000000cb0000000000000000cb0000000000000000cb3fb4fa6440095505cb3fa07be14fade9",
    "31cb0000000000000000cb3fa07be14fade932cb3fe7b7d3e73acfd2af707265636973696f6e",
    "5f7072696f72cb0000000000000000af707265636973696f6e5f7363616c65cb3f93bffa7dae",
    "fb93a4626574619193cbc01508e3b4719472cb3fdfd0199aa0292dcbbfc601000ecce389a170",
    "9199cb3f981567fdefedd3cb3f76191820efb699cb3f39f96be0750fd2cb3f76191820efb699",
    "cb3f95d8820e22f7f4cbbf3b59deffdb8197cb3f39f96be0750fd2cbbf3b59deffdb8197cb3f",
    "7e49ac57baa62ca47369673291cb3fd56a2425f6d817a47773696791cb40316f298de2d470a2",
    "776a91cb4032fa43c5d33776a9726f77735f7365656e1ea972657369645f7661729191cb3fd5",
    "6a2425f6d817a772657369645f779191cb40316f298de2d470a5647269667490a77265736964",
    "5f7190a86175746f636f727290a76d65747269637390a9636f6e666f726d616c9092a1628aa5",
    "636c6f636b84aa707265765f636c6f636bc0ac707265765f73657373696f6ec0a770656e6469",
    "6e67cb0000000000000000a773746172746564c3a66d6f64656c739182ae736368656d615f76",
    "657273696f6e02a56d6f64656c81a64b616c6d616e87a36366678caa6e5f6665617475726573",
    "02a96e5f7461726765747301ad6164645f696e74657263657074c3a5646563617981a848616c",
    "666c696665cb4034000000000000a868616c666c69666591cb4049000000000000a171c0a76f",
    "62735f766172c0a27030cb3ff0000000000000a773686172655f70c2ab6d696e5f706572696f",
    "6473cb4008000000000000af7265766572745f68616c666c69666591cb7ff0000000000000ab",
    "7374616e64617264697a65c3a3636f7687a16b03a5775f73756dcb4032fa43c5d33776ab7072",
    "696f725f7363616c65cb3fd76ce51f6a0bc1a16d93cb3ff0000000000000cbbf29efaa8bbbc2",
    "00cb40274abb4ddad35fa16399cb0000000000000000cb0000000000000000cb000000000000",
    "0000cb0000000000000000cb3fb727dbee15497ecbbf8c2f01b471f6dccb0000000000000000",
    "cbbf8c2f01b471f6dbcb3fe9b58cd7ddd77baf707265636973696f6e5f7072696f72cb000000",
    "0000000000af707265636973696f6e5f7363616c65cb3f93bffa7daefb93a4626574619193cb",
    "c013ee2fa869d4cbcb3fdc9d51c653e975cbbfb470a26a5039a2a1709199cb3fb84b2d427ac5",
    "44cb3f83068d7bcced84cbbf8389c1672f6aaccb3f83068d7bcced84cb3fac953daad25a69cb",
    "bf2582a7889067d4cbbf8389c1672f6aaccbbf2582a7889067d4cb3f97e26e44caf74ca47369",
    "673291cb3ffe4acfa891a607a47773696791cb40316f298de2d470a2776a91cb4032fa43c5d3",
    "3776a9726f77735f7365656e1ea972657369645f7661729191cb3ffe4acfa891a607a7726573",
    "69645f779191cb40316f298de2d470a5647269667490a772657369645f7190a86175746f636f",
    "727290a76d65747269637390a9636f6e666f726d616c909292a1618aa5636c6f636b84aa7072",
    "65765f636c6f636bc0ac707265765f73657373696f6ec0a770656e64696e67cb000000000000",
    "0000a773746172746564c3a66d6f64656c739182ae736368656d615f76657273696f6e02a56d",
    "6f64656c81a353676485a36366678caa6e5f666561747572657302a96e5f7461726765747301",
    "ad6164645f696e74657263657074c3a5646563617981a848616c666c696665cb403400000000",
    "0000a46c6f7373a773717561726564ad6c6561726e696e675f72617465cb3fa999999999999a",
    "a87363686564756c65a8636f6e7374616e74a26c32cb0000000000000000ab6d696e5f706572",
    "696f6473cb4008000000000000ae7363616c655f6665617475726573c3ad636c69705f677261",
    "6469656e74cb408f400000000000aa636f6e73747261696e7483a26c6f92cbc0140000000000",
    "00cbc014000000000000a2686992cb7ff0000000000000cb7ff0000000000000a373756dc0a6",
    "7363616c657287a16b03a5775f73756dcb4032fa43c5d33776ab7072696f725f7363616c65cb",
    "3fd76ce51f6a0bc1a16d93cb3ff0000000000000cbbf8bfebea2d2fd92cb4026c5a2bbc5e554",
    "a16399cb0000000000000000cb0000000000000000cb0000000000000000cb00000000000000",
    "00cb3fb4fa6440095505cb3fa07be14fade931cb0000000000000000cb3fa07be14fade932cb",
    "3fe7b7d3e73acfd2af707265636973696f6e5f7072696f72cb0000000000000000af70726563",
    "6973696f6e5f7363616c65cb3f93bffa7daefb93a4626574619193cbc011e49deaea6401cb3f",
    "e148a4e6bf0aaccb3fc8b63dfaa47788a2673290a5775f73756dcb4032fa43c5d33776a9726f",
    "77735f7365656e1ea972657369645f7661729191cb4016a7c54b5a13dea772657369645f7791",
    "91cb40316f298de2d470a5647269667490a772657369645f7190a86175746f636f727290a76d",
    "65747269637390a9636f6e666f726d616c9092a1628aa5636c6f636b84aa707265765f636c6f",
    "636bc0ac707265765f73657373696f6ec0a770656e64696e67cb0000000000000000a7737461",
    "72746564c3a66d6f64656c739182ae736368656d615f76657273696f6e02a56d6f64656c81a3",
    "53676485a36366678caa6e5f666561747572657302a96e5f7461726765747301ad6164645f69",
    "6e74657263657074c3a5646563617981a848616c666c696665cb4034000000000000a46c6f73",
    "73a773717561726564ad6c6561726e696e675f72617465cb3fa999999999999aa87363686564",
    "756c65a8636f6e7374616e74a26c32cb0000000000000000ab6d696e5f706572696f6473cb40",
    "08000000000000ae7363616c655f6665617475726573c3ad636c69705f6772616469656e74cb",
    "408f400000000000aa636f6e73747261696e7483a26c6f92cbc014000000000000cbc0140000",
    "00000000a2686992cb7ff0000000000000cb7ff0000000000000a373756dc0a67363616c6572",
    "87a16b03a5775f73756dcb4032fa43c5d33776ab7072696f725f7363616c65cb3fd76ce51f6a",
    "0bc1a16d93cb3ff0000000000000cbbf29efaa8bbbc200cb40274abb4ddad35fa16399cb0000",
    "000000000000cb0000000000000000cb0000000000000000cb0000000000000000cb3fb727db",
    "ee15497ecbbf8c2f01b471f6dccb0000000000000000cbbf8c2f01b471f6dbcb3fe9b58cd7dd",
    "d77baf707265636973696f6e5f7072696f72cb0000000000000000af707265636973696f6e5f",
    "7363616c65cb3f93bffa7daefb93a4626574619193cbc0112e2d24cc4831cb3fd3937651f345",
    "28cbbfe58400dfadafbaa2673290a5775f73756dcb4032fa43c5d33776a9726f77735f736565",
    "6e1ea972657369645f7661729191cb4020a35247cea7a8a772657369645f779191cb40316f29",
    "8de2d470a5647269667490a772657369645f7190a86175746f636f727290a76d657472696373",
    "90a9636f6e666f726d616c90a8726f77735f6665643c",
);

/// The same stream the fixture was built from: every row a function of its
/// index alone (integer arithmetic into an exactly representable `f64`, no
/// `sin`), so any window is reproducible on any platform.
fn frame(start: usize, n: usize) -> DataFrame {
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in start..start + n {
        let mut s = (i as u64) ^ 0x9E37_79B9_7F4A_7C15;
        let a = lcg(&mut s) - 0.5;
        let b = 10.0 + 3.0 * lcg(&mut s);
        let e = 0.1 * (lcg(&mut s) - 0.5);
        x0.push(a);
        x1.push(b);
        y.push(2.0 * a - 0.5 * b + 0.25 + e);
        g.push(if i % 2 == 0 { "a" } else { "b" }.to_string());
    }
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn bytes() -> Vec<u8> {
    (0..SCHEMA2_STATE_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&SCHEMA2_STATE_HEX[i..i + 2], 16).unwrap())
        .collect()
}

/// The specs are recovered *from* the fixture: a file carries its own, and
/// restating them here would only test that two copies agree.
fn specs_from_fixture() -> Vec<Spec> {
    Bank::load_bytes(&bytes(), None)
        .expect("schema-2 fixture should load without spec expectations")
        .specs()
        .to_vec()
}

fn header(bytes: &[u8]) -> serde_json::Value {
    rmp_serde::from_slice(bytes).expect("a bank file is a msgpack map")
}

/// One frame per spec, unnested, so the comparison sees every field.
fn unnested(n: usize, cols: Vec<Column>) -> Vec<DataFrame> {
    let df = DataFrame::new(n, cols).unwrap();
    ["k", "s"]
        .iter()
        .map(|name| df.select([*name]).unwrap().unnest([*name], None).unwrap())
        .collect()
}

fn assert_same(want: &[DataFrame], got: &[DataFrame], what: &str) {
    for (w, g) in want.iter().zip(got) {
        assert!(w.equals_missing(g), "{what}:\n{w}\n{g}");
    }
}

#[test]
fn the_fixture_is_what_it_says() {
    let h = header(&bytes());
    assert_eq!(h["format_version"], 2);
    assert_eq!(h["schema_version"], 2);
    let specs = specs_from_fixture();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].name, "k");
    assert_eq!(specs[1].name, "s");
    // Both standardize, or the fixture would not exercise the conversion.
    let json = serde_json::to_value(&specs).unwrap();
    assert_eq!(json[0]["model"]["type"], "kalman");
    assert_eq!(json[0]["model"]["standardize"], true);
    assert_eq!(json[1]["model"]["type"], "sgd");
    assert_eq!(json[1]["model"]["scale_features"], true);
}

#[test]
fn a_schema_2_state_file_still_loads() {
    let bank = Bank::load_bytes(&bytes(), None);
    assert!(
        bank.is_ok(),
        "schema-2 state failed to load: {:?}",
        bank.err()
    );
    let specs = specs_from_fixture();
    assert!(Bank::load_bytes(&bytes(), Some(&specs)).is_ok());
    // Written before the last learned row travelled with the state
    // (docs/PLAN.md task 34): a row of nulls per group, for both specs.
    let mut bank = bank.unwrap();
    for si in 0..specs.len() {
        let (keys, col) = bank.last_row(si, None).unwrap();
        assert!(!keys.is_empty());
        for f in col.struct_().unwrap().fields_as_series() {
            assert_eq!(f.null_count(), keys.len(), "{}", f.name());
        }
    }
    // And before the data summary did (task 35): null but for what the
    // stream always kept, as loaded and after more rows -- a count that
    // began partway would read as the whole history.
    let no_summary = |bank: &Bank, what: &str| {
        for (si, spec) in specs.iter().enumerate() {
            let s = bank.summary(si, None).unwrap();
            assert!(s.height() > 0);
            for c in s.columns() {
                match c.name().as_str() {
                    "group" | "rows_processed" => {
                        assert_eq!(c.null_count(), 0, "{what}: {}", c.name())
                    }
                    _ => assert_eq!(
                        c.null_count(),
                        s.height(),
                        "{what}: {} should be null",
                        c.name()
                    ),
                }
            }
            let d = bank.describe(si, None).unwrap();
            assert_eq!(
                d.height(),
                s.height() * (spec.features.len() + spec.targets.len())
            );
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
        }
    };
    no_summary(&bank, "as loaded");
    bank.fit_predict(&frame(60, 20)).unwrap();
    no_summary(&bank, "after more rows");
    let again = Bank::load_bytes(&bank.save_bytes().unwrap(), Some(&specs)).unwrap();
    no_summary(&again, "after save/load");
}

#[test]
fn a_schema_2_state_continues_the_stream_to_the_bit() {
    let specs = specs_from_fixture();
    let mut fresh = Bank::new(specs.clone()).unwrap();
    fresh.fit_predict(&frame(0, 60)).unwrap();
    let mut restored = Bank::load_bytes(&bytes(), Some(&specs)).unwrap();

    // The state as loaded: what the models report before another row.
    for si in 0..specs.len() {
        assert_eq!(
            restored.coef(si, None).unwrap(),
            fresh.coef(si, None).unwrap(),
            "spec {si}: coefficients differ after the conversion"
        );
    }
    // And the stream from here, in fit and in predict.
    let next = frame(60, 40);
    let want = unnested(next.height(), fresh.fit_predict(&next).unwrap());
    let got = unnested(next.height(), restored.fit_predict(&next).unwrap());
    assert_same(
        &want,
        &got,
        "a schema-2 state did not continue identically to a schema-3 one",
    );
    let probe = frame(100, 10);
    let want = unnested(probe.height(), fresh.predict(&probe).unwrap());
    let got = unnested(probe.height(), restored.predict(&probe).unwrap());
    assert_same(&want, &got, "predict differs");
}

#[test]
fn a_schema_2_state_saves_as_schema_3_and_loads_again() {
    let specs = specs_from_fixture();
    let restored = Bank::load_bytes(&bytes(), Some(&specs)).unwrap();
    let upgraded = restored.save_bytes().unwrap();
    assert_ne!(upgraded, bytes(), "saving should write the current schema");
    let h = header(&upgraded);
    assert_eq!(h["format_version"], 2, "the envelope did not move");
    assert_eq!(h["schema_version"], online_core::SCHEMA_VERSION);
    assert_eq!(h["schema_version"], 3);
    let mut again = Bank::load_bytes(&upgraded, Some(&specs)).unwrap();
    let mut fresh = Bank::new(specs).unwrap();
    fresh.fit_predict(&frame(0, 60)).unwrap();
    let next = frame(60, 20);
    let want = unnested(next.height(), fresh.fit_predict(&next).unwrap());
    let got = unnested(next.height(), again.fit_predict(&next).unwrap());
    assert_same(
        &want,
        &got,
        "the re-saved state did not continue identically",
    );
}
