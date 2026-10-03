//! The spec layer of `marginal`'s newer parameters (docs/PLAN.md tasks 123,
//! 126 and 131): how `shards`, `bin_budget` and `cross_lags` read from JSON
//! and TOML, write back, and are refused (review 2026-09-26, D missing
//! 7-9). The models' own rules are pinned in `online-core`; this is the
//! crossing.

use online_polars::{ModelKind, ShardSpec, Spec};

fn spec_json(model: &str) -> Result<Spec, String> {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "marginal"{model}}},
            "targets": ["y"], "features": ["x0", "x1"], "half_life": 20.0}}"#
    ))
    .map_err(|e| e.to_string())
}

fn shards_of(spec: &Spec) -> Option<ShardSpec> {
    match &spec.model {
        ModelKind::Marginal { shards, .. } => *shards,
        other => panic!("{other:?}"),
    }
}

/// A count of at least one or `"auto"`, and nothing else: a float, zero, a
/// negative, another word and a count past `usize` are refused as what
/// they are, from JSON and from TOML alike.
#[test]
fn shards_read_a_count_or_auto_and_refuse_the_rest() {
    assert_eq!(
        shards_of(&spec_json(r#", "shards": 1"#).unwrap()),
        Some(ShardSpec::Count(1))
    );
    assert_eq!(
        shards_of(&spec_json(r#", "shards": 7"#).unwrap()),
        Some(ShardSpec::Count(7))
    );
    assert_eq!(
        shards_of(&spec_json(r#", "shards": "auto""#).unwrap()),
        Some(ShardSpec::Auto)
    );
    assert_eq!(shards_of(&spec_json("").unwrap()), None);
    let max = format!(r#", "shards": {}"#, usize::MAX);
    assert_eq!(
        shards_of(&spec_json(&max).unwrap()),
        Some(ShardSpec::Count(usize::MAX)),
        "the core clamps a count to the width"
    );
    for (bad, what) in [
        (r#", "shards": 2.0"#, "floating point"),
        (r#", "shards": 0"#, "0"),
        (r#", "shards": -1"#, "-1"),
        (r#", "shards": "AUTO""#, "AUTO"),
        (r#", "shards": true"#, "boolean"),
        (
            r#", "shards": 340282366920938463463374607431768211456"#,
            "number",
        ),
    ] {
        let err = spec_json(bad).unwrap_err();
        assert!(err.contains(what), "{bad}: {err}");
        assert!(
            err.contains("at least 1") || err.contains("number"),
            "{bad}: {err}"
        );
    }
    let toml_spec = |shards: &str| -> Result<Spec, String> {
        toml::from_str(&format!(
            "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\nhalf_life = 20.0\n\
             [model]\ntype = \"marginal\"\nshards = {shards}\n"
        ))
        .map_err(|e| e.to_string())
    };
    assert_eq!(
        shards_of(&toml_spec("4").unwrap()),
        Some(ShardSpec::Count(4))
    );
    assert_eq!(
        shards_of(&toml_spec("\"auto\"").unwrap()),
        Some(ShardSpec::Auto)
    );
    for bad in ["2.5", "0", "-3", "\"many\""] {
        assert!(toml_spec(bad).is_err(), "{bad}");
    }
}

/// A spec writes its `shards` back as it read them, through JSON and the
/// state file's msgpack, and a spec without them writes none.
#[test]
fn shards_round_trip_through_json_and_msgpack() {
    for given in [r#", "shards": 3"#, r#", "shards": "auto""#, ""] {
        let spec = spec_json(given).unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json.contains("shards"), !given.is_empty(), "{json}");
        let back: Spec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec, "{given}");
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: Spec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, spec, "{given} through msgpack");
    }
}

/// `bin_budget`: refused by name where it means nothing -- zero, negative,
/// `-inf`, NaN, a string that is not a number word, and without bins to
/// bound -- and taken beside given edges as beside learned bins.
#[test]
fn bin_budget_is_held_to_what_it_can_mean() {
    for bad in ["0", "-1", "\"-inf\"", "\"nan\""] {
        let mut spec = spec_json(&format!(r#", "bins": 4, "bin_budget": {bad}"#)).unwrap();
        let err = spec.check().unwrap_err();
        assert!(err.contains("bin_budget"), "{bad}: {err}");
    }
    let err = spec_json(r#", "bins": 4, "bin_budget": "10""#).unwrap_err();
    assert!(err.contains("10"), "{err}");
    let mut alone = spec_json(r#", "bin_budget": 64"#).unwrap();
    let err = alone.check().unwrap_err();
    assert!(
        err.contains("bin_budget") && err.contains("bins or bin_edges"),
        "{err}"
    );
    let mut edges = spec_json(r#", "bin_edges": [[0.0], [1.0]], "bin_budget": 64"#).unwrap();
    assert!(edges.check().is_ok());
    let mut inf = spec_json(r#", "bins": 4, "bin_budget": "inf""#).unwrap();
    assert!(inf.check().is_ok(), "no bound");
}

/// `cross_lags`: an empty list is kept as written (every cross term off),
/// `null` and absent are the default (every lag), and cross lags without
/// lags are refused by name.
#[test]
fn cross_lags_write_back_as_given_and_need_lags() {
    let empty = spec_json(r#", "lags": [1, 2], "cross_lags": []"#).unwrap();
    let json = serde_json::to_string(&empty).unwrap();
    assert!(json.contains(r#""cross_lags":[]"#), "{json}");
    match &empty.model {
        ModelKind::Marginal { cross_lags, .. } => assert_eq!(cross_lags.as_deref(), Some(&[][..])),
        other => panic!("{other:?}"),
    }
    let default = spec_json(r#", "lags": [1, 2]"#).unwrap();
    assert!(
        !serde_json::to_string(&default)
            .unwrap()
            .contains("cross_lags")
    );
    match &default.model {
        ModelKind::Marginal { cross_lags, .. } => assert!(cross_lags.is_none()),
        other => panic!("{other:?}"),
    }
    let mut without = spec_json(r#", "cross_lags": [1]"#).unwrap();
    let err = without.check().unwrap_err();
    assert!(err.contains("cross_lags"), "{err}");
    let mut outside = spec_json(r#", "lags": [1, 2], "cross_lags": [3]"#).unwrap();
    let err = outside.check().unwrap_err();
    assert!(err.contains("cross_lags"), "{err}");
}
