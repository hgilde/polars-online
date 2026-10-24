//! `huber_scale` and `huber_scale_half_life` on `huber` and on `sgd`'s Huber
//! loss (docs/PLAN.md task 271): the words, the memory, the loss that owns
//! them, and the default a spec builds -- the MAD -- where a state written
//! before schema 63 loads as the RMS (`online_core::HuberScale`).

use super::Spec;
use crate::stream::{AnyModel, huber_model, sgd_model};
use online_core::{Decay, HuberScale};

fn spec(model: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"],
            "features": ["x"], "half_life": 10}}"#
    ))
    .unwrap()
}

/// The scale and its memory a spec's model is built with.
fn built(model: &str) -> (HuberScale, Option<Decay>) {
    let mut s = spec(model);
    s.check().unwrap();
    let decay = s.decays().unwrap()[0].1;
    match (huber_model(&s, decay), sgd_model(&s, decay)) {
        (Ok(AnyModel::Robust(m)), _) => (m.cfg().huber_scale, m.cfg().huber_scale_decay),
        (_, Ok(AnyModel::Sgd(m))) => (m.cfg().huber_scale, m.cfg().huber_scale_decay),
        _ => panic!("{model}: neither huber nor sgd"),
    }
}

#[test]
fn a_spec_builds_the_mad_unless_it_names_the_rms() {
    for (model, want) in [
        (r#"{"type": "huber"}"#, (HuberScale::Mad, None)),
        (
            r#"{"type": "huber", "huber_scale": "rms"}"#,
            (HuberScale::Rms, None),
        ),
        (
            r#"{"type": "huber", "huber_scale_half_life": 2.5}"#,
            (HuberScale::Mad, Some(Decay::Halflife(2.5))),
        ),
        (
            r#"{"type": "huber", "huber_scale_half_life": "inf"}"#,
            (HuberScale::Mad, Some(Decay::Halflife(f64::INFINITY))),
        ),
        (
            r#"{"type": "sgd", "loss": "huber"}"#,
            (HuberScale::Mad, None),
        ),
        (
            r#"{"type": "sgd", "loss": "huber", "huber_scale": "rms", "huber_scale_half_life": 4}"#,
            (HuberScale::Rms, Some(Decay::Halflife(4.0))),
        ),
        // Every other loss reads no Huber scale.
        (r#"{"type": "sgd"}"#, (HuberScale::Rms, None)),
        (
            r#"{"type": "sgd", "loss": "logistic"}"#,
            (HuberScale::Rms, None),
        ),
    ] {
        assert_eq!(built(model), want, "{model}");
    }
}

#[test]
fn a_huber_scale_is_refused_by_name() {
    for (model, want) in [
        (
            r#"{"type": "huber", "huber_scale": "MAD"}"#,
            r#"huber_scale must be one of "mad", "rms"; got "MAD""#,
        ),
        (
            r#"{"type": "huber", "huber_scale_half_life": 0}"#,
            "huber_scale_half_life must be > 0",
        ),
        (
            r#"{"type": "sgd", "loss": "huber", "huber_scale_half_life": -1}"#,
            "huber_scale_half_life must be > 0",
        ),
        (
            r#"{"type": "sgd", "huber_scale": "mad"}"#,
            r#"sgd huber_scale is for loss "huber"; loss "squared" does not use it"#,
        ),
        (
            r#"{"type": "sgd", "loss": "quantile", "quantile": 0.5, "huber_scale_half_life": 3}"#,
            r#"sgd huber_scale_half_life is for loss "huber"; loss "quantile" does not use it"#,
        ),
    ] {
        let err = spec(model).check().unwrap_err();
        assert!(err.contains(want), "{model}: {err}");
    }
}

/// Left out, neither key is written: a spec from before them serializes as
/// it did.
#[test]
fn a_spec_without_them_serializes_as_before() {
    let json = serde_json::to_string(&spec(r#"{"type": "huber"}"#)).unwrap();
    assert!(!json.contains("huber_scale"), "{json}");
    let json = serde_json::to_string(&spec(r#"{"type": "huber", "huber_scale": "rms"}"#)).unwrap();
    assert!(json.contains(r#""huber_scale":"rms""#), "{json}");
}
