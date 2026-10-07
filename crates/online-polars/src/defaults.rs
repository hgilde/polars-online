//! What a spec runs with where it leaves a parameter out
//! (docs/RELEASE-READINESS.md, "Keeping the API stable").
//!
//! About fifty defaults are chosen on this side rather than in the Python
//! builders, which pass `None` for them, so a builder's signature cannot
//! show them. [`resolved_defaults`] reads them off the bank's own build of a
//! spec: [`Spec::check`], the door every spec comes in by, then
//! [`Stream::new`], the constructor every group's stream comes from. Nothing
//! here chooses a value. `tests/api_surface.txt` renders the result for
//! every kind, so a default that moves is a diff there (review 2026-10-06,
//! AP1, YA1, TB1, DB4).

use online_core::{ClockCfg, SessionGap, WindowBudget};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};

use crate::spec::{ModelKind, Spec};
use crate::stream::{AnyModel, Stream, marginal_shards};

/// The configuration a spec resolves to, as JSON, from its first model
/// instance (the instances of a grid differ in their decay alone):
///
/// - `model`: the model's configuration as built, every field;
/// - `derived`: the values a model derives from a field the configuration
///   leaves `None`, by a rule of its own -- `bocpd`'s priors, `rcov`'s ring
///   and pre-averaging window, `marginal`'s bin budget -- where it has one;
/// - `stream`: what the stream applies around the model -- the warm-up and
///   readiness gates, the diagnostics' settings, the window budget, and
///   `marginal`'s shards;
/// - `clock`: the clock policy.
///
/// A non-finite number is written as `"inf"`, `"-inf"` or `"nan"`, the
/// spelling a spec takes. Refused with the spec's own refusal for a spec
/// the bank would refuse.
pub fn resolved_defaults(spec: &Spec) -> Result<Value, String> {
    let mut spec = spec.clone();
    spec.check()?;
    let stream = Stream::new(&spec)?;
    let (_, model) = stream
        .models
        .first()
        .ok_or_else(|| format!("spec {:?} builds no model instance", spec.name))?;
    let mut out = Map::new();
    out.insert("model".into(), model.cfg_json()?);
    if let Some(d) = derived(model) {
        out.insert("derived".into(), d);
    }
    out.insert("stream".into(), stream_settings(&spec, &stream, model));
    out.insert("clock".into(), clock(&spec.clock_cfg()?)?);
    Ok(Value::Object(out))
}

/// `v` as JSON, refused unless it reads back as itself: `serde_json`
/// writes NaN and the infinities as `null` without a word, which a
/// configuration field without `online_core::humanfloat` would pass on as
/// a value no spec can give. Compared as msgpack, as
/// `Bank::save_json_string` compares a state, so a NaN matches itself.
pub(crate) fn faithful_json<T: Serialize + DeserializeOwned>(v: &T) -> Result<Value, String> {
    let json = serde_json::to_value(v).map_err(|e| e.to_string())?;
    let back: T = serde_json::from_value(json.clone()).map_err(|e| {
        format!(
            "a configuration does not read back from its JSON ({e}); a non-finite float \
             needs `online_core::humanfloat` on its field"
        )
    })?;
    let bytes = |t: &T| rmp_serde::to_vec_named(t).map_err(|e| e.to_string());
    if bytes(&back)? != bytes(v)? {
        return Err(
            "a configuration's JSON reads back as another value; a field's human-readable \
             encoding does not round-trip"
                .into(),
        );
    }
    Ok(json)
}

/// A number as a spec writes one: finite as itself, the others as words.
fn number(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else if v.is_nan() {
        json!("nan")
    } else if v > 0.0 {
        json!("inf")
    } else {
        json!("-inf")
    }
}

fn numbers(v: &[f64]) -> Value {
    Value::Array(v.iter().copied().map(number).collect())
}

/// What a model derives by a rule of its own from a field its configuration
/// leaves `None`, read through the model's own functions.
fn derived(model: &AnyModel) -> Option<Value> {
    match model {
        AnyModel::Bocpd(m) => {
            let c = m.cfg();
            Some(json!({
                "prior_mean": numbers(&c.mu0()),
                "prior_nu": number(c.nu0()),
                "prior_scale": numbers(&c.psi0()),
            }))
        }
        AnyModel::Rcov(m) => {
            let c = m.cfg();
            Some(json!({
                "max_bandwidth": c.ring_for(),
                "preavg_rows": c.window_for(),
            }))
        }
        AnyModel::Marginal(m) => m
            .cfg()
            .bins
            .as_ref()
            .map(|b| json!({ "bin_budget": number(b.budget_mib_or_default()) })),
        _ => None,
    }
}

/// What the stream applies around the model, each read through the
/// function the stream reads it with.
fn stream_settings(spec: &Spec, stream: &Stream, model: &AnyModel) -> Value {
    let budget = |b: WindowBudget| match b {
        WindowBudget::Thin(m) => json!({ "thin": number(m) }),
        WindowBudget::Refuse(m) => json!({ "refuse": number(m) }),
    };
    let mut s = Map::new();
    s.insert("min_weight".into(), numbers(stream.min_weight()));
    s.insert(
        "min_settled_frac".into(),
        number(spec.min_settled_frac_or_default()),
    );
    s.insert(
        "max_error_inflation".into(),
        number(spec.max_error_inflation_or_default()),
    );
    s.insert("drift_action".into(), json!(spec.drift_action));
    s.insert("drift_delta".into(), number(spec.drift_delta_or_default()));
    s.insert(
        "drift_threshold".into(),
        number(spec.drift_threshold_or_default()),
    );
    s.insert(
        "conformal_rate".into(),
        number(spec.conformal_rate_or_default()),
    );
    s.insert(
        "resid_autocorr_lag".into(),
        json!(spec.resid_autocorr_lag_or_default()),
    );
    s.insert("average_eta".into(), number(spec.average_eta_or_default()));
    s.insert(
        "window_budget".into(),
        spec.model.window_budget().map_or(Value::Null, budget),
    );
    if matches!(spec.model, ModelKind::Marginal { .. }) {
        s.insert("shards".into(), json!(marginal_shards(spec, model)));
    }
    Value::Object(s)
}

/// The clock policy a spec resolves to ([`Spec::clock_cfg`]).
fn clock(c: &ClockCfg) -> Result<Value, String> {
    Ok(json!({
        "gap_cap": number(c.gap_cap),
        "min_backwards_jump": number(c.min_backwards_jump),
        "on_clock_reset": serde_json::to_value(c.on_clock_reset).map_err(|e| e.to_string())?,
        "session_gap": match c.session_gap {
            None => Value::Null,
            Some(SessionGap::Gap(g)) => number(g),
            Some(SessionGap::Reset) => json!("reset"),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::{faithful_json, number};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[test]
    fn a_number_is_written_as_a_spec_writes_it() {
        assert_eq!(number(1.5), json!(1.5));
        assert_eq!(number(f64::INFINITY), json!("inf"));
        assert_eq!(number(f64::NEG_INFINITY), json!("-inf"));
        assert_eq!(number(f64::NAN), json!("nan"));
    }

    #[derive(Serialize, Deserialize)]
    struct Bare {
        x: f64,
    }

    #[derive(Serialize, Deserialize)]
    struct Optional {
        x: Option<f64>,
    }

    /// `serde_json` writes an infinity as `null`: a bare field then does not
    /// read back at all, and an optional one reads back as `None`, a value
    /// that was not there. Both are refused; a finite value is written.
    #[test]
    fn an_infinity_json_would_write_as_null_is_refused() {
        assert_eq!(faithful_json(&Bare { x: 2.0 }).unwrap(), json!({"x": 2.0}));
        assert!(faithful_json(&Bare { x: f64::INFINITY }).is_err());
        let err = faithful_json(&Optional {
            x: Some(f64::INFINITY),
        })
        .unwrap_err();
        assert!(err.contains("reads back as another value"), "{err}");
        assert_eq!(
            faithful_json(&Optional { x: None }).unwrap(),
            json!({"x": null})
        );
    }
}
