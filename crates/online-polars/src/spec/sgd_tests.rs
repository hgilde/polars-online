//! `sgd`'s spec keys, kept in a file of their own: with task 225's
//! `gram_threads`, `spec.rs` passed the repository's 250 KB cap for a
//! source file (`tests/test_repo_hygiene.py`).

use super::Spec;

fn sgd(model: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "sgd"{model}}}, "targets": ["y"],
            "features": ["x"], "half_life": 10}}"#
    ))
    .unwrap()
}

/// Task 160, YA8b: a parameter of a loss or a schedule the spec does not
/// use was taken and ignored by every door; the builder refuses it now,
/// and so does the validation every spec meets, a dict's or a TOML
/// file's, naming the parameter and what reads it. A null is the
/// default, not a value given.
#[test]
fn a_parameter_its_loss_or_schedule_does_not_read_is_refused() {
    for (model, want) in [
        (
            r#", "huber_delta": 0.5"#,
            r#"sgd huber_delta is for loss "huber"; loss "squared" does not use it"#,
        ),
        (
            r#", "quantile": 0.3"#,
            r#"sgd quantile is for loss "quantile"; loss "squared" does not use it"#,
        ),
        (
            r#", "loss": "huber", "eps": 0.2"#,
            r#"sgd eps is for loss "epsilon_insensitive"; loss "huber" does not use it"#,
        ),
        (
            r#", "power": 0.9"#,
            r#"sgd power is for schedule "inv_scaling"; schedule "constant" does not use it"#,
        ),
        (
            r#", "schedule": "adagrad", "power": 0.9"#,
            r#"sgd power is for schedule "inv_scaling"; schedule "adagrad" does not use it"#,
        ),
    ] {
        let err = sgd(model).check().unwrap_err();
        assert!(
            err.contains(&format!("spec \"m\": {want}")),
            "{model}: {err}"
        );
    }
    // Each beside what reads it, and every one null, is a spec.
    for model in [
        r#", "loss": "huber", "huber_delta": 0.5"#,
        r#", "loss": "quantile", "quantile": 0.3"#,
        r#", "loss": "epsilon_insensitive", "eps": 0.2"#,
        r#", "schedule": "inv_scaling", "power": 0.9"#,
        r#", "huber_delta": null, "quantile": null, "eps": null, "power": null"#,
    ] {
        assert!(sgd(model).check().is_ok(), "{model}");
    }
    // An unknown loss is named as unknown, not as the wrong owner.
    let err = sgd(r#", "loss": "nope", "eps": 0.2"#).check().unwrap_err();
    assert!(err.contains("unknown sgd loss \"nope\""), "{err}");
}
