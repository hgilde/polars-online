//! `pa`'s spec keys, kept in a file of their own beside `sgd_tests.rs`:
//! `spec.rs` stands near the repository's 250 KB cap for a source file
//! (`tests/test_repo_hygiene.py`).

use super::Spec;

fn pa(model: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "pa"{model}}}, "targets": ["y"],
            "features": ["x"], "half_life": 10}}"#
    ))
    .unwrap()
}

/// `c` is the cap of `"pa1"` and the damping of `"pa2"`; `"pa"` reads
/// neither, and took a `c` without a word. A parameter whose switch is off
/// is refused, as `sgd` refuses `power` beside a schedule that does not
/// read it: by every door, a builder's, a dict's or a TOML file's, naming
/// the parameter and the modes that read it. A null is the default, not a
/// value given.
#[test]
fn a_c_under_the_mode_that_reads_none_is_refused() {
    for model in [
        r#", "mode": "pa", "c": 0.5"#,
        r#", "mode": "pa", "c": "inf""#,
    ] {
        let err = pa(model).check().unwrap_err();
        assert!(
            err.contains(r#"spec "m": pa c is for mode "pa1" or "pa2"; mode "pa" does not use it"#),
            "{model}: {err}"
        );
    }
    for model in [
        r#", "mode": "pa""#,
        r#", "mode": "pa", "c": null"#,
        r#", "mode": "pa1", "c": 0.5"#,
        r#", "mode": "pa2", "c": 0.5"#,
        r#", "c": 0.5"#,
    ] {
        assert!(pa(model).check().is_ok(), "{model}");
    }
    // A `c` out of range is named for its range first, and an unknown mode
    // as unknown.
    let err = pa(r#", "mode": "pa", "c": 0.0"#).check().unwrap_err();
    assert!(err.contains("pa c must be > 0"), "{err}");
    let err = pa(r#", "mode": "pa3", "c": 0.5"#).check().unwrap_err();
    assert!(err.contains("unknown pa mode \"pa3\""), "{err}");
}
