use super::Spec;

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

/// A column a model reads from the targets slot, named and not listed:
/// `fill_defaults` fills `targets` with it, as the builders write them,
/// not with `features[0]`, which was then read as the hazard or the
/// exogenous series (review 2026-09-12, C20).
#[test]
fn a_named_hazard_or_exogenous_column_fills_the_targets() {
    for (model, col) in [
        (r#"{"type": "bocpd", "hazard_col": "h"}"#, "h"),
        (
            r#"{"type": "hmm", "n_components": 2, "precision_prior": 0.1, "exog_tvtp": "z"}"#,
            "z",
        ),
    ] {
        let mut s = spec(&format!(
            r#"{{"name": "m", "model": {model}, "features": ["x"]}}"#
        ));
        s.fill_defaults();
        assert_eq!(*s.targets, vec![col.to_string()], "{model}");
    }
}

/// And a spec that lists another column in that slot is refused, naming
/// both: the model reads `targets[0]` whatever `hazard_col` says. The
/// message prints the names (review 2026-09-26, D5: `Targets`' derived
/// `Debug` printed every field).
#[test]
fn a_hazard_column_beside_another_target_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "bocpd", "hazard_col": "h"},
            "targets": ["y"], "features": ["x"]}"#,
    );
    let err = s.validate().unwrap_err();
    assert!(err.contains("hazard_col") && err.contains("\"h\""), "{err}");
    assert!(err.contains("(got [\"y\"])"), "{err}");
}

/// The slot is the column itself: a table named like the hazard but
/// reading another column is refused, and so is the hazard column under
/// another name, for `bocpd` and `hmm` alike (review 2026-09-26, D4: the
/// names were compared, and the first put column `z` in the slot).
#[test]
fn a_renamed_table_cannot_smuggle_another_column_into_the_hazard_slot() {
    // `hmm` wants its `n_components`, a prior, the tvtp coefficients and a
    // half-life, which `bocpd` refuses.
    for (model, key, top) in [
        ("\"bocpd\"", "hazard_col", ""),
        (
            "\"hmm\", \"n_components\": 2, \"precision_prior\": 1.0, \
             \"tvtp_coef\": [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]",
            "exog_tvtp",
            ", \"half_life\": 20.0",
        ),
    ] {
        for targets in [
            r#"[{"column": "z", "name": "h"}]"#,
            r#"[{"column": "h", "name": "hz"}]"#,
        ] {
            let s = spec(&format!(
                r#"{{"name": "m", "model": {{"type": {model}, "{key}": "h"}},
                    "targets": {targets}, "features": ["x"]{top}}}"#
            ));
            let err = s.validate().unwrap_err();
            assert!(
                err.contains(key) && err.contains("the column itself"),
                "{err}"
            );
        }
        let s = spec(&format!(
            r#"{{"name": "m", "model": {{"type": {model}, "{key}": "h"}},
                "targets": ["h"], "features": ["x"]{top}}}"#
        ));
        assert!(
            s.validate().is_ok(),
            "{model}: the column itself: {:?}",
            s.validate()
        );
    }
}
