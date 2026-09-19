//! Where `inf` means something both layers take it, and where it means
//! nothing both refuse it by name (review 2026-09-12, S27; the user's
//! decision of 2026-09-15).
//!
//! The layers disagreed where `inf` came from TOML. A plain `f64` field
//! refused JSON's string `"inf"` at the parser, so Python never met the
//! question, while `validate` let TOML's real `inf` through wherever its
//! check was `> 0` or `>= 0` alone. JSON's route is
//! `tests/test_error_messages.py`'s; this is TOML's, where `validate` is the
//! only gate.

use online_polars::{Bank, Spec};

/// A spec with `top` beside the defaults and `model` as its model table.
fn spec(top: &str, model: &str) -> Spec {
    let base = if top.contains("features") {
        ""
    } else {
        "features = [\"x0\"]\nhalflife = 10.0\n"
    };
    let text = format!("name = \"m\"\ntargets = [\"y\"]\n{base}{top}\n[model]\n{model}\n");
    toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"))
}

/// Where `inf` is a limit with a name: least squares, the whole history, no
/// forgetting, a step nothing caps, the argmin.
const MEANS_SOMETHING: [(&str, &str); 7] = [
    ("", "type = \"huber\"\nhuber_delta = inf"),
    ("", "type = \"sgd\"\nloss = \"huber\"\nhuber_delta = inf"),
    (
        "session = \"s\"\nsession_gap = 1.0",
        "type = \"ew_ridge\"\nsession_shrink = 0.5\nlong_halflife = inf",
    ),
    (
        "",
        "type = \"lasso\"\nlasso_path = [0.1]\nselect_halflife = inf",
    ),
    ("features = []", "type = \"holt\"\nlevel_halflife = inf"),
    ("", "type = \"pa\"\nc = inf"),
    (
        "emit_averaged = true\naverage_eta = inf",
        "type = \"ew_ridge\"\nridge = [1e-6, 1.0]",
    ),
];

/// Where it is no setting: a step, a penalty, a tube or a threshold at `inf`.
const MEANS_NOTHING: [(&str, &str, &str); 13] = [
    // The disorder checks: `inf` would refuse every second jump, or every
    // jump; `0` is the way to switch one off (design note of 2026-09-19).
    (
        "clock = \"t\"\nmax_dclock = 10.0\nmin_session_clock = inf",
        "type = \"ew_ridge\"",
        "min_session_clock",
    ),
    (
        "clock = \"t\"\nmax_dclock = 10.0\nbackwards_jitter_ratio = inf",
        "type = \"ew_ridge\"",
        "backwards_jitter_ratio",
    ),
    (
        "emit_drift = true\ndrift_delta = inf",
        "type = \"ew_ridge\"",
        "drift_delta",
    ),
    (
        "emit_drift = true\ndrift_threshold = inf",
        "type = \"ew_ridge\"",
        "drift_threshold",
    ),
    (
        "",
        "type = \"quantile\"\nquantile = 0.5\nquantile_eps = inf",
        "quantile_eps",
    ),
    ("", "type = \"pa\"\neps = inf", "eps"),
    ("", "type = \"ftrl\"\nalpha = inf", "alpha"),
    ("", "type = \"ftrl\"\nbeta = inf", "beta"),
    ("", "type = \"ftrl\"\nl1 = inf", "l1"),
    ("", "type = \"ftrl\"\nl2 = inf", "l2"),
    ("", "type = \"sgd\"\nlearning_rate = inf", "learning_rate"),
    ("", "type = \"ew_ridge\"\nridge = inf", "ridge"),
    (
        "",
        "type = \"kalman\"\ncoef_halflife = 10.0\nq = [inf, 0.5]",
        "q",
    ),
];

#[test]
fn inf_is_taken_where_it_means_something() {
    let refused: Vec<String> = MEANS_SOMETHING
        .iter()
        .filter_map(|(top, model)| {
            let mut s = spec(top, model);
            let built = s.check().and_then(|()| Bank::new(vec![s]).map(|_| ()));
            built.err().map(|e| format!("{model}: {e}"))
        })
        .collect();
    assert!(refused.is_empty(), "{refused:#?}");
}

#[test]
fn inf_is_refused_by_name_where_it_means_nothing() {
    let wrong: Vec<String> = MEANS_NOTHING
        .iter()
        .filter_map(|(top, model, name)| match spec(top, model).check() {
            Ok(()) => Some(format!("{name} = inf was accepted")),
            Err(e) if !(e.contains(name) && e.contains("finite")) => Some(format!("{name}: {e}")),
            Err(_) => None,
        })
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Where NaN is no setting at all -- which is everywhere -- and the spec
/// layer leaves the check to the core: the fields whose core `validate`
/// tested `v <= 0.0` / `v < 0.0`, which a NaN passes (review 2026-09-18,
/// B4). Each is refused, by the spec or by the bank, with its name.
const NAN_IS_NO_SETTING: [(&str, &str, &str); 9] = [
    ("", "type = \"sgd\"\nclip_gradient = nan", "clip_gradient"),
    ("", "type = \"sgd\"\nl2 = nan", "l2"),
    (
        "",
        "type = \"sgd\"\nloss = \"epsilon_insensitive\"\neps = nan",
        "eps",
    ),
    (
        "",
        "type = \"sgd\"\nschedule = \"inv_scaling\"\npower = nan",
        "power",
    ),
    ("", "type = \"kalman\"\ncoef_halflife = nan", "halflife"),
    (
        "",
        "type = \"kalman\"\ncoef_halflife = 10.0\np0 = nan",
        "p0",
    ),
    (
        "",
        "type = \"kalman\"\ncoef_halflife = 10.0\nq = [nan, 0.5]",
        "q",
    ),
    (
        "",
        "type = \"kalman\"\ncoef_halflife = 10.0\nobs_var = nan",
        "obs_var",
    ),
    (
        "",
        "type = \"rls\"\ncoef_prior = [[nan, 0.0]]",
        "coef_prior",
    ),
];

#[test]
fn nan_is_refused_by_name_where_the_core_is_the_only_gate() {
    let wrong: Vec<String> = NAN_IS_NO_SETTING
        .iter()
        .filter_map(|(top, model, name)| {
            let mut s = spec(top, model);
            let built = s.check().and_then(|()| Bank::new(vec![s]).map(|_| ()));
            match built {
                Ok(()) => Some(format!("{name} = nan was accepted")),
                Err(e) if !e.contains(name) => {
                    Some(format!("{name}: refused without its name: {e}"))
                }
                Err(_) => None,
            }
        })
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `Num` reads JSON's `"nan"` as it reads `"inf"`, so each field that became
/// one needs a `validate` that refuses NaN; TOML's `nan` reaches the same
/// check. `sgd`'s `huber_delta` had none, in the spec or in the core, and a
/// NaN there reached `f64::clamp`, which panics on a NaN bound.
#[test]
fn nan_is_refused_where_inf_means_something() {
    let accepted: Vec<String> = MEANS_SOMETHING
        .iter()
        .map(|(top, model)| {
            (
                top.replace("= inf", "= nan"),
                model.replace("= inf", "= nan"),
            )
        })
        .filter(|(top, model)| spec(top, model).check().is_ok())
        .map(|(_, model)| model)
        .collect();
    assert!(accepted.is_empty(), "nan accepted: {accepted:#?}");
}
