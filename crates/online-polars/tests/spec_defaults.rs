//! The resolved default of `min_backwards_jump`: `max_dclock`, pinned as a
//! number rather than through a refusal. `max_dclock` is the most two
//! adjacent rows can be apart and a session is longer than that, so a jump
//! back by less cannot be a boundary (2026-09-20). An infinite cap gives the
//! check nothing to compare against, so there the default is 0, off. Explicit
//! values win and `0` is off; `inf` is refused as no setting (`spec_inf.rs`).
//! The halflife plays no part: it is the model's memory, not a session.

use online_polars::Spec;

fn resolved(top: &str) -> f64 {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\n{top}\n[model]\ntype = \"ew_ridge\"\n"
    );
    let spec: Spec = toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"));
    spec.clock_cfg()
        .unwrap_or_else(|e| panic!("{e}"))
        .min_backwards_jump
}

#[test]
fn the_default_is_max_dclock() {
    assert_eq!(
        resolved("halflife = 10.0\nclock = \"t\"\nmax_dclock = 100.0"),
        100.0
    );
}

#[test]
fn the_halflife_plays_no_part() {
    assert_eq!(
        resolved("halflife = 100000.0\nclock = \"t\"\nmax_dclock = 100.0"),
        100.0
    );
}

#[test]
fn an_infinite_cap_defaults_to_off() {
    assert_eq!(
        resolved("halflife = 10.0\nclock = \"t\"\nmax_dclock = inf"),
        0.0
    );
}

#[test]
fn an_explicit_value_wins_including_zero_and_under_an_infinite_cap() {
    let base = "halflife = 10.0\nclock = \"t\"\nmax_dclock = 100.0\n";
    assert_eq!(resolved(&format!("{base}min_backwards_jump = 5.0")), 5.0);
    assert_eq!(resolved(&format!("{base}min_backwards_jump = 0.0")), 0.0);
    let inf = "halflife = 10.0\nclock = \"t\"\nmax_dclock = inf\n";
    assert_eq!(resolved(&format!("{inf}min_backwards_jump = 5.0")), 5.0);
}

/// The two readiness gates (docs/WARMUP-AND-CONVERGENCE.md §2), resolved:
/// `min_settled_frac` is off, `max_error_inflation` is `sqrt(2)` -- the
/// estimation variance equal to the noise -- and `ew_ridge`, which that gate
/// serves, no longer takes the `k + 1` floor `min_periods` gave it, where
/// every model without a noise statistic keeps its own default.
fn parsed(top: &str, model: &str) -> Spec {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\", \"x1\"]\nhalflife = 10.0\n{top}\n[model]\n{model}\n"
    );
    toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"))
}

#[test]
fn the_settled_gate_is_off_and_the_noise_gate_is_root_two() {
    let spec = parsed("", "type = \"ew_ridge\"");
    assert_eq!(spec.min_settled_frac_or_default(), 0.0);
    assert_eq!(spec.max_error_inflation_or_default(), 2f64.sqrt());
    let spec = parsed(
        "min_settled_frac = 0.5\nmax_error_inflation = 1.1",
        "type = \"ew_ridge\"",
    );
    assert_eq!(spec.min_settled_frac_or_default(), 0.5);
    assert_eq!(spec.max_error_inflation_or_default(), 1.1);
}

#[test]
fn ew_ridge_drops_the_count_floor_and_the_others_keep_theirs() {
    assert_eq!(
        parsed("", "type = \"ew_ridge\"").min_periods_per_target(),
        vec![0.0]
    );
    assert_eq!(
        parsed("", "type = \"lasso\"\nlasso_path = [0.1]").min_periods_per_target(),
        vec![3.0]
    );
    assert_eq!(
        parsed("", "type = \"rls\"").min_periods_per_target(),
        vec![3.0]
    );
    assert_eq!(
        parsed("min_periods = 7.0", "type = \"ew_ridge\"").min_periods_per_target(),
        vec![7.0]
    );
}
