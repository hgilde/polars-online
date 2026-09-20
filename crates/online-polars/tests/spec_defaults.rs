//! The resolved default of `min_session_clock`: the larger of `max_dclock`
//! and the halflife, pinned as numbers rather than through a refusal.
//!
//! `max_dclock` alone is a row-scale bound on a session-scale quantity: at
//! `max_dclock = 10` it caught none of the 2,965 backwards jumps a query
//! engine's block reordering produced (spans of 71 .. 2.5e6 clock units),
//! while the halflife refused at the seventh (2026-09-19). The Python tests
//! cross that threshold; these pin the value the spec resolves to, including
//! a `lam` read back as the halflife it is, and the `inf` cases.

use online_polars::Spec;

fn resolved(top: &str) -> f64 {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\n{top}\n[model]\ntype = \"ew_ridge\"\n"
    );
    let spec: Spec = toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"));
    spec.clock_cfg()
        .unwrap_or_else(|e| panic!("{e}"))
        .min_session_clock
}

#[test]
fn the_halflife_lifts_the_default_above_max_dclock() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = 100.0\nhalflife = 1000.0"),
        1000.0
    );
}

#[test]
fn max_dclock_stays_the_floor_where_the_halflife_is_smaller() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = 100.0\nhalflife = 10.0"),
        100.0
    );
}

#[test]
fn a_lam_is_read_as_the_halflife_it_is() {
    let lam = 0.5f64.powf(1.0 / 1000.0);
    let got = resolved(&format!(
        "clock = \"t\"\nmax_dclock = 100.0\nlam = {lam:.17}"
    ));
    assert!((got - 1000.0).abs() < 1e-6, "{got}");
}

#[test]
fn a_halflife_list_takes_its_longest_member() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = 100.0\nhalflife = [50.0, 2000.0, 300.0]"),
        2000.0
    );
}

#[test]
fn max_dclock_inf_falls_to_the_halflife_rather_than_off() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = inf\nhalflife = 1000.0"),
        1000.0
    );
}

#[test]
fn halflife_inf_falls_to_max_dclock() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = 100.0\nhalflife = inf"),
        100.0
    );
}

#[test]
fn neither_finite_is_off() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = inf\nhalflife = inf"),
        0.0
    );
}

#[test]
fn an_explicit_value_wins() {
    assert_eq!(
        resolved("clock = \"t\"\nmax_dclock = 100.0\nhalflife = 1000.0\nmin_session_clock = 5.0"),
        5.0
    );
}
