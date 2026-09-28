//! The clock's settings as task 120 decided them (the user, 2026-09-28).
//! `max_dclock` is a cap on a step and nothing else: finite and above 0.
//! `min_backwards_jump` has no default derived from it -- the caller says
//! what a late row is: it is required with `on_clock_reset =
//! "reset_state"`, refused with `"error"`, the default, which refuses every
//! step back and reads no minimum, and what is given is what the clock
//! reads, `0` included. `session_gap` is finite or `"reset"`.

use online_core::OnClockReset;
use online_polars::Spec;

fn cfg(top: &str) -> Result<online_core::ClockCfg, String> {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\n{top}\n[model]\ntype = \"ew_ridge\"\n"
    );
    let spec: Spec = toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"));
    spec.clock_cfg()
}

const CLOCK: &str = "halflife = 10.0\nclock = \"t\"\nmax_dclock = 100.0\n";

#[test]
fn the_default_policy_refuses_a_step_back_and_reads_no_minimum() {
    let c = cfg(CLOCK).unwrap();
    assert_eq!(c.on_clock_reset, OnClockReset::Error);
    assert_eq!(c.min_backwards_jump, 0.0);
    let err = cfg(&format!("{CLOCK}min_backwards_jump = 5.0")).unwrap_err();
    assert!(
        err.contains("applies only under on_clock_reset = \"reset_state\""),
        "{err}"
    );
}

#[test]
fn reset_state_requires_a_minimum_and_reads_the_one_given() {
    let reset = format!("{CLOCK}on_clock_reset = \"reset_state\"\n");
    let err = cfg(&reset).unwrap_err();
    assert!(err.contains("min_backwards_jump is required"), "{err}");
    // The cap plays no part: a minimum above it is the caller's to give.
    for v in [0.0, 5.0, 500.0] {
        let c = cfg(&format!("{reset}min_backwards_jump = {v:?}")).unwrap();
        assert_eq!(c.min_backwards_jump, v);
        assert_eq!(c.on_clock_reset, OnClockReset::ResetState);
    }
}

#[test]
fn max_dclock_is_finite_and_above_zero() {
    let with = |v: &str| cfg(&format!("halflife = 10.0\nclock = \"t\"\nmax_dclock = {v}"));
    let err = with("0.0").unwrap_err();
    assert!(
        err.contains("must be > 0") && err.contains("halflife = \"inf\""),
        "{err}"
    );
    let err = with("inf").unwrap_err();
    assert!(err.contains("must be finite"), "{err}");
    assert!(with("-5.0").unwrap_err().contains("finite number > 0"));
    assert_eq!(with("1e-9").unwrap().max_dclock, 1e-9);
}

#[test]
fn session_gap_is_finite_or_reset() {
    let with = |v: &str| cfg(&format!("{CLOCK}session = \"s\"\nsession_gap = {v}"));
    let err = with("inf").unwrap_err();
    assert!(
        err.contains("must be finite") && err.contains("\"reset\""),
        "{err}"
    );
    assert!(with("0.0").is_ok() && with("\"reset\"").is_ok());
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
