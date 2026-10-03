//! The clock's settings as task 120 decided them (the user, 2026-09-28).
//! `gap_cap` is a cap on a step and nothing else: finite and above 0.
//! `restart_after_step_back` has no default derived from it -- the caller
//! says what a late row is: unset, the default, every step back is refused
//! and no minimum is read; given, what is given is what the clock reads, `0`
//! included (one name since task 144). `session_gap` is finite or `"reset"`.

use online_core::OnClockReset;
use online_polars::Spec;

fn cfg(top: &str) -> Result<online_core::ClockCfg, String> {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\n{top}\n[model]\ntype = \"ew_ridge\"\n"
    );
    let spec: Spec = toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"));
    spec.clock_cfg()
}

const CLOCK: &str = "half_life = 10.0\nclock = \"t\"\ngap_cap = 100.0\n";

#[test]
fn unset_refuses_a_step_back_and_reads_no_minimum() {
    let c = cfg(CLOCK).unwrap();
    assert_eq!(c.on_clock_reset, OnClockReset::Error);
    assert_eq!(c.min_backwards_jump, 0.0);
}

#[test]
fn a_restart_threshold_reads_as_given() {
    // The cap plays no part: a minimum above it is the caller's to give.
    for v in [0.0, 5.0, 500.0] {
        let c = cfg(&format!("{CLOCK}restart_after_step_back = {v:?}")).unwrap();
        assert_eq!(c.min_backwards_jump, v);
        assert_eq!(c.on_clock_reset, OnClockReset::ResetState);
    }
    let err = cfg(&format!("{CLOCK}restart_after_step_back = -1.0")).unwrap_err();
    assert!(
        err.contains("restart_after_step_back must be finite and >= 0"),
        "{err}"
    );
}

#[test]
fn max_dclock_is_finite_and_above_zero() {
    let with = |v: &str| cfg(&format!("half_life = 10.0\nclock = \"t\"\ngap_cap = {v}"));
    let err = with("0.0").unwrap_err();
    assert!(
        err.contains("must be > 0") && err.contains("half_life = \"inf\""),
        "{err}"
    );
    let err = with("inf").unwrap_err();
    assert!(err.contains("must be finite"), "{err}");
    assert!(with("-5.0").unwrap_err().contains("finite number > 0"));
    assert_eq!(with("1e-9").unwrap().gap_cap, 1e-9);
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
/// serves, no longer takes the `k + 1` floor `min_weight` gave it, where
/// every model without a noise statistic keeps its own default.
fn parsed(top: &str, model: &str) -> Spec {
    let text = format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\", \"x1\"]\nhalf_life = 10.0\n{top}\n[model]\n{model}\n"
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
        parsed("min_weight = 7.0", "type = \"ew_ridge\"").min_periods_per_target(),
        vec![7.0]
    );
}
