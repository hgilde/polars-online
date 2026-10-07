//! The clock's settings as task 120 decided them (the user, 2026-09-28).
//! `gap_cap` is a cap on a step and nothing else: finite and above 0.
//! `restart_after_step_back` has no default derived from it -- the caller
//! says what a late row is: unset, the default, every step back is refused
//! and no minimum is read; given, what is given is what the clock reads, `0`
//! included (one name since task 144). `session_gap` is finite or `"reset"`.

use online_core::OnClockReset;
use online_polars::Spec;
use serde_json::{Value, json};

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

// The defaults a spec leaves to this side, read the way the API snapshot
// reads them (`online_polars::resolved_defaults`, `tests/api_surface.txt`'s
// `[resolved defaults]`). Each expected value is the one the documentation
// states -- the README, the spec's and the models' doc comments, the paper a
// rule comes from -- not one read back from the code (review 2026-10-06,
// AP1, DB4).

/// What a spec written in TOML, as the CLI reads one, resolves to.
fn resolved(text: &str) -> Value {
    let spec: Spec = toml::from_str(text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"));
    online_polars::resolved_defaults(&spec).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// A spec over two features and one target with a half-life of 10, `top`
/// beside it, and `model` as its `[model]` table.
fn resolved_with(top: &str, model: &str) -> Value {
    resolved(&format!(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\", \"x1\"]\nhalf_life = 10.0\n{top}\n\
         [model]\n{model}\n"
    ))
}

/// The two defaults the stability table said nothing pinned (DB4):
/// `emit_averaged`'s sharpness `average_eta = 1` and `sgd`'s gradient clip
/// `clip_gradient = 1e3`.
#[test]
fn average_eta_is_one_and_the_sgd_clip_a_thousand() {
    let r = resolved_with("", "type = \"sgd\"");
    assert_eq!(r["stream"]["average_eta"], 1.0);
    assert_eq!(r["model"]["clip_gradient"], 1e3);
    let r = resolved_with("", "type = \"sgd\"\nclip_gradient = 5.0");
    assert_eq!(r["model"]["clip_gradient"], 5.0);
}

/// The diagnostics' defaults as `Spec`'s fields document them, the readiness
/// gates as the README's *Warm-up* table does, and each given value taken.
#[test]
fn the_stream_settings_resolve_as_documented() {
    let r = resolved_with("", "type = \"ew_ridge\"");
    let s = &r["stream"];
    assert_eq!(s["drift_action"], "flag");
    assert_eq!(s["drift_delta"], 0.5);
    assert_eq!(s["drift_threshold"], 20.0);
    assert_eq!(s["conformal_rate"], 0.05);
    assert_eq!(s["resid_autocorr_lag"], 1);
    assert_eq!(s["average_eta"], 1.0);
    assert_eq!(s["min_settled_frac"], 0.0);
    assert_eq!(s["max_error_inflation"], 2f64.sqrt());
    assert_eq!(s["window_budget"], Value::Null);
    assert!(s.get("shards").is_none(), "{s}");
    let r = resolved_with(
        "emit_drift = true\ndrift_action = \"reset\"\ndrift_delta = 0.25\n\
         drift_threshold = 5.0\nconformal = 0.9\nconformal_rate = 0.1\nemit_autocorr = true\n\
         resid_autocorr_lag = 3\nemit_averaged = true\naverage_eta = 2.0\n\
         min_settled_frac = 0.5\nmax_error_inflation = \"inf\"",
        "type = \"ew_ridge\"\nridge = [0.1, 1.0]",
    );
    let s = &r["stream"];
    assert_eq!(s["drift_action"], "reset");
    assert_eq!(s["drift_delta"], 0.25);
    assert_eq!(s["drift_threshold"], 5.0);
    assert_eq!(s["conformal_rate"], 0.1);
    assert_eq!(s["resid_autocorr_lag"], 3);
    assert_eq!(s["average_eta"], 2.0);
    assert_eq!(s["min_settled_frac"], 0.5);
    assert_eq!(s["max_error_inflation"], "inf");
}

/// `ew_ridge`'s gate is its noise statistic, so its `min_weight` resolves
/// to 0 (the README's last row), while the model keeps a row per unknown as
/// the floor of its first solve (`build_bare`'s comment); every other
/// regression gates on that count itself.
#[test]
fn the_ridge_keeps_its_count_floor_in_the_model_and_gates_at_zero() {
    let r = resolved_with("", "type = \"ew_ridge\"");
    assert_eq!(r["stream"]["min_weight"], json!([0.0]));
    assert_eq!(r["model"]["min_weight"], 3.0);
    let r = resolved_with("", "type = \"rls\"");
    assert_eq!(r["stream"]["min_weight"], json!([3.0]));
    assert_eq!(r["model"]["min_weight"], 3.0);
}

/// The clock policy as task 120 decided it: without a clock, a row is one
/// step with no cap and every step back is refused; given, each setting is
/// what the spec says.
#[test]
fn the_clock_policy_resolves_as_task_120_decided() {
    let r = resolved_with("", "type = \"ew_ridge\"");
    assert_eq!(
        r["clock"],
        json!({"gap_cap": "inf", "min_backwards_jump": 0.0, "on_clock_reset": "error",
               "session_gap": null})
    );
    let clocked = "clock = \"t\"\ngap_cap = 5.0\nrestart_after_step_back = 2.0\nsession = \"s\"";
    let r = resolved_with(
        &format!("{clocked}\nsession_gap = 3.0"),
        "type = \"ew_ridge\"",
    );
    assert_eq!(
        r["clock"],
        json!({"gap_cap": 5.0, "min_backwards_jump": 2.0, "on_clock_reset": "reset_state",
               "session_gap": 3.0})
    );
    let r = resolved_with(
        &format!("{clocked}\nsession_gap = \"reset\""),
        "type = \"ew_ridge\"",
    );
    assert_eq!(r["clock"]["session_gap"], "reset");
}

/// A window refuses past 256 MiB where its spec names no budget
/// (`ModelKind::window_budget`), and runs under the one it names.
#[test]
fn a_window_runs_under_a_256_mib_refusing_budget_unless_told() {
    let r = resolved_with("", "type = \"ew_ridge\"\nwindow_size = 100.0");
    assert_eq!(r["stream"]["window_budget"], json!({"refuse": 256.0}));
    let r = resolved_with(
        "",
        "type = \"ew_ridge\"\nwindow_size = 100.0\nwindow_budget = { thin = \"inf\" }",
    );
    assert_eq!(r["stream"]["window_budget"], json!({"thin": "inf"}));
}

/// The values a model derives from a field left unset, by its documented
/// rule: `bocpd`'s `nu0` (3 under `diag`, `d + 2` under `gaussian`) and the
/// first `warm_rows = d + 2` rows' mean and scale, which no build of a spec
/// can show and read "data" (`BocpdCfg`; docs/PLAN.md task 195, U4 and U5);
/// `rcov`'s pre-averaging window `ceil(theta * n^0.6)` (CKP
/// Eq. 16, `theta` 1) and ring `ceil(c* * n^0.6)` with BNHLS's Parzen
/// `c* = 3.5134`; `marginal`'s bin budget of 256 MiB; and a single shard.
#[test]
fn the_models_derive_what_their_spec_leaves_unset() {
    let bocpd = |extra: &str| {
        resolved(&format!(
            "name = \"m\"\nfeatures = [\"x0\", \"x1\"]\n[model]\ntype = \"bocpd\"\n{extra}\n"
        ))
    };
    assert_eq!(
        bocpd("")["derived"],
        json!({"prior_mean": "data", "prior_nu": 3.0, "prior_scale": "data", "warm_rows": 4})
    );
    assert_eq!(
        bocpd("emission = \"gaussian\"\nwarm_rows = 9")["derived"],
        json!({"prior_mean": "data", "prior_nu": 4.0, "prior_scale": "data", "warm_rows": 9})
    );
    let given = bocpd("prior_nu = 7.0\nprior_scale = [2.0]\nprior_mean = [1.0, -1.0]");
    assert_eq!(
        given["derived"],
        json!({"prior_mean": [1.0, -1.0], "prior_nu": 7.0, "prior_scale": [2.0, 0.0, 0.0, 2.0], "warm_rows": null})
    );
    let rcov = resolved(
        "name = \"m\"\nfeatures = [\"x0\", \"x1\"]\ngroup = \"g\"\ngroup_close = \"monotone\"\n\
         [model]\ntype = \"rcov\"\nblock_rows = 100\n",
    );
    let n06 = 100f64.powf(0.6);
    assert_eq!(rcov["derived"]["preavg_rows"], n06.ceil() as u64);
    assert_eq!(
        rcov["derived"]["max_bandwidth"],
        (3.5134 * n06).ceil() as u64
    );
    let marginal = resolved_with("", "type = \"marginal\"\nbins = 4");
    assert_eq!(marginal["derived"], json!({"bin_budget": 256.0}));
    assert_eq!(marginal["stream"]["shards"], 1);
    let marginal = resolved_with(
        "",
        "type = \"marginal\"\nbins = 4\nbin_budget = 64.0\nshards = 3",
    );
    assert_eq!(marginal["derived"], json!({"bin_budget": 64.0}));
    assert_eq!(marginal["stream"]["shards"], 3);
    assert!(
        resolved_with("", "type = \"marginal\"")
            .get("derived")
            .is_none()
    );
    assert!(
        resolved_with("", "type = \"ew_ridge\"")
            .get("derived")
            .is_none()
    );
}

/// A spec the bank refuses is refused here, with the bank's words.
#[test]
fn a_spec_the_bank_refuses_resolves_to_its_refusal() {
    let spec: Spec = toml::from_str(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\n[model]\ntype = \"ew_ridge\"\n",
    )
    .unwrap();
    let err = online_polars::resolved_defaults(&spec).unwrap_err();
    assert!(err.contains("one of half_life/lam is required"), "{err}");
}
