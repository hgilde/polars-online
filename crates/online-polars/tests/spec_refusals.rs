//! What a spec is refused for, and the words: the spec, the parameter and
//! the value, from every door a spec comes in by (`Spec::check`, which the
//! bank, a run config and every builder go through). Review round 4
//! (2026-10-06): ceilings on what a spec value sizes (CD10, CF2, CE9),
//! parameters a mode does not read (CF6, CE4, PC6), `warm_rows` below `k`
//! (PC8), the empties (YA5), a ridge of `-0.0` (PC10), the value in every
//! refusal (PC11) and the spec's name on the core's (YA4).

use online_polars::{Spec, output_fields};

/// A spec from JSON, filled, validated and built, as every door does it.
fn check(json: &str) -> Result<(), String> {
    let mut s: Spec = serde_json::from_str(json).map_err(|e| format!("did not parse: {e}"))?;
    s.check()
}

/// A supervised spec over `y` from `x0` and `x1`, with `extra` keys.
fn sup(model: &str, extra: &str) -> String {
    format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0", "x1"],
            "half_life": 10.0{extra}}}"#
    )
}

/// A spec of a model with no target over `x0` and `x1`.
fn unsup(model: &str, extra: &str) -> String {
    format!(r#"{{"name": "m", "model": {model}, "features": ["x0", "x1"]{extra}}}"#)
}

fn refused(json: &str, want: &[&str]) {
    let e = check(json).expect_err(&format!("accepted: {json}"));
    for w in want {
        assert!(e.contains(w), "wanted {w:?} in: {e}");
    }
    assert!(
        e.starts_with("spec \"m\": "),
        "the spec is not named first: {e}"
    );
}

fn accepted(json: &str) {
    check(json).unwrap_or_else(|e| panic!("refused: {e}\n{json}"));
}

/// CD10: a lag sizes a ring before the first row, and `n_perm` the draws
/// held for a quantile: 2^20 each, refused by name with the value.
/// `marginal(lags=[2^62])` panicked "capacity overflow" inside the builder.
#[test]
fn a_lag_or_n_perm_past_the_ceiling_is_refused_by_name() {
    let window = r#""kind": "window", "span_rows": 10"#;
    refused(
        &unsup(
            &format!(r#"{{"type": "corrchange", {window}, "n_perm": 1048577}}"#),
            "",
        ),
        &[
            "corrchange: n_perm must be at most 1048576 (2^20)",
            "got 1048577",
        ],
    );
    accepted(&unsup(
        &format!(r#"{{"type": "corrchange", {window}, "n_perm": 1048576}}"#),
        "",
    ));
    for lag in [1_048_577u64, 1 << 62] {
        refused(
            &sup(
                &format!(r#"{{"type": "marginal", "lags": [1, {lag}]}}"#),
                "",
            ),
            &[
                "marginal: lags must be at most 1048576 (2^20)",
                &format!("got {lag}"),
            ],
        );
        refused(
            &unsup(
                &format!(r#"{{"type": "ew_cov", "lags": [{lag}]}}"#),
                r#", "half_life": 10.0"#,
            ),
            &["ew_cov lags must be at most 1048576", &format!("got {lag}")],
        );
        refused(
            &sup(
                r#"{"type": "ew_ridge"}"#,
                &format!(r#", "emit_autocorr": true, "resid_autocorr_lag": {lag}"#),
            ),
            &[
                "resid_autocorr_lag must be at most 1048576",
                &format!("got {lag}"),
            ],
        );
    }
}

/// CF2: `k` has a ceiling and the warm-up buffer a budget, each refused by
/// name with the value; micro's linkage holds a matrix of 128 MiB at most
/// (4,096 potential summaries) and `O(m)` memory past it, so its cap needs
/// no ceiling.
#[test]
fn kmeans_sizes_have_ceilings_and_micro_needs_none() {
    refused(
        &unsup(
            r#"{"type": "kmeans", "k": 65537, "warm_rows": 65537}"#,
            r#", "half_life": 10.0"#,
        ),
        &["kmeans: k must be at most 65536 (2^16)", "got 65537"],
    );
    refused(
        &unsup(
            r#"{"type": "kmeans", "k": 3, "warm_rows": 5592406}"#,
            r#", "half_life": 10.0"#,
        ),
        &[
            "kmeans: the warm-up buffer would hold 256.00003 MiB",
            "warm_rows 5592406 rows of 2 features",
        ],
    );
    accepted(&unsup(
        r#"{"type": "micro", "eps": 0.3, "max_clusters": 10000000}"#,
        r#", "half_life": 10.0"#,
    ));
}

/// `hmm`'s `k` sizes its transition matrix, `k²` cells, and its states
/// before the first row: 2^10, refused by name with the value, where
/// `k = 2^62` grew memory without bound (review 2026-10-06, CF2's sibling).
#[test]
fn hmm_k_has_a_ceiling() {
    let hmm = |k: u64| {
        unsup(
            &format!(r#"{{"type": "hmm", "k": {k}, "warm_rows": {k}, "precision_prior": 0.1}}"#),
            r#", "half_life": 10.0"#,
        )
    };
    for k in [1025u64, 1 << 62] {
        refused(
            &hmm(k),
            &["hmm: k must be at most 1024 (2^10)", &format!("got {k}")],
        );
    }
    accepted(&hmm(1024));
}

/// CE9: rcov's lagged products, `(ring + 1)·k²` doubles, are held to 256
/// MiB before the first row.
#[test]
fn rcov_lagged_products_have_a_byte_ceiling() {
    let ten = (0..10)
        .map(|i| format!("\"x{i}\""))
        .collect::<Vec<_>>()
        .join(", ");
    refused(
        &format!(
            r#"{{"name": "m", "model": {{"type": "rcov", "bandwidth": 1048576}},
                "features": [{ten}], "group": "g", "group_close": "monotone"}}"#
        ),
        &[
            "rcov: the lagged products would take 800 MiB",
            "over the 256 MiB",
        ],
    );
}

/// CF6 and PC8: kmeans' `dead_frac` beside `split_merge = 0`, and
/// `warm_rows` below `k`, refused by name; left out, `dead_frac` is 0 under
/// `split_merge = 0` and `warm_rows` is at least `k`.
#[test]
fn kmeans_refuses_a_dead_rule_with_no_check_and_too_few_warm_rows() {
    let km = |model: &str| unsup(model, r#", "half_life": 10.0"#);
    refused(
        &km(r#"{"type": "kmeans", "k": 3, "split_merge": 0.0, "dead_frac": 0.5}"#),
        &["kmeans: dead_frac", "split_merge = 0", "0.5"],
    );
    accepted(&km(r#"{"type": "kmeans", "k": 3, "split_merge": 0.0}"#));
    refused(
        &km(r#"{"type": "kmeans", "k": 3, "warm_rows": 1}"#),
        &["kmeans: warm_rows must be at least k (3)", "got 1"],
    );
    accepted(&km(r#"{"type": "kmeans", "k": 600}"#));
}

/// CE4: a parameter its mode does not read is refused by name: hmm's
/// `transition` and `transition_prior` beside `tvtp_coef`, its
/// `warm_rows`, `seed_rule` and `seed` beside given states; rcov's
/// `jitter` and `max_bandwidth` outside `"kernel"`, `theta` outside
/// `"preavg"`.
#[test]
fn a_parameter_its_mode_does_not_read_is_refused() {
    let hmm = |extra: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "hmm", "k": 2, "precision_prior": 0.1{extra}}},
                "features": ["x0", "x1"], "half_life": 10.0}}"#
        )
    };
    let tvtp = r#", "exog_tvtp": "z", "tvtp_coef": [[0, 0, 0, 0], [0, 0, 0, 0]]"#;
    let given = r#", "means": [-1, -1, 1, 1], "covs": [1, 0, 0, 1, 1, 0, 0, 1]"#;
    let with_z = |s: String| s.replace(r#""features""#, r#""targets": ["z"], "features""#);
    for (param, value) in [
        ("transition", "[0.9, 0.1, 0.2, 0.8]"),
        ("transition_prior", "2.0"),
    ] {
        refused(
            &with_z(hmm(&format!(r#"{tvtp}, "{param}": {value}"#))),
            &[&format!("hmm: {param} does not apply with tvtp_coef")],
        );
    }
    accepted(&with_z(hmm(tvtp)));
    for (param, value) in [
        ("warm_rows", "20"),
        ("seed_rule", "\"first\""),
        ("seed", "3"),
    ] {
        refused(
            &hmm(&format!(r#"{given}, "{param}": {value}"#)),
            &[&format!(
                "hmm: {param} does not apply with means and covs given"
            )],
        );
    }
    accepted(&hmm(given));
    let rcov = |model: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "features": ["x0", "x1"], "group": "g",
                "group_close": "monotone"}}"#
        )
    };
    for (kind, param, value, owner) in [
        ("plain", "jitter", "3", "\"kernel\""),
        ("plain", "max_bandwidth", "5", "\"kernel\""),
        ("plain", "theta", "2.0", "\"preavg\""),
        ("kernel", "theta", "2.0", "\"preavg\""),
        ("preavg", "jitter", "3", "\"kernel\""),
        ("preavg", "max_bandwidth", "5", "\"kernel\""),
    ] {
        refused(
            &rcov(&format!(
                r#"{{"type": "rcov", "kind": "{kind}", "block_rows": 100, "{param}": {value}}}"#
            )),
            &[&format!(
                "rcov: {param} applies to kind = {owner}, not \"{kind}\""
            )],
        );
    }
    accepted(&rcov(
        r#"{"type": "rcov", "kind": "kernel", "block_rows": 100, "jitter": 3}"#,
    ));
    accepted(&rcov(
        r#"{"type": "rcov", "kind": "preavg", "block_rows": 100, "theta": 2.0}"#,
    ));
}

/// A `covariance` shape not offered is refused naming the model it was
/// given to, the value and the shapes there are: `hmm`'s said "unknown
/// ew_class covariance" (task 190's find, folded into task 193).
#[test]
fn a_covariance_refusal_names_its_own_model() {
    refused(
        &unsup(
            r#"{"type": "hmm", "k": 2, "precision_prior": 0.1, "covariance": "lda"}"#,
            r#", "half_life": 10.0"#,
        ),
        &["unknown hmm covariance \"lda\" (expected full, shared or diagonal)"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"],
            "precision_prior": 1.0, "covariance": "lda"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
        &["unknown ew_class covariance \"lda\" (expected full, shared or diagonal)"],
    );
}

/// PC6: `kalman` takes `coef_half_life` or `q`, exactly one: `q` alone was
/// refused as a missing field, and the pair took `q` and ignored the
/// half-life.
#[test]
fn kalman_takes_exactly_one_of_coef_half_life_and_q() {
    accepted(&sup(r#"{"type": "kalman", "q": [0.0, 0.1, 0.1]}"#, ""));
    accepted(&sup(r#"{"type": "kalman", "coef_half_life": 50.0}"#, ""));
    refused(
        &sup(
            r#"{"type": "kalman", "coef_half_life": 50.0, "q": [0.0, 0.1, 0.1]}"#,
            "",
        ),
        &["kalman takes coef_half_life or q, not both"],
    );
    refused(
        &sup(r#"{"type": "kalman"}"#, ""),
        &["kalman needs coef_half_life", "or q"],
    );
}

/// YA5: an empty `mahal_quantiles` and an empty target name are refused,
/// as an empty `resid_quantiles` and `po.target("")` are.
#[test]
fn the_empties_are_refused() {
    refused(
        &unsup(
            r#"{"type": "ew_cov", "stats": ["mahal"], "precision_prior": 1.0, "mahal_quantiles": []}"#,
            r#", "half_life": 10.0"#,
        ),
        &["mahal_quantiles must be non-empty"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": [""], "features": ["x0"],
            "half_life": 10.0}"#,
        &["targets must not contain an empty name"],
    );
}

/// PC10: `-0.0` is `0.0`: a ridge grid of the two is one value listed
/// twice, and a `-0.0` names its field `__r0`.
#[test]
fn minus_zero_is_zero_in_a_grid_and_a_field_name() {
    refused(
        &sup(r#"{"type": "ew_ridge", "ridge": [0.0, -0.0]}"#, ""),
        &["ridge lists 0 more than once"],
    );
    let mut s: Spec =
        serde_json::from_str(&sup(r#"{"type": "ew_ridge", "ridge": [-0.0, 1.0]}"#, "")).unwrap();
    s.check().unwrap();
    let fields = output_fields(&s);
    assert!(
        fields.contains(&"pred_y__r0".to_string()) && !fields.iter().any(|f| f.contains("-0")),
        "{fields:?}"
    );
}

/// PC11: a refusal of a value names it. A sample across the shared
/// parameters and the models, each refused with `got <value>`.
#[test]
fn a_refused_value_is_named() {
    for (json, want) in [
        (
            sup(r#"{"type": "ew_ridge", "ridge": -1.0}"#, ""),
            "ridge must be finite and >= 0, got -1",
        ),
        (
            sup(r#"{"type": "ew_ridge"}"#, r#", "min_weight": -2.0"#),
            "min_weight must be >= 0, got -2",
        ),
        (
            sup(r#"{"type": "ew_ridge", "solve_every": -3.0}"#, ""),
            "solve_every must be finite and >= 0 (0 solves every row), got -3",
        ),
        (
            sup(r#"{"type": "ew_ridge"}"#, r#", "conformal": 1.5"#),
            "got 1.5",
        ),
        (
            sup(r#"{"type": "lasso", "lasso_path": [0.1], "tol": 0.0}"#, ""),
            "tol must be finite and > 0, got 0",
        ),
        (
            sup(
                r#"{"type": "kalman", "coef_half_life": 50.0, "p0": -1.0}"#,
                "",
            ),
            "p0 must be finite and > 0, got -1",
        ),
        (
            sup(r#"{"type": "ftrl", "alpha": 0.0}"#, ""),
            "ftrl alpha must be finite and > 0, got 0",
        ),
        (
            sup(r#"{"type": "sgd", "learning_rate": -0.5}"#, ""),
            "learning_rate must be finite and > 0, got -0.5",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "k": 2, "dead_frac": -1.0}"#,
                r#", "half_life": 10.0"#,
            ),
            "dead_frac must be finite and >= 0 (0 disables it), got -1",
        ),
        (
            unsup(
                r#"{"type": "micro", "eps": -0.5}"#,
                r#", "half_life": 10.0"#,
            ),
            "micro eps must be finite and > 0, got -0.5",
        ),
        (
            r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
                "features": ["x0"], "half_life": [10.0, -4.0]}"#
                .to_string(),
            "half_life must be > 0 (\"inf\" for no decay), got -4",
        ),
    ] {
        refused(&json, &[want]);
    }
}

/// YA4: a refusal raised by a model's own check names the spec, as the
/// spec's own refusals do; and a window that is not finite is refused as
/// "finite and > 0", which "> 0 (got inf)" was untrue of.
#[test]
fn a_models_refusal_names_the_spec() {
    refused(
        &sup(r#"{"type": "sgd", "clip_gradient": 0.0}"#, ""),
        &["sgd: clip_gradient must be > 0"],
    );
    for model in [
        r#"{"type": "lasso", "lasso_path": [0.1], "window_size": "inf"}"#,
        r#"{"type": "marginal", "window_size": "inf"}"#,
    ] {
        refused(
            &sup(model, ""),
            &["window_size must be finite and > 0 (got inf)"],
        );
    }
}

/// PC11: an old name is refused naming the new one, and cites nothing a
/// wheel's user does not have.
#[test]
fn a_renamed_key_cites_no_plan_task() {
    let msg = online_polars::name_renamed("unknown field `halflife`, expected one of `x`");
    assert!(msg.contains("halflife was renamed half_life"), "{msg}");
    assert!(!msg.contains("PLAN") && !msg.contains("task"), "{msg}");
}
