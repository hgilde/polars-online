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
    let window = r#""kind": "window", "window_rows": 10"#;
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
                r#"{"type": "ewridge"}"#,
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
            r#"{"type": "kmeans", "n_clusters": 65537, "warm_rows": 65537}"#,
            r#", "half_life": 10.0"#,
        ),
        &[
            "kmeans: n_clusters must be at most 65536 (2^16)",
            "got 65537",
        ],
    );
    refused(
        &unsup(
            r#"{"type": "kmeans", "n_clusters": 3, "warm_rows": 5592406}"#,
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
            &format!(
                r#"{{"type": "hmm", "n_components": {k}, "warm_rows": {k}, "precision_prior": 0.1}}"#
            ),
            r#", "half_life": 10.0"#,
        )
    };
    for k in [1025u64, 1 << 62] {
        refused(
            &hmm(k),
            &[
                "hmm: n_components must be at most 1024 (2^10)",
                &format!("got {k}"),
            ],
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
        &km(r#"{"type": "kmeans", "n_clusters": 3, "split_merge": 0.0, "dead_frac": 0.5}"#),
        &["kmeans: dead_frac", "split_merge = 0", "0.5"],
    );
    accepted(&km(
        r#"{"type": "kmeans", "n_clusters": 3, "split_merge": 0.0}"#,
    ));
    refused(
        &km(r#"{"type": "kmeans", "n_clusters": 3, "warm_rows": 1}"#),
        &["kmeans: warm_rows must be at least n_clusters (3)", "got 1"],
    );
    accepted(&km(r#"{"type": "kmeans", "n_clusters": 600}"#));
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
            r#"{{"name": "m", "model": {{"type": "hmm", "n_components": 2, "precision_prior": 0.1{extra}}},
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
    for (param, value) in [("warm_rows", "20"), ("init", "\"first\""), ("seed", "3")] {
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

/// A `covariance_type` shape not offered is refused naming the spec, the
/// parameter, the value and the shapes there are: `hmm`'s said "unknown
/// ew_class covariance" (task 190's find, folded into task 193), and the
/// parameter is one word on every surface now (task 244, P-4).
#[test]
fn a_covariance_refusal_names_the_spec_and_the_parameter() {
    refused(
        &unsup(
            r#"{"type": "hmm", "n_components": 2, "precision_prior": 0.1, "covariance_type": "lda"}"#,
            r#", "half_life": 10.0"#,
        ),
        &["covariance_type must be one of \"full\", \"tied\", \"diag\"; got \"lda\""],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"],
            "precision_prior": 1.0, "covariance_type": "lda"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
        &["covariance_type must be one of \"full\", \"tied\", \"diag\"; got \"lda\""],
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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": [""], "features": ["x0"],
            "half_life": 10.0}"#,
        &["targets must not contain an empty name"],
    );
    // An empty `lags` asked for nothing, and `marginal` took it as no lags
    // where `ew_cov` refuses it; `cross_lags = []` is a choice, the least
    // memory, and stays.
    refused(
        &unsup(
            r#"{"type": "ew_cov", "lags": []}"#,
            r#", "half_life": 10.0"#,
        ),
        &["ew_cov lags must be non-empty"],
    );
    refused(
        &sup(r#"{"type": "marginal", "lags": []}"#, ""),
        &["marginal lags must be non-empty; leave it out for none"],
    );
    refused(
        &sup(r#"{"type": "marginal", "lags": [], "cross_lags": []}"#, ""),
        &["marginal lags must be non-empty"],
    );
    accepted(&sup(
        r#"{"type": "marginal", "lags": [1, 2], "cross_lags": []}"#,
        "",
    ));
    accepted(&sup(r#"{"type": "marginal", "lags": null}"#, ""));
}

/// PC10: `-0.0` is `0.0`: a ridge grid of the two is one value listed
/// twice, and a `-0.0` names its field `__r0`.
#[test]
fn minus_zero_is_zero_in_a_grid_and_a_field_name() {
    refused(
        &sup(r#"{"type": "ewridge", "ridge": [0.0, -0.0]}"#, ""),
        &["ridge lists 0 more than once"],
    );
    let mut s: Spec =
        serde_json::from_str(&sup(r#"{"type": "ewridge", "ridge": [-0.0, 1.0]}"#, "")).unwrap();
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
            sup(r#"{"type": "ewridge", "ridge": -1.0}"#, ""),
            "ridge must be finite and >= 0, got -1",
        ),
        (
            sup(r#"{"type": "ewridge"}"#, r#", "min_weight": -2.0"#),
            "min_weight must be >= 0, got -2",
        ),
        (
            sup(r#"{"type": "ewridge", "solve_every": -3.0}"#, ""),
            "solve_every must be finite and >= 0 (0 solves every row), got -3",
        ),
        (
            sup(r#"{"type": "ewridge"}"#, r#", "conformal": 1.5"#),
            "got 1.5",
        ),
        (
            sup(r#"{"type": "lasso", "penalties": [0.1], "tol": 0.0}"#, ""),
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
                r#"{"type": "kmeans", "n_clusters": 2, "dead_frac": -1.0}"#,
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
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
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
        r#"{"type": "lasso", "penalties": [0.1], "window_size": "inf"}"#,
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

// ---- Task 252: each refusal below had no test that reached it. ----

/// A count that must be at least 1, given as 0, is refused by name with
/// the value, for every parameter that sizes or schedules something.
#[test]
fn a_count_of_zero_is_refused_by_name() {
    const HL: &str = r#", "half_life": 10.0"#;
    let cases: Vec<(String, &str)> = vec![
        (
            unsup(r#"{"type": "kmeans", "n_clusters": 0}"#, HL),
            "kmeans n_clusters must be >= 1, got 0",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "n_clusters": 2, "update_every_rows": 0}"#,
                HL,
            ),
            "update_every_rows must be >= 1, got 0",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "n_clusters": 2, "split_merge_every_rows": 0}"#,
                HL,
            ),
            "split_merge_every_rows must be >= 1, got 0",
        ),
        (
            unsup(r#"{"type": "micro", "eps": 0.5, "max_clusters": 0}"#, HL),
            "max_clusters must be >= 1, got 0",
        ),
        (
            sup(
                r#"{"type": "ewridge"}"#,
                r#", "emit_specification": true, "ljung_box_lags": 0"#,
            ),
            "ljung_box_lags must be >= 1, got 0",
        ),
        (
            sup(
                r#"{"type": "ewridge"}"#,
                r#", "emit_autocorr": true, "resid_autocorr_lag": 0"#,
            ),
            "resid_autocorr_lag must be >= 1, got 0",
        ),
    ];
    for (json, want) in cases {
        refused(&json, &[want]);
    }
}

/// `horizon_rows` sizes Newey and West's ring before the first row: past
/// half the lag ceiling it is refused with the ceiling, and at it accepted.
#[test]
fn a_horizon_past_half_the_lag_ceiling_is_refused() {
    let half = online_polars::online_core::MAX_LAG / 2;
    let at = |h: usize| {
        sup(
            r#"{"type": "ewridge"}"#,
            &format!(r#", "emit_calibration": true, "horizon_rows": {h}"#),
        )
    };
    refused(
        &at(half + 1),
        &[
            &format!(
                "horizon_rows must be at most {half} (2^19), got {}",
                half + 1
            ),
            "Newey and West",
        ],
    );
    accepted(&at(half));
}

/// `gram_threads = 0` is refused by the reader, and a spec built in Rust
/// with it is refused by `check`, naming the parameter and the way out.
#[test]
fn a_spec_built_in_rust_with_no_gram_threads_is_refused() {
    let mut s: Spec = serde_json::from_str(&sup(r#"{"type": "ewridge"}"#, "")).unwrap();
    let online_polars::ModelKind::EwRidge { gram_threads, .. } = &mut s.model else {
        unreachable!("an ewridge spec");
    };
    *gram_threads = Some(online_polars::CountOrAuto::Count(0));
    let e = s.check().expect_err("gram_threads = 0 accepted");
    assert!(
        e.starts_with("spec \"m\": gram_threads must be >= 1") && e.contains("got 0"),
        "{e}"
    );
}

/// What a model cannot take: holt's features, `ew_class`'s second target,
/// a `coef_prior` of the wrong shape, a feature set naming a column that is
/// not a feature, and `corrchange`'s sequential kind without its history.
#[test]
fn a_model_given_what_it_cannot_read_is_refused() {
    refused(
        r#"{"name": "m", "model": {"type": "holt"}, "targets": ["y"], "features": ["x0"],
            "half_life": 10.0}"#,
        &["holt takes no features (got 1)"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0},
            "targets": ["y", "z"], "features": ["x0"], "half_life": 10.0}"#,
        &["ew_class takes exactly one target, the label column (got 2)"],
    );
    refused(
        &sup(r#"{"type": "rls", "coef_prior": [[1.0, 2.0]]}"#, ""),
        &["coef_prior must be n_targets x (n_features + intercept)"],
    );
    refused(
        &sup(
            r#"{"type": "ewridge", "feature_sets": [["a", ["x0", "nope"]]]}"#,
            "",
        ),
        &["feature set \"a\" references unknown feature \"nope\""],
    );
    let e = check(&unsup(
        r#"{"type": "deco", "blocks": [["a", ["x0", "zz"]], ["b", ["x1", "x2"]]]}"#,
        r#", "half_life": 10.0"#,
    ))
    .expect_err("a deco block naming no feature accepted");
    assert!(e.contains("zz"), "{e}");
    let e = check(&unsup(
        r#"{"type": "corrchange", "kind": "sequential"}"#,
        "",
    ))
    .expect_err("sequential without window_rows accepted");
    assert!(
        e.contains("kind = \"sequential\" needs `window_rows`")
            && e.contains("the rows of history each monitoring period is tested against"),
        "{e}"
    );
}

/// A formula target is refused, with the reason, by a model that does not
/// regress on a real-valued target: `seqtest` tests signs, `sgd` under the
/// logistic loss fits a probability and under the Poisson one a log rate.
/// (A comparison is a `seqtest`, refused for its signs before its sides.)
#[test]
fn a_formula_target_is_refused_by_a_model_it_does_not_apply_to() {
    let target = r#"[{"name": "fwd", "formula": ["rewm_mean", ["col", "y"], {"half_life": 5.0, "window_size": 10.0}]}]"#;
    let with = |model: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "targets": {target}, "features": ["x0"],
                "half_life": 10.0, "embargo": 10.0}}"#
        )
    };
    for (model, why) in [
        (
            r#"{"type": "sgd", "loss": "logistic"}"#,
            "fits a probability to a 0/1 target (loss = \"logistic\")",
        ),
        (
            r#"{"type": "sgd", "loss": "poisson"}"#,
            "fits a log rate to a count (loss = \"poisson\")",
        ),
    ] {
        refused(&with(model), &["a formula target", why]);
    }
    let e = check(&format!(
        r#"{{"name": "m", "model": {{"type": "seqtest"}}, "targets": {target},
            "features": []}}"#
    ))
    .expect_err("a seqtest over a formula target accepted");
    assert!(e.contains("tests the signs of its targets"), "{e}");
}

/// A session column needs a `session_gap` (or a session close), and a
/// word that is not `"reset"` is refused naming the forms it can take.
#[test]
fn a_session_without_its_rule_or_with_a_word_it_does_not_know_is_refused() {
    let clocked = r#", "clock": "t", "gap_cap": 5.0, "session": "s""#;
    let e = check(&sup(r#"{"type": "ewridge"}"#, clocked)).expect_err("no session_gap");
    assert!(
        e.contains("session_gap is required when session is given")
            && e.contains("group_close = \"session\""),
        "{e}"
    );
    // The reader takes no word but "reset"; a spec built in Rust is held to
    // the same.
    let mut s: Spec = serde_json::from_str(&sup(
        r#"{"type": "ewridge"}"#,
        &format!(r#"{clocked}, "session_gap": "reset""#),
    ))
    .unwrap();
    s.session_gap = Some(online_polars::SessionGapSpec::Word("restart".into()));
    let e = s.check().expect_err("a session_gap word accepted");
    assert!(
        e.contains("session_gap must be a number, a duration or \"reset\", got \"restart\""),
        "{e}"
    );
    // A negative integer is read as a number, and refused as one.
    let e = check(&sup(
        r#"{"type": "ewridge"}"#,
        &format!(r#"{clocked}, "session_gap": -2"#),
    ))
    .expect_err("a negative session_gap accepted");
    assert!(e.contains("session_gap must be >= 0 or \"reset\""), "{e}");
}

/// The window operators' run says the same about a cap that is not finite,
/// and about a session and its gap, in a window's words rather than a
/// model's: a gap without a session is refused, and a session needs none.
#[test]
fn a_window_runs_clock_rules_are_said_in_a_windows_words() {
    use online_polars::{WindowsConfig, WindowsRun};
    let schema = polars::prelude::Schema::from_iter([
        ("t".into(), polars::prelude::DataType::Float64),
        ("s".into(), polars::prelude::DataType::String),
        ("x".into(), polars::prelude::DataType::Float64),
    ]);
    let refusal = |extra: &str| {
        let config: WindowsConfig = serde_json::from_str(&format!(
            r#"{{"formulas": [{{"name": "f", "tree": ["ewm_mean", ["col", "x"], {{"half_life": 5.0}}]}}],
                "clock": "t"{extra}}}"#
        ))
        .unwrap();
        WindowsRun::new(config, &schema)
            .err()
            .expect("the run started")
    };
    let e = refusal(r#", "gap_cap": "inf""#);
    assert!(
        e.contains("gap_cap must be finite; it is the longest step a window spans"),
        "{e}"
    );
    // A session needs no gap here: every window ends at a session change,
    // so no window reads the number (task 278).
    let config: WindowsConfig = serde_json::from_str(
        r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"], {"half_life": 5.0}]}],
            "clock": "t", "gap_cap": 5.0, "session": "s"}"#,
    )
    .unwrap();
    WindowsRun::new(config, &schema).expect("a session without a gap refused");
    let e = refusal(r#", "gap_cap": 5.0, "session_gap": 1.0"#);
    assert!(
        e.starts_with("with_windows: session_gap needs session;"),
        "{e}"
    );
}

/// A deprecated model key forwarded beside the key it became is refused;
/// one forwarded alone keeps a value that is not its word for the new key
/// to refuse in its own words, and says so.
#[test]
fn a_deprecated_model_key_beside_its_new_name_is_refused() {
    use online_polars::forward_deprecated_in_models;
    for json in [
        r#"{"type": "bocpd", "emission": "robust", "covariance_type": "diag"}"#,
        r#"{"type": "corrchange", "reset_on_flag": true, "change_action": "flag"}"#,
    ] {
        let mut v: serde_json::Value = serde_json::from_str(json).unwrap();
        let e = forward_deprecated_in_models(&mut v).unwrap_err();
        assert!(
            e.contains("was renamed") && e.contains("both are given"),
            "{e}"
        );
    }
    // A word that never was one goes on as itself, with no robust_beta.
    let mut v: serde_json::Value =
        serde_json::from_str(r#"{"type": "bocpd", "emission": "laplace"}"#).unwrap();
    let notices = forward_deprecated_in_models(&mut v).unwrap();
    assert_eq!(v["covariance_type"], "laplace");
    assert!(v.get("robust_beta").is_none(), "{v}");
    assert_eq!(
        notices,
        [
            "emission = \"laplace\" is deprecated: it is covariance_type = \"laplace\". It is \
          read so until the next major version, which refuses it"
        ]
    );
    // A value that is not a word at all goes on unchanged.
    let mut v: serde_json::Value =
        serde_json::from_str(r#"{"type": "bocpd", "emission": 3}"#).unwrap();
    forward_deprecated_in_models(&mut v).unwrap();
    assert_eq!(v["covariance_type"], 3);
    // `reset_on_flag = false` is the flag, and a value that is not a
    // boolean goes on unchanged.
    let mut v: serde_json::Value =
        serde_json::from_str(r#"{"type": "corrchange", "reset_on_flag": false}"#).unwrap();
    forward_deprecated_in_models(&mut v).unwrap();
    assert_eq!(v["change_action"], "flag");
    let mut v: serde_json::Value =
        serde_json::from_str(r#"{"type": "corrchange", "reset_on_flag": "yes"}"#).unwrap();
    let notices = forward_deprecated_in_models(&mut v).unwrap();
    assert_eq!(v["change_action"], "yes");
    assert!(
        notices[0].contains("change_action = \"flag\""),
        "{notices:?}"
    );
}

/// A word refusal from serde with an empty list of words is not one this
/// library can reword: passed on as it is (`None`).
#[test]
fn a_variant_message_with_no_words_is_not_reworded() {
    assert_eq!(
        online_polars::word_refusal_body("unknown variant `x`, expected `` or ``"),
        None
    );
    assert!(online_polars::word_refusal_body("unknown variant `x`, expected `a` or `b`").is_some());
}

/// A word a model does not know is refused naming the word, whichever of
/// the model's word keys it is given to.
#[test]
fn a_model_word_it_does_not_know_is_refused_naming_it() {
    for (model, word) in [
        (r#"{"type": "ew_cov", "stats": ["kurtosis"]}"#, "kurtosis"),
        (r#"{"type": "sgd", "loss": "hinge"}"#, "hinge"),
        (r#"{"type": "sgd", "schedule": "cosine"}"#, "cosine"),
        (r#"{"type": "pa", "mode": "pa3"}"#, "pa3"),
        (
            r#"{"type": "marginal", "feature_moments": "pooled"}"#,
            "pooled",
        ),
    ] {
        let json = if model.contains("ew_cov") {
            unsup(model, r#", "half_life": 10.0"#)
        } else {
            sup(model, "")
        };
        let e = check(&json).expect_err(model);
        assert!(e.contains(word), "{model}: {e}");
    }
}

/// What needs a decay reads `lam` as it reads a half-life: `lam < 1`
/// forgets, and `lam = 1` does not, for `min_settled_frac` and for feature
/// health alike; `lam` beside `half_life` is refused.
#[test]
fn lam_is_a_decay_where_a_decay_is_needed() {
    let at = |lam: &str, extra: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "lam": {lam}{extra}}}"#
        )
    };
    for extra in [
        r#", "min_settled_frac": 0.5"#,
        r#", "emit_feature_health": true"#,
    ] {
        accepted(&at("0.99", extra));
        let e = check(&at("1.0", extra)).expect_err(extra);
        assert!(e.starts_with("spec \"m\": "), "{e}");
    }
    let mut both: Spec = serde_json::from_str(&at("0.99", "")).unwrap();
    both.half_life = Some(online_polars::SpanList::One(online_polars::Span::Units(
        10.0,
    )));
    assert_eq!(
        both.decays().unwrap_err(),
        "spec \"m\": half_life and lam are mutually exclusive"
    );
}

/// A relative target with no column or name is refused with the removal's
/// own words; a formula target's name must not be empty.
#[test]
fn a_target_table_with_nothing_to_name_is_refused() {
    let parse = |targets: &str| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": {targets},
                "features": ["x0"], "half_life": 10.0}}"#
        ))
        .expect_err("parsed")
        .to_string()
    };
    let e = parse(r#"[{"relative_to": "x0"}]"#);
    assert!(!e.contains("target \""), "no column to name: {e}");
    assert!(e.contains("relative"), "{e}");
    let e = parse(r#"[{"name": "", "formula": ["col", "y"]}]"#);
    assert!(e.contains("formula target: name must not be empty"), "{e}");
}
