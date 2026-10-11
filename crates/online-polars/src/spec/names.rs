//! The words a spec's parameters take and how a word outside them is
//! refused (task 244, P-4), and the names a spec took for 1.0 forwarded
//! from the ones they replaced, model by model (task 244): moved out of
//! `spec.rs` under the 250 KB a source file is held to.

use super::DEPRECATED;

/// The words `kmeans`' and `hmm`'s `init` takes: scikit-learn's `KMeans`
/// spelling of k-means++ (task 244), where `seed_rule` said `"kmeanspp"`.
pub const INIT_WORDS: [&str; 4] = ["first", "farthest", "k-means++", "lloyd"];

/// The one refusal of a word outside the words a parameter takes (task 244,
/// P-4): `<param> must be one of "a", "b"; got "x"`, after the spec's name
/// where the caller has one ([`spec_words_refusal`]). Every word parameter
/// says it the same way -- a builder in Python, a spec dict, a TOML file --
/// so the words read the same wherever a caller meets them, and the API
/// snapshot reads them from this one shape.
pub fn words_refusal(param: &str, words: &[&str], got: &str) -> String {
    format!("{param} {}", words_body(words, got))
}

/// [`words_refusal`] without the parameter: `must be one of "a", "b"; got
/// "x"`.
pub fn words_body(words: &[&str], got: &str) -> String {
    let list = words
        .iter()
        .map(|w| format!("{w:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("must be one of {list}; got {got:?}")
}

/// [`words_refusal`] after `spec "<name>": `.
pub fn spec_words_refusal(spec: &str, param: &str, words: &[&str], got: &str) -> String {
    format!("spec {spec:?}: {}", words_refusal(param, words, got))
}

/// A deserializer's refusal of a word, `msg`, as [`words_body`] words it:
/// `must be one of "a", "b"; got "x"`. serde words an enum's unknown
/// variant as ``unknown variant `x`, expected `a` or `b` `` (or ``expected
/// one of `a`, `b`, `c` ``), and a spec's own deserializers say ``must be
/// ...`` already. `None` for any other message, which is not about a word.
pub fn word_refusal_body(msg: &str) -> Option<String> {
    if msg.starts_with("must be ") {
        return Some(msg.to_string());
    }
    let rest = msg.strip_prefix("unknown variant `")?;
    let (got, rest) = rest.split_once('`')?;
    let list = rest.strip_prefix(", expected ")?;
    let list = list.strip_prefix("one of ").unwrap_or(list);
    let words: Vec<&str> = list
        .split(['`', ','])
        .map(str::trim)
        .filter(|w| !w.is_empty() && *w != "or")
        .collect();
    if words.is_empty() {
        return None;
    }
    Some(words_body(&words, got))
}

/// The keys renamed in one model, `(model type, old, new)`: read in a model
/// table whose `type` is the model's as the new name, with a notice, until
/// the next major version refuses them (task 244). A model's own table, not
/// [`DEPRECATED`], because one old name went two ways (`kmeans`' `k` is
/// `n_clusters`, `hmm`'s `n_components`) and because a name another model
/// still takes, or a feature called `k` in a table keyed by feature, must
/// not move. Its twin is `python/polars_online/_warnings.py`'s
/// `_DEPRECATED_IN`, which `tests/test_deprecation.py` holds equal to it.
pub const DEPRECATED_IN: &[(&str, &str, &str)] = &[
    ("kmeans", "k", "n_clusters"),
    ("kmeans", "seed_rule", "init"),
    ("hmm", "k", "n_components"),
    ("hmm", "seed_rule", "init"),
    ("hmm", "covariance", "covariance_type"),
    ("ew_class", "covariance", "covariance_type"),
    ("ewridge", "window_every", "snapshot_every"),
    ("lasso", "window_every", "snapshot_every"),
    ("ew_cov", "window_every", "snapshot_every"),
    ("ew_class", "window_every", "snapshot_every"),
    ("marginal", "window_every", "snapshot_every"),
    ("ew_cov", "max_rows_between_pca", "max_rows_between_pcas"),
    ("lasso", "lasso_path", "penalties"),
    ("bocpd", "prior_mean", "mean_prior"),
    ("bocpd", "prior_kappa", "mean_precision_prior"),
    ("bocpd", "prior_nu", "degrees_of_freedom_prior"),
    ("bocpd", "prior_scale", "covariance_prior"),
    ("corrchange", "span_rows", "window_rows"),
];

/// The words renamed in one model's parameter, `(model type, key, old,
/// new)`, read as [`DEPRECATED_IN`] reads a key: scikit-learn's spellings
/// (task 244). Its twin is `_warnings.py`'s `_DEPRECATED_WORDS_IN`.
pub const DEPRECATED_WORDS_IN: &[(&str, &str, &str, &str)] = &[
    ("kmeans", "init", "kmeanspp", "k-means++"),
    ("hmm", "init", "kmeanspp", "k-means++"),
    ("hmm", "covariance_type", "shared", "tied"),
    ("hmm", "covariance_type", "diagonal", "diag"),
    ("ew_class", "covariance_type", "shared", "tied"),
    ("ew_class", "covariance_type", "diagonal", "diag"),
];

/// What a deprecated word is told, as [`deprecation_notice`] tells a key.
pub fn word_deprecation_notice(key: &str, old: &str, new: &str) -> String {
    format!(
        "{key} = {old:?} is deprecated: it was renamed {new:?}. It is read as {new:?} until the \
         next major version, which refuses it"
    )
}

/// `v` with every model table's renamed keys and words read as the new
/// ones ([`DEPRECATED_IN`], [`DEPRECATED_WORDS_IN`]), at any depth, and two
/// keys that became others with a value of their own: `bocpd`'s `emission`
/// (`"gaussian"` is `covariance_type = "full"`, `"diag"` is `"diag"`, and
/// `"robust"` is `"diag"` with `robust_beta`, 0.1 where none is given, its
/// default then) and `corrchange`'s `reset_on_flag` (`true` is
/// `change_action = "reset"`, `false` is `"flag"`). A notice for each; an old
/// name beside its new one is refused, naming both.
pub fn forward_deprecated_in_models(v: &mut serde_json::Value) -> Result<Vec<String>, String> {
    let mut notices = Vec::new();
    forward_models(v, &mut notices)?;
    Ok(notices)
}

fn forward_models(v: &mut serde_json::Value, notices: &mut Vec<String>) -> Result<(), String> {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(kind) = map.get("type").and_then(|t| t.as_str()).map(str::to_string) {
                forward_model(&kind, map, notices)?;
            }
            for value in map.values_mut() {
                forward_models(value, notices)?;
            }
        }
        serde_json::Value::Array(items) => {
            for value in items {
                forward_models(value, notices)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn forward_model(
    kind: &str,
    map: &mut serde_json::Map<String, serde_json::Value>,
    notices: &mut Vec<String>,
) -> Result<(), String> {
    let both = |old: &str, new: &str| {
        format!("{old} was renamed {new}, and both are given: give {new} alone")
    };
    for (model, old, new) in DEPRECATED_IN {
        if *model == kind
            && let Some(value) = map.remove(*old)
        {
            if map.contains_key(*new) {
                return Err(both(old, new));
            }
            map.insert((*new).to_string(), value);
            notices.push(deprecation_notice(old, new));
        }
    }
    if kind == "bocpd"
        && let Some(value) = map.remove("emission")
    {
        if map.contains_key("covariance_type") {
            return Err(both("emission", "covariance_type"));
        }
        // A value that is no word, or a word that never was one, is passed
        // on for `covariance_type` to refuse in its own words.
        let word = value.as_str().unwrap_or_default().to_string();
        let (covariance, beta) = match word.as_str() {
            "gaussian" => ("full", false),
            "robust" => ("diag", true),
            other => (other, false),
        };
        if value.is_string() {
            map.insert("covariance_type".into(), covariance.into());
        } else {
            map.insert("covariance_type".into(), value);
        }
        let robust = if beta {
            map.entry("robust_beta").or_insert(0.1.into());
            " with robust_beta, 0.1 where none is given"
        } else {
            ""
        };
        notices.push(format!(
            "emission = {word:?} is deprecated: it is covariance_type = {covariance:?}{robust}. \
             It is read so until the next major version, which refuses it"
        ));
    }
    if kind == "corrchange"
        && let Some(value) = map.remove("reset_on_flag")
    {
        if map.contains_key("change_action") {
            return Err(both("reset_on_flag", "change_action"));
        }
        let action = if value.as_bool() == Some(true) {
            "reset"
        } else {
            "flag"
        };
        if value.is_boolean() {
            map.insert("change_action".into(), action.into());
        } else {
            map.insert("change_action".into(), value.clone());
        }
        notices.push(format!(
            "reset_on_flag = {value} is deprecated: it is change_action = {action:?}. It is read \
             so until the next major version, which refuses it"
        ));
    }
    for (model, key, old, new) in DEPRECATED_WORDS_IN {
        if *model == kind && map.get(*key).and_then(|w| w.as_str()) == Some(*old) {
            map.insert((*key).to_string(), (*new).into());
            notices.push(word_deprecation_notice(key, old, new));
        }
    }
    Ok(())
}

/// What a deprecated name is told: that it was renamed, that it still
/// works, and until when.
pub fn deprecation_notice(old: &str, new: &str) -> String {
    format!(
        "{old} is deprecated: it was renamed {new}. It is read as {new} until the next major \
         version, which refuses it"
    )
}

/// `v` -- a spec, a list of specs or a windows config, as JSON -- with each
/// key [`DEPRECATED`] names renamed to its new name, at any depth (a model's
/// own parameters are a level down), and a notice for each. A map naming
/// both an old name and its new one is refused, naming both.
pub fn forward_deprecated(v: &mut serde_json::Value) -> Result<Vec<String>, String> {
    let mut notices = forward_deprecated_with(v, DEPRECATED)?;
    notices.extend(forward_deprecated_in_models(v)?);
    Ok(notices)
}

/// [`forward_deprecated`] by `table`.
pub fn forward_deprecated_with(
    v: &mut serde_json::Value,
    table: &[(&str, &str)],
) -> Result<Vec<String>, String> {
    let mut notices = Vec::new();
    forward_into(v, table, &mut notices)?;
    Ok(notices)
}

fn forward_into(
    v: &mut serde_json::Value,
    table: &[(&str, &str)],
    notices: &mut Vec<String>,
) -> Result<(), String> {
    match v {
        serde_json::Value::Object(map) => {
            for (old, new) in table {
                if let Some(value) = map.remove(*old) {
                    if map.contains_key(*new) {
                        return Err(format!(
                            "{old} was renamed {new}, and both are given: give {new} alone"
                        ));
                    }
                    map.insert((*new).to_string(), value);
                    notices.push(deprecation_notice(old, new));
                }
            }
            for value in map.values_mut() {
                forward_into(value, table, notices)?;
            }
        }
        serde_json::Value::Array(items) => {
            for value in items {
                forward_into(value, table, notices)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The builder keywords a spec table spells another way, `(keyword,
/// builder, the table's key)`: a builder names its columns for what they
/// are to the model, and every table names them `targets` and `features`
/// (task 244, C-7). A table that says the builder's word is refused naming
/// the table's ([`super::name_renamed`]).
pub const BUILDER_KEYWORDS: &[(&str, &str, &str)] = &[
    ("label", "ew_class", "targets = [\"<the label column>\"]"),
    ("columns", "audit", "features = [...]"),
];

#[cfg(test)]
mod tests {
    /// The forwarding table is empty before 1.0: a rename made before it is
    /// refused by name (task 144's rule), never forwarded, and no name is
    /// both refused and forwarded (docs/PLAN.md task 198, D2).
    #[test]
    fn the_forwarding_table_is_empty_before_1_0() {
        let major: u32 = env!("CARGO_PKG_VERSION")
            .split('.')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if major < 1 {
            assert!(
                crate::spec::DEPRECATED.is_empty(),
                "{:?}",
                crate::spec::DEPRECATED
            );
        }
        for (old, _) in crate::spec::DEPRECATED {
            assert!(
                crate::spec::RENAMED.iter().all(|(r, _)| r != old),
                "{old} is both refused and forwarded"
            );
        }
    }

    /// An entry forwards an old name to the new one at any depth -- a spec's
    /// own key and its model's -- with a notice each, and the spec it makes
    /// is the one the new name makes; an old and a new name side by side are
    /// refused, naming both.
    #[test]
    fn a_deprecated_name_is_forwarded_with_a_notice() {
        let table = [("half_lyfe", "half_life"), ("rigde", "ridge")];
        let mut old: serde_json::Value = serde_json::from_str(
            r#"[{"name": "m", "model": {"type": "ewridge", "rigde": 0.5}, "targets": ["y"],
                 "features": ["x"], "half_lyfe": 10.0}]"#,
        )
        .unwrap();
        let notices = crate::spec::forward_deprecated_with(&mut old, &table).unwrap();
        assert_eq!(
            notices,
            [
                crate::spec::deprecation_notice("half_lyfe", "half_life"),
                crate::spec::deprecation_notice("rigde", "ridge"),
            ]
        );
        assert!(notices[0].contains("renamed half_life") && notices[0].contains("next major"));
        let new: serde_json::Value = serde_json::from_str(
            r#"[{"name": "m", "model": {"type": "ewridge", "ridge": 0.5}, "targets": ["y"],
                 "features": ["x"], "half_life": 10.0}]"#,
        )
        .unwrap();
        let (a, b): (Vec<crate::spec::Spec>, Vec<crate::spec::Spec>) = (
            serde_json::from_value(old).unwrap(),
            serde_json::from_value(new).unwrap(),
        );
        assert_eq!(a, b);
        let mut both: serde_json::Value =
            serde_json::from_str(r#"{"half_lyfe": 1.0, "half_life": 2.0}"#).unwrap();
        let err = crate::spec::forward_deprecated_with(&mut both, &table).unwrap_err();
        assert!(
            err.contains("half_lyfe") && err.contains("half_life"),
            "{err}"
        );
        // The production table, empty, leaves a spec as it is.
        let mut plain: serde_json::Value = serde_json::from_str(r#"{"half_life": 3.0}"#).unwrap();
        let before = plain.clone();
        assert!(
            crate::spec::forward_deprecated(&mut plain)
                .unwrap()
                .is_empty()
        );
        assert_eq!(plain, before);
    }

    /// A spec table that says a builder keyword is told the table's key
    /// (task 244, C-7): `ew_class`'s `label` is `targets`, `audit`'s
    /// `columns` is `features`, from JSON and from TOML.
    #[test]
    fn a_builder_keyword_in_a_table_is_told_the_tables_key() {
        let json = serde_json::from_str::<crate::spec::Spec>(
            r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"],
                "precision_prior": 1.0}, "label": "y", "features": ["x"], "half_life": 10}"#,
        )
        .unwrap_err()
        .to_string();
        let json = crate::spec::name_renamed(&json);
        assert!(
            json.contains(
                "`label` is ew_class's builder keyword; a spec table says \
                 `targets = [\"<the label column>\"]`"
            ),
            "{json}"
        );
        let toml = toml::from_str::<crate::spec::Spec>(
            "name = \"a\"\ncolumns = [\"x\", \"y\"]\n[model]\ntype = \"audit\"\n",
        )
        .unwrap_err()
        .to_string();
        let toml = crate::spec::name_renamed(&toml);
        assert!(
            toml.contains(
                "`columns` is audit's builder keyword; a spec table says `features = [...]`"
            ),
            "{toml}"
        );
    }

    /// The renames made for 1.0, one model's at a time (task 244): a key and
    /// a word read as the new ones with a notice each, only in a table of
    /// that model's type, and an old name beside its new one refused.
    #[test]
    fn a_models_renamed_key_and_word_are_forwarded_in_its_own_table_alone() {
        let mut old: serde_json::Value = serde_json::from_str(
            r#"[{"name": "m", "model": {"type": "kmeans", "k": 2, "seed_rule": "kmeanspp"},
                 "features": ["x"], "half_life": 10.0},
                {"name": "e", "model": {"type": "ewridge", "k": 2}, "targets": ["y"],
                 "features": ["k"], "half_life": 10.0}]"#,
        )
        .unwrap();
        let notices = crate::spec::forward_deprecated(&mut old).unwrap();
        assert_eq!(
            notices,
            [
                crate::spec::deprecation_notice("k", "n_clusters"),
                crate::spec::deprecation_notice("seed_rule", "init"),
                crate::spec::word_deprecation_notice("init", "kmeanspp", "k-means++"),
            ]
        );
        assert_eq!(old[0]["model"]["n_clusters"], 2);
        assert_eq!(old[0]["model"]["init"], "k-means++");
        // Another model's `k`, and a feature called `k`, are left alone.
        assert_eq!(old[1]["model"]["k"], 2);
        assert_eq!(old[1]["features"][0], "k");
        let mut bocpd: serde_json::Value =
            serde_json::from_str(r#"{"type": "bocpd", "emission": "robust"}"#).unwrap();
        crate::spec::forward_deprecated_in_models(&mut bocpd).unwrap();
        assert_eq!(bocpd["covariance_type"], "diag");
        assert_eq!(bocpd["robust_beta"], 0.1);
        let mut corr: serde_json::Value =
            serde_json::from_str(r#"{"type": "corrchange", "reset_on_flag": true}"#).unwrap();
        crate::spec::forward_deprecated_in_models(&mut corr).unwrap();
        assert_eq!(corr["change_action"], "reset");
        let mut both: serde_json::Value =
            serde_json::from_str(r#"{"type": "hmm", "k": 2, "n_components": 2}"#).unwrap();
        let err = crate::spec::forward_deprecated_in_models(&mut both).unwrap_err();
        assert!(
            err.contains("k was renamed n_components, and both"),
            "{err}"
        );
    }

    /// Every word refusal reads one way: the parameter, the words, the word
    /// given; and serde's enum refusal reads the same once rewritten.
    #[test]
    fn a_word_refusal_has_one_shape() {
        assert_eq!(
            crate::spec::words_refusal("closed", &["right", "both"], "left"),
            "closed must be one of \"right\", \"both\"; got \"left\""
        );
        assert_eq!(
            crate::spec::word_refusal_body("unknown variant `x`, expected `mean` or `sum`"),
            Some("must be one of \"mean\", \"sum\"; got \"x\"".to_string())
        );
        assert_eq!(
            crate::spec::word_refusal_body("unknown variant `x`, expected one of `a`, `b`, `c`"),
            Some("must be one of \"a\", \"b\", \"c\"; got \"x\"".to_string())
        );
        assert_eq!(crate::spec::word_refusal_body("invalid type: map"), None);
    }
}
