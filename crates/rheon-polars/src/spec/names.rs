//! The words a spec's parameters take and how a word outside them is
//! refused (task 244, P-4): moved out of `spec.rs` under the 250 KB a
//! source file is held to.

/// The words `kmeans`' and `hmm`'s `init` takes: scikit-learn's `KMeans`
/// spelling of k-means++ (task 244), where `seed_rule` said `"kmeanspp"`.
pub const INIT_WORDS: [&str; 4] = ["first", "farthest", "k-means++", "lloyd"];

/// The one refusal of a word outside the words a parameter takes (task 244,
/// P-4), in the shape every message takes (WRITING §11): `<param> must be
/// "a", "b" or "c", but it is "x". Use one of those words.`, after the
/// spec's name where the caller has one ([`spec_words_refusal`]). Every word
/// parameter says it the same way -- a builder in Python, a spec dict, a
/// TOML file -- so the words read the same wherever a caller meets them,
/// and the API snapshot reads them from this one shape.
pub fn words_refusal(param: &str, words: &[&str], got: &str) -> String {
    format!("{param} {}", words_body(words, got))
}

/// `"a"`, `"a" or "b"`, `"a", "b" or "c"`: the words a parameter takes, as
/// a refusal lists them.
pub fn words_listed(words: &[&str]) -> String {
    let quoted: Vec<String> = words.iter().map(|w| format!("{w:?}")).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => quoted.join(""),
    }
}

/// [`words_refusal`] without the parameter, which its remedy does not
/// need: `must be "a" or "b", but it is "x". Use one of those words.` A
/// word renamed before 1.0 ([`super::RENAMED_VALUES`]) is told its new
/// name.
pub fn words_body(words: &[&str], got: &str) -> String {
    let listed = words_listed(words);
    if let Some((_, new)) = super::RENAMED_VALUES
        .iter()
        .find(|(old, new)| *old == got && words.contains(new))
    {
        return format!(
            "must be {listed}, but it is {got:?}, the old name of {new:?}. Use {new:?}."
        );
    }
    let remedy = match words {
        [one] => format!("Use {one:?}."),
        _ => "Use one of those words.".to_string(),
    };
    format!("must be {listed}, but it is {got:?}. {remedy}")
}

/// [`words_refusal`] after `spec "<name>": `.
pub fn spec_words_refusal(spec: &str, param: &str, words: &[&str], got: &str) -> String {
    format!("spec {spec:?}: {}", words_refusal(param, words, got))
}

/// A deserializer's refusal of a word, `msg`, as [`words_body`] words it:
/// `must be "a" or "b", but it is "x". Use one of those words.` serde words an enum's unknown
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

    /// `init`'s words reach the core's seeding rules one for one, and a
    /// windowed `closed` of Polars' two the model cannot honour is refused
    /// with its reason.
    #[test]
    fn init_words_reach_the_seeding_rules_and_closed_left_says_why() {
        use rheon_core::SeedRule;
        for (word, rule) in [
            (None, SeedRule::Lloyd),
            (Some("lloyd"), SeedRule::Lloyd),
            (Some("k-means++"), SeedRule::Kmeanspp),
            (Some("farthest"), SeedRule::Farthest),
            (Some("first"), SeedRule::First),
        ] {
            assert_eq!(crate::stream::init_rule(word).unwrap(), rule, "{word:?}");
        }
        let e = crate::stream::init_rule(Some("kmeanspp")).unwrap_err();
        assert!(e.starts_with("init must be \"first\", "), "{e}");
        let e = serde_json::from_str::<crate::spec::ModelClosed>(r#""left""#)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(
                "must be \"right\" or \"both\", but it is \"left\", which leaves out the \
                 current row"
            ),
            "{e}"
        );
    }

    /// `kmeans`' and `micro`'s `scale_floor` floors the variance `standardize`
    /// measures in, and is refused beside `standardize = false` (task 244,
    /// P-5); each alone is taken. And `ew_cov`'s row cap is listed under its
    /// name with the clock cadence it pairs with.
    #[test]
    fn scale_floor_needs_standardize_and_the_pca_cap_is_listed() {
        for model in [
            r#"{"type": "kmeans", "n_clusters": 2"#,
            r#"{"type": "micro", "eps": 0.3"#,
        ] {
            let spec = |extra: &str| -> crate::spec::Spec {
                serde_json::from_str(&format!(
                    r#"{{"name": "m", "model": {model}{extra}}}, "features": ["x0", "x1"],
                        "half_life": 10.0}}"#
                ))
                .unwrap()
            };
            let mut s = spec(r#", "standardize": false, "scale_floor": 0.1"#);
            s.fill_defaults();
            let e = s.validate().unwrap_err();
            assert!(
                e.contains("scale_floor floors the variance"),
                "{model}: {e}"
            );
            for ok in [r#", "standardize": false"#, r#", "scale_floor": 0.1"#] {
                let mut s = spec(ok);
                s.fill_defaults();
                s.validate().unwrap_or_else(|e| panic!("{model}{ok}: {e}"));
            }
        }
        let cov: crate::spec::ModelKind =
            serde_json::from_str(r#"{"type": "ew_cov", "pca": 1, "max_rows_between_pcas": 5}"#)
                .unwrap();
        assert!(
            cov.row_caps()
                .contains(&("max_rows_between_pcas", Some(5), "pca_every")),
            "{:?}",
            cov.row_caps()
        );
    }

    /// `bocpd`'s tempered predictive is `robust_beta` given, above 0 and
    /// finite, under `covariance_type = "diag"` alone (task 244, P-2).
    #[test]
    fn robust_beta_turns_on_the_tempered_predictive_under_diag_alone() {
        let spec = |extra: &str| -> crate::spec::Spec {
            let mut s: crate::spec::Spec = serde_json::from_str(&format!(
                r#"{{"name": "m", "model": {{"type": "bocpd"{extra}}}, "features": ["x0", "x1"]}}"#
            ))
            .unwrap();
            s.fill_defaults();
            s
        };
        let cfg = crate::stream::bocpd_cfg(&spec(r#", "robust_beta": 0.1"#)).unwrap();
        assert_eq!(cfg.emission, rheon_core::BocpdEmission::Robust);
        assert_eq!(cfg.robust_beta, 0.1);
        let full = crate::stream::bocpd_cfg(&spec(r#", "covariance_type": "full""#)).unwrap();
        assert_eq!(full.emission, rheon_core::BocpdEmission::Gaussian);
        let plain = crate::stream::bocpd_cfg(&spec("")).unwrap();
        assert_eq!(plain.emission, rheon_core::BocpdEmission::Diag);
        for bad in ["0.0", "-0.1"] {
            let e = crate::stream::bocpd_cfg(&spec(&format!(r#", "robust_beta": {bad}"#)))
                .map(|_| ())
                .unwrap_err();
            assert!(
                e.contains("robust_beta must be finite and above 0"),
                "{bad}: {e}"
            );
        }
        let e =
            crate::stream::bocpd_cfg(&spec(r#", "covariance_type": "full", "robust_beta": 0.1"#))
                .map(|_| ())
                .unwrap_err();
        assert!(
            e.contains("robust_beta applies only under covariance_type=\"diag\""),
            "{e}"
        );
    }

    /// Every word refusal reads one way: the parameter, the words, the word
    /// given; and serde's enum refusal reads the same once rewritten.
    #[test]
    fn a_word_refusal_has_one_shape() {
        assert_eq!(
            crate::spec::words_refusal("closed", &["right", "both"], "left"),
            "closed must be \"right\" or \"both\", but it is \"left\". Use one of those words."
        );
        assert_eq!(
            crate::spec::word_refusal_body("unknown variant `x`, expected `mean` or `sum`"),
            Some(
                "must be \"mean\" or \"sum\", but it is \"x\". Use one of those words.".to_string()
            )
        );
        assert_eq!(
            crate::spec::word_refusal_body("unknown variant `x`, expected one of `a`, `b`, `c`"),
            Some(
                "must be \"a\", \"b\" or \"c\", but it is \"x\". Use one of those words."
                    .to_string()
            )
        );
        assert_eq!(
            crate::spec::words_refusal("kernel", &["parzen"], "bartlett"),
            "kernel must be \"parzen\", but it is \"bartlett\". Use \"parzen\"."
        );
        assert_eq!(
            crate::spec::words_refusal("stats", &["mean", "lag_corr"], "lagcorr"),
            "stats must be \"mean\" or \"lag_corr\", but it is \"lagcorr\", the old name of \
             \"lag_corr\". Use \"lag_corr\"."
        );
        assert_eq!(crate::spec::word_refusal_body("invalid type: map"), None);
    }
}
