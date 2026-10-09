//! Model bank specs (docs/PLAN.md §3, §5): serde-deserializable from JSON
//! (Python) and TOML (CLI), with common-parameter validation and the output
//! struct layout.

use online_core::{ClockCfg, Decay, ExactCaps, OnClockReset, SessionGap, TargetGaps};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use crate::span::{Span, SpanList};
use crate::targets::Targets;

fn default_true() -> bool {
    true
}

/// A float that also accepts the JSON strings `"inf"` / `"-inf"`, since JSON
/// has no infinity literal and `half_life = inf` is meaningful (it pins a
/// coefficient, docs/PLAN.md §4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num(pub f64);

/// A window's memory budget as a spec writes it: `{"thin": MiB}` or
/// `{"refuse": MiB}`, the MiB a [`Num`] so that `"inf"`, no bound, reads
/// from JSON (review 2026-09-12, P4; the user's decision of 2026-09-15).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowBudgetSpec {
    Thin(Num),
    Refuse(Num),
}

impl WindowBudgetSpec {
    pub fn mib(&self) -> f64 {
        match *self {
            WindowBudgetSpec::Thin(m) | WindowBudgetSpec::Refuse(m) => m.0,
        }
    }

    pub fn to_core(self) -> online_core::WindowBudget {
        match self {
            WindowBudgetSpec::Thin(m) => online_core::WindowBudget::Thin(m.0),
            WindowBudgetSpec::Refuse(m) => online_core::WindowBudget::Refuse(m.0),
        }
    }
}

mod readiness;

/// A number rendered for a **field name** (`__r{ridge}`, `@h{half_life}`,
/// `abs_resid_q{level}`, `__l{lambda}`).
///
/// Rust's `Display` for `f64` never uses scientific notation, so a perfectly
/// legal `ridge = 1e-300` produced a **311-character field name** (three
/// hundred zeros). Values outside [1e-6, 1e7) render as compact scientific
/// (`1e-300`, `2.5e8`) instead; everything inside renders exactly as before,
/// so no existing name changes.
///
/// These strings are **public API** — users index the output struct by them —
/// and this function is deliberately the only place the rendering lives.
/// `tests/test_api_surface.py` pins the results, so a change here (or a change
/// in rustc's float formatting, which has happened historically) fails a test
/// instead of silently renaming users' columns.
#[cfg(test)]
mod kinds_tests {
    use super::ModelKind;

    /// serde's unknown-variant error names every variant the enum has -- the
    /// one place that list exists outside the enum itself.
    #[test]
    fn kinds_lists_every_variant_in_order() {
        let err = serde_json::from_str::<ModelKind>(r#"{"type": "nope"}"#)
            .unwrap_err()
            .to_string();
        let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
        assert_eq!(quoted[0], "nope", "{err}");
        assert_eq!(
            &quoted[1..],
            ModelKind::KINDS,
            "ModelKind::KINDS is out of date"
        );
    }

    /// A key no spec has -- a typo in a TOML -- is refused, naming it and the
    /// keys there are, rather than left at its default without a word. Both
    /// levels: the spec's own keys and the model's.
    #[test]
    fn an_unknown_key_is_refused_at_either_level() {
        let spec = serde_json::from_str::<super::Spec>(
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x"], "halflfe": 10}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            spec.contains("unknown field `halflfe`") && spec.contains("`half_life`"),
            "{spec}"
        );
        let model = serde_json::from_str::<ModelKind>(r#"{"type": "ewridge", "rigde": 0.1}"#)
            .unwrap_err()
            .to_string();
        assert!(
            model.contains("unknown field `rigde`") && model.contains("`ridge`"),
            "{model}"
        );
    }

    /// Task 144: an old name is refused naming the new one, from JSON and
    /// from TOML, at the spec's level and the model's.
    #[test]
    fn an_old_name_is_refused_naming_the_new_one() {
        let json = serde_json::from_str::<super::Spec>(
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x"], "halflife": 10}"#,
        )
        .unwrap_err()
        .to_string();
        let json = super::name_renamed(&json);
        assert!(json.contains("halflife was renamed half_life"), "{json}");
        let toml = toml::from_str::<super::Spec>(
            "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x\"]\nhalf_life = 10.0\n\
             [model]\ntype = \"ewridge\"\nwindow = 10.0\n",
        )
        .unwrap_err()
        .to_string();
        let toml = super::name_renamed(&toml);
        assert!(toml.contains("window was renamed window_size"), "{toml}");
        for (old, new) in super::RENAMED {
            let msg = super::name_renamed(&format!("unknown field `{old}`, expected one of `x`"));
            assert!(msg.contains(&format!("{old} was renamed {new}")), "{msg}");
        }
        assert_eq!(
            super::name_renamed("unknown field `rigde`"),
            "unknown field `rigde`"
        );
        // `rls`'s `ridge` is `delta` (task 195, N11): named where the model
        // takes `delta`, and not where another model refuses a `ridge` it
        // never had.
        let rls = serde_json::from_str::<super::Spec>(
            r#"{"name": "m", "model": {"type": "rls", "ridge": 1.0}, "targets": ["y"],
                "features": ["x"], "half_life": 10}"#,
        )
        .unwrap_err()
        .to_string();
        let rls = super::name_renamed(&rls);
        assert!(rls.contains("ridge was renamed delta"), "{rls}");
        let sgd = serde_json::from_str::<super::Spec>(
            r#"{"name": "m", "model": {"type": "sgd", "ridge": 1.0}, "targets": ["y"],
                "features": ["x"], "half_life": 10}"#,
        )
        .unwrap_err()
        .to_string();
        let sgd = super::name_renamed(&sgd);
        assert!(
            sgd.contains("unknown field `ridge`") && !sgd.contains("renamed"),
            "{sgd}"
        );
    }

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
            assert!(super::DEPRECATED.is_empty(), "{:?}", super::DEPRECATED);
        }
        for (old, _) in super::DEPRECATED {
            assert!(
                super::RENAMED.iter().all(|(r, _)| r != old),
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
        let notices = super::forward_deprecated_with(&mut old, &table).unwrap();
        assert_eq!(
            notices,
            [
                super::deprecation_notice("half_lyfe", "half_life"),
                super::deprecation_notice("rigde", "ridge"),
            ]
        );
        assert!(notices[0].contains("renamed half_life") && notices[0].contains("next major"));
        let new: serde_json::Value = serde_json::from_str(
            r#"[{"name": "m", "model": {"type": "ewridge", "ridge": 0.5}, "targets": ["y"],
                 "features": ["x"], "half_life": 10.0}]"#,
        )
        .unwrap();
        let (a, b): (Vec<super::Spec>, Vec<super::Spec>) = (
            serde_json::from_value(old).unwrap(),
            serde_json::from_value(new).unwrap(),
        );
        assert_eq!(a, b);
        let mut both: serde_json::Value =
            serde_json::from_str(r#"{"half_lyfe": 1.0, "half_life": 2.0}"#).unwrap();
        let err = super::forward_deprecated_with(&mut both, &table).unwrap_err();
        assert!(
            err.contains("half_lyfe") && err.contains("half_life"),
            "{err}"
        );
        // The production table, empty, leaves a spec as it is.
        let mut plain: serde_json::Value = serde_json::from_str(r#"{"half_life": 3.0}"#).unwrap();
        let before = plain.clone();
        assert!(super::forward_deprecated(&mut plain).unwrap().is_empty());
        assert_eq!(plain, before);
    }
}

#[cfg(test)]
mod fill_tests {
    use super::Spec;

    fn spec(json: &str) -> Spec {
        serde_json::from_str(json).unwrap()
    }

    /// A column a model reads from the targets slot, named and not listed:
    /// `fill_defaults` fills `targets` with it, as the builders write them,
    /// not with `features[0]`, which was then read as the hazard or the
    /// exogenous series (review 2026-09-12, C20).
    #[test]
    fn a_named_hazard_or_exogenous_column_fills_the_targets() {
        for (model, col) in [
            (r#"{"type": "bocpd", "hazard_col": "h"}"#, "h"),
            (
                r#"{"type": "hmm", "k": 2, "precision_prior": 0.1, "exog_tvtp": "z"}"#,
                "z",
            ),
        ] {
            let mut s = spec(&format!(
                r#"{{"name": "m", "model": {model}, "features": ["x"]}}"#
            ));
            s.fill_defaults();
            assert_eq!(*s.targets, vec![col.to_string()], "{model}");
        }
    }

    /// And a spec that lists another column in that slot is refused, naming
    /// both: the model reads `targets[0]` whatever `hazard_col` says. The
    /// message prints the names (review 2026-09-26, D5: `Targets`' derived
    /// `Debug` printed every field).
    #[test]
    fn a_hazard_column_beside_another_target_is_refused() {
        let s = spec(
            r#"{"name": "m", "model": {"type": "bocpd", "hazard_col": "h"},
                "targets": ["y"], "features": ["x"]}"#,
        );
        let err = s.validate().unwrap_err();
        assert!(err.contains("hazard_col") && err.contains("\"h\""), "{err}");
        assert!(err.contains("(got [\"y\"])"), "{err}");
    }

    /// The slot is the column itself: a table named like the hazard but
    /// reading another column is refused, and so is the hazard column under
    /// another name, for `bocpd` and `hmm` alike (review 2026-09-26, D4: the
    /// names were compared, and the first put column `z` in the slot).
    #[test]
    fn a_renamed_table_cannot_smuggle_another_column_into_the_hazard_slot() {
        // `hmm` wants its `k`, a prior, the tvtp coefficients and a
        // half-life, which `bocpd` refuses.
        for (model, key, top) in [
            ("\"bocpd\"", "hazard_col", ""),
            (
                "\"hmm\", \"k\": 2, \"precision_prior\": 1.0, \
                 \"tvtp_coef\": [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]",
                "exog_tvtp",
                ", \"half_life\": 20.0",
            ),
        ] {
            for targets in [
                r#"[{"column": "z", "name": "h"}]"#,
                r#"[{"column": "h", "name": "hz"}]"#,
            ] {
                let s = spec(&format!(
                    r#"{{"name": "m", "model": {{"type": {model}, "{key}": "h"}},
                        "targets": {targets}, "features": ["x"]{top}}}"#
                ));
                let err = s.validate().unwrap_err();
                assert!(
                    err.contains(key) && err.contains("the column itself"),
                    "{err}"
                );
            }
            let s = spec(&format!(
                r#"{{"name": "m", "model": {{"type": {model}, "{key}": "h"}},
                    "targets": ["h"], "features": ["x"]{top}}}"#
            ));
            assert!(
                s.validate().is_ok(),
                "{model}: the column itself: {:?}",
                s.validate()
            );
        }
    }
}

#[cfg(test)]
mod num_label_tests {
    use super::num_label;

    #[test]
    fn ordinary_values_render_exactly_as_before() {
        // The compact form must not change any name the suite already pins.
        assert_eq!(num_label(1e-6), "0.000001");
        assert_eq!(num_label(1e-7 * 10.0), "0.000001");
        assert_eq!(num_label(0.1), "0.1");
        assert_eq!(num_label(0.5), "0.5");
        assert_eq!(num_label(100.0), "100");
        assert_eq!(num_label(250.5), "250.5");
        assert_eq!(num_label(3200.0), "3200");
        assert_eq!(num_label(0.0), "0");
        // One number, one name (review 2026-10-06, PC10).
        assert_eq!(num_label(-0.0), "0");
    }

    #[test]
    fn extreme_values_render_compactly() {
        // The bug this exists for: 1e-300 was a 311-character field name.
        assert_eq!(num_label(1e-300), "1e-300");
        assert_eq!(num_label(2.5e8), "2.5e8");
        assert_eq!(num_label(1e7), "1e7");
        assert_eq!(num_label(1e9), "1e9");
        assert_eq!(num_label(-1e-300), "-1e-300");
        assert!(num_label(1e-300).len() < 10);
    }

    #[test]
    fn the_thresholds_are_where_the_doc_says() {
        // Just inside: plain. Just outside: scientific.
        assert_eq!(num_label(1e-6), "0.000001");
        assert_eq!(num_label(9.999e-7), "9.999e-7");
        assert_eq!(num_label(9_999_999.0), "9999999");
        assert_eq!(num_label(1e7), "1e7");
    }
}

pub fn num_label(v: f64) -> String {
    // `-0.0` is `0.0` (`==`), so one name: a ridge grid of the two was two
    // instances under `__r0` and `__r-0` (review 2026-10-06, PC10).
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if v.is_finite() && !(1e-6..1e7).contains(&a) {
        // `{:e}` gives `1e-300` / `2.5e8`; normalize the `e0` exponent Rust
        // emits for values that just crossed the threshold.
        let s = format!("{v:e}");
        match s.strip_suffix("e0") {
            Some(t) => t.to_string(),
            None => s,
        }
    } else {
        format!("{v}")
    }
}

/// `v > 0` as a named predicate, so that `!positive(v)` refuses NaN too (a
/// plain `v <= 0.0` lets it through, and NaN in any of these parameters is
/// a state that never washes out).
fn positive(v: f64) -> bool {
    v > 0.0
}

fn non_negative(v: f64) -> bool {
    v >= 0.0
}

/// The first value that appears twice in a grid, if any. Grid entries become
/// field-name suffixes, so a repeated value is always a mistake. Compared
/// as numbers, so `-0.0` repeats `0.0`: bit by bit the two were two
/// instances of one model (review 2026-10-06, PC10).
fn first_duplicate(vals: &[f64]) -> Option<f64> {
    vals.iter()
        .enumerate()
        .find(|(i, v)| vals[..*i].iter().any(|u| u == *v))
        .map(|(_, v)| *v)
}

/// `ridge` for the IRLS models: finite and non-negative (zero is plain least
/// squares).
fn check_ridge(name: &str, ridge: Option<f64>) -> Result<(), String> {
    if let Some(r) = ridge.filter(|r| !non_negative(*r) || !r.is_finite()) {
        return Err(format!(
            "spec {name:?}: ridge must be finite and >= 0, got {r}"
        ));
    }
    Ok(())
}

/// `solve_every`: clock units between solves; zero solves every row. A
/// negative value silently meant "every row" too, NaN meant "never".
fn check_solve_every(name: &str, v: Option<&Span>) -> Result<(), String> {
    if let Some(v) = v.filter(|v| !non_negative(v.value()) || !v.value().is_finite()) {
        return Err(format!(
            "spec {name:?}: solve_every must be finite and >= 0 (0 solves every row), got {v}"
        ));
    }
    Ok(())
}

impl Serialize for Num {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            s.serialize_f64(self.0)
        } else if self.0 == f64::INFINITY {
            s.serialize_str("inf")
        } else if self.0 == f64::NEG_INFINITY {
            s.serialize_str("-inf")
        } else {
            s.serialize_str("nan")
        }
    }
}

impl Num {
    /// The words accepted in place of a number for the non-finite values.
    pub(crate) fn from_word(w: &str) -> Option<Num> {
        match w.to_ascii_lowercase().as_str() {
            "inf" | "+inf" | "infinity" | "+infinity" => Some(Num(f64::INFINITY)),
            "-inf" | "-infinity" => Some(Num(f64::NEG_INFINITY)),
            "nan" => Some(Num(f64::NAN)),
            _ => None,
        }
    }
}

// Hand-written visitors rather than `#[serde(untagged)]`: an untagged enum
// that matches nothing reports "data did not match any variant of untagged
// enum FloatOrList", which names a Rust type and not what was expected. A
// visitor says `invalid type: string "10", expected a number or a list of
// numbers ("inf" allowed)`, and does so for JSON, TOML and the msgpack state
// file alike (all three are self-describing, so `deserialize_any` is exactly
// what the untagged form used underneath).

struct NumVisitor;

impl serde::de::Visitor<'_> for NumVisitor {
    type Value = Num;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number or \"inf\"/\"-inf\"")
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Num, E> {
        Ok(Num(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Num, E> {
        Num::from_word(v).ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
    }
}

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NumVisitor)
    }
}

/// A float or a list of floats (grids; `half_life` lists mean one accumulator
/// per value, docs/PLAN.md §4.1).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FloatOrList {
    Float(Num),
    List(Vec<Num>),
}

impl<'de> Deserialize<'de> for FloatOrList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = FloatOrList;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number or a list of numbers (\"inf\" allowed)")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<FloatOrList, E> {
                Num::from_word(v)
                    .map(FloatOrList::Float)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                seq: A,
            ) -> Result<FloatOrList, A::Error> {
                Vec::<Num>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(FloatOrList::List)
            }
        }

        d.deserialize_any(V)
    }
}

impl FloatOrList {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrList::Float(f) => vec![f.0],
            FloatOrList::List(v) => v.iter().map(|n| n.0).collect(),
        }
    }
}

/// `session_gap`: clock units, a duration under a temporal clock, or
/// "reset". The gap is a [`Span`], so an infinite one ("never") survives
/// JSON, which has no infinity literal.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SessionGapSpec {
    Gap(Span),
    Word(String),
}

impl<'de> Deserialize<'de> for SessionGapSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl serde::de::Visitor<'_> for V {
            type Value = SessionGapSpec;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a gap in clock units (\"inf\" for never), a duration such as \"10m\", \
                     or the word \"reset\"",
                )
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<SessionGapSpec, E> {
                if v == "reset" {
                    return Ok(SessionGapSpec::Word(v.to_string()));
                }
                if let Some(n) = Num::from_word(v) {
                    return Ok(SessionGapSpec::Gap(Span::Units(n.0)));
                }
                crate::span::Duration::parse(v)
                    .map(|d| SessionGapSpec::Gap(Span::Duration(d)))
                    .map_err(|e| E::custom(format!("{e}, or the word \"reset\"")))
            }
        }

        d.deserialize_any(V)
    }
}

/// How many ranges of features a `marginal` splits its pair work into
/// (docs/PLAN.md task 126): a count of at least one, or `"auto"`, which the
/// bank sizes from the model's width and its own pool. A setting, not state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardSpec {
    Count(usize),
    Auto,
}

impl Serialize for ShardSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ShardSpec::Count(n) => s.serialize_u64(*n as u64),
            ShardSpec::Auto => s.serialize_str("auto"),
        }
    }
}

/// A count or `"auto"`, each read by its own rule, so a wrong value is
/// named as what it is.
impl<'de> Deserialize<'de> for ShardSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = ShardSpec;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a number of shards of at least 1, or \"auto\"")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<ShardSpec, E> {
                usize::try_from(v)
                    .ok()
                    .filter(|n| *n >= 1)
                    .map(ShardSpec::Count)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Unsigned(v), &self))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<ShardSpec, E> {
                match u64::try_from(v) {
                    Ok(v) => self.visit_u64(v),
                    Err(_) => Err(E::invalid_value(serde::de::Unexpected::Signed(v), &self)),
                }
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ShardSpec, E> {
                match v {
                    "auto" => Ok(ShardSpec::Auto),
                    _ => Err(E::invalid_value(serde::de::Unexpected::Str(v), &self)),
                }
            }
        }
        d.deserialize_any(V)
    }
}

/// What `ewridge`'s ridge is scaled against (docs/PLAN.md task 151): a
/// penalty on the mean moments, permanent, or a prior on the decaying sums,
/// fading (`"sum"`, classic RLS regularization; task 144 named the
/// difference, where `ridge_decay` was a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RidgeScale {
    #[default]
    Mean,
    Sum,
}

/// A windowed model's `closed`: the window operators' name and two of its
/// values (`crate::windows::Closed`, Polars' `rolling_*_by(closed=)`;
/// docs/PLAN.md task 196, N17), which edge of the window holds the row
/// exactly `window_size` old. `Right`, the default, keeps the rows less
/// than `window_size` old -- a row exactly that old has left, as it has in
/// the window operators and in `rolling_*_by` -- and `Both` keeps it too.
/// Polars' `"left"` and `"none"` leave out the row the window ends at,
/// which a model cannot do: it reads its fit after it has learned that row.
/// The two are refused with that reason as the spec is read, and any other
/// word with the two this takes, so the words the parameter takes are the
/// words it accepts (review round 5, E2: the API snapshot pinned all four).
/// Written as `Closed` writes the same two, so a state file does not move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelClosed {
    #[default]
    Right,
    Both,
}

impl<'de> Deserialize<'de> for ModelClosed {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = ModelClosed;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("`right` or `both`")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ModelClosed, E> {
                match v {
                    "right" => Ok(ModelClosed::Right),
                    "both" => Ok(ModelClosed::Both),
                    "left" | "none" => Err(E::custom(format!(
                        "closed = \"{v}\" leaves out the current row, the one the window ends \
                         at; a model reads its fit after it has learned that row, so its window \
                         always holds it: give \"right\" (the default) or \"both\""
                    ))),
                    _ => Err(E::unknown_variant(v, &["right", "both"])),
                }
            }
        }
        d.deserialize_str(V)
    }
}

/// The edge a windowed model's ring keeps for its `closed`.
fn model_edge(closed: ModelClosed) -> online_core::WindowClosed {
    match closed {
        ModelClosed::Right => online_core::WindowClosed::Right,
        ModelClosed::Both => online_core::WindowClosed::Both,
    }
}

/// Model choice + model-specific params (docs/PLAN.md §4).
///
/// A key no variant has is an error, not ignored: a spec is typed by hand in
/// TOML, and a misspelt parameter that silently kept its default would change
/// the model without a word.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
/// **State files written before 2026-09-07 do not load.** The naming pass
/// that day renamed six spec keys with no aliases, and a spec is
/// deserialized with unknown fields denied, so a file naming `coef0` or
/// `n_max` is refused rather than silently losing the field.
///
/// That is a deliberate exception to hard rule 5, taken while the library is
/// days old and pre-1.0, on the grounds that the names are worth more than
/// the compatibility: `MIN_SCHEMA_VERSION` became 6, the fixtures that
/// proved older files load went, and a state saved by 0.2.0 has to be refit.
/// It was taken again on 2026-09-14 (`MIN_SCHEMA_VERSION` 8, docs/PLAN.md
/// task 81): a state saved before `target_gaps` has to be refit too. And on
/// 2026-09-15 (`MIN_SCHEMA_VERSION` 9, the code review's S29/S30 and C24): a
/// state saved before `holt`'s weighted means and `ftrl`'s proximal sum.
pub enum ModelKind {
    /// `type = "ewridge"`, the builder's, the README's and the core's
    /// spelling; `"ew_ridge"` is refused naming it ([`RENAMED_VALUES`],
    /// docs/PLAN.md task 196, N1).
    #[serde(rename = "ewridge")]
    EwRidge {
        #[serde(default)]
        ridge: Option<FloatOrList>,
        #[serde(default)]
        feature_sets: Option<Vec<(String, Vec<String>)>>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        ridge_scale: RidgeScale,
        /// Shrink toward these coefficients instead of toward zero, one vector
        /// per target of length `n_features + intercept`, in original units.
        #[serde(default)]
        coef_prior: Option<Vec<Vec<f64>>>,
        /// On a session change, fit on this share of a slow-moving twin's
        /// moments, the rest today's, at today's weight: 0 keeps today's
        /// fit, 1 takes the long run's. Needs `long_half_life`.
        #[serde(default)]
        session_shrink: Option<f64>,
        /// Half-life of that twin; `"inf"` makes the long run the whole
        /// history (review 2026-09-12, S27).
        #[serde(default)]
        long_half_life: Option<Span>,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        /// Rows of the Gram update held back and merged as one block
        /// (docs/ENHANCEMENTS.md E51): `0` or absent updates the `k×k`
        /// matrix on every row; `256` is the measured setting, `6.6×` faster
        /// at a thousand features. Needs a solve cadence (`solve_every > 0`
        /// and `max_rows_between_solves > 1`), does not combine with
        /// `window_size`, and the merged sum is not bit-identical to the per-row
        /// one. Chunk invariance holds either way.
        #[serde(default)]
        gram_block_rows: Option<usize>,
        /// Threads the Gram's `k²` work runs on (docs/PLAN.md task 225):
        /// a block's merge under `gram_block_rows`, the per-row update
        /// otherwise. The same bits at every count; absent is one, and `0`
        /// is refused. Skipped when absent, so only a spec that sets it
        /// writes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gram_threads: Option<usize>,
        /// Which rows a target's fit is read from where the target is null
        /// on some (docs/PLAN.md task 81): `"own_rows"`, the default, fits it
        /// on exactly its rows; `"pairwise"` reads the Gram over every row
        /// and its cross-moments over its own.
        #[serde(default)]
        target_gaps: TargetGaps,
        /// Clock units of history the fit sees, with a **hard** cutoff: a row
        /// older than this is not in the Gram at all, where the exponential
        /// weight alone would leave `0.5^(age/half_life)` of it. Inside the
        /// window the weights are still exponential (docs/PLAN.md §13). A
        /// half-life grid is one instance per entry, each with its own ring.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the snapshots the window is computed from, as
        /// `solve_every` is between solves: a number of the clock column's
        /// units, or a duration on a temporal clock, `0` every row
        /// (docs/PLAN.md task 162). With `max_rows_between_snapshots` too,
        /// whichever comes first; with neither, every row.
        #[serde(default)]
        window_every: Option<Span>,
        /// At most this many rows between the window's snapshots, as
        /// `max_rows_between_solves` is between solves; `1` is every row,
        /// and `0`, no schedule, is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    Lasso {
        /// Decreasing penalties on standardized stats; required.
        lasso_path: Vec<f64>,
        /// 1.0 = lasso, < 1.0 = elastic net.
        #[serde(default)]
        l1_ratio: Option<f64>,
        /// Half-life of the EW squared error used to select lambda; `"inf"`
        /// selects on the plain mean over every row so far.
        #[serde(default)]
        select_half_life: Option<Span>,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        max_iter: Option<u32>,
        #[serde(default)]
        tol: Option<f64>,
        /// Which rows a target's path is fitted from where the target is
        /// null on some, as for `EwRidge` (docs/PLAN.md task 81).
        #[serde(default)]
        target_gaps: TargetGaps,
        /// Clock units of history the path is fitted from, with a **hard**
        /// cutoff (docs/PLAN.md §13). The selection error follows the same
        /// window, so the chosen `lambda` fits the rows the model reports on.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        window_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    Kalman {
        /// Per-factor coefficient half-life (scalar or one per slot, intercept
        /// first). `inf` pins a coefficient. Note this is the COEFFICIENT
        /// half-life; the spec-level `half_life` drives the standardization and
        /// residual-variance statistics. Exactly one of it and `q` is given:
        /// it was required, and ignored beside a `q` (review 2026-10-06,
        /// PC6). Optional, so a state written when it was required reads.
        #[serde(default)]
        coef_half_life: Option<SpanList>,
        /// The process noise per slot, given outright rather than derived from
        /// `coef_half_life`, which is then refused -- as `_spec.py`'s
        /// `kalman` docstring says (review 2026-09-12, S22 and V16; review
        /// 2026-10-06, PC6).
        #[serde(default)]
        q: Option<Vec<Num>>,
        #[serde(default)]
        obs_var: Option<f64>,
        #[serde(default)]
        p0: Option<f64>,
        #[serde(default)]
        share_p: bool,
        /// Per-slot reversion half-life (scalar or one per slot, intercept
        /// first): the coefficient mean shrinks toward zero by `2^(-d/r_i)`
        /// per row. `inf` (the default) is the random walk
        /// (docs/ENHANCEMENTS.md E41).
        #[serde(default)]
        revert_half_life: Option<SpanList>,
        /// Standardize features internally (default true). Off makes the filter
        /// a plain Bayesian linear regression on the features' own scale.
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Huber regression (docs/PLAN.md §4.5).
    Huber {
        /// Cut point in units of the EW residual std. Default 1.345, the
        /// constant of 95% efficiency at the normal (Huber 1981), filled
        /// where the model is built ([`crate::build_models`]); `"inf"` cuts
        /// nothing, which is least squares.
        #[serde(default)]
        huber_delta: Option<Num>,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
    },
    /// Quantile regression at level `quantile` (docs/PLAN.md §4.5).
    Quantile {
        quantile: f64,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        /// Half-width of the band the fit takes its Newton step in, in units
        /// of the EW residual std; default 0.2 (review 2026-09-12, N9), and
        /// never narrower than `(k/n)^(2/5)` for the target's effective sample
        /// `n` (the second review's F3). A band holding under one row per
        /// coefficient takes least-squares rows until it holds rows again,
        /// which rebuilds a fit a row at the input bound left behind
        /// (`online_core::robust`'s module docs).
        #[serde(default)]
        quantile_eps: Option<f64>,
    },
    /// Online logistic regression via FTRL-proximal (docs/PLAN.md §4.6).
    /// `pred` is a probability; `resid = y - p`.
    Ftrl {
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        #[serde(default)]
        l1: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        /// Refuse a chunk whose target is not 0 or 1, naming the row, rather
        /// than clamp it into [0, 1] (review 2026-09-12, S31). Logistic loss
        /// only.
        #[serde(default)]
        strict_binary: bool,
        /// "logistic" (default, binary targets, `pred` is a probability) or
        /// "squared" (continuous targets, sparse linear regression).
        #[serde(default)]
        loss: Option<String>,
    },
    /// EW moments of the feature columns, no regression (docs/PLAN.md §4.7).
    /// `targets` is ignored; every column of interest goes in `features`.
    EwCov {
        /// Any of "mean", "var", "std", "cov", "corr", "partial_corr", "mahal",
        /// "lag_corr" (the last with `lags`).
        /// Default: mean + std + corr.
        #[serde(default)]
        stats: Option<Vec<String>>,
        /// Prior for the precision matrix, required by "partial_corr" and "mahal".
        #[serde(default)]
        precision_prior: Option<f64>,
        /// Quantile levels of the past Mahalanobis scores, exponentially
        /// weighted at the model's half-life, one `mahal_q<p>` field each;
        /// needs "mahal" in `stats` (E37, docs/PLAN.md task 146).
        #[serde(default)]
        mahal_quantiles: Option<Vec<f64>>,
        /// Principal components to track, each `pc<j>_var`, `pc<j>_share`,
        /// `pc<j>_<feature>` per feature and `pc<j>_score` (E38).
        #[serde(default)]
        pca: Option<usize>,
        /// Clock units between refreshes of the components, as `solve_every`
        /// is the regressions': a number of the clock column's units, or a
        /// duration on a temporal clock, `0` every row (docs/PLAN.md task
        /// 161). With `max_rows_between_pca` too, whichever comes first;
        /// with neither, every row.
        #[serde(default)]
        pca_every: Option<Span>,
        /// At most this many rows between refreshes, as
        /// `max_rows_between_solves` is the regressions'; `1` is every row,
        /// and `0` is refused (tasks 161 and 196).
        #[serde(default)]
        max_rows_between_pca: Option<u32>,
        /// Lags to accumulate cross-moments at, in output order
        /// (docs/ENHANCEMENTS.md E56): strictly increasing, `>= 1` and at
        /// most 2^20 (`online_core::MAX_LAG`, the ring being sized before the
        /// first row), counted in *learned rows within the group*. Read from `gram()`
        /// as `lags`/`lag_comoments`, or emitted as `lag_corr_<a>_<b>_l<l>`
        /// by adding `"lag_corr"` to `stats`.
        #[serde(default)]
        lags: Option<Vec<usize>>,
        /// Threads the per-row update of the `k × k` co-moments runs on
        /// (docs/PLAN.md task 225), as `EwRidge`'s: the same bits at every
        /// count, absent is one, `0` is refused, skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gram_threads: Option<usize>,
        /// Clock units of history the statistics see, with a **hard** cutoff:
        /// a row older than this contributes nothing, where the exponential
        /// weight alone would leave `0.5^(age/half_life)` of it. Inside the
        /// window the weights are still exponential -- it is not a flat
        /// window (docs/PLAN.md §13).
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the snapshots the window is computed from, as
        /// for `EwRidge`, `0` every row (docs/PLAN.md task 162). Every row,
        /// the default, is the tightest boundary; a coarser cadence divides
        /// the memory and shortens the effective window by at most one
        /// spacing -- never lengthens it.
        #[serde(default)]
        window_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    /// Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).
    /// O(k) per row, no solves, and the only model here that takes count
    /// targets (via `loss = "poisson"`).
    Sgd {
        /// "squared" (default), "huber", "quantile", "epsilon_insensitive",
        /// "poisson" or "logistic".
        #[serde(default)]
        loss: Option<String>,
        /// Huber cut, in units of the target's EW residual std, as
        /// `huber`'s; default 1.345. `"inf"` clips nothing, which is the
        /// squared loss. The residual's std, where `eps` is in the target's
        /// own: a clipped gradient keeps learning while early residuals
        /// widen the cut, and a zero one inside a band does not
        /// (docs/PLAN.md task 202).
        #[serde(default)]
        huber_delta: Option<Num>,
        #[serde(default)]
        quantile: Option<f64>,
        /// Half-width of the insensitive tube, in units of the target's
        /// own EW std, the spread of `y` around its EW mean (docs/PLAN.md
        /// task 202). Default 0.01: errors under 1% of the target's own
        /// spread do not move the fit. In units of the noise the tube is
        /// `eps / √(1 − R²)` wide, `R²` the fit's: 0.067 noise stds at R²
        /// 0.978, 2.2 at 0.99998, where a tube of 0.1, 22 noise stds, held
        /// the fit wherever it first landed inside it (out-of-sample error
        /// 24 noise variances above the noise against 0.20, `inv_scaling`;
        /// 0.010 against 0.015 at R² 0.978). Task 207 kept 0.01.
        #[serde(default)]
        eps: Option<f64>,
        #[serde(default)]
        learning_rate: Option<f64>,
        /// "constant" (default), "inv_scaling" or "adagrad".
        #[serde(default)]
        schedule: Option<String>,
        /// Exponent for `inv_scaling`.
        #[serde(default)]
        power: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        #[serde(default)]
        /// `Num`, not `f64`: the documented way to disable clipping is
        /// `inf`, and JSON cannot carry Infinity as a number -- `Num` accepts
        /// the string "inf", exactly as the half-life fields do.
        clip_gradient: Option<Num>,
        /// Standardize features against their running moments before the
        /// gradient step, unscaling the coefficients on the way out.
        /// Default true, as `kalman`'s: a gradient step's one learning rate
        /// has to suit every feature (docs/PLAN.md task 195, U2).
        #[serde(default = "default_true")]
        standardize: bool,
        /// Lower bound per slope (ENHANCEMENTS E40): one number for every
        /// feature or a list with one entry per feature; "-inf" for none.
        /// The intercept is never bounded. Imposed by Euclidean projection
        /// after each update, in the space the step is taken in.
        #[serde(default)]
        coef_min: Option<FloatOrList>,
        /// Upper bound per slope, as `coef_min`; "inf" for none.
        #[serde(default)]
        coef_max: Option<FloatOrList>,
        /// The slopes sum to this, in the caller's units. With `coef_min =
        /// 0` and `coef_sum = 1` the slopes are weights on the simplex.
        #[serde(default)]
        coef_sum: Option<f64>,
        /// Refuse a chunk whose target is not 0 or 1, naming the row, rather
        /// than clamp it into [0, 1]: `ftrl`'s rule. Logistic loss only
        /// (docs/PLAN.md task 195, S4).
        #[serde(default)]
        strict_binary: bool,
    },
    /// Passive-aggressive regression (ENHANCEMENTS E17). No learning rate:
    /// each row's update is the smallest change that satisfies it.
    Pa {
        /// "pa1" (default), "pa" (unbounded) or "pa2" (damped).
        #[serde(default)]
        mode: Option<String>,
        /// Aggressiveness: under "pa1" a cap on the step, in the target's
        /// units over `‖z‖²`'s, so the same `c` binds by the target's scale
        /// (review round 5, G5); under "pa2" the damping `1/(2c)` beside
        /// `‖z‖²`, free of the target's units. Default 1. Ignored by "pa";
        /// `"inf"` caps nothing, so either bounded mode is "pa".
        #[serde(default)]
        c: Option<Num>,
        /// Insensitive tube, in units of the target's own EW std, the
        /// spread of `y` around its EW mean: rows already this close leave
        /// the fit alone (docs/PLAN.md task 202). Default 0.01: errors under
        /// 1% of the target's own spread do not move the fit. In units of
        /// the noise it is `eps / √(1 − R²)` wide, `R²` the fit's: 0.067
        /// noise stds at R² 0.978, where a binding cap damps and a wider
        /// tube would too (out-of-sample error 1.15 noise variances above
        /// the noise at the defaults, 0.33 at `c = 0.1`, 0.57 at `eps =
        /// 0.1`, on a target of spread about 2); 2.2 at 0.99998, where a
        /// tube of 0.1, 22 noise stds, holds the fit wherever it first lands
        /// inside it (0.14 at the defaults, 65 at `eps = 0.1`). Task 207
        /// kept 0.01.
        #[serde(default)]
        eps: Option<f64>,
        /// Bounds and sum on the slopes, as for `sgd` (ENHANCEMENTS E40).
        #[serde(default)]
        coef_min: Option<FloatOrList>,
        #[serde(default)]
        coef_max: Option<FloatOrList>,
        #[serde(default)]
        coef_sum: Option<f64>,
        /// Standardize features against `sgd`'s running moments. Default
        /// true (docs/PLAN.md task 195, U2).
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Holt's linear trend method (ENHANCEMENTS E25): level plus slope, no
    /// features. The baseline a feature-based model should have to beat.
    /// The level's half-life is the spec's own `half_life` (or `lam`),
    /// `"inf"` included: no forgetting. It had a second name,
    /// `level_half_life`, which built a second dict shape and is refused
    /// naming `half_life` ([`RENAMED`]; review 2026-10-06, TA6; docs/PLAN.md
    /// task 196, N16).
    Holt {
        /// Half-life of the trend; `"inf"` forgets no slope, so the trend is
        /// the whole history's drift. Defaults to four times the level's
        /// half-life.
        #[serde(default)]
        trend_half_life: Option<Span>,
        /// `false` fits the level alone: the trend is held at zero and the
        /// forecast is flat, simple exponential smoothing (docs/PLAN.md task
        /// 115, S30). `trend_half_life` is refused beside it. Skipped when
        /// absent: the trend is on.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend: Option<bool>,
    },
    Rls {
        /// Prior strength: `A0 = delta I`, i.e. `P0 = I / delta`, the
        /// classic RLS name; `ew_ridge`'s `ridge` under `ridge_scale =
        /// "sum"`, not its `ridge` (docs/PLAN.md task 195, N11). Default 1.
        /// Scalar only (baked into the state).
        #[serde(default)]
        delta: Option<f64>,
        #[serde(default)]
        coef_prior: Option<Vec<Vec<f64>>>,
    },
    /// Exponentially weighted k-means (docs/CLUSTERING.md §6.2; PLAN §11a,
    /// task 23). No targets: every column of interest goes in `features`, and
    /// the outputs are the nearest centre's index and two distances, read
    /// before the row is learned.
    #[serde(rename = "kmeans")]
    KMeans {
        /// Number of clusters, `1 ..= 2^16` (`online_core::KMeans::MAX_K`).
        k: usize,
        /// Learned rows buffered before seeding, at least `k`: fewer is
        /// refused by name, as `hmm` refuses it, where it was floored to `k`
        /// in silence (review 2026-10-06, PC8). Default 500, or `k` where
        /// that is more. The buffer is held to 256 MiB
        /// (`online_core::KMeans::WARM_BUDGET_MIB`).
        #[serde(default)]
        warm_rows: Option<usize>,
        /// "lloyd" (default), "kmeanspp", "farthest" or "first".
        #[serde(default)]
        seed_rule: Option<String>,
        /// Seed of the generator behind "kmeanspp" and "lloyd". Default 0.
        #[serde(default)]
        seed: Option<u64>,
        /// Learned rows between centre updates. Default 1 (every row).
        #[serde(default)]
        update_every_rows: Option<u32>,
        /// Merge the two closest clusters when their centres are closer than
        /// this many summed radii, re-placing the freed centre at the
        /// farthest row seen. Default 0.5; `0` disables split–merge.
        #[serde(default)]
        split_merge: Option<f64>,
        /// Learned rows between split–merge checks. Default 100.
        #[serde(default)]
        split_merge_every_rows: Option<u32>,
        /// A cluster lighter than `dead_frac · n_eff / k` at a check is
        /// re-placed. Default 0.05; `0` disables the dead rule. The rule runs
        /// at a split–merge check, so under `split_merge = 0` the default is
        /// 0 and a value above it is refused (review 2026-10-06, CF6).
        #[serde(default)]
        dead_frac: Option<f64>,
        /// Measure distances in units of each feature's EW standard
        /// deviation. Default true.
        #[serde(default)]
        standardize: Option<bool>,
        /// Floor the metric's variance at this fraction of the feature's
        /// long-run (undecayed) variance. Default 0.1; `0` is the EW
        /// variance alone.
        #[serde(default)]
        scale_floor: Option<f64>,
    },
    /// DenStream-style micro-clusters with a linkage macro step
    /// (docs/CLUSTERING.md §6.5; PLAN §11a, task 24). No targets: the
    /// outputs are the nearest cluster's label and distance, the
    /// micro-cluster id the row goes to, an outlier flag and two counts,
    /// read before the row is learned; the potential summaries ride in
    /// `coef`, one row each.
    #[serde(rename = "micro")]
    Micro {
        /// Bound on a summary's RMS radius per standardized coordinate:
        /// `eps √p` in the metric. Required, finite, `> 0`.
        eps: f64,
        /// Weight at which a summary becomes potential (DenStream's βµ).
        /// Default 3.
        #[serde(default)]
        beta_mu: Option<f64>,
        /// Cap on live summaries; at the cap the lightest outlier summary is
        /// evicted, else the lightest potential one. Default 200.
        #[serde(default)]
        max_clusters: Option<usize>,
        /// Clock units between checkpoints (pruning, then linkage), as
        /// `solve_every` is the regressions': a number of the clock column's
        /// units, or a duration on a temporal clock, `0` every row
        /// (docs/PLAN.md task 163). With `max_rows_between_prunes` too,
        /// whichever comes first; with neither, every 100 learned rows.
        #[serde(default)]
        prune_every: Option<Span>,
        /// At most this many learned rows between checkpoints (task 163).
        #[serde(default)]
        max_rows_between_prunes: Option<u32>,
        /// Single-linkage threshold in units of `eps √p`; `0` links nothing.
        /// Default: derived from the spacing of the potential summaries at
        /// each checkpoint.
        #[serde(default)]
        macro_link: Option<f64>,
        /// Measure distances in units of each feature's EW standard
        /// deviation. Default true.
        #[serde(default)]
        standardize: Option<bool>,
        /// Floor the metric's variance at this fraction of the feature's
        /// long-run (undecayed) variance. Default 0.1; `0` is the EW
        /// variance alone.
        #[serde(default)]
        scale_floor: Option<f64>,
    },
    /// Class-conditional Gaussian classifier on exponentially weighted
    /// class moments (docs/ENHANCEMENTS.md E39; PLAN §11a, task 27). The
    /// one target is the label column, which names its class by value; the
    /// outputs are the most probable class and one posterior per class,
    /// read before the row is learned, and `coef` holds the class means.
    #[serde(rename = "ew_class")]
    EwClass {
        /// The classes, in output order; a label value not listed here is
        /// an error, a null label scores the row without learning from it.
        classes: Vec<String>,
        /// "full" (default: one covariance per class, QDA), "shared" (one
        /// pooled covariance, LDA) or "diagonal" (naive Bayes).
        #[serde(default)]
        covariance: Option<String>,
        /// Ridge on every class covariance, finite and `> 0`; it decays as
        /// the class accumulates data, like `ew_cov`'s `precision_prior`.
        precision_prior: f64,
        /// Clock units of history each class's moments are computed from,
        /// with a **hard** cutoff (docs/PLAN.md §13). `covariance = "full"`
        /// pays one factorization per class per row under a window, because
        /// the truncated covariance moves every row where a decayed one does
        /// not; the other shapes are unaffected.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        window_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    /// Sequential test of a sign by betting (docs/ENHANCEMENTS.md E42;
    /// PLAN §11a, task 30): per target, two e-processes, one for "positive"
    /// and one for "negative", each a Kelly bettor with a
    /// Krichevsky–Trofimov stake on the sign counts so far. The outputs are
    /// the two log e-values and the two counts, read before the row is
    /// learned, so `exp(log_e) >= 1/alpha` at any row is evidence at level
    /// `alpha` -- anytime-valid, under no assumption but that the signs are
    /// not predictable from the past. No features, no decay, no `weight`
    /// (every learned row is one trial); a null or zero target bets nothing.
    ///
    /// Two specs of the same bank are compared by naming them as `a` and
    /// `b`: each target `t` then names a residual field both specs carry
    /// (`resid_<t>`, plus the side's grid suffix when it has one), the sign
    /// tested is that of `|resid_b| - |resid_a|` -- positive when `a` was
    /// closer on the row -- and the fields are `log_e_a_<t>`, `log_e_b_<t>`,
    /// `wins_a_<t>`, `wins_b_<t>`. The bank runs `a` and `b` first, so the
    /// comparison reads the same out-of-sample residuals the two structs
    /// report.
    #[serde(rename = "seqtest")]
    SeqTest {
        /// The spec whose predictions are hoped to be better.
        #[serde(default)]
        a: Option<String>,
        /// The spec it is compared with.
        #[serde(default)]
        b: Option<String>,
        /// The grid suffix of `a`'s residual field, when `a` is a grid:
        /// `resid_<t><a_suffix>`, e.g. `__r0.1` or `@h50`. Default none.
        #[serde(default)]
        a_suffix: Option<String>,
        /// The same for `b`.
        #[serde(default)]
        b_suffix: Option<String>,
    },
    /// Exponentially weighted moments of every feature against every target,
    /// one pair at a time (docs/ENHANCEMENTS.md E44; PLAN §11a, task 37):
    /// per (target, feature) the two means, the two variances, the
    /// covariance, and the target's `Σw` and `Σw²` -- `O(p·T)` state where
    /// an `ew_cov` over the same columns keeps `O((p+T)²)`. It emits nothing
    /// per row but `n_eff`; the pairs are read from the bank as a long frame
    /// (`ModelBank.marginal`), each with its correlation, the slope of the
    /// target on the feature, Kish's effective sample size and the
    /// t-statistic of the correlation at that size. A pair's correlation
    /// is the one an `ew_cov` over the two columns reports, to the bit. A
    /// null target ages that target's pairs and moves nothing else; a
    /// null feature drops the row for every pair, as everywhere.
    /// `half_life`/`lam`, `weight`, `clock` and `min_weight` are the spec's;
    /// the lags, the bins, the window and the shards below are its own.
    /// `min_weight` (default 3) is the
    /// weight a target needs before its pairs' `corr`, `beta` and `t` are
    /// reported -- a correlation of two rows is ±1 whatever the data.
    #[serde(rename = "marginal")]
    Marginal {
        /// Lags to accumulate pair moments at (docs/ENHANCEMENTS.md E66):
        /// strictly increasing, `>= 1` and at most 2^20
        /// (`online_core::MAX_LAG`), counted in **learned rows within the
        /// group**. Gives the two autocorrelations and both
        /// cross-correlations per pair, and the serial-dependence-corrected
        /// `n_serial` when `serial_rule` asks for it.
        ///
        /// **Skipped when absent**, unlike the spec keys that predate
        /// 2026-09-07: a key that writes `null` moves every spec's bytes and
        /// costs a `SCHEMA_VERSION` bump for a model nobody using it has
        /// heard of. New optional keys skip, so adding one to a model
        /// changes only the states that use it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lags: Option<Vec<usize>>,
        /// `"truncated"`, `"bartlett"` or `"geometric"`: how `n_serial` is
        /// formed from the lags. Needs `lags`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        serial_rule: Option<String>,
        /// The lags to keep the cross-correlations at (E70, docs/PLAN.md
        /// task 123): strictly increasing, each one of `lags`, empty for
        /// none. Absent keeps them at every lag. `n_serial` reads the
        /// autocorrelations alone, which every lag keeps. Needs `lags`.
        /// Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cross_lags: Option<Vec<usize>>,
        /// Bins per feature for the binned target moments
        /// (docs/ENHANCEMENTS.md E67): the feature's response curve, and the
        /// best single split of it, which is what a threshold or a V shows
        /// up in when `corr` cannot see it. Skipped when absent, as `lags`
        /// is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bins: Option<usize>,
        /// `"quantile"` (equal weights, the default) or `"fixed"` (equal
        /// widths): how `bins` becomes edges. Refused with `bin_edges`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_rule: Option<String>,
        /// Learned rows to hold before fixing the edges, default 1,000. The
        /// rows are held and replayed, not spent. Refused with `bin_edges`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_warm_rows: Option<usize>,
        /// Explicit interior edges, one strictly increasing list per feature
        /// in feature order. Exact and comparable across runs, and skips the
        /// warm-up entirely; `bins`, `bin_rule` and `bin_warm_rows` are the
        /// learned kind's and refused beside it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_edges: Option<Vec<Vec<f64>>>,
        /// The MiB the bins' warm-up hold and histogram may each take, per
        /// model instance -- every group keeps its own, and so does every
        /// half-life of a grid -- before the model refuses to build
        /// (docs/PLAN.md task 131): 256 when absent, `"inf"` no bound. Needs
        /// `bins` or `bin_edges`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_budget: Option<Num>,
        /// Ranges of features to split the pair work into, run on the bank's
        /// pool a batch of rows at a time (docs/PLAN.md task 126): a count,
        /// or `"auto"` for as many as the width can keep busy. The numbers
        /// are the same to the bit whatever it says, so it is a setting,
        /// not state. Absent is one, the pairs row by row on the group's
        /// own thread. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shards: Option<ShardSpec>,
        /// Clock units of history the pairs are computed from, with a
        /// **hard** cutoff: a row older than this contributes nothing
        /// (docs/PLAN.md §13). Inside the window the weights are still
        /// exponential.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        window_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
        /// Accept `lags` under a `window_size` at its price (docs/PLAN.md task
        /// 137): each window snapshot then also holds the lag moments, `L·T
        /// + (L + 2C)·p·T` doubles beside the `(3p + 5)·T` it holds without
        /// them, adding about `(L + 2C)/3` times as many. Refused without both a
        /// window and lags; the pair is refused without it. Skipped when
        /// absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_lags: Option<bool>,
        /// Where the feature moments are kept (docs/PLAN.md task 125):
        /// `"per_target"`, the default, keeps each pair's mean and variance of
        /// the feature over its target's rows; `"shared"` keeps one per
        /// feature over every learned row, `p` where the default keeps `p·T`,
        /// and under `lags` the feature's autocovariance with them. Refused
        /// with a window, as the core says. Skipped when absent, as
        /// `window_lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        feature_moments: Option<String>,
    },
    /// Dynamic equicorrelation (Engle & Kelly 2012; docs/ENHANCEMENTS.md
    /// E55): one number for the whole correlation matrix, `O(m)` a row where
    /// a full one is `O(m²)`.
    ///
    /// The row is standardised against an [`online_core::EwDiag`]'s pre-row
    /// means and variances, and the average pairwise correlation of the
    /// standardised row is a closed form in two sums. `blocks` estimates one
    /// number per named block and one per pair of blocks instead of a single
    /// one, which is the useful middle between "one correlation" and "all of
    /// them". Outputs `u` (the row's own estimate), `rho` (the level before
    /// the row), `loglik` (the row's Gaussian density under it) and `n_eff`;
    /// with blocks, one `u_*` and one `rho_*` per value and a single
    /// `loglik`. `half_life`/`lam` are the spec's and are required, since
    /// both the standardiser and the level decay on them.
    #[serde(rename = "deco")]
    Deco {
        /// `"ew"` (default): `rho' = a·rho + b·u`, the exponentially
        /// weighted mean of the row estimates. `"linear"`: Engle–Kelly
        /// eq. 21 with correlation targeting, `rho' = (1 − α − β)·rho_bar' +
        /// α·u + β·rho`, which needs `alpha` and `beta`.
        #[serde(default)]
        dynamics: Option<String>,
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        /// Named groups of features, in emission order; every feature must
        /// be in exactly one, and a block needs at least two. Omitted, the
        /// model estimates one number over every feature.
        /// `feature_sets`' shape: a Python dict, serialised as a list of
        /// `[name, columns]` pairs so the order the caller wrote is the
        /// order the outputs come in.
        #[serde(default)]
        blocks: Option<Vec<(String, Vec<String>)>>,
    },
    /// A block's realised covariance, robust to microstructure noise
    /// (docs/ENHANCEMENTS.md E57). Rows are **returns**; there is no decay
    /// and no per-row output but `n_eff`, because the model's value is its
    /// state at the group's close -- so it requires `group` and
    /// `group_close`, and the block rides in that row.
    #[serde(rename = "rcov")]
    Rcov {
        /// `"kernel"` (default), `"preavg"` or `"plain"`.
        #[serde(default)]
        kind: Option<String>,
        /// Only `"parzen"`.
        #[serde(default)]
        kernel: Option<String>,
        /// A fixed `H`, or omitted for BNHLS's automatic rule (which needs
        /// `block_rows`).
        #[serde(default)]
        bandwidth: Option<usize>,
        /// Observations averaged at each end; default 2, `1` is none.
        /// `"kernel"` only: the other kinds read none, and a value given to
        /// them is refused (review 2026-10-06, CE4).
        #[serde(default)]
        jitter: Option<usize>,
        /// Pre-averaging window scale, `kₙ = ⌈θ·block_rows^0.6⌉` under `psd`
        /// (the default) and `⌊θ√block_rows⌋` without it; default 1.
        /// `"preavg"` only, refused under the other kinds (CE4).
        #[serde(default)]
        theta: Option<f64>,
        /// Clip negative eigenvalues at close; default true.
        #[serde(default)]
        psd: Option<bool>,
        /// The block's expected length, which sizes the ring before the
        /// first row.
        #[serde(default)]
        block_rows: Option<usize>,
        /// Ring depth, if not the default from `block_rows`. `"kernel"`
        /// only, refused under the other kinds (CE4); the ring's lagged
        /// products, `(ring + 1)·k²` doubles, are held to 256 MiB (CE9).
        #[serde(default)]
        max_bandwidth: Option<usize>,
        /// A fixed pre-averaging length, in rows, instead of `⌊θ√block_rows⌋`.
        #[serde(default)]
        preavg_rows: Option<usize>,
        /// Subsampling stride for the noise estimate; default 1.
        #[serde(default)]
        noise_stride: Option<usize>,
        /// Subsampling stride for the sparse variance; default 20.
        #[serde(default)]
        iv_stride: Option<usize>,
    },
    /// A Gaussian hidden Markov model, filtered online
    /// (docs/ENHANCEMENTS.md E60). `ew_class` without the labels: the state
    /// is hidden, and a transition matrix carries information from one row
    /// to the next.
    #[serde(rename = "hmm")]
    Hmm {
        /// Hidden states, `2 ..= 1024` (`online_core::Hmm::MAX_K`).
        k: usize,
        /// `"full"` (default), `"shared"` or `"diagonal"`, as `ew_class`.
        #[serde(default)]
        covariance: Option<String>,
        /// Ridge on every state covariance; **required**, since a state's
        /// centred co-moments start at zero.
        precision_prior: f64,
        /// Update the states and the transition counts; default true.
        #[serde(default)]
        learn: Option<bool>,
        /// Dirichlet pseudo-count per cell of the transition matrix
        /// (default 1). With `transition` it is spread over that matrix, so
        /// the given one is the prior mean. Refused beside `tvtp_coef`,
        /// which learns no count (review 2026-10-06, CE4).
        #[serde(default)]
        transition_prior: Option<f64>,
        /// A `K x K` row-stochastic matrix, flattened row-major. Refused
        /// beside `tvtp_coef`, as `transition_prior` is.
        #[serde(default)]
        transition: Option<Vec<f64>>,
        /// State means, `K x d` row-major; with `covs`, there is no warm-up.
        #[serde(default)]
        means: Option<Vec<f64>>,
        /// State covariances, `K` matrices of `d x d`, row-major.
        #[serde(default)]
        covs: Option<Vec<f64>>,
        /// Learned rows buffered before the states are seeded (default 50).
        /// With `means` and `covs` given nothing is seeded, and it, like
        /// `seed_rule` and `seed`, is refused (review 2026-10-06, CE4).
        #[serde(default)]
        warm_rows: Option<usize>,
        /// `"first"`, `"farthest"`, `"kmeanspp"` or `"lloyd"` (default), as
        /// `kmeans`.
        #[serde(default)]
        seed_rule: Option<String>,
        #[serde(default)]
        seed: Option<u64>,
        /// A column whose value drives the transition matrix through
        /// `tvtp_coef`; declared like `weight`, not a feature.
        #[serde(default)]
        exog_tvtp: Option<String>,
        /// `[A, B]`, each `K x K` row-major: `Π(t) = softmax(A + B·z)`.
        #[serde(default)]
        tvtp_coef: Option<Vec<Vec<f64>>>,
    },
    /// Has the correlation structure changed? (docs/ENHANCEMENTS.md E59)
    ///
    /// `"monitor"` is Wied, Krämer & Dehling's closed-sample constancy
    /// test, run over consecutive spans of `span_rows` rows; `"sequential"`
    /// is Wied & Galeano's detector, a history of `span_rows` rows and then
    /// every row of a monitoring period tested against it; `"window"` is
    /// the size of the change between two adjacent blocks of `span_rows`,
    /// against a fixed threshold or a permutation critical value. A
    /// parameter that belongs to another kind is refused.
    #[serde(rename = "corrchange")]
    CorrChange {
        /// `"monitor"` (default), `"sequential"` or `"window"`.
        #[serde(default)]
        kind: Option<String>,
        /// Rows per comparison block, required by every kind: the span
        /// `"monitor"` tests for constancy (at least 8), the history
        /// `"sequential"` monitors against (at least 8), or the length of
        /// each of the two adjacent blocks `"window"` compares (at least 3).
        #[serde(default)]
        span_rows: Option<usize>,
        /// Nominal level; default 0.05.
        #[serde(default)]
        alpha: Option<f64>,
        /// `"bonferroni"` (default) over the pairs, or `"none"`.
        #[serde(default)]
        alpha_adjust: Option<String>,
        /// `"monitor"` and `"sequential"`: the Bartlett bandwidth, `⌊ln T⌋`
        /// (or `⌊ln span_rows⌋`) when unset.
        #[serde(default)]
        bandwidth: Option<usize>,
        /// `"monitor"` and `"sequential"`: test the equicorrelation of the
        /// standardised row instead of every pair.
        #[serde(default)]
        scalar: Option<bool>,
        /// `"window"`: a fixed critical value; unset draws a permutation
        /// one. `"sequential"`: replaces Wied & Galeano's.
        #[serde(default)]
        crit: Option<f64>,
        /// `"window"`: permutation draws per critical value; default 200, at
        /// least 20 and at most 2^20 (`online_core::MAX_PERM`).
        #[serde(default)]
        n_perm: Option<usize>,
        #[serde(default)]
        permute_every_rows: Option<usize>,
        /// Permute blocks of this many consecutive rows (default 1), so
        /// serial dependence does not make the null too liberal.
        #[serde(default)]
        perm_block: Option<usize>,
        /// `"l1"` (default) or `"linf"` over the strict upper triangle.
        #[serde(default)]
        norm: Option<String>,
        #[serde(default)]
        seed: Option<u64>,
        /// Empty the rings at a flag and start over (`"window"` only;
        /// `"monitor"`'s spans are disjoint already, and a `"sequential"`
        /// cycle ends at its flag).
        #[serde(default)]
        reset_on_flag: Option<bool>,
        /// `"sequential"`: the rows monitored after each history, Wied &
        /// Galeano's `⌊mT⌋`; default `span_rows` (`T = 1`).
        #[serde(default)]
        monitor_rows: Option<usize>,
        /// `"sequential"`: the boundary's exponent `γ`, `0 ≤ γ < 1/2`;
        /// default 0.
        #[serde(default)]
        boundary_gamma: Option<f64>,
    },
    /// Bayesian online changepoint detection (docs/ENHANCEMENTS.md E61): a
    /// posterior over how long the current run has lasted, updated one row
    /// at a time.
    #[serde(rename = "bocpd")]
    Bocpd {
        /// A number: the expected rows between changepoints, `H = 1/hazard`
        /// the chance of a break after each row, on any clock. A duration,
        /// which only a temporal clock reads: the expected time between
        /// them, the step of `d` into each row carrying the chance `1 −
        /// exp(−d/hazard)`, applied before the row is read (task 179).
        /// Default 250 rows. In [`DURATION_OR_UNIT_FREE_FIELDS`].
        #[serde(default)]
        hazard: Option<Span>,
        /// A column carrying a per-row hazard instead of one number: the
        /// chance of a break after its row. Refused beside a duration
        /// `hazard`.
        #[serde(default)]
        hazard_col: Option<String>,
        /// `"gaussian"`, `"diag"` (default) or `"robust"`.
        #[serde(default)]
        emission: Option<String>,
        /// `μ₀`; absent, the first `warm_rows` learned rows' sample mean.
        #[serde(default)]
        prior_mean: Option<Vec<f64>>,
        #[serde(default)]
        prior_kappa: Option<f64>,
        /// `ν₀`; absent, `d + 2` under `"gaussian"` and 3 under `"diag"`
        /// and `"robust"` (docs/PLAN.md task 195, U5).
        #[serde(default)]
        prior_nu: Option<f64>,
        /// A scalar for `sI`, or a `d x d` matrix, in the data's units;
        /// absent, the first `warm_rows` learned rows' sample covariance
        /// (its diagonal under `"diag"` and `"robust"`; task 195, U4).
        #[serde(default)]
        prior_scale: Option<Vec<f64>>,
        /// `"robust"`'s β.
        #[serde(default)]
        robust_beta: Option<f64>,
        #[serde(default)]
        prune_below: Option<f64>,
        #[serde(default)]
        max_run: Option<usize>,
        /// Learned rows held, reporting nothing, to set the priors left out
        /// from; default `d + 2`. Refused beside both `prior_mean` and
        /// `prior_scale` (task 195, U4).
        #[serde(default)]
        warm_rows: Option<usize>,
    },
}

/// The two sides of a `seqtest` comparison, as [`ModelKind::compares`]
/// reads them off the spec: each side's spec name and the grid suffix its
/// residual fields carry (`""` for a single instance).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compare<'a> {
    pub a: &'a str,
    pub b: &'a str,
    pub a_suffix: &'a str,
    pub b_suffix: &'a str,
}

impl Compare<'_> {
    /// The residual field target `t` names on each side.
    pub fn fields(&self, t: &str) -> (String, String) {
        (
            format!("resid_{t}{}", self.a_suffix),
            format!("resid_{t}{}", self.b_suffix),
        )
    }
}

impl ModelKind {
    /// Every model the bank can build, by the `type` a spec names it with,
    /// in declaration order. The registry the Python side and the tests
    /// check themselves against (docs/EXTENDING.md); `kinds_tests` holds it
    /// to the enum, so a new variant fails a test until it is listed here.
    pub const KINDS: &'static [&'static str] = &[
        "ewridge",
        "lasso",
        "kalman",
        "huber",
        "quantile",
        "ftrl",
        "ew_cov",
        "sgd",
        "pa",
        "holt",
        "rls",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "rcov",
        "hmm",
        "corrchange",
        "bocpd",
    ];

    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ewridge",
            ModelKind::Rls { .. } => "rls",
            ModelKind::Lasso { .. } => "lasso",
            ModelKind::Kalman { .. } => "kalman",
            ModelKind::Huber { .. } => "huber",
            ModelKind::Quantile { .. } => "quantile",
            ModelKind::Ftrl { .. } => "ftrl",
            ModelKind::EwCov { .. } => "ew_cov",
            ModelKind::Sgd { .. } => "sgd",
            ModelKind::Pa { .. } => "pa",
            ModelKind::Holt { .. } => "holt",
            ModelKind::KMeans { .. } => "kmeans",
            ModelKind::Micro { .. } => "micro",
            ModelKind::EwClass { .. } => "ew_class",
            ModelKind::SeqTest { .. } => "seqtest",
            ModelKind::Marginal { .. } => "marginal",
            ModelKind::Deco { .. } => "deco",
            ModelKind::Rcov { .. } => "rcov",
            ModelKind::Hmm { .. } => "hmm",
            ModelKind::CorrChange { .. } => "corrchange",
            ModelKind::Bocpd { .. } => "bocpd",
        }
    }

    /// The column a model with no target reads from the targets slot, when
    /// the spec names one: `bocpd`'s `hazard_col` and `hmm`'s `exog_tvtp`.
    /// It fills `targets` ([`Spec::fill_defaults`]) so the column is read
    /// from the targets slot (review 2026-09-12, C20, S24).
    pub fn targets_slot_column(&self) -> Option<&str> {
        match self {
            ModelKind::Bocpd { hazard_col, .. } => hazard_col.as_deref(),
            ModelKind::Hmm { exog_tvtp, .. } => exog_tvtp.as_deref(),
            _ => None,
        }
    }

    /// False for the models that report no coefficients -- no `coef` field
    /// in their output: `ew_cov`, `seqtest`, `marginal`, `rcov`, `corrchange`
    /// and `bocpd`. `tests/test_model_registry.py` holds this to the field
    /// names, which are rendered elsewhere.
    pub fn has_coef(&self) -> bool {
        !matches!(
            self,
            ModelKind::EwCov { .. }
                | ModelKind::SeqTest { .. }
                | ModelKind::Marginal { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
        )
    }

    /// `gram_threads` as the spec gives it, for the two kinds that take it
    /// (docs/PLAN.md task 225); `None` for every other kind.
    pub fn gram_threads_given(&self) -> Option<usize> {
        match self {
            ModelKind::EwRidge { gram_threads, .. } | ModelKind::EwCov { gram_threads, .. } => {
                *gram_threads
            }
            _ => None,
        }
    }

    /// The threads a model's Gram update runs on: the spec's
    /// `gram_threads`, or one. Configuration, not state: the stream sets it
    /// on every model it builds or restores.
    pub fn gram_threads(&self) -> usize {
        self.gram_threads_given().unwrap_or(1)
    }

    /// A windowed kind's `window_size` and `window_budget`; `None` for a kind with
    /// no window to bound.
    pub fn window_parts(&self) -> Option<(Option<f64>, Option<WindowBudgetSpec>)> {
        match self {
            ModelKind::EwRidge {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Lasso {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwCov {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwClass {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Marginal {
                window_size: window,
                window_budget,
                ..
            } => Some((window.as_ref().map(Span::value), *window_budget)),
            _ => None,
        }
    }

    /// The budget a model's window runs under: the spec's, or refuse past
    /// 256 MiB where it names none (`online_core::WindowBudget::DEFAULT`).
    /// `None` without a window.
    pub fn window_budget(&self) -> Option<online_core::WindowBudget> {
        let (window, budget) = self.window_parts()?;
        window.map(|_| {
            budget.map_or(
                online_core::WindowBudget::DEFAULT,
                WindowBudgetSpec::to_core,
            )
        })
    }

    /// A windowed kind's `window_every` and `max_rows_between_snapshots`, as
    /// the spec gives them; `None` for a kind with no window (docs/PLAN.md
    /// task 162).
    pub fn window_cadence(&self) -> Option<(Option<&Span>, Option<u32>)> {
        match self {
            ModelKind::EwRidge {
                window_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Lasso {
                window_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwCov {
                window_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwClass {
                window_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Marginal {
                window_every,
                max_rows_between_snapshots,
                ..
            } => Some((window_every.as_ref(), *max_rows_between_snapshots)),
            _ => None,
        }
    }

    /// A windowed kind's `closed`, as the spec gives it; `None` for a kind
    /// with no window (docs/PLAN.md task 196, N17).
    pub fn window_closed(&self) -> Option<ModelClosed> {
        match self {
            ModelKind::EwRidge { closed, .. }
            | ModelKind::Lasso { closed, .. }
            | ModelKind::EwCov { closed, .. }
            | ModelKind::EwClass { closed, .. }
            | ModelKind::Marginal { closed, .. } => Some(*closed),
            _ => None,
        }
    }

    /// The edge a windowed model's ring keeps ([`online_core::WindowClosed`]):
    /// the spec's `closed`, `Right` by default and for a kind with no window,
    /// which ignores it. A `closed` a model refuses never gets here: it is
    /// refused as the spec is read ([`ModelClosed`]).
    pub fn window_edge(&self) -> online_core::WindowClosed {
        self.window_closed().map(model_edge).unwrap_or_default()
    }

    /// Every `max_rows_between_*` the model takes, as given, beside the
    /// clock parameter it caps: what [`Spec::validate`] holds to `>= 1`
    /// (docs/PLAN.md task 196, U7). The spec's own `max_rows_between_coefs`
    /// is checked there.
    pub fn row_caps(&self) -> Vec<(&'static str, Option<u32>, &'static str)> {
        let mut caps = Vec::new();
        match self {
            ModelKind::EwRidge {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Lasso {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Huber {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Quantile {
                max_rows_between_solves,
                ..
            } => caps.push((
                "max_rows_between_solves",
                *max_rows_between_solves,
                "solve_every",
            )),
            ModelKind::EwCov {
                max_rows_between_pca,
                ..
            } => caps.push(("max_rows_between_pca", *max_rows_between_pca, "pca_every")),
            ModelKind::Micro {
                max_rows_between_prunes,
                ..
            } => caps.push((
                "max_rows_between_prunes",
                *max_rows_between_prunes,
                "prune_every",
            )),
            _ => {}
        }
        if let Some((_, rows)) = self.window_cadence() {
            caps.push(("max_rows_between_snapshots", rows, "window_every"));
        }
        caps
    }

    /// The cadence a windowed model's ring takes its snapshots on, as the
    /// model maps its configuration ([`online_core::Cadence::of`]): every
    /// row unless given (docs/PLAN.md task 162).
    pub fn snapshot_cadence(&self) -> Option<online_core::Cadence> {
        let (every, rows) = self.window_cadence()?;
        Some(online_core::Cadence::of(
            every.map(Span::value),
            rows.map(|r| r as usize),
        ))
    }

    /// The `window_size` and snapshot cadence of a windowed model that
    /// predicts a target: what the stream cuts its residual spread with
    /// (review 2026-09-12, S1), the model's own cadence, so the spread's
    /// boundary is the fit's. The other windowed models predict none, so
    /// the stream keeps no spread for them.
    pub fn window_and_cadence(&self) -> Option<(f64, online_core::Cadence)> {
        match self {
            ModelKind::EwRidge {
                window_size: Some(w),
                ..
            }
            | ModelKind::Lasso {
                window_size: Some(w),
                ..
            } => Some((w.value(), self.snapshot_cadence()?)),
            _ => None,
        }
    }

    /// True for the models that learn from no target column: `ew_cov`,
    /// `kmeans`, `micro`, `deco`, `rcov`, `hmm`, `corrchange` and `bocpd`,
    /// the list `_spec.py`'s `UNSUPERVISED` keeps too. Their `targets` mirror
    /// `features[0]` for plumbing, so a target that is also a feature is not
    /// a leak for them -- except the column [`Self::targets_slot_column`]
    /// names, which is read as a target is.
    pub fn is_unsupervised(&self) -> bool {
        matches!(
            self,
            ModelKind::EwCov { .. }
                | ModelKind::KMeans { .. }
                | ModelKind::Micro { .. }
                | ModelKind::Deco { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::Hmm { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
        )
    }

    /// True for the models that predict no target as a number: the
    /// unsupervised ones, `ew_class`, whose target is a label it
    /// classifies, `seqtest`, whose targets are the signs it tests, and
    /// `marginal`, whose targets are the columns it correlates the features
    /// with. Their outputs are statistics, assignments, posteriors or
    /// e-values read from the state *before* each row (or, for `marginal`,
    /// nothing at all), and nothing residual-based (`sigma`, `zscore`,
    /// metrics, quantiles, conformal, autocorrelation, drift, selection,
    /// averaging) applies to them; their slots are whatever rides in
    /// `pred`, not targets × combos.
    pub fn predicts_no_target(&self) -> bool {
        self.is_unsupervised()
            || matches!(
                self,
                ModelKind::EwClass { .. } | ModelKind::SeqTest { .. } | ModelKind::Marginal { .. }
            )
    }

    /// The two specs a `seqtest` compares, when it compares rather than
    /// tests columns. Such a spec's targets name residual fields, read from
    /// the two specs' own output in the bank, not columns of the frame; the
    /// bank runs it after them.
    pub fn compares(&self) -> Option<Compare<'_>> {
        match self {
            ModelKind::SeqTest {
                a: Some(a),
                b: Some(b),
                a_suffix,
                b_suffix,
            } => Some(Compare {
                a,
                b,
                a_suffix: a_suffix.as_deref().unwrap_or(""),
                b_suffix: b_suffix.as_deref().unwrap_or(""),
            }),
            _ => None,
        }
    }
}

/// Every parameter measured in clock units, by where it sits: `"*"` for the
/// spec's own, otherwise the model's `type`. Each is a [`Span`] or a
/// [`SpanList`], so each takes a duration under a temporal clock
/// (docs/PLAN.md task 88), and [`Spec::clock_spans`] reads them from here,
/// so a field listed here is one the spec's consistency checks see.
///
/// `clock_fields_are_exactly_the_fields_that_take_a_duration` walks every
/// field serde knows and holds this table to the types: a clock-unit field
/// left off it, or a field on it that refuses a duration, fails the test.
pub const CLOCK_FIELDS: &[(&str, &[&str])] = &[
    (
        "*",
        &[
            "half_life",
            "gap_cap",
            "restart_after_step_back",
            "session_gap",
            "embargo",
            "drift_threshold",
            "coef_every",
        ],
    ),
    (
        "ewridge",
        &[
            "long_half_life",
            "solve_every",
            "window_size",
            "window_every",
        ],
    ),
    (
        "lasso",
        &[
            "select_half_life",
            "solve_every",
            "window_size",
            "window_every",
        ],
    ),
    ("kalman", &["coef_half_life", "revert_half_life"]),
    ("huber", &["solve_every"]),
    ("quantile", &["solve_every"]),
    ("ew_cov", &["pca_every", "window_size", "window_every"]),
    ("holt", &["trend_half_life"]),
    ("micro", &["prune_every"]),
    ("ew_class", &["window_size", "window_every"]),
    ("marginal", &["window_size", "window_every"]),
];

/// The parameters that take a duration on a temporal clock or a number that
/// binds to no clock unit, each meaning something of its own: `bocpd`'s
/// `hazard`, a number of rows between changepoints on any spec, or a
/// duration, the expected time between them (docs/PLAN.md task 179). A
/// number here is legal beside durations, so these are not
/// [`CLOCK_FIELDS`]; [`Spec::clock_spans`] reads one only when it is a
/// duration, which then meets every rule a clock parameter does.
/// `clock_fields_are_exactly_the_fields_that_take_a_duration` holds the two
/// tables together to the types, and `a_duration_or_a_count_is_read_only_
/// as_a_duration` holds this one to `clock_spans`.
pub const DURATION_OR_UNIT_FREE_FIELDS: &[(&str, &[&str])] = &[("bocpd", &["hazard"])];

/// The parameters that are a rate *per* clock unit: a decay factor per unit
/// (`lam`) and a variance per unit (`kalman`'s `q`). A rate has no duration
/// form, so a temporal clock refuses one; the half-life it stands in for
/// takes a duration instead.
pub const CLOCK_RATES: &[(&str, &[&str])] = &[("*", &["lam"]), ("kalman", &["q"])];

/// What a spec's clock-unit parameters are written in, which decides the
/// clock column it can read (docs/PLAN.md task 88).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockScale {
    /// No parameter is tied to a unit: each is `0`, `inf` or left out. A
    /// numeric clock and a temporal one both read the spec.
    Free,
    /// Plain numbers of clock units, which only a numeric clock (or none)
    /// can give a meaning. Names the first such parameter.
    Numbers(&'static str),
    /// Durations, which only a temporal clock can measure. Names the first.
    Durations(&'static str),
}

/// A clock policy in a caller's own words: a spec's, or `po.stream.with_windows`'
/// (docs/PLAN.md task 78: one set of clock rules, stated once).
pub(crate) struct ClockPolicy<'a> {
    /// How a message names the caller: `spec "name"`, or `windows`.
    pub who: &'a str,
    pub clock: Option<&'a str>,
    pub gap_cap: Option<&'a Span>,
    pub restart_after_step_back: Option<&'a Span>,
    pub session: Option<&'a str>,
    pub session_gap: Option<&'a SessionGapSpec>,
    /// For a spec, whether it closes its groups on a session change, which
    /// stands in for `session_gap`; `None` for a caller with no model, whose
    /// messages speak of its windows.
    pub spec_closes_on_session: Option<bool>,
    /// The group column: each group's windows close on its own clock.
    pub group: Option<&'a str>,
    /// Whether a window looks ahead: a `with_windows` formula's forward
    /// operator, or a spec's formula target, which always holds one.
    pub looks_ahead: bool,
}

/// The [`ClockCfg`] a clock policy asks for, or why it cannot run.
pub(crate) fn clock_cfg_of(p: &ClockPolicy<'_>) -> Result<ClockCfg, String> {
    let who = p.who;
    let model = p.spec_closes_on_session.is_some();
    if p.clock.is_some() && p.gap_cap.is_none() {
        return Err(format!("{who}: gap_cap is required when clock is given"));
    }
    // The cap is on the clock's step. Without a clock every row is one step,
    // and a cap was taken and applied to it: `gap_cap = 0.5` halved every
    // decay step, marked every row capped and cleared every lag at every row
    // (task 160, PB2). Refused, as `restart_after_step_back` is.
    if p.clock.is_none() && p.gap_cap.is_some() {
        return Err(format!(
            "{who}: gap_cap needs clock; it caps the step from one clock value to the \
             next, and without a clock every row is one step"
        ));
    }
    // A window looking ahead closes when a row past its far edge arrives on
    // its group's clock, and on a clock a group silent past `gap_cap` is cut.
    // Without a clock column a group's clock counts only its own rows and
    // there is no cap, so a silent group left its windows open, and
    // `with_windows`, which returns rows in input order, held every later
    // row of every group behind them to the end of the input: 500,001 rows
    // and 117 MiB more over 2M rows, where the docs promise about one
    // window (task 173, PC1). Refused, by `with_windows` and by a spec with
    // a window target alike, the user's call: one rule, one message.
    if p.looks_ahead && p.group.is_some() && p.clock.is_none() {
        return Err(format!(
            "{who}: a window looking ahead with group needs a clock column. Without one a \
             group's clock counts only its own rows, so a group that falls silent leaves its \
             windows open, and every row waiting on them waits for as long as it is silent, \
             with nothing to cut them: gap_cap, which cuts a silent group, needs a clock. \
             Name a clock column, with gap_cap, or leave out group"
        ));
    }
    // The cap is a cap on a step and nothing else (task 120, decided
    // 2026-09-28). A negative one clipped every delta to it, so the decay
    // *grew* (`n_eff` ran to 6e7 on 50 rows with `gap_cap = -5`); NaN
    // poisons the clock. Zero froze the clock, and every positive gap
    // then read as a break, so `embargo` released each held label on
    // the next row (since task 153 a break releases nothing early): no
    // decay is `half_life = "inf"`, or no clock. Infinity
    // took the break away, and handed a model an infinite step at a
    // session change.
    if let Some(m) = p.gap_cap {
        let v = m.value();
        if v == 0.0 {
            return Err(format!(
                "{who}: gap_cap must be > 0; a cap of 0 froze the clock. For \
                     no decay set half_life = \"inf\", or leave out the clock to count rows"
            ));
        }
        if v == f64::INFINITY {
            return Err(if model {
                format!(
                    "{who}: gap_cap must be finite; it is the longest step a model \
                     decays across, and a longer gap is a break. Give a cap longer than \
                     any gap the stream should decay across in full"
                )
            } else {
                format!(
                    "{who}: gap_cap must be finite; it is the longest step a window \
                     spans, and a longer gap ends every window open across it. Give a cap \
                     longer than any gap a window should span"
                )
            });
        }
        if !(v.is_finite() && positive(v)) {
            return Err(format!("{who}: gap_cap must be a finite number > 0"));
        }
    }
    let session_gap = match p.session_gap {
        None => None,
        Some(SessionGapSpec::Gap(g)) if g.value() == f64::INFINITY => {
            return Err(format!(
                "{who}: session_gap must be finite; to start over at a session \
                     change set session_gap = \"reset\""
            ));
        }
        Some(SessionGapSpec::Gap(g)) if !(g.value().is_finite() && non_negative(g.value())) => {
            return Err(format!("{who}: session_gap must be >= 0 or \"reset\""));
        }
        Some(SessionGapSpec::Gap(g)) => Some(SessionGap::Gap(g.value())),
        Some(SessionGapSpec::Word(w)) if w == "reset" => Some(SessionGap::Reset),
        Some(SessionGapSpec::Word(w)) => {
            return Err(format!(
                "{who}: session_gap must be a number, a duration or \"reset\", got {w:?}"
            ));
        }
    };
    // `group_close = "session"` is itself the prescription for a session
    // change -- emit the span and start over -- so it takes the place of
    // `session_gap` rather than sitting beside it (E54); `validate`
    // refuses the pair.
    if p.session.is_some() && session_gap.is_none() && p.spec_closes_on_session != Some(true) {
        return Err(if model {
            format!(
                "{who}: session_gap is required when session is given (or \
                 group_close = \"session\", which closes the group at the change instead)"
            )
        } else {
            format!("{who}: session_gap is required when session is given")
        });
    }
    // What a late row is belongs to the caller: unset, every step back is
    // refused; given, a step back no larger than it is a late row and is
    // refused, a larger one starts over. It has no default from the cap
    // (task 120; one name since task 144). `validate` names a missing
    // clock first.
    if p.restart_after_step_back
        .is_some_and(|v| !(v.value().is_finite() && non_negative(v.value())))
    {
        return Err(format!(
            "{who}: restart_after_step_back must be finite and >= 0 (0 starts {} over \
                 at every step back)",
            if model { "the model" } else { "every window" }
        ));
    }
    Ok(ClockCfg {
        // A row-count clock steps by one row and has no cap.
        gap_cap: p.gap_cap.map_or(f64::INFINITY, Span::value),
        on_clock_reset: if p.restart_after_step_back.is_some() {
            OnClockReset::ResetState
        } else {
            OnClockReset::Error
        },
        session_gap,
        min_backwards_jump: p.restart_after_step_back.map_or(0.0, Span::value),
    })
}

impl Spec {
    /// Every clock-unit value this spec sets, with its parameter's name, in
    /// [`CLOCK_FIELDS`] order. A `half_life` grid gives one entry per value.
    /// `every_clock_field_is_walked` holds this to the table.
    pub fn clock_spans(&self) -> Vec<(&'static str, Span)> {
        fn put(out: &mut Vec<(&'static str, Span)>, name: &'static str, s: Option<&Span>) {
            out.extend(s.map(|s| (name, s.clone())));
        }
        fn put_list(out: &mut Vec<(&'static str, Span)>, name: &'static str, l: Option<&SpanList>) {
            if let Some(l) = l {
                out.extend(l.spans().iter().map(|s| (name, s.clone())));
            }
        }
        let mut out = Vec::new();
        put_list(&mut out, "half_life", self.half_life.as_ref());
        put(&mut out, "gap_cap", self.gap_cap.as_ref());
        put(
            &mut out,
            "restart_after_step_back",
            self.restart_after_step_back.as_ref(),
        );
        if let Some(SessionGapSpec::Gap(g)) = &self.session_gap {
            put(&mut out, "session_gap", Some(g));
        }
        put(&mut out, "embargo", self.embargo.as_ref());
        put(&mut out, "drift_threshold", self.drift_threshold.as_ref());
        put(&mut out, "coef_every", self.coef_every.as_ref());
        // A formula target's operators measure in the same clock (review
        // R1, D7): with them here a number beside durations is refused at
        // the spec, and the embargo check compares like units.
        for t in self.targets.defs() {
            if let Some(tree) = &t.formula {
                for op in tree.operators() {
                    put(&mut out, "half_life", op.half_life.as_ref());
                    put(&mut out, "window_size", op.window_size.as_ref());
                }
            }
        }
        match &self.model {
            ModelKind::EwRidge {
                long_half_life,
                solve_every,
                window_size: window,
                window_every,
                ..
            } => {
                put(&mut out, "long_half_life", long_half_life.as_ref());
                put(&mut out, "solve_every", solve_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "window_every", window_every.as_ref());
            }
            ModelKind::Lasso {
                select_half_life,
                solve_every,
                window_size: window,
                window_every,
                ..
            } => {
                put(&mut out, "select_half_life", select_half_life.as_ref());
                put(&mut out, "solve_every", solve_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "window_every", window_every.as_ref());
            }
            ModelKind::Kalman {
                coef_half_life,
                revert_half_life,
                ..
            } => {
                put_list(&mut out, "coef_half_life", coef_half_life.as_ref());
                put_list(&mut out, "revert_half_life", revert_half_life.as_ref());
            }
            ModelKind::Huber { solve_every, .. } | ModelKind::Quantile { solve_every, .. } => {
                put(&mut out, "solve_every", solve_every.as_ref());
            }
            ModelKind::EwCov {
                pca_every,
                window_size: window,
                window_every,
                ..
            } => {
                put(&mut out, "pca_every", pca_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "window_every", window_every.as_ref());
            }
            ModelKind::EwClass {
                window_size: window,
                window_every,
                ..
            }
            | ModelKind::Marginal {
                window_size: window,
                window_every,
                ..
            } => {
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "window_every", window_every.as_ref());
            }
            ModelKind::Holt {
                trend_half_life, ..
            } => {
                put(&mut out, "trend_half_life", trend_half_life.as_ref());
            }
            ModelKind::Micro { prune_every, .. } => {
                put(&mut out, "prune_every", prune_every.as_ref());
            }
            // A number of rows binds to no clock unit
            // ([`DURATION_OR_UNIT_FREE_FIELDS`]).
            ModelKind::Bocpd {
                hazard: Some(h), ..
            } if h.is_duration() => {
                put(&mut out, "hazard", Some(h));
            }
            _ => {}
        }
        out
    }

    /// The first rate per clock unit the spec sets to something a unit
    /// changes: `lam` other than 1, or a nonzero `q`.
    fn clock_rate(&self) -> Option<&'static str> {
        if self.lam.is_some_and(|l| l != 1.0) {
            return Some("lam");
        }
        match &self.model {
            ModelKind::Kalman { q: Some(q), .. } if q.iter().any(|v| v.0 != 0.0) => Some("q"),
            _ => None,
        }
    }

    /// Whether the spec's clock parameters are numbers, durations or neither,
    /// refusing a spec that mixes the two: a half-life of `"10m"` beside a
    /// `gap_cap` of `300` says nothing about what the `300` is in.
    pub fn clock_scale(&self) -> Result<ClockScale, String> {
        let spans = self.clock_spans();
        let duration = spans.iter().find(|(_, s)| s.is_duration()).map(|(f, _)| *f);
        // A rate first: it has no duration form, and its message names the
        // parameter to give instead. With a finite cap required beside it
        // (task 120), the cap would otherwise always be the number named.
        let number = self.clock_rate().or_else(|| {
            spans
                .iter()
                .find(|(_, s)| s.is_unit_bound_number())
                .map(|(f, _)| *f)
        });
        match (duration, number) {
            (Some(d), Some("lam")) => Err(format!(
                "spec {:?}: {d} is a duration, and lam is a decay per clock unit, which has no \
                 duration form; give half_life as a duration instead",
                self.name
            )),
            (Some(d), Some("q")) => Err(format!(
                "spec {:?}: {d} is a duration, and q is the noise a row one clock unit after \
                 the last adds (q times the step squared for a longer one), which has no \
                 duration form; leave q out and give coef_half_life as a duration, which \
                 derives it",
                self.name
            )),
            (Some(d), Some(n)) if d == n => Err(format!(
                "spec {:?}: {d} mixes durations and plain numbers, and a number says nothing \
                 about its unit; give every value the same way -- durations for a Datetime, \
                 Date or Duration clock, numbers for a numeric one",
                self.name
            )),
            (Some(d), Some(n)) => Err(format!(
                "spec {:?}: {d} is a duration but {n} is a plain number, and a number says \
                 nothing about its unit; give every clock parameter the same way -- durations \
                 for a Datetime, Date or Duration clock, numbers for a numeric one",
                self.name
            )),
            (Some(d), None) if self.clock.is_none() => Err(format!(
                "spec {:?}: {d} is a duration, which needs a clock column to measure it; with \
                 no clock a row is one unit, so give it as a number of rows",
                self.name
            )),
            (Some(d), None) => Ok(ClockScale::Durations(d)),
            (None, Some(n)) => Ok(ClockScale::Numbers(n)),
            (None, None) => Ok(ClockScale::Free),
        }
    }
}

/// The parameters task 144 renamed (docs/PLAN.md): a spec dict, a TOML file
/// or a windows config naming the old one is refused naming the new one, in
/// place of a bare "unknown field". No alias.
pub const RENAMED: &[(&str, &str)] = &[
    ("halflife", "half_life"),
    ("long_halflife", "long_half_life"),
    ("coef_halflife", "coef_half_life"),
    ("revert_halflife", "revert_half_life"),
    ("select_halflife", "select_half_life"),
    // Straight to the spec's `half_life`: `level_half_life` is refused too
    // (task 196).
    ("level_halflife", "half_life"),
    ("trend_halflife", "trend_half_life"),
    ("label_delay", "embargo"),
    ("max_dclock", "gap_cap"),
    ("window", "window_size"),
    ("min_periods", "min_weight"),
    ("emit_resid_z", "emit_zscore"),
    ("scale_features", "standardize"),
    ("on_clock_reset", "restart_after_step_back"),
    ("min_backwards_jump", "restart_after_step_back"),
    ("ridge_decay", "ridge_scale"),
    ("add_intercept", "fit_intercept"),
    ("max_cd_iters", "max_iter"),
    ("cd_tol", "tol"),
    ("reset", "reset_on_flag"),
    // Task 196 (docs/PLAN.md §18, N14): a count of rows says so, so that a
    // clock form can come later under the plain name.
    ("update_every", "update_every_rows"),
    ("split_merge_every", "split_merge_every_rows"),
    ("permute_every", "permute_every_rows"),
    // N16: holt's level takes the spec's `half_life`, one knob under one
    // name.
    ("level_half_life", "half_life"),
];

/// The parameters renamed in one model whose old name another model keeps:
/// `rls`'s `ridge` is `delta` (docs/PLAN.md task 195, N11), and `ridge` is
/// still `ewridge`'s, `huber`'s and `quantile`'s. Named only where the
/// refused field's model takes the new name -- where serde's list of the
/// fields it expected holds it -- so a model with neither is told nothing
/// about another's rename.
pub const RENAMED_WHERE_EXPECTED: &[(&str, &str)] = &[("ridge", "delta")];

/// The values task 196 renamed (docs/PLAN.md §18): the model `type` tag
/// (N1, the builder's, the README's and the core's spelling) and an
/// `ew_cov` statistic (N10, beside `partial_corr`). A spec naming the old
/// one is refused naming the new one, as [`RENAMED`] does for a key.
pub const RENAMED_VALUES: &[(&str, &str)] = &[("ew_ridge", "ewridge"), ("lagcorr", "lag_corr")];
/// The parameters renamed after 1.0, `(old, new)`: a spec dict, a TOML file
/// or a windows config naming the old one is read as naming the new one,
/// with a deprecation notice ([`deprecation_notice`]; in Python a
/// `polars_online.PolarsOnlineDeprecationWarning`, on the command line a
/// line on stderr), until the next major version removes the entry and the
/// name is refused through [`RENAMED`] (docs/PLAN.md task 198; review round
/// 4, D2). Empty: every rename so far was made before 1.0, and stays
/// refused by name, task 144's rule. A rename after 1.0 goes here, beside
/// its twin in `python/polars_online/_warnings.py`'s `_DEPRECATED`.
pub const DEPRECATED: &[(&str, &str)] = &[];

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
    forward_deprecated_with(v, DEPRECATED)
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

/// `msg`, a deserialization error, with the rename named when the field it
/// refuses as unknown is an old name, or the model `type` it refuses as an
/// unknown variant is an old tag.
pub fn name_renamed(msg: &str) -> String {
    // No citation: a wheel's user has no docs/PLAN.md (review 2026-10-06,
    // PC11).
    for (old, new) in RENAMED {
        if msg.contains(&format!("unknown field `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_WHERE_EXPECTED {
        if msg.contains(&format!("unknown field `{old}`")) && msg.contains(&format!("`{new}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_VALUES {
        if msg.contains(&format!("unknown variant `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    msg.to_string()
}

/// One model spec: common parameters (docs/PLAN.md §3) + the model. An
/// unknown key is refused, naming the keys there are (see [`ModelKind`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Output struct column name.
    pub name: String,
    pub model: ModelKind,
    /// Columns to learn against, each a name, a table naming a column
    /// under another name, or a table holding a formula of the row's future
    /// ([`Targets`]); reads as the names. Optional for a model that learns from no
    /// target (`ModelKind::is_unsupervised`), where
    /// [`Self::fill_defaults`] mirrors `features[0]` the way the Python
    /// builders do (docs/ENHANCEMENTS.md E53); required otherwise.
    #[serde(default)]
    pub targets: Targets,
    pub features: Vec<String>,
    #[serde(default = "default_true")]
    pub fit_intercept: bool,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub half_life: Option<SpanList>,
    #[serde(default)]
    pub lam: Option<f64>,
    /// Ceiling on the clock delta, in clock units: finite and positive,
    /// required with a clock and refused without one (task 160, PB2), where
    /// every row is one step. It caps the step between two rows a model
    /// learns from, so a run of skipped rows hands the row after it at most
    /// this much clock (review 2026-09-12, S3); a step the ceiling cut is a
    /// break, which clears `ew_cov`'s and `marginal`'s lagged co-moments,
    /// adjacency being broken -- under `embargo`, when the row after the
    /// break is learned (docs/PLAN.md task 153).
    #[serde(default)]
    pub gap_cap: Option<Span>,
    /// The clock stepping back within a group. Unset, every step back is
    /// refused, naming the row, the bank untouched. Given, in clock units:
    /// a step back no larger than this is a late row and is refused the
    /// same way; a larger one starts the model over. `0` starts over at
    /// every step back. Finite and `>= 0`; needs `clock` (task 120, decided
    /// 2026-09-28; one name since task 144).
    #[serde(default)]
    pub restart_after_step_back: Option<Span>,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    /// Warmup in `n_eff` units. A scalar applies to every target; a list gives
    /// one threshold per target, in `targets` order (ENHANCEMENTS E7) — a
    /// 5-minute-ahead target and a 1-day-ahead target rarely deserve the same
    /// warmup. Warmup gates *output*, not learning: the model still updates
    /// from rows whose predictions are withheld. Each target's threshold is
    /// checked against that target's own weight -- the rows it was present
    /// on, inside the window under one -- where the model keeps one
    /// (`ewridge`, `lasso`, `kalman`, `huber`, `quantile`, `holt`, and since
    /// task 158 `sgd`, `pa`, `ftrl` and `rls`), and against the shared
    /// `n_eff` otherwise; the emitted `n_eff` is the shared weight either way
    /// (review 2026-09-12, S2).
    pub min_weight: Option<FloatOrList>,
    /// Withhold predictions until the decay window has filled this far
    /// toward steady state: `settled_frac = 1 − 2^(−T/h)`, `T` the decay
    /// time the models have seen (docs/WARMUP-AND-CONVERGENCE.md §2). A
    /// fraction in `[0, 1)`, so it cannot be set unreachably; `0`, the
    /// default, is off. Off because under a stationary process the
    /// mean-form fit is unbiased from its first row and its variance is
    /// `max_error_inflation`'s to gate; what this guards is a history that
    /// does not represent the process -- regimes or seasons the half-life
    /// was chosen to average across -- which only the user can judge. Needs
    /// a decay to settle toward.
    #[serde(default)]
    pub min_settled_frac: Option<Num>,
    /// Withhold predictions while the estimation error is expected to
    /// inflate the prediction error over the noise floor by more than this:
    /// `error_inflation = sqrt(1 + estimation variance / noise)` (§2.1).
    /// `ewridge`: `sqrt(1 + edf / n_kish)`, the effective degrees of freedom
    /// the last solve used over Kish's effective sample size behind the fit,
    /// default `sqrt(2)`, the estimation variance no larger than the noise
    /// being fitted. Off (`inf`) unless set on the others (docs/PLAN.md task
    /// 116): `rls`, `sqrt(1 + k_total / n_kish)`, its edf's bound under the
    /// fading prior; `lasso`, `sqrt(1 + df / n_kish)` per path point, `df`
    /// the active count plus the intercept; `kalman`, per row and exact,
    /// `sqrt(1 + z' P⁻ z / R)`, the prior predictive variance over the noise.
    /// Tracks the model, so adding a feature moves the gate with it, and
    /// reads Kish's `n` rather than the weight, so uneven weights withhold
    /// for longer. A ratio above 1; `inf` is off. Refused on every other
    /// model, which is left to `min_weight`.
    #[serde(default)]
    pub max_error_inflation: Option<Num>,
    /// Emit `error_inflation_<slot>`: the same ratio for *this* row's
    /// features, `sqrt(1 + h(x))`, so a row leaning on a direction the data
    /// never showed reads large where the stream average cannot see it.
    /// `ewridge`: `h = x' Σ̂⁻¹ x / n_kish`, the row's leverage against the
    /// factor its fit came from; `rls`: `‖R⁻ᵀ z‖² s₂ / s₁`, the same in sum
    /// form against its kept factor; `kalman`: `z' P⁻ z / R`, the gate's own
    /// value. One triangular solve or quadratic form a row, `O(k²)`, and
    /// `ewridge` keeps its factors -- which is why it is opt-in. Refused
    /// elsewhere, `lasso` included: it keeps no factor.
    #[serde(default)]
    pub emit_error_inflation: bool,
    /// Emit `se_coef`: each coefficient's standard error, on `coef`'s rows
    /// and laid out like it, in `coef`'s own units, the intercept's included
    /// -- `T Cov Tᵀ`'s diagonal, `T` the map `coef` is read out by
    /// (docs/PLAN.md task 116, F). `ewridge`: `Cov = σ̂² M`,
    /// `M = Σ̂⁻¹ / n_kish` (WARMUP §7.1), which leaves out the ridge's
    /// sandwich and so errs large; `rls`: `Cov = σ̂² (s₂ / s₁) A⁻¹`, the same
    /// in sum form, `O(k³)` on the `coef` schedule; `kalman`: `Cov = P`, its
    /// posterior, exact. `σ̂` is the slot's EW out-of-sample residual std as
    /// the row reads it (`sigma`), which carries the estimation error too,
    /// `sqrt(1 + h)` larger than the noise in warm-up, so the error errs
    /// large there; null before there is one. A report, not a gate. Refused
    /// for `lasso` (post-selection), `huber` and `quantile` (an
    /// M-estimator's covariance is a sandwich) and the gradient models (no
    /// second moment).
    #[serde(default)]
    pub emit_se_coef: bool,
    /// Emit `scored_clock` and `learned_clock` on every scored row: the row's
    /// own clock, and the clock of the newest row the models had learned
    /// from, at a positive weight, when the row was scored (docs/PLAN.md
    /// task 152). In the clock column's own type; the group's row index
    /// with no clock. One pair per spec, since every instance learns the
    /// same rows at the same time. Default false.
    #[serde(default)]
    pub emit_clocks: bool,
    /// When the `coef` field is filled, as `solve_every` schedules a solve
    /// (docs/PLAN.md task 178): a `coef` row once the clock has moved this
    /// far since the group's last one -- a number of the clock column's
    /// units, or a duration on a temporal clock, the clock measured as the
    /// models are stepped on it, a gap capped at `gap_cap` counting as the
    /// cap -- and `0` every row. With `max_rows_between_coefs` too,
    /// whichever comes first. Without a clock column the clock is the row's
    /// number, the first row being 1, so `N` is every `N` rows. With neither
    /// given, **each group's** last row within every chunk: one row per
    /// group per chunk, the one schedule that follows the chunking, while
    /// every other field, and `coef` under a cadence, is chunk-invariant.
    #[serde(default)]
    pub coef_every: Option<Span>,
    /// At most this many accepted rows between `coef` rows, rows of weight
    /// zero and rows with a null target included, as `coef_every` counted
    /// them before it read the clock; at least 1 (task 178).
    #[serde(default)]
    pub max_rows_between_coefs: Option<u32>,
    /// Emit `sigma_<slot>`: the EW standard deviation of this slot's
    /// out-of-sample residuals, read from the state *before* each row. Off by
    /// default because it widens the output struct. Its weight ages on every
    /// row, a row with no prediction or with weight 0 included (review
    /// 2026-09-12, N6), and under a `window_size` it is the window's, as the fit
    /// is (S1).
    #[serde(default)]
    pub emit_sigma: bool,
    /// Emit `zscore_<slot>` = `resid / sigma`: how surprising this row was, in
    /// units of the model's own recent error. Off by default.
    #[serde(default)]
    pub emit_zscore: bool,
    /// Emit `ic_<slot>`, `r2_<slot>` and `hit_rate_<slot>`: exponentially
    /// weighted evaluation metrics kept beside the model (ENHANCEMENTS E22).
    /// `polars_online.eval` computes the same things in Polars over collected
    /// output, which needs the whole frame; this is the O(state) version, so a
    /// long-running stream or the CLI can report how the fit is doing without
    /// keeping the rows.
    ///
    /// On a `sgd` or `ftrl` fit with `loss = "logistic"`, `pred` is a
    /// probability and `y` a 0/1 label rather than a signed regression
    /// target, and two of the three read differently (docs/PLAN.md task
    /// 76): `hit_rate` is accuracy at a 0.5 threshold instead of sign
    /// agreement (the sign test always agrees on two positive numbers, and
    /// read 1.0 for every such fit before this), `r2` is the Brier skill
    /// score against the running base rate, and `ic` is the point-biserial
    /// correlation between the probability and the label -- both under
    /// their usual names, since the formula does not change. There is no
    /// log loss here; `polars_online.eval.metrics(..., binary=True)` adds
    /// it over the collected frame (a streaming version would put a `ln`
    /// result into the state, which `docs/PLAN.md` §11a's B4 rule forbids).
    ///
    /// The sign test is about zero for every target, so a target that sits
    /// about 1, such as a plain ratio of two prices, reads 1.0 whatever the
    /// fit: two positive numbers always agree. A return is better a
    /// difference or a log ratio, which sit about zero.
    #[serde(default)]
    pub emit_metrics: bool,
    /// Emit `lo_<slot>`, `hi_<slot>` and `coverage_<slot>`: an
    /// adaptive conformal interval `pred ± q` at this coverage level, with
    /// the realized coverage beside it (ENHANCEMENTS E36). `q` is a tracked
    /// quantile of `|resid|` — `q ← max(0, q + rate·sigma·(w/w̄)·(miss −
    /// α))`, `miss = 1{|resid| > q}`, `α = 1 − coverage`, `w̄` the EW mean
    /// weight — so the long-run coverage weighted by `w/w̄` is the target
    /// whatever the residual distribution is and however it moves, where
    /// `sigma` gives a Gaussian interval (review round 4, CA5). Read before
    /// the row, like every other diagnostic. A coverage level strictly
    /// between 0 and 1.
    #[serde(default)]
    pub conformal: Option<f64>,
    /// Step of the conformal radius per unit of the slot's `sigma`. Default
    /// 0.05: a miss widens the interval by `0.05·sigma·(1 − α)`, a hit
    /// narrows it by `0.05·sigma·α`, each times the row's weight over the
    /// scored rows' EW mean weight (docs/PLAN.md task 147).
    #[serde(default)]
    pub conformal_rate: Option<f64>,
    /// Emit `abs_resid_q<p>_<slot>` for each level in `resid_quantiles`: the
    /// exponentially weighted quantile of `|resid|` at the model's half-life
    /// and each row's weight (ENHANCEMENTS E23, docs/PLAN.md task 146), from
    /// one decaying sketch per slot within `tanh(1/128)` of the exact one --
    /// a distribution-free interval where `sigma` only gives a Gaussian one.
    #[serde(default)]
    pub resid_quantiles: Option<Vec<f64>>,
    /// Emit `autocorr_<slot>`: EW lag-`resid_autocorr_lag` autocorrelation of
    /// the out-of-sample residuals. A residual stream should look like noise;
    /// autocorrelation is the classic sign that it does not.
    #[serde(default)]
    pub emit_autocorr: bool,
    /// Lag for `emit_autocorr`. Default 1.
    #[serde(default)]
    pub resid_autocorr_lag: Option<usize>,
    /// Emit `drift_<slot>`: a Page-Hinkley detector on each slot's absolute
    /// out-of-sample residual, true on the row where a break is detected.
    /// Under `embargo` a residual reaches the detector when its label is
    /// released, so the flag is on the row whose clock released the label
    /// that tripped it; that label's own row is out already (task 160,
    /// PB1). Complements the half-life: decay forgets smoothly and always,
    /// drift detection notices a break and says so.
    #[serde(default)]
    pub emit_drift: bool,
    /// Change magnitude the drift detector tolerates before accumulating,
    /// in units of the slot's own EW residual std. Default 0.5.
    #[serde(default)]
    pub drift_delta: Option<f64>,
    /// Accumulated excess that counts as drift, in `sigma` times clock units:
    /// the detector sums each row's excess times its clock step. A number of
    /// the clock column's units, or a duration on a temporal clock (`"20m"`
    /// is one `sigma` of excess held for twenty minutes). Default 20 without
    /// a clock column, where a row is one unit and it is the classic test;
    /// required with one, as `gap_cap` is, since no number means the same
    /// evidence on every clock (task 168; review 2026-10-05, CC1).
    #[serde(default)]
    pub drift_threshold: Option<Span>,
    /// What a detection does besides setting the flag: `"flag"` (default) or
    /// `"reset"`, which restarts this stream's models and their residual
    /// diagnostics on the flagged row, before it is scored under `embargo`
    /// and after it is learned from otherwise. Unlike a clock reset it
    /// keeps the rows `embargo` holds: they teach the restarted models as
    /// they are released.
    #[serde(default)]
    pub drift_action: Option<String>,
    /// Emit `pred_<target>__averaged`: an exponentially weighted average of
    /// every slot's prediction, with weights `softmax(−eta · σ²/σ²_best)`,
    /// each slot's EW squared error as a ratio to the best slot's
    /// (ENHANCEMENTS E14; a ratio since the code review's S23, so `eta` does
    /// not depend on the target's units). The soft counterpart of
    /// `emit_selected`: averaging hedges where selection commits, which is
    /// usually the better trade when several slots are close. The weights
    /// come from each slot's EW *mean* squared error, so they stay bounded
    /// and do not sharpen as rows accumulate, as a weighting by summed
    /// losses (river's `EWARegressor`) does; and a slot with no prediction
    /// or no `sigma` on the row is left out of that row's average, which is
    /// null only when every slot is (code review of 2026-09-12, S23).
    #[serde(default)]
    pub emit_averaged: bool,
    /// Sharpness of the averaging weights, `exp(−eta · (σ²/σ²_best − 1))`:
    /// at 1, a slot whose error is twice the best's weighs `e⁻¹` of it. Large
    /// values approach `emit_selected`'s argmin, and `"inf"` is it, a tie
    /// shared; small values approach an equal-weight mean. Default 1.
    #[serde(default)]
    pub average_eta: Option<Num>,
    /// Emit `selected_<target>` and `pred_<target>__selected`: online model
    /// selection across every grid slot for that target (ridge values, feature
    /// sets and half-lives), by lowest EW out-of-sample error. Generalizes the
    /// lasso's `lam_selected`. Requires more than one slot per target.
    #[serde(default)]
    pub emit_selected: bool,
    /// Hold each row back from *learning* until the model's clock has moved
    /// `embargo` further on, in the same units the decay uses
    /// (docs/ENHANCEMENTS.md E47). `None` (the default) learns from a row
    /// where it sits.
    ///
    /// A target that is a forward quantity over `h` clock units is not known
    /// at the row it sits on. Learning it there hands the model `h` of the
    /// future before it predicts the rows in between, and every
    /// "out-of-sample" number after that is contaminated -- with an
    /// autocorrelated feature even a pure noise column then shows a
    /// correlation with its target. With a delay the row is buffered, and
    /// released into the model, the residual, `sigma`, the metrics, drift,
    /// the conformal interval, `n_eff` and `min_weight` only once its label
    /// would really have been known.
    ///
    /// The delay counts the time that passed: the clock column's own steps,
    /// skipped rows' time included, not the capped delta the models decay by
    /// (`gap_cap` and `session_gap` say how much a model forgets across a
    /// break, not how long it lasted), and the session's gap where a session
    /// change restarts the clock (docs/PLAN.md task 153). A row exactly the
    /// delay later releases a row, and the time is held exactly: in integer
    /// nanoseconds on a temporal clock, by one subtraction of the two rows'
    /// raw values on a number clock (task 176). Release depends on the rows
    /// alone, which makes it chunk-invariant. With no `clock`
    /// column that is one unit per row of the group, a skipped row included,
    /// so `embargo = 20` is twenty rows. A reset drops the buffer (the
    /// state it would teach is gone). A break releases nothing early: its
    /// events -- the lag rings' clear, `session_shrink`'s blend -- wait with
    /// the row after it and run when that row is learned, so the models run
    /// one delay behind, events included.
    #[serde(default)]
    pub embargo: Option<Span>,
    /// One state per key.
    #[serde(default)]
    pub group: Option<String>,
    /// Emit a group's accumulators when the stream can prove no further row
    /// will join it, and drop the stream (docs/ENHANCEMENTS.md E54).
    ///
    /// `"monotone"`: the group column is non-decreasing, so a key smaller
    /// than the largest key of the chunk just fed is finished. Requires
    /// `group`, and refuses a chunk whose keys are out of order, naming the
    /// row.
    ///
    /// `"session"`: a group's span ends where its `session` value changes.
    /// Requires `session`; refused with `session_gap`, which is a second
    /// prescription for the same event (the close *is* the reset, with an
    /// emission).
    ///
    /// `None` (the default) keeps every group for the life of the bank,
    /// which is what makes a bank over an unbounded key space grow without
    /// bound. The closed rows queue in the bank and are read with
    /// `closed_groups()`, written to the `closed_groups` sidecar by a run,
    /// or both.
    ///
    /// Refused with `embargo`: a closed group cannot release the rows it
    /// is holding, so its row would differ from the `gram()` a driver reads
    /// at the same point -- which is the one thing the closed row promises.
    #[serde(default)]
    pub group_close: Option<String>,
}

impl Spec {
    /// Fill in what a spec is allowed to leave out
    /// (docs/ENHANCEMENTS.md E53), so that everything downstream sees a spec
    /// with every field set.
    ///
    /// Two rules. `ew_cov`, `kmeans` and `micro` learn from no target,
    /// and their `targets` mirror `features[0]` for the plumbing's sake. The
    /// Python builders write that line; a TOML author should not have to
    /// invent a target for a model that has none, and a spec that left it
    /// out used to be refused with "targets must be non-empty". Filling it
    /// with the same value the builders use is what makes a spec written in
    /// TOML byte-identical to the same spec written in Python -- so a state
    /// saved from one resumes under the other.
    ///
    /// And `drift_action` is spelled out as `"flag"`, which is what it means
    /// when it is absent. It is the one other field the Python builders write
    /// and a TOML author would not, and leaving it unfilled would make the
    /// two surfaces' specs differ in a byte for no reason at all.
    ///
    /// Idempotent, and a no-op for a spec that already says both.
    pub fn fill_defaults(&mut self) {
        if self.targets.is_empty() && self.model.is_unsupervised() {
            // A column the model reads from the slot fills it, as the
            // builders write it. `features[0]` went there, and was then read
            // as the hazard or the exogenous series (review 2026-09-12, C20).
            if let Some(col) = self.model.targets_slot_column() {
                self.targets = vec![col.to_string()].into();
            } else if let Some(first) = self.features.first() {
                self.targets = vec![first.clone()].into();
            }
        }
        if self.drift_action.is_none() {
            self.drift_action = Some("flag".into());
        }
    }

    /// Does this spec close a group when its `session` value changes
    /// (docs/ENHANCEMENTS.md E54)?
    pub fn closes_on_session(&self) -> bool {
        self.group_close.as_deref() == Some("session")
    }

    /// Does this spec close a group once a larger key has been seen?
    pub fn closes_monotone(&self) -> bool {
        self.group_close.as_deref() == Some("monotone")
    }

    pub fn k(&self) -> usize {
        self.features.len()
    }

    pub fn m(&self) -> usize {
        self.targets.len()
    }

    /// Decay values, one per half-life grid entry (one model instance each).
    pub fn decays(&self) -> Result<Vec<(String, Decay)>, String> {
        match (&self.half_life, self.lam) {
            (Some(_), Some(_)) => Err(format!(
                "spec {:?}: half_life and lam are mutually exclusive",
                self.name
            )),
            (None, None) => match &self.model {
                // An e-process does not forget: its validity is the product
                // of every bet made, so there is nothing a decay could apply
                // to. One undecayed instance, and `half_life`/`lam` refused
                // below rather than ignored.
                // A realised covariance is a sum over a block, not a
                // decayed mean: the block boundary is `group_close`'s, and
                // `half_life`/`lam` are refused below rather than ignored.
                ModelKind::SeqTest { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. } => {
                    Ok(vec![(String::new(), Decay::Halflife(f64::INFINITY))])
                }
                _ => Err(format!(
                    "spec {:?}: one of half_life/lam is required",
                    self.name
                )),
            },
            (None, Some(l)) => {
                if !(0.0 < l && l <= 1.0) {
                    return Err(format!(
                        "spec {:?}: lam must be in (0, 1], got {l}",
                        self.name
                    ));
                }
                Ok(vec![(String::new(), Decay::Lam(l))])
            }
            (Some(h), None) => {
                let hs = h.to_vec();
                // An empty grid built no model instance at all, and the
                // output struct had no field (task 160, PB3).
                if hs.is_empty() {
                    return Err(format!(
                        "spec {:?}: half_life names no half-life; give one or a grid",
                        self.name
                    ));
                }
                // `!(h > 0)` rather than `h <= 0` so NaN is refused too: it
                // decays every accumulator to NaN and nothing washes it out.
                if let Some(bad) = h.spans().iter().find(|s| !positive(s.value())) {
                    return Err(format!(
                        "spec {:?}: half_life must be > 0 (\"inf\" for no decay), got {bad}",
                        self.name
                    ));
                }
                if let Some(dup) = first_duplicate(&hs) {
                    // "5m" and "300s" are one half-life written two ways.
                    let label = h
                        .spans()
                        .iter()
                        .rev()
                        .find(|s| s.value().to_bits() == dup.to_bits())
                        .map_or_else(|| num_label(dup), Span::label);
                    return Err(format!(
                        "spec {:?}: half_life lists {} more than once; each value is one \
                         model instance and the two would be the same model",
                        self.name, label
                    ));
                }
                if hs.len() == 1 {
                    Ok(vec![(String::new(), Decay::Halflife(hs[0]))])
                } else {
                    // A grid names each instance by its half-life as written:
                    // `@h600` for a number, `@h10m` for a duration.
                    Ok(h.spans()
                        .iter()
                        .map(|s| (format!("@h{}", s.label()), Decay::Halflife(s.value())))
                        .collect())
                }
            }
        }
    }

    /// `gap_cap` and `session_gap` in integer nanoseconds where they are
    /// durations, which only a temporal clock reads: what a row's stamp, its
    /// decayed clock held exactly, caps a step at ([`ExactCaps`],
    /// docs/PLAN.md task 175).
    pub fn exact_caps(&self) -> ExactCaps {
        let nanos = |s: &Span| match s {
            Span::Duration(d) => Some(d.nanos),
            Span::Units(_) => None,
        };
        ExactCaps {
            gap_cap_ns: self.gap_cap.as_ref().and_then(nanos),
            session_gap_ns: match &self.session_gap {
                Some(SessionGapSpec::Gap(g)) => nanos(g),
                _ => None,
            },
        }
    }

    pub fn clock_cfg(&self) -> Result<ClockCfg, String> {
        clock_cfg_of(&ClockPolicy {
            who: &format!("spec {:?}", self.name),
            clock: self.clock.as_deref(),
            gap_cap: self.gap_cap.as_ref(),
            restart_after_step_back: self.restart_after_step_back.as_ref(),
            session: self.session.as_deref(),
            session_gap: self.session_gap.as_ref(),
            spec_closes_on_session: Some(self.closes_on_session()),
            group: self.group.as_deref(),
            looks_ahead: self.targets.defs().iter().any(|t| {
                t.formula
                    .as_ref()
                    .is_some_and(crate::formula::Node::has_forward_operator)
            }),
        })
    }

    /// Threshold per target, in `targets` order.
    pub fn min_periods_per_target(&self) -> Vec<f64> {
        match &self.min_weight {
            None => vec![self.default_min_periods(); self.m()],
            Some(FloatOrList::Float(v)) => vec![v.0; self.m()],
            Some(FloatOrList::List(v)) => v.iter().map(|n| n.0).collect(),
        }
    }

    fn default_min_periods(&self) -> f64 {
        match self.model {
            // An e-value is valid from the first row (it is 1 before it).
            ModelKind::SeqTest { .. } => 0.0,
            // A pair's statistics are over two columns whatever the feature
            // count: two rows give a correlation of ±1, three the first one
            // with any content.
            ModelKind::Marginal { .. } => 3.0,
            // The standardiser needs a variance per feature before the row
            // can be standardised at all; three rows is where it has one.
            ModelKind::Deco { .. } => 3.0,
            // Nothing is gated: `rcov` reports nothing per row.
            ModelKind::Rcov { .. } => 0.0,
            // The states are seeded from `warm_rows`, which is the real
            // gate; `min_weight` on top of it would be a second one.
            ModelKind::Hmm { .. } => 0.0,
            // The span or the windows are the gate.
            ModelKind::CorrChange { .. } => 0.0,
            // On row one the run-length posterior has no run older than
            // one, so `P(r <= 1)` is 1 whatever the row: one row of
            // warm-up, and the prior is the gate after it.
            ModelKind::Bocpd { .. } => 1.0,
            // The noise gate is this model's readiness gate
            // (docs/WARMUP-AND-CONVERGENCE.md §2.1): `max_error_inflation`
            // tracks the model where a count of weight cannot, and the
            // `k + 1` rows its first solve needs are the model's own floor
            // (`build_one`), not a setting. An explicit value still floors.
            ModelKind::EwRidge { .. } => 0.0,
            // No intercept to count: `fit_intercept` has nothing to act on in
            // these, and counted, it moved the first reported row (review
            // 2026-09-12, S22). `k + 1` is what the builders' default gave.
            ModelKind::EwCov { .. }
            | ModelKind::KMeans { .. }
            | ModelKind::Micro { .. }
            | ModelKind::EwClass { .. }
            | ModelKind::Holt { .. } => (self.k() + 1) as f64,
            _ => (self.k() + usize::from(self.fit_intercept)) as f64,
        }
    }

    /// The threshold the *model* uses: the smallest across targets, so a model
    /// starts predicting as soon as any target is ready. Per-target gating of
    /// the reported values happens in the stream layer.
    pub fn min_periods_or_default(&self) -> f64 {
        self.min_periods_per_target()
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    }

    /// The settled-fraction gate, `0` (off) unless set
    /// (docs/WARMUP-AND-CONVERGENCE.md §4.1.1).
    pub fn min_settled_frac_or_default(&self) -> f64 {
        self.min_settled_frac.map_or(0.0, |v| v.0)
    }

    /// The noise gate: on `ewridge`, `sqrt(2)` unless set -- the estimation
    /// variance no larger than the noise (§2), the gate that replaced its
    /// `k + 1` floor -- and on every other model off, infinite, unless set:
    /// those keep their `min_weight` defaults, and no default moves
    /// (docs/PLAN.md task 116, D8).
    pub fn max_error_inflation_or_default(&self) -> f64 {
        let default = if matches!(self.model, ModelKind::EwRidge { .. }) {
            2f64.sqrt()
        } else {
            f64::INFINITY
        };
        self.max_error_inflation.map_or(default, |v| v.0)
    }

    // The diagnostics' defaults, each read here and nowhere else, so the
    // stream that builds a diagnostic and `crate::defaults`, which the API
    // snapshot pins them through, read one value (review 2026-10-06, AP1,
    // DB4).

    /// The drift detector's tolerance, `0.5` of the slot's residual std
    /// unless set.
    pub fn drift_delta_or_default(&self) -> f64 {
        self.drift_delta.unwrap_or(0.5)
    }

    /// The drift detector's threshold in `sigma` times clock units: `20`
    /// unless set, which only a spec without a clock column can leave it
    /// (task 168).
    pub fn drift_threshold_or_default(&self) -> f64 {
        self.drift_threshold.as_ref().map_or(20.0, Span::value)
    }

    /// The conformal radius' step per unit of `sigma`, `0.05` unless set.
    pub fn conformal_rate_or_default(&self) -> f64 {
        self.conformal_rate.unwrap_or(0.05)
    }

    /// The lag of `emit_autocorr`, `1` unless set.
    pub fn resid_autocorr_lag_or_default(&self) -> usize {
        self.resid_autocorr_lag.unwrap_or(1)
    }

    /// The sharpness of `emit_averaged`'s weights, `1` unless set.
    pub fn average_eta_or_default(&self) -> f64 {
        self.average_eta.map_or(1.0, |n| n.0)
    }

    /// Whether any of this spec's model instances forgets: a finite
    /// half-life, or a `lam` below 1. What `settled_frac` needs to be a
    /// fraction of.
    pub fn has_decay(&self) -> bool {
        self.decays().is_ok_and(|ds| {
            ds.iter().any(|(_, d)| match d {
                Decay::Halflife(h) => h.is_finite(),
                Decay::Lam(l) => *l < 1.0,
            })
        })
    }

    /// The weight-share cadence (docs/PLAN.md task 115 (b)), the default
    /// where `solve_every` is left out under a finite half-life: a solve once
    /// the weight learned since the last reaches `ln 2 / 50` of the weight
    /// the fit holds. `None` with an explicit `solve_every`, which keeps its
    /// clock, and under `lam` or an infinite half-life, which solve every row.
    pub fn solve_share_default(&self, solve_every: Option<&Span>, decay: Decay) -> Option<f64> {
        match (solve_every, decay) {
            (None, Decay::Halflife(h)) if h.is_finite() => Some(online_core::DEFAULT_SOLVE_SHARE),
            _ => None,
        }
    }

    /// The clock cadence a spec without `solve_every` carries: half-life/50, or
    /// 0 (every row) under `lam` or an infinite half-life. Under a finite
    /// half-life the solves go by weight instead ([`Self::solve_share_default`],
    /// task 115 (b)); this value stays in the cfg, where `gram_block_rows`
    /// reads whether there is a cadence at all.
    pub fn solve_every_default(&self, decay: Decay) -> f64 {
        match decay {
            Decay::Halflife(h) if h.is_finite() => h / 50.0,
            _ => 0.0, // lam decay / infinite half_life: solve every row
        }
    }

    /// Everything a spec meets before a bank runs it, in one place: the
    /// defaults it may leave out ([`Self::fill_defaults`]), its own rules
    /// ([`Self::validate`]) and its models' (`build_models`, which builds each
    /// instance and drops it). Every door a spec comes in by -- the bank, a
    /// run config, the builders' check, `output_fields`, `output_index` and
    /// `coef_fields` -- goes through this, so a spec is accepted or refused
    /// alike at each. They met one, two or all three of the steps, by door
    /// (review 2026-09-12, S25).
    pub fn check(&mut self) -> Result<(), String> {
        self.fill_defaults();
        self.validate()?;
        crate::stream::build_models(self)?;
        // Two outputs rendered to one field name, named by their inputs
        // (task 144); the bank's tripwire for a spec built some other way.
        match crate::bank::duplicate_field(self) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        // Clock parameters given as durations and as numbers at once, or
        // durations with no clock to measure them (task 88).
        self.clock_scale()?;
        // The bank's tables put `spec` and `group` columns beside a struct
        // named after the spec (`ModelBank.last_row`), and an empty name names
        // no struct at all (review 2026-09-12, S7).
        if self.name.is_empty() || self.name == "spec" || self.name == "group" {
            return Err(format!(
                "spec {:?}: the name must be non-empty and neither \"spec\" nor \"group\", \
                 which the bank's tables use for their own columns",
                self.name
            ));
        }
        if self.targets.is_empty() {
            // An unsupervised spec is allowed to omit them, and
            // `fill_defaults` mirrors `features[0]` (E53) -- so an empty list
            // here means either a model that needs targets, or one that would
            // have had them filled but names no features either.
            return Err(if self.model.is_unsupervised() {
                format!(
                    "spec {:?}: features must be non-empty ({} learns from no target, so its \
                     targets are filled from features[0])",
                    self.name,
                    self.model.kind_name()
                )
            } else {
                format!("spec {:?}: targets must be non-empty", self.name)
            });
        }
        // An empty name names no column and writes fields called `pred_` and
        // `resid_`; `po.target("")` and an empty table name were refused, a
        // plain `""` was not (review 2026-10-06, YA5).
        if self.targets.defs().iter().any(|t| t.name.is_empty()) {
            return Err(format!(
                "spec {:?}: targets must not contain an empty name; its fields would be called \
                 pred_ and resid_",
                self.name
            ));
        }
        if let Some(gc) = &self.group_close {
            if !["monotone", "session"].contains(&gc.as_str()) {
                return Err(format!(
                    "spec {:?}: group_close must be \"monotone\" or \"session\" (got {gc:?})",
                    self.name
                ));
            }
            if self.group.is_none() {
                return Err(format!(
                    "spec {:?}: group_close needs a group column; without one the bank keeps a \
                     single stream, which is never finished",
                    self.name
                ));
            }
            if self.embargo.is_some() {
                return Err(format!(
                    "spec {:?}: group_close does not work with embargo; a closed group \
                     cannot release the rows it is still holding, so its row would not equal \
                     the gram() read at the same point",
                    self.name
                ));
            }
            if gc == "session" {
                if self.session.is_none() {
                    return Err(format!(
                        "spec {:?}: group_close = \"session\" needs a session column to close on",
                        self.name
                    ));
                }
                if self.session_gap.is_some() {
                    return Err(format!(
                        "spec {:?}: group_close = \"session\" and session_gap are two \
                         prescriptions for one event; the close is the reset, with an emission",
                        self.name
                    ));
                }
            }
        }
        if let Some(d) = self.embargo.as_ref().map(Span::value)
            && (d.is_nan() || !d.is_finite() || d <= 0.0)
        {
            return Err(format!(
                "spec {:?}: embargo must be finite and > 0 (got {d}); leave it out for no \
                     delay",
                self.name
            ));
        }
        // Holt has no features by construction, and neither has seqtest;
        // every other model needs at least one to regress on. Both
        // directions are errors: silently ignoring features passed to Holt
        // would look like they were used.
        if matches!(self.model, ModelKind::Holt { .. }) {
            if !self.features.is_empty() {
                return Err(format!(
                    "spec {:?}: holt takes no features (got {}); it extrapolates the \
                     target's own level and trend. Use a regression model to use features.",
                    self.name,
                    self.features.len()
                ));
            }
        } else if let ModelKind::SeqTest {
            a,
            b,
            a_suffix,
            b_suffix,
        } = &self.model
        {
            if !self.features.is_empty() {
                return Err(format!(
                    "spec {:?}: seqtest takes no features (got {}); it tests the sign of \
                     each target. Put the columns whose sign is tested in targets.",
                    self.name,
                    self.features.len()
                ));
            }
            // The two are one switch: a comparison needs both sides.
            match (a, b) {
                (Some(_), None) | (None, Some(_)) => {
                    return Err(format!(
                        "spec {:?}: seqtest a and b go together (got {}); name both specs \
                         to compare them, or neither to test the sign of the targets",
                        self.name,
                        if a.is_some() { "a only" } else { "b only" }
                    ));
                }
                // One spec against itself is a tie on every row -- unless
                // the suffixes pick two instances of its grid, which is a
                // comparison like any other.
                (Some(a), Some(b)) if a == b && a_suffix == b_suffix => {
                    return Err(format!(
                        "spec {:?}: seqtest a and b are both {a:?} with the same suffix; a \
                         spec against itself is a tie on every row (a_suffix/b_suffix \
                         compare two instances of its grid)",
                        self.name
                    ));
                }
                (Some(a), Some(b)) if a == &self.name || b == &self.name => {
                    return Err(format!(
                        "spec {:?}: seqtest a/b name the spec itself; it has no residuals",
                        self.name
                    ));
                }
                (None, None) if a_suffix.is_some() || b_suffix.is_some() => {
                    return Err(format!(
                        "spec {:?}: seqtest a_suffix/b_suffix pick a side's grid instance; \
                         they need a and b",
                        self.name
                    ));
                }
                _ => {}
            }
            // The stake is a function of the trial counts, so a trial is a
            // row, not a weight, and the wealth is a product that no decay
            // can apply to. Refused rather than ignored, as elsewhere.
            if self.weight.is_some() {
                return Err(format!(
                    "spec {:?}: weight does not apply to seqtest (every learned row is one \
                     trial; null the target to skip a row)",
                    self.name
                ));
            }
            if self.half_life.is_some() || self.lam.is_some() {
                return Err(format!(
                    "spec {:?}: half_life/lam do not apply to seqtest (an e-process does not \
                     forget; use session or restart_after_step_back to restart it)",
                    self.name
                ));
            }
        } else if self.features.is_empty() {
            return Err(format!("spec {:?}: features must be non-empty", self.name));
        }
        // A duplicated column silently splits its coefficient across identical
        // slots on an exactly singular system (the jitter fallback rescues the
        // solve, so nothing else complains), and a duplicated target collides
        // in the output struct. Both are always mistakes.
        for (label, cols) in [("features", &self.features), ("targets", &self.targets)] {
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = cols.iter().find(|c| !seen.insert(c.as_str())) {
                return Err(format!(
                    "spec {:?}: {label} lists {dup:?} more than once",
                    self.name
                ));
            }
        }
        // A target used as a feature reads the *current row's* target to
        // predict that same row: perfect leakage, measured as corr(pred, y)
        // = 1.0. Hard rule 2 (out-of-sample by construction) protects the
        // target as target; this is the other door, and it must be locked --
        // it is exactly the accident a long column list invites, and the
        // resulting backtest looks wonderful right up until deployment.
        //
        // The unsupervised models are exempt *by design*, not oversight: they
        // predict no target. Their "targets" mirror their columns for
        // plumbing, and their outputs are read from the state BEFORE each
        // row, which is what makes an ew_cov statistic or a kmeans
        // assignment safe to use as a same-row feature (E1).
        let unsupervised = self.model.is_unsupervised();
        // The leak is a feature that is a target's *column* (a renamed
        // target's as much as a plain one's); a target merely named like a
        // feature reads another column (review 2026-09-26, D7: the names
        // were matched too, and refused that).
        // A formula target's columns may be features (docs/PLAN.md task 104,
        // reviewed): a window over `mid` looking ahead is not the row's
        // `mid`, and `closed` decides whether the row's own value is in it.
        let is_target = |f: &String| {
            self.targets
                .defs()
                .iter()
                .any(|t| t.value_column() == Some(f.as_str()))
        };
        if let Some(leak) = (!unsupervised)
            .then(|| self.features.iter().find(|f| is_target(f)))
            .flatten()
        {
            return Err(format!(
                "spec {:?}: {leak:?} is both a target and a feature; a feature is read \
                 from the current row, so this would predict the target with itself \
                 (use a lagged copy of the column if you mean its past values)",
                self.name
            ));
        }
        // A formula target (docs/PLAN.md task 104) is a regression target.
        // Where the targets slot holds something else -- a column an
        // unsupervised model mirrors, a label, a sign, a 0/1 a probability is
        // fitted to -- it is refused by name rather than turned into a number
        // that means nothing.
        if self.targets.any_formula() {
            let what = "a formula target (a window expression looking ahead)";
            let why = match &self.model {
                m if m.is_unsupervised() => Some("learns from no target"),
                ModelKind::EwClass { .. } => Some("classifies its target as a label"),
                ModelKind::SeqTest { .. } => Some("tests the signs of its targets"),
                ModelKind::Ftrl { loss, .. }
                    if loss.as_deref().unwrap_or("logistic") == "logistic" =>
                {
                    Some("fits a probability to a 0/1 target (loss = \"logistic\")")
                }
                // `sgd` fits the same probability under the same loss, and a
                // log rate to a count under "poisson" (review 2026-09-26, D2).
                ModelKind::Sgd { loss, .. } if loss.as_deref() == Some("logistic") => {
                    Some("fits a probability to a 0/1 target (loss = \"logistic\")")
                }
                ModelKind::Sgd { loss, .. } if loss.as_deref() == Some("poisson") => {
                    Some("fits a log rate to a count (loss = \"poisson\")")
                }
                _ => None,
            };
            if let Some(why) = why {
                return Err(format!(
                    "spec {:?}: {what} does not apply: this model {why}",
                    self.name
                ));
            }
        }
        // A formula target is resolved by the bank's window core, which
        // needs the stream's clock and runs the spec's clock policy
        // (docs/PLAN.md task 104). A closed group cannot release the rows
        // it holds, as under `embargo`.
        if self.targets.any_formula() {
            if self.group_close.is_some() {
                return Err(format!(
                    "spec {:?}: group_close does not work with a formula target; a closed \
                     group cannot release the rows whose windows are still open",
                    self.name
                ));
            }
            if self.model.compares().is_some() {
                return Err(format!(
                    "spec {:?}: a comparison reads two other specs' residuals, not a formula \
                     target",
                    self.name
                ));
            }
            // The bank adds the target beside the columns its formula reads,
            // so it cannot be one of them. The builder has said so since task
            // 159 (P1); a spec written as a dict or in TOML was refused only
            // at its first chunk (task 160, PB5).
            if let Some(t) = self.targets.defs().iter().find(|t| {
                t.formula
                    .as_ref()
                    .is_some_and(|tree| tree.columns().contains(&t.name))
            }) {
                return Err(format!(
                    "spec {:?}: target {:?} is named after a column its formula reads, so it \
                     could never be added beside that column: give it a name of its own",
                    self.name, t.name
                ));
            }
        }
        self.decays()?;
        self.clock_cfg()?;
        let mp = self.min_periods_per_target();
        if mp.len() != self.m() {
            return Err(format!(
                "spec {:?}: min_weight list has {} entries but there are {} targets",
                self.name,
                mp.len(),
                self.m()
            ));
        }
        if let Some(v) = mp.iter().find(|v| **v < 0.0 || v.is_nan()) {
            return Err(format!(
                "spec {:?}: min_weight must be >= 0, got {v}",
                self.name
            ));
        }
        if let Some(Num(f)) = self.min_settled_frac {
            if !(0.0..1.0).contains(&f) {
                return Err(format!(
                    "spec {:?}: min_settled_frac must be a finite fraction of steady state in [0, 1) \
                     (0 is off; 0.5 is one half_life, 0.75 two), got {f}",
                    self.name
                ));
            }
            if f > 0.0 && !self.has_decay() {
                return Err(format!(
                    "spec {:?}: min_settled_frac needs a decay to settle toward (a finite \
                     half_life, or lam < 1); without one every row is settled and the gate \
                     would do nothing",
                    self.name
                ));
            }
        }
        if let Some(Num(r)) = self.max_error_inflation {
            if r.is_nan() || r <= 1.0 {
                return Err(format!(
                    "spec {:?}: max_error_inflation must be a ratio above 1 -- how much \
                     estimation error may inflate a prediction's error over the noise floor \
                     (sqrt(2) is the default; inf switches the gate off), got {r}",
                    self.name
                ));
            }
            // Range-checked and then dropped on every other model before
            // release 0.11.1 (docs/PLAN.md task 109): refused by name, as
            // `emit_error_inflation` is, with the reason (task 116).
            if !self.has_error_inflation() {
                return Err(format!(
                    "spec {:?}: max_error_inflation needs a model with a noise statistic to \
                     gate on (ewridge, rls, kalman, lasso); {} has none: {}. It is held by \
                     min_weight alone",
                    self.name,
                    self.model.kind_name(),
                    self.no_noise_statistic()
                ));
            }
        }
        if self.emit_se_coef && !self.has_se_coef() {
            let why = match self.model {
                ModelKind::Lasso { .. } => {
                    "lasso's fit is post-selection: its active set is chosen from the same rows, \
                     and a covariance that ignores the choice understates it"
                }
                ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                    "an M-estimator's covariance is a sandwich of the loss's curvature and the \
                     scores' spread, a different formula from a least-squares fit's"
                }
                ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                    "a gradient fit keeps no second moment of the features to read a covariance \
                     from"
                }
                _ => "it keeps no covariance of coefficients",
            };
            return Err(format!(
                "spec {:?}: emit_se_coef needs a model that keeps its coefficients' covariance \
                 (ewridge, rls, kalman); {} has none: {why}",
                self.name,
                self.model.kind_name()
            ));
        }
        if self.emit_error_inflation && !self.has_row_error_inflation() {
            let why = if matches!(self.model, ModelKind::Lasso { .. }) {
                "lasso keeps no factor to read a row's own leverage from (its gate reads the \
                 active count over Kish's n instead)"
            } else {
                self.no_noise_statistic()
            };
            return Err(format!(
                "spec {:?}: emit_error_inflation needs a model that reads one row's estimation \
                 variance (ewridge, rls, kalman); {} has none: {why}",
                self.name,
                self.model.kind_name()
            ));
        }
        if let Some(a) = &self.drift_action
            && !["flag", "reset"].contains(&a.as_str())
        {
            return Err(format!(
                "spec {:?}: drift_action must be \"flag\" or \"reset\", got {a:?}",
                self.name
            ));
        }
        // A tolerance or a threshold of `inf` is a detector that never
        // fires: no setting (review 2026-09-12, S27).
        if let Some(v) = self.drift_delta.filter(|v| *v < 0.0 || !v.is_finite()) {
            return Err(format!(
                "spec {:?}: drift_delta must be finite and >= 0, got {v}",
                self.name
            ));
        }
        if let Some(v) = self
            .drift_threshold
            .as_ref()
            .filter(|v| v.value() <= 0.0 || !v.value().is_finite())
        {
            return Err(format!(
                "spec {:?}: drift_threshold must be finite and > 0, got {v}",
                self.name
            ));
        }
        if let Some(qs) = &self.resid_quantiles {
            if qs.is_empty() {
                return Err(format!(
                    "spec {:?}: resid_quantiles must be non-empty",
                    self.name
                ));
            }
            if let Some(q) = qs
                .iter()
                .find(|q| !(0.0..=1.0).contains(*q) || **q == 0.0 || **q == 1.0)
            {
                return Err(format!(
                    "spec {:?}: resid_quantiles must be strictly between 0 and 1, got {q}",
                    self.name
                ));
            }
        }
        if let Some(c) = self.conformal.filter(|c| !(*c > 0.0 && *c < 1.0)) {
            return Err(format!(
                "spec {:?}: conformal must be a coverage level strictly between 0 and 1, got {c}",
                self.name
            ));
        }
        if let Some(r) = self.conformal_rate.filter(|r| !(*r > 0.0 && r.is_finite())) {
            return Err(format!(
                "spec {:?}: conformal_rate must be finite and > 0, got {r}",
                self.name
            ));
        }
        if self.conformal_rate.is_some() && self.conformal.is_none() {
            return Err(format!(
                "spec {:?}: conformal_rate needs conformal (the coverage level) to be set",
                self.name
            ));
        }
        if self.resid_autocorr_lag.is_some_and(|l| l == 0) {
            return Err(format!(
                "spec {:?}: resid_autocorr_lag must be >= 1, got 0",
                self.name
            ));
        }
        // The tracker's buffer is sized from the lag before the first row,
        // as the models' lag rings are (review 2026-10-06, CD10).
        if let Some(l) = self.resid_autocorr_lag {
            online_core::check_lag_ceiling("resid_autocorr_lag", l)
                .map_err(|e| format!("spec {:?}: {e}", self.name))?;
        }
        if let Some(v) = self.average_eta.filter(|v| v.0 <= 0.0 || v.0.is_nan()) {
            return Err(format!(
                "spec {:?}: average_eta must be > 0 (\"inf\" is emit_selected's argmin), got {}",
                self.name, v.0
            ));
        }
        // A knob whose switch is off does nothing: refused rather than
        // ignored, as elsewhere here (review 2026-09-12, S22). `drift_action
        // = "flag"` is what `fill_defaults` writes on every spec, so only
        // `"reset"` asks for anything.
        if !self.emit_drift {
            if self.drift_action.as_deref() == Some("reset") {
                return Err(format!(
                    "spec {:?}: drift_action = \"reset\" needs emit_drift; the detector is built \
                     only under that flag",
                    self.name
                ));
            }
            for (knob, set) in [
                ("drift_delta", self.drift_delta.is_some()),
                ("drift_threshold", self.drift_threshold.is_some()),
            ] {
                if set {
                    return Err(format!("spec {:?}: {knob} needs emit_drift", self.name));
                }
            }
        } else if self.clock.is_some() && self.drift_threshold.is_none() {
            return Err(format!(
                "spec {:?}: drift_threshold is required when clock is given, as gap_cap is: the \
                 drift detector sums each row's excess times its clock step, so its threshold is \
                 sigma times clock time, and no default means the same on every clock; give it \
                 in the clock column's units, or as a duration on a Datetime, Date or Duration \
                 clock (\"20m\": one sigma of excess held for twenty minutes)",
                self.name
            ));
        }
        if self.average_eta.is_some() && !self.emit_averaged {
            return Err(format!(
                "spec {:?}: average_eta needs emit_averaged",
                self.name
            ));
        }
        if self.resid_autocorr_lag.is_some() && !self.emit_autocorr {
            return Err(format!(
                "spec {:?}: resid_autocorr_lag needs emit_autocorr",
                self.name
            ));
        }
        if self.session_gap.is_some() && self.session.is_none() {
            return Err(format!("spec {:?}: session_gap needs session", self.name));
        }
        // A step back is read off the clock; given without one it would be
        // silently ignored, and a key that does nothing is refused here.
        if self.restart_after_step_back.is_some() && self.clock.is_none() {
            return Err(format!(
                "spec {:?}: restart_after_step_back needs clock",
                self.name
            ));
        }
        // The snapshot cadence spaces a window's snapshots, so each part of
        // it needs a window; the clock spacing is clock units, finite and
        // `>= 0` as `solve_every` is, `0` for every row (docs/PLAN.md task
        // 162). A negative value would have meant every row unsaid, NaN never.
        if let (Some((window, _)), Some((every, rows))) =
            (self.model.window_parts(), self.model.window_cadence())
        {
            for (key, given) in [
                ("window_every", every.is_some()),
                ("max_rows_between_snapshots", rows.is_some()),
            ] {
                if given && window.is_none() {
                    return Err(format!("spec {:?}: {key} needs `window_size`", self.name));
                }
            }
            if let Some(e) = every.filter(|e| !non_negative(e.value()) || !e.value().is_finite()) {
                return Err(format!(
                    "spec {:?}: window_every must be finite and >= 0 clock units (0 snapshots \
                     every row), got {e}",
                    self.name
                ));
            }
        }
        // The window's edge, Polars' `closed` (docs/PLAN.md task 196, N17):
        // of its four values the two that leave out the window's last row
        // cannot apply, and are refused as the spec is read
        // ([`ModelClosed`]); `"both"` with no window is a key that does
        // nothing (review S10's rule).
        if let (Some((window, _)), Some(closed)) =
            (self.model.window_parts(), self.model.window_closed())
            && closed == ModelClosed::Both
            && window.is_none()
        {
            return Err(format!(
                "spec {:?}: closed needs `window_size` (it says which edge holds a row \
                 exactly one window old)",
                self.name
            ));
        }
        // A row cap of no rows is no schedule, in every `max_rows_between_*`,
        // as `max_rows_between_coefs`'s is since task 178; the clock form's
        // `0` is every row. `0` was every row for four of them and refused
        // for the fifth (review 2026-10-06, PC3 CB4 YA2; docs/PLAN.md task
        // 196, U7).
        for (key, cap, clock) in self.model.row_caps() {
            if cap == Some(0) {
                return Err(format!(
                    "spec {:?}: {key} must be >= 1 ({clock} = 0 is every row), got 0",
                    self.name
                ));
            }
        }
        // A count of no threads (docs/PLAN.md task 225).
        if self.model.gram_threads_given() == Some(0) {
            return Err(format!(
                "spec {:?}: gram_threads must be >= 1 (the threads the Gram's update runs \
                 on; leave it out for one), got 0",
                self.name
            ));
        }
        // A budget bounds a window's snapshots, so it needs a window, and a
        // budget of no bytes bounds nothing (review 2026-09-12, P4).
        if let Some((window, Some(budget))) = self.model.window_parts() {
            if window.is_none() {
                return Err(format!(
                    "spec {:?}: window_budget needs `window_size`",
                    self.name
                ));
            }
            let mib = budget.mib();
            if mib <= 0.0 || mib.is_nan() {
                return Err(format!(
                    "spec {:?}: window_budget must be > 0 MiB (\"inf\" is no bound), got {mib}",
                    self.name
                ));
            }
        }
        // `coef_every` and `max_rows_between_coefs` schedule the `coef`
        // field, which a model with no coefficients does not have (review
        // 2026-09-12, S22). `0` was the default and passed there; since task
        // 178 it is every row, and refused there as any other value is.
        for (key, given) in [
            ("coef_every", self.coef_every.is_some()),
            (
                "max_rows_between_coefs",
                self.max_rows_between_coefs.is_some(),
            ),
        ] {
            if given && !self.model.has_coef() {
                return Err(format!(
                    "spec {:?}: {key} does not apply to {} (it reports no coefficients)",
                    self.name,
                    self.model.kind_name()
                ));
            }
        }
        if let Some(e) = self
            .coef_every
            .as_ref()
            .filter(|e| !(e.value().is_finite() && non_negative(e.value())))
        {
            return Err(format!(
                "spec {:?}: coef_every must be finite and >= 0 clock units (0 writes `coef` on \
                 every row), got {e}",
                self.name
            ));
        }
        // A cap of no rows is no schedule: every row is `coef_every = 0`.
        if self.max_rows_between_coefs == Some(0) {
            return Err(format!(
                "spec {:?}: max_rows_between_coefs must be >= 1 (coef_every = 0 writes `coef` \
                 on every row), got 0",
                self.name
            ));
        }
        // Nothing residual-based applies to a model that predicts no target.
        // Refused rather than ignored: a flag that silently emits nothing
        // looks like a bug in the output, not in the spec.
        if self.model.predicts_no_target() {
            let asked = [
                ("emit_sigma", self.emit_sigma),
                ("emit_zscore", self.emit_zscore),
                ("emit_metrics", self.emit_metrics),
                ("resid_quantiles", self.resid_quantiles.is_some()),
                ("conformal", self.conformal.is_some()),
                ("emit_autocorr", self.emit_autocorr),
                ("emit_drift", self.emit_drift),
                ("emit_averaged", self.emit_averaged),
                ("emit_selected", self.emit_selected),
            ];
            if let Some((flag, _)) = asked.iter().find(|(_, on)| *on) {
                return Err(format!(
                    "spec {:?}: {flag} does not apply to {} (it has no predictions, so no \
                     residuals)",
                    self.name,
                    self.model.kind_name()
                ));
            }
        }
        if self.emit_selected {
            let n_slots = self.decays()?.len() * crate::combo_labels(self).len();
            if n_slots < 2 {
                return Err(format!(
                    "spec {:?}: emit_selected needs more than one slot per target; add a \
                     ridge/feature_set/half_life grid or a lasso path",
                    self.name
                ));
            }
        }
        match &self.model {
            ModelKind::Holt {
                trend_half_life,
                trend,
            } => {
                if *trend == Some(false) && trend_half_life.is_some() {
                    return Err(format!(
                        "spec {:?}: holt trend_half_life applies only with a trend; trend = false \
                         holds it at zero",
                        self.name
                    ));
                }
                if let Some(h) = trend_half_life
                    .as_ref()
                    .filter(|h| h.value() <= 0.0 || h.value().is_nan())
                {
                    return Err(format!(
                        "spec {:?}: trend_half_life must be > 0 (\"inf\" forgets no slope), got {h}",
                        self.name
                    ));
                }
            }
            ModelKind::Pa { mode, c, eps, .. } => {
                if let Some(md) = mode
                    && !["pa", "pa1", "pa2"].contains(&md.as_str())
                {
                    return Err(format!(
                        "spec {:?}: unknown pa mode {md:?}; expected pa, pa1 or pa2",
                        self.name
                    ));
                }
                if let Some(v) = c.filter(|v| v.0 <= 0.0 || v.0.is_nan()) {
                    return Err(format!(
                        "spec {:?}: pa c must be > 0 (\"inf\" caps nothing: mode \"pa\"), got {}",
                        self.name, v.0
                    ));
                }
                // A tube of `inf` holds every row: a model that never learns
                // (review 2026-09-12, S27).
                if let Some(v) = eps.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: pa eps must be finite and >= 0, got {v}",
                        self.name
                    ));
                }
            }
            ModelKind::Sgd {
                loss,
                huber_delta,
                quantile,
                eps,
                learning_rate,
                schedule,
                power,
                strict_binary,
                ..
            } => {
                if let Some(l) = loss {
                    const OK: [&str; 6] = [
                        "squared",
                        "huber",
                        "quantile",
                        "epsilon_insensitive",
                        "poisson",
                        "logistic",
                    ];
                    if !OK.contains(&l.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown sgd loss {l:?}; expected one of {}",
                            self.name,
                            OK.join(", ")
                        ));
                    }
                    if l == "quantile" && quantile.is_none() {
                        return Err(format!(
                            "spec {:?}: sgd loss \"quantile\" needs a `quantile` level",
                            self.name
                        ));
                    }
                }
                if let Some(sc) = schedule
                    && !["constant", "inv_scaling", "adagrad"].contains(&sc.as_str())
                {
                    return Err(format!(
                        "spec {:?}: unknown sgd schedule {sc:?}; expected constant, \
                             inv_scaling or adagrad",
                        self.name
                    ));
                }
                if let Some(v) = learning_rate.filter(|v| *v <= 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: learning_rate must be finite and > 0, got {v}",
                        self.name
                    ));
                }
                // Unchecked, a NaN reached the core's `f64::clamp`, which
                // panics on a NaN bound.
                if let Some(d) = huber_delta.filter(|d| !positive(d.0)) {
                    return Err(format!(
                        "spec {:?}: huber_delta must be > 0 (\"inf\" is the squared loss), got {}",
                        self.name, d.0
                    ));
                }
                // A parameter of a loss or a schedule the spec does not use
                // is refused, as a switch that is off is: each was taken and
                // ignored. The builder has refused them since task 160 (YA8);
                // a dict or a TOML file is refused here, in its words (YA8b).
                // A null is the default, not a value given.
                let loss = loss.as_deref().unwrap_or("squared");
                let schedule = schedule.as_deref().unwrap_or("constant");
                for (key, given, what, owner, chosen) in [
                    ("huber_delta", huber_delta.is_some(), "loss", "huber", loss),
                    ("quantile", quantile.is_some(), "loss", "quantile", loss),
                    ("eps", eps.is_some(), "loss", "epsilon_insensitive", loss),
                    ("strict_binary", *strict_binary, "loss", "logistic", loss),
                    (
                        "power",
                        power.is_some(),
                        "schedule",
                        "inv_scaling",
                        schedule,
                    ),
                ] {
                    if given && chosen != owner {
                        return Err(format!(
                            "spec {:?}: sgd {key} is for {what} {owner:?}; {what} {chosen:?} \
                             does not use it",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::EwCov {
                stats,
                precision_prior,
                mahal_quantiles,
                pca,
                pca_every,
                max_rows_between_pca,
                lags,
                // Checked with `ew_ridge`'s (`gram_threads_given`).
                gram_threads: _,
                window_size: window,
                // Checked with every windowed kind's (`window_cadence`).
                window_every: _,
                max_rows_between_snapshots: _,
                window_budget: _,
                closed: _,
            } => {
                if let Some(w) = window
                    && (!w.value().is_finite() || w.value() <= 0.0)
                {
                    return Err(format!(
                        "spec {:?}: window_size must be finite and > 0 (got {w}); it is clock \
                         units of history to keep",
                        self.name
                    ));
                }
                const OK: [&str; 8] = [
                    "mean",
                    "var",
                    "std",
                    "cov",
                    "corr",
                    "partial_corr",
                    "mahal",
                    "lag_corr",
                ];
                if let Some(stats) = stats {
                    // `stats = []` stays legal: it is the accumulate-only use
                    // (docs/ENHANCEMENTS.md E43), which writes `weight_sum`
                    // alone and keeps the Gram, not a spelling of absent.
                    for st in stats {
                        if !OK.contains(&st.as_str()) {
                            let renamed = RENAMED_VALUES
                                .iter()
                                .find(|(old, _)| *old == st)
                                .map_or(String::new(), |(old, new)| {
                                    format!("; {old} was renamed {new}")
                                });
                            return Err(format!(
                                "spec {:?}: unknown ew_cov statistic {st:?}; expected one of \
                                 {}{renamed}",
                                self.name,
                                OK.join(", ")
                            ));
                        }
                    }
                    let pairwise =
                        |st: &String| st == "cov" || st == "corr" || st == "partial_corr";
                    if self.k() < 2 && stats.iter().any(pairwise) {
                        return Err(format!(
                            "spec {:?}: ew_cov cov/corr/partial_corr need at least two features",
                            self.name
                        ));
                    }
                    if stats.iter().any(|st| st == "partial_corr") && precision_prior.is_none() {
                        return Err(format!(
                            "spec {:?}: ew_cov partial_corr needs `precision_prior`",
                            self.name
                        ));
                    }
                    if stats.iter().any(|st| st == "mahal") && precision_prior.is_none() {
                        return Err(format!(
                            "spec {:?}: ew_cov mahal needs `precision_prior`",
                            self.name
                        ));
                    }
                }
                if let Some(p) = precision_prior.filter(|p| *p <= 0.0 || !p.is_finite()) {
                    return Err(format!(
                        "spec {:?}: precision_prior must be finite and > 0, got {p}",
                        self.name
                    ));
                }
                let has_lag_corr = stats
                    .as_ref()
                    .is_some_and(|st| st.iter().any(|s| s == "lag_corr"));
                if has_lag_corr && lags.as_ref().is_none_or(|l| l.is_empty()) {
                    return Err(format!(
                        "spec {:?}: ew_cov lag_corr needs `lags` (which lags to accumulate, e.g. \
                         lags = [1, 2, 5])",
                        self.name
                    ));
                }
                if let Some(lags) = lags {
                    // The list's own rules are the accumulator's, so the CLI
                    // and the bank get one message.
                    online_core::EwLagCov::new(self.k().max(1), lags.clone())
                        .map_err(|e| format!("spec {:?}: ew_cov {e}", self.name))?;
                }
                if let Some(levels) = mahal_quantiles {
                    // An empty list asked for nothing and was taken without a
                    // word, where `resid_quantiles = []` is refused (review
                    // 2026-10-06, YA5).
                    if levels.is_empty() {
                        return Err(format!(
                            "spec {:?}: ew_cov mahal_quantiles must be non-empty; leave it out \
                             for none",
                            self.name
                        ));
                    }
                    let has_mahal = stats
                        .as_ref()
                        .is_some_and(|st| st.iter().any(|s| s == "mahal"));
                    if !has_mahal {
                        return Err(format!(
                            "spec {:?}: ew_cov mahal_quantiles needs \"mahal\" in `stats`",
                            self.name
                        ));
                    }
                    for &q in levels {
                        if !(q > 0.0 && q < 1.0) {
                            return Err(format!(
                                "spec {:?}: ew_cov mahal_quantiles must be strictly between 0 and 1, got {q}",
                                self.name
                            ));
                        }
                    }
                }
                // `0` was a second spelling of absent (review 2026-10-06,
                // PC9; docs/PLAN.md task 196, U7).
                if *pca == Some(0) {
                    return Err(format!(
                        "spec {:?}: ew_cov pca = 0 asks for no component; leave it out for none",
                        self.name
                    ));
                }
                if let Some(r) = pca
                    && *r > self.k()
                {
                    return Err(format!(
                        "spec {:?}: ew_cov pca asks for {r} components of {} features",
                        self.name,
                        self.k()
                    ));
                }
                if let Some(e) = pca_every
                    .as_ref()
                    .filter(|e| !non_negative(e.value()) || !e.value().is_finite())
                {
                    return Err(format!(
                        "spec {:?}: ew_cov pca_every must be finite and >= 0 clock units (0 \
                         refreshes on every row), got {e}",
                        self.name
                    ));
                }
                for (key, given) in [
                    ("pca_every", pca_every.is_some()),
                    ("max_rows_between_pca", max_rows_between_pca.is_some()),
                ] {
                    if given && pca.is_none_or(|r| r == 0) {
                        return Err(format!(
                            "spec {:?}: ew_cov {key} needs `pca` (the number of components)",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::KMeans {
                k,
                seed_rule,
                update_every_rows,
                split_merge,
                split_merge_every_rows,
                dead_frac,
                scale_floor,
                ..
            } => {
                if *k == 0 {
                    return Err(format!(
                        "spec {:?}: kmeans k must be >= 1, got 0",
                        self.name
                    ));
                }
                if let Some(rule) = seed_rule {
                    const OK: [&str; 4] = ["first", "farthest", "kmeanspp", "lloyd"];
                    if !OK.contains(&rule.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown kmeans seed_rule {rule:?}; expected one of {}",
                            self.name,
                            OK.join(", ")
                        ));
                    }
                }
                if update_every_rows.is_some_and(|v| v == 0) {
                    return Err(format!(
                        "spec {:?}: update_every_rows must be >= 1, got 0",
                        self.name
                    ));
                }
                if split_merge_every_rows.is_some_and(|v| v == 0) {
                    return Err(format!(
                        "spec {:?}: split_merge_every_rows must be >= 1, got 0",
                        self.name
                    ));
                }
                if let Some(v) = split_merge.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: split_merge must be finite and >= 0 (0 disables it), got {v}",
                        self.name
                    ));
                }
                if let Some(v) = dead_frac.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: dead_frac must be finite and >= 0 (0 disables it), got {v}",
                        self.name
                    ));
                }
                if let Some(v) = scale_floor.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: scale_floor must be finite and >= 0 (0 is the EW variance \
                         alone), got {v}",
                        self.name
                    ));
                }
            }
            ModelKind::Micro {
                eps,
                beta_mu,
                max_clusters,
                prune_every,
                macro_link,
                scale_floor,
                ..
            } => {
                if !(eps.is_finite() && *eps > 0.0) {
                    return Err(format!(
                        "spec {:?}: micro eps must be finite and > 0, got {eps}",
                        self.name
                    ));
                }
                if let Some(v) = beta_mu.filter(|v| !(v.is_finite() && *v > 0.0)) {
                    return Err(format!(
                        "spec {:?}: beta_mu must be finite and > 0, got {v}",
                        self.name
                    ));
                }
                if max_clusters.is_some_and(|v| v == 0) {
                    return Err(format!(
                        "spec {:?}: max_clusters must be >= 1, got 0",
                        self.name
                    ));
                }
                if let Some(e) = prune_every
                    .as_ref()
                    .filter(|e| !non_negative(e.value()) || !e.value().is_finite())
                {
                    return Err(format!(
                        "spec {:?}: micro prune_every must be finite and >= 0 clock units (0 \
                         checkpoints on every row), got {e}",
                        self.name
                    ));
                }
                if let Some(v) = macro_link.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: macro_link must be finite and >= 0 (0 links nothing), got {v}",
                        self.name
                    ));
                }
                if let Some(v) = scale_floor.filter(|v| *v < 0.0 || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: scale_floor must be finite and >= 0 (0 is the EW variance \
                         alone), got {v}",
                        self.name
                    ));
                }
            }
            ModelKind::EwClass {
                classes,
                covariance,
                precision_prior,
                window_size: window,
                // Checked with every windowed kind's (`window_cadence`).
                window_every: _,
                max_rows_between_snapshots: _,
                window_budget: _,
                closed: _,
            } => {
                if let Some(w) = window
                    && (!w.value().is_finite() || w.value() <= 0.0)
                {
                    return Err(format!(
                        "spec {:?}: window_size must be finite and > 0 (got {w})",
                        self.name
                    ));
                }
                if self.targets.len() != 1 {
                    return Err(format!(
                        "spec {:?}: ew_class takes exactly one target, the label column (got {})",
                        self.name,
                        self.targets.len()
                    ));
                }
                if classes.len() < 2 {
                    return Err(format!(
                        "spec {:?}: ew_class classes must list at least 2 classes (got {})",
                        self.name,
                        classes.len()
                    ));
                }
                let mut seen = std::collections::HashSet::new();
                if let Some(dup) = classes.iter().find(|c| !seen.insert(c.as_str())) {
                    return Err(format!(
                        "spec {:?}: ew_class classes lists {dup:?} more than once",
                        self.name
                    ));
                }
                if classes.iter().any(|c| c.is_empty()) {
                    return Err(format!(
                        "spec {:?}: ew_class classes must not contain an empty name",
                        self.name
                    ));
                }
                if let Some(c) = covariance {
                    online_core::Covariance::parse("ew_class", c)
                        .map_err(|e| format!("spec {:?}: {e}", self.name))?;
                }
                if !(precision_prior.is_finite() && *precision_prior > 0.0) {
                    return Err(format!(
                        "spec {:?}: ew_class precision_prior must be finite and > 0, got \
                         {precision_prior}",
                        self.name
                    ));
                }
            }
            // Checked above, with the features: its refusals come before the
            // shared checks so that `half_life` is named for what it is here.
            ModelKind::SeqTest { .. } => {}
            // The shared checks (non-empty features and targets, no column on
            // both sides, a decay, `min_weight` per target, no residual
            // diagnostics) and one of its own: given edges and the knobs for
            // learning them are two ways of saying where the bins are, and a
            // spec that says both is a spec with a mistake in it. The core
            // checks the edges and the lags themselves.
            ModelKind::Marginal {
                bins,
                bin_rule,
                bin_warm_rows,
                bin_edges,
                bin_budget,
                window_size: window,
                lags,
                cross_lags,
                window_lags,
                feature_moments,
                ..
            } => {
                // Where the feature moments are kept (docs/PLAN.md task 125).
                // The core refuses `"shared"` with a window or lags too, and
                // says why; here it is refused where the spec is made.
                match feature_moments.as_deref() {
                    None | Some("per_target") => {}
                    Some("shared") if window.is_some() => {
                        return Err(format!(
                            "spec {:?}: marginal feature_moments = \"shared\" takes no window: \
                             a window subtracts each pair's moments centred on the pair's own \
                             mean, and the shared mean also moves on rows the pair's target \
                             missed. Use feature_moments = \"per_target\" with a window.",
                            self.name
                        ));
                    }
                    Some("shared") => {}
                    Some(other) => {
                        return Err(format!(
                            "spec {:?}: marginal feature_moments must be \"per_target\" or \
                             \"shared\", got {other:?}",
                            self.name
                        ));
                    }
                }
                // Lags under a window cost the snapshot its lag moments, and
                // the spec says so by name before it pays (the user,
                // 2026-09-28: "implement but with an api parameter that
                // documents the impact"; docs/PLAN.md task 137).
                let n_lags = lags.as_ref().map_or(0, Vec::len);
                match (window.is_some(), n_lags > 0, window_lags.unwrap_or(false)) {
                    (true, true, false) => {
                        let (p, t) = (self.k() as f64, self.m() as f64);
                        let l = n_lags as f64;
                        let c = cross_lags.as_ref().map_or(l, |c| c.len() as f64);
                        let base = (3.0 * p + 5.0) * t;
                        let more = l * t + (l + 2.0 * c) * p * t;
                        return Err(format!(
                            "spec {:?}: marginal lags under a window need window_lags = true. \
                             Each window snapshot then also holds the lag moments: here {more} \
                             doubles beside the {base} it holds without them, {:.1} times the \
                             size (L·T + (L + 2C)·p·T against (3p + 5)·T, C the cross lags; \
                             cross_lags = [] costs least, and n_serial does not read the cross \
                             terms). Set window_lags = true to accept that, or drop the window \
                             or the lags.",
                            self.name,
                            (base + more) / base
                        ));
                    }
                    (window_set, lags_set, true) if !(window_set && lags_set) => {
                        return Err(format!(
                            "spec {:?}: marginal window_lags applies only with both a window and \
                             lags",
                            self.name
                        ));
                    }
                    _ => {}
                }
                if bin_edges.is_some()
                    && (bins.is_some() || bin_rule.is_some() || bin_warm_rows.is_some())
                {
                    return Err(format!(
                        "spec {:?}: marginal bin_edges fixes the bins outright; bins, bin_rule \
                         and bin_warm_rows describe learning them and do not apply with it",
                        self.name
                    ));
                }
                if bin_budget.is_some() && bins.is_none() && bin_edges.is_none() {
                    return Err(format!(
                        "spec {:?}: marginal bin_budget bounds the bins' memory and needs bins \
                         or bin_edges",
                        self.name
                    ));
                }
                // The lag ring is sized before the first row: `[2^62]`
                // panicked inside the builder (review 2026-10-06, CD10).
                if let Some(&l) = lags.as_ref().and_then(|l| l.iter().max()) {
                    online_core::check_lag_ceiling("marginal: lags", l)
                        .map_err(|e| format!("spec {:?}: {e}", self.name))?;
                }
            }
            // Every parameter check is `DecoCfg::validate`'s, so that the
            // CLI and the bank get the same messages; only
            // the block *names* are resolved here, where the feature list is.
            ModelKind::Bocpd {
                hazard, hazard_col, ..
            } => {
                // The model reads `targets[0]` as the hazard whatever the name
                // here says, so the two must be one column (review
                // 2026-09-12, C20).
                // The slot is the column itself, not a table: a table's name
                // means nothing here, and its column is what the model would
                // read (review 2026-09-26, D4: the names were compared, so a
                // table named like the hazard put another column in the slot).
                if let Some(h) = hazard_col
                    && self.targets.defs() != [crate::targets::TargetDef::plain(h.clone())]
                {
                    return Err(format!(
                        "spec {:?}: bocpd reads hazard_col {h:?} from the targets slot, so \
                             targets must be [{h:?}], the column itself (got {:?})",
                        self.name,
                        self.targets.as_slice()
                    ));
                }
                if self.half_life.is_some() || self.lam.is_some() {
                    return Err(format!(
                        "spec {:?}: half_life/lam do not apply to bocpd; the run-length posterior \
                         is what forgets, and `hazard` is how fast",
                        self.name
                    ));
                }
                // A column's value is the chance of a break after its row,
                // and a duration's the chance of the step before each row:
                // a null falling back to the duration would put both on one
                // boundary (task 179).
                if let (Some(h), Some(d)) =
                    (hazard_col, hazard.as_ref().filter(|d| d.is_duration()))
                {
                    return Err(format!(
                        "spec {:?}: bocpd's hazard_col {h:?} reads a per-row hazard, the chance \
                         of a break after its row, and hazard is a duration ({d}), the chance of \
                         the step before each row; one stream cannot take both. Give hazard as a \
                         number of rows beside hazard_col, or leave hazard_col out",
                        self.name
                    ));
                }
                crate::stream::bocpd_cfg(self).map_err(|e| format!("spec {:?}: {e}", self.name))?;
            }
            ModelKind::CorrChange { scalar, .. } => {
                if !scalar.unwrap_or(false) && (self.half_life.is_some() || self.lam.is_some()) {
                    return Err(format!(
                        "spec {:?}: half_life/lam apply to corrchange only with scalar = true \
                         (they parametrise the standardiser); neither kind decays anything else",
                        self.name
                    ));
                }
                // The model's own checks too, `n_perm`'s ceiling among them,
                // so a spec meets them here (review 2026-10-06, CD10).
                crate::stream::corrchange_cfg(self)
                    .and_then(|c| c.validate())
                    .map_err(|e| format!("spec {:?}: {e}", self.name))?;
            }
            ModelKind::Hmm { k, exog_tvtp, .. } => {
                // `k` sizes the transition matrix and the states before the
                // first row, as `kmeans`' sizes its centres; held here before
                // the configuration is built from it (review 2026-10-06,
                // CF2's sibling).
                online_core::Hmm::check_k(*k).map_err(|e| format!("spec {:?}: {e}", self.name))?;
                // As `bocpd`'s `hazard_col` (review 2026-09-12, C20).
                if let Some(z) = exog_tvtp
                    && self.targets.defs() != [crate::targets::TargetDef::plain(z.clone())]
                {
                    return Err(format!(
                        "spec {:?}: hmm reads exog_tvtp {z:?} from the targets slot, so \
                             targets must be [{z:?}], the column itself (got {:?})",
                        self.name,
                        self.targets.as_slice()
                    ));
                }
                crate::stream::hmm_cfg(self).map_err(|e| format!("spec {:?}: {e}", self.name))?;
            }
            ModelKind::Rcov { .. } => {
                if self.group.is_none() || self.group_close.is_none() {
                    return Err(format!(
                        "spec {:?}: rcov needs `group` and `group_close`; its value is the block \
                         it emits when the group closes, and a stream with no close never emits \
                         one",
                        self.name
                    ));
                }
                if self.half_life.is_some() || self.lam.is_some() {
                    return Err(format!(
                        "spec {:?}: half_life/lam do not apply to rcov (a realised covariance is \
                         a sum over a block, not a decayed mean); the block boundary is \
                         group_close's",
                        self.name
                    ));
                }
                crate::stream::rcov_cfg(self).map_err(|e| format!("spec {:?}: {e}", self.name))?;
            }
            ModelKind::Deco { blocks, .. } => {
                if let Some(blocks) = blocks {
                    for (name, cols) in blocks {
                        for c in cols {
                            if !self.features.contains(c) {
                                return Err(format!(
                                    "spec {:?}: deco block {name:?} names {c:?}, which is not a \
                                     feature of this spec",
                                    self.name
                                ));
                            }
                        }
                    }
                }
                crate::stream::deco_cfg(self).map_err(|e| format!("spec {:?}: {e}", self.name))?;
            }
            ModelKind::Ftrl {
                alpha,
                beta,
                l1,
                l2,
                ..
            } => {
                // At `inf` each is no setting: `alpha` leaves `-z/l2`, and
                // `beta`, `l1` or `l2` zeroes every coordinate for ever
                // (review 2026-09-12, S27).
                if let Some(a) = alpha.filter(|a| *a <= 0.0 || !a.is_finite()) {
                    return Err(format!(
                        "spec {:?}: ftrl alpha must be finite and > 0, got {a}",
                        self.name
                    ));
                }
                for (name, v) in [("beta", beta), ("l1", l1), ("l2", l2)] {
                    if let Some(v) = v.filter(|v| *v < 0.0 || !v.is_finite()) {
                        return Err(format!(
                            "spec {:?}: ftrl {name} must be finite and >= 0, got {v}",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::Huber {
                huber_delta,
                ridge,
                solve_every,
                ..
            } => {
                if let Some(d) = huber_delta.filter(|d| !positive(d.0)) {
                    return Err(format!(
                        "spec {:?}: huber_delta must be > 0 (\"inf\" is least squares), got {}",
                        self.name, d.0
                    ));
                }
                check_ridge(&self.name, *ridge)?;
                check_solve_every(&self.name, solve_every.as_ref())?;
            }
            ModelKind::Quantile {
                quantile,
                quantile_eps,
                ridge,
                solve_every,
                ..
            } => {
                if !(0.0 < *quantile && *quantile < 1.0) {
                    return Err(format!(
                        "spec {:?}: quantile must be in (0, 1), got {quantile}",
                        self.name
                    ));
                }
                // A floor of `inf` weighs every row 0: a model that never
                // learns (review 2026-09-12, S27).
                if let Some(e) = quantile_eps.filter(|e| !positive(*e) || !e.is_finite()) {
                    return Err(format!(
                        "spec {:?}: quantile_eps must be finite and > 0, got {e}",
                        self.name
                    ));
                }
                check_ridge(&self.name, *ridge)?;
                check_solve_every(&self.name, solve_every.as_ref())?;
            }
            ModelKind::Kalman {
                coef_half_life,
                q,
                obs_var,
                p0,
                revert_half_life,
                ..
            } => {
                let k_total = self.k() + usize::from(self.fit_intercept);
                // One way of giving the process noise: `coef_half_life` was
                // required, and ignored beside `q` (review 2026-10-06, PC6).
                match (coef_half_life, q) {
                    (Some(_), Some(_)) => {
                        return Err(format!(
                            "spec {:?}: kalman takes coef_half_life or q, not both: q is the \
                             process noise given outright, which coef_half_life derives, and the \
                             half-life was ignored beside it",
                            self.name
                        ));
                    }
                    (None, None) => {
                        return Err(format!(
                            "spec {:?}: kalman needs coef_half_life (how fast a coefficient may \
                             drift) or q (the process noise given outright)",
                            self.name
                        ));
                    }
                    _ => {}
                }
                if let Some(hl) = coef_half_life {
                    let hs = hl.to_vec();
                    if hs.len() != 1 && hs.len() != k_total {
                        return Err(format!(
                            "spec {:?}: coef_half_life must be scalar or length {k_total}, got {} \
                             values",
                            self.name,
                            hs.len()
                        ));
                    }
                    if let Some(h) = hs.iter().find(|&&h| !positive(h)) {
                        return Err(format!(
                            "spec {:?}: coef_half_life must be > 0 (\"inf\" pins a coefficient), \
                             got {h}",
                            self.name
                        ));
                    }
                }
                if let Some(rs) = revert_half_life {
                    let rs = rs.to_vec();
                    if rs.len() != 1 && rs.len() != k_total {
                        return Err(format!(
                            "spec {:?}: revert_half_life must be scalar or length {k_total}, got \
                             {} values",
                            self.name,
                            rs.len()
                        ));
                    }
                    if let Some(r) = rs.iter().find(|&&r| !positive(r)) {
                        return Err(format!(
                            "spec {:?}: revert_half_life must be > 0 (\"inf\" is the random \
                             walk), got {r}",
                            self.name
                        ));
                    }
                }
                if let Some(q) = q.as_ref().filter(|q| q.len() != k_total) {
                    return Err(format!(
                        "spec {:?}: q must have length {k_total}, got {}",
                        self.name,
                        q.len()
                    ));
                }
                if let Some(v) = q
                    .as_ref()
                    .and_then(|q| q.iter().find(|v| !non_negative(v.0) || !v.0.is_finite()))
                {
                    return Err(format!(
                        "spec {:?}: q values must be finite and >= 0 (0 pins a coefficient), got \
                         {}",
                        self.name, v.0
                    ));
                }
                if let Some(v) = obs_var.filter(|v| !positive(*v) || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: obs_var must be finite and > 0, got {v}",
                        self.name
                    ));
                }
                if let Some(v) = p0.filter(|v| !positive(*v) || !v.is_finite()) {
                    return Err(format!(
                        "spec {:?}: p0 must be finite and > 0, got {v}",
                        self.name
                    ));
                }
            }
            ModelKind::Lasso {
                lasso_path,
                l1_ratio,
                select_half_life,
                solve_every,
                tol,
                ..
            } => {
                if lasso_path.is_empty() {
                    return Err(format!(
                        "spec {:?}: lasso_path must be non-empty",
                        self.name
                    ));
                }
                if let Some(l) = lasso_path
                    .iter()
                    .find(|l| !non_negative(**l) || !l.is_finite())
                {
                    return Err(format!(
                        "spec {:?}: lasso_path values must be finite and >= 0, got {l}",
                        self.name
                    ));
                }
                // Strictly: a repeated penalty is two identical slots with the
                // same field name.
                if let Some(w) = lasso_path.windows(2).find(|w| w[0] <= w[1]) {
                    return Err(format!(
                        "spec {:?}: lasso_path must be strictly decreasing, got {} then {}",
                        self.name, w[0], w[1]
                    ));
                }
                if let Some(r) = l1_ratio.filter(|r| !(0.0..=1.0).contains(r)) {
                    return Err(format!(
                        "spec {:?}: l1_ratio must be in [0, 1], got {r}",
                        self.name
                    ));
                }
                if let Some(h) = select_half_life.as_ref().filter(|h| !positive(h.value())) {
                    return Err(format!(
                        "spec {:?}: select_half_life must be > 0, got {h}",
                        self.name
                    ));
                }
                check_solve_every(&self.name, solve_every.as_ref())?;
                if let Some(t) = tol.filter(|t| !positive(*t) || !t.is_finite()) {
                    return Err(format!(
                        "spec {:?}: tol must be finite and > 0, got {t}",
                        self.name
                    ));
                }
            }
            ModelKind::Rls { delta, coef_prior } => {
                if let Some(r) = delta.filter(|r| !positive(*r) || !r.is_finite()) {
                    return Err(format!(
                        "spec {:?}: rls delta must be finite and > 0, got {r}",
                        self.name
                    ));
                }
                let k_total = self.k() + usize::from(self.fit_intercept);
                if let Some(c) = coef_prior
                    && (c.len() != self.m() || c.iter().any(|v| v.len() != k_total))
                {
                    return Err(format!(
                        "spec {:?}: coef_prior must be n_targets x (n_features + intercept)",
                        self.name
                    ));
                }
            }
            ModelKind::EwRidge {
                ridge,
                feature_sets,
                coef_prior,
                session_shrink,
                long_half_life,
                solve_every,
                ..
            } => {
                if let Some(r) = ridge {
                    let rs = r.to_vec();
                    // A negative ridge makes the system indefinite and the
                    // coefficients garbage; NaN/inf make them zero. Zero is
                    // legal: plain least squares, rescued by the jitter
                    // fallback when singular.
                    if let Some(r) = rs.iter().find(|r| !non_negative(**r) || !r.is_finite()) {
                        return Err(format!(
                            "spec {:?}: ridge must be finite and >= 0, got {r}",
                            self.name
                        ));
                    }
                    if let Some(dup) = first_duplicate(&rs) {
                        return Err(format!(
                            "spec {:?}: ridge lists {} more than once; each value is one \
                             grid slot and the two would produce the same field names",
                            self.name,
                            num_label(dup)
                        ));
                    }
                }
                check_solve_every(&self.name, solve_every.as_ref())?;
                if let Some(h) = long_half_life.as_ref().filter(|h| !positive(h.value())) {
                    return Err(format!(
                        "spec {:?}: long_half_life must be > 0, got {h}",
                        self.name
                    ));
                }
                if let Some(f) = session_shrink.filter(|f| !(0.0..=1.0).contains(f)) {
                    return Err(format!(
                        "spec {:?}: session_shrink must be in [0, 1], got {f}",
                        self.name
                    ));
                }
                if session_shrink.is_some() && long_half_life.is_none() {
                    return Err(format!(
                        "spec {:?}: session_shrink needs long_half_life",
                        self.name
                    ));
                }
                if session_shrink.is_some() && self.session.is_none() {
                    return Err(format!(
                        "spec {:?}: session_shrink needs a `session` column to react to",
                        self.name
                    ));
                }
                // Two prescriptions for one row. Under `session_gap = "reset"`
                // the reset wins and the blend never runs; under `group_close
                // = "session"` the stream starts over at the change, so the
                // twin is never read (review 2026-09-12, S22).
                if session_shrink.is_some() {
                    if matches!(&self.session_gap, Some(SessionGapSpec::Word(w)) if w == "reset") {
                        return Err(format!(
                            "spec {:?}: session_shrink does not apply with session_gap = \
                             \"reset\" (the reset replaces the state the blend would mix)",
                            self.name
                        ));
                    }
                    if self.closes_on_session() {
                        return Err(format!(
                            "spec {:?}: session_shrink does not apply with group_close = \
                             \"session\" (the stream starts over at the change, so the twin is \
                             never read)",
                            self.name
                        ));
                    }
                }
                if long_half_life.is_some() && session_shrink.is_none() {
                    return Err(format!(
                        "spec {:?}: long_half_life needs session_shrink; it is the half_life of \
                         the twin a session boundary blends toward",
                        self.name
                    ));
                }
                if let Some(c) = coef_prior {
                    let k_total = self.k() + usize::from(self.fit_intercept);
                    if c.len() != self.m() || c.iter().any(|v| v.len() != k_total) {
                        return Err(format!(
                            "spec {:?}: coef_prior must be {} vectors of length {k_total}",
                            self.name,
                            self.m()
                        ));
                    }
                }
                if let Some(fs) = feature_sets {
                    // `[]` is every feature, one set, to the model, and no
                    // slot at all to the field names, which rendered none for
                    // a model emitting one per ridge (review 2026-09-12, V23).
                    if fs.is_empty() {
                        return Err(format!(
                            "spec {:?}: feature_sets names no set; leave it out to fit every \
                             feature",
                            self.name
                        ));
                    }
                    let mut names = std::collections::HashSet::new();
                    for (name, cols) in fs {
                        // A name twice renders two slots under one field name,
                        // which only the bank's tripwire caught (S7).
                        if !names.insert(name.as_str()) {
                            return Err(format!(
                                "spec {:?}: feature_sets names {name:?} more than once",
                                self.name
                            ));
                        }
                        if cols.is_empty() {
                            return Err(format!(
                                "spec {:?}: feature set {name:?} is empty",
                                self.name
                            ));
                        }
                        let mut seen = std::collections::HashSet::new();
                        for c in cols {
                            if !self.features.contains(c) {
                                return Err(format!(
                                    "spec {:?}: feature set {name:?} references unknown feature {c:?}",
                                    self.name
                                ));
                            }
                            // A column twice split its coefficient across two
                            // identical slots, without a word (S7).
                            if !seen.insert(c.as_str()) {
                                return Err(format!(
                                    "spec {:?}: feature set {name:?} lists {c:?} more than once",
                                    self.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod clock_tests {
    use super::{CLOCK_FIELDS, ClockScale, DURATION_OR_UNIT_FREE_FIELDS, ModelKind, Spec};
    use crate::span::Span;
    use std::collections::BTreeSet;

    /// The fields serde knows for one JSON object, read from its
    /// unknown-field message: the one list of them outside the type.
    fn fields_of<T: serde::de::DeserializeOwned>(json: &str) -> Vec<String> {
        let err = match serde_json::from_str::<T>(json) {
            Ok(_) => panic!("{json} has no unknown field"),
            Err(e) => e.to_string(),
        };
        let expected = err.split("expected").nth(1).unwrap_or("");
        expected
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    /// Whether a field reads a duration as clock units: it takes `"10m"`
    /// (at most another field is then missing) and refuses a word that is
    /// no duration *as* no duration, which a plain text field would take.
    fn takes_duration<T: serde::de::DeserializeOwned>(json: impl Fn(&str) -> String) -> bool {
        let fits = match serde_json::from_str::<T>(&json("\"10m\"")) {
            Ok(_) => true,
            Err(e) => e.to_string().starts_with("missing field"),
        };
        let refuses = serde_json::from_str::<T>(&json("\"nonsense\""))
            .err()
            .is_some_and(|e| e.to_string().contains("is not a duration"));
        fits && refuses
    }

    #[test]
    fn clock_fields_are_exactly_the_fields_that_take_a_duration() {
        let mut found: BTreeSet<(String, String)> = BTreeSet::new();
        let base =
            r#""name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"]"#;
        for f in fields_of::<Spec>(&format!("{{{base}, \"zzz\": 1}}")) {
            if takes_duration::<Spec>(|v| format!("{{{base}, \"{f}\": {v}}}")) {
                found.insert(("*".into(), f));
            }
        }
        for kind in ModelKind::KINDS {
            for f in fields_of::<ModelKind>(&format!(r#"{{"type": "{kind}", "zzz": 1}}"#)) {
                if takes_duration::<ModelKind>(|v| format!(r#"{{"type": "{kind}", "{f}": {v}}}"#)) {
                    found.insert((kind.to_string(), f));
                }
            }
        }
        let table = |t: &[(&str, &[&str])]| -> BTreeSet<(String, String)> {
            t.iter()
                .flat_map(|(owner, fields)| {
                    fields.iter().map(|f| (owner.to_string(), f.to_string()))
                })
                .collect()
        };
        // Two tables: a clock parameter, and one that takes a duration or a
        // number bound to no clock unit (task 179). A field is in one.
        let (clock, either) = (table(CLOCK_FIELDS), table(DURATION_OR_UNIT_FREE_FIELDS));
        assert!(
            clock.is_disjoint(&either),
            "{:?}",
            clock.intersection(&either)
        );
        assert!(
            found.len() > 15,
            "the walk found too little to mean anything: {found:?}"
        );
        assert_eq!(
            found,
            &clock | &either,
            "CLOCK_FIELDS and DURATION_OR_UNIT_FREE_FIELDS against the fields that take a duration"
        );
    }

    /// A field of [`DURATION_OR_UNIT_FREE_FIELDS`] is read by `clock_spans`
    /// as a duration and not at all as a number (task 179): `bocpd`'s
    /// `hazard` of `"1h"` binds the spec to a temporal clock, and one of 250
    /// rows binds it to nothing, so it stands beside durations. A duration
    /// there meets every rule a clock parameter does: refused without a
    /// clock, and beside a plain number.
    #[test]
    fn a_duration_or_a_count_is_read_only_as_a_duration() {
        // Filled, as a bank fills a spec before it runs it: the model with
        // no target takes `features[0]` for its targets slot.
        let build = |owner: &str, field: &str, value: &str, rest: &str| -> Spec {
            let mut spec: Spec = serde_json::from_str(&format!(
                r#"{{"name": "m", "model": {{"type": "{owner}", "{field}": {value}}},
                    "features": ["x"]{rest}}}"#
            ))
            .unwrap_or_else(|e| panic!("{owner}.{field} = {value}: {e}"));
            spec.fill_defaults();
            spec
        };
        let mut walked = 0;
        for (owner, fields) in DURATION_OR_UNIT_FREE_FIELDS {
            for &field in *fields {
                let clock = r#", "clock": "t", "gap_cap": "1h""#;
                let duration = build(owner, field, r#""10m""#, clock);
                assert!(
                    duration
                        .clock_spans()
                        .iter()
                        .any(|(f, s)| *f == field && s.is_duration()),
                    "{owner}.{field}: a duration is not read"
                );
                duration.validate().unwrap();
                let number = build(owner, field, "250", clock);
                assert!(
                    number.clock_spans().iter().all(|(f, _)| *f != field),
                    "{owner}.{field}: a number is read as clock units"
                );
                assert_eq!(number.clock_scale(), Ok(ClockScale::Durations("gap_cap")));
                number.validate().unwrap();
                // Nor does a number bind a spec with no clock at all.
                assert_eq!(
                    build(owner, field, "250", "").clock_scale(),
                    Ok(ClockScale::Free)
                );
                let err = build(owner, field, r#""10m""#, "").validate().unwrap_err();
                assert!(
                    err.contains(&format!("{field} is a duration, which needs a clock")),
                    "{err}"
                );
                let mixed = r#", "clock": "t", "gap_cap": 300"#;
                let err = build(owner, field, r#""10m""#, mixed)
                    .validate()
                    .unwrap_err();
                assert!(
                    err.contains(&format!(
                        "{field} is a duration but gap_cap is a plain number"
                    )),
                    "{err}"
                );
                walked += 1;
            }
        }
        assert_eq!(walked, 1);
    }

    /// A spec whose one clock parameter under test is a duration.
    fn with_duration(owner: &str, field: &str) -> Spec {
        let model = match owner {
            "*" => r#"{"type": "ewridge"}"#.to_string(),
            kind => {
                let required = match kind {
                    "lasso" => r#", "lasso_path": [0.1]"#,
                    "kalman" if field != "coef_half_life" => r#", "coef_half_life": "1h""#,
                    "quantile" => r#", "quantile": 0.5"#,
                    "ew_class" => r#", "classes": ["a", "b"], "precision_prior": 1.0"#,
                    "micro" => r#", "eps": 0.3"#,
                    _ => "",
                };
                format!(r#"{{"type": "{kind}", "{field}": "10m"{required}}}"#)
            }
        };
        let own = if owner == "*" {
            format!(r#", "{field}": "10m""#)
        } else {
            String::new()
        };
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x"],
                "clock": "t"{own}}}"#
        ))
        .unwrap_or_else(|e| panic!("{owner}.{field}: {e}"))
    }

    #[test]
    fn every_clock_field_is_walked() {
        for (owner, fields) in CLOCK_FIELDS {
            for &field in *fields {
                let spec = with_duration(owner, field);
                assert!(
                    spec.clock_spans()
                        .iter()
                        .any(|(f, s)| *f == field && s.is_duration()),
                    "{owner}.{field} is in CLOCK_FIELDS but clock_spans does not read it"
                );
            }
        }
    }

    fn spec(extra: &str) -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x"]{extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn a_spec_is_numbers_durations_or_neither() {
        let durations = spec(r#", "clock": "t", "half_life": "10m", "gap_cap": "5m""#);
        assert_eq!(
            durations.clock_scale(),
            Ok(ClockScale::Durations("half_life"))
        );
        let numbers = spec(r#", "clock": "t", "half_life": 600, "gap_cap": 300"#);
        assert_eq!(numbers.clock_scale(), Ok(ClockScale::Numbers("half_life")));
        // 0 and inf mean the same in every unit, so they bind a spec to neither.
        let free = spec(r#", "clock": "t", "half_life": "inf""#);
        assert_eq!(free.clock_scale(), Ok(ClockScale::Free));
        let beside = spec(r#", "clock": "t", "half_life": "inf", "gap_cap": "5m""#);
        assert_eq!(beside.clock_scale(), Ok(ClockScale::Durations("gap_cap")));
        assert!(durations.validate().is_ok());
    }

    #[test]
    fn a_mixture_is_refused_naming_both_parameters() {
        let err = spec(r#", "clock": "t", "half_life": "10m", "gap_cap": 300"#)
            .validate()
            .unwrap_err();
        assert!(
            err.contains("half_life is a duration") && err.contains("gap_cap is a plain number"),
            "{err}"
        );
        let err = spec(r#", "clock": "t", "lam": 0.99, "gap_cap": "5m""#)
            .validate()
            .unwrap_err();
        assert!(err.contains("lam is a decay per clock unit"), "{err}");
    }

    #[test]
    fn a_duration_needs_a_clock() {
        let err = spec(r#", "half_life": "10m""#).validate().unwrap_err();
        assert!(err.contains("needs a clock column"), "{err}");
    }

    #[test]
    fn a_duration_is_read_in_seconds_and_names_a_grid_as_written() {
        let s = spec(r#", "clock": "t", "half_life": ["5m", "1h"], "gap_cap": "1d""#);
        let decays = s.decays().unwrap();
        let labels: Vec<&str> = decays.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(labels, vec!["@h5m", "@h1h"]);
        assert_eq!(
            decays[0].1,
            online_core::Decay::Halflife(300.0),
            "a duration is read in seconds"
        );
        let dup = spec(r#", "clock": "t", "half_life": ["5m", "300s"], "gap_cap": "1d""#);
        let err = dup.decays().unwrap_err();
        assert!(err.contains("300s more than once"), "{err}");
        assert!(matches!(
            s.half_life.as_ref().unwrap().spans()[0],
            Span::Duration(_)
        ));
    }

    /// Task 160, PB2: the cap is on the clock's step, and without a clock a
    /// row is one step. A cap of 0.5 was taken and halved every decay step,
    /// marked every row capped and cleared every lag at every row.
    #[test]
    fn gap_cap_needs_a_clock() {
        let err = spec(r#", "half_life": 10, "gap_cap": 0.5"#)
            .check()
            .unwrap_err();
        assert!(err.contains("spec \"m\": gap_cap needs clock"), "{err}");
        assert!(
            spec(r#", "clock": "t", "half_life": 10, "gap_cap": 0.5"#)
                .check()
                .is_ok()
        );
    }

    /// Task 173, PC1: a window target under `group` is one window core per
    /// group on that group's clock, which without a clock column counts
    /// its own rows, so a group that falls silent leaves its windows open
    /// with nothing to cut them. Refused with `with_windows`' message, the
    /// rule stated once (`clock_cfg_of`); with a clock, or without groups,
    /// the same spec is a spec.
    #[test]
    fn a_formula_target_with_group_needs_a_clock() {
        let formula = |extra: &str| {
            serde_json::from_str::<Spec>(&format!(
                r#"{{"name": "m", "model": {{"type": "ewridge"}}, "features": ["x"],
                    "targets": [{{"name": "fwd", "formula": ["-", ["rewm_mean", ["col", "mid"],
                        {{"half_life": 5, "window_size": 10}}], ["col", "mid"]]}}],
                    "half_life": 50, "embargo": 10{extra}}}"#
            ))
            .unwrap()
        };
        let err = formula(r#", "group": "g""#).check().unwrap_err();
        assert!(
            err.starts_with("spec \"m\": a window looking ahead with group needs a clock column")
                && err.ends_with("Name a clock column, with gap_cap, or leave out group"),
            "{err}"
        );
        assert_eq!(
            formula(r#", "group": "g", "clock": "t", "gap_cap": 100"#).check(),
            Ok(())
        );
        assert_eq!(formula("").check(), Ok(()));
        // A plain target under groups and no clock is a spec as before.
        assert_eq!(spec(r#", "half_life": 10, "group": "g""#).check(), Ok(()));
    }

    /// Task 160, PB3: an empty half-life grid built no model instance.
    #[test]
    fn an_empty_half_life_grid_is_refused() {
        let err = spec(r#", "half_life": []"#).check().unwrap_err();
        assert!(
            err.contains("spec \"m\": half_life names no half-life; give one or a grid"),
            "{err}"
        );
    }

    /// The drift detector sums each row's excess times its clock step, so
    /// its threshold is sigma times clock units: a number of the clock
    /// column's units, a duration on a temporal clock, and required with a
    /// clock, as `gap_cap` is; 20 without one, where a row is one unit and
    /// it is the classic test (task 168). The default 20 on a temporal clock
    /// was 20 sigma-seconds, a unit no one chose, and fired on noise at a
    /// row a minute (review 2026-10-05, CC1).
    #[test]
    fn drift_threshold_is_a_clock_parameter() {
        let durations =
            r#", "clock": "t", "half_life": "10m", "gap_cap": "5m", "emit_drift": true"#;
        let err = spec(durations).validate().unwrap_err();
        assert!(
            err.contains("spec \"m\": drift_threshold is required when clock is given"),
            "{err}"
        );
        let given = spec(&format!(r#"{durations}, "drift_threshold": "20m""#));
        assert_eq!(given.validate(), Ok(()));
        assert_eq!(
            given.drift_threshold.as_ref().map(Span::value),
            Some(1200.0),
            "a duration is read on the clock's scale, seconds"
        );
        let err = spec(&format!(r#"{durations}, "drift_threshold": 20"#))
            .validate()
            .unwrap_err();
        assert!(err.contains("drift_threshold is a plain number"), "{err}");
        let numbers = r#", "clock": "t", "half_life": 600, "gap_cap": 300, "emit_drift": true"#;
        assert_eq!(
            spec(&format!(r#"{numbers}, "drift_threshold": 20"#)).validate(),
            Ok(())
        );
        let err = spec(&format!(r#"{numbers}, "drift_threshold": "20m""#))
            .validate()
            .unwrap_err();
        assert!(err.contains("drift_threshold is a duration"), "{err}");
        for bad in ["0", "-1", "\"inf\""] {
            let err = spec(&format!(r#"{numbers}, "drift_threshold": {bad}"#))
                .validate()
                .unwrap_err();
            assert!(
                err.contains("drift_threshold must be finite and > 0"),
                "{bad}: {err}"
            );
        }
        // Without a clock a row is one unit: 20 is the classic test.
        assert_eq!(
            spec(r#", "half_life": 600, "emit_drift": true"#).validate(),
            Ok(())
        );
    }

    /// `coef_every` reads the clock, as `solve_every` does (task 178): a
    /// number of the clock column's units or a duration on a temporal clock,
    /// the mixtures refused as for every clock parameter; `0` every row and
    /// free of a unit; finite and `>= 0`. `max_rows_between_coefs` is at
    /// least 1, and both need a model with coefficients, `0` included, which
    /// was the default and passed there.
    #[test]
    fn coef_every_is_a_clock_parameter_and_its_row_cap_a_count() {
        let durations = r#", "clock": "t", "half_life": "10m", "gap_cap": "5m""#;
        let given = spec(&format!(r#"{durations}, "coef_every": "15m""#));
        assert_eq!(given.validate(), Ok(()));
        assert_eq!(given.coef_every.as_ref().map(Span::value), Some(900.0));
        let err = spec(&format!(r#"{durations}, "coef_every": 900"#))
            .validate()
            .unwrap_err();
        assert!(err.contains("coef_every is a plain number"), "{err}");
        // `0` means the same in every unit, so a temporal spec takes it.
        assert_eq!(
            spec(&format!(r#"{durations}, "coef_every": 0"#)).validate(),
            Ok(())
        );
        let numbers = r#", "clock": "t", "half_life": 600, "gap_cap": 300"#;
        assert_eq!(
            spec(&format!(r#"{numbers}, "coef_every": 900"#)).validate(),
            Ok(())
        );
        let err = spec(&format!(r#"{numbers}, "coef_every": "15m""#))
            .validate()
            .unwrap_err();
        assert!(err.contains("coef_every is a duration"), "{err}");
        for bad in ["-1", "-0.5", "\"inf\""] {
            let err = spec(&format!(r#"{numbers}, "coef_every": {bad}"#))
                .validate()
                .unwrap_err();
            assert!(
                err.contains("coef_every must be finite and >= 0 clock units"),
                "{bad}: {err}"
            );
        }
        let err = spec(r#", "half_life": 60, "max_rows_between_coefs": 0"#)
            .validate()
            .unwrap_err();
        assert!(err.contains("max_rows_between_coefs must be >= 1"), "{err}");
        assert_eq!(
            spec(r#", "half_life": 60, "coef_every": 7, "max_rows_between_coefs": 1"#).validate(),
            Ok(())
        );
        for cadence in [r#""coef_every": 0"#, r#""max_rows_between_coefs": 5"#] {
            let cov: Spec = serde_json::from_str(&format!(
                r#"{{"name": "c", "model": {{"type": "ew_cov"}}, "targets": ["x"],
                    "features": ["x", "z"], "half_life": 60, {cadence}}}"#
            ))
            .unwrap();
            let err = cov.validate().unwrap_err();
            let key = cadence.split('"').nth(1).unwrap();
            assert!(
                err.contains(&format!(
                    "{key} does not apply to ew_cov (it reports no coefficients)"
                )),
                "{err}"
            );
        }
    }

    /// Task 160, PB9: no delay is no embargo, not an embargo of 0.
    #[test]
    fn a_zero_embargo_is_told_to_leave_it_out() {
        let err = spec(r#", "half_life": 10, "embargo": 0"#)
            .check()
            .unwrap_err();
        assert!(
            err.contains("embargo must be finite and > 0 (got 0); leave it out for no delay"),
            "{err}"
        );
    }
}

#[cfg(test)]
mod sgd_tests;
