//! A spec's targets, each a column to learn against or a column taken
//! against another of its own row (docs/PLAN.md task 107a).
//!
//! The user, 2026-09-11: "every target should have the option to be
//! relative". A level -- a price, a VWAP -- is rarely what a regression
//! should predict; where it goes from *now* is. So a target may name a
//! reference column, read **at the target's own row**, and a way to take the
//! target against it:
//!
//! ```text
//! "difference"  y − r
//! "ratio"       y / r        where y > 0 and r > 0
//! "log_ratio"   ln(y / r)    where y > 0 and r > 0
//! ```
//!
//! `r` is known when the row arrives, so a relative target is exactly as
//! honest as the column it is built from: it adds no look-ahead. A value
//! either side cannot use -- null, not finite, past the input bound -- or,
//! for the two ratios, a `y` or `r` that is not positive, makes the target
//! null on that row: not learned from, never a NaN in the state (hard rule
//! 9). So does a result the model cannot use: a difference past the bound,
//! or a ratio that over- or underflows (three hundred orders of magnitude
//! apart, inside the bound on both sides), is null on that row too, by the
//! same rule every target is held to. Everything downstream is on the
//! relative scale: `pred`, `resid`, `sigma`, the metrics and the interval.
//!
//! On every surface a target is a string or a table: in Python
//! `po.target("price_5m", relative_to="mid")`, in the CLI's TOML
//! `targets = ["ret_5m", { column = "price_5m", relative_to = "mid" }]`. A
//! table's `name`, which the output fields carry, is its `column` unless
//! given. A spec whose targets are all plain columns writes them as the
//! strings it always did, so its bytes do not move.

use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// How a relative target is taken against its reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relative {
    /// `y − r`.
    #[default]
    Difference,
    /// `y / r`, where both are positive.
    Ratio,
    /// `ln(y / r)`, where both are positive.
    LogRatio,
}

impl Relative {
    /// The target's value from the column's `y` and the reference's `r`, both
    /// already usable ([`crate::stream::usable`]): NaN, a null target, where
    /// a ratio has a side that is not positive.
    #[inline]
    pub fn of(self, y: f64, r: f64) -> f64 {
        match self {
            Relative::Difference => y - r,
            Relative::Ratio if y > 0.0 && r > 0.0 => y / r,
            Relative::LogRatio if y > 0.0 && r > 0.0 => (y / r).ln(),
            Relative::Ratio | Relative::LogRatio => f64::NAN,
        }
    }
}

/// One target: the name its output fields carry, the column its values are
/// read from, and for a relative target the column of the same row they are
/// taken against, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetDef {
    pub name: String,
    pub column: String,
    pub relative_to: Option<String>,
    pub relative: Relative,
}

impl TargetDef {
    /// A plain column, named after itself.
    pub fn plain(column: impl Into<String>) -> Self {
        let column = column.into();
        Self {
            name: column.clone(),
            column,
            relative_to: None,
            relative: Relative::Difference,
        }
    }

    /// A column read as it is, under its own name: what a string target is.
    pub fn is_plain(&self) -> bool {
        self.relative_to.is_none() && self.name == self.column
    }
}

/// A spec's targets. Reads as the list of their names, which is what nearly
/// every use wants; [`Targets::defs`] has the rest.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Targets {
    names: Vec<String>,
    defs: Vec<TargetDef>,
}

impl Targets {
    /// Each target in order, with its column and reference.
    pub fn defs(&self) -> &[TargetDef] {
        &self.defs
    }

    /// Whether any target is taken against a reference.
    pub fn any_relative(&self) -> bool {
        self.defs.iter().any(|d| d.relative_to.is_some())
    }
}

impl Deref for Targets {
    type Target = Vec<String>;

    fn deref(&self) -> &Vec<String> {
        &self.names
    }
}

impl<'a> IntoIterator for &'a Targets {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.names.iter()
    }
}

impl From<Vec<TargetDef>> for Targets {
    fn from(defs: Vec<TargetDef>) -> Self {
        Self {
            names: defs.iter().map(|d| d.name.clone()).collect(),
            defs,
        }
    }
}

impl From<Vec<String>> for Targets {
    fn from(columns: Vec<String>) -> Self {
        columns
            .into_iter()
            .map(TargetDef::plain)
            .collect::<Vec<_>>()
            .into()
    }
}

impl<'a> From<Vec<&'a str>> for Targets {
    fn from(columns: Vec<&'a str>) -> Self {
        columns
            .into_iter()
            .map(TargetDef::plain)
            .collect::<Vec<_>>()
            .into()
    }
}

/// A target as it is written: a column's name, or a table.
#[derive(Serialize)]
#[serde(untagged)]
enum Written {
    Name(String),
    Table(Table),
}

/// A string or a table, each read by its own rules, so a table with a
/// misspelt key says which key -- where an untagged enum says only that
/// nothing matched. A table is a map, never a positional array: its absent
/// fields are skipped, so an array would be ambiguous (the compact-encoding
/// rule, `online-core`'s `tests/state_encoding.rs`), and a spec is written
/// named.
impl<'de> Deserialize<'de> for Written {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Written;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "a column name, or a table with `column` and optionally `relative_to`, \
                     `relative` and `name`",
                )
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Written, E> {
                Ok(Written::Name(v.to_owned()))
            }

            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Written, E> {
                Ok(Written::Name(v))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<Written, A::Error> {
                Table::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                    .map(Written::Table)
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    column: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative: Option<Relative>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl Serialize for Targets {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let written: Vec<Written> = self
            .defs
            .iter()
            .map(|d| {
                if d.is_plain() {
                    Written::Name(d.column.clone())
                } else {
                    Written::Table(Table {
                        column: d.column.clone(),
                        relative: d.relative_to.as_ref().map(|_| d.relative),
                        relative_to: d.relative_to.clone(),
                        name: (d.name != d.column).then(|| d.name.clone()),
                    })
                }
            })
            .collect();
        written.serialize(s)
    }
}

impl<'de> Deserialize<'de> for Targets {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let written = Vec::<Written>::deserialize(d)?;
        let mut defs = Vec::with_capacity(written.len());
        for w in written {
            defs.push(match w {
                Written::Name(column) => TargetDef::plain(column),
                Written::Table(t) => {
                    if t.relative.is_some() && t.relative_to.is_none() {
                        return Err(serde::de::Error::custom(format!(
                            "target {:?}: relative needs relative_to, the column of the \
                             same row the target is taken against",
                            t.column
                        )));
                    }
                    // An empty name would give fields called `pred_`; an
                    // empty column or reference names no column (review
                    // 2026-09-26, D missing 5).
                    for (what, value) in [
                        ("column", Some(&t.column)),
                        ("name", t.name.as_ref()),
                        ("relative_to", t.relative_to.as_ref()),
                    ] {
                        if value.is_some_and(String::is_empty) {
                            return Err(serde::de::Error::custom(format!(
                                "target {:?}: {what} must not be empty",
                                t.column
                            )));
                        }
                    }
                    TargetDef {
                        name: t.name.unwrap_or_else(|| t.column.clone()),
                        column: t.column,
                        relative_to: t.relative_to,
                        relative: t.relative.unwrap_or_default(),
                    }
                }
            });
        }
        Ok(defs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(v: &str) -> Result<Targets, String> {
        serde_json::from_str(v).map_err(|e| e.to_string())
    }

    /// A string is a plain column and writes back as the string, so a spec
    /// without a relative target keeps its bytes; a table writes back as a
    /// table, with only what it said.
    #[test]
    fn a_target_is_a_string_or_a_table_and_writes_back_as_it_was() {
        let t = json(r#"["y", {"column": "p", "relative_to": "mid"}]"#).unwrap();
        assert_eq!(t.as_slice(), ["y", "p"], "it reads as the names");
        assert_eq!(t.defs()[0], TargetDef::plain("y"));
        assert_eq!(
            t.defs()[1],
            TargetDef {
                name: "p".into(),
                column: "p".into(),
                relative_to: Some("mid".into()),
                relative: Relative::Difference,
            }
        );
        assert!(t.any_relative());
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            r#"["y",{"column":"p","relative_to":"mid","relative":"difference"}]"#
        );
        let plain = json(r#"["a", "b"]"#).unwrap();
        assert_eq!(serde_json::to_string(&plain).unwrap(), r#"["a","b"]"#);
        assert!(!plain.any_relative());
        // Named apart from its column, with no reference: a renamed column.
        let named = json(r#"[{"column": "p", "name": "price"}]"#).unwrap();
        assert_eq!(named.as_slice(), ["price"]);
        assert_eq!(
            serde_json::to_string(&named).unwrap(),
            r#"[{"column":"p","name":"price"}]"#
        );
    }

    /// The CLI's TOML form, as docs/PLAN.md task 107 writes it.
    #[test]
    fn the_toml_table_form_reads_the_same() {
        #[derive(Deserialize)]
        struct Holder {
            targets: Targets,
        }
        let h: Holder = toml::from_str(
            r#"targets = ["ret_5m", { column = "price_5m", relative_to = "mid", relative = "log_ratio" }]"#,
        )
        .unwrap();
        let want = json(
            r#"["ret_5m", {"column": "price_5m", "relative_to": "mid", "relative": "log_ratio"}]"#,
        )
        .unwrap();
        assert_eq!(h.targets, want);
        assert_eq!(h.targets.defs()[1].relative, Relative::LogRatio);
    }

    #[test]
    fn a_table_that_says_too_little_or_too_much_is_refused() {
        let err = json(r#"[{"column": "p", "relative": "ratio"}]"#).unwrap_err();
        assert!(err.contains("relative needs relative_to"), "{err}");
        let err = json(r#"[{"column": "p", "relative_to": "m", "relative": "sum"}]"#).unwrap_err();
        assert!(err.contains("unknown variant `sum`"), "{err}");
        let err = json(r#"[{"column": "p", "against": "m"}]"#).unwrap_err();
        assert!(err.contains("unknown field `against`"), "{err}");
        let err = json(r#"[3]"#).unwrap_err();
        assert!(err.contains("a column name, or a table"), "{err}");
    }

    /// The three ways, and a ratio's refusal of a side that is not positive.
    #[test]
    fn the_three_ways_to_take_a_target() {
        assert_eq!(Relative::Difference.of(5.0, 2.0), 3.0);
        assert_eq!(Relative::Difference.of(-5.0, 2.0), -7.0);
        assert_eq!(Relative::Ratio.of(5.0, 2.0), 2.5);
        assert_eq!(Relative::LogRatio.of(5.0, 2.0), 2.5_f64.ln());
        for how in [Relative::Ratio, Relative::LogRatio] {
            assert!(how.of(0.0, 2.0).is_nan());
            assert!(how.of(5.0, 0.0).is_nan());
            assert!(how.of(-5.0, 2.0).is_nan());
            assert!(how.of(5.0, -2.0).is_nan());
        }
    }
}
