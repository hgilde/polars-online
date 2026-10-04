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
//!
//! A target may also be a **formula of the row's future** (docs/PLAN.md task
//! 104): a table with a `name` and a `formula`, task 143's compact tree of
//! window operators -- `po.rewm_mean("mid", half_life="10s",
//! window_size="1m") - pl.col("mid")` in Python, the same tree in TOML --
//! holding at least one forward operator. Its value is not known at its row:
//! the bank's window core resolves it when the window closes, and the row is
//! learned from then, under the spec's `embargo`.

use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::formula::Node;

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
/// taken against, and how; or, for a formula target, the formula of the
/// row's future (`column` is then empty and never read).
#[derive(Debug, Clone, PartialEq)]
pub struct TargetDef {
    pub name: String,
    pub column: String,
    pub relative_to: Option<String>,
    pub relative: Relative,
    /// A formula over window operators, at least one looking ahead
    /// (docs/PLAN.md task 104); `None` for a column target.
    pub formula: Option<Node>,
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
            formula: None,
        }
    }

    /// A formula target under `name`.
    pub fn formula(name: impl Into<String>, tree: Node) -> Self {
        Self {
            name: name.into(),
            column: String::new(),
            relative_to: None,
            relative: Relative::Difference,
            formula: Some(tree),
        }
    }

    /// A column read as it is, under its own name: what a string target is.
    pub fn is_plain(&self) -> bool {
        self.formula.is_none() && self.relative_to.is_none() && self.name == self.column
    }

    /// Whether the target is a formula of the row's future, resolved by the
    /// bank's window core rather than read from a column.
    pub fn is_formula(&self) -> bool {
        self.formula.is_some()
    }

    /// The column the target's value is read from, at its own row: `None`
    /// for a formula target.
    pub fn value_column(&self) -> Option<&str> {
        if self.formula.is_some() {
            None
        } else {
            Some(self.column.as_str())
        }
    }

    /// Every column the target reads: its own and its reference, or the
    /// columns of its formula.
    pub fn columns(&self) -> Vec<String> {
        match &self.formula {
            Some(tree) => tree.columns(),
            None => std::iter::once(self.column.clone())
                .chain(self.relative_to.clone())
                .collect(),
        }
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

    /// Whether any target is a formula of the row's future.
    pub fn any_formula(&self) -> bool {
        self.defs.iter().any(TargetDef::is_formula)
    }

    /// The positions of the formula targets, in `targets` order.
    pub fn formula_slots(&self) -> Vec<usize> {
        self.defs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.is_formula().then_some(i))
            .collect()
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
                     `relative` and `name`, or a table with `name` and `formula`",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative: Option<Relative>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    formula: Option<Node>,
}

impl Serialize for Targets {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let written: Vec<Written> = self
            .defs
            .iter()
            .map(|d| {
                if d.is_plain() {
                    Written::Name(d.column.clone())
                } else if let Some(tree) = &d.formula {
                    Written::Table(Table {
                        column: None,
                        relative: None,
                        relative_to: None,
                        name: Some(d.name.clone()),
                        formula: Some(tree.clone()),
                    })
                } else {
                    Written::Table(Table {
                        column: Some(d.column.clone()),
                        relative: d.relative_to.as_ref().map(|_| d.relative),
                        relative_to: d.relative_to.clone(),
                        name: (d.name != d.column).then(|| d.name.clone()),
                        formula: None,
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
                    // A formula target (docs/PLAN.md task 104): a name and
                    // the tree, nothing of a column target's.
                    if let Some(tree) = t.formula {
                        let Some(name) = t.name else {
                            return Err(serde::de::Error::custom(
                                "a formula target needs a name, which its output fields \
                                 carry (pred_<name> and the rest)",
                            ));
                        };
                        if name.is_empty() {
                            return Err(serde::de::Error::custom(
                                "formula target: name must not be empty",
                            ));
                        }
                        if t.column.is_some() || t.relative_to.is_some() || t.relative.is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "target {name:?}: a formula target has a name and a formula and \
                                 no column; a relative target is a column taken against \
                                 another, so put the subtraction in the formula instead"
                            )));
                        }
                        if !tree.has_forward_operator() {
                            // Name the call that makes the column: with_windows
                            // for a formula over operators looking back, Polars'
                            // with_columns for one of the row alone, which
                            // with_windows refuses in turn.
                            let make = if tree.operators().is_empty() {
                                "Polars' with_columns, since it holds no operator at all"
                            } else {
                                "po.stream.with_windows"
                            };
                            return Err(serde::de::Error::custom(format!(
                                "target {name:?}: a formula target holds at least one operator \
                                 looking ahead (rewm_mean, rewm_sum or rewm_rate); a formula \
                                 known at its own row is a column: make it with {make}"
                            )));
                        }
                        defs.push(TargetDef::formula(name, tree));
                        continue;
                    }
                    let Some(column) = t.column else {
                        return Err(serde::de::Error::custom(
                            "a target table names a `column`, or a `name` and a `formula`",
                        ));
                    };
                    if t.relative.is_some() && t.relative_to.is_none() {
                        return Err(serde::de::Error::custom(format!(
                            "target {:?}: relative needs relative_to, the column of the \
                             same row the target is taken against",
                            column
                        )));
                    }
                    // An empty name would give fields called `pred_`; an
                    // empty column or reference names no column (review
                    // 2026-09-26, D missing 5).
                    for (what, value) in [
                        ("column", Some(&column)),
                        ("name", t.name.as_ref()),
                        ("relative_to", t.relative_to.as_ref()),
                    ] {
                        if value.is_some_and(String::is_empty) {
                            return Err(serde::de::Error::custom(format!(
                                "target {:?}: {what} must not be empty",
                                column
                            )));
                        }
                    }
                    TargetDef {
                        name: t.name.unwrap_or_else(|| column.clone()),
                        column,
                        relative_to: t.relative_to,
                        relative: t.relative.unwrap_or_default(),
                        formula: None,
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
                formula: None,
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

    /// A formula target: a name and task 143's tree, written back as it
    /// was read, in JSON and in TOML (docs/PLAN.md task 104). Its columns
    /// are the tree's; it reads from no column of its own.
    #[test]
    fn a_formula_target_is_a_name_and_a_tree() {
        let tree = r#"["-", ["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]"#;
        let t = json(&format!(r#"["y", {{"name": "fwd", "formula": {tree}}}]"#)).unwrap();
        assert_eq!(t.as_slice(), ["y", "fwd"]);
        assert!(t.any_formula());
        assert_eq!(t.formula_slots(), [1]);
        let f = &t.defs()[1];
        assert!(f.is_formula() && !f.is_plain());
        assert_eq!(f.value_column(), None);
        assert_eq!(f.columns(), ["mid"]);
        assert_eq!(t.defs()[0].columns(), ["y"]);
        let written = serde_json::to_string(&t).unwrap();
        assert!(
            written
                .starts_with(r#"["y",{"name":"fwd","formula":["-",["rewm_mean",["col","mid"],{"#),
            "{written}"
        );
        assert_eq!(json(&written).unwrap(), t);
        #[derive(Deserialize)]
        struct Holder {
            targets: Targets,
        }
        let h: Holder = toml::from_str(
            r#"targets = ["y", { name = "fwd", formula = ["-", ["rewm_mean", ["col", "mid"], { half_life = "10s", window_size = "1m" }], ["col", "mid"]] }]"#,
        )
        .unwrap();
        assert_eq!(h.targets, t);
    }

    /// What a formula target may not be: nameless, a column too, or a
    /// formula known at its own row.
    #[test]
    fn a_formula_target_that_is_not_one_is_refused() {
        let fwd = r#"["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}]"#;
        let err = json(&format!(r#"[{{"formula": {fwd}}}]"#)).unwrap_err();
        assert!(err.contains("needs a name"), "{err}");
        let err = json(&format!(
            r#"[{{"name": "f", "column": "mid", "formula": {fwd}}}]"#
        ))
        .unwrap_err();
        assert!(err.contains("no column"), "{err}");
        let back = r#"["ewm_mean", ["col", "mid"], {"half_life": "10s"}]"#;
        let err = json(&format!(r#"[{{"name": "f", "formula": {back}}}]"#)).unwrap_err();
        assert!(err.contains("looking ahead"), "{err}");
        // A formula that looks back is a column with_windows makes; one of the
        // row alone is Polars' own with_columns, which with_windows refuses.
        // Each refusal names the call that works.
        assert!(err.contains("po.stream.with_windows"), "{err}");
        let plain = r#"["-", ["col", "price"], ["col", "mid"]]"#;
        let err = json(&format!(r#"[{{"name": "f", "formula": {plain}}}]"#)).unwrap_err();
        assert!(err.contains("looking ahead"), "{err}");
        assert!(err.contains("with_columns"), "{err}");
        assert!(!err.contains("with_windows"), "{err}");
        let err = json(r#"[{"name": "f"}]"#).unwrap_err();
        assert!(
            err.contains("names a `column`, or a `name` and a `formula`"),
            "{err}"
        );
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
