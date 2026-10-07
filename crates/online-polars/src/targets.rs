//! A spec's targets, each a column to learn against, under its own name or
//! another, or a formula of the row's future.
//!
//! On every surface a target is a string or a table: in Python a column's
//! name or `po.target("p", name="price")`, in the CLI's TOML
//! `targets = ["ret_5m", { column = "p", name = "price" }]`. A table's
//! `name`, which the output fields carry, is its `column` unless given. A
//! spec whose targets are all plain columns writes them as strings, so its
//! bytes do not move.
//!
//! A target may also be a **formula of the row's future** (docs/PLAN.md task
//! 104): a table with a `name` and a `formula`, task 143's compact tree of
//! window operators -- `po.rewm_mean("mid", half_life="10s",
//! window_size="1m") - pl.col("mid")` in Python, the same tree in TOML --
//! holding at least one forward operator. Its value is not known at its row:
//! the bank's window core resolves it when the window closes, and the row is
//! learned from then, under the spec's `embargo`.
//!
//! **Relative targets were removed** (docs/PLAN.md task 201). Task 107a let a
//! table name a reference column of its own row, `relative_to`, and a way to
//! take the target against it, `relative` (a difference, a ratio or a log
//! ratio). Polars computes the same target as a column -- `with_columns(
//! ret=pl.col("p") - pl.col("mid"))`, in the columns' own type -- and the bank
//! learned it to the bit, so the table form duplicated an expression. A table
//! naming either key is refused by name, saying to derive the column
//! upstream.

use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::formula::Node;

/// What a table naming `relative_to` or `relative` is told: the two keys
/// were task 107a's relative target, which task 201 removed.
pub const RELATIVE_REMOVED: &str = "relative targets were removed; derive the target column \
     upstream (in Polars, with_columns(ret=pl.col(\"p\") - pl.col(\"mid\")), or for a return \
     the log ratio (pl.col(\"p\") / pl.col(\"mid\")).log()) and name it in targets";

/// One target: the name its output fields carry and the column its values
/// are read from; or, for a formula target, the formula of the row's future
/// (`column` is then empty and never read).
#[derive(Debug, Clone, PartialEq)]
pub struct TargetDef {
    pub name: String,
    pub column: String,
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
            formula: None,
        }
    }

    /// A formula target under `name`.
    pub fn formula(name: impl Into<String>, tree: Node) -> Self {
        Self {
            name: name.into(),
            column: String::new(),
            formula: Some(tree),
        }
    }

    /// A column read as it is, under its own name: what a string target is.
    pub fn is_plain(&self) -> bool {
        self.formula.is_none() && self.name == self.column
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

    /// Every column the target reads: its own, or the columns of its
    /// formula.
    pub fn columns(&self) -> Vec<String> {
        match &self.formula {
            Some(tree) => tree.columns(),
            None => vec![self.column.clone()],
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
    /// Each target in order, with its column or formula.
    pub fn defs(&self) -> &[TargetDef] {
        &self.defs
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
                    "a column name, or a table with `column` and optionally `name`, or a table \
                     with `name` and `formula`",
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
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    formula: Option<Node>,
    /// Task 107a's two keys, read only to be refused by name
    /// ([`RELATIVE_REMOVED`]) where `deny_unknown_fields` would say only
    /// "unknown field"; never written.
    #[serde(default, skip_serializing)]
    relative_to: Option<serde::de::IgnoredAny>,
    #[serde(default, skip_serializing)]
    relative: Option<serde::de::IgnoredAny>,
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
                        column: d.formula.is_none().then(|| d.column.clone()),
                        name: Some(d.name.clone()),
                        formula: d.formula.clone(),
                        relative_to: None,
                        relative: None,
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
                    if t.relative_to.is_some() || t.relative.is_some() {
                        let which = t.column.as_ref().or(t.name.as_ref());
                        return Err(serde::de::Error::custom(match which {
                            Some(c) => format!("target {c:?}: {RELATIVE_REMOVED}"),
                            None => RELATIVE_REMOVED.to_string(),
                        }));
                    }
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
                        if t.column.is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "target {name:?}: a formula target has a name and a formula and \
                                 no column; to take a window against a column of the row, put \
                                 the subtraction in the formula"
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
                    // An empty name would give fields called `pred_`; an
                    // empty column names no column (review 2026-09-26, D
                    // missing 5).
                    for (what, value) in [("column", Some(&column)), ("name", t.name.as_ref())] {
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
    /// of plain columns keeps its bytes; a table writes back as a table, with
    /// only what it said, and a table naming its own column is the column.
    #[test]
    fn a_target_is_a_string_or_a_table_and_writes_back_as_it_was() {
        let plain = json(r#"["a", "b"]"#).unwrap();
        assert_eq!(serde_json::to_string(&plain).unwrap(), r#"["a","b"]"#);
        assert_eq!(plain.defs()[0], TargetDef::plain("a"));
        // Named apart from its column: a renamed column.
        let named = json(r#"["y", {"column": "p", "name": "price"}]"#).unwrap();
        assert_eq!(named.as_slice(), ["y", "price"], "it reads as the names");
        assert_eq!(
            named.defs()[1],
            TargetDef {
                name: "price".into(),
                column: "p".into(),
                formula: None,
            }
        );
        assert_eq!(named.defs()[1].columns(), ["p"]);
        assert!(!named.defs()[1].is_plain());
        assert_eq!(
            serde_json::to_string(&named).unwrap(),
            r#"["y",{"column":"p","name":"price"}]"#
        );
        let own = json(r#"[{"column": "p"}, {"column": "q", "name": "q"}]"#).unwrap();
        assert_eq!(serde_json::to_string(&own).unwrap(), r#"["p","q"]"#);
    }

    /// The CLI's TOML form.
    #[test]
    fn the_toml_table_form_reads_the_same() {
        #[derive(Deserialize)]
        struct Holder {
            targets: Targets,
        }
        let h: Holder =
            toml::from_str(r#"targets = ["ret_5m", { column = "p", name = "price" }]"#).unwrap();
        let want = json(r#"["ret_5m", {"column": "p", "name": "price"}]"#).unwrap();
        assert_eq!(h.targets, want);
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
        let err = json(r#"[{"column": "p", "against": "m"}]"#).unwrap_err();
        assert!(err.contains("unknown field `against`"), "{err}");
        let err = json(r#"[3]"#).unwrap_err();
        assert!(err.contains("a column name, or a table"), "{err}");
        for (table, what) in [
            (r#"{"column": "", "name": "p"}"#, "column"),
            (r#"{"column": "p", "name": ""}"#, "name"),
        ] {
            let err = json(&format!("[{table}]")).unwrap_err();
            assert!(err.contains(&format!("{what} must not be empty")), "{err}");
        }
    }

    /// Task 201: relative targets were removed. A table naming `relative_to`
    /// or `relative` -- a spec dict, the command line's TOML, a saved state
    /// -- is refused by name, saying what replaces it, where serde would
    /// have said only "unknown field".
    #[test]
    fn a_relative_target_is_refused_by_name() {
        let removed = "relative targets were removed; derive the target column upstream";
        let fwd = r#"["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}]"#;
        for table in [
            r#"{"column": "p", "relative_to": "mid"}"#.to_string(),
            r#"{"column": "p", "relative_to": "mid", "relative": "log_ratio"}"#.to_string(),
            r#"{"column": "p", "relative": "ratio"}"#.to_string(),
            r#"{"column": "p", "name": "r", "relative_to": "mid"}"#.to_string(),
            format!(r#"{{"name": "f", "formula": {fwd}, "relative_to": "mid"}}"#),
        ] {
            let err = json(&format!(r#"["y", {table}]"#)).unwrap_err();
            assert!(err.contains(removed), "{table}: {err}");
        }
        #[derive(Debug, Deserialize)]
        struct Holder {
            #[allow(dead_code)]
            targets: Targets,
        }
        let err = toml::from_str::<Holder>(
            r#"targets = ["ret_5m", { column = "price_5m", relative_to = "mid", relative = "log_ratio" }]"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains(removed), "{err}");
        assert!(err.contains(r#"target "price_5m""#), "{err}");
    }
}
