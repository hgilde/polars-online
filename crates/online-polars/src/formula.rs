//! A formula: a Polars expression over the window operators, kept as this
//! library's own compact tree (docs/PLAN.md task 143, decided 2026-10-02).
//!
//! Python reads an expression's own tree (`expr.meta.serialize`) at build
//! time and walks seven node kinds into nested lists; this side reads the
//! lists and rebuilds the expression through Polars' public builders, which
//! are stable where the serialized tree is not:
//!
//! ```text
//! ["-", ["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]
//! ```
//!
//! Leaves are `["col", name]` and `["lit", value]`; the element-wise nodes
//! are the binary operators `+ - * / ** == != < <= > >= & |`, the unary
//! `neg abs exp sqrt is_null is_not_null`, `["log", x, base]`,
//! `["clip", x, lo, hi]` (either bound `["lit"]`, a bare `null` read too), `["fill_null", x, y]`,
//! `["when", p, then, else]`, `["cast", x, "Float64"]` and
//! `["alias", x, name]`. An operator is `[name, input, {params}]` for
//! `ewm_mean`, `rewm_mean`, `ewm_sum`, `rewm_sum`, `ewm_rate` and
//! `rewm_rate`, and `["increment", input]`. Nothing else is read, so a
//! `shift`, a `cum_sum`, a rolling window or an aggregation -- which would
//! depend on the chunking (hard rule 3) -- is refused by name, at build time.

use std::fmt;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::span::Span;
use crate::windows::{Closed, Direction, OpKind, Partial, Stat};

/// A literal, as JSON carries it.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Literal {
    fn from_json(v: &Value) -> Result<Self, String> {
        Ok(match v {
            Value::Null => Literal::Null,
            Value::Bool(b) => Literal::Bool(*b),
            Value::Number(n) => {
                match n.as_i64() {
                    Some(i) => Literal::Int(i),
                    None => Literal::Float(n.as_f64().ok_or_else(|| {
                        format!("a literal must be a number JSON can hold, got {n}")
                    })?),
                }
            }
            Value::String(s) => Literal::Str(s.clone()),
            other => return Err(format!("a literal must be a scalar, got {other}")),
        })
    }

    fn to_json(&self) -> Value {
        match self {
            Literal::Null => Value::Null,
            Literal::Bool(b) => json!(b),
            Literal::Int(i) => json!(i),
            Literal::Float(f) => json!(f),
            Literal::Str(s) => json!(s),
        }
    }

    fn to_expr(&self) -> Expr {
        match self {
            Literal::Null => lit(NULL),
            Literal::Bool(b) => lit(*b),
            Literal::Int(i) => lit(*i),
            Literal::Float(f) => lit(*f),
            Literal::Str(s) => lit(s.as_str()),
        }
    }
}

/// A window operator: what it computes, over which input, on which kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct OpNode {
    pub kind: OpKind,
    /// An element-wise formula of the row, `increment` nodes included; no
    /// window operator.
    pub input: Box<Node>,
    /// Required for a window operator; none for `increment`.
    pub half_life: Option<Span>,
    pub window_size: Option<Span>,
    pub closed: Closed,
    pub min_samples: u32,
    /// Unset takes the direction's default: `keep` backward, `null` forward.
    pub partial: Option<Partial>,
}

impl OpNode {
    /// Backward or forward; `increment` reads back one row and is backward.
    pub fn direction(&self) -> Direction {
        self.kind.direction()
    }

    /// The compact JSON of this node alone: what a hidden column is named by,
    /// so two formulas asking for one operator share it.
    pub fn key(&self) -> String {
        Node::Op(self.clone()).to_json().to_string()
    }

    fn params_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(h) = &self.half_life {
            m.insert(
                "half_life".into(),
                serde_json::to_value(h).expect("a span is JSON"),
            );
        }
        if let Some(w) = &self.window_size {
            m.insert(
                "window_size".into(),
                serde_json::to_value(w).expect("a span is JSON"),
            );
        }
        m.insert("closed".into(), json!(self.closed.name()));
        m.insert("min_samples".into(), json!(self.min_samples));
        if let Some(p) = self.partial {
            m.insert("partial".into(), json!(p.name()));
        }
        Value::Object(m)
    }
}

/// One node of a formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Col(String),
    Lit(Literal),
    /// An element-wise node: its name and its arguments, in the order the
    /// list form has them. `cast` and `alias` carry their text as a `Lit`.
    Call(String, Vec<Node>),
    Op(OpNode),
}

/// The element-wise nodes and their arities.
const CALLS: &[(&str, usize)] = &[
    ("+", 2),
    ("-", 2),
    ("*", 2),
    ("/", 2),
    ("**", 2),
    ("==", 2),
    ("!=", 2),
    ("<", 2),
    ("<=", 2),
    (">", 2),
    (">=", 2),
    ("&", 2),
    ("|", 2),
    ("neg", 1),
    ("abs", 1),
    ("exp", 1),
    ("sqrt", 1),
    ("is_null", 1),
    ("is_not_null", 1),
    ("log", 2),
    ("clip", 3),
    ("fill_null", 2),
    ("when", 3),
    ("cast", 2),
    ("alias", 2),
];

/// The dtypes a `cast` may name, as Polars spells them.
const DTYPES: &[(&str, DataType)] = &[
    ("Float64", DataType::Float64),
    ("Float32", DataType::Float32),
    ("Int64", DataType::Int64),
    ("Int32", DataType::Int32),
    ("Int16", DataType::Int16),
    ("Int8", DataType::Int8),
    ("UInt64", DataType::UInt64),
    ("UInt32", DataType::UInt32),
    ("UInt16", DataType::UInt16),
    ("UInt8", DataType::UInt8),
    ("Boolean", DataType::Boolean),
    ("String", DataType::String),
];

fn dtype_of(name: &str) -> Result<DataType, String> {
    DTYPES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t.clone())
        .ok_or_else(|| {
            format!(
                "cast to {name:?} is not read; a formula casts to one of {:?}",
                DTYPES.iter().map(|(n, _)| *n).collect::<Vec<_>>()
            )
        })
}

impl Node {
    /// Read the list form. An unknown node is refused by name: what is not
    /// element-wise never gets as far as a kernel.
    ///
    /// # Errors
    ///
    /// A node that is not a list headed by a name, an unknown name, a wrong
    /// arity, a bad literal, a `cast` to a dtype not read, an operator with
    /// bad parameters, or a window operator inside an operator's input.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let Value::Array(items) = v else {
            return Err(format!(
                "a formula node is a list headed by its kind, got {}",
                short(v)
            ));
        };
        let Some(Value::String(head)) = items.first() else {
            return Err(format!(
                "a formula node is a list headed by its kind, got {}",
                short(v)
            ));
        };
        let args = &items[1..];
        let arity = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!(
                    "{head:?} takes {n} argument{}, got {}",
                    if n == 1 { "" } else { "s" },
                    args.len()
                ))
            }
        };
        match head.as_str() {
            "col" => {
                arity(1)?;
                match &args[0] {
                    Value::String(s) => Ok(Node::Col(s.clone())),
                    other => Err(format!("\"col\" names a column, got {}", short(other))),
                }
            }
            "lit" => {
                // `["lit"]` is null, the form TOML can carry; `["lit", null]`
                // is read too.
                if args.is_empty() {
                    return Ok(Node::Lit(Literal::Null));
                }
                arity(1)?;
                Ok(Node::Lit(Literal::from_json(&args[0])?))
            }
            "cast" | "alias" => {
                // A cast is strict unless a third element says `"non_strict"`
                // (review R1, B2: Python's default is strict, Rust's `cast`
                // is not, and a failed cast became a silent null).
                let non_strict = head == "cast" && args.len() == 3;
                if !non_strict {
                    arity(2)?;
                }
                let x = Node::from_json(&args[0])?;
                let Value::String(text) = &args[1] else {
                    return Err(format!(
                        "{head:?} takes a name as its second argument, got {}",
                        short(&args[1])
                    ));
                };
                if head == "cast" {
                    dtype_of(text)?;
                }
                let mut call = vec![x, Node::Lit(Literal::Str(text.clone()))];
                if non_strict {
                    match &args[2] {
                        Value::String(s) if s == "non_strict" => {
                            call.push(Node::Lit(Literal::Str(s.clone())));
                        }
                        other => {
                            return Err(format!(
                                "\"cast\" takes \"non_strict\" as its third argument, got {}",
                                short(other)
                            ));
                        }
                    }
                }
                Ok(Node::Call(head.clone(), call))
            }
            name if CALLS.iter().any(|(n, _)| *n == name) => {
                let n = CALLS
                    .iter()
                    .find(|(c, _)| *c == name)
                    .map_or(0, |(_, a)| *a);
                arity(n)?;
                let nodes = args
                    .iter()
                    .map(|a| {
                        if name == "clip" && a.is_null() {
                            Ok(Node::Lit(Literal::Null))
                        } else {
                            Node::from_json(a)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Node::Call(name.to_string(), nodes))
            }
            "increment" => {
                arity(1)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator("increment")?;
                Ok(Node::Op(OpNode {
                    kind: OpKind::Increment,
                    input: Box::new(input),
                    half_life: None,
                    window_size: None,
                    closed: Closed::Right,
                    min_samples: 1,
                    partial: None,
                }))
            }
            name if OpKind::parse(name).is_some() => {
                let kind = OpKind::parse(name).expect("checked");
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(name)?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "{name:?} takes its parameters as an object, got {}",
                        short(&args[1])
                    ));
                };
                let mut op = OpNode {
                    kind,
                    input: Box::new(input),
                    half_life: None,
                    window_size: None,
                    closed: Closed::Right,
                    min_samples: 1,
                    partial: None,
                };
                for (k, v) in params {
                    match k.as_str() {
                        "half_life" => {
                            op.half_life = Some(span_of(name, k, v)?);
                        }
                        "window_size" => {
                            op.window_size = Some(span_of(name, k, v)?);
                        }
                        "closed" => {
                            op.closed =
                                Closed::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                                    format!(
                                        "{name}: closed must be \"right\", \"left\", \"both\" or \
                                     \"none\", got {}",
                                        short(v)
                                    )
                                })?;
                        }
                        "min_samples" => {
                            op.min_samples = v
                                .as_u64()
                                .and_then(|n| u32::try_from(n).ok())
                                .filter(|&n| n >= 1)
                                .ok_or_else(|| {
                                    format!(
                                        "{name}: min_samples must be a count of at least 1, got {}",
                                        short(v)
                                    )
                                })?;
                        }
                        "partial" => {
                            op.partial = Some(
                                Partial::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                                    format!(
                                        "{name}: partial must be \"keep\", \"null\" or \"drop\", \
                                         got {}",
                                        short(v)
                                    )
                                })?,
                            );
                        }
                        other => {
                            return Err(format!(
                                "{name}: unknown parameter {other:?}; the parameters are half_life, \
                                 window_size, closed, min_samples and partial"
                            ));
                        }
                    }
                }
                let Some(h) = &op.half_life else {
                    return Err(format!("{name}: half_life is required"));
                };
                if h.value().is_nan() || h.value() <= 0.0 {
                    return Err(format!("{name}: half_life must be above 0, got {h}"));
                }
                match &op.window_size {
                    Some(w) if !(w.value().is_finite() && w.value() > 0.0) => {
                        return Err(format!(
                            "{name}: window_size must be finite and above 0, got {w}"
                        ));
                    }
                    None if kind.direction() == Direction::Forward => {
                        return Err(format!(
                            "{name}: a forward operator needs a window_size; the rows ahead of a \
                             row have no end without one"
                        ));
                    }
                    None if h.value().is_infinite() && kind.stat() == Some(Stat::Mean) => {
                        return Err(format!(
                            "{name}: half_life = inf needs a window_size; a value held from \
                             before the first row has no finite mass to weigh it by"
                        ));
                    }
                    _ => {}
                }
                Ok(Node::Op(op))
            }
            other => Err(format!(
                "formula node {other:?} is not read: a formula is element-wise, from columns, \
                 literals, arithmetic, comparisons, log/exp/abs/sqrt/clip/fill_null, when/then, \
                 cast and alias over the window operators; a shift, a cumulative or rolling \
                 function or an aggregation would depend on the chunking"
            )),
        }
    }

    /// Read the list form from its text.
    ///
    /// # Errors
    ///
    /// As [`Node::from_json`], and text that is not JSON.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("not a formula: {e}"))?;
        Node::from_json(&v)
    }

    /// The list form.
    pub fn to_json(&self) -> Value {
        match self {
            Node::Col(c) => json!(["col", c]),
            // A null literal is `["lit"]`: TOML has no null (review R2, P3).
            Node::Lit(Literal::Null) => json!(["lit"]),
            Node::Lit(l) => json!(["lit", l.to_json()]),
            Node::Call(name, args) => {
                let mut items = vec![json!(name)];
                match name.as_str() {
                    "cast" | "alias" => {
                        items.push(args[0].to_json());
                        if let Node::Lit(Literal::Str(s)) = &args[1] {
                            items.push(json!(s));
                        }
                        // A non-strict cast carries its third argument
                        // (review R2, F2: it was dropped, and a saved spec
                        // ran a strict cast).
                        if let Some(Node::Lit(Literal::Str(s))) = args.get(2) {
                            items.push(json!(s));
                        }
                    }
                    // A bound left out is the null literal, `["lit"]`, the
                    // form TOML can carry, as Python writes it; a bare
                    // `null` is still read (task 159, F5).
                    "clip" => items.extend(args.iter().map(Node::to_json)),
                    _ => items.extend(args.iter().map(Node::to_json)),
                }
                Value::Array(items)
            }
            Node::Op(op) => {
                let mut items = vec![json!(op.kind.name()), op.input.to_json()];
                if op.kind != OpKind::Increment {
                    items.push(op.params_json());
                }
                Value::Array(items)
            }
        }
    }

    /// Every operator in the tree, outermost first, in reading order.
    pub fn operators(&self) -> Vec<&OpNode> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a OpNode>) {
        match self {
            Node::Col(_) | Node::Lit(_) => {}
            Node::Call(_, args) => args.iter().for_each(|a| a.collect(out)),
            Node::Op(op) => {
                out.push(op);
                op.input.collect(out);
            }
        }
    }

    /// The columns of the input the tree reads, `increment` inputs included.
    pub fn columns(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns(&self, out: &mut Vec<String>) {
        match self {
            Node::Col(c) => {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
            Node::Lit(_) => {}
            Node::Call(_, args) => args.iter().for_each(|a| a.collect_columns(out)),
            Node::Op(op) => op.input.collect_columns(out),
        }
    }

    /// Whether the tree holds a call of `name` anywhere (an operator's
    /// input included).
    pub fn contains_call(&self, name: &str) -> bool {
        match self {
            Node::Col(_) | Node::Lit(_) => false,
            Node::Call(n, args) => n == name || args.iter().any(|a| a.contains_call(name)),
            Node::Op(op) => op.input.contains_call(name),
        }
    }

    /// [`Self::columns`], borrowed from the tree.
    pub fn columns_ref(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_columns_ref(&mut out);
        out
    }

    fn collect_columns_ref<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Node::Col(c) => {
                if !out.contains(&c.as_str()) {
                    out.push(c.as_str());
                }
            }
            Node::Lit(_) => {}
            Node::Call(_, args) => args.iter().for_each(|a| a.collect_columns_ref(out)),
            Node::Op(op) => op.input.collect_columns_ref(out),
        }
    }

    /// Whether the tree holds a window operator (an `increment` is not one).
    pub fn has_window_operator(&self) -> bool {
        self.operators()
            .iter()
            .any(|op| op.kind != OpKind::Increment)
    }

    /// Whether the tree holds a forward operator.
    pub fn has_forward_operator(&self) -> bool {
        self.operators()
            .iter()
            .any(|op| op.direction() == Direction::Forward)
    }

    fn no_window_operator(&self, who: &str) -> Result<(), String> {
        if self.has_window_operator() {
            return Err(format!(
                "{who}: an operator's input is an element-wise formula of the row, increments \
                 included, not another window operator; a formula over an operator's output is \
                 a second with_windows call, or a target of the first's"
            ));
        }
        Ok(())
    }

    /// The Polars expression, with every operator resolved by `resolve` --
    /// to the hidden column that carries it, in a run.
    pub fn to_expr(&self, resolve: &dyn Fn(&OpNode) -> Expr) -> Expr {
        match self {
            Node::Col(c) => col(c.as_str()),
            Node::Lit(l) => l.to_expr(),
            Node::Op(op) => resolve(op),
            Node::Call(name, args) => {
                let e = |i: usize| args[i].to_expr(resolve);
                match name.as_str() {
                    "+" => e(0) + e(1),
                    "-" => e(0) - e(1),
                    "*" => e(0) * e(1),
                    // Python's `/`: a true division, whatever the dtypes.
                    "/" => binary_expr(e(0), Operator::TrueDivide, e(1)),
                    "**" => e(0).pow(e(1)),
                    "==" => e(0).eq(e(1)),
                    "!=" => e(0).neq(e(1)),
                    "<" => e(0).lt(e(1)),
                    "<=" => e(0).lt_eq(e(1)),
                    ">" => e(0).gt(e(1)),
                    ">=" => e(0).gt_eq(e(1)),
                    "&" => e(0).and(e(1)),
                    "|" => e(0).or(e(1)),
                    "neg" => -e(0),
                    "abs" => e(0).abs(),
                    "exp" => e(0).exp(),
                    "sqrt" => e(0).sqrt(),
                    "is_null" => e(0).is_null(),
                    "is_not_null" => e(0).is_not_null(),
                    "log" => e(0).log(e(1)),
                    "clip" => match (&args[1], &args[2]) {
                        (Node::Lit(Literal::Null), Node::Lit(Literal::Null)) => e(0),
                        (Node::Lit(Literal::Null), _) => e(0).clip_max(e(2)),
                        (_, Node::Lit(Literal::Null)) => e(0).clip_min(e(1)),
                        _ => e(0).clip(e(1), e(2)),
                    },
                    "fill_null" => e(0).fill_null(e(1)),
                    "when" => when(e(0)).then(e(1)).otherwise(e(2)),
                    "cast" => match &args[1] {
                        Node::Lit(Literal::Str(t)) => {
                            let dtype = dtype_of(t).expect("checked when read");
                            if args.len() == 3 {
                                e(0).cast(dtype)
                            } else {
                                e(0).strict_cast(dtype)
                            }
                        }
                        _ => unreachable!("cast carries its dtype as text"),
                    },
                    "alias" => match &args[1] {
                        Node::Lit(Literal::Str(n)) => e(0).alias(n.as_str()),
                        _ => unreachable!("alias carries its name as text"),
                    },
                    other => unreachable!("{other} is not a node kind this reads"),
                }
            }
        }
    }

    /// The name an `alias` at the root gives, if any.
    pub fn alias(&self) -> Option<&str> {
        match self {
            Node::Call(name, args) if name == "alias" => match &args[1] {
                Node::Lit(Literal::Str(n)) => Some(n.as_str()),
                _ => None,
            },
            _ => None,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

impl Serialize for Node {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        Node::from_json(&v).map_err(serde::de::Error::custom)
    }
}

/// A named formula: one output column of a `with_windows` call, or a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Formula {
    pub name: String,
    pub tree: Node,
}

fn span_of(name: &str, key: &str, v: &Value) -> Result<Span, String> {
    serde_json::from_value::<Span>(v.clone())
        .map_err(|e| format!("{name}: {key} must be a number or a duration: {e}"))
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    // On a character boundary: a slice at byte 60 panicked inside a
    // multibyte name (review R1, B3).
    match s.char_indices().nth(60) {
        Some((cut, _)) => format!("{}...", &s[..cut]),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    /// Review R1, B3: the message's cut of a long value fell inside a
    /// multibyte character and panicked.
    #[test]
    fn a_long_name_is_cut_on_a_character_boundary() {
        let name = "€".repeat(70);
        let err = super::Node::from_json(&serde_json::json!({ "co": name })).unwrap_err();
        assert!(err.contains("..."), "{err}");
    }

    use super::*;

    fn tree(text: &str) -> Node {
        Node::parse(text).unwrap()
    }

    /// Every node kind survives the list form in both directions.
    #[test]
    fn every_node_kind_round_trips() {
        for text in [
            r#"["col","a"]"#,
            r#"["lit",1.5]"#,
            r#"["lit",2]"#,
            r#"["lit","x"]"#,
            r#"["lit",true]"#,
            r#"["lit"]"#,
            r#"["+",["col","a"],["col","b"]]"#,
            r#"["/",["col","a"],["lit",2]]"#,
            r#"["**",["col","a"],["lit",2]]"#,
            r#"["==",["col","s"],["lit","x"]]"#,
            r#"["&",[">",["col","a"],["lit",0]],["is_null",["col","b"]]]"#,
            r#"["neg",["col","a"]]"#,
            r#"["log",["col","a"],["lit",2.718281828459045]]"#,
            r#"["clip",["col","a"],["lit",0],["lit"]]"#,
            r#"["clip",["col","a"],["lit"],["lit",1]]"#,
            r#"["fill_null",["col","a"],["lit",0.0]]"#,
            r#"["when",[">",["col","a"],["lit",0]],["col","b"],["lit"]]"#,
            r#"["cast",["col","a"],"Float64"]"#,
            r#"["cast",["col","a"],"Int8","non_strict"]"#,
            r#"["fill_null",["col","a"],["lit"]]"#,
            r#"["alias",["col","a"],"z"]"#,
            r#"["ewm_mean",["col","mid"],{"closed":"right","half_life":"10s","min_samples":1,"window_size":"1m"}]"#,
            r#"["rewm_sum",["*",["col","p"],["col","q"]],{"closed":"both","half_life":5.0,"min_samples":3,"partial":"keep","window_size":60.0}]"#,
            r#"["ewm_rate",["increment",["col","cum"]],{"closed":"right","half_life":30.0,"min_samples":1}]"#,
            r#"["increment",["col","cum"]]"#,
        ] {
            let node = tree(text);
            assert_eq!(node.to_json().to_string(), text, "{text}");
            assert_eq!(Node::parse(&node.to_json().to_string()).unwrap(), node);
        }
        // A null written the old way, `["lit", null]`, still reads, and
        // writes back as `["lit"]`, the form TOML can carry (review R2, P3).
        let old = tree(r#"["fill_null",["col","a"],["lit",null]]"#);
        assert_eq!(old, tree(r#"["fill_null",["col","a"],["lit"]]"#));
        assert_eq!(
            old.to_json().to_string(),
            r#"["fill_null",["col","a"],["lit"]]"#
        );
    }

    /// The rebuilt expression computes what the formula says, on a frame.
    #[test]
    fn the_expression_computes_the_formula() {
        let df = df!("a" => [1.0, 4.0, f64::NAN], "b" => [2.0, 0.5, 1.0], "s" => ["x", "y", "x"])
            .unwrap()
            .lazy()
            .with_columns([col("a").fill_nan(lit(NULL))])
            .collect()
            .unwrap();
        let cases = [
            (
                r#"["-",["col","a"],["lit",1.5]]"#,
                vec![Some(-0.5), Some(2.5), None],
            ),
            (
                r#"["/",["col","a"],["col","b"]]"#,
                vec![Some(0.5), Some(8.0), None],
            ),
            (
                r#"["fill_null",["col","a"],["lit",0.0]]"#,
                vec![Some(1.0), Some(4.0), Some(0.0)],
            ),
            (
                r#"["when",["==",["col","s"],["lit","x"]],["col","b"],["lit",null]]"#,
                vec![Some(2.0), None, Some(1.0)],
            ),
            (r#"["sqrt",["col","a"]]"#, vec![Some(1.0), Some(2.0), None]),
            (
                r#"["log",["col","a"],["lit",2.0]]"#,
                vec![Some(0.0), Some(2.0), None],
            ),
            (
                r#"["clip",["col","a"],["lit",2.0],null]"#,
                vec![Some(2.0), Some(4.0), None],
            ),
            (
                r#"["neg",["col","b"]]"#,
                vec![Some(-2.0), Some(-0.5), Some(-1.0)],
            ),
            (
                r#"["cast",["**",["col","b"],["lit",2]],"Float64"]"#,
                vec![Some(4.0), Some(0.25), Some(1.0)],
            ),
        ];
        for (text, want) in cases {
            let e = tree(text)
                .to_expr(&|_| unreachable!("no operator here"))
                .alias("out");
            let got = df.clone().lazy().select([e]).collect().unwrap();
            let got: Vec<Option<f64>> = got.column("out").unwrap().f64().unwrap().iter().collect();
            assert_eq!(got, want, "{text}");
        }
    }

    /// An operator is resolved to whatever the caller hands over, so a run
    /// can stand a hidden column in for it.
    #[test]
    fn operators_are_resolved_by_the_caller_and_collected_once_each() {
        let f = tree(
            r#"["/",["rewm_sum",["col","n"],{"half_life":1.0,"window_size":2.0}],["rewm_sum",["col","q"],{"half_life":1.0,"window_size":2.0}]]"#,
        );
        let ops = f.operators();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].kind, OpKind::RewmSum);
        assert_eq!(ops[0].direction(), Direction::Forward);
        assert_ne!(ops[0].key(), ops[1].key());
        assert_eq!(f.columns(), vec!["n".to_string(), "q".to_string()]);
        let df = df!("@0" => [1.0], "@1" => [4.0]).unwrap();
        let e = f
            .to_expr(&|op| {
                col(if op.key().contains("\"n\"") {
                    "@0"
                } else {
                    "@1"
                })
            })
            .alias("out");
        let got = df.lazy().select([e]).collect().unwrap();
        assert_eq!(got.column("out").unwrap().f64().unwrap().get(0), Some(0.25));
    }

    /// What is not element-wise, or not a formula at all, is refused by
    /// name before any kernel exists.
    #[test]
    fn what_is_not_read_is_refused_by_name() {
        for (text, says) in [
            (
                r#"["shift",["col","a"],["lit",1]]"#,
                "\"shift\" is not read",
            ),
            (r#"["cum_sum",["col","a"]]"#, "\"cum_sum\" is not read"),
            (r#"["mean",["col","a"]]"#, "\"mean\" is not read"),
            (r#"{"col":"a"}"#, "a list headed by its kind"),
            (r#"["col",1]"#, "names a column"),
            (r#"["+",["col","a"]]"#, "\"+\" takes 2 arguments, got 1"),
            (
                r#"["cast",["col","a"],"Date"]"#,
                "cast to \"Date\" is not read",
            ),
            (r#"["lit",[1,2]]"#, "a literal must be a scalar"),
            (r#"["ewm_mean",["col","a"],{}]"#, "half_life is required"),
            (
                r#"["ewm_mean",["col","a"],{"half_life":0}]"#,
                "half_life must be above 0",
            ),
            (
                r#"["rewm_mean",["col","a"],{"half_life":1}]"#,
                "needs a window_size",
            ),
            (
                r#"["ewm_mean",["col","a"],{"half_life":"inf"}]"#,
                "half_life = inf needs a window_size",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"window_size":0}]"#,
                "window_size must be finite and above 0",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"closed":"up"}]"#,
                "closed must be",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"min_samples":0}]"#,
                "min_samples must be a count of at least 1",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"partial":"maybe"}]"#,
                "partial must be",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"horizon":3}]"#,
                "unknown parameter \"horizon\"",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1},3]"#,
                "takes 2 arguments, got 3",
            ),
            (
                r#"["ewm_mean",["ewm_sum",["col","a"],{"half_life":1}],{"half_life":1}]"#,
                "not another window operator",
            ),
            (r#"not json"#, "not a formula"),
        ] {
            let err = Node::parse(text).unwrap_err();
            assert!(err.contains(says), "{text}: {err}");
        }
        // An increment inside an operator's input is the one nesting allowed.
        assert!(Node::parse(r#"["ewm_sum",["increment",["col","c"]],{"half_life":1}]"#).is_ok());
        let f = tree(r#"["alias",["-",["col","a"],["col","b"]],"spread"]"#);
        assert_eq!(f.alias(), Some("spread"));
        assert!(!f.has_window_operator());
    }
}
