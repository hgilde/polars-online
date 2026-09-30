//! `po.stream.with_windows` on polars frames (docs/PLAN.md task 78d): a call's
//! descriptions expanded into the core's windows and named, each chunk's
//! columns read into rows for [`Windows`], the rows the core still holds
//! kept as the input's own chunks, and the state saved with them.
//!
//! The rows a forward window holds are the input's chunks as they came --
//! sliced, never copied -- so the memory they cost is one horizon of input,
//! whatever the row width, and the output is those slices with the computed
//! columns beside them (docs/PLAN.md task 78, *How rows move*).

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;

use online_core::{ClockCfg, ClockValue, OnClockReset};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arrow::nanos_array;
use crate::bank::session_hash;
use crate::span::{Span, SpanList, format_duration};
use crate::spec::{ClockPolicy, SessionGapSpec, clock_cfg_of};
use crate::stream::usable;
use crate::windows::{
    Direction, Emitted, Partial, Refusal, RowIn, SameClock, SplitDef, SplitValue, Unlisted,
    WindowDef, Windows,
};

/// Which way a description looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Ewm,
    LookaheadRewm,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Ewm => "ewm",
            Kind::LookaheadRewm => "lookahead_rewm",
        }
    }
}

/// A listed value of a split column: text, or an integer matched as its
/// text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SplitWord {
    Int(i64),
    Text(String),
}

impl SplitWord {
    fn text(&self) -> String {
        match self {
            SplitWord::Int(i) => i.to_string(),
            SplitWord::Text(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitSpec {
    pub column: String,
    pub values: Vec<SplitWord>,
}

fn unlisted_error() -> Unlisted {
    Unlisted::Error
}

fn yes() -> bool {
    true
}

/// One `po.window.*` description, as the Python builders write it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Description {
    pub kind: Kind,
    pub columns: Vec<String>,
    pub halflife: SpanList,
    #[serde(default)]
    pub horizon: Option<SpanList>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub split: Option<SplitSpec>,
    #[serde(default = "unlisted_error")]
    pub unlisted: Unlisted,
    #[serde(default = "yes")]
    pub total: bool,
    #[serde(default)]
    pub same_clock: Option<SameClock>,
    #[serde(default)]
    pub partial: Option<Partial>,
    #[serde(default)]
    pub complete: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// The spec behind `like=`: its name, for messages, and the columns whose
/// values must all be usable for it to learn from a row -- its features and
/// its weight, the rule `stream.rs` applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Like {
    pub spec: String,
    pub accept: Vec<String>,
}

/// One `po.stream.with_windows` call: its descriptions and its clock policy, in
/// the spec's own words (`spec.rs`), which is how `like=` fills it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowsConfig {
    pub windows: Vec<Description>,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub max_dclock: Option<Span>,
    #[serde(default)]
    pub on_clock_reset: OnClockReset,
    #[serde(default)]
    pub min_backwards_jump: Option<Span>,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub like: Option<Like>,
}

const WHO: &str = "with_windows";
const WINDOWS_MAGIC: &str = "polars-online windows";
const WINDOWS_VERSION: u32 = 1;

/// What [`WindowsRun::save_bytes`] writes: the call, the core, and the rows
/// the core holds as Arrow IPC, in the input's own columns.
#[derive(Serialize)]
struct FileOut<'a> {
    magic: &'a str,
    version: u32,
    config: &'a WindowsConfig,
    core: &'a Windows,
    held: &'a [u8],
}

#[derive(Deserialize)]
struct FileIn {
    magic: String,
    version: u32,
    config: WindowsConfig,
    core: Windows,
    held: Vec<u8>,
}

/// A description's name in a message: its place in the `windows` list,
/// its kind and its columns.
fn label(i: usize, d: &Description) -> String {
    format!(
        "windows[{i}] ({} of {})",
        d.kind.name(),
        d.columns.join(", ")
    )
}

/// `template` with each `{field}` replaced. A field the window has no value
/// for, one the template may not use, and an unmatched brace are refused.
fn fill(
    template: &str,
    fields: &[(&str, Option<&str>)],
    who: &str,
    key: &str,
) -> Result<String, String> {
    let known = || {
        fields
            .iter()
            .map(|(f, _)| format!("{{{f}}}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = String::new();
    let mut rest = template;
    while let Some(open) = rest.find(['{', '}']) {
        if rest.as_bytes()[open] == b'}' {
            return Err(format!("{who}: {key} {template:?} has a }} with no {{"));
        }
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            return Err(format!("{who}: {key} {template:?} has a {{ with no }}"));
        };
        let field = &after[..close];
        match fields.iter().find(|(f, _)| *f == field) {
            Some((_, Some(v))) => out.push_str(v),
            Some((_, None)) => {
                return Err(format!(
                    "{who}: {key} {template:?} uses {{{field}}}, and this window has no {field}"
                ));
            }
            None => {
                return Err(format!(
                    "{who}: {key} {template:?} has the field {{{field}}}; the fields it may use are {}",
                    known()
                ));
            }
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A column's dtype; the caller has checked the input has it.
fn dtype_of<'a>(input: &'a Schema, column: &str) -> &'a DataType {
    input.get(column).expect("a column the input has")
}

/// A column the core reads as numbers: primitive numbers, booleans, or a
/// column of nulls. Not a temporal one, whose internal integers would
/// become values, nor text.
fn numeric(dtype: &DataType) -> bool {
    dtype.is_primitive_numeric() || matches!(dtype, DataType::Boolean | DataType::Null)
}

/// The windows, their output names and the columns they read: a call's
/// descriptions checked and expanded, columns × halflives × horizons in that
/// order.
struct Plan {
    defs: Vec<WindowDef>,
    values: Vec<String>,
    weights: Vec<String>,
    splits: Vec<(String, Vec<String>)>,
    outputs: Vec<String>,
    completes: Vec<(String, usize)>,
}

fn index_of(list: &mut Vec<String>, name: &str) -> usize {
    match list.iter().position(|c| c == name) {
        Some(i) => i,
        None => {
            list.push(name.to_string());
            list.len() - 1
        }
    }
}

fn plan(config: &WindowsConfig, input: &Schema) -> Result<Plan, String> {
    if config.windows.is_empty() {
        return Err(format!(
            "{WHO}: no windows; give at least one po.window description"
        ));
    }
    let has = |role: &str, c: &str| -> Result<(), String> {
        if input.contains(c) {
            Ok(())
        } else {
            Err(format!(
                "{WHO}: no {role} column {c:?} in the frame; it has {:?}",
                input.iter_names().map(|n| n.as_str()).collect::<Vec<_>>()
            ))
        }
    };
    let mut p = Plan {
        defs: Vec::new(),
        values: Vec::new(),
        weights: Vec::new(),
        splits: Vec::new(),
        outputs: Vec::new(),
        completes: Vec::new(),
    };
    // Every name a call adds, with who adds it, for the collision check.
    let mut owner: HashMap<String, String> = HashMap::new();
    let mut claim = |name: String, who: &str, label: &str| -> Result<String, String> {
        if input.contains(&name) {
            return Err(format!(
                "{who}: output column {name:?} is already a column of the frame; give the window \
                 a name= that does not collide"
            ));
        }
        if let Some(first) = owner.insert(name.clone(), label.to_string()) {
            return Err(format!(
                "{who}: output column {name:?} is also {first}'s; a name template needs a field \
                 -- {{column}}, {{halflife}}, {{horizon}} or {{split}} -- for whatever tells two \
                 windows apart"
            ));
        }
        Ok(name)
    };
    for (i, d) in config.windows.iter().enumerate() {
        let this = label(i, d);
        let who = format!("{WHO}: {this}");
        let direction = match d.kind {
            Kind::Ewm => Direction::Backward,
            Kind::LookaheadRewm => Direction::Forward,
        };
        if d.columns.is_empty() {
            return Err(format!("{who}: columns is empty"));
        }
        if d.kind == Kind::Ewm && d.same_clock.is_some() {
            return Err(format!(
                "{who}: same_clock is for lookahead_rewm; ewm has no later rows"
            ));
        }
        if d.kind == Kind::LookaheadRewm && d.horizon.is_none() {
            return Err(format!("{who}: lookahead_rewm needs a horizon"));
        }
        for c in &d.columns {
            has("value", c)?;
            if !numeric(dtype_of(input, c)) {
                return Err(format!(
                    "{who}: column {c:?} has dtype {}, and a window's values must be numbers",
                    dtype_of(input, c)
                ));
            }
        }
        let weight = match &d.weight {
            None => None,
            Some(w) => {
                has("weight", w)?;
                if !numeric(dtype_of(input, w)) {
                    return Err(format!(
                        "{who}: weight column {w:?} has dtype {}, and a weight must be a number",
                        dtype_of(input, w)
                    ));
                }
                Some(index_of(&mut p.weights, w))
            }
        };
        let split = match &d.split {
            None => None,
            Some(s) => {
                has("split", &s.column)?;
                if s.values.is_empty() {
                    return Err(format!("{who}: split lists no values; give at least one"));
                }
                let text: Vec<String> = s.values.iter().map(SplitWord::text).collect();
                if text.iter().collect::<HashSet<_>>().len() != text.len() {
                    return Err(format!("{who}: split lists a value twice: {text:?}"));
                }
                // The same column split by the same list is read once.
                let slot = match p
                    .splits
                    .iter()
                    .position(|(c, v)| *c == s.column && *v == text)
                {
                    Some(k) => k,
                    None => {
                        p.splits.push((s.column.clone(), text.clone()));
                        p.splits.len() - 1
                    }
                };
                Some((slot, text))
            }
        };
        let horizons: Vec<Option<&Span>> = match &d.horizon {
            None => vec![None],
            Some(h) => h.spans().iter().map(Some).collect(),
        };
        let partial = d.partial.unwrap_or(match direction {
            Direction::Backward => Partial::Keep,
            Direction::Forward => Partial::Null,
        });
        let template = d.name.clone().unwrap_or_else(|| {
            match (d.kind, d.horizon.is_some()) {
                (Kind::Ewm, true) => "{column}_ewm_{halflife}_{horizon}{split}",
                (Kind::Ewm, false) => "{column}_ewm_{halflife}{split}",
                (Kind::LookaheadRewm, _) => "{column}_rewm_{halflife}_{horizon}{split}",
            }
            .to_string()
        });
        // One `complete` column per horizon: every window of a description
        // with one horizon has the same flags, whatever its column or
        // halflife.
        let mut complete_of: HashMap<Option<String>, usize> = HashMap::new();
        for c in &d.columns {
            let value = index_of(&mut p.values, c);
            for h in d.halflife.spans() {
                for &z in &horizons {
                    let def = WindowDef {
                        direction,
                        value,
                        weight,
                        halflife: h.value(),
                        horizon: z.map(Span::value),
                        split: split.as_ref().map(|(slot, text)| SplitDef {
                            column: *slot,
                            categories: text.len(),
                            total: d.total,
                            unlisted: d.unlisted,
                        }),
                        same_clock: d.same_clock.unwrap_or(SameClock::Include),
                        partial,
                    };
                    def.check().map_err(|e| format!("{who}: {e}"))?;
                    let w = p.defs.len();
                    let (hl, zl) = (h.label(), z.map(Span::label));
                    let mut parts: Vec<Option<String>> = Vec::new();
                    if split.is_none() || d.total {
                        parts.push(None);
                    }
                    if let Some((_, text)) = &split {
                        parts.extend(text.iter().map(|t| Some(t.clone())));
                    }
                    for part in parts {
                        let sfield = part.map_or_else(String::new, |t| format!("_{t}"));
                        let name = fill(
                            &template,
                            &[
                                ("column", Some(c)),
                                ("halflife", Some(&hl)),
                                ("horizon", zl.as_deref()),
                                ("split", Some(&sfield)),
                            ],
                            &who,
                            "name",
                        )?;
                        p.outputs.push(claim(name, &who, &this)?);
                    }
                    if let Some(tmpl) = &d.complete {
                        let name = fill(tmpl, &[("horizon", zl.as_deref())], &who, "complete")?;
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            complete_of.entry(zl.clone())
                        {
                            e.insert(w);
                            p.completes.push((claim(name, &who, &this)?, w));
                        }
                    }
                    p.defs.push(def);
                }
            }
        }
    }
    Ok(p)
}

/// The kind of clock a run is fed. A state resumes only on the kind that
/// wrote it: the other cannot be ordered against its last row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockKind {
    Numeric,
    Temporal,
}

fn kind_of(v: ClockValue) -> ClockKind {
    match v {
        ClockValue::F64(_) => ClockKind::Numeric,
        ClockValue::Ns(_) => ClockKind::Temporal,
    }
}

/// The clock policy's quantities, checked against the clock column: all
/// durations on a temporal clock, all numbers on a numeric one or none, and
/// a cap or late-row threshold no finer than the clock's own step -- the
/// rules a spec's clock obeys (`arrow.rs`, `check_clocks`).
fn check_scale(config: &WindowsConfig, input: &Schema) -> Result<(), String> {
    let mut spans: Vec<(&str, &Span)> = Vec::new();
    for d in &config.windows {
        spans.extend(d.halflife.spans().iter().map(|s| ("halflife", s)));
        if let Some(h) = &d.horizon {
            spans.extend(h.spans().iter().map(|s| ("horizon", s)));
        }
    }
    spans.extend(config.max_dclock.iter().map(|s| ("max_dclock", s)));
    spans.extend(
        config
            .min_backwards_jump
            .iter()
            .map(|s| ("min_backwards_jump", s)),
    );
    if let Some(SessionGapSpec::Gap(g)) = &config.session_gap {
        spans.push(("session_gap", g));
    }
    let duration = spans.iter().find(|(_, s)| s.is_duration()).map(|(f, _)| *f);
    let number = spans
        .iter()
        .find(|(_, s)| s.is_unit_bound_number())
        .map(|(f, _)| *f);
    if let (Some(d), Some(n)) = (duration, number) {
        return Err(format!(
            "{WHO}: {d} is a duration but {n} is a plain number, and a number says nothing about \
             its unit; give every clock quantity the same way -- durations for a Datetime, Date \
             or Duration clock, numbers for a numeric one"
        ));
    }
    let Some(clock) = &config.clock else {
        return match duration {
            Some(d) => Err(format!(
                "{WHO}: {d} is a duration, which needs a clock column to measure it; with no \
                 clock a row is one unit, so give it as a number of rows"
            )),
            None => Ok(()),
        };
    };
    let dtype = dtype_of(input, clock);
    if matches!(dtype, DataType::Time) {
        return Err(format!(
            "{WHO}: clock column {clock:?} is a time of day, which starts again at midnight, so \
             it cannot be a clock; combine it with its date into a Datetime, e.g. \
             pl.col(\"date\").dt.combine(pl.col({clock:?}))"
        ));
    }
    if dtype.is_temporal() {
        if let Some(n) = number {
            return Err(format!(
                "{WHO}: clock column {clock:?} has dtype {dtype}, a temporal clock, but {n} is a \
                 plain number, which it cannot read: a number has no unit. Give {n} and the \
                 other clock quantities as durations, e.g. pl.duration(minutes=10), \
                 timedelta(minutes=10) or \"10m\"; or cast the clock to the unit you mean, e.g. \
                 pl.col({clock:?}).dt.epoch(\"s\").cast(pl.Float64), and use that column."
            ));
        }
        let tick = match dtype {
            DataType::Date => 86_400.0,
            DataType::Datetime(TimeUnit::Milliseconds, _)
            | DataType::Duration(TimeUnit::Milliseconds) => 1e-3,
            DataType::Datetime(TimeUnit::Microseconds, _)
            | DataType::Duration(TimeUnit::Microseconds) => 1e-6,
            _ => 1e-9,
        };
        for (param, s) in &spans {
            let why = match *param {
                "max_dclock" => {
                    "every step would be capped to it, so the clock would count rows rather \
                     than measure time"
                }
                "min_backwards_jump" => {
                    "no step back could be as small, so every one would start the windows over, \
                     which 0 says directly"
                }
                _ => continue,
            };
            let v = s.value();
            if s.is_duration() && v > 0.0 && v < tick {
                return Err(format!(
                    "{WHO}: {param} is {s}, less than the smallest step clock column {clock:?} \
                     ({dtype}) can take: {why}"
                ));
            }
        }
    } else if !numeric(dtype) || matches!(dtype, DataType::Boolean) {
        return Err(format!(
            "{WHO}: clock column {clock:?} has dtype {dtype}; a clock is numeric or temporal"
        ));
    } else if let Some(d) = duration {
        return Err(format!(
            "{WHO}: {d} is a duration, but clock column {clock:?} has dtype {dtype}, which has no \
             unit to measure it in. Use a temporal clock -- a Datetime, Date or Duration column, \
             e.g. pl.from_epoch({clock:?}, time_unit=\"s\") -- or give {d} as a number of the \
             clock's own units."
        ));
    }
    Ok(())
}

/// A frame fed through [`Windows`]: the rows it holds as the input's chunks.
pub struct WindowsRun {
    config: WindowsConfig,
    core: Windows,
    values: Vec<String>,
    weights: Vec<String>,
    splits: Vec<(String, Vec<String>)>,
    accept: Vec<String>,
    outputs: Vec<String>,
    completes: Vec<(String, usize)>,
    /// The input's schema, as the call was built against it.
    input: Schema,
    /// The rows the core holds, oldest first, as slices of the input.
    held: VecDeque<DataFrame>,
    /// The core's row count at this run's first row: an error names the
    /// row of this run's input.
    run_base: u64,
}

impl WindowsRun {
    /// A run of `config` over an input with schema `input`.
    ///
    /// # Errors
    ///
    /// A description or clock policy that cannot run, a column the input has
    /// not got or cannot give, and an output name that collides.
    pub fn new(config: WindowsConfig, input: &Schema) -> Result<Self, String> {
        for (role, c) in [
            ("clock", &config.clock),
            ("session", &config.session),
            ("group", &config.group),
        ] {
            if let Some(c) = c {
                if !input.contains(c) {
                    return Err(format!(
                        "{WHO}: no {role} column {c:?} in the frame; it has {:?}",
                        input.iter_names().map(|n| n.as_str()).collect::<Vec<_>>()
                    ));
                }
            }
        }
        let cfg: ClockCfg = clock_cfg_of(&ClockPolicy {
            who: WHO,
            clock: config.clock.as_deref(),
            max_dclock: config.max_dclock.as_ref(),
            on_clock_reset: config.on_clock_reset,
            min_backwards_jump: config.min_backwards_jump.as_ref(),
            session: config.session.as_deref(),
            session_gap: config.session_gap.as_ref(),
            spec_closes_on_session: None,
        })?;
        check_scale(&config, input)?;
        let p = plan(&config, input)?;
        let accept = config
            .like
            .as_ref()
            .map_or_else(Vec::new, |l| l.accept.clone());
        for c in &accept {
            if !input.contains(c) {
                return Err(format!(
                    "{WHO}: like= spec {:?} reads column {c:?}, which the frame has not got",
                    config.like.as_ref().map_or("", |l| l.spec.as_str())
                ));
            }
        }
        for (c, _) in &p.splits {
            Series::new_empty(c.as_str().into(), dtype_of(input, c))
                .cast(&DataType::String)
                .map(drop)
                .map_err(|_| {
                    format!(
                        "{WHO}: split column {c:?} has dtype {}, which has no text form to match \
                         the listed values against",
                        dtype_of(input, c)
                    )
                })?;
        }
        let core = Windows::new(p.defs, cfg).map_err(|e| format!("{WHO}: {e}"))?;
        Ok(Self {
            config,
            core,
            values: p.values,
            weights: p.weights,
            splits: p.splits,
            accept,
            outputs: p.outputs,
            completes: p.completes,
            input: input.clone(),
            held: VecDeque::new(),
            run_base: 0,
        })
    }

    /// The output's schema: the input's columns, then each output, then
    /// each `complete` column.
    pub fn schema(&self) -> Schema {
        let mut out = self.input.clone();
        for n in &self.outputs {
            out.insert(n.as_str().into(), DataType::Float64);
        }
        for (n, _) in &self.completes {
            out.insert(n.as_str().into(), DataType::Boolean);
        }
        out
    }

    /// Rows fed and not yet out.
    pub fn held(&self) -> usize {
        self.core.held()
    }

    /// Contributing rows the windows' queues hold: what their sums cost.
    pub fn queued(&self) -> usize {
        self.core.queued()
    }

    /// The columns the windows read, whatever else the output keeps: what a
    /// source must read however narrow the query's projection.
    pub fn needed(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut put = |c: &String| {
            if !out.contains(c) {
                out.push(c.clone());
            }
        };
        for c in [&self.config.clock, &self.config.session, &self.config.group]
            .into_iter()
            .flatten()
        {
            put(c);
        }
        self.values.iter().for_each(&mut put);
        self.weights.iter().for_each(&mut put);
        self.splits.iter().map(|(c, _)| c).for_each(&mut put);
        self.accept.iter().for_each(&mut put);
        out
    }

    /// Feed one chunk and return the rows it resolved, in input order, less
    /// any a `"drop"` window was partial on. With `limit`, rows are fed only
    /// until `limit` rows are out, so the state is the state after the row
    /// that resolved the last of them, whatever the chunk size.
    ///
    /// # Errors
    ///
    /// A chunk whose columns are not the input's, and every row refusal the
    /// core makes, each naming the row.
    pub fn feed(&mut self, df: &DataFrame, limit: Option<usize>) -> PolarsResult<DataFrame> {
        let n = df.height();
        let f64s = |names: &[String]| -> PolarsResult<Vec<Vec<f64>>> {
            names
                .iter()
                .map(|c| {
                    let s = df.column(c.as_str())?.cast(&DataType::Float64)?;
                    Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
                })
                .collect()
        };
        let values = f64s(&self.values)?;
        let weights = f64s(&self.weights)?;
        let accepts = f64s(&self.accept)?;
        let split_cols: Vec<Column> = self
            .splits
            .iter()
            .map(|(c, _)| df.column(c.as_str())?.cast(&DataType::String))
            .collect::<PolarsResult<_>>()?;
        let split_codes: Vec<Vec<SplitValue>> = split_cols
            .iter()
            .zip(&self.splits)
            .map(|(col, (_, listed))| {
                let index: HashMap<&str, u32> = listed
                    .iter()
                    .enumerate()
                    .map(|(k, t)| (t.as_str(), k as u32))
                    .collect();
                Ok(col
                    .str()?
                    .iter()
                    .map(|v| match v {
                        None => SplitValue::Null,
                        Some(t) => index
                            .get(t)
                            .map_or(SplitValue::Unlisted, |&k| SplitValue::Listed(k)),
                    })
                    .collect())
            })
            .collect::<PolarsResult<_>>()?;
        let clock = self.clock_values(df)?;
        let session: Option<Vec<u64>> = match &self.config.session {
            None => None,
            Some(c) => {
                let s = df.column(c.as_str())?.cast(&DataType::String)?;
                Some(s.str()?.iter().map(session_hash).collect())
            }
        };
        let group_col = match &self.config.group {
            None => None,
            Some(c) => Some(df.column(c.as_str())?.cast(&DataType::String)?),
        };
        let groups = group_col.as_ref().map(|c| c.str()).transpose()?;

        // The core's count at this chunk's first row: a refused row is
        // `seq - chunk_base` rows into the chunk.
        let chunk_base = self.core.next_seq();
        let mut row_values = vec![0.0; self.values.len()];
        let mut row_weights = vec![0.0; self.weights.len()];
        let mut row_splits = vec![SplitValue::Null; self.splits.len()];
        let mut consumed = n;
        for i in 0..n {
            if let Some(l) = limit {
                if self.core.ready_kept() >= l {
                    consumed = i;
                    break;
                }
            }
            for (k, col) in values.iter().enumerate() {
                row_values[k] = col[i];
            }
            for (k, col) in weights.iter().enumerate() {
                row_weights[k] = col[i];
            }
            for (k, col) in split_codes.iter().enumerate() {
                row_splits[k] = col[i];
            }
            let key = match groups {
                None => Some(""),
                Some(g) => g.get(i),
            };
            let group = self.core.group(key);
            let row = RowIn {
                group,
                clock: clock.as_ref().map(|c| c[i]),
                session: session.as_ref().map(|s| s[i]),
                values: &row_values,
                weights: &row_weights,
                splits: &row_splits,
                accept: accepts.iter().all(|col| usable(col[i])),
            };
            if let Err(r) = self.core.push(&row) {
                // The rows before it are in the core: hold them.
                if i > 0 {
                    self.held.push_back(df.slice(0, i));
                }
                return Err(self.refusal(r, chunk_base, &split_cols));
            }
        }
        if consumed > 0 {
            self.held.push_back(df.slice(0, consumed));
        }
        let e = match limit {
            None => self.core.drain(),
            Some(l) => self.core.drain_kept(l),
        };
        self.assemble(e, df.schema())
    }

    /// The end of the input: every row still held goes out, each forward
    /// window still open over it null.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        let e = self.core.finish();
        let schema = self.input.clone();
        self.assemble(e, &schema)
    }

    /// Clock values of a chunk, in the form the column has them: a temporal
    /// clock as nanoseconds, a numeric one as numbers; null, NaN and
    /// infinity refused by row, and the other kind than the state's.
    fn clock_values(&self, df: &DataFrame) -> PolarsResult<Option<Vec<ClockValue>>> {
        let Some(c) = &self.config.clock else {
            return Ok(None);
        };
        let col = df.column(c.as_str())?;
        let base = self.core.next_seq() - self.run_base;
        let bad = |i: usize| {
            polars_err!(ComputeError:
                "{WHO}: row {} has a null or non-finite clock {c:?}", base + i as u64)
        };
        let values: Vec<ClockValue> = if col.dtype().is_temporal() {
            let ns = nanos_array(col.as_materialized_series(), base as usize)?;
            (0..ns.len())
                .map(|i| ns.get(i).map(ClockValue::Ns).ok_or_else(|| bad(i)))
                .collect::<PolarsResult<_>>()?
        } else {
            let s = col.cast(&DataType::Float64)?;
            s.f64()?
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.filter(|v| v.is_finite())
                        .map(ClockValue::F64)
                        .ok_or_else(|| bad(i))
                })
                .collect::<PolarsResult<_>>()?
        };
        if let (Some(first), Some(last)) = (values.first(), self.core.shared_clock().last_clock()) {
            if kind_of(*first) != kind_of(last) {
                polars_bail!(ComputeError:
                    "{WHO}: clock column {c:?} is {}, and the state it resumes was fed a {} one; \
                     resume a state on the kind of clock that wrote it",
                    col.dtype(),
                    match kind_of(last) { ClockKind::Numeric => "numeric", ClockKind::Temporal => "temporal" }
                );
            }
        }
        Ok(Some(values))
    }

    /// A row the core refused, as the error naming it.
    fn refusal(&self, r: Refusal, chunk_base: u64, split_cols: &[Column]) -> PolarsError {
        let row = |seq: u64| seq - self.run_base;
        let at = |seq: u64| usize::try_from(seq - chunk_base).expect("a row of this chunk");
        match r {
            Refusal::Backwards {
                seq,
                back,
                prev,
                now,
                min_backwards_jump,
            } => {
                let column = self.config.clock.as_deref().unwrap_or("<row count>");
                let step = match (now, prev) {
                    (Some(ClockValue::Ns(c)), Some(ClockValue::Ns(p))) => {
                        i64::try_from(i128::from(p) - i128::from(c))
                            .map_or_else(|_| back.to_string(), format_duration)
                    }
                    _ => crate::spec::num_label(back),
                };
                match min_backwards_jump {
                    None => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} goes backwards by {step} at row {} \
                         (on_clock_reset = \"error\", the default). The input must be in clock \
                         order across groups: sort it by the clock; or, if a step back starts \
                         the stream over, set on_clock_reset = \"reset_state\" with a \
                         min_backwards_jump.",
                        row(seq)
                    ),
                    Some(m) => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} goes backwards by {step} at row {}, no \
                         more than min_backwards_jump = {}: a late row, not a new start. Sort \
                         the input by the clock, or lower min_backwards_jump if a step back \
                         this small starts the stream over (0 starts over at every one).",
                        row(seq),
                        self.config.min_backwards_jump.as_ref().map_or_else(|| crate::spec::num_label(m), ToString::to_string)
                    ),
                }
            }
            Refusal::Unlisted { seq, window } => {
                let def = &self.core.defs()[window];
                let slot = def.split.as_ref().expect("an unlisted row is split").column;
                let (column, listed) = &self.splits[slot];
                let value = split_cols[slot]
                    .str()
                    .ok()
                    .and_then(|s| s.get(at(seq)))
                    .map_or_else(|| "null".to_string(), |v| format!("{v:?}"));
                polars_err!(ComputeError:
                    "{WHO}: row {} has {column:?} = {value}, which the split does not list \
                     ({listed:?}); unlisted = \"error\" refuses a row that counts in a window \
                     with a value not listed, null included. List it, or set unlisted = \
                     \"total\" to count it in the total only or \"ignore\" to count it nowhere.",
                    row(seq)
                )
            }
            Refusal::NegativeWeight {
                seq,
                window,
                weight,
            } => {
                let def = &self.core.defs()[window];
                let column = &self.weights[def.weight.expect("a weight was read")];
                polars_err!(ComputeError:
                    "{WHO}: row {} has weight {column:?} = {weight}, below 0; a weight is how \
                     much a row counts, and a null or zero one counts nothing",
                    row(seq)
                )
            }
        }
    }

    /// The first `e.rows` held rows, with the computed columns beside them,
    /// less the dropped ones.
    fn assemble(&mut self, e: Emitted, schema: &Schema) -> PolarsResult<DataFrame> {
        let mut out: Option<DataFrame> = None;
        let mut parts = 0;
        let mut left = e.rows;
        while left > 0 {
            let h = self.held.front().map_or(0, DataFrame::height);
            assert!(h > 0, "the core holds no row the frames do not");
            let part = if h <= left {
                self.held.pop_front().expect("checked")
            } else {
                let front = self.held.front_mut().expect("checked");
                let part = front.slice(0, left);
                *front = front.slice(left as i64, h - left);
                part
            };
            left -= part.height();
            parts += 1;
            match &mut out {
                None => out = Some(part),
                Some(o) => {
                    o.vstack_mut(&part)?;
                }
            }
        }
        let mut frame = out.unwrap_or_else(|| DataFrame::empty_with_schema(schema));
        // Many one-row chunks make every later operation slow; a few are
        // what vstack is for.
        if parts > 8 {
            frame.rechunk_mut();
        }
        let mut cols: Vec<Column> = Vec::with_capacity(self.outputs.len() + self.completes.len());
        for (name, v) in self.outputs.iter().zip(&e.values) {
            let ca: Float64Chunked = v.iter().map(|&x| (!x.is_nan()).then_some(x)).collect();
            cols.push(ca.with_name(name.as_str().into()).into_column());
        }
        for (name, w) in &self.completes {
            cols.push(Column::new(name.as_str().into(), e.complete[*w].as_slice()));
        }
        frame.hstack_mut(&cols)?;
        if e.drop.iter().any(|&d| d) {
            let keep: BooleanChunked = e.drop.iter().map(|&d| Some(!d)).collect();
            frame = frame.filter(&keep)?;
        }
        Ok(frame)
    }

    /// The state as bytes: the call, the core, and the rows it holds, which
    /// the next run emits first. Versioned msgpack, one state one file.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let mut rows = DataFrame::empty_with_schema(&self.input);
        for part in &self.held {
            rows.vstack_mut(part).map_err(|e| e.to_string())?;
        }
        let mut held = Vec::new();
        IpcWriter::new(&mut held)
            .finish(&mut rows)
            .map_err(|e| e.to_string())?;
        rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            version: WINDOWS_VERSION,
            config: &self.config,
            core: &self.core,
            held: &held,
        })
        .map_err(|e| e.to_string())
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.save_bytes().map_err(std::io::Error::other)?;
        crate::atomic::write(path, &bytes)
    }

    /// A run that goes on from a saved state: the call must be the one that
    /// saved it, and the input must have the columns it held. Rows are
    /// counted from the start of the new input.
    ///
    /// # Errors
    ///
    /// Bytes that are not a windows state, a version this build does not
    /// read, another call, or held rows whose columns the input has not got.
    pub fn load_bytes(bytes: &[u8], config: WindowsConfig, input: &Schema) -> Result<Self, String> {
        let file: FileIn = rmp_serde::from_slice(bytes)
            .map_err(|e| format!("{WHO}: not a state it saved ({e})"))?;
        if file.magic != WINDOWS_MAGIC {
            return Err(format!("{WHO}: not a state it saved"));
        }
        if file.version != WINDOWS_VERSION {
            return Err(format!(
                "{WHO} state version {} not supported (this build reads {WINDOWS_VERSION})",
                file.version
            ));
        }
        if file.config != config {
            return Err(format!(
                "{WHO}: the state was saved by another call -- other windows or another clock \
                 policy -- and resumes only the call that saved it"
            ));
        }
        let mut run = Self::new(config, input)?;
        if file.core.defs() != run.core.defs() || file.core.clock_cfg() != run.core.clock_cfg() {
            return Err(format!(
                "{WHO}: the state's windows do not match its own call"
            ));
        }
        let held = IpcReader::new(Cursor::new(file.held))
            .finish()
            .map_err(|e| format!("{WHO}: the state's held rows cannot be read ({e})"))?;
        if held.height() != file.core.held() {
            return Err(format!(
                "{WHO}: the state holds {} rows and its core {}; it is damaged",
                held.height(),
                file.core.held()
            ));
        }
        if held.height() > 0 && held.schema().as_ref() != input {
            return Err(format!(
                "{WHO}: the state holds {} rows with columns {:?}, and the frame has {:?}; a \
                 state resumes on the input it was saved from",
                held.height(),
                held.schema()
                    .iter()
                    .map(|(n, t)| format!("{n}: {t}"))
                    .collect::<Vec<_>>(),
                input
                    .iter()
                    .map(|(n, t)| format!("{n}: {t}"))
                    .collect::<Vec<_>>()
            ));
        }
        run.run_base = file.core.next_seq();
        run.core = file.core;
        if held.height() > 0 {
            run.held.push_back(held);
        }
        Ok(run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(json: &str) -> WindowsConfig {
        serde_json::from_str(json).unwrap()
    }

    const FORWARD: &str = r#"{"windows": [{"kind": "lookahead_rewm", "columns": ["x"],
        "halflife": [1], "horizon": [2]}], "clock": "t", "max_dclock": 100}"#;

    fn frame(t: &[f64]) -> DataFrame {
        df!(
            "t" => t,
            "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
            "wide" => t.iter().map(|v| format!("row {v} {}", "pad".repeat(20))).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn values_ptr(df: &DataFrame, c: &str) -> *const f64 {
        df.column(c)
            .unwrap()
            .f64()
            .unwrap()
            .downcast_iter()
            .next()
            .unwrap()
            .values()
            .as_ptr()
    }

    /// The rows a forward window holds are the input's chunks, sliced and
    /// never copied: what makes their memory independent of row width.
    #[test]
    fn held_rows_are_the_input_itself() {
        let df = frame(&(0..100).map(|i| f64::from(i) * 0.01).collect::<Vec<_>>());
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let out = run.feed(&df, None).unwrap();
        assert_eq!(out.height(), 0);
        assert_eq!(run.held.len(), 1);
        assert_eq!(values_ptr(&run.held[0], "x"), values_ptr(&df, "x"));
        let wide = |d: &DataFrame| {
            d.column("wide")
                .unwrap()
                .str()
                .unwrap()
                .downcast_iter()
                .next()
                .unwrap()
                .views()
                .as_ptr()
        };
        assert_eq!(wide(&run.held[0]), wide(&df));
        // Rows out are the same buffers too, the computed column beside them.
        let late = frame(&[5.0]);
        let out = run.feed(&late, None).unwrap();
        assert_eq!(out.height(), 100);
        assert_eq!(values_ptr(&out, "x"), values_ptr(&df, "x"));
    }

    /// Under a limit the run reads up to the row that resolved the last row
    /// wanted, and no further: the state is the state after it.
    #[test]
    fn a_limit_stops_at_the_row_that_resolves() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let out = run.feed(&df, Some(1)).unwrap();
        assert_eq!(out.height(), 1);
        // Row 0 is resolved by the row at 2, the third row read.
        assert_eq!(run.core.next_seq(), 3);
        assert_eq!(run.held(), 2);
        assert_eq!(
            out.column("x_rewm_1_2").unwrap().f64().unwrap().get(0),
            Some(10.0)
        );
    }

    /// A resumed run names the rows of its own input, the held rows of the
    /// last one going out first.
    #[test]
    fn a_resumed_run_counts_its_own_rows() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0]);
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let first = run.feed(&df, None).unwrap();
        assert_eq!(first.height(), 2);
        let bytes = run.save_bytes().unwrap();
        let mut run = WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).unwrap();
        // The row at 9 resolves the two held rows, and is held itself.
        let out = run.feed(&frame(&[9.0]), None).unwrap();
        assert_eq!(
            out.column("t").unwrap().f64().unwrap().to_vec(),
            [Some(2.0), Some(3.0)]
        );
        // This run's rows are 9, 10 and 1: the third, row 2, steps back.
        let e = run
            .feed(&frame(&[10.0, 1.0]), None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("goes backwards by 9 at row 2"), "{e}");
    }

    /// A state resumes only on the kind of clock, and the columns, it held.
    #[test]
    fn a_state_holds_its_clock_kind_and_columns() {
        let df = frame(&[0.0, 1.0]);
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        run.feed(&df, None).unwrap();
        let bytes = run.save_bytes().unwrap();
        let other = df.drop("wide").unwrap();
        let e = WindowsRun::load_bytes(&bytes, config(FORWARD), other.schema())
            .err()
            .unwrap();
        assert!(e.contains("resumes on the input it was saved from"), "{e}");
        let temporal = df!("t" => [3i64], "x" => [1.0], "wide" => ["w"])
            .unwrap()
            .lazy()
            .with_column(col("t").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)))
            .collect()
            .unwrap();
        let mut run = WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).unwrap();
        let e = run.feed(&temporal, None).unwrap_err().to_string();
        assert!(
            e.contains("resume a state on the kind of clock that wrote it"),
            "{e}"
        );
    }

    #[test]
    fn a_name_template_is_filled_or_refused() {
        let f = |t: &str| {
            fill(
                t,
                &[
                    ("column", Some("x")),
                    ("horizon", None),
                    ("split", Some("")),
                ],
                "w",
                "name",
            )
        };
        assert_eq!(f("{column}_a{split}").unwrap(), "x_a");
        assert!(f("{column").unwrap_err().contains("with no }"));
        assert!(f("column}").unwrap_err().contains("with no {"));
        assert!(f("{horizon}").unwrap_err().contains("has no horizon"));
        assert!(
            f("{nope}")
                .unwrap_err()
                .contains("{column}, {horizon}, {split}")
        );
    }
}
