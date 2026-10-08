//! `po.stream.with_windows` over a frame (docs/PLAN.md tasks 78, 143 and
//! 144): formulas over the window operators, run a chunk at a time with the
//! window core, in input order, O(window) memory.
//!
//! A chunk goes through four passes. Polars evaluates every `increment`'s
//! input; this runner takes each increment one row back within its group's
//! session (a step back past `restart_after_step_back` starts it over), as
//! a hidden column of the chunk. Polars evaluates every operator's input --
//! an element-wise formula of the row, increments included -- into a
//! number per row. The core takes the rows, one kernel per distinct
//! setting, every operator on it reading its own input, and gives back
//! the rows it resolved, each with its operators' values. Polars then
//! evaluates the formulas over the held rows with the operators standing
//! in as hidden columns, and the hidden columns are dropped.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;

use online_core::{ClockCfg, ClockValue};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arrow::{NanosRole, fits_64, key_text, nanos_array, wide_integer_clock};
use crate::formula::{Formula, Node, OpNode};
use crate::span::{Span, format_duration};
use crate::spec::{ClockPolicy, SessionGapSpec, clock_cfg_of};
use crate::stream::usable;
use crate::windows::{
    Closed, Direction, KernelDef, OpDef, OpKind, Partial, Peek, Refusal, RowIn, Stat, Windows,
};

/// The spec behind `like=`: its name, for messages, and the columns whose
/// values must all be usable for it to learn from a row -- its features and
/// its weight, the rule `stream.rs` applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Like {
    pub spec: String,
    pub accept: Vec<String>,
}

/// One `po.stream.with_windows` call: its formulas and its clock policy, in
/// the spec's own words (`spec.rs`), which is how `like=` fills it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowsConfig {
    pub formulas: Vec<Formula>,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub gap_cap: Option<Span>,
    #[serde(default)]
    pub restart_after_step_back: Option<Span>,
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
/// 9 since task 212: a queue keeps which of its operators are variances
/// (`ewm_var`, `ewm_std`), its arenas six wide for one where every other
/// operator's are three, and an operator keeps its `bias`;
/// 8 since task 200: an integer clock rides in a row's `off` as its own
/// value, beside the form `off` holds, where a bool said nanoseconds or
/// the bits of a double, and an increment of an integer input keeps its
/// previous value as an integer; 7 since task 159 (F2): the group and
/// session columns' dtypes, which a resumed input must match; 6 since task 159 (W3): a number clock's raw
/// value rides in a row's `off`, where 5 left it 0; 5 since review R6
/// (D1): the last row read, beside the rows held, as a
/// sliced state's identity of its input; 4 since review R5 (C5): the skip
/// a sliced state carries counts the rows of its input consumed so far,
/// and the state knows that input by its first clock (round four changed
/// both under 3); 3 since review R2 (W1, W4, W5): a row's raw clock beside
/// its policy time in the queues, and the rows a resume skips; 2 since
/// task 143: formulas over operators, where 1 held descriptions. An older
/// state would load with defaults and misbehave. The bank's schema moves
/// with this number (review R4, A2; R6, D5).
const WINDOWS_VERSION: u32 = 9;

/// What [`WindowsRun::save_bytes`] writes: the call, the core, the rows the
/// core holds as Arrow IPC with their increment columns, and each group's
/// increment state.
#[derive(Serialize)]
struct FileOut<'a> {
    magic: &'a str,
    version: u32,
    config: &'a WindowsConfig,
    core: &'a Windows,
    held: &'a [u8],
    increments: &'a [IncrState],
    stream_clock: Option<ClockValue>,
    resume_skip: usize,
    resume_first: Option<ClockValue>,
    /// That row's session hash: with its clock, what tells the saved input
    /// from a next file that starts at the same stamp (review R7, E2).
    resume_first_session: Option<u64>,
    /// The last row the run read, the input's columns as Arrow IPC: with
    /// the rows held, the identity of a sliced state's input (review R6,
    /// D1); empty unless saved under a slice (R7, E3).
    last_row: &'a [u8],
    /// The group and session columns' dtypes, by role (task 159, F2).
    key_dtypes: &'a [(String, String)],
}

/// The rest of the file, after [`Header`]'s two fields.
#[derive(Deserialize)]
struct FileIn {
    config: WindowsConfig,
    core: Windows,
    held: Vec<u8>,
    increments: Vec<IncrState>,
    #[serde(default)]
    stream_clock: Option<ClockValue>,
    /// Rows of the input this state was saved under that the run consumed,
    /// for a run resumed on that input: see [`WindowsRun::consumed`]
    /// (review R2, W4; R4, B1).
    #[serde(default)]
    resume_skip: usize,
    /// That input's first row's clock: a resume on an input that starts
    /// elsewhere skips nothing (review R4, B3).
    #[serde(default)]
    resume_first: Option<ClockValue>,
    resume_first_session: Option<u64>,
    last_row: Vec<u8>,
    key_dtypes: Vec<(String, String)>,
}

/// The file's first two fields, read before the rest: a state another
/// version wrote is refused by its version, not by a field it lacks
/// (review R2, W5).
#[derive(Deserialize)]
struct Header {
    magic: String,
    version: u32,
}

/// A group's increment state: the previous row's inputs, session and clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct IncrState {
    key: Option<String>,
    /// Per increment: the previous value, NaN for none; a temporal input in
    /// nanoseconds; an integer input in integers (docs/PLAN.md task 200),
    /// whose step is taken in integers before it becomes a double.
    prev: Vec<f64>,
    prev_ns: Vec<Option<i64>>,
    prev_int: Vec<Option<i128>>,
    session: Option<u64>,
    clock: Option<ClockValue>,
}

/// The hidden column an operator stands in as, in a formula.
fn hidden(op: &OpNode) -> String {
    format!("@po:{}", op.key())
}

/// The hidden column an operator's input is evaluated into.
fn input_column(i: usize) -> String {
    format!("@in:{i}")
}

/// What the plan made of the call.
struct Plan {
    kernels: Vec<KernelDef>,
    ops: Vec<OpDef>,
    /// Distinct operator inputs, each a formula of the row and the
    /// increments, by the index `OpDef::input` names.
    inputs: Vec<Node>,
    /// Distinct window operators: the hidden column and the core output.
    columns: Vec<(String, usize)>,
    /// Distinct increments: the hidden column, the input formula and whether
    /// the input is temporal (an increment in seconds).
    increments: Vec<(String, Node, bool)>,
    /// Every span a kernel reads, with its parameter's name, for the scale
    /// check.
    spans: Vec<(&'static str, Span)>,
}

fn dtype_of<'a>(input: &'a Schema, column: &str) -> &'a DataType {
    input.get(column).expect("checked by the plan")
}

fn numeric(dtype: &DataType) -> bool {
    dtype.is_primitive_numeric() || matches!(dtype, DataType::Boolean | DataType::Decimal(..))
}

/// The dtypes `exprs` give over an empty frame with `schema`.
fn dtypes_of(schema: &Schema, exprs: Vec<Expr>) -> Result<Schema, String> {
    DataFrame::empty_with_schema(schema)
        .lazy()
        .with_columns(exprs)
        .collect_schema()
        .map(|s| s.as_ref().clone())
        .map_err(|e| e.to_string())
}

fn plan(config: &WindowsConfig, input: &Schema) -> Result<Plan, String> {
    if config.formulas.is_empty() {
        return Err(format!("{WHO}: no formulas; give at least one expression"));
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
    let mut names: HashSet<&str> = HashSet::new();
    for f in &config.formulas {
        if f.name.is_empty() || f.name.starts_with("@po:") || f.name.starts_with("@in:") {
            return Err(format!(
                "{WHO}: {:?} is no name for an output; name the expression with .alias() or a keyword",
                f.name
            ));
        }
        if input.contains(&f.name) {
            return Err(format!(
                "{WHO}: output {:?} is already a column of the frame; a positional expression is \
                 named after its leftmost column unless .alias() names it, so give it another name",
                f.name
            ));
        }
        if !names.insert(&f.name) {
            return Err(format!("{WHO}: two expressions are named {:?}", f.name));
        }
        // Each formula holds an operator (one rule with the Python side,
        // review R1, N7), and reads no column under a reserved prefix (N8).
        if f.tree.operators().is_empty() {
            return Err(format!(
                "{WHO}: {:?} holds no operator; a formula of the row alone is Polars' with_columns",
                f.name
            ));
        }
        for c in f.tree.columns() {
            if c.starts_with("@po:") || c.starts_with("@in:") {
                return Err(format!(
                    "{WHO}: {:?} reads column {c:?}; names starting with \"@po:\" and \"@in:\" are \
                     reserved for the operators' own columns",
                    f.name
                ));
            }
            has("input", &c)?;
        }
    }
    let mut p = Plan {
        kernels: Vec::new(),
        ops: Vec::new(),
        inputs: Vec::new(),
        columns: Vec::new(),
        increments: Vec::new(),
        spans: Vec::new(),
    };
    // Increments first: an operator's input may read them.
    for f in &config.formulas {
        for op in f.tree.operators() {
            if op.kind != OpKind::Increment {
                continue;
            }
            let key = hidden(op);
            if p.increments.iter().any(|(k, _, _)| *k == key) {
                continue;
            }
            if op
                .input
                .operators()
                .iter()
                .any(|o| o.kind == OpKind::Increment)
            {
                return Err(format!(
                    "{WHO}: {}: an increment's input is a formula of the row's columns, not of \
                     another increment",
                    f.name
                ));
            }
            let dtype = dtypes_of(
                input,
                vec![op.input.to_expr(&|_| unreachable!()).alias("@t")],
            )
            .map_err(|e| {
                format!(
                    "{WHO}: {}: the increment's input cannot be evaluated: {e}",
                    f.name
                )
            })?;
            let dtype = dtype.get("@t").expect("aliased").clone();
            let temporal = dtype.is_temporal();
            if !(numeric(&dtype) || temporal) {
                return Err(format!(
                    "{WHO}: {}: an increment's input is a number or a temporal column, got {dtype}",
                    f.name
                ));
            }
            p.increments.push((key, (*op.input).clone(), temporal));
        }
    }
    let incr_schema = {
        let mut s = input.clone();
        for (k, _, _) in &p.increments {
            s.insert(k.as_str().into(), DataType::Float64);
        }
        s
    };
    let resolve_incr = |op: &OpNode| col(hidden(op).as_str());
    for f in &config.formulas {
        for op in f.tree.operators() {
            if op.kind == OpKind::Increment {
                continue;
            }
            let key = hidden(op);
            if p.columns.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let direction = op.direction();
            let stat = op.kind.stat().expect("a window operator");
            let half_life = op.half_life.clone().expect("checked when read");
            p.spans.push(("half_life", half_life.clone()));
            if let Some(w) = &op.window_size {
                p.spans.push(("window_size", w.clone()));
            }
            let kernel = KernelDef {
                direction,
                half_life: half_life.value(),
                window_size: op.window_size.as_ref().map(Span::value),
                window_ns: match &op.window_size {
                    Some(Span::Duration(d)) => Some(d.nanos),
                    _ => None,
                },
                closed: op.closed,
            };
            kernel
                .check()
                .map_err(|e| format!("{WHO}: {}: {}: {e}", f.name, op.kind.name()))?;
            let ki = match p.kernels.iter().position(|k| *k == kernel) {
                Some(i) => i,
                None => {
                    p.kernels.push(kernel);
                    p.kernels.len() - 1
                }
            };
            let input_i = match p.inputs.iter().position(|n| *n == *op.input) {
                Some(i) => i,
                None => {
                    let dtype = dtypes_of(
                        &incr_schema,
                        vec![op.input.to_expr(&resolve_incr).alias("@t")],
                    )
                    .map_err(|e| {
                        format!(
                            "{WHO}: {}: {}'s input cannot be evaluated: {e}",
                            f.name,
                            op.kind.name()
                        )
                    })?;
                    let dtype = dtype.get("@t").expect("aliased").clone();
                    if !numeric(&dtype) {
                        return Err(format!(
                            "{WHO}: {}: {}'s input must be a number, got {dtype}",
                            f.name,
                            op.kind.name()
                        ));
                    }
                    p.inputs.push((*op.input).clone());
                    p.inputs.len() - 1
                }
            };
            p.ops.push(OpDef {
                kernel: ki,
                stat,
                input: input_i,
                min_samples: op.min_samples,
                partial: op.partial.unwrap_or(match direction {
                    Direction::Backward => Partial::Keep,
                    Direction::Forward => Partial::Null,
                }),
                bias: op.bias,
            });
            p.columns.push((key, p.ops.len() - 1));
        }
    }
    debug_assert!(
        !(p.ops.is_empty() && p.increments.is_empty()),
        "checked per formula"
    );
    Ok(p)
}

/// The kind of clock a run is fed. A state resumes only on the kind that
/// wrote it: the other cannot be ordered against its last row, and an
/// integer clock's edges are decided in integers, a float clock's in
/// doubles (docs/PLAN.md task 200).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockKind {
    Numeric,
    Temporal,
    Integer,
}

impl ClockKind {
    fn name(self) -> &'static str {
        match self {
            ClockKind::Numeric => "a float",
            ClockKind::Temporal => "a temporal",
            ClockKind::Integer => "an integer",
        }
    }
}

fn kind_of(v: ClockValue) -> ClockKind {
    match v {
        ClockValue::F64(_) => ClockKind::Numeric,
        ClockValue::Ns(_) => ClockKind::Temporal,
        ClockValue::I64(_) => ClockKind::Integer,
    }
}

/// Whether the clock stepped back from `prev` to `now` by more than
/// `restart`, which the policy reads as a new start: on an integer clock
/// the step in integers against it, exactly (task 200).
fn back_past(prev: ClockValue, now: ClockValue, restart: f64) -> bool {
    if let Some(step) = now.int_step(prev) {
        return step < 0
            && online_core::cmp_int_f64(-step, restart) == Some(std::cmp::Ordering::Greater);
    }
    let back = match (prev, now) {
        (ClockValue::Ns(p), ClockValue::Ns(c)) => {
            online_core::seconds_of_ns(i128::from(p) - i128::from(c))
        }
        (p, c) => p.seconds() - c.seconds(),
    };
    back > 0.0 && back > restart
}

/// An integer column's values, as `i128` so an unsigned 64-bit one is
/// whole too, and an `Int128` (review round 5, B3); null where null.
fn integer_values(s: &Series) -> PolarsResult<Vec<Option<i128>>> {
    Ok(match s.dtype() {
        DataType::UInt64 => s.u64()?.iter().map(|v| v.map(i128::from)).collect(),
        DataType::Int128 => s.i128()?.iter().collect(),
        _ => s
            .cast(&DataType::Int64)?
            .i64()?
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
    })
}

/// Whether an increment's input is read as its integers: every integer
/// dtype [`integer_values`] holds whole, the 64-bit forms and an `Int128`.
fn integer_input(dtype: &DataType) -> bool {
    fits_64(dtype) || *dtype == DataType::Int128
}

/// `now − prev` as a double, rounded once: the difference of two `i128`s
/// fits a `u128` either way round, where `i128` subtraction overflows
/// between the type's two ends.
fn integer_step(now: i128, prev: i128) -> f64 {
    let size = now.abs_diff(prev) as f64;
    if now >= prev { size } else { -size }
}

/// Durations and numbers must match the clock, as a spec's must
/// (docs/PLAN.md task 88): a duration needs a temporal clock, a number a
/// numeric one, and a duration below the clock's resolution is refused.
fn check_scale(
    config: &WindowsConfig,
    spans: &[(&'static str, Span)],
    input: &Schema,
) -> Result<(), String> {
    let mut all: Vec<(&str, &Span)> = spans.iter().map(|(n, s)| (*n, s)).collect();
    all.extend(config.gap_cap.iter().map(|s| ("gap_cap", s)));
    all.extend(
        config
            .restart_after_step_back
            .iter()
            .map(|s| ("restart_after_step_back", s)),
    );
    if let Some(SessionGapSpec::Gap(g)) = &config.session_gap {
        all.push(("session_gap", g));
    }
    let duration = all.iter().find(|(_, s)| s.is_duration()).map(|(f, _)| *f);
    let number = all
        .iter()
        .find(|(_, s)| s.is_unit_bound_number())
        .map(|(f, _)| *f);
    let clock = config.clock.as_deref();
    let temporal = clock.is_some_and(|c| dtype_of(input, c).is_temporal());
    match (duration, number) {
        (Some(d), Some(n)) => {
            return Err(format!(
                "{WHO}: {d} is a duration and {n} a number; on one clock every clock parameter \
                 is written the same way"
            ));
        }
        (Some(d), None) if !temporal => {
            return Err(format!(
                "{WHO}: {d} is a duration, which needs a temporal clock column (Datetime, \
                 Date or Duration); clock is {}",
                clock.map_or("not given".to_string(), |c| {
                    format!("{c:?} ({})", dtype_of(input, c))
                })
            ));
        }
        (None, Some(n)) if temporal => {
            return Err(format!(
                "{WHO}: {n} is a number, and clock column {:?} is {}; write it as a duration \
                 (\"10m\", a timedelta or pl.duration)",
                clock.unwrap_or(""),
                dtype_of(input, clock.unwrap_or(""))
            ));
        }
        _ => {}
    }
    if let (Some(c), true) = (clock, temporal) {
        let dtype = dtype_of(input, c);
        let tick = match dtype {
            DataType::Date => 86_400.0,
            DataType::Datetime(unit, _) | DataType::Duration(unit) => match unit {
                TimeUnit::Milliseconds => 1e-3,
                TimeUnit::Microseconds => 1e-6,
                TimeUnit::Nanoseconds => 1e-9,
            },
            _ => 1e-9,
        };
        for (param, s) in &all {
            let why = match *param {
                "gap_cap" => {
                    "every step would be capped to it, so the clock would count rows rather \
                     than measure time"
                }
                "restart_after_step_back" => {
                    "no step back could be as small, so every one would start the windows over, \
                     which 0 says directly"
                }
                _ => continue,
            };
            let v = s.value();
            if s.is_duration() && v > 0.0 && v < tick {
                return Err(format!(
                    "{WHO}: {param} is {s}, less than the smallest step clock column {c:?} \
                     ({dtype}) can take: {why}"
                ));
            }
        }
    }
    Ok(())
}

/// A session value as the core sees it: a hash of its text, so any dtype
/// can mark sessions and only a change is read.
/// The bank's session hash (`fnv1a`), which std's `DefaultHasher` is not: a
/// hash persisted in the state must be the same on every toolchain (review
/// R1, S6).
fn session_hash(v: Option<&str>) -> u64 {
    crate::bank::session_hash(v)
}

/// One `with_windows` run: the core, its plan, and the input rows it holds.
#[derive(Clone)]
pub struct WindowsRun {
    config: WindowsConfig,
    core: Windows,
    inputs: Vec<Node>,
    columns: Vec<(String, usize)>,
    increments: Vec<(String, Node, bool)>,
    accept: Vec<String>,
    /// The input's schema, as the call was built against it.
    input: Schema,
    /// The group and session columns' dtypes, by role: a key is its value's
    /// text, so a column of another type on a resumed input would start
    /// every key over, silently (task 159, F2).
    key_dtypes: Vec<(String, String)>,
    /// The output's schema: the input's columns, then each formula.
    output: Schema,
    /// The rows the core holds, oldest first, as slices of the input with
    /// the increment columns beside them.
    held: VecDeque<DataFrame>,
    /// Per group, the increments' state; the null key apart, so a row's
    /// key is looked up as text without a copy.
    incr_state: Vec<IncrState>,
    incr_index: HashMap<String, usize>,
    incr_null: Option<usize>,
    /// The previous row's clock on the stream, across groups: a step back
    /// past `restart_after_step_back` there restarts every group's windows
    /// in the core, so it restarts every group's increments too (review
    /// R1, S4).
    stream_clock: Option<ClockValue>,
    /// The core's row count at this run's first row: an error names the
    /// row of this run's input.
    run_base: u64,
    /// Rows at the start of the input to skip, from a state saved under a
    /// slice of the same input (review R2, W4; R4, B1), the first clock of
    /// the input that state was saved under (R4, B3), and how many rows
    /// this run has skipped, which an error's row number counts.
    resume_skip: usize,
    resume_first: Option<ClockValue>,
    resume_first_session: Option<u64>,
    skipped: u64,
    /// The identity of the input a sliced state resumes on (review R5, C3;
    /// R6, D1 and D2): the skip as loaded; the last rows the state read --
    /// the rows it holds that are this input's, or the last row read where
    /// it holds none -- and the skipped rows at those positions as they go
    /// by.
    resume_total: usize,
    resume_held: Option<DataFrame>,
    resume_rows: Vec<DataFrame>,
    /// The last row read, skipped or fed, as the chunk had it (the hidden
    /// columns beside the input's when fed): what a sliced state saves as
    /// its identity beside the rows it holds (review R6, D1).
    last_row: Option<DataFrame>,
    /// The first row's clock and session of this run's input, and whether
    /// a row has been seen: what a state saved by this run remembers its
    /// input by (review R4, B3; R7, E2).
    first_clock: Option<ClockValue>,
    first_session: Option<u64>,
    started: bool,
}

impl WindowsRun {
    /// A run of `config` over an input with schema `input`.
    ///
    /// # Errors
    ///
    /// A formula or clock policy that cannot run, a column the input has
    /// not got, and an output name that collides.
    pub fn new(config: WindowsConfig, input: &Schema) -> Result<Self, String> {
        for (role, c) in [
            ("clock", &config.clock),
            ("session", &config.session),
            ("group", &config.group),
        ] {
            if let Some(c) = c
                && !input.contains(c)
            {
                return Err(format!(
                    "{WHO}: no {role} column {c:?} in the frame; it has {:?}",
                    input.iter_names().map(|n| n.as_str()).collect::<Vec<_>>()
                ));
            }
        }
        let cfg: ClockCfg = clock_cfg_of(&ClockPolicy {
            who: WHO,
            clock: config.clock.as_deref(),
            gap_cap: config.gap_cap.as_ref(),
            restart_after_step_back: config.restart_after_step_back.as_ref(),
            session: config.session.as_deref(),
            session_gap: config.session_gap.as_ref(),
            spec_closes_on_session: None,
            // A bank's window targets run one core per group, each built
            // with no group column (`resolvers.rs`): the spec refuses its
            // own groups without a clock, with this same rule (task 173).
            group: config.group.as_deref(),
            looks_ahead: config
                .formulas
                .iter()
                .any(|f| f.tree.has_forward_operator()),
        })?;
        let p = plan(&config, input)?;
        check_scale(&config, &p.spans, input)?;
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
        // The output's dtypes: the formulas over the hidden columns, every
        // one a Float64, on an empty frame.
        let mut hidden_schema = input.clone();
        for (k, _, _) in &p.increments {
            hidden_schema.insert(k.as_str().into(), DataType::Float64);
        }
        for (k, _) in &p.columns {
            hidden_schema.insert(k.as_str().into(), DataType::Float64);
        }
        let exprs: Vec<Expr> = config
            .formulas
            .iter()
            .map(|f| {
                f.tree
                    .to_expr(&|op| col(hidden(op).as_str()))
                    .alias(f.name.as_str())
            })
            .collect();
        let evaluated = dtypes_of(&hidden_schema, exprs)
            .map_err(|e| format!("{WHO}: a formula cannot be evaluated: {e}"))?;
        let mut output = input.clone();
        for f in &config.formulas {
            let dtype = evaluated.get(&f.name).expect("aliased").clone();
            output.insert(f.name.as_str().into(), dtype);
        }
        let mut core = if p.ops.is_empty() {
            // Increments alone: the core carries a kernel that reads nothing,
            // so rows pass straight through it. A sum over no window, read
            // at every row.
            Windows::new(
                vec![KernelDef {
                    direction: Direction::Backward,
                    half_life: f64::INFINITY,
                    window_size: Some(1.0),
                    // Its edge is never read, but on a temporal clock an
                    // integer window converts nothing (review R4).
                    window_ns: Some(1_000_000_000),
                    closed: Closed::Left,
                }],
                vec![OpDef {
                    kernel: 0,
                    stat: Stat::Sum,
                    input: 0,
                    min_samples: 1,
                    partial: Partial::Keep,
                    bias: false,
                }],
                cfg,
            )
            .map_err(|e| format!("{WHO}: {e}"))?
        } else {
            Windows::new(p.kernels, p.ops, cfg).map_err(|e| format!("{WHO}: {e}"))?
        };
        core.set_grouped(config.group.is_some());
        let inputs = if p.inputs.is_empty() {
            vec![Node::Lit(crate::formula::Literal::Float(0.0))]
        } else {
            p.inputs
        };
        let key_dtypes: Vec<(String, String)> =
            [("group", &config.group), ("session", &config.session)]
                .into_iter()
                .filter_map(|(role, c)| {
                    let c = c.as_ref()?;
                    let dtype = input.get(c.as_str()).map(ToString::to_string)?;
                    Some((role.to_string(), dtype))
                })
                .collect();
        Ok(Self {
            config,
            core,
            inputs,
            columns: p.columns,
            increments: p.increments,
            accept,
            input: input.clone(),
            key_dtypes,
            output,
            held: VecDeque::new(),
            incr_state: Vec::new(),
            incr_index: HashMap::new(),
            incr_null: None,
            stream_clock: None,
            run_base: 0,
            resume_skip: 0,
            resume_first: None,
            resume_first_session: None,
            skipped: 0,
            resume_total: 0,
            resume_held: None,
            resume_rows: Vec::new(),
            last_row: None,
            first_clock: None,
            first_session: None,
            started: false,
        })
    }

    /// The output's schema: the input's columns, then each formula.
    pub fn schema(&self) -> Schema {
        self.output.clone()
    }

    /// Rows fed and not yet out.
    pub fn held(&self) -> usize {
        self.core.held()
    }

    /// Contributing rows the kernels' queues hold: what their sums cost.
    pub fn queued(&self) -> usize {
        self.core.queued()
    }

    /// The columns the formulas read, whatever else the output keeps: what a
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
        for f in &self.config.formulas {
            f.tree.columns().iter().for_each(&mut put);
        }
        self.accept.iter().for_each(&mut put);
        out
    }

    /// The hidden columns a held chunk carries beside the input's.
    fn hidden_names(&self) -> Vec<String> {
        self.increments.iter().map(|(k, _, _)| k.clone()).collect()
    }

    /// Feed one chunk and return the rows it resolved, in input order, less
    /// any a `"drop"` operator was partial on. With `limit`, rows are fed
    /// only until `limit` rows are out, so the state is the state after
    /// the row that resolved the last of them, whatever the chunk size.
    ///
    /// # Errors
    ///
    /// A chunk whose columns are not the input's, and every row refusal the
    /// core makes, each naming the row.
    pub fn feed(&mut self, df: &DataFrame, limit: Option<usize>) -> PolarsResult<DataFrame> {
        self.feed_inner(df, limit, None)
    }

    /// Feed one chunk for a bank learning formula targets (docs/PLAN.md
    /// task 104): every row it resolved, dropped ones included, as a frame
    /// of the formulas' values -- null where a `"drop"` operator was partial
    /// -- beside `@po:seq`, the row's number as the caller counts it (the
    /// chunk's `@po:row` column), and `@po:at`, the number of the row that
    /// resolved it (the row that closed, cut or discarded its windows),
    /// read from `rows`, the chunk's rows' numbers.
    ///
    /// # Errors
    ///
    /// As [`Self::feed`].
    pub fn feed_resolving(&mut self, df: &DataFrame, rows: &[u64]) -> PolarsResult<DataFrame> {
        polars_ensure!(
            rows.len() == df.height() && df.get_column_names().iter().any(|c| c.as_str() == "@po:row"),
            ComputeError: "feed_resolving: a row number per row, and an @po:row column"
        );
        self.feed_inner(df, None, Some(rows))
    }

    fn feed_inner(
        &mut self,
        df: &DataFrame,
        limit: Option<usize>,
        resolving: Option<&[u64]>,
    ) -> PolarsResult<DataFrame> {
        // The clock of the whole chunk first: the row numbers an error
        // names count the rows skipped below.
        let mut clock = self.clock_values(df)?;
        // A state saved under a slice recorded the rows of its input the
        // run consumed; a run resumed on that input skips them (review R2,
        // W4; R4, B1). The input's first clock says whether this is that
        // input: one that starts elsewhere, the next file, skips nothing
        // (R4, B3). Without a clock column the rows are skipped either way.
        if !self.started && df.height() > 0 {
            // The verdict first, the run's record of its input after it: a
            // refused first chunk leaves the run as it was, so fed again it
            // is refused again, where a run marked started before the
            // verdict took the same chunk whole on the next call (task 159,
            // F1).
            let first_clock = clock.as_ref().map(|c| c[0]);
            let (key, session) = self.first_key_session(df)?;
            if self.resume_skip > 0
                && (first_clock != self.resume_first || session != self.resume_first_session)
            {
                // Not the same input from its start: its first row's clock or
                // session is not the saved input's (review R7, E2: a clock
                // that starts over at the same stamp each day gives the next
                // file the same first clock, and the session tells them
                // apart). The row is put to the clock policy against the
                // last row the state read: a step forward or a new start (a
                // step back past restart_after_step_back, a new session) is
                // the next file, which skips nothing; the same stamp is a
                // file boundary inside a tied stamp or the same input sliced
                // inside it, which cannot be told apart; a step back the
                // policy refuses is the same input sliced by hand or an
                // overlapping file (review R5, C4; R6, D3 and D4).
                match self.core.peek(key.as_deref(), first_clock, session) {
                    Ok(Peek::NewStart) => self.resume_skip = 0,
                    Ok(Peek::Forward) if self.config.clock.is_some() => self.resume_skip = 0,
                    // A row-count clock steps forward at every row, so a step
                    // forward is evidence of nothing there: without a clock
                    // column the next file begins with a new session (review
                    // R8, F1) -- of its first row's group under `group`, so a
                    // group the state has not read shows none (R9, G1).
                    Ok(Peek::Forward) => polars_bail!(ComputeError:
                        "{WHO}: the state was saved under a slice of another input: without a \
                         clock column, this input does not begin with a new session: its first \
                         row continues the session last read, or is of a group the state has not \
                         read. A sliced state resumes on the input it was saved from, unsliced, \
                         or on the next file, which begins with a new session (under group, of a \
                         group the state has read)"
                    ),
                    Ok(Peek::SameStamp) => polars_bail!(ComputeError:
                        "{WHO}: the state was saved under a slice of another input: this input \
                         starts at the last stamp the state read, a file boundary inside a tied \
                         stamp or the same input sliced inside it. A sliced state resumes on the \
                         input it was saved from, unsliced, or on the next file, which starts \
                         after the last stamp it read"
                    ),
                    // The policy's own words beside the rule's (review R8, F2):
                    // under `group` a next-day file's step back is refused on
                    // the stream's clock whatever its session, as the
                    // unsliced path refuses it.
                    Err(r) => {
                        let why = self.refusal(r, self.core.next_seq()).to_string();
                        let why = why.strip_prefix(&format!("{WHO}: ")).unwrap_or(&why);
                        polars_bail!(ComputeError:
                            "{WHO}: the state was saved under a slice of another input: this \
                             input starts before the last row the state read, a step back the \
                             clock policy refuses. A sliced state resumes on the input it was \
                             saved from, unsliced, or on the next file, which starts after the \
                             last row it read. The policy: {why}"
                        )
                    }
                }
            } else if self.resume_skip == 0 && self.config.clock.is_some() && resolving.is_none() {
                // A state saved without a slice goes on with the next file,
                // and refuses one that starts at the last stamp it read, as a
                // sliced state does: a file boundary inside a tied stamp
                // cannot be told from an input that repeats the rows read
                // there, which would come out twice (task 158, E13). A fresh
                // run has read no stamp, and a new session at that stamp is a
                // new start; a step back is the policy's to refuse, below.
                // Not for a core the bank's resolver feeds (`resolving`): it
                // gets the rows of one stream the bank's own clock has
                // checked, so a chunk boundary inside a tied stamp is that
                // stream going on, and the same words refused every bank
                // with a window target saved there (task 159, B1).
                if let Ok(Peek::SameStamp) = self.core.peek(key.as_deref(), first_clock, session) {
                    polars_bail!(ComputeError:
                        "{WHO}: this input starts at the last stamp the state read, so it may be \
                         another input that repeats rows the state read: a file boundary inside \
                         a tied stamp cannot be told from one, whose repeated rows would come \
                         out twice. Resume on the next file, which starts after the last stamp \
                         the state read"
                    );
                }
            }
            self.started = true;
            self.first_clock = first_clock;
            self.first_session = session;
        }
        let skipped: DataFrame;
        let mut df = df;
        let mut resolving = resolving;
        let skip = self.resume_skip.min(df.height());
        if skip > 0 {
            // The skipped rows inside the held window -- the last
            // `resume_held` of the `resume_total` rows consumed -- are kept
            // for the identity check once the skip completes (review R5, C3).
            let before = usize::try_from(self.skipped).expect("rows fit");
            let held = self.resume_held.as_ref().map_or(0, DataFrame::height);
            let lo = self.resume_total.saturating_sub(held).max(before);
            let hi = (before + skip).min(self.resume_total);
            if hi > lo {
                self.resume_rows
                    .push(df.slice((lo - before) as i64, hi - lo));
            }
            self.resume_skip -= skip;
            self.skipped += skip as u64;
            // Once the skip completes, the last row skipped is the last row
            // read; until then the loaded one still is, as `consumed` counts
            // the rows pending (review R6, D1; R7, E4).
            if self.resume_skip == 0 {
                self.last_row = Some(df.slice(skip as i64 - 1, 1));
            }
            skipped = df.slice(skip as i64, df.height() - skip);
            df = &skipped;
            resolving = resolving.map(|r| &r[skip..]);
            if let Some(c) = &mut clock {
                c.drain(..skip);
            }
            if self.resume_skip == 0 {
                self.check_resumed_input()?;
            }
        }
        let n = df.height();
        // Keys are text, a zoned Datetime's its instant, as the bank keys
        // them: the cast to text failed on one (task 160, PA4b).
        let session: Option<Vec<u64>> = match &self.config.session {
            None => None,
            Some(c) => {
                let s = key_text(df.column(c.as_str())?.as_materialized_series())?;
                Some(s.str()?.iter().map(session_hash).collect())
            }
        };
        let group_col = match &self.config.group {
            None => None,
            Some(c) => Some(key_text(df.column(c.as_str())?.as_materialized_series())?),
        };
        let groups = group_col.as_ref().map(|c| c.str()).transpose()?;
        let keys: Vec<Option<&str>> = (0..n)
            .map(|i| match groups {
                None => Some(""),
                Some(g) => g.get(i),
            })
            .collect();

        // Pass one: the increments, one row back within the group's
        // session, as hidden columns of the chunk. On a snapshot: the state
        // must end at the last row the core took, and a slice (`limit`) or
        // a refusal stops short of the chunk's end (review R1, B1).
        let snapshot = (
            self.incr_state.clone(),
            self.incr_index.clone(),
            self.incr_null,
            self.stream_clock,
        );
        let chunk = self.with_increments(df, &keys, &clock, session.as_deref())?;

        // Pass two: Polars evaluates every operator's input.
        let resolve_incr = |op: &OpNode| col(hidden(op).as_str());
        // `with_columns` then `select`: a bare literal input (the stand-in
        // where only increments run) broadcasts to the chunk's rows.
        let evaluated = chunk
            .clone()
            .lazy()
            .with_columns(
                self.inputs
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        node.to_expr(&resolve_incr)
                            .cast(DataType::Float64)
                            .alias(input_column(i).as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .select(
                (0..self.inputs.len())
                    .map(|i| col(input_column(i).as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()?;
        let values: Vec<Vec<f64>> = (0..self.inputs.len())
            .map(|i| {
                let s = evaluated.column(input_column(i).as_str())?;
                Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            })
            .collect::<PolarsResult<_>>()?;
        let accepts: Vec<Vec<f64>> = self
            .accept
            .iter()
            .map(|c| {
                let s = df.column(c.as_str())?.cast(&DataType::Float64)?;
                Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            })
            .collect::<PolarsResult<_>>()?;

        // Pass three: the core.
        let chunk_base = self.core.next_seq();
        let mut row_values = vec![0.0; self.inputs.len()];
        let mut consumed = n;
        // Resolving: per resolved row, the number of the row that resolved
        // it, read off the core's ready count after each push. Nothing is
        // ready before the chunk: every resolving feed drains everything.
        let mut at: Vec<u64> = Vec::new();
        if let Some(rows) = resolving {
            at.resize(self.core.ready(), rows.first().copied().unwrap_or(0));
        }
        for i in 0..n {
            if let Some(l) = limit
                && self.core.ready_kept() >= l
            {
                consumed = i;
                break;
            }
            for (k, col) in values.iter().enumerate() {
                row_values[k] = col[i];
            }
            let group = self.core.group(keys[i]);
            let row = RowIn {
                group,
                clock: clock.as_ref().map(|c| c[i]),
                session: session.as_ref().map(|s| s[i]),
                values: &row_values,
                accept: accepts.iter().all(|col| usable(col[i])),
            };
            if let Err(r) = self.core.push(&row) {
                // The rows before it are in the core: hold them, and put the
                // increments' state where they end.
                if i > 0 {
                    self.held.push_back(chunk.slice(0, i));
                    self.last_row = Some(chunk.slice(i as i64 - 1, 1));
                }
                self.rewind_increments(snapshot, df, &keys, &clock, session.as_deref(), i)?;
                return Err(self.refusal(r, chunk_base));
            }
            if let Some(rows) = resolving {
                at.resize(self.core.ready(), rows[i]);
            }
        }
        if consumed > 0 {
            self.held.push_back(chunk.slice(0, consumed));
            self.last_row = Some(chunk.slice(consumed as i64 - 1, 1));
        }
        if consumed < n {
            self.rewind_increments(snapshot, df, &keys, &clock, session.as_deref(), consumed)?;
        }
        let e = match limit {
            None => self.core.drain(),
            Some(l) => self.core.drain_kept(l),
        };
        if resolving.is_some() {
            debug_assert_eq!(at.len(), e.rows);
            self.assemble_resolved(e, &at)
        } else {
            self.assemble(e)
        }
    }

    /// [`Self::assemble`] for [`Self::feed_resolving`]: the formulas over
    /// every resolved row, null on a dropped one, with the row numbers.
    fn assemble_resolved(
        &mut self,
        e: crate::windows::Emitted,
        at: &[u64],
    ) -> PolarsResult<DataFrame> {
        let resolved_at: UInt64Chunked = at.iter().map(|&a| Some(a)).collect();
        let mut frame = self.assemble_rows(e)?;
        frame.hstack_mut(&[resolved_at.with_name("@po:at".into()).into_column()])?;
        // An operator partial under `"drop"` gives no value, so a formula
        // over it is null of itself; the row's other formulas keep theirs
        // (review R1, D4: every formula of the row was nulled).
        let exprs: Vec<Expr> = self
            .config
            .formulas
            .iter()
            .map(|f| {
                f.tree
                    .to_expr(&|op| col(hidden(op).as_str()))
                    .alias(f.name.as_str())
            })
            .collect();
        let keep: Vec<Expr> = self
            .config
            .formulas
            .iter()
            .map(|f| col(f.name.as_str()))
            .chain([col("@po:row").alias("@po:seq"), col("@po:at")])
            .collect();
        frame.lazy().with_columns(exprs).select(keep).collect()
    }

    /// Rows of this run's input consumed so far: skipped as already read, a
    /// loaded skip not yet applied (review R5, C1), or fed to the core.
    /// Under a slice, what a run resumed on the same input must skip
    /// ([`Self::save_bytes_with`]). Rows the state held from an earlier run
    /// and returned are none of this input's (review R4, B1 and B2).
    pub fn consumed(&self) -> usize {
        let fed = usize::try_from(self.core.next_seq() - self.run_base).expect("rows fit");
        usize::try_from(self.skipped).expect("rows fit") + self.resume_skip + fed
    }

    /// The first row's group key and session hash, as the core reads them.
    fn first_key_session(&self, df: &DataFrame) -> PolarsResult<(Option<String>, Option<u64>)> {
        let head = df.slice(0, 1);
        let key = match &self.config.group {
            None => Some(String::new()),
            Some(c) => key_text(head.column(c.as_str())?.as_materialized_series())?
                .str()?
                .get(0)
                .map(str::to_string),
        };
        let session = match &self.config.session {
            None => None,
            Some(c) => {
                let s = key_text(head.column(c.as_str())?.as_materialized_series())?;
                Some(session_hash(s.str()?.get(0)))
            }
        };
        Ok((key, session))
    }

    /// The identity of the input a sliced state resumes on (review R5, C3
    /// and C4; R6, D1): the rows skipped where the state was cut must be
    /// the last rows the state read -- the rows it held that are the
    /// input's own, or the last row alone where it held none. Another input
    /// that starts at the same clock, or the same input sliced by hand, is
    /// refused by name rather than fed with rows skipped or doubled.
    fn check_resumed_input(&mut self) -> PolarsResult<()> {
        let rows = std::mem::take(&mut self.resume_rows);
        let names: Vec<&str> = self.input.iter_names().map(|n| n.as_str()).collect();
        let stack = |parts: &[DataFrame]| -> PolarsResult<DataFrame> {
            let mut out = DataFrame::empty_with_schema(&self.input);
            for part in parts {
                out.vstack_mut(&part.select(names.iter().copied())?)?;
            }
            Ok(out)
        };
        if let Some(want) = self.resume_held.take() {
            // The last rows the state read, as loaded: a run may have
            // drained some of the held ones before the skip completes.
            let got = stack(&rows)?;
            if got.height() != want.height() || !got.equals_missing(&want) {
                polars_bail!(ComputeError:
                    "{WHO}: the state was saved under a slice of another input: the last {} rows \
                     it read are not this input's rows there. A sliced state resumes on the input \
                     it was saved from, unsliced",
                    want.height()
                );
            }
        }
        Ok(())
    }

    /// The rows a sliced state still expects to skip: an input that ends
    /// first is another input (review R5, C3).
    fn resumed_input_ended_early(&self) -> Option<String> {
        (self.resume_skip > 0).then(|| {
            format!(
                "{WHO}: the state was saved under a slice of another input: {} rows were \
                 consumed, and this input ended after {}. A sliced state resumes on the input \
                 it was saved from, unsliced",
                self.resume_total, self.skipped
            )
        })
    }

    /// The end of the input: every row still held goes out, each forward
    /// window still open over it null.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        if let Some(msg) = self.resumed_input_ended_early() {
            polars_bail!(ComputeError: "{}", msg);
        }
        let e = self.core.finish();
        self.assemble(e)
    }

    /// Put the increments' state back to `snapshot` and run the first
    /// `rows` rows of `df` through it again, so it ends where the core
    /// stopped: the values the held rows carry are the same, since an
    /// increment reads only the rows before it.
    #[allow(clippy::type_complexity)]
    fn rewind_increments(
        &mut self,
        snapshot: (
            Vec<IncrState>,
            HashMap<String, usize>,
            Option<usize>,
            Option<ClockValue>,
        ),
        df: &DataFrame,
        keys: &[Option<&str>],
        clock: &Option<Vec<ClockValue>>,
        session: Option<&[u64]>,
        rows: usize,
    ) -> PolarsResult<()> {
        (
            self.incr_state,
            self.incr_index,
            self.incr_null,
            self.stream_clock,
        ) = snapshot;
        if rows > 0 {
            let clock = clock.as_ref().map(|c| c[..rows].to_vec());
            self.with_increments(
                &df.slice(0, rows),
                &keys[..rows],
                &clock,
                session.map(|s| &s[..rows]),
            )?;
        }
        Ok(())
    }

    /// The chunk with one hidden column per increment: `x_i - x_{i-1}`
    /// within the group's session, null on a session's first row and after
    /// a step back the policy reads as a new start; seconds for a temporal
    /// input.
    fn with_increments(
        &mut self,
        df: &DataFrame,
        keys: &[Option<&str>],
        clock: &Option<Vec<ClockValue>>,
        session: Option<&[u64]>,
    ) -> PolarsResult<DataFrame> {
        if self.increments.is_empty() {
            return Ok(df.clone());
        }
        let n = df.height();
        let inputs = df
            .clone()
            .lazy()
            .select(
                self.increments
                    .iter()
                    .enumerate()
                    .map(|(i, (_, node, _))| {
                        node.to_expr(&|_| unreachable!("no operator in an increment's input"))
                            .alias(input_column(i).as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .collect()?;
        let restart = self.core.clock_cfg().min_backwards_jump;
        let restarts = matches!(
            self.core.clock_cfg().on_clock_reset,
            online_core::OnClockReset::ResetState
        );
        let mut cols: Vec<Column> = Vec::with_capacity(self.increments.len());
        let mut out: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n); self.increments.len()];
        // An integer input's values as integers (task 200): its step is
        // taken in integers and then converted, where two doubles of an
        // epoch-nanosecond column near 1.8e18 tied at a step of 1.
        let integer: Vec<Option<Vec<Option<i128>>>> = (0..self.increments.len())
            .map(|i| {
                let c = inputs.column(input_column(i).as_str())?;
                if self.increments[i].2 || !integer_input(c.dtype()) {
                    Ok(None)
                } else {
                    integer_values(c.as_materialized_series()).map(Some)
                }
            })
            .collect::<PolarsResult<_>>()?;
        let numeric: Vec<Option<Vec<Option<f64>>>> = (0..self.increments.len())
            .map(|i| {
                if self.increments[i].2 || integer[i].is_some() {
                    Ok(None)
                } else {
                    let s = inputs
                        .column(input_column(i).as_str())?
                        .cast(&DataType::Float64)?;
                    Ok(Some(s.f64()?.iter().collect()))
                }
            })
            .collect::<PolarsResult<_>>()?;
        let temporal: Vec<Option<Vec<Option<i64>>>> = (0..self.increments.len())
            .map(|i| {
                if self.increments[i].2 {
                    // A time of day is read too, in nanoseconds since
                    // midnight (review round 4, PD5).
                    let c = inputs.column(input_column(i).as_str())?;
                    let ns = nanos_array(c.as_materialized_series(), 0, NanosRole::Increment)?;
                    Ok(Some((0..ns.len()).map(|r| ns.get(r)).collect()))
                } else {
                    Ok(None)
                }
            })
            .collect::<PolarsResult<_>>()?;
        for r in 0..n {
            let found = match keys[r] {
                Some(k) => self.incr_index.get(k).copied(),
                None => self.incr_null,
            };
            let gi = match found {
                Some(gi) => gi,
                None => {
                    let gi = self.incr_state.len();
                    self.incr_state.push(IncrState {
                        key: keys[r].map(str::to_string),
                        prev: vec![f64::NAN; self.increments.len()],
                        prev_ns: vec![None; self.increments.len()],
                        prev_int: vec![None; self.increments.len()],
                        session: None,
                        clock: None,
                    });
                    match keys[r] {
                        Some(k) => {
                            self.incr_index.insert(k.to_string(), gi);
                        }
                        None => self.incr_null = Some(gi),
                    }
                    gi
                }
            };
            // A step back on the stream's clock past the threshold restarts
            // every group in the core, so every group's increments start
            // over too (review R1, S4).
            let now_clock = clock.as_ref().map(|c| c[r]);
            if let (Some(now), Some(prev)) = (now_clock, self.stream_clock)
                && restarts
                && back_past(prev, now, restart)
            {
                for st in &mut self.incr_state {
                    st.prev.iter_mut().for_each(|p| *p = f64::NAN);
                    st.prev_ns.iter_mut().for_each(|p| *p = None);
                    st.prev_int.iter_mut().for_each(|p| *p = None);
                }
            }
            self.stream_clock = now_clock;
            let st = &mut self.incr_state[gi];
            let now_session = session.map(|s| s[r]);
            let mut new_start = st.session != now_session && st.session.is_some();
            if let (Some(now), Some(prev)) = (now_clock, st.clock)
                && restarts
                && back_past(prev, now, restart)
            {
                new_start = true;
            }
            if new_start {
                st.prev.iter_mut().for_each(|p| *p = f64::NAN);
                st.prev_ns.iter_mut().for_each(|p| *p = None);
                st.prev_int.iter_mut().for_each(|p| *p = None);
            }
            st.session = now_session;
            st.clock = now_clock;
            for i in 0..self.increments.len() {
                if let Some(vals) = &integer[i] {
                    let v = vals[r];
                    out[i].push(match (v, st.prev_int[i]) {
                        (Some(c), Some(p)) => Some(integer_step(c, p)),
                        // A previous value a float input of another chunk left.
                        (Some(c), None) if !st.prev[i].is_nan() => Some(c as f64 - st.prev[i]),
                        _ => None,
                    });
                    if let Some(c) = v {
                        st.prev_int[i] = Some(c);
                        st.prev[i] = c as f64;
                    }
                } else if let Some(vals) = &numeric[i] {
                    let v = match vals[r] {
                        Some(x) if usable(x) => x,
                        _ => f64::NAN,
                    };
                    out[i].push(if v.is_nan() || st.prev[i].is_nan() {
                        None
                    } else {
                        Some(v - st.prev[i])
                    });
                    if !v.is_nan() {
                        st.prev[i] = v;
                        st.prev_int[i] = None;
                    }
                } else if let Some(vals) = &temporal[i] {
                    let v = vals[r];
                    out[i].push(match (v, st.prev_ns[i]) {
                        (Some(c), Some(p)) => {
                            Some(online_core::seconds_of_ns(i128::from(c) - i128::from(p)))
                        }
                        _ => None,
                    });
                    if v.is_some() {
                        st.prev_ns[i] = v;
                    }
                }
            }
        }
        for (i, (name, _, _)) in self.increments.iter().enumerate() {
            let ca: Float64Chunked = std::mem::take(&mut out[i]).into_iter().collect();
            cols.push(ca.with_name(name.as_str().into()).into_column());
        }
        let mut chunk = df.clone();
        chunk.hstack_mut(&cols)?;
        Ok(chunk)
    }

    /// Clock values of a chunk, in the form the column has them: a temporal
    /// clock as nanoseconds, a numeric one as numbers; null, NaN and
    /// infinity refused by row, and the other kind than the state's.
    fn clock_values(&self, df: &DataFrame) -> PolarsResult<Option<Vec<ClockValue>>> {
        let Some(c) = &self.config.clock else {
            return Ok(None);
        };
        let col = df.column(c.as_str())?;
        let base = self.core.next_seq() - self.run_base + self.skipped;
        let bad = |i: usize| {
            polars_err!(ComputeError:
                "{WHO}: row {} has a null or non-finite clock {c:?}", base + i as u64)
        };
        let values: Vec<ClockValue> = if col.dtype().is_temporal() {
            let ns = nanos_array(
                col.as_materialized_series(),
                base as usize,
                NanosRole::Clock,
            )?;
            (0..ns.len())
                .map(|i| ns.get(i).map(ClockValue::Ns).ok_or_else(|| bad(i)))
                .collect::<PolarsResult<_>>()?
        } else if col.dtype().is_integer() && fits_64(col.dtype()) {
            // An integer clock as its integers (task 200), as the bank reads
            // one; an unsigned value past an `i64` is refused by row.
            integer_values(col.as_materialized_series())?
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let v = v.ok_or_else(|| bad(i))?;
                    i64::try_from(v).map(ClockValue::I64).map_err(|_| {
                        polars_err!(ComputeError:
                            "{WHO}: row {} has clock {c:?} = {v}, past {}, the largest value \
                             an integer clock holds (an Int64's)",
                            base + i as u64, i64::MAX)
                    })
                })
                .collect::<PolarsResult<_>>()?
        } else if col.dtype().is_integer() {
            polars_bail!(ComputeError: "{}", wide_integer_clock(WHO, c, col.dtype()));
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
        if let (Some(first), Some(last)) = (values.first(), self.core.shared_clock().last_clock())
            && kind_of(*first) != kind_of(last)
        {
            polars_bail!(ComputeError:
                "{WHO}: clock column {c:?} is {}, and the state it resumes was fed {} one; \
                 resume a state on the kind of clock that wrote it",
                col.dtype(),
                kind_of(last).name()
            );
        }
        Ok(Some(values))
    }

    /// A row the core refused, as the error naming it.
    fn refusal(&self, r: Refusal, _chunk_base: u64) -> PolarsError {
        let row = |seq: u64| seq - self.run_base + self.skipped;
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
                    // In integers, exactly (task 200).
                    (Some(ClockValue::I64(c)), Some(ClockValue::I64(p))) => {
                        (i128::from(p) - i128::from(c)).to_string()
                    }
                    _ => crate::spec::num_label(back),
                };
                match min_backwards_jump {
                    None => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} goes backwards by {step} at row {} \
                         (restart_after_step_back is unset, so every step back is refused). \
                         The input must be in clock order across groups: sort it by the \
                         clock; or, if a step back this large starts the stream over, set \
                         restart_after_step_back to the smallest one that does.",
                        row(seq)
                    ),
                    Some(m) => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} goes backwards by {step} at row {}, no \
                         more than restart_after_step_back = {}: a late row, not a new start. \
                         Sort the input by the clock, or lower restart_after_step_back if a \
                         step back this small starts the stream over (0 starts over at every \
                         one).",
                        row(seq),
                        self.config.restart_after_step_back.as_ref().map_or_else(|| crate::spec::num_label(m), ToString::to_string)
                    ),
                }
            }
        }
    }

    /// The first `e.rows` held rows with the formulas evaluated beside
    /// them, less the dropped ones and the hidden columns.
    fn assemble(&mut self, e: crate::windows::Emitted) -> PolarsResult<DataFrame> {
        let drop = e.drop.clone();
        let mut frame = self.assemble_rows(e)?;
        if drop.iter().any(|&d| d) {
            let keep: BooleanChunked = drop.iter().map(|&d| Some(!d)).collect();
            frame = frame.filter(&keep)?;
        }
        // Pass four: the formulas, over the rows with their operators.
        let exprs: Vec<Expr> = self
            .config
            .formulas
            .iter()
            .map(|f| {
                f.tree
                    .to_expr(&|op| col(hidden(op).as_str()))
                    .alias(f.name.as_str())
            })
            .collect();
        // The input's columns the chunk has (a projection may have narrowed
        // it), then the formulas; the hidden columns stay behind.
        let present: Vec<String> = frame
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .filter(|n| self.input.contains(n.as_str()))
            .collect();
        let keep: Vec<Expr> = present
            .iter()
            .map(|n| col(n.as_str()))
            .chain(self.config.formulas.iter().map(|f| col(f.name.as_str())))
            .collect();
        frame.lazy().with_columns(exprs).select(keep).collect()
    }

    /// The emitted rows as one frame: the held input rows with their
    /// increment columns, and each operator's values as its hidden column.
    fn assemble_rows(&mut self, e: crate::windows::Emitted) -> PolarsResult<DataFrame> {
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
        let held_schema = {
            let mut s = self.input.clone();
            for k in self.hidden_names() {
                s.insert(k.as_str().into(), DataType::Float64);
            }
            s
        };
        let mut frame = out.unwrap_or_else(|| DataFrame::empty_with_schema(&held_schema));
        // Many one-row chunks make every later operation slow; a few are
        // what vstack is for.
        if parts > 8 {
            frame.rechunk_mut();
        }
        let mut cols: Vec<Column> = Vec::with_capacity(self.columns.len());
        for (name, o) in &self.columns {
            let ca: Float64Chunked = e.values[*o]
                .iter()
                .map(|&x| (!x.is_nan()).then_some(x))
                .collect();
            cols.push(ca.with_name(name.as_str().into()).into_column());
        }
        frame.hstack_mut(&cols)?;
        Ok(frame)
    }

    /// The state as bytes: the call, the core, the rows it holds, which the
    /// next run emits first, and the increments' state. Versioned msgpack,
    /// one state one file.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        self.save_bytes_with(0, false)
    }

    /// [`Self::save_bytes`] for a state saved under a slice: a run resumed
    /// on the same input skips its first `skip_on_resume` rows, the rows
    /// consumed so far ([`Self::consumed`]); the state knows the input by
    /// its first row's clock and session, and by the last rows it read: the
    /// rows it holds that are the input's, or the last row alone (review
    /// R2, W4; R4, B1 to B3; R5, C1 to C4; R6, D1; R7, E2). `input_ended`
    /// says the input ran out rather
    /// than a slice being satisfied: a skip still pending is then another
    /// input, refused.
    pub fn save_bytes_with(
        &self,
        skip_on_resume: usize,
        input_ended: bool,
    ) -> Result<Vec<u8>, String> {
        // Only the caller knows whether the input ended or a slice was
        // satisfied first; a skip still pending is another input only in
        // the first case (review R5, C3).
        if input_ended && let Some(msg) = self.resumed_input_ended_early() {
            return Err(msg);
        }
        let held_schema = {
            let mut s = self.input.clone();
            for k in self.hidden_names() {
                s.insert(k.as_str().into(), DataType::Float64);
            }
            s
        };
        let mut rows = DataFrame::empty_with_schema(&held_schema);
        for part in &self.held {
            rows.vstack_mut(part).map_err(|e| e.to_string())?;
        }
        let mut held = Vec::new();
        IpcWriter::new(&mut held)
            .finish(&mut rows)
            .map_err(|e| e.to_string())?;
        // The last row read is a sliced state's identity and nothing else's
        // (review R7, E3): the bank saves every core unsliced.
        let mut last_row = Vec::new();
        if let (true, Some(r)) = (skip_on_resume > 0, &self.last_row) {
            let names: Vec<&str> = self.input.iter_names().map(|n| n.as_str()).collect();
            let mut r = r.select(names.iter().copied()).map_err(|e| e.to_string())?;
            IpcWriter::new(&mut last_row)
                .finish(&mut r)
                .map_err(|e| e.to_string())?;
        }
        rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            version: WINDOWS_VERSION,
            config: &self.config,
            core: &self.core,
            held: &held,
            increments: &self.incr_state,
            stream_clock: self.stream_clock,
            resume_skip: skip_on_resume,
            // A run that saw no row keeps the identity it loaded (review R5, C1).
            resume_first: if self.started {
                self.first_clock
            } else {
                self.resume_first
            },
            resume_first_session: if self.started {
                self.first_session
            } else {
                self.resume_first_session
            },
            last_row: &last_row,
            key_dtypes: &self.key_dtypes,
        })
        .map_err(|e| e.to_string())
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.save_with(path, 0, false)
    }

    /// [`Self::save_bytes_with`] to `path`, replacing it whole or not at
    /// all.
    pub fn save_with(
        &self,
        path: &std::path::Path,
        skip_on_resume: usize,
        input_ended: bool,
    ) -> std::io::Result<()> {
        let bytes = self
            .save_bytes_with(skip_on_resume, input_ended)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        crate::atomic::write(path, &bytes)
    }

    /// A run that goes on from a saved state: the call must be the one that
    /// saved it, and the input must have the columns it held. Rows are
    /// counted from the start of the new input, the rows the state says to
    /// skip included ([`Self::save_bytes_with`]).
    ///
    /// # Errors
    ///
    /// Bytes that are not a windows state, a version this build does not
    /// read, another call, or held rows whose columns the input has not got.
    pub fn load_bytes(bytes: &[u8], config: WindowsConfig, input: &Schema) -> Result<Self, String> {
        let header: Header = rmp_serde::from_slice(bytes).map_err(|e| {
            format!("{WHO}: the state cannot be read: not a state it saved, or damaged ({e})")
        })?;
        if header.magic != WINDOWS_MAGIC {
            return Err(format!("{WHO}: not a state it saved"));
        }
        if header.version != WINDOWS_VERSION {
            return Err(format!(
                "{WHO} state version {} not supported (this build reads {WINDOWS_VERSION})",
                header.version
            ));
        }
        let file: FileIn = rmp_serde::from_slice(bytes).map_err(|e| {
            format!(
                "{WHO}: the state is damaged, or was written by a build that changed a field \
                 without bumping the version ({e})"
            )
        })?;
        // A duration compares by its length, so "5000ms" is the call that
        // saved "5s" (task 160, PC3).
        if file.config != config {
            return Err(format!(
                "{WHO}: the state was saved by another call -- other formulas or another clock \
                 policy -- and resumes only the call that saved it"
            ));
        }
        let mut run = Self::new(config, input)?;
        if file.key_dtypes != run.key_dtypes {
            let name = |role: &str| {
                if role == "group" {
                    run.config.group.clone()
                } else {
                    run.config.session.clone()
                }
                .unwrap_or_default()
            };
            let saved: Vec<String> = file
                .key_dtypes
                .iter()
                .map(|(r, t)| format!("{r} column {:?}: {t}", name(r)))
                .collect();
            let now: Vec<String> = run
                .key_dtypes
                .iter()
                .map(|(r, t)| format!("{r} column {:?}: {t}", name(r)))
                .collect();
            return Err(format!(
                "{WHO}: the state keyed its groups and sessions by {}, and this input has {}; \
                 a key is its value's text, so a column of another type starts every key \
                 over. Cast the column to the type the state was saved with",
                saved.join(", "),
                now.join(", ")
            ));
        }
        if file.core.kernels() != run.core.kernels()
            || file.core.ops() != run.core.ops()
            || file.core.clock_cfg() != run.core.clock_cfg()
        {
            return Err(format!(
                "{WHO}: the state's operators do not match its own call"
            ));
        }
        // The core's own invariants, beyond the call's (review 2026-10-06,
        // PD1): a damaged core that still decoded panicked at the next call.
        file.core
            .check()
            .map_err(|e| format!("{WHO}: the state is damaged: {e}"))?;
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
        let hidden_names = run.hidden_names();
        let held_input: Vec<(String, DataType)> = held
            .schema()
            .iter()
            .filter(|(n, _)| !hidden_names.iter().any(|h| h == n.as_str()))
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        let want: Vec<(String, DataType)> = input
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        if held.height() > 0 && held_input != want {
            return Err(format!(
                "{WHO}: the state holds {} rows with columns {:?}, and the frame has {:?}; a \
                 state resumes on the input it was saved from",
                held.height(),
                held_input
                    .iter()
                    .map(|(n, t)| format!("{n}: {t}"))
                    .collect::<Vec<_>>(),
                want.iter()
                    .map(|(n, t)| format!("{n}: {t}"))
                    .collect::<Vec<_>>()
            ));
        }
        run.run_base = file.core.next_seq();
        run.core = file.core;
        if held.height() > 0 {
            run.held.push_back(held);
        }
        if file.resume_skip > 0 && !file.last_row.is_empty() {
            let last = IpcReader::new(Cursor::new(file.last_row))
                .finish()
                .map_err(|e| format!("{WHO}: the state's last row cannot be read ({e})"))?;
            let have: Vec<(String, DataType)> = last
                .schema()
                .iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect();
            if have != want {
                return Err(format!(
                    "{WHO}: the state's last row has columns {:?}, and the frame has {:?}; a \
                     state resumes on the input it was saved from",
                    have.iter()
                        .map(|(n, t)| format!("{n}: {t}"))
                        .collect::<Vec<_>>(),
                    want.iter()
                        .map(|(n, t)| format!("{n}: {t}"))
                        .collect::<Vec<_>>()
                ));
            }
            run.last_row = Some(last);
        }
        run.stream_clock = file.stream_clock;
        run.resume_skip = file.resume_skip;
        run.resume_first = file.resume_first;
        run.resume_first_session = file.resume_first_session;
        run.resume_total = file.resume_skip;
        if file.resume_skip > 0 {
            // The identity of this input: the rows the state holds that are
            // this input's -- the last `resume_skip` of them at most, since
            // a state saved on the next file under a slice still holds the
            // previous file's unresolved rows ahead of its own, rows going
            // out in order (review R6, D2) -- or, where it holds none (an
            // operator that resolves a row at its push), the last row it
            // read (R6, D1).
            let names: Vec<&str> = run.input.iter_names().map(|n| n.as_str()).collect();
            let mut want = DataFrame::empty_with_schema(&run.input);
            for part in &run.held {
                want.vstack_mut(
                    &part
                        .select(names.iter().copied())
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            let extra = want.height().saturating_sub(file.resume_skip);
            if extra > 0 {
                want = want.slice(extra as i64, want.height() - extra);
            }
            if want.height() == 0
                && let Some(last) = &run.last_row
            {
                want = last.clone();
            }
            if want.height() == 0 {
                // This build writes the last row read under every skip, so a
                // state with none has lost it (review R8, F5): refused rather
                // than resumed on anything by a check that compares nothing.
                return Err(format!(
                    "{WHO}: the state was saved under a slice of {} rows and carries no row to \
                     know its input by; it is damaged",
                    file.resume_skip
                ));
            }
            run.resume_held = Some(want);
        }
        for (i, st) in file.increments.into_iter().enumerate() {
            match &st.key {
                Some(k) => {
                    run.incr_index.insert(k.clone(), i);
                }
                None => run.incr_null = Some(i),
            }
            run.incr_state.push(st);
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

    const FORWARD: &str = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
        {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#;

    fn frame(t: &[f64]) -> DataFrame {
        df!(
            "t" => t,
            "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
            "wide" => t.iter().map(|v| format!("row {v} {}", "pad".repeat(20))).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Task 160, PB2: without a clock a row is one step, and a cap on it cut
    /// every window at every row (a gap of 1 past a cap of 0.5).
    #[test]
    fn gap_cap_needs_a_clock() {
        let input = frame(&[0.0, 1.0]).schema().clone();
        let err = WindowsRun::new(
            config(
                r#"{"formulas": [{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {"half_life": 10, "window_size": 3}]}], "gap_cap": 0.5}"#,
            ),
            &input,
        )
        .err()
        .expect("refused");
        assert!(err.contains("with_windows: gap_cap needs clock"), "{err}");
    }

    /// Task 173, PC1: without a clock column each group's clock counts its
    /// own rows, so a group that falls silent never closed its windows, and
    /// every later row of every group waited for them to the end of the
    /// input (500,001 rows held, 117 MiB more over 2M rows). Refused, naming
    /// a clock column or no group; the same call with a clock runs, and so
    /// do a window looking back under groups and one looking ahead without.
    #[test]
    fn a_window_looking_ahead_with_group_needs_a_clock() {
        let df = df!(
            "t" => [0.0, 1.0, 2.0, 3.0],
            "g" => ["a", "b", "b", "b"],
            "x" => [1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let ahead = r#"[{"name": "f", "tree": ["-", ["rewm_sum", ["col", "x"],
            {"half_life": 10, "window_size": 2}], ["col", "x"]]}]"#;
        let back = r#"[{"name": "f", "tree": ["ewm_sum", ["col", "x"], {"half_life": 10}]}]"#;
        let run = |formulas: &str, policy: &str| {
            WindowsRun::new(
                config(&format!(r#"{{"formulas": {formulas}{policy}}}"#)),
                df.schema(),
            )
        };
        let err = run(ahead, r#", "group": "g""#).err().expect("refused");
        assert!(
            err.starts_with("with_windows: a window looking ahead with group needs a clock column")
                && err.ends_with("Name a clock column, with gap_cap, or leave out group"),
            "{err}"
        );
        // `like=` takes a spec's policy, and is held to the same rule.
        let like = r#", "group": "g", "like": {"spec": "m", "accept": ["x"]}"#;
        let err = run(ahead, like).err().expect("refused");
        assert!(err.contains("needs a clock column"), "{err}");
        let mut clocked = run(ahead, r#", "group": "g", "clock": "t", "gap_cap": 1.5"#).unwrap();
        let mut out = clocked.feed(&df, None).unwrap();
        out.vstack_mut(&clocked.finish().unwrap()).unwrap();
        assert_eq!(out.height(), 4);
        run(back, r#", "group": "g""#).unwrap();
        run(ahead, "").unwrap();
    }

    /// Task 173, PC2: a spec's window target runs the same core, through
    /// `feed_resolving`, and a reset keeps the window whose far edge is the
    /// last row before it whole there too: row 1's window `(1, 3]` resolves
    /// at the step back with `4 + 8`, where it resolved discarded, null.
    /// The windows reaching past that row resolve discarded, null.
    #[test]
    fn a_target_window_whose_far_edge_is_the_last_row_before_a_reset_resolves_whole() {
        let df = df!(
            "t" => [0.0, 1.0, 2.0, 3.0, -7.0, -6.0],
            "x" => [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "@po:row" => [100u64, 101, 102, 103, 104, 105],
        )
        .unwrap();
        let mut run = WindowsRun::new(
            config(
                r#"{"formulas": [{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {"half_life": "inf", "window_size": 2}]}], "clock": "t", "gap_cap": 100,
                "restart_after_step_back": 1}"#,
            ),
            df.schema(),
        )
        .unwrap();
        let out = run
            .feed_resolving(&df, &[100, 101, 102, 103, 104, 105])
            .unwrap();
        let rows = |name: &str| -> Vec<u64> {
            out.column(name)
                .unwrap()
                .u64()
                .unwrap()
                .iter()
                .flatten()
                .collect()
        };
        let f: Vec<Option<f64>> = out.column("f").unwrap().f64().unwrap().iter().collect();
        assert_eq!(rows("@po:seq"), [100, 101, 102, 103]);
        assert_eq!(rows("@po:at"), [103, 104, 104, 104]);
        assert_eq!(f, [Some(6.0), Some(12.0), None, None]);
    }

    /// Task 160, PC3: a state compares its call by length, not by how a
    /// length was written: "5000ms" resumes the call that saved "5s", and
    /// the rows it holds go on to the same values.
    #[test]
    fn a_state_resumes_another_spelling_of_one_length() {
        let df = df!(
            "t" => Series::new("t".into(), [0i64, 1_000, 2_000, 3_000, 4_000, 5_000])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        .unwrap();
        let call = |cap: &str, window: &str| {
            config(&format!(
                r#"{{"formulas": [{{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {{"half_life": "1h", "window_size": "{window}"}}]}}], "clock": "t",
                "gap_cap": "{cap}"}}"#
            ))
        };
        let mut whole = WindowsRun::new(call("5s", "2s"), df.schema()).unwrap();
        let mut want = whole.feed(&df, None).unwrap();
        want.vstack_mut(&whole.finish().unwrap()).unwrap();
        let mut first = WindowsRun::new(call("5s", "2s"), df.schema()).unwrap();
        let got = first.feed(&df.slice(0, 3), None).unwrap();
        let bytes = first.save_bytes().unwrap();
        for (cap, window) in [("5000ms", "2s"), ("5s", "2000ms")] {
            let mut second = WindowsRun::load_bytes(&bytes, call(cap, window), df.schema())
                .unwrap_or_else(|e| panic!("{cap}, {window}: {e}"));
            let mut out = got.clone();
            out.vstack_mut(&second.feed(&df.slice(3, 3), None).unwrap())
                .unwrap();
            out.vstack_mut(&second.finish().unwrap()).unwrap();
            assert!(out.equals_missing(&want), "{cap}, {window}");
        }
        assert!(
            WindowsRun::load_bytes(&bytes, call("6s", "2s"), df.schema())
                .err()
                .expect("another length is another call")
                .contains("another call")
        );
    }

    fn values_ptr(df: &DataFrame, c: &str) -> *const f64 {
        df.column(c)
            .unwrap()
            .f64()
            .unwrap()
            .cont_slice()
            .unwrap()
            .as_ptr()
    }

    /// The rows a look-ahead holds are the input's own chunks, not copies:
    /// the output's values sit at the input's addresses.
    /// Review R1, N7 and N8: a formula with no operator beside one with an
    /// operator was accepted (Python refused it), and an input under the
    /// operators' prefix would have collided with a hidden column at run
    /// time.
    #[test]
    fn each_formula_holds_an_operator_and_reads_no_reserved_column() {
        let input = frame(&[0.0, 1.0]).schema().clone();
        let err = WindowsRun::new(
            config(
                r#"{"formulas": [{"name": "y", "tree": ["ewm_sum", ["col", "x"], {"half_life": 1.0}]},
                                 {"name": "z", "tree": ["-", ["col", "x"], ["lit", 1.0]]}],
                   "clock": "t", "gap_cap": 100.0}"#,
            ),
            &input,
        )
        .err()
        .expect("refused");
        assert!(err.contains("\"z\" holds no operator"), "{err}");
        let mut with_hidden: Schema = (*input).clone();
        with_hidden.insert("@po:x".into(), DataType::Float64);
        let err = WindowsRun::new(
            config(
                r#"{"formulas": [{"name": "y", "tree": ["ewm_sum", ["col", "@po:x"], {"half_life": 1.0}]}],
                   "clock": "t", "gap_cap": 100.0}"#,
            ),
            &with_hidden,
        )
        .err()
        .expect("refused");
        assert!(err.contains("reserved"), "{err}");
    }

    #[test]
    fn held_rows_are_the_inputs_own_chunks() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let out = run.feed(&df, None).unwrap();
        assert_eq!(out.height(), 3, "rows 0..3 resolved by row 5");
        assert_eq!(values_ptr(&out, "x"), values_ptr(&df, "x"));
        assert_eq!(run.held(), 3);
        let rest = run.finish().unwrap();
        assert_eq!(rest.height(), 3);
        assert!(
            rest.column("f")
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .all(|v| v.is_none())
        );
    }

    /// The schema is the input's columns then the formulas, typed by what
    /// they compute.
    #[test]
    fn the_schema_types_each_formula() {
        let df = frame(&[0.0, 1.0]);
        let cfg = config(
            r#"{"formulas": [
                {"name": "m", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]},
                {"name": "up", "tree": [">", ["ewm_sum", ["col", "x"], {"half_life": 1}], ["lit", 0]]},
                {"name": "dx", "tree": ["increment", ["col", "x"]]}
            ], "clock": "t", "gap_cap": 100}"#,
        );
        let run = WindowsRun::new(cfg, df.schema()).unwrap();
        let s = run.schema();
        assert_eq!(
            s.iter_names().map(|n| n.to_string()).collect::<Vec<_>>(),
            ["t", "x", "wide", "m", "up", "dx"]
        );
        assert_eq!(s.get("up"), Some(&DataType::Boolean));
        assert_eq!(s.get("dx"), Some(&DataType::Float64));
        assert_eq!(run.needed(), vec!["t".to_string(), "x".to_string()]);
    }

    /// A formula that cannot run is refused while the plan is built, by
    /// name.
    #[test]
    fn a_formula_that_cannot_run_is_refused_by_name() {
        let df = frame(&[0.0, 1.0]);
        for (json, says) in [
            (
                r#"{"formulas": [], "clock": "t", "gap_cap": 1}"#,
                "no formulas",
            ),
            (
                r#"{"formulas": [{"name": "x", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
                "already a column of the frame",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]}, {"name": "a", "tree": ["ewm_sum", ["col", "x"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
                "two expressions are named",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "nope"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
                "no input column \"nope\"",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "wide"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
                "input must be a number, got str",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["-", ["col", "x"], ["lit", 1]]}], "clock": "t", "gap_cap": 1}"#,
                "\"a\" holds no operator",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "x"], {"half_life": "1s"}]}], "clock": "t", "gap_cap": 1}"#,
                "half_life is a duration",
            ),
            (
                r#"{"formulas": [{"name": "a", "tree": ["shift", ["col", "x"]]}], "clock": "t", "gap_cap": 1}"#,
                "\"shift\" is not read",
            ),
        ] {
            let err = match serde_json::from_str::<WindowsConfig>(json) {
                Ok(c) => WindowsRun::new(c, df.schema()).err().expect("refused"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(says), "{json}: {err}");
        }
    }

    /// An increment is one row back within the group's session, and seconds
    /// on a temporal column.
    #[test]
    fn an_increment_is_one_row_back_within_the_session() {
        let df = df!(
            "t" => [0.0, 1.0, 2.0, 3.0, 4.0],
            "s" => ["a", "a", "b", "b", "b"],
            "c" => [10.0, 12.0, 15.0, 15.5, 20.0],
        )
        .unwrap();
        let cfg = config(
            r#"{"formulas": [{"name": "dc", "tree": ["increment", ["col", "c"]]},
                {"name": "dt", "tree": ["increment", ["col", "t"]]}],
                "clock": "t", "gap_cap": 100, "session": "s", "session_gap": 1}"#,
        );
        let mut run = WindowsRun::new(cfg, df.schema()).unwrap();
        let mut out = run.feed(&df.slice(0, 2), None).unwrap();
        out.vstack_mut(&run.feed(&df.slice(2, 3), None).unwrap())
            .unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        let dc: Vec<Option<f64>> = out.column("dc").unwrap().f64().unwrap().iter().collect();
        assert_eq!(dc, vec![None, Some(2.0), None, Some(0.5), Some(4.5)]);
        let dt: Vec<Option<f64>> = out.column("dt").unwrap().f64().unwrap().iter().collect();
        assert_eq!(dt, vec![None, Some(1.0), None, Some(1.0), Some(1.0)]);
    }

    /// A state resumes only its own call, and goes on from where it was.
    #[test]
    fn a_state_resumes_its_own_call() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut whole = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut want = whole.feed(&df, None).unwrap();
        want.vstack_mut(&whole.finish().unwrap()).unwrap();
        let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut got = first.feed(&df.slice(0, 3), None).unwrap();
        let bytes = first.save_bytes().unwrap();
        let other = config(
            r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 2, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#,
        );
        let err = WindowsRun::load_bytes(&bytes, other, df.schema())
            .err()
            .expect("refused");
        assert!(err.contains("another call"));
        let mut second = WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).unwrap();
        got.vstack_mut(&second.feed(&df.slice(3, 4), None).unwrap())
            .unwrap();
        got.vstack_mut(&second.finish().unwrap()).unwrap();
        assert!(got.equals_missing(&want));
    }

    /// Review R2, W4, then R4, B1 to B3: a state saved under a slice records
    /// the rows of its input the run consumed, skipped or fed, and the
    /// input's first clock. A run resumed on the same input skips them, so a
    /// chain of sliced runs reads the rest as one run would, rows held from
    /// an earlier run and returned counted as none of this input's; a run
    /// on an input that starts at another clock skips nothing.
    #[test]
    fn a_state_saved_under_a_slice_resumes_on_the_same_input() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut whole = one.feed(&df, None).unwrap();
        whole.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut out: Option<DataFrame> = None;
        let mut state: Option<Vec<u8>> = None;
        for _ in 0..4 {
            let mut run = match &state {
                None => WindowsRun::new(config(FORWARD), df.schema()).unwrap(),
                Some(b) => WindowsRun::load_bytes(b, config(FORWARD), df.schema()).unwrap(),
            };
            let part = run.feed(&df, Some(1)).unwrap();
            assert_eq!(part.height(), 1);
            state = Some(run.save_bytes_with(run.consumed(), false).unwrap());
            match &mut out {
                None => out = Some(part),
                Some(o) => {
                    o.vstack_mut(&part).unwrap();
                }
            }
        }
        let state = state.unwrap();
        let mut rest = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
        let mut tail = rest.feed(&df, None).unwrap();
        tail.vstack_mut(&rest.finish().unwrap()).unwrap();
        let mut both = out.unwrap();
        both.vstack_mut(&tail).unwrap();
        assert!(both.equals_missing(&whole), "{both} against {whole}");
        // The next file starts at another clock: none of its rows skipped.
        let later = frame(&[10.0, 11.0, 12.0]);
        let mut next = WindowsRun::load_bytes(&state, config(FORWARD), later.schema()).unwrap();
        let mut got = next.feed(&later, None).unwrap();
        got.vstack_mut(&next.finish().unwrap()).unwrap();
        let fed_later = got
            .column("t")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .filter(|t| *t >= 10.0)
            .count();
        assert_eq!(fed_later, 3, "{got}");
    }

    /// Review R2, W5: a state written by version 2 (before round one)
    /// loaded with defaults for the fields round one added, and
    /// misbehaved; the version is read before the rest, and refused by
    /// number.
    #[test]
    fn a_state_of_another_version_is_refused_by_its_version() {
        #[derive(Serialize)]
        struct Old<'a> {
            magic: &'a str,
            version: u32,
        }
        let input = frame(&[0.0]).schema().clone();
        let old = rmp_serde::to_vec_named(&Old {
            magic: WINDOWS_MAGIC,
            version: 2,
        })
        .unwrap();
        let err = WindowsRun::load_bytes(&old, config(FORWARD), &input)
            .err()
            .expect("refused");
        assert!(
            err.contains("state version 2 not supported (this build reads 9)"),
            "{err}"
        );
        let other = rmp_serde::to_vec_named(&Old {
            magic: "something else",
            version: WINDOWS_VERSION,
        })
        .unwrap();
        let err = WindowsRun::load_bytes(&other, config(FORWARD), &input)
            .err()
            .expect("refused");
        assert!(err.contains("not a state it saved"), "{err}");
    }

    /// Review R5, C1: a resumed run that satisfies its limit from the rows the
    /// state held, while a loaded skip still spans the chunks it has fed, saves
    /// the rows consumed of the input so far -- the loaded skip included.
    #[test]
    fn a_resumed_run_fed_one_row_at_a_time_saves_the_whole_count() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut whole = one.feed(&df, None).unwrap();
        whole.vstack_mut(&one.finish().unwrap()).unwrap();
        // Run 1: head(2) reads rows 0..5: row 0 closes at t = 3, and row 4 at
        // t = 10 closes rows 1..3 at once, so rows 2 and 3 stay held, resolved.
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut out = run.feed(&df, Some(2)).unwrap();
        assert_eq!(run.consumed(), 5);
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        // Run 2, one row a chunk, head(1): the first chunk is skipped, the
        // held row 2 satisfies the limit, the skip still spans four chunks.
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
        let mut i = 0;
        loop {
            let part = run.feed(&df.slice(i, 1), Some(1)).unwrap();
            i += 1;
            if part.height() == 1 {
                out.vstack_mut(&part).unwrap();
                break;
            }
        }
        assert_eq!(run.consumed(), 5, "the loaded skip not yet applied counts");
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        // Run 3 on the whole input: the rest, as one run would give it.
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
        let mut rest = run.feed(&df, None).unwrap();
        rest.vstack_mut(&run.finish().unwrap()).unwrap();
        out.vstack_mut(&rest).unwrap();
        assert!(out.equals_missing(&whole), "{out} against {whole}");
    }

    /// Review R5, C3/C4 and C6: a sliced state knows its input by the rows it
    /// held and its first clock, and refuses another input by name; a state
    /// cut short is reported as damaged.
    #[test]
    fn a_sliced_state_refuses_another_input_and_a_damaged_file_says_so() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        run.feed(&df, Some(2)).unwrap();
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        // The same first clock, other rows after it.
        let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), other.schema()).unwrap();
        let err = run.feed(&other, None).expect_err("refused");
        assert!(err.to_string().contains("another input"), "{err}");
        // The input sliced by hand.
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
        let err = run.feed(&df.slice(2, 5), None).expect_err("refused");
        assert!(err.to_string().contains("another input"), "{err}");
        // A file cut short.
        let cut = &state[..state.len() / 2];
        let err = WindowsRun::load_bytes(cut, config(FORWARD), df.schema())
            .err()
            .expect("refused");
        assert!(err.contains("damaged"), "{err}");
    }

    /// Review R4, A2 and R6, D5: a windows state version moves the bank's
    /// schema with it, since a bank file embeds one per formula target and
    /// refuses an older form by number, not at a group's first chunk.
    #[test]
    fn a_windows_state_version_moves_the_banks_schema_with_it() {
        // 28 to 39 moved for `ew_cov`'s PCA cadence,
        // `micro`'s pruning, the models' window cadence, `lasso`'s
        // per-target thresholds, the windows' stamps, the quantile fit's
        // band systems, the embargo's elapsed clock, `coef_every` on the
        // clock, `bocpd`'s hazard on the clock, the solve, component and
        // checkpoint cadences on the exact clock, `ewridge`'s kept
        // systems, the stream's and the bank's state and task 195's
        // residual scales, scaler, warm-up and per-target thresholds (tasks
        // 161, 163, 162, 174, 175, 170, 176, 178, 179, 180, 186, 194, 195),
        // and 41 for the models' window edge and task 196's names, the
        // windows state unchanged: a schema may move alone, a windows
        // version may not. 42 (task 200) moved with windows state 8, an
        // integer clock held as an integer in its rows; 43 for the relative
        // targets' removal (task 201), 44 for task 202's target spreads,
        // 45 for task 116's readiness statistics, 46 for the readiness
        // notices' waits (task 208) and 47 for task 206's warm-up of a
        // standardizing fit, the windows state unchanged; 48 (task 212)
        // with windows state 9, a variance's queue six wide.
        assert_eq!((WINDOWS_VERSION, online_core::SCHEMA_VERSION), (9, 48));
    }

    /// Review R6, D2: a run on the next file under a slice keeps the first
    /// file's unresolved rows held ahead of its own -- rows go out in order,
    /// so none of its own went out while one of theirs waited -- and the
    /// state then holds more rows than it consumed of this input. The
    /// identity is the last `consumed` of the held rows, this input's own;
    /// a resume on the second file goes on.
    #[test]
    fn a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held() {
        let first = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let second = frame(&[10.0, 11.0, 12.0]);
        let whole = first.vstack(&second).unwrap();
        let mut one = WindowsRun::new(config(FORWARD), whole.schema()).unwrap();
        let mut want = one.feed(&whole, None).unwrap();
        want.vstack_mut(&one.finish().unwrap()).unwrap();
        // The first file, unsliced: rows 3..6 stay held, their windows open.
        let mut run = WindowsRun::new(config(FORWARD), first.schema()).unwrap();
        let mut out = run.feed(&first, None).unwrap();
        assert_eq!(run.held(), 3);
        let state = run.save_bytes().unwrap();
        // The second file under head(1): its first row closes the three, one
        // goes out, two stay held ahead of it; one row of it consumed.
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), second.schema()).unwrap();
        out.vstack_mut(&run.feed(&second, Some(1)).unwrap())
            .unwrap();
        assert_eq!((run.consumed(), run.held()), (1, 3));
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        // The second file resumed, unsliced: the rest, as one run would give it.
        let mut run = WindowsRun::load_bytes(&state, config(FORWARD), second.schema()).unwrap();
        out.vstack_mut(&run.feed(&second, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{out} against {want}");
    }

    /// Review R6, D1: a backward operator under "left" resolves a row at its
    /// own push, so a sliced run holds no row, and the held rows identify
    /// nothing; the state keeps the last row it read and refuses another
    /// input by it, with a clock column and without (a row-count clock
    /// repeats no stamp, so every operator resolves at the push).
    #[test]
    fn a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        const BARE: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}]}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5]);
        for (cfg, other) in [(LEFT, other), (BARE, df.slice(3, 4))] {
            let mut one = WindowsRun::new(config(cfg), df.schema()).unwrap();
            let mut want = one.feed(&df, None).unwrap();
            want.vstack_mut(&one.finish().unwrap()).unwrap();
            let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
            let mut out = run.feed(&df, Some(3)).unwrap();
            assert_eq!(
                (out.height(), run.consumed(), run.held()),
                (3, 3, 0),
                "{cfg}"
            );
            let state = run.save_bytes_with(run.consumed(), false).unwrap();
            let mut run = WindowsRun::load_bytes(&state, config(cfg), df.schema()).unwrap();
            let err = run.feed(&other, None).expect_err("refused");
            assert!(err.to_string().contains("another input"), "{cfg}: {err}");
            let mut run = WindowsRun::load_bytes(&state, config(cfg), df.schema()).unwrap();
            out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
            out.vstack_mut(&run.finish().unwrap()).unwrap();
            assert!(out.equals_missing(&want), "{cfg}: {out} against {want}");
        }
    }

    /// Review R6, D3 and D4: an input that starts elsewhere is the next file
    /// when the clock policy takes its first row as a step forward or a new
    /// start (a step back past `restart_after_step_back`, a new session),
    /// and is refused when it starts at the last stamp the state read (a
    /// file boundary inside a tied stamp cannot be told from the same input
    /// sliced inside it) or steps back where the policy refuses it.
    #[test]
    fn a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file() {
        let day = |t: &[f64], s: &str| {
            df!(
                "t" => t,
                "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
                "s" => vec![s; t.len()],
            )
            .unwrap()
        };
        let day1 = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "a");
        let day2 = day(&[0.5, 1.5, 2.5, 3.5], "b");
        let restart = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "restart_after_step_back": 2}"#;
        let session = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "session": "s", "session_gap": 1}"#;
        for cfg in [restart, session] {
            let whole = day1.vstack(&day2).unwrap();
            let mut one = WindowsRun::new(config(cfg), whole.schema()).unwrap();
            let mut want = one.feed(&whole, None).unwrap();
            want.vstack_mut(&one.finish().unwrap()).unwrap();
            // Day 1 under a slice its input exhausts: a state with a skip.
            let mut run = WindowsRun::new(config(cfg), day1.schema()).unwrap();
            let mut out = run.feed(&day1, Some(100)).unwrap();
            assert!(run.held() > 0 && run.consumed() == 7);
            let state = run.save_bytes_with(run.consumed(), true).unwrap();
            // Day 2 steps back: a new start by the policy's word, the next file.
            let mut run = WindowsRun::load_bytes(&state, config(cfg), day2.schema()).unwrap();
            out.vstack_mut(&run.feed(&day2, None).unwrap()).unwrap();
            out.vstack_mut(&run.finish().unwrap()).unwrap();
            assert!(out.equals_missing(&want), "{cfg}: {out} against {want}");
        }
        // Review R7, E2: day 2 at the same first stamp as day 1. With a
        // session column the first row's session tells the file from the
        // saved input, and the policy takes it; without one it is the saved
        // input until the rows differ, refused by name.
        let same = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "b");
        let whole = day1.vstack(&same).unwrap();
        let mut one = WindowsRun::new(config(session), whole.schema()).unwrap();
        let mut want = one.feed(&whole, None).unwrap();
        want.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut run = WindowsRun::new(config(session), day1.schema()).unwrap();
        let mut out = run.feed(&day1, Some(100)).unwrap();
        let state = run.save_bytes_with(run.consumed(), true).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(session), same.schema()).unwrap();
        out.vstack_mut(&run.feed(&same, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{out} against {want}");
        let mut run = WindowsRun::new(config(restart), day1.schema()).unwrap();
        run.feed(&day1, Some(100)).unwrap();
        let state = run.save_bytes_with(run.consumed(), true).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(restart), same.schema()).unwrap();
        let err = run.feed(&same, None).expect_err("refused");
        assert!(err.to_string().contains("another input"), "{err}");
        // Under the default policy a step back is refused, and so is a tie.
        let plain = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#;
        let mut run = WindowsRun::new(config(plain), day1.schema()).unwrap();
        run.feed(&day1, Some(100)).unwrap();
        let state = run.save_bytes_with(run.consumed(), true).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(plain), day2.schema()).unwrap();
        let err = run.feed(&day2, None).expect_err("refused");
        assert!(
            err.to_string()
                .contains("before the last row the state read"),
            "{err}"
        );
        let tied = day(&[6.0, 7.0, 8.0], "a");
        let mut run = WindowsRun::load_bytes(&state, config(plain), tied.schema()).unwrap();
        let err = run.feed(&tied, None).expect_err("refused");
        assert!(
            err.to_string().contains("at the last stamp the state read"),
            "{err}"
        );
    }

    /// Review R5, C1 (the plan's row, unpinned) and C3: a resumed run that
    /// saw no row saves the identity it loaded, so the chain goes on from
    /// that state as from the one before it; a save that says the input
    /// ended refuses a skip still pending, one that does not takes it.
    #[test]
    fn a_resumed_run_that_saw_no_row_saves_the_identity_it_loaded() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut whole = one.feed(&df, None).unwrap();
        whole.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        let mut out = run.feed(&df, Some(2)).unwrap();
        assert_eq!(run.consumed(), 5);
        let first = run.save_bytes_with(run.consumed(), false).unwrap();
        // Loaded and saved again without a row: the same identity and count.
        let run = WindowsRun::load_bytes(&first, config(FORWARD), df.schema()).unwrap();
        assert_eq!(run.consumed(), 5);
        let again = run.save_bytes_with(run.consumed(), false).unwrap();
        let mut run = WindowsRun::load_bytes(&again, config(FORWARD), df.schema()).unwrap();
        out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&whole), "{out} against {whole}");
        // Mid-skip: the input is not over unless the caller says so.
        let mut run = WindowsRun::load_bytes(&first, config(FORWARD), df.schema()).unwrap();
        run.feed(&df.slice(0, 2), None).unwrap();
        assert_eq!(run.consumed(), 5);
        run.save_bytes_with(run.consumed(), false)
            .expect("a slice satisfied mid-skip");
        let err = run
            .save_bytes_with(run.consumed(), true)
            .expect_err("refused");
        assert!(err.contains("ended after 2"), "{err}");
        // Review R7, E5: the identity itself survives a re-save without a
        // row -- the last row read, where nothing is held, refuses another
        // input with the same first clock.
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, Some(3)).unwrap();
        let first = run.save_bytes_with(run.consumed(), false).unwrap();
        let run = WindowsRun::load_bytes(&first, config(LEFT), df.schema()).unwrap();
        let again = run.save_bytes_with(run.consumed(), false).unwrap();
        let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5]);
        let mut run = WindowsRun::load_bytes(&again, config(LEFT), other.schema()).unwrap();
        let err = run.feed(&other, None).expect_err("refused");
        assert!(err.to_string().contains("another input"), "{err}");
    }

    /// Review R7, E3: the last row read is a sliced state's identity and
    /// nothing else's, so an unsliced state that holds no rows resumes on an
    /// input with a column more, as it did before the row was kept; a
    /// sliced one names the columns it has against the frame's.
    #[test]
    fn the_last_row_binds_a_sliced_state_only() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0]);
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, None).unwrap();
        assert_eq!(run.held(), 0);
        let state = run.save_bytes().unwrap();
        // The next file, with a column more.
        let wider = frame(&[4.0, 5.0, 6.0, 7.0])
            .lazy()
            .with_columns([lit(1.0).alias("extra")])
            .collect()
            .unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), wider.schema()).unwrap();
        let out = run.feed(&wider, None).unwrap();
        assert_eq!(out.height(), 4);
        assert!(out.get_column_names().iter().any(|c| c.as_str() == "extra"));
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, Some(2)).unwrap();
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        let err = WindowsRun::load_bytes(&state, config(LEFT), wider.schema())
            .err()
            .expect("refused");
        assert!(err.contains("the state's last row has columns"), "{err}");
    }

    /// Task 158, E13: a state saved without a slice refuses an input that
    /// starts at the last stamp it read -- here one that repeats its last
    /// row, which came out twice -- as a sliced state does, and goes on with
    /// one that starts after it.
    #[test]
    fn an_unsliced_state_refuses_an_input_at_the_last_stamp_it_read() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0]);
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, None).unwrap();
        let state = run.save_bytes().unwrap();
        let again = frame(&[3.0, 4.0, 5.0]);
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), again.schema()).unwrap();
        let err = run.feed(&again, None).expect_err("refused");
        assert!(
            err.to_string().contains("at the last stamp the state read"),
            "{err}"
        );
        let next = frame(&[4.0, 5.0, 6.0]);
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), next.schema()).unwrap();
        assert_eq!(run.feed(&next, None).unwrap().height(), 3);
        // A fresh run has read no stamp.
        let mut run = WindowsRun::new(config(LEFT), again.schema()).unwrap();
        assert_eq!(run.feed(&again, None).unwrap().height(), 3);
    }

    /// Task 159 (F1): a run marked itself started, and kept its input's
    /// first row, before the identity verdict, so a refused first chunk was
    /// taken whole on the next call. A refused first chunk leaves the run as
    /// it was: fed again, it is refused again, with the same words.
    #[test]
    fn a_refused_first_chunk_is_refused_again() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0]);
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, None).unwrap();
        let state = run.save_bytes().unwrap();
        let again = frame(&[3.0, 4.0, 5.0]);
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), again.schema()).unwrap();
        let first = run.feed(&again, None).expect_err("refused").to_string();
        let second = run
            .feed(&again, None)
            .expect_err("refused again")
            .to_string();
        assert!(
            first.contains("at the last stamp the state read"),
            "{first}"
        );
        assert_eq!(first, second);
        // And the run goes on with the next file as if nothing had been fed.
        let next = frame(&[4.0, 5.0, 6.0]);
        assert_eq!(run.feed(&next, None).unwrap().height(), 3);
    }

    /// Task 159 (W1): a restart on the stream's clock starts every group's
    /// clock over with its windows, so a group's next row within
    /// `restart_after_step_back` of its last is the first on a fresh clock,
    /// not a late row: C at 52, 8 after C at 60, once B at 50 restarted the
    /// stream after A at 100.
    #[test]
    fn a_stream_restart_restarts_every_groups_clock() {
        const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_sum", ["col", "x"],
            {"half_life": 1e300}]}], "clock": "t", "gap_cap": 1e9, "group": "g",
            "restart_after_step_back": 10}"#;
        let df = df!(
            "t" => [0.0, 60.0, 100.0, 50.0, 52.0],
            "g" => ["A", "C", "A", "B", "C"],
            "x" => [1.0; 5]
        )
        .unwrap();
        let mut run = WindowsRun::new(config(GROUPED), df.schema()).unwrap();
        let fed = run.feed(&df, None).expect("every row taken").height();
        let rest = run.finish().unwrap().height();
        assert_eq!(fed + rest, 5);
    }

    /// Task 159 (F2): a key is its value's text, so a group column of another
    /// dtype on a resumed input started every key over, silently. The state
    /// carries the group and session columns' dtypes, and refuses another.
    #[test]
    fn a_state_refuses_a_group_column_of_another_dtype() {
        const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}], "clock": "t", "gap_cap": 100, "group": "g"}"#;
        let df =
            df!("t" => [0.0, 1.0, 2.0, 3.0], "x" => [1.0, 2.0, 3.0, 4.0], "g" => [1i64, 1, 2, 2])
                .unwrap();
        let mut run = WindowsRun::new(config(GROUPED), df.schema()).unwrap();
        run.feed(&df, None).unwrap();
        let state = run.save_bytes().unwrap();
        let next = df!("t" => [4.0, 5.0], "x" => [5.0, 6.0], "g" => [1.0f64, 2.0]).unwrap();
        let err = WindowsRun::load_bytes(&state, config(GROUPED), next.schema())
            .err()
            .expect("refused");
        assert!(
            err.contains("group column \"g\": i64") && err.contains("f64"),
            "{err}"
        );
        let same = df!("t" => [4.0, 5.0], "x" => [5.0, 6.0], "g" => [1i64, 2]).unwrap();
        WindowsRun::load_bytes(&state, config(GROUPED), same.schema()).unwrap();
    }

    /// Review R8, F1: on a row-count clock every row is a step forward, so
    /// a step forward says nothing; without a clock column the next file
    /// begins with a new session, and a hand slice starting in the session
    /// the state last read is refused.
    #[test]
    fn without_a_clock_the_next_file_begins_with_a_new_session() {
        const ROWS: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}], "session": "s", "session_gap": 1}"#;
        let rows = |x: &[f64], s: &[&str]| df!("x" => x, "s" => s).unwrap();
        let df = rows(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &["a", "a", "a", "b", "b", "b"],
        );
        let mut run = WindowsRun::new(config(ROWS), df.schema()).unwrap();
        let mut out = run.feed(&df, Some(4)).unwrap();
        assert_eq!((out.height(), run.consumed(), run.held()), (4, 4, 0));
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        for cut in [3, 5] {
            let mut run = WindowsRun::load_bytes(&state, config(ROWS), df.schema()).unwrap();
            let err = run
                .feed(&df.slice(cut, df.height() - cut as usize), None)
                .expect_err("refused");
            assert!(
                err.to_string()
                    .contains("does not begin with a new session"),
                "{cut}: {err}"
            );
        }
        let next = rows(&[7.0, 8.0], &["c", "c"]);
        let whole = df.head(Some(4)).vstack(&next).unwrap();
        let mut one = WindowsRun::new(config(ROWS), whole.schema()).unwrap();
        let mut want = one.feed(&whole, None).unwrap();
        want.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(ROWS), next.schema()).unwrap();
        out.vstack_mut(&run.feed(&next, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{out} against {want}");
    }

    /// Review R8, F2: where the policy refuses the first row of an input
    /// that starts elsewhere, the refusal says what the policy said -- under
    /// `group` a next-day file's step back is refused on the stream's clock
    /// whatever its session, as the unsliced path refuses it, naming
    /// `restart_after_step_back`.
    #[test]
    fn a_refused_first_row_carries_the_policys_words() {
        const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "group": "g", "session": "s", "session_gap": 1}"#;
        let day = |t: &[f64], s: &str| {
            df!(
                "t" => t,
                "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
                "g" => vec!["k"; t.len()],
                "s" => vec![s; t.len()],
            )
            .unwrap()
        };
        let day1 = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "a");
        let day2 = day(&[0.0, 1.0, 2.0, 3.0], "b");
        let mut run = WindowsRun::new(config(GROUPED), day1.schema()).unwrap();
        run.feed(&day1, Some(100)).unwrap();
        let state = run.save_bytes_with(run.consumed(), true).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(GROUPED), day2.schema()).unwrap();
        let err = run.feed(&day2, None).expect_err("refused").to_string();
        assert!(
            err.contains("another input") && err.contains("restart_after_step_back"),
            "{err}"
        );
    }

    /// Review R7, E4 and R8, F3: a run that stops inside the skip (a limit
    /// of 0 on a chunk shorter than the skip) saves the count and the
    /// identity it loaded, not the partial skip's last row, so the next
    /// resume on the same input goes on.
    #[test]
    fn a_save_inside_a_pending_skip_keeps_the_loaded_identity() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut one = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        let mut whole = one.feed(&df, None).unwrap();
        whole.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        let mut out = run.feed(&df, Some(3)).unwrap();
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), df.schema()).unwrap();
        assert_eq!(run.feed(&df.slice(0, 1), Some(0)).unwrap().height(), 0);
        assert_eq!(run.consumed(), 3);
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(LEFT), df.schema()).unwrap();
        out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&whole), "{out} against {whole}");
    }

    /// Review R8, F5: a state saved under a slice carries the rows it knows
    /// its input by; one that carries none is damaged, refused at load.
    #[test]
    fn a_sliced_state_without_its_identity_is_damaged() {
        const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
        let df = frame(&[0.0, 1.0, 2.0, 3.0]);
        let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
        run.feed(&df, Some(2)).unwrap();
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        let file: FileIn = rmp_serde::from_slice(&state).unwrap();
        let stripped = rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            version: WINDOWS_VERSION,
            config: &file.config,
            core: &file.core,
            held: &file.held,
            increments: &file.increments,
            stream_clock: file.stream_clock,
            key_dtypes: &file.key_dtypes,
            resume_skip: file.resume_skip,
            resume_first: file.resume_first,
            resume_first_session: file.resume_first_session,
            last_row: &[],
        })
        .unwrap();
        let err = WindowsRun::load_bytes(&stripped, config(LEFT), df.schema())
            .err()
            .expect("refused");
        assert!(err.contains("damaged"), "{err}");
    }

    /// Review 2026-10-06, PD1: the core a state carries is held to its own
    /// invariants at load, beyond the kernels, the operators, the clock and
    /// the held count the loader compared: a held value short, an output
    /// count below the operators, a queue of another width, a silent-list
    /// link past the groups, more rows ready than held, or a waiting row
    /// before the held ones loaded, and `finish`'s `emit` or the next
    /// `push` indexed `held_values[r * n_out + o]` past its end.
    #[test]
    fn a_state_whose_core_is_damaged_is_refused() {
        let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
        first.feed(&df.slice(0, 4), None).unwrap();
        let bytes = first.save_bytes().unwrap();
        let file: FileIn = rmp_serde::from_slice(&bytes).unwrap();
        assert!(file.core.held() > 0, "rows wait for their forward windows");
        for what in [
            "held_values",
            "n_outputs",
            "queue",
            "link",
            "ready",
            "waiting",
        ] {
            let mut core = file.core.clone();
            core.damage(what);
            let damaged = rmp_serde::to_vec_named(&FileOut {
                magic: WINDOWS_MAGIC,
                version: WINDOWS_VERSION,
                config: &file.config,
                core: &core,
                held: &file.held,
                increments: &file.increments,
                stream_clock: file.stream_clock,
                key_dtypes: &file.key_dtypes,
                resume_skip: file.resume_skip,
                resume_first: file.resume_first,
                resume_first_session: file.resume_first_session,
                last_row: &file.last_row,
            })
            .unwrap();
            match WindowsRun::load_bytes(&damaged, config(FORWARD), df.schema()) {
                Err(err) => assert!(err.contains("the state is damaged"), "{what}: {err}"),
                Ok(mut run) => {
                    let fed = run.feed(&df.slice(4, 2), None).map(|f| f.height());
                    let done = run.finish().map(|f| f.height());
                    panic!("{what}: loaded, fed {fed:?} and finished {done:?}");
                }
            }
        }
        assert!(WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).is_ok());
    }
}
