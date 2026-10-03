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

use crate::arrow::nanos_array;
use crate::formula::{Formula, Node, OpNode};
use crate::span::{Span, format_duration};
use crate::spec::{ClockPolicy, SessionGapSpec, clock_cfg_of};
use crate::stream::usable;
use crate::windows::{
    Closed, Direction, KernelDef, OpDef, OpKind, Partial, Refusal, RowIn, Stat, Windows,
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
/// 2 since task 143: formulas over operators, where 1 held descriptions.
const WINDOWS_VERSION: u32 = 2;

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
}

#[derive(Deserialize)]
struct FileIn {
    magic: String,
    version: u32,
    config: WindowsConfig,
    core: Windows,
    held: Vec<u8>,
    increments: Vec<IncrState>,
}

/// A group's increment state: the previous row's inputs, session and clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct IncrState {
    key: Option<String>,
    /// Per increment: the previous value, NaN for none; a temporal input in
    /// nanoseconds.
    prev: Vec<f64>,
    prev_ns: Vec<Option<i64>>,
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
                "{WHO}: output {:?} is already a column of the frame; give the expression another \
                 name",
                f.name
            ));
        }
        if !names.insert(&f.name) {
            return Err(format!("{WHO}: two expressions are named {:?}", f.name));
        }
        for c in f.tree.columns() {
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
            });
            p.columns.push((key, p.ops.len() - 1));
        }
    }
    if p.ops.is_empty() && p.increments.is_empty() {
        return Err(format!(
            "{WHO}: no formula holds an operator; a formula of the row alone is Polars' \
             with_columns"
        ));
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
fn session_hash(v: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// One `with_windows` run: the core, its plan, and the input rows it holds.
pub struct WindowsRun {
    config: WindowsConfig,
    core: Windows,
    inputs: Vec<Node>,
    columns: Vec<(String, usize)>,
    increments: Vec<(String, Node, bool)>,
    accept: Vec<String>,
    /// The input's schema, as the call was built against it.
    input: Schema,
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
    /// The core's row count at this run's first row: an error names the
    /// row of this run's input.
    run_base: u64,
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
            gap_cap: config.gap_cap.as_ref(),
            restart_after_step_back: config.restart_after_step_back.as_ref(),
            session: config.session.as_deref(),
            session_gap: config.session_gap.as_ref(),
            spec_closes_on_session: None,
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
        let core = if p.ops.is_empty() {
            // Increments alone: the core carries a kernel that reads nothing,
            // so rows pass straight through it. A sum over no window, read
            // at every row.
            Windows::new(
                vec![KernelDef {
                    direction: Direction::Backward,
                    half_life: f64::INFINITY,
                    window_size: Some(1.0),
                    closed: Closed::Left,
                }],
                vec![OpDef {
                    kernel: 0,
                    stat: Stat::Sum,
                    input: 0,
                    min_samples: 1,
                    partial: Partial::Keep,
                }],
                cfg,
            )
            .map_err(|e| format!("{WHO}: {e}"))?
        } else {
            Windows::new(p.kernels, p.ops, cfg).map_err(|e| format!("{WHO}: {e}"))?
        };
        let inputs = if p.inputs.is_empty() {
            vec![Node::Lit(crate::formula::Literal::Float(0.0))]
        } else {
            p.inputs
        };
        Ok(Self {
            config,
            core,
            inputs,
            columns: p.columns,
            increments: p.increments,
            accept,
            input: input.clone(),
            output,
            held: VecDeque::new(),
            incr_state: Vec::new(),
            incr_index: HashMap::new(),
            incr_null: None,
            run_base: 0,
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
        let n = df.height();
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
        let keys: Vec<Option<&str>> = (0..n)
            .map(|i| match groups {
                None => Some(""),
                Some(g) => g.get(i),
            })
            .collect();

        // Pass one: the increments, one row back within the group's
        // session, as hidden columns of the chunk.
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
            let group = self.core.group(keys[i]);
            let row = RowIn {
                group,
                clock: clock.as_ref().map(|c| c[i]),
                session: session.as_ref().map(|s| s[i]),
                values: &row_values,
                accept: accepts.iter().all(|col| usable(col[i])),
            };
            if let Err(r) = self.core.push(&row) {
                // The rows before it are in the core: hold them.
                if i > 0 {
                    self.held.push_back(chunk.slice(0, i));
                }
                return Err(self.refusal(r, chunk_base));
            }
        }
        if consumed > 0 {
            self.held.push_back(chunk.slice(0, consumed));
        }
        let e = match limit {
            None => self.core.drain(),
            Some(l) => self.core.drain_kept(l),
        };
        self.assemble(e)
    }

    /// The end of the input: every row still held goes out, each forward
    /// window still open over it null.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        let e = self.core.finish();
        self.assemble(e)
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
        let numeric: Vec<Option<Vec<Option<f64>>>> = (0..self.increments.len())
            .map(|i| {
                if self.increments[i].2 {
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
                    let c = inputs.column(input_column(i).as_str())?;
                    let ns = nanos_array(c.as_materialized_series(), 0)?;
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
            let st = &mut self.incr_state[gi];
            let now_session = session.map(|s| s[r]);
            let now_clock = clock.as_ref().map(|c| c[r]);
            let mut new_start = st.session != now_session && st.session.is_some();
            if let (Some(now), Some(prev)) = (now_clock, st.clock) {
                let back = match (prev, now) {
                    (ClockValue::Ns(p), ClockValue::Ns(c)) => {
                        online_core::seconds_of_ns(i128::from(p) - i128::from(c))
                    }
                    (p, c) => p.seconds() - c.seconds(),
                };
                if restarts && back > 0.0 && back > restart {
                    new_start = true;
                }
            }
            if new_start {
                st.prev.iter_mut().for_each(|p| *p = f64::NAN);
                st.prev_ns.iter_mut().for_each(|p| *p = None);
            }
            st.session = now_session;
            st.clock = now_clock;
            for i in 0..self.increments.len() {
                if let Some(vals) = &numeric[i] {
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
    fn refusal(&self, r: Refusal, _chunk_base: u64) -> PolarsError {
        let row = |seq: u64| seq - self.run_base;
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
        if e.drop.iter().any(|&d| d) {
            let keep: BooleanChunked = e.drop.iter().map(|&d| Some(!d)).collect();
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

    /// The state as bytes: the call, the core, the rows it holds, which the
    /// next run emits first, and the increments' state. Versioned msgpack,
    /// one state one file.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
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
        rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            version: WINDOWS_VERSION,
            config: &self.config,
            core: &self.core,
            held: &held,
            increments: &self.incr_state,
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
                "{WHO}: the state was saved by another call -- other formulas or another clock \
                 policy -- and resumes only the call that saved it"
            ));
        }
        let mut run = Self::new(config, input)?;
        if file.core.kernels() != run.core.kernels()
            || file.core.ops() != run.core.ops()
            || file.core.clock_cfg() != run.core.clock_cfg()
        {
            return Err(format!(
                "{WHO}: the state's operators do not match its own call"
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
                "no formula holds an operator",
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
}
