//! A spec's formula targets' resolvers (docs/PLAN.md task 104): task 143's
//! window core over the targets' formulas, one core per group under the
//! spec's clock policy, fed every row of every chunk before the streams run,
//! and what each chunk resolved, handed to the streams as `FormulaBundle`.
//! The bank (bank.rs) owns one `TargetWindows` per spec, calls
//! `prepare_targets` for every spec before feeding any (review R2, F1), then
//! `resolve_targets`, and saves the cores as `SavedResolvers`.

use std::collections::HashMap;

use polars::prelude::*;

use crate::arrow::{ArrowChunk, Form};
use crate::bank::{GroupKey, Layout, gathered};
use crate::formula::Formula;
use crate::spec::Spec;
use crate::stream::{FormulaTargets, Resolutions};
use crate::windows_frame::{WindowsConfig, WindowsRun};

/// One chunk's formula-target inputs for a spec's streams (docs/PLAN.md
/// task 104): which targets are formulas, each row's number in the bank's
/// stream in the laid-out row order, and what each group's core resolved.
pub(crate) struct FormulaBundle {
    slots: Vec<usize>,
    seqs: Vec<u64>,
    groups: HashMap<GroupKey, Resolutions>,
}

impl FormulaBundle {
    pub(crate) fn targets(&self, key: &GroupKey) -> FormulaTargets<'_> {
        FormulaTargets {
            slots: &self.slots,
            seqs: &self.seqs,
            resolved: self.groups.get(key),
        }
    }
}

/// A spec's formula targets' resolver (docs/PLAN.md task 104): task 143's
/// window core over the targets' formulas, under the spec's clock policy,
/// fed every row of every chunk -- rows the spec skips included, since a
/// target depends on the prices ahead and not on the row's features -- in
/// chunk order, before the streams run. One core per group, as the bank
/// keeps one stream per group: a group's rows resolve in their own order on
/// their own clock, so a resolution is known at the row that made it
/// whatever the other groups hold open. (One core over every group emits a
/// row only once every row before it, of any group, has resolved -- a wait
/// that moved with the chunking.) Each core is built at its group's first
/// chunk, which says what the columns are.
#[derive(Clone, Default)]
pub(crate) struct TargetWindows {
    pub(crate) runs: HashMap<GroupKey, WindowsRun>,
    /// A loaded bank's cores, each resumed at its group's first chunk.
    pub(crate) saved: HashMap<GroupKey, Vec<u8>>,
}

/// The row-number column a resolver's frame carries: a row's number in the
/// bank's stream, which a resolution names it by.
const ROW_NUMBER: &str = "@po:row";

/// The resolvers as a bank file carries them: per spec index, each group's
/// core as `WindowsRun::save_bytes` writes it.
pub(crate) type SavedResolvers = Vec<(usize, Vec<(GroupKey, Vec<u8>)>)>;

/// The call a spec's formula targets amount to: the formulas under the
/// spec's clock policy, over one group, with no `like=` (the core sees
/// every row).
fn resolver_config(spec: &Spec) -> WindowsConfig {
    WindowsConfig {
        formulas: spec
            .targets
            .defs()
            .iter()
            .filter_map(|t| {
                t.formula.as_ref().map(|tree| Formula {
                    name: t.name.clone(),
                    tree: tree.clone(),
                })
            })
            .collect(),
        clock: spec.clock.clone(),
        gap_cap: spec.gap_cap.clone(),
        restart_after_step_back: spec.restart_after_step_back.clone(),
        session: spec.session.clone(),
        session_gap: spec.session_gap.clone(),
        group: None,
        like: None,
    }
}

/// The columns a spec's resolver reads, each with the forms it takes them
/// in: the clock and the session as the chunk holds them, and a formula's
/// columns as numbers or text. The group is the core's key, not a column.
fn resolver_columns(spec: &Spec) -> Vec<(String, &'static [Form])> {
    const CLOCK: &[Form] = &[Form::Clock, Form::Key, Form::Number];
    const SESSION: &[Form] = &[Form::Text, Form::Key, Form::Number];
    // A column holds the boolean form only when it is a boolean, so that
    // form comes first: a boolean another role reads as a number has both
    // (review R2, F3). A formula's columns come first too, so a session
    // column a formula reads keeps its own form.
    const VALUE: &[Form] = &[Form::Bool, Form::Number, Form::Text, Form::Key];
    let mut out: Vec<(String, &'static [Form])> = Vec::new();
    let mut put = |c: &str, forms: &'static [Form]| {
        if !out.iter().any(|(n, _)| n == c) {
            out.push((c.to_string(), forms));
        }
    };
    for t in spec.targets.defs().iter().filter(|t| t.is_formula()) {
        for c in t.columns() {
            put(&c, VALUE);
        }
    }
    if let Some(c) = &spec.clock {
        put(c, CLOCK);
    }
    if let Some(c) = &spec.session {
        put(c, SESSION);
    }
    out
}

/// The frame a spec's resolver is fed: its columns, from the chunk, and
/// each row's number from `first_row` on.
fn resolver_frame(
    chunk: &ArrowChunk,
    spec: &Spec,
    columns: &[(String, &'static [Form])],
    first_row: u64,
) -> PolarsResult<DataFrame> {
    let n = chunk.height();
    let mut cols: Vec<Column> = Vec::with_capacity(columns.len() + 1);
    for (name, forms) in columns {
        let Some(s) = chunk.series(name, forms) else {
            let have: Vec<&str> = chunk.names().iter().map(|n| n.as_str()).collect();
            polars_bail!(ColumnNotFound:
                "spec {:?}: formula target column {:?} not found; the input has columns {:?}",
                spec.name, name, have
            );
        };
        cols.push(s.into_column());
    }
    let rows: UInt64Chunked = (0..n as u64).map(|i| Some(first_row + i)).collect();
    cols.push(rows.with_name(ROW_NUMBER.into()).into_column());
    DataFrame::new(n, cols)
}

impl TargetWindows {
    /// The group's core, built -- or resumed from the loaded state -- at
    /// its first chunk.
    fn run(
        &mut self,
        key: &GroupKey,
        spec: &Spec,
        frame: &DataFrame,
    ) -> PolarsResult<&mut WindowsRun> {
        if !self.runs.contains_key(key) {
            let config = resolver_config(spec);
            let schema = frame.schema();
            let run = match self.saved.remove(key) {
                None => WindowsRun::new(config, schema),
                Some(bytes) => WindowsRun::load_bytes(&bytes, config, schema),
            }
            .map_err(
                |e| polars_err!(ComputeError: "spec {:?}: formula target: {}", spec.name, without_who(&e)),
            )?;
            self.runs.insert(key.clone(), run);
        }
        Ok(self.runs.get_mut(key).expect("inserted above"))
    }

    /// Every group's core as bytes, by key: the running ones, and the
    /// loaded ones no chunk has reached yet.
    pub(crate) fn save(&self) -> Result<Vec<(GroupKey, Vec<u8>)>, String> {
        let mut out: Vec<(GroupKey, Vec<u8>)> = Vec::with_capacity(self.runs.len());
        for (key, run) in &self.runs {
            out.push((key.clone(), run.save_bytes()?));
        }
        for (key, bytes) in &self.saved {
            out.push((key.clone(), bytes.clone()));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// The longest `window_size` among the formulas' forward operators,
    /// which the embargo must cover for a prediction to be out of sample.
    pub(crate) fn longest_forward_window(spec: &Spec) -> Option<crate::span::Span> {
        spec.targets
            .defs()
            .iter()
            .filter_map(|t| t.formula.as_ref())
            .flat_map(|tree| {
                tree.operators()
                    .into_iter()
                    .filter(|op| op.direction() == crate::windows::Direction::Forward)
                    .filter_map(|op| op.window_size.clone())
                    .collect::<Vec<_>>()
            })
            .max_by(|a, b| a.value().total_cmp(&b.value()))
    }
}

/// The frame a spec's resolver is fed, with the cores of every group of
/// the chunk built or resumed: everything that can refuse before a row
/// goes in, so it is done for every spec before any spec is fed (review
/// R2, F1: a later spec's refusal left an earlier spec's core fed).
pub(crate) fn prepare_targets(
    spec: &Spec,
    resolver: &mut TargetWindows,
    chunk: &ArrowChunk,
    groups: &[(GroupKey, Vec<usize>)],
    first_row: u64,
) -> PolarsResult<DataFrame> {
    let n = chunk.height();
    let frame = resolver_frame(chunk, spec, &resolver_columns(spec), first_row)?;
    for (key, idx) in groups {
        if !resolver.runs.contains_key(key) {
            let sub = if idx.len() == n {
                frame.clone()
            } else {
                let take: IdxCa =
                    IdxCa::from_vec("".into(), idx.iter().map(|&i| i as IdxSize).collect());
                frame.take(&take)?
            };
            resolver.run(key, spec, &sub)?;
        }
    }
    Ok(frame)
}

/// Feed one chunk to a spec's resolver, group by group in the chunk's row
/// order, and gather what it resolved: per resolved row, by its number in
/// the bank's stream, the number of the row that resolved it and each
/// formula target's value; and each row's number in the laid-out order the
/// streams read.
pub(crate) fn resolve_targets(
    spec: &Spec,
    resolver: &mut TargetWindows,
    frame: &DataFrame,
    groups: &[(GroupKey, Vec<usize>)],
    layout: Layout<'_>,
    first_row: u64,
) -> PolarsResult<FormulaBundle> {
    let n = frame.height();
    let names: Vec<&str> = spec
        .targets
        .defs()
        .iter()
        .filter(|t| t.is_formula())
        .map(|t| t.name.as_str())
        .collect();
    let mut by_group: HashMap<GroupKey, Resolutions> = HashMap::with_capacity(groups.len());
    for (key, idx) in groups {
        let rows: Vec<u64> = idx.iter().map(|&i| first_row + i as u64).collect();
        let sub = if idx.len() == n {
            frame.clone()
        } else {
            let take: IdxCa =
                IdxCa::from_vec("".into(), idx.iter().map(|&i| i as IdxSize).collect());
            frame.take(&take)?
        };
        let run = resolver.run(key, spec, &sub)?;
        let out = run.feed_resolving(&sub, &rows).map_err(
            |e| polars_err!(ComputeError: "spec {:?}: formula target: {}", spec.name, without_who(&e.to_string())),
        )?;
        let h = out.height();
        if h == 0 {
            continue;
        }
        let width = names.len();
        let mut values: Vec<f64> = vec![f64::NAN; h * width];
        for (k, name) in names.iter().enumerate() {
            let s = out.column(name)?.cast(&DataType::Float64)?;
            for (r, v) in s.f64()?.iter().enumerate() {
                if let Some(v) = v {
                    values[r * width + k] = v;
                }
            }
        }
        let seqs: Vec<u64> = out.column("@po:seq")?.u64()?.iter().flatten().collect();
        let ats: Vec<u64> = out.column("@po:at")?.u64()?.iter().flatten().collect();
        debug_assert!(seqs.len() == h && ats.len() == h, "no null row number");
        debug_assert!(
            seqs.windows(2).all(|w| w[0] < w[1]),
            "a group resolves in row order"
        );
        by_group.insert(
            key.clone(),
            Resolutions {
                seqs,
                ats,
                values,
                width,
            },
        );
    }
    let seqs = gathered((0..n).map(|i| first_row + i as u64).collect(), layout);
    Ok(FormulaBundle {
        slots: spec.targets.formula_slots(),
        seqs,
        groups: by_group,
    })
}

/// A core's message without its own caller's name: a bank's refusal names
/// the bank's spec, not `with_windows` (task 159, P1).
fn without_who(message: &str) -> &str {
    message
        .strip_prefix("with_windows: ")
        .or_else(|| message.strip_prefix("with_windows "))
        .unwrap_or(message)
}
