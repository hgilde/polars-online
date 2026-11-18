//! A spec's formula targets' resolvers (docs/PLAN.md task 104): task 143's
//! window core over the targets' formulas, one core per group under the
//! spec's clock policy, fed every row of every chunk before the streams run,
//! and what each chunk resolved, handed to the streams as `FormulaBundle`.
//! The bank (bank.rs) owns one `TargetWindows` per spec, calls
//! `prepare_targets` for every spec before feeding any (review R2, F1), then
//! `resolve_targets`, and saves the cores as `SavedResolvers`.

use online_core::FixedState;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars::prelude::*;

use crate::arrow::{ArrowChunk, Form};
use crate::bank::{GroupKey, Layout, PAR_MIN_ROWS, gathered};
use crate::formula::Formula;
use crate::spec::Spec;
use crate::stream::{FormulaTargets, Resolutions};
use crate::windows_frame::{Given, WindowsConfig, WindowsRun};

/// One chunk's formula-target inputs for a spec's streams (docs/PLAN.md
/// task 104): which targets are formulas, each row's number in the bank's
/// stream in the laid-out row order, and what each group's core resolved.
pub(crate) struct FormulaBundle {
    slots: Vec<usize>,
    seqs: Vec<u64>,
    groups: HashMap<GroupKey, Resolutions, FixedState>,
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
    pub(crate) runs: HashMap<GroupKey, WindowsRun, FixedState>,
    /// A loaded bank's cores, each resumed at its group's first chunk.
    pub(crate) saved: HashMap<GroupKey, Vec<u8>, FixedState>,
    /// A core that has read no row: what every group's core of this spec
    /// shares its call and plan with, so that a copy of the cores (the
    /// bank's, before a chunk that may be refused) and their memory are each
    /// group's state alone.
    proto: Option<WindowsRun>,
}

/// From this many groups a chunk's groups resolve on the pool's threads
/// whatever its rows: a group's fixed cost -- its rows taken, its core's
/// bookkeeping, and where a formula's inputs could not be evaluated once for
/// the chunk, the plan of Polars' lazy engine a group then makes, about 85
/// microseconds -- is a dispatch's many times over.
const PAR_MIN_GROUPS: usize = 8;

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
                "spec {:?}: the formula target reads column {:?}, which is not in the input, \
                 whose columns are {:?}. Add the column to the input, or change the formula to \
                 read a column the input has.",
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
            if self.proto.is_none() {
                self.proto = WindowsRun::new(config.clone(), schema).ok();
            }
            let mut run = match self.saved.remove(key) {
                None => WindowsRun::new(config, schema),
                Some(bytes) => WindowsRun::load_bytes(&bytes, config, schema),
            }
            .map_err(|e| {
                // The window run's refusal, said behind the spec's where.
                let prefix = format!("spec {:?}: formula target", spec.name);
                polars_err!(ComputeError: "{prefix}: {}", without_who(&e))
            })?;
            if let Some(proto) = &self.proto {
                run.share_with(proto);
            }
            self.runs.insert(key.clone(), run);
        }
        Ok(self.runs.get_mut(key).expect("inserted above"))
    }

    /// The cores of `groups` as they are, for a chunk that may refuse after
    /// feeding them: a chunk changes no other core, so these are all a
    /// refusal has to put back (see [`Self::restore`]). A group with no core
    /// yet is left out.
    pub(crate) fn snapshot(
        &self,
        groups: &[(GroupKey, Vec<usize>)],
    ) -> Vec<(GroupKey, WindowsRun)> {
        groups
            .iter()
            .filter_map(|(key, _)| self.runs.get(key).map(|run| (key.clone(), run.clone())))
            .collect()
    }

    /// Put back the cores [`Self::snapshot`] took.
    pub(crate) fn restore(&mut self, cores: Vec<(GroupKey, WindowsRun)>) {
        for (key, run) in cores {
            self.runs.insert(key, run);
        }
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
            resolver.run(key, spec, &group_frame(&frame, idx, n)?)?;
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
    // A core a caller has not prepared is made here, serially, as
    // `prepare_targets` makes it. Then each group's core is taken out of the
    // map to be fed: they are independent, and a group resolves on whichever
    // thread takes it. A chunk's groups are distinct keys.
    for (key, idx) in groups {
        if !resolver.runs.contains_key(key) {
            resolver.run(key, spec, &group_frame(frame, idx, n)?)?;
        }
    }
    let mut cores: Vec<WindowsRun> = groups
        .iter()
        .map(|(key, _)| resolver.runs.remove(key).expect("made above"))
        .collect();
    // A chunk of one group, or of too few rows and groups to repay a
    // dispatch, runs on this thread; the cores and the rows each is fed are
    // the same either way, and the results are read in the groups' order, so
    // the first refusal is the one a serial pass reports.
    // The operator inputs a formula computes (a product, a when/then, a
    // cast) over the whole chunk in one plan of Polars' lazy engine, where a
    // plan a group cost tens of microseconds each; a group is handed its
    // rows' values. Every input is element-wise, so a value does not depend
    // on the rows beside it. So are an increment's input and the operator
    // inputs that read an increment, which each group's own previous row
    // gives: the increments are read off each group's state first
    // (`WindowsRun::share`), and the inputs evaluated over the chunk with
    // them. A plan that refuses is left for the groups' own, which report
    // it by group. A chunk of one group has nothing to share.
    let shared = (groups.len() > 1)
        .then(|| {
            let rows: Vec<&[usize]> = groups.iter().map(|(_, idx)| idx.as_slice()).collect();
            WindowsRun::share(&cores, frame, &rows)
        })
        .flatten();
    let chunk = Chunk {
        spec,
        frame,
        names: &names,
        first_row,
    };
    let one = |(key, idx): &(GroupKey, Vec<usize>), run: &mut WindowsRun| {
        let given = shared.as_ref().map(|s| s.of_group(idx));
        resolve_group(&chunk, run, key, idx, given)
    };
    let results: Vec<Resolved> =
        if groups.len() > 1 && (n >= PAR_MIN_ROWS || groups.len() >= PAR_MIN_GROUPS) {
            in_parallel(&mut cores, groups, one)
        } else {
            // Up to the first refusal, as a pass over the groups in order
            // stops at it.
            let mut done = Vec::with_capacity(cores.len());
            for (run, g) in cores.iter_mut().zip(groups) {
                let r = one(g, run);
                let refused = r.is_err();
                done.push(r);
                if refused {
                    break;
                }
            }
            done
        };
    for ((key, _), run) in groups.iter().zip(cores) {
        resolver.runs.insert(key.clone(), run);
    }
    let by_group = settle(spec, resolver, groups, results, &names)?;
    let seqs = gathered((0..n).map(|i| first_row + i as u64).collect(), layout);
    Ok(FormulaBundle {
        slots: spec.targets.formula_slots(),
        seqs,
        groups: by_group,
    })
}

/// What one group's step gives: its key and rows, `None` where nothing
/// resolved.
type Resolved = PolarsResult<Option<(GroupKey, Rows)>>;

/// One group's resolved rows, as its step leaves them.
enum Rows {
    /// With the formulas' values, as [`Resolutions`] holds them.
    Done(Resolutions),
    /// Before the formulas, which [`WindowsRun::resolved_formulas`] evaluates
    /// for every group of the chunk at once: the rows' held columns, the
    /// operators' hidden ones, `@po:row` and `@po:at`.
    Pending(DataFrame),
}

thread_local! {
    /// Whether this thread is inside [`in_parallel`]'s loop.
    static LOOPING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the thread as in [`in_parallel`]'s loop until dropped.
struct Looping {
    /// Whether it was in the loop already.
    was: bool,
}

impl Looping {
    fn enter() -> Self {
        Self {
            was: LOOPING.with(|l| l.replace(true)),
        }
    }
}

impl Drop for Looping {
    fn drop(&mut self) {
        LOOPING.with(|l| l.set(self.was));
    }
}

/// `one` on every group's core, on the pool's threads, the results in the
/// groups' order.
///
/// A group's step can block its thread: Polars' lazy engine hands its work
/// to Polars' own pool and waits for it. (A step does so only where the
/// chunk's formulas could not be evaluated once for all its groups -- a plan
/// that refuses, a run that skips rows it read before it was saved -- since the group's rows are taken a
/// column at a time on its own thread and the formulas are evaluated
/// outside this loop.) A rayon thread that
/// waits does not idle, it runs other jobs of its own pool on top of its
/// stack, so a parallel iterator over the groups nests: the thread blocked
/// in one group's step starts another group's, which blocks and starts
/// another, to a depth that grows with the number of groups. A debug build's
/// 2 MiB thread stack held about 400 groups on two threads, and the
/// process aborted past that (800 and 2,000 measured); a release build's held
/// a few thousand.
///
/// So the groups are not split into jobs. The calling thread and up to one
/// job for each other thread run a loop that takes the next group from a
/// shared counter, and a job that finds its thread in the loop already --
/// one it picked up while blocked in a step -- leaves at once: the loop that
/// thread is in, resumed when its step returns, takes the groups left. Each
/// thread runs one group's step at a time, so its stack holds one step
/// however many groups there are, and the threads stay as busy as they were,
/// each taking a group as it finishes the last. The caller's loop runs
/// whether or not its thread was in one, which is what makes every group
/// taken.
fn in_parallel<F>(
    cores: &mut Vec<WindowsRun>,
    groups: &[(GroupKey, Vec<usize>)],
    one: F,
) -> Vec<Resolved>
where
    F: Fn(&(GroupKey, Vec<usize>), &mut WindowsRun) -> Resolved + Sync,
{
    // Each core is locked once, by the thread that takes its group, so the
    // lock is never waited for.
    let steps: Vec<Mutex<(WindowsRun, Option<Resolved>)>> = std::mem::take(cores)
        .into_iter()
        .map(|run| Mutex::new((run, None)))
        .collect();
    let next = AtomicUsize::new(0);
    let drain = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(step) = steps.get(i) else {
                return;
            };
            let mut step = step
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (run, out) = &mut *step;
            *out = Some(one(&groups[i], run));
        }
    };
    rayon::scope(|s| {
        for _ in 1..rayon::current_num_threads().min(steps.len()) {
            s.spawn(|_| {
                let looping = Looping::enter();
                if !looping.was {
                    drain();
                }
            });
        }
        let _looping = Looping::enter();
        drain();
    });
    let mut results = Vec::with_capacity(steps.len());
    for step in steps {
        let (run, out) = step
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cores.push(run);
        results.push(out.expect("the caller's loop takes every group before the scope ends"));
    }
    results
}

/// The rows of the chunk's frame that are one group's: the frame itself
/// where the group is all of it. Each column is taken on this thread:
/// `DataFrame::take` hands its columns to Polars' own pool and waits, which
/// is what [`in_parallel`] must not nest.
fn group_frame(frame: &DataFrame, idx: &[usize], n: usize) -> PolarsResult<DataFrame> {
    if idx.len() == n {
        return Ok(frame.clone());
    }
    let take: IdxCa = IdxCa::from_vec("".into(), idx.iter().map(|&i| i as IdxSize).collect());
    let cols = frame
        .columns()
        .iter()
        .map(|c| c.take(&take))
        .collect::<PolarsResult<Vec<_>>>()?;
    DataFrame::new(idx.len(), cols)
}

/// A resolved frame's columns, read once: each formula's values as doubles,
/// the rows' numbers and the numbers of the rows that resolved them.
struct ResolvedColumns {
    values: Vec<Float64Chunked>,
    seqs: Vec<u64>,
    ats: Vec<u64>,
}

impl ResolvedColumns {
    fn of(out: &DataFrame, names: &[&str]) -> PolarsResult<Self> {
        let values = names
            .iter()
            .map(|name| Ok(out.column(name)?.cast(&DataType::Float64)?.f64()?.clone()))
            .collect::<PolarsResult<_>>()?;
        let seqs = out.column("@po:seq")?.u64()?.iter().flatten().collect();
        let ats = out.column("@po:at")?.u64()?.iter().flatten().collect();
        Ok(Self { values, seqs, ats })
    }

    /// The `h` rows from `lo`.
    fn resolutions(&self, lo: usize, h: usize) -> Resolutions {
        let width = self.values.len();
        let mut values: Vec<f64> = vec![f64::NAN; h * width];
        for (k, ca) in self.values.iter().enumerate() {
            for (r, v) in ca.slice(lo as i64, h).iter().enumerate() {
                if let Some(v) = v {
                    values[r * width + k] = v;
                }
            }
        }
        let seqs = self.seqs[lo..lo + h].to_vec();
        let ats = self.ats[lo..lo + h].to_vec();
        debug_assert!(
            seqs.windows(2).all(|w| w[0] < w[1]),
            "a group resolves in row order"
        );
        Resolutions {
            seqs,
            ats,
            values,
            width,
        }
    }
}

/// What every group of a chunk is fed from.
#[derive(Clone, Copy)]
struct Chunk<'a> {
    spec: &'a Spec,
    frame: &'a DataFrame,
    /// The formula targets' names.
    names: &'a [&'a str],
    first_row: u64,
}

/// One group's rows fed to its core: what the core resolved, with the
/// formulas' values by resolved row and the rows' numbers where the core
/// evaluates the formulas itself, and the rows before them where the chunk's
/// groups' are evaluated together; `None` where nothing resolved. `given`
/// is what the chunk's inputs evaluated beforehand gave at this group's
/// rows.
fn resolve_group(
    chunk: &Chunk<'_>,
    run: &mut WindowsRun,
    key: &GroupKey,
    idx: &[usize],
    given: Option<Given>,
) -> Resolved {
    let Chunk {
        spec,
        frame,
        names,
        first_row,
    } = *chunk;
    let rows: Vec<u64> = idx.iter().map(|&i| first_row + i as u64).collect();
    let sub = group_frame(frame, idx, frame.height())?;
    let pending = run.formulas_pending();
    let out = run.feed_resolving_with(&sub, &rows, given).map_err(|e| {
        let prefix = format!("spec {:?}: formula target", spec.name);
        polars_err!(ComputeError: "{prefix}: {}", without_who(&e.to_string()))
    })?;
    let h = out.height();
    if h == 0 {
        return Ok(None);
    }
    if pending {
        return Ok(Some((key.clone(), Rows::Pending(out))));
    }
    let resolutions = ResolvedColumns::of(&out, names)?.resolutions(0, h);
    Ok(Some((key.clone(), Rows::Done(resolutions))))
}

/// The formulas over every group's pending rows in one plan, and every
/// group's resolutions, the first refusal in the groups' order being the one
/// reported: a group's own step's, or its rows' formulas', each as if the
/// groups were taken one by one.
///
/// The formulas are element-wise, so a row's value does not depend on the
/// rows stacked beside it. A formula that refuses the stacked rows (a strict
/// cast of a value out of range, whose message counts the failing values) is
/// evaluated group by group again, so that the refusal is the first group's
/// own.
fn settle(
    spec: &Spec,
    resolver: &TargetWindows,
    groups: &[(GroupKey, Vec<usize>)],
    results: Vec<Resolved>,
    names: &[&str],
) -> PolarsResult<HashMap<GroupKey, Resolutions, FixedState>> {
    // A group's own refusal comes after the formulas' refusals of the groups
    // before it, and before those of the groups after it.
    let mut taken: Vec<Option<(GroupKey, Rows)>> = Vec::with_capacity(results.len());
    let mut refusal: Option<PolarsError> = None;
    for r in results {
        match r {
            Ok(r) => taken.push(r),
            Err(e) => {
                refusal = Some(e);
                break;
            }
        }
    }
    let pending: Vec<usize> = (0..taken.len())
        .filter(|&i| matches!(taken[i], Some((_, Rows::Pending(_)))))
        .collect();
    let mut resolved: HashMap<usize, Resolutions> = HashMap::new();
    if !pending.is_empty() {
        let run = &resolver.runs[&groups[0].0];
        let frames = |i: usize| match &taken[i] {
            Some((_, Rows::Pending(f))) => f,
            _ => unreachable!("pending above"),
        };
        let stacked = (|| {
            let mut all = frames(pending[0]).clone();
            for &i in &pending[1..] {
                all.vstack_mut(frames(i))?;
            }
            all.rechunk_mut();
            run.resolved_formulas(all)
        })();
        match stacked {
            Ok(out) => {
                let cols = ResolvedColumns::of(&out, names)?;
                let mut lo = 0;
                for &i in &pending {
                    let h = frames(i).height();
                    resolved.insert(i, cols.resolutions(lo, h));
                    lo += h;
                }
            }
            Err(_) => {
                for &i in &pending {
                    let out = run.resolved_formulas(frames(i).clone()).map_err(|e| {
                        let prefix = format!("spec {:?}: formula target", spec.name);
                        polars_err!(ComputeError: "{prefix}: {}", without_who(&e.to_string()))
                    })?;
                    let h = out.height();
                    resolved.insert(i, ResolvedColumns::of(&out, names)?.resolutions(0, h));
                }
            }
        }
    }
    if let Some(e) = refusal {
        return Err(e);
    }
    let mut by_group: HashMap<GroupKey, Resolutions, FixedState> =
        HashMap::with_capacity_and_hasher(groups.len(), FixedState::default());
    for (i, r) in taken.into_iter().enumerate() {
        match r {
            None => {}
            Some((key, Rows::Done(res))) => {
                by_group.insert(key, res);
            }
            Some((key, Rows::Pending(_))) => {
                by_group.insert(
                    key,
                    resolved.remove(&i).expect("every pending group settled"),
                );
            }
        }
    }
    Ok(by_group)
}

/// A core's message without its own caller's name: a bank's refusal names
/// the bank's spec, not `with_windows` (task 159, P1).
fn without_who(message: &str) -> &str {
    message
        .strip_prefix("with_windows: ")
        .or_else(|| message.strip_prefix("with_windows "))
        .unwrap_or(message)
}
