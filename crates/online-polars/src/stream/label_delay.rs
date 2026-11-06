//! The rows an `embargo` or a formula target holds, and how a chunk's plan
//! list holds and releases them (moved out of `stream.rs` under the 250 KB
//! cap a source file is held to).

#[allow(unused_imports)]
use super::*;

/// One accepted row waiting for its label to mature (`embargo`,
/// docs/ENHANCEMENTS.md E47).
///
/// It carries everything the models need to be stepped with later: the row
/// itself, the clock delta it arrived with -- so replaying the buffer in
/// order gives the models exactly the gap sequence they would have seen
/// without the delay -- its place on the elapsed clock, which its release
/// is measured from, and the break it arrived after, whose events run when
/// it is learned (docs/PLAN.md task 153).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingRow {
    /// The row's place on the elapsed clock, held exactly
    /// ([`online_core::ClockAdvance::elapsed_stamp`]): it is learned from
    /// once an accepted row's place is at least `embargo` past it. The
    /// elapsed clock counts the clock column's own steps rather than the
    /// capped delta, so a break never releases a row before its delay has
    /// passed (task 153), and two places compare exactly -- in integer
    /// nanoseconds on a temporal clock, by one subtraction of raw values on
    /// a number clock -- where an embargo counted down in doubles released
    /// rows a row late or early (task 176). `None` only in a damaged state,
    /// which [`Stream::restore`] refuses.
    #[serde(default)]
    pub arrived: Option<Stamp>,
    /// The delta the row arrived with, replayed when it is released.
    pub d_clock: f64,
    /// The step the row reads as a fresh stream's first accepted row,
    /// that stream begun right after the accepted row before it
    /// (`RowPlan::d_fresh`): what an instance a drift reset restarted
    /// learns the row with when it is the first held row after the one
    /// that fired. Nonzero only for a row after skipped rows under
    /// `drift_action = "reset"`, and written only then, so no other
    /// state's bytes move. Schema 64.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub d_fresh: f64,
    /// The row's stamp, its decayed clock held exactly, which a window keys
    /// the row's snapshot by when it is released (docs/PLAN.md task 175).
    #[serde(default)]
    pub stamp: Option<Stamp>,
    pub w: f64,
    pub xs: Vec<f64>,
    /// `None` for a target that was null on the row.
    pub ys: Vec<Option<f64>>,
    /// The break before this row -- a session change (with the long-run
    /// blend `session_shrink` asks for) or a gap past `gap_cap`, on this
    /// row or on rows skipped since the last accepted one -- applied to the
    /// models when the row is learned, after every row before the break.
    /// Applied when the row arrived, it forced every held row out first.
    #[serde(default)]
    pub session_changed: bool,
    #[serde(default)]
    pub blend: bool,
    #[serde(default)]
    pub capped: bool,
    /// The row's clock as the fields show it (task 152).
    #[serde(default)]
    pub clock: Option<ClockValue>,
    /// The row's number in the stream the spec's window core was fed
    /// (docs/PLAN.md task 104): what a resolution names. 0 for a spec
    /// without a formula target, where nothing reads it.
    #[serde(default)]
    pub seq: u64,
    /// Whether every formula target of the row is known: true from the
    /// start for a spec without one, and once the window core has resolved
    /// the row otherwise. A row is released only when it is.
    #[serde(default = "yes")]
    pub resolved: bool,
    /// Under `label_clock`, the row's label time: the buffer is ordered by
    /// it, ties in arrival order, and the row is released at the first
    /// accepted row whose clock is at or past it, to be learned at it
    /// (task 341, schema 67). `None` under `embargo`, whose release is
    /// decided on the elapsed clock by the delay, as before, and not
    /// written, so an `embargo` state's bytes do not move: a stored
    /// `clock + delay` would be compared as a sum where the delay is
    /// compared as a difference, and on a number clock the two round
    /// apart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<ClockValue>,
}

/// What `label_clock` hands the release (task 341): each row's label time,
/// the clock the models have learned the released rows at, the bound on
/// the rows held, and the counts of labels that never arrive or are
/// dropped.
pub(super) struct LabelRun<'a> {
    times: &'a LabelTimes,
    cfg: online_core::ClockCfg,
    caps: ExactCaps,
    learned: &'a mut ClockState,
    max_held: Option<u64>,
    drop_oldest: bool,
    missing: &'a mut u64,
    dropped: &'a mut u64,
    /// For each row held, by its `ri`: where it went into the buffer, and
    /// the row a full buffer dropped after it ([`u32::MAX`] for none),
    /// which `run_instance` keeps each instance's records in step with.
    places: &'a mut Vec<(u32, u32)>,
}

impl<'a> LabelRun<'a> {
    /// The run for `spec`, `None` for a spec without `label_clock` or a
    /// chunk that carries no label column (which the bank never hands one).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn of(
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        caps: ExactCaps,
        times: Option<&'a LabelTimes>,
        learned: &'a mut Option<ClockState>,
        missing: &'a mut u64,
        dropped: &'a mut u64,
        places: &'a mut Vec<(u32, u32)>,
    ) -> Option<Self> {
        spec.label_clock.as_ref()?;
        Some(LabelRun {
            times: times?,
            cfg: *cfg,
            caps,
            learned: learned.get_or_insert_with(ClockState::new),
            max_held: spec.max_held_rows,
            drop_oldest: spec.drops_oldest_held(),
            missing,
            dropped,
            places,
        })
    }
}

/// A place in the buffer as a plan carries it: four bytes, which a group's
/// held rows never outgrow -- each is a row of features held in memory.
fn held_index(at: usize) -> u32 {
    u32::try_from(at).expect("fewer than 2^32 rows held in one group")
}

/// The held row that arrived first: the earliest clock, the first in
/// buffer order among rows that arrived at one clock.
fn oldest(pending: &VecDeque<PendingRow>) -> usize {
    let mut at = 0;
    for (i, r) in pending.iter().enumerate().skip(1) {
        if let (Some(c), Some(o)) = (r.clock, pending[at].clock)
            && c.is_before(o)
        {
            at = i;
        }
    }
    at
}

impl LabelRefusal {
    /// The refusal as the bank says it, naming the spec, the column and the
    /// row of the frame.
    pub fn message(&self, spec: &Spec, row_base: usize) -> String {
        let label = spec.label_clock.as_ref().map_or("", |l| l.column());
        match *self {
            LabelRefusal::Before {
                row,
                label: l,
                clock,
            } => {
                let by = match (l, clock) {
                    (ClockValue::Ns(a), ClockValue::Ns(b)) => {
                        crate::span::format_duration(b.saturating_sub(a))
                    }
                    (ClockValue::I64(a), ClockValue::I64(b)) => {
                        (i128::from(b) - i128::from(a)).to_string()
                    }
                    _ => format!("{:.3}", clock.seconds() - l.seconds()),
                };
                format!(
                    "spec {:?}: label_clock column {label:?} is {by} before clock column {:?} at \
                     row {}, and a label is never known before its row; the bank was not \
                     updated. Give each row's label time, at or after its clock.",
                    spec.name,
                    spec.clock.as_deref().unwrap_or(""),
                    row_base + row
                )
            }
            LabelRefusal::Full { row, held } => format!(
                "spec {:?}: row {} would put {} rows of its group waiting for their label times, \
                 past max_held_rows = {held}; the bank was not updated. Raise max_held_rows, or \
                 set held_overflow=\"drop_oldest\" to drop the row that arrived first.",
                spec.name,
                row_base + row,
                held + 1
            ),
        }
    }
}

/// Why a chunk is refused under `label_clock` before any stream is touched
/// ([`Stream::check_labels`]): the row, by its place in the chunk, and
/// what is wrong with it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelRefusal {
    /// The row's label time is before its own clock.
    Before {
        row: usize,
        label: ClockValue,
        clock: ClockValue,
    },
    /// Holding the row puts the group past `max_held_rows`.
    Full { row: usize, held: u64 },
}

fn yes() -> bool {
    true
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

impl Stream {
    /// Rewrite a chunk's plan list so each accepted row is *scored* where it
    /// sits and *learned from* only once `embargo` has passed
    /// (docs/ENHANCEMENTS.md E47). A no-op, and one `Option` test, for a
    /// spec without a delay.
    ///
    /// The rule is one line: at each accepted row, every buffered row whose
    /// wait has run out is released -- in arrival order, each replaying the
    /// clock delta it arrived with, so the models see exactly the sequence of
    /// gaps they would have seen without the delay -- and then the row itself
    /// is scored and buffered. The wait counts the time that passed, the
    /// clock column's own steps ([`online_core::ClockAdvance::elapsed`]), not
    /// the capped delta: `gap_cap` and `session_gap` say how much a model
    /// forgets across a break, not how long it lasted. Release therefore
    /// depends on the rows alone, which is what makes it chunk-invariant.
    ///
    /// A row's wait has run out when the accepted row's place on the
    /// elapsed clock is at least the embargo past the row's own
    /// ([`online_core::ClockAdvance::elapsed_stamp`], docs/PLAN.md task
    /// 176): two integers of nanoseconds on a temporal clock, one
    /// subtraction of raw values on a number clock, two row counts without a
    /// clock. Each held row counted its embargo down in doubles until then,
    /// and the rounded steps drifted: on rows 1 ms apart under `"2s"` every
    /// row was learned a row late.
    ///
    /// A break releases nothing early (docs/PLAN.md task 153). Its events --
    /// the lag rings' clear at a gap past `gap_cap` or a session change,
    /// and `session_shrink`'s blend -- wait in the buffer with the row that
    /// raised them, or with the next accepted row when a skipped row raised
    /// them, and run when that row is learned: after every row before the
    /// break, before every row after it, so no ring pairs rows across it
    /// (docs/REVIEW-E54-E64.md L2). Applied when the row arrived, they had
    /// forced every held row out first, and a label shorter-lived than the
    /// delay was learned before it was known. The models run one delay
    /// behind, events included. Only a clock **reset** acts on arrival: it
    /// drops the buffer, since the state those rows would teach is gone. A
    /// drift a released row trips is flagged on the row releasing it
    /// ([`RowPlan::drift_ri`]), and a drift reset restarts the models there,
    /// keeping the buffer: the rows it holds arrived after the one that
    /// tripped it.
    ///
    /// A formula target (docs/PLAN.md task 104) is not known at its row:
    /// the row waits, with no embargo as well, until the spec's window core
    /// has resolved it -- at the row that closed, cut or discarded its
    /// windows, which `formulas` carries by sequence number -- and its wait
    /// has run out, whichever is later. Resolutions are applied in row
    /// order, at the row that made them, so a coarse chunking releases
    /// nothing a fine one holds (the review under task 104: with an embargo
    /// equal to the window, a row exactly one window later has the wait run
    /// out while the window is still open).
    ///
    /// Returns the released rows' values, indexed by `RowPlan::pending`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_label_delay(
        embargo: Option<Embargo>,
        pending: &mut VecDeque<PendingRow>,
        held_break: &mut HeldBreak,
        plans: &mut Vec<RowPlan>,
        // Each plan's place on the elapsed clock, by its `ri`: every plan
        // handed here is one [`Stream::schedule`] made, one per row, so it
        // is indexed rather than looked up -- a plan without its stamp
        // fails loudly instead of reading as no place at all.
        elapsed: &[Option<Stamp>],
        features: &FeatureRows,
        targets: &[Vec<f64>],
        formulas: Option<FormulaTargets<'_>>,
        // Where to keep each scored row's `Stream::front_cut`, by its `ri`,
        // for a stream whose drift resets restart instances it may score
        // for; `None` elsewhere.
        mut front_cuts: Option<&mut Vec<Option<f64>>>,
        // Under `label_clock`, each row's label time and what the release
        // keeps with it (task 341); `None` elsewhere.
        mut labels: Option<&mut LabelRun<'_>>,
    ) -> Vec<PendingRow> {
        if embargo.is_none() && formulas.is_none() && labels.is_none() {
            return Vec::new();
        }
        if let Some(cuts) = front_cuts.as_deref_mut() {
            cuts.clear();
            cuts.resize(plans.len(), None);
        }
        if let Some(lr) = labels.as_deref_mut() {
            lr.places.clear();
            lr.places.resize(plans.len(), (u32::MAX, u32::MAX));
        }
        let mut released: Vec<PendingRow> = Vec::new();
        let mut out: Vec<RowPlan> = Vec::with_capacity(plans.len());
        // The resolved rows are a prefix of the buffer (below): its length,
        // found by bisection once a chunk and kept as rows resolve, leave
        // and join, where a pass over it and a `Vec` shifted at each release
        // cost O(rows held) a row (task 270, M-3).
        let mut resolved = pending.partition_point(|r| r.resolved);
        // One template for every replayed row: only `pending` and the
        // break's events differ, and a replay never emits, never wants
        // coefficients and never counts as a row of the frame. A drift it
        // detects is flagged on `at`, the row releasing it (task 160, PB1).
        let replay = |slot: usize, row: &PendingRow, at: usize| RowPlan {
            ri: usize::MAX,
            i: usize::MAX,
            pending: slot,
            d_clock: row.d_clock,
            d_fresh: row.d_fresh,
            elapsed_step: row.d_clock,
            stamp: row.stamp,
            clock: row.clock,
            reset: false,
            blend: row.blend,
            session_changed: row.session_changed,
            backwards: false,
            capped: row.capped,
            accept: true,
            want_coef: false,
            emit: false,
            drift_ri: at,
            learn: true,
            buffered: false,
            parks: false,
            w: row.w,
        };
        // The events of rows skipped since the last accepted one fall before
        // the next accepted row, so they wait with it, across a chunk
        // boundary too (`held_break` is the stream's).
        for plan in plans.drain(..) {
            if plan.reset {
                // The models restart at this row -- and the reset is applied
                // here, not a delay later, because the point of it is that
                // the state is no longer about this stream. The rows waiting
                // to teach that state go with it.
                // Under `label_clock` they are counted, and the clock of the
                // labels learned starts over with the models.
                if let Some(lr) = labels.as_deref_mut() {
                    *lr.dropped = lr.dropped.saturating_add(pending.len() as u64);
                    *lr.learned = ClockState::new();
                }
                pending.clear();
                resolved = 0;
                *held_break = HeldBreak::default();
            }
            // The break's events run when its row is learned, not here.
            let quiet = RowPlan {
                session_changed: false,
                blend: false,
                capped: false,
                ..plan
            };
            // The resolutions made at or before this row, in order, at
            // every row of the group -- a skipped row resolves windows as
            // any row does, and the next accepted row may be chunks away.
            // The resolved rows are a prefix of the buffer (windows close in
            // clock order, a cut or a discard takes every open one), so the
            // first unresolved row still waiting ends the pass. A resolution
            // is made by a row of this group in this chunk, so it is never
            // left for a later one.
            if !plan.reset
                && let Some(res) = formulas.and_then(|f| f.resolved)
            {
                let f = formulas.expect("checked");
                let now = f.seqs[plan.i];
                for row in pending.range_mut(resolved..) {
                    match res.find(row.seq) {
                        Some((at, ys)) if at <= now => {
                            for (&slot, &y) in f.slots.iter().zip(ys) {
                                row.ys[slot] = Some(y).filter(|v| usable(*v));
                            }
                            row.resolved = true;
                            resolved += 1;
                        }
                        _ => break,
                    }
                }
            }
            if !plan.accept {
                // A skipped row teaches nothing and waits for nothing; its
                // time passes on the elapsed clock, which the next accepted
                // row's place carries, and its events wait with that row.
                held_break.session_changed |= plan.session_changed;
                held_break.blend |= plan.blend;
                held_break.capped |= plan.capped;
                out.push(quiet);
                continue;
            }
            if let Some(lr) = labels.as_deref_mut() {
                Self::hold_for_label(
                    lr,
                    pending,
                    held_break,
                    (&plan, quiet),
                    elapsed,
                    features,
                    targets,
                    (&mut released, &mut out),
                );
                continue;
            }
            if !plan.reset {
                // With no embargo a formula row waits for its window alone.
                let now = elapsed[plan.ri];
                while let Some(row) = pending.pop_front_if(|r| {
                    r.resolved && embargo.is_none_or(|e| e.passed(now, r.arrived))
                }) {
                    resolved = resolved.saturating_sub(1);
                    released.push(row);
                    out.push(replay(
                        released.len() - 1,
                        released.last().unwrap(),
                        plan.ri,
                    ));
                }
            }
            // The row itself: scored from the state as it now stands, and
            // buffered with the delta it arrived with, its place on the
            // elapsed clock and the break before it.
            let i = plan.i;
            resolved += usize::from(formulas.is_none());
            // What a restarted instance's fresh bank does not count of the
            // rows held ahead of this one (`Stream::front_cut`).
            if let Some(cuts) = front_cuts.as_deref_mut() {
                cuts[plan.ri] = pending.front().map(|r| r.d_clock - r.d_fresh);
            }
            pending.push_back(PendingRow {
                arrived: elapsed[plan.ri],
                d_clock: plan.d_clock,
                d_fresh: plan.d_fresh,
                stamp: plan.stamp,
                w: plan.w,
                xs: features.row(i).to_vec(),
                ys: targets
                    .iter()
                    .map(|t| Some(t[i]).filter(|v| usable(*v)))
                    .collect(),
                session_changed: held_break.session_changed || plan.session_changed,
                blend: held_break.blend || plan.blend,
                capped: held_break.capped || plan.capped,
                clock: plan.clock,
                seq: formulas.map_or(0, |f| f.seqs[i]),
                resolved: formulas.is_none(),
                label: None,
            });
            *held_break = HeldBreak::default();
            out.push(RowPlan {
                learn: false,
                buffered: true,
                ..quiet
            });
        }
        *plans = out;
        released
    }
}

impl Stream {
    /// The part of the clock of the rows held under `embargo` that a fresh
    /// bank fed the rows after a drift reset's row never sees: the first
    /// held row's `d_clock`, its step from the row that fired, less its
    /// `d_fresh`, the step it reads as that fresh bank's first accepted
    /// row. What an instance restarted since that row arrived takes out of
    /// the held clock for a row it scores after every row held now; `None`
    /// where no row is held, so that the row scored is the first such an
    /// instance will learn.
    pub(super) fn front_cut(&self) -> Option<f64> {
        self.persisted
            .pending
            .front()
            .map(|r| r.d_clock - r.d_fresh)
    }

    /// The clock the rows held under `embargo` have covered, as instance
    /// `i`'s next row reads it: the held rows' summed steps, or, for an
    /// instance restarted since the first of them arrived, those a fresh
    /// bank fed the rows after the restart reads -- the first held row's
    /// step from the row that fired taken out ([`Self::front_cut`]),
    /// none while no row is held.
    pub(super) fn held_clock(&self, i: usize) -> f64 {
        let held = self.persisted.pending_clock.get(i).copied().unwrap_or(0.0);
        if self.persisted.pending_clock.is_empty()
            || !self.persisted.restarted.get(i).copied().unwrap_or(false)
        {
            return held;
        }
        self.front_cut().map_or(0.0, |cut| held - cut)
    }

    /// Whether pass 1 keeps each row's place on the elapsed clock: where a
    /// buffer reads it -- an `embargo`'s release, a formula target's, and
    /// every held row's `arrived`, which a saved state carries.
    pub(super) fn keeps_elapsed(&self, formulas: bool) -> bool {
        self.embargo.is_some() || formulas || self.persisted.label_learned.is_some()
    }

    /// One accepted row of [`Self::apply_label_delay`] under `label_clock`
    /// (task 341): the held rows whose label time the row's clock has
    /// reached are released, in order of label time and then arrival, each
    /// learned at its label time -- its step the one from the label time
    /// before it, on the clock of labels learned, capped as any step is --
    /// and then the row is scored and held, in label-time order, unless it
    /// has nothing to teach: a weight of 0, or a null label time, which is
    /// counted. A full buffer drops the row that arrived first, under
    /// `held_overflow = "drop_oldest"`; under `"refuse"` the bank refused
    /// the chunk before any stream ran ([`Self::check_labels`]).
    ///
    /// The held rows' clock (`Instance::pending_clock`) counts each scored
    /// row's own step, held or not, and gives back each released row's
    /// step, so that `settled_frac` reads the clock since the stream's
    /// first row as it does under `embargo`, and `embargo = d` and a label
    /// time of `clock + d` give the same numbers wherever the clock adds
    /// up exactly.
    #[allow(clippy::too_many_arguments)]
    fn hold_for_label(
        lr: &mut LabelRun<'_>,
        pending: &mut VecDeque<PendingRow>,
        held_break: &mut HeldBreak,
        (plan, quiet): (&RowPlan, RowPlan),
        elapsed: &[Option<Stamp>],
        features: &FeatureRows,
        targets: &[Vec<f64>],
        (released, out): (&mut Vec<PendingRow>, &mut Vec<RowPlan>),
    ) {
        let now = plan.clock.expect("a label_clock spec reads a clock column");
        if !plan.reset {
            while pending
                .front()
                .and_then(|r| r.label)
                .is_some_and(|l| !now.is_before(l))
            {
                let row = pending.pop_front().expect("checked");
                let adv = lr
                    .learned
                    .advance_stamped(&lr.cfg, &lr.caps, row.label, None, true);
                released.push(row);
                let row = released.last().expect("pushed");
                out.push(RowPlan {
                    ri: usize::MAX,
                    i: usize::MAX,
                    pending: released.len() - 1,
                    d_clock: adv.d_clock,
                    d_fresh: 0.0,
                    elapsed_step: adv.d_clock,
                    stamp: adv.stamp,
                    clock: row.clock,
                    reset: false,
                    blend: false,
                    session_changed: false,
                    backwards: false,
                    capped: adv.capped,
                    accept: true,
                    want_coef: false,
                    emit: false,
                    drift_ri: plan.ri,
                    learn: true,
                    buffered: false,
                    parks: false,
                    w: row.w,
                });
            }
        }
        *held_break = HeldBreak::default();
        let i = plan.i;
        let label = lr.times.at(i);
        if label.is_none() {
            *lr.missing = lr.missing.saturating_add(1);
        }
        let Some(label) = label.filter(|_| plan.w > 0.0) else {
            out.push(RowPlan {
                learn: false,
                parks: true,
                ..quiet
            });
            return;
        };
        debug_assert!(!label.is_before(now), "refused by check_labels");
        let at = pending.partition_point(|r| r.label.is_some_and(|l| !label.is_before(l)));
        pending.insert(
            at,
            PendingRow {
                arrived: elapsed[plan.ri],
                d_clock: plan.d_clock,
                d_fresh: 0.0,
                stamp: plan.stamp,
                w: plan.w,
                xs: features.row(i).to_vec(),
                ys: targets
                    .iter()
                    .map(|t| Some(t[i]).filter(|v| usable(*v)))
                    .collect(),
                session_changed: false,
                blend: false,
                capped: false,
                clock: plan.clock,
                seq: 0,
                resolved: true,
                label: Some(label),
            },
        );
        let mut drop_at = u32::MAX;
        if lr.max_held.is_some_and(|m| pending.len() as u64 > m) {
            debug_assert!(lr.drop_oldest, "refused by check_labels");
            let oldest = oldest(pending);
            pending.remove(oldest);
            drop_at = held_index(oldest);
            *lr.dropped = lr.dropped.saturating_add(1);
        }
        lr.places[plan.ri] = (held_index(at), drop_at);
        out.push(RowPlan {
            learn: false,
            buffered: true,
            ..quiet
        });
    }

    /// Under `label_clock`, whether the chunk's rows of this stream can be
    /// held (task 341): a label time before its row's clock is refused, as a
    /// column that is not the label times, and so is a row that would put
    /// the group past `max_held_rows` under `held_overflow = "refuse"`.
    /// Run over every stream before any learns, on pass 1's schedule, so a
    /// refusal leaves the bank as it was. A no-op for a spec without
    /// `label_clock`.
    #[allow(clippy::too_many_arguments)]
    pub fn check_labels(
        &self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &crate::rows::FeatureCols,
        clock: Option<&ClockCol>,
        label: Option<&LabelTimes>,
        session: Option<&[u64]>,
        weight: Option<&[f64]>,
        rows: &[usize],
        base: usize,
    ) -> Result<(), LabelRefusal> {
        let Some(times) = label.filter(|_| spec.label_clock.is_some()) else {
            return Ok(());
        };
        let features = &features.rows(base, rows.len());
        // A clock the policy refuses is `check_clock`'s to name.
        let Ok(Scheduled { plans, .. }) = self.schedule(
            spec,
            cfg,
            features,
            clock,
            session,
            weight,
            rows,
            base,
            None,
            false,
            &mut self.persisted.restart_fold.clone(),
        ) else {
            return Ok(());
        };
        let bound = spec.max_held_rows.filter(|_| !spec.drops_oldest_held());
        let mut held: VecDeque<ClockValue> = self
            .persisted
            .pending
            .iter()
            .filter_map(|r| r.label)
            .collect();
        for plan in &plans {
            if plan.reset {
                held.clear();
            }
            if !plan.accept {
                continue;
            }
            let Some(now) = plan.clock else { continue };
            while held.front().is_some_and(|&l| !now.is_before(l)) {
                held.pop_front();
            }
            let Some(l) = times.at(plan.i) else { continue };
            if l.is_before(now) {
                return Err(LabelRefusal::Before {
                    row: rows[plan.ri],
                    label: l,
                    clock: now,
                });
            }
            if plan.w <= 0.0 {
                continue;
            }
            let Some(max) = bound else { continue };
            let at = held.partition_point(|&h| !l.is_before(h));
            held.insert(at, l);
            if held.len() as u64 > max {
                return Err(LabelRefusal::Full {
                    row: rows[plan.ri],
                    held: max,
                });
            }
        }
        Ok(())
    }
}

/// A saved stream's held rows against `spec` on load (task 341): rows held
/// for their label times are released by them, and rows held under
/// `embargo` by the delay, so each spec reads its own kind; a stream under
/// `label_clock` keeps the clock of the labels it has learned.
pub(super) fn held_kind(spec: &Spec, p: &mut Persisted) -> Result<(), String> {
    let by_label = spec.label_clock.is_some();
    if p.pending.iter().any(|r| r.label.is_some() != by_label)
        || (p.label_learned.is_some() && !by_label)
    {
        return Err(if by_label {
            "saved state holds rows for a delay, and this spec releases rows by \
             label_clock; resume it under the spec it was saved with"
        } else {
            "saved state holds rows for their label times, and this spec has no \
             label_clock to release them by; resume it under the spec it was saved with"
        }
        .into());
    }
    if by_label && p.label_learned.is_none() {
        p.label_learned = Some(ClockState::new());
    }
    Ok(())
}
