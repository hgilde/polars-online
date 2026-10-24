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
}

fn yes() -> bool {
    true
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
        pending: &mut Vec<PendingRow>,
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
    ) -> Vec<PendingRow> {
        if embargo.is_none() && formulas.is_none() {
            return Vec::new();
        }
        let mut released: Vec<PendingRow> = Vec::new();
        let mut out: Vec<RowPlan> = Vec::with_capacity(plans.len());
        // One template for every replayed row: only `pending` and the
        // break's events differ, and a replay never emits, never wants
        // coefficients and never counts as a row of the frame. A drift it
        // detects is flagged on `at`, the row releasing it (task 160, PB1).
        let replay = |slot: usize, row: &PendingRow, at: usize| RowPlan {
            ri: usize::MAX,
            i: usize::MAX,
            pending: slot,
            d_clock: row.d_clock,
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
                pending.clear();
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
                for row in pending.iter_mut().skip_while(|r| r.resolved) {
                    match res.find(row.seq) {
                        Some((at, ys)) if at <= now => {
                            for (&slot, &y) in f.slots.iter().zip(ys) {
                                row.ys[slot] = Some(y).filter(|v| usable(*v));
                            }
                            row.resolved = true;
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
            if !plan.reset {
                // With no embargo a formula row waits for its window alone.
                let now = elapsed[plan.ri];
                let ready = pending
                    .iter()
                    .take_while(|r| r.resolved && embargo.is_none_or(|e| e.passed(now, r.arrived)))
                    .count();
                for row in pending.drain(..ready) {
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
            pending.push(PendingRow {
                arrived: elapsed[plan.ri],
                d_clock: plan.d_clock,
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
