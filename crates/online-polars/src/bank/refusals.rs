//! The refusals of a chunk the bank cannot learn, in the shape every
//! message takes (WRITING §11): a hard window over its budget, a step back
//! of the clock, a step past the largest double, and a clock amount as the
//! spec gives it. Split from bank.rs, which `tests/test_repo_hygiene.py`
//! holds under 250 KB.
use super::*;

/// The error for a window past a refusing `window_budget` (review
/// 2026-09-12, P4; the user's decision of 2026-09-15): names the spec, the
/// group, the ring's size and the ways out. It is found as the rows go in --
/// after the run of rows in which the ring stopped at the budget -- so the
/// bank that raises it has learned part of the chunk, and marks itself
/// `broken`.
pub(super) fn over_budget(
    spec: &Spec,
    key: &GroupKey,
    bytes: usize,
    cadence: online_core::Cadence,
    learned: bool,
) -> PolarsError {
    polars_err!(ComputeError: "{}", over_budget_message(spec, key, bytes, cadence, learned))
}

/// A clock spacing as the spec writes `snapshot_every`: its own text where the
/// ring runs at it, otherwise in the same form -- a duration on a temporal
/// spec, a number of clock units on any other.
fn spacing_label(spec: &Spec, spacing: f64) -> String {
    match spec.model.window_cadence().and_then(|(every, _)| every) {
        Some(every) if every.value() == spacing => every.label(),
        Some(crate::span::Span::Duration(_)) => {
            crate::span::format_duration((spacing * 1e9).round() as i64)
        }
        _ => crate::spec::num_label(spacing),
    }
}

/// How the refusal says to keep fewer snapshots, from the cadence the ring
/// ran at (docs/PLAN.md task 162): raise what is in force -- the clock
/// spacing, the row cap, or both, since whichever comes first takes a
/// snapshot -- and at every row, give either one.
fn sparser_ring(spec: &Spec, cadence: online_core::Cadence) -> String {
    match (cadence.clock(), cadence.row_cap()) {
        (Some(spacing), Some(rows)) => format!(
            "increase snapshot_every ({} now) and max_rows_between_snapshots ({rows} now) \
             together, since whichever comes first takes a snapshot",
            spacing_label(spec, spacing)
        ),
        (Some(spacing), None) => {
            format!(
                "increase snapshot_every ({} now)",
                spacing_label(spec, spacing)
            )
        }
        (None, Some(1)) => "set snapshot_every or max_rows_between_snapshots (a snapshot every \
                            row now)"
            .to_string(),
        (None, Some(rows)) => format!(
            "increase max_rows_between_snapshots ({rows} now) or give snapshot_every in its place"
        ),
        (None, None) => "set snapshot_every or max_rows_between_snapshots".to_string(),
    }
}

/// The words of [`over_budget`]: how far past which budget, and each way
/// out spelled as the spec writes it -- a larger budget (the user,
/// 2026-09-28: "on failure it should explain how to raise it"), a sparser
/// ring, or thinning in place of refusing. A spec that sets no budget is
/// told it ran under the default, which it never wrote. `learned`: the
/// ring passed the budget as the rows went in, so the bank learned part of
/// the chunk and cannot go on, and the way out starts from a saved state.
pub(super) fn over_budget_message(
    spec: &Spec,
    key: &GroupKey,
    bytes: usize,
    cadence: online_core::Cadence,
    learned: bool,
) -> String {
    let mib = spec.model.window_budget().map_or(0.0, |b| b.mib());
    let group = key
        .0
        .as_deref()
        .filter(|_| spec.group.is_some())
        .map_or(String::new(), |g| format!(" group {g:?}"));
    let named = spec.model.window_parts().and_then(|(_, b)| b).is_some();
    let which = if named {
        format!("window_budget = {{\"refuse\": {mib}}}")
    } else {
        format!("the default window_budget of {mib} MiB, which a spec gets when it sets none")
    };
    // Rounded up: the figure then stays above the budget it is said to pass.
    let held =
        crate::figures::three_figures(bytes as f64 / (1024.0 * 1024.0), crate::figures::Round::Up);
    // Past the budget mid-chunk, the bank learned part of it: the setting
    // is the spec's, which a load compares, so the way out is a new bank
    // fitted from the start.
    let (did, then) = if learned {
        (
            "the bank stopped after learning part of the chunk",
            ", then build the bank again and fit it from the start of its input",
        )
    } else {
        ("the bank refused the chunk", "")
    };
    format!(
        "spec {:?}{group}: the snapshots its hard window keeps hold {held} MiB, above {which}, \
         so {did}. Set window_budget = {{\"refuse\": MiB}} to a larger number of MiB (inf, or \
         float(\"inf\") in Python, for no bound), {} to keep fewer snapshots, or set \
         window_budget = {{\"thin\": MiB}} to thin the snapshots in place of refusing{then}.",
        spec.name,
        sparser_ring(spec, cadence)
    )
}

/// Where a step-back error sends its reader: the how-to that says how to
/// put rows in clock order, at the section on a refused chunk. A message
/// link is never changed once shipped, so that heading stays.
pub(crate) const CLOCK_ORDER_HOW_TO: &str = "https://github.com/hgilde/polars-online/blob/main/\
    docs/how-to/put-rows-in-clock-order.md#fix-a-chunk-the-bank-refused";

/// The error for a refused backwards clock, in the parts every message
/// takes (WRITING §11): where (the spec), what is wrong and the value (the
/// clock column, the size of the step back and the row, counted from 0),
/// what the bank did, and what to do (the ways out, and the how-to that
/// chooses among them). With `restart_after_step_back` unset, the default
/// (`disorder` is `None`), or given and no smaller than the step, the ways
/// out are a sort, skipping what a resumed state has learned, or a setting
/// below the step. Either way the bank was not updated. A step on a number
/// clock is printed to three significant figures, and the setting to stay
/// below it, rounded down, so the remedy holds; a step in integers or
/// nanoseconds is exact.
pub(super) fn backwards_clock(spec: &Spec, refusal: ClockRefusal, row_base: usize) -> PolarsError {
    use crate::figures::{Round, three_figures};
    let column = spec.clock.as_deref().unwrap_or("<row count>");
    let row = row_base + refusal.row;
    // The step exactly: in integer nanoseconds on a temporal clock, where the
    // double of seconds resolves more than a nanosecond above about 104 days;
    // in integers on an integer clock (task 200).
    let (step, below) = match (
        refusal.back_ns.and_then(|ns| i64::try_from(ns).ok()),
        refusal.back_int,
    ) {
        (Some(ns), _) => {
            let s = crate::span::format_duration(ns);
            (s.clone(), s)
        }
        (None, Some(back)) => (back.to_string(), back.to_string()),
        (None, None)
            if spec
                .gap_cap
                .as_ref()
                .is_some_and(crate::span::Span::is_duration) =>
        {
            let s = clock_amount(spec, -refusal.raw);
            (s.clone(), s)
        }
        // Rounded down both times: the step printed is then never above a
        // setting it is "no more than", and the setting to stay below holds.
        (None, None) => {
            let s = three_figures(-refusal.raw, Round::Down);
            (s.clone(), s)
        }
    };
    // Each form written whole, so a reading of the message finds its where
    // and its remedy in one string (WRITING §13).
    match refusal.disorder {
        None => polars_err!(ComputeError:
            "spec {:?}: clock column {column:?} steps back by {step} at row {row} (counting \
             from 0), and restart_after_step_back is not set, so the bank refused the chunk and \
             learned none of it. Sort the rows by the clock, skip the rows a saved state has \
             learned with ModelBank.skip_learned (or --skip-learned), or set \
             restart_after_step_back below {below} to reset the model at such a step back (0 \
             resets it at every one). See {CLOCK_ORDER_HOW_TO} for which to choose.",
            spec.name
        ),
        Some(Disorder {
            min_backwards_jump, ..
        }) => polars_err!(ComputeError:
            "spec {:?}: clock column {column:?} steps back by {step} at row {row} (counting \
             from 0), no more than restart_after_step_back = {}, so the bank refused the chunk \
             and learned none of it. Sort the rows by the clock, skip the rows a saved state has \
             learned with ModelBank.skip_learned (or --skip-learned), or set \
             restart_after_step_back below {below} to reset the model at such a step back (0 \
             resets it at every one). See {CLOCK_ORDER_HOW_TO} for which to choose.",
            spec.name,
            // As the spec wrote it: a duration's text is exact.
            spec.restart_after_step_back
                .as_ref()
                .map_or_else(|| clock_amount(spec, min_backwards_jump), ToString::to_string)
        ),
    }
}

/// The error for a clock step that is not a number of clock units: two
/// values of a number clock whose difference is past the largest double
/// (review 2026-10-06, PB7). Names the spec, the column, the two values and
/// the row; the bank was not updated.
pub(super) fn non_finite_step(spec: &Spec, refusal: StepRefusal, row_base: usize) -> PolarsError {
    polars_err!(ComputeError:
        "spec {:?}: clock column {:?} steps from {} to {} at row {} (counting from 0), a \
         difference too large for a float, so the bank refused the chunk. Shift or rescale the \
         clock column so that the step between two of its values is finite.",
        spec.name,
        spec.clock.as_deref().unwrap_or("<row count>"),
        crate::spec::num_label(refusal.prev),
        crate::spec::num_label(refusal.now),
        row_base + refusal.row
    )
}

/// A clock amount as the spec measures it: on a temporal clock a duration
/// (`1d`, not `86400`), whose every clock parameter is one, `gap_cap`
/// among them (task 120: a `Date` clock's refusal said `86400`); a number of
/// the column's units otherwise.
pub(crate) fn clock_amount(spec: &Spec, v: f64) -> String {
    if spec
        .gap_cap
        .as_ref()
        .is_some_and(crate::span::Span::is_duration)
    {
        crate::span::format_duration((v * 1e9).round() as i64)
    } else {
        crate::spec::num_label(v)
    }
}
