//! The residual spread a row reads, moved out of `stream.rs`, which is held
//! to the size a source file may reach (task 368).

use super::*;

/// A slot's `sigma` as the row finds it: read from the state BEFORE this
/// row's residual is folded in, so `zscore` is out-of-sample like the
/// prediction it scales -- under a window (`ring`), inside it, where the
/// model reads its fit (review 2026-09-12, S1). NaN for a slot with no
/// weight yet, or none inside the window.
///
/// One slot at a time, where it is read: a held row's record reads it for
/// the conformal radius the row is shown (the tracker's multiplier times
/// it, task 368), and the residual loop reads it beside each residual. A
/// pass of its own over every slot, before the record, put a call and a
/// loop on every row of every spec, conformal or not (task 372).
#[inline]
pub(super) fn pre_row_sigma(
    ring: Option<&ResidWindow>,
    decay: Decay,
    (resid_w, resid_var): (&[f64], &[f64]),
    slot: usize,
) -> f64 {
    let w = resid_w[slot];
    if w <= 0.0 {
        return f64::NAN;
    }
    let var = match ring {
        Some(ring) => ring.inside(decay, slot, w, resid_var[slot]),
        None => Some(resid_var[slot]),
    };
    var.map_or(f64::NAN, |var| var.max(0.0).sqrt())
}
