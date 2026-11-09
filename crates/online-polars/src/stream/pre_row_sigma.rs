//! The residual spread a row reads, moved out of `stream.rs`, which is held
//! to the size a source file may reach (task 368).

use super::*;

/// Each slot's `sigma` as the row finds it, into `sig`: read from the state
/// BEFORE this row's residual is folded in, so `zscore` is out-of-sample like
/// the prediction it scales -- under a window (`ring`), inside it, where the
/// model reads its fit (review 2026-09-12, S1). NaN for a slot with no weight
/// yet; empty for a model that predicts no target (`residuals` false). Read
/// before a held row's record is kept, since the conformal radius the row is
/// shown is the tracker's multiplier times it (task 368). The instance's
/// fields, not the instance: its scratch is borrowed beside them.
pub(super) fn pre_row_sigma(
    residuals: bool,
    ring: Option<&SpreadRef<'_>>,
    decay: Decay,
    (resid_w, resid_var): (&[f64], &[f64]),
    n_slots: usize,
    sig: &mut Vec<f64>,
) {
    sig.clear();
    if !residuals {
        return;
    }
    sig.resize(n_slots, f64::NAN);
    let ring = ring.map(SpreadRef::get);
    for (slot, s) in sig.iter_mut().enumerate() {
        if resid_w[slot] <= 0.0 {
            continue;
        }
        let var = match ring {
            Some(ring) => ring.inside(decay, slot, resid_w[slot], resid_var[slot]),
            None => Some(resid_var[slot]),
        };
        if let Some(var) = var {
            *s = var.max(0.0).sqrt();
        }
    }
}
