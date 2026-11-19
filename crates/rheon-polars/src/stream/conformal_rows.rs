//! The conformal interval's per-row work, out of `run_instance`'s row loop
//! (task 375): reached only for a spec with a conformal tracker, and never
//! inlined, so the loop every other spec runs does not carry its code. Its
//! presence there, never run, cost `bank_seqtest` about 37 instructions a
//! row on x86 (spills and reloads in the inlined loop).

use super::*;

/// The radius each slot's interval is shown with, appended to a held row's
/// record: the tracker's multiplier times `sigma` as the row finds it.
/// `spread` is `None` for a model with no residual.
#[inline(never)]
pub(super) fn shown_radii(
    record: &mut Vec<f64>,
    cs: &[Conformal],
    ring: Option<&ResidWindow>,
    decay: Decay,
    spread: Option<(&[f64], &[f64])>,
) {
    record.extend(cs.iter().enumerate().map(|(slot, c)| {
        let sig = spread.map_or(f64::NAN, |s| pre_row_sigma(ring, decay, s, slot));
        c.radius(sig).unwrap_or(f64::NAN)
    }));
}

/// Where a row's interval is written: its row `ri` of `n_rows`, and the
/// `block` of each of the three columns.
pub(super) struct At {
    pub ri: usize,
    pub n_rows: usize,
    pub block: usize,
}

/// The conformal interval is `pred ± q`, `q = m·sigma` read before the
/// row, then the row's own score moves `m`. `sigma` is the pre-row value,
/// the one the row's `sigma` field shows. `held` is the rows waiting for
/// their labels, read under `conformal_rate = "auto"` (0 otherwise), the
/// released one's record already off the front.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn step_conformal(
    cs: &mut [Conformal],
    out: &mut [f64],
    at: At,
    (pred, resid, sig): (&[f64], &[f64], &[f64]),
    (emit, learn): (bool, bool),
    shown_radius: Option<&[f64]>,
    (lam, w): (f64, f64),
    held: usize,
) {
    for (slot, c) in cs.iter_mut().enumerate() {
        if emit {
            let i = slot * at.n_rows + at.ri;
            let p = pred[slot];
            let q = c
                .radius(sig.get(slot).copied().unwrap_or(f64::NAN))
                .unwrap_or(f64::NAN);
            out[i] = p - q;
            out[at.block + i] = p + q;
            out[2 * at.block + i] = c.coverage().unwrap_or(f64::NAN);
        }
        if learn {
            let shown = shown_radius.map(|q| q[slot]);
            c.update_against(resid[slot], sig[slot], lam, w, shown, held);
        }
    }
}
