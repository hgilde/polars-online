//! The residual spread's window (review 2026-09-12, S1; the user's decision
//! of 2026-09-15).
//!
//! Under a `window` the fit is read from the rows inside it, and `sigma`
//! describes those rows too -- with `resid_z`, the drift detector's scale,
//! the conformal band and the ranking of `emit_selected` and
//! `emit_averaged`, which all read it. The spread is the stream's own, one
//! EW mean of squared residuals per slot (`Stream::resid_var`), so it is cut
//! the way the models cut theirs ([`online_core::Snapshots`]): a ring of the
//! spread as it stood before each learned row, decayed to that row,
//! subtracted at the oldest snapshot still inside the window. It is taken on
//! the rows the model learns, with its `window` and `window_every`, keyed by
//! a clock summed from the deltas the model was stepped with, and read
//! before the row, where the model reads its window -- so its boundary is
//! the fit's. A thinning budget can move the two boundaries apart, each
//! still inside the window, which is the promise.

use online_core::{Decay, Footprint, Snapshots, WindowBudget, truncated_scalar};
use serde::{Deserialize, Serialize};

/// The spread before a learned row, decayed to it: per slot, the weight and
/// the mean of the squared residuals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Spread {
    w: Vec<f64>,
    var: Vec<f64>,
}

impl Footprint for Spread {
    fn footprint(&self) -> usize {
        std::mem::size_of_val(self.w.as_slice()) + std::mem::size_of_val(self.var.as_slice())
    }
}

/// One model instance's ring (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidWindow {
    /// The clock of the last row learned, summed from the deltas.
    clock: f64,
    snaps: Snapshots<Spread>,
}

impl ResidWindow {
    pub fn new(window: f64, every: usize, budget: Option<WindowBudget>) -> Result<Self, String> {
        let mut snaps = Snapshots::new(window, every)?;
        snaps.set_budget(budget);
        Ok(Self { clock: 0.0, snaps })
    }

    /// Slot `slot`'s mean squared residual inside the window, given the whole
    /// history's weight and mean as they stand before the row; `None` when
    /// nothing is left inside. With no snapshot yet there is nothing to
    /// subtract.
    pub fn inside(&self, decay: Decay, slot: usize, w: f64, var: f64) -> Option<f64> {
        let Some((t, old)) = self.snaps.boundary() else {
            return Some(var);
        };
        let f = decay.factor(self.clock - t);
        truncated_scalar(w, var, old.w[slot], old.var[slot], f).map(|(_, v)| v)
    }

    /// Before a learned row's residuals are folded: offer the spread as it
    /// stands, decayed to the row by `lam`, move the clock to the row, and
    /// drop what can no longer be the boundary.
    pub fn learn(&mut self, d_clock: f64, lam: f64, w: &[f64], var: &[f64]) {
        let t = self.clock + d_clock;
        self.snaps.offer(t, || Spread {
            w: w.iter().map(|w| w * lam).collect(),
            var: var.to_vec(),
        });
        self.clock = t;
        self.snaps.trim(t);
    }

    /// Whether this ring's snapshots are `n_slots` wide: a file written for
    /// another spec's slots is not restored into this one.
    pub fn fits(&self, n_slots: usize) -> bool {
        self.snaps
            .boundary()
            .is_none_or(|(_, s)| s.w.len() == n_slots && s.var.len() == n_slots)
    }

    /// The budget is configuration, which a state does not carry.
    pub fn set_budget(&mut self, budget: Option<WindowBudget>) {
        self.snaps.set_budget(budget);
    }

    /// A refusing budget's overrun: the ring's bytes and its spacing.
    pub fn over_budget(&self) -> Option<(usize, usize)> {
        self.snaps.over_budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring's reading is the spread of the residuals inside the window,
    /// summed directly: the rows whose age at the last learned row is at
    /// most the window, each at `0.5^(age/h)`, across an irregular clock
    /// and a burst the window drops.
    #[test]
    fn the_spread_inside_is_the_direct_sum_over_the_window() {
        let (h, window) = (7.0, 10.0);
        let decay = Decay::Halflife(h);
        let mut ring = ResidWindow::new(window, 1, None).unwrap();
        let (mut w, mut var) = (vec![0.0], vec![0.0]);
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let mut clock = 0.0;
        for i in 0..60u32 {
            let d = if i % 7 == 3 { 2.5 } else { 1.0 };
            let r = if (20..25).contains(&i) {
                30.0
            } else {
                f64::from(i % 5) - 2.0
            };
            if let Some(&(last, _)) = rows.last() {
                let (mut num, mut den) = (0.0, 0.0);
                for &(t, r) in rows.iter().filter(|(t, _)| last - t <= window) {
                    let a = decay.factor(last - t);
                    num += a * r * r;
                    den += a;
                }
                let got = ring.inside(decay, 0, w[0], var[0]).unwrap();
                let want = num / den;
                assert!(
                    (got - want).abs() <= 1e-10 * want,
                    "row {i}: {got} vs {want}"
                );
            }
            let lam = decay.factor(d);
            ring.learn(d, lam, &w, &var);
            clock += d;
            let w_new = lam * w[0] + 1.0;
            var[0] = (lam * w[0] * var[0] + r * r) / w_new;
            w[0] = w_new;
            rows.push((clock, r));
        }
    }
}
