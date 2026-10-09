//! The residual spread's window (review 2026-09-12, S1; the user's decision
//! of 2026-09-15).
//!
//! Under a `window` the fit is read from the rows inside it, and `sigma`
//! describes those rows too -- with `zscore`, the drift detector's scale,
//! the conformal band and the ranking of `emit_selected` and
//! `emit_averaged`, which all read it. The spread is the stream's own, one
//! EW mean of squared residuals per slot (`Stream::resid_var`), so it is cut
//! the way the models cut theirs ([`online_core::Snapshots`]): a ring of the
//! spread as it stood before each learned row, decayed to that row,
//! subtracted at the oldest snapshot still inside the window. It is taken on
//! the rows the model learns, with its `window` and its snapshot cadence
//! (`window_every` clock units or `max_rows_between_snapshots` rows,
//! whichever comes first: [`online_core::Cadence`], docs/PLAN.md task 162),
//! keyed by the stamps the model's window is keyed by, each row's decayed
//! clock held exactly ([`online_core::Stamp`], task 175), beside a clock
//! summed from the deltas the model was stepped with, which decays a
//! snapshot forward; and read before the row, where the model reads its
//! window -- so its boundary is the fit's. A thinning budget can move the
//! two boundaries apart, each still inside the window, which is the
//! promise.

use online_core::{
    Cadence, Decay, Footprint, Snapshots, Stamp, WindowBudget, WindowClosed, WindowShadow,
    truncated_scalar,
};
use serde::{Deserialize, Serialize};

/// The spread before a learned row, decayed to it: per slot, the weight and
/// the mean of the squared residuals, and the count of residuals of
/// positive weight folded so far ([`ResidWindow::rows`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Spread {
    w: Vec<f64>,
    var: Vec<f64>,
    #[serde(default)]
    n: Vec<u64>,
}

impl Footprint for Spread {
    fn footprint(&self) -> usize {
        std::mem::size_of_val(self.w.as_slice())
            + std::mem::size_of_val(self.var.as_slice())
            + std::mem::size_of_val(self.n.as_slice())
    }
}

/// One model instance's ring (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidWindow {
    /// The clock of the last row learned, summed from the deltas.
    clock: f64,
    snaps: Snapshots<Spread>,
    /// Per slot, the residuals of positive weight folded so far, which each
    /// snapshot keeps as it stood before its row: the window holds a slot's
    /// rows when the count now exceeds the boundary's, exactly, where the
    /// weights' difference is a remainder that rounding leaves above 0. A
    /// slot whose target was absent for some 1,060 half-lives kept a weight
    /// stuck a few subnormal steps above 0, and its window read a `sigma` of
    /// 0 where a slot never seen reads none (docs/PLAN.md task 217). A row
    /// of weight 0 is not counted (hard rule 9). Sized at the first row.
    #[serde(default)]
    rows: Vec<u64>,
}

impl ResidWindow {
    /// The model's `window`, snapshot `cadence` and edge `closed`
    /// ([`crate::spec::ModelKind::window_and_cadence`],
    /// [`crate::spec::ModelKind::window_edge`]), so the spread's boundary is
    /// the fit's.
    pub fn new(
        window: f64,
        cadence: Cadence,
        closed: WindowClosed,
        budget: Option<WindowBudget>,
    ) -> Result<Self, String> {
        let mut snaps = Snapshots::with_cadence(window, cadence)?.closed(closed);
        snaps.set_budget(budget);
        Ok(Self {
            clock: 0.0,
            snaps,
            rows: Vec::new(),
        })
    }

    /// Slot `slot`'s mean squared residual inside the window, given the whole
    /// history's weight and mean as they stand before the row; `None` when
    /// nothing is left inside: no residual of positive weight since the
    /// boundary, counted ([`Self::rows`]), or no weight the subtraction can
    /// read. With no snapshot yet there is nothing to subtract.
    pub fn inside(&self, decay: Decay, slot: usize, w: f64, var: f64) -> Option<f64> {
        let Some((t, old)) = self.snaps.boundary() else {
            return Some(var);
        };
        if let Some(&then) = old.n.get(slot)
            && self.rows.get(slot).is_none_or(|&now| now <= then)
        {
            return None;
        }
        let f = decay.factor(self.clock - t);
        truncated_scalar(w, var, old.w[slot], old.var[slot], f).map(|(_, v)| v)
    }

    /// A learned row: offer the spread as it stands before the row's
    /// residuals are folded, decayed to the row by `lam`, move the clock to
    /// the row, and drop what can no longer be the boundary, at the row's
    /// `stamp` -- the one the model's window was handed -- or the summed
    /// clock for `None`. Then count each slot whose residual `resid` the
    /// row folds at a positive `weight`.
    pub fn learn(
        &mut self,
        d_clock: f64,
        stamp: Option<Stamp>,
        lam: f64,
        (w, var): (&[f64], &[f64]),
        (resid, weight): (&[f64], f64),
    ) {
        if let Some(stamp) = stamp {
            self.snaps.stamp_next(stamp);
        }
        if self.rows.len() != w.len() {
            self.rows = vec![0; w.len()];
        }
        let rows = &self.rows;
        self.snaps.learn(&mut self.clock, d_clock, || Spread {
            w: w.iter().map(|w| w * lam).collect(),
            var: var.to_vec(),
            n: rows.clone(),
        });
        if weight > 0.0 {
            for (n, r) in self.rows.iter_mut().zip(resid) {
                if r.is_finite() {
                    *n += 1;
                }
            }
        }
    }

    /// Whether this ring's snapshots and counts are `n_slots` wide: a file
    /// written for another spec's slots is not restored into this one.
    pub fn fits(&self, n_slots: usize) -> bool {
        (self.rows.is_empty() || self.rows.len() == n_slots)
            && self.snaps.boundary().is_none_or(|(_, s)| {
                s.w.len() == n_slots && s.var.len() == n_slots && s.n.len() == n_slots
            })
    }

    /// The budget is configuration, which a state does not carry.
    pub fn set_budget(&mut self, budget: Option<WindowBudget>) {
        self.snaps.set_budget(budget);
    }

    /// The ring's shadow (docs/PLAN.md task 115 (d)): each snapshot a spread
    /// of `n_slots` slots, as `learn` takes one.
    pub fn shadow(&self, n_slots: usize) -> WindowShadow {
        WindowShadow::new(self.clock, &self.snaps, || Spread {
            w: vec![0.0; n_slots],
            var: vec![0.0; n_slots],
            n: vec![0; n_slots],
        })
    }

    /// A refusing budget's overrun: the ring's bytes and its cadence.
    pub fn over_budget(&self) -> Option<(usize, Cadence)> {
        self.snaps.over_budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spread's snapshot counts every vector it holds in its footprint,
    /// and holds nothing else: a field added to it and not to the footprint
    /// fails here, as one did in two of the models' snapshots
    /// (docs/PLAN.md task 130).
    #[test]
    fn the_footprint_counts_every_vector() {
        let snap = Spread {
            w: vec![1.0; 3],
            var: vec![2.0; 5],
            n: vec![4; 2],
        };
        let state = serde_json::to_value(&snap).unwrap();
        let values = state.as_object().unwrap().values();
        let numbers: usize = values.map(|v| v.as_array().map_or(1, Vec::len)).sum();
        assert_eq!(snap.footprint(), numbers * std::mem::size_of::<f64>());
    }

    /// **A window that holds none of a slot's residuals reads none**
    /// (docs/PLAN.md task 217), by the ring's count against the boundary's.
    /// Sixty residuals at a weight of `1e-300`, then 300 rows with none at
    /// `lam = 0.75`: the slot's weight ages into the subnormal range and
    /// sticks a few steps above 0, and the window of 5 read the remainder
    /// as a spread (a `sigma` of 0, where a slot never seen reads none). A
    /// residual at weight 0 is no row of the window; one at `1e-310` is.
    #[test]
    fn a_window_holding_none_of_a_slots_residuals_reads_none() {
        let (lam, decay) = (0.75, Decay::Lam(0.75));
        for last in [None, Some(0.0), Some(1e-310)] {
            let mut ring =
                ResidWindow::new(5.0, Cadence::EVERY_ROW, WindowClosed::Right, None).unwrap();
            let (mut w, mut var) = (vec![0.0], vec![0.0]);
            let mut stuck = f64::NAN;
            for i in 0..361 {
                if i == 360 {
                    stuck = w[0];
                }
                let (r, weight) = match i {
                    0..60 => (f64::from(i % 5) - 2.0, 1e-300),
                    360 => last.map_or((f64::NAN, 1.0), |wt| (1.5, wt)),
                    _ => (f64::NAN, 1.0),
                };
                let d = if i == 0 { 0.0 } else { 1.0 };
                let step = if i == 0 { 1.0 } else { lam };
                ring.learn(d, None, step, (&w, &var), (&[r], weight));
                if r.is_finite() {
                    let w_new = step * w[0] + weight;
                    if w_new > 0.0 {
                        var[0] = (step * w[0] * var[0] + weight * r * r) / w_new;
                        w[0] = w_new;
                    }
                } else {
                    w[0] *= step;
                }
            }
            assert!(
                stuck > 0.0 && stuck < 1e-320,
                "the fixture: a stuck weight, {stuck:e}"
            );
            let got = ring.inside(decay, 0, w[0], var[0]);
            match last {
                Some(wt) if wt > 0.0 => assert!(
                    got.is_some_and(|v| (v - 2.25).abs() < 1e-6),
                    "a residual of weight {wt:e} inside: {got:?}"
                ),
                _ => assert_eq!(got, None, "the last row's residual {last:?}"),
            }
        }
    }

    /// The ring's reading is the spread of the residuals inside the window,
    /// summed directly: the rows whose age at the last learned row is less
    /// than the window (at most it, under `closed = "both"`), each at
    /// `0.5^(age/h)`, across an irregular clock -- whose steps of 1 and 2
    /// put rows exactly one window back -- and a burst the window drops.
    #[test]
    fn the_spread_inside_is_the_direct_sum_over_the_window() {
        for closed in [WindowClosed::Right, WindowClosed::Both] {
            spread_inside_is_the_direct_sum(closed);
        }
    }

    fn spread_inside_is_the_direct_sum(closed: WindowClosed) {
        let (h, window) = (7.0, 10.0);
        let decay = Decay::Halflife(h);
        let inside = |age: f64| match closed {
            WindowClosed::Right => age < window,
            WindowClosed::Both => age <= window,
        };
        let mut on_the_edge = 0;
        let mut ring = ResidWindow::new(window, Cadence::EVERY_ROW, closed, None).unwrap();
        let (mut w, mut var) = (vec![0.0], vec![0.0]);
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let mut clock = 0.0;
        for i in 0..60u32 {
            let d = if i % 4 == 3 { 2.0 } else { 1.0 };
            let r = if (20..25).contains(&i) {
                30.0
            } else {
                f64::from(i % 5) - 2.0
            };
            if let Some(&(last, _)) = rows.last() {
                let (mut num, mut den) = (0.0, 0.0);
                on_the_edge += rows.iter().filter(|(t, _)| last - t == window).count();
                for &(t, r) in rows.iter().filter(|(t, _)| inside(last - t)) {
                    let a = decay.factor(last - t);
                    num += a * r * r;
                    den += a;
                }
                let got = ring.inside(decay, 0, w[0], var[0]).unwrap();
                let want = num / den;
                assert!(
                    (got - want).abs() <= 1e-10 * want,
                    "{closed:?}, row {i}: {got} vs {want}"
                );
            }
            let lam = decay.factor(d);
            ring.learn(d, None, lam, (&w, &var), (&[r], 1.0));
            clock += d;
            let w_new = lam * w[0] + 1.0;
            var[0] = (lam * w[0] * var[0] + r * r) / w_new;
            w[0] = w_new;
            rows.push((clock, r));
        }
        assert!(
            on_the_edge > 5,
            "rows must sit exactly one window back for the edge to be tested"
        );
    }
}
