//! A hard cutoff on an exponentially weighted accumulator (docs/PLAN.md §13).
//!
//! An EW mean never forgets: a halflife of `h` still leaves 12.5% of the
//! weight on data older than `3h`. Some questions need the other thing —
//! *nothing* from before a point, as a guarantee. The identity that makes it
//! cheap is that an EW sum contains its own past:
//!
//! ```text
//! A(t) = lam^(t-u) * A(u)  +  (everything after u)
//! ```
//!
//! so the part to discard is the accumulator's own earlier value, decayed
//! forward, and truncation is a subtraction rather than a recomputation.
//! What a model has to keep is therefore not the rows but a ring of past
//! *snapshots*, one per row (or one per `every` rows), which is what
//! this module holds.
//!
//! **The boundary is chosen conservatively, and the direction matters.** A
//! snapshot taken before the row at `t_j` carries everything strictly older
//! than `t_j`; subtracting it retains rows at `t_j` and after. Honouring
//! "nothing older than `window`" therefore needs `t_j >= t - window`, so the
//! boundary is the *oldest snapshot still inside the window* and a coarse
//! `every` discards slightly more than asked, never less. Rounding the other
//! way would keep rows the window promised to exclude, which is the one
//! failure this design refuses to have.
//!
//! That needs a snapshot inside the window. With a coarse `every` there was
//! none after a clock gap longer than the window, or wherever `every` rows
//! span more clock than it: the newest snapshot was older than the window,
//! it stayed the boundary, and rows the window excludes stayed in the fit.
//! So a row that finds the newest snapshot outside the window is
//! snapshotted whatever the cadence (review 2026-09-12, S6).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::EwCov;

/// What a window does when its snapshots pass a memory budget, and the
/// budget in MiB (review 2026-09-12, P4; the user's decision of
/// 2026-09-15). Past it the ring either thins -- every other snapshot
/// dropped and the spacing doubled, as often as it takes, so the boundary
/// grows coarser and never keeps an older row -- or stops: the snapshot that
/// would cross is not kept, none is made after it, and the overrun is
/// recorded for the caller to refuse the run on, naming the size and
/// `window_every`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowBudget {
    Thin(f64),
    Refuse(f64),
}

impl WindowBudget {
    /// Refuse past 256 MiB: the figure `gram_block_rows` refuses at, and
    /// what a spec that names no budget gets.
    pub const DEFAULT: WindowBudget = WindowBudget::Refuse(256.0);

    /// The budget in MiB.
    pub fn mib(&self) -> f64 {
        match *self {
            WindowBudget::Thin(m) | WindowBudget::Refuse(m) => m,
        }
    }

    fn bytes(&self) -> f64 {
        self.mib() * 1024.0 * 1024.0
    }
}

/// The heap a snapshot holds, in bytes: what a window's budget counts.
pub trait Footprint {
    fn footprint(&self) -> usize;
}

/// Bytes in a slice of floats, for a [`Footprint`].
pub(crate) fn floats(v: &[f64]) -> usize {
    std::mem::size_of_val(v)
}

/// Snapshots of an accumulator, oldest first, spanning at most one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshots<S> {
    /// Clock units of history the window keeps.
    window: f64,
    /// Snapshot every `every` rows; `1` snapshots them all and makes the
    /// boundary as tight as the data allows. A row whose newest snapshot has
    /// left the window is snapshotted whatever the count.
    every: usize,
    /// Rows since the last snapshot, so the cadence is counted in the
    /// *stream* and never in the chunk (hard rule 3).
    since: usize,
    /// `(clock of the row the snapshot precedes, snapshot)`.
    ring: VecDeque<(f64, S)>,
    /// The budget, and a refusing budget's overrun: configuration the caller
    /// sets after building or restoring the model, so not part of the
    /// state, and two rings that differ only here are equal.
    #[serde(skip)]
    limit: Limit,
}

#[derive(Debug, Clone, Default)]
struct Limit {
    budget: Option<WindowBudget>,
    over: Option<usize>,
    /// The bytes the ring holds, kept as snapshots come and go so a push
    /// costs no walk of the ring; counted afresh whenever a budget is set.
    held: usize,
}

impl PartialEq for Limit {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<S> Snapshots<S> {
    /// `window` in clock units, `every` rows between snapshots.
    pub fn new(window: f64, every: usize) -> Result<Self, String> {
        if !window.is_finite() || window <= 0.0 {
            return Err(format!("window must be > 0 (got {window})"));
        }
        if every == 0 {
            return Err("window_every must be >= 1".into());
        }
        Ok(Self {
            window,
            every,
            since: usize::MAX, // the first row always snapshots
            ring: VecDeque::new(),
            limit: Limit::default(),
        })
    }

    pub fn window(&self) -> f64 {
        self.window
    }

    /// Every snapshot held, oldest first, for a caller that must rewrite them
    /// in place: a state converted as it is read.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.ring.iter_mut().map(|(_, s)| s)
    }

    /// The snapshot to subtract, and the clock it is referenced at.
    pub fn boundary(&self) -> Option<&(f64, S)> {
        self.ring.front()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

impl<S: Footprint> Snapshots<S> {
    /// Drop what can never be the boundary again: everything strictly older
    /// than `now - window`. What is left at the front is the oldest snapshot
    /// inside the window, which is the boundary; [`Self::offer`] at `now`
    /// has seen to it that there is one. (A stale front did not "subtract
    /// the whole accumulator", as the comment here said: it subtracts the
    /// state before the row it precedes, which keeps that row -- review
    /// 2026-09-12, S6.)
    pub fn trim(&mut self, now: f64) {
        let oldest = now - self.window;
        while self.ring.len() > 1 && self.ring[0].0 < oldest {
            if let Some((_, s)) = self.ring.pop_front() {
                self.limit.held = self.limit.held.saturating_sub(s.footprint());
            }
        }
    }

    /// Offer a snapshot of the state *as it stands before* the row at
    /// `clock`, already decayed to that row. Taken when the cadence is due,
    /// and whatever the cadence when the newest snapshot is older than the
    /// window, so the boundary is always inside it (the module docs); `make`
    /// is not called otherwise. Past a thinning budget the ring then thins;
    /// a refusing one keeps no snapshot that would cross it, and makes none
    /// after ([`WindowBudget`]).
    pub fn offer(&mut self, clock: f64, make: impl FnOnce() -> S) {
        self.since = self.since.saturating_add(1);
        if self.limit.over.is_some() {
            return;
        }
        let stale = self
            .ring
            .back()
            .is_none_or(|&(t, _)| t < clock - self.window);
        if self.since >= self.every || stale {
            self.since = 0;
            let snap = make();
            let held = self.limit.held + snap.footprint();
            // The ring stops short of a refusing budget: the caller refuses
            // the run, and until it looks, the memory stays bounded.
            let refuse = matches!(
                self.limit.budget,
                Some(b @ WindowBudget::Refuse(_)) if held as f64 > b.bytes()
            );
            if refuse {
                self.limit.over = Some(held);
                return;
            }
            self.limit.held = held;
            self.ring.push_back((clock, snap));
            self.enforce();
        }
    }

    /// Bound the ring ([`WindowBudget`]), `None` for no bound. A ring already
    /// past the new budget thins, or records its overrun, at once.
    pub fn set_budget(&mut self, budget: Option<WindowBudget>) {
        self.limit = Limit {
            budget,
            over: None,
            held: self.bytes(),
        };
        self.enforce();
    }

    /// A refusing budget's overrun: the bytes the ring reached, and the
    /// spacing it reached them at (`window_every`, doubled by any thinning).
    pub fn over_budget(&self) -> Option<(usize, usize)> {
        self.limit.over.map(|bytes| (bytes, self.every))
    }

    /// The bytes the ring's snapshots hold.
    pub fn bytes(&self) -> usize {
        self.ring.iter().map(|(_, s)| s.footprint()).sum()
    }

    fn enforce(&mut self) {
        let Some(budget) = self.limit.budget else {
            return;
        };
        let limit = budget.bytes();
        if self.limit.held as f64 <= limit {
            return;
        }
        match budget {
            // Only a ring given a budget it is already past gets here: a
            // snapshot that would cross one is turned away in `offer`.
            WindowBudget::Refuse(_) => self.limit.over = Some(self.limit.held),
            WindowBudget::Thin(_) => {
                // Keep the newest and every second one before it, and take
                // them half as often, until the ring fits. The front snapshot
                // is the boundary, so losing it moves the boundary later: the
                // window drops more rows, and never keeps an older one.
                while self.limit.held as f64 > limit && self.ring.len() > 1 {
                    let n = self.ring.len();
                    self.ring = std::mem::take(&mut self.ring)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| (n - 1 - i) % 2 == 0)
                        .map(|(_, e)| e)
                        .collect();
                    self.every = self.every.saturating_mul(2);
                    self.limit.held = self.bytes();
                }
            }
        }
    }
}

/// The smallest remainder, relative to the weight it was subtracted from, that
/// is still something. The window's weight is `W - f·W_u`, a difference of two
/// positives that are equal in exact arithmetic when every row of the history
/// has aged out -- and in `f64` then comes out at about `±W·1e-16`, not 0. A
/// positive crumb that size would pass as "a window with rows in it" and turn
/// every mean divided by it into noise, so anything below this fraction is an
/// empty window (review 2026-09-12, C2: "clamp the truncated weight at 0
/// before the test"). A genuine window never gets near it: its newest row
/// alone carries weight 1 against a history of at most `1/(1 - lam)`.
pub const EMPTY_FRACTION: f64 = 1e-12;

/// An [`EwCov`]'s data, decayed to the clock of the row it precedes. Means and
/// centred co-moments do not move under decay; only the two weight sums do,
/// which is why a snapshot is this and not a whole accumulator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub w: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q: Option<f64>,
    pub m: Vec<f64>,
    pub c: Vec<f64>,
}

impl Moments {
    /// The accumulator as it stands, decayed by `lam` -- what the snapshot
    /// before the row at `t` must hold for `t` to be retained when it is
    /// subtracted.
    pub fn of(cov: &EwCov, lam: f64) -> Self {
        Self {
            w: cov.n_eff() * lam,
            q: cov.q_sum().map(|q| q * lam * lam),
            m: cov.means().to_vec(),
            c: cov.comoments().to_vec(),
        }
    }
}

impl Footprint for Moments {
    fn footprint(&self) -> usize {
        2 * std::mem::size_of::<f64>() + floats(&self.m) + floats(&self.c)
    }
}

/// `cov` with everything before the row the snapshot precedes removed:
/// `A(t) - f·A(u)`, in the mean form the accumulator stores. That row, at
/// the snapshot's clock `u`, and every row after it are kept (the module
/// docs say why the boundary is drawn there).
///
/// `None` when the window holds no weight -- a clock gap longer than it, or a
/// boundary that is the whole accumulator. The caller reports nothing rather
/// than dividing by zero (hard rule 9).
///
/// **Centred, never through raw moments.** With `W` the live weight, `W_u` the
/// snapshot's weight decayed forward and `W_R = W - W_u` what the window
/// keeps, the pooling identity for two sets of weighted moments,
///
/// ```text
/// W·C = W_R·C_R + W_u·C_u + (W_u·W / W_R)·(m_u - m)(m_u - m)ᵀ
/// ```
///
/// solved for the remainder gives `C_R`, and `m_R = m - (W_u/W_R)·(m_u - m)`.
/// Every term is a centred moment or a difference of two means, so nothing is
/// ever the size of `m²`: the window keeps its precision at any offset. The
/// earlier form went back through `E[x x'] = C + m m'`, subtracted, and
/// re-centred, which at a level of `1e8` and unit variance left the windowed
/// variance with a resolution of about 2 -- nothing (review 2026-09-12, C17;
/// `tests/test_second_opinion.py` holds it to `numpy.cov` at that offset).
///
/// What precision remains to lose is the fraction discarded: the result is a
/// difference of positives of size `C`, so it is negligible at `window =
/// 3·halflife`, where the correction is an eighth, and worse as the window
/// shortens toward the halflife.
pub fn truncated(cov: &EwCov, old: &Moments, f: f64) -> Option<EwCov> {
    let k = cov.k();
    let w_now = cov.n_eff();
    let w_old = f * old.w;
    let w = w_now - w_old;
    if w <= EMPTY_FRACTION * w_now || !w.is_finite() {
        return None;
    }
    // `ratio = W_u / W_R` and `g = W / W_R`, so `C_R = g·C - ratio·C_u -
    // ratio·g·d dᵀ` with `d = m_u - m`.
    let (ratio, g) = (w_old / w, w_now / w);
    let d: Vec<f64> = (0..k).map(|i| old.m[i] - cov.mean(i)).collect();
    let mean: Vec<f64> = (0..k).map(|i| cov.mean(i) - ratio * d[i]).collect();
    let c_now = cov.comoments();
    let mut cen = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let ij = i * k + j;
            cen[ij] = g * c_now[ij] - ratio * old.c[ij] - ratio * g * d[i] * d[j];
        }
        // What is left to lose is a difference of positives; a variance it
        // takes a hair below zero is zero (review V3).
        cen[i * k + i] = cen[i * k + i].max(0.0);
    }
    let q = match (cov.q_sum(), old.q) {
        (Some(q_now), Some(q_then)) => Some((q_now - f * f * q_then).max(0.0)),
        _ => None,
    };
    let mut out = cov.clone();
    out.set_moments(&mean, &cen, w, q);
    Some(out)
}

/// One mean-form accumulator's worth of subtraction: `(w, mean)` truncated
/// against a snapshot, for the per-target sums that sit beside a Gram.
/// `None` when nothing is left inside the window.
pub fn truncated_mean(
    w_now: f64,
    mean: &[f64],
    w_old: f64,
    mean_old: &[f64],
    f: f64,
) -> Option<(f64, Vec<f64>)> {
    let w = w_now - f * w_old;
    if w <= EMPTY_FRACTION * w_now || !w.is_finite() {
        return None;
    }
    let out = mean
        .iter()
        .zip(mean_old)
        .map(|(m, m0)| (w_now * m - f * w_old * m0) / w)
        .collect();
    Some((w, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight bytes a snapshot, for rings of counters.
    impl Footprint for usize {
        fn footprint(&self) -> usize {
            8
        }
    }

    /// Past a thinning budget the ring keeps every other snapshot and
    /// doubles its spacing until it fits, and its boundary stays inside the
    /// window: it drops rows, never keeps an older one (review 2026-09-12,
    /// P4).
    #[test]
    fn past_a_thinning_budget_the_ring_fits_and_its_boundary_stays_inside() {
        let window = 100.0;
        let mut snaps = Snapshots::new(window, 1).unwrap();
        // Ten snapshots' worth.
        snaps.set_budget(Some(WindowBudget::Thin(80.0 / (1024.0 * 1024.0))));
        for i in 0..1000u32 {
            let t = f64::from(i);
            snaps.offer(t, || i as usize);
            snaps.trim(t);
            assert!(snaps.bytes() <= 80, "row {i}: {} bytes", snaps.bytes());
            assert_eq!(snaps.limit.held, snaps.bytes(), "row {i}: the count");
            let &(b, _) = snaps.boundary().unwrap();
            assert!(b >= t - window, "row {i}: bounded at {b}");
        }
        assert!(snaps.every > 1, "and it takes its snapshots less often");
        assert_eq!(
            snaps.over_budget(),
            None,
            "a thinning budget refuses nothing"
        );
    }

    /// Past a refusing budget the ring records the overrun, with its
    /// spacing, for the caller to refuse the run on -- and stops there: the
    /// snapshot that crossed is not kept and none is made after it, so the
    /// memory the refusal bounds stays bounded until the caller checks.
    #[test]
    fn past_a_refusing_budget_the_ring_records_the_overrun() {
        let mut snaps = Snapshots::new(100.0, 1).unwrap();
        snaps.set_budget(Some(WindowBudget::Refuse(80.0 / (1024.0 * 1024.0))));
        for i in 0..10u32 {
            snaps.offer(f64::from(i), || i as usize);
            assert_eq!(snaps.over_budget(), None, "row {i}: ten fit");
        }
        snaps.offer(10.0, || 10);
        assert_eq!(snaps.over_budget(), Some((88, 1)));
        assert_eq!(snaps.bytes(), 80, "the snapshot that crossed is not kept");
        let mut made = 0;
        for i in 11..1000u32 {
            snaps.offer(f64::from(i), || {
                made += 1;
                i as usize
            });
            snaps.trim(f64::from(i));
            assert!(snaps.bytes() <= 80, "row {i}: {} bytes", snaps.bytes());
            assert_eq!(snaps.limit.held, snaps.bytes(), "row {i}: the count");
        }
        assert_eq!(made, 0, "no snapshot is made past a refusal");
        assert_eq!(snaps.over_budget(), Some((88, 1)), "the first overrun");
    }

    /// The module doc's boundary, which `trim`'s comment had the other way
    /// round (review 2026-09-12, S6): a snapshot is the state before the
    /// row it precedes, so subtracting it keeps that row. Across a gap of
    /// twice the window the one snapshot left is the one taken before the
    /// row after the gap, and what the subtraction leaves is that row.
    #[test]
    fn the_boundary_keeps_the_row_it_precedes() {
        let (window, lam) = (5.0, 0.9f64);
        let mut cov = EwCov::new(1);
        let mut snaps = Snapshots::new(window, 1).unwrap();
        let mut t = 0.0;
        for (i, d) in [0.0, 1.0, 1.0, 1.0, 2.0 * window].into_iter().enumerate() {
            t += d;
            let l = lam.powf(d);
            snaps.offer(t, || Moments::of(&cov, l));
            snaps.trim(t);
            cov.update(&[i as f64], l, 1.0);
        }
        assert_eq!(
            snaps.len(),
            1,
            "one snapshot is left after a gap of twice the window"
        );
        let (clock, old) = snaps.boundary().unwrap();
        assert_eq!(*clock, t);
        let kept = truncated(&cov, old, 1.0).expect("the row after the gap is in the window");
        assert!(
            (kept.n_eff() - 1.0).abs() < 1e-12,
            "exactly that row's weight: {}",
            kept.n_eff()
        );
        assert!(
            (kept.mean(0) - 4.0).abs() < 1e-12,
            "and its value: {}",
            kept.mean(0)
        );
    }

    /// The module doc's promise -- a coarse `every` discards more than
    /// asked, never less -- held only while a snapshot stayed inside the
    /// window. After a clock gap longer than the window, or where `every`
    /// rows span more than the window, the newest snapshot was older than
    /// the window, `trim` kept it as the boundary, and rows the window
    /// excludes stayed in the fit (found testing review 2026-09-12, S6).
    #[test]
    fn the_boundary_is_never_older_than_the_window() {
        let ramp = |n: u32| (0..n).map(f64::from).collect::<Vec<_>>();
        let gap = |mut v: Vec<f64>| {
            let end = v[v.len() - 1];
            v.extend([end + 100.0, end + 101.0, end + 102.0]);
            v
        };
        for (every, window, clocks) in [
            // A gap longer than the window.
            (5, 20.0, gap(ramp(101))),
            // No gap: five rows between snapshots span more than the window.
            (5, 2.5, ramp(30)),
            // The control: every row is snapshotted.
            (1, 20.0, gap(ramp(101))),
        ] {
            let mut snaps = Snapshots::new(window, every).unwrap();
            for (i, &t) in clocks.iter().enumerate() {
                snaps.offer(t, || i);
                snaps.trim(t);
                let &(b, _) = snaps.boundary().unwrap();
                assert!(
                    b >= t - window,
                    "every {every}, window {window}: the row at {t} is bounded at {b}, \
                     outside the window"
                );
            }
        }
    }
}
