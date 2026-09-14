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
//! *snapshots*, one per learned row (or one per `every` rows), which is what
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

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::EwCov;

/// Snapshots of an accumulator, oldest first, spanning at most one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshots<S> {
    /// Clock units of history the window keeps.
    window: f64,
    /// Snapshot every `every` learned rows; `1` snapshots them all and makes
    /// the boundary as tight as the data allows.
    every: usize,
    /// Learned rows since the last snapshot, so the cadence is counted in the
    /// *stream* and never in the chunk (hard rule 3).
    since: usize,
    /// `(clock of the row the snapshot precedes, snapshot)`.
    ring: VecDeque<(f64, S)>,
}

impl<S> Snapshots<S> {
    /// `window` in clock units, `every` learned rows between snapshots.
    pub fn new(window: f64, every: usize) -> Result<Self, String> {
        if !window.is_finite() || window <= 0.0 {
            return Err(format!("window must be > 0 (got {window})"));
        }
        if every == 0 {
            return Err("window_every must be >= 1".into());
        }
        Ok(Self {
            window,
            every: every.max(1),
            since: usize::MAX, // the first row always snapshots
            ring: VecDeque::new(),
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

    /// Offer a snapshot of the state *as it stands before* the row at
    /// `clock`, already decayed to that row. Taken only when the cadence is
    /// due, so `make` is not called otherwise.
    pub fn offer(&mut self, clock: f64, make: impl FnOnce() -> S) {
        self.since = self.since.saturating_add(1);
        if self.since >= self.every {
            self.since = 0;
            self.ring.push_back((clock, make()));
        }
    }

    /// Drop what can never be the boundary again: everything strictly older
    /// than `now - window`. What is left at the front is the oldest snapshot
    /// inside the window, which is the boundary.
    pub fn trim(&mut self, now: f64) {
        let oldest = now - self.window;
        while self.ring.len() > 1 && self.ring[0].0 < oldest {
            self.ring.pop_front();
        }
        // A single entry older than the window still bounds the answer: it
        // says the whole accumulator is out of date, and the model reports
        // an empty window rather than stale numbers.
        if self.ring.len() == 1 && self.ring[0].0 < oldest {
            // Keep it. `boundary` is what decides, and it will subtract the
            // whole accumulator, which is the truthful "nothing in the
            // window" answer.
        }
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

/// `cov` with everything at or before the snapshot's clock removed:
/// `A(t) - f·A(u)`, in the mean form the accumulator stores.
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
