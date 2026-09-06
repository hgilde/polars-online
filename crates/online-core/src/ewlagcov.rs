//! `EwLagCov`: exponentially weighted **lagged** cross-moments,
//! `E_w[d_t d'_{t−ℓ}]`, beside an [`EwCov`] (docs/ENHANCEMENTS.md E56).
//!
//! The contemporaneous co-moments answer "how do these columns move
//! together *now*". A realised autocovariance, the lagged decay functions of
//! an Epps decomposition and the Bartlett standard error of a correlation
//! between autocorrelated series all need the same object one step further
//! out: how column `a` now moves with column `b` `ℓ` rows ago.
//!
//! # The recursion
//!
//! Per learned row, with `W` the accumulator's weight *before* the row and
//! `m` its mean before the row -- the same operands [`EwCov::update`] uses,
//! in the same expressions, so the same bits:
//!
//! ```text
//! W' = lam*W + w      a = lam*W/W'      b = w/W'
//! d_t = x_t − m       d_{t−ℓ} = x_{t−ℓ} − m        (both against the OLD mean)
//! C_ℓ' = a·C_ℓ + a·b·d_t·d'_{t−ℓ}       when x_{t−ℓ} is in the ring
//! C_ℓ' = a·C_ℓ                          when it is not yet
//! ```
//!
//! Centring **both** legs at the pre-row mean is the choice that makes
//! `ℓ = 0` coincide with [`EwCov`]'s own co-moments to the bit, and a unit
//! test says so. The alternative -- each leg against the mean as it stood
//! when that row arrived -- is a different statistic, and one that cannot be
//! computed from a ring of raw rows.
//!
//! # What is in the ring
//!
//! The last `max(lags)` rows that **entered the accumulator**: a skipped row
//! never gets here, and a zero-weight row is fed (so the lag matrices decay
//! with the accumulator) but is *not* pushed, since it taught nothing and is
//! not a row for a later one to be lagged against.
//!
//! Lags are counted in learned rows within the group, not in clock units,
//! because the recursion is row-based. [`EwLagCov::clear`] empties the ring
//! -- and only the ring -- on the two events after which "the row `ℓ` back"
//! stops meaning a row `ℓ` ago: a session change and a capped clock gap
//! (`docs/PLAN.md` task 47).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// EW lagged cross-moments for a list of lags; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwLagCov {
    k: usize,
    /// The lags, strictly increasing and `>= 1`. The list order is the
    /// output order.
    lags: Vec<usize>,
    /// The last `max(lags)` learned rows, oldest first.
    ring: VecDeque<Vec<f64>>,
    /// `L * k * k`, row-major within a lag: `c[li*k*k + i*k + j]` is
    /// `E_w[d_i(t)·d_j(t−lags[li])]`.
    c: Vec<f64>,
}

impl EwLagCov {
    /// `Err` naming the problem: an empty list, a zero lag, or a list that
    /// is not strictly increasing (the order is the output order, so it is
    /// not sorted in silence).
    pub fn new(k: usize, lags: Vec<usize>) -> Result<Self, String> {
        if lags.is_empty() {
            return Err("lags must be non-empty".into());
        }
        if lags.contains(&0) {
            return Err(
                "lags must be >= 1; lag 0 is the contemporaneous co-moment matrix, which is \
                 already there as `comoments`"
                    .into(),
            );
        }
        if lags.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!(
                "lags must be strictly increasing (got {lags:?}); the list order is the output \
                 order, so it is not sorted for you"
            ));
        }
        let l = lags.len();
        let max = *lags.iter().max().expect("non-empty");
        Ok(Self {
            k,
            lags,
            ring: VecDeque::with_capacity(max),
            c: vec![0.0; l * k * k],
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// The lags, in output order.
    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// The matrices, `L * k * k` row-major within a lag.
    pub fn comoments(&self) -> &[f64] {
        &self.c
    }

    /// `E_w[d_a(t)·d_b(t − lags[li])]`.
    #[inline]
    pub fn get(&self, li: usize, a: usize, b: usize) -> f64 {
        self.c[li * self.k * self.k + a * self.k + b]
    }

    /// How many learned rows are in the ring; a lag beyond it has not been
    /// seen since the last clear and contributes nothing to its matrix.
    pub fn depth(&self) -> usize {
        self.ring.len()
    }

    /// Drop the ring, and only the ring (`OnlineModel::clear_lags`). The
    /// matrices are decayed statistics and go on decaying; what is no longer
    /// true is that the rows behind this one are `1, 2, ... ` rows *ago*.
    pub fn clear(&mut self) {
        self.ring.clear();
    }

    /// One learned row, with the accumulator's pre-row weight and mean.
    /// `lam` and `w` are [`EwCov::update`]'s, and the guards are the same: a
    /// negative weight is a caller's bug, and a row that leaves the total
    /// weight at zero changes nothing (hard rule 9).
    pub fn update(&mut self, x: &[f64], m: &[f64], w_sum: f64, lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert_eq!(m.len(), self.k);
        if w < 0.0 {
            return;
        }
        let w_new = lam * w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * w_sum / w_new;
        let b = w / w_new;
        let k = self.k;
        for (li, &lag) in self.lags.iter().enumerate() {
            let block = &mut self.c[li * k * k..(li + 1) * k * k];
            match self.ring.len().checked_sub(lag).map(|i| &self.ring[i]) {
                Some(past) => {
                    for i in 0..k {
                        let ab_di = a * b * (x[i] - m[i]);
                        let row = &mut block[i * k..(i + 1) * k];
                        for (cj, (&pj, &mj)) in row.iter_mut().zip(past.iter().zip(m)) {
                            *cj = a * *cj + ab_di * (pj - mj);
                        }
                    }
                }
                // The lag is deeper than the ring: nothing to pair this row
                // with, so the matrix only ages ([`EwAutoCorr`]'s rule).
                None => block.iter_mut().for_each(|c| *c *= a),
            }
        }
        // A zero-weight row aged the matrices above and teaches nothing, so
        // it is not a row for a later one to be lagged against.
        if w > 0.0 {
            // The depth is `max(lags)`, read from the lags themselves and
            // never from `ring.capacity()`: `VecDeque::clone` (which
            // `state()` goes through) and deserialization both allocate
            // exactly `len`, so a ring saved while it was short would keep
            // that length for ever and the deeper lags would only decay.
            let max_lag = *self.lags.last().expect("non-empty, strictly increasing");
            if self.ring.len() >= max_lag {
                self.ring.pop_front();
            }
            self.ring.push_back(x.to_vec());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EwCov;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn rows(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..n)
            .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
            .collect()
    }

    /// Lag 0 -- the row paired with itself -- reproduces `EwCov`'s own
    /// co-moments **to the bit**. That is the guard on "both legs against
    /// the pre-row mean": any other centring gives a different matrix here.
    ///
    /// `new` refuses lag 0, so the ring is primed with the current row and
    /// lag 1 reads it back, which is the same arithmetic.
    #[test]
    fn lag_zero_is_the_contemporaneous_comoments() {
        let k = 3;
        let mut cov = EwCov::new(k);
        let mut lag = EwLagCov::new(k, vec![1]).unwrap();
        for (i, x) in rows(200, k, 5).iter().enumerate() {
            let (lam, w) = (
                0.97,
                if i % 7 == 0 {
                    0.0
                } else {
                    1.0 + (i % 3) as f64
                },
            );
            lag.ring.clear();
            lag.ring.push_back(x.clone());
            lag.update(x, cov.means(), cov.n_eff(), lam, w);
            cov.update(x, lam, w);
        }
        assert_eq!(lag.comoments(), cov.comoments());
    }

    /// The recursion against a direct computation with explicit weights.
    #[test]
    fn the_recursion_is_the_weighted_lagged_cross_moment() {
        let (k, n, lag) = (2usize, 120usize, 3usize);
        let xs = rows(n, k, 11);
        let lam = 0.95;
        let mut cov = EwCov::new(k);
        let mut lc = EwLagCov::new(k, vec![lag]).unwrap();
        // Longhand: the same recursion, written without the ring.
        let mut direct = vec![0.0; k * k];
        for (t, x) in xs.iter().enumerate() {
            let w = 1.0;
            let (w_sum, m) = (cov.n_eff(), cov.means().to_vec());
            let w_new = lam * w_sum + w;
            let (a, b) = (lam * w_sum / w_new, w / w_new);
            if t >= lag {
                for i in 0..k {
                    for j in 0..k {
                        direct[i * k + j] =
                            a * direct[i * k + j] + a * b * (x[i] - m[i]) * (xs[t - lag][j] - m[j]);
                    }
                }
            } else {
                direct.iter_mut().for_each(|c| *c *= a);
            }
            lc.update(x, &m, w_sum, lam, w);
            cov.update(x, lam, w);
        }
        assert_eq!(lc.comoments(), &direct[..]);
    }

    #[test]
    fn a_cleared_ring_starts_the_pairing_over_and_keeps_the_matrices() {
        let k = 2;
        let mut cov = EwCov::new(k);
        let mut lc = EwLagCov::new(k, vec![1, 2]).unwrap();
        for x in rows(40, k, 3) {
            lc.update(&x, cov.means(), cov.n_eff(), 0.98, 1.0);
            cov.update(&x, 0.98, 1.0);
        }
        let before = lc.comoments().to_vec();
        assert_eq!(lc.depth(), 2);
        lc.clear();
        assert_eq!(lc.depth(), 0);
        assert_eq!(lc.comoments(), &before[..], "clear() is not a reset");
    }

    #[test]
    fn a_zero_weight_row_ages_the_matrices_and_stays_out_of_the_ring() {
        let k = 2;
        let mut cov = EwCov::new(k);
        let mut lc = EwLagCov::new(k, vec![1]).unwrap();
        let xs = rows(5, k, 9);
        for x in &xs[..3] {
            lc.update(x, cov.means(), cov.n_eff(), 0.9, 1.0);
            cov.update(x, 0.9, 1.0);
        }
        let depth = lc.depth();
        let before = lc.comoments().to_vec();
        lc.update(&[1e6, -1e6], cov.means(), cov.n_eff(), 0.9, 0.0);
        cov.update(&[1e6, -1e6], 0.9, 0.0);
        assert_eq!(lc.depth(), depth, "a zero-weight row is not a lagged row");
        // `a = 1` at `w = 0` in mean form, so nothing moves either.
        assert_eq!(lc.comoments(), &before[..]);
    }

    #[test]
    fn a_zero_weight_first_row_is_legal() {
        let mut lc = EwLagCov::new(2, vec![1]).unwrap();
        lc.update(&[1.0, 2.0], &[0.0, 0.0], 0.0, 0.9, 0.0);
        assert_eq!(lc.depth(), 0);
        assert!(lc.comoments().iter().all(|c| *c == 0.0));
    }

    #[test]
    fn the_ring_never_grows_past_the_largest_lag() {
        let mut lc = EwLagCov::new(2, vec![1, 4]).unwrap();
        let mut cov = EwCov::new(2);
        for x in rows(50, 2, 1) {
            lc.update(&x, cov.means(), cov.n_eff(), 0.99, 1.0);
            cov.update(&x, 0.99, 1.0);
            assert!(lc.depth() <= 4);
        }
        assert_eq!(lc.depth(), 4);
    }

    /// The depth is `max(lags)`, not whatever the ring happens to have
    /// allocated. `VecDeque::clone` and `serde` both hand back a deque with
    /// capacity `len`, so a ring copied while it was short used to be stuck
    /// at that length: the deeper lags then only decayed, for ever. Copy at
    /// every depth from empty to full and check the copy goes on filling.
    #[test]
    fn a_ring_copied_while_short_goes_on_filling() {
        let all = rows(12, 2, 7);
        for cut in 0..=5usize {
            let (mut lc, mut cov) = (EwLagCov::new(2, vec![1, 3]).unwrap(), EwCov::new(2));
            for x in &all[..cut] {
                lc.update(x, cov.means(), cov.n_eff(), 0.99, 1.0);
                cov.update(x, 0.99, 1.0);
            }
            // Both a plain clone and a msgpack round-trip: `state()` goes
            // through the first, a state file through the second.
            let bytes = rmp_serde::to_vec(&lc).unwrap();
            for copy in [lc.clone(), rmp_serde::from_slice(&bytes).unwrap()] {
                let (mut copy, mut ccov) = (copy, cov.clone());
                let (mut lc, mut cov) = (lc.clone(), cov.clone());
                for x in &all[cut..] {
                    copy.update(x, ccov.means(), ccov.n_eff(), 0.99, 1.0);
                    ccov.update(x, 0.99, 1.0);
                    lc.update(x, cov.means(), cov.n_eff(), 0.99, 1.0);
                    cov.update(x, 0.99, 1.0);
                }
                assert_eq!(copy.depth(), 3, "cut {cut}: the copy's ring stayed short");
                assert_eq!(
                    copy.comoments(),
                    lc.comoments(),
                    "cut {cut}: the copy drifted from the run it came from"
                );
            }
        }
    }

    #[test]
    fn a_bad_lag_list_is_refused_by_name() {
        for (lags, msg) in [
            (vec![], "must be non-empty"),
            (vec![0, 1], "must be >= 1"),
            (vec![2, 1], "strictly increasing"),
            (vec![1, 1], "strictly increasing"),
        ] {
            let e = EwLagCov::new(2, lags).unwrap_err();
            assert!(e.contains(msg), "{e}");
        }
    }
}
