//! Lagged pair moments for [`crate::Marginal`] (docs/ENHANCEMENTS.md E66,
//! `docs/MARGINAL-LAGS-AND-BINS.md`).
//!
//! `marginal` reports `t = corr·sqrt((n_kish − 2)/(1 − corr²))`. `n_kish` is
//! the right count for unequal *weights* and says nothing about serial
//! dependence: on a smooth stream consecutive rows are nearly the same
//! observation, and the variance of a sample correlation is not `1/n` but,
//! to first order (Bartlett 1935),
//!
//! ```text
//! Var(r) ≈ (1/n)·[ 1 + 2·Σ_{ℓ≥1} ρ_x(ℓ)·ρ_y(ℓ) ]
//! ```
//!
//! so the count that makes `Var(r) = 1/n` true is `n_kish` divided by that
//! bracket. This accumulator is what the bracket needs: the two series'
//! autocorrelations at a few lags, and — as a by-product worth having — the
//! cross-correlations in both orientations, which say whether a feature
//! leads the target or follows it.
//!
//! # Shape
//!
//! [`crate::EwLagCov`] answers the same question densely, `k×k` per lag, and
//! that is exactly what `marginal` exists to avoid: at `p = 10,000` a dense
//! lag matrix is 100M doubles *per lag*. So this keeps `marginal`'s sparse
//! shape — per lag, one autocovariance per target, and per pair a feature
//! autocovariance and both cross-covariances:
//!
//! ```text
//! cyy[ℓ][t]      E_w[dy_t · dy_{t−ℓ}]          T   per lag
//! cxx[ℓ][t][j]   E_w[dx_t · dx_{t−ℓ}]          p·T per lag
//! cxy[ℓ][t][j]   E_w[dx_t · dy_{t−ℓ}]          p·T per lag   (x now, y back)
//! cyx[ℓ][t][j]   E_w[dy_t · dx_{t−ℓ}]          p·T per lag   (y now, x back)
//! ```
//!
//! # The recursion, and why it is [`crate::EwLagCov`]'s
//!
//! Per learned row, with `W` the target's weight *before* the row and `m` the
//! means before it — the same operands [`crate::Marginal`]'s own update uses,
//! in the same expressions, so lag 0 would agree with the contemporaneous
//! co-moments to the bit:
//!
//! ```text
//! a = lam·W/W'      b = w/W'
//! d_now = v_t − m       d_lag = v_{t−ℓ} − m       (both against the OLD mean)
//! C_ℓ' = a·C_ℓ + a·b·d_now·d_lag     when row t−ℓ is in the ring
//! C_ℓ' = a·C_ℓ                       when it is not yet
//! ```
//!
//! # What a lag counts
//!
//! **Learned rows within the group**, not rows where a particular target was
//! present. The ring is shared across targets, so with several targets that
//! appear on different rows, `ℓ = 1` means "the previous learned row",
//! whichever targets that row happened to carry. For the usual one-target
//! spec the distinction does not arise; for a sparsely present target it
//! means the lag is a *row* distance and not an observation distance, which
//! is the honest thing to say in the docs and the reason `n_serial` is
//! documented as a correction rather than an exact count.
//!
//! A zero-weight row leaves the moments where they are -- the pair update's
//! own mix on such a row is `a = 1, b = 0` -- and is not pushed: it taught
//! nothing, so it is not something a later row can be `ℓ` rows after. A row
//! where a target is absent *is* pushed, since the ring is shared, and that
//! target's moments hold, as its pair moments do.
//!
//! # Why a missing target holds rather than decays
//!
//! These are normalized moments, `E_w[·]`, and decay reaches them only
//! through `a = lam·W/W'` on the rows that learn: `W` carries the ageing,
//! and `E_w` of a history that has merely aged is the same number. Ageing
//! them by `lam` on their own on a row where the target is absent -- which
//! is how the first version of this file read -- takes an `E_w` toward zero
//! by a factor of `lam` per missing row, and nothing ever puts it back: on a
//! stream where a target is absent one row in `k`, every lagged
//! autocorrelation was low by about `1/k`, and after a hundred missing rows
//! at a halflife of twenty, by `2⁻⁵`. The pair moments held all along
//! (`W_t·lam`, `Q_t·lam²`, and the means and centred moments untouched), and
//! `n_serial` divides one by the other, so the two families must age the
//! same way. `tests/test_marginal_lags.py` holds a lag autocorrelation
//! across a run of null targets.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Lagged moments beside a [`crate::Marginal`]'s pair moments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalLags {
    p: usize,
    t: usize,
    /// Strictly increasing, all `>= 1`.
    lags: Vec<usize>,
    /// `[lag][target]`
    cyy: Vec<Vec<f64>>,
    /// `[lag][target*p + feature]`
    cxx: Vec<Vec<f64>>,
    cxy: Vec<Vec<f64>>,
    cyx: Vec<Vec<f64>>,
    /// The last `max(lags)` learned rows, oldest first: the features, and
    /// each target's value where it was present.
    ring_x: VecDeque<Vec<f64>>,
    ring_y: VecDeque<Vec<Option<f64>>>,
}

/// What a lagged update must borrow from the pair update it accompanies, so
/// the two centre and mix identically: the target's feature means `mx` and
/// target mean `my` as they stand *before* this row, and the mixing weights
/// `a` and `b` the pair update is about to use.
#[derive(Debug, Clone, Copy)]
pub struct PairMix<'a> {
    pub mx: &'a [f64],
    pub my: f64,
    pub a: f64,
    pub b: f64,
}

impl MarginalLags {
    pub fn new(p: usize, t: usize, lags: Vec<usize>) -> Result<Self, String> {
        if lags.is_empty() {
            return Err("marginal: lags must not be empty".into());
        }
        if lags[0] < 1 {
            return Err("marginal: lags must be >= 1 (lag 0 is the pair itself)".into());
        }
        if lags.windows(2).any(|w| w[1] <= w[0]) {
            return Err("marginal: lags must be strictly increasing".into());
        }
        let l = lags.len();
        let max = *lags.last().expect("lags is non-empty");
        Ok(Self {
            p,
            t,
            lags,
            cyy: vec![vec![0.0; t]; l],
            cxx: vec![vec![0.0; p * t]; l],
            cxy: vec![vec![0.0; p * t]; l],
            cyx: vec![vec![0.0; p * t]; l],
            ring_x: VecDeque::with_capacity(max),
            ring_y: VecDeque::with_capacity(max),
        })
    }

    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// `E_w[dy_t·dy_{t−ℓ}]` for target `t`, at the `li`-th configured lag.
    pub fn cyy(&self, li: usize, t: usize) -> f64 {
        self.cyy[li][t]
    }

    /// `E_w[dx_t·dx_{t−ℓ}]` for the pair.
    pub fn cxx(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cxx[li][t * self.p + j]
    }

    /// `E_w[dx_t·dy_{t−ℓ}]`: the feature now against the target `ℓ` rows ago.
    pub fn cxy(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cxy[li][t * self.p + j]
    }

    /// `E_w[dy_t·dx_{t−ℓ}]`: the target now against the feature `ℓ` rows ago.
    pub fn cyx(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cyx[li][t * self.p + j]
    }

    /// Empty the ring, keeping the moments: a session change or a clock gap
    /// beyond `max_dclock` means the next row is not `1` after the last one.
    pub fn clear(&mut self) {
        self.ring_x.clear();
        self.ring_y.clear();
    }

    /// One target's lagged moments, before its pair moments advance.
    pub fn update_target(&mut self, t: usize, x: &[f64], yt: f64, mix: PairMix<'_>) {
        let PairMix { mx, my, a, b } = mix;
        let depth = self.ring_x.len();
        let dy_now = yt - my;
        for (li, &lag) in self.lags.iter().enumerate() {
            let row = t * self.p;
            if lag > depth {
                // Nothing that far back yet: the moments age and wait.
                self.cyy[li][t] *= a;
                for i in row..row + self.p {
                    self.cxx[li][i] *= a;
                    self.cxy[li][i] *= a;
                    self.cyx[li][i] *= a;
                }
                continue;
            }
            let back = depth - lag;
            let x_lag = &self.ring_x[back];
            let y_lag = self.ring_y[back][t];
            for (j, (xj, mxj)) in x.iter().zip(mx).enumerate() {
                let i = row + j;
                let dx_now = xj - mxj;
                let dx_lag = x_lag[j] - mxj;
                self.cxx[li][i] = a * self.cxx[li][i] + a * b * dx_now * dx_lag;
                self.cyx[li][i] = a * self.cyx[li][i] + a * b * dy_now * dx_lag;
                // The two that need the target `lag` rows ago: a row where it
                // was absent contributes nothing but the decay.
                match y_lag {
                    Some(v) => {
                        let dy_lag = v - my;
                        self.cxy[li][i] = a * self.cxy[li][i] + a * b * dx_now * dy_lag;
                    }
                    None => self.cxy[li][i] *= a,
                }
            }
            match y_lag {
                Some(v) => {
                    let dy_lag = v - my;
                    self.cyy[li][t] = a * self.cyy[li][t] + a * b * dy_now * dy_lag;
                }
                None => self.cyy[li][t] *= a,
            }
        }
    }

    /// Push a learned row, dropping what has fallen off the deepest lag. The
    /// dropped row's buffers are reused, so a full ring allocates nothing
    /// per row.
    pub fn push(&mut self, x: &[f64], y: &[Option<f64>]) {
        debug_assert!(self.t == y.len());
        let max_lag = *self.lags.last().expect("lags is non-empty");
        let (mut bx, mut by) = if self.ring_x.len() >= max_lag {
            let bx = self.ring_x.pop_front().expect("non-empty");
            let by = self.ring_y.pop_front().expect("non-empty");
            (bx, by)
        } else {
            (Vec::with_capacity(x.len()), Vec::with_capacity(y.len()))
        };
        bx.clear();
        bx.extend_from_slice(x);
        by.clear();
        by.extend_from_slice(y);
        self.ring_x.push_back(bx);
        self.ring_y.push_back(by);
        debug_assert_eq!(self.ring_x.len(), self.ring_y.len());
    }
}
