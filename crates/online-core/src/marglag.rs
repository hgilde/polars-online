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
//! cxy[ℓ][t][j]   E_w[dx_t · dy_{t−ℓ}]          p·T per cross lag   (x now, y back)
//! cyx[ℓ][t][j]   E_w[dy_t · dx_{t−ℓ}]          p·T per cross lag   (y now, x back)
//! ```
//!
//! The cross lags are every lag unless `cross_lags` names fewer (E70,
//! docs/PLAN.md task 123). `n_serial` reads the two autocorrelations alone,
//! so the cross terms are the lead/lag by-product: worth having at the
//! first lag or two, and two of the three lagged moments at every lag kept,
//! the cheaper part of each lag's work (docs/PERFORMANCE.md §23).
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
    /// `[cross lag][target*p + feature]`: one per entry of
    /// [`Self::cross_lags`], which is every lag unless `cross_lags` names
    /// fewer.
    cxy: Vec<Vec<f64>>,
    cyx: Vec<Vec<f64>>,
    /// The last `max(lags)` learned rows, oldest first: the features, and
    /// each target's value where it was present.
    ring_x: VecDeque<Vec<f64>>,
    ring_y: VecDeque<Vec<Option<f64>>>,
    /// The lags the cross moments are kept at, a subsequence of `lags`;
    /// `None` is every lag, which is what a state written before E70 holds
    /// (docs/PLAN.md task 123). Last, and not skipped, so both encodings
    /// read a state without it (`tests/state_encoding.rs`).
    #[serde(default)]
    cross_lags: Option<Vec<usize>>,
}

/// What a lagged update must borrow from the pair update it accompanies, so
/// the two centre and mix identically: the target's feature means `mx` and
/// target mean `my` as they stand *before* this row, each with what its
/// double leaves out (`mx_lo`, `my_lo`: the means are pairs,
/// [`crate::comp`]), and the mixing weights `a` and `b` the pair update is
/// about to use.
#[derive(Debug, Clone, Copy)]
pub struct PairMix<'a> {
    pub mx: &'a [f64],
    pub mx_lo: &'a [f64],
    pub my: f64,
    pub my_lo: f64,
    pub a: f64,
    pub b: f64,
}

impl MarginalLags {
    /// `cross_lags` is `None` for a cross moment at every lag, or the lags
    /// to keep them at: strictly increasing, each one of `lags`, and empty
    /// for none.
    pub fn new(
        p: usize,
        t: usize,
        lags: Vec<usize>,
        cross_lags: Option<Vec<usize>>,
    ) -> Result<Self, String> {
        if lags.is_empty() {
            return Err("marginal: lags must not be empty".into());
        }
        if lags[0] < 1 {
            return Err("marginal: lags must be >= 1 (lag 0 is the pair itself)".into());
        }
        if lags.windows(2).any(|w| w[1] <= w[0]) {
            return Err("marginal: lags must be strictly increasing".into());
        }
        if let Some(c) = cross_lags.as_ref() {
            if c.windows(2).any(|w| w[1] <= w[0]) {
                return Err("marginal: cross_lags must be strictly increasing".into());
            }
            if let Some(bad) = c.iter().find(|l| !lags.contains(l)) {
                return Err(format!(
                    "marginal: cross_lags must each be one of lags, and {bad} is not: a cross \
                     term reads the ring `lags` keeps"
                ));
            }
        }
        let l = lags.len();
        let n_cross = cross_lags.as_ref().map_or(l, Vec::len);
        let max = *lags.last().expect("lags is non-empty");
        Ok(Self {
            p,
            t,
            lags,
            cyy: vec![vec![0.0; t]; l],
            cxx: vec![vec![0.0; p * t]; l],
            cxy: vec![vec![0.0; p * t]; n_cross],
            cyx: vec![vec![0.0; p * t]; n_cross],
            ring_x: VecDeque::with_capacity(max),
            ring_y: VecDeque::with_capacity(max),
            cross_lags,
        })
    }

    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// The lags the cross moments are kept at, in order: `cxy` and `cyx`
    /// are indexed by position here.
    pub fn cross_lags(&self) -> &[usize] {
        self.cross_lags.as_deref().unwrap_or(&self.lags)
    }

    /// Whether every matrix and both rings are those of `p` features, `t`
    /// targets, `lags` and `cross_lags`: what a restored state must hold to
    /// be updated (review 2026-09-18, B3).
    pub fn has_shape(
        &self,
        p: usize,
        t: usize,
        lags: &[usize],
        cross_lags: Option<&[usize]>,
    ) -> bool {
        let l = lags.len();
        let n_cross = cross_lags.map_or(l, <[usize]>::len);
        self.p == p
            && self.t == t
            && self.lags.as_slice() == lags
            && self.cross_lags.as_deref() == cross_lags
            && self.cyy.len() == l
            && self.cyy.iter().all(|v| v.len() == t)
            && self.cxx.len() == l
            && [&self.cxy, &self.cyx].iter().all(|m| m.len() == n_cross)
            && [&self.cxx, &self.cxy, &self.cyx]
                .iter()
                .all(|m| m.iter().all(|v| v.len() == p * t))
            && self.ring_x.iter().all(|r| r.len() == p)
            && self.ring_y.iter().all(|r| r.len() == t)
            // As many target rows as feature rows, and no more than the
            // deepest lag reads: a deeper ring never shrinks, and reads
            // every lag a row too recent (review 2026-09-26, B3).
            && self.ring_x.len() == self.ring_y.len()
            && self.ring_x.len() <= lags.last().copied().unwrap_or(0)
    }

    /// `E_w[dy_t·dy_{t−ℓ}]` for target `t`, at the `li`-th configured lag.
    pub fn cyy(&self, li: usize, t: usize) -> f64 {
        self.cyy[li][t]
    }

    /// `E_w[dx_t·dx_{t−ℓ}]` for the pair.
    pub fn cxx(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cxx[li][t * self.p + j]
    }

    /// `E_w[dx_t·dy_{t−ℓ}]`: the feature now against the target `ℓ` rows
    /// ago, at the `ci`-th of [`Self::cross_lags`].
    pub fn cxy(&self, ci: usize, t: usize, j: usize) -> f64 {
        self.cxy[ci][t * self.p + j]
    }

    /// `E_w[dy_t·dx_{t−ℓ}]`: the target now against the feature `ℓ` rows
    /// ago, at the `ci`-th of [`Self::cross_lags`].
    pub fn cyx(&self, ci: usize, t: usize, j: usize) -> f64 {
        self.cyx[ci][t * self.p + j]
    }

    /// Empty the ring, keeping the moments: a session change or a clock gap
    /// beyond `max_dclock` means the next row is not `1` after the last one.
    pub fn clear(&mut self) {
        self.ring_x.clear();
        self.ring_y.clear();
    }

    /// One target's lagged moments, before its pair moments advance.
    pub fn update_target(&mut self, t: usize, x: &[f64], yt: f64, mix: PairMix<'_>) {
        if self.cross_lags.is_none() {
            self.update_every_lag(t, x, yt, mix);
        } else {
            self.update_some_lags(t, x, yt, mix);
        }
    }

    /// [`Self::update_target`] with the cross terms at every lag: the loop
    /// as it was before `cross_lags` existed (E70). One loop for both cases
    /// measured 1% slower on the default (docs/PERFORMANCE.md §23), so the
    /// default keeps its own, and `cross_lags_keep_the_default_cross_terms_at_their_lags`
    /// holds the two loops to the same bits with every lag named.
    fn update_every_lag(&mut self, t: usize, x: &[f64], yt: f64, mix: PairMix<'_>) {
        let PairMix {
            mx,
            mx_lo,
            my,
            my_lo,
            a,
            b,
        } = mix;
        use crate::comp::dev;
        let depth = self.ring_x.len();
        let dy_now = dev(yt, my, my_lo);
        let r = t * self.p..(t + 1) * self.p;
        debug_assert_eq!(
            mx.len(),
            mx_lo.len(),
            "the means' low parts are sized first"
        );
        for (li, &lag) in self.lags.iter().enumerate() {
            if lag > depth {
                // Nothing that far back yet: the moments age and wait.
                self.cyy[li][t] *= a;
                wait(a, &mut self.cxx[li][r.clone()]);
                wait(a, &mut self.cxy[li][r.clone()]);
                wait(a, &mut self.cyx[li][r.clone()]);
                continue;
            }
            let back = depth - lag;
            // The target `lag` rows ago, against its mean now: a row where
            // it was absent contributes nothing but the decay.
            let dy_lag = self.ring_y[back][t].map(|v| dev(v, my, my_lo));
            step_all(
                Lagged {
                    a,
                    b,
                    dy_now,
                    dy_lag,
                    x,
                    x_lag: &self.ring_x[back],
                    mx,
                    mx_lo,
                },
                &mut self.cxx[li][r.clone()],
                &mut self.cxy[li][r.clone()],
                &mut self.cyx[li][r.clone()],
            );
            self.cyy[li][t] = step_cyy(a, b, dy_now, dy_lag, self.cyy[li][t]);
        }
    }

    /// [`Self::update_target`] under `cross_lags`: the cross terms only at
    /// the lags it names, each moment in the expression the loop above
    /// gives it. Never inlined, so that the default path's caller stays the
    /// size it was (docs/PERFORMANCE.md §23).
    #[inline(never)]
    fn update_some_lags(&mut self, t: usize, x: &[f64], yt: f64, mix: PairMix<'_>) {
        let PairMix {
            mx,
            mx_lo,
            my,
            my_lo,
            a,
            b,
        } = mix;
        use crate::comp::dev;
        let depth = self.ring_x.len();
        let dy_now = dev(yt, my, my_lo);
        let r = t * self.p..(t + 1) * self.p;
        debug_assert_eq!(
            mx.len(),
            mx_lo.len(),
            "the means' low parts are sized first"
        );
        // The next cross lag to meet, walking `lags` in order: `cross_lags`
        // is a subsequence of it.
        let cross = self.cross_lags.as_deref().unwrap_or(&[]);
        let mut next_cross = 0;
        for (li, &lag) in self.lags.iter().enumerate() {
            let ci = (cross.get(next_cross) == Some(&lag)).then(|| {
                next_cross += 1;
                next_cross - 1
            });
            if lag > depth {
                // Nothing that far back yet: the moments age and wait.
                self.cyy[li][t] *= a;
                wait(a, &mut self.cxx[li][r.clone()]);
                if let Some(ci) = ci {
                    wait(a, &mut self.cxy[ci][r.clone()]);
                    wait(a, &mut self.cyx[ci][r.clone()]);
                }
                continue;
            }
            let back = depth - lag;
            let dy_lag = self.ring_y[back][t].map(|v| dev(v, my, my_lo));
            let lagged = Lagged {
                a,
                b,
                dy_now,
                dy_lag,
                x,
                x_lag: &self.ring_x[back],
                mx,
                mx_lo,
            };
            match ci {
                Some(ci) => step_all(
                    lagged,
                    &mut self.cxx[li][r.clone()],
                    &mut self.cxy[ci][r.clone()],
                    &mut self.cyx[ci][r.clone()],
                ),
                // No cross terms at this lag: the feature's autocovariance
                // alone.
                None => step_xx(lagged, &mut self.cxx[li][r.clone()]),
            }
            self.cyy[li][t] = step_cyy(a, b, dy_now, dy_lag, self.cyy[li][t]);
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

    /// The learned rows the ring holds.
    pub(crate) fn depth(&self) -> usize {
        self.ring_x.len()
    }

    /// The deepest lag, which is how many rows the ring keeps.
    pub(crate) fn max_lag(&self) -> usize {
        *self.lags.last().expect("lags is non-empty")
    }

    /// The ring's `i`-th row, oldest first: its features and targets.
    pub(crate) fn ring_row(&self, i: usize) -> (&[f64], &[Option<f64>]) {
        (&self.ring_x[i], &self.ring_y[i])
    }

    /// Keep the newest `n` rows of the ring and drop the rest: what a
    /// caller that held rows back does before pushing them, so the rows the
    /// ring already holds are not copied (review 2026-09-26, A8).
    pub(crate) fn keep_last(&mut self, n: usize) {
        while self.ring_x.len() > n {
            self.ring_x.pop_front();
            self.ring_y.pop_front();
        }
    }

    /// Target `t`'s own lagged moment at the `li`-th lag, which moves with
    /// the target alone and so is advanced where the target's scalars are.
    pub(crate) fn cyy_mut(&mut self, li: usize, t: usize) -> &mut f64 {
        &mut self.cyy[li][t]
    }

    /// Per lag, the position of its cross moments in `cxy` and `cyx`, or
    /// `None` where it keeps none.
    pub(crate) fn cross_of(&self) -> Vec<Option<usize>> {
        let cross = self.cross_lags();
        let mut next = 0;
        self.lags
            .iter()
            .map(|lag| {
                (cross.get(next) == Some(lag)).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    /// The three per-pair moment families, `[lag][t·p + j]`, for a caller
    /// that hands disjoint ranges of them to several writers, and the ring
    /// they read back from.
    pub(crate) fn parts(&mut self) -> LagParts<'_> {
        LagParts {
            cxx: &mut self.cxx,
            cxy: &mut self.cxy,
            cyx: &mut self.cyx,
            ring_x: &self.ring_x,
        }
    }
}

/// [`MarginalLags::parts`].
pub(crate) struct LagParts<'a> {
    pub cxx: &'a mut [Vec<f64>],
    pub cxy: &'a mut [Vec<f64>],
    pub cyx: &'a mut [Vec<f64>],
    pub ring_x: &'a VecDeque<Vec<f64>>,
}

/// What one target's lagged step reads over a range of features: its mix
/// (`a`, `b`), its deviation now and `lag` rows back (`None` where it was
/// absent then), and the features now and `lag` rows back with their means
/// before the row, all against the same old means ([`PairMix`]).
#[derive(Clone, Copy)]
pub(crate) struct Lagged<'a> {
    pub a: f64,
    pub b: f64,
    pub dy_now: f64,
    pub dy_lag: Option<f64>,
    pub x: &'a [f64],
    pub x_lag: &'a [f64],
    pub mx: &'a [f64],
    pub mx_lo: &'a [f64],
}

/// Nothing `lag` rows back yet: the moments age and wait.
#[inline]
pub(crate) fn wait(a: f64, c: &mut [f64]) {
    for v in c {
        *v *= a;
    }
}

/// The feature's autocovariance alone, where a lag keeps no cross terms.
#[inline]
pub(crate) fn step_xx(l: Lagged<'_>, cxx: &mut [f64]) {
    use crate::comp::dev;
    let (a, b) = (l.a, l.b);
    let means = l.mx.iter().zip(l.mx_lo);
    for (((&xj, &xl), (&m, &lo)), c) in l.x.iter().zip(l.x_lag).zip(means).zip(cxx) {
        let dx_now = dev(xj, m, lo);
        let dx_lag = dev(xl, m, lo);
        *c = a * *c + a * b * dx_now * dx_lag;
    }
}

/// All three moments at a lag: the feature's autocovariance, the target
/// now against the feature back (`cyx`), and the feature now against the
/// target back (`cxy`), which a row where the target was absent then ages
/// alone.
#[inline]
pub(crate) fn step_all(l: Lagged<'_>, cxx: &mut [f64], cxy: &mut [f64], cyx: &mut [f64]) {
    use crate::comp::dev;
    let (a, b, dy_now) = (l.a, l.b, l.dy_now);
    let means = l.mx.iter().zip(l.mx_lo);
    let cells = cxx.iter_mut().zip(cxy.iter_mut()).zip(cyx.iter_mut());
    let each = l.x.iter().zip(l.x_lag).zip(means).zip(cells);
    match l.dy_lag {
        Some(dy_lag) => {
            for (((&xj, &xl), (&m, &lo)), ((xx, xy), yx)) in each {
                let dx_now = dev(xj, m, lo);
                let dx_lag = dev(xl, m, lo);
                *xx = a * *xx + a * b * dx_now * dx_lag;
                *yx = a * *yx + a * b * dy_now * dx_lag;
                *xy = a * *xy + a * b * dx_now * dy_lag;
            }
        }
        None => {
            for (((&xj, &xl), (&m, &lo)), ((xx, xy), yx)) in each {
                let dx_now = dev(xj, m, lo);
                let dx_lag = dev(xl, m, lo);
                *xx = a * *xx + a * b * dx_now * dx_lag;
                *yx = a * *yx + a * b * dy_now * dx_lag;
                *xy *= a;
            }
        }
    }
}

/// The target's own lagged moment `c`, stepped: `dy_lag` is its deviation
/// `lag` rows back, `None` where it was absent then, which ages it alone.
#[inline]
pub(crate) fn step_cyy(a: f64, b: f64, dy_now: f64, dy_lag: Option<f64>, c: f64) -> f64 {
    match dy_lag {
        Some(d) => a * c + a * b * dy_now * d,
        None => c * a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A restored state must hold what its config asks for, or it is refused
    /// before the first row indexes past an end (review 2026-09-18, B3). One
    /// corruption per condition `has_shape` checks, each refused alone.
    #[test]
    fn a_lag_state_of_another_shape_is_refused() {
        let (p, t, lags) = (3, 2, vec![1usize, 2, 5]);
        let good = |cross: Option<Vec<usize>>| {
            let mut m = MarginalLags::new(p, t, lags.clone(), cross).unwrap();
            m.push(&[1.0, 2.0, 3.0], &[Some(1.0), None]);
            m
        };
        let every = good(None);
        assert!(every.has_shape(p, t, &lags, None));
        let one = good(Some(vec![2]));
        assert!(one.has_shape(p, t, &lags, Some(&[2])));

        let refused = |m: &MarginalLags, cross: Option<&[usize]>, what: &str| {
            assert!(!m.has_shape(p, t, &lags, cross), "{what}");
        };
        refused(
            &every,
            Some(&[2]),
            "the config names cross lags the state does not keep",
        );
        refused(
            &one,
            None,
            "the state keeps fewer cross lags than the config's every lag",
        );
        refused(&one, Some(&[5]), "a different cross lag");
        assert!(!every.has_shape(p + 1, t, &lags, None), "another width");
        assert!(!every.has_shape(p, t + 1, &lags, None), "more targets");
        assert!(!every.has_shape(p, t, &[1, 2], None), "other lags");
        let mut m = good(None);
        m.cyy.pop();
        refused(&m, None, "a lag short in cyy");
        let mut m = good(None);
        m.cyy[0].pop();
        refused(&m, None, "a target short in cyy");
        let mut m = good(None);
        m.cxx.pop();
        refused(&m, None, "a lag short in cxx");
        let mut m = good(Some(vec![2]));
        m.cxy.push(vec![0.0; p * t]);
        refused(&m, Some(&[2]), "a cross lag too many in cxy");
        let mut m = good(Some(vec![2]));
        m.cyx.clear();
        refused(&m, Some(&[2]), "no cross lag in cyx");
        let mut m = good(None);
        m.cxx[1].pop();
        refused(&m, None, "a pair short in cxx");
        let mut m = good(None);
        m.cyx[2].push(0.0);
        refused(&m, None, "a pair too many in cyx");
        let mut m = good(None);
        m.ring_x[0].pop();
        refused(&m, None, "a ring row short of a feature");
        let mut m = good(None);
        m.ring_y[0].push(None);
        refused(&m, None, "a ring row with a target too many");
    }

    /// A restored ring deeper than the deepest lag, or with fewer target
    /// rows than feature rows, is refused as the wrong shape: the first
    /// would read every lag a row too recent for good, the second would
    /// panic (review 2026-09-26, B3).
    #[test]
    fn a_lag_state_with_an_overlong_or_uneven_ring_is_refused() {
        let mut l = MarginalLags::new(2, 1, vec![1, 2], None).unwrap();
        for i in 0..3 {
            l.push(&[i as f64, 0.0], &[Some(1.0)]);
        }
        assert!(l.has_shape(2, 1, &[1, 2], None));
        l.ring_x.push_back(vec![9.0, 9.0]);
        l.ring_y.push_back(vec![None]);
        assert!(
            !l.has_shape(2, 1, &[1, 2], None),
            "three rows for a deepest lag of two"
        );
        l.ring_x.pop_back();
        assert!(
            !l.has_shape(2, 1, &[1, 2], None),
            "more target rows than feature rows"
        );
    }

    /// A state written before `cross_lags` existed is nine positional
    /// fields; the tenth reads as `None` from a compact array of nine,
    /// serde's rule for a defaulted trailing field, pinned here (review
    /// 2026-09-26, B missing 5).
    #[test]
    fn a_compact_state_without_cross_lags_reads_with_every_lag() {
        #[derive(Serialize)]
        struct Before {
            p: usize,
            t: usize,
            lags: Vec<usize>,
            cyy: Vec<Vec<f64>>,
            cxx: Vec<Vec<f64>>,
            cxy: Vec<Vec<f64>>,
            cyx: Vec<Vec<f64>>,
            ring_x: VecDeque<Vec<f64>>,
            ring_y: VecDeque<Vec<Option<f64>>>,
        }
        let before = Before {
            p: 1,
            t: 1,
            lags: vec![1],
            cyy: vec![vec![0.5]],
            cxx: vec![vec![0.25]],
            cxy: vec![vec![0.1]],
            cyx: vec![vec![0.2]],
            ring_x: VecDeque::from(vec![vec![1.0]]),
            ring_y: VecDeque::from(vec![vec![Some(2.0)]]),
        };
        let bytes = rmp_serde::to_vec(&before).unwrap();
        let l: MarginalLags = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(l.cross_lags, None);
        assert_eq!(l.cross_lags(), &[1]);
        assert!(l.has_shape(1, 1, &[1], None));
        assert_eq!(l.cxy(0, 0, 0), 0.1);
    }
}
