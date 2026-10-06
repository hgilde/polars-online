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
//! at a half-life of twenty, by `2⁻⁵`. The pair moments held all along
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
    /// The feature's autocovariance kept once per feature, over every
    /// learned row, where it is kept per target otherwise
    /// ([`crate::FeatureMomentLayout::Shared`], docs/PLAN.md task 125): `cxx`
    /// is `p` a lag, not `p·T`. Last, and not skipped, as `cross_lags` is.
    #[serde(default)]
    shared: bool,
}

/// A target's side of a row's lagged moments under shared feature moments
/// ([`MarginalLags::update_shared`]): its mix `a = lam·W_t/W'_t` and `b =
/// w/W'_t`, its value on the row, and its mean before the row, with the
/// part the double leaves out.
#[derive(Debug, Clone, Copy)]
pub struct TargetLag {
    pub a: f64,
    pub b: f64,
    pub yt: f64,
    pub my: f64,
    pub my_lo: f64,
}

/// The lag moments alone, as a `window`'s snapshot holds them (docs/PLAN.md
/// task 137). They are normalized moments and do not decay; a window
/// truncates them in sum form with the target's weight, `W_t·C − f·W_u·C_u`
/// over the window's weight -- the increments made inside the window, each
/// centred at the mean as it stood, since a lagged moment has no re-centring
/// identity (review 2026-09-12, V12) -- beside the pair moments the same
/// snapshot holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LagMoments {
    p: usize,
    cyy: Vec<Vec<f64>>,
    cxx: Vec<Vec<f64>>,
    cxy: Vec<Vec<f64>>,
    cyx: Vec<Vec<f64>>,
}

impl LagMoments {
    pub fn cyy(&self, li: usize, t: usize) -> f64 {
        self.cyy[li][t]
    }

    pub fn cxx(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cxx[li][t * self.p + j]
    }

    pub fn cxy(&self, ci: usize, t: usize, j: usize) -> f64 {
        self.cxy[ci][t * self.p + j]
    }

    pub fn cyx(&self, ci: usize, t: usize, j: usize) -> f64 {
        self.cyx[ci][t * self.p + j]
    }

    /// Whether these are the moments of the lags `lags` holds: the same
    /// matrices at the same widths (review 2026-09-18, B3).
    pub fn matches(&self, lags: &MarginalLags) -> bool {
        let dims = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.len() == y.len())
        };
        self.p == lags.p
            && dims(&self.cyy, &lags.cyy)
            && dims(&self.cxx, &lags.cxx)
            && dims(&self.cxy, &lags.cxy)
            && dims(&self.cyx, &lags.cyx)
    }
}

impl crate::Footprint for LagMoments {
    fn footprint(&self) -> usize {
        [&self.cyy, &self.cxx, &self.cxy, &self.cyx]
            .iter()
            .flat_map(|m| m.iter())
            .map(|v| crate::window::floats(v))
            .sum::<usize>()
            + std::mem::size_of::<usize>()
    }
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
        shared: bool,
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
            cxx: vec![vec![0.0; if shared { p } else { p * t }]; l],
            cxy: vec![vec![0.0; p * t]; n_cross],
            cyx: vec![vec![0.0; p * t]; n_cross],
            ring_x: VecDeque::with_capacity(max),
            ring_y: VecDeque::with_capacity(max),
            cross_lags,
            shared,
        })
    }

    /// Whether the feature's autocovariance is kept once per feature.
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// The moments as they stand, for a `window`'s snapshot (task 137); the
    /// ring of raw rows is not in it, being the last `max(lags)` learned
    /// rows, inside any window.
    pub fn moments(&self) -> LagMoments {
        LagMoments {
            p: self.p,
            cyy: self.cyy.clone(),
            cxx: self.cxx.clone(),
            cxy: self.cxy.clone(),
            cyx: self.cyx.clone(),
        }
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
            && self
                .cxx
                .iter()
                .all(|v| v.len() == if self.shared { p } else { p * t })
            && [&self.cxy, &self.cyx]
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

    /// `E_w[dx_t·dx_{t−ℓ}]` for the pair: the feature's own under shared
    /// feature moments, whatever the target.
    pub fn cxx(&self, li: usize, t: usize, j: usize) -> f64 {
        self.cxx[li][if self.shared { j } else { t * self.p + j }]
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
    /// beyond `gap_cap` means the next row is not `1` after the last one.
    pub fn clear(&mut self) {
        self.ring_x.clear();
        self.ring_y.clear();
    }

    /// Every lagged moment of one learned row under shared feature moments
    /// (docs/PLAN.md task 125): per lag, the feature's autocovariance once,
    /// with the row's own mix `row` over the model's weight, and each
    /// present target's terms with its own, every deviation against the
    /// means before the row -- the shared feature means `mx`, each target's
    /// `my`. Element for element the expressions [`Self::update_target`]
    /// uses, so where every target is on every learned row the moments are
    /// its moments, to the bit.
    pub fn update_shared(
        &mut self,
        x: &[f64],
        row: Option<(f64, f64)>,
        targets: &[Option<TargetLag>],
        mx: &[f64],
        mx_lo: &[f64],
    ) {
        use crate::comp::dev;
        let (p, depth) = (self.p, self.ring_x.len());
        let cross_of = self.cross_of();
        for (li, &lag) in self.lags.iter().enumerate() {
            let ci = cross_of[li];
            if lag > depth {
                // Nothing that far back yet: the moments age and wait.
                if let Some((a, _)) = row {
                    wait(a, &mut self.cxx[li]);
                }
                for (t, m) in targets.iter().enumerate() {
                    let Some(m) = m else {
                        continue;
                    };
                    self.cyy[li][t] *= m.a;
                    if let Some(ci) = ci {
                        let r = t * p..(t + 1) * p;
                        wait(m.a, &mut self.cxy[ci][r.clone()]);
                        wait(m.a, &mut self.cyx[ci][r]);
                    }
                }
                continue;
            }
            let back = depth - lag;
            let x_lag = &self.ring_x[back];
            if let Some((a, b)) = row {
                let l = Lagged {
                    a,
                    b,
                    dy_now: 0.0,
                    dy_lag: None,
                    x,
                    x_lag,
                    mx,
                    mx_lo,
                };
                step_xx(l, &mut self.cxx[li]);
            }
            for (t, m) in targets.iter().enumerate() {
                let Some(m) = m else {
                    continue;
                };
                let dy_now = dev(m.yt, m.my, m.my_lo);
                // The target `lag` rows ago, against its mean now: a row
                // where it was absent contributes nothing but the decay.
                let dy_lag = self.ring_y[back][t].map(|v| dev(v, m.my, m.my_lo));
                if let Some(ci) = ci {
                    let r = t * p..(t + 1) * p;
                    let l = Lagged {
                        a: m.a,
                        b: m.b,
                        dy_now,
                        dy_lag,
                        x,
                        x_lag,
                        mx,
                        mx_lo,
                    };
                    let (cxy, cyx) = (&mut self.cxy[ci], &mut self.cyx[ci]);
                    step_cross(l, &mut cxy[r.clone()], &mut cyx[r]);
                }
                self.cyy[li][t] = step_cyy(m.a, m.b, dy_now, dy_lag, self.cyy[li][t]);
            }
        }
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

/// The cross terms at a lag alone, where the feature's autocovariance is
/// stepped once for every target ([`MarginalLags::update_shared`]): the
/// expressions [`step_all`] gives them.
#[inline]
pub(crate) fn step_cross(l: Lagged<'_>, cxy: &mut [f64], cyx: &mut [f64]) {
    use crate::comp::dev;
    let (a, b, dy_now) = (l.a, l.b, l.dy_now);
    let means = l.mx.iter().zip(l.mx_lo);
    let cells = cxy.iter_mut().zip(cyx.iter_mut());
    let each = l.x.iter().zip(l.x_lag).zip(means).zip(cells);
    match l.dy_lag {
        Some(dy_lag) => {
            for (((&xj, &xl), (&m, &lo)), (xy, yx)) in each {
                let dx_now = dev(xj, m, lo);
                let dx_lag = dev(xl, m, lo);
                *yx = a * *yx + a * b * dy_now * dx_lag;
                *xy = a * *xy + a * b * dx_now * dy_lag;
            }
        }
        None => {
            for (((_, &xl), (&m, &lo)), (xy, yx)) in each {
                let dx_lag = dev(xl, m, lo);
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
            let mut m = MarginalLags::new(p, t, lags.clone(), cross, false).unwrap();
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
        let mut l = MarginalLags::new(2, 1, vec![1, 2], None, false).unwrap();
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

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The module doc's recursion, written out longhand: per learned row and
    /// lag `ℓ`, `C' = a·C + a·b·d_now·d_lag` where row `t − ℓ` is in the ring
    /// and carried what `d_lag` reads, and `C' = a·C` where it is not, every
    /// deviation against the means before the row. Cross moments are kept at
    /// every lag here; a model keeping fewer is compared at the ones it keeps.
    struct Longhand {
        lags: Vec<usize>,
        /// `[lag][target]`
        cyy: Vec<Vec<f64>>,
        /// `[lag][target][feature]`, and `[lag][feature]` for the shared
        /// layout's feature autocovariance.
        cxx: Vec<Vec<Vec<f64>>>,
        cxx_shared: Vec<Vec<f64>>,
        cxy: Vec<Vec<Vec<f64>>>,
        cyx: Vec<Vec<Vec<f64>>>,
        ring: Vec<(Vec<f64>, Vec<Option<f64>>)>,
        /// Evidence the two branches the test is for ran: steps that met a
        /// target absent `ℓ` rows back, and steps with nothing `ℓ` rows back
        /// that aged a moment that was not zero.
        absent_back: usize,
        aged: usize,
    }

    impl Longhand {
        fn new(p: usize, t: usize, lags: &[usize]) -> Self {
            let l = lags.len();
            Self {
                lags: lags.to_vec(),
                cyy: vec![vec![0.0; t]; l],
                cxx: vec![vec![vec![0.0; p]; t]; l],
                cxx_shared: vec![vec![0.0; p]; l],
                cxy: vec![vec![vec![0.0; p]; t]; l],
                cyx: vec![vec![vec![0.0; p]; t]; l],
                ring: Vec::new(),
                absent_back: 0,
                aged: 0,
            }
        }

        /// Target `t`'s moments: its autocovariance and the cross terms, and
        /// the feature's autocovariance where `own_xx` (the per-target
        /// layout).
        #[allow(clippy::too_many_arguments)]
        fn target(
            &mut self,
            t: usize,
            x: &[f64],
            y: f64,
            mx: &[f64],
            my: f64,
            a: f64,
            b: f64,
            own_xx: bool,
        ) {
            for li in 0..self.lags.len() {
                let lag = self.lags[li];
                if lag > self.ring.len() {
                    let before = self.cyy[li][t];
                    self.cyy[li][t] = a * before;
                    for j in 0..x.len() {
                        if own_xx {
                            self.cxx[li][t][j] *= a;
                        }
                        self.cxy[li][t][j] *= a;
                        self.cyx[li][t][j] *= a;
                    }
                    if before != 0.0 {
                        self.aged += 1;
                    }
                    continue;
                }
                let (xb, yb) = &self.ring[self.ring.len() - lag];
                let dy_now = y - my;
                let dy_lag = yb[t].map(|v| v - my);
                for j in 0..x.len() {
                    let dx_now = x[j] - mx[j];
                    let dx_lag = xb[j] - mx[j];
                    if own_xx {
                        let c = self.cxx[li][t][j];
                        self.cxx[li][t][j] = a * c + a * b * dx_now * dx_lag;
                    }
                    let c = self.cyx[li][t][j];
                    self.cyx[li][t][j] = a * c + a * b * dy_now * dx_lag;
                    let c = self.cxy[li][t][j];
                    self.cxy[li][t][j] = match dy_lag {
                        Some(d) => a * c + a * b * dx_now * d,
                        None => a * c,
                    };
                }
                let c = self.cyy[li][t];
                self.cyy[li][t] = match dy_lag {
                    Some(d) => a * c + a * b * dy_now * d,
                    None => a * c,
                };
                if dy_lag.is_none() {
                    self.absent_back += 1;
                }
            }
        }

        /// The shared layout's feature autocovariance, with the row's mix.
        fn features(&mut self, x: &[f64], mx: &[f64], a: f64, b: f64) {
            for li in 0..self.lags.len() {
                let lag = self.lags[li];
                for j in 0..x.len() {
                    let c = self.cxx_shared[li][j];
                    self.cxx_shared[li][j] = if lag > self.ring.len() {
                        a * c
                    } else {
                        let xb = &self.ring[self.ring.len() - lag].0;
                        a * c + a * b * (x[j] - mx[j]) * (xb[j] - mx[j])
                    };
                }
            }
        }

        fn push(&mut self, x: &[f64], y: &[Option<f64>]) {
            self.ring.push((x.to_vec(), y.to_vec()));
            if self.ring.len() > *self.lags.last().unwrap() {
                self.ring.remove(0);
            }
        }
    }

    /// One stream of fourteen learned rows, two features and two targets,
    /// the second absent on every third row; the ring cleared after the
    /// eighth, as a break in the clock clears it. Per row: the features, the
    /// targets, each target's feature means, target mean and mix, and the
    /// row's own mix for the shared layout.
    type Row = (
        Vec<f64>,
        Vec<Option<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<(f64, f64)>,
        (f64, f64),
    );

    fn rows() -> Vec<Row> {
        let mut s = 17u64;
        (0..14)
            .map(|r| {
                let x = vec![lcg(&mut s), 2.0 * lcg(&mut s)];
                let y = vec![Some(lcg(&mut s)), (r % 3 != 1).then(|| lcg(&mut s))];
                let rf = r as f64;
                let mx = (0..2)
                    .map(|t| {
                        (0..2)
                            .map(|j| 0.03 * rf - 0.1 * t as f64 + 0.05 * j as f64)
                            .collect()
                    })
                    .collect();
                let my = (0..2).map(|t| 0.02 * rf + 0.1 * t as f64).collect();
                let mix = (0..2)
                    .map(|t| {
                        (
                            0.8 + 0.05 * t as f64 + 0.01 * (r % 4) as f64,
                            0.15 + 0.02 * t as f64,
                        )
                    })
                    .collect();
                (x, y, mx, my, mix, (0.85, 0.12 + 0.01 * (r % 3) as f64))
            })
            .collect()
    }

    const BREAK_AFTER: usize = 7;

    fn near(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-13 * (1.0 + want.abs()),
            "{what}: {got} vs {want}"
        );
    }

    /// The per-target layout's moments are the recursion, at every lag and
    /// with the cross terms at every lag or at some, through a target absent
    /// `ℓ` rows back and through the rows after a break, where nothing is
    /// `ℓ` rows back and the moments age and wait. A window's snapshot reads
    /// the same numbers as the model does.
    #[test]
    fn the_lag_moments_are_the_recursion_written_out() {
        let (p, t, lags) = (2usize, 2usize, vec![1usize, 3]);
        for cross in [None, Some(vec![3usize])] {
            let mut l = MarginalLags::new(p, t, lags.clone(), cross.clone(), false).unwrap();
            let mut o = Longhand::new(p, t, &lags);
            for (r, (x, y, mx, my, mix, _)) in rows().iter().enumerate() {
                for tt in 0..t {
                    let Some(yt) = y[tt] else { continue };
                    let (a, b) = mix[tt];
                    let zeros = [0.0; 2];
                    let pm = PairMix {
                        mx: &mx[tt],
                        mx_lo: &zeros,
                        my: my[tt],
                        my_lo: 0.0,
                        a,
                        b,
                    };
                    l.update_target(tt, x, yt, pm);
                    o.target(tt, x, yt, &mx[tt], my[tt], a, b, true);
                }
                l.push(x, y);
                o.push(x, y);
                if r == BREAK_AFTER {
                    l.clear();
                    o.ring.clear();
                }
            }
            assert!(
                o.absent_back > 0 && o.aged > 0,
                "{} {}",
                o.absent_back,
                o.aged
            );
            let snap = l.moments();
            for li in 0..lags.len() {
                for tt in 0..t {
                    near(l.cyy(li, tt), o.cyy[li][tt], "cyy");
                    near(snap.cyy(li, tt), o.cyy[li][tt], "the snapshot's cyy");
                    for j in 0..p {
                        near(l.cxx(li, tt, j), o.cxx[li][tt][j], "cxx");
                        near(snap.cxx(li, tt, j), o.cxx[li][tt][j], "the snapshot's cxx");
                    }
                }
            }
            for (ci, lag) in l.cross_lags().to_vec().into_iter().enumerate() {
                let li = lags.iter().position(|&v| v == lag).unwrap();
                for tt in 0..t {
                    for j in 0..p {
                        let what = format!("{cross:?}, lag {lag}, target {tt}, feature {j}");
                        near(l.cxy(ci, tt, j), o.cxy[li][tt][j], &format!("cxy {what}"));
                        near(l.cyx(ci, tt, j), o.cyx[li][tt][j], &format!("cyx {what}"));
                        near(
                            snap.cxy(ci, tt, j),
                            o.cxy[li][tt][j],
                            &format!("snap cxy {what}"),
                        );
                        near(
                            snap.cyx(ci, tt, j),
                            o.cyx[li][tt][j],
                            &format!("snap cyx {what}"),
                        );
                    }
                }
            }
        }
    }

    /// The shared layout's moments are the same recursion: the feature's
    /// autocovariance once, with the row's mix and the shared means, and
    /// each present target's terms with its own mix, through the same
    /// absences and the same break.
    #[test]
    fn the_shared_lag_moments_are_the_recursion_written_out() {
        let (p, t, lags) = (2usize, 2usize, vec![1usize, 3]);
        for cross in [None, Some(vec![1usize])] {
            let mut l = MarginalLags::new(p, t, lags.clone(), cross.clone(), true).unwrap();
            assert!(l.is_shared());
            let mut o = Longhand::new(p, t, &lags);
            for (r, (x, y, mx, my, mix, row)) in rows().iter().enumerate() {
                // One set of feature means for every target: the first's.
                let shared_mx = &mx[0];
                let targets: Vec<Option<TargetLag>> = (0..t)
                    .map(|tt| {
                        y[tt].map(|yt| TargetLag {
                            a: mix[tt].0,
                            b: mix[tt].1,
                            yt,
                            my: my[tt],
                            my_lo: 0.0,
                        })
                    })
                    .collect();
                l.update_shared(x, Some(*row), &targets, shared_mx, &[0.0; 2]);
                o.features(x, shared_mx, row.0, row.1);
                for tt in 0..t {
                    if let Some(yt) = y[tt] {
                        o.target(tt, x, yt, shared_mx, my[tt], mix[tt].0, mix[tt].1, false);
                    }
                }
                l.push(x, y);
                o.push(x, y);
                if r == BREAK_AFTER {
                    l.clear();
                    o.ring.clear();
                }
            }
            assert!(
                o.absent_back > 0 && o.aged > 0,
                "{} {}",
                o.absent_back,
                o.aged
            );
            for li in 0..lags.len() {
                for tt in 0..t {
                    near(l.cyy(li, tt), o.cyy[li][tt], "cyy");
                    for j in 0..p {
                        near(l.cxx(li, tt, j), o.cxx_shared[li][j], "the shared cxx");
                    }
                }
            }
            for (ci, lag) in l.cross_lags().to_vec().into_iter().enumerate() {
                let li = lags.iter().position(|&v| v == lag).unwrap();
                for tt in 0..t {
                    for j in 0..p {
                        let what = format!("{cross:?}, lag {lag}, target {tt}, feature {j}");
                        near(l.cxy(ci, tt, j), o.cxy[li][tt][j], &format!("cxy {what}"));
                        near(l.cyx(ci, tt, j), o.cyx[li][tt][j], &format!("cyx {what}"));
                    }
                }
            }
        }
    }

    /// A snapshot matches the lags it was taken from, and no others: each
    /// dimension `matches` reads, changed alone, is refused.
    #[test]
    fn a_snapshot_matches_only_the_lags_it_was_taken_from() {
        let l = MarginalLags::new(2, 2, vec![1, 3], Some(vec![3]), false).unwrap();
        let snap = l.moments();
        assert!(snap.matches(&l));
        type Edit = (&'static str, fn(&mut LagMoments));
        let edits: [Edit; 7] = [
            ("another width", |s| s.p = 3),
            ("a lag too many in cyy", |s| s.cyy.push(vec![0.0; 2])),
            ("a target short in cyy", |s| {
                s.cyy[0].pop();
            }),
            ("a lag short in cxx", |s| {
                s.cxx.pop();
            }),
            ("a pair short in cxx", |s| {
                s.cxx[1].pop();
            }),
            ("a cross lag too many in cxy", |s| s.cxy.push(vec![0.0; 4])),
            ("a pair too many in cyx", |s| s.cyx[0].push(0.0)),
        ];
        for (what, edit) in edits {
            let mut s = snap.clone();
            edit(&mut s);
            assert!(!s.matches(&l), "{what}");
        }
    }

    /// A model whose feature moments are shared keeps its lags that way, and
    /// its state, which says so, restores.
    #[test]
    fn a_shared_model_with_lags_restores() {
        use crate::{Decay, FeatureMomentLayout, Marginal, MarginalCfg, OnlineModel};
        let cfg = MarginalCfg {
            n_features: 2,
            n_targets: 2,
            decay: Decay::Halflife(20.0),
            min_weight: vec![0.0; 2],
            lags: vec![1, 2],
            serial_rule: None,
            cross_lags: None,
            bins: None,
            feature_moments: FeatureMomentLayout::Shared,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        };
        let mut m = Marginal::new(cfg).unwrap();
        for i in 0..10 {
            let v = f64::from(i);
            m.step(
                &[v, 1.0 - v * v],
                &[Some(v), (i % 2 == 0).then_some(-v)],
                1.0,
                1.0,
            );
        }
        assert_eq!(Marginal::restore(&m.state()).unwrap(), m);
    }
}
