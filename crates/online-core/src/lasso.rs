//! Lasso path on top of the EW-ridge accumulators (docs/PLAN.md §4.3).
//!
//! Coordinate descent on the standardized centered statistics -- each
//! target's Gram and centred cross-moments, the accumulators `ewridge` keeps
//! (`crate::gaps`) -- over a decreasing `lasso_path`, warm-started along the
//! path and across solves. `target_gaps` says which rows a target's Gram is
//! over, as for `ewridge` (docs/PLAN.md task 81).
//!
//! For standardized features (unit variance, zero mean) and a centered target,
//! the coordinate update for feature `i` at penalty `l` is
//!
//! ```text
//! rho_i = c_i - sum_{j != i} C_ij b_j          (C = correlation matrix, c = cov(x, y) / s_x)
//! b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))
//! ```
//!
//! with `soft(v, t) = sign(v) * max(|v| - t, 0)`; `l1_ratio = 1` is pure lasso,
//! `< 1` is elastic net. `C` is the correlation matrix of the target's Gram
//! and `c` each feature's covariance with the target over the feature's
//! standard deviation -- the target is centred but not scaled, so the
//! threshold `l * l1_ratio` is in the target's units while the ridge part
//! `l * (1 - l1_ratio)`, added to a correlation, has none -- centred at the
//! means over the target's own rows, from the centred cross-moments `ewridge` keeps since the code
//! review's N1. This kept them raw and centred them by subtraction, which lost
//! the fit at a level (N2). Coefficients are unscaled afterwards and the
//! intercept recovered as `ȳ − m_j · beta`, `m_j` the column means over the
//! target's rows.
//!
//! Lambda selection is free: predictions for every path point are computed
//! anyway, so `penalty_selected_j` is the argmin over the path of an EW mean of
//! squared out-of-sample error with half-life `select_half_life`, over the
//! rows from the one where target `j`'s own weight reaches its own
//! `min_weight` (`LassoCfg::target_min_weight`; review 2026-10-05, CA3).

use serde::{Deserialize, Serialize};

use crate::gaps::{Acc, AccSnap, AccView, Cross, gram_parts};
use crate::model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::since::Since;
use crate::solve::dot_aug;
use crate::{Decay, EwCov, GramPart, TargetGaps, TargetMoments};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LassoCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    /// Decreasing penalties. Applied to standardized (correlation-form) stats.
    pub lasso_path: Vec<f64>,
    /// 1.0 = lasso, < 1.0 = elastic net (docs/PLAN.md §4.3, [`Self::validate`]).
    pub l1_ratio: f64,
    /// Half-life of the EW squared-error used to pick lambda; defaults to the
    /// model half-life when None.
    pub select_half_life: Option<f64>,
    /// The weight, over every row, the model predicts from: the smallest
    /// of the targets' thresholds in a bank, which then withholds each
    /// target's output until its own weight reaches its own.
    pub min_weight: f64,
    /// Per target, the threshold its own weight before a row is checked
    /// against, as a bank checks it on the output (hard rule 8): a row
    /// below it adds no error to the target's selection, so one target's
    /// choice does not depend on another's threshold (review 2026-10-05,
    /// CA3). Empty: `min_weight` for every target.
    #[serde(default)]
    pub target_min_weight: Vec<f64>,
    /// Solve cadence in clock units; <= 0 solves every row. Measured from
    /// the last solve on the rows' stamps, the decayed clock held exactly,
    /// where the caller hands them (`since::Since`, docs/PLAN.md task 180).
    pub solve_every: f64,
    pub max_rows_between_solves: u32,
    /// The default cadence (docs/PLAN.md task 115 (b)): solve once the weight
    /// learned since the last solve reaches this share of the weight the fit
    /// holds, in place of `solve_every`'s clock. In steady state that is the
    /// clock's own `half_life / 50` at a share of `ln 2 / 50`; where they part
    /// -- warm-up, after a gap, a half-life far longer than the stream -- it
    /// keeps the fit that close to its data, where the clock solved once and
    /// never again. `None` keeps the clock.
    #[serde(default)]
    pub solve_share: Option<f64>,
    /// The coordinate descent's sweeps per solve, at least 1.
    pub max_iter: u32,
    pub tol: f64,
    /// Which rows a target's Gram is taken over where the target is null on
    /// some (docs/PLAN.md task 81; [`TargetGaps`]): its own, the default, or
    /// every row.
    #[serde(default)]
    pub target_gaps: TargetGaps,
    /// Clock units of history the path is fitted from, with a **hard** cutoff:
    /// a row older than this is not in the Gram at all (docs/PLAN.md §13).
    /// Inside the window the weights are still exponential. The selection
    /// error follows the same window, so the chosen `lambda` is chosen on the
    /// rows the fit uses.
    ///
    /// **Last, with `window_every` and `max_rows_between_snapshots`, and
    /// they must stay last**: the compact msgpack encoding writes a struct
    /// as an array, so a `skip_serializing_if` field anywhere else shifts
    /// the fields after it. Only the row cap skips.
    #[serde(default)]
    pub window: Option<f64>,
    /// Clock units between the window's snapshots, `0` every row; with
    /// `max_rows_between_snapshots` too, whichever comes first, and with
    /// neither, every row (`EwRidgeCfg::window_every`; docs/PLAN.md task
    /// 162).
    #[serde(default)]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots, counted on
    /// every row the model is stepped with, rows of weight zero included;
    /// `0` or `1` is every row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows_between_snapshots: Option<usize>,
}

impl LassoCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn n_lambdas(&self) -> usize {
        self.lasso_path.len()
    }

    /// Target `j`'s own threshold ([`Self::target_min_weight`]).
    pub fn min_weight_of(&self, j: usize) -> f64 {
        self.target_min_weight
            .get(j)
            .copied()
            .unwrap_or(self.min_weight)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("lasso: {e}"))?;
        if self
            .solve_share
            .is_some_and(|f| !(f.is_finite() && f > 0.0))
        {
            return Err("solve_share must be finite and > 0".into());
        }
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        let own = &self.target_min_weight;
        let bad = |t: &f64| t.is_nan() || *t < 0.0;
        if !(own.is_empty() || own.len() == self.n_targets) || own.iter().any(bad) {
            return Err(
                "lasso: target_min_weight must be empty or one value >= 0 per target".into(),
            );
        }
        if self.lasso_path.is_empty() {
            return Err("lasso_path must have at least one value".into());
        }
        if self.lasso_path.iter().any(|&l| l < 0.0) {
            return Err("lasso_path values must be >= 0".into());
        }
        if !self.lasso_path.windows(2).all(|w| w[0] >= w[1]) {
            return Err("lasso_path must be decreasing".into());
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err("l1_ratio must be in [0, 1]".into());
        }
        // No sweep is no descent: every solve a failure, and the output the
        // start of one -- the intercept at the mean, every slope 0 -- that
        // looked like a fit (review 2026-10-05, PB7).
        if self.max_iter < 1 {
            return Err(
                "lasso: max_iter must be >= 1: it is the coordinate descent's sweeps, and with \
                 none every solve fails and leaves the slopes at 0"
                    .into(),
            );
        }
        crate::window::check_cadence(
            "lasso",
            self.window,
            self.window_every,
            self.max_rows_between_snapshots,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lasso {
    cfg: LassoCfg,
    /// The Grams, and per target its weight, centred cross-moments and
    /// moments (see `crate::gaps::Acc`), as `ewridge` keeps them.
    acc: Acc,
    /// Per target, per path point: coefficients in original units (`k_total`).
    beta: Option<Vec<Vec<Vec<f64>>>>,
    /// Per target, per path point: EW mean squared out-of-sample error.
    sel_err: Vec<Vec<f64>>,
    sel_w: Vec<f64>,
    /// Per target: index into the path chosen by `sel_err`. It starts at
    /// `np − 1`, the path's last point (its lightest penalty), and stays
    /// there until the target's first scored row; from then it is the
    /// argmin of `sel_err`, the first point on a tie, so the heavier
    /// penalty of two that predict alike (review 2026-10-05, CA4).
    sel_idx: Vec<usize>,
    /// Where `solve_every`'s clock stands: the stamp of the last solve, the
    /// decayed clock held exactly (docs/PLAN.md task 180).
    since_solve: Since,
    rows_since_solve: u32,
    /// Weight learned since the last solve, for `solve_share`.
    #[serde(default)]
    weight_since_solve: f64,
    /// Coordinate descents that ran out of sweeps (`max_iter`) before
    /// meeting `tol`, one per target and path point; such a fit is where
    /// the descent stopped (review 2026-09-12, S11: nothing wrote this).
    pub solve_failures: u64,
    /// The hard-cutoff window, when the spec asks for one. Last, for the
    /// reason `LassoCfg::window` gives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
    #[serde(skip)]
    zbuf: Vec<f64>,
}

/// One Gram's statistics in correlation form (`Lasso::standardized`): the
/// correlation matrix, then per target the standardized cross-correlations,
/// the feature scales, and per target the column means.
type Standardized = (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>);

/// The accumulators a windowed path is fitted from, truncated to the window.
/// The selection reads its errors per target ([`Lasso::window_sel_err`]).
struct LassoView {
    acc: AccView,
}

/// The window's clock and the snapshots it subtracts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Windowed {
    clock: f64,
    snaps: crate::Snapshots<LassoMoments>,
}

/// Every accumulator the path is read from, before a row and decayed to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LassoMoments {
    acc: AccSnap,
    sel_w: Vec<f64>,
    sel_err: Vec<Vec<f64>>,
}

impl crate::Footprint for LassoMoments {
    fn footprint(&self) -> usize {
        crate::Footprint::footprint(&self.acc)
            + crate::window::floats(&self.sel_w)
            + self
                .sel_err
                .iter()
                .map(|v| crate::window::floats(v))
                .sum::<usize>()
    }
}

impl Lasso {
    pub fn new(cfg: LassoCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k_total = cfg.k_total();
        let (m, np) = (cfg.n_targets, cfg.n_lambdas());
        let cadence = crate::Cadence::of(cfg.window_every, cfg.max_rows_between_snapshots);
        let win = match cfg.window {
            Some(w) => Some(Windowed {
                clock: 0.0,
                snaps: crate::Snapshots::with_cadence(w, cadence)?,
            }),
            None => None,
        };
        Ok(Self {
            acc: Acc::new(m, k_total, 0, cfg.window.is_some()),
            beta: None,
            sel_err: vec![vec![0.0; np]; m],
            sel_w: vec![0.0; m],
            sel_idx: vec![np - 1; m],
            since_solve: Since::default(),
            rows_since_solve: 0,
            weight_since_solve: 0.0,
            solve_failures: 0,
            win,
            zbuf: vec![0.0; k_total],
            cfg,
        })
    }

    pub fn cfg(&self) -> &LassoCfg {
        &self.cfg
    }

    /// Coefficients per (target, path point), in original units.
    pub fn coefficients(&self) -> Option<&Vec<Vec<Vec<f64>>>> {
        self.beta.as_ref()
    }

    /// Selected lambda per target.
    pub fn lam_selected(&self) -> Vec<f64> {
        self.sel_idx
            .iter()
            .map(|&i| self.cfg.lasso_path[i])
            .collect()
    }

    /// The Grams the path is read from, one entry per Gram with the targets
    /// that read it, and the target moments: see `EwRidge::gram_parts`.
    pub fn gram_parts(&self) -> (Vec<GramPart>, Option<TargetMoments>) {
        let (of, gaps) = (&self.acc.grams.of, self.cfg.target_gaps);
        match self.view() {
            Some(v) => (
                gram_parts(&v.acc.grams, of, &v.acc.cross, &v.acc.wj, gaps),
                v.acc.tm,
            ),
            None => (
                gram_parts(
                    &self.acc.grams.grams,
                    of,
                    &self.acc.cross,
                    &self.acc.wj,
                    gaps,
                ),
                Some(self.acc.tm.clone()),
            ),
        }
    }

    /// Per-target **uncentered** cross-moments `E[z·y_t]` over each target's
    /// rows, `k_total` long; see `EwRidge::cross_moments`.
    pub fn cross_moments(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| self.acc.cross.raw(j))
            .collect()
    }

    /// Per-target accumulated weight behind [`Self::cross_moments`].
    pub fn target_weights(&self) -> &[f64] {
        &self.acc.wj
    }

    /// Per-target mean, variance and `Sum w^2` (docs/ENHANCEMENTS.md E45).
    pub fn target_moments(&self) -> Option<&TargetMoments> {
        Some(&self.acc.tm)
    }

    /// The decay the selection error ages by: `select_half_life`'s, or the
    /// model's where it is not set.
    fn select_decay(&self) -> Decay {
        self.cfg
            .select_half_life
            .map_or(self.cfg.decay, Decay::Halflife)
    }

    /// The accumulated weight over every row: under a `window`, the weight
    /// inside it.
    pub fn n_eff(&self) -> f64 {
        self.window_weights().map_or(self.acc.cross.w, |(w, _)| w)
    }

    /// The accumulators the path is fitted from: the live ones, or -- with a
    /// `window` -- the same ones with everything older than the window
    /// subtracted off (docs/PLAN.md §13), as `EwRidge::view` takes them. The
    /// selection error is truncated with them, so the `lambda` chosen is the
    /// one that fits the window rather than one that fitted a history the
    /// window has dropped. An empty window is an empty view, never the live
    /// state, and a target with no row inside it keeps weight 0 while the
    /// others stay windowed (review 2026-09-12, C2).
    fn view(&self) -> Option<LassoView> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        let f = self.cfg.decay.factor(win.clock - u);
        let acc = self.acc.window(&old.acc, f)?;
        Some(LassoView { acc })
    }

    /// The selection errors [`Self::view`] carried, every target's, until the
    /// selection read them one target at a time (review 2026-09-12, P1):
    /// kept, as it was, to hold [`Self::window_sel_err`] to it. `None` where
    /// the view is.
    #[cfg(test)]
    fn view_sel_err(&self) -> Option<Vec<Vec<f64>>> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        let f = self.cfg.decay.factor(win.clock - u);
        let acc = self.acc.window(&old.acc, f)?;
        let f_sel = self.select_decay().factor(win.clock - u);
        let mut sel_err = self.sel_err.clone();
        if acc.cross.w > 0.0 {
            for (j, err) in sel_err.iter_mut().enumerate() {
                if old.sel_w[j] == 0.0 {
                    continue;
                }
                if let Some((_w, e)) = crate::truncated_mean(
                    self.sel_w[j],
                    &self.sel_err[j],
                    old.sel_w[j],
                    &old.sel_err[j],
                    f_sel,
                ) {
                    *err = e.into_iter().map(|v| v.max(0.0)).collect();
                }
            }
        }
        Some(sel_err)
    }

    /// The window's weights alone, as [`Self::view`] has them: what `predict`
    /// and `n_eff` read on every row, without the view's O(k²) truncation
    /// (review 2026-09-12, P1). `None` where the live accumulators are the
    /// answer.
    fn window_weights(&self) -> Option<(f64, Vec<f64>)> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        let f = self.cfg.decay.factor(win.clock - u);
        self.acc.window_weights(&old.acc, f)
    }

    /// `predict` from the window's weights, [`Self::window_weights`], which
    /// `step` reads once for this and for its selection.
    fn predict_with(&self, x: &[f64], weights: Option<&(f64, Vec<f64>)>) -> Step {
        let (m, np) = (self.cfg.n_targets, self.cfg.n_lambdas());
        // Under a `window`, the `n_eff` reported and gated on is the weight
        // inside it, as `EwRidge::predict` reports it (review 2026-09-12, C9).
        // The weights alone for that and for the per-target test (C2), not
        // the O(k²) view, which is the solve's (review 2026-09-12, P1).
        let (n_eff, wj) = match weights {
            Some((w, wj)) => (*w, wj.as_slice()),
            None => (self.acc.cross.w, self.acc.wj.as_slice()),
        };
        let mut pred = vec![f64::NAN; m * np];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_weight, &self.beta) {
            for j in 0..m {
                if wj[j] > 0.0 {
                    for li in 0..np {
                        pred[j * np + li] = dot_aug(&beta[j][li], x, self.cfg.fit_intercept);
                    }
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: Some(Extra::Lasso {
                lam_selected: self.lam_selected(),
            }),
        }
    }

    /// Target `j`'s selection errors inside the window, as it stands at the
    /// row just learned: the errors less the boundary snapshot's, aged by
    /// the selection's own decay; the live ones where there is no window or
    /// no error inside it. One target's, without the Gram truncation: the
    /// selection loop built the whole view once per target per row (review
    /// 2026-09-12, P1).
    ///
    /// The truncation ages the snapshot by `select_half_life`, the decay the
    /// errors age by, where it used the model's (PLAN task 95). Whether the
    /// window is empty is the errors' own question: it used to be the
    /// Gram's, which the choice now reads before this row is learned. An
    /// empty window -- no scored row of this target inside it -- is `None`,
    /// and the choice stands: the whole-history errors it used to fall back
    /// on are the rows the window has dropped (review 2026-09-25).
    fn window_sel_err(&self, j: usize) -> Option<Vec<f64>> {
        let live = || Some(self.sel_err[j].clone());
        let Some(win) = self.win.as_ref() else {
            return live();
        };
        let Some((u, old)) = win.snaps.boundary() else {
            return live();
        };
        if old.sel_w[j] == 0.0 {
            // Nothing has aged out of the errors: the live ones, to the bit,
            // rather than the same numbers through a subtraction of zero.
            return live();
        }
        let f = self.select_decay().factor(win.clock - u);
        crate::truncated_mean(
            self.sel_w[j],
            &self.sel_err[j],
            old.sel_w[j],
            &old.sel_err[j],
            f,
        )
        .map(|(_w, e)| e.into_iter().map(|v| v.max(0.0)).collect())
    }

    /// One Gram's statistics in correlation form, for the targets `readers`
    /// that read it: the correlation matrix `c`, each target's standardized
    /// cross-correlation `d` -- centred at its own means -- the feature scales
    /// `s`, and each target's column means over its rows, which the intercept
    /// is recovered from.
    fn standardized(&self, acc: &EwCov, cross: &Cross, readers: &[usize]) -> Standardized {
        let k = self.cfg.n_features;
        let off = usize::from(self.cfg.fit_intercept);
        if off == 0 {
            // No intercept: nothing to centre on. Scale by the raw second
            // moment and keep the raw cross-moments -- `E[x x']` and `E[x y]`,
            // whose zero-penalty solution is least squares through the origin
            // -- as `EwRidge`'s no-intercept branch does. This centred the Gram
            // whatever `fit_intercept` said, so without one it solved a hybrid
            // of the centred and the raw problem, least squares only when
            // every feature has mean zero (review 2026-09-12, C8).
            let s: Vec<f64> = (0..k).map(|i| acc.raw(i, i).max(0.0).sqrt()).collect();
            let mut c = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    c[i * k + j] = if s[i] > 0.0 && s[j] > 0.0 {
                        acc.raw(i, j) / (s[i] * s[j])
                    } else {
                        f64::from(i == j)
                    };
                }
            }
            let d = readers
                .iter()
                .map(|&j| {
                    let rj = cross.raw(j);
                    (0..k)
                        .map(|i| if s[i] > 0.0 { rj[i] / s[i] } else { 0.0 })
                        .collect::<Vec<f64>>()
                })
                .collect();
            return (c, d, s, vec![vec![0.0; k]; readers.len()]);
        }
        // Centered co-moments come straight from the accumulator; deriving them
        // as raw - mean*mean would reintroduce the cancellation the Welford
        // representation exists to avoid.
        let mut cov = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = acc.cov(i + off, j + off);
            }
        }
        let s: Vec<f64> = (0..k)
            .map(|i| {
                let v = cov[i * k + i];
                if crate::variance_is_usable(v, acc.raw(i + off, i + off)) {
                    v.sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                c[i * k + j] = if s[i] > 0.0 && s[j] > 0.0 {
                    cov[i * k + j] / (s[i] * s[j])
                } else {
                    f64::from(i == j)
                };
            }
        }
        // Each target's cross-covariance is the centred one it keeps, so
        // nothing level-sized is subtracted here (N2).
        let d = readers
            .iter()
            .map(|&j| {
                (0..k)
                    .map(|i| {
                        if s[i] > 0.0 {
                            cross.c[j][i + off] / s[i]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();
        // `m_j` as `EwRidge::solve_centred` takes it: the Gram's mean, or
        // the target's own mean under `pairwise`, where the Gram is over
        // every row.
        let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
        let means = readers
            .iter()
            .map(|&j| {
                (0..k)
                    .map(|i| {
                        if pairwise {
                            cross.mj[j][i + off]
                        } else {
                            acc.mean(i + off)
                        }
                    })
                    .collect()
            })
            .collect();
        (c, d, s, means)
    }

    fn solve(&mut self) {
        let k = self.cfg.n_features;
        let k_total = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        // With a `window`, the path is fitted from the truncated
        // accumulators: no row older than the window is in a Gram, the
        // right-hand side, or the selection error. The selection error is
        // read where the path point is chosen, in `step`.
        let view = self.view();
        let (grams, cross, wj) = match view.as_ref() {
            Some(v) => (v.acc.grams.as_slice(), &v.acc.cross, &v.acc.wj),
            None => (
                self.acc.grams.grams.as_slice(),
                &self.acc.cross,
                &self.acc.wj,
            ),
        };
        let np = self.cfg.n_lambdas();
        let mut out = vec![vec![vec![0.0; k_total]; np]; self.cfg.n_targets];
        let mut unconverged = 0u64;

        // One set of statistics per Gram, for the targets that read it.
        for (g, gram) in grams.iter().enumerate() {
            let readers = self.acc.grams.readers(g);
            let (c, d, s, means) = self.standardized(gram, cross, &readers);
            for (jj, &j) in readers.iter().enumerate() {
                if wj[j] <= 0.0 {
                    // Under a window, no row of this target is inside it: no
                    // fit to report (C2). Without one, a target never seen
                    // keeps the zeros it always had.
                    if view.is_some() {
                        out[j] = vec![vec![f64::NAN; k_total]; np];
                    }
                    continue;
                }
                // Warm start from the previous solve's largest-penalty
                // solution -- unless that was an empty window's NaN, which
                // would poison the descent rather than start it.
                let mut b = vec![0.0; k];
                if let Some(prev) = &self.beta {
                    for i in 0..k {
                        let p = prev[j][0][i + off];
                        if s[i] > 0.0 && p.is_finite() {
                            b[i] = p * s[i];
                        }
                    }
                }
                for (li, &lam) in self.cfg.lasso_path.iter().enumerate() {
                    // Coordinate descent, warm-started along the path.
                    let l1 = lam * self.cfg.l1_ratio;
                    let l2 = lam * (1.0 - self.cfg.l1_ratio);
                    let mut converged = false;
                    for _ in 0..self.cfg.max_iter {
                        let mut max_delta: f64 = 0.0;
                        for i in 0..k {
                            if s[i] <= 0.0 {
                                b[i] = 0.0;
                                continue;
                            }
                            let mut rho = d[jj][i];
                            for (jn, bj) in b.iter().enumerate() {
                                if jn != i {
                                    rho -= c[i * k + jn] * bj;
                                }
                            }
                            let denom = c[i * k + i] + l2;
                            let newb = if rho > l1 {
                                (rho - l1) / denom
                            } else if rho < -l1 {
                                (rho + l1) / denom
                            } else {
                                0.0
                            };
                            max_delta = max_delta.max((newb - b[i]).abs());
                            b[i] = newb;
                        }
                        if max_delta < self.cfg.tol {
                            converged = true;
                            break;
                        }
                    }
                    // Out of sweeps before `tol`: the fit is where the
                    // descent stopped, and it is counted (review 2026-09-12,
                    // S11: nothing wrote `solve_failures`).
                    unconverged += u64::from(!converged);
                    // Unscale and recover the intercept, from the target's
                    // own means.
                    let coefs = &mut out[j][li];
                    for i in 0..k {
                        coefs[i + off] = if s[i] > 0.0 { b[i] / s[i] } else { 0.0 };
                    }
                    if self.cfg.fit_intercept {
                        let mut b0 = cross.my[j];
                        for i in 0..k {
                            b0 -= means[jj][i] * coefs[i + off];
                        }
                        coefs[0] = b0;
                    }
                }
            }
        }
        self.solve_failures += unconverged;
        self.beta = Some(out);
        self.since_solve.restart();
        self.rows_since_solve = 0;
        self.weight_since_solve = 0.0;
    }
}

impl OnlineModel for Lasso {
    fn set_solve_share(&mut self, share: Option<f64>) {
        self.cfg.solve_share = share;
    }

    fn solve_share(&self) -> Option<f64> {
        self.cfg.solve_share
    }

    /// The window keys the row's snapshot by its stamp (task 175), and the
    /// solve cadence measures its clock by it (task 180).
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.stamp_next(stamp);
        }
        self.since_solve.stamp_next(stamp);
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        self.win.as_ref().and_then(|win| win.snaps.over_budget())
    }

    /// The ring's shadow, its snapshots formed as the step forms them
    /// (docs/PLAN.md task 115 (d)).
    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        let win = self.win.as_ref()?;
        Some(crate::WindowShadow::new(win.clock, &win.snaps, || {
            LassoMoments {
                acc: self.acc.snapshot(1.0),
                sel_w: self.sel_w.clone(),
                sel_err: self.sel_err.clone(),
            }
        }))
    }
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        match self.window_weights() {
            Some((_, wj)) => out.extend_from_slice(&wj),
            None => out.extend_from_slice(&self.acc.wj),
        }
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let m = self.cfg.n_targets;
        let np = self.cfg.n_lambdas();
        let lam_decay = self.cfg.decay.factor(d_clock);
        if self.zbuf.len() != self.cfg.k_total() {
            self.zbuf = vec![0.0; self.cfg.k_total()];
        }
        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict every path point (state before the update) ----
        // `lam_selected` is read here too: the selection as it stood coming
        // into this row, like everything else a `Step` reports, so pairing it
        // with this row's `pred_<lambda>` slots is an out-of-sample choice.
        let weights = self.window_weights();
        let out = self.predict_with(x, weights.as_ref());
        let pred = &out.pred;
        // Each target's own weight before the row, inside the window under
        // one: what its own `min_weight` is checked against, here as on the
        // output (hard rule 8).
        let own: &[f64] = weights.as_ref().map_or(&self.acc.wj, |(_, wj)| wj);

        // ---- the window moves to this row ----
        // The snapshot is every accumulator before this row, decayed to this
        // row's clock, so subtracting it later retains this row and after.
        // The selection's part too: its errors and weights before this row's
        // error is folded in, aged over the row by the selection's own decay.
        // It was taken after, with the weight aged twice, and the choice
        // below read the window as it stood a row earlier (PLAN task 95).
        let sel_lam = self.select_decay().factor(d_clock);
        if let Some(win) = self.win.as_mut() {
            // Built inside the closure so the snapshot is only formed on the
            // rows `offer` keeps, not on every row (review 2026-09-18, P1).
            win.snaps.learn(&mut win.clock, d_clock, || LassoMoments {
                acc: self.acc.snapshot(lam_decay),
                sel_w: self.sel_w.iter().map(|w| w * sel_lam).collect(),
                sel_err: self.sel_err.clone(),
            });
        }

        // ---- lambda selection: EW mean squared OOS error, free from preds ----
        // A target's errors count from the row where its own weight reaches
        // its own `min_weight`: a row its threshold withholds from the
        // output adds none, so its choice is the same whatever the other
        // targets' thresholds, which the model's own gate takes the least of
        // (review 2026-10-05, CA3).
        for j in 0..m {
            let aged = sel_lam * self.sel_w[j];
            let shown = own[j] >= self.cfg.min_weight_of(j);
            match y[j].filter(|_| shown && pred[j * np].is_finite()) {
                Some(yj) if weight > 0.0 => {
                    let w_new = aged + weight;
                    let (a, b) = (aged / w_new, weight / w_new);
                    for li in 0..np {
                        let e = yj - pred[j * np + li];
                        self.sel_err[j][li] = a * self.sel_err[j][li] + b * e * e;
                    }
                    self.sel_w[j] = w_new;
                }
                _ => {
                    // No error to fold -- the target is null, not predicted
                    // or below its own threshold, or the row has no weight --
                    // so the errors only age. A zero-weight row folded in the
                    // mean form is `0/0` before the first error, and the NaN
                    // it left in every `sel_err` never washed out: every
                    // comparison below false, the choice stuck at the
                    // heaviest penalty (hard rule 9; review 2026-09-12, C7).
                    self.sel_w[j] = aged;
                    if aged <= 0.0 {
                        continue;
                    }
                }
            }
            // Under a `window`, the path point is chosen on the error
            // *inside* it: selecting on the whole history while fitting on
            // the window would pick a lambda for rows the coefficients no
            // longer see. And it is chosen again on every row, scored or
            // not, since the rows inside the window change on every row; a
            // choice left standing from the last scored row read errors the
            // window had since dropped (PLAN task 95). Without a window only
            // the common age moved, and the choice is the one it was.
            if let Some(err) = self.window_sel_err(j) {
                let mut best = 0usize;
                for li in 1..np {
                    if err[li] < err[best] {
                        best = li;
                    }
                }
                self.sel_idx[j] = best;
            }
        }

        // ---- update accumulators ----
        self.acc
            .learn(&self.zbuf, y, lam_decay, weight, self.cfg.target_gaps);

        // The clock since the last solve, on the row's stamp (task 180).
        self.since_solve.step(d_clock);
        self.rows_since_solve += 1;
        if weight.is_finite() && weight > 0.0 {
            self.weight_since_solve += weight;
        }
        let by_cadence = match self.cfg.solve_share {
            Some(share) => self.weight_since_solve >= share * self.n_eff(),
            None => self.cfg.solve_every <= 0.0 || self.since_solve.reached(self.cfg.solve_every),
        };
        let due = by_cadence
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.n_eff() >= self.cfg.min_weight);
        if due {
            self.solve();
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        self.predict_with(x, self.window_weights().as_ref())
    }

    fn state(&self) -> State {
        State::new(ModelState::Lasso(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Lasso(m) => {
                let mut m = (**m).clone();
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                // The selection fields too: `sel_idx` indexes `lasso_path`
                // in `lam_selected`, and a window is carried exactly when
                // the cfg asks for one (review 2026-09-18, B3).
                let np = m.cfg.n_lambdas();
                let selection = m.sel_err.len() == n
                    && m.sel_err.iter().all(|e| e.len() == np)
                    && m.sel_w.len() == n
                    && m.sel_idx.len() == n
                    && m.sel_idx.iter().all(|&i| i < np)
                    && (m.cfg.target_min_weight.is_empty() || m.cfg.target_min_weight.len() == n)
                    && m.beta.as_ref().is_none_or(|b| {
                        b.len() == n
                            && b.iter()
                                .all(|t| t.len() == np && t.iter().all(|c| c.len() == k))
                    })
                    && m.win.is_some() == m.cfg.window.is_some();
                // A state written before schema 17 kept each target's own
                // mean as an offset (`crate::gaps::Cross`).
                if s.schema_version < 17 {
                    m.acc.offsets_to_means();
                    if let Some(win) = m.win.as_mut() {
                        win.snaps.iter_mut().for_each(|s| s.acc.offsets_to_means());
                    }
                }
                if !m.acc.has_shape(n, k) || !selection {
                    return Err(StateError::Invalid(
                        "lasso: the accumulators have the wrong shape".into(),
                    ));
                }
                // The runs follow the window, not the file (review 2026-09-26,
                // C4; `EwCovModel::restore` says why).
                match m.cfg.window {
                    None => m.acc.set_runs_off(),
                    Some(_) if !m.acc.keeps_runs() => {
                        return Err(StateError::Invalid(
                            "lasso: a windowed state whose runs are off".into(),
                        ));
                    }
                    Some(_) => {}
                }
                m.zbuf = vec![0.0; k];
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "lasso",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.n_lambdas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A selection index past the path is refused, where it loaded and
    /// panicked in `lam_selected` (review 2026-09-18, B3); so is a threshold
    /// for a target there is not (CA3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Lasso::new(cfg(2, 1, vec![0.1, 0.01])).unwrap();
        let corrupt: [fn(&mut Lasso); 2] = [
            |l| l.sel_idx[0] = 2,
            |l| l.cfg.target_min_weight = vec![1.0; 2],
        ];
        for f in corrupt {
            let mut s = m.state();
            let ModelState::Lasso(inner) = &mut s.model else {
                unreachable!()
            };
            f(inner);
            match Lasso::restore(&s) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
                other => panic!("{other:?}"),
            }
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// A target absent on a row whose features stand at the input bound
    /// (review 2026-09-26, G3, from the contract's proptest): the cross
    /// accumulator kept each target's own feature mean as an offset from the
    /// all-row mean, and their sum, two numbers of `1e99`, resolved nothing
    /// below `1e83`, so the next present row's deviation of `0.4` read as
    /// `1e83`, the cross-moment as `1e132`, and the prediction at `1e100`
    /// as `-inf`, windowed or not. Own means are kept as their own pairs
    /// now, and every prediction is a number.
    #[test]
    fn a_target_absent_on_a_row_at_the_bound_leaves_the_fit_finite() {
        use crate::OnlineModel;
        for window in [Some(7.0), None] {
            let mut c = cfg(2, 1, vec![0.1, 0.0]);
            c.decay = Decay::Halflife(20.0);
            c.min_weight = 3.0;
            c.max_iter = 100;
            c.tol = 1e-10;
            c.window = window;
            let mut m = Lasso::new(c).unwrap();
            let rows: Vec<([f64; 2], Option<f64>, f64)> = vec![
                ([0.0, -2.570413849510098e99], None, 1.0),
                ([0.0, 0.0], None, 1.0),
                ([0.0, 0.0], Some(0.0), 1.0),
                (
                    [0.0, -0.39761046383362925],
                    Some(-5.567947375697575e49),
                    1.0,
                ),
                ([1.4616143340058023, 1e100], Some(0.0), 0.01),
                ([1e100, 0.0], None, 1.0),
            ];
            for (i, (x, y0, w)) in rows.iter().enumerate() {
                let out = m.step(x, &[*y0], if i == 0 { 0.0 } else { 1.0 }, *w);
                assert!(
                    out.pred.iter().all(|p| p.is_nan() || p.is_finite()),
                    "window {window:?}, row {i}: {:?}",
                    out.pred
                );
            }
            // The own mean of `z` over the target's rows is those rows' mean
            // -- `x0` a hundredth of `1.46` over two rows' weight, `x1` a
            // hundredth of `1e100` over the same -- untouched by the `1e99`
            // on the row the target was absent on.
            let mj = &m.acc.cross.mj[0];
            assert!(mj[1] > 0.0 && mj[1] < 0.02, "{mj:?}");
            assert!(mj[2] > 1e97 && mj[2] < 1e98, "{mj:?}");
        }
    }

    fn cfg(k: usize, m: usize, path: Vec<f64>) -> LassoCfg {
        LassoCfg {
            n_features: k,
            n_targets: m,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            lasso_path: path,
            l1_ratio: 1.0,
            select_half_life: None,
            min_weight: (k + 1) as f64,
            target_min_weight: Vec::new(),
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
            target_gaps: TargetGaps::OwnRows,
            max_iter: 200,
            tol: 1e-12,
        }
    }

    /// Feed a deterministic stream with `k` informative features.
    fn fit(cfg: LassoCfg, n: usize, seed: u64) -> (Lasso, Vec<(Vec<f64>, f64)>) {
        let k = cfg.n_features;
        let mut m = Lasso::new(cfg).unwrap();
        let mut s = seed;
        let mut rows = Vec::new();
        for i in 0..n {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y = 1.5 * x[0] - 0.75 * x[1] + 0.25 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            rows.push((x, y));
        }
        (m, rows)
    }

    #[test]
    fn every_path_point_satisfies_the_kkt_conditions() {
        // The coordinate descent is checked against the optimality conditions
        // of the problem it claims to solve, rather than against a golden
        // number: for the objective 1/2 b'Cb - d'b + l1|b|_1 + l2/2 |b|^2, at
        // the optimum the gradient g = Cb - d + l2 b satisfies g_i = -l1 sign(b_i)
        // on the active set and |g_i| <= l1 off it. This is the same check the
        // Python suite makes; having it here makes the solver's arithmetic
        // visible to `cargo test`, and so to mutation testing.
        for l1_ratio in [1.0, 0.5] {
            let mut c = cfg(4, 1, vec![0.5, 0.1, 0.01]);
            c.l1_ratio = l1_ratio;
            c.tol = 1e-14;
            c.max_iter = 2000;
            let (m, _) = fit(c.clone(), 500, 11);

            // Rebuild the standardized normal equations the solver works in.
            let k = c.n_features;
            let off = usize::from(c.fit_intercept);
            let s: Vec<f64> = (0..k)
                .map(|i| m.acc.grams.grams[0].cov(i + off, i + off).sqrt())
                .collect();
            let beta = m.coefficients().unwrap();
            for (li, &lam) in c.lasso_path.iter().enumerate() {
                let (l1, l2) = (lam * c.l1_ratio, lam * (1.0 - c.l1_ratio));
                // Back to the scaled parameterization the objective is in.
                let b: Vec<f64> = (0..k).map(|i| beta[0][li][i + off] * s[i]).collect();
                for i in 0..k {
                    let mut g = 0.0;
                    for jj in 0..k {
                        g += m.acc.grams.grams[0].cov(i + off, jj + off) / (s[i] * s[jj]) * b[jj];
                    }
                    let d_i = m.acc.cross.c[0][i + off] / s[i];
                    g -= d_i;
                    g += l2 * b[i];
                    if b[i].abs() > 1e-9 {
                        assert!(
                            (g + l1 * b[i].signum()).abs() < 1e-6,
                            "lam {lam} ratio {l1_ratio} coef {i}: active KKT {g}"
                        );
                    } else {
                        assert!(
                            g.abs() <= l1 + 1e-6,
                            "lam {lam} ratio {l1_ratio} coef {i}: inactive KKT {g} > {l1}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_path_is_monotone_in_sparsity_and_shrinkage() {
        // A larger penalty can only zero more coefficients and shrink the rest;
        // the path must be ordered, which pins the direction of every
        // soft-threshold comparison.
        let c = cfg(6, 1, vec![1.0, 0.3, 0.1, 0.03, 0.0]);
        let (m, _) = fit(c.clone(), 600, 13);
        let beta = &m.coefficients().unwrap()[0];
        let nnz: Vec<usize> = beta
            .iter()
            .map(|b| b[1..].iter().filter(|v| v.abs() > 1e-9).count())
            .collect();
        for w in nnz.windows(2) {
            assert!(
                w[0] <= w[1],
                "sparsity should relax along the path: {nnz:?}"
            );
        }
        assert!(
            nnz[0] < nnz[nnz.len() - 1],
            "the path must do something: {nnz:?}"
        );

        // The strongest signal's magnitude grows as the penalty falls.
        let mags: Vec<f64> = beta.iter().map(|b| b[1].abs()).collect();
        for w in mags.windows(2) {
            assert!(w[0] <= w[1] + 1e-9, "shrinkage should relax: {mags:?}");
        }
        assert!((mags[mags.len() - 1] - 1.5).abs() < 0.05, "{mags:?}");
    }

    #[test]
    fn the_intercept_is_recovered_from_the_centered_fit() {
        // The features are centered before the solve, so the intercept is
        // reconstructed as mean(y) - sum(b_i mean(x_i)) rather than fitted.
        let mut c = cfg(2, 1, vec![0.0]);
        c.min_weight = 3.0;
        let (m, rows) = fit(c, 500, 17);
        let b = &m.coefficients().unwrap()[0][0];
        let n = rows.len() as f64;
        let ybar: f64 = rows.iter().map(|(_, y)| y).sum::<f64>() / n;
        let xbar: Vec<f64> = (0..2)
            .map(|i| rows.iter().map(|(x, _)| x[i]).sum::<f64>() / n)
            .collect();
        let want = ybar - b[1] * xbar[0] - b[2] * xbar[1];
        assert!((b[0] - want).abs() < 1e-6, "{} vs {want}", b[0]);
    }

    #[test]
    fn lambda_selection_tracks_the_out_of_sample_error() {
        // `sel_err` is an EW mean of each path point's squared OOS error and
        // `lam_selected` reports the argmin. Both are checked against the
        // predictions the model itself emitted, so a selection reading the
        // wrong slot, or the wrong direction of the comparison, is caught.
        let mut c = cfg(3, 1, vec![1.0, 0.05, 0.0]);
        c.min_weight = 4.0;
        c.select_half_life = Some(f64::INFINITY);
        let np = c.n_lambdas();
        let mut m = Lasso::new(c).unwrap();

        let mut sums = vec![0.0; np];
        let mut count = 0.0;
        let mut s = 19u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] + 0.02 * lcg(&mut s);
            let step = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if step.pred[0].is_finite() {
                count += 1.0;
                for (li, sum) in sums.iter_mut().enumerate() {
                    *sum += (y - step.pred[li]).powi(2);
                }
            }
        }
        assert!(count > 100.0);
        // An infinite select_half_life makes the EW mean a plain mean.
        for (li, sum) in sums.iter().enumerate() {
            assert!(
                (m.sel_err[0][li] - sum / count).abs() < 1e-9,
                "path point {li}: {} vs {}",
                m.sel_err[0][li],
                sum / count
            );
        }
        let best = (0..np)
            .min_by(|&a, &b| sums[a].partial_cmp(&sums[b]).unwrap())
            .unwrap();
        assert_eq!(m.sel_idx[0], best, "selection must be the argmin: {sums:?}");
        assert_eq!(m.lam_selected(), vec![m.cfg.lasso_path[best]]);
        // Only one feature matters, so the heaviest penalty must not win.
        assert!(
            best > 0,
            "a penalty of 1.0 should not be selected: {sums:?}"
        );
    }

    #[test]
    fn a_null_target_and_a_zero_weight_row_only_decay() {
        let mut c = cfg(2, 1, vec![0.0]);
        c.decay = Decay::Halflife(10.0);
        c.min_weight = 3.0;
        let (mut m, _) = fit(c, 60, 23);
        let (wj, c, my, sel_w) = (
            m.acc.wj[0],
            m.acc.cross.c[0].clone(),
            m.acc.cross.my[0],
            m.sel_w[0],
        );

        let lam = 0.5f64.powf(2.0 / 10.0);
        m.step(&[0.5, -0.5], &[None], 2.0, 1.0);
        assert!((m.acc.wj[0] - wj * lam).abs() < 1e-12, "null decays wj");
        assert_eq!(m.acc.cross.c[0], c, "and leaves the cross-moments alone");
        assert_eq!(m.acc.cross.my[0], my);
        assert!((m.sel_w[0] - sel_w * lam).abs() < 1e-12);
    }

    /// Hard rule 9 in the selection: a zero-weight row on the first row
    /// with a prediction met a selection weight of 0 and formed `0/0`, and
    /// the NaN it left in every `sel_err` never washed out -- each
    /// comparison false, the choice held at the heaviest penalty for the
    /// life of the state (review 2026-09-12, C7). The errors are those of
    /// `lambda_selection_tracks_the_out_of_sample_error`, which leave the
    /// zero-weight row out as its weight does.
    #[test]
    fn a_zero_weight_row_on_the_first_prediction_leaves_the_selection_alone() {
        let mut c = cfg(3, 1, vec![1.0, 0.05, 0.0]);
        c.min_weight = 1.0;
        c.select_half_life = Some(f64::INFINITY);
        let np = c.n_lambdas();
        let mut m = Lasso::new(c).unwrap();
        let (mut sums, mut count, mut met) = (vec![0.0; np], 0.0, false);
        let mut s = 19u64;
        for i in 0..402 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] + 0.02 * lcg(&mut s);
            let w = if i == 1 { 0.0 } else { 1.0 };
            let step = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
            if step.pred[0].is_finite() {
                if w == 0.0 {
                    met = true;
                } else {
                    count += 1.0;
                    for (li, sum) in sums.iter_mut().enumerate() {
                        *sum += (y - step.pred[li]).powi(2);
                    }
                }
            }
            assert!(
                m.sel_err[0].iter().all(|e| e.is_finite()),
                "row {i}: {:?}",
                m.sel_err[0]
            );
        }
        assert!(
            met,
            "the zero-weight row must be the first with a prediction"
        );
        for (li, sum) in sums.iter().enumerate() {
            assert!(
                (m.sel_err[0][li] - sum / count).abs() < 1e-9,
                "path point {li}: {} vs {}",
                m.sel_err[0][li],
                sum / count
            );
        }
        let best = (0..np)
            .min_by(|&a, &b| sums[a].partial_cmp(&sums[b]).unwrap())
            .unwrap();
        assert!(
            best > 0,
            "one feature matters, so the heaviest penalty loses: {sums:?}"
        );
        assert_eq!(m.lam_selected(), vec![m.cfg.lasso_path[best]]);
    }

    /// Each target's selection counts its errors from the row where its own
    /// weight before the row reaches its own `min_weight` (review 2026-10-05,
    /// CA3; hard rule 8). `y1`, present on every other row, waits for six
    /// rows of its own, where it counted from the model's first prediction;
    /// and one target's selection is the same whatever the other's threshold.
    #[test]
    fn each_targets_selection_counts_from_its_own_min_weight() {
        let t = 6.0;
        let run = |own: [f64; 2]| {
            let mut c = cfg(2, 2, vec![1.0, 0.1, 0.0]);
            c.min_weight = own[0].min(own[1]);
            c.target_min_weight = own.to_vec();
            let mut m = Lasso::new(c).unwrap();
            let mut s = 7u64;
            (0..40)
                .map(|i| {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = [1.0 + 2.0 * x[0], -1.0 + x[1]].map(|v| v + 0.3 * lcg(&mut s));
                    let own_before = m.acc.wj[1];
                    let y1 = (i % 2 == 0).then_some(y[1]);
                    m.step(&x, &[Some(y[0]), y1], f64::from(u8::from(i > 0)), 1.0);
                    (own_before, m.sel_w.clone(), m.sel_err.clone())
                })
                .collect::<Vec<_>>()
        };
        let (hi, lo, none) = (run([t, t]), run([0.0, t]), run([0.0, 0.0]));
        for (i, (own, sel_w, _)) in hi.iter().enumerate() {
            // Row 12 is `y1`'s first with six of its rows before it.
            assert_eq!(
                sel_w[1] > 0.0,
                i >= 12,
                "row {i}: own weight {own}, {sel_w:?}"
            );
            // `y1` alike under `[0, t]` and `[t, t]`, `y0` under `[0, t]`
            // and `[0, 0]`, to the bit.
            let pick = |r: &(f64, Vec<f64>, Vec<Vec<f64>>), j: usize| (r.1[j], r.2[j].clone());
            assert_eq!(pick(&lo[i], 1), pick(&hi[i], 1), "row {i}");
            assert_eq!(pick(&lo[i], 0), pick(&none[i], 0), "row {i}");
        }
        assert!(
            none[11].1[1] > 0.0,
            "with no threshold `y1` counts from the start"
        );
        // None, or one threshold of at least 0 per target.
        let with = |own: Vec<f64>| {
            let mut c = cfg(2, 2, vec![1.0]);
            c.target_min_weight = own;
            c.validate()
        };
        for bad in [vec![t], vec![t, -1.0], vec![f64::NAN, t]] {
            let err = with(bad).unwrap_err();
            assert!(err.contains("target_min_weight"), "{err}");
        }
        with(vec![0.0, f64::INFINITY]).unwrap();
    }

    /// Coordinate descent that runs out of sweeps before it meets `tol`
    /// is the failure this model has, and `solve_failures` counts it: one
    /// per target and path point left unconverged (review 2026-09-12, S11:
    /// the field was reported and never written).
    #[test]
    fn a_descent_that_runs_out_of_sweeps_is_a_solve_failure() {
        let mut c = cfg(4, 1, vec![0.5, 0.1, 0.01]);
        c.tol = 1e-14;
        c.max_iter = 1;
        let (short, _) = fit(c.clone(), 200, 11);
        assert!(
            short.solve_failures > 0,
            "one sweep cannot converge every point"
        );
        c.max_iter = 2000;
        let (long, _) = fit(c, 200, 11);
        assert_eq!(
            long.solve_failures, 0,
            "and enough sweeps converge them all"
        );
    }

    /// `predict`, `n_eff` and the selection read the window without the
    /// O(k²) view (review 2026-09-12, P1), and must read what it had, to the
    /// bit: the weights, and each target's selection errors as the view
    /// carried them.
    /// PLAN task 95. Under a `window`, `lam_selected` is the path point with
    /// the least EW squared out-of-sample error over the rows inside the
    /// window (the builder's docstring): each scored row at `w *
    /// 0.5^(age / select_half_life)`, its age counted from the last row
    /// learned, and the rows inside the window those at most `window`
    /// older; a row scores a target once the target's own weight inside the
    /// window, before the row, reaches `min_weight` (review 2026-10-05,
    /// CA3). Recomputed here from each row's own predictions, which `step`
    /// reports, it is held on every row where it is decided: not a tie,
    /// and not an empty window. Three departures moved it on 14 to 90 rows
    /// of 277 in `tests/test_oracles_lasso_paths.py`: the snapshot took the
    /// selection after the row's own error, with its weight aged twice; the
    /// choice read the window as it stood a row earlier; and the truncation
    /// aged by the model's half-life where `select_half_life` differs.
    #[test]
    fn penalty_selected_under_a_window_is_the_argmin_inside_it() {
        let path = vec![0.4, 0.1, 0.02, 0.0];
        let np = path.len();
        for (select, window) in [(Some(6.0), 12.0), (None, 12.0), (Some(40.0), 7.5)] {
            // Two targets, the second first seen at row 25 and missing on
            // its own rows after, so each target's errors are its own.
            let mut c = cfg(2, 2, path.clone());
            c.decay = Decay::Halflife(20.0);
            c.select_half_life = select;
            c.window = Some(window);
            c.max_rows_between_snapshots = Some(1);
            c.min_weight = 2.0;
            let sel_h = select.unwrap_or(20.0);
            let mut m = Lasso::new(c).unwrap();
            let mut s = 11u64;
            let (mut t, mut t_prev) = (0.0, 0.0);
            // Per target, (clock, weight, squared error per path point) of
            // each row that scored it, and (clock, weight) of each it was on:
            // a row scores it once that weight inside the window, before
            // the row, reaches `min_weight` (CA3).
            let mut scored: Vec<Vec<(f64, f64, Vec<f64>)>> = vec![Vec::new(), Vec::new()];
            let mut on: Vec<Vec<(f64, f64)>> = vec![Vec::new(), Vec::new()];
            let mut held = [0usize; 2];
            for i in 0..300usize {
                let x = [lcg(&mut s), lcg(&mut s)];
                // A fit that moves, so the penalty chosen moves with it.
                let slope = if (i / 60) % 2 == 0 { 1.0 } else { 0.1 };
                let y0 = (i % 9 != 4).then_some(slope * x[0] - 0.3 * x[1] + 0.3 * lcg(&mut s));
                let y1 = (i >= 25 && i % 7 != 3)
                    .then_some(0.5 * x[1] - slope * x[0] + 0.2 * lcg(&mut s));
                let y = [y0, y1];
                let d = match i {
                    0 => 0.0,
                    150 => 30.0,
                    _ => [0.25, 0.5, 1.0, 1.5][i % 4],
                };
                let w = if i % 13 == 6 {
                    0.0
                } else {
                    0.5 + ((i * 37) % 10) as f64 / 10.0
                };
                t += d;
                // The choice in force for this row: from the rows before it,
                // inside the window as it stood at the last of them.
                let mut want = [None; 2];
                for (j, rows) in scored.iter().enumerate() {
                    let mut sums = vec![0.0; np];
                    let mut any = false;
                    for (tj, wj, e2) in rows {
                        if *tj >= t_prev - window {
                            let om = wj * (-((t_prev - tj) / sel_h)).exp2();
                            for (acc, e) in sums.iter_mut().zip(e2) {
                                *acc += om * e;
                            }
                            any = true;
                        }
                    }
                    let mut order: Vec<usize> = (0..np).collect();
                    order.sort_by(|&a, &b| sums[a].total_cmp(&sums[b]));
                    if any && sums[order[1]] - sums[order[0]] > 1e-9 * sums[order[0]] {
                        want[j] = Some(path[order[0]]);
                    }
                }
                let own = on.iter().map(|rows| {
                    rows.iter()
                        .filter(|(tr, _)| *tr >= t_prev - window)
                        .map(|(tr, wr)| wr * (-((t_prev - tr) / 20.0)).exp2())
                        .sum::<f64>()
                });
                let own: Vec<f64> = own.collect();
                let out = m.step(&x, &y, d, w);
                let Some(Extra::Lasso { lam_selected }) = &out.extra else {
                    panic!("a lasso reports its selection");
                };
                for j in 0..2 {
                    if let Some(lam) = want[j] {
                        assert_eq!(
                            lam_selected[j], lam,
                            "select {select:?}, window {window}: row {i}, target {j}"
                        );
                        held[j] += 1;
                    }
                    if y[j].is_some() {
                        on[j].push((t, w));
                    }
                    if let (Some(yv), true) = (y[j], out.pred[j * np].is_finite())
                        && w > 0.0
                        && own[j] >= 2.0
                    {
                        let e2 = out.pred[j * np..(j + 1) * np]
                            .iter()
                            .map(|p| (yv - p).powi(2))
                            .collect();
                        scored[j].push((t, w, e2));
                    }
                }
                t_prev = t;
            }
            assert!(
                held.iter().all(|&h| h > 150),
                "select {select:?}: held {held:?}"
            );
        }
    }

    #[test]
    fn the_window_weights_and_errors_are_the_views_to_the_bit() {
        let mut c = cfg(2, 2, vec![0.1, 0.0]);
        c.decay = Decay::Halflife(15.0);
        c.window = Some(12.0);
        c.max_rows_between_snapshots = Some(3);
        c.min_weight = 3.0;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 73u64;
        for i in 0..120 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = (i % 5 != 2).then_some(x[0] - x[1]);
            let y1 = (i % 3 != 1).then_some(x[0] + 0.5);
            let d = match i {
                0 => 0.0,
                60 => 40.0,
                _ => 1.0,
            };
            m.step(&x, &[y0, y1], d, if i % 11 == 4 { 0.0 } else { 1.0 });
            let view = m.view();
            let live = (m.acc.cross.w, m.acc.wj.clone());
            let cheap = m.window_weights().unwrap_or_else(|| live.clone());
            let full = view
                .as_ref()
                .map_or_else(|| live.clone(), |v| (v.acc.cross.w, v.acc.wj.clone()));
            assert_eq!(cheap.0.to_bits(), full.0.to_bits(), "row {i}");
            assert_eq!(cheap.1, full.1, "row {i}");
            let errs = m.view_sel_err();
            for j in 0..2 {
                let want = errs
                    .as_ref()
                    .map_or_else(|| m.sel_err[j].clone(), |e| e[j].clone());
                if let Some(got) = m.window_sel_err(j) {
                    assert_eq!(got, want, "row {i} target {j}");
                }
            }
        }
    }

    /// A window with no scored row of a target keeps the lambda that target
    /// last chose: the whole-history errors the choice used to fall back on
    /// are the rows the window has dropped (review 2026-09-25, tasks 94-97,
    /// finding 2). Scored to clock 50, then absent for 60 units of a
    /// 24-unit window at half-life 40.
    #[test]
    fn an_empty_selection_window_keeps_the_choice() {
        let mut c = cfg(2, 1, vec![0.3, 0.03, 0.003]);
        c.decay = Decay::Halflife(40.0);
        c.window = Some(24.0);
        c.max_rows_between_snapshots = Some(1);
        c.min_weight = 3.0;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 11u64;
        let mut chosen = None;
        let mut empty_rows = 0;
        for i in 0..120 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i <= 50).then_some(0.7 * x[0] + 0.1 * lcg(&mut s));
            m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i == 50 {
                chosen = Some(m.sel_idx[0]);
            }
            if i > 50 + 24 + 2 {
                assert!(
                    m.window_sel_err(0).is_none(),
                    "row {i}: the window has no scored row"
                );
                assert_eq!(Some(m.sel_idx[0]), chosen, "row {i}: the choice stands");
                empty_rows += 1;
            }
        }
        assert!(empty_rows > 30, "{empty_rows}");
    }

    #[test]
    fn zero_penalty_matches_ols() {
        // lambda = 0 => the lasso solution is the OLS solution.
        let mut m = Lasso::new(cfg(3, 1, vec![0.0])).unwrap();
        let mut s = 5u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.75 * x[1] + 2.0 * x[2] + 0.25;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0][0];
        assert!((b[0] - 0.25).abs() < 1e-6, "intercept {}", b[0]);
        assert!((b[1] - 1.5).abs() < 1e-6);
        assert!((b[2] + 0.75).abs() < 1e-6);
        assert!((b[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn large_penalty_zeroes_coefficients() {
        let mut m = Lasso::new(cfg(2, 1, vec![100.0, 0.0])).unwrap();
        let mut s = 6u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap();
        assert_eq!(
            &b[0][0][1..],
            &[0.0, 0.0],
            "heavy penalty must zero features"
        );
        assert!((b[0][1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn selects_sparse_lambda_when_features_are_noise() {
        // y depends on x0 only; x1..x3 are noise. A middling penalty should beat
        // lambda = 0 on out-of-sample error, so selection must not pick 0.
        let mut c = cfg(4, 1, vec![0.5, 0.2, 0.05, 0.0]);
        c.decay = Decay::Halflife(500.0);
        c.min_weight = 20.0;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..1500 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 1.0 * x[0] + 0.9 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap();
        // at the largest penalty the noise features are zero
        assert_eq!(&b[0][0][2..], &[0.0, 0.0, 0.0]);
        // and the selected lambda is a real choice from the path
        assert!(m.cfg.lasso_path.contains(&m.lam_selected()[0]));
    }

    #[test]
    fn elastic_net_shrinks_less_sparsely() {
        let mut c = cfg(2, 1, vec![0.3]);
        c.l1_ratio = 0.5;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 8u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0][0];
        // elastic net with this penalty shrinks but does not zero out
        assert!(b[1] > 0.1 && b[1] < 1.0, "{}", b[1]);
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Lasso::new(cfg(2, 1, vec![0.5, 0.0])).unwrap();
        let mut s = 9u64;
        let rows: Vec<([f64; 2], f64)> = (0..80)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..40].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Lasso::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[40..] {
            let a = m1.step(x, &[Some(*y)], 1.0, 1.0);
            let b = m2.step(x, &[Some(*y)], 1.0, 1.0);
            assert_eq!(a.pred, b.pred);
            assert_eq!(a.extra, b.extra);
        }
    }

    /// N2, found while fixing N1: the lasso kept its cross-moments raw and
    /// centred them by subtraction, `E[z·y] − m·ȳ`, which loses `L²·ε` at a
    /// level `L`. It reads `ewridge`'s centred ones now (docs/PLAN.md task
    /// 81), so a level costs the path nothing: the same stream at the origin
    /// and shifted by `1e8`, features and target alike, gives the same slopes
    /// at every path point and predictions that differ by the shift. The
    /// tolerances are the data's resolution at `1e8`, `ulp ≈ 1.5e-8`.
    #[test]
    fn a_level_costs_the_path_nothing() {
        let run = |level: f64| {
            let mut c = cfg(2, 1, vec![0.05, 0.0]);
            c.decay = Decay::Halflife(200.0);
            c.min_weight = 10.0;
            let mut m = Lasso::new(c).unwrap();
            let mut s = 29u64;
            let mut preds = Vec::new();
            for i in 0..600 {
                let u = [lcg(&mut s), lcg(&mut s)];
                let x = [level + u[0], level + u[1]];
                let y = level + 2.0 * u[0] - u[1] + 0.1 * lcg(&mut s);
                let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                preds.extend(st.pred.iter().map(|p| p - level));
            }
            (preds, m.coefficients().unwrap()[0].clone())
        };
        let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
        for (li, (a, b)) in b0.iter().zip(&b8).enumerate() {
            for i in 1..3 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-6,
                    "path point {li}, slope {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
        let mut worst = 0.0f64;
        for (a, b) in p0.iter().zip(&p8) {
            assert_eq!(a.is_finite(), b.is_finite());
            if a.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "the predictions part by {worst}");
    }

    /// `own_rows` (docs/PLAN.md task 81): a target of a lasso bank is fitted
    /// on exactly its rows, as a lasso of that target alone is -- the Gram
    /// to the bit, and the path.
    #[test]
    fn under_own_rows_each_target_is_the_path_of_that_target_alone() {
        let c = cfg(2, 2, vec![0.1, 0.0]);
        let mut bank = Lasso::new(c.clone()).unwrap();
        let one = LassoCfg {
            n_targets: 1,
            ..c.clone()
        };
        let mut alone = [Lasso::new(one.clone()).unwrap(), Lasso::new(one).unwrap()];
        let mut s = 37u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + 2.0 * x[0] - x[1] + 0.05 * lcg(&mut s);
            let ys = [Some(y), (x[0] < 0.4).then_some(3.0 - y)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            bank.step(&x, &ys, d, 1.0);
            for (j, a) in alone.iter_mut().enumerate() {
                a.step(&x, &[ys[j]], d, 1.0);
            }
        }
        assert_eq!(bank.acc.grams.grams.len(), 2);
        for (j, a) in alone.iter().enumerate() {
            let g = &bank.acc.grams.grams[bank.acc.grams.of[j]];
            assert_eq!(g, &a.acc.grams.grams[0], "target {j}'s Gram");
            let (b, want) = (
                &bank.coefficients().unwrap()[j],
                &a.coefficients().unwrap()[0],
            );
            for (p, q) in b.iter().flatten().zip(want.iter().flatten()) {
                assert!(
                    (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                    "target {j}: {b:?} vs {want:?}"
                );
            }
        }
    }

    /// Runs are a window's (docs/PLAN.md task 128).
    #[test]
    fn only_a_windowed_lasso_keeps_runs() {
        for window in [None, Some(40.0)] {
            let mut c = cfg(2, 1, vec![0.1]);
            c.window = window;
            let m = Lasso::new(c).unwrap();
            assert_eq!(
                m.acc.grams.grams[0].keeps_runs(),
                window.is_some(),
                "window {window:?}"
            );
        }
    }

    /// The window's snapshot counts every vector it holds in its footprint,
    /// the cross-moments' low parts included (docs/PLAN.md task 130).
    /// Every target present on every row, so each snapshot holds one Gram.
    /// The path is a dimension too (of the selection errors), so the larger
    /// snapshot has a longer one: the check wants every vector to grow.
    #[test]
    fn the_window_footprint_counts_every_vector() {
        let snap = |k: usize, t: usize| {
            let path = if k > 2 {
                vec![0.3, 0.1, 0.0]
            } else {
                vec![0.1, 0.0]
            };
            let mut c = cfg(k, t, path);
            c.window = Some(12.0);
            let mut m = Lasso::new(c).unwrap();
            let mut s = 7u64;
            for i in 0..40 {
                let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
                let y: Vec<Option<f64>> = (0..t).map(|j| Some(x[j % k] + lcg(&mut s))).collect();
                m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
        };
        crate::window::assert_footprint_counts_every_vector(&snap(2, 1), &snap(5, 3), "lasso");
    }

    /// Two points of the path that zero every coefficient predict the same
    /// number, so their selection errors tie to the bit, and the choice
    /// stays with the first, the heaviest penalty (review 2026-09-26, C2: a
    /// mutant taking the later one was excused as unreachable).
    #[test]
    fn a_tie_in_the_selection_error_keeps_the_first_of_the_path() {
        let mut c = cfg(2, 1, vec![100.0, 50.0, 0.0]);
        c.select_half_life = Some(f64::INFINITY);
        let mut m = Lasso::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..100 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.3 * x[0] + lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(
            m.sel_err[0][0].to_bits(),
            m.sel_err[0][1].to_bits(),
            "the tie is real: {} vs {}",
            m.sel_err[0][0],
            m.sel_err[0][1]
        );
        assert_eq!(m.lam_selected(), vec![100.0]);
    }

    /// As `ewridge`'s: the runs follow the window on restore (review
    /// 2026-09-26, C4).
    #[test]
    fn restore_sets_the_runs_from_the_window() {
        let build = |window: Option<f64>| {
            let mut c = cfg(2, 1, vec![0.1, 0.0]);
            c.window = window;
            let mut m = Lasso::new(c).unwrap();
            let mut s = 3u64;
            for i in 0..30 {
                let x = [lcg(&mut s), lcg(&mut s)];
                m.step(
                    &x,
                    &[Some(x[0] - x[1])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            serde_json::to_value(m.state()).unwrap()
        };
        let restore = |v: serde_json::Value| Lasso::restore(&serde_json::from_value(v).unwrap());
        let mut v = build(None);
        crate::window::json_edit(&mut v, "runs", &mut |x| {
            *x = serde_json::json!({"x": [], "start": []});
        });
        assert!(
            restore(v)
                .unwrap()
                .acc
                .grams
                .grams
                .iter()
                .all(|g| !g.keeps_runs())
        );
        let mut v = build(Some(12.0));
        crate::window::json_edit(&mut v, "runs", &mut |x| {
            x["off"] = serde_json::json!(true);
        });
        assert!(matches!(restore(v), Err(StateError::Invalid(_))));
    }

    // --- the mutation baseline's survivors (docs/PLAN.md task 113) --------

    /// A weighted row: features, two targets, the weight.
    type Row = (Vec<f64>, [Option<f64>; 2], f64);

    /// Weighted rows: `k` features at levels, two targets, the second absent
    /// on every third row.
    fn weighted_rows(k: usize, n: usize, seed: u64) -> Vec<Row> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let x: Vec<f64> = (0..k)
                    .map(|j| 1.5 * lcg(&mut s) + 0.7 * j as f64 + 0.5)
                    .collect();
                let y0 = 1.0
                    + x.iter()
                        .enumerate()
                        .map(|(j, v)| (j as f64 - 0.8) * v)
                        .sum::<f64>()
                    + 0.2 * lcg(&mut s);
                let y1 = -0.5 + 0.9 * x[0] + 0.2 * lcg(&mut s);
                (
                    x,
                    [Some(y0), (i % 3 != 1).then_some(y1)],
                    0.5 + (lcg(&mut s) + 1.0),
                )
            })
            .collect()
    }

    fn run(c: LassoCfg, rows: &[Row]) -> Lasso {
        use crate::OnlineModel;
        let m = c.n_targets;
        let mut model = Lasso::new(c).unwrap();
        for (i, (x, y, w)) in rows.iter().enumerate() {
            model.step(x, &y[..m], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        model
    }

    /// With no intercept and no penalty the descent lands on least squares
    /// through the origin, `E[xx'] β = E[xy]`, whatever the features' means
    /// (review 2026-09-12, C8's form, now held to its closed form).
    #[test]
    fn without_an_intercept_and_no_penalty_it_is_least_squares_through_the_origin() {
        let mut c = cfg(3, 1, vec![0.0]);
        c.fit_intercept = false;
        c.min_weight = 0.0;
        c.max_iter = 100_000;
        c.tol = 1e-15;
        let rows = weighted_rows(3, 80, 3);
        let m = run(c, &rows);
        let ws: f64 = rows.iter().map(|r| r.2).sum();
        let e = |f: &dyn Fn(&Row) -> f64| rows.iter().map(|r| r.2 * f(r)).sum::<f64>() / ws;
        let a: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| e(&|r| r.0[i] * r.0[j])).collect())
            .collect();
        let b: Vec<f64> = (0..3).map(|i| e(&|r| r.0[i] * r.1[0].unwrap())).collect();
        let want = crate::oracle::solve(&a.concat(), &b);
        let got = &m.coefficients().unwrap()[0][0];
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-8 * (1.0 + w.abs()),
                "{got:?} against {want:?}"
            );
        }
    }

    /// A feature that never moves gets no slope and leaves the others at
    /// the fit without it: with no penalty, the moving feature's slope is
    /// `Cov(x, y) / Var(x)` and the intercept `ȳ − m·β`.
    #[test]
    fn a_held_feature_gets_no_slope_and_moves_nothing_else() {
        let mut c = cfg(2, 1, vec![0.0]);
        c.min_weight = 0.0;
        let mut rows = weighted_rows(2, 60, 7);
        for r in &mut rows {
            r.0[1] = 7.25;
        }
        let m = run(c, &rows);
        let ws: f64 = rows.iter().map(|r| r.2).sum();
        let e = |f: &dyn Fn(&Row) -> f64| rows.iter().map(|r| r.2 * f(r)).sum::<f64>() / ws;
        let (mx, my) = (e(&|r| r.0[0]), e(&|r| r.1[0].unwrap()));
        let slope = e(&|r| (r.0[0] - mx) * (r.1[0].unwrap() - my)) / e(&|r| (r.0[0] - mx).powi(2));
        let got = &m.coefficients().unwrap()[0][0];
        assert_eq!(got[2], 0.0, "{got:?}");
        assert!(
            (got[1] - slope).abs() <= 1e-9 * slope.abs(),
            "{got:?} against {slope}"
        );
        assert!(
            (got[0] - (my - mx * slope)).abs() <= 1e-9 * (1.0 + my.abs()),
            "{got:?}"
        );
    }

    /// Under `pairwise` a target's intercept is centred on its own rows'
    /// feature means, not the Gram's (`EwRidge` holds the same).
    #[test]
    fn a_pairwise_intercept_is_centred_on_the_targets_own_rows() {
        let mut c = cfg(2, 2, vec![0.05]);
        c.target_gaps = TargetGaps::Pairwise;
        c.min_weight = 0.0;
        let rows = weighted_rows(2, 60, 11);
        let m = run(c, &rows);
        let beta = &m.coefficients().unwrap()[1][0];
        let own: Vec<_> = rows.iter().filter(|r| r.1[1].is_some()).collect();
        let w: f64 = own.iter().map(|r| r.2).sum();
        let my = own.iter().map(|r| r.2 * r.1[1].unwrap()).sum::<f64>() / w;
        let mx: Vec<f64> = (0..2)
            .map(|i| own.iter().map(|r| r.2 * r.0[i]).sum::<f64>() / w)
            .collect();
        let want = my - mx[0] * beta[1] - mx[1] * beta[2];
        assert!(
            (beta[0] - want).abs() <= 1e-10 * (1.0 + want.abs()),
            "{} against {want}",
            beta[0]
        );
    }

    /// The descent starts from the last solve's answer: on the same data it
    /// is already there, and one sweep finds nothing to move.
    #[test]
    fn a_re_solve_starts_where_the_last_one_ended() {
        let mut c = cfg(4, 1, vec![0.01]);
        c.min_weight = 0.0;
        let rows = weighted_rows(4, 80, 13);
        let mut m = run(c, &rows);
        let before = m.solve_failures;
        m.cfg.max_iter = 1;
        m.cfg.tol = 1e-9;
        m.solve();
        assert_eq!(m.solve_failures, before, "the warm start was not used");
    }

    /// An empty window leaves the fit NaN, and the next solve does not start
    /// from it: every coefficient is a number again once rows return.
    #[test]
    fn a_fit_after_an_empty_window_does_not_start_from_its_nan() {
        use crate::OnlineModel;
        let mut c = cfg(2, 1, vec![0.01]);
        c.window = Some(10.0);
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 0.0;
        let mut m = Lasso::new(c).unwrap();
        let rows = weighted_rows(2, 60, 17);
        for (i, (x, y, w)) in rows.iter().take(30).enumerate() {
            m.step(x, &y[..1], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        // A zero-weight row after a gap longer than the window: nothing is in
        // the window, and the fit says so.
        m.step(&rows[30].0, &[None], 50.0, 0.0);
        let emptied = m.coefficients().unwrap()[0][0].clone();
        assert!(
            emptied.iter().any(|v| v.is_nan()),
            "the window was not emptied: {emptied:?}"
        );
        for (x, y, w) in rows.iter().skip(31) {
            m.step(x, &y[..1], 1.0, *w);
        }
        let back = &m.coefficients().unwrap()[0][0];
        assert!(back.iter().all(|v| v.is_finite()), "{back:?}");
    }

    #[test]
    fn a_window_budget_reports_its_overrun_and_clears() {
        use crate::OnlineModel;
        let mut c = cfg(2, 1, vec![0.01]);
        c.window = Some(40.0);
        c.max_rows_between_snapshots = Some(2);
        c.decay = Decay::Halflife(30.0);
        c.min_weight = 0.0;
        let rows = weighted_rows(2, 100, 19);
        let mut m = run(c, &rows);
        assert_eq!(m.window_over_budget(), None);
        m.set_window_budget(Some(crate::WindowBudget::Refuse(1e-6)));
        match m.window_over_budget() {
            Some((bytes, every)) => {
                assert!(bytes > 1, "{bytes}");
                assert_eq!(every, crate::Cadence::of(None, Some(2)));
            }
            None => panic!("a ring of snapshots is past a budget of one byte"),
        }
        m.set_window_budget(None);
        assert_eq!(m.window_over_budget(), None);
    }

    /// A coefficient vector of the wrong length inside a path of the right
    /// length is refused, not loaded.
    #[test]
    fn a_fit_of_the_wrong_inner_shape_is_refused() {
        use crate::{ModelState, OnlineModel};
        let mut c = cfg(2, 1, vec![0.1, 0.01]);
        c.min_weight = 0.0;
        let m = run(c, &weighted_rows(2, 20, 23));
        let mut s = m.state();
        let ModelState::Lasso(inner) = &mut s.model else {
            unreachable!()
        };
        inner.beta.as_mut().unwrap()[0][1].pop();
        assert!(Lasso::restore(&s).is_err());
    }

    /// Without an intercept a feature that is all zeros has no scale: it
    /// gets no slope, and the rest is least squares through the origin on
    /// the others.
    #[test]
    fn without_an_intercept_an_all_zero_feature_moves_nothing() {
        let mut c = cfg(3, 1, vec![0.0]);
        c.fit_intercept = false;
        c.min_weight = 0.0;
        c.max_iter = 100_000;
        c.tol = 1e-15;
        let mut rows = weighted_rows(3, 80, 29);
        for r in &mut rows {
            r.0[1] = 0.0;
        }
        let m = run(c, &rows);
        let ws: f64 = rows.iter().map(|r| r.2).sum();
        let e = |f: &dyn Fn(&Row) -> f64| rows.iter().map(|r| r.2 * f(r)).sum::<f64>() / ws;
        let keep = [0usize, 2];
        let a: Vec<Vec<f64>> = keep
            .iter()
            .map(|&i| keep.iter().map(|&j| e(&|r| r.0[i] * r.0[j])).collect())
            .collect();
        let b: Vec<f64> = keep
            .iter()
            .map(|&i| e(&|r| r.0[i] * r.1[0].unwrap()))
            .collect();
        let want = crate::oracle::solve(&a.concat(), &b);
        let got = &m.coefficients().unwrap()[0][0];
        assert_eq!(got[1], 0.0, "{got:?}");
        for (g, w) in [got[0], got[2]].iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-8 * (1.0 + w.abs()),
                "{got:?} against {want:?}"
            );
        }
    }

    /// What the bank exports from a fit: the Grams with their readers, each
    /// target's raw cross-moments `E[z·y_j]` and weight over its own rows,
    /// the target moments, and the count of targets.
    #[test]
    fn the_exported_moments_are_each_targets_own() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2, vec![0.01]);
        c.min_weight = 0.0;
        let rows = weighted_rows(2, 40, 31);
        let m = run(c, &rows);
        assert_eq!(m.n_targets(), 2);
        let (parts, moments) = m.gram_parts();
        assert_eq!(parts.len(), 2);
        assert!(moments.is_some() && m.target_moments().is_some());
        let mut out = Vec::new();
        assert!(m.target_n_eff_into(&mut out));
        let cross = m.cross_moments();
        for t in 0..2 {
            let own: Vec<_> = rows.iter().filter(|r| r.1[t].is_some()).collect();
            let w: f64 = own.iter().map(|r| r.2).sum();
            assert!((m.target_weights()[t] - w).abs() <= 1e-12 * w);
            assert!((out[t] - w).abs() <= 1e-12 * w);
            assert_eq!(cross[t].len(), 3);
            for (i, &got) in cross[t].iter().enumerate() {
                let z = |r: &&Row| if i == 0 { 1.0 } else { r.0[i - 1] };
                let want = own
                    .iter()
                    .map(|r| r.2 * z(r) * r.1[t].unwrap())
                    .sum::<f64>()
                    / w;
                assert!(
                    (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
                    "target {t}, slot {i}"
                );
            }
        }
    }

    /// The schedule: every `max_rows_between_solves` rows, every
    /// `solve_every` of clock, and the first fit as soon as `min_weight` is
    /// met -- counted as the rows on which the fit changed.
    #[test]
    fn the_solve_schedule_counts_rows_and_clock() {
        use crate::OnlineModel;
        let changes = |solve_every: f64, max_rows: u32| {
            let mut c = cfg(2, 1, vec![0.01]);
            c.min_weight = 3.0;
            c.solve_every = solve_every;
            c.max_rows_between_solves = max_rows;
            let mut m = Lasso::new(c).unwrap();
            let (mut last, mut n) = (None::<Vec<f64>>, 0);
            for (i, (x, y, _)) in weighted_rows(2, 30, 37).iter().enumerate() {
                m.step(x, &y[..1], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                let now = m.coefficients().map(|b| b[0][0].clone());
                if now != last {
                    n += 1;
                    last = now;
                }
            }
            n
        };
        // The first fit at row 3 (3 rows of weight 1), then every 4 rows:
        // rows 3, 7, ..., 27.
        assert_eq!(changes(1e9, 4), 7);
        // Every 5 units of clock, one a row: rows 3, 8, 13, ..., 28.
        assert_eq!(changes(5.0, 1_000_000), 6);
    }

    #[test]
    fn a_config_without_features_or_targets_is_refused() {
        assert!(Lasso::new(cfg(0, 1, vec![0.1])).is_err());
        assert!(Lasso::new(cfg(2, 0, vec![0.1])).is_err());
        assert!(Lasso::new(cfg(2, 1, vec![0.1])).is_ok());
    }

    /// Through the bytes a file holds, and read as schema 17 (own means,
    /// already): a loaded model goes on exactly as the one saved.
    #[test]
    fn a_saved_fit_goes_on_as_it_would_have() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2, vec![0.05, 0.01]);
        c.target_gaps = TargetGaps::Pairwise;
        c.min_weight = 0.0;
        let rows = weighted_rows(2, 60, 41);
        let m = run(c, &rows[..40]);
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let mut s17: crate::State = rmp_serde::from_slice(&bytes).unwrap();
        let mut loaded = Lasso::restore(&s17).unwrap();
        s17.schema_version = 17;
        let mut read17 = Lasso::restore(&s17).unwrap();
        let mut whole = m;
        for (x, y, w) in &rows[40..] {
            let a = whole.step(x, &y[..], 1.0, *w).pred;
            let b = loaded.step(x, &y[..], 1.0, *w).pred;
            let c = read17.step(x, &y[..], 1.0, *w).pred;
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b));
            assert_eq!(bits(&a), bits(&c));
        }
    }

    /// The empty window's NaN fit is no start at all: a descent from it is
    /// a descent from zero, sweep for sweep, as a model with no fit takes.
    #[test]
    fn a_nan_fit_is_started_from_zero_like_no_fit() {
        use crate::OnlineModel;
        let mut c = cfg(3, 1, vec![0.01]);
        c.window = Some(10.0);
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 0.0;
        let mut m = Lasso::new(c).unwrap();
        let rows = weighted_rows(3, 60, 43);
        for (i, (x, y, w)) in rows.iter().take(30).enumerate() {
            m.step(x, &y[..1], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        m.step(&rows[30].0, &[None], 50.0, 0.0);
        for (x, y, w) in rows.iter().skip(31).take(5) {
            m.step(x, &y[..1], 1.0, *w);
        }
        m.cfg.max_iter = 1;
        m.beta.as_mut().unwrap()[0][0].fill(f64::NAN);
        let mut cold = m.clone();
        cold.beta = None;
        m.solve();
        cold.solve();
        assert_eq!(
            m.coefficients().unwrap()[0][0],
            cold.coefficients().unwrap()[0][0]
        );
    }

    /// A target no row has taught predicts nothing, from `predict` as from
    /// `step`: its fit is the empty system's, not an estimate.
    #[test]
    fn a_target_with_no_rows_predicts_nothing() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2, vec![0.1, 0.01]);
        c.min_weight = 0.0;
        let mut m = Lasso::new(c).unwrap();
        for (i, (x, y, w)) in weighted_rows(2, 20, 47).iter().enumerate() {
            m.step(x, &[y[0], None], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        let pred = m.predict(&[0.3, 1.2], 1.0).pred;
        assert!(pred[..2].iter().all(|v| v.is_finite()), "{pred:?}");
        assert!(pred[2..].iter().all(|v| v.is_nan()), "{pred:?}");
    }

    // --- the weekly pass's survivors (docs/PLAN.md task 158) --------------

    /// `solve_share` is a positive, finite fraction: 0, a negative, `inf`
    /// and NaN are refused, and a fraction accepted.
    #[test]
    fn solve_share_must_be_finite_and_positive() {
        let with = |f: f64| {
            let mut c = cfg(2, 1, vec![0.1]);
            c.solve_share = Some(f);
            c.validate()
        };
        for bad in [0.0, -0.5, f64::INFINITY, f64::NAN] {
            let err = with(bad).unwrap_err();
            assert!(err.contains("solve_share"), "{bad}: {err}");
        }
        with(0.25).unwrap();
    }

    /// `max_iter = 0` is refused: the descent never ran, every solve was a
    /// failure, and the output looked like a fit -- the intercept at the
    /// mean and every slope 0 (review 2026-10-05, PB7). One sweep is
    /// accepted.
    #[test]
    fn cfg_validation_refuses_a_descent_of_no_sweeps() {
        let with = |max_iter: u32| {
            let mut c = cfg(2, 1, vec![0.1]);
            c.max_iter = max_iter;
            c.validate()
        };
        let err = with(0).unwrap_err();
        assert!(err.contains("max_iter must be >= 1"), "{err}");
        assert!(Lasso::new(cfg(2, 1, vec![0.1])).is_ok());
        with(1).unwrap();
    }

    /// The share the model runs at is the one it is given, by its cfg or
    /// set on it later.
    #[test]
    fn the_solve_share_is_the_one_set() {
        use crate::OnlineModel;
        let mut c = cfg(2, 1, vec![0.1]);
        c.solve_share = Some(0.25);
        let mut m = Lasso::new(c).unwrap();
        assert_eq!(m.solve_share(), Some(0.25));
        m.set_solve_share(Some(0.75));
        assert_eq!(m.solve_share(), Some(0.75));
        m.set_solve_share(None);
        assert_eq!(m.solve_share(), None);
    }

    /// Under `solve_share` a solve is due once the weight learned since the
    /// last one reaches that share of `n_eff` (the row's own included), and
    /// at the first row that meets `min_weight`; only a positive, finite
    /// weight counts toward it. Held to that rule kept beside the model,
    /// at weights that move and with a decay (`max_rows_between_solves` out
    /// of reach).
    #[test]
    fn solve_share_solves_when_the_weight_since_reaches_its_share() {
        use crate::OnlineModel;
        let mut c = cfg(2, 1, vec![0.1]);
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 0.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_share = Some(0.3);
        let mut m = Lasso::new(c).unwrap();
        let (mut w_sum, mut since, mut first) = (0.0f64, 0.0f64, true);
        let (mut s, mut solves) = (109u64, 0);
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let w = if i % 7 == 3 {
                0.0
            } else {
                0.25 + (lcg(&mut s) + 1.0)
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.step(&x, &[Some(x[0] - x[1])], d, w);
            w_sum = 0.5f64.powf(d / 20.0) * w_sum + w;
            since += w;
            let due = since >= 0.3 * w_sum || first;
            assert_eq!(m.rows_since_solve == 0, due, "row {i}");
            if due {
                (since, first, solves) = (0.0, false, solves + 1);
            }
        }
        assert!((20..150).contains(&solves), "{solves} solves");
        // A weight that is not finite does not count toward the share.
        let before = m.weight_since_solve;
        m.step(&[0.1, 0.2], &[Some(0.0)], 1.0, f64::INFINITY);
        assert_eq!(m.weight_since_solve, before);
    }

    /// The ring's shadow is the ring: learning the rows the model learns, it
    /// is past a refusing budget exactly when the model's ring is (docs/PLAN.md
    /// task 115 (d)); a model without a window has none.
    #[test]
    fn the_window_shadow_follows_the_ring() {
        use crate::OnlineModel;
        let plain = Lasso::new(cfg(2, 1, vec![0.01])).unwrap();
        assert!(plain.window_shadow().is_none());
        let mut c = cfg(2, 1, vec![0.01]);
        c.window = Some(40.0);
        c.max_rows_between_snapshots = Some(2);
        c.decay = Decay::Halflife(30.0);
        c.min_weight = 0.0;
        let rows = weighted_rows(2, 30, 117);
        let mut m = run(c, &rows[..10]);
        m.set_window_budget(Some(crate::WindowBudget::Refuse(0.002)));
        let mut shadow = m.window_shadow().expect("a windowed model has a shadow");
        let mut over = 0;
        for (x, y, w) in &rows[10..] {
            shadow.learn(1.0, None);
            m.step(x, &y[..1], 1.0, *w);
            assert_eq!(shadow.over_budget(), m.window_over_budget());
            over += usize::from(m.window_over_budget().is_some());
        }
        assert!(over > 0, "the ring went past its budget");
    }

    /// Task 180: `solve_every` reads the stamps its caller hands: every
    /// 2,000th 1 ms row under 2 s after the first solve, where the clock
    /// summed since the last solve, read when none is handed, is a row late
    /// each time, as it always was.
    #[test]
    fn solve_every_measures_the_stamps_it_is_handed() {
        use crate::OnlineModel;
        for (stamped, want) in [(true, [0, 2000, 4000]), (false, [0, 2001, 4002])] {
            let mut c = cfg(1, 1, vec![0.0]);
            c.min_weight = 0.0;
            c.solve_every = 2.0;
            c.max_rows_between_solves = u32::MAX;
            let mut m = Lasso::new(c).unwrap();
            let got = crate::since::events_on_millisecond_rows(4_100, stamped, |i, stamp, d| {
                if let Some(s) = stamp {
                    m.stamp_next(s);
                }
                let x = (i % 7) as f64;
                m.step(&[x], &[Some(1.0 + 2.0 * x)], d, 1.0);
                m.rows_since_solve == 0
            });
            assert_eq!(got, want, "stamped: {stamped}");
        }
    }
}
