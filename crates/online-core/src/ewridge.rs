//! EW-ridge on sufficient statistics (docs/PLAN.md §4.1) — the workhorse.
//!
//! Math (per row, decay factor `lam = decay.factor(d_clock)`, weight `w`,
//! `z = [1, x]` when the intercept is configured):
//!
//! ```text
//! W'   = lam W + w                          (n_eff)
//! S'   = (lam W S + w z z^T) / W'           (EW mean of z z^T; shared)
//! W_j' = lam W_j + w                        (only when y_j present)
//! r_j' = (lam W_j r_j + w z y_j) / W_j'     (per target)
//! ```
//!
//! Solve (per grid combo = feature set x ridge value), scheduled by
//! `solve_every` clock units / `max_rows_between_solves`:
//! - plain:        `(S + ridge D) beta = r_j`, D = I minus the intercept slot;
//! - standardized: centered stats scaled to correlation form, solved, unscaled,
//!   intercept recovered as `ybar - m . beta`; ~zero-variance features dropped;
//! - ridge_decay:  `(W S + prior_scale * ridge I) beta = W r_j` — a decaying
//!   prior on the sum scale, penalizing the intercept: exactly classic RLS
//!   regularization (used by the RLS agreement test, task 9).
//!
//! Predictions use the last solved coefficients (out-of-sample by construction:
//! the solve happens after the row's update, the pred before it).

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{dot_aug, solve_spd};
use crate::{Decay, EwCov, TargetMoments};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwRidgeCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Ridge grid, expanded at solve time; length >= 1.
    pub ridge: Vec<f64>,
    /// Named subsets of feature indices (0-based, excluding the intercept).
    /// Empty means one full set named "all".
    pub feature_sets: Vec<(String, Vec<usize>)>,
    pub standardize: bool,
    /// Decaying sum-scale prior (classic RLS regularization). Incompatible with
    /// `standardize` and with grids; the intercept is penalized.
    pub ridge_decay: bool,
    /// Shrink toward these coefficients instead of toward zero
    /// (ENHANCEMENTS E15): the solve becomes `(S + ridge·D)β = r + ridge·D·β₀`.
    /// One vector per target, each `k_total` long, in the features' original
    /// units; the intercept slot is unpenalized and therefore ignored.
    ///
    /// **Whether the prior fades depends on `ridge_decay`, and the difference
    /// matters.** `S` here is a weighted *mean*, not a sum, so it does not grow
    /// with the sample: a plain `ridge` is a fixed per-observation penalty and
    /// its pull toward `coef0` is **permanent** — "always stay near this
    /// belief". With `ridge_decay` the prior sits on the sum scale and its
    /// weight decays with the data, which is the usual warm start: "begin at
    /// yesterday's fit and let evidence take over".
    #[serde(default)]
    pub coef0: Option<Vec<Vec<f64>>>,
    /// Blend toward a slow-moving twin on a session change, instead of the
    /// all-or-nothing choice between `session_gap` and a full reset
    /// (ENHANCEMENTS E6, PLAN §12 open question 1).
    ///
    /// A second accumulator runs alongside the main one with `long_halflife`,
    /// representing the long-run relationship. On a session boundary the two
    /// are mixed, weight-respectingly, with the slow one taking share
    /// `session_shrink`:
    ///
    /// ```text
    /// W'  = (1−f)·W_fast + f·W_slow
    /// S'  = ((1−f)·W_fast·S_fast + f·W_slow·S_slow) / W'
    /// ```
    ///
    /// so `0` keeps today's fit, `1` reverts fully to the long run, and
    /// anything between says "overnight, drift partway back". Unlike
    /// `session_gap` this changes *what the model believes*, not merely how
    /// confident it is.
    #[serde(default)]
    pub session_shrink: Option<f64>,
    /// Halflife of the slow twin. Required by `session_shrink`.
    #[serde(default)]
    pub long_halflife: Option<f64>,
    /// Outputs are null until `n_eff` (before the row's update) reaches this.
    pub min_periods: f64,
    /// Solve cadence in clock units; <= 0 solves every row.
    pub solve_every: f64,
    /// Row cap between solves; 1 solves every row.
    pub max_rows_between_solves: u32,
}

impl EwRidgeCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    /// (feature-set index, ridge index) pairs in output order.
    fn combos(&self) -> Vec<(usize, usize)> {
        let nf = self.feature_sets.len().max(1);
        let nr = self.ridge.len();
        (0..nf).flat_map(|f| (0..nr).map(move |r| (f, r))).collect()
    }

    pub fn n_combos(&self) -> usize {
        self.feature_sets.len().max(1) * self.ridge.len()
    }

    /// Human-readable combo labels, used by the bank for output field names.
    pub fn combo_labels(&self) -> Vec<String> {
        let fs: Vec<&str> = if self.feature_sets.is_empty() {
            vec!["all"]
        } else {
            self.feature_sets.iter().map(|(n, _)| n.as_str()).collect()
        };
        let mut out = Vec::new();
        for f in &fs {
            for r in &self.ridge {
                out.push(format!("{f}_r{r}"));
            }
        }
        out
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.ridge.is_empty() {
            return Err("ridge grid must have at least one value".into());
        }
        if self.ridge_decay && (self.standardize || self.n_combos() > 1) {
            return Err("ridge_decay is incompatible with standardize and grids".into());
        }
        match (self.session_shrink, self.long_halflife) {
            (Some(f), _) if !(0.0..=1.0).contains(&f) => {
                return Err("session_shrink must be in [0, 1]".into());
            }
            (Some(_), None) => {
                return Err("session_shrink needs long_halflife (the slow twin's decay)".into());
            }
            (None, Some(_)) => {
                return Err("long_halflife has no effect without session_shrink".into());
            }
            (Some(_), Some(h)) if h <= 0.0 || h.is_nan() => {
                return Err("long_halflife must be > 0".into());
            }
            _ => {}
        }
        if let Some(c) = &self.coef0 {
            if c.len() != self.n_targets || c.iter().any(|v| v.len() != self.k_total()) {
                return Err(format!(
                    "coef0 must be {} vector{} of length {}",
                    self.n_targets,
                    if self.n_targets == 1 { "" } else { "s" },
                    self.k_total()
                ));
            }
            if c.iter().flatten().any(|v| !v.is_finite()) {
                return Err("coef0 values must be finite".into());
            }
        }
        for (name, idx) in &self.feature_sets {
            if idx.is_empty() || idx.iter().any(|&i| i >= self.n_features) {
                return Err(format!("feature set {name:?} has out-of-range indices"));
            }
        }
        Ok(())
    }
}

/// The long-run twin's accumulators (see [`EwRidgeCfg::session_shrink`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SlowState {
    cov: EwCov,
    wj: Vec<f64>,
    r: Vec<Vec<f64>>,
    /// The twin's own target moments, so a blend mixes two complete sets
    /// (docs/ENHANCEMENTS.md E45). `None` in a state written before task 38.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tm: Option<TargetMoments>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwRidge {
    cfg: EwRidgeCfg,
    cov: EwCov,
    /// Per-target EW weight sums and cross-moment means (see module docs).
    wj: Vec<f64>,
    r: Vec<Vec<f64>>,
    /// Per-target EW residual variance and its weight sum.
    wsig: Vec<f64>,
    sig2: Vec<f64>,
    /// Per-target mean, variance and `Sum w^2`, the other half of the
    /// sufficient statistic the Gram export hands back
    /// (docs/ENHANCEMENTS.md E45). `None` in a state written before task 38:
    /// the moments cannot be reconstructed from the cross-moments, so such a
    /// state reports `None` rather than a partial answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tm: Option<TargetMoments>,
    /// Last solved coefficients per output slot (target-major, then combo),
    /// each of length `k_total` (zeros outside a combo's feature set).
    beta: Option<Vec<Vec<f64>>>,
    /// Slow-moving twin for `session_shrink`: the same statistics under
    /// `long_halflife`, representing the long-run relationship.
    #[serde(default)]
    slow: Option<Box<SlowState>>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    pub solve_failures: u64,
    // scratch buffers (serialized for simplicity; tiny)
    #[serde(skip)]
    zbuf: Vec<f64>,
}

impl EwRidge {
    pub fn new(cfg: EwRidgeCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k_total = cfg.k_total();
        let m = cfg.n_targets;
        let slow = cfg.session_shrink.map(|_| {
            Box::new(SlowState {
                cov: EwCov::new(k_total),
                wj: vec![0.0; m],
                r: vec![vec![0.0; k_total]; m],
                tm: Some(TargetMoments::new(m)),
            })
        });
        Ok(Self {
            slow,
            cov: EwCov::new(k_total),
            wj: vec![0.0; m],
            r: vec![vec![0.0; k_total]; m],
            tm: Some(TargetMoments::new(m)),
            wsig: vec![0.0; m],
            sig2: vec![0.0; m],
            beta: None,
            clock_since_solve: 0.0,
            rows_since_solve: 0,
            solve_failures: 0,
            zbuf: vec![0.0; k_total],
            cfg,
        })
    }

    pub fn cfg(&self) -> &EwRidgeCfg {
        &self.cfg
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    pub fn n_eff(&self) -> f64 {
        self.cov.n_eff()
    }

    /// Current coefficients per output slot, if solved.
    /// The feature accumulator: EW means and the centered co-moment matrix
    /// over `k_total` columns (the intercept column included when
    /// `add_intercept`, where it is constant 1 and so has zero variance).
    /// See [`EwCov::comoments`] (docs/ENHANCEMENTS.md E30).
    pub fn cov(&self) -> &EwCov {
        &self.cov
    }

    /// Per-target **uncentered** cross-moments `r[t]`, each `k_total` long:
    /// the EW mean of `z·y_t`, where `z` is the feature row with the intercept
    /// slot as a constant 1.
    ///
    /// Uncentered deliberately — it is what the solve consumes, paired with
    /// the *raw* second moment. Mixing it with the centered
    /// [`EwCov::comoments`] silently gives the wrong coefficients; the
    /// identity that holds is
    /// `(comoments + means⊗means) · beta == cross_moments`.
    pub fn cross_moments(&self) -> &[Vec<f64>] {
        &self.r
    }

    /// Per-target accumulated weight, the denominator behind
    /// [`Self::cross_moments`]. This is `n_eff` *per target*, which differs
    /// from the shared [`Self::n_eff`] when targets have different null
    /// patterns.
    pub fn target_weights(&self) -> &[f64] {
        &self.wj
    }

    /// Per-target mean, variance and `Sum w^2` -- the other half of what a
    /// saved Gram needs (docs/ENHANCEMENTS.md E45). `None` for a state
    /// written before task 38; see the field.
    pub fn target_moments(&self) -> Option<&TargetMoments> {
        self.tm.as_ref()
    }

    pub fn coefficients(&self) -> Option<&[Vec<f64>]> {
        self.beta.as_deref()
    }

    /// Re-solve and return the first target's first slope. Test helper: after
    /// a blend the coefficients are stale until the next solve.
    #[cfg(test)]
    pub(crate) fn coefficients_after_blend(&mut self) -> f64 {
        self.solve();
        self.coefficients().unwrap()[0][1]
    }

    /// Mix the main accumulators toward the slow twin, as a session boundary
    /// asks for (see [`EwRidgeCfg::session_shrink`]). A no-op when the twin is
    /// not configured, or before it has seen anything.
    pub fn blend_toward_long_run(&mut self) {
        let Some(f) = self.cfg.session_shrink else {
            return;
        };
        let Some(slow) = &self.slow else { return };
        if f <= 0.0 {
            return;
        }
        let k = self.cfg.k_total();

        // Weight-respecting mixture of two weighted means.
        let (wf, ws) = (self.cov.n_eff(), slow.cov.n_eff());
        let w_new = (1.0 - f) * wf + f * ws;
        if w_new > 0.0 {
            let mut blended = EwCov::new(k);
            let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
            // EwCov holds means, so mix means directly and restore the weight
            // by replaying a single synthetic observation is not possible;
            // instead rebuild from the mixed moments.
            let mixed_mean: Vec<f64> = (0..k)
                .map(|i| af * self.cov.mean(i) + as_ * slow.cov.mean(i))
                .collect();
            let mut mixed_c = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    // Mix raw second moments, then re-center on the mixed mean:
                    // centered moments are not additive across differing means.
                    let raw = af * self.cov.raw(i, j) + as_ * slow.cov.raw(i, j);
                    mixed_c[i * k + j] = raw - mixed_mean[i] * mixed_mean[j];
                }
            }
            // `Q` mixes by the same coefficients as the moments; see
            // `TargetMoments::blend` for why a union would be wrong here.
            let q = match (self.cov.q_sum(), slow.cov.q_sum()) {
                (Some(qf), Some(qs)) => Some(af * qf + as_ * qs),
                _ => None,
            };
            blended.set_moments(&mixed_mean, &mixed_c, w_new, q);
            self.cov = blended;
        }

        // A twin restored from a pre-task-38 state has no target moments, and
        // the mixture of a set with nothing is nothing (E45).
        if slow.tm.is_none() {
            self.tm = None;
        }
        for j in 0..self.cfg.n_targets {
            let (wf, ws) = (self.wj[j], slow.wj[j]);
            let w_new = (1.0 - f) * wf + f * ws;
            if w_new > 0.0 {
                let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
                for i in 0..k {
                    self.r[j][i] = af * self.r[j][i] + as_ * slow.r[j][i];
                }
                self.wj[j] = w_new;
                if let (Some(tm), Some(stm)) = (self.tm.as_mut(), slow.tm.as_ref()) {
                    tm.blend(stm, j, af, as_);
                }
            }
        }
    }

    fn z(&mut self, x: &[f64]) {
        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }
    }

    /// Indices into z for a combo's feature set (intercept first if configured).
    fn combo_z_indices(&self, fs_idx: usize) -> Vec<usize> {
        let off = usize::from(self.cfg.add_intercept);
        let mut idx: Vec<usize> = if self.cfg.feature_sets.is_empty() {
            (0..self.cfg.n_features).map(|i| i + off).collect()
        } else {
            self.cfg.feature_sets[fs_idx]
                .1
                .iter()
                .map(|&i| i + off)
                .collect()
        };
        if self.cfg.add_intercept {
            idx.insert(0, 0);
        }
        idx
    }

    fn solve(&mut self) {
        let k_total = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let combos = self.cfg.combos();
        let mut beta = vec![vec![0.0; k_total]; m * combos.len()];

        for (ci, &(fs_idx, r_idx)) in combos.iter().enumerate() {
            let ridge = self.cfg.ridge[r_idx];
            let zidx = self.combo_z_indices(fs_idx);
            let kc = zidx.len();

            // Gather the sub-block of S and the per-target rhs.
            let mut a = vec![0.0; kc * kc];
            for (ai, &zi) in zidx.iter().enumerate() {
                for (aj, &zj) in zidx.iter().enumerate() {
                    a[ai * kc + aj] = self.cov.raw(zi, zj);
                }
            }
            let mut b = vec![0.0; kc * m];
            for j in 0..m {
                for (ai, &zi) in zidx.iter().enumerate() {
                    b[j * kc + ai] = self.r[j][zi];
                }
            }

            let solved: Option<Vec<f64>> = if self.cfg.ridge_decay {
                // (W S + prior_scale * ridge I) beta = W r  — intercept penalized.
                let w = self.cov.n_eff();
                let ps = self.cov.prior_scale();
                for i in 0..kc {
                    for j in 0..kc {
                        a[i * kc + j] *= w;
                    }
                    a[i * kc + i] += ps * ridge;
                }
                for v in b.iter_mut() {
                    *v *= w;
                }
                // Warm start: the prior enters on the same decaying sum scale,
                // so its weight falls away as data accumulates.
                if let Some(c0) = &self.cfg.coef0 {
                    for j in 0..m {
                        for (ai, &zi) in zidx.iter().enumerate() {
                            b[j * kc + ai] += ps * ridge * c0[j][zi];
                        }
                    }
                }
                self.run_solve(&a, &b, kc, m)
            } else if !self.cfg.standardize {
                let off = usize::from(self.cfg.add_intercept);
                for i in off..kc {
                    a[i * kc + i] += ridge;
                }
                // Warm prior: shrink toward coef0 rather than toward zero, by
                // moving the penalty's target into the right-hand side.
                if let Some(c0) = &self.cfg.coef0 {
                    for j in 0..m {
                        for (ai, &zi) in zidx.iter().enumerate() {
                            if ai >= off {
                                b[j * kc + ai] += ridge * c0[j][zi];
                            }
                        }
                    }
                }
                self.run_solve(&a, &b, kc, m)
            } else {
                self.solve_standardized(&zidx, &b, kc, m, ridge)
            };

            if let Some(sol) = solved {
                for j in 0..m {
                    for (ai, &zi) in zidx.iter().enumerate() {
                        beta[j * combos.len() + ci][zi] = sol[j * kc + ai];
                    }
                }
            } else if let Some(prev) = &self.beta {
                // Total failure even with jitter: keep the previous coefficients.
                for j in 0..m {
                    beta[j * combos.len() + ci] = prev[j * combos.len() + ci].clone();
                }
            }
        }
        self.beta = Some(beta);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
    }

    fn run_solve(&mut self, a: &[f64], b: &[f64], k: usize, m: usize) -> Option<Vec<f64>> {
        match solve_spd(a, b, k, m) {
            Some((x, jit)) => {
                self.solve_failures += u64::from(jit);
                Some(x)
            }
            None => {
                self.solve_failures += 1;
                None
            }
        }
    }

    /// Standardized solve on one combo's sub-block. `a` is the raw-moment
    /// sub-matrix (intercept row 0 when configured), `b` the per-target rhs.
    /// `zidx` maps this combo's slots to accumulator indices, so the centered
    /// statistics can be read from `EwCov` directly rather than re-derived from
    /// raw moments (which would reintroduce the cancellation the centered
    /// representation exists to avoid).
    fn solve_standardized(
        &mut self,
        zidx: &[usize],
        b: &[f64],
        kc: usize,
        m: usize,
        ridge: f64,
    ) -> Option<Vec<f64>> {
        if !self.cfg.add_intercept {
            // No intercept: scale by the raw second-moment diagonals (there is
            // no centering here, so no cancellation either).
            let s: Vec<f64> = (0..kc)
                .map(|i| self.cov.raw(zidx[i], zidx[i]).max(0.0).sqrt())
                .collect();
            // No centering here, so no cancellation: any strictly positive raw
            // moment is usable.
            let keep: Vec<usize> = (0..kc).filter(|&i| s[i] > 0.0).collect();
            let kk = keep.len();
            if kk == 0 {
                return Some(vec![0.0; kc * m]);
            }
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &j) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = self.cov.raw(zidx[i], zidx[j]) / (s[i] * s[j]);
                }
                asub[i2 * kk + i2] += ridge;
            }
            let mut bsub = vec![0.0; kk * m];
            for j in 0..m {
                for (i2, &i) in keep.iter().enumerate() {
                    bsub[j * kk + i2] = b[j * kc + i] / s[i];
                }
            }
            let sol = self.run_solve(&asub, &bsub, kk, m)?;
            let mut out = vec![0.0; kc * m];
            for j in 0..m {
                for (i2, &i) in keep.iter().enumerate() {
                    out[j * kc + i] = sol[j * kk + i2] / s[i];
                }
            }
            return Some(out);
        }
        // With intercept: center, scale to correlation form, solve, unscale,
        // recover the intercept. Feature slots are 1..kc.
        let kf = kc - 1;
        // Materialized up front: the solve below borrows `self` mutably.
        let means: Vec<f64> = zidx.iter().map(|&z| self.cov.mean(z)).collect();
        let mean = |i: usize| means[i];
        let mut c = vec![0.0; kf * kf];
        for i in 0..kf {
            for j in 0..kf {
                c[i * kf + j] = self.cov.cov(zidx[i + 1], zidx[j + 1]);
            }
        }
        let s: Vec<f64> = (0..kf).map(|i| c[i * kf + i].max(0.0).sqrt()).collect();
        // A genuinely constant feature is dropped (coefficient 0) rather than
        // blowing up; with centered accumulators its variance is exactly zero.
        let keep: Vec<usize> = (0..kf)
            .filter(|&i| {
                crate::variance_is_usable(c[i * kf + i], self.cov.raw(zidx[i + 1], zidx[i + 1]))
            })
            .collect();
        let kk = keep.len();
        let mut out = vec![0.0; kc * m];
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &j) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = c[i * kf + j] / (s[i] * s[j]);
                }
                asub[i2 * kk + i2] += ridge;
            }
            let mut bsub = vec![0.0; kk * m];
            for j in 0..m {
                let ybar = b[j * kc];
                for (i2, &i) in keep.iter().enumerate() {
                    bsub[j * kk + i2] = (b[j * kc + i + 1] - mean(i + 1) * ybar) / s[i];
                    // The prior lives in original units; on the standardized
                    // scale a coefficient is beta * sd, so scale it in.
                    if let Some(c0) = &self.cfg.coef0 {
                        bsub[j * kk + i2] += ridge * c0[j][zidx[i + 1]] * s[i];
                    }
                }
            }
            let sol = self.run_solve(&asub, &bsub, kk, m)?;
            for j in 0..m {
                for (i2, &i) in keep.iter().enumerate() {
                    out[j * kc + i + 1] = sol[j * kk + i2] / s[i];
                }
            }
        }
        for j in 0..m {
            let mut b0 = b[j * kc]; // ybar
            for i in 0..kf {
                b0 -= mean(i + 1) * out[j * kc + i + 1];
            }
            out[j * kc] = b0;
        }
        Some(out)
    }
}

impl OnlineModel for EwRidge {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        let m = self.cfg.n_targets;
        let nc = self.cfg.n_combos();
        let lam = self.cfg.decay.factor(d_clock);
        self.z(x);

        // ---- predict (state before this row's update) ----
        let out = self.predict(x, d_clock);
        let pred = &out.pred;

        // ---- update ----
        // The slow twin sees the same rows under its own, longer halflife.
        if let (Some(slow), Some(h)) = (self.slow.as_mut(), self.cfg.long_halflife) {
            let slow_lam = Decay::Halflife(h).factor(d_clock);
            slow.cov.update(&self.zbuf, slow_lam, weight);
            let SlowState {
                wj: swj,
                r: sr,
                tm: stm,
                ..
            } = &mut **slow;
            for (j, yj) in y.iter().enumerate() {
                let (wj, r) = (&mut swj[j], &mut sr[j]);
                match yj {
                    // `wj_new == 0` means this row carries no weight and none
                    // has ever been carried, so there is nothing to blend and
                    // `a`/`b` would both be 0/0. Same guard as `EwCov::update`.
                    Some(yj) if slow_lam * *wj + weight > 0.0 => {
                        let wj_new = slow_lam * *wj + weight;
                        let a = slow_lam * *wj / wj_new;
                        let b = weight / wj_new;
                        for (ri, zi) in r.iter_mut().zip(&self.zbuf) {
                            *ri = a * *ri + b * zi * yj;
                        }
                        *wj = wj_new;
                        if let Some(tm) = stm.as_mut() {
                            tm.learn(j, *yj, a, b, slow_lam, weight);
                        }
                    }
                    Some(_) => {}
                    None => {
                        *wj *= slow_lam;
                        if let Some(tm) = stm.as_mut() {
                            tm.age(j, slow_lam);
                        }
                    }
                }
            }
        }
        self.cov.update(&self.zbuf, lam, weight);
        for j in 0..m {
            match y[j] {
                // `wj_new == 0` means this row carries no weight and none has
                // ever been carried -- a zero-weight row at the head of a
                // stream. `a` and `b` would both be 0/0, and the NaN would
                // never wash out: `wj` stays NaN, `NaN > 0.0` is false, and the
                // model silently stops predicting forever. Same guard as
                // `EwCov::update` already has.
                Some(yj) if lam * self.wj[j] + weight > 0.0 => {
                    let wj_new = lam * self.wj[j] + weight;
                    let a = lam * self.wj[j] / wj_new;
                    let b = weight / wj_new;
                    for (ri, zi) in self.r[j].iter_mut().zip(&self.zbuf) {
                        *ri = a * *ri + b * zi * yj;
                    }
                    self.wj[j] = wj_new;
                    // The other half of the sufficient statistic, on the same
                    // `a`/`b` as the cross-moments (E45).
                    if let Some(tm) = self.tm.as_mut() {
                        tm.learn(j, yj, a, b, lam, weight);
                    }
                    // EW residual variance from the primary (first-combo) pred.
                    let p = pred[j * nc];
                    let ws_new = lam * self.wsig[j] + weight;
                    if p.is_finite() && ws_new > 0.0 {
                        let resid = yj - p;
                        self.sig2[j] =
                            (lam * self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                        self.wsig[j] = ws_new;
                    }
                }
                Some(_) => {}
                None => {
                    self.wj[j] *= lam;
                    self.wsig[j] *= lam;
                    if let Some(tm) = self.tm.as_mut() {
                        tm.age(j, lam);
                    }
                }
            }
        }

        // ---- solve schedule ----
        self.clock_since_solve += d_clock;
        self.rows_since_solve += 1;
        let due = self.cfg.solve_every <= 0.0
            || self.clock_since_solve >= self.cfg.solve_every
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.cov.n_eff() >= self.cfg.min_periods);
        if due {
            self.solve();
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        let (m, nc) = (self.cfg.n_targets, self.cfg.n_combos());
        let n_eff = self.cov.n_eff();
        let mut pred = vec![f64::NAN; m * nc];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_periods, &self.beta) {
            for j in 0..m {
                if self.wj[j] > 0.0 {
                    for c in 0..nc {
                        pred[j * nc + c] = dot_aug(&beta[j * nc + c], x, self.cfg.add_intercept);
                    }
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::EwRidge(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::EwRidge(m) => {
                let mut m = (**m).clone();
                m.zbuf = vec![0.0; m.cfg.k_total()];
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ew_ridge",
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
        self.cfg.n_targets * self.cfg.n_combos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(k: usize, m: usize) -> EwRidgeCfg {
        EwRidgeCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-8],
            feature_sets: vec![],
            standardize: false,
            ridge_decay: false,
            coef0: None,
            session_shrink: None,
            long_halflife: None,
            min_periods: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
        }
    }

    /// With `ridge_decay` the prior sits on the sum scale, so it starts the
    /// fit and then fades: the usual warm start.
    #[test]
    fn coef0_with_ridge_decay_warms_the_start_then_fades() {
        let mut c = cfg(2, 1);
        c.ridge = vec![10.0];
        c.ridge_decay = true;
        c.standardize = false;
        c.coef0 = Some(vec![vec![0.0, 5.0, -5.0]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();

        // With no target seen yet the fit is essentially the prior. (Not
        // exactly: the feature row itself has already entered S, which pulls a
        // little even with no target.)
        let mut s = 71u64;
        let x0 = [lcg(&mut s), lcg(&mut s)];
        m.step(&x0, &[None], 0.0, 1.0);
        let early = m.coefficients().unwrap()[0].clone();
        assert!(
            (early[1] - 5.0).abs() < 0.5 && (early[2] + 5.0).abs() < 0.5,
            "cold start should sit at the prior: {early:?}"
        );

        // With enough contradicting evidence it moves to the truth.
        for i in 0..20000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.5 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let late = m.coefficients().unwrap()[0].clone();
        assert!(
            (late[1] - 1.5).abs() < 0.05,
            "prior did not wash out: {late:?}"
        );
        assert!((late[2] + 0.5).abs() < 0.05);
    }

    /// Without `ridge_decay` the pull is permanent, because `S` is a weighted
    /// mean and never outgrows a fixed `ridge`. Worth pinning: it is the
    /// opposite of the usual "the prior washes out" intuition.
    #[test]
    fn coef0_without_ridge_decay_pulls_forever() {
        let mut c = cfg(1, 1);
        c.ridge = vec![10.0];
        c.coef0 = Some(vec![vec![0.0, 5.0]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 72u64;
        for i in 0..50000 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(1.5 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap()[0][1];
        assert!(
            b > 3.0,
            "a fixed ridge should keep pulling toward the prior forever, got {b}"
        );
    }

    #[test]
    fn coef0_shrinks_toward_the_prior_not_zero() {
        // Same ridge, same data, different priors => the fits differ, and each
        // sits between the data's answer and its own prior.
        let run = |prior: Option<Vec<Vec<f64>>>| {
            let mut c = cfg(1, 1);
            c.ridge = vec![50.0];
            c.coef0 = prior;
            c.min_periods = 0.0;
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 73u64;
            for i in 0..200 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients().unwrap()[0][1]
        };
        let toward_zero = run(None);
        let toward_ten = run(Some(vec![vec![0.0, 10.0]]));
        assert!(
            toward_zero < 2.0,
            "no prior should shrink toward 0: {toward_zero}"
        );
        assert!(
            toward_ten > 2.0,
            "a prior of 10 should pull up: {toward_ten}"
        );
    }

    #[test]
    fn coef0_works_with_standardization() {
        // The prior is stated in original units; on badly scaled features the
        // standardized path must still honour it.
        let mut c = cfg(1, 1);
        c.ridge = vec![1e6];
        c.standardize = true;
        c.coef0 = Some(vec![vec![0.0, 0.02]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 79u64;
        for i in 0..500 {
            let x = [100.0 * lcg(&mut s)];
            let y = 0.05 * x[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // An overwhelming ridge pins the fit at the prior, in original units.
        let b = m.coefficients().unwrap()[0][1];
        assert!(
            (b - 0.02).abs() < 5e-3,
            "expected ~0.02 in original units, got {b}"
        );
    }

    /// A session break should be able to revert partway toward the long run,
    /// rather than only choosing between "carry on" and "start over".
    #[test]
    fn session_shrink_reverts_toward_the_long_run() {
        // Long run: slope 1. Today: slope -1 for a while. After a session
        // break with shrink f, the fit should sit between the two.
        let build = |f: Option<f64>| {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(50.0);
            c.session_shrink = f;
            c.long_halflife = f.map(|_| 100_000.0);
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let run = |m: &mut EwRidge| {
            let mut s = 91u64;
            // a long history at slope +1
            for i in 0..4000 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            // then a shorter stretch at slope -1
            for _ in 0..300 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(-x[0])], 1.0, 1.0);
            }
            m.coefficients().unwrap()[0][1]
        };

        let mut plain = build(None);
        let before_plain = run(&mut plain);
        let mut shrunk = build(Some(0.9));
        let before_shrunk = run(&mut shrunk);
        // Both have been dragged to roughly -1 by the recent regime.
        assert!(before_plain < -0.5 && before_shrunk < -0.5);

        // The session boundary reverts the shrinking one toward +1.
        shrunk.blend_toward_long_run();
        plain.blend_toward_long_run(); // no twin configured: a no-op
        let after_shrunk = shrunk.coefficients_after_blend();
        let after_plain = plain.coefficients_after_blend();

        assert!(
            (after_plain - before_plain).abs() < 1e-9,
            "no twin configured should mean no change: {before_plain} -> {after_plain}"
        );
        assert!(
            after_shrunk > before_shrunk + 0.5,
            "shrink should pull back toward the long run: {before_shrunk} -> {after_shrunk}"
        );
    }

    /// E45: the target moments must survive a blend as a mixture, not go
    /// stale, and `f = 0` (or a twin identical to the model) must leave them
    /// exactly where they were -- the invariant the means, co-moments and
    /// weights already hold.
    #[test]
    fn a_blend_mixes_the_target_moments() {
        let build = |f: f64| {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(50.0);
            c.session_shrink = Some(f);
            c.long_halflife = Some(50.0); // the same halflife: an identical twin
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let run = |m: &mut EwRidge| {
            let mut s = 7u64;
            for i in 0..500 {
                let x = [lcg(&mut s)];
                m.step(
                    &x,
                    &[Some(3.0 + 2.0 * x[0])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
        };
        let mut twin = build(0.5);
        run(&mut twin);
        let (m0, v0, q0) = {
            let tm = twin.target_moments().unwrap();
            (tm.means()[0], tm.vars()[0], tm.q()[0])
        };
        twin.blend_toward_long_run();
        let tm = twin.target_moments().unwrap();
        assert!(
            (tm.means()[0] - m0).abs() < 1e-12,
            "{} vs {m0}",
            tm.means()[0]
        );
        assert!((tm.vars()[0] - v0).abs() < 1e-9, "{} vs {v0}", tm.vars()[0]);
        assert!((tm.q()[0] - q0).abs() < 1e-9, "{} vs {q0}", tm.q()[0]);

        // A genuinely slower twin moves them toward the long run. The level
        // sits at 0 for a long stretch and jumps to 10 for a short one, so
        // the fast window's mean is ~10 and the twin's is much lower.
        let mut slow = {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(20.0);
            c.session_shrink = Some(0.9);
            c.long_halflife = Some(5000.0);
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let mut s = 11u64;
        for i in 0..2000 {
            let x = [lcg(&mut s)];
            let level = if i < 1900 { 0.0 } else { 10.0 };
            slow.step(
                &x,
                &[Some(level + x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = slow.target_moments().unwrap().means()[0];
        assert!(
            before > 9.0,
            "the fast window should be at the new level: {before}"
        );
        slow.blend_toward_long_run();
        let tm = slow.target_moments().unwrap();
        assert!(
            tm.means()[0] < before - 5.0,
            "the blend should pull the mean toward the long run: {before} -> {}",
            tm.means()[0]
        );
        // And the mixture is still a valid set of moments.
        assert!(tm.vars()[0] > 0.0);
        assert!(tm.q()[0] > 0.0 && tm.n_kish(slow.target_weights())[0].unwrap() > 1.0);
    }

    #[test]
    fn session_shrink_config_is_validated() {
        let mut c = cfg(1, 1);
        c.session_shrink = Some(0.5);
        assert!(EwRidge::new(c).is_err(), "shrink without long_halflife");
        let mut c = cfg(1, 1);
        c.long_halflife = Some(1000.0);
        assert!(EwRidge::new(c).is_err(), "long_halflife without shrink");
        let mut c = cfg(1, 1);
        c.session_shrink = Some(1.5);
        c.long_halflife = Some(1000.0);
        assert!(EwRidge::new(c).is_err(), "shrink out of range");
    }

    #[test]
    fn coef0_shape_is_validated() {
        let mut c = cfg(2, 1);
        c.coef0 = Some(vec![vec![0.0, 1.0]]); // too short
        assert!(EwRidge::new(c).is_err());
        let mut c = cfg(2, 1);
        c.coef0 = Some(vec![vec![0.0, 1.0, f64::NAN]]);
        assert!(EwRidge::new(c).is_err());
        // coef0 *is* allowed with ridge_decay -- that combination is the
        // fading warm start -- so only the shape rules above are enforced.
        let mut c = cfg(2, 1);
        c.ridge_decay = true;
        c.standardize = false;
        c.coef0 = Some(vec![vec![0.0, 1.0, 2.0]]);
        assert!(EwRidge::new(c).is_ok());
    }

    /// Deterministic pseudo-random stream (no external rng dependency).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn recovers_static_beta() {
        let beta = [0.5, -1.0, 2.0];
        let mut m = EwRidge::new(cfg(3, 1)).unwrap();
        let mut s = 42u64;
        let mut last = Step {
            pred: vec![],
            n_eff: 0.0,
            extra: None,
        };
        for i in 0..500 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>() + 3.0;
            let d = if i == 0 { 0.0 } else { 1.0 };
            last = m.step(&x, &[Some(y)], d, 1.0);
        }
        let c = &m.coefficients().unwrap()[0];
        assert!((c[0] - 3.0).abs() < 1e-6, "intercept {}", c[0]);
        for i in 0..3 {
            assert!((c[i + 1] - beta[i]).abs() < 1e-6, "beta[{i}] {}", c[i + 1]);
        }
        assert!(last.pred[0].is_finite());
    }

    #[test]
    fn pred_is_out_of_sample_and_warmup_nan() {
        let mut m = EwRidge::new(cfg(2, 1)).unwrap();
        let mut s = 7u64;
        for i in 0..3 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let st = m.step(
                &x,
                &[Some(lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            assert!(st.pred[0].is_nan(), "warmup row {i} must be NaN");
        }
        let st = m.step(&[0.1, 0.2], &[Some(0.3)], 1.0, 1.0);
        assert!(st.pred[0].is_finite());
    }

    #[test]
    fn solve_schedule_staleness() {
        // With a large solve_every, coefficients stay fixed between solves.
        let mut c = cfg(1, 1);
        c.solve_every = 1e9;
        c.max_rows_between_solves = 10;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 9u64;
        let mut snapshots = vec![];
        for i in 0..40 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            snapshots.push(m.coefficients().map(|b| b[0].clone()));
        }
        // Between solve rows the coefficients are bit-identical.
        let mut changes = 0;
        for w in snapshots.windows(2) {
            if w[0] != w[1] {
                changes += 1;
            }
        }
        assert!(
            changes <= 5,
            "expected sparse solves, got {changes} changes"
        );
    }

    #[test]
    fn multi_target_and_grids() {
        let mut c = cfg(3, 2);
        c.ridge = vec![1e-8, 10.0];
        c.feature_sets = vec![("a".into(), vec![0, 1]), ("b".into(), vec![2])];
        let m = EwRidge::new(c.clone()).unwrap();
        assert_eq!(m.n_outputs(), 2 * 4);
        assert_eq!(
            c.combo_labels(),
            vec!["a_r0.00000001", "a_r10", "b_r0.00000001", "b_r10"]
        );

        let mut m = EwRidge::new(c).unwrap();
        let mut s = 3u64;
        let mut last = None;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y0 = x[0] - x[1];
            let y1 = 3.0 * x[2];
            last = Some(m.step(
                &x,
                &[Some(y0), Some(y1)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            ));
        }
        let st = last.unwrap();
        assert_eq!(st.pred.len(), 8);
        assert!(st.pred.iter().all(|p| p.is_finite()));
        // combo "b" (only x2) predicts y1 well, y0 badly; heavy ridge shrinks.
        let b = m.coefficients().unwrap();
        // target 1 (y1), combo index 2 = fs "b", small ridge: coef on x2 ~ 3
        assert!((b[4 + 2][3] - 3.0).abs() < 0.05, "{:?}", b[6]);
        // heavy-ridge combo shrinks toward zero
        assert!(b[4 + 3][3].abs() < b[4 + 2][3].abs());
        // feature set "b" never touches x0/x1
        assert_eq!(b[4 + 2][1], 0.0);
        assert_eq!(b[4 + 2][2], 0.0);
    }

    #[test]
    fn standardize_matches_plain_when_ridge_tiny() {
        // On well-conditioned data the centered/standardized solve and the raw
        // solve are algebraically identical (ridge ~ 0), so they must agree
        // tightly. (On badly scaled data they diverge because the raw normal
        // equations are ill-conditioned -- which is why standardize exists.)
        // Ridge is dropped to 1e-12 and the feature scales are put ~1e3 apart,
        // so the centering and scaling matrices are far from the identity and
        // an operation applied in the wrong direction cannot hide inside the
        // tolerance. The invariance is exact only at ridge = 0, because the
        // penalty lands on the raw scale in one path and the standardized
        // scale in the other.
        let mut ca = cfg(2, 1);
        ca.ridge = vec![1e-12];
        let mut cb = ca.clone();
        cb.standardize = true;
        let mut ma = EwRidge::new(ca).unwrap();
        let mut mb = EwRidge::new(cb).unwrap();
        let mut s = 11u64;
        for i in 0..300 {
            let x = [400.0 + 3.0 * lcg(&mut s), 0.002 * lcg(&mut s)];
            let y = 7.0 + 0.5 * x[0] - 300.0 * x[1] + 0.001 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            ma.step(&x, &[Some(y)], d, 1.0);
            mb.step(&x, &[Some(y)], d, 1.0);
        }
        let a = &ma.coefficients().unwrap()[0];
        let b = &mb.coefficients().unwrap()[0];
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
                "coef {i}: plain {} vs standardized {}",
                a[i],
                b[i]
            );
        }
        // And the standardized path recovered the generating relationship, so
        // the agreement is not two paths failing the same way. The intercept is
        // reconstructed from the centered fit, which is the step most easily
        // lost.
        assert!((b[0] - 7.0).abs() < 0.1, "intercept {}", b[0]);
        assert!((b[1] - 0.5).abs() < 1e-3, "slope 0 {}", b[1]);
        assert!((b[2] + 300.0).abs() < 5.0, "slope 1 {}", b[2]);
    }

    /// A model with a slow twin, fed `n` rows of a deterministic stream.
    fn blended_pair(shrink: f64) -> EwRidge {
        let mut c = cfg(2, 1);
        c.session_shrink = Some(shrink);
        c.long_halflife = Some(400.0);
        c.decay = Decay::Halflife(20.0);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 23u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            // A relationship that flips halfway, so fast and slow genuinely
            // disagree by the time the session boundary arrives.
            let sign = if i < 100 { 1.0 } else { -1.0 };
            let y = sign * (2.0 * x[0] - x[1]);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m
    }

    #[test]
    fn coef0_solves_the_ridge_problem_it_claims_to() {
        // With `coef0 = c`, the penalty shrinks toward `c` rather than zero:
        //     beta = (C + rI)^-1 (d + r c)
        // on the centered accumulators, with the intercept recovered after.
        // The existing coef0 tests check the *direction* of the pull; this
        // pins the closed form, computed by hand from the model's own state.
        let (r, c0) = (0.7, vec![vec![0.0, 3.0, -2.0]]);
        let mut cfg_ = cfg(2, 1);
        cfg_.ridge = vec![r];
        cfg_.coef0 = Some(c0.clone());
        cfg_.min_periods = 3.0;
        let mut m = EwRidge::new(cfg_).unwrap();
        let mut s = 127u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let y = 1.0 + 2.0 * x[0] - x[1] + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }

        // The 2x2 penalized normal equations on the centered moments.
        let (c00, c01, c11) = (m.cov.cov(1, 1), m.cov.cov(1, 2), m.cov.cov(2, 2));
        // d_i = E[x_i y] - E[x_i] E[y], from the tracked cross-moment means.
        let d0 = m.r[0][1] - m.cov.mean(1) * m.r[0][0];
        let d1 = m.r[0][2] - m.cov.mean(2) * m.r[0][0];
        let (a00, a11) = (c00 + r, c11 + r);
        let (rhs0, rhs1) = (d0 + r * c0[0][1], d1 + r * c0[0][2]);
        let det = a00 * a11 - c01 * c01;
        let want = [
            (rhs0 * a11 - c01 * rhs1) / det,
            (a00 * rhs1 - c01 * rhs0) / det,
        ];

        let got = &m.coefficients().unwrap()[0];
        for i in 0..2 {
            assert!(
                (got[i + 1] - want[i]).abs() < 1e-9 * (1.0 + want[i].abs()),
                "slope {i}: {} vs {}",
                got[i + 1],
                want[i]
            );
        }
        // The intercept is reconstructed, not fitted: mean(y) - b'mean(x).
        let want0 = m.r[0][0] - got[1] * m.cov.mean(1) - got[2] * m.cov.mean(2);
        assert!((got[0] - want0).abs() < 1e-9, "{} vs {want0}", got[0]);

        // An overwhelming penalty must land on coef0 exactly.
        let mut cfg_ = cfg(2, 1);
        cfg_.ridge = vec![1e12];
        cfg_.coef0 = Some(c0.clone());
        cfg_.min_periods = 3.0;
        let mut m = EwRidge::new(cfg_).unwrap();
        let mut s = 131u64;
        for i in 0..100 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let got = &m.coefficients().unwrap()[0];
        assert!((got[1] - 3.0).abs() < 1e-3, "slope 0 -> coef0: {}", got[1]);
        assert!((got[2] + 2.0).abs() < 1e-3, "slope 1 -> coef0: {}", got[2]);
    }

    #[test]
    fn a_singular_solve_is_counted_and_the_previous_fit_is_kept() {
        // `run_solve` records both outcomes: a solve rescued by jitter and a
        // total failure. Two perfectly collinear features with no ridge give
        // a rank-deficient system; the model must never emit NaN and must say
        // that something went wrong.
        let mut c = cfg(2, 1);
        c.ridge = vec![0.0];
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 137u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
        let beta = &m.coefficients().unwrap()[0];
        assert!(beta.iter().all(|v| v.is_finite()), "never NaN: {beta:?}");

        // A well-conditioned stream records nothing.
        let mut c = cfg(2, 1);
        c.min_periods = 2.0;
        let mut ok = EwRidge::new(c).unwrap();
        let mut s = 139u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            ok.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert_eq!(ok.solve_failures, 0, "a healthy fit must record no failure");
    }

    #[test]
    fn the_slow_twin_is_the_same_model_at_the_long_halflife() {
        // The twin's accumulators are updated by a second copy of the update
        // block inside `step`, which nothing else reaches. The oracle is the
        // obvious one: a standalone model configured at `long_halflife` and
        // fed the same rows must end up with identical statistics.
        let mut c = cfg(2, 1);
        c.session_shrink = Some(0.4);
        c.long_halflife = Some(300.0);
        c.decay = Decay::Halflife(15.0);
        c.min_periods = 3.0;

        let mut twin_cfg = cfg(2, 1);
        twin_cfg.decay = Decay::Halflife(300.0);
        twin_cfg.min_periods = 3.0;

        let mut m = EwRidge::new(c).unwrap();
        let mut reference = EwRidge::new(twin_cfg).unwrap();
        let mut s = 43u64;
        for i in 0..150 {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
            // Every fourth row has a null target and an irregular gap, so the
            // twin's null branch (`*wj *= slow_lam`) is exercised too.
            let y = if i % 4 == 3 {
                None
            } else {
                Some(1.5 * x[0] - 0.5 * x[1])
            };
            let d = if i == 0 { 0.0 } else { 1.0 + (i % 3) as f64 };
            m.step(&x, &[y], d, 1.0);
            reference.step(&x, &[y], d, 1.0);
        }

        let slow = m.slow.as_ref().unwrap();
        let k = m.cfg.k_total();
        assert!((slow.cov.n_eff() - reference.cov.n_eff()).abs() < 1e-9);
        for i in 0..k {
            assert!(
                (slow.cov.mean(i) - reference.cov.mean(i)).abs() < 1e-9,
                "mean {i}"
            );
            for j in 0..k {
                assert!((slow.cov.cov(i, j) - reference.cov.cov(i, j)).abs() < 1e-9);
            }
        }
        assert!((slow.wj[0] - reference.wj[0]).abs() < 1e-9);
        for i in 0..k {
            assert!((slow.r[0][i] - reference.r[0][i]).abs() < 1e-9, "r[{i}]");
        }
        // And it is genuinely slower than the fast side, or the test would
        // pass with the twin wired to the wrong decay.
        assert!(
            slow.cov.n_eff() > 3.0 * m.cov.n_eff(),
            "{} vs {}",
            slow.cov.n_eff(),
            m.cov.n_eff()
        );
    }

    #[test]
    fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
        // `sigma2` is only surfaced through the Polars layer, so nothing in
        // this crate pinned its recursion. It is an EW mean on the model's own
        // clock, over the *predicted* residual -- and rows before the first
        // prediction contribute nothing.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(25.0);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();

        let (mut want, mut wsig) = (0.0, 0.0);
        let mut s = 47u64;
        for i in 0..120 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let lam = 0.5f64.powf(d / 25.0);
            let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
            if p.is_finite() {
                let resid = y - p;
                let ws_new = lam * wsig + 1.0;
                want = (lam * wsig * want + resid * resid) / ws_new;
                wsig = ws_new;
            }
            assert!(
                (m.sigma2()[0] - want).abs() < 1e-12,
                "row {i}: {} vs {want}",
                m.sigma2()[0]
            );
        }
        // The weight saturates at 1/(1 - lam) ~ 36.6 for this halflife.
        assert!(
            wsig > 30.0,
            "the recursion should have run, not been skipped"
        );
        assert!(
            want > 0.0 && want < 1.0,
            "plausible residual variance: {want}"
        );
    }

    #[test]
    fn null_targets_decay_the_residual_variance_weight_without_adding_to_it() {
        // The `None` arm decays `wj` and `wsig` but must not fold a residual
        // in -- otherwise a gap in the target inflates or freezes sigma.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(10.0);
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 53u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(2.0 * x[0] + 0.1)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (sig, w, wj) = (m.sigma2()[0], m.wsig[0], m.wj[0]);
        assert!(sig > 0.0 && w > 0.0);

        let lam = 0.5f64.powf(3.0 / 10.0);
        m.step(&[0.5], &[None], 3.0, 1.0);
        assert_eq!(m.sigma2()[0], sig, "a null target must not move sigma2");
        assert!((m.wsig[0] - w * lam).abs() < 1e-12, "but its weight decays");
        assert!((m.wj[0] - wj * lam).abs() < 1e-12);
    }

    #[test]
    fn the_solve_schedule_controls_when_coefficients_move() {
        // Almost every test here solves on every row, which masks the three
        // clauses of the `due` condition. Each is checked on its own.
        let coefs = |c: EwRidgeCfg, ds: &[f64]| {
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 59u64;
            let mut out = Vec::new();
            for (i, &d) in ds.iter().enumerate() {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(3.0 * x[0])], if i == 0 { 0.0 } else { d }, 1.0);
                out.push(m.coefficients().map(|b| b[0][1]));
            }
            out
        };
        let ones = vec![1.0; 24];

        // Row-counted. `solve_every = 0` means "every row" and short-circuits
        // the rest of the condition, so the row cap is only visible with a
        // clock schedule that will not fire.
        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.solve_every = 1e9;
        c.max_rows_between_solves = 5;
        let by_rows = coefs(c, &ones);
        let changes: Vec<usize> = by_rows
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] != w[1])
            .map(|(i, _)| i + 1)
            .collect();
        // First solve as soon as min_periods is met, then strictly every 5
        // accepted rows: nothing in between, and it does not stop.
        assert_eq!(changes, vec![1, 6, 11, 16, 21], "24 rows, cap of 5");

        // Clock-counted: the same stream on a clock that advances 2 per row
        // must solve half as often when `solve_every` is 4.
        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 4.0;
        let slow = coefs(c, &[2.0; 24]);
        let n_slow = slow.windows(2).filter(|w| w[0] != w[1]).count();

        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 0.0; // 0 means "every row"
        let every = coefs(c, &ones);
        let n_every = every.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(
            n_slow < n_every,
            "solve_every should throttle: {n_slow} vs {n_every}"
        );

        // The first solve is not throttled: it happens as soon as min_periods
        // is met, however long the schedule says to wait.
        let mut c = cfg(1, 1);
        c.min_periods = 3.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 1e9;
        let first = coefs(c, &ones);
        assert!(
            first.iter().take(6).any(|b| b.is_some()),
            "the first solve must not wait for the schedule"
        );
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        // One case per rejection in `EwRidgeCfg::validate`, each matched on the
        // message so a mutation that reports the wrong reason is caught too --
        // and each accompanied by the nearest *valid* config, so a validator
        // that rejects everything cannot pass either.
        let bad = |f: &dyn Fn(&mut EwRidgeCfg), want: &str| {
            let mut c = cfg(2, 1);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        let good = |f: &dyn Fn(&mut EwRidgeCfg)| {
            let mut c = cfg(2, 1);
            f(&mut c);
            c.validate().expect("should be accepted");
        };

        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");
        bad(&|c| c.ridge = vec![], "at least one value");

        // ridge_decay alone is fine; it is the combination that is refused.
        good(&|c| c.ridge_decay = true);
        bad(
            &|c| {
                c.ridge_decay = true;
                c.standardize = true;
            },
            "incompatible",
        );
        bad(
            &|c| {
                c.ridge_decay = true;
                c.ridge = vec![1e-6, 1.0];
            },
            "incompatible",
        );

        bad(&|c| c.session_shrink = Some(-0.1), "in [0, 1]");
        bad(&|c| c.session_shrink = Some(1.5), "in [0, 1]");
        bad(&|c| c.session_shrink = Some(0.5), "needs long_halflife");
        bad(&|c| c.long_halflife = Some(100.0), "no effect without");
        bad(
            &|c| {
                c.session_shrink = Some(0.5);
                c.long_halflife = Some(0.0);
            },
            "must be > 0",
        );
        bad(
            &|c| {
                c.session_shrink = Some(0.5);
                c.long_halflife = Some(f64::NAN);
            },
            "must be > 0",
        );
        good(&|c| {
            c.session_shrink = Some(0.0);
            c.long_halflife = Some(100.0);
        });
        good(&|c| {
            c.session_shrink = Some(1.0);
            c.long_halflife = Some(100.0);
        });

        // coef0 is one vector per target, each of length k_total (2 + intercept).
        bad(
            &|c| c.coef0 = Some(vec![vec![0.0; 3], vec![0.0; 3]]),
            "1 vector of",
        );
        bad(&|c| c.coef0 = Some(vec![vec![0.0; 2]]), "length 3");
        bad(
            &|c| c.coef0 = Some(vec![vec![0.0, 0.0, f64::NAN]]),
            "finite",
        );
        bad(
            &|c| c.coef0 = Some(vec![vec![0.0, 0.0, f64::INFINITY]]),
            "finite",
        );
        good(&|c| c.coef0 = Some(vec![vec![1.0, 2.0, 3.0]]));

        bad(
            &|c| c.feature_sets = vec![("a".into(), vec![])],
            "out-of-range",
        );
        bad(
            &|c| c.feature_sets = vec![("a".into(), vec![2])],
            "out-of-range",
        );
        good(&|c| c.feature_sets = vec![("a".into(), vec![0]), ("b".into(), vec![0, 1])]);

        cfg(2, 1).validate().expect("the baseline config is valid");
    }

    #[test]
    fn blend_is_the_weight_respecting_mixture() {
        // Every arithmetic step of `blend_toward_long_run` is checked against
        // the same quantities recomputed by hand from the pre-blend state, so
        // a factor applied to the wrong side, a missing re-centering, or a
        // swapped index all show up.
        let f = 0.3;
        let mut m = blended_pair(f);
        let before = m.clone();
        let slow = before.slow.as_ref().unwrap();
        let k = m.cfg.k_total();

        let (wf, ws) = (before.cov.n_eff(), slow.cov.n_eff());
        assert!(wf > 0.0 && ws > wf, "the slow twin should hold more weight");
        let w_new = (1.0 - f) * wf + f * ws;
        let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);

        m.blend_toward_long_run();

        assert!((m.cov.n_eff() - w_new).abs() < 1e-12);
        for i in 0..k {
            let want = af * before.cov.mean(i) + as_ * slow.cov.mean(i);
            assert!(
                (m.cov.mean(i) - want).abs() < 1e-12,
                "mean {i}: {} vs {want}",
                m.cov.mean(i)
            );
        }
        for i in 0..k {
            for j in 0..k {
                // Raw moments mix linearly; the centered moment must then be
                // re-derived against the *mixed* mean, not either input's.
                let raw = af * before.cov.raw(i, j) + as_ * slow.cov.raw(i, j);
                let want = raw - m.cov.mean(i) * m.cov.mean(j);
                assert!(
                    (m.cov.cov(i, j) - want).abs() < 1e-10,
                    "cov {i},{j}: {} vs {want}",
                    m.cov.cov(i, j)
                );
            }
        }
        for j in 0..m.cfg.n_targets {
            let (wf, ws) = (before.wj[j], slow.wj[j]);
            let w_new = (1.0 - f) * wf + f * ws;
            let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
            assert!((m.wj[j] - w_new).abs() < 1e-12);
            for i in 0..k {
                let want = af * before.r[j][i] + as_ * slow.r[j][i];
                assert!((m.r[j][i] - want).abs() < 1e-12, "r[{j}][{i}]");
            }
        }
    }

    #[test]
    fn blend_endpoints_are_identity_and_full_replacement() {
        // f = 0 must not touch the state; f = 1 must land exactly on the twin.
        let mut zero = blended_pair(0.0);
        let before = zero.clone();
        zero.blend_toward_long_run();
        assert_eq!(zero, before, "session_shrink = 0 must be a no-op");

        let mut one = blended_pair(1.0);
        let slow = one.slow.clone().unwrap();
        one.blend_toward_long_run();
        let k = one.cfg.k_total();
        assert!((one.cov.n_eff() - slow.cov.n_eff()).abs() < 1e-12);
        for i in 0..k {
            assert!(
                (one.cov.mean(i) - slow.cov.mean(i)).abs() < 1e-12,
                "mean {i}"
            );
        }
        for j in 0..one.cfg.n_targets {
            assert!((one.wj[j] - slow.wj[j]).abs() < 1e-12);
            for i in 0..k {
                assert!((one.r[j][i] - slow.r[j][i]).abs() < 1e-12, "r[{j}][{i}]");
            }
        }
    }

    #[test]
    fn blend_before_any_data_is_a_no_op() {
        // The other half of the doc comment's promise: a no-op when the twin
        // is not configured, *or before it has seen anything*. With both sides
        // at zero weight the mixture's denominator is zero, and the guard has
        // to catch that rather than divide.
        let mut c = cfg(2, 1);
        c.session_shrink = Some(0.5);
        c.long_halflife = Some(200.0);
        let mut m = EwRidge::new(c).unwrap();
        assert!(m.slow.is_some());
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before, "nothing seen yet: nothing to blend");
        for i in 0..m.cfg.k_total() {
            assert!(m.cov.mean(i).is_finite(), "and no NaN got in");
        }

        // Still safe on a session boundary that arrives with a session's worth
        // of zero-weight rows behind it.
        for _ in 0..5 {
            m.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 0.0);
        }
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before, "zero-weight rows carry no weight to blend");
    }

    #[test]
    fn blend_without_a_twin_is_a_no_op() {
        // No `session_shrink` at all: there is no twin to blend with, and the
        // method must return before touching anything.
        let mut c = cfg(2, 1);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 29u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.slow.is_none());
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before);
    }

    #[test]
    fn blend_moves_the_fit_toward_the_long_run_relationship() {
        // The point of the feature, not just its arithmetic: the last session
        // ran at the opposite sign to the long run, and blending must pull the
        // fit back -- monotonically in the shrink parameter.
        let slope = |f: f64| {
            let mut m = blended_pair(f);
            m.blend_toward_long_run();
            m.coefficients_after_blend()
        };
        let (none, half, full) = (slope(0.0), slope(0.5), slope(1.0));
        assert!(none < 0.0, "the last session was negative: {none}");
        assert!(
            full > none,
            "the long run should pull it up: {full} vs {none}"
        );
        assert!(
            none < half && half < full,
            "reversion should be monotone: {none} < {half} < {full}"
        );
    }

    #[test]
    fn standardize_without_intercept_matches_plain_when_ridge_tiny() {
        // `solve_standardized` has a second, quite different branch for
        // `add_intercept = false`: it scales by the *raw* second-moment
        // diagonals rather than centering first. The invariance is the same --
        // a diagonal rescale of the normal equations cannot move the solution
        // when the penalty is negligible -- so a scaling applied in the wrong
        // direction, or to only one of A, b and the unscaled result, shows up
        // here. Features are deliberately on very different scales (~4 and
        // ~0.003) so the scaling matrix is far from the identity.
        let mut ca = cfg(2, 1);
        ca.add_intercept = false;
        ca.min_periods = 2.0;
        // The invariance is exact only at ridge = 0: a penalty applies on the
        // raw scale in one path and the standardized scale in the other, and
        // the two feature scales here differ by ~1e3, so 1e-8 is enough to
        // separate them at 1e-6.
        ca.ridge = vec![1e-12];
        let mut cb = ca.clone();
        cb.standardize = true;
        let mut ma = EwRidge::new(ca).unwrap();
        let mut mb = EwRidge::new(cb).unwrap();
        let mut s = 17u64;
        for i in 0..300 {
            let x = [4.0 + lcg(&mut s), 0.003 * lcg(&mut s)];
            let y = 0.25 * x[0] - 40.0 * x[1] + 0.001 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            ma.step(&x, &[Some(y)], d, 1.0);
            mb.step(&x, &[Some(y)], d, 1.0);
        }
        let a = &ma.coefficients().unwrap()[0];
        let b = &mb.coefficients().unwrap()[0];
        assert_eq!(a.len(), 2, "no intercept slot when add_intercept is false");
        for i in 0..2 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
                "coef {i}: plain {} vs standardized {}",
                a[i],
                b[i]
            );
        }
        // And it actually recovered the generating coefficients, so the
        // agreement is not two paths being wrong the same way.
        assert!((b[0] - 0.25).abs() < 1e-3, "{}", b[0]);
        assert!((b[1] + 40.0).abs() < 1.0, "{}", b[1]);
    }

    #[test]
    fn standardize_without_intercept_drops_a_zero_column() {
        // The `s[i] > 0.0` guard in the no-intercept branch: a feature that is
        // identically zero has zero raw second moment, so it cannot be scaled
        // and must come out with a zero coefficient rather than a NaN.
        let mut c = cfg(2, 1);
        c.add_intercept = false;
        c.standardize = true;
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 19u64;
        for i in 0..100 {
            let x = [1.0 + lcg(&mut s), 0.0];
            let y = 2.0 * x[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0];
        assert_eq!(b[1], 0.0, "the all-zero feature must be dropped, not NaN");
        assert!((b[0] - 2.0).abs() < 1e-6, "{}", b[0]);
    }

    #[test]
    fn zero_variance_feature_dropped_in_standardized_solve() {
        let mut c = cfg(2, 1);
        c.standardize = true;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 13u64;
        for i in 0..100 {
            let x = [lcg(&mut s), 5.0]; // constant second feature
            let y = 2.0 * x[0];
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i > 10 {
                assert!(st.pred[0].is_finite(), "row {i}");
            }
        }
        let b = &m.coefficients().unwrap()[0];
        assert_eq!(b[2], 0.0, "full coef {b:?}"); // dropped, not blown up
        assert!((b[1] - 2.0).abs() < 1e-6); // 1e-8 ridge itself shifts this by ~2e-8
    }

    #[test]
    fn state_roundtrip_continues_identically() {
        let mut m1 = EwRidge::new(cfg(2, 1)).unwrap();
        let mut s = 5u64;
        let mut rows = vec![];
        for _ in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + 0.1 * lcg(&mut s);
            rows.push((x, y));
        }
        for (i, (x, y)) in rows[..30].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let restored: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut m2 = EwRidge::restore(&restored).unwrap();
        for (x, y) in &rows[30..] {
            let a = m1.step(x, &[Some(*y)], 1.0, 1.0);
            let b = m2.step(x, &[Some(*y)], 1.0, 1.0);
            assert_eq!(a.pred, b.pred);
            assert_eq!(a.n_eff, b.n_eff);
        }
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = EwRidge::new(cfg(1, 2)).unwrap();
        let mut s = 17u64;
        for i in 0..50 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(2.0 * x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.r[1].clone();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred[1].is_finite()); // pred still emitted
        // r for target 1 unchanged in value terms (mean form: no data added)
        assert_eq!(m.r[1], before);
    }
}
