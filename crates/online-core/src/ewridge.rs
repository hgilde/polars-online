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
use crate::solve::solve_spd;
use crate::{Decay, EwCov};

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
                    "coef0 must be {} vectors of length {}",
                    self.n_targets,
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
            })
        });
        Ok(Self {
            slow,
            cov: EwCov::new(k_total),
            wj: vec![0.0; m],
            r: vec![vec![0.0; k_total]; m],
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
    pub fn coefficients(&self) -> Option<&[Vec<f64>]> {
        self.beta.as_deref()
    }

    /// Mix the main accumulators toward the slow twin, as a session boundary
    /// asks for (see [`EwRidgeCfg::session_shrink`]). A no-op when the twin is
    /// not configured, or before it has seen anything.
    /// Re-solve and return the first target's first slope. Test helper: after
    /// a blend the coefficients are stale until the next solve.
    #[cfg(test)]
    pub(crate) fn coefficients_after_blend(&mut self) -> f64 {
        self.solve();
        self.coefficients().unwrap()[0][1]
    }

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
            blended.set_moments(&mixed_mean, &mixed_c, w_new);
            self.cov = blended;
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
        let n_eff = self.cov.n_eff();
        let ready = n_eff >= self.cfg.min_periods && self.beta.is_some();
        let mut pred = vec![f64::NAN; m * nc];
        if ready {
            let beta = self.beta.as_ref().unwrap();
            for j in 0..m {
                if self.wj[j] > 0.0 {
                    for c in 0..nc {
                        let b = &beta[j * nc + c];
                        let mut p = 0.0;
                        for (zi, bi) in self.zbuf.iter().zip(b) {
                            p += zi * bi;
                        }
                        pred[j * nc + c] = p;
                    }
                }
            }
        }

        // ---- update ----
        // The slow twin sees the same rows under its own, longer halflife.
        if let (Some(slow), Some(h)) = (self.slow.as_mut(), self.cfg.long_halflife) {
            let slow_lam = Decay::Halflife(h).factor(d_clock);
            slow.cov.update(&self.zbuf, slow_lam, weight);
            for ((wj, r), yj) in slow.wj.iter_mut().zip(slow.r.iter_mut()).zip(y) {
                match yj {
                    Some(yj) => {
                        let wj_new = slow_lam * *wj + weight;
                        let a = slow_lam * *wj / wj_new;
                        let b = weight / wj_new;
                        for (ri, zi) in r.iter_mut().zip(&self.zbuf) {
                            *ri = a * *ri + b * zi * yj;
                        }
                        *wj = wj_new;
                    }
                    None => *wj *= slow_lam,
                }
            }
        }
        self.cov.update(&self.zbuf, lam, weight);
        for j in 0..m {
            match y[j] {
                Some(yj) => {
                    let wj_new = lam * self.wj[j] + weight;
                    let a = lam * self.wj[j] / wj_new;
                    let b = weight / wj_new;
                    for (ri, zi) in self.r[j].iter_mut().zip(&self.zbuf) {
                        *ri = a * *ri + b * zi * yj;
                    }
                    self.wj[j] = wj_new;
                    // EW residual variance from the primary (first-combo) pred.
                    let p = pred[j * nc];
                    if p.is_finite() {
                        let resid = yj - p;
                        let ws_new = lam * self.wsig[j] + weight;
                        self.sig2[j] =
                            (lam * self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                        self.wsig[j] = ws_new;
                    }
                }
                None => {
                    self.wj[j] *= lam;
                    self.wsig[j] *= lam;
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

        Step {
            pred,
            coef: None,
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
            coef: None,
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
        let ca = cfg(2, 1);
        let mut cb = cfg(2, 1);
        cb.standardize = true;
        let mut ma = EwRidge::new(ca).unwrap();
        let mut mb = EwRidge::new(cb).unwrap();
        let mut s = 11u64;
        for i in 0..300 {
            let x = [2.0 + lcg(&mut s), 3.0 * lcg(&mut s)];
            let y = 0.5 * x[0] - 1.5 * x[1] + 0.01 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            ma.step(&x, &[Some(y)], d, 1.0);
            mb.step(&x, &[Some(y)], d, 1.0);
        }
        let a = &ma.coefficients().unwrap()[0];
        let b = &mb.coefficients().unwrap()[0];
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()), // ridge applies on different scales
                "coef {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
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
