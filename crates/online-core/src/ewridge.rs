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
        for (name, idx) in &self.feature_sets {
            if idx.is_empty() || idx.iter().any(|&i| i >= self.n_features) {
                return Err(format!("feature set {name:?} has out-of-range indices"));
            }
        }
        Ok(())
    }
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
        Ok(Self {
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
                self.run_solve(&a, &b, kc, m)
            } else if !self.cfg.standardize {
                let off = usize::from(self.cfg.add_intercept);
                for i in off..kc {
                    a[i * kc + i] += ridge;
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
            min_periods: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
        }
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
