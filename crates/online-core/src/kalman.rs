//! Kalman / random-walk-beta dynamic linear model (docs/PLAN.md §4.4).
//!
//! State per target: coefficient mean `b_j` and covariance `P_j` (k x k).
//! Observation `y_j = z' b_j + e`, `e ~ N(0, R_j)`; coefficients follow a random
//! walk `b_j <- b_j + w`, `w ~ N(0, Q)`.
//!
//! Per row (clock delta `d`, weight `w_row`):
//!
//! ```text
//! P_j <- P_j + Q * d / w_row          (predict; Q scaled by elapsed clock)
//! s    = z' P_j z + R_j / w_row       (innovation variance)
//! k    = P_j z / s                    (gain)
//! b_j <- b_j + k (y_j - z' b_j)
//! P_j <- P_j - k z' P_j
//! ```
//!
//! **Process noise from a per-factor halflife.** On standardized features, the
//! steady-state gain of a random-walk-beta filter matches EW-RLS with halflife
//! `h_i` when `q_i = sigma^2 * (ln2 / h_i)^2` (docs/PLAN.md §4.4). `halflife`
//! may be scalar or per factor; `halflife = inf` gives `q_i = 0`, pinning that
//! coefficient. An explicit `q` overrides the derivation.
//!
//! Features are standardized internally using a shared [`EwCov`] over `z`, so
//! `q_i` is on a comparable scale across features; `R_j` defaults to the EW
//! residual variance `sigma^2_j` unless `obs_var` is given.
//!
//! `P` is per target because the Riccati recursion depends on `R_j`. With
//! `share_p` the filter keeps one `P` driven by the mean `sigma^2` across
//! targets (docs/PLAN.md §4.4 [validate]).

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwCov};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    /// Decay used for the standardization statistics and the EW residual
    /// variance (NOT for the coefficients: those follow the random walk).
    pub decay: Decay,
    /// Per-factor coefficient halflife in clock units (length 1 or `k_total`).
    /// `f64::INFINITY` pins a coefficient. Ignored when `q` is given.
    pub halflife: Vec<f64>,
    /// Explicit process-noise variances (length `k_total`), overriding
    /// `halflife`.
    pub q: Option<Vec<f64>>,
    /// Fixed observation variance; defaults to the EW residual variance.
    pub obs_var: Option<f64>,
    /// Initial coefficient covariance (diagonal).
    pub p0: f64,
    pub share_p: bool,
    pub min_periods: f64,
    /// Standardize features internally before filtering (default).
    ///
    /// On by default because the halflife-derived process noise
    /// `q_i = sigma^2 (ln2/h_i)^2` is only comparable across features on a
    /// common scale. Turn it off when the features are already on a sensible
    /// scale and you want the filter to operate on them directly — that makes
    /// this exactly a Bayesian linear regression (with `q = 0` and a fixed
    /// `obs_var`), which is how it is cross-checked against river.
    #[serde(default = "default_true")]
    pub standardize: bool,
}

fn default_true() -> bool {
    true
}

impl KalmanCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        let k = self.k_total();
        if let Some(q) = &self.q {
            if q.len() != k {
                return Err(format!("kalman: q must have length {k}"));
            }
            if q.iter().any(|&v| v < 0.0) {
                return Err("kalman: q values must be >= 0".into());
            }
        } else {
            if self.halflife.len() != 1 && self.halflife.len() != k {
                return Err(format!("kalman: halflife must have length 1 or {k}"));
            }
            if self.halflife.iter().any(|&h| h <= 0.0) {
                return Err("kalman: halflife values must be > 0 (inf pins)".into());
            }
        }
        if self.p0 <= 0.0 {
            return Err("kalman: p0 must be > 0".into());
        }
        if self.obs_var.is_some_and(|v| v <= 0.0) {
            return Err("kalman: obs_var must be > 0".into());
        }
        Ok(())
    }

    /// Per-slot halflife, broadcast to `k_total`.
    fn halflife_per_slot(&self) -> Vec<f64> {
        let k = self.k_total();
        if self.halflife.len() == 1 {
            vec![self.halflife[0]; k]
        } else {
            self.halflife.clone()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kalman {
    cfg: KalmanCfg,
    /// Standardization stats over `z` (shared across targets).
    cov: EwCov,
    /// Per target: coefficient mean on the standardized scale.
    beta: Vec<Vec<f64>>,
    /// Per target (or one when `share_p`): covariance, row-major `k*k`.
    p: Vec<Vec<f64>>,
    /// EW residual variance per target and its weight sum.
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    zs: Vec<f64>,
    #[serde(skip)]
    pz: Vec<f64>,
    #[serde(skip)]
    gain: Vec<f64>,
}

impl Kalman {
    pub fn new(cfg: KalmanCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        let mut p_init = vec![0.0; k * k];
        for i in 0..k {
            p_init[i * k + i] = cfg.p0;
        }
        Ok(Self {
            cov: EwCov::new(k),
            beta: vec![vec![0.0; k]; m],
            p: vec![p_init; n_p],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            wj: vec![0.0; m],
            zbuf: vec![0.0; k],
            zs: vec![0.0; k],
            pz: vec![0.0; k],
            gain: vec![0.0; k],
            cfg,
        })
    }

    pub fn cfg(&self) -> &KalmanCfg {
        &self.cfg
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Predictive variance of the *last* prediction, per target:
    /// `zᵀ P_j z + R_j` — parameter uncertainty plus observation noise
    /// (ENHANCEMENTS E12).
    ///
    /// This is the piece `sigma` alone cannot give. `sigma` is the spread of
    /// realized errors; this also knows how unsure the filter is about its own
    /// coefficients, so it is wide during warmup and after a gap, and narrows
    /// as evidence accumulates. Only Kalman tracks `P`, so only Kalman can
    /// report it exactly.
    pub fn pred_var(&self) -> Vec<f64> {
        let k = self.cfg.k_total();
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                let p = &self.p[pi];
                let mut quad = 0.0;
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * self.zs[jj];
                    }
                    quad += self.zs[i] * acc;
                }
                let r = self.cfg.obs_var.unwrap_or_else(|| {
                    let s2 = if self.cfg.share_p {
                        self.sig2.iter().sum::<f64>() / self.cfg.n_targets as f64
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 { s2 } else { f64::NAN }
                });
                quad + r
            })
            .collect()
    }

    pub fn n_eff(&self) -> f64 {
        self.cov.n_eff()
    }

    /// Feature scales used for standardization: sd for features, 1 for the
    /// intercept slot. Zero-variance features get scale 1 (their standardized
    /// value is then their centered value, i.e. 0). All ones when
    /// `standardize` is off.
    fn scales(&self) -> Vec<f64> {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.add_intercept);
        if !self.cfg.standardize {
            return vec![1.0; k];
        }
        (0..k)
            .map(|i| {
                if i < off {
                    1.0
                } else {
                    let v = self.cov.var(i);
                    let raw = self.cov.raw(i, i);
                    if crate::variance_is_usable(v, raw) {
                        v.sqrt()
                    } else {
                        1.0
                    }
                }
            })
            .collect()
    }

    /// Coefficients in the ORIGINAL feature units, per target.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        if !self.cfg.standardize {
            return self.beta.clone();
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.add_intercept);
        let s = self.scales();
        let mut out = Vec::with_capacity(self.cfg.n_targets);
        for b in &self.beta {
            let mut c = vec![0.0; k];
            for (i, ci) in c.iter_mut().enumerate().skip(off) {
                *ci = b[i] / s[i];
            }
            if self.cfg.add_intercept {
                // b0_std is on centered features: unshift by the feature means.
                let mut b0 = b[0];
                for (i, ci) in c.iter().enumerate().skip(off) {
                    b0 -= ci * self.cov.mean(i);
                }
                c[0] = b0;
            }
            out.push(c);
        }
        out
    }

    /// Process-noise variances for this row: `q_i = sigma^2 * (ln2 / h_i)^2`
    /// (steady-state gain matching with EW-RLS on standardized features).
    fn q_vec(&self, sigma2: f64) -> Vec<f64> {
        if let Some(q) = &self.cfg.q {
            return q.clone();
        }
        self.cfg
            .halflife_per_slot()
            .into_iter()
            .map(|h| {
                if h.is_infinite() {
                    0.0
                } else {
                    let r = std::f64::consts::LN_2 / h;
                    sigma2 * r * r
                }
            })
            .collect()
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.zs = vec![0.0; k];
            self.pz = vec![0.0; k];
            self.gain = vec![0.0; k];
        }
    }
}

impl OnlineModel for Kalman {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.add_intercept);
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // Standardized regressors from the stats BEFORE this row's update.
        let s = self.scales();
        for (i, zs) in self.zs.iter_mut().enumerate() {
            *zs = if !self.cfg.standardize {
                self.zbuf[i]
            } else if i < off {
                1.0
            } else {
                (self.zbuf[i] - self.cov.mean(i)) / s[i]
            };
        }

        // ---- predict (state before the update) ----
        let n_eff = self.cov.n_eff();
        let ready = n_eff >= self.cfg.min_periods;
        let mut pred = vec![f64::NAN; m];
        if ready {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = self.zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum();
                }
            }
        }

        // ---- Kalman update per target ----
        for j in 0..m {
            let pi = if self.cfg.share_p { 0 } else { j };
            let sigma2 = self.cfg.obs_var.unwrap_or_else(|| {
                let s2 = if self.cfg.share_p {
                    self.sig2.iter().sum::<f64>() / m as f64
                } else {
                    self.sig2[j]
                };
                if s2 > 0.0 { s2 } else { 1.0 }
            });
            // Process step: P += Q * d_clock (only for the target that owns P,
            // or once when shared).
            if !self.cfg.share_p || j == 0 {
                let q = self.q_vec(sigma2);
                let p = &mut self.p[pi];
                for i in 0..k {
                    p[i * k + i] += q[i] * d_clock;
                }
            }
            let Some(yj) = y[j] else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            if weight <= 0.0 {
                continue;
            }
            // pz = P z
            {
                let p = &self.p[pi];
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * self.zs[jj];
                    }
                    self.pz[i] = acc;
                }
            }
            let zpz: f64 = self.zs.iter().zip(&self.pz).map(|(z, p)| z * p).sum();
            let s_inn = zpz + sigma2 / weight;
            if s_inn > 0.0 {
                for i in 0..k {
                    self.gain[i] = self.pz[i] / s_inn;
                }
                let pred_now: f64 = self.zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum();
                let err = yj - pred_now;
                for (b, g) in self.beta[j].iter_mut().zip(&self.gain) {
                    *b += g * err;
                }
                let p = &mut self.p[pi];
                for i in 0..k {
                    let gi = self.gain[i];
                    let row = i * k;
                    for jj in 0..k {
                        p[row + jj] -= gi * self.pz[jj];
                    }
                }
            }
            // EW residual variance from the out-of-sample prediction.
            if pred[j].is_finite() {
                let resid = yj - pred[j];
                let ws_new = lam * self.wsig[j] + weight;
                self.sig2[j] =
                    (lam * self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                self.wsig[j] = ws_new;
            }
            self.wj[j] = lam * self.wj[j] + weight;
        }

        // Standardization stats update last, so this row's z used the prior stats.
        self.cov.update(&self.zbuf, lam, weight);

        Step {
            pred,
            coef: None,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Kalman(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Kalman(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "kalman",
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(k: usize, m: usize, hl: Vec<f64>) -> KalmanCfg {
        KalmanCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(200.0),
            halflife: hl,
            q: None,
            obs_var: None,
            p0: 1.0,
            share_p: false,
            min_periods: 10.0,
            standardize: true,
        }
    }

    /// Feed a deterministic stream, returning the fitted filter.
    fn fit(cfg: KalmanCfg, n: usize, seed: u64) -> Kalman {
        let m = cfg.n_targets;
        let mut model = Kalman::new(cfg).unwrap();
        let mut s = seed;
        for i in 0..n {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let ys: Vec<Option<f64>> = (0..m)
                .map(|j| Some((j as f64 + 1.0) * (2.0 * x[0] - x[1]) + 0.1 * lcg(&mut s)))
                .collect();
            model.step(&x, &ys, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        model
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let bad = |f: &dyn Fn(&mut KalmanCfg), want: &str| {
            let mut c = cfg(2, 1, vec![100.0]);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        let good = |f: &dyn Fn(&mut KalmanCfg)| {
            let mut c = cfg(2, 1, vec![100.0]);
            f(&mut c);
            c.validate().expect("should be accepted");
        };

        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");

        // `q` is the process noise per slot: one entry per coefficient,
        // including the intercept, and zero means "pinned".
        bad(&|c| c.q = Some(vec![0.0; 2]), "length 3");
        bad(&|c| c.q = Some(vec![0.0, 0.0, -1e-9]), "must be >= 0");
        good(&|c| c.q = Some(vec![0.0; 3]));

        // Without `q`, the halflives are broadcast: one value, or one per slot.
        bad(&|c| c.halflife = vec![1.0, 2.0], "length 1 or 3");
        bad(&|c| c.halflife = vec![0.0], "must be > 0");
        bad(&|c| c.halflife = vec![-1.0], "must be > 0");
        good(&|c| c.halflife = vec![f64::INFINITY]);
        good(&|c| c.halflife = vec![1.0, 2.0, 3.0]);

        // p0 is the prior variance and obs_var the measurement noise; both
        // divide, so neither may be zero. obs_var may be absent (inferred).
        bad(&|c| c.p0 = 0.0, "p0 must be > 0");
        bad(&|c| c.p0 = -1.0, "p0 must be > 0");
        bad(&|c| c.obs_var = Some(0.0), "obs_var must be > 0");
        bad(&|c| c.obs_var = Some(-1.0), "obs_var must be > 0");
        good(&|c| c.obs_var = None);
        good(&|c| c.obs_var = Some(1e-9));

        cfg(2, 1, vec![100.0]).validate().unwrap();
    }

    #[test]
    fn standardize_defaults_to_on_when_a_state_file_omits_it() {
        // `#[serde(default = "default_true")]`: a state written before the
        // field existed must load with standardization on, which is the
        // behaviour that state was produced under. Defaulting to `false`
        // instead would silently change every restored model's numbers.
        let json = r#"{
            "n_features": 2, "n_targets": 1, "add_intercept": true,
            "decay": {"Halflife": 200.0}, "halflife": [100.0], "q": null,
            "obs_var": null, "p0": 1.0, "share_p": false, "min_periods": 10.0
        }"#;
        let cfg: KalmanCfg = serde_json::from_str(json).expect("should load without the field");
        assert!(cfg.standardize, "the omitted field must default to true");
    }

    #[test]
    fn coefficients_are_reported_in_the_callers_units() {
        // The filter works on standardized, centered features; `coefficients`
        // has to undo both -- divide by the scale, then unshift the intercept
        // by the feature means -- or the numbers a caller reads are not the
        // ones their data is in.
        // A coefficient halflife rather than a pinned one, so the filter keeps
        // re-learning as the standardization stats settle. With `q = 0` and a
        // near-zero observation noise it would instead converge in a handful of
        // rows, locking its betas into the standardized space of the first few
        // rows while `coefficients` unscales with the current stats -- which is
        // why the Bayesian-regression correspondence test turns standardization
        // off rather than working around it.
        let mut c = cfg(2, 1, vec![500.0]);
        c.min_periods = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 149u64;
        // Features on very different scales and far from zero, so a missing
        // unscale or a missing unshift is unmistakable.
        for i in 0..20_000 {
            let x = [500.0 + 10.0 * lcg(&mut s), 0.01 * lcg(&mut s)];
            let y = 12.0 + 0.25 * x[0] - 800.0 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!((b[1] - 0.25).abs() < 0.02, "slope 0: {}", b[1]);
        assert!((b[2] + 800.0).abs() < 10.0, "slope 1: {}", b[2]);
        // The intercept carries the accumulated slope error times mean(x0), so
        // it is the loosest of the three -- but it must be near 12, not near
        // the ~137 that dropping the unshift would give.
        assert!((b[0] - 12.0).abs() < 12.0, "intercept: {}", b[0]);
    }

    #[test]
    fn pred_var_is_the_quadratic_form_plus_observation_noise() {
        // `pred_var` is surfaced only through the Polars layer, so its
        // arithmetic had no test in this crate. It is z' P z + R, and both
        // halves are checked: the quadratic form against a longhand loop over
        // the stored covariance, and R against the configured or inferred
        // observation noise.
        let mut c = cfg(2, 1, vec![100.0]);
        c.obs_var = Some(0.25);
        c.min_periods = 3.0;
        let m = fit(c, 80, 61);
        let k = m.cfg.k_total();

        let mut want = 0.0;
        for i in 0..k {
            for j in 0..k {
                want += m.zs[i] * m.p[0][i * k + j] * m.zs[j];
            }
        }
        want += 0.25;
        let got = m.pred_var()[0];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!(got > 0.25, "must exceed the observation noise: {got}");

        // Without a configured obs_var it falls back to the tracked residual
        // variance of that target.
        let mut c = cfg(2, 1, vec![100.0]);
        c.min_periods = 3.0;
        let m = fit(c, 80, 61);
        let got = m.pred_var()[0];
        assert!(got > m.sigma2()[0], "{got} vs {}", m.sigma2()[0]);
    }

    #[test]
    fn share_p_shares_one_covariance_and_averages_the_noise() {
        // With `share_p` the process step runs once rather than once per
        // target, one covariance is kept, and the inferred observation noise
        // is the mean across targets rather than each target's own.
        let mut shared = cfg(2, 2, vec![100.0]);
        shared.share_p = true;
        shared.min_periods = 3.0;
        let ms = fit(shared, 200, 71);
        assert_eq!(ms.p.len(), 1, "one covariance for all targets");
        // Both targets read the same P, so their pred_var differs only through
        // ... nothing: z is shared too. They must be identical.
        let pv = ms.pred_var();
        assert!((pv[0] - pv[1]).abs() < 1e-12, "{pv:?}");
        // And that shared noise is the mean of the per-target residual
        // variances, which here differ by construction (target 1 is 2x target 0).
        let s2 = ms.sigma2();
        assert!(s2[1] > 2.0 * s2[0], "targets differ: {s2:?}");
        let mean = s2.iter().sum::<f64>() / 2.0;
        let quad = pv[0] - mean;
        assert!(
            quad > 0.0 && quad < mean,
            "R should be the mean: {pv:?} {s2:?}"
        );

        let mut separate = cfg(2, 2, vec![100.0]);
        separate.min_periods = 3.0;
        let msep = fit(separate, 200, 71);
        assert_eq!(msep.p.len(), 2, "one covariance per target");
        let pv2 = msep.pred_var();
        assert!(
            (pv2[0] - pv2[1]).abs() > 1e-6,
            "unshared targets should differ: {pv2:?}"
        );
    }

    #[test]
    fn a_null_target_decays_its_weights_and_leaves_the_filter_alone() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.decay = Decay::Halflife(10.0);
        c.min_periods = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 73u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.beta[0].clone();
        let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

        let lam = 0.5f64.powf(4.0 / 10.0);
        m.step(&[0.3, -0.2], &[None], 4.0, 1.0);
        assert_eq!(m.beta[0], beta, "no target, no correction");
        assert_eq!(m.sig2[0], sig2);
        assert!((m.wj[0] - wj * lam).abs() < 1e-12);
        assert!((m.wsig[0] - wsig * lam).abs() < 1e-12);
    }

    #[test]
    fn a_zero_weight_row_does_not_correct_the_filter() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.min_periods = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 79u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.beta[0].clone();
        m.step(&[0.3, -0.2], &[Some(-500.0)], 1.0, 0.0);
        assert_eq!(m.beta[0], beta, "weight 0 must not move the coefficients");
    }

    #[test]
    fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
        let mut c = cfg(1, 1, vec![f64::INFINITY]);
        c.decay = Decay::Halflife(25.0);
        c.min_periods = 3.0;
        c.standardize = false;
        c.obs_var = Some(0.5);
        let mut m = Kalman::new(c).unwrap();

        let (mut want, mut wsig) = (0.0, 0.0);
        let mut s = 83u64;
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
        assert!(wsig > 30.0 && want > 0.0);
    }

    /// With `standardize = false`, `q = 0` and a fixed `obs_var`, the filter is
    /// exactly a Bayesian linear regression: coefficients converge to the ridge
    /// solution with penalty `obs_var / p0`.
    #[test]
    fn unstandardized_with_no_process_noise_is_bayesian_regression() {
        let (p0, obs_var) = (10.0, 0.25);
        let mut m = Kalman::new(KalmanCfg {
            n_features: 2,
            n_targets: 1,
            add_intercept: false,
            decay: Decay::Halflife(f64::INFINITY),
            halflife: vec![f64::INFINITY],
            q: Some(vec![0.0, 0.0]),
            obs_var: Some(obs_var),
            p0,
            share_p: false,
            min_periods: 0.0,
            standardize: false,
        })
        .unwrap();
        // Accumulate the normal equations alongside, then compare with the
        // closed-form ridge solution (obs_var / p0 is the implied penalty).
        let mut s = 55u64;
        let (mut xtx, mut xty) = ([[0.0f64; 2]; 2], [0.0f64; 2]);
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.25 * x[0] - 0.5 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            for a in 0..2 {
                xty[a] += x[a] * y;
                for b in 0..2 {
                    xtx[a][b] += x[a] * x[b];
                }
            }
        }
        let lam = obs_var / p0;
        let (a, b, c, d) = (xtx[0][0] + lam, xtx[0][1], xtx[1][0], xtx[1][1] + lam);
        let det = a * d - b * c;
        let want = [
            (d * xty[0] - b * xty[1]) / det,
            (-c * xty[0] + a * xty[1]) / det,
        ];
        let got = &m.coefficients()[0];
        for i in 0..2 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "coef {i}: {} vs ridge closed form {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn tracks_a_static_beta() {
        let mut m = Kalman::new(cfg(2, 1, vec![500.0])).unwrap();
        let mut s = 3u64;
        for i in 0..2000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - 1.0 * x[1] + 0.5 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let c = &m.coefficients()[0];
        assert!((c[1] - 2.0).abs() < 0.1, "slope0 {}", c[1]);
        assert!((c[2] + 1.0).abs() < 0.1, "slope1 {}", c[2]);
        assert!((c[0] - 0.5).abs() < 0.1, "intercept {}", c[0]);
    }

    #[test]
    fn tracks_a_drifting_beta_better_than_a_pinned_one() {
        // Same data through a responsive filter and a pinned one: the responsive
        // filter must have lower out-of-sample error.
        let mut fast = Kalman::new(cfg(1, 1, vec![50.0])).unwrap();
        let mut pinned = Kalman::new(cfg(1, 1, vec![f64::INFINITY])).unwrap();
        let mut s = 4u64;
        let (mut e_fast, mut e_pin) = (0.0f64, 0.0f64);
        let mut beta = 1.0f64;
        for i in 0..3000 {
            beta += 0.01 * lcg(&mut s); // random walk
            let x = [lcg(&mut s)];
            let y = beta * x[0] + 0.05 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = fast.step(&x, &[Some(y)], d, 1.0);
            let b = pinned.step(&x, &[Some(y)], d, 1.0);
            if i > 500 {
                if a.pred[0].is_finite() {
                    e_fast += (y - a.pred[0]).powi(2);
                }
                if b.pred[0].is_finite() {
                    e_pin += (y - b.pred[0]).powi(2);
                }
            }
        }
        assert!(e_fast < e_pin, "fast {e_fast} should beat pinned {e_pin}");
    }

    #[test]
    fn infinite_halflife_pins_the_coefficient() {
        // Per-factor: slot 1 (x0) pinned, slot 2 (x1) free.
        let mut m = Kalman::new(cfg(2, 1, vec![1e9, f64::INFINITY, 30.0])).unwrap();
        let mut s = 5u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let q = m.q_vec(1.0);
        assert_eq!(q[1], 0.0, "pinned factor must have zero process noise");
        assert!(q[2] > 0.0);
    }

    #[test]
    fn explicit_q_overrides_halflife() {
        let mut c = cfg(1, 1, vec![10.0]);
        c.q = Some(vec![0.0, 0.25]);
        let m = Kalman::new(c).unwrap();
        assert_eq!(m.q_vec(99.0), vec![0.0, 0.25]);
    }

    #[test]
    fn share_p_keeps_one_covariance() {
        let mut c = cfg(2, 3, vec![100.0]);
        c.share_p = true;
        let mut m = Kalman::new(c).unwrap();
        assert_eq!(m.p.len(), 1);
        let mut s = 6u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = x[0];
            let y1 = x[1];
            let y2 = x[0] + x[1];
            m.step(
                &x,
                &[Some(y0), Some(y1), Some(y2)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert!(
            m.coefficients()
                .iter()
                .all(|c| c.iter().all(|v| v.is_finite()))
        );
    }

    /// Predictive variance must start wide and narrow with evidence — that is
    /// the whole reason to report it alongside `sigma`.
    #[test]
    fn predictive_variance_narrows_with_evidence() {
        let mut m = Kalman::new(cfg(2, 1, vec![f64::INFINITY])).unwrap();
        let mut s = 44u64;
        let mut early = 0.0;
        let mut late = 0.0;
        for i in 0..3000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i == 20 {
                early = m.pred_var()[0];
            }
            if i == 2999 {
                late = m.pred_var()[0];
            }
        }
        assert!(early.is_finite() && late.is_finite());
        assert!(
            late < early,
            "predictive variance should narrow: {early} -> {late}"
        );
    }

    #[test]
    fn predictive_variance_exceeds_the_observation_noise() {
        // It is parameter uncertainty PLUS noise, so it can never be smaller
        // than the noise alone.
        let mut c = cfg(2, 1, vec![f64::INFINITY]);
        c.obs_var = Some(0.25);
        let mut m = Kalman::new(c).unwrap();
        let mut s = 46u64;
        for i in 0..500 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            assert!(m.pred_var()[0] >= 0.25 - 1e-12);
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Kalman::new(cfg(2, 1, vec![100.0])).unwrap();
        let mut s = 7u64;
        let rows: Vec<([f64; 2], f64)> = (0..120)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..60].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Kalman::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Kalman::new(cfg(1, 2, vec![100.0])).unwrap();
        let mut s = 8u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.beta[1].clone();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred[1].is_finite());
        assert_eq!(m.beta[1], before);
    }
}
