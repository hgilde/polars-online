//! Stochastic gradient descent with pluggable losses (docs/ENHANCEMENTS.md E16).
//!
//! The cheap baseline: one gradient step per row, no solves, O(k) per row rather
//! than O(k²). Also the only model here that handles **count targets**, via the
//! Poisson loss with a log link — none of the exact solvers cover those.
//!
//! Every loss shares the same shape. With `eta = z·b` the linear predictor,
//! `p = link(eta)` the prediction, and `d = dL/d(eta)`:
//!
//! | loss | link | `p` | `d` |
//! |---|---|---|---|
//! | `Squared` | identity | `eta` | `p − y` |
//! | `Huber` | identity | `eta` | `clamp(p − y, ±delta)` |
//! | `Quantile` | identity | `eta` | `1{y < p} − tau` |
//! | `EpsilonInsensitive` | identity | `eta` | `0` if `|p − y| ≤ eps`, else `sign(p − y)` |
//! | `Poisson` | log | `exp(eta)` | `p − y` |
//! | `Logistic` | sigmoid | `sigmoid(eta)` | `p − y` |
//!
//! then `g_i = d · z_i · w + l2 · b_i` and `b_i -= lr_i · g_i`.
//!
//! Learning rates ([`LearningRate`]): a constant, an inverse-scaling schedule
//! that anneals with `n_eff`, or AdaGrad's per-coordinate `lr / (sqrt(G_i) + eps)`.
//! AdaGrad's accumulator and `n_eff` are both decayed on the model's clock, so
//! an annealed or adapted rate re-opens after a long gap instead of staying
//! frozen at whatever it had converged to.

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// Loss function, and with it the link (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SgdLoss {
    Squared,
    /// `delta` is in target units (not residual std, unlike `robust`'s Huber).
    Huber {
        delta: f64,
    },
    Quantile {
        tau: f64,
    },
    /// Ignores residuals within `eps` — the SVR loss.
    EpsilonInsensitive {
        eps: f64,
    },
    /// Log link, for non-negative count targets.
    Poisson,
    Logistic,
}

/// Per-coordinate step size.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningRate {
    Constant,
    /// `lr / (1 + n_eff)^power`.
    InvScaling {
        power: f64,
    },
    /// `lr / (sqrt(G_i) + 1e-8)`, `G_i` the decayed sum of squared gradients.
    AdaGrad,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SgdCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    pub loss: SgdLoss,
    pub learning_rate: f64,
    pub schedule: LearningRate,
    /// Ridge penalty added to the gradient. The intercept is never penalized.
    pub l2: f64,
    pub min_periods: f64,
    /// Standardize features against their own running moments before the
    /// gradient step (ENHANCEMENTS E24), unscaling the coefficients on the way
    /// out so they stay in the caller's units.
    ///
    /// Gradient methods are the ones that need this: a single learning rate has
    /// to suit every coordinate, so a feature measured in thousands and one
    /// measured in basis points cannot both converge. The exact solvers do not
    /// care (they standardize inside the solve, or not at all).
    #[serde(default)]
    pub scale_features: bool,
    /// Cap on `|gradient|` before the step. **Finite by default** (`1e3` via the
    /// spec layer), not because ordinary losses need it but because a log-link
    /// loss does: with `Poisson`, `p = exp(eta)`, so one row that pushes `eta`
    /// up makes the next gradient exponentially larger, and a constant learning
    /// rate diverges within a few thousand rows. Measured on a 30k-row Poisson
    /// stream: unclipped the intercept ran to -4e10, clipped at 1e3 it recovers
    /// the true `[0.4, 0.8]`. The cap does not bind for identity-link losses at
    /// ordinary scales — a squared-loss fit is bit-identical with and without
    /// it. `inf` disables it.
    pub clip_gradient: f64,
}

impl SgdCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("sgd: n_features and n_targets must be >= 1".into());
        }
        if self.learning_rate <= 0.0 || self.learning_rate.is_nan() {
            return Err("sgd: learning_rate must be > 0".into());
        }
        if self.l2 < 0.0 {
            return Err("sgd: l2 must be >= 0".into());
        }
        if self.clip_gradient <= 0.0 {
            return Err("sgd: clip_gradient must be > 0 (use inf to disable)".into());
        }
        match self.loss {
            SgdLoss::Huber { delta } if delta <= 0.0 => {
                return Err("sgd: huber delta must be > 0".into());
            }
            SgdLoss::Quantile { tau }
                if !(0.0..=1.0).contains(&tau) || tau == 0.0 || tau == 1.0 =>
            {
                return Err("sgd: quantile must be in (0, 1)".into());
            }
            SgdLoss::EpsilonInsensitive { eps } if eps < 0.0 => {
                return Err("sgd: eps must be >= 0".into());
            }
            _ => {}
        }
        if let LearningRate::InvScaling { power } = self.schedule {
            if power < 0.0 {
                return Err("sgd: inv_scaling power must be >= 0".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sgd {
    cfg: SgdCfg,
    /// Running feature moments, when `scale_features` is on.
    #[serde(default)]
    scaler: Option<crate::EwCov>,
    /// Coefficients per target.
    beta: Vec<Vec<f64>>,
    /// AdaGrad accumulators per target (empty for the other schedules).
    g2: Vec<Vec<f64>>,
    w_sum: f64,
    #[serde(skip)]
    zbuf: Vec<f64>,
}

impl Sgd {
    pub fn new(cfg: SgdCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let g2 = if matches!(cfg.schedule, LearningRate::AdaGrad) {
            vec![vec![0.0; k]; m]
        } else {
            Vec::new()
        };
        let scaler = cfg.scale_features.then(|| crate::EwCov::new(k));
        Ok(Self {
            scaler,
            beta: vec![vec![0.0; k]; m],
            g2,
            w_sum: 0.0,
            zbuf: vec![0.0; k],
            cfg,
        })
    }

    pub fn cfg(&self) -> &SgdCfg {
        &self.cfg
    }

    /// Coefficients in the caller's units. With `scale_features` the model
    /// fits on standardized inputs, so they are unscaled here and the intercept
    /// absorbs the shift.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let Some(sc) = &self.scaler else {
            return self.beta.clone();
        };
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.add_intercept);
        let scales = self.scales();
        self.beta
            .iter()
            .map(|b| {
                let mut c = vec![0.0; k];
                for i in off..k {
                    c[i] = b[i] / scales[i];
                }
                if self.cfg.add_intercept {
                    let mut b0 = b[0];
                    for (i, ci) in c.iter().enumerate().skip(off) {
                        b0 -= ci * sc.mean(i);
                    }
                    c[0] = b0;
                }
                c
            })
            .collect()
    }

    /// Per-slot scale: the running sd for features, 1 for the intercept and for
    /// a feature with no spread yet.
    fn scales(&self) -> Vec<f64> {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.add_intercept);
        match &self.scaler {
            None => vec![1.0; k],
            Some(sc) => (0..k)
                .map(|i| {
                    if i < off {
                        return 1.0;
                    }
                    let v = sc.var(i);
                    if crate::variance_is_usable(v, sc.raw(i, i)) {
                        v.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect(),
        }
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// `p = link(eta)`.
    fn link(&self, eta: f64) -> f64 {
        match self.cfg.loss {
            SgdLoss::Poisson => eta.clamp(-30.0, 30.0).exp(),
            SgdLoss::Logistic => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            _ => eta,
        }
    }

    /// `dL/d(eta)` for the configured loss.
    fn dloss(&self, p: f64, y: f64) -> f64 {
        let r = p - y;
        match self.cfg.loss {
            SgdLoss::Squared | SgdLoss::Poisson | SgdLoss::Logistic => r,
            SgdLoss::Huber { delta } => r.clamp(-delta, delta),
            SgdLoss::Quantile { tau } => f64::from(y < p) - tau,
            SgdLoss::EpsilonInsensitive { eps } => {
                if r.abs() <= eps {
                    0.0
                } else {
                    r.signum()
                }
            }
        }
    }

    fn ensure_buffers(&mut self) {
        if self.zbuf.len() != self.cfg.k_total() {
            self.zbuf = vec![0.0; self.cfg.k_total()];
        }
    }
}

impl OnlineModel for Sgd {
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

        // Standardize against the moments from BEFORE this row, so the scaling
        // cannot see the row it is scaling (ENHANCEMENTS E24). The raw values
        // are kept to update the scaler afterwards.
        let raw_z: Vec<f64> = if self.scaler.is_some() {
            self.zbuf.clone()
        } else {
            Vec::new()
        };
        if let Some(sc) = &self.scaler {
            let means: Vec<f64> = (0..k).map(|i| sc.mean(i)).collect();
            let scales = self.scales();
            for (i, z) in self.zbuf.iter_mut().enumerate().skip(off) {
                *z = (*z - means[i]) / scales[i];
            }
        }

        // Decay first, so a long gap re-opens an annealed or adapted rate.
        if lam != 1.0 {
            for g in self.g2.iter_mut() {
                for v in g.iter_mut() {
                    *v *= lam;
                }
            }
        }
        // `n_eff` is the weight *before* this row's update and *before* its
        // decay, which is the convention every other model reports and gates
        // on (see `EwRidgeCfg::min_periods`). Decaying it here would make
        // `min_periods` mean a slightly different number of rows for `sgd`
        // than for `ewridge`, which is exactly the kind of quiet divergence
        // the cross-model semantics suite exists to catch.
        let n_eff = self.w_sum;

        let ready = n_eff >= self.cfg.min_periods;
        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            let eta: f64 = self
                .zbuf
                .iter()
                .zip(&self.beta[j])
                .map(|(z, b)| z * b)
                .sum();
            let p = self.link(eta);
            if ready {
                pred[j] = p;
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 || !yj.is_finite() {
                continue;
            }
            let d = self.dloss(p, yj);
            for i in 0..k {
                let mut g = d * self.zbuf[i] * weight;
                if i >= off {
                    g += self.cfg.l2 * self.beta[j][i];
                }
                g = g.clamp(-self.cfg.clip_gradient, self.cfg.clip_gradient);
                let lr = match self.cfg.schedule {
                    LearningRate::Constant => self.cfg.learning_rate,
                    LearningRate::InvScaling { power } => {
                        self.cfg.learning_rate / (1.0 + n_eff).powf(power)
                    }
                    LearningRate::AdaGrad => {
                        self.g2[j][i] += g * g;
                        self.cfg.learning_rate / (self.g2[j][i].sqrt() + 1e-8)
                    }
                };
                self.beta[j][i] -= lr * g;
            }
        }
        if let Some(sc) = &mut self.scaler {
            sc.update(&raw_z, lam, weight);
        }
        self.w_sum = lam * self.w_sum + weight;

        Step {
            pred,
            coef: None,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Sgd(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Sgd(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "sgd",
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

    fn cfg(k: usize, loss: SgdLoss) -> SgdCfg {
        SgdCfg {
            n_features: k,
            n_targets: 1,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss,
            learning_rate: 0.05,
            schedule: LearningRate::Constant,
            l2: 0.0,
            min_periods: 5.0,
            clip_gradient: 1e12,
            scale_features: false,
        }
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let bad = |f: &dyn Fn(&mut SgdCfg), want: &str| {
            let mut c = cfg(2, SgdLoss::Squared);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");
        bad(&|c| c.learning_rate = 0.0, "learning_rate must be > 0");
        bad(&|c| c.learning_rate = f64::NAN, "learning_rate must be > 0");
        bad(&|c| c.l2 = -1e-9, "l2 must be >= 0");
        // clip_gradient is a magnitude bound, so inf disables it rather than
        // being an error, but zero would clip everything to nothing.
        bad(&|c| c.clip_gradient = 0.0, "clip_gradient must be > 0");
        let mut ok = cfg(2, SgdLoss::Squared);
        ok.clip_gradient = f64::INFINITY;
        ok.validate().unwrap();

        bad(&|c| c.loss = SgdLoss::Huber { delta: 0.0 }, "huber delta");
        for t in [0.0, 1.0, 1.5] {
            bad(
                &|c| c.loss = SgdLoss::Quantile { tau: t },
                "quantile must be in",
            );
        }
        bad(
            &|c| c.loss = SgdLoss::EpsilonInsensitive { eps: -1.0 },
            "eps must be >= 0",
        );
        // eps = 0 is a legal (if pointless) epsilon-insensitive loss.
        cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.0 })
            .validate()
            .unwrap();

        bad(
            &|c| c.schedule = LearningRate::InvScaling { power: -0.1 },
            "power must be >= 0",
        );
        let mut ok = cfg(2, SgdLoss::Squared);
        ok.schedule = LearningRate::InvScaling { power: 0.0 };
        ok.validate().unwrap();
    }

    #[test]
    fn each_loss_has_the_gradient_it_claims() {
        // `dloss` is dL/d(eta) and `link` maps eta to the prediction. Both are
        // small and entirely arithmetic, and every model output depends on
        // them, so they are checked directly rather than through a fit.
        let m = |loss| Sgd::new(cfg(1, loss)).unwrap();

        // Squared: the plain residual, in both directions.
        assert_eq!(m(SgdLoss::Squared).dloss(3.0, 1.0), 2.0);
        assert_eq!(m(SgdLoss::Squared).dloss(1.0, 3.0), -2.0);

        // Huber: the residual, clipped symmetrically at delta.
        let h = m(SgdLoss::Huber { delta: 1.5 });
        assert_eq!(h.dloss(1.0, 0.0), 1.0, "inside the band: squared");
        assert_eq!(h.dloss(9.0, 0.0), 1.5, "outside: clipped");
        assert_eq!(h.dloss(-9.0, 0.0), -1.5);

        // Quantile: a constant gradient whose sign depends on which side of
        // the prediction the target fell, asymmetric except at tau = 0.5.
        let q = m(SgdLoss::Quantile { tau: 0.9 });
        assert_eq!(q.dloss(5.0, 1.0), 1.0 - 0.9, "over-predicted");
        assert_eq!(q.dloss(1.0, 5.0), -0.9, "under-predicted");
        let med = m(SgdLoss::Quantile { tau: 0.5 });
        assert_eq!(
            med.dloss(5.0, 1.0),
            -med.dloss(1.0, 5.0),
            "symmetric at 0.5"
        );

        // Epsilon-insensitive: exactly zero inside the tube.
        let e = m(SgdLoss::EpsilonInsensitive { eps: 1.0 });
        assert_eq!(e.dloss(0.5, 0.0), 0.0);
        assert_eq!(e.dloss(1.0, 0.0), 0.0, "the boundary is inside");
        assert!(e.dloss(3.0, 0.0) > 0.0);
        assert!(e.dloss(-3.0, 0.0) < 0.0);

        // Links. Squared and the robust losses are the identity; logistic and
        // Poisson are not, and both must stay finite at extreme inputs.
        assert_eq!(m(SgdLoss::Squared).link(-7.0), -7.0);
        let lg = m(SgdLoss::Logistic);
        assert_eq!(lg.link(0.0), 0.5);
        for eta in [900.0, -900.0] {
            let p = lg.link(eta);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "link({eta}) = {p}"
            );
        }
        let po = m(SgdLoss::Poisson);
        assert_eq!(po.link(0.0), 1.0);
        assert!(po.link(1e9).is_finite(), "the exponent is clamped");
        assert!(po.link(-1e9) > 0.0, "a rate is positive");
    }

    /// The case scaling exists for: one feature in thousands, one in
    /// thousandths. A single learning rate cannot suit both.
    #[test]
    fn scaling_rescues_badly_scaled_features() {
        let run = |scale: bool| {
            let mut c = cfg(2, SgdLoss::Squared);
            c.scale_features = scale;
            c.learning_rate = 0.01;
            c.min_periods = 0.0;
            let mut m = Sgd::new(c).unwrap();
            let mut s = 61u64;
            for i in 0..20000 {
                let x = [1000.0 * lcg(&mut s), 0.001 * lcg(&mut s)];
                let y = 0.002 * x[0] + 900.0 * x[1];
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients()[0].clone()
        };
        let plain = run(false);
        let scaled = run(true);
        let err = |b: &[f64]| (b[1] - 0.002).abs() / 0.002 + (b[2] - 900.0).abs() / 900.0;
        assert!(
            err(&scaled) < err(&plain),
            "scaled {scaled:?} should beat unscaled {plain:?} (truth [_, 0.002, 900])"
        );
        assert!(err(&scaled) < 0.2, "scaled fit still poor: {scaled:?}");
    }

    #[test]
    fn scaling_reports_coefficients_in_the_callers_units() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.scale_features = true;
        c.learning_rate = 0.1;
        c.min_periods = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 63u64;
        for i in 0..20000 {
            let x = [500.0 + 100.0 * lcg(&mut s)];
            let y = 0.05 * x[0] + 3.0;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!(
            (b[1] - 0.05).abs() < 5e-3,
            "slope in original units: {}",
            b[1]
        );
        assert!(
            (b[0] - 3.0).abs() < 1.0,
            "intercept absorbs the shift: {}",
            b[0]
        );
    }

    #[test]
    fn scaling_is_out_of_sample() {
        // The scaler must not see the row it is scaling: an enormous first row
        // should not be normalized away by its own magnitude.
        let mut c = cfg(1, SgdLoss::Squared);
        c.scale_features = true;
        c.min_periods = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let before = m.coefficients()[0].clone();
        m.step(&[1e6], &[Some(1.0)], 0.0, 1.0);
        assert_ne!(
            m.coefficients()[0],
            before,
            "the row should still have moved the fit"
        );
    }

    /// Runs a stream and returns the final coefficients.
    fn fit(
        cfg: SgdCfg,
        n: usize,
        seed: u64,
        mut f: impl FnMut(&[f64], &mut u64) -> f64,
    ) -> Vec<f64> {
        let k = cfg.n_features;
        let mut m = Sgd::new(cfg).unwrap();
        let mut s = seed;
        for i in 0..n {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y = f(&x, &mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    }

    #[test]
    fn squared_loss_recovers_the_coefficients() {
        let b = fit(cfg(2, SgdLoss::Squared), 20000, 1, |x, _| {
            1.5 * x[0] - 0.5 * x[1] + 0.25
        });
        assert!((b[0] - 0.25).abs() < 0.05, "intercept {}", b[0]);
        assert!((b[1] - 1.5).abs() < 0.05, "slope0 {}", b[1]);
        assert!((b[2] + 0.5).abs() < 0.05, "slope1 {}", b[2]);
    }

    #[test]
    fn poisson_loss_recovers_a_log_rate() {
        // y ~ counts with log-rate 0.4 + 0.8 x0. The canonical link means the
        // coefficients live on the log scale.
        let mut c = cfg(1, SgdLoss::Poisson);
        c.learning_rate = 0.02;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..40000 {
            let x = [lcg(&mut s)];
            let rate = (0.4 + 0.8 * x[0]).exp();
            // Poisson draw by inversion (rate is small enough for this to be cheap)
            let mut kdraw = 0.0;
            let mut prod = (lcg(&mut s) + 1.0) / 2.0;
            let limit = (-rate).exp();
            while prod > limit && kdraw < 50.0 {
                prod *= (lcg(&mut s) + 1.0) / 2.0;
                kdraw += 1.0;
            }
            m.step(&x, &[Some(kdraw)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!((b[0] - 0.4).abs() < 0.15, "log-intercept {}", b[0]);
        assert!((b[1] - 0.8).abs() < 0.15, "log-slope {}", b[1]);
    }

    #[test]
    fn poisson_predictions_are_non_negative() {
        let mut c = cfg(1, SgdLoss::Poisson);
        c.min_periods = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 12u64;
        for i in 0..2000 {
            let x = [lcg(&mut s) * 5.0];
            let st = m.step(&x, &[Some(3.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if st.pred[0].is_finite() {
                assert!(st.pred[0] >= 0.0, "poisson predicted {}", st.pred[0]);
            }
        }
    }

    #[test]
    fn quantile_loss_tracks_the_level() {
        let lo = fit(cfg(1, SgdLoss::Quantile { tau: 0.1 }), 20000, 13, |_, s| {
            1.0 + 2.0 * lcg(s)
        });
        let hi = fit(cfg(1, SgdLoss::Quantile { tau: 0.9 }), 20000, 13, |_, s| {
            1.0 + 2.0 * lcg(s)
        });
        assert!(
            lo[0] < hi[0],
            "tau=0.1 intercept {} !< tau=0.9 {}",
            lo[0],
            hi[0]
        );
    }

    /// `y = 2x`, with every 20th row replaced by a gross outlier.
    fn contaminated() -> impl FnMut(&[f64], &mut u64) -> f64 {
        let mut row = 0u64;
        move |x: &[f64], s: &mut u64| {
            row += 1;
            if row % 20 == 0 {
                500.0 * lcg(s)
            } else {
                2.0 * x[0]
            }
        }
    }

    #[test]
    fn huber_resists_outliers() {
        let hub = fit(
            cfg(1, SgdLoss::Huber { delta: 1.0 }),
            20000,
            17,
            contaminated(),
        );
        let sq = fit(cfg(1, SgdLoss::Squared), 20000, 17, contaminated());
        assert!(
            (hub[1] - 2.0).abs() < (sq[1] - 2.0).abs(),
            "huber {} should beat squared {} (truth 2.0)",
            hub[1],
            sq[1]
        );
    }

    #[test]
    fn epsilon_insensitive_needs_an_annealed_rate_to_settle() {
        // Its subgradient is sign-valued (+/-1), so the step size does not
        // shrink near the optimum: with a constant rate the coefficient
        // oscillates in a band, and only an annealed schedule settles. Both
        // halves are asserted, because the constant-rate behaviour is a real
        // property of this loss rather than a bug.
        let mut constant = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.2 });
        constant.learning_rate = 0.01;
        let b_const = fit(constant, 30000, 19, |x, s| 2.0 * x[0] + 0.1 * lcg(s));

        let mut annealed = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.2 });
        annealed.learning_rate = 0.5;
        annealed.schedule = LearningRate::InvScaling { power: 0.5 };
        let b_anneal = fit(annealed, 30000, 19, |x, s| 2.0 * x[0] + 0.1 * lcg(s));

        assert!(
            (b_anneal[1] - 2.0).abs() < (b_const[1] - 2.0).abs(),
            "annealed {} should beat constant {} (truth 2.0)",
            b_anneal[1],
            b_const[1]
        );
        assert!(
            (b_anneal[1] - 2.0).abs() < 0.2,
            "annealed slope {}",
            b_anneal[1]
        );
    }

    #[test]
    fn epsilon_insensitive_ignores_residuals_inside_the_tube() {
        // A pure-noise target smaller than eps must leave the fit at zero.
        let mut c = cfg(1, SgdLoss::EpsilonInsensitive { eps: 1.0 });
        c.learning_rate = 0.01;
        let b = fit(c, 5000, 21, |_, s| 0.3 * lcg(s));
        assert!(b[0].abs() < 1e-12 && b[1].abs() < 1e-12, "moved: {b:?}");
    }

    #[test]
    fn adagrad_and_inv_scaling_also_converge() {
        for schedule in [
            LearningRate::AdaGrad,
            LearningRate::InvScaling { power: 0.25 },
        ] {
            let mut c = cfg(2, SgdLoss::Squared);
            c.schedule = schedule;
            c.learning_rate = if matches!(schedule, LearningRate::AdaGrad) {
                0.5
            } else {
                0.1
            };
            let b = fit(c, 30000, 23, |x, _| 1.5 * x[0] - 0.5 * x[1]);
            assert!((b[1] - 1.5).abs() < 0.15, "{schedule:?}: slope0 {}", b[1]);
            assert!((b[2] + 0.5).abs() < 0.15, "{schedule:?}: slope1 {}", b[2]);
        }
    }

    #[test]
    fn l2_shrinks_toward_zero() {
        let plain = fit(cfg(1, SgdLoss::Squared), 5000, 29, |x, _| 2.0 * x[0]);
        let mut c = cfg(1, SgdLoss::Squared);
        c.l2 = 1.0;
        let shrunk = fit(c, 5000, 29, |x, _| 2.0 * x[0]);
        assert!(shrunk[1].abs() < plain[1].abs());
    }

    /// A log-link loss diverges without the cap. Deterministic rather than
    /// stochastic: one large count is enough to push `eta` into the exp clamp,
    /// after which the gradient is ~1e13 and a single step throws the
    /// coefficients to ~1e11. (The same thing happens on real Poisson data,
    /// where the heavy tail supplies the large count — measured on a 30k-row
    /// stream, the intercept ran to -4e10.)
    #[test]
    fn poisson_diverges_without_gradient_clipping() {
        let run = |clip: f64| {
            let mut c = cfg(1, SgdLoss::Poisson);
            c.learning_rate = 0.02;
            c.clip_gradient = clip;
            c.min_periods = 0.0;
            let mut m = Sgd::new(c).unwrap();
            for i in 0..5 {
                m.step(&[1.0], &[Some(1e6)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients()[0].clone()
        };
        let unclipped = run(f64::INFINITY);
        assert!(
            unclipped.iter().any(|b| b.abs() > 1e9),
            "expected the unclipped fit to blow up, got {unclipped:?}"
        );
        let clipped = run(1e3);
        assert!(
            clipped.iter().all(|b| b.abs() < 1e3),
            "the cap should bound the coefficients, got {clipped:?}"
        );
    }

    #[test]
    fn clip_gradient_bounds_the_step() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.clip_gradient = 1.0;
        c.min_periods = 0.0;
        let mut m = Sgd::new(c).unwrap();
        // one absurd row: with lr 0.05 and the clip at 1, |db| <= 0.05
        m.step(&[1.0], &[Some(1e9)], 0.0, 1.0);
        assert!(m.coefficients()[0][1].abs() <= 0.05 + 1e-12);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.min_periods = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 31u64;
        for i in 0..50 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.coefficients()[0].clone();
        let st = m.step(&[0.5], &[None], 1.0, 1.0);
        assert!(st.pred[0].is_finite());
        assert_eq!(m.coefficients()[0], before);
    }

    #[test]
    fn state_roundtrip() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.schedule = LearningRate::AdaGrad;
        let mut m1 = Sgd::new(c).unwrap();
        let mut s = 37u64;
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
        let mut m2 = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.learning_rate = 0.0;
        assert!(Sgd::new(c).is_err());
        assert!(Sgd::new(cfg(1, SgdLoss::Quantile { tau: 0.0 })).is_err());
        assert!(Sgd::new(cfg(1, SgdLoss::Huber { delta: -1.0 })).is_err());
    }
}
