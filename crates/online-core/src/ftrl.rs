//! Online logistic regression via FTRL-proximal (docs/PLAN.md §4.6).
//!
//! For binary targets (direction, "signal accurate now"). Per-coordinate
//! adaptive learning rates, following McMahan et al. (2013), with the
//! accumulators decayed by the same clock as every other model here so it
//! forgets on the same schedule.
//!
//! Per row (`z` includes the intercept when configured, `p` the predicted
//! probability, `g_i = (p - y) * z_i * w` the gradient):
//!
//! ```text
//! decay:   n_i <- lam * n_i ;  zz_i <- lam * zz_i     (lam from the clock)
//! predict: b_i = 0 if |zz_i| <= l1
//!              = -(zz_i - sign(zz_i) l1) / ((beta + sqrt(n_i)) / alpha + l2)
//!          p   = sigmoid(z . b)
//! update:  s_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
//!          zz_i += g_i - s_i b_i
//!          n_i  += g_i^2
//! ```
//!
//! `pred` is the probability computed from the state *before* the update, so it
//! is out-of-sample like every other model; `resid = y - p`.
//!
//! Defaults `alpha = 0.1`, `beta = 1.0`, `l1 = 0.0`, `l2 = 1.0` follow the paper's
//! guidance (docs/PLAN.md marks them [validate]).

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FtrlCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Learning-rate scale.
    pub alpha: f64,
    /// Learning-rate smoothing.
    pub beta: f64,
    pub l1: f64,
    pub l2: f64,
    pub min_periods: f64,
    /// Reject targets that are not 0/1 instead of clamping them.
    pub strict_binary: bool,
}

impl FtrlCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.alpha <= 0.0 {
            return Err("ftrl: alpha must be > 0".into());
        }
        if self.beta < 0.0 || self.l1 < 0.0 || self.l2 < 0.0 {
            return Err("ftrl: beta, l1 and l2 must be >= 0".into());
        }
        Ok(())
    }
}

#[inline]
fn sigmoid(v: f64) -> f64 {
    // Numerically stable both ways.
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ftrl {
    cfg: FtrlCfg,
    /// Per target: squared-gradient accumulators and the FTRL `z` state.
    n: Vec<Vec<f64>>,
    zz: Vec<Vec<f64>>,
    w_sum: f64,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    coef: Vec<f64>,
}

impl Ftrl {
    pub fn new(cfg: FtrlCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        Ok(Self {
            n: vec![vec![0.0; k]; m],
            zz: vec![vec![0.0; k]; m],
            w_sum: 0.0,
            zbuf: vec![0.0; k],
            coef: vec![0.0; k],
            cfg,
        })
    }

    pub fn cfg(&self) -> &FtrlCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Proximal weights implied by the current FTRL state, per target.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| (0..self.cfg.k_total()).map(|i| self.weight(j, i)).collect())
            .collect()
    }

    #[inline]
    fn weight(&self, j: usize, i: usize) -> f64 {
        let zz = self.zz[j][i];
        if zz.abs() <= self.cfg.l1 {
            0.0
        } else {
            let sgn = if zz < 0.0 { -1.0 } else { 1.0 };
            -(zz - sgn * self.cfg.l1)
                / ((self.cfg.beta + self.n[j][i].sqrt()) / self.cfg.alpha + self.cfg.l2)
        }
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.coef = vec![0.0; k];
        }
    }
}

impl OnlineModel for Ftrl {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // Decay first: the accumulators forget on the model's clock.
        if lam != 1.0 {
            for j in 0..m {
                for i in 0..k {
                    self.n[j][i] *= lam;
                    self.zz[j][i] *= lam;
                }
            }
        }
        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_periods;

        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            // Proximal weights from the state before this row's update.
            for i in 0..k {
                self.coef[i] = self.weight(j, i);
            }
            let logit: f64 = self.zbuf.iter().zip(&self.coef).map(|(z, b)| z * b).sum();
            let p = sigmoid(logit);
            if ready {
                pred[j] = p;
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 {
                continue;
            }
            let yb = if self.cfg.strict_binary {
                if yj != 0.0 && yj != 1.0 {
                    continue; // caller asked for strictness: skip, do not learn
                }
                yj
            } else {
                yj.clamp(0.0, 1.0)
            };
            let err = p - yb;
            for i in 0..k {
                let g = err * self.zbuf[i] * weight;
                let n_new = self.n[j][i] + g * g;
                let s = (n_new.sqrt() - self.n[j][i].sqrt()) / self.cfg.alpha;
                self.zz[j][i] += g - s * self.coef[i];
                self.n[j][i] = n_new;
            }
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
        State::new(ModelState::Ftrl(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Ftrl(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ftrl",
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

    fn cfg(k: usize, m: usize) -> FtrlCfg {
        FtrlCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 1.0,
            min_periods: 10.0,
            strict_binary: false,
        }
    }

    #[test]
    fn learns_a_separable_rule() {
        // y = 1 when x0 > 0. After training, predictions must be on the right
        // side of 0.5 for clear cases.
        let mut m = Ftrl::new(cfg(1, 1)).unwrap();
        let mut s = 91u64;
        for i in 0..5000 {
            let x = [lcg(&mut s)];
            let y = f64::from(x[0] > 0.0);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let pos = m.step(&[0.9], &[None], 1.0, 1.0).pred[0];
        let neg = m.step(&[-0.9], &[None], 1.0, 1.0).pred[0];
        assert!(pos > 0.6, "p(x=0.9) = {pos}");
        assert!(neg < 0.4, "p(x=-0.9) = {neg}");
    }

    #[test]
    fn predictions_are_probabilities() {
        let mut m = Ftrl::new(cfg(2, 1)).unwrap();
        let mut s = 92u64;
        for i in 0..500 {
            let x = [lcg(&mut s) * 100.0, lcg(&mut s) * 100.0];
            let y = f64::from(x[0] + x[1] > 0.0);
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if st.pred[0].is_finite() {
                assert!((0.0..=1.0).contains(&st.pred[0]), "{}", st.pred[0]);
            }
        }
    }

    #[test]
    fn base_rate_is_learned_by_the_intercept() {
        // Features carry no information; p must converge toward the base rate.
        let mut m = Ftrl::new(cfg(1, 1)).unwrap();
        let mut s = 93u64;
        let mut last = 0.0;
        for i in 0..20000 {
            let x = [lcg(&mut s)];
            let y = f64::from(lcg(&mut s) < 0.4); // base rate 0.7
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if st.pred[0].is_finite() {
                last = st.pred[0];
            }
        }
        assert!((last - 0.7).abs() < 0.15, "converged to {last}, want ~0.7");
    }

    #[test]
    fn l1_shrinks_noise_features_and_can_zero_everything() {
        // FTRL's L1 zeroes a coordinate only while |z_i| <= l1, and z_i grows
        // with the accumulated gradient, so a moderate penalty shrinks noise
        // features rather than pinning them at exactly zero forever.
        let run = |l1: f64| {
            let mut c = cfg(3, 1);
            c.l1 = l1;
            let mut m = Ftrl::new(c).unwrap();
            let mut s = 94u64;
            for i in 0..3000 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                let y = f64::from(x[0] > 0.0); // only x0 matters
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients()[0].clone()
        };
        let plain = run(0.0);
        let penalized = run(2.0);
        assert!(
            penalized[1].abs() > 5.0 * penalized[2].abs().max(penalized[3].abs()),
            "signal {} should dominate noise {:?}",
            penalized[1],
            &penalized[2..]
        );
        for i in 2..4 {
            assert!(
                penalized[i].abs() < plain[i].abs(),
                "L1 must shrink noise feature {i}"
            );
        }
        // A large enough penalty zeroes every coordinate exactly.
        assert_eq!(run(1e9), vec![0.0; 4]);
    }

    #[test]
    fn forgets_on_the_clock() {
        // A regime flip: with a short halflife the model must follow it.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(200.0);
        let mut m = Ftrl::new(c).unwrap();
        let mut s = 95u64;
        for i in 0..3000 {
            let x = [lcg(&mut s)];
            let y = f64::from(x[0] > 0.0);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // flip the rule
        for _ in 0..3000 {
            let x = [lcg(&mut s)];
            let y = f64::from(x[0] < 0.0);
            m.step(&x, &[Some(y)], 1.0, 1.0);
        }
        let p = m.step(&[0.9], &[None], 1.0, 1.0).pred[0];
        assert!(p < 0.4, "after the flip p(x=0.9) should be low, got {p}");
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Ftrl::new(cfg(2, 1)).unwrap();
        let mut s = 96u64;
        let rows: Vec<([f64; 2], f64)> = (0..200)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = f64::from(x[0] + x[1] > 0.0);
                (x, y)
            })
            .collect();
        for (i, (x, y)) in rows[..100].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Ftrl::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[100..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }
}
