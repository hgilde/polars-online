//! Passive-aggressive regression (docs/ENHANCEMENTS.md E17).
//!
//! Crammer et al. (2006). Each row poses a constraint — "get within `eps` of
//! this target" — and the update makes the *smallest* change to the
//! coefficients that satisfies it. Passive when the constraint already holds,
//! aggressive when it does not; there is no learning rate to tune.
//!
//! With `p = z·b`, `loss = max(0, |y − p| − eps)` and `s = ||z||²`:
//!
//! ```text
//! PA    tau = loss / s                    (unbounded step)
//! PA-I  tau = min(C, loss / s)            (step capped at C)
//! PA-II tau = loss / (s + 1 / (2C))       (step damped by C)
//! b    += tau * sign(y − p) * z
//! ```
//!
//! **Decay note.** Unlike every other model here, PA keeps no accumulators, so
//! there is nothing for the clock to decay: each step fully satisfies the
//! current row's constraint and older rows survive only through the
//! coefficients they left behind. `n_eff` is still decayed on the clock so
//! `min_periods` means the same thing as elsewhere, but the coefficients
//! themselves have no half-life. Use PA-I/PA-II (a finite `c`) when that
//! aggressiveness is a problem: an outlier otherwise moves the fit as far as it
//! takes to satisfy the outlier.

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// Which passive-aggressive variant (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaMode {
    /// Unbounded step: satisfies the constraint exactly.
    Pa,
    /// Step capped at `c`.
    #[default]
    Pa1,
    /// Step damped by `c`.
    Pa2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    pub mode: PaMode,
    /// Aggressiveness. Ignored by [`PaMode::Pa`].
    pub c: f64,
    /// Width of the insensitive tube: rows already this close are passive.
    pub eps: f64,
    pub min_periods: f64,
}

impl PaCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("pa: n_features and n_targets must be >= 1".into());
        }
        if self.c <= 0.0 || self.c.is_nan() {
            return Err("pa: c must be > 0".into());
        }
        if self.eps < 0.0 || self.eps.is_nan() {
            return Err("pa: eps must be >= 0".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pa {
    cfg: PaCfg,
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    #[serde(skip)]
    zbuf: Vec<f64>,
}

impl Pa {
    pub fn new(cfg: PaCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        Ok(Self {
            beta: vec![vec![0.0; k]; cfg.n_targets],
            w_sum: 0.0,
            zbuf: vec![0.0; k],
            cfg,
        })
    }

    pub fn cfg(&self) -> &PaCfg {
        &self.cfg
    }

    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.beta
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    fn ensure_buffers(&mut self) {
        if self.zbuf.len() != self.cfg.k_total() {
            self.zbuf = vec![0.0; self.cfg.k_total()];
        }
    }
}

impl OnlineModel for Pa {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // Before this row's update and before its decay -- the convention
        // every model reports and gates on.
        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_periods;
        let sq_norm: f64 = self.zbuf.iter().map(|z| z * z).sum();

        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            let p: f64 = self
                .zbuf
                .iter()
                .zip(&self.beta[j])
                .map(|(z, b)| z * b)
                .sum();
            if ready {
                pred[j] = p;
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 || !yj.is_finite() || sq_norm <= 0.0 {
                continue;
            }
            let err = yj - p;
            let loss = (err.abs() - self.cfg.eps).max(0.0);
            if loss == 0.0 {
                continue; // passive: the constraint already holds
            }
            // The row weight scales the aggressiveness, so a half-weight row
            // moves the fit half as far.
            let tau = weight
                * match self.cfg.mode {
                    PaMode::Pa => loss / sq_norm,
                    PaMode::Pa1 => (loss / sq_norm).min(self.cfg.c),
                    PaMode::Pa2 => loss / (sq_norm + 0.5 / self.cfg.c),
                };
            let step = tau * err.signum();
            for (b, z) in self.beta[j].iter_mut().zip(&self.zbuf) {
                *b += step * z;
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
        State::new(ModelState::Pa(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Pa(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "pa",
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

    fn cfg(k: usize, mode: PaMode) -> PaCfg {
        PaCfg {
            n_features: k,
            n_targets: 1,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            mode,
            c: 1.0,
            eps: 0.01,
            min_periods: 5.0,
        }
    }

    fn fit(
        cfg: PaCfg,
        n: usize,
        seed: u64,
        mut f: impl FnMut(&[f64], &mut u64) -> f64,
    ) -> Vec<f64> {
        let k = cfg.n_features;
        let mut m = Pa::new(cfg).unwrap();
        let mut s = seed;
        for i in 0..n {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y = f(&x, &mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    }

    #[test]
    fn recovers_a_noiseless_relationship() {
        for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
            let b = fit(cfg(2, mode), 5000, 1, |x, _| 1.5 * x[0] - 0.5 * x[1] + 0.25);
            assert!((b[0] - 0.25).abs() < 0.05, "{mode:?} intercept {}", b[0]);
            assert!((b[1] - 1.5).abs() < 0.05, "{mode:?} slope0 {}", b[1]);
            assert!((b[2] + 0.5).abs() < 0.05, "{mode:?} slope1 {}", b[2]);
        }
    }

    #[test]
    fn passive_inside_the_tube() {
        // With a wide tube and a target already inside it, nothing moves.
        let mut c = cfg(1, PaMode::Pa1);
        c.eps = 10.0;
        c.min_periods = 0.0;
        let mut m = Pa::new(c).unwrap();
        for _ in 0..100 {
            m.step(&[1.0], &[Some(0.5)], 1.0, 1.0);
        }
        assert_eq!(m.coefficients()[0], vec![0.0, 0.0]);
    }

    #[test]
    fn plain_pa_satisfies_the_constraint_exactly() {
        // One aggressive step must land the prediction on the tube edge.
        let mut c = cfg(1, PaMode::Pa);
        c.eps = 0.0;
        c.min_periods = 0.0;
        let mut m = Pa::new(c).unwrap();
        m.step(&[2.0], &[Some(7.0)], 0.0, 1.0);
        let b = &m.coefficients()[0];
        let p = b[0] + 2.0 * b[1];
        assert!(
            (p - 7.0).abs() < 1e-12,
            "pa did not satisfy the constraint: {p}"
        );
    }

    #[test]
    fn c_caps_the_aggressiveness() {
        // PA-1 with a small c must move far less than plain PA on the same row.
        let step = |mode, c| {
            let mut cf = cfg(1, mode);
            cf.c = c;
            cf.eps = 0.0;
            cf.min_periods = 0.0;
            let mut m = Pa::new(cf).unwrap();
            m.step(&[1.0], &[Some(100.0)], 0.0, 1.0);
            m.coefficients()[0][1]
        };
        let unbounded = step(PaMode::Pa, 1.0);
        let capped = step(PaMode::Pa1, 0.1);
        let damped = step(PaMode::Pa2, 0.1);
        assert!(capped < unbounded, "PA-1 {capped} !< PA {unbounded}");
        assert!(damped < unbounded, "PA-2 {damped} !< PA {unbounded}");
    }

    #[test]
    fn bounded_variants_resist_an_outlier() {
        let mut row = 0u64;
        let contaminated = move |x: &[f64], s: &mut u64| {
            row += 1;
            if row % 25 == 0 {
                500.0 * lcg(s)
            } else {
                2.0 * x[0]
            }
        };
        let mut c1 = cfg(1, PaMode::Pa1);
        c1.c = 0.05;
        let capped = fit(c1, 5000, 3, contaminated);
        let mut row2 = 0u64;
        let unbounded = fit(
            cfg(1, PaMode::Pa),
            5000,
            3,
            move |x: &[f64], s: &mut u64| {
                row2 += 1;
                if row2 % 25 == 0 {
                    500.0 * lcg(s)
                } else {
                    2.0 * x[0]
                }
            },
        );
        assert!(
            (capped[1] - 2.0).abs() < (unbounded[1] - 2.0).abs(),
            "PA-1 {} should beat PA {} under contamination (truth 2.0)",
            capped[1],
            unbounded[1]
        );
    }

    #[test]
    fn row_weight_scales_the_step() {
        let step_for = |w: f64| {
            let mut c = cfg(1, PaMode::Pa);
            c.eps = 0.0;
            c.min_periods = 0.0;
            let mut m = Pa::new(c).unwrap();
            m.step(&[1.0], &[Some(4.0)], 0.0, w);
            m.coefficients()[0][1]
        };
        assert!((step_for(0.5) - 0.5 * step_for(1.0)).abs() < 1e-12);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut c = cfg(1, PaMode::Pa1);
        c.min_periods = 0.0;
        let mut m = Pa::new(c).unwrap();
        let mut s = 7u64;
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
        let mut m1 = Pa::new(cfg(2, PaMode::Pa2)).unwrap();
        let mut s = 11u64;
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
        let mut m2 = Pa::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut c = cfg(1, PaMode::Pa1);
        c.c = 0.0;
        assert!(Pa::new(c).is_err());
        let mut c = cfg(1, PaMode::Pa1);
        c.eps = -1.0;
        assert!(Pa::new(c).is_err());
    }
}
