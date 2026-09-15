//! Online regression via FTRL-proximal (docs/PLAN.md §4.6).
//!
//! Per-coordinate adaptive learning rates following McMahan et al. (2013),
//! with the sums decayed on the model's clock -- which, for a closed form in
//! sums, is not the forgetting every other model here does (below).
//!
//! Two losses, which differ only in the link and the gradient:
//!
//! - [`FtrlLoss::Logistic`] for binary targets (direction, "signal accurate
//!   now"): `p = sigmoid(z·b)` and `g = (p − y)·z`. `pred` is a probability.
//! - [`FtrlLoss::Squared`] for continuous targets: `p = z·b` and the same
//!   `g = (p − y)·z`. This is the sparse linear regression river gets from
//!   `optim.FTRLProximal` with a squared loss — cheap (no solves) and L1-capable
//!   where `ew_ridge` is not.
//!
//! Per row (`z` includes the intercept when configured, `p` the predicted
//! probability, `g_i = (p - y) * z_i * w` the gradient):
//!
//! ```text
//! decay:   n_i <- lam·n_i ;  zz_i <- lam·zz_i ;  d_i <- lam·d_i   (lam from the clock)
//! predict: b_i = 0 if |zz_i| <= l1
//!              = -(zz_i - sign(zz_i) l1) / (beta/alpha + d_i + l2)          under a halflife
//!              = -(zz_i - sign(zz_i) l1) / ((beta + sqrt(n_i))/alpha + l2)  without one
//!          p   = sigmoid(z . b)
//! update:  s_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
//!          zz_i += g_i - s_i b_i ;  n_i += g_i^2 ;  d_i += s_i
//! ```
//!
//! **What a halflife does here** (review 2026-09-12, C24). The proximal
//! weight is a closed form in sums, and the sums decay. Without decay `d`
//! telescopes to `sqrt(n)/alpha`, and the model is river's `FTRLProximal`,
//! computed as river computes it (T-R1). Under a halflife `d` is the
//! discounted sum of the steps themselves; decaying `n` inside the square
//! root instead shrank every coefficient toward zero on every row, and a
//! constant target of 5 settled at 2.25 at `halflife = 100`. What remains is
//! the penalties: `beta`, `l1` and `l2` are constants on the sums' scale, so
//! under a halflife they act as a mean-scale ridge of `(1 − lam)·(beta/alpha
//! + l2)` -- the constant 5 settles at `5/(1 + (1 − lam)(beta/alpha + l2))`,
//! 4.65 at `halflife = 100` and 4.96 at 1000 -- and a gap still costs: a row
//! with no target and a clock of `t` scales every coefficient by
//! `lam^t·(d + c)/(lam^t·d + c)`, `c = beta/alpha + l2`, about 0.75 over one
//! halflife at 100, where a mean-form model like `ewridge` does not move.
//! Constant penalties are what keep the undecayed model river's; for
//! forgetting without the shrinkage, reset a `halflife = inf` model on a
//! `session`, or use `sgd` or `ewridge`.
//!
//! `pred` is the probability computed from the state *before* the update, so it
//! is out-of-sample like every other model; `resid = y - p`.
//!
//! Defaults `alpha = 0.1`, `beta = 1.0`, `l1 = 0.0`, `l2 = 1.0` follow the paper's
//! guidance (docs/PLAN.md marks them [validate]).

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// Which loss the FTRL updates follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FtrlLoss {
    /// Binary targets; `pred` is a probability in [0, 1].
    #[default]
    Logistic,
    /// Continuous targets; `pred` is the linear prediction.
    Squared,
}

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
    /// A target that is not 0 or 1 is not learned from, where the default
    /// clamps it into [0, 1]. Logistic only. The bank never hands the model
    /// such a row: it refuses the chunk, naming the row (review 2026-09-12,
    /// S31).
    pub strict_binary: bool,
    /// Logistic (default) or squared loss.
    #[serde(default)]
    pub loss: FtrlLoss,
}

impl FtrlCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        // NaN compares false both ways, so each check says what it accepts;
        // all four took NaN, and every coefficient was NaN from the first
        // row (review 2026-09-12, S32).
        if self.alpha <= 0.0 || self.alpha.is_nan() {
            return Err("ftrl: alpha must be > 0".into());
        }
        if [self.beta, self.l1, self.l2]
            .iter()
            .any(|&v| v < 0.0 || v.is_nan())
        {
            return Err("ftrl: beta, l1 and l2 must be >= 0".into());
        }
        if self.strict_binary && self.loss == FtrlLoss::Squared {
            return Err("ftrl: strict_binary applies to the logistic loss only".into());
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
    /// Per target: the discounted sum of the proximal steps `s_i`, the rate's
    /// term under a halflife, where `sqrt(n)/alpha` is without one (C24).
    #[serde(default)]
    prox: Vec<Vec<f64>>,
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
            prox: vec![vec![0.0; k]; m],
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
        self.weight_of(self.zz[j][i], self.n[j][i], self.prox[j][i])
    }

    /// Whether the sums decay at all. Without a halflife the rate is river's
    /// `(beta + sqrt(n))/alpha`, computed as river computes it.
    fn forgets(&self) -> bool {
        match self.cfg.decay {
            Decay::Halflife(h) => h.is_finite(),
            Decay::Lam(l) => l != 1.0,
        }
    }

    /// The proximal weight for one coordinate's `(z, n, d)` -- the closed
    /// form of the FTRL-Proximal update, shared by `step` and `predict`.
    /// Under a halflife the rate's term is `d`, the discounted sum of the
    /// proximal steps; without one it is `sqrt(n)/alpha`, which `d`
    /// telescopes to (review 2026-09-12, C24; the module docs).
    fn weight_of(&self, zz: f64, n: f64, prox: f64) -> f64 {
        if zz.abs() <= self.cfg.l1 {
            0.0
        } else {
            let sgn = if zz < 0.0 { -1.0 } else { 1.0 };
            let rate = if self.forgets() {
                self.cfg.beta / self.cfg.alpha + prox
            } else {
                (self.cfg.beta + n.sqrt()) / self.cfg.alpha
            };
            -(zz - sgn * self.cfg.l1) / (rate + self.cfg.l2)
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
                    self.prox[j][i] *= lam;
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
            let raw: f64 = self.zbuf.iter().zip(&self.coef).map(|(z, b)| z * b).sum();
            // The two losses share everything but the link; the gradient is
            // `(p - y) * z` either way.
            let p = match self.cfg.loss {
                FtrlLoss::Logistic => sigmoid(raw),
                FtrlLoss::Squared => raw,
            };
            if ready {
                pred[j] = p;
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 {
                continue;
            }
            let yb = match self.cfg.loss {
                FtrlLoss::Squared => yj,
                FtrlLoss::Logistic if self.cfg.strict_binary => {
                    if yj != 0.0 && yj != 1.0 {
                        continue; // caller asked for strictness: skip, do not learn
                    }
                    yj
                }
                FtrlLoss::Logistic => yj.clamp(0.0, 1.0),
            };
            let err = p - yb;
            // `n_i += g^2` never decays an `inf` away, so a row whose squared
            // gradient would overflow (a feature at the input bound with a
            // comparable weight or, under the squared loss, a comparable
            // error) is skipped rather than learned from
            // (docs/IMPROVEMENTS.md C2).
            let g_max = err.abs() * weight * self.zbuf.iter().fold(0.0_f64, |m, z| m.max(z.abs()));
            if !(g_max * g_max).is_finite() {
                continue;
            }
            for i in 0..k {
                let g = err * self.zbuf[i] * weight;
                let n_new = self.n[j][i] + g * g;
                let s = (n_new.sqrt() - self.n[j][i].sqrt()) / self.cfg.alpha;
                self.zz[j][i] += g - s * self.coef[i];
                self.n[j][i] = n_new;
                self.prox[j][i] += s;
            }
        }
        self.w_sum = lam * self.w_sum + weight;

        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        let lam = self.cfg.decay.factor(d_clock);
        let n_eff = self.w_sum;
        let m = self.cfg.n_targets;
        let mut pred = vec![f64::NAN; m];
        if n_eff >= self.cfg.min_periods {
            let k = self.cfg.k_total();
            let off = usize::from(self.cfg.add_intercept);
            for (j, p) in pred.iter_mut().enumerate() {
                // The proximal weights `step` would derive after decaying the
                // accumulators by this row's clock, computed without storing
                // the decay.
                let raw: f64 = (0..k)
                    .map(|i| {
                        let z = if i < off { 1.0 } else { x[i - off] };
                        z * self.weight_of(
                            self.zz[j][i] * lam,
                            self.n[j][i] * lam,
                            self.prox[j][i] * lam,
                        )
                    })
                    .sum();
                *p = match self.cfg.loss {
                    FtrlLoss::Logistic => sigmoid(raw),
                    FtrlLoss::Squared => raw,
                };
            }
        }
        Step {
            pred,
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

    #[test]
    fn sigmoid_is_stable_at_both_extremes() {
        // The two branches are algebraically identical, so only the extremes
        // distinguish them: taking the wrong one at a large magnitude gives
        // inf/inf = NaN rather than saturating.
        for v in [800.0, 80.0, 1.0, 0.0, -1.0, -80.0, -800.0] {
            let p = sigmoid(v);
            assert!(p.is_finite(), "sigmoid({v}) = {p}");
            assert!((0.0..=1.0).contains(&p), "sigmoid({v}) = {p}");
        }
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(800.0), 1.0);
        assert_eq!(sigmoid(-800.0), 0.0);
        // Symmetric about zero.
        for v in [0.3, 2.0, 17.0] {
            assert!((sigmoid(v) + sigmoid(-v) - 1.0).abs() < 1e-15, "{v}");
        }
    }

    /// The model alone skips such a row. The bank never hands it one: it
    /// refuses a chunk whose `strict_binary` target is not 0 or 1, naming the
    /// row, before any stream sees it (review 2026-09-12, S31).
    #[test]
    fn strict_binary_skips_a_non_binary_target_instead_of_clamping_it() {
        // Two policies for a target outside {0, 1}: clamp it (the default) or
        // refuse to learn from it. The difference is only visible in the state
        // afterwards, since both still predict.
        let fit = |strict: bool, y: f64| {
            let mut c = cfg(2, 1);
            c.strict_binary = strict;
            c.min_periods = 0.0;
            let mut m = Ftrl::new(c).unwrap();
            for i in 0..20 {
                m.step(
                    &[1.0, -1.0],
                    &[Some(1.0)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            let before = m.zz[0].clone();
            m.step(&[1.0, -1.0], &[Some(y)], 1.0, 1.0);
            (before, m.zz[0].clone())
        };

        let (before, after) = fit(true, 0.7);
        assert_eq!(before, after, "strict_binary must not learn from y = 0.7");
        let (before, after) = fit(false, 0.7);
        assert_ne!(before, after, "the default clamps and learns");

        // Values inside {0, 1} are learned from under either policy.
        let (before, after) = fit(true, 0.0);
        assert_ne!(before, after, "y = 0 is binary and must be learned from");

        // Out-of-range values are clamped rather than extrapolated.
        let (_, clamped) = fit(false, 5.0);
        let (_, at_one) = fit(false, 1.0);
        assert_eq!(clamped, at_one, "y = 5 must behave exactly like y = 1");
    }

    /// `g = err * z * weight`: the weight multiplies the row's loss, so its
    /// gradient scales by `w` and its squared gradient, which sets the
    /// coordinate's learning rate, by `w²` -- river's and sklearn's
    /// convention. A row at weight 4 is therefore *not* four rows at weight
    /// 1, as a weight is for the accumulators elsewhere here: it moves `zz`
    /// four times as far only on the first row, where the coefficient is
    /// still 0, and slows its coordinate's rate more than four light rows
    /// would (review 2026-09-12, D10).
    #[test]
    fn the_row_weight_scales_the_gradient() {
        let run = |w: f64, reps: usize| {
            let mut c = cfg(1, 1);
            c.loss = FtrlLoss::Squared;
            c.min_periods = 0.0;
            c.l1 = 0.0;
            c.l2 = 0.0;
            let mut m = Ftrl::new(c).unwrap();
            for _ in 0..reps {
                m.step(&[1.0], &[Some(1.0)], 0.0, w);
            }
            m.zz[0][0]
        };
        // Zero weight is a no-op.
        assert_eq!(run(0.0, 5), 0.0);
        // A heavier row moves further than a lighter one, in the same direction.
        let (light, heavy) = (run(1.0, 1), run(4.0, 1));
        assert!(heavy.abs() > light.abs(), "{heavy} vs {light}");
        assert_eq!(heavy.signum(), light.signum());
        // And exactly four times as far on the first row, where the state is
        // still zero so the gradient is linear in the weight.
        assert!((heavy - 4.0 * light).abs() < 1e-12, "{heavy} vs {light}");
    }

    #[test]
    fn shape_accessors_report_the_configured_shape() {
        let m = Ftrl::new(cfg(3, 2)).unwrap();
        assert_eq!(OnlineModel::n_features(&m), 3);
        assert_eq!(OnlineModel::n_targets(&m), 2);
        assert_eq!(m.cfg().k_total(), 4, "3 features plus an intercept");
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        // One case per rejection in `FtrlCfg::validate`, matched on the message,
        // each paired with the nearest accepted config so a validator that
        // refuses everything fails too.
        let bad = |f: &dyn Fn(&mut FtrlCfg), want: &str| {
            let mut c = cfg(2, 1);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        let good = |f: &dyn Fn(&mut FtrlCfg)| {
            let mut c = cfg(2, 1);
            f(&mut c);
            c.validate().expect("should be accepted");
        };

        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");

        // alpha divides the learning rate, so zero is as fatal as negative.
        bad(&|c| c.alpha = 0.0, "alpha must be > 0");
        bad(&|c| c.alpha = -1.0, "alpha must be > 0");
        good(&|c| c.alpha = 1e-9);

        // beta/l1/l2 may be zero -- only negative is meaningless.
        bad(&|c| c.beta = -1e-9, "must be >= 0");
        bad(&|c| c.l1 = -1e-9, "must be >= 0");
        bad(&|c| c.l2 = -1e-9, "must be >= 0");
        good(&|c| {
            c.beta = 0.0;
            c.l1 = 0.0;
            c.l2 = 0.0;
        });

        // NaN compares false both ways, so each check must say what it
        // accepts (review 2026-09-12, S32: all four took NaN, and every
        // coefficient was NaN from the first row).
        bad(&|c| c.alpha = f64::NAN, "alpha must be > 0");
        bad(&|c| c.beta = f64::NAN, "must be >= 0");
        bad(&|c| c.l1 = f64::NAN, "must be >= 0");
        bad(&|c| c.l2 = f64::NAN, "must be >= 0");

        // strict_binary checks that y is 0/1, which the squared loss does not require.
        bad(
            &|c| {
                c.strict_binary = true;
                c.loss = FtrlLoss::Squared;
            },
            "logistic loss only",
        );
        good(&|c| c.strict_binary = true);
        good(&|c| c.loss = FtrlLoss::Squared);

        cfg(2, 1).validate().expect("the baseline config is valid");
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
            loss: FtrlLoss::Logistic,
        }
    }

    #[test]
    fn squared_loss_fits_a_continuous_target() {
        let mut c = cfg(2, 1);
        c.loss = FtrlLoss::Squared;
        c.alpha = 0.5;
        c.l2 = 0.01;
        c.min_periods = 5.0;
        let mut m = Ftrl::new(c).unwrap();
        let mut s = 97u64;
        let mut last = 0.0;
        for i in 0..20000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.5 * x[1];
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i > 19000 && st.pred[0].is_finite() {
                last = (y - st.pred[0]).abs();
            }
        }
        assert!(
            last < 0.15,
            "squared-loss FTRL did not converge: |err| {last}"
        );
        let b = &m.coefficients()[0];
        assert!((b[1] - 1.5).abs() < 0.2, "slope0 {}", b[1]);
        assert!((b[2] + 0.5).abs() < 0.2, "slope1 {}", b[2]);
    }

    #[test]
    fn squared_loss_predictions_are_not_probabilities() {
        // The logistic link would squash these into [0, 1]; the squared loss
        // must not.
        let mut c = cfg(1, 1);
        c.loss = FtrlLoss::Squared;
        c.alpha = 0.5;
        c.l2 = 0.01;
        c.min_periods = 2.0;
        let mut m = Ftrl::new(c).unwrap();
        let mut s = 98u64;
        let mut seen_big = false;
        for i in 0..5000 {
            let x = [lcg(&mut s)];
            let y = 20.0 * x[0];
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if st.pred[0] > 1.5 {
                seen_big = true;
            }
        }
        assert!(
            seen_big,
            "squared-loss predictions were squashed into [0, 1]"
        );
    }

    #[test]
    fn strict_binary_is_rejected_for_the_squared_loss() {
        let mut c = cfg(1, 1);
        c.loss = FtrlLoss::Squared;
        c.strict_binary = true;
        assert!(Ftrl::new(c).is_err());
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

    /// The recursion under decay written out longhand, for the squared loss
    /// on an intercept alone: `n` and `zz` are decayed sums, and so is the
    /// proximal term, `d = Σ λ^age·σ_s`, which `√n/α` telescopes to without
    /// decay and does not with it (review 2026-09-12, C24). Each row's
    /// prediction, and the last state `(zz, n, d)`.
    fn decayed_longhand(c: &FtrlCfg, y: f64, rows: usize) -> (Vec<f64>, f64, f64, f64) {
        let lam = c.decay.factor(1.0);
        let (mut zz, mut n, mut d) = (0.0f64, 0.0f64, 0.0f64);
        let mut preds = Vec::with_capacity(rows);
        for i in 0..rows {
            if i > 0 {
                zz *= lam;
                n *= lam;
                d *= lam;
            }
            let b = if zz.abs() <= c.l1 {
                0.0
            } else {
                -(zz - zz.signum() * c.l1) / (c.beta / c.alpha + d + c.l2)
            };
            preds.push(b);
            let g = b - y;
            let s = ((n + g * g).sqrt() - n.sqrt()) / c.alpha;
            zz += g - s * b;
            n += g * g;
            d += s;
        }
        (preds, zz, n, d)
    }

    fn intercept_only(decay: Decay) -> FtrlCfg {
        let mut c = cfg(1, 1);
        c.loss = FtrlLoss::Squared;
        c.decay = decay;
        c.min_periods = 0.0;
        c
    }

    /// A constant target, with the one feature at 0 so only the intercept
    /// learns.
    fn fit_constant(c: &FtrlCfg, y: f64, rows: usize) -> (Vec<f64>, Ftrl) {
        let mut m = Ftrl::new(c.clone()).unwrap();
        let preds = (0..rows)
            .map(|i| {
                let d = if i == 0 { 0.0 } else { 1.0 };
                m.step(&[0.0], &[Some(y)], d, 1.0).pred[0]
            })
            .collect();
        (preds, m)
    }

    /// Under a halflife the proximal term is a decayed sum of its own. The
    /// fit is the longhand's on every row, and a constant target of 5 settles
    /// where sum-scale penalties put it, `5 / (1 + (1 − λ)(β/α + l2))` -- a
    /// mean-scale ridge of `(1 − λ)(β/α + l2)` -- 4.65 at `halflife = 100`.
    /// Decaying `n` inside `√n` had put it at 2.25 (review 2026-09-12, C24).
    #[test]
    fn a_decaying_state_keeps_its_proximal_sum() {
        let c = intercept_only(Decay::Halflife(100.0));
        let (preds, _) = fit_constant(&c, 5.0, 20_000);
        let (want, ..) = decayed_longhand(&c, 5.0, 20_000);
        for (i, (p, w)) in preds.iter().zip(&want).enumerate() {
            assert!(
                (p - w).abs() <= 1e-12 * w.abs().max(1.0),
                "row {i}: {p} vs {w}"
            );
        }
        let lam = c.decay.factor(1.0);
        let settled = 5.0 / (1.0 + (1.0 - lam) * (c.beta / c.alpha + c.l2));
        let last = preds[preds.len() - 1];
        assert!(
            (last - settled).abs() < 0.005 * settled,
            "{last} vs {settled}"
        );
    }

    /// What a gap still costs: the penalties are constants and the sums
    /// decay, so a row with no target and a clock of `d` scales every
    /// coefficient by `λ^d·(D + c)/(λ^d·D + c)`, `c = β/α + l2`, `D` the
    /// proximal sum -- 0.75 over 100 clock units at `halflife = 100`, where
    /// it was 0.70. The review asked for 1, which needs the penalties scaled
    /// by the weight, and that is not river's at `inf` (C24).
    #[test]
    fn a_gap_shrinks_the_coefficients_by_the_stated_factor() {
        let c = intercept_only(Decay::Halflife(100.0));
        let (_, mut m) = fit_constant(&c, 5.0, 5_000);
        let (_, _, _, d) = decayed_longhand(&c, 5.0, 5_000);
        let before = m.coefficients()[0][0];
        m.step(&[0.0], &[None], 100.0, 1.0);
        let after = m.coefficients()[0][0];
        let f = c.decay.factor(100.0);
        let k = c.beta / c.alpha + c.l2;
        let want = f * (d + k) / (f * d + k);
        assert!(
            (after / before - want).abs() < 1e-12,
            "{} vs {want}",
            after / before
        );
    }

    /// Without decay the proximal term telescopes to `√n/α`, and the weight
    /// is river's closed form computed as river computes it, so T-R1 holds
    /// to the bit and the repair of C24 changes nothing at `halflife = inf`.
    #[test]
    fn without_decay_the_weights_are_rivers_closed_form() {
        let c = intercept_only(Decay::Halflife(f64::INFINITY));
        let (_, m) = fit_constant(&c, 5.0, 500);
        let (zz, n) = (m.zz[0][0], m.n[0][0]);
        let river = -zz / ((c.beta + n.sqrt()) / c.alpha + c.l2);
        assert_eq!(m.coefficients()[0][0].to_bits(), river.to_bits());
    }
}
