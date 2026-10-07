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
//!   where `ewridge` is not.
//!
//! Per row (`z` includes the intercept when configured, `p` the predicted
//! probability, `g_i = (p - y) * z_i * w` the gradient):
//!
//! ```text
//! decay:   n_i <- lam·n_i ;  zz_i <- lam·zz_i ;  d_i <- lam·d_i   (lam from the clock)
//! predict: b_i = 0 if |zz_i| <= l1
//!              = -(zz_i - sign(zz_i) l1) / (beta/alpha + d_i + l2)          under a half_life,
//!                with l1, l2 and beta/alpha times the target's scale m (below)
//!              = -(zz_i - sign(zz_i) l1) / ((beta + sqrt(n_i))/alpha + l2)  without one
//!          p   = sigmoid(z . b)
//! update:  s_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
//!          zz_i += g_i - s_i b_i ;  n_i += g_i^2 ;  d_i += s_i
//! ```
//!
//! **What a half-life does here** (review 2026-09-12, C24). The proximal
//! weight is a closed form in sums, and the sums decay. Without decay `d`
//! telescopes to `sqrt(n)/alpha`, and the model is river's `FTRLProximal`,
//! computed as river computes it (T-R1). Under a half-life `d` is the
//! discounted sum of the steps themselves; decaying `n` inside the square
//! root instead shrank every coefficient toward zero on every row, and a
//! constant target of 5 settled at 2.25 at `half_life = 100`.
//!
//! **The penalties under a half-life** (docs/PLAN.md task 115 (d)). `beta`,
//! `l1` and `l2` are river's constants on the sums' scale, and the sums
//! decay; held constant, they shrank the fit toward zero on every row that
//! taught it nothing, 0.75 of itself over one half-life at 100, where
//! `ewridge` does not move. So each target's penalties take a scale,
//!
//! ```text
//! b_i  = -(zz_i - sign(zz_i)·l1·m) / (beta/alpha·m + d_i + l2·m),  0 if |zz_i| <= l1·m
//! m    = W / W*       W  = the target's weight, decayed on every row
//!                     W* = the same on a clock that runs only on the rows that teach it
//! ```
//!
//! A row that teaches the target nothing -- absent, at weight 0, a label
//! `strict_binary` refuses, or a gradient whose square would overflow --
//! ages `zz`, `d` and `m` by the same factor, so the fit does not move, and
//! adds nothing to the target's weight; the rows that teach it bring `m`
//! back toward 1.
//! Without a half-life `W = W*`, `m = 1`, and the model is river's to the
//! bit, and Vowpal Wabbit's to its single precision. The steady state is as
//! before: the penalties act as a mean-scale ridge of
//! `(1 − lam)·(beta/alpha + l2)`, and a constant 5 settles at
//! `5/(1 + (1 − lam)(beta/alpha + l2))`, 4.65 at `half_life = 100` and 4.96
//! at 1000 -- the prior against the
//! window's worth of evidence, which vanishes as the half-life grows. The
//! model keeps a target's decay until the next row that teaches it, so the
//! fit read from the sums as that row left them is the frozen one exactly,
//! and a gap of any length cannot take the sums into the subnormal range.
//!
//! **Row weights** (docs/PLAN.md task 151). The gradient is `(p − y)·z·w`,
//! an importance weight as Vowpal Wabbit's, and `l1`, `l2` and `beta` are a
//! prior of fixed mass against evidence that grows with weight and density:
//! FTRL minimizes the cumulative loss plus a fixed regularizer, which its
//! regret bound rests on, so under a half-life the effective penalty is
//! `l1 / W` for the weight the window holds. The sum-scale family, beside
//! `rls`'s ridge, `ridge_scale` and `kalman`'s observation precision; the
//! mean-scale sparse fit, invariant to both, is `lasso`.
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
    pub fit_intercept: bool,
    pub decay: Decay,
    /// Learning-rate scale.
    pub alpha: f64,
    /// Learning-rate smoothing.
    pub beta: f64,
    pub l1: f64,
    pub l2: f64,
    pub min_weight: f64,
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
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("ftrl: {e}"))?;
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
    /// term under a half-life, where `sqrt(n)/alpha` is without one (C24).
    #[serde(default)]
    prox: Vec<Vec<f64>>,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it, decayed: what its
    /// `min_weight` is checked against (hard rule 8, docs/PLAN.md task 115
    /// (d)). `w_sum` stood in for it, so ten rows with a null target met
    /// `min_weight = 10` with every coefficient at zero and `pred = 0.5`
    /// (review 2026-09-12, S31). A row that taught it nothing does not
    /// count: a label `strict_binary` refuses, and a row whose gradient
    /// would overflow, which is skipped (review 2026-10-05, CC3).
    w_target: Vec<f64>,
    /// Per target, the scale on the penalties `beta/alpha`, `l1` and `l2`
    /// under a half-life, as the last row that taught it left it: `W/W*`, its
    /// weight over its weight on a clock that runs only on the rows that
    /// teach it (docs/PLAN.md task 115 (d); the module docs). `1` without a
    /// half-life.
    scale: Vec<f64>,
    /// Per target, `W*`: its weight on the clock of the rows that teach it.
    w_taught: Vec<f64>,
    /// Per target, the decay the clock has run since the last row that
    /// taught it, not yet applied to its sums: they stay as that row left
    /// them, so the fit read from them is the frozen one exactly, and a
    /// long gap cannot run them into the subnormal range.
    pending: Vec<f64>,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    coef: Vec<f64>,
}

/// A target's penalties as its weights read them ([`Ftrl::penalties`]):
/// under a half-life `l1`, `beta / alpha` and `l2` each times the target's
/// scale, else river's `l1`, `beta` and `l2`.
#[derive(Clone, Copy)]
struct Penalties {
    forgets: bool,
    l1: f64,
    base: f64,
    l2: f64,
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
            w_target: vec![0.0; m],
            scale: vec![1.0; m],
            w_taught: vec![0.0; m],
            pending: vec![1.0; m],
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

    /// Per target, the weight of the rows that carried it: what its
    /// `min_weight` is checked against.
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// Proximal weights implied by the current FTRL state, per target.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| {
                let pen = self.penalties(j);
                (0..self.cfg.k_total())
                    .map(|i| self.weight(pen, j, i))
                    .collect()
            })
            .collect()
    }

    #[inline]
    fn weight(&self, pen: Penalties, j: usize, i: usize) -> f64 {
        self.weight_of(pen, self.zz[j][i], self.n[j][i], self.prox[j][i])
    }

    /// Target `j`'s penalties as its weights read them, taken once for all
    /// of its coordinates: `k` of them a row, each recomputing the same
    /// products and the `beta / alpha` quotient, had cost `ftrl` 18% of its
    /// rows a second once the penalties took the target's scale. The same
    /// operations in the same order, so the same bits.
    fn penalties(&self, j: usize) -> Penalties {
        if self.forgets() {
            let scale = self.scale[j];
            Penalties {
                forgets: true,
                l1: self.cfg.l1 * scale,
                base: self.cfg.beta / self.cfg.alpha * scale,
                l2: self.cfg.l2 * scale,
            }
        } else {
            Penalties {
                forgets: false,
                l1: self.cfg.l1,
                base: self.cfg.beta,
                l2: self.cfg.l2,
            }
        }
    }

    /// Whether the sums decay at all. Without a half-life the rate is river's
    /// `(beta + sqrt(n))/alpha`, computed as river computes it.
    fn forgets(&self) -> bool {
        match self.cfg.decay {
            Decay::Halflife(h) => h.is_finite(),
            Decay::Lam(l) => l != 1.0,
        }
    }

    /// The proximal weight for one coordinate's `(z, n, d)` -- the closed
    /// form of the FTRL-Proximal update, shared by `step` and `predict`.
    /// Under a half-life the rate's term is `d`, the discounted sum of the
    /// proximal steps, and the penalties take the target's `scale`; without
    /// one it is `sqrt(n)/alpha`, which `d` telescopes to, and the penalties
    /// are river's (review 2026-09-12, C24; the module docs).
    fn weight_of(&self, pen: Penalties, zz: f64, n: f64, prox: f64) -> f64 {
        if pen.forgets {
            if zz.abs() <= pen.l1 {
                return 0.0;
            }
            let sgn = if zz < 0.0 { -1.0 } else { 1.0 };
            // `beta / alpha * scale + prox + l2 * scale`, left to right.
            let rate = pen.base + prox + pen.l2;
            // No evidence and no prior left -- a total gap takes both to 0 --
            // is no fit, not 0/0.
            return if rate > 0.0 {
                -(zz - sgn * pen.l1) / rate
            } else {
                0.0
            };
        }
        if zz.abs() <= pen.l1 {
            0.0
        } else {
            // Never zero here: `|zz| > l1 >= 0`, or `zz` is NaN, so `< 0` and
            // `<= 0` agree (scripts/mutants_equivalent.toml).
            let sgn = if zz < 0.0 { -1.0 } else { 1.0 };
            let rate = (pen.base + n.sqrt()) / self.cfg.alpha + pen.l2;
            // No evidence and no prior -- `beta = l2 = 0` and a gradient
            // whose square underflowed, which moves `z` and not `n` -- is
            // no fit, as under a half-life above: `-z / 0` was an infinite
            // weight, and every row after it skipped (review 2026-10-05,
            // CC2).
            if rate > 0.0 {
                -(zz - sgn * pen.l1) / rate
            } else {
                0.0
            }
        }
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.coef = vec![0.0; k];
        }
    }

    /// Learn target `j` from the row in `zbuf`, its prediction `p` read
    /// with the weights in `coef`, and say whether the row taught it. A row
    /// that teaches nothing -- the target absent, a weight of 0, a label
    /// `strict_binary` refuses, or a gradient whose square would overflow
    /// -- touches none of the target's state: its sums keep the decay they
    /// are owed, `W*` and the scale stay as the last row that taught left
    /// them, so the fit does not move (the module docs). The overflow skip
    /// came after the owed decay, `W*` and the scale had moved as though
    /// the row taught, and the next prediction moved 2.2% (review
    /// 2026-10-05, CC3).
    fn teach(&mut self, j: usize, p: f64, y: Option<f64>, lam: f64, weight: f64) -> bool {
        let Some(yj) = y else { return false };
        if weight <= 0.0 {
            return false;
        }
        let yb = match self.cfg.loss {
            FtrlLoss::Squared => yj,
            FtrlLoss::Logistic if self.cfg.strict_binary => {
                if yj != 0.0 && yj != 1.0 {
                    return false; // caller asked for strictness: skip, do not learn
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
            return false;
        }
        let k = self.cfg.k_total();
        // The row teaches the target: its sums take the decay they were
        // owed, and its weight on the teaching clock this row's.
        let owed = self.pending[j];
        if owed != 1.0 {
            for i in 0..k {
                self.n[j][i] *= owed;
                self.zz[j][i] *= owed;
                self.prox[j][i] *= owed;
            }
            self.pending[j] = 1.0;
        }
        self.w_taught[j] = lam * self.w_taught[j] + weight;
        // The penalties' scale `W/W*`, with `W` as the ageing after this
        // leaves it: rows that teach bring it back toward 1 after a gap took
        // it down with the sums. Without a half-life it stays 1.
        if self.forgets() {
            self.scale[j] = (lam * self.w_target[j] + weight) / self.w_taught[j];
        }
        for i in 0..k {
            let g = err * self.zbuf[i] * weight;
            let n_new = self.n[j][i] + g * g;
            let s = (n_new.sqrt() - self.n[j][i].sqrt()) / self.cfg.alpha;
            self.zz[j][i] += g - s * self.coef[i];
            self.n[j][i] = n_new;
            self.prox[j][i] += s;
        }
        true
    }
}

impl OnlineModel for Ftrl {
    /// Each target's own weight, the rows that carried it, for the bank's
    /// per-target `min_weight` gate: without it the bank checked a list of
    /// thresholds against the shared `n_eff` (task 158, E14).
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.w_target);
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // The accumulators forget on the model's clock. Each target's decay
        // waits in `pending` until a row teaches it: the fit read from the
        // sums as that row left them, under the scale it left, is the one
        // the decayed sums and penalties give (the module docs).
        if lam != 1.0 {
            for p in &mut self.pending {
                *p *= lam;
            }
        }
        // Each target's `min_weight` reads its own weight.
        let n_eff = self.w_sum;

        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            // Proximal weights from the state before this row's update.
            let pen = self.penalties(j);
            for i in 0..k {
                self.coef[i] = self.weight(pen, j, i);
            }
            let raw: f64 = self.zbuf.iter().zip(&self.coef).map(|(z, b)| z * b).sum();
            // The two losses share everything but the link; the gradient is
            // `(p - y) * z` either way.
            let p = match self.cfg.loss {
                FtrlLoss::Logistic => sigmoid(raw),
                FtrlLoss::Squared => raw,
            };
            if self.w_target[j] >= self.cfg.min_weight {
                pred[j] = p;
            }
            let taught = self.teach(j, p, y[j], lam, weight);
            // The target's own weight, aged by the row and carrying it only
            // where it taught (`crate::model::age_target_weights`'
            // expression): read above, and in `teach`, as it stood before
            // the row, and by no other target.
            self.w_target[j] = lam * self.w_target[j] + if taught { weight } else { 0.0 };
        }
        self.w_sum = lam * self.w_sum + weight;
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    /// The row scored from the fit as the last row that taught each target
    /// left it: a row that teaches nothing does not move it, so the clock
    /// since does not either (the module docs).
    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let n_eff = self.w_sum;
        let m = self.cfg.n_targets;
        let mut pred = vec![f64::NAN; m];
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        for (j, p) in pred.iter_mut().enumerate() {
            if self.w_target[j] < self.cfg.min_weight {
                continue;
            }
            let pen = self.penalties(j);
            let raw: f64 = (0..k)
                .map(|i| {
                    let z = if i < off { 1.0 } else { x[i - off] };
                    z * self.weight(pen, j, i)
                })
                .sum();
            *p = match self.cfg.loss {
                FtrlLoss::Logistic => sigmoid(raw),
                FtrlLoss::Squared => raw,
            };
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
                crate::model::check_cfg("ftrl", m.cfg.validate())?;
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                let rows = |v: &[Vec<f64>]| v.len() == n && v.iter().all(|r| r.len() == k);
                // The three per-target accumulators at the cfg's width.
                // `prox` is `serde(default)`, so a state without it loaded
                // as `[]` and `self.prox[j][i]` panicked on the first `step`
                // (review 2026-09-18, B3).
                if !rows(&m.n) || !rows(&m.zz) || !rows(&m.prox) {
                    return Err(StateError::Invalid(
                        "ftrl: the accumulators have the wrong shape".into(),
                    ));
                }
                if m.w_target.len() != n {
                    return Err(StateError::Invalid(
                        "ftrl: the target weights have the wrong shape".into(),
                    ));
                }
                if [&m.scale, &m.w_taught, &m.pending]
                    .iter()
                    .any(|v| v.len() != n)
                {
                    return Err(StateError::Invalid(
                        "ftrl: the penalties' scale has the wrong shape".into(),
                    ));
                }
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

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step`: here the `serde(default)` case, a
    /// file without `prox` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Ftrl::new(cfg(2, 1)).unwrap();
        let mut s = m.state();
        let ModelState::Ftrl(inner) = &mut s.model else {
            unreachable!()
        };
        inner.prox.clear();
        match Ftrl::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

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

    /// A target's `min_weight` counts only the rows that carried it, and
    /// `n_eff` stays every row's (hard rule 8, docs/PLAN.md task 115 (d)):
    /// S31's ten rows with no target met `min_weight = 3` with every
    /// coefficient at zero, and `pred` was 0.5. A zero-weight row with the
    /// target (row 12) only ages it, and a label `strict_binary` refuses
    /// (the second target's 0.5) never counts.
    #[test]
    fn min_periods_counts_only_the_rows_that_carried_the_target() {
        let mut c = cfg(2, 2);
        c.min_weight = 3.0;
        c.strict_binary = true;
        let mut m = Ftrl::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..16usize {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = (i >= 10).then(|| f64::from(u8::from(x[0] > x[1])));
            let out = m.step(&x, &[y0, Some(0.5)], 1.0, if i == 12 { 0.0 } else { 1.0 });
            assert_eq!(out.n_eff, (i - usize::from(i > 12)) as f64, "row {i}");
            assert_eq!(out.pred[0].is_nan(), i < 14, "row {i}");
            assert!(out.pred[1].is_nan(), "row {i}");
            let served = m.predict(&x, 1.0).pred;
            assert_eq!(served[0].is_nan(), i < 13, "row {i}");
            assert!(served[1].is_nan(), "row {i}");
        }
        assert_eq!(m.target_weights(), &[5.0, 0.0]);
    }

    /// A state without target weights, or without the penalties' scale, is
    /// refused, where it loaded with each target at the shared weight and
    /// each penalty whole -- the layouts before schema 20, which this build
    /// refuses by their version, and the repair a damaged file reached
    /// (docs/PLAN.md task 198).
    #[test]
    fn a_state_without_target_weights_is_refused() {
        use crate::{ModelState, State};
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(50.0);
        let mut m = Ftrl::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..8 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[(i >= 5).then_some(1.0), None], 1.0, 1.0);
        }
        assert!(m.target_weights()[0] < m.n_eff() && m.target_weights()[1] == 0.0);
        for key in ["w_target", "scale", "w_taught", "pending"] {
            let mut v = serde_json::to_value(&m).unwrap();
            assert!(v.as_object_mut().unwrap().remove(key).is_some());
            let err = serde_json::from_value::<Ftrl>(v).unwrap_err().to_string();
            assert!(err.contains(&format!("missing field `{key}`")), "{err}");
            let mut short = serde_json::to_value(&m).unwrap();
            short[key] = serde_json::json!([]);
            let old: Ftrl = serde_json::from_value(short).unwrap();
            let err = Ftrl::restore(&State::new(ModelState::Ftrl(Box::new(old)))).unwrap_err();
            assert!(err.to_string().contains("wrong shape"), "{key}: {err}");
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
            c.min_weight = 0.0;
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
            c.min_weight = 0.0;
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
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 1.0,
            min_weight: 10.0,
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
        c.min_weight = 5.0;
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
        c.min_weight = 2.0;
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
        // A regime flip: with a short half-life the model must follow it.
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
    /// decay and does not with it (review 2026-09-12, C24). The penalties
    /// carry the scale `m = W/W*`: `W` the target's weight, decayed on every
    /// row, `W*` the same on a clock that runs only on the rows that teach
    /// it (docs/PLAN.md task 115 (d)). Every row here is `(clock, target,
    /// weight)`; a row teaches when the target is there and the weight is
    /// above 0. Each row's prediction, and the last state `(zz, n, d)`.
    fn decayed_longhand(
        c: &FtrlCfg,
        rows: &[(f64, Option<f64>, f64)],
    ) -> (Vec<f64>, f64, f64, f64) {
        let (mut zz, mut n, mut d) = (0.0f64, 0.0f64, 0.0f64);
        let (mut w, mut w_taught) = (0.0f64, 0.0f64);
        let mut preds = Vec::with_capacity(rows.len());
        for &(clock, y, weight) in rows {
            let lam = c.decay.factor(clock);
            zz *= lam;
            n *= lam;
            d *= lam;
            w *= lam;
            let m = if w_taught > 0.0 { w / w_taught } else { 1.0 };
            let b = if zz.abs() <= c.l1 * m {
                0.0
            } else {
                -(zz - zz.signum() * c.l1 * m) / (c.beta / c.alpha * m + d + c.l2 * m)
            };
            preds.push(b);
            let Some(y) = y.filter(|_| weight > 0.0) else {
                continue;
            };
            let g = (b - y) * weight;
            let s = ((n + g * g).sqrt() - n.sqrt()) / c.alpha;
            zz += g - s * b;
            n += g * g;
            d += s;
            w += weight;
            w_taught = lam * w_taught + weight;
        }
        (preds, zz, n, d)
    }

    /// `rows` rows of the constant target `y` at weight 1, a clock of 1
    /// between them, the first at 0.
    fn constant_rows(y: f64, rows: usize) -> Vec<(f64, Option<f64>, f64)> {
        (0..rows)
            .map(|i| (if i == 0 { 0.0 } else { 1.0 }, Some(y), 1.0))
            .collect()
    }

    fn intercept_only(decay: Decay) -> FtrlCfg {
        let mut c = cfg(1, 1);
        c.loss = FtrlLoss::Squared;
        c.decay = decay;
        c.min_weight = 0.0;
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

    /// Under a half-life the proximal term is a decayed sum of its own. The
    /// fit is the longhand's on every row, and a constant target of 5 settles
    /// where sum-scale penalties put it, `5 / (1 + (1 − λ)(β/α + l2))` -- a
    /// mean-scale ridge of `(1 − λ)(β/α + l2)` -- 4.65 at `half_life = 100`.
    /// Decaying `n` inside `√n` had put it at 2.25 (review 2026-09-12, C24).
    #[test]
    fn a_decaying_state_keeps_its_proximal_sum() {
        let c = intercept_only(Decay::Halflife(100.0));
        let (preds, _) = fit_constant(&c, 5.0, 20_000);
        let (want, ..) = decayed_longhand(&c, &constant_rows(5.0, 20_000));
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

    /// A row that teaches the target nothing leaves the fit where it was
    /// (docs/PLAN.md task 115 (d); the user, 2026-09-29: "Build it"): the
    /// penalties age with the sums, so the target absent, a zero weight, a
    /// clock of 100 or of a million, and a hundred thousand half-lives of
    /// rows without it all leave every coefficient and every prediction to
    /// the bit. With constant penalties 100 clock units took the
    /// coefficient to 0.75 of itself at `half_life = 100` (review 2026-09-12,
    /// C24).
    #[test]
    fn a_row_that_teaches_nothing_leaves_the_fit() {
        for l1 in [0.0, 0.5] {
            let mut c = intercept_only(Decay::Halflife(100.0));
            c.l1 = l1;
            let (_, mut m) = fit_constant(&c, 5.0, 5_000);
            let before = m.coefficients()[0][0].to_bits();
            let pred = m.predict(&[0.0], 1.0).pred[0].to_bits();
            assert_eq!(pred, before, "the next row is predicted with the fit");
            for (clock, y, w) in [(100.0, None, 1.0), (1.0, Some(5.0), 0.0), (1e6, None, 0.0)] {
                let step = m.step(&[0.0], &[y], clock, w);
                assert_eq!(step.pred[0].to_bits(), before, "{clock} {y:?} {w}");
                assert_eq!(
                    m.coefficients()[0][0].to_bits(),
                    before,
                    "{clock} {y:?} {w}"
                );
            }
            // After a total gap the next row that teaches is scored with the
            // frozen fit, and learns from there: the old sums are gone, and the
            // proximal step centres on that fit.
            let after = m.step(&[0.0], &[Some(5.0)], 1.0, 1.0).pred[0];
            assert_eq!(after.to_bits(), before, "the first row after a total gap");
            let next = m.coefficients()[0][0];
            assert!(next.is_finite() && (next - 5.0).abs() < 5.0, "{next}");
            // A row whose squared gradient would overflow -- a feature and a
            // target at the input bound -- is skipped, and a skipped row
            // teaches nothing: the fit, the target's weight and its scale
            // after it are those after the same row with the target absent
            // (review 2026-10-05, CC3: the skip came after the owed decay,
            // `W*` and the scale had moved as though it taught).
            let mut c2 = intercept_only(Decay::Halflife(20.0));
            c2.l1 = l1;
            let (_, fitted) = fit_constant(&c2, 5.0, 300);
            let (mut skipped, mut absent) = (fitted.clone(), fitted);
            skipped.step(&[1e100], &[Some(1e100)], 1.0, 1.0);
            absent.step(&[1e100], &[None], 1.0, 1.0);
            assert_eq!(
                skipped.coefficients()[0][0].to_bits(),
                absent.coefficients()[0][0].to_bits(),
                "the overflow row moved the fit"
            );
            assert_eq!(skipped.target_weights(), absent.target_weights());
            assert_eq!(skipped.scale, absent.scale);
            assert_eq!(skipped.w_taught, absent.w_taught);
            let (a, b) = (
                skipped.step(&[0.5], &[Some(5.0)], 1.0, 1.0).pred[0],
                absent.step(&[0.5], &[Some(5.0)], 1.0, 1.0).pred[0],
            );
            assert_eq!(a.to_bits(), b.to_bits(), "the next prediction");
            let c_short = intercept_only(Decay::Halflife(1.0));
            let (_, mut short) = fit_constant(&c_short, 5.0, 200);
            let frozen = short.coefficients()[0][0].to_bits();
            for _ in 0..100_000 {
                short.step(&[0.0], &[None], 1.0, 1.0);
            }
            assert_eq!(short.coefficients()[0][0].to_bits(), frozen, "no underflow");
            assert_eq!(short.predict(&[0.0], 1.0).pred[0].to_bits(), frozen);
        }
    }

    /// The fit after rows that teach nothing, and after the rows that teach
    /// it again, is the longhand's: the decayed sums over the decayed
    /// penalties, the scale `W/W*` coming back toward 1 as the target's rows
    /// refill the window. Gaps of one clock unit to 2,000 -- `2^-50` of the
    /// weight at `half_life = 40`, where the longhand's decayed sums are still
    /// normal numbers; a total gap is the test above -- a zero weight and
    /// weights of 0.3 to 2, under both penalties' settings.
    #[test]
    fn the_penalty_scale_is_the_longhands_across_gaps() {
        for l1 in [0.0, 0.2] {
            let mut c = intercept_only(Decay::Halflife(40.0));
            c.l1 = l1;
            let mut s = 29u64;
            let rows: Vec<(f64, Option<f64>, f64)> = (0..3_000)
                .map(|i| {
                    let clock = match i {
                        0 => 0.0,
                        _ if i % 500 == 250 => 2_000.0,
                        _ if i % 97 == 13 => 60.0,
                        _ => 1.0,
                    };
                    let y = (i % 5 != 2 && !(700..900).contains(&i)).then(|| 5.0 + lcg(&mut s));
                    let w = if i % 11 == 3 {
                        0.0
                    } else {
                        1.15 + 0.85 * lcg(&mut s)
                    };
                    (clock, y, w)
                })
                .collect();
            let (want, ..) = decayed_longhand(&c, &rows);
            let mut m = Ftrl::new(c).unwrap();
            for (i, &(clock, y, w)) in rows.iter().enumerate() {
                let got = m.step(&[0.0], &[y], clock, w).pred[0];
                assert!(
                    (got - want[i]).abs() <= 1e-12 * want[i].abs().max(1.0),
                    "l1 {l1}, row {i}: {got} vs {}",
                    want[i]
                );
            }
        }
    }

    /// Without decay the proximal term telescopes to `√n/α`, and the weight
    /// is river's closed form computed as river computes it, so T-R1 holds
    /// to the bit and the repair of C24 changes nothing at `half_life = inf`.
    #[test]
    fn without_decay_the_weights_are_rivers_closed_form() {
        let c = intercept_only(Decay::Halflife(f64::INFINITY));
        let (_, m) = fit_constant(&c, 5.0, 500);
        let (zz, n) = (m.zz[0][0], m.n[0][0]);
        let river = -zz / ((c.beta + n.sqrt()) / c.alpha + c.l2);
        assert_eq!(m.coefficients()[0][0].to_bits(), river.to_bits());
    }

    /// A decay given as a per-unit factor is the half-life it equals: at a
    /// clock of one unit a row `Lam(2^(-1/h))` and `Halflife(h)` age the
    /// sums by the same double, so the two models are one to the bit -- the
    /// forgetting form included, where `Lam(lam)` once read as "no decay"
    /// would have taken river's penalties. `Lam(1)` forgets nothing, as an
    /// infinite half-life does.
    #[test]
    fn a_decay_factor_is_the_half_life_it_equals() {
        let h = 7.0;
        let lam = Decay::Halflife(h).factor(1.0);
        assert!(lam < 1.0 && Decay::Lam(lam).factor(1.0) == lam);
        for (a, b) in [
            (Decay::Lam(lam), Decay::Halflife(h)),
            (Decay::Lam(1.0), Decay::Halflife(f64::INFINITY)),
        ] {
            let mk = |decay: Decay| {
                let mut c = cfg(2, 2);
                c.decay = decay;
                c.l1 = 0.05;
                c.min_weight = 0.0;
                c.loss = FtrlLoss::Squared;
                Ftrl::new(c).unwrap()
            };
            let (mut ma, mut mb) = (mk(a), mk(b));
            let mut s = 31u64;
            for i in 0..60 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = [Some(2.0 * x[0] - x[1]), (i % 3 != 0).then(|| x[1] + 0.5)];
                let d = if i == 0 { 0.0 } else { 1.0 };
                let (pa, pb) = (ma.step(&x, &y, d, 1.0), mb.step(&x, &y, d, 1.0));
                assert_eq!(pa, pb, "{a:?} against {b:?} at row {i}");
            }
            assert_eq!(ma.coefficients(), mb.coefficients(), "{a:?} against {b:?}");
            assert_eq!(ma.forgets(), mb.forgets(), "{a:?} against {b:?}");
        }
    }

    /// River's closed form without decay, `b = -(z - sign(z)·l1) /
    /// ((beta + sqrt(n))/alpha + l2)` (McMahan et al. 2013, Algorithm 1),
    /// after one row from zero: the squared loss on `y = 10` at `p = 0`
    /// gives `g = -10`, `n = 100`, `z = -10`, and with `alpha = 0.1`, `beta
    /// = l1 = l2 = 1` the intercept is `9 / 111`. The penalty pulls toward
    /// zero from either side, so `y = -10` gives `-9 / 111`.
    #[test]
    fn the_l1_shrinkage_pulls_toward_zero_from_either_side() {
        for (y, want) in [(10.0, 9.0 / 111.0), (-10.0, -9.0 / 111.0)] {
            let mut c = cfg(1, 1);
            c.l1 = 1.0;
            c.min_weight = 0.0;
            c.loss = FtrlLoss::Squared;
            let mut m = Ftrl::new(c).unwrap();
            assert!(!m.forgets());
            m.step(&[0.0], &[Some(y)], 0.0, 1.0);
            assert_eq!(m.zz[0][0], -y, "z after one row");
            let b = m.coefficients()[0][0];
            assert!((b - want).abs() < 1e-12, "y = {y}: {b} against {want}");
            assert_eq!(m.coefficients()[0][1], 0.0, "a zero feature stays at zero");
        }
    }

    /// No prior and no curvature yet is no fit. With `beta = l2 = 0`, a
    /// target so small that its squared gradient underflows moves `z` but
    /// adds nothing to `n` or to the proximal sum, so the rate is 0: the
    /// weight is 0, not `-z / 0 = ±inf` -- under a half-life, and without
    /// one, where the rate is river's `(beta + √n)/alpha` (review
    /// 2026-10-05, CC2: there an infinite coefficient made every later
    /// prediction infinite and every later row a skipped one).
    #[test]
    fn a_rate_of_zero_is_no_fit() {
        for decay in [Decay::Halflife(50.0), Decay::Halflife(f64::INFINITY)] {
            let mut c = cfg(1, 1);
            c.decay = decay;
            c.beta = 0.0;
            c.l2 = 0.0;
            c.min_weight = 0.0;
            c.loss = FtrlLoss::Squared;
            let mut m = Ftrl::new(c).unwrap();
            m.step(&[0.0], &[Some(1e-170)], 0.0, 1.0);
            assert_eq!(m.forgets(), decay.factor(1.0) < 1.0, "{decay:?}");
            assert_eq!(m.zz[0][0], -1e-170, "{decay:?}: the row moved z");
            assert_eq!(
                (m.n[0][0], m.prox[0][0]),
                (0.0, 0.0),
                "{decay:?}: and no sum"
            );
            assert_eq!(m.coefficients(), vec![vec![0.0, 0.0]], "{decay:?}");
            let p = m.predict(&[1.0], 1.0).pred[0];
            assert_eq!(p, 0.0, "{decay:?}");
        }
    }

    /// A row of weight `1e-170` teaches a gradient whose square underflows,
    /// so after it the sums hold a `z` and no rate; the rows after it are
    /// ordinary and must be learned from. Without decay at `beta = l2 = 0`
    /// the weight read from such sums was `±inf`: every prediction after it
    /// infinite, and every row skipped by the overflow guard, for the rest
    /// of the stream (review 2026-10-05, CC2). The fit is held to the stream
    /// without the light row, which it can differ from only by what a
    /// gradient of `2e-170` moved.
    #[test]
    fn a_row_too_light_to_square_leaves_the_rows_after_it_learning() {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY);
        c.beta = 0.0;
        c.l2 = 0.0;
        c.min_weight = 0.0;
        c.loss = FtrlLoss::Squared;
        let (mut light, mut without) = (Ftrl::new(c.clone()).unwrap(), Ftrl::new(c).unwrap());
        light.step(&[0.5], &[Some(2.0)], 0.0, 1e-170);
        for i in 1..12 {
            let y = 2.0 + 0.1 * f64::from(i);
            let (a, b) = (
                light.step(&[0.5], &[Some(y)], 1.0, 1.0).pred[0],
                without
                    .step(&[0.5], &[Some(y)], if i == 1 { 0.0 } else { 1.0 }, 1.0)
                    .pred[0],
            );
            assert!(a.is_finite(), "row {i}: {a}");
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "row {i}: {a} against {b}"
            );
        }
        let last = light.predict(&[0.5], 1.0).pred[0];
        assert!(
            last > 0.5,
            "the fit learned the rows after the light one: {last}"
        );
    }

    /// The bank's per-target `min_weight` gate reads each target's own
    /// weight through the trait: `true`, and one entry a target, the decayed
    /// weight of the rows that carried it -- a sparse second target its own
    /// rows only (task 158, E14; the mutation run of 2026-10-05 left the
    /// body replaced by `true` or `false` alive).
    #[test]
    fn the_trait_reports_each_targets_own_weight() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(10.0);
        c.loss = FtrlLoss::Squared;
        let mut m = Ftrl::new(c).unwrap();
        let lam = Decay::Halflife(10.0).factor(1.0);
        let (mut want0, mut want1) = (0.0f64, 0.0f64);
        let mut s = 3u64;
        for i in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let w = 0.5 + lcg(&mut s).abs();
            let y1 = (i % 4 == 1).then_some(x[1]);
            m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, w);
            let f = if i == 0 { 1.0 } else { lam };
            want0 = f * want0 + w;
            want1 = f * want1 + if y1.is_some() { w } else { 0.0 };
        }
        let mut out = vec![7.0; 5];
        assert!(m.target_n_eff_into(&mut out));
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(
            (out[0] - want0).abs() <= 1e-12 * want0,
            "{out:?} against {want0}"
        );
        assert!(
            (out[1] - want1).abs() <= 1e-12 * want1,
            "{out:?} against {want1}"
        );
        assert!(
            out[1] < 0.5 * out[0],
            "the sparse target's own weight: {out:?}"
        );
    }

    /// One corruption per accumulator and per part of the penalties' scale,
    /// each refused alone; the three scale parts all absent are a state
    /// written before them, which loads at a scale of 1.
    #[test]
    fn each_part_of_a_state_is_checked_alone() {
        use crate::{ModelState, StateError};
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(20.0);
        let mut m = Ftrl::new(c).unwrap();
        m.step(&[0.5, -1.0], &[Some(1.0), Some(0.0)], 0.0, 1.0);
        m.step(&[1.5, 0.2], &[None, Some(1.0)], 1.0, 1.0);
        let restored = |f: &dyn Fn(&mut Ftrl)| {
            let mut s = m.state();
            let ModelState::Ftrl(inner) = &mut s.model else {
                unreachable!()
            };
            f(inner);
            Ftrl::restore(&s)
        };
        assert_eq!(restored(&|_| {}).unwrap(), m);
        type Corruption<'a> = (&'a str, &'a dyn Fn(&mut Ftrl));
        let parts: [Corruption; 7] = [
            ("a target short in n", &|f: &mut Ftrl| {
                f.n.pop();
            }),
            ("a coordinate short in n", &|f: &mut Ftrl| {
                f.n[1].pop();
            }),
            ("a target short in zz", &|f: &mut Ftrl| {
                f.zz.pop();
            }),
            ("no prox", &|f: &mut Ftrl| f.prox.clear()),
            ("no scale", &|f: &mut Ftrl| f.scale.clear()),
            ("no pending decay", &|f: &mut Ftrl| f.pending.clear()),
            ("neither W* nor the pending decay", &|f: &mut Ftrl| {
                f.w_taught.clear();
                f.pending.clear();
            }),
        ];
        for (what, corrupt) in parts {
            match restored(corrupt) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
                other => panic!("{what}: {other:?}"),
            }
        }
        // All three absent, a state written before them, is refused too,
        // where it loaded with the penalties whole (docs/PLAN.md task 198).
        match restored(&|f| {
            f.scale.clear();
            f.w_taught.clear();
            f.pending.clear();
        }) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("all three absent: {other:?}"),
        }
    }
}
