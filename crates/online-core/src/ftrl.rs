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
//! A row that teaches the target nothing -- absent, at weight 0, or a label
//! `strict_binary` refuses -- ages `zz`, `d` and `m` by the same factor, so
//! the fit does not move; the rows that teach it bring `m` back toward 1.
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
    /// (review 2026-09-12, S31). A label `strict_binary` refuses does not
    /// count.
    #[serde(default)]
    w_target: Vec<f64>,
    /// Per target, the scale on the penalties `beta/alpha`, `l1` and `l2`
    /// under a half-life, as the last row that taught it left it: `W/W*`, its
    /// weight over its weight on a clock that runs only on the rows that
    /// teach it (docs/PLAN.md task 115 (d); the module docs). `1` without a
    /// half-life, and in a state written before it: the penalties that state
    /// was made with.
    #[serde(default)]
    scale: Vec<f64>,
    /// Per target, `W*`: its weight on the clock of the rows that teach it.
    /// A state written before it loads with `W`, a scale of 1.
    #[serde(default)]
    w_taught: Vec<f64>,
    /// Per target, the decay the clock has run since the last row that
    /// taught it, not yet applied to its sums: they stay as that row left
    /// them, so the fit read from them is the frozen one exactly, and a
    /// long gap cannot run them into the subnormal range. `1` in a state
    /// written before it, whose sums were decayed to its last row.
    #[serde(default)]
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
            let rate = (pen.base + n.sqrt()) / self.cfg.alpha;
            -(zz - sgn * pen.l1) / (rate + pen.l2)
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
            // The penalties' scale `W/W*`, with `W` as the ageing below
            // leaves it: rows that teach bring it back toward 1 after a gap
            // took it down with the sums. Without a half-life it stays 1.
            if self.forgets() {
                self.scale[j] = (lam * self.w_target[j] + weight) / self.w_taught[j];
            }
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
        let strict = self.cfg.strict_binary && matches!(self.cfg.loss, FtrlLoss::Logistic);
        crate::model::age_target_weights(
            &mut self.w_target,
            |j| y[j].is_some_and(|v| v.is_finite() && (!strict || v == 0.0 || v == 1.0)),
            lam,
            weight,
        );
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    /// The row scored from the fit as the last row that taught each target
    /// left it: a row that teaches nothing does not move it, so the clock
    /// since does not either (the module docs).
    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
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
                if !crate::model::restore_target_weights(&mut m.w_target, m.w_sum, n) {
                    return Err(StateError::Invalid(
                        "ftrl: the target weights have the wrong shape".into(),
                    ));
                }
                // A state written before the penalties' scale: its sums were
                // decayed to its last row and its penalties whole.
                if m.scale.is_empty() && m.w_taught.is_empty() && m.pending.is_empty() {
                    m.scale = vec![1.0; n];
                    m.w_taught = m.w_target.clone();
                    m.pending = vec![1.0; n];
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

    /// A schema-19 state keeps no target weights: it loads with each target
    /// at the shared weight, the one its gate read.
    #[test]
    fn a_state_without_target_weights_loads_at_the_shared_weight() {
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
        let mut v = serde_json::to_value(&m).unwrap();
        assert!(v.as_object_mut().unwrap().remove("w_target").is_some());
        let old: Ftrl = serde_json::from_value(v).unwrap();
        let back = Ftrl::restore(&State::new(ModelState::Ftrl(Box::new(old)))).unwrap();
        assert_eq!(back.target_weights(), &[m.n_eff(), m.n_eff()]);
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
}
