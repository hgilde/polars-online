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
//! | `Huber` | identity | `eta` | `clamp(p − y, ±delta·s)` |
//! | `Quantile` | identity | `eta` | `1{y < p} − tau` |
//! | `EpsilonInsensitive` | identity | `eta` | `0` if `|p − y| ≤ eps·s_y`, else `sign(p − y)` |
//! | `Poisson` | log | `exp(clamp(eta, −30, 30))` | `p − y` |
//! | `Logistic` | sigmoid | `sigmoid(eta)` | `p − clamp(y, 0, 1)` |
//!
//! then `g_i = d · z_i · w + l2 · b_i` for a slope and `g_0 = d · w` for the
//! intercept, which is not penalised; then `b_i -= lr_i · g_i`. The Poisson
//! link clamps its exponent at `±30`, so a rate is between `e^−30 ≈ 9.4e-14`
//! and `e^30 ≈ 1.1e13`, and a linear predictor past either end predicts as
//! that end does: the prediction and its gradient stay finite, where `exp`
//! alone overflows past 709.
//!
//! **`delta` is in units of `s`**, the target's EW residual standard
//! deviation as the row arrives, as `huber`'s `huber_delta` is
//! (`crate::Robust`); **`eps` is in units of `s_y`**, the target's own EW
//! standard deviation as the row arrives. A cut and a tube in the target's
//! own units fitted one scale and failed the others -- a target in
//! thousandths never left a tube of 0.1, so the model was passive for
//! ever, and one in thousands never entered it (docs/PLAN.md task 195,
//! review round 4 CC4). Under `huber`, `s²` is the EW mean of the squared
//! out-of-sample residual `y − p`, aged on every row on the model's clock
//! and learned from a row with the target, a weight above 0 and a
//! prediction, after the row has used it:
//!
//! ```text
//! W_s ← lam·W_s;   s² ← (W_s·s² + w·(y − p)²) / (W_s + w),   W_s ← W_s + w
//! ```
//!
//! skipped where it would not be finite, as `huber`'s is. Under
//! `epsilon_insensitive`, `s_y²` is the EW variance of `y` around its EW
//! mean over the rows with the target and a weight above 0, whatever the
//! fit, the row's own `y` joining after it is judged ([`crate::spread`]).
//! Before the target has one -- no residual yet, or every one so far
//! exactly 0, for `s`; fewer than two weighted rows, or every `y` the same,
//! for `s_y` -- there is nothing to draw a cut or a tube in: the Huber row
//! is a squared-loss row and the tube has no width, `huber`'s rule for its
//! rows before a scale. The squared loss is then equivariant in the
//! target's scale -- a target scaled by `c` fits as the unscaled one,
//! scaled by `c` -- and so is the Huber loss; the quantile and
//! epsilon-insensitive losses step by the sign of the residual, so their
//! `learning_rate` is in the target's units.
//!
//! **Why two units.** The fit starts from zero coefficients, so its first
//! residuals are the target's whole level. Outside the Huber cut the
//! gradient is clipped, not zero, so a cut drawn wide by those residuals
//! still lets every row teach, and narrows as they leave the mean. Inside
//! the tube the gradient is zero: a tube drawn in the residual's spread on
//! a target at 1,000 in a spread of 2 was about 100 wide, held every row,
//! and without decay never narrowed, so the fit stopped where it stood
//! (docs/PLAN.md task 202). `y`'s own spread does not read the fit.
//!
//! **A logistic label outside {0, 1}** is clamped into `[0, 1]`, as `ftrl`
//! clamps it; `strict_binary` instead learns nothing from it, and the bank
//! refuses the chunk naming the row. `p − y` with `y = 5` pushed the linear
//! predictor up on every row for ever (review round 4, CC9).
//!
//! Learning rates ([`LearningRate`]): a constant, an inverse-scaling schedule
//! that anneals with `n_eff`, or AdaGrad's per-coordinate `lr / (sqrt(G_i) + eps)`.
//! AdaGrad's accumulator and `n_eff` are both decayed on the model's clock, so
//! an annealed or adapted rate re-opens after a long gap instead of staying
//! frozen at whatever it had converged to. The coefficients do not decay:
//! under a constant rate their memory is in rows, about `1 / (lr · E[z²])`,
//! whatever the clock between them (docs/PLAN.md task 146).

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::spread::TargetSpread;
use crate::{Constraint, Decay, EwDiag};

/// Loss function, and with it the link (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SgdLoss {
    Squared,
    /// `delta` is in units of the target's EW residual std, as `robust`'s
    /// Huber is: a clipped gradient keeps learning, so a cut a fit's first
    /// residuals widen does not stop it (the module docs).
    Huber {
        delta: f64,
    },
    Quantile {
        tau: f64,
    },
    /// Ignores residuals within `eps` of the target's own EW std — the SVR
    /// loss. Not the residual's: inside the tube the gradient is zero, and
    /// a tube a fit's first residuals widen stops it (the module docs).
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
    pub fit_intercept: bool,
    pub decay: Decay,
    pub loss: SgdLoss,
    pub learning_rate: f64,
    pub schedule: LearningRate,
    /// Ridge penalty added to the gradient. The intercept is never penalized.
    pub l2: f64,
    pub min_weight: f64,
    /// Standardize features against their own running moments before the
    /// gradient step (ENHANCEMENTS E24), unscaling the coefficients on the way
    /// out so they stay in the caller's units.
    ///
    /// Gradient methods are the ones that need this: a single learning rate has
    /// to suit every coordinate, so a feature measured in thousands and one
    /// measured in basis points cannot both converge. The exact solvers do not
    /// care (they standardize inside the solve, or not at all).
    ///
    /// The moments a row is standardized against **include the row**, as one
    /// more row of the stream at unit weight ([`EwDiag::including`]): that
    /// is sklearn's `partial_fit` then `transform`, and it bounds a
    /// standardized value by `sqrt(n_eff)` whatever the running variance
    /// happens to be (the bound is reached when the history has no spread
    /// at all). Against the moments from *before* the row -- the
    /// original E24 -- a variance estimate a few rows old can be tiny by
    /// chance, the standardized value huge, and one step with
    /// `lr · |z|² > 2` throws a coefficient where the next hundred rows do
    /// not bring it back: R² of −6.9 over rows 25–50 of 200-row groups
    /// where `ewridge` scored 0.97, and a wide fit is that on every row
    /// (docs/PLAN.md task 74). Not a leak: the row's features are known at
    /// prediction time, and the rule is about the target, which enters
    /// nothing until after the prediction. The one standardized row serves
    /// the prediction and the gradient, so a row is standardized the way
    /// every row the coefficients were learned from was (sklearn's loop
    /// predicts against the moments from before the row instead, which is
    /// two standardizations a row and a prediction with no bound). The
    /// row's own weight applies to what it teaches, not to where it sits
    /// among the rows seen, so a prediction never depends on the weight and
    /// `predict` gives the step's number exactly.
    ///
    /// The coefficients are read out with the moments as they stand now,
    /// while each step was taken in the coordinates of the moments on its
    /// own row: `coefficients()`, `predict` and a constraint's projection
    /// read the betas through today's scaler, so a coefficient "per unit of
    /// `x`" moves with the scaler as well as with the fit, as `kalman`'s
    /// does. And one gap only `sgd` has: the step standardizes with the row
    /// admitted at unit weight, and the scaler then learns the row at its
    /// own weight, so for a weight other than 1 the coordinates the gradient
    /// was taken in and the ones the projection and the coefficients use a
    /// moment later differ by that weight (review 2026-09-12, D5).
    #[serde(default)]
    pub standardize: bool,
    /// Cap on `|gradient|` before the step. **Finite by default** (`1e3` via the
    /// spec layer), not because ordinary losses need it but because a log-link
    /// loss does: with `Poisson`, `p = exp(eta)`, so one row that pushes `eta`
    /// up makes the next gradient exponentially larger, and a constant learning
    /// rate diverges within a few thousand rows. Measured on a 30k-row Poisson
    /// stream: unclipped the intercept ran to -4e10, clipped at 1e3 it recovers
    /// the true `[0.4, 0.8]`. The cap does not bind for identity-link losses at
    /// ordinary scales — a squared-loss fit is bit-identical with and without
    /// it. `inf` disables it.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub clip_gradient: f64,
    /// Box and/or sum constraint on the slopes, imposed by Euclidean
    /// projection after each update (ENHANCEMENTS E40); the intercept is
    /// free. The projection is taken in the space the step is taken in: the
    /// caller's units, or the standardized coordinates under
    /// `standardize`, where a caller bound `lo_i` on `c_i` is the bound
    /// `lo_i * scale_i` on `b_i` and the sum is `sum(b_i / scale_i)`.
    /// The reported coefficients satisfy the constraint after every learned
    /// row, and the initial `0` is projected too, so a simplex starts uniform.
    #[serde(default)]
    pub constraint: Option<Constraint>,
    /// A logistic label that is not 0 or 1 is not learned from, where the
    /// default clamps it into `[0, 1]`: `ftrl`'s rule. Logistic only. The
    /// bank never hands the model such a row: it refuses the chunk, naming
    /// the row (docs/PLAN.md task 195, S4).
    #[serde(default)]
    pub strict_binary: bool,
}

impl SgdCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("sgd: {e}"))?;
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("sgd: n_features and n_targets must be >= 1".into());
        }
        if self.learning_rate <= 0.0 || self.learning_rate.is_nan() {
            return Err("sgd: learning_rate must be > 0".into());
        }
        // NaN passes `v < 0.0` and `v <= 0.0` alike, so each bound below
        // names it: a NaN `clip_gradient` reached `f64::clamp`, which panics
        // on a NaN bound, on the first learned row; a NaN `l2`, `eps` or
        // `power` poisoned every coefficient or rate (review 2026-09-18, B4).
        if self.l2.is_nan() || self.l2 < 0.0 {
            return Err("sgd: l2 must be >= 0".into());
        }
        if self.clip_gradient.is_nan() || self.clip_gradient <= 0.0 {
            return Err("sgd: clip_gradient must be > 0 (use inf to disable)".into());
        }
        match self.loss {
            // NaN too: `f64::clamp` panics on a NaN bound. `inf` clips
            // nothing, which is the squared loss.
            SgdLoss::Huber { delta } if delta <= 0.0 || delta.is_nan() => {
                return Err("sgd: huber delta must be > 0".into());
            }
            SgdLoss::Quantile { tau }
                if !(0.0..=1.0).contains(&tau) || tau == 0.0 || tau == 1.0 =>
            {
                return Err("sgd: quantile must be in (0, 1)".into());
            }
            SgdLoss::EpsilonInsensitive { eps } if eps.is_nan() || eps < 0.0 => {
                return Err("sgd: eps must be >= 0".into());
            }
            _ => {}
        }
        if let LearningRate::InvScaling { power } = self.schedule
            && (power.is_nan() || power < 0.0)
        {
            return Err("sgd: inv_scaling power must be >= 0".into());
        }
        if let Some(c) = &self.constraint {
            c.validate(self.n_features, "sgd")?;
        }
        if self.strict_binary && self.loss != SgdLoss::Logistic {
            return Err("sgd: strict_binary applies to the logistic loss only".into());
        }
        // What the spec layer refuses, refused here too, so the Rust API
        // and a state file are held to it (review round 5, C6): a NaN
        // `min_weight` is never reached, and a negative one is no floor.
        // `inf` never predicts, which is legal.
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err(format!(
                "sgd: min_weight must be >= 0, got {}",
                self.min_weight
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SgdV3")]
pub struct Sgd {
    cfg: SgdCfg,
    /// Running feature means and variances, when `standardize` is on.
    scaler: Option<EwDiag>,
    /// Coefficients per target.
    beta: Vec<Vec<f64>>,
    /// AdaGrad accumulators per target (empty for the other schedules).
    g2: Vec<Vec<f64>>,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it, decayed: what its
    /// `min_weight` is checked against (hard rule 8, docs/PLAN.md task 115
    /// (d)). `w_sum` stood in for it, so ten rows with a null target met
    /// `min_weight = 10` with every coefficient at zero.
    w_target: Vec<f64>,
    /// Per target, the EW variance of the out-of-sample residual, `s²`, and
    /// its weight: the scale `huber_delta` is in (the module docs;
    /// docs/PLAN.md task 195). Kept under the Huber loss only, and empty
    /// under every other.
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    /// Per target, the EW mean and variance of the target itself: the unit
    /// `eps` is in (the module docs; docs/PLAN.md task 202). Kept under the
    /// epsilon-insensitive loss only, and empty under every other.
    spread: TargetSpread,
    /// The standardized row `[1, z]` under a scaler; unused without one,
    /// where the model reads `x` in place.
    #[serde(skip)]
    zbuf: Vec<f64>,
    /// The raw row `[1, x]` the scaler is updated with, under a scaler.
    #[serde(skip)]
    rawbuf: Vec<f64>,
    /// Scratch for the projection.
    #[serde(skip)]
    pbuf: crate::constraint::Scratch,
    /// Which targets stepped this row (only kept under a constraint).
    #[serde(skip)]
    learned: Vec<bool>,
}

/// The layout `Sgd` loads, checked on the way in: what the cfg asks for,
/// the state carries, and nothing else. (A schema-2 `scaler` was a full
/// `EwCov` read through its diagonal; the minimum schema has passed it,
/// docs/PLAN.md tasks 198 and 194-202, and [`EwDiag`] refuses an `EwCov`'s
/// fields and shape.)
#[derive(Deserialize)]
struct SgdV3 {
    cfg: SgdCfg,
    scaler: Option<EwDiag>,
    beta: Vec<Vec<f64>>,
    g2: Vec<Vec<f64>>,
    w_sum: f64,
    w_target: Vec<f64>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    spread: TargetSpread,
}

impl TryFrom<SgdV3> for Sgd {
    type Error = String;

    fn try_from(v: SgdV3) -> Result<Self, String> {
        let (cfg, scaler, beta, g2, w_sum) = (v.cfg, v.scaler, v.beta, v.g2, v.w_sum);
        let (sig2, wsig, spread) = (v.sig2, v.wsig, v.spread);
        let w_target = v.w_target;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        // What the cfg asks for, the state carries, and nothing else: a
        // scaler exactly when `standardize` is on, and AdaGrad's sums, one
        // per target, exactly under AdaGrad. A file that lost its scaler
        // loaded as a model reading raw inputs with coefficients learned on
        // standardized ones, and one that lost its sums panicked on the first
        // step (review 2026-09-12, S16).
        if cfg.standardize != scaler.is_some() {
            return Err("sgd: the state's scaler does not match its cfg's standardize".into());
        }
        let sums = if matches!(cfg.schedule, LearningRate::AdaGrad) {
            m
        } else {
            0
        };
        if g2.len() != sums {
            return Err("sgd: the state's AdaGrad sums do not match its cfg's schedule".into());
        }
        if scaler.as_ref().is_some_and(|sc| sc.k() != k)
            || beta.len() != m
            || beta.iter().any(|b| b.len() != k)
            || g2.iter().any(|g| g.len() != k)
        {
            return Err("sgd: state has the wrong shape".into());
        }
        if w_target.len() != m {
            return Err("sgd: the state's target weights have the wrong shape".into());
        }
        // The scales a loss draws in, one per target, exactly under the
        // loss that draws in it: the residual's under Huber, the target's
        // own under epsilon-insensitive (docs/PLAN.md task 202).
        let (residual, own) = scales_kept(&cfg.loss);
        let scales = |kept: bool| if kept { m } else { 0 };
        if sig2.len() != scales(residual) || wsig.len() != scales(residual) {
            return Err("sgd: the state's residual variances do not match its cfg's loss".into());
        }
        if !spread.has_shape(scales(own)) {
            return Err("sgd: the state's target spreads do not match its cfg's loss".into());
        }
        Ok(Self {
            cfg,
            scaler,
            beta,
            g2,
            w_sum,
            w_target,
            sig2,
            wsig,
            spread,
            zbuf: vec![],
            rawbuf: vec![],
            pbuf: crate::constraint::Scratch::default(),
            learned: Vec::new(),
        })
    }
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
        let scaler = cfg.standardize.then(|| EwDiag::new(k));
        let mut beta = vec![vec![0.0; k]; m];
        if let Some(c) = &cfg.constraint {
            let off = usize::from(cfg.fit_intercept);
            let mut scratch = crate::constraint::Scratch::default();
            for b in beta.iter_mut() {
                c.project(&mut b[off..], None, &mut scratch);
            }
        }
        let (residual, own) = scales_kept(&cfg.loss);
        let residual = if residual { m } else { 0 };
        Ok(Self {
            scaler,
            beta,
            g2,
            w_sum: 0.0,
            w_target: vec![0.0; m],
            sig2: vec![0.0; residual],
            wsig: vec![0.0; residual],
            spread: if own {
                TargetSpread::new(m)
            } else {
                TargetSpread::none()
            },
            zbuf: vec![0.0; k],
            rawbuf: vec![0.0; k],
            pbuf: crate::constraint::Scratch::default(),
            learned: Vec::new(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &SgdCfg {
        &self.cfg
    }

    /// Coefficients in the caller's units. With `standardize` the model
    /// fits on standardized inputs, so they are unscaled here and the intercept
    /// absorbs the shift.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let Some(sc) = &self.scaler else {
            return self.beta.clone();
        };
        unscaled(&self.beta, sc, self.cfg.fit_intercept)
    }

    /// Per-slot scale: the running sd for features, 1 for the intercept and for
    /// a feature with no spread yet.
    fn scales(&self) -> Vec<f64> {
        match &self.scaler {
            None => vec![1.0; self.cfg.k_total()],
            Some(sc) => scales_of(sc, usize::from(self.cfg.fit_intercept)),
        }
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Per target, the weight of the rows that carried it: what its
    /// `min_weight` is checked against.
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// `p = link(eta)`; the Poisson link is `exp(clamp(eta, −30, 30))` (the
    /// module docs).
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

    /// Per target, the EW variance of the out-of-sample residual, `s²`,
    /// under the Huber loss; empty under every other (the module docs).
    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Per target, the EW variance of the target itself, `s_y²`, under the
    /// epsilon-insensitive loss; empty under every other (the module docs).
    pub fn target_variance(&self) -> Vec<f64> {
        self.spread.vars()
    }

    /// The unit target `j`'s cut or tube is drawn in, as the row arrives:
    /// under Huber the residual's scale `s = √s²` once `s²` is finite and
    /// above 0; under epsilon-insensitive the target's own `s_y` once it has
    /// a spread. `None` before -- and under every other loss -- where there
    /// is no cut and no tube to draw (`huber`'s rule, `Robust`;
    /// [`crate::spread`]).
    fn scale(&self, j: usize) -> Option<f64> {
        match self.cfg.loss {
            SgdLoss::Huber { .. } => {
                let s2 = self.sig2[j];
                (s2 > 0.0 && s2.is_finite()).then(|| s2.sqrt())
            }
            SgdLoss::EpsilonInsensitive { .. } => {
                let unit = self.spread.band(j, 1.0);
                (unit > 0.0).then_some(unit)
            }
            _ => None,
        }
    }

    /// `dL/d(eta)` for the configured loss, with `scale` the unit its cut
    /// or tube is drawn in ([`Sgd::scale`]; `None` before it has one: no
    /// cut, and a tube of no width).
    fn dloss(&self, p: f64, y: f64, scale: Option<f64>) -> f64 {
        let r = p - y;
        match self.cfg.loss {
            SgdLoss::Squared | SgdLoss::Poisson | SgdLoss::Logistic => r,
            SgdLoss::Huber { delta } => match scale {
                Some(s) => r.clamp(-delta * s, delta * s),
                None => r,
            },
            SgdLoss::Quantile { tau } => f64::from(y < p) - tau,
            SgdLoss::EpsilonInsensitive { eps } => {
                if r.abs() <= scale.map_or(0.0, |s| eps * s) {
                    0.0
                } else {
                    r.signum()
                }
            }
        }
    }

    /// The label a row teaches target `j` with: `y` itself, except under
    /// the logistic loss, where a label outside {0, 1} is clamped into
    /// `[0, 1]`, or under `strict_binary` teaches nothing (`None`), `ftrl`'s
    /// rule (docs/PLAN.md task 195, S4).
    fn label(&self, y: f64) -> Option<f64> {
        match self.cfg.loss {
            SgdLoss::Logistic if self.cfg.strict_binary => (y == 0.0 || y == 1.0).then_some(y),
            SgdLoss::Logistic => Some(y.clamp(0.0, 1.0)),
            _ => Some(y),
        }
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
        }
        if self.rawbuf.len() != k {
            self.rawbuf = vec![0.0; k];
        }
    }
}

/// Which scales `loss` draws in: `(residual, own)`, the residual's spread
/// for the Huber cut and the target's own for the epsilon-insensitive tube
/// (the module docs). A state keeps each, one per target, exactly under
/// the loss that reads it.
fn scales_kept(loss: &SgdLoss) -> (bool, bool) {
    (
        matches!(loss, SgdLoss::Huber { .. }),
        matches!(loss, SgdLoss::EpsilonInsensitive { .. }),
    )
}

thread_local! {
    /// `predict`'s standardized row (see there).
    static PREDICT_Z: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The scaler's per-slot scale, `k` slots with `off` of them the
/// intercept: the running sd for a feature, 1 for the intercept and for a
/// feature with no spread yet; through the origin (`off = 0`) the root of
/// the raw second moment, the scale the step standardizes with (C13).
/// `pa` reads it too (docs/PLAN.md task 195, U2).
pub(crate) fn scales_of(sc: &EwDiag, off: usize) -> Vec<f64> {
    (0..sc.k())
        .map(|i| {
            if i < off {
                return 1.0;
            }
            if off == 0 {
                // No intercept: the raw second moment's scale, the
                // one the step standardizes with (C13).
                let raw = sc.raw(i);
                return if raw > 0.0 { raw.sqrt() } else { 1.0 };
            }
            let v = sc.var(i);
            if crate::variance_is_usable(v, sc.raw(i)) {
                v.sqrt()
            } else {
                1.0
            }
        })
        .collect()
}

/// Coefficients learned in the coordinates `sc` standardizes into, read
/// out in the caller's units with the moments as they stand: each slope
/// over its feature's scale, and the intercept absorbing the shift of the
/// means. `pa` reads them out the same way (docs/PLAN.md task 195, U2).
pub(crate) fn unscaled(beta: &[Vec<f64>], sc: &EwDiag, fit_intercept: bool) -> Vec<Vec<f64>> {
    let off = usize::from(fit_intercept);
    let scales = scales_of(sc, off);
    let k = scales.len();
    beta.iter()
        .map(|b| {
            let mut c = vec![0.0; k];
            for i in off..k {
                c[i] = b[i] / scales[i];
            }
            if fit_intercept {
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

/// `x` standardized against the moments of `sc` with the row itself
/// admitted at unit weight after a decay of `lam` ([`EwDiag::including`];
/// `SgdCfg::standardize` says why the row is in): feature `i` sits in
/// slot `off + i` of the scaler, and is divided by that sd, or by 1 while
/// it has no spread (the per-slot scale of [`scales_of`]). Lazy: `step`
/// writes it into the model's buffer and `predict` into the thread's, with
/// no vector of scales in between. `pa` standardizes with it too.
#[inline]
pub(crate) fn standardized<'a>(
    sc: &'a EwDiag,
    off: usize,
    x: &'a [f64],
    lam: f64,
) -> impl Iterator<Item = f64> + 'a {
    let inc = sc.including(lam);
    x.iter().enumerate().map(move |(i, &xi)| {
        let (mean, v, dev) = inc.moments(off + i, xi);
        if off == 0 {
            // No intercept to absorb a shift: scale by the raw second moment
            // and do not centre. Centring gave every prediction a hidden
            // intercept `-Σ b_i m_i / s_i` that `coefficients()` has no slot
            // for, and took a constraint's projection in coordinates shifted in
            // a way the bounds do not know about (review 2026-09-12, C13).
            let raw = v + mean * mean;
            return if raw > 0.0 { xi / raw.sqrt() } else { xi };
        }
        let scale = if crate::variance_is_usable(v, v + mean * mean) {
            v.sqrt()
        } else {
            1.0
        };
        dev / scale
    })
}

/// Partial sums the dot product is accumulated in. A single running sum is
/// a chain of dependent additions, three or four cycles each whatever the
/// core could do in parallel; eight independent ones run at the adder's
/// throughput instead, and were most of a wide step (docs/PERFORMANCE.md
/// §20). The order is fixed by this constant and the code below, not by
/// the hardware, so every platform sums the same bits.
const DOT_LANES: usize = 8;

/// `beta · [1, z]`: `z` against the feature coefficients in [`DOT_LANES`]
/// interleaved partial sums (slot `i` into sum `i % DOT_LANES`), the sums
/// folded pairwise, and the intercept's coefficient -- its `z` is the
/// constant 1 -- added last.
#[inline]
fn dot(beta: &[f64], off: usize, z: &[f64]) -> f64 {
    let b = &beta[off..];
    debug_assert_eq!(b.len(), z.len());
    let mut acc = [0.0f64; DOT_LANES];
    let (zc, zr) = z.as_chunks::<DOT_LANES>();
    let (bc, br) = b.as_chunks::<DOT_LANES>();
    for (zs, bs) in zc.iter().zip(bc) {
        for l in 0..DOT_LANES {
            acc[l] += zs[l] * bs[l];
        }
    }
    for (l, (zi, bi)) in zr.iter().zip(br).enumerate() {
        acc[l] += zi * bi;
    }
    let s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    if off == 1 { beta[0] + s } else { s }
}

impl OnlineModel for Sgd {
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
        // (`OnlineModel`): a feature that is not a number made every
        // coefficient NaN (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.fit_intercept);
        let lam = self.cfg.decay.factor(d_clock);

        // The row the model sees. Without a scaler it is `x` itself, the
        // intercept's constant 1 folded into `dot` and the update below:
        // copying `[1, x]` in front of every row was a tenth of a wide step
        // (docs/PERFORMANCE.md §20). With one, `[1, z]` is standardized
        // against the moments with this row admitted -- read off the scaler
        // as it stands, the way `predict` reads them (E24, and
        // `SgdCfg::standardize` for the order); the raw `[1, x]` is kept
        // to update the scaler afterwards, at the row's actual weight.
        if self.scaler.is_some() {
            self.ensure_buffers();
            let Self {
                scaler,
                zbuf,
                rawbuf,
                ..
            } = self;
            let sc = scaler.as_ref().expect("checked above");
            if off == 1 {
                zbuf[0] = 1.0;
                rawbuf[0] = 1.0;
            }
            rawbuf[off..].copy_from_slice(x);
            for (z, v) in zbuf[off..].iter_mut().zip(standardized(sc, off, x, lam)) {
                *z = v;
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
        // on (see `EwRidgeCfg::min_weight`). Decaying it here would make
        // `min_weight` mean a slightly different number of rows for `sgd`
        // than for `ewridge`, which is exactly the kind of quiet divergence
        // the cross-model semantics suite exists to catch. Each target's
        // `min_weight` reads its own weight, the rows that carried it.
        let n_eff = self.w_sum;

        let mut pred = vec![f64::NAN; m];
        if self.cfg.constraint.is_some() {
            // Skipped by serde, so sized here rather than in `new`.
            self.learned.clear();
            self.learned.resize(m, false);
        }
        // The rate every slot shares this row, or `None` under AdaGrad,
        // whose rate is per slot. `inv_scaling`'s power is one `powf` a
        // row, not one a slot.
        let lr_row = match self.cfg.schedule {
            LearningRate::Constant => Some(self.cfg.learning_rate),
            LearningRate::InvScaling { power } => {
                Some(self.cfg.learning_rate / (1.0 + n_eff).powf(power))
            }
            LearningRate::AdaGrad => None,
        };
        let (lr0, l2, clip) = (self.cfg.learning_rate, self.cfg.l2, self.cfg.clip_gradient);
        let z: &[f64] = if self.scaler.is_some() {
            &self.zbuf[off..]
        } else {
            x
        };
        for j in 0..m {
            let eta = dot(&self.beta[j], off, z);
            let p = self.link(eta);
            if self.w_target[j] >= self.cfg.min_weight {
                pred[j] = p;
            }
            // The cut and the tube are drawn in their scale as the row
            // arrives (the module docs). Under Huber, `s²`'s weight ages on
            // every row, as `huber`'s does, whatever the row then teaches;
            // under epsilon-insensitive the target's spread does, and
            // learns the row's `y` at once, whatever the fit.
            let scale = self.scale(j);
            if let Some(ws) = self.wsig.get_mut(j) {
                *ws *= lam;
            }
            if !self.spread.is_empty() {
                self.spread.update(j, y[j], lam, weight);
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 || !yj.is_finite() {
                continue;
            }
            let Some(yj) = self.label(yj) else { continue };
            let d = self.dloss(p, yj, scale);
            // The row's own residual joins `s²` afterwards, from the
            // prediction the row reports, as `huber`'s does: a row with no
            // prediction (`min_weight` unmet) adds nothing.
            if !self.sig2.is_empty() && pred[j].is_finite() {
                let resid = yj - p;
                let ws_new = self.wsig[j] + weight;
                let s2 = (self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                // Skipped where it would not be finite, as `huber`'s is: an
                // `inf` scale would make every cut and tube infinite for good.
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
            // Per slot: `g = d * z_i * w`, plus `l2 * beta_i` off the
            // intercept, clipped; `beta_i -= lr * g`. The intercept's `z`
            // is 1 (`d * 1 * w` is `d * w` exactly), and it is not
            // penalised.
            let beta = &mut self.beta[j];
            // A row any of whose gradients is not a finite number teaches
            // nothing, and is skipped before `beta` moves, as a row with no
            // weight is; under AdaGrad so is one whose squared gradient is
            // not, which `g2` would keep for good (decay never takes an
            // `inf` away; `ftrl`'s guard). With `clip_gradient = inf`, which
            // disables the clip, two rows inside the input bound made a
            // coefficient infinite and every later prediction null, where
            // `pa`, `ftrl` and `kalman` each skip such a row (review round
            // 4, CC2). The gradients are formed here as the update forms
            // them, so a row that passes moves `beta` to the same bits.
            let usable = |g: f64| {
                if lr_row.is_some() {
                    g.is_finite()
                } else {
                    (g * g).is_finite()
                }
            };
            if (off == 1 && !usable((d * 1.0 * weight).clamp(-clip, clip)))
                || beta[off..]
                    .iter()
                    .zip(z)
                    .any(|(b, &zi)| !usable((d * zi * weight + l2 * *b).clamp(-clip, clip)))
            {
                continue;
            }
            match lr_row {
                Some(lr) => {
                    if off == 1 {
                        let g = (d * 1.0 * weight).clamp(-clip, clip);
                        beta[0] -= lr * g;
                    }
                    for (b, &zi) in beta[off..].iter_mut().zip(z) {
                        let g = (d * zi * weight + l2 * *b).clamp(-clip, clip);
                        *b -= lr * g;
                    }
                }
                None => {
                    let g2 = &mut self.g2[j];
                    if off == 1 {
                        let g = (d * 1.0 * weight).clamp(-clip, clip);
                        g2[0] += g * g;
                        beta[0] -= lr0 / (g2[0].sqrt() + 1e-8) * g;
                    }
                    for ((b, g2), &zi) in beta[off..].iter_mut().zip(&mut g2[off..]).zip(z) {
                        let g = (d * zi * weight + l2 * *b).clamp(-clip, clip);
                        *g2 += g * g;
                        *b -= lr0 / (g2.sqrt() + 1e-8) * g;
                    }
                }
            }
            if let Some(l) = self.learned.get_mut(j) {
                *l = true;
            }
        }
        if let Some(sc) = &mut self.scaler {
            sc.update(&self.rawbuf, lam, weight);
        }
        // Project after the scaler moved: the bounds live in the caller's
        // units, and in standardized coordinates they move with the scales,
        // so every target is re-projected when the scales changed (a
        // positive weight), otherwise only the ones this row stepped.
        if self.cfg.constraint.is_some() {
            let scales = self.scaler.is_some().then(|| self.scales());
            let rescaled = self.scaler.is_some() && weight > 0.0;
            let Self {
                cfg,
                beta,
                pbuf,
                learned,
                ..
            } = self;
            let c = cfg.constraint.as_ref().expect("checked above");
            for (j, b) in beta.iter_mut().enumerate() {
                if rescaled || learned[j] {
                    c.project(&mut b[off..], scales.as_deref().map(|s| &s[off..]), pbuf);
                }
            }
        }
        self.w_sum = lam * self.w_sum + weight;
        // A label `strict_binary` refuses carries no weight for its target,
        // as in `ftrl`: it taught nothing.
        let strict = self.cfg.strict_binary;
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

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let n_eff = self.w_sum;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        if self.w_target.iter().any(|w| *w >= self.cfg.min_weight) {
            let off = usize::from(self.cfg.fit_intercept);
            let lam = self.cfg.decay.factor(d_clock);
            let predict_with = |z: &[f64], pred: &mut [f64]| {
                for (j, p) in pred.iter_mut().enumerate() {
                    if self.w_target[j] >= self.cfg.min_weight {
                        *p = self.link(dot(&self.beta[j], off, z));
                    }
                }
            };
            match &self.scaler {
                None => predict_with(x, &mut pred),
                // `&self`, so the standardized row goes into a buffer of
                // the thread's rather than one of the model's, and not a
                // fresh vector per row.
                Some(sc) => PREDICT_Z.with_borrow_mut(|z| {
                    z.clear();
                    z.extend(standardized(sc, off, x, lam));
                    predict_with(z, &mut pred);
                }),
            }
        }
        Step {
            pred,
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
                // The shapes are checked as the state is read (`SgdV3`).
                crate::model::check_cfg("sgd", m.cfg.validate())?;
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
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss,
            learning_rate: 0.05,
            schedule: LearningRate::Constant,
            l2: 0.0,
            min_weight: 5.0,
            clip_gradient: 1e12,
            constraint: None,
            standardize: false,
            strict_binary: false,
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
        // NaN passed every `<= 0.0` / `< 0.0` test here: a NaN clip then
        // panicked in `f64::clamp` on the first learned row, a NaN `l2`
        // poisoned every coefficient, a NaN `eps` emptied the tube and a NaN
        // `power` made every rate NaN (review 2026-09-18, B4).
        bad(&|c| c.clip_gradient = f64::NAN, "clip_gradient must be > 0");
        bad(&|c| c.l2 = f64::NAN, "l2 must be >= 0");
        bad(
            &|c| c.loss = SgdLoss::EpsilonInsensitive { eps: f64::NAN },
            "eps must be >= 0",
        );
        bad(
            &|c| c.schedule = LearningRate::InvScaling { power: f64::NAN },
            "power must be >= 0",
        );
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
        let one = Some(1.0);
        assert_eq!(m(SgdLoss::Squared).dloss(3.0, 1.0, one), 2.0);
        assert_eq!(m(SgdLoss::Squared).dloss(1.0, 3.0, one), -2.0);

        // Huber: the residual, clipped symmetrically at delta residual
        // stds, and not at all before the target has a scale.
        let h = m(SgdLoss::Huber { delta: 1.5 });
        assert_eq!(h.dloss(1.0, 0.0, one), 1.0, "inside the band: squared");
        assert_eq!(h.dloss(9.0, 0.0, one), 1.5, "outside: clipped");
        assert_eq!(h.dloss(-9.0, 0.0, one), -1.5);
        assert_eq!(h.dloss(9.0, 0.0, Some(2.0)), 3.0, "delta stds of 2");
        assert_eq!(h.dloss(9.0, 0.0, None), 9.0, "no scale: no cut");

        // Quantile: a constant gradient whose sign depends on which side of
        // the prediction the target fell, asymmetric except at tau = 0.5.
        let q = m(SgdLoss::Quantile { tau: 0.9 });
        assert_eq!(q.dloss(5.0, 1.0, one), 1.0 - 0.9, "over-predicted");
        assert_eq!(q.dloss(1.0, 5.0, one), -0.9, "under-predicted");
        let med = m(SgdLoss::Quantile { tau: 0.5 });
        assert_eq!(
            med.dloss(5.0, 1.0, one),
            -med.dloss(1.0, 5.0, one),
            "symmetric at 0.5"
        );

        // Epsilon-insensitive: exactly zero inside the tube of eps of the
        // target's std, and a tube of no width before the target has one.
        let e = m(SgdLoss::EpsilonInsensitive { eps: 1.0 });
        assert_eq!(e.dloss(0.5, 0.0, one), 0.0);
        assert_eq!(e.dloss(1.0, 0.0, one), 0.0, "the boundary is inside");
        assert!(e.dloss(3.0, 0.0, one) > 0.0);
        assert!(e.dloss(-3.0, 0.0, one) < 0.0);
        assert_eq!(e.dloss(3.0, 0.0, Some(4.0)), 0.0, "a tube of 4");
        assert_eq!(e.dloss(0.5, 0.0, None), 1.0, "no scale: no tube");
        assert_eq!(e.dloss(0.0, 0.0, None), 0.0, "on the target");

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
        // Where the module docs put the clamp: `exp(clamp(eta, ±30))`, so a
        // linear predictor past 30 either way predicts as 30 does (review
        // 2026-10-05, CC6: the table said `exp(eta)`).
        assert_eq!(po.link(31.0), po.link(30.0));
        assert_eq!(po.link(-31.0), po.link(-30.0));
        assert_eq!(po.link(30.0), 30.0f64.exp());
        assert!(po.link(29.0) < po.link(30.0), "inside the clamp it is exp");
    }

    /// The case scaling exists for: one feature in thousands, one in
    /// thousandths. A single learning rate cannot suit both.
    #[test]
    fn scaling_rescues_badly_scaled_features() {
        let run = |scale: bool| {
            let mut c = cfg(2, SgdLoss::Squared);
            c.standardize = scale;
            c.learning_rate = 0.01;
            c.min_weight = 0.0;
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
        c.standardize = true;
        c.learning_rate = 0.1;
        c.min_weight = 0.0;
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
    fn scaling_admits_the_row_it_scales() {
        // The row is standardized against moments that include it (docs/PLAN.md
        // task 74). The first row of a stream is then the whole history: its
        // feature standardizes to 0, so only the intercept learns from it, and
        // an enormous first row cannot throw the slope.
        let mut c = cfg(1, SgdLoss::Squared);
        c.standardize = true;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c.clone()).unwrap();
        let step = m.step(&[1e6], &[Some(1.0)], 0.0, 1.0);
        assert_eq!(step.pred[0], 0.0, "an empty fit predicts 0");
        let coef = &m.coefficients()[0];
        assert_eq!(
            coef[1], 0.0,
            "the first row is z = 0: the slope cannot move"
        );
        assert_eq!(coef[0], 0.05, "the intercept learned from it: lr * (1 - 0)");

        // With the row inside the moments a standardized value is bounded by
        // sqrt(lam * W): z = a d / sqrt(a c + a b d^2) <= sqrt(a / b), with
        // equality when the history has no spread (c = 0). So a row a million
        // deviations out predicts within |b0| + |b1| sqrt(W) of zero, where
        // against the moments from *before* the row -- the order this
        // replaced -- the same row predicted about 1.7e6.
        let mut m = Sgd::new(c).unwrap();
        for i in 0..10 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            m.step(&[x], &[Some(0.5 * x)], f64::from(u8::from(i > 0)), 1.0);
        }
        let b = m.beta[0].clone(); // in standardized coordinates
        let w = m.scaler.as_ref().unwrap().n_eff();
        assert_eq!(w, 10.0);
        let step = m.step(&[1e6], &[Some(0.0)], 1.0, 1.0);
        let bound = b[0].abs() + b[1].abs() * w.sqrt();
        assert!(
            step.pred[0].abs() <= bound,
            "pred {} exceeds the bound {bound} of a standardized row",
            step.pred[0]
        );
        assert!(
            step.pred[0].abs() > 0.5 * bound,
            "the row is at the bound's edge"
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
        c.min_weight = 0.0;
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
            if row.is_multiple_of(20) {
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
        // property of this loss rather than a bug. The band is the slope's
        // range over the last 2000 of 30000 rows: a final value alone is one
        // draw from it. The tube, `eps = 0.02` of the target's std of about
        // 1.15, is narrower than the noise of up to 0.1, so most rows step;
        // at `eps = 0.2` it held every residual once the slope reached 1.87
        // under the constant rate and 1.89 under the annealed one, and both
        // fits stopped there (docs/PLAN.md task 202).
        let band = |c: SgdCfg| {
            let mut m = Sgd::new(c).unwrap();
            let mut s = 19u64;
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..30000 {
                let x = [lcg(&mut s)];
                let y = 2.0 * x[0] + 0.1 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                if i >= 28000 {
                    let b = m.coefficients()[0][1];
                    (lo, hi) = (lo.min(b), hi.max(b));
                }
            }
            (lo, hi)
        };
        let mut constant = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.02 });
        constant.learning_rate = 0.01;
        let (c_lo, c_hi) = band(constant);

        let mut annealed = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.02 });
        annealed.learning_rate = 0.5;
        annealed.schedule = LearningRate::InvScaling { power: 0.5 };
        let (a_lo, a_hi) = band(annealed);

        assert!(
            a_hi - a_lo < 0.75 * (c_hi - c_lo),
            "annealed band [{a_lo}, {a_hi}] should be narrower than constant [{c_lo}, {c_hi}]"
        );
        assert!(
            (a_lo - 2.0).abs() < 0.05 && (a_hi - 2.0).abs() < 0.05,
            "annealed band [{a_lo}, {a_hi}] (truth 2.0)"
        );
    }

    /// The tube does not shrink as the fit improves, so a fit stops wherever
    /// every residual is inside it, and the builder's default must sit
    /// inside a good fit's errors (docs/PLAN.md task 203). `y = 2x` plus
    /// noise of up to 0.01, `x` in `[-1, 1]`, so the target's spread is
    /// about `2/√3 = 1.155`: a fit with every residual inside `eps` of it
    /// is off by `|b0| + |b1 − 2| ≤ 1.155·eps − 0.01` at most. At 0.01 that
    /// is 0.0016, and an annealed rate ends within 0.005 (the scaler still
    /// moves the coefficients a little); at 0.1 it is 0.105, and the fit
    /// stopped more than 0.01 off.
    #[test]
    fn a_tube_inside_a_good_fits_errors_reaches_the_slope() {
        let off = |eps: f64| {
            let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps });
            cf.learning_rate = 0.5;
            cf.schedule = LearningRate::InvScaling { power: 0.5 };
            cf.standardize = true;
            cf.clip_gradient = 1e3;
            cf.min_weight = 2.0;
            let mut m = Sgd::new(cf).unwrap();
            let mut s = 19u64;
            for i in 0..20_000 {
                let x = [lcg(&mut s)];
                let y = 2.0 * x[0] + 0.01 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let b = &m.coefficients()[0];
            b[0].abs() + (b[1] - 2.0).abs()
        };
        let (inside, outside) = (off(0.01), off(0.1));
        assert!(inside < 0.005, "eps 0.01: {inside}");
        assert!(outside > 0.01, "eps 0.1: {outside}");
    }

    /// A residual inside the tube, `eps` of the target's own std wide,
    /// leaves the fit alone, and one outside it steps: the "insensitive" of
    /// the name. The first two rows, 1 and 3, have no spread to draw a tube
    /// in, so each steps whatever its residual; they give the target a
    /// spread of 1, the unit the next rows are judged in (docs/PLAN.md task
    /// 202). The Huber scale is not kept under this loss.
    #[test]
    fn epsilon_insensitive_ignores_residuals_inside_the_tube() {
        let mut c = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.8 });
        c.learning_rate = 0.01;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c).unwrap();
        m.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
        assert_eq!(m.coefficients()[0], vec![0.01, 0.01], "no spread: a step");
        assert_eq!(m.target_variance(), vec![0.0]);
        m.step(&[1.0], &[Some(3.0)], 1.0, 1.0);
        let second = m.coefficients()[0].clone();
        assert_eq!(second, vec![0.02, 0.02], "one row has no spread: a step");
        assert_eq!(m.target_variance(), vec![1.0]);
        assert!(m.sigma2().is_empty(), "no residual scale under this loss");
        // The prediction is now 0.04: a residual of 0.5 is inside 0.8 · 1.
        m.step(&[1.0], &[Some(0.54)], 1.0, 1.0);
        assert_eq!(m.coefficients()[0], second, "inside the tube");
        // The third row joined the spread; a residual just past the tube
        // it draws now steps.
        let tube = 0.8 * m.target_variance()[0].sqrt();
        m.step(&[1.0], &[Some(0.04 + tube + 0.01)], 1.0, 1.0);
        assert_ne!(m.coefficients()[0], second, "outside the tube");
    }

    /// Each scale a loss draws in is kept exactly under that loss, one per
    /// target: the residual's under Huber, the target's own under
    /// epsilon-insensitive, neither under the rest. A state carrying one
    /// its loss does not read, or missing one it does, is refused as it is
    /// read (docs/PLAN.md task 202).
    #[test]
    fn a_state_is_refused_when_its_scales_disagree_with_its_loss() {
        let json = |loss: SgdLoss| {
            let mut c = cfg(2, loss);
            c.decay = Decay::Halflife(50.0); // JSON has no `inf`
            c.n_targets = 2;
            serde_json::to_value(Sgd::new(c).unwrap()).unwrap()
        };
        let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone());
        let huber = json(SgdLoss::Huber { delta: 1.345 });
        let tube = json(SgdLoss::EpsilonInsensitive { eps: 0.1 });
        let squared = json(SgdLoss::Squared);
        for v in [&huber, &tube, &squared] {
            assert!(loads(v).is_ok(), "the control");
        }
        assert_eq!(huber["sig2"].as_array().unwrap().len(), 2);
        assert_eq!(tube["spread"].as_array().unwrap().len(), 2);
        for field in ["sig2", "wsig", "spread"] {
            assert!(squared[field].as_array().unwrap().is_empty(), "{field}");
        }
        assert!(huber["spread"].as_array().unwrap().is_empty());
        assert!(tube["sig2"].as_array().unwrap().is_empty());

        let mut lost = huber.clone();
        lost["sig2"] = serde_json::json!([]);
        let err = loads(&lost).unwrap_err().to_string();
        assert!(err.contains("residual variances do not match"), "{err}");
        let mut stray = squared.clone();
        stray["wsig"] = huber["wsig"].clone();
        let err = loads(&stray).unwrap_err().to_string();
        assert!(err.contains("residual variances do not match"), "{err}");

        let mut lost = tube.clone();
        lost["spread"].as_array_mut().unwrap().pop();
        let err = loads(&lost).unwrap_err().to_string();
        assert!(err.contains("target spreads do not match"), "{err}");
        let mut stray = huber.clone();
        stray["spread"] = tube["spread"].clone();
        let err = loads(&stray).unwrap_err().to_string();
        assert!(err.contains("target spreads do not match"), "{err}");
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
            c.min_weight = 0.0;
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

    /// A row any of whose gradients is not a finite number -- or, under
    /// AdaGrad, one whose square is not, which `g2` never decays away --
    /// teaches nothing, before `beta` moves, as `pa`, `ftrl` and `kalman`
    /// skip theirs: with `clip_gradient = inf` ("disables it"), two rows
    /// inside the input bound made a coefficient infinite and every later
    /// prediction null (review round 4, CC2), and under AdaGrad one froze
    /// the slope where it started, for good. The stream after them learns
    /// the line.
    #[test]
    fn a_row_whose_gradient_overflows_teaches_nothing() {
        for schedule in [LearningRate::Constant, LearningRate::AdaGrad] {
            let mut c = cfg(1, SgdLoss::Squared);
            c.clip_gradient = f64::INFINITY;
            c.min_weight = 0.0;
            c.schedule = schedule;
            c.learning_rate = 0.2;
            let mut m = Sgd::new(c).unwrap();
            m.step(&[1e100], &[Some(1e100)], 0.0, 1.0);
            let before = m.coefficients()[0].clone();
            m.step(&[1e100], &[Some(0.0)], 1.0, 1.0);
            assert_eq!(
                m.coefficients()[0],
                before,
                "{schedule:?}: a row whose gradient overflows moved the fit"
            );
            let mut s = 37u64;
            let mut last = f64::NAN;
            for i in 0..20_000 {
                let x = [lcg(&mut s)];
                let p = m.step(&x, &[Some(1.0 + 0.5 * x[0])], 1.0, 1.0).pred[0];
                assert!(p.is_finite(), "{schedule:?}, row {i}: {p}");
                last = p - (1.0 + 0.5 * x[0]);
            }
            let beta = m.coefficients()[0].clone();
            assert!(
                (beta[0] - 1.0).abs() < 0.01 && (beta[1] - 0.5).abs() < 0.01,
                "{schedule:?}: {beta:?}, last error {last}"
            );
        }
    }

    #[test]
    fn clip_gradient_bounds_the_step() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.clip_gradient = 1.0;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c).unwrap();
        // one absurd row: with lr 0.05 and the clip at 1, |db| <= 0.05
        m.step(&[1.0], &[Some(1e9)], 0.0, 1.0);
        assert!(m.coefficients()[0][1].abs() <= 0.05 + 1e-12);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut c = cfg(1, SgdLoss::Squared);
        c.min_weight = 0.0;
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

    fn constrained(k: usize, lo: f64, hi: f64, sum: Option<f64>) -> SgdCfg {
        let mut c = cfg(k, SgdLoss::Squared);
        c.constraint = Some(Constraint {
            lo: vec![lo; k],
            hi: vec![hi; k],
            sum,
        });
        c
    }

    #[test]
    fn a_constraint_is_validated_by_name() {
        let mut c = constrained(2, 0.0, 1.0, Some(3.0));
        assert_eq!(
            c.validate().unwrap_err(),
            "sgd: coef_sum = 3 is outside what the bounds allow, [0, 2]"
        );
        c.constraint.as_mut().unwrap().lo = vec![0.0];
        assert!(
            c.validate()
                .unwrap_err()
                .contains("coef_min lists 1 bounds")
        );
    }

    #[test]
    fn a_simplex_starts_uniform_and_stays_on_it() {
        // Truth on the simplex: the fit lands on it, and every intermediate
        // coefficient vector is on it too (sum exact to rounding, slopes >= 0).
        let mut m = Sgd::new(constrained(3, 0.0, f64::INFINITY, Some(1.0))).unwrap();
        let start = &m.coefficients()[0];
        assert_eq!(start[0], 0.0);
        for b in &start[1..] {
            assert!((b - 1.0 / 3.0).abs() <= 1e-15);
        }
        let mut s = 11u64;
        for i in 0..20000 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 0.2 * x[0] + 0.5 * x[1] + 0.3 * x[2] + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let b = &m.coefficients()[0][1..];
            assert!(b.iter().all(|v| *v >= 0.0), "row {i}: {b:?}");
            assert!(
                (b.iter().sum::<f64>() - 1.0).abs() <= 1e-12,
                "row {i}: {b:?}"
            );
        }
        let b = &m.coefficients()[0];
        for (got, want) in b[1..].iter().zip([0.2, 0.5, 0.3]) {
            assert!((got - want).abs() < 0.03, "{b:?}");
        }
    }

    #[test]
    fn a_box_holds_where_the_truth_lies_outside_it() {
        // Uncorrelated unit-scale features: the constrained least-squares
        // optimum is the truth clamped into the box.
        let mut c = constrained(2, 0.0, 0.5, None);
        c.learning_rate = 0.02;
        let b = fit(c, 20000, 3, |x, s| -0.4 * x[0] + 0.9 * x[1] + 0.1 * lcg(s));
        assert_eq!(b[1], 0.0, "{b:?}");
        assert_eq!(b[2], 0.5, "{b:?}");
        assert!(b[0].abs() < 0.05, "{b:?}");
    }

    #[test]
    fn a_pinned_slope_never_moves_and_the_rest_learn() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.constraint = Some(Constraint {
            lo: vec![0.7, f64::NEG_INFINITY],
            hi: vec![0.7, f64::INFINITY],
            sum: None,
        });
        let b = fit(c, 20000, 5, |x, _| 1.5 * x[0] - 0.5 * x[1] + 0.25);
        assert_eq!(b[1], 0.7);
        assert!((b[2] + 0.5).abs() < 0.05, "{b:?}");
    }

    #[test]
    fn the_constraint_holds_in_the_callers_units_under_scaling() {
        // The bound is on c_i = b_i / scale_i; the projection is taken with
        // the scales the coefficients are reported with, after every row.
        let mut c = constrained(2, 0.0, 0.01, Some(0.01));
        c.standardize = true;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 17u64;
        for i in 0..5000 {
            let x = [1000.0 * lcg(&mut s), 0.001 * lcg(&mut s)];
            let y = 0.002 * x[0] + 900.0 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let b = &m.coefficients()[0][1..];
            assert!(
                b.iter().all(|v| *v >= -1e-15 && *v <= 0.01 + 1e-15),
                "row {i}: {b:?}"
            );
            assert!(
                (b.iter().sum::<f64>() - 0.01).abs() <= 1e-12,
                "row {i}: {b:?}"
            );
        }
        // The truth (0.002, 900) has one slope inside the box and one far
        // outside. The fit keeps the first at its truth and gives the second
        // whatever the sum leaves: in the standardized metric that is the
        // nearest feasible point, since a unit of c_1 is 10^6 times cheaper
        // than a unit of c_0.
        let b = &m.coefficients()[0];
        assert!((b[1] - 0.002).abs() <= 2e-4, "{b:?}");
        assert!((b[2] - 0.008).abs() <= 2e-4, "{b:?}");
    }

    #[test]
    fn a_zero_weight_row_leaves_a_constrained_fit_alone() {
        let mut m = Sgd::new(constrained(2, 0.0, f64::INFINITY, Some(1.0))).unwrap();
        let mut s = 23u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.coefficients().to_vec();
        m.step(&[0.3, -0.2], &[Some(5.0)], 1.0, 0.0);
        m.step(&[0.3, -0.2], &[None], 1.0, 1.0);
        assert_eq!(m.coefficients(), &before[..]);
    }

    #[test]
    fn a_constrained_state_roundtrips() {
        let mut m1 = Sgd::new(constrained(2, 0.0, f64::INFINITY, Some(1.0))).unwrap();
        let mut s = 29u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m1.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(m2.cfg().constraint, m1.cfg().constraint);
        for _ in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            assert_eq!(
                m1.step(&x, &[Some(x[0])], 1.0, 1.0).pred,
                m2.step(&x, &[Some(x[0])], 1.0, 1.0).pred
            );
            assert_eq!(m1.coefficients(), m2.coefficients());
        }
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

    /// A target's `min_weight` counts only the rows that carried it, and
    /// `n_eff` stays every row's (hard rule 8, docs/PLAN.md task 115 (d)):
    /// ten rows with no target met `min_weight = 3` with every coefficient
    /// at zero. A zero-weight row with the target (row 12) only ages it; the
    /// scaler, which a null target does not stop, is on.
    #[test]
    fn min_periods_counts_only_the_rows_that_carried_the_target() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.min_weight = 3.0;
        c.standardize = true;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..16usize {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i >= 10).then(|| x[0] - x[1]);
            let out = m.step(&x, &[y], 1.0, if i == 12 { 0.0 } else { 1.0 });
            assert_eq!(out.n_eff, (i - usize::from(i > 12)) as f64, "row {i}");
            assert_eq!(out.pred[0].is_nan(), i < 14, "row {i}");
            assert_eq!(m.predict(&x, 1.0).pred[0].is_nan(), i < 13, "row {i}");
        }
        assert_eq!(m.target_weights(), &[5.0]);
    }

    /// A state without target weights is refused, where it loaded with each
    /// target at the shared weight -- the layout of schema 19, which this
    /// build refuses by its version, and the repair a damaged file reached
    /// (docs/PLAN.md task 198).
    #[test]
    fn a_state_without_target_weights_is_refused() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.decay = Decay::Halflife(50.0); // JSON has no `inf`
        let mut m = Sgd::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..8 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[(i >= 5).then(|| x[0])], 1.0, 1.0);
        }
        assert!(m.target_weights()[0] < m.n_eff());
        let mut v = serde_json::to_value(&m).unwrap();
        assert!(v.as_object_mut().unwrap().remove("w_target").is_some());
        let err = serde_json::from_value::<Sgd>(v).unwrap_err().to_string();
        assert!(err.contains("missing field `w_target`"), "{err}");
        let mut short = serde_json::to_value(&m).unwrap();
        short["w_target"] = serde_json::json!([]);
        let err = serde_json::from_value::<Sgd>(short)
            .unwrap_err()
            .to_string();
        assert!(err.contains("target weights have the wrong shape"), "{err}");
    }

    /// Without `standardize` there is no scaler in either schema, and a
    /// file written before the field existed (no `scaler` key at all) still
    /// loads as one without.
    #[test]
    fn a_state_without_a_scaler_loads_in_every_layout() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.decay = Decay::Halflife(50.0); // JSON has no `inf`
        let m = Sgd::new(c).unwrap();
        let mut v = serde_json::to_value(&m).unwrap();
        assert!(v["scaler"].is_null());
        let back: Sgd = serde_json::from_value(v.clone()).unwrap();
        assert!(back.scaler.is_none());
        v.as_object_mut().unwrap().remove("scaler");
        let back: Sgd = serde_json::from_value(v).unwrap();
        assert!(back.scaler.is_none());
    }

    /// A state whose cfg says `standardize` carries its scaler, and one
    /// on AdaGrad its sums, one per target -- and neither carries what its
    /// cfg does not ask for. A file that lost the scaler loaded as a model
    /// reading raw inputs with coefficients learned on standardized ones;
    /// one that lost its sums panicked on its first AdaGrad step
    /// (review 2026-09-12, S16).
    #[test]
    fn a_state_is_refused_when_its_scaler_or_sums_disagree_with_its_cfg() {
        let json = |c: &SgdCfg| serde_json::to_value(Sgd::new(c.clone()).unwrap()).unwrap();
        let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone()).is_ok();
        let mut plain = cfg(2, SgdLoss::Squared);
        plain.decay = Decay::Halflife(50.0); // JSON has no `inf`

        let mut scaled = plain.clone();
        scaled.standardize = true;
        let with = json(&scaled);
        assert!(loads(&with), "the control");
        let mut lost = with.clone();
        lost.as_object_mut().unwrap().remove("scaler");
        assert!(!loads(&lost), "standardize without its scaler");
        let mut stray = json(&plain);
        stray["scaler"] = with["scaler"].clone();
        assert!(!loads(&stray), "a scaler the cfg does not ask for");

        let mut ada = plain.clone();
        ada.schedule = LearningRate::AdaGrad;
        let with = json(&ada);
        assert!(loads(&with), "the control");
        let mut lost = with.clone();
        lost["g2"] = serde_json::json!([]);
        assert!(!loads(&lost), "AdaGrad without its sums");
        let mut stray = json(&plain);
        stray["g2"] = with["g2"].clone();
        assert!(!loads(&stray), "sums a constant rate does not keep");
    }

    /// The named msgpack of `value` with `edit` applied, read back: the path
    /// a saved state takes into the model, whose check is in its `TryFrom`.
    fn reread<T: serde::Serialize + serde::de::DeserializeOwned>(
        value: &T,
        edit: impl FnOnce(&mut rmpv::Value),
    ) -> Result<T, String> {
        let bytes = rmp_serde::to_vec_named(value).unwrap();
        let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        edit(&mut v);
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        rmp_serde::from_slice(&out).map_err(|e| e.to_string())
    }

    /// The first coefficient vector of the state, one entry short.
    fn shorten_first(v: &mut rmpv::Value, field: &str) {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map")
        };
        let (_, x) = entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(field))
            .unwrap_or_else(|| panic!("no {field}"));
        let rmpv::Value::Array(rows) = x else {
            panic!("{field} is a list")
        };
        let rmpv::Value::Array(first) = &mut rows[0] else {
            panic!("{field}[0] is a list")
        };
        first.pop();
    }

    /// A state whose coefficients are not its cfg's shape is refused as it is
    /// read, which is how a saved state reaches `restore`, where it loaded and
    /// panicked on the first step (review 2026-09-18, B3; docs/PLAN.md task
    /// 111: every other model had this test).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        let m = Sgd::new(cfg(2, SgdLoss::Squared)).unwrap();
        assert!(reread(&m, |_| {}).is_ok(), "the control");
        let err = reread(&m, |v| shorten_first(v, "beta")).unwrap_err();
        assert!(err.contains("wrong shape"), "{err}");
    }

    /// Each shape check refuses on its own: a state one target short of its
    /// cfg, and one whose scaler is another width than the row, are each
    /// refused with every other part of the state in shape (task 158).
    #[test]
    fn a_state_missing_a_target_or_with_a_scaler_of_another_width_is_refused() {
        let json = |c: &SgdCfg| serde_json::to_value(Sgd::new(c.clone()).unwrap()).unwrap();
        let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone());
        let mut two = cfg(2, SgdLoss::Squared);
        two.decay = Decay::Halflife(50.0); // JSON has no `inf`
        two.n_targets = 2;
        let mut v = json(&two);
        assert!(loads(&v).is_ok(), "the control");
        v["beta"].as_array_mut().unwrap().pop();
        let err = loads(&v).unwrap_err().to_string();
        assert!(err.contains("sgd: state has the wrong shape"), "{err}");

        let mut scaled = two.clone();
        scaled.standardize = true;
        let mut wider = scaled.clone();
        wider.n_features = 3;
        let mut v = json(&scaled);
        assert!(loads(&v).is_ok(), "the control");
        v["scaler"] = json(&wider)["scaler"].clone();
        let err = loads(&v).unwrap_err().to_string();
        assert!(err.contains("sgd: state has the wrong shape"), "{err}");
    }

    /// `dot` is `beta · [1, z]` whatever the lanes it sums in: 3, 8, 11 and
    /// 19 features, every slot nonzero, against the plain sum (task 158).
    #[test]
    fn dot_is_the_plain_sum_in_every_lane() {
        let mut s = 71u64;
        for k in [3usize, 5, 7, 8, 11, 19] {
            for off in [0usize, 1] {
                let beta: Vec<f64> = (0..k + off).map(|_| 0.5 + lcg(&mut s)).collect();
                let z: Vec<f64> = (0..k).map(|_| 1.5 + lcg(&mut s)).collect();
                let plain: f64 = (0..k).map(|i| beta[off + i] * z[i]).sum::<f64>()
                    + if off == 1 { beta[0] } else { 0.0 };
                let got = dot(&beta, off, &z);
                assert!(
                    (got - plain).abs() <= 1e-13 * plain.abs().max(1.0),
                    "k {k}, off {off}: {got} against {plain}"
                );
            }
        }
    }

    /// The logistic link is `1 / (1 + e^-eta)` on both sides of 0, and the
    /// quantile loss's gradient at `y = p` is `-tau` (`1{y < p} − tau`,
    /// the module table) (task 158).
    #[test]
    fn the_logistic_link_is_the_sigmoid_and_a_quantile_tie_is_minus_tau() {
        let lg = Sgd::new(cfg(1, SgdLoss::Logistic)).unwrap();
        for eta in [-3.0, -0.5, 0.25, 2.0] {
            let want = 1.0 / (1.0 + f64::exp(-eta));
            let got = lg.link(eta);
            assert!(
                (got - want).abs() <= 1e-15,
                "eta {eta}: {got} against {want}"
            );
        }
        let q = Sgd::new(cfg(1, SgdLoss::Quantile { tau: 0.9 })).unwrap();
        assert_eq!(q.dloss(1.0, 1.0, Some(1.0)), -0.9);
    }

    /// Without an intercept, `standardize` divides each feature by the root
    /// of its raw second moment, the row admitted at unit weight, and leaves
    /// a feature with none (here one always 0) as it is; the coefficients are
    /// the betas over the root of the raw moment as it stands. Held to SGD
    /// written from the module docs over the raw sums `Σ w`, `Σ w x²`
    /// (task 158).
    #[test]
    fn standardizing_without_an_intercept_is_the_raw_moment_scaling() {
        let mut c = cfg(3, SgdLoss::Squared);
        c.fit_intercept = false;
        c.standardize = true;
        c.min_weight = 0.0;
        c.learning_rate = 0.1;
        c.l2 = 0.01;
        c.decay = Decay::Halflife(10.0);
        let mut m = Sgd::new(c).unwrap();
        let (mut w_sum, mut s2, mut b) = (0.0f64, [0.0f64; 3], [0.0f64; 3]);
        let mut s = 73u64;
        for i in 0..300 {
            let x = [3.0 + 2.0 * lcg(&mut s), 0.01 * lcg(&mut s), 0.0];
            let y = 0.4 * x[0] - 50.0 * x[1] + 0.1 * lcg(&mut s);
            let w = 0.5 + 0.5 * (lcg(&mut s) + 1.0);
            let d_clock = if i == 0 { 0.0 } else { 1.0 };
            let lam = 0.5f64.powf(d_clock / 10.0);
            let z: Vec<f64> = (0..3)
                .map(|f| {
                    let raw = (lam * s2[f] + x[f] * x[f]) / (lam * w_sum + 1.0);
                    if raw > 0.0 { x[f] / raw.sqrt() } else { x[f] }
                })
                .collect();
            let want: f64 = (0..3).map(|f| b[f] * z[f]).sum();
            let got = m.step(&x, &[Some(y)], d_clock, w).pred[0];
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "row {i}: {got} against {want}"
            );
            for f in 0..3 {
                b[f] -= 0.1 * ((want - y) * z[f] * w + 0.01 * b[f]);
                s2[f] = lam * s2[f] + w * x[f] * x[f];
            }
            w_sum = lam * w_sum + w;
        }
        let got = &m.coefficients()[0];
        for f in 0..3 {
            let raw = s2[f] / w_sum;
            let want = if raw > 0.0 { b[f] / raw.sqrt() } else { b[f] };
            assert!(
                (got[f] - want).abs() <= 1e-10 * want.abs().max(1e-3),
                "slot {f}: {} against {want}",
                got[f]
            );
        }
        assert_eq!(got[2], 0.0, "the feature with no moment");
    }

    /// AdaGrad, from the module docs: each slot's sum of squared gradients
    /// decays on the clock and grows by `g²`, and the slot steps by
    /// `lr / (√G + 1e-8) · g`, `g = d z w (+ l2 b off the intercept)`. With
    /// and without an intercept, at weights other than 1 and through a gap
    /// of 40 half-lives that re-opens the rate (task 158).
    #[test]
    fn adagrad_is_its_recursion() {
        for fit_intercept in [true, false] {
            let mut c = cfg(3, SgdLoss::Squared);
            c.fit_intercept = fit_intercept;
            c.schedule = LearningRate::AdaGrad;
            c.min_weight = 0.0;
            c.learning_rate = 0.3;
            c.l2 = 0.05;
            c.decay = Decay::Halflife(5.0);
            let mut m = Sgd::new(c).unwrap();
            let off = usize::from(fit_intercept);
            let k = 3 + off;
            let (mut b, mut g2) = (vec![0.0f64; k], vec![0.0f64; k]);
            let mut s = 79u64;
            for i in 0..200 {
                let x = [lcg(&mut s), 2.0 * lcg(&mut s), 0.5 * lcg(&mut s)];
                let y = 1.0 + 0.7 * x[0] - 0.3 * x[1] + 0.05 * lcg(&mut s);
                let w = 0.25 + 1.5 * (lcg(&mut s) + 1.0) / 2.0;
                let d_clock = match i {
                    0 => 0.0,
                    100 => 200.0,
                    _ => 1.0,
                };
                let lam = 0.5f64.powf(d_clock / 5.0);
                g2.iter_mut().for_each(|g| *g *= lam);
                let zrow: Vec<f64> = (0..off).map(|_| 1.0).chain(x).collect();
                let want: f64 = (0..k).map(|f| b[f] * zrow[f]).sum();
                let got = m.step(&x, &[Some(y)], d_clock, w).pred[0];
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "intercept {fit_intercept}, row {i}: {got} against {want}"
                );
                for f in 0..k {
                    let pen = if f < off { 0.0 } else { 0.05 * b[f] };
                    let g = (want - y) * zrow[f] * w + pen;
                    g2[f] += g * g;
                    b[f] -= 0.3 / (g2[f].sqrt() + 1e-8) * g;
                }
            }
        }
    }

    /// The scaler learns the row as `[1, x]`: its intercept slot holds the
    /// constant 1 (task 158).
    #[test]
    fn the_scaler_learns_the_intercepts_constant() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.standardize = true;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 83u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let sc = m.scaler.as_ref().unwrap();
        assert_eq!((sc.mean(0), sc.var(0)), (1.0, 0.0));
    }

    /// Under `standardize` a row that moves the scales re-projects every
    /// target, also one that did not learn from it: target 1 carries a value
    /// on one row in three, and its coefficients stay in the box on every
    /// row, where the scales shrinking would carry them past it. A row of
    /// weight 0 moves no scale and leaves every coefficient where it was, to
    /// the bit (task 158).
    #[test]
    fn every_target_is_reprojected_when_the_scales_move() {
        let mut c = constrained(2, 0.0, 0.01, None);
        c.n_targets = 2;
        c.standardize = true;
        c.min_weight = 0.0;
        c.decay = Decay::Halflife(20.0);
        let mut m = Sgd::new(c).unwrap();
        let mut s = 89u64;
        let mut moved = 0;
        for i in 0..400 {
            let x = [10.0 * lcg(&mut s), 0.1 * lcg(&mut s)];
            let y = 0.5 * x[0] + 5.0 * x[1];
            let y1 = (i % 3 == 0).then_some(y);
            let before = m.beta[1].clone();
            m.step(&x, &[Some(y), y1], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            moved += usize::from(y1.is_none() && m.beta[1] != before);
            for (j, b) in m.coefficients().iter().enumerate() {
                assert!(
                    b[1..].iter().all(|v| *v >= -1e-15 && *v <= 0.01 + 1e-15),
                    "row {i}, target {j}: {b:?}"
                );
            }
        }
        assert!(
            moved > 0,
            "target 1 was re-projected on rows it did not learn"
        );

        let mut c = constrained(2, f64::NEG_INFINITY, f64::INFINITY, Some(0.3));
        c.n_targets = 2;
        c.standardize = true;
        c.decay = Decay::Halflife(20.0);
        let mut m = Sgd::new(c).unwrap();
        for i in 0..100 {
            let x = [10.0 * lcg(&mut s), 0.1 * lcg(&mut s)];
            let y = 0.5 * x[0] + 5.0 * x[1];
            m.step(
                &x,
                &[Some(y), Some(-y)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.beta.clone();
        m.step(&[3.0, -0.05], &[Some(1.0), Some(1.0)], 1.0, 0.0);
        assert_eq!(m.beta, before);
    }

    /// The bank's per-target `min_weight` gate reads each target's own
    /// weight through the trait: `true`, and one entry a target, the decayed
    /// weight of the rows that carried it -- a sparse second target its own
    /// rows only (task 158, E14; the mutation run of 2026-10-05 left the
    /// body replaced by `true` or `false` alive).
    #[test]
    fn the_trait_reports_each_targets_own_weight() {
        let mut c = cfg(2, SgdLoss::Squared);
        c.n_targets = 2;
        c.decay = Decay::Halflife(10.0);
        let mut m = Sgd::new(c).unwrap();
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

    /// Bits of a prediction, every NaN one value: a withheld row is NaN
    /// whatever its sign or payload.
    fn bits_of(v: f64) -> u64 {
        if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        }
    }

    /// `huber_delta` is in units of the target's EW residual standard
    /// deviation σ, as `huber`'s is (`robust.rs`): a target scaled by a power
    /// of two fits as the unscaled one does, scaled by it, to the bit, since
    /// every number the step forms -- the residual, σ, the cut, the gradient
    /// -- scales with it exactly. In the target's units the cut bound on the
    /// outliers at scale 1, never at 2^-10 and on every row at 2^10 (review
    /// round 4, CC4; docs/PLAN.md task 195, U1). The squared loss, which has
    /// no cut, is the control.
    #[test]
    fn the_huber_cut_is_in_units_of_the_residual_std() {
        for loss in [SgdLoss::Huber { delta: 1.345 }, SgdLoss::Squared] {
            let run = |c: f64| -> Vec<u64> {
                let mut cf = cfg(2, loss);
                cf.standardize = true;
                cf.clip_gradient = f64::INFINITY;
                let mut m = Sgd::new(cf).unwrap();
                let mut s = 5u64;
                (0..400)
                    .map(|i| {
                        let x = [lcg(&mut s), lcg(&mut s)];
                        let noise = lcg(&mut s);
                        let outlier = if i % 37 == 5 { 20.0 } else { 0.0 };
                        let y = c * (0.5 + 2.0 * x[0] - x[1] + 0.3 * noise + outlier);
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        bits_of(m.step(&x, &[Some(y)], d, 1.0).pred[0] / c)
                    })
                    .collect()
            };
            let one = run(1.0);
            for c in [2f64.powi(-10), 2f64.powi(10)] {
                let scaled = run(c);
                let differ = scaled.iter().zip(&one).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differ, 0,
                    "{loss:?} at scale {c}: {differ} of 400 rows differ"
                );
            }
        }
    }

    /// Under `epsilon_insensitive` a target scaled by a power of two, with
    /// the learning rate scaled by it -- the rate of a sign-valued gradient
    /// is in the target's units (the module docs) -- fits as the unscaled
    /// one does, scaled by it, to the bit: the band is in units of the
    /// target's own spread, which scales with it exactly. The stream sits at
    /// a level of 1,000 in a spread of 2, under a decay and irregular
    /// weights (docs/PLAN.md task 202).
    #[test]
    fn the_band_scales_with_the_target() {
        let run = |c: f64| -> Vec<u64> {
            let mut cf = cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.1 });
            cf.standardize = true;
            cf.clip_gradient = f64::INFINITY;
            cf.decay = Decay::Halflife(50.0);
            cf.learning_rate = 5.0 * c;
            cf.schedule = LearningRate::InvScaling { power: 0.5 };
            let mut m = Sgd::new(cf).unwrap();
            let mut s = 5u64;
            (0..600)
                .map(|i| {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = c * (1000.5 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s));
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let w = 0.5 + lcg(&mut s).abs();
                    bits_of(m.step(&x, &[Some(y)], d, w).pred[0] / c)
                })
                .collect()
        };
        let one = run(1.0);
        for c in [2f64.powi(-10), 2f64.powi(10)] {
            let scaled = run(c);
            let differ = scaled.iter().zip(&one).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "at scale {c}: {differ} of 600 rows differ");
        }
    }

    /// A target held at one value has no spread, so under
    /// `epsilon_insensitive` it has no band: every row teaches, and an
    /// annealed rate takes the fit to the value from zero, at a level of
    /// 1,000, with or without decay. A band in the residual's spread was
    /// drawn from the first residuals, the whole level, and without decay
    /// the fit stopped short of the value by most of it for good
    /// (docs/PLAN.md task 202).
    #[test]
    fn a_target_held_constant_is_learned_under_epsilon_insensitive() {
        for half_life in [f64::INFINITY, 500.0] {
            let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.1 });
            cf.standardize = true;
            cf.decay = Decay::Halflife(half_life);
            cf.learning_rate = 10.0;
            cf.schedule = LearningRate::InvScaling { power: 0.5 };
            cf.min_weight = 2.0;
            let mut m = Sgd::new(cf).unwrap();
            let mut s = 3u64;
            let mut last = f64::NAN;
            for i in 0..20_000 {
                let x = [lcg(&mut s)];
                let d = if i == 0 { 0.0 } else { 1.0 };
                last = m.step(&x, &[Some(1000.0)], d, 1.0).pred[0];
            }
            assert!(
                (last - 1000.0).abs() < 1.0,
                "half-life {half_life}: the last prediction {last}"
            );
        }
    }

    /// The Huber loss and the squared loss are where they were before the
    /// band moved to the target's spread, to the bit: every prediction of a
    /// 400-row stream, under a decay, irregular weights, null targets and
    /// outliers the Huber cut binds on, digested, and three of them pinned
    /// as they print. `huber_delta` stays in units of the residual's
    /// spread: outside its cut the gradient is clipped, not zero, so a fit
    /// from zero still learns from rows the cut holds (docs/PLAN.md task
    /// 202). The digests were BASE's, task 195's build.
    ///
    /// No call into the platform's libm is on the stream's path, so the bits
    /// are every platform's: the decay is a literal factor taken once per
    /// clock unit (`lam^1` is `lam`, `lam^0` is 1, exactly), and `sqrt` is
    /// correctly rounded everywhere. Under `Halflife(80)` with irregular
    /// steps each row's factor was an `exp2`, whose last bit glibc and
    /// Apple's libm round differently: the digests pinned on macOS failed
    /// on Linux (CI, 2026-10-08), and were re-pinned here.
    #[test]
    fn the_huber_and_squared_losses_did_not_move() {
        let run = |loss: SgdLoss| {
            let mut cf = cfg(2, loss);
            cf.standardize = true;
            cf.decay = Decay::Lam(0.9914);
            cf.learning_rate = 0.05;
            cf.min_weight = 3.0;
            let mut m = Sgd::new(cf).unwrap();
            let mut s = 17u64;
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            let mut got = Vec::new();
            for i in 0..400 {
                let x = [lcg(&mut s), 3.0 + lcg(&mut s)];
                let outlier = if i % 41 == 7 { 25.0 } else { 0.0 };
                let y =
                    (i % 17 != 3).then(|| 5.0 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s) + outlier);
                let d = if i == 0 { 0.0 } else { 1.0 };
                let w = if i % 23 == 9 {
                    0.0
                } else {
                    0.5 + lcg(&mut s).abs()
                };
                let p = m.step(&x, &[y], d, w).pred[0];
                for byte in bits_of(p).to_le_bytes() {
                    h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
                if matches!(i, 50 | 200 | 399) {
                    got.push(p);
                }
            }
            (h, got)
        };
        let huber = run(SgdLoss::Huber { delta: 1.345 });
        let squared = run(SgdLoss::Squared);
        println!("huber {huber:x?}\nsquared {squared:x?}");
        assert_ne!(huber.0, squared.0, "the cut binds on the outliers");
        for ((h, got), (digest, picks)) in [huber, squared].into_iter().zip([
            (
                0x95a9_bba1_2435_1e24_u64,
                [2.5454166364986803, 0.22878767638248743, 4.291195395028382],
            ),
            (
                0x5948_2068_29a3_3663_u64,
                [3.8900811448866905, 0.6435315190771895, 4.854641183478482],
            ),
        ]) {
            assert_eq!(h, digest, "picks {got:?}");
            for (g, w) in got.iter().zip(picks) {
                assert_eq!(g.to_bits(), f64::to_bits(w), "{got:?}");
            }
        }
    }

    /// `eps` is in units of the target's spread: a target whose spread is
    /// under 0.1 of its own units -- a return in decimal -- is learned. In
    /// the target's units every residual sat inside the tube, so the model
    /// was passive for ever and every coefficient stayed 0 (review round 4,
    /// CC4).
    #[test]
    fn a_target_smaller_than_the_tube_in_its_own_units_is_learned() {
        let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.1 });
        cf.standardize = true;
        cf.learning_rate = 1e-4;
        let mut m = Sgd::new(cf).unwrap();
        let mut s = 9u64;
        for i in 0..2000 {
            let x = [lcg(&mut s)];
            let y = 1e-3 * (0.5 + 2.0 * x[0] + 0.3 * lcg(&mut s));
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let c = &m.coefficients()[0];
        assert!(
            (c[1] - 2e-3).abs() < 1e-3,
            "the slope is 2e-3 in the target's units: {c:?}"
        );
    }

    /// A logistic label outside {0, 1} is clamped into [0, 1], as `ftrl`
    /// clamps it: a label of 5 teaches what a label of 1 does and one of -2
    /// what 0 does, to the bit. `p - y` with `p` in (0, 1) and `y = 5` pushed
    /// the linear predictor up on every row for ever (review round 4, CC9;
    /// docs/PLAN.md task 195, S4).
    #[test]
    fn a_logistic_label_outside_zero_and_one_is_clamped() {
        let fit = |hi: f64, lo: f64| {
            let mut m = Sgd::new(cfg(1, SgdLoss::Logistic)).unwrap();
            let mut s = 21u64;
            for i in 0..300 {
                let x = [lcg(&mut s)];
                let y = if x[0] + 0.5 * lcg(&mut s) > 0.0 {
                    hi
                } else {
                    lo
                };
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients()[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(fit(5.0, -2.0), fit(1.0, 0.0));
        assert_eq!(fit(1.0, -0.5), fit(1.0, 0.0));
    }

    /// Under `strict_binary` a label that is not 0 or 1 teaches nothing and
    /// counts nothing toward its target's weight, as in `ftrl`; the option
    /// is the logistic loss's alone (docs/PLAN.md task 195, S4).
    #[test]
    fn strict_binary_skips_a_label_that_is_not_zero_or_one() {
        let mut c = cfg(1, SgdLoss::Logistic);
        c.strict_binary = true;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c.clone()).unwrap();
        m.step(&[0.5], &[Some(1.0)], 0.0, 1.0);
        let (before, w) = (m.coefficients(), m.target_weights()[0]);
        m.step(&[0.5], &[Some(0.7)], 1.0, 1.0);
        assert_eq!(m.coefficients(), before, "a label of 0.7 taught nothing");
        assert_eq!(m.target_weights()[0], w, "and counted nothing");
        m.step(&[0.5], &[Some(0.0)], 1.0, 1.0);
        assert_ne!(m.coefficients(), before, "a label of 0 teaches");
        let mut bad = c;
        bad.loss = SgdLoss::Squared;
        let err = bad.validate().unwrap_err();
        assert!(err.contains("strict_binary"), "{err}");
    }

    /// `s²` is the EW mean of the squared out-of-sample residuals of the
    /// rows that carried the target, a weight above 0 and a prediction
    /// (the module docs): held to that mean from its definition, every
    /// qualifying row's weight aged by the clock since it, over irregular
    /// steps and weights, null targets, zero weights and the rows before
    /// `min_weight`, which add nothing.
    #[test]
    fn the_residual_variance_is_the_ew_mean_of_the_squared_residuals() {
        let mut c = cfg(2, SgdLoss::Huber { delta: 1.345 });
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 3.0;
        let mut m = Sgd::new(c.clone()).unwrap();
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let mut s = 7u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i % 9 != 4).then(|| 1.0 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s));
            let w = if i % 13 == 6 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
            let lam = c.decay.factor(d);
            let p = m.step(&x, &[y], d, w).pred[0];
            for (_, wi) in seen.iter_mut() {
                *wi *= lam;
            }
            if let Some(yv) = y
                && w > 0.0
                && p.is_finite()
            {
                seen.push(((yv - p) * (yv - p), w));
            }
            let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
            let want = if total > 0.0 {
                seen.iter().map(|(r2, wi)| r2 * wi).sum::<f64>() / total
            } else {
                0.0
            };
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "row {i}: s² {got} against {want}"
            );
        }
        assert!(seen.len() > 150, "{} rows taught it", seen.len());
    }

    /// Under `epsilon_insensitive` the tube's unit is its definition:
    /// the target's EW variance around its EW mean, `Σ ωᵢ (yᵢ − m)² / Σ ωᵢ`
    /// with `m = Σ ωᵢ yᵢ / Σ ωᵢ`, over the rows that carried the target
    /// with a weight above 0, the rows before `min_weight` included, `ωᵢ`
    /// aged by the model's decay over every row since, rebuilt from the
    /// history at every row in two passes, not by the recursion. Irregular
    /// steps and weights, null targets and rows of weight 0, at a level of
    /// 1,000 (docs/PLAN.md task 202).
    #[test]
    fn the_target_variance_is_the_ew_variance_of_the_target() {
        let mut c = cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.1 });
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 3.0;
        c.standardize = true;
        let mut m = Sgd::new(c.clone()).unwrap();
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let mut s = 7u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i % 9 != 4).then(|| 1000.0 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s));
            let w = if i % 13 == 6 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
            let lam = c.decay.factor(d);
            m.step(&x, &[y], d, w);
            for (_, wi) in seen.iter_mut() {
                *wi *= lam;
            }
            if let Some(v) = y
                && w > 0.0
            {
                seen.push((v, w));
            }
            let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
            let want = if seen.len() < 2 {
                0.0
            } else {
                let mean = seen.iter().map(|(v, wi)| v * wi).sum::<f64>() / total;
                seen.iter()
                    .map(|(v, wi)| wi * (v - mean) * (v - mean))
                    .sum::<f64>()
                    / total
            };
            let got = m.target_variance()[0];
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "row {i}: s_y² {got} against {want}"
            );
            let unit = m.scale(0);
            assert_eq!(unit, (got > 0.0).then(|| got.sqrt()), "row {i}");
        }
        assert!(seen.len() > 150, "{} rows taught it", seen.len());
        assert!(m.sigma2().is_empty(), "no residual scale under this loss");
    }
}
