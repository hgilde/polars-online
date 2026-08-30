//! Model bank specs (docs/PLAN.md §3, §5): serde-deserializable from JSON
//! (Python) and TOML (CLI), with common-parameter validation and the output
//! struct layout.

use online_core::{ClockCfg, Decay, OnClockReset, SessionGap};
use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

/// A float that also accepts the JSON strings `"inf"` / `"-inf"`, since JSON
/// has no infinity literal and `halflife = inf` is meaningful (it pins a
/// coefficient, docs/PLAN.md §4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num(pub f64);

impl Serialize for Num {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            s.serialize_f64(self.0)
        } else if self.0 == f64::INFINITY {
            s.serialize_str("inf")
        } else if self.0 == f64::NEG_INFINITY {
            s.serialize_str("-inf")
        } else {
            s.serialize_str("nan")
        }
    }
}

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(f64),
            Word(String),
        }
        match Raw::deserialize(d)? {
            Raw::Num(v) => Ok(Num(v)),
            Raw::Word(w) => match w.to_ascii_lowercase().as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(Num(f64::INFINITY)),
                "-inf" | "-infinity" => Ok(Num(f64::NEG_INFINITY)),
                "nan" => Ok(Num(f64::NAN)),
                other => Err(serde::de::Error::custom(format!(
                    "expected a number or \"inf\"/\"-inf\", got {other:?}"
                ))),
            },
        }
    }
}

/// A float or a list of floats (grids; `halflife` lists mean one accumulator
/// per value, docs/PLAN.md §4.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FloatOrList {
    Float(Num),
    List(Vec<Num>),
}

impl FloatOrList {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrList::Float(f) => vec![f.0],
            FloatOrList::List(v) => v.iter().map(|n| n.0).collect(),
        }
    }
}

/// `session_gap`: clock units, or "reset".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionGapSpec {
    Gap(f64),
    Word(String),
}

/// Model choice + model-specific params (docs/PLAN.md §4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelKind {
    EwRidge {
        #[serde(default)]
        ridge: Option<FloatOrList>,
        #[serde(default)]
        feature_sets: Option<Vec<(String, Vec<String>)>>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        ridge_decay: bool,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
    },
    Lasso {
        /// Decreasing penalties on standardized stats; required.
        lasso_path: Vec<f64>,
        /// 1.0 = lasso, < 1.0 = elastic net.
        #[serde(default)]
        l1_ratio: Option<f64>,
        /// Halflife of the EW squared error used to select lambda.
        #[serde(default)]
        select_halflife: Option<f64>,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        max_cd_iters: Option<u32>,
        #[serde(default)]
        cd_tol: Option<f64>,
    },
    Kalman {
        /// Per-factor coefficient halflife (scalar or one per slot, intercept
        /// first). `inf` pins a coefficient. Note this is the COEFFICIENT
        /// halflife; the spec-level `halflife` drives the standardization and
        /// residual-variance statistics.
        coef_halflife: FloatOrList,
        #[serde(default)]
        q: Option<Vec<Num>>,
        #[serde(default)]
        obs_var: Option<f64>,
        #[serde(default)]
        p0: Option<f64>,
        #[serde(default)]
        share_p: bool,
        /// Standardize features internally (default true). Off makes the filter
        /// a plain Bayesian linear regression on the features' own scale.
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Huber regression (docs/PLAN.md §4.5).
    Huber {
        /// Cut point in units of the EW residual std. Default 1.5 [validate].
        #[serde(default)]
        huber_delta: Option<f64>,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
    },
    /// Quantile regression at level `quantile` (docs/PLAN.md §4.5).
    Quantile {
        quantile: f64,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<f64>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        quantile_eps: Option<f64>,
    },
    /// Online logistic regression via FTRL-proximal (docs/PLAN.md §4.6).
    /// `pred` is a probability; `resid = y - p`.
    Ftrl {
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        #[serde(default)]
        l1: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        /// Error on targets that are not 0/1 rather than clamping them.
        /// Logistic loss only.
        #[serde(default)]
        strict_binary: bool,
        /// "logistic" (default, binary targets, `pred` is a probability) or
        /// "squared" (continuous targets, sparse linear regression).
        #[serde(default)]
        loss: Option<String>,
    },
    /// EW moments of the feature columns, no regression (docs/PLAN.md §4.7).
    /// `targets` is ignored; every column of interest goes in `features`.
    EwCov {
        /// Any of "mean", "var", "std", "cov", "corr". Default: mean + std + corr.
        #[serde(default)]
        stats: Option<Vec<String>>,
    },
    /// Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).
    /// O(k) per row, no solves, and the only model here that takes count
    /// targets (via `loss = "poisson"`).
    Sgd {
        /// "squared" (default), "huber", "quantile", "epsilon_insensitive",
        /// "poisson" or "logistic".
        #[serde(default)]
        loss: Option<String>,
        /// Huber cut, in target units.
        #[serde(default)]
        huber_delta: Option<f64>,
        #[serde(default)]
        quantile: Option<f64>,
        /// Width of the insensitive tube.
        #[serde(default)]
        eps: Option<f64>,
        #[serde(default)]
        learning_rate: Option<f64>,
        /// "constant" (default), "inv_scaling" or "adagrad".
        #[serde(default)]
        schedule: Option<String>,
        /// Exponent for `inv_scaling`.
        #[serde(default)]
        power: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        #[serde(default)]
        clip_gradient: Option<f64>,
    },
    /// Passive-aggressive regression (ENHANCEMENTS E17). No learning rate:
    /// each row's update is the smallest change that satisfies it.
    Pa {
        /// "pa1" (default), "pa" (unbounded) or "pa2" (damped).
        #[serde(default)]
        mode: Option<String>,
        /// Aggressiveness cap. Ignored by "pa".
        #[serde(default)]
        c: Option<f64>,
        /// Insensitive tube: rows already this close leave the fit alone.
        #[serde(default)]
        eps: Option<f64>,
    },
    Rls {
        /// Prior strength: `P0 = I / ridge`. Scalar only (baked into P0).
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        coef0: Option<Vec<Vec<f64>>>,
    },
}

impl ModelKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ew_ridge",
            ModelKind::Rls { .. } => "rls",
            ModelKind::Lasso { .. } => "lasso",
            ModelKind::Kalman { .. } => "kalman",
            ModelKind::Huber { .. } => "huber",
            ModelKind::Quantile { .. } => "quantile",
            ModelKind::Ftrl { .. } => "ftrl",
            ModelKind::EwCov { .. } => "ew_cov",
            ModelKind::Sgd { .. } => "sgd",
            ModelKind::Pa { .. } => "pa",
        }
    }
}

/// One model spec: common parameters (docs/PLAN.md §3) + the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    /// Output struct column name.
    pub name: String,
    pub model: ModelKind,
    pub targets: Vec<String>,
    pub features: Vec<String>,
    #[serde(default = "default_true")]
    pub add_intercept: bool,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub halflife: Option<FloatOrList>,
    #[serde(default)]
    pub lam: Option<f64>,
    #[serde(default)]
    pub max_dclock: Option<f64>,
    #[serde(default)]
    pub on_clock_reset: OnClockReset,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub min_periods: Option<f64>,
    /// 0 = never; coefficients are also emitted on the last row of every chunk.
    #[serde(default)]
    pub coef_every: u32,
    /// Emit `sigma_<slot>`: the EW standard deviation of this slot's
    /// out-of-sample residuals, read from the state *before* each row. Off by
    /// default because it widens the output struct.
    #[serde(default)]
    pub emit_sigma: bool,
    /// Emit `resid_z_<slot>` = `resid / sigma`: how surprising this row was, in
    /// units of the model's own recent error. Off by default.
    #[serde(default)]
    pub emit_resid_z: bool,
    /// Emit `selected_<target>` and `pred_<target>__selected`: online model
    /// selection across every grid slot for that target (ridge values, feature
    /// sets and halflives), by lowest EW out-of-sample error. Generalizes the
    /// lasso's `lam_selected`. Requires more than one slot per target.
    #[serde(default)]
    pub emit_selected: bool,
    /// ModelBank/CLI only; one state per key. The expression API uses `.over()`.
    #[serde(default)]
    pub group: Option<String>,
}

impl Spec {
    pub fn k(&self) -> usize {
        self.features.len()
    }

    pub fn m(&self) -> usize {
        self.targets.len()
    }

    /// Decay values, one per halflife grid entry (one model instance each).
    pub fn decays(&self) -> Result<Vec<(String, Decay)>, String> {
        match (&self.halflife, self.lam) {
            (Some(_), Some(_)) => Err(format!(
                "spec {:?}: halflife and lam are mutually exclusive",
                self.name
            )),
            (None, None) => Err(format!(
                "spec {:?}: one of halflife/lam is required",
                self.name
            )),
            (None, Some(l)) => {
                if !(0.0 < l && l <= 1.0) {
                    return Err(format!("spec {:?}: lam must be in (0, 1]", self.name));
                }
                Ok(vec![(String::new(), Decay::Lam(l))])
            }
            (Some(h), None) => {
                let hs = h.to_vec();
                if hs.iter().any(|&h| h <= 0.0) {
                    return Err(format!("spec {:?}: halflife must be > 0", self.name));
                }
                if hs.len() == 1 {
                    Ok(vec![(String::new(), Decay::Halflife(hs[0]))])
                } else {
                    Ok(hs
                        .iter()
                        .map(|&h| (format!("@h{h}"), Decay::Halflife(h)))
                        .collect())
                }
            }
        }
    }

    pub fn clock_cfg(&self) -> Result<ClockCfg, String> {
        if self.clock.is_some() && self.max_dclock.is_none() {
            return Err(format!(
                "spec {:?}: max_dclock is required when clock is given",
                self.name
            ));
        }
        let session_gap = match &self.session_gap {
            None => None,
            Some(SessionGapSpec::Gap(g)) => Some(SessionGap::Gap(*g)),
            Some(SessionGapSpec::Word(w)) if w == "reset" => Some(SessionGap::Reset),
            Some(SessionGapSpec::Word(w)) => {
                return Err(format!(
                    "spec {:?}: session_gap must be a number or \"reset\", got {w:?}",
                    self.name
                ));
            }
        };
        if self.session.is_some() && session_gap.is_none() {
            return Err(format!(
                "spec {:?}: session_gap is required when session is given",
                self.name
            ));
        }
        Ok(ClockCfg {
            max_dclock: self.max_dclock.unwrap_or(f64::INFINITY),
            on_clock_reset: self.on_clock_reset,
            session_gap,
        })
    }

    pub fn min_periods_or_default(&self) -> f64 {
        self.min_periods
            .unwrap_or((self.k() + usize::from(self.add_intercept)) as f64)
    }

    /// Default solve cadence: halflife/50 (docs/PLAN.md §4.1, [validate]).
    pub fn solve_every_default(&self, decay: Decay) -> f64 {
        match decay {
            Decay::Halflife(h) if h.is_finite() => h / 50.0,
            _ => 0.0, // lam decay / infinite halflife: solve every row
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.targets.is_empty() || self.features.is_empty() {
            return Err(format!(
                "spec {:?}: targets and features must be non-empty",
                self.name
            ));
        }
        self.decays()?;
        self.clock_cfg()?;
        if self.emit_selected {
            if matches!(self.model, ModelKind::EwCov { .. }) {
                return Err(format!(
                    "spec {:?}: emit_selected does not apply to ew_cov (it has no predictions)",
                    self.name
                ));
            }
            let n_slots = self.decays()?.len() * crate::combo_labels(self).len();
            if n_slots < 2 {
                return Err(format!(
                    "spec {:?}: emit_selected needs more than one slot per target; add a \
                     ridge/feature_set/halflife grid or a lasso path",
                    self.name
                ));
            }
        }
        match &self.model {
            ModelKind::Pa { mode, c, eps } => {
                if let Some(md) = mode {
                    if !["pa", "pa1", "pa2"].contains(&md.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown pa mode {md:?}; expected pa, pa1 or pa2",
                            self.name
                        ));
                    }
                }
                if c.is_some_and(|v| v <= 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: pa c must be > 0", self.name));
                }
                if eps.is_some_and(|v| v < 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: pa eps must be >= 0", self.name));
                }
            }
            ModelKind::Sgd {
                loss,
                quantile,
                learning_rate,
                schedule,
                ..
            } => {
                if let Some(l) = loss {
                    const OK: [&str; 6] = [
                        "squared",
                        "huber",
                        "quantile",
                        "epsilon_insensitive",
                        "poisson",
                        "logistic",
                    ];
                    if !OK.contains(&l.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown sgd loss {l:?}; expected one of {}",
                            self.name,
                            OK.join(", ")
                        ));
                    }
                    if l == "quantile" && quantile.is_none() {
                        return Err(format!(
                            "spec {:?}: sgd loss \"quantile\" needs a `quantile` level",
                            self.name
                        ));
                    }
                }
                if let Some(sc) = schedule {
                    if !["constant", "inv_scaling", "adagrad"].contains(&sc.as_str()) {
                        return Err(format!(
                            "spec {:?}: unknown sgd schedule {sc:?}; expected constant, \
                             inv_scaling or adagrad",
                            self.name
                        ));
                    }
                }
                if learning_rate.is_some_and(|v| v <= 0.0 || v.is_nan()) {
                    return Err(format!("spec {:?}: learning_rate must be > 0", self.name));
                }
            }
            ModelKind::EwCov { stats } => {
                if let Some(stats) = stats {
                    for st in stats {
                        if !["mean", "var", "std", "cov", "corr"].contains(&st.as_str()) {
                            return Err(format!(
                                "spec {:?}: unknown ew_cov statistic {st:?}; expected one of \
                                 mean, var, std, cov, corr",
                                self.name
                            ));
                        }
                    }
                    if self.k() < 2 && stats.iter().any(|st| st == "cov" || st == "corr") {
                        return Err(format!(
                            "spec {:?}: ew_cov cov/corr need at least two features",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::Ftrl {
                alpha,
                beta,
                l1,
                l2,
                ..
            } => {
                if alpha.is_some_and(|a| a <= 0.0 || a.is_nan()) {
                    return Err(format!("spec {:?}: ftrl alpha must be > 0", self.name));
                }
                for (name, v) in [("beta", beta), ("l1", l1), ("l2", l2)] {
                    if v.is_some_and(|v| v < 0.0 || v.is_nan()) {
                        return Err(format!("spec {:?}: ftrl {name} must be >= 0", self.name));
                    }
                }
            }
            ModelKind::Huber { huber_delta, .. } => {
                if huber_delta.is_some_and(|d| d <= 0.0 || d.is_nan()) {
                    return Err(format!("spec {:?}: huber_delta must be > 0", self.name));
                }
            }
            ModelKind::Quantile {
                quantile,
                quantile_eps,
                ..
            } => {
                if !(0.0 < *quantile && *quantile < 1.0) {
                    return Err(format!("spec {:?}: quantile must be in (0, 1)", self.name));
                }
                if quantile_eps.is_some_and(|e| e <= 0.0) {
                    return Err(format!("spec {:?}: quantile_eps must be > 0", self.name));
                }
            }
            ModelKind::Kalman {
                coef_halflife,
                q,
                obs_var,
                p0,
                ..
            } => {
                let k_total = self.k() + usize::from(self.add_intercept);
                let hs = coef_halflife.to_vec();
                if hs.len() != 1 && hs.len() != k_total {
                    return Err(format!(
                        "spec {:?}: coef_halflife must be scalar or length {k_total}",
                        self.name
                    ));
                }
                if hs.iter().any(|&h| h <= 0.0) {
                    return Err(format!("spec {:?}: coef_halflife must be > 0", self.name));
                }
                if q.as_ref().is_some_and(|q| q.len() != k_total) {
                    return Err(format!(
                        "spec {:?}: q must have length {k_total}",
                        self.name
                    ));
                }
                if obs_var.is_some_and(|v| v <= 0.0) {
                    return Err(format!("spec {:?}: obs_var must be > 0", self.name));
                }
                if p0.is_some_and(|v| v <= 0.0) {
                    return Err(format!("spec {:?}: p0 must be > 0", self.name));
                }
            }
            ModelKind::Lasso {
                lasso_path,
                l1_ratio,
                ..
            } => {
                if lasso_path.is_empty() {
                    return Err(format!(
                        "spec {:?}: lasso_path must be non-empty",
                        self.name
                    ));
                }
                if !lasso_path.windows(2).all(|w| w[0] >= w[1]) {
                    return Err(format!(
                        "spec {:?}: lasso_path must be decreasing",
                        self.name
                    ));
                }
                if l1_ratio.is_some_and(|r| !(0.0..=1.0).contains(&r)) {
                    return Err(format!("spec {:?}: l1_ratio must be in [0, 1]", self.name));
                }
            }
            ModelKind::Rls { ridge, coef0 } => {
                if ridge.is_some_and(|r| r <= 0.0 || r.is_nan()) {
                    return Err(format!("spec {:?}: rls ridge must be > 0", self.name));
                }
                let k_total = self.k() + usize::from(self.add_intercept);
                if let Some(c) = coef0 {
                    if c.len() != self.m() || c.iter().any(|v| v.len() != k_total) {
                        return Err(format!(
                            "spec {:?}: coef0 must be n_targets x (n_features + intercept)",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::EwRidge { feature_sets, .. } => {
                if let Some(fs) = feature_sets {
                    for (name, cols) in fs {
                        for c in cols {
                            if !self.features.contains(c) {
                                return Err(format!(
                                    "spec {:?}: feature set {name:?} references unknown feature {c:?}",
                                    self.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
