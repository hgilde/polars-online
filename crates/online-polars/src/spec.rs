//! Model bank specs (docs/PLAN.md §3, §5): serde-deserializable from JSON
//! (Python) and TOML (CLI), with common-parameter validation and the output
//! struct layout.

use online_core::{ClockCfg, Decay, OnClockReset, SessionGap};
use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

/// A float or a list of floats (grids; `halflife` lists mean one accumulator
/// per value, docs/PLAN.md §4.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FloatOrList {
    Float(f64),
    List(Vec<f64>),
}

impl FloatOrList {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrList::Float(f) => vec![*f],
            FloatOrList::List(v) => v.clone(),
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
}

impl ModelKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ew_ridge",
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
        match &self.model {
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
