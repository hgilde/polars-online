//! One (spec, group) stream: clock state + model instances (one per halflife
//! grid entry), row-by-row processing with the docs/PLAN.md §3 null policy.

use online_core::{
    ClockState, Decay, EwRidge, EwRidgeCfg, ModelState, OnlineModel, State, StateError,
};
use serde::{Deserialize, Serialize};

use crate::spec::{FloatOrList, ModelKind, Spec};

/// Enum dispatch over the models the bank can run (serde-friendly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnyModel {
    EwRidge(Box<EwRidge>),
}

impl AnyModel {
    pub fn step(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
    ) -> online_core::Step {
        match self {
            AnyModel::EwRidge(m) => m.step(x, y, d_clock, weight),
        }
    }

    pub fn n_outputs(&self) -> usize {
        match self {
            AnyModel::EwRidge(m) => m.n_outputs(),
        }
    }

    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        match self {
            AnyModel::EwRidge(m) => m.coefficients().map(|b| b.to_vec()),
        }
    }

    pub fn state(&self) -> State {
        match self {
            AnyModel::EwRidge(m) => m.state(),
        }
    }

    pub fn restore(s: &State) -> Result<Self, StateError> {
        match &s.model {
            ModelState::EwRidge(_) => Ok(AnyModel::EwRidge(Box::new(EwRidge::restore(s)?))),
            other => Err(StateError::WrongModel {
                expected: "a bank-supported model",
                found: other.kind(),
            }),
        }
    }
}

/// Build the model instances for a spec: one per halflife grid entry.
pub fn build_models(spec: &Spec) -> Result<Vec<(String, AnyModel)>, String> {
    let decays = spec.decays()?;
    decays
        .into_iter()
        .map(|(suffix, decay)| {
            let m = build_one(spec, decay)?;
            Ok((suffix, m))
        })
        .collect()
}

fn build_one(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            standardize,
            ridge_decay,
            solve_every,
            max_rows_between_solves,
        } => {
            let fs = feature_sets
                .as_ref()
                .map(|sets| {
                    sets.iter()
                        .map(|(name, cols)| {
                            let idx = cols
                                .iter()
                                .map(|c| spec.features.iter().position(|f| f == c).unwrap())
                                .collect();
                            (name.clone(), idx)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let cfg = EwRidgeCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                ridge: ridge
                    .as_ref()
                    .map(FloatOrList::to_vec)
                    .unwrap_or_else(|| vec![1e-6]),
                feature_sets: fs,
                standardize: *standardize,
                ridge_decay: *ridge_decay,
                min_periods: spec.min_periods_or_default(),
                solve_every: solve_every.unwrap_or_else(|| spec.solve_every_default(decay)),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
            };
            Ok(AnyModel::EwRidge(Box::new(EwRidge::new(cfg)?)))
        }
    }
}

/// Combo labels per model instance ("" when there is only one combo).
pub fn combo_labels(spec: &Spec) -> Vec<String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            ..
        } => {
            let nr = ridge.as_ref().map(|r| r.to_vec().len()).unwrap_or(1);
            let nf = feature_sets.as_ref().map(|f| f.len()).unwrap_or(0).max(1);
            if nr * nf == 1 {
                return vec![String::new()];
            }
            let ridges = ridge
                .as_ref()
                .map(FloatOrList::to_vec)
                .unwrap_or(vec![1e-6]);
            let fs_names: Vec<String> = feature_sets
                .as_ref()
                .map(|f| f.iter().map(|(n, _)| n.clone()).collect())
                .unwrap_or_else(|| vec!["all".to_string()]);
            let mut out = Vec::new();
            for f in &fs_names {
                for r in &ridges {
                    out.push(if nf == 1 {
                        format!("__r{r}")
                    } else if nr == 1 {
                        format!("__{f}")
                    } else {
                        format!("__{f}_r{r}")
                    });
                }
            }
            out
        }
    }
}

/// Serialized per-stream state: the clock plus each halflife's model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamState {
    pub clock: ClockState,
    pub models: Vec<State>,
    pub rows_seen: u64,
}

/// Live per-stream state.
pub struct Stream {
    pub clock: ClockState,
    pub models: Vec<(String, AnyModel)>,
    pub rows_seen: u64,
}

/// Output of one row for one stream: `None` = skipped/emit-all-null.
pub struct RowOut {
    /// Per model instance, per output slot (NaN = null).
    pub pred: Vec<Vec<f64>>,
    /// `y_j - pred_slot`; NaN when the target is null or the pred not ready.
    pub resid: Vec<Vec<f64>>,
    pub n_eff: Vec<f64>,
    pub coef: Option<Vec<Vec<Vec<f64>>>>,
    pub extra: Vec<Option<online_core::Extra>>,
}

impl Stream {
    pub fn new(spec: &Spec) -> Result<Self, String> {
        Ok(Self {
            clock: ClockState::new(),
            models: build_models(spec)?,
            rows_seen: 0,
        })
    }

    pub fn save(&self) -> StreamState {
        StreamState {
            clock: self.clock.clone(),
            models: self.models.iter().map(|(_, m)| m.state()).collect(),
            rows_seen: self.rows_seen,
        }
    }

    pub fn restore(spec: &Spec, saved: &StreamState) -> Result<Self, String> {
        let fresh = build_models(spec)?;
        if fresh.len() != saved.models.len() {
            return Err("saved state has a different number of model instances".into());
        }
        let models = fresh
            .into_iter()
            .zip(&saved.models)
            .map(|((suffix, _), st)| {
                Ok((suffix, AnyModel::restore(st).map_err(|e| e.to_string())?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            clock: saved.clock.clone(),
            models,
            rows_seen: saved.rows_seen,
        })
    }

    fn reset_models(&mut self, spec: &Spec) {
        self.models = build_models(spec).expect("spec was already validated");
    }

    /// Process one row. `emit_coef` forces a coefficient snapshot (last row of
    /// chunk); `coef_every` counts accepted rows.
    #[allow(clippy::too_many_arguments)]
    pub fn process_row(
        &mut self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        x: &[Option<f64>],
        y: &[Option<f64>],
        clock: Option<f64>,
        session: Option<u64>,
        weight: Option<f64>,
        emit_coef: bool,
    ) -> Option<RowOut> {
        let accept = x.iter().all(|v| v.is_some_and(f64::is_finite))
            && weight.map(|w| w.is_finite()).unwrap_or(true);
        let adv = self.clock.advance(cfg, clock, session, accept);
        if adv.reset {
            self.reset_models(spec);
        }
        if !accept {
            return None;
        }
        self.rows_seen += 1;
        let xs: Vec<f64> = x.iter().map(|v| v.unwrap()).collect();
        let ys: Vec<Option<f64>> = y.iter().map(|v| v.filter(|f| f.is_finite())).collect();
        let w = weight.unwrap_or(1.0);

        let want_coef =
            emit_coef || (spec.coef_every > 0 && self.rows_seen % u64::from(spec.coef_every) == 0);
        let mut pred = Vec::with_capacity(self.models.len());
        let mut resid = Vec::with_capacity(self.models.len());
        let mut n_eff = Vec::with_capacity(self.models.len());
        let mut extra = Vec::with_capacity(self.models.len());
        let mut coef = if want_coef { Some(Vec::new()) } else { None };
        let m_targets = ys.len();
        for (_, m) in self.models.iter_mut() {
            let step = m.step(&xs, &ys, adv.d_clock, w);
            let nc = step.pred.len() / m_targets;
            let r: Vec<f64> = step
                .pred
                .iter()
                .enumerate()
                .map(|(slot, p)| match ys[slot / nc] {
                    Some(yj) if p.is_finite() => yj - p,
                    _ => f64::NAN,
                })
                .collect();
            pred.push(step.pred);
            resid.push(r);
            n_eff.push(step.n_eff);
            extra.push(step.extra);
            if let Some(c) = &mut coef {
                c.push(m.coefficients().unwrap_or_default());
            }
        }
        Some(RowOut {
            pred,
            resid,
            n_eff,
            coef,
            extra,
        })
    }
}
