//! Task 221's diagnostics for one model instance: the accumulators that keep
//! a memory of their own, defaulting to the instance's half-life, `inf` the
//! run-once form (docs/PLAN.md task 221).
//!
//! Each reads the row's out-of-sample prediction and residual as the other
//! residual diagnostics do -- before the row folds into it, so a field never
//! includes the row it describes -- and each folds the row only when the
//! row is learned: under `embargo` at its release, with the prediction the
//! row was scored with. Their per-slot values ride in one output buffer,
//! [`crate::ChunkOut::checks`], [`value_names`]' count of them per slot.

use online_core::{Breaks, Calibration, Decay, Sandwich, TwinFit};
use serde::{Deserialize, Serialize};

use crate::spec::{ModelKind, Spec};

/// The values one slot writes, in their order in the `checks` buffer, with
/// the field each becomes (`<name>_<slot>`).
pub fn value_names(spec: &Spec) -> Vec<&'static str> {
    let mut out = Vec::new();
    if spec.emit_calibration {
        out.extend([
            "calibration_slope",
            "calibration_intercept",
            "calibration_wald",
        ]);
    }
    if spec.emit_breaks {
        out.extend(["studentized", "cusum", "cusum_sq", "break_wald"]);
    }
    out
}

/// What one instance's diagnostics are configured with, from the spec:
/// each one's decay -- its own memory, or the instance's -- and, for the
/// breaks' twin fits, the features each slot's coefficients cover.
#[derive(Debug, Clone)]
pub struct CheckCfg {
    calibration: Decay,
    breaks: Decay,
    robust: Decay,
    /// Newey and West's lags, `0` for HC0 alone.
    lags: usize,
    /// Per combo (slot `s` is combo `s % n_combos`), the features its fit
    /// reads, as positions in the spec's features: an `ewridge` feature
    /// set's, or every feature.
    combo_features: Vec<Vec<usize>>,
    /// Whether the fits carry an intercept: the spec's, and `holt`'s level,
    /// which has no feature to stand on.
    intercept: bool,
    /// The coefficients per slot, as `coef` lays them out: the features and
    /// the spec's intercept.
    width: usize,
}

impl CheckCfg {
    pub fn of(spec: &Spec, model: Decay) -> Self {
        let all: Vec<usize> = (0..spec.k()).collect();
        let combo_features = match &spec.model {
            ModelKind::EwRidge {
                feature_sets: Some(sets),
                ridge,
                ..
            } => {
                let nr = ridge.as_ref().map_or(1, |r| r.to_vec().len());
                sets.iter()
                    .flat_map(|(_, cols)| {
                        let idx: Vec<usize> = cols
                            .iter()
                            .filter_map(|c| spec.features.iter().position(|f| f == c))
                            .collect();
                        std::iter::repeat_n(idx, nr)
                    })
                    .collect()
            }
            _ => vec![all; crate::stream::combos(spec).len()],
        };
        Self {
            calibration: Spec::diagnostic_decay(spec.calibration_half_life.as_ref(), model),
            breaks: Spec::diagnostic_decay(spec.breaks_half_life.as_ref(), model),
            robust: Spec::diagnostic_decay(spec.robust_se_half_life.as_ref(), model),
            lags: spec.robust_se_lags_or_default(),
            combo_features,
            intercept: spec.fit_intercept || spec.k() == 0,
            width: spec.k() + usize::from(spec.fit_intercept),
        }
    }

    /// Whether `se_coef_hac` is written: a lag to weigh.
    pub fn hac(&self) -> bool {
        self.lags > 0
    }
}

/// One row as the diagnostics read it, per slot of the instance.
pub struct RowView<'a> {
    /// Each slot's prediction: under an embargo, at a release, the one the
    /// row was scored with.
    pub preds: &'a [f64],
    /// Each slot's residual, NaN where there is none.
    pub resid: &'a [f64],
    /// Each slot's error inflation for the row's features, where the model
    /// reads one (`ewridge`, `rls`, `kalman`); `None` stands in 1.
    pub inflation: Option<&'a [f64]>,
    /// The targets and the features.
    pub ys: &'a [Option<f64>],
    pub xs: &'a [f64],
    /// Slots per target: slot `s` belongs to target `s / nc`.
    pub nc: usize,
}

/// One model instance's task-221 accumulators, empty where their switch is
/// off. Persisted with the stream ([`crate::stream::Persisted::checks`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Checks {
    /// Mincer–Zarnowitz per slot, under `emit_calibration`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calibration: Vec<Calibration>,
    /// The studentized residuals' CUSUM sums per slot, under `emit_breaks`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub breaks: Vec<Breaks>,
    /// The fast and slow fits per target, under `emit_breaks`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub twin: Vec<TwinFit>,
    /// The robust covariances' sums per slot, under `emit_robust_se`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sandwich: Vec<Sandwich>,
}

impl Checks {
    /// Fresh accumulators for `n_slots` slots, as `spec` switches them on.
    pub fn new(spec: &Spec, n_slots: usize) -> Self {
        let on = |flag: bool, n: usize| if flag { n } else { 0 };
        Self {
            calibration: vec![Calibration::new(); on(spec.emit_calibration, n_slots)],
            breaks: vec![Breaks::new(); on(spec.emit_breaks, n_slots)],
            twin: vec![TwinFit::new(spec.k()); on(spec.emit_breaks, spec.m())],
            sandwich: vec![
                Sandwich::new(
                    spec.k(),
                    spec.fit_intercept,
                    spec.robust_se_lags_or_default()
                );
                on(spec.emit_robust_se, n_slots)
            ],
        }
    }

    /// Whether any switch is on: a spec with none keeps no `Checks`.
    pub fn any(spec: &Spec) -> bool {
        spec.emit_calibration || spec.emit_breaks || spec.emit_robust_se
    }

    /// Whether a restored set is shaped as `fresh`, this spec's: the same
    /// accumulators, each shaped to be read and folded.
    pub fn fits(&self, fresh: &Self, spec: &Spec) -> bool {
        let (k, lags) = (spec.k(), spec.robust_se_lags_or_default());
        self.calibration.len() == fresh.calibration.len()
            && self.breaks.len() == fresh.breaks.len()
            && self.twin.len() == fresh.twin.len()
            && self.sandwich.len() == fresh.sandwich.len()
            && self.calibration.iter().all(Calibration::has_shape)
            && self.breaks.iter().all(Breaks::has_shape)
            && self.twin.iter().all(|t| t.has_shape(k))
            && self
                .sandwich
                .iter()
                .all(|s| s.has_shape(k, spec.fit_intercept, lags))
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.sandwich.iter_mut().for_each(Sandwich::clear_lags);
    }

    /// Every slot's robust standard errors, flattened in `coef`'s layout,
    /// HC0's or (`hac`) Newey and West's: `None` until any slot has one.
    pub fn robust_se(&self, cfg: &CheckCfg, hac: bool) -> Option<Vec<f64>> {
        if self.sandwich.is_empty() {
            return None;
        }
        let nc = cfg.combo_features.len().max(1);
        let width = cfg.width;
        let mut any = false;
        let mut out = Vec::with_capacity(self.sandwich.len() * width);
        for (slot, s) in self.sandwich.iter().enumerate() {
            match cfg
                .combo_features
                .get(slot % nc)
                .and_then(|idx| s.standard_errors(idx, hac))
            {
                Some(v) => {
                    any = true;
                    out.extend(v);
                }
                None => out.extend(std::iter::repeat_n(f64::NAN, width)),
            }
        }
        any.then_some(out)
    }

    /// Each slot's recursive residual into `v`, under `emit_breaks`:
    /// `resid / error_inflation`, 1 standing in for the inflation on a model
    /// without one, and NaN where there is no residual or the inflation is
    /// not finite (a model that has not solved). Its studentized residual
    /// is [`Breaks::studentized`]'s, against the spread before the row.
    pub fn recursive(&self, row: &RowView<'_>, v: &mut Vec<f64>) {
        v.clear();
        if self.breaks.is_empty() {
            return;
        }
        v.extend(row.resid.iter().enumerate().map(|(slot, &r)| {
            let infl = row
                .inflation
                .map_or(1.0, |v| v.get(slot).copied().unwrap_or(f64::NAN));
            if r.is_finite() && infl.is_finite() && infl > 0.0 {
                r / infl
            } else {
                f64::NAN
            }
        }));
    }

    /// Write the values each slot reads before the row -- the row's own
    /// studentized residual, from its recursive residual `rec`, beside the
    /// sums before it -- value `v` of slot `s` at `(v·n_slots + s)·n_rows +
    /// ri` of `out`, the instance's block.
    pub fn read(
        &self,
        cfg: &CheckCfg,
        rec: &[f64],
        n_slots: usize,
        n_rows: usize,
        ri: usize,
        out: &mut [f64],
    ) {
        let mut put = |v: usize, slot: usize, x: Option<f64>| {
            out[(v * n_slots + slot) * n_rows + ri] = x.unwrap_or(f64::NAN);
        };
        let mut v = 0;
        if !self.calibration.is_empty() {
            for (slot, c) in self.calibration.iter().enumerate() {
                put(v, slot, c.slope());
                put(v + 1, slot, c.intercept());
                put(v + 2, slot, c.wald());
            }
            v += 3;
        }
        if !self.breaks.is_empty() {
            let nc = cfg.combo_features.len().max(1);
            for (slot, b) in self.breaks.iter().enumerate() {
                put(v, slot, rec.get(slot).and_then(|&r| b.studentized(r)));
                put(v + 1, slot, b.cusum());
                put(v + 2, slot, b.cusum_sq());
                let wald = self
                    .twin
                    .get(slot / nc)
                    .and_then(|t| t.wald(cfg.combo_features.get(slot % nc)?, cfg.intercept));
                put(v + 3, slot, wald);
            }
        }
    }

    /// Fold one learned row: `row` and its recursive residuals `rec`, the
    /// row's clock step and weight. A slot with no prediction or no target,
    /// and a row of weight 0, only age (CLAUDE.md hard rule 9).
    pub fn learn(&mut self, cfg: &CheckCfg, d_clock: f64, row: &RowView<'_>, rec: &[f64], w: f64) {
        let nc = row.nc.max(1);
        if !self.calibration.is_empty() {
            let lam = cfg.calibration.factor(d_clock);
            for (slot, c) in self.calibration.iter_mut().enumerate() {
                let y = row.ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
                c.update(row.preds[slot], y, lam, w);
            }
        }
        if !self.breaks.is_empty() {
            let lam = cfg.breaks.factor(d_clock);
            for (slot, b) in self.breaks.iter_mut().enumerate() {
                b.update(rec.get(slot).copied().unwrap_or(f64::NAN), lam, w);
            }
            for (t, fit) in self.twin.iter_mut().enumerate() {
                let y = row.ys.get(t).copied().flatten().unwrap_or(f64::NAN);
                fit.update(row.xs, y, lam, w);
            }
        }
        if !self.sandwich.is_empty() {
            let lam = cfg.robust.factor(d_clock);
            for (slot, s) in self.sandwich.iter_mut().enumerate() {
                s.update(
                    row.xs,
                    row.resid.get(slot).copied().unwrap_or(f64::NAN),
                    lam,
                    w,
                );
            }
        }
    }
}
