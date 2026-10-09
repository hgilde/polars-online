//! Task 221's diagnostics for one model instance: the accumulators that keep
//! a memory of their own, defaulting to the instance's half-life, `inf` the
//! run-once form (docs/PLAN.md task 221).
//!
//! Each reads the row's out-of-sample prediction and residual as the other
//! residual diagnostics do -- before the row folds into it, so a field never
//! includes the row it describes -- and each folds the row only when the
//! row is learned: under `embargo` at its release, with the prediction the
//! row was scored with. Their per-slot values ride in one output buffer,
//! [`crate::ChunkOut::checks`], `n_values` of them per slot.

use online_core::{Calibration, Decay};
use serde::{Deserialize, Serialize};

use crate::spec::Spec;

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
    out
}

/// What one instance's diagnostics decay by: each its own memory, or the
/// instance's.
#[derive(Debug, Clone, Copy)]
pub struct CheckDecays {
    calibration: Decay,
}

impl CheckDecays {
    pub fn of(spec: &Spec, model: Decay) -> Self {
        Self {
            calibration: Spec::diagnostic_decay(spec.calibration_half_life.as_ref(), model),
        }
    }
}

/// One model instance's task-221 accumulators, each per output slot, empty
/// where its switch is off. Persisted with the stream
/// ([`crate::stream::Persisted::checks`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Checks {
    /// Mincer–Zarnowitz per slot, under `emit_calibration`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calibration: Vec<Calibration>,
}

impl Checks {
    /// Fresh accumulators for `n_slots` slots, as `spec` switches them on.
    pub fn new(spec: &Spec, n_slots: usize) -> Self {
        Self {
            calibration: if spec.emit_calibration {
                vec![Calibration::new(); n_slots]
            } else {
                Vec::new()
            },
        }
    }

    /// Whether any switch is on: a spec with none keeps no `Checks`.
    pub fn any(spec: &Spec) -> bool {
        spec.emit_calibration
    }

    /// Whether a restored set is shaped as `fresh`, this spec's: the same
    /// accumulators, each shaped to be read and folded.
    pub fn fits(&self, fresh: &Self) -> bool {
        self.calibration.len() == fresh.calibration.len()
            && self.calibration.iter().all(Calibration::has_shape)
    }

    /// Write the values each slot reads before the row: value `v` of slot
    /// `s` at `(v·n_slots + s)·n_rows + ri` of `out`, the instance's block.
    pub fn read(&self, n_slots: usize, n_rows: usize, ri: usize, out: &mut [f64]) {
        let mut v = 0;
        let mut put = |v: usize, slot: usize, x: Option<f64>| {
            out[(v * n_slots + slot) * n_rows + ri] = x.unwrap_or(f64::NAN);
        };
        for (slot, c) in self.calibration.iter().enumerate() {
            put(v, slot, c.slope());
            put(v + 1, slot, c.intercept());
            put(v + 2, slot, c.wald());
        }
        if !self.calibration.is_empty() {
            v += 3;
        }
        let _ = v;
    }

    /// Fold one learned row: each slot's prediction (`preds`, slot `s`
    /// belonging to target `s / nc`), the targets, the row's clock step and
    /// weight. A slot with no prediction or no target, and a row of weight
    /// 0, only age (CLAUDE.md hard rule 9).
    pub fn learn(
        &mut self,
        decays: &CheckDecays,
        d_clock: f64,
        preds: &[f64],
        ys: &[Option<f64>],
        nc: usize,
        w: f64,
    ) {
        if !self.calibration.is_empty() {
            let lam = decays.calibration.factor(d_clock);
            for (slot, c) in self.calibration.iter_mut().enumerate() {
                let y = ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
                c.update(preds[slot], y, lam, w);
            }
        }
    }
}
