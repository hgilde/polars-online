//! The diagnostics with a memory of their own (docs/PLAN.md task 221):
//! each switch's memory key, and the decay it resolves to.

use online_core::Decay;

use super::Spec;
use crate::span::Span;

impl Spec {
    /// The diagnostics with a memory of their own (docs/PLAN.md task 221):
    /// each memory's key, its switch, whether the switch is on, and the
    /// memory as given.
    pub fn diagnostic_memories(&self) -> [(&'static str, &'static str, bool, Option<&Span>); 2] {
        [
            (
                "calibration_half_life",
                "emit_calibration",
                self.emit_calibration,
                self.calibration_half_life.as_ref(),
            ),
            (
                "breaks_half_life",
                "emit_breaks",
                self.emit_breaks,
                self.breaks_half_life.as_ref(),
            ),
        ]
    }

    /// A diagnostic's decay for the model instance decaying by `model`: its
    /// own half-life where given, else the instance's (task 221).
    pub fn diagnostic_decay(memory: Option<&Span>, model: Decay) -> Decay {
        memory.map_or(model, |h| Decay::Halflife(h.value()))
    }
}
