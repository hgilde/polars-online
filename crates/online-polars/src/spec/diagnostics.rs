//! The diagnostics with a memory of their own (docs/PLAN.md task 221):
//! each switch's memory key, and the decay it resolves to.

use online_core::Decay;

use super::Spec;
use crate::span::Span;

impl Spec {
    /// The diagnostics with a memory of their own (docs/PLAN.md task 221):
    /// each memory's key, its switch, whether the switch is on, and the
    /// memory as given.
    pub fn diagnostic_memories(&self) -> [(&'static str, &'static str, bool, Option<&Span>); 3] {
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
            (
                "robust_se_half_life",
                "emit_robust_se",
                self.emit_robust_se,
                self.robust_se_half_life.as_ref(),
            ),
        ]
    }

    /// Whether this spec's model takes `emit_robust_se`: the least-squares
    /// fits, whose coefficients the sandwich is of (task 221 (c)).
    pub fn has_robust_se(&self) -> bool {
        matches!(
            self.model,
            super::ModelKind::EwRidge { .. } | super::ModelKind::Rls { .. }
        )
    }

    /// Newey and West's lags: `robust_se_lags` where given; else twice the
    /// target's horizon in rows -- `embargo`, rounded up, on a spec with no
    /// clock column, where a row is a unit (a look-ahead target in a fit
    /// needs an embargo as long as its window); else `0`, HC0 alone.
    ///
    /// Twice, not once: a horizon of `h` rows correlates the residuals up
    /// to `h − 1` rows apart, and Bartlett's weights `1 − l/(L+1)` count a
    /// lag at less than its whole. Measured on overlapping labels with a
    /// persistent feature (task 221 (c)), the standard error read 83-97%
    /// of the coefficients' true spread at `L = h` and 90-104% at `2h`.
    pub fn robust_se_lags_or_default(&self) -> usize {
        if let Some(l) = self.robust_se_lags {
            return l;
        }
        match (&self.clock, &self.embargo) {
            (None, Some(Span::Units(h))) if h.is_finite() && *h > 0.0 => {
                ((2.0 * h).ceil() as usize).min(online_core::MAX_LAG)
            }
            _ => 0,
        }
    }

    /// A diagnostic's decay for the model instance decaying by `model`: its
    /// own half-life where given, else the instance's (task 221).
    pub fn diagnostic_decay(memory: Option<&Span>, model: Decay) -> Decay {
        memory.map_or(model, |h| Decay::Halflife(h.value()))
    }
}
