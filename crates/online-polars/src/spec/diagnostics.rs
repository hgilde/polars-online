//! The diagnostics with a memory of their own (docs/PLAN.md task 221):
//! each switch's memory key, and the decay it resolves to.

use online_core::Decay;

use super::Spec;
use crate::span::Span;

impl Spec {
    /// The diagnostics with a memory of their own (docs/PLAN.md task 221):
    /// each memory's key, its switch, whether the switch is on, and the
    /// memory as given.
    pub fn diagnostic_memories(&self) -> [(&'static str, &'static str, bool, Option<&Span>); 6] {
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
            (
                "specification_half_life",
                "emit_specification",
                self.emit_specification,
                self.specification_half_life.as_ref(),
            ),
            (
                "tails_half_life",
                "emit_tails",
                self.emit_tails,
                self.tails_half_life.as_ref(),
            ),
            (
                "influence_half_life",
                "emit_influence",
                self.emit_influence,
                self.influence_half_life.as_ref(),
            ),
        ]
    }

    /// Whether a switch reads the row's own error inflation (task 221):
    /// its field, and the recursive residual of the breaks and the tails.
    pub fn reads_row_inflation(&self) -> bool {
        self.emit_error_inflation || self.emit_breaks || self.emit_tails || self.emit_influence
    }

    /// The target's horizon in rows: `embargo`, rounded up, on a spec with
    /// no clock column, and 0 otherwise -- the rows a look-ahead target's
    /// residuals are correlated over by construction (task 221).
    pub fn horizon_rows(&self) -> usize {
        match (&self.clock, &self.embargo) {
            (None, Some(Span::Units(h))) if h.is_finite() && *h > 0.0 => {
                (h.ceil() as usize).min(online_core::MAX_LAG)
            }
            _ => 0,
        }
    }

    /// The lags Ljung and Box skip: the horizon's `h − 1`, inside which a
    /// look-ahead target's residuals share their shocks (task 221 (d)).
    pub fn ljung_box_skip(&self) -> usize {
        self.horizon_rows().saturating_sub(1)
    }

    /// Ljung and Box's lags past the skipped ones: 10 unless set.
    pub fn ljung_box_lags_or_default(&self) -> usize {
        self.ljung_box_lags.unwrap_or(10)
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

    /// How many of the instance's half-lives a diagnostic's memory is when
    /// its `*_half_life` is left out (task 221): 4 for the calibration, 1
    /// for the others.
    ///
    /// The calibration's 4: a fit that forgets absorbs a miscalibration at
    /// its own pace, so a calibration read at the fit's memory is
    /// conservative. Measured beside `ewridge` at half-lives of 50 and 200
    /// (400 streams of 3,000 rows), Wald's statistic passed 5.99, its 5%
    /// value, on 0.3-0.6% of the rows of calibrated fits at 1x and on 0-0.3%
    /// at 4x; on a weak fit whose slope was 0.7, on 5% of rows at 1x and
    /// 42% at 4x (89% run once), and on one at 0.91, 2% and 4%. The breaks keep 1x: at 4x their `cusum`
    /// passed 1.96 on 0.03% of rows with no break (0.6% at 1x) and found an
    /// intercept break in 224 rows rather than 100, and `cusum_sq` a
    /// variance break in 36 rather than 20 -- a longer memory only slowed
    /// them. The specification tests keep 1x: they read the residuals'
    /// second moments, which a fit does not absorb, and passed their 5%
    /// values on 4.2-6.1% of no-break rows at 1x and 4x alike, finding what
    /// they look for on 98-100% of rows at either. So do the tails: Jarque
    /// and Bera's statistic passed 5.99 on 4.7-5.0% of the rows of Gaussian
    /// residuals at 1x, 4x and run once, and on every row of Student's t
    /// with 5 degrees of freedom at each. The influence's scale too: with
    /// one row in 200 planted at six spreads out and six off the line, its
    /// clean rows' 99.9th percentile read 0.29 at 1x and 4x, and the planted
    /// rows' median 1.19 and 1.18.
    pub fn memory_multiple(key: &str) -> u32 {
        if key == "calibration_half_life" { 4 } else { 1 }
    }

    /// A diagnostic's decay for the model instance decaying by `model`: its
    /// own half-life where given, else `multiple` of the instance's, `inf`
    /// staying `inf` (task 221). A `lam` decay's multiple of 4 is its fourth
    /// root, two square roots, which every libm rounds exactly.
    pub fn diagnostic_decay(memory: Option<&Span>, model: Decay, multiple: u32) -> Decay {
        if let Some(h) = memory {
            return Decay::Halflife(h.value());
        }
        match (model, multiple) {
            (_, 1) => model,
            (Decay::Halflife(h), m) => Decay::Halflife(h * f64::from(m)),
            (Decay::Lam(l), 4) => Decay::Lam(l.sqrt().sqrt()),
            (Decay::Lam(l), 2) => Decay::Lam(l.sqrt()),
            (Decay::Lam(l), m) => Decay::Lam(l.powf(1.0 / f64::from(m))),
        }
    }
}
