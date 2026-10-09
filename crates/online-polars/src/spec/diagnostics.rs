//! The diagnostics with a memory of their own (docs/PLAN.md task 221):
//! each switch's memory key, and the decay it resolves to.

use online_core::Decay;

use super::Spec;
use crate::span::Span;

impl Spec {
    /// The diagnostics with a memory of their own (docs/PLAN.md task 221):
    /// each memory's key, its switch, whether the switch is on, and the
    /// memory as given.
    pub fn diagnostic_memories(&self) -> [(&'static str, &'static str, bool, Option<&Span>); 7] {
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
            (
                "feature_health_half_life",
                "emit_feature_health",
                self.emit_feature_health,
                self.feature_health_half_life.as_ref(),
            ),
        ]
    }

    /// Whether a switch reads the row's own error inflation (task 221):
    /// its field, and the recursive residual of the breaks and the tails.
    pub fn reads_row_inflation(&self) -> bool {
        self.emit_error_inflation || self.emit_breaks || self.emit_tails || self.emit_influence
    }

    /// The target's horizon in rows, `h`: the rows a look-ahead target's
    /// residuals are correlated over by construction (task 221). One number
    /// for every diagnostic that reads it (task 232 (3); review round 6,
    /// B-1, B-2, A-4, A-8, F-4, G-5): `horizon_rows` where given; else
    /// `embargo`, rounded up, on a spec with no clock column, where a row is
    /// a unit; else 0. On a clock column `embargo` is in clock units and the
    /// rows it spans are not known before the stream, so a spec there gives
    /// `horizon_rows` itself, and a notice says so where it does not
    /// ([`Self::horizon_notices`]).
    pub fn horizon(&self) -> usize {
        if let Some(h) = self.horizon_rows {
            return h.min(online_core::MAX_LAG);
        }
        match (&self.clock, &self.embargo) {
            (None, Some(Span::Units(h))) if h.is_finite() && *h > 0.0 => {
                (h.ceil() as usize).min(online_core::MAX_LAG)
            }
            _ => 0,
        }
    }

    /// Newey and West's lags for the diagnostics under a horizon: twice it,
    /// as `se_coef_hac`'s default ([`Self::robust_se_lags_or_default`] has
    /// the measurement behind twice), `0` without one (task 232 (3)).
    pub fn nw_lags(&self) -> usize {
        (2 * self.horizon()).min(online_core::MAX_LAG)
    }

    /// The diagnostics that read the horizon and are on: each switch's
    /// name and what it does with the horizon (task 232 (3)).
    pub fn horizon_readers(&self) -> Vec<(&'static str, &'static str)> {
        [
            (
                self.emit_calibration,
                "emit_calibration",
                "the calibration's Wald test takes Newey and West's variance",
            ),
            (
                self.emit_breaks,
                "emit_breaks",
                "the CUSUMs and break_wald take the long-run variance",
            ),
            (
                self.emit_specification,
                "emit_specification",
                "Ljung-Box skips the lags inside it, and Breusch-Pagan and RESET take Newey \
                 and West's variance",
            ),
            (
                self.emit_robust_se,
                "emit_robust_se",
                "se_coef_hac's lags default to twice it",
            ),
        ]
        .into_iter()
        .filter(|(on, _, _)| *on)
        .map(|(_, name, what)| (name, what))
        .collect()
    }

    /// What a spec on a clock column with an `embargo` and no
    /// `horizon_rows` is told, once per diagnostic that reads the horizon
    /// (task 232 (3); review round 6, B-2, A-8, F-4, G-5): its horizon is
    /// 0, so a look-ahead target's overlapping residuals read as a fault.
    pub fn horizon_notices(&self) -> Vec<String> {
        if self.clock.is_none() || self.embargo.is_none() || self.horizon_rows.is_some() {
            return Vec::new();
        }
        self.horizon_readers()
            .into_iter()
            .map(|(name, what)| {
                format!(
                    "{name} reads the target's horizon -- the rows a look-ahead target's \
                     residuals overlap -- and on a clock column it cannot be read from embargo, \
                     which is in clock units, so it is 0: under a horizon {what}, and without it \
                     a target that looks ahead flags the test on most rows. Give horizon_rows, \
                     the rows the target looks ahead (0 if it does not; docs/DIAGNOSTICS.md)."
                )
            })
            .collect()
    }

    /// The level a quantile fit predicts -- `quantile`'s, or `sgd`'s under
    /// `loss="quantile"` -- whose calibration is a coverage and whose CUSUM
    /// sums indicators (task 232 (6); review round 6, G-1, F-2). `None` for
    /// a fit of the mean.
    pub fn quantile_level(&self) -> Option<f64> {
        match &self.model {
            super::ModelKind::Quantile { quantile, .. } => Some(*quantile),
            super::ModelKind::Sgd { loss, quantile, .. } if loss.as_deref() == Some("quantile") => {
                *quantile
            }
            _ => None,
        }
    }

    /// The lags Ljung and Box skip: the horizon's `h − 1`, inside which a
    /// look-ahead target's residuals share their shocks (task 221 (d)).
    pub fn ljung_box_skip(&self) -> usize {
        self.horizon().saturating_sub(1)
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
    /// target's horizon in rows ([`Self::horizon`]: `horizon_rows`, or
    /// `embargo` rounded up on a spec with no clock column, where a row is a
    /// unit -- a look-ahead target in a fit needs an embargo as long as its
    /// window); else `0`, HC0 alone.
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
        if self.horizon_rows.is_some() {
            return self.nw_lags();
        }
        match (&self.clock, &self.embargo) {
            (None, Some(Span::Units(h))) if h.is_finite() && *h > 0.0 => {
                ((2.0 * h).ceil() as usize).min(online_core::MAX_LAG)
            }
            _ => 0,
        }
    }

    /// How many of the fit's memories ([`Self::fit_memory`]: the instance's
    /// half-life but beside a window or a `kalman`) a diagnostic's memory is
    /// when its `*_half_life` is left out (task 221): 4 for the calibration,
    /// 1 for the others.
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
    /// rows' median 1.19 and 1.18. The feature health keeps 1x for its
    /// fast memory, its slow one four times that: on docs/DATA-ISSUES.md's
    /// frozen feed (half-life 20) the held feature's spread ratio fell below
    /// 0.5 2.9 half-lives in at 1x and 14 at 4x, where `rls`'s slope passed
    /// 1 at 52.6; and a one-spread move read `|mean_shift| > 0.3` within 190
    /// rows at 1x (half-life 200) and never at 4x.
    pub fn memory_multiple(key: &str) -> u32 {
        if key == "calibration_half_life" { 4 } else { 1 }
    }

    /// The fit's own memory, which a diagnostic's default memory is a
    /// multiple of (task 232 (2); review round 6, A-2, A-3, B-4, G-4, F-5):
    /// the instance's decay `model`, but for two models whose fit forgets
    /// on another clock.
    ///
    /// - **Under `window_size`** the half-life whose exponential weights
    ///   have the window's Kish size. A window of `W` rows holds `W` rows of
    ///   Kish's size; weights `λ^i` hold `(1 + λ) / (1 − λ)`, so
    ///   `λ = (W − 1) / (W + 1)` and the half-life is
    ///   `ln 2 / ln((W + 1) / (W − 1))`, about `W / 2.885` (`W ln 2 / 2`).
    ///   Beside a decay `λ` as well, the window's weights `λ^i, i < W`, have
    ///   Kish's size `n = (1 + λ)(1 − λ^W) / ((1 − λ)(1 + λ^W))`, and the
    ///   half-life is the one whose weights have `n`. On a clock column the
    ///   window is `W` clock units and the rows' rate is unknown, so the
    ///   continuous form is taken: a decay rate `κ` (`ln 2` over the
    ///   half-life, 0 with none) cut at `W` has the Kish size of the rate
    ///   `κ / tanh(κ W / 2)`, which is `2 / W` with no decay -- a half-life
    ///   of `W ln 2 / 2` clock units.
    /// - **`kalman`** forgets its coefficients at `coef_half_life`, not at
    ///   the `half_life` its noise statistics read: the shortest finite one
    ///   where a list gives one per coefficient. A `q` given outright, or
    ///   every coefficient pinned (`inf`), leaves the instance's decay.
    pub fn fit_memory(&self, model: Decay) -> Decay {
        if let super::ModelKind::Kalman {
            coef_half_life: Some(list),
            ..
        } = &self.model
        {
            let h = list
                .to_vec()
                .into_iter()
                .filter(|h| h.is_finite() && *h > 0.0)
                .fold(f64::INFINITY, f64::min);
            if h.is_finite() {
                return Decay::Halflife(h);
            }
        }
        match self.model.window_parts() {
            Some((Some(w), _)) if w.is_finite() && w > 0.0 => {
                Decay::Halflife(window_half_life(w, model, self.clock.is_some()))
            }
            _ => model,
        }
    }

    /// A diagnostic's default memory, `multiple` of the fit's own
    /// ([`Self::fit_memory`]) for the instance decaying by `model`, or its
    /// own half-life `memory` where given.
    pub fn diagnostic_decay_of(&self, memory: Option<&Span>, model: Decay, multiple: u32) -> Decay {
        Self::diagnostic_decay(memory, self.fit_memory(model), multiple)
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

/// The half-life whose exponential weights have the Kish size of a window
/// of `w` -- rows, or clock units on a clock (`clocked`) -- under the
/// decay `model` ([`Spec::fit_memory`] has the formulas).
pub(crate) fn window_half_life(w: f64, model: Decay, clocked: bool) -> f64 {
    let ln2 = std::f64::consts::LN_2;
    // The decay rate per clock unit, 0 for none.
    let kappa = match model {
        Decay::Halflife(h) if h.is_finite() && h > 0.0 => ln2 / h,
        Decay::Lam(l) if l > 0.0 && l < 1.0 => -l.ln(),
        _ => 0.0,
    };
    if !clocked {
        // A window of `w` rows: Kish's size of its weights, exactly.
        let n = if kappa == 0.0 {
            w
        } else {
            let lam = (-kappa).exp();
            let lw = lam.powf(w);
            (1.0 + lam) * (1.0 - lw) / ((1.0 - lam) * (1.0 + lw))
        };
        if n > 1.0 {
            return ln2 / ((n + 1.0) / (n - 1.0)).ln();
        }
    }
    // The continuous form: the rate whose weights have the window's Kish
    // size, `κ / tanh(κ w / 2)`, `2 / w` with no decay.
    let rate = if kappa == 0.0 {
        2.0 / w
    } else {
        kappa / (kappa * w / 2.0).tanh()
    };
    ln2 / rate
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kish's size of a set of weights.
    fn kish(w: impl Iterator<Item = f64> + Clone) -> f64 {
        let s: f64 = w.clone().sum();
        s * s / w.map(|v| v * v).sum::<f64>()
    }

    /// The half-life a window resolves to has the window's Kish size, by
    /// the weights summed one by one: `W` rows, the decayed weights cut at
    /// `W` rows, and on a clock the continuous weights cut at `W` units.
    #[test]
    fn a_windows_half_life_has_its_kish_size() {
        let inf = Decay::Halflife(f64::INFINITY);
        for w in [12.0, 100.0, 500.0] {
            let h = window_half_life(w, inf, false);
            let lam = 0.5f64.powf(1.0 / h);
            let n = kish((0..200_000).map(|i| lam.powi(i)));
            assert!((n - w).abs() < 1e-6 * w, "{w}: {n}");
            let clocked = window_half_life(w, inf, true);
            assert!((clocked - w * std::f64::consts::LN_2 / 2.0).abs() < 1e-12 * w);
        }
        let (w, h) = (300.0, 100.0);
        let lam = 0.5f64.powf(1.0 / h);
        let want = kish((0..300).map(|i| lam.powi(i)));
        let got = window_half_life(w, Decay::Halflife(h), false);
        let mu = 0.5f64.powf(1.0 / got);
        let n = kish((0..200_000).map(|i| mu.powi(i)));
        assert!((n - want).abs() < 1e-6 * want, "{n} vs {want}");
        // On a clock: a rate's weights `e^(−κt)` on a fine grid, cut at `w`.
        let dt = 1e-3;
        let kappa = std::f64::consts::LN_2 / h;
        let cut = kish((0..(w / dt) as usize).map(|i| (-kappa * i as f64 * dt).exp())) * dt;
        let got = window_half_life(w, Decay::Halflife(h), true);
        let rate = std::f64::consts::LN_2 / got;
        let full =
            kish((0..(40.0 * got / dt) as usize).map(|i| (-rate * i as f64 * dt).exp())) * dt;
        assert!((cut - full).abs() < 1e-3 * cut, "{cut} vs {full}");
    }
}
