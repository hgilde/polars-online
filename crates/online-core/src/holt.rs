//! Holt's linear trend method (docs/ENHANCEMENTS.md E25).
//!
//! The forecasting baseline every other model here should have to beat: no
//! features at all, just a level and a slope extrapolated forward. If a
//! regression cannot outperform "the series is going up at about this rate",
//! the features are not earning their place.
//!
//! Per row, with clock delta `d` and halflife-derived rates
//! `alpha = 1 − 0.5^(d/level_halflife)`, `beta = 1 − 0.5^(d/trend_halflife)`:
//!
//! ```text
//! pred  = l + b·d                       (extrapolate d clock units ahead)
//! l'    = alpha·y + (1 − alpha)·pred
//! b'    = beta·(l' − l)/d + (1 − beta)·b
//! ```
//!
//! Deriving the rates from halflives, rather than taking them directly, keeps
//! the parameter meaning the same as everywhere else in this library: a
//! halflife is in clock units, so an irregular clock is handled correctly
//! instead of every row counting the same. `trend_halflife = inf` pins the
//! trend at zero, which reduces this to a plain exponentially weighted level.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoltCfg {
    pub n_targets: usize,
    /// Halflife of the level, in clock units.
    pub level_halflife: f64,
    /// Halflife of the trend. `inf` pins the trend at zero, giving a plain
    /// EW level.
    pub trend_halflife: f64,
    pub min_periods: f64,
}

impl HoltCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err("holt: n_targets must be >= 1".into());
        }
        if self.level_halflife <= 0.0 || self.level_halflife.is_nan() {
            return Err("holt: level_halflife must be > 0".into());
        }
        if self.trend_halflife <= 0.0 || self.trend_halflife.is_nan() {
            return Err("holt: trend_halflife must be > 0 (inf pins the trend)".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Holt {
    cfg: HoltCfg,
    level: Vec<f64>,
    trend: Vec<f64>,
    seen: Vec<bool>,
    w_sum: f64,
}

impl Holt {
    pub fn new(cfg: HoltCfg) -> Result<Self, String> {
        cfg.validate()?;
        let m = cfg.n_targets;
        Ok(Self {
            level: vec![0.0; m],
            trend: vec![0.0; m],
            seen: vec![false; m],
            w_sum: 0.0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &HoltCfg {
        &self.cfg
    }

    pub fn level(&self) -> &[f64] {
        &self.level
    }

    pub fn trend(&self) -> &[f64] {
        &self.trend
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Reported as `coef`: `[level, trend]` per target, which is the whole
    /// state and the only thing worth inspecting.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| vec![self.level[j], self.trend[j]])
            .collect()
    }
}

impl OnlineModel for Holt {
    fn step(&mut self, _x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let m = self.cfg.n_targets;
        let alpha = 1.0 - 0.5f64.powf(d_clock / self.cfg.level_halflife);
        let beta = if self.cfg.trend_halflife.is_infinite() {
            0.0
        } else {
            1.0 - 0.5f64.powf(d_clock / self.cfg.trend_halflife)
        };

        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_periods;
        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            if !self.seen[j] {
                // Nothing to extrapolate from yet.
                if let Some(yj) = y[j] {
                    if yj.is_finite() && weight > 0.0 {
                        self.level[j] = yj;
                        self.seen[j] = true;
                    }
                }
                continue;
            }
            // Extrapolate over this row's own elapsed clock, so an irregular
            // clock forecasts the right distance ahead.
            let p = self.level[j] + self.trend[j] * d_clock;
            if ready {
                pred[j] = p;
            }
            let Some(yj) = y[j] else { continue };
            if !yj.is_finite() || weight <= 0.0 {
                continue;
            }
            let prev_level = self.level[j];
            self.level[j] = alpha * yj + (1.0 - alpha) * p;
            // `d_clock > 0.0` cannot be the deciding condition -- beta is
            // `1 - 0.5^(d/halflife)`, which is 0 whenever d is -- but it is the
            // guard that names why the division below is safe, so both stay.
            if beta > 0.0 && d_clock > 0.0 {
                let observed_slope = (self.level[j] - prev_level) / d_clock;
                self.trend[j] = beta * observed_slope + (1.0 - beta) * self.trend[j];
            }
        }
        self.w_sum = self.w_sum * 0.5f64.powf(d_clock / self.cfg.level_halflife) + weight;

        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn predict(&self, _x: &[f64], d_clock: f64) -> Step {
        let n_eff = self.w_sum;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        if n_eff >= self.cfg.min_periods {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.seen[j] {
                    *p = self.level[j] + self.trend[j] * d_clock;
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Holt(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Holt(m) => Ok((**m).clone()),
            other => Err(StateError::WrongModel {
                expected: "holt",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// Holt uses no features.
    fn n_features(&self) -> usize {
        0
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

    fn cfg(level: f64, trend: f64) -> HoltCfg {
        HoltCfg {
            n_targets: 1,
            level_halflife: level,
            trend_halflife: trend,
            min_periods: 0.0,
        }
    }

    fn run(cfg: HoltCfg, ys: &[f64], d: f64) -> (Vec<f64>, Holt) {
        let mut m = Holt::new(cfg).unwrap();
        let mut preds = Vec::new();
        for (i, y) in ys.iter().enumerate() {
            let step = m.step(&[], &[Some(*y)], if i == 0 { 0.0 } else { d }, 1.0);
            preds.push(step.pred[0]);
        }
        (preds, m)
    }

    /// The recursion written out longhand, against an irregular clock, so the
    /// test cannot share a mistake with the implementation.
    fn reference(cfg: &HoltCfg, ys: &[f64], ds: &[f64]) -> (Vec<f64>, f64, f64, f64) {
        let (mut level, mut trend, mut w_sum) = (0.0, 0.0, 0.0);
        let mut seen = false;
        let mut preds = Vec::new();
        for (&y, &d) in ys.iter().zip(ds) {
            let decay = 0.5f64.powf(d / cfg.level_halflife);
            let alpha = 1.0 - decay;
            let beta = if cfg.trend_halflife.is_infinite() {
                0.0
            } else {
                1.0 - 0.5f64.powf(d / cfg.trend_halflife)
            };
            if !seen {
                level = y;
                seen = true;
                preds.push(f64::NAN);
            } else {
                let p = level + trend * d;
                preds.push(if w_sum >= cfg.min_periods {
                    p
                } else {
                    f64::NAN
                });
                let prev = level;
                level = alpha * y + (1.0 - alpha) * p;
                if beta > 0.0 && d > 0.0 {
                    trend = beta * ((level - prev) / d) + (1.0 - beta) * trend;
                }
            }
            w_sum = w_sum * decay + 1.0;
        }
        (preds, level, trend, w_sum)
    }

    #[test]
    fn every_step_matches_the_recursion_written_out() {
        // Pins each arithmetic step: the extrapolation distance, the level
        // blend, the per-clock-unit slope, the trend blend, and the decayed
        // weight -- on a clock whose gaps vary, so `d` cannot cancel out.
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let mut s = 5u64;
        let ys: Vec<f64> = (0..ds.len())
            .map(|i| 3.0 + 0.8 * i as f64 + lcg(&mut s))
            .collect();
        let c = HoltCfg {
            n_targets: 1,
            level_halflife: 4.0,
            trend_halflife: 9.0,
            min_periods: 2.5,
        };
        let (want_pred, want_level, want_trend, want_w) = reference(&c, &ys, &ds);

        let mut m = Holt::new(c).unwrap();
        for (i, (&y, &d)) in ys.iter().zip(&ds).enumerate() {
            let step = m.step(&[], &[Some(y)], d, 1.0);
            match (step.pred[0].is_nan(), want_pred[i].is_nan()) {
                (true, true) => {}
                (false, false) => assert!(
                    (step.pred[0] - want_pred[i]).abs() < 1e-12,
                    "row {i}: {} vs {}",
                    step.pred[0],
                    want_pred[i]
                ),
                _ => panic!("row {i}: {} vs {}", step.pred[0], want_pred[i]),
            }
        }
        assert!((m.level()[0] - want_level).abs() < 1e-12);
        assert!((m.trend()[0] - want_trend).abs() < 1e-12);
        assert!((m.n_eff() - want_w).abs() < 1e-12);
        assert_eq!(m.coefficients(), vec![vec![want_level, want_trend]]);
    }

    #[test]
    fn n_eff_decays_on_the_clock_and_gates_output() {
        let c = HoltCfg {
            n_targets: 1,
            level_halflife: 10.0,
            trend_halflife: 40.0,
            min_periods: 3.0,
        };
        let mut m = Holt::new(c).unwrap();
        assert_eq!(m.n_eff(), 0.0, "nothing seen yet");

        // n_eff is the weight before the row, so it lags the row count by one.
        let mut want = 0.0;
        for i in 0..6 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let step = m.step(&[], &[Some(i as f64)], d, 1.0);
            assert!((step.n_eff - want).abs() < 1e-12, "row {i}");
            assert_eq!(
                step.pred[0].is_nan(),
                want < 3.0,
                "row {i}: gated at n_eff = {want}"
            );
            want = want * 0.5f64.powf(d / 10.0) + 1.0;
        }

        // A long gap decays it rather than resetting it.
        let before = m.n_eff();
        m.step(&[], &[Some(99.0)], 100.0, 1.0);
        let after = m.n_eff();
        assert!(
            (after - (before * 0.5f64.powf(10.0) + 1.0)).abs() < 1e-12,
            "{before} -> {after}"
        );
    }

    #[test]
    fn the_level_is_seeded_only_by_a_usable_first_row() {
        // Two independent reasons a row cannot seed the level: the target is
        // not a number, or the row carries no weight. Either alone must leave
        // the model unseeded, still predicting nothing.
        for (y, w) in [
            (Some(f64::NAN), 1.0),
            (Some(f64::INFINITY), 1.0),
            (None, 1.0),
            (Some(5.0), 0.0),
            (Some(5.0), -1.0),
        ] {
            let mut m = Holt::new(cfg(10.0, 40.0)).unwrap();
            let step = m.step(&[], &[y], 0.0, w);
            assert!(step.pred[0].is_nan(), "({y:?}, {w}) must not predict");
            assert!(!m.seen[0], "({y:?}, {w}) must not seed the level");
            assert_eq!(m.level()[0], 0.0);

            // A usable row afterwards still seeds it, at its own value.
            m.step(&[], &[Some(9.0)], 1.0, 1.0);
            assert!(m.seen[0]);
            assert_eq!(m.level()[0], 9.0, "the first usable row seeds the level");
        }
    }

    #[test]
    fn a_zero_gap_leaves_the_trend_alone() {
        // With d = 0 the observed slope would divide by zero, so the trend must
        // be held; the level still updates, but alpha is 0 at d = 0, so the
        // repeated timestamp changes nothing at all.
        let c = cfg(5.0, 20.0);
        let mut m = Holt::new(c).unwrap();
        for i in 0..30 {
            m.step(
                &[],
                &[Some(2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        assert!(t > 0.5, "there should be a trend to preserve: {t}");
        m.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        assert_eq!(m.trend()[0], t, "a zero gap must not move the trend");
        assert_eq!(m.level()[0], l, "alpha is 0 at d = 0");
    }

    #[test]
    fn targets_are_independent() {
        // `n_targets` and the per-target loop: two targets with unrelated
        // series must not contaminate each other, and both must be reported.
        let c = HoltCfg {
            n_targets: 2,
            level_halflife: 8.0,
            trend_halflife: 30.0,
            min_periods: 0.0,
        };
        let mut m = Holt::new(c).unwrap();
        for i in 0..600 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let (a, b) = (10.0 + 3.0 * i as f64, 500.0 - 1.0 * i as f64);
            m.step(&[], &[Some(a), Some(b)], d, 1.0);
        }
        let coef = m.coefficients();
        assert_eq!(coef.len(), 2, "one [level, trend] pair per target");
        assert_eq!(coef[0].len(), 2);
        assert!(
            (coef[0][1] - 3.0).abs() < 0.05,
            "target 0 trend {}",
            coef[0][1]
        );
        assert!(
            (coef[1][1] + 1.0).abs() < 0.05,
            "target 1 trend {}",
            coef[1][1]
        );
        assert!(coef[0][0] > 1000.0, "levels are far apart: {}", coef[0][0]);
        assert!(
            coef[1][0] < 0.0,
            "target 1 has fallen below zero: {}",
            coef[1][0]
        );
    }

    #[test]
    fn a_null_target_predicts_without_learning() {
        // The `let Some(yj) = y[j] else { continue }` arm: prediction still
        // happens, state does not move -- which is what makes the output
        // out-of-sample.
        let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
        for i in 0..40 {
            m.step(
                &[],
                &[Some(1.0 + 2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        let step = m.step(&[], &[None], 1.0, 1.0);
        assert!((step.pred[0] - (l + t)).abs() < 1e-12, "still extrapolates");
        assert_eq!(m.level()[0], l);
        assert_eq!(m.trend()[0], t);
    }

    #[test]
    fn a_zero_weight_row_is_pure_decay() {
        let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
        for i in 0..40 {
            m.step(
                &[],
                &[Some(1.0 + 2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t, w) = (m.level()[0], m.trend()[0], m.n_eff());
        m.step(&[], &[Some(-500.0)], 1.0, 0.0);
        assert_eq!(m.level()[0], l, "weight 0 must not fold the row in");
        assert_eq!(m.trend()[0], t);
        assert!((m.n_eff() - w * 0.5f64.powf(1.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn tracks_a_constant_level() {
        let ys: Vec<f64> = vec![7.0; 200];
        let (preds, m) = run(cfg(10.0, 20.0), &ys, 1.0);
        assert!((preds[199] - 7.0).abs() < 1e-9);
        assert!(m.trend()[0].abs() < 1e-9, "no trend in a flat series");
    }

    #[test]
    fn extrapolates_a_linear_trend() {
        // y = 3 + 2t. A level-only model would always lag; with a trend the
        // one-step-ahead prediction should be right.
        let ys: Vec<f64> = (0..500).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, m) = run(cfg(5.0, 5.0), &ys, 1.0);
        assert!(
            (m.trend()[0] - 2.0).abs() < 0.05,
            "trend should be ~2, got {}",
            m.trend()[0]
        );
        assert!(
            (preds[499] - ys[499]).abs() < 0.5,
            "should predict the next value closely: {} vs {}",
            preds[499],
            ys[499]
        );
    }

    #[test]
    fn a_pinned_trend_is_a_plain_ew_level() {
        let ys: Vec<f64> = (0..500).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, m) = run(cfg(5.0, f64::INFINITY), &ys, 1.0);
        assert_eq!(m.trend()[0], 0.0, "an infinite trend halflife pins it");
        // and it therefore lags a trending series badly
        assert!(preds[499] < ys[499] - 1.0, "a level-only fit must lag");
    }

    #[test]
    fn an_irregular_clock_extrapolates_the_right_distance() {
        // Same series sampled every 5 clock units: the trend is per clock unit,
        // so the prediction must step 5 units ahead, not 1.
        let mut m = Holt::new(cfg(20.0, 20.0)).unwrap();
        let mut last = f64::NAN;
        for i in 0..400 {
            let t = i as f64 * 5.0;
            let y = 3.0 + 2.0 * t;
            let step = m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 5.0 }, 1.0);
            last = step.pred[0];
        }
        let truth = 3.0 + 2.0 * (399.0 * 5.0);
        assert!(
            (last - truth).abs() / truth < 0.01,
            "expected ~{truth}, got {last}"
        );
        assert!(
            (m.trend()[0] - 2.0).abs() < 0.1,
            "trend per clock unit: {}",
            m.trend()[0]
        );
    }

    #[test]
    fn first_value_seeds_the_level() {
        let (preds, m) = run(cfg(10.0, 10.0), &[42.0, 42.0], 1.0);
        assert!(
            preds[0].is_nan(),
            "nothing to predict from on the first row"
        );
        assert!((m.level()[0] - 42.0).abs() < 1e-9);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Holt::new(cfg(10.0, 10.0)).unwrap();
        for i in 0..20 {
            m.step(&[], &[Some(i as f64)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = (m.level()[0], m.trend()[0]);
        let step = m.step(&[], &[None], 1.0, 1.0);
        assert!(step.pred[0].is_finite());
        assert_eq!((m.level()[0], m.trend()[0]), before);
    }

    #[test]
    fn state_roundtrip() {
        let ys: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.3).sin() + 0.1 * i as f64)
            .collect();
        let (_, m1) = run(cfg(10.0, 30.0), &ys[..50], 1.0);
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Holt::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        let mut m1 = m1;
        for y in &ys[50..] {
            assert_eq!(
                m1.step(&[], &[Some(*y)], 1.0, 1.0).pred,
                m2.step(&[], &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(Holt::new(cfg(0.0, 10.0)).is_err());
        assert!(Holt::new(cfg(10.0, 0.0)).is_err());
    }
}
