//! Holt's linear trend method (docs/ENHANCEMENTS.md E25).
//!
//! The forecasting baseline every other model here should have to beat: no
//! features at all, just a level and a slope extrapolated forward. If a
//! regression cannot outperform "the series is going up at about this rate",
//! the features are not earning their place.
//!
//! Per row and target, with `s` the clock since the target was last observed
//! -- this row's delta included -- `w` the row's weight, and `W`, `V` the
//! weight the level and the trend have gathered, each decayed on its own
//! halflife, `λ_l = 0.5^(s/level_halflife)` and `λ_b = 0.5^(s/trend_halflife)`:
//!
//! ```text
//! pred  = l + b·s                                (extrapolate s clock units ahead)
//! l'    = (λ_l·W·pred + w·y) / (λ_l·W + w)         W' = λ_l·W + w
//! b'    = (λ_b·V·b + w·(l' − l)/s) / (λ_b·V + w)   V' = λ_b·V + w
//! ```
//!
//! Level and trend are weighted means, of what the row observes and what the
//! state forecast, as every accumulator here is: a row at weight `w` counts
//! `w` times, `halflife = inf` forgets nothing and fits the whole history,
//! and the first observation seeds the level at its own value. At `w = 1`
//! with `W` saturated this is the textbook recursion, `α = 1 − λ_l`,
//! `β = 1 − λ_b`, and statsmodels' `Holt` agrees with it from there; before
//! that the gains `w/(λW + w)` are larger than the fixed ones, so a new
//! series is followed sooner. The textbook form read the weight only as
//! "learn or not" -- a row at weight 0.5 moved the level as far as one at 1
//! -- and at an infinite halflife its rate was 0, so the level froze at the
//! first row and a trend never learned (review 2026-09-12, S29/S30). So
//! `trend_halflife = inf` is the whole history's drift, not a trend pinned at
//! zero; and a row at the last row's clock (`s = 0`) is a second observation
//! the level takes in, where it had changed nothing -- the trend holds, since
//! a move over no clock has no slope.
//!
//! On a row that observes every target, `s` is the row's own delta `d`.
//! Deriving the decays from halflives keeps the parameter meaning the same
//! as everywhere else in this library: a halflife is in clock units, so an
//! irregular clock is handled correctly instead of every row counting the
//! same.
//!
//! **A row the model cannot learn from is transparent.** A null target, or a
//! zero weight, leaves the level and slope where the last observation put
//! them and adds the row's delta to that target's `s`, so the next observed
//! row forecasts, and decays, over the whole gap: the same numbers as if the
//! row were absent and its clock folded into the next one, which is what
//! `clock.rs` does with a row it skips. The level used to stand still across
//! such a row, so the row after it forecast one trend step short (review
//! 2026-09-12, C22).

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoltCfg {
    pub n_targets: usize,
    /// Halflife of the level, in clock units. `inf` forgets nothing: the
    /// level is the whole history's.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub level_halflife: f64,
    /// Halflife of the trend. `inf` forgets no slope: the trend is the whole
    /// history's drift (it pinned the trend at zero before review S30).
    #[serde(with = "crate::humanfloat::f64_or_tag")]
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
            return Err("holt: trend_halflife must be > 0 (inf forgets no slope)".into());
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
    /// Per target, the clock since its last observation, which a row it could
    /// not learn from adds to and the next observed row extrapolates over
    /// (see the module docs). A schema-6 state has none and loads with every
    /// target at zero: what the old recursion kept.
    #[serde(default)]
    since: Vec<f64>,
    /// Per target, the weight the level has gathered, decayed on the level's
    /// halflife over the clock since the target was last observed: `W` in
    /// the module docs.
    #[serde(default)]
    w_level: Vec<f64>,
    /// Per target, the weight the trend has gathered, on the trend's
    /// halflife: `V` in the module docs.
    #[serde(default)]
    w_trend: Vec<f64>,
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
            since: vec![0.0; m],
            w_level: vec![0.0; m],
            w_trend: vec![0.0; m],
            cfg,
        })
    }

    /// `(λ_l, λ_b)`, the level's and the trend's decay over `s` clock units
    /// since the last observation; an infinite halflife forgets nothing.
    fn decays(&self, s: f64) -> (f64, f64) {
        let f = |h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                (-(s / h)).exp2()
            }
        };
        (f(self.cfg.level_halflife), f(self.cfg.trend_halflife))
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
    /// Each target's weight is its level's, decayed over the clock since the
    /// target was last observed: the rows it was present on (S2).
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(
            self.w_level
                .iter()
                .zip(&self.since)
                .map(|(&w, &s)| w * self.decays(s).0),
        );
        true
    }

    fn step(&mut self, _x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let m = self.cfg.n_targets;
        // The decays for a target observed on the previous row, where `s` is
        // this row's delta: every target of a stream with no gaps.
        let row_decays = self.decays(d_clock);

        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_periods;
        let mut pred = vec![f64::NAN; m];
        for j in 0..m {
            let yj = y[j].filter(|v| v.is_finite() && weight > 0.0);
            if !self.seen[j] {
                // Nothing to extrapolate from yet: the first observation is
                // the level, at its own weight.
                if let Some(yj) = yj {
                    self.level[j] = yj;
                    self.w_level[j] = weight;
                    self.seen[j] = true;
                }
                continue;
            }
            // Extrapolate over the clock since the target was last observed,
            // so an irregular clock, or a gap in the target, forecasts the
            // right distance ahead.
            let s = self.since[j] + d_clock;
            let p = self.level[j] + self.trend[j] * s;
            if ready {
                pred[j] = p;
            }
            let Some(yj) = yj else {
                // Nothing to learn from: carry the clock to the next
                // observation (C22).
                self.since[j] = s;
                continue;
            };
            let (lam_l, lam_b) = if self.since[j] == 0.0 {
                row_decays
            } else {
                self.decays(s)
            };
            // Weighted means of what the row observes and what the state
            // held; `weight > 0` here, so neither denominator is 0 (hard
            // rule 9).
            let held = lam_l * self.w_level[j];
            let prev_level = self.level[j];
            self.level[j] = (held * p + weight * yj) / (held + weight);
            self.w_level[j] = held + weight;
            // A move over no clock has no slope: the trend holds.
            if s > 0.0 {
                let held = lam_b * self.w_trend[j];
                let slope = (self.level[j] - prev_level) / s;
                self.trend[j] = (held * self.trend[j] + weight * slope) / (held + weight);
                self.w_trend[j] = held + weight;
            }
            self.since[j] = 0.0;
        }
        self.w_sum = self.w_sum * row_decays.0 + weight;

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
                    *p = self.level[j] + self.trend[j] * (self.since[j] + d_clock);
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
            ModelState::Holt(m) => {
                let mut m = (**m).clone();
                let n = m.cfg.n_targets;
                if m.since.len() != n {
                    m.since = vec![0.0; n];
                }
                // The rest are checked, not repaired: a level or a weight
                // vector of the wrong length loaded and panicked on the
                // first `step` (review 2026-09-18, B3).
                if m.seen.len() != n
                    || [&m.level, &m.trend, &m.w_level, &m.w_trend]
                        .iter()
                        .any(|v| v.len() != n)
                {
                    return Err(StateError::Invalid(
                        "holt: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
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

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Holt::new(cfg(10.0, 20.0)).unwrap();
        let mut s = m.state();
        let ModelState::Holt(inner) = &mut s.model else {
            unreachable!()
        };
        inner.level.pop();
        match Holt::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

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
    /// test cannot share a mistake with the implementation: level and trend
    /// are each a weighted mean of what the row observes and what the state
    /// held, `(λ·W·old + w·new)/(λ·W + w)`, with `W` the weight each has
    /// gathered and `λ` its decay over the clock since the target was last
    /// observed (review 2026-09-12, S29/S30).
    fn reference(
        cfg: &HoltCfg,
        ys: &[Option<f64>],
        ds: &[f64],
        ws: &[f64],
    ) -> (Vec<f64>, f64, f64, f64) {
        let decay = |s: f64, h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                0.5f64.powf(s / h)
            }
        };
        let (mut level, mut trend, mut w_level, mut w_trend) = (0.0, 0.0, 0.0, 0.0);
        let (mut w_sum, mut since, mut seen) = (0.0, 0.0, false);
        let mut preds = Vec::new();
        for ((&y, &d), &w) in ys.iter().zip(ds).zip(ws) {
            let y = y.filter(|v| v.is_finite() && w > 0.0);
            if !seen {
                preds.push(f64::NAN);
                if let Some(y) = y {
                    (level, w_level, seen) = (y, w, true);
                }
            } else {
                let s = since + d;
                let p = level + trend * s;
                preds.push(if w_sum >= cfg.min_periods {
                    p
                } else {
                    f64::NAN
                });
                if let Some(y) = y {
                    let held = decay(s, cfg.level_halflife) * w_level;
                    let prev = level;
                    level = (held * p + w * y) / (held + w);
                    w_level = held + w;
                    if s > 0.0 {
                        let held = decay(s, cfg.trend_halflife) * w_trend;
                        trend = (held * trend + w * (level - prev) / s) / (held + w);
                        w_trend = held + w;
                    }
                    since = 0.0;
                } else {
                    since = s;
                }
            }
            w_sum = w_sum * decay(d, cfg.level_halflife) + w;
        }
        (preds, level, trend, w_sum)
    }

    #[test]
    fn every_step_matches_the_recursion_written_out() {
        // Pins each arithmetic step -- the extrapolation distance, both
        // weighted means, the per-clock-unit slope and the decayed weight --
        // on a clock whose gaps vary, rows whose weights vary (one of them
        // 0), and a null, so no factor can cancel out.
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        let c = HoltCfg {
            n_targets: 1,
            level_halflife: 4.0,
            trend_halflife: 9.0,
            min_periods: 2.5,
        };
        let (want_pred, want_level, want_trend, want_w) = reference(&c, &ys, &ds, &ws);

        let mut m = Holt::new(c).unwrap();
        for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
            let step = m.step(&[], &[y], d, w);
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

    /// A repeated timestamp is a second observation at the same time: the
    /// level takes it in, as a weighted mean takes any row, and the trend is
    /// held, since a move over no clock has no slope. The textbook form's
    /// rate was 0 at `d = 0`, so such a row changed nothing (S29).
    #[test]
    fn a_zero_gap_folds_the_row_into_the_level_and_holds_the_trend() {
        let c = cfg(5.0, 20.0);
        let mut ys: Vec<Option<f64>> = (0..30).map(|i| Some(2.0 * i as f64)).collect();
        let mut ds: Vec<f64> = (0..30).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut m = Holt::new(c.clone()).unwrap();
        for (y, d) in ys.iter().zip(&ds) {
            m.step(&[], &[*y], *d, 1.0);
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        assert!(t > 0.5, "there should be a trend to hold: {t}");
        m.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        assert_eq!(m.trend()[0], t, "a zero gap must not move the trend");
        assert!(
            m.level()[0] < l,
            "the row is taken in: {} vs {l}",
            m.level()[0]
        );
        ys.push(Some(-1000.0));
        ds.push(0.0);
        let ws = vec![1.0; ys.len()];
        let (_, want, _, _) = reference(&c, &ys, &ds, &ws);
        assert!(
            (m.level()[0] - want).abs() < 1e-9,
            "{} vs {want}",
            m.level()[0]
        );
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
        // Prediction still happens; the level and slope stay where the last
        // observation put them, and the row's clock is carried to the next
        // one (`a_row_it_cannot_learn_from_is_as_if_absent`).
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
        assert_eq!(m.since[0], 1.0, "the clock is carried");
    }

    /// Weight 0 folds nothing in: the level and slope stay, the clock is
    /// carried as across a null, and `n_eff` decays.
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
        assert_eq!(m.since[0], 1.0, "the clock is carried");
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

    /// An infinite trend halflife forgets no slope: the trend is the mean of
    /// every one observed, which is what `inf` means for every other model
    /// here. It pinned the trend at zero once -- the textbook form's rate at
    /// an infinite halflife is 0, a trend that never learns (S30).
    #[test]
    fn an_infinite_trend_halflife_is_the_whole_history_drift() {
        let ys: Vec<f64> = (0..500).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, m) = run(cfg(5.0, f64::INFINITY), &ys, 1.0);
        assert!((m.trend()[0] - 2.0).abs() < 0.05, "trend {}", m.trend()[0]);
        assert!(
            (preds[499] - ys[499]).abs() < 1.0,
            "{} vs {}",
            preds[499],
            ys[499]
        );
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

    /// A row the model cannot learn from -- a null target, or a present one
    /// at weight zero -- is transparent: the same numbers as if it were
    /// absent and its clock folded into the next row's, which is what
    /// `clock.rs` does with a row it skips (review 2026-09-12, C22). The
    /// textbook form's rate is not additive in the clock, `α(d₁ + d₂) ≠
    /// α(d₂)`, so advancing the level with the row's own rate would not do:
    /// each target keeps its clock since its last observation and the next
    /// observed row extrapolates, and forms its rates, over all of it.
    #[test]
    fn a_row_it_cannot_learn_from_is_as_if_absent() {
        let c = cfg(6.0, 25.0);
        let mut s = 11u64;
        let ys: Vec<f64> = (0..91)
            .map(|i| 3.0 + 2.0 * i as f64 + lcg(&mut s))
            .collect();
        for zero_weight in [false, true] {
            let mut full = Holt::new(c.clone()).unwrap();
            let mut kept = Holt::new(c.clone()).unwrap();
            let mut folded = 0.0;
            for (i, &y) in ys.iter().enumerate() {
                let d = if i == 0 { 0.0 } else { 1.0 };
                if i % 2 == 1 {
                    // The value is one no fit could absorb, so a row that
                    // leaked into the state would show.
                    let (yv, w) = if zero_weight {
                        (Some(-500.0), 0.0)
                    } else {
                        (None, 1.0)
                    };
                    full.step(&[], &[yv], d, w);
                    folded += d;
                    continue;
                }
                let a = full.step(&[], &[Some(y)], d, 1.0).pred[0];
                let b = kept.step(&[], &[Some(y)], d + folded, 1.0).pred[0];
                folded = 0.0;
                assert!(
                    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                    "zero weight {zero_weight}, row {i}: {a} vs {b}"
                );
            }
            assert_eq!(full.coefficients(), kept.coefficients());
        }
    }

    /// `y = 3 + 2t` with every other target null. Before C22 the observed
    /// rows were forecast from a level that had stood still across the null,
    /// one trend step short -- about 2 -- on every one of them.
    #[test]
    fn a_trend_is_extrapolated_across_a_missing_row() {
        let mut m = Holt::new(cfg(5.0, 5.0)).unwrap();
        let mut miss = f64::NAN;
        for i in 0..500 {
            let y = 3.0 + 2.0 * i as f64;
            let yv = (i % 2 == 0).then_some(y);
            let step = m.step(&[], &[yv], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            // The null row is forecast one step out, the next two.
            miss = step.pred[0] - y;
        }
        assert!(miss.abs() < 0.5, "the last forecast missed by {miss}");
    }

    /// `predict` extrapolates over the clock since the target was last
    /// observed plus its own delta, as the next `step` would.
    #[test]
    fn predict_extrapolates_over_the_clock_since_the_last_observation() {
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
        let on_the_null = m.step(&[], &[None], 3.0, 1.0).pred[0];
        assert!((on_the_null - (l + 3.0 * t)).abs() < 1e-12);
        let p = m.predict(&[], 2.0).pred[0];
        assert!((p - (l + 5.0 * t)).abs() < 1e-12, "{p} vs {}", l + 5.0 * t);
    }

    /// A row's weight is how much it counts: the level after a row at weight
    /// 2 is the level after the same row twice at weight 1, the second at no
    /// clock -- the review's exact check for S29. It passed on the textbook
    /// form too, where neither row's weight counted and a row at no clock
    /// changed nothing; `a_lighter_row_moves_the_level_less` is the one that
    /// failed. The trend reads a move over the clock, which the second row
    /// does not have, so the identity is the level's and `n_eff`'s.
    #[test]
    fn a_row_at_weight_two_is_the_row_given_twice() {
        let feed = |m: &mut Holt| {
            for i in 0..25 {
                let d = if i == 0 { 0.0 } else { 1.0 };
                m.step(&[], &[Some(1.0 + 0.5 * i as f64)], d, 1.0);
            }
        };
        let mut once = Holt::new(cfg(6.0, 25.0)).unwrap();
        let mut twice = Holt::new(cfg(6.0, 25.0)).unwrap();
        feed(&mut once);
        feed(&mut twice);
        once.step(&[], &[Some(40.0)], 1.0, 2.0);
        twice.step(&[], &[Some(40.0)], 1.0, 1.0);
        twice.step(&[], &[Some(40.0)], 0.0, 1.0);
        let (a, b) = (once.level()[0], twice.level()[0]);
        assert!((a - b).abs() <= 1e-12 * b.abs(), "{a} vs {b}");
        assert!((once.n_eff() - twice.n_eff()).abs() < 1e-12);
    }

    /// The review's test for S29: the same stream with its last 40 rows at
    /// weight 0.5 or at 1. A lighter row counts for less against the history
    /// before it, so the two levels differ, each what the weighted means
    /// give; the textbook form read the weight only as "learn or not", and
    /// the two were the same to the last digit.
    #[test]
    fn a_lighter_row_moves_the_level_less() {
        let c = cfg(6.0, 25.0);
        let mut s = 17u64;
        let ys: Vec<Option<f64>> = (0..80)
            .map(|i| Some(1.0 + 2.0 * i as f64 + 3.0 * lcg(&mut s)))
            .collect();
        let ds: Vec<f64> = (0..80).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut levels = Vec::new();
        for late in [0.5, 1.0] {
            let ws: Vec<f64> = (0..80).map(|i| if i < 40 { 1.0 } else { late }).collect();
            let mut m = Holt::new(c.clone()).unwrap();
            for ((y, d), w) in ys.iter().zip(&ds).zip(&ws) {
                m.step(&[], &[*y], *d, *w);
            }
            let (_, want, _, _) = reference(&c, &ys, &ds, &ws);
            let got = m.level()[0];
            assert!((got - want).abs() < 1e-9, "weight {late}: {got} vs {want}");
            levels.push(got);
        }
        assert_ne!(levels[0], levels[1], "the weight must count");
    }

    /// Weights are relative: every row at 0.5 is every row at 1, to the bit,
    /// as for any weighted mean. (Every weight is scaled by the same power of
    /// two, so the arithmetic is exact.)
    #[test]
    fn a_constant_weight_cancels() {
        let mut s = 23u64;
        let ys: Vec<f64> = (0..60)
            .map(|i| 5.0 - 0.3 * i as f64 + lcg(&mut s))
            .collect();
        let fit = |w: f64| {
            let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
            let preds: Vec<f64> = ys
                .iter()
                .enumerate()
                .map(|(i, y)| {
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    m.step(&[], &[Some(*y)], d, w).pred[0]
                })
                .collect();
            (preds, m.coefficients())
        };
        let (half, whole) = (fit(0.5), fit(1.0));
        assert_eq!(half.1, whole.1);
        for (a, b) in half.0.iter().zip(&whole.0) {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{a} vs {b}"
            );
        }
    }

    /// An infinite halflife forgets nothing, so the fit is the whole
    /// history's: on `y = 3 + 2t` level and trend follow the line, the first
    /// rows' slopes damped by the level they are read from. The textbook
    /// form's rate is 0 at an infinite halflife, so it forecast the first
    /// row's 3 on every row for ever while `n_eff` climbed (review
    /// 2026-09-12, S30).
    #[test]
    fn an_infinite_halflife_is_the_cumulative_fit() {
        let c = cfg(f64::INFINITY, f64::INFINITY);
        let ys: Vec<f64> = (0..200).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, _) = run(c.clone(), &ys, 1.0);
        let opt: Vec<Option<f64>> = ys.iter().map(|y| Some(*y)).collect();
        let ds: Vec<f64> = (0..200).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let (want, _, _, _) = reference(&c, &opt, &ds, &[1.0; 200]);
        assert!(
            (preds[199] - want[199]).abs() < 1e-9,
            "{} vs {}",
            preds[199],
            want[199]
        );
        assert!(
            (preds[199] - ys[199]).abs() < 2.5,
            "{} vs {}",
            preds[199],
            ys[199]
        );
    }
}
