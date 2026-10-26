//! Holt's linear trend method (docs/ENHANCEMENTS.md E25).
//!
//! The forecasting baseline every other model here should have to beat: no
//! features at all, just a level and a slope extrapolated forward. If a
//! regression cannot outperform "the series is going up at about this rate",
//! the features are not earning their place.
//!
//! Per row and target, with `s` the clock since the target was last observed
//! -- this row's delta included -- `w` the row's weight, `e = y − pred` the
//! forecast error, and `W`, `V` the weight the level and the trend have
//! gathered, each decayed on its own half-life, `λ_l = 0.5^(s/level_half_life)`
//! and `λ_b = 0.5^(s/trend_half_life)`:
//!
//! ```text
//! pred  = l + b·s                                  (extrapolate s clock units ahead)
//! textbook:  g_t = w/(λ_l·W + w)                   W' = λ_l·W + w
//!            h_t = w/(λ_b·V + w) · g/s             V' = λ_b·V + w    (s > 0)
//! line fit:  S0' = S0 + w     ā' = S0·(ā + s)/S0'    Q' = Q + S0·w·(ā + s)²/S0'
//!            g_f = w·(1/S0' + ā'²/Q')              h_f = w·ā'/Q'
//! l' = pred + g·e     g = max(g_f, g_t)
//! b' = b + h·e        h = max(h_f, h_t)
//! ```
//!
//! **The textbook gains** are weighted means, of what the row observes and
//! what the state forecast, as every accumulator here is: `l'` is then
//! `(λ_l·W·pred + w·y)/(λ_l·W + w)`, and `b'` the weighted mean of `b` and
//! the slope the level moved at, `(l' − l)/s`. A row at weight `w` counts `w`
//! times. At `w = 1` with `W` saturated this is the textbook recursion, `α =
//! 1 − λ_l`, `β = 1 − λ_b`. The textbook form read the weight only as "learn
//! or not" -- a row at weight 0.5 moved the level as far as one at 1 -- and at
//! an infinite half-life its rate was 0, so the level froze at the first row
//! and a trend never learned (review 2026-09-12, S29/S30).
//!
//! **The line fit's gains** are those of the weighted least-squares line
//! through every observation so far, on the clock, read at the last: `S0` its
//! weight, `ā` the observations' weighted mean age and `Q = Σ w·(age − ā)²`
//! the spread of the ages, none decayed. In the coordinates (level now,
//! slope) the fit's information is `[[S0, −S0·ā], [−S0·ā, Q + S0·ā²]]`, and
//! `(g_f, h_f)` is `w` times the first column of its inverse. With every
//! observation at one clock (`Q = 0`) no slope is determined: `g_f = w/S0'`,
//! `h_f = 0`. The weighted means alone gained `1/n` on both the level and the
//! trend, which telescoped the trend to `(l − y₁)/(t − t₁)`, pinned to the
//! first observation, and the level, which adds the trend every row, carried
//! that observation's error times `ln n` (docs/PLAN.md task 278, D-H: 10.96
//! off a flat series at 40,000 rows without decay, the worst of 50 seeds at a
//! noise of 0.5; 6.33 at 10,000 rows at `half_life = 10,000`).
//!
//! **A row takes the larger gain**, on the level and on the slope. An infinite
//! half-life has no textbook gain to give (its fixed gain `1 − λ` is 0), so
//! without decay the model *is* the weighted line fit, to rounding, whatever
//! the weights and the clock. Under a half-life the line fit's gains fall as
//! `1/n` and `1/n²` and the textbook's settle at `1 − λ`, so after a warm-up of
//! a few half-lives every row takes the textbook's for good, and statsmodels'
//! `Holt` agrees with the model from there; before, the warm-up is the line
//! fit, never slower than the textbook. `trend_half_life = inf` is the whole
//! history's drift, the line fit's slope. A row at the last row's clock (`s =
//! 0`) is a second observation the level takes in; the textbook trend holds,
//! since a move over no clock has no slope, and the line fit's moves by its
//! own gain, as a least-squares slope does.
//!
//! **A state saved before the line fit** (schema 63 and earlier) keeps none
//! and goes on with the textbook gains alone, the recursion it was fitted
//! under, to the bit.
//!
//! **Without a trend** (`trend = false`) the trend is held at zero and the
//! forecast is flat, `pred = l`: simple exponential smoothing. The level is
//! then the weighted mean of the observations, each at its weight times
//! `λ_l` per clock unit of its age, `Σ w·λ_l^age·y / Σ w·λ_l^age`, which
//! pandas' `ewm(adjust=True)` computes for unit weights (docs/PLAN.md task
//! 115, S30). Since S30 an infinite `trend_half_life` is the whole history's
//! drift, so this is the only way to ask for no trend.
//!
//! On a row that observes every target, `s` is the row's own delta `d`.
//! Deriving the decays from half-lives keeps the parameter meaning the same
//! as everywhere else in this library: a half-life is in clock units, so an
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
    /// Half-life of the level, in clock units. `inf` forgets nothing: the
    /// level is the whole history's.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub level_half_life: f64,
    /// Half-life of the trend. `inf` forgets no slope: the trend is the whole
    /// history's drift (it pinned the trend at zero before review S30).
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub trend_half_life: f64,
    pub min_weight: f64,
    /// `false` holds the trend at zero: the level alone, with a flat
    /// forecast (see the module docs). A config written before the switch
    /// has none and loads with the trend on, what it was produced under.
    #[serde(default = "default_true")]
    pub trend: bool,
}

fn default_true() -> bool {
    true
}

impl HoltCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err("holt: n_targets must be >= 1".into());
        }
        if self.level_half_life <= 0.0 || self.level_half_life.is_nan() {
            return Err("holt: level_half_life must be > 0".into());
        }
        if self.trend_half_life <= 0.0 || self.trend_half_life.is_nan() {
            return Err("holt: trend_half_life must be > 0 (inf forgets no slope)".into());
        }
        // A NaN `min_weight` is never reached, and a negative one is no
        // floor; `inf` never predicts, which is legal (review round 5, C6).
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err(format!(
                "holt: min_weight must be >= 0, got {}",
                self.min_weight
            ));
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
    /// (see the module docs).
    since: Vec<f64>,
    /// Per target, the weight the level has gathered, decayed on the level's
    /// half-life over the clock since the target was last observed: `W` in
    /// the module docs.
    #[serde(default)]
    w_level: Vec<f64>,
    /// Per target, the weight the trend has gathered, on the trend's
    /// half-life: `V` in the module docs.
    #[serde(default)]
    w_trend: Vec<f64>,
    /// Per target, the line fit to every observation so far, undecayed
    /// (docs/PLAN.md task 278, D-H): its weight `S0`, its weighted mean age
    /// `ā` on the clock, and the spread of the ages about it `Q =
    /// Σ w·(age − ā)²`. They give the exact weighted least-squares gains,
    /// which a row takes where they are the larger (the module docs). Empty
    /// in a state saved before them (schema 63 and earlier), which goes on
    /// with the textbook gains alone, as it was fitted.
    #[serde(default)]
    fit_weight: Vec<f64>,
    #[serde(default)]
    fit_age: Vec<f64>,
    #[serde(default)]
    fit_spread: Vec<f64>,
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
            fit_weight: vec![0.0; m],
            fit_age: vec![0.0; m],
            fit_spread: vec![0.0; m],
            cfg,
        })
    }

    /// Whether the state keeps the line fit: every state made by this
    /// build, and none saved before it (`fit_weight`).
    fn has_line_fit(&self) -> bool {
        !self.fit_weight.is_empty()
    }

    /// Fold target `j`'s observation at weight `w`, `s` clock units after
    /// the last, into its line fit, and return the fit's gains `(g, h)`:
    /// on the level, and on the slope per unit of the forecast error. In
    /// the coordinates (level now, slope) the fit's information is
    /// `[[S0, −S0·ā], [−S0·ā, Q + S0·ā²]]`, so the gain `w·A⁻¹·(1, 0)ᵀ` is
    /// `g = w·(1/S0 + ā²/Q)` and `h = w·ā/Q`, read after the row. With every
    /// observation at one clock (`Q = 0`) the slope is not determined: the
    /// level is their weighted mean and the slope does not move.
    fn line_fit_gains(&mut self, j: usize, s: f64, w: f64) -> (f64, f64) {
        let s0 = self.fit_weight[j];
        let aged = self.fit_age[j] + s;
        let s0_new = s0 + w;
        let age = s0 * aged / s0_new;
        let spread = self.fit_spread[j] + s0 * w * aged * aged / s0_new;
        self.fit_weight[j] = s0_new;
        self.fit_age[j] = age;
        self.fit_spread[j] = spread;
        if spread > 0.0 {
            (w * (1.0 / s0_new + age * age / spread), w * age / spread)
        } else {
            (w / s0_new, 0.0)
        }
    }

    /// `(λ_l, λ_b)`, the level's and the trend's decay over `s` clock units
    /// since the last observation; an infinite half-life forgets nothing.
    fn decays(&self, s: f64) -> (f64, f64) {
        let f = |h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                (-(s / h)).exp2()
            }
        };
        (f(self.cfg.level_half_life), f(self.cfg.trend_half_life))
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
    /// state and the only thing worth inspecting. NaN, which the stream
    /// writes as null, for a target not yet observed: it has no level to
    /// report, and the `[0, 0]` it reported was a level no row had given
    /// (docs/OUTPUTS.md; PLAN task 97).
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| {
                if self.seen[j] {
                    vec![self.level[j], self.trend[j]]
                } else {
                    vec![f64::NAN, f64::NAN]
                }
            })
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
        // A target or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`); Holt reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            return refused;
        }
        let m = self.cfg.n_targets;
        // The decays for a target observed on the previous row, where `s` is
        // this row's delta: every target of a stream with no gaps.
        let row_decays = self.decays(d_clock);

        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_weight;
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
                    if self.has_line_fit() {
                        self.fit_weight[j] = weight;
                    }
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
            // The line fit's gains, where the state keeps one and fits a
            // trend; without a trend the level's weighted mean below is
            // already the larger gain (`λ·W ≤ S0`).
            let line =
                (self.cfg.trend && self.has_line_fit()).then(|| self.line_fit_gains(j, s, weight));
            // The textbook gain: the level as a weighted mean of what the
            // row observes and what the state held; `weight > 0` here, so
            // no denominator is 0 (hard rule 9). An infinite half-life has
            // no textbook gain to give: its fixed gain `1 − λ` is 0, and the
            // line fit is the whole history's.
            let held = lam_l * self.w_level[j];
            let gain = weight / (held + weight);
            let prev_level = self.level[j];
            let error = yj - p;
            let level_gain = match line {
                Some((g, _)) if g > gain || self.cfg.level_half_life.is_infinite() => {
                    self.level[j] = p + g * error;
                    g
                }
                _ => {
                    self.level[j] = (held * p + weight * yj) / (held + weight);
                    gain
                }
            };
            self.w_level[j] = held + weight;
            if self.cfg.trend {
                // The textbook trend: a weighted mean of the slopes the
                // level moved at, whose gain on the error is `w/(λ_b·V + w)
                // · g/s`. A move over no clock has no slope, and an infinite
                // trend half-life none to give where the line fit is kept:
                // the line fit's slope is then the whole history's drift.
                let textbook = s > 0.0 && (line.is_none() || self.cfg.trend_half_life.is_finite());
                let held_b = lam_b * self.w_trend[j];
                let take_line = match line {
                    Some((_, h)) => !textbook || h > weight / (held_b + weight) * level_gain / s,
                    None => false,
                };
                if take_line {
                    self.trend[j] += line.map_or(0.0, |(_, h)| h) * error;
                } else if textbook {
                    let slope = (self.level[j] - prev_level) / s;
                    self.trend[j] = (held_b * self.trend[j] + weight * slope) / (held_b + weight);
                }
                if s > 0.0 {
                    self.w_trend[j] = held_b + weight;
                }
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
        if n_eff >= self.cfg.min_weight {
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
                let m = (**m).clone();
                crate::model::check_cfg("holt", m.cfg.validate())?;
                let n = m.cfg.n_targets;
                // Every vector is checked, not repaired: a level or a weight
                // vector of the wrong length loaded and panicked on the
                // first `step` (review 2026-09-18, B3), and a clock since an
                // observation of the wrong length was set to zero, the
                // repair for a schema-6 state (docs/PLAN.md task 198).
                // The line fit is all three vectors or none: none is a state
                // saved before it, which goes on with the textbook gains.
                let fit = [&m.fit_weight, &m.fit_age, &m.fit_spread];
                let fit_len = m.fit_weight.len();
                if m.seen.len() != n
                    || [&m.level, &m.trend, &m.w_level, &m.w_trend, &m.since]
                        .iter()
                        .any(|v| v.len() != n)
                    || (fit_len != 0 && fit_len != n)
                    || fit.iter().any(|v| v.len() != fit_len)
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
            level_half_life: level,
            trend_half_life: trend,
            min_weight: 0.0,
            trend: true,
        }
    }

    /// PLAN task 97. Before a target's first observation there is no level
    /// and no trend to report, and `coef` is null there, as docs/OUTPUTS.md
    /// says it is "before the model has anything to report" (NaN here, which
    /// the stream writes as null). It was `[0, 0]`, a level of zero that no
    /// row had given. A null value or a row of weight 0 observes nothing,
    /// and the first observation reports its own value as the level.
    #[test]
    fn a_target_not_yet_observed_reports_no_coefficients() {
        let mut m = Holt::new(HoltCfg {
            n_targets: 2,
            ..cfg(10.0, 10.0)
        })
        .unwrap();
        let unseen = |c: &[f64]| c.iter().all(|v| v.is_nan());
        let coef = m.coefficients();
        assert!(
            unseen(&coef[0]) && unseen(&coef[1]),
            "nothing yet: {coef:?}"
        );
        // Target 0 observed; target 1 null, then observed at weight 0.
        m.step(&[], &[Some(4.0), None], 0.0, 1.0);
        m.step(&[], &[Some(5.0), Some(9.0)], 1.0, 0.0);
        let coef = m.coefficients();
        assert_eq!(coef[0][0], 4.0, "the first observation is the level");
        assert_eq!(coef[0][1], 0.0, "and no slope yet: one point has none");
        assert!(
            unseen(&coef[1]),
            "a null and a weight of 0 observe nothing: {coef:?}"
        );
        m.step(&[], &[Some(6.0), Some(-2.0)], 1.0, 1.0);
        let coef = m.coefficients();
        assert_eq!(coef[1], vec![-2.0, 0.0], "its first observation");
        assert!(coef[0].iter().all(|v| v.is_finite()));
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

    /// The textbook recursion alone, written out longhand: level and trend
    /// are each a weighted mean of what the row observes and what the state
    /// held, `(λ·W·old + w·new)/(λ·W + w)`, with `W` the weight each has
    /// gathered and `λ` its decay over the clock since the target was last
    /// observed (review 2026-09-12, S29/S30). The whole model until the line
    /// fit (task 278, D-H), and still a state's saved before it.
    fn textbook(
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
                preds.push(if w_sum >= cfg.min_weight { p } else { f64::NAN });
                if let Some(y) = y {
                    let held = decay(s, cfg.level_half_life) * w_level;
                    let prev = level;
                    level = (held * p + w * y) / (held + w);
                    w_level = held + w;
                    if s > 0.0 {
                        let held = decay(s, cfg.trend_half_life) * w_trend;
                        trend = (held * trend + w * (level - prev) / s) / (held + w);
                        w_trend = held + w;
                    }
                    since = 0.0;
                } else {
                    since = s;
                }
            }
            w_sum = w_sum * decay(d, cfg.level_half_life) + w;
        }
        (preds, level, trend, w_sum)
    }

    /// The model written out from its definitions, against an irregular
    /// clock, so the test cannot share a mistake with the implementation:
    /// the line fit's weight, mean age and spread summed afresh over every
    /// observation so far (`crate::oracle::sum`), its gains `g = w·(1/S0 +
    /// ā²/Q)` and `h = w·ā/Q`, and the textbook's weighted means; a row
    /// takes the larger gain, on the level and on the slope, and an infinite
    /// half-life has no textbook gain (task 278, D-H).
    fn reference(
        cfg: &HoltCfg,
        ys: &[Option<f64>],
        ds: &[f64],
        ws: &[f64],
    ) -> (Vec<f64>, f64, f64, f64) {
        use crate::oracle::sum;
        let decay = |s: f64, h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                0.5f64.powf(s / h)
            }
        };
        let (mut level, mut trend, mut w_level, mut w_trend) = (0.0, 0.0, 0.0, 0.0);
        let (mut w_sum, mut since, mut seen) = (0.0, 0.0, false);
        let (mut clock, mut obs) = (0.0, Vec::<(f64, f64)>::new());
        let mut preds = Vec::new();
        for ((&y, &d), &w) in ys.iter().zip(ds).zip(ws) {
            clock += d;
            let y = y.filter(|v| v.is_finite() && w > 0.0);
            if !seen {
                preds.push(f64::NAN);
                if let Some(y) = y {
                    (level, w_level, seen) = (y, w, true);
                    obs.push((clock, w));
                }
            } else {
                let s = since + d;
                let p = level + trend * s;
                preds.push(if w_sum >= cfg.min_weight { p } else { f64::NAN });
                if let Some(y) = y {
                    obs.push((clock, w));
                    let s0 = sum(obs.iter().map(|o| o.1));
                    let age = sum(obs.iter().map(|o| o.1 * (clock - o.0))) / s0;
                    let q = sum(obs.iter().map(|o| o.1 * (clock - o.0 - age).powi(2)));
                    let (g_fit, h_fit) = if q > 0.0 {
                        (w / s0 + w * age * age / q, w * age / q)
                    } else {
                        (w / s0, 0.0)
                    };
                    let e = y - p;
                    let held = decay(s, cfg.level_half_life) * w_level;
                    let g_book = if cfg.level_half_life.is_infinite() {
                        0.0
                    } else {
                        w / (held + w)
                    };
                    let g = g_fit.max(g_book);
                    level = p + g * e;
                    w_level = held + w;
                    if s > 0.0 {
                        let held = decay(s, cfg.trend_half_life) * w_trend;
                        let h_book = if cfg.trend_half_life.is_infinite() {
                            0.0
                        } else {
                            w / (held + w) * g / s
                        };
                        trend += h_fit.max(h_book) * e;
                        w_trend = held + w;
                    } else {
                        trend += h_fit * e;
                    }
                    since = 0.0;
                } else {
                    since = s;
                }
            }
            w_sum = w_sum * decay(d, cfg.level_half_life) + w;
        }
        (preds, level, trend, w_sum)
    }

    /// A model whose state keeps no line fit: a state saved before it.
    fn without_line_fit(c: HoltCfg) -> Holt {
        let mut m = Holt::new(c).unwrap();
        m.fit_weight.clear();
        m.fit_age.clear();
        m.fit_spread.clear();
        Holt::restore(&m.state()).expect("a state without the line fit loads")
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
            level_half_life: 4.0,
            trend_half_life: 9.0,
            min_weight: 2.5,
            trend: true,
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
            level_half_life: 10.0,
            trend_half_life: 40.0,
            min_weight: 3.0,
            trend: true,
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
    /// level takes it in, as a weighted mean takes any row. The textbook
    /// trend holds, since a move over no clock has no slope; the line fit's
    /// does not, since a second observation at one time still moves a
    /// least-squares slope, by the fit's own gain (task 278, D-H). The
    /// textbook form's rate was 0 at `d = 0`, so such a row once changed
    /// nothing (S29).
    #[test]
    fn a_zero_gap_folds_the_row_into_the_level_and_the_line_fits_slope() {
        let c = cfg(5.0, 20.0);
        let mut ys: Vec<Option<f64>> = (0..30).map(|i| Some(2.0 * i as f64)).collect();
        let mut ds: Vec<f64> = (0..30).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut m = Holt::new(c.clone()).unwrap();
        let mut old = without_line_fit(c.clone());
        for (y, d) in ys.iter().zip(&ds) {
            m.step(&[], &[*y], *d, 1.0);
            old.step(&[], &[*y], *d, 1.0);
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        assert!(t > 0.5, "there should be a trend: {t}");
        let t_old = old.trend()[0];
        m.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        old.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        assert_eq!(
            old.trend()[0],
            t_old,
            "the textbook trend holds at a zero gap"
        );
        assert!(m.trend()[0] < t, "the line fit's slope takes the row in");
        assert!(
            m.level()[0] < l,
            "the row is taken in: {} vs {l}",
            m.level()[0]
        );
        ys.push(Some(-1000.0));
        ds.push(0.0);
        let ws = vec![1.0; ys.len()];
        let (_, want, want_t, _) = reference(&c, &ys, &ds, &ws);
        assert!(
            (m.level()[0] - want).abs() < 1e-9,
            "{} vs {want}",
            m.level()[0]
        );
        assert!(
            (m.trend()[0] - want_t).abs() < 1e-12,
            "{} vs {want_t}",
            m.trend()[0]
        );
    }

    #[test]
    fn targets_are_independent() {
        // `n_targets` and the per-target loop: two targets with unrelated
        // series must not contaminate each other, and both must be reported.
        let c = HoltCfg {
            n_targets: 2,
            level_half_life: 8.0,
            trend_half_life: 30.0,
            min_weight: 0.0,
            trend: true,
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

    /// An infinite trend half-life forgets no slope: the trend is the mean of
    /// every one observed, which is what `inf` means for every other model
    /// here. It pinned the trend at zero once -- the textbook form's rate at
    /// an infinite half-life is 0, a trend that never learns (S30).
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

    /// An infinite half-life forgets nothing, so the fit is the whole
    /// history's: the weighted least-squares line through every observation
    /// on the clock, read at the last one (task 278, D-H). Held against
    /// faer's LU solve of the normal equations (`crate::oracle`), on an
    /// irregular clock with uneven weights, nulls and zero weights. The
    /// weighted means before it pinned the trend to the first observation,
    /// `(l − y₁)/(t − t₁)`, and the level's error grew like that
    /// observation's noise times `ln n`; the textbook form's rate is 0
    /// there, so it forecast the first row for ever (review 2026-09-12,
    /// S30).
    #[test]
    fn an_infinite_halflife_is_the_weighted_line_fit() {
        let c = cfg(f64::INFINITY, f64::INFINITY);
        let mut st = 29u64;
        let n = 3000;
        let mut clock = 0.0;
        let (mut ys, mut ds, mut ws, mut rows) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let d = if i == 0 {
                0.0
            } else {
                [0.0, 0.5, 1.0, 3.0][i % 4]
            };
            let w = [1.0, 0.5, 2.0, 0.0, 1.5][i % 5];
            clock += d;
            let y = 2.0 + 0.3 * clock + 0.7 * lcg(&mut st);
            let y = (i % 7 != 3).then_some(y);
            if let Some(y) = y.filter(|_| w > 0.0) {
                rows.push((clock, w, y));
            }
            ds.push(d);
            ws.push(w);
            ys.push(y);
        }
        let mut m = Holt::new(c).unwrap();
        for i in 0..n {
            m.step(&[], &[ys[i]], ds[i], ws[i]);
        }
        // In clock units from the last observation, so the oracle's
        // intercept is the level.
        use crate::oracle::{solve, sum};
        let t_last = rows.last().unwrap().0;
        let a = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| r.1 * f(r.0 - t_last)));
        let b = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| r.1 * r.2 * f(r.0 - t_last)));
        let (s0, s1, s2) = (a(&|_| 1.0), a(&|t| t), a(&|t| t * t));
        let fit = solve(&[s0, s1, s1, s2], &[b(&|_| 1.0), b(&|t| t)]);
        let (level, trend) = (m.level()[0], m.trend()[0]);
        assert!(
            (level - fit[0]).abs() <= 1e-10 * fit[0].abs(),
            "level {level} vs {}",
            fit[0]
        );
        assert!(
            (trend - fit[1]).abs() <= 1e-10 * fit[1].abs(),
            "trend {trend} vs {}",
            fit[1]
        );
    }

    /// The defect itself (task 278, D-H): a flat series whose first
    /// observation is 10 off. The weighted means carried that 10 into the
    /// trend, `(l − y₁)/(t − t₁)`, so the level's error grew like `−10·ln
    /// n` without decay and through every warm-up under one. The line fit's
    /// first error falls as `1/n`.
    #[test]
    fn the_first_observations_error_does_not_grow() {
        for (h, rows, bound) in [(f64::INFINITY, 10_000, 0.02), (1000.0, 1000, 0.1)] {
            let mut m = Holt::new(cfg(h, 4.0 * h)).unwrap();
            let mut st = 3u64;
            for i in 0..rows {
                let y = 3.0 + 0.5 * lcg(&mut st) + if i == 0 { 10.0 } else { 0.0 };
                m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let off = m.level()[0] - 3.0;
            assert!(
                off.abs() < bound,
                "half-life {h}: the level is {off} off after {rows} rows"
            );
        }
    }

    /// Once the line fit's gains fall below the textbook's, every row takes
    /// the textbook's, so the model is the textbook recursion from there:
    /// beside a state that never kept a line fit, on the same rows, the
    /// predictions agree to 1e-12 once the warm-ups' differences have
    /// decayed at the trend's half-life (`tests/test_second_opinion.py`
    /// holds the same against statsmodels' `Holt`).
    #[test]
    fn settled_it_is_the_textbook_recursion() {
        let c = cfg(6.0, 25.0);
        let mut m = Holt::new(c.clone()).unwrap();
        let mut old = without_line_fit(c);
        let mut st = 41u64;
        let mut level = 3.0;
        for i in 0..2000 {
            level += 0.4 + 0.5 * lcg(&mut st);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = m.step(&[], &[Some(level)], d, 1.0).pred[0];
            let b = old.step(&[], &[Some(level)], d, 1.0).pred[0];
            if i >= 1000 {
                assert!((a - b).abs() <= 1e-12 * b.abs(), "row {i}: {a} vs {b}");
            }
        }
    }

    /// A state saved before the line fit (schema 63 and earlier) has none,
    /// and goes on with the textbook gains alone, the recursion it was
    /// fitted under (hard rule 5; the kept fixtures hold it to the bit): a
    /// row the line fit would have taken does not reach it.
    #[test]
    fn a_state_saved_before_the_line_fit_goes_on_with_the_textbook_gains() {
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        for c in [cfg(4.0, 9.0), cfg(f64::INFINITY, f64::INFINITY)] {
            let (want, level, trend, _) = textbook(&c, &ys, &ds, &ws);
            let mut m = without_line_fit(c.clone());
            for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
                let p = m.step(&[], &[y], d, w).pred[0];
                // The decays here run through `powf`, the model's through
                // `exp2`; the kept fixtures hold the bits (state_fixtures).
                assert!(
                    (p - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs())
                        || (p.is_nan() && want[i].is_nan()),
                    "row {i}: {p} vs {}",
                    want[i]
                );
            }
            assert!((m.level()[0] - level).abs() <= 1e-13 * (1.0 + level.abs()));
            assert!((m.trend()[0] - trend).abs() <= 1e-13 * (1.0 + trend.abs()));
            assert!(m.fit_weight.is_empty(), "and it saves none");
        }
        // The line fit is all three vectors or none.
        let mut m = Holt::new(cfg(4.0, 9.0)).unwrap();
        m.fit_age.clear();
        assert!(
            Holt::restore(&m.state()).is_err(),
            "a part of the line fit is refused"
        );
    }

    /// `trend = false` holds the trend at zero (docs/PLAN.md task 115, S30):
    /// the forecast is flat and the level is the weighted mean of the
    /// observations, each at its weight times `0.5^(age/level_half_life)`,
    /// written here from that definition. The clock is irregular, and a null
    /// and a zero weight are among the rows, which observe nothing.
    #[test]
    fn without_a_trend_the_level_is_the_weighted_mean_of_the_observations() {
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        let mut m = Holt::new(HoltCfg {
            trend: false,
            ..cfg(4.0, 9.0)
        })
        .unwrap();
        let (mut clock, mut seen) = (0.0, Vec::<(f64, f64, f64)>::new());
        for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
            clock += d;
            let pred = m.step(&[], &[y], d, w).pred[0];
            let at = |&(t, w, _): &(f64, f64, f64)| w * 0.5f64.powf((clock - t) / 4.0);
            let num: f64 = seen.iter().map(|o| at(o) * o.2).sum();
            let den: f64 = seen.iter().map(at).sum();
            if seen.is_empty() {
                assert!(pred.is_nan(), "row {i}: nothing observed yet");
            } else {
                let want = num / den;
                assert!(
                    (pred - want).abs() < 1e-12 * want.abs(),
                    "row {i}: {pred} vs {want}"
                );
            }
            if let Some(y) = y.filter(|_| w > 0.0) {
                seen.push((clock, w, y));
            }
            assert_eq!(m.trend()[0], 0.0, "row {i}: the trend never moves");
            assert_eq!(m.coefficients()[0][1], 0.0, "row {i}: and reports 0");
        }
        assert_eq!(seen.len(), 10, "a null and a zero weight observe nothing");
    }

    /// A config written before the switch has no `trend`, and loads with the
    /// trend on: what that state was produced under.
    #[test]
    fn a_config_written_before_the_trend_switch_keeps_its_trend() {
        let json = r#"{"n_targets": 1, "level_half_life": 10.0, "trend_half_life": 40.0, "min_weight": 0.0}"#;
        let c: HoltCfg = serde_json::from_str(json).expect("loads without the field");
        assert!(c.trend, "the omitted field is the trend on");
    }

    /// Each target's weight is its level's, decayed over the clock since the
    /// target was last seen: `Σᵢ wᵢ·2^(−(T − tᵢ)/h)` over the rows that
    /// carried it, `T` the clock now and `h` the level's half-life. The
    /// second target is absent on the last rows, so its weight has aged
    /// since; a zero-weight row carries nothing.
    #[test]
    fn a_targets_weight_is_its_rows_decayed_to_now() {
        let h = 4.0;
        let mut m = Holt::new(HoltCfg {
            n_targets: 2,
            ..cfg(h, 8.0)
        })
        .unwrap();
        let mut clock = 0.0;
        let mut carried: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
        for i in 0..12usize {
            let d = 0.5 + 0.75 * (i % 3) as f64;
            clock += d;
            let w = if i == 5 {
                0.0
            } else {
                0.5 + 0.25 * (i % 4) as f64
            };
            let y = [Some(i as f64), (i % 4 != 3 && i < 9).then(|| -(i as f64))];
            m.step(&[], &y, d, w);
            for t in 0..2 {
                if y[t].is_some() && w > 0.0 {
                    carried[t].push((clock, w));
                }
            }
        }
        let mut out = Vec::new();
        assert!(m.target_n_eff_into(&mut out));
        assert_eq!(out.len(), 2);
        for t in 0..2 {
            let want: f64 = carried[t]
                .iter()
                .map(|(at, w)| w * (-(clock - at) / h).exp2())
                .sum();
            assert!(
                (out[t] - want).abs() <= 1e-12 * want,
                "target {t}: {} vs {want}",
                out[t]
            );
        }
        assert!(m.since[1] > 0.0, "the second target has aged since");
    }

    /// Task 252: a model with no targets is refused by name.
    #[test]
    fn a_model_with_no_targets_is_refused_by_name() {
        let e = Holt::new(HoltCfg {
            n_targets: 0,
            ..cfg(6.0, 25.0)
        })
        .unwrap_err();
        assert_eq!(e, "holt: n_targets must be >= 1");
    }
}
