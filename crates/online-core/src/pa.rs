//! Passive-aggressive regression (docs/ENHANCEMENTS.md E17).
//!
//! Crammer et al. (2006). Each row poses a constraint — "get within `eps` of
//! this target" — and the update makes the *smallest* change to the
//! coefficients that satisfies it. Passive when the constraint already holds,
//! aggressive when it does not; there is no learning rate to tune.
//!
//! With `p = z·b`, `loss = max(0, |y − p| − eps·σ_y)` and `s = ||z||²`:
//!
//! ```text
//! PA    tau = loss / s                    (unbounded step)
//! PA-I  tau = min(C, loss / s)            (step capped at C)
//! PA-II tau = loss / (s + 1 / (2C))       (step damped by C)
//! b    += min(w, 1) * tau * sign(y − p) * z
//! ```
//!
//! With an intercept, `z` carries it as a constant 1, so it is inside
//! `s = ||z||²` -- Crammer et al.'s augmented form -- where river's
//! `PARegressor` and sklearn keep the intercept outside the norm: the one
//! difference T-S18 has to map (review 2026-09-12, D10).
//!
//! **The tube is in units of `σ_y`**, the target's own EW standard
//! deviation as the row arrives: the spread of `y` around its EW mean, over
//! the rows that carried the target with a weight above 0, on the model's
//! clock, the row's own `y` joining after it is judged
//! ([`crate::spread`]). Until the target has a spread -- fewer than two
//! weighted rows, or every `y` the same -- the tube has no width and every
//! row teaches. A tube in the target's own units left a target in
//! hundredths inside it on every row, passive for ever: every prediction
//! 0.0 and R² -0.051 where the unscaled target scored 0.961 (docs/PLAN.md
//! task 195, review round 4 CC4). Under the unbounded step a target scaled
//! by `c` then fits as the unscaled one, scaled by `c`; `C` caps `tau` in
//! the target's units over `s`'s. The tube was drawn in the residual's
//! spread at first, as `huber`'s cut is, and a fit from zero coefficients
//! has the target's whole level for its first residuals: a target at 1,000
//! in a spread of 2 drew a tube about 100 wide, held every later row in it,
//! and without decay never narrowed it, R² −52 at a half-life of 1e9
//! (docs/PLAN.md task 202). `y`'s own spread does not read the fit.
//!
//! **`standardize`** reads `z` as the features standardized against their
//! EW moments with the row admitted, `sgd`'s scaler and its rule
//! ([`crate::SgdCfg::standardize`]), so `s` and `C` stop being in the
//! features' units; the coefficients are read out in the caller's units
//! through the moments as they stand. A box or a sum is projected in the
//! standardized coordinates, the bounds scaled as `sgd` scales them.
//!
//! **Weight note.** A row weight below 1 scales the step; a weight above 1
//! counts as 1. The update is a projection onto the row's constraint, and
//! repeating a projection changes nothing, so there is no "two observations"
//! to emulate -- scaling past the projection would overshoot it. sklearn's
//! `sample_weight` scales the step uncapped and river's `PARegressor` takes
//! no weight, so a comparison with either holds for `w <= 1` only (T-S18;
//! D10).
//!
//! **Decay note.** The coefficients keep no accumulators, so there is
//! nothing in them for the clock to decay: each step fully satisfies the
//! current row's constraint and older rows survive only through the
//! coefficients they left behind. The clock decays `n_eff`, so
//! `min_weight` means the same thing as elsewhere, and it decays each
//! target's own weight, the scaler under `standardize` and the target's
//! spread the tube is drawn in. The coefficients have no half-life -- so after
//! a gap `min_weight` can withhold a fit exactly as good as the one before
//! it, which is `ewridge`'s behaviour too (a mean-form fit does not move on
//! a gap either): the library's convention rather than `pa`'s (CLAUDE.md
//! rule 8; D10). Use PA-I/PA-II (a finite `c`) when that aggressiveness is
//! a problem: an outlier otherwise moves the fit as far as it takes to
//! satisfy the outlier.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::sgd::{scales_of, standardized, unscaled};
use crate::solve::dot_aug;
use crate::spread::TargetSpread;
use crate::{Constraint, Decay, EwDiag};

/// Which passive-aggressive variant (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaMode {
    /// Unbounded step: satisfies the constraint exactly.
    Pa,
    /// Step capped at `c`.
    #[default]
    Pa1,
    /// Step damped by `c`.
    Pa2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    pub mode: PaMode,
    /// Aggressiveness, a cap on `tau`, which is in the target's units over
    /// `||z||²`'s: the features' own, or standardized ones under
    /// `standardize`. Ignored by [`PaMode::Pa`]; `inf` caps nothing, so
    /// either bounded mode is [`PaMode::Pa`] exactly.
    pub c: f64,
    /// Half-width of the insensitive tube, in units of the target's own EW
    /// standard deviation: rows already this close are passive (the module
    /// docs).
    pub eps: f64,
    pub min_weight: f64,
    /// Box and/or sum constraint on the slopes, imposed by Euclidean
    /// projection after each update (ENHANCEMENTS E40); the intercept is
    /// free. The initial `0` is projected too. A projected step no longer
    /// satisfies the row's margin exactly -- it is the closest feasible
    /// coefficient to the one that would. Under `standardize` the
    /// projection is taken in the standardized coordinates, as `sgd`'s is.
    #[serde(default)]
    pub constraint: Option<Constraint>,
    /// Standardize the features against their EW moments, `sgd`'s scaler
    /// (the module docs; docs/PLAN.md task 195, U2).
    #[serde(default)]
    pub standardize: bool,
}

impl PaCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("pa: {e}"))?;
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("pa: n_features and n_targets must be >= 1".into());
        }
        if self.c <= 0.0 || self.c.is_nan() {
            return Err("pa: c must be > 0".into());
        }
        if self.eps < 0.0 || self.eps.is_nan() {
            return Err("pa: eps must be >= 0".into());
        }
        if let Some(c) = &self.constraint {
            c.validate(self.n_features, "pa")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pa {
    cfg: PaCfg,
    /// Running feature means and variances, under `standardize`.
    scaler: Option<EwDiag>,
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it, decayed: what its
    /// `min_weight` is checked against (hard rule 8, docs/PLAN.md task 115
    /// (d)). `w_sum` stood in for it, so ten rows with a null target met
    /// `min_weight = 10` with every coefficient at zero.
    #[serde(default)]
    w_target: Vec<f64>,
    /// Per target, the EW mean and variance of the target itself: the unit
    /// `eps` is in (the module docs; docs/PLAN.md task 202).
    spread: TargetSpread,
    /// The row the model sees, `[1, z]`: the features as they stand, or
    /// standardized under a scaler.
    #[serde(skip)]
    zbuf: Vec<f64>,
    /// The raw row `[1, x]` the scaler is updated with.
    #[serde(skip)]
    rawbuf: Vec<f64>,
    /// Scratch for the projection.
    #[serde(skip)]
    pbuf: crate::constraint::Scratch,
    /// Which targets stepped this row (kept under a constraint only).
    #[serde(skip)]
    learned: Vec<bool>,
}

thread_local! {
    /// `predict`'s standardized row, as `sgd`'s.
    static PREDICT_Z: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Pa {
    pub fn new(cfg: PaCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let mut beta = vec![vec![0.0; k]; m];
        if let Some(c) = &cfg.constraint {
            let off = usize::from(cfg.fit_intercept);
            let mut scratch = crate::constraint::Scratch::default();
            for b in beta.iter_mut() {
                c.project(&mut b[off..], None, &mut scratch);
            }
        }
        Ok(Self {
            scaler: cfg.standardize.then(|| EwDiag::new(k)),
            beta,
            w_sum: 0.0,
            w_target: vec![0.0; m],
            spread: TargetSpread::new(m),
            zbuf: vec![0.0; k],
            rawbuf: vec![0.0; k],
            pbuf: crate::constraint::Scratch::default(),
            learned: Vec::new(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &PaCfg {
        &self.cfg
    }

    /// Coefficients in the caller's units: under `standardize` the slopes
    /// over each feature's scale as it stands, the intercept absorbing the
    /// means, as `sgd`'s are read out.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        match &self.scaler {
            None => self.beta.clone(),
            Some(sc) => unscaled(&self.beta, sc, self.cfg.fit_intercept),
        }
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Per target, the weight of the rows that carried it: what its
    /// `min_weight` is checked against.
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// Per target, the EW variance of the target itself, `σ_y²`: the
    /// square of the unit `eps` is in (the module docs).
    pub fn target_variance(&self) -> Vec<f64> {
        self.spread.vars()
    }

    /// Target `j`'s tube half-width: `eps·σ_y` once `σ_y²` is finite and
    /// above 0, and no width before -- fewer than two weighted rows, or
    /// every `y` the same ([`crate::spread`]).
    fn tube(&self, j: usize) -> f64 {
        self.spread.band(j, self.cfg.eps)
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
        }
        if self.rawbuf.len() != k {
            self.rawbuf = vec![0.0; k];
        }
    }
}

impl OnlineModel for Pa {
    /// Each target's own weight, the rows that carried it, for the bank's
    /// per-target `min_weight` gate: without it the bank checked a list of
    /// thresholds against the shared `n_eff` (task 158, E14).
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.w_target);
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        self.ensure_buffers();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);
        let off = usize::from(self.cfg.fit_intercept);

        // The row the model sees: `[1, x]`, or `[1, z]` standardized
        // against the moments with the row admitted, `sgd`'s rule; the raw
        // row updates the scaler afterwards, at the row's weight.
        if off == 1 {
            self.zbuf[0] = 1.0;
            self.rawbuf[0] = 1.0;
        }
        self.rawbuf[off..].copy_from_slice(x);
        match &self.scaler {
            Some(sc) => {
                for (z, v) in self.zbuf[off..]
                    .iter_mut()
                    .zip(standardized(sc, off, x, lam))
                {
                    *z = v;
                }
            }
            None => self.zbuf[off..].copy_from_slice(x),
        }

        // Before this row's update and before its decay -- the convention
        // every model reports and gates on, each target on its own weight.
        let n_eff = self.w_sum;
        let sq_norm: f64 = self.zbuf.iter().map(|z| z * z).sum();

        let mut pred = vec![f64::NAN; m];
        if self.cfg.constraint.is_some() {
            self.learned.clear();
            self.learned.resize(m, false);
        }
        for j in 0..m {
            let p: f64 = self
                .zbuf
                .iter()
                .zip(&self.beta[j])
                .map(|(z, b)| z * b)
                .sum();
            if self.w_target[j] >= self.cfg.min_weight {
                pred[j] = p;
            }
            // The tube is drawn in the target's spread as the row arrives;
            // the row's own `y` joins it afterwards, whatever the fit, and
            // a row without the target or of weight 0 only ages it (the
            // module docs).
            let tube = self.tube(j);
            self.spread.update(j, y[j], lam, weight);
            let Some(yj) = y[j] else { continue };
            // `p` overflows when a feature at the input bound meets a large
            // coefficient; a step from an infinite loss would be permanent.
            if weight <= 0.0 || !yj.is_finite() || !p.is_finite() {
                continue;
            }
            let err = yj - p;
            if sq_norm <= 0.0 {
                continue;
            }
            let loss = (err.abs() - tube).max(0.0);
            if loss == 0.0 {
                continue; // passive: the constraint already holds
            }
            // A weight below 1 scales the step, so a half-weight row moves the
            // fit half as far. A weight above 1 counts as 1: the update is a
            // projection onto this row's constraint, and a projection repeated
            // is the same projection -- scaling past it would overshoot (a
            // constant weight of 2 makes plain PA oscillate for ever) and,
            // since nothing here decays, a weight at the input bound would
            // move the fit by 1e100 with no way back (docs/IMPROVEMENTS.md C2).
            let tau = weight.min(1.0)
                * match self.cfg.mode {
                    PaMode::Pa => loss / sq_norm,
                    PaMode::Pa1 => (loss / sq_norm).min(self.cfg.c),
                    PaMode::Pa2 => loss / (sq_norm + 0.5 / self.cfg.c),
                };
            let step = tau * err.signum();
            for (b, z) in self.beta[j].iter_mut().zip(&self.zbuf) {
                *b += step * z;
            }
            if let Some(l) = self.learned.get_mut(j) {
                *l = true;
            }
        }
        if let Some(sc) = &mut self.scaler {
            sc.update(&self.rawbuf, lam, weight);
        }
        // Project after the scaler moved, as `sgd` does: the bounds live in
        // the caller's units, and in standardized coordinates they move with
        // the scales, so every target is re-projected when the scales
        // changed (a positive weight), otherwise only the ones this row
        // stepped. Without a scaler that is each stepped target, where its
        // step left it.
        if let Some(c) = &self.cfg.constraint {
            let scales = self.scaler.as_ref().map(|sc| scales_of(sc, off));
            let rescaled = self.scaler.is_some() && weight > 0.0;
            for (j, b) in self.beta.iter_mut().enumerate() {
                if rescaled || self.learned[j] {
                    c.project(
                        &mut b[off..],
                        scales.as_deref().map(|s| &s[off..]),
                        &mut self.pbuf,
                    );
                }
            }
        }
        self.w_sum = lam * self.w_sum + weight;
        crate::model::age_target_weights(
            &mut self.w_target,
            |j| y[j].is_some_and(f64::is_finite),
            lam,
            weight,
        );

        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let n_eff = self.w_sum;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        let fit_intercept = self.cfg.fit_intercept;
        let predict_with = |z: &[f64], pred: &mut [f64]| {
            for ((p, beta), w) in pred.iter_mut().zip(&self.beta).zip(&self.w_target) {
                if *w >= self.cfg.min_weight {
                    *p = dot_aug(beta, z, fit_intercept);
                }
            }
        };
        match &self.scaler {
            None => predict_with(x, &mut pred),
            // The row the step would see, standardized against the moments
            // with it admitted after the step's decay, so `predict` gives
            // the step's number exactly, as `sgd`'s does.
            Some(sc) => PREDICT_Z.with_borrow_mut(|z| {
                let off = usize::from(fit_intercept);
                let lam = self.cfg.decay.factor(d_clock);
                z.clear();
                z.extend(standardized(sc, off, x, lam));
                predict_with(z, &mut pred);
            }),
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Pa(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Pa(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("pa", m.cfg.validate())?;
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                // One coefficient row per target at the cfg's width; a
                // short one loaded and panicked on the first `step` (review
                // 2026-09-18, B3).
                if m.beta.len() != n || m.beta.iter().any(|b| b.len() != k) {
                    return Err(StateError::Invalid(
                        "pa: the coefficients have the wrong shape".into(),
                    ));
                }
                if !crate::model::restore_target_weights(&mut m.w_target, m.w_sum, n) {
                    return Err(StateError::Invalid(
                        "pa: the target weights have the wrong shape".into(),
                    ));
                }
                if !m.spread.has_shape(n) {
                    return Err(StateError::Invalid(
                        "pa: the target spreads have the wrong shape".into(),
                    ));
                }
                // A scaler exactly when `standardize` is on, at the cfg's
                // width, as `sgd`'s state is held to (S16).
                if m.cfg.standardize != m.scaler.is_some()
                    || m.scaler.as_ref().is_some_and(|sc| sc.k() != k)
                {
                    return Err(StateError::Invalid(
                        "pa: the state's scaler does not match its cfg's standardize".into(),
                    ));
                }
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "pa",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A target's `min_weight` counts only the rows that carried it, and
    /// `n_eff` stays every row's (hard rule 8, docs/PLAN.md task 115 (d)):
    /// ten rows with no target met `min_weight = 3` with every coefficient
    /// at zero, and `pred` was 0. A zero-weight row with the target (row 12)
    /// only ages it.
    #[test]
    fn min_periods_counts_only_the_rows_that_carried_the_target() {
        use crate::OnlineModel;
        let mut c = cfg(2, PaMode::Pa1);
        c.min_weight = 3.0;
        let mut m = Pa::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..16usize {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i >= 10).then(|| x[0] - x[1]);
            let out = m.step(&x, &[y], 1.0, if i == 12 { 0.0 } else { 1.0 });
            assert_eq!(out.n_eff, (i - usize::from(i > 12)) as f64, "row {i}");
            assert_eq!(out.pred[0].is_nan(), i < 14, "row {i}");
            assert_eq!(m.predict(&x, 1.0).pred[0].is_nan(), i < 13, "row {i}");
        }
        assert_eq!(m.target_weights(), &[5.0]);
    }

    /// A schema-19 state keeps no target weights: it loads with each target
    /// at the shared weight, the one its gate read.
    #[test]
    fn a_state_without_target_weights_loads_at_the_shared_weight() {
        use crate::{ModelState, OnlineModel, State};
        let mut m = Pa::new(cfg(2, PaMode::Pa1)).unwrap();
        let mut s = 7u64;
        for i in 0..8 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[(i >= 5).then(|| x[0])], 1.0, 1.0);
        }
        assert_eq!(m.target_weights(), &[3.0]);
        let mut v = serde_json::to_value(&m).unwrap();
        assert!(v.as_object_mut().unwrap().remove("w_target").is_some());
        let old: Pa = serde_json::from_value(v).unwrap();
        let back = Pa::restore(&State::new(ModelState::Pa(Box::new(old)))).unwrap();
        assert_eq!(back.target_weights(), &[8.0]);
    }

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Pa::new(cfg(2, PaMode::Pa1)).unwrap();
        let mut s = m.state();
        let ModelState::Pa(inner) = &mut s.model else {
            unreachable!()
        };
        inner.beta[0].pop();
        match Pa::restore(&s) {
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

    fn cfg(k: usize, mode: PaMode) -> PaCfg {
        PaCfg {
            n_features: k,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            mode,
            c: 1.0,
            eps: 0.01,
            min_weight: 5.0,
            constraint: None,
            standardize: false,
        }
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let bad = |f: &dyn Fn(&mut PaCfg), want: &str| {
            let mut c = cfg(2, PaMode::Pa1);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");
        // c is a divisor in PA-2 and a cap in PA-1, so zero is as bad as negative.
        bad(&|c| c.c = 0.0, "c must be > 0");
        bad(&|c| c.c = -1.0, "c must be > 0");
        bad(&|c| c.c = f64::NAN, "c must be > 0");
        // eps is an insensitivity band, so zero is legal (fit exactly).
        bad(&|c| c.eps = -1e-9, "eps must be >= 0");
        bad(&|c| c.eps = f64::NAN, "eps must be >= 0");
        let mut ok = cfg(2, PaMode::Pa1);
        ok.eps = 0.0;
        ok.validate().unwrap();
        cfg(2, PaMode::Pa1).validate().unwrap();
    }

    #[test]
    fn the_three_modes_take_the_step_their_formula_prescribes() {
        // tau is `loss / |z|^2` capped or damped by `c`, and the update is
        // `beta += tau * sign(err) * z`. On the first learning row the state is
        // known exactly, so each mode's step can be computed by hand.
        let one_step = |mode: PaMode, c: f64, y: f64| {
            let mut cfg = cfg(1, mode);
            cfg.c = c;
            cfg.eps = 0.1;
            cfg.min_weight = 0.0;
            let mut m = Pa::new(cfg).unwrap();
            m.step(&[2.0], &[Some(y)], 0.0, 1.0);
            m.coefficients()[0].clone()
        };
        // z = [1, 2] (intercept first), so |z|^2 = 5. beta starts at 0, so
        // err = y; and the target has no spread yet, so the tube has no
        // width and loss = |y| (the module docs; docs/PLAN.md task 202).
        let (y, sq_norm) = (3.0, 5.0);
        let loss = y;

        let pa = one_step(PaMode::Pa, 1.0, y);
        let tau = loss / sq_norm;
        assert!((pa[0] - tau).abs() < 1e-12, "intercept {pa:?}");
        assert!((pa[1] - 2.0 * tau).abs() < 1e-12, "slope {pa:?}");

        // PA-1 caps tau at c: with c above the uncapped value nothing changes,
        // with c below it the step is exactly c * z.
        let pa1_loose = one_step(PaMode::Pa1, 10.0, y);
        assert!(
            (pa1_loose[1] - pa[1]).abs() < 1e-12,
            "uncapped: {pa1_loose:?}"
        );
        let pa1_tight = one_step(PaMode::Pa1, 0.05, y);
        assert!(
            (pa1_tight[1] - 2.0 * 0.05).abs() < 1e-12,
            "capped: {pa1_tight:?}"
        );

        // PA-2 damps the denominator by 1/(2c) rather than capping.
        let c = 0.5;
        let pa2 = one_step(PaMode::Pa2, c, y);
        let tau2 = loss / (sq_norm + 0.5 / c);
        assert!((pa2[1] - 2.0 * tau2).abs() < 1e-12, "{pa2:?}");
        assert!(tau2 < tau, "PA-2 must take a smaller step than PA");

        // The step follows the sign of the error.
        let down = one_step(PaMode::Pa, 1.0, -y);
        assert!((down[1] + pa[1]).abs() < 1e-12, "{down:?} vs {pa:?}");
    }

    #[test]
    fn inside_the_insensitivity_band_nothing_moves() {
        // `loss == 0.0 => continue`: an error smaller than eps of the
        // target's std leaves the coefficients untouched, which is the
        // "passive" half of the name. Two rows, 4 and 6, give the target a
        // spread of 1, `((4 - 5)² + (6 - 5)²) / 2`; before it, no band, so
        // both teach.
        let mut c = cfg(1, PaMode::Pa);
        c.eps = 0.5;
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        m.step(&[1.0], &[Some(4.0)], 0.0, 1.0);
        assert_eq!(m.target_variance(), vec![0.0], "one row: no spread");
        let first = m.coefficients()[0].clone();
        m.step(&[1.0], &[Some(6.0)], 1.0, 1.0);
        let moved = m.coefficients()[0].clone();
        assert!(moved != first, "the second row teaches: no band yet");
        assert_eq!(m.target_variance(), vec![1.0]);

        // Now feed a row it already predicts to within eps·σ_y = 0.5.
        let p: f64 = moved[0] + moved[1];
        assert_eq!(p, 6.0, "the unbounded step lands on the row");
        m.step(&[1.0], &[Some(p + 0.4)], 1.0, 1.0);
        assert_eq!(m.coefficients()[0], moved, "inside the band: passive");
        // The third row joined the spread: `σ_y²` is the variance of 4, 6
        // and 6.4 about their mean, written out below.
        let ys = [4.0, 6.0, 6.4];
        let mean = ys.iter().sum::<f64>() / 3.0;
        let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / 3.0;
        let band = 0.5 * m.target_variance()[0].sqrt();
        assert!((band - 0.5 * var.sqrt()).abs() < 1e-12, "{band}");
        m.step(&[1.0], &[Some(p + band + 0.01)], 1.0, 1.0);
        assert_ne!(m.coefficients()[0], moved, "outside the band: aggressive");
    }

    fn fit(
        cfg: PaCfg,
        n: usize,
        seed: u64,
        mut f: impl FnMut(&[f64], &mut u64) -> f64,
    ) -> Vec<f64> {
        let k = cfg.n_features;
        let mut m = Pa::new(cfg).unwrap();
        let mut s = seed;
        for i in 0..n {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y = f(&x, &mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    }

    #[test]
    fn recovers_a_noiseless_relationship() {
        for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
            let b = fit(cfg(2, mode), 5000, 1, |x, _| 1.5 * x[0] - 0.5 * x[1] + 0.25);
            assert!((b[0] - 0.25).abs() < 0.05, "{mode:?} intercept {}", b[0]);
            assert!((b[1] - 1.5).abs() < 0.05, "{mode:?} slope0 {}", b[1]);
            assert!((b[2] + 0.5).abs() < 0.05, "{mode:?} slope1 {}", b[2]);
        }
    }

    #[test]
    fn passive_inside_the_tube() {
        // With a wide tube and a target already inside it, nothing moves
        // after the first two rows, which have no spread to draw a tube in
        // and so teach; their two values, 0.4 and 0.6, give the target a
        // spread every later row is well inside.
        let mut c = cfg(1, PaMode::Pa1);
        c.eps = 10.0;
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        m.step(&[1.0], &[Some(0.4)], 0.0, 1.0);
        let first = m.coefficients()[0].clone();
        assert_ne!(first, vec![0.0, 0.0], "the first row teaches");
        m.step(&[1.0], &[Some(0.6)], 1.0, 1.0);
        let second = m.coefficients()[0].clone();
        assert_ne!(second, first, "and so does the second");
        for i in 0..100 {
            m.step(&[1.0], &[Some(0.5 + 0.01 * f64::from(i % 3))], 1.0, 1.0);
        }
        assert_eq!(m.coefficients()[0], second);
    }

    #[test]
    fn plain_pa_satisfies_the_constraint_exactly() {
        // One aggressive step must land the prediction on the tube edge.
        let mut c = cfg(1, PaMode::Pa);
        c.eps = 0.0;
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        m.step(&[2.0], &[Some(7.0)], 0.0, 1.0);
        let b = &m.coefficients()[0];
        let p = b[0] + 2.0 * b[1];
        assert!(
            (p - 7.0).abs() < 1e-12,
            "pa did not satisfy the constraint: {p}"
        );
    }

    #[test]
    fn c_caps_the_aggressiveness() {
        // PA-1 with a small c must move far less than plain PA on the same row.
        let step = |mode, c| {
            let mut cf = cfg(1, mode);
            cf.c = c;
            cf.eps = 0.0;
            cf.min_weight = 0.0;
            let mut m = Pa::new(cf).unwrap();
            m.step(&[1.0], &[Some(100.0)], 0.0, 1.0);
            m.coefficients()[0][1]
        };
        let unbounded = step(PaMode::Pa, 1.0);
        let capped = step(PaMode::Pa1, 0.1);
        let damped = step(PaMode::Pa2, 0.1);
        assert!(capped < unbounded, "PA-1 {capped} !< PA {unbounded}");
        assert!(damped < unbounded, "PA-2 {damped} !< PA {unbounded}");
    }

    #[test]
    fn bounded_variants_resist_an_outlier() {
        let mut row = 0u64;
        let contaminated = move |x: &[f64], s: &mut u64| {
            row += 1;
            if row.is_multiple_of(25) {
                500.0 * lcg(s)
            } else {
                2.0 * x[0]
            }
        };
        let mut c1 = cfg(1, PaMode::Pa1);
        c1.c = 0.05;
        let capped = fit(c1, 5000, 3, contaminated);
        let mut row2 = 0u64;
        let unbounded = fit(
            cfg(1, PaMode::Pa),
            5000,
            3,
            move |x: &[f64], s: &mut u64| {
                row2 += 1;
                if row2.is_multiple_of(25) {
                    500.0 * lcg(s)
                } else {
                    2.0 * x[0]
                }
            },
        );
        assert!(
            (capped[1] - 2.0).abs() < (unbounded[1] - 2.0).abs(),
            "PA-1 {} should beat PA {} under contamination (truth 2.0)",
            capped[1],
            unbounded[1]
        );
    }

    #[test]
    fn row_weight_scales_the_step() {
        let step_for = |w: f64| {
            let mut c = cfg(1, PaMode::Pa);
            c.eps = 0.0;
            c.min_weight = 0.0;
            let mut m = Pa::new(c).unwrap();
            m.step(&[1.0], &[Some(4.0)], 0.0, w);
            m.coefficients()[0][1]
        };
        assert!((step_for(0.5) - 0.5 * step_for(1.0)).abs() < 1e-12);
    }

    /// A weight above 1 counts as 1, `weight.min(1.0)`: the step is a
    /// projection onto the row's constraint, and a projection repeated is
    /// the same projection. A row at weight 2 moves every coefficient as the
    /// same row at weight 1 does, to the bit, in each mode, where
    /// `row_weight_scales_the_step`'s half-weight row holds of a step with
    /// no cap as well (review 2026-10-06, CC10). The row is no passive one:
    /// at weight 0.5 it moves the fit elsewhere.
    #[test]
    fn a_weight_above_one_steps_as_a_weight_of_one() {
        for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
            let fit = |w: f64| {
                let mut c = cfg(2, mode);
                c.eps = 0.0;
                c.min_weight = 0.0;
                let mut m = Pa::new(c).unwrap();
                let mut s = 11u64;
                for i in 0..20 {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = 1.5 * x[0] - 0.5 * x[1] + 0.3;
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    m.step(&x, &[Some(y)], d, if i == 7 { w } else { 1.0 });
                }
                m.coefficients()[0].clone()
            };
            let bits = |b: &[f64]| b.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let one = fit(1.0);
            assert_eq!(bits(&fit(2.0)), bits(&one), "{mode:?}");
            assert_eq!(
                bits(&fit(1e100)),
                bits(&one),
                "{mode:?}: at the input bound"
            );
            assert_ne!(
                bits(&fit(0.5)),
                bits(&one),
                "{mode:?}: the row moves the fit"
            );
        }
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut c = cfg(1, PaMode::Pa1);
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..50 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.coefficients()[0].clone();
        let st = m.step(&[0.5], &[None], 1.0, 1.0);
        assert!(st.pred[0].is_finite());
        assert_eq!(m.coefficients()[0], before);
    }

    fn constrained(k: usize, lo: f64, hi: f64, sum: Option<f64>) -> PaCfg {
        let mut c = cfg(k, PaMode::Pa1);
        c.constraint = Some(Constraint {
            lo: vec![lo; k],
            hi: vec![hi; k],
            sum,
        });
        c
    }

    #[test]
    fn a_constraint_is_validated_by_name() {
        let c = constrained(2, 1.0, 0.0, None);
        assert_eq!(
            c.validate().unwrap_err(),
            "pa: coef_min[0] = 1 is above coef_max[0] = 0"
        );
    }

    #[test]
    fn the_projection_follows_every_update() {
        // Slopes on the simplex after every row, including the start, and a
        // truth on it is recovered; the fit is no longer exact on each row.
        let mut m = Pa::new(constrained(3, 0.0, f64::INFINITY, Some(1.0))).unwrap();
        for b in &m.coefficients()[0][1..] {
            assert!((b - 1.0 / 3.0).abs() <= 1e-15);
        }
        let mut s = 5u64;
        for i in 0..5000 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 0.2 * x[0] + 0.5 * x[1] + 0.3 * x[2];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let b = &m.coefficients()[0][1..];
            assert!(b.iter().all(|v| *v >= 0.0), "row {i}: {b:?}");
            assert!(
                (b.iter().sum::<f64>() - 1.0).abs() <= 1e-12,
                "row {i}: {b:?}"
            );
        }
        let b = &m.coefficients()[0];
        for (got, want) in b[1..].iter().zip([0.2, 0.5, 0.3]) {
            assert!((got - want).abs() < 0.03, "{b:?}");
        }
    }

    #[test]
    fn a_box_holds_where_the_truth_lies_outside_it() {
        // A truth outside the box is never realizable, so PA keeps stepping;
        // the projection keeps every step inside the box, and with a small
        // cap `c` the fit sits against the walls the truth is behind.
        let mut c = constrained(2, 0.0, 0.5, None);
        c.c = 0.01;
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        let mut s = 3u64;
        let mut bound = 0;
        for i in 0..5000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = -0.4 * x[0] + 0.9 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let b = &m.coefficients()[0][1..];
            assert!(b.iter().all(|v| (0.0..=0.5).contains(v)), "row {i}: {b:?}");
            bound += usize::from(b[0] == 0.0 || b[1] == 0.5);
        }
        let b = &m.coefficients()[0];
        assert!(b[1] < 0.05 && b[2] > 0.45, "{b:?}");
        assert!(bound > 4000, "the box bound on {bound} of 5000 rows");
    }

    #[test]
    fn a_constrained_state_roundtrips() {
        let mut m1 = Pa::new(constrained(2, -1.0, 1.0, Some(0.5))).unwrap();
        let mut s = 31u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m1.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Pa::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(m2.cfg().constraint, m1.cfg().constraint);
        for _ in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            assert_eq!(
                m1.step(&x, &[Some(x[0])], 1.0, 1.0).pred,
                m2.step(&x, &[Some(x[0])], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Pa::new(cfg(2, PaMode::Pa2)).unwrap();
        let mut s = 11u64;
        let rows: Vec<([f64; 2], f64)> = (0..120)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..60].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Pa::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut c = cfg(1, PaMode::Pa1);
        c.c = 0.0;
        assert!(Pa::new(c).is_err());
        let mut c = cfg(1, PaMode::Pa1);
        c.eps = -1.0;
        assert!(Pa::new(c).is_err());
    }

    /// Rows that teach nothing move nothing, to the bit. Under a sum
    /// constraint that holds only if such a row is never projected: the
    /// projection is not idempotent in the last bits (projecting its own
    /// output moves a coordinate by an ulp about two times in five). So: a
    /// zero-weight row however far off the fit (hard rule 9), and a target
    /// that is not finite.
    #[test]
    fn a_row_that_teaches_nothing_moves_nothing() {
        use crate::OnlineModel;
        let bits = |m: &Pa| {
            m.coefficients()
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        let mut c = cfg(3, PaMode::Pa1);
        c.min_weight = 0.0;
        c.constraint = Some(Constraint {
            lo: vec![f64::NEG_INFINITY; 3],
            hi: vec![f64::INFINITY; 3],
            sum: Some(1.0),
        });
        let mut m = Pa::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..40 {
            let x = [3.0 * lcg(&mut s), 3.0 * lcg(&mut s), 3.0 * lcg(&mut s)];
            let y = x[0] - 2.0 * x[1] + 0.5 * lcg(&mut s);
            m.step(&x, &[Some(y)], 1.0, 1.0);
            let before = bits(&m);
            m.step(&x, &[Some(y + 10.0)], 1.0, 0.0);
            assert_eq!(bits(&m), before, "row {i}: a zero weight");
            m.step(&x, &[Some(f64::INFINITY)], 1.0, 1.0);
            assert_eq!(bits(&m), before, "row {i}: an infinite target");
        }
    }

    /// A row whose prediction overflows takes no step: its loss is infinite,
    /// and a step from it would be permanent. One tiny feature makes a
    /// coefficient of `1e204`; a coefficient of `1e300` takes a feature at
    /// [`crate::INPUT_BOUND`] past `f64::MAX`. That one is written into the
    /// state: a step of the size that makes it overflows `tau` first, and a
    /// feature of `1e300`, which reached the same overflow from `1e204`, is
    /// past the bound and refused before it is read (task 183).
    #[test]
    fn an_overflowing_prediction_takes_no_step() {
        use crate::OnlineModel;
        let mut c = cfg(1, PaMode::Pa);
        c.fit_intercept = false;
        c.min_weight = 0.0;
        let mut m = Pa::new(c).unwrap();
        m.step(&[1e-104], &[Some(1e100)], 1.0, 1.0);
        let big = m.coefficients()[0][0];
        assert!(big > 1e200 && big.is_finite(), "{big}");
        m.beta[0][0] = 1e300;
        let p = m.predict(&[crate::INPUT_BOUND], 1.0).pred[0];
        assert!(p.is_infinite(), "the prediction overflows: {p}");
        m.step(&[crate::INPUT_BOUND], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(m.coefficients()[0][0], 1e300);
        // A feature past the bound is refused, and moves nothing either.
        m.step(&[1e300], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(m.coefficients()[0][0], 1e300);
    }

    /// The bank's per-target `min_weight` gate reads each target's own
    /// weight through the trait: `true`, and one entry a target, the decayed
    /// weight of the rows that carried it -- a sparse second target its own
    /// rows only (task 158, E14; the mutation run of 2026-10-05 left the
    /// body replaced by `true` or `false` alive).
    #[test]
    fn the_trait_reports_each_targets_own_weight() {
        use crate::OnlineModel;
        let mut c = cfg(2, PaMode::Pa1);
        c.n_targets = 2;
        c.decay = Decay::Halflife(10.0);
        let mut m = Pa::new(c).unwrap();
        let lam = Decay::Halflife(10.0).factor(1.0);
        let (mut want0, mut want1) = (0.0f64, 0.0f64);
        let mut s = 3u64;
        for i in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let w = 0.5 + lcg(&mut s).abs();
            let y1 = (i % 4 == 1).then_some(x[1]);
            m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, w);
            let f = if i == 0 { 1.0 } else { lam };
            want0 = f * want0 + w;
            want1 = f * want1 + if y1.is_some() { w } else { 0.0 };
        }
        let mut out = vec![7.0; 5];
        assert!(m.target_n_eff_into(&mut out));
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(
            (out[0] - want0).abs() <= 1e-12 * want0,
            "{out:?} against {want0}"
        );
        assert!(
            (out[1] - want1).abs() <= 1e-12 * want1,
            "{out:?} against {want1}"
        );
        assert!(
            out[1] < 0.5 * out[0],
            "the sparse target's own weight: {out:?}"
        );
    }

    /// `eps` is in units of the target's own EW standard deviation, and
    /// that unit scales with the target: a target scaled by a power of two
    /// fits as the unscaled one does, scaled by it, to the bit, under the
    /// unbounded step, whose `tau = loss / |z|²` scales with the target (a
    /// finite `c` caps it in the target's units). The stream sits at a level
    /// of 1,000 in a spread of 2, with and without the scaler. In the
    /// target's units a target in thousandths sat inside the tube on every
    /// row, passive for ever, and one in thousands never did (review round
    /// 4, CC4; docs/PLAN.md task 195, U1, and task 202).
    #[test]
    fn the_band_scales_with_the_target() {
        for standardize in [false, true] {
            let run = |c: f64| -> Vec<u64> {
                let mut cf = cfg(2, PaMode::Pa);
                cf.eps = 0.1;
                cf.min_weight = 0.0;
                cf.standardize = standardize;
                let mut m = Pa::new(cf).unwrap();
                let mut s = 5u64;
                (0..400)
                    .map(|i| {
                        let x = [lcg(&mut s), lcg(&mut s)];
                        let y = c * (1000.5 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s));
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        let p = m.step(&x, &[Some(y)], d, 1.0).pred[0] / c;
                        if p.is_nan() {
                            f64::NAN.to_bits()
                        } else {
                            p.to_bits()
                        }
                    })
                    .collect()
            };
            let one = run(1.0);
            for c in [2f64.powi(-10), 2f64.powi(10)] {
                let scaled = run(c);
                let differ = scaled.iter().zip(&one).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differ, 0,
                    "standardize {standardize}, at scale {c}: {differ} of 400 rows differ"
                );
            }
        }
    }

    /// `y = level + 0.5 + 2x + 0.3·noise`, `x` and the noise of unit
    /// variance: a spread of about 2 around `level`. Rows of `(x, y)`.
    fn level_stream(level: f64, n: usize, seed: u64) -> Vec<(f64, f64)> {
        let mut s = seed;
        let root3 = 3f64.sqrt();
        (0..n)
            .map(|_| {
                let x = root3 * lcg(&mut s);
                (x, level + 0.5 + 2.0 * x + 0.3 * root3 * lcg(&mut s))
            })
            .collect()
    }

    /// R² of the predictions of rows `from..` against their targets.
    fn r2_from(preds: &[f64], ys: &[f64], from: usize) -> f64 {
        let (p, y) = (&preds[from..], &ys[from..]);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let sse: f64 = p.iter().zip(y).map(|(p, y)| (y - p) * (y - p)).sum();
        let sst: f64 = y.iter().map(|y| (y - mean) * (y - mean)).sum();
        1.0 - sse / sst
    }

    /// `pa` at the builder's defaults -- `pa1`, `c = 1`, `eps = 0.1`, the
    /// scaler on, `min_weight` 2 -- on a target at a level of 1,000 in a
    /// spread of about 2 fits rows 10,000 to 20,000 as it fits the same
    /// target at 0, with or without decay. The fit starts from zero
    /// coefficients, so its first residuals are the whole level: a band in
    /// units of the residual's spread learned from them was about 100 wide,
    /// every later row inside it, and without decay nothing ever narrowed
    /// it: R² -52 at a half-life of 1e9 against +0.954 at 500 (task 195's
    /// report; docs/PLAN.md task 202). The target's own spread is about 2
    /// from its second row on, whatever the fit.
    #[test]
    fn a_target_far_from_zero_is_learned_with_or_without_decay() {
        for half_life in [1e9, 500.0] {
            for level in [0.0, 1000.0] {
                let mut m = Pa::new(PaCfg {
                    n_features: 1,
                    n_targets: 1,
                    fit_intercept: true,
                    decay: Decay::Halflife(half_life),
                    mode: PaMode::Pa1,
                    c: 1.0,
                    eps: 0.1,
                    min_weight: 2.0,
                    constraint: None,
                    standardize: true,
                })
                .unwrap();
                let rows = level_stream(level, 20_000, 2);
                let preds: Vec<f64> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, (x, y))| {
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        m.step(&[*x], &[Some(*y)], d, 1.0).pred[0]
                    })
                    .collect();
                let ys: Vec<f64> = rows.iter().map(|(_, y)| *y).collect();
                let r2 = r2_from(&preds, &ys, 10_000);
                assert!(
                    r2 > 0.9,
                    "half-life {half_life}, level {level}: R² of rows 10k-20k {r2:.3}"
                );
            }
        }
    }

    /// A target held at one value has no spread, so no band: every row
    /// teaches, and the fit reaches the value, from zero, in every mode: at
    /// a level of 1,000 under the default cap of 1, which takes a thousand
    /// rows to climb, and at 1e8 under a cap that does not bind. A band in
    /// the residual's spread was drawn from the climb's residuals, the
    /// level itself, and without decay the capped fit stopped short of the
    /// value by a share of it for good (docs/PLAN.md task 202).
    #[test]
    fn a_target_held_constant_is_learned() {
        for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
            for (level, c) in [(1000.0, 1.0), (1e8, 1e9)] {
                let mut cf = cfg(1, mode);
                cf.c = c;
                cf.eps = 0.1;
                cf.min_weight = 2.0;
                cf.standardize = true;
                let mut m = Pa::new(cf).unwrap();
                let mut s = 3u64;
                let mut last = f64::NAN;
                for i in 0..4000 {
                    let x = [lcg(&mut s)];
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    last = m.step(&x, &[Some(level)], d, 1.0).pred[0];
                }
                assert!(
                    (last - level).abs() <= 1e-9 * level,
                    "{mode:?} at {level}: the last prediction {last}"
                );
            }
        }
    }

    /// Under `standardize` the step reads the features standardized against
    /// `sgd`'s scaler, so features scaled by a power of two predict the same
    /// numbers, to the bit, in every mode, and the coefficients come back
    /// in each scale's own units, a slope scaled by the inverse; `predict`
    /// is the next step's number exactly; and a state round-trips mid-stream
    /// (docs/PLAN.md task 195, U2). The raw step is the control.
    #[test]
    fn standardize_makes_the_fit_free_of_the_features_units() {
        for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
            let run = |c: f64, standardize: bool| {
                let mut cf = cfg(2, mode);
                cf.standardize = standardize;
                cf.eps = 0.1;
                let mut m = Pa::new(cf).unwrap();
                let mut s = 5u64;
                let mut preds = Vec::new();
                for i in 0..300 {
                    let x = [c * lcg(&mut s), c * (3.0 + lcg(&mut s))];
                    let y = 0.5 + 2.0 * x[0] / c - x[1] / c + 0.2 * lcg(&mut s);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let ahead = m.predict(&x, d).pred[0];
                    let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
                    assert_eq!(ahead.to_bits(), p.to_bits(), "{mode:?}, row {i}");
                    preds.push(p.to_bits());
                }
                let back = Pa::restore(&m.state()).unwrap();
                assert_eq!(back.coefficients(), m.coefficients(), "{mode:?}");
                (preds, m.coefficients()[0].clone())
            };
            let (one, coef) = run(1.0, true);
            let (scaled, scaled_coef) = run(128.0, true);
            assert_eq!(scaled, one, "{mode:?}");
            for i in 1..3 {
                assert!(
                    (scaled_coef[i] * 128.0 - coef[i]).abs() <= 1e-12 * coef[i].abs(),
                    "{mode:?}: {scaled_coef:?} against {coef:?}"
                );
            }
            assert!(
                (coef[1] - 2.0).abs() < 0.2 && (coef[2] + 1.0).abs() < 0.2,
                "{coef:?}"
            );
            assert_ne!(run(128.0, false).0, run(1.0, false).0, "{mode:?}: raw");
        }
    }

    /// The tube's unit is its definition: target `j`'s EW variance around
    /// its EW mean, `Σ ωᵢ (yᵢ − m)² / Σ ωᵢ` with `m = Σ ωᵢ yᵢ / Σ ωᵢ`, over
    /// the rows that carried the target with a weight above 0 -- those before
    /// `min_weight` and those the fit got wrong included, since it does not
    /// read the fit -- `ωᵢ` aged by the model's decay over every row since,
    /// rebuilt from the history at every row in two passes, not by the
    /// recursion. Irregular steps and weights, null targets and rows of
    /// weight 0, at a level of 1,000 (docs/PLAN.md task 202).
    #[test]
    fn the_target_variance_is_the_ew_variance_of_the_target() {
        use crate::OnlineModel;
        let mut c = cfg(2, PaMode::Pa1);
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 3.0;
        c.standardize = true;
        let mut m = Pa::new(c.clone()).unwrap();
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let mut s = 7u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i % 9 != 4).then(|| 1000.0 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s));
            let w = if i % 13 == 6 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
            let lam = c.decay.factor(d);
            m.step(&x, &[y], d, w);
            for (_, wi) in seen.iter_mut() {
                *wi *= lam;
            }
            if let Some(v) = y
                && w > 0.0
            {
                seen.push((v, w));
            }
            let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
            let want = if seen.len() < 2 {
                0.0
            } else {
                let mean = seen.iter().map(|(v, wi)| v * wi).sum::<f64>() / total;
                seen.iter()
                    .map(|(v, wi)| wi * (v - mean) * (v - mean))
                    .sum::<f64>()
                    / total
            };
            let got = m.target_variance()[0];
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "row {i}: σ_y² {got} against {want}"
            );
            assert_eq!(m.tube(0), c.eps * got.sqrt(), "row {i}");
        }
        assert!(seen.len() > 150, "{} rows taught it", seen.len());
    }
}
