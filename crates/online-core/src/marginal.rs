//! `marginal`: exponentially weighted moments of every feature against every
//! target, one pair at a time (docs/ENHANCEMENTS.md E44).
//!
//! An `ew_cov` over `[x_1..x_p, y_1..y_T]` keeps the whole `(p+T)²` Gram.
//! This model keeps the diagonal and the cross column -- per (target `t`,
//! feature `j`) the two means, the two centred variances and the covariance
//! -- which is `O(p·T)` state and `O(p·T)` work per row where the Gram is
//! `O(p²)`. It emits nothing per row but `n_eff`; its value is its state,
//! read back per pair with [`Marginal::pair`].
//!
//! Per target `t`, on a row where `y_t` is present, with weight `w` and decay
//! factor `lam`:
//!
//! ```text
//! W'_t       = lam·W_t + w            a = lam·W_t / W'_t        b = w / W'_t
//! Q'_t       = lam²·Q_t + w²                                    (Σw², for Kish)
//! dy         = y_t − m_y[t]           dx_j = x_j − m_x[t,j]
//! S'_yy[t]   = a·S_yy[t]   + a·b·dy·dy
//! S'_xx[t,j] = a·S_xx[t,j] + a·b·dx_j·dx_j
//! S'_xy[t,j] = a·S_xy[t,j] + a·b·dx_j·dy
//! m_y[t]    += b·dy                   m_x[t,j] += b·dx_j
//! ```
//!
//! This is the weighted Welford form [`crate::EwCov`] uses, operation for
//! operation, so the pair `(x_j, y_t)` here and the pair in an `ew_cov` over
//! `[x_j, y_t]` fed the same rows give the same correlation to the bit
//! (`tests/test_marginal.py` holds them to it). A missing `y_t` ages the
//! target's accumulators (`W_t·lam`, `Q_t·lam²`) and moves nothing else, as
//! a missing target does in `ew_ridge`. The feature moments are kept per
//! *pair*, not per feature: they are over the rows where the target was
//! present, so both sides of a correlation are over the same rows whatever
//! the target's missingness.
//!
//! Read back per pair ([`Pair`]): `n_eff = W_t`; `n_kish = W_t² / Q_t`,
//! Kish's effective sample size -- `(1+lam)/(1−lam)` in the limit for unit
//! weights, about twice `n_eff`, and the `n` a standard error wants; the
//! moments; and from them `corr = S_xy / √(S_xx·S_yy)`, `beta = S_xy /
//! S_xx` (the slope of `y` on `x`) and `t = corr·√((n_kish − 2) / (1 −
//! corr²))`, the t-statistic of the correlation at Kish's `n`. `corr`, `beta`
//! and `t` are NaN while `W_t < min_periods`; the moments are always
//! reported. The t is descriptive: the rows of a stream are rarely
//! independent, and nothing here pretends otherwise.

use serde::{Deserialize, Serialize};

use crate::{Decay, OnlineModel, State, StateError, Step};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub decay: Decay,
    /// Weight each target must have accumulated before its pairs' `corr`,
    /// `beta` and `t` are reported, one entry per target; the moments never
    /// wait.
    pub min_periods: Vec<f64>,
}

impl MarginalCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("marginal: at least one feature is required".into());
        }
        if self.n_targets == 0 {
            return Err("marginal: at least one target is required".into());
        }
        if self.min_periods.len() != self.n_targets {
            return Err(format!(
                "marginal: min_periods has {} entries for {} targets",
                self.min_periods.len(),
                self.n_targets
            ));
        }
        if let Some(bad) = self
            .min_periods
            .iter()
            .find(|v| v.is_nan() || v.is_infinite() || **v < 0.0)
        {
            return Err(format!(
                "marginal: min_periods must be finite and >= 0, got {bad}"
            ));
        }
        Ok(())
    }
}

/// One (feature, target) pair as the state stands (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    /// Accumulated weight behind the pair: `W_t`, the rows where the target
    /// was present.
    pub n_eff: f64,
    /// Kish's effective sample size `W_t² / Q_t`; NaN before the first row.
    pub n_kish: f64,
    pub mean_x: f64,
    /// Centred variance of the feature over the rows the target was present.
    pub var_x: f64,
    pub mean_y: f64,
    pub var_y: f64,
    /// Centred covariance.
    pub cov: f64,
    /// `cov / √(var_x·var_y)`, clamped to `[-1, 1]`; NaN when either side is
    /// constant, or below `min_periods`.
    pub corr: f64,
    /// The slope of `y` on `x`, `cov / var_x`; NaN when the feature is
    /// constant, or below `min_periods`.
    pub beta: f64,
    /// `corr·√((n_kish − 2) / (1 − corr²))`; NaN when `n_kish <= 2`, or
    /// below `min_periods`. Enormous or `±inf` for a perfect correlation,
    /// which is the honest value.
    pub t: f64,
}

/// See the module doc. Vectors indexed `[t * n_features + j]` are per pair;
/// the ones of length `n_targets` are per target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marginal {
    cfg: MarginalCfg,
    /// Accumulated weight of every learned row, targets present or not: the
    /// model's `n_eff`.
    w_sum: f64,
    /// `W_t`: accumulated weight of the rows where target `t` was present.
    wt: Vec<f64>,
    /// `Q_t`: accumulated squared weight of the same rows.
    qt: Vec<f64>,
    my: Vec<f64>,
    syy: Vec<f64>,
    mx: Vec<f64>,
    sxx: Vec<f64>,
    sxy: Vec<f64>,
}

impl Marginal {
    pub fn new(cfg: MarginalCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (p, t) = (cfg.n_features, cfg.n_targets);
        Ok(Self {
            cfg,
            w_sum: 0.0,
            wt: vec![0.0; t],
            qt: vec![0.0; t],
            my: vec![0.0; t],
            syy: vec![0.0; t],
            mx: vec![0.0; p * t],
            sxx: vec![0.0; p * t],
            sxy: vec![0.0; p * t],
        })
    }

    pub fn cfg(&self) -> &MarginalCfg {
        &self.cfg
    }

    /// The accumulated weight of every learned row, as the next row's
    /// `n_eff` reports it (CLAUDE.md rule 8).
    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// `W_t`, the weight behind target `t`'s pairs.
    pub fn target_weight(&self, t: usize) -> f64 {
        self.wt[t]
    }

    /// The statistics of feature `j` against target `t`. The moments are
    /// reported at any weight; `corr`, `beta` and `t` wait for
    /// `min_periods`.
    pub fn pair(&self, t: usize, j: usize) -> Pair {
        let i = t * self.cfg.n_features + j;
        let n_eff = self.wt[t];
        // 0/0 before the first row, which is the NaN it should be.
        let n_kish = n_eff * n_eff / self.qt[t];
        let var_x = self.sxx[i].max(0.0);
        let var_y = self.syy[t].max(0.0);
        let cov = self.sxy[i];
        let (corr, beta, t_stat) = if n_eff >= self.cfg.min_periods[t] {
            // The same product of the same two roots `ew_cov` takes, so the
            // correlations agree to the bit.
            let d = var_x.sqrt() * var_y.sqrt();
            let corr = if d > 0.0 {
                (cov / d).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            };
            let beta = if var_x > 0.0 { cov / var_x } else { f64::NAN };
            let t_stat = if n_kish > 2.0 {
                corr * ((n_kish - 2.0) / (1.0 - corr * corr)).sqrt()
            } else {
                f64::NAN
            };
            (corr, beta, t_stat)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        Pair {
            n_eff,
            n_kish,
            mean_x: self.mx[i],
            var_x,
            mean_y: self.my[t],
            var_y,
            cov,
            corr,
            beta,
            t: t_stat,
        }
    }

    fn learn(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        debug_assert!(w >= 0.0, "marginal requires a non-negative weight, got {w}");
        if w < 0.0 {
            return;
        }
        let p = self.cfg.n_features;
        // The model-level weight: every row, present targets or not. A
        // zero-weight first row leaves it at zero, which is legal (rule 9).
        self.w_sum = lam * self.w_sum + w;
        for (t, yt) in y.iter().enumerate() {
            let Some(yt) = *yt else {
                // Time passes for a target that is not there: its weight
                // ages, its moments hold, as `ew_ridge` treats a missing
                // target.
                self.wt[t] *= lam;
                self.qt[t] *= lam * lam;
                continue;
            };
            let w_new = lam * self.wt[t] + w;
            if w_new <= 0.0 {
                // No weight in the history and none on this row: nothing to
                // average, and `a`/`b` would be 0/0 (CLAUDE.md rule 9).
                continue;
            }
            let a = lam * self.wt[t] / w_new;
            let b = w / w_new;
            let dy = yt - self.my[t];
            // Co-moments from the deviations against the OLD means, then the
            // means advance -- `EwCov::update`'s order, operation for
            // operation, so the pair agrees with `ew_cov` to the bit.
            self.syy[t] = a * self.syy[t] + a * b * dy * dy;
            let row = t * p;
            for (i, xj) in (row..row + p).zip(x) {
                let dx = xj - self.mx[i];
                self.sxx[i] = a * self.sxx[i] + a * b * dx * dx;
                self.sxy[i] = a * self.sxy[i] + a * b * dx * dy;
            }
            for (mi, xj) in self.mx[row..row + p].iter_mut().zip(x) {
                *mi += b * (xj - *mi);
            }
            self.my[t] += b * dy;
            self.wt[t] = w_new;
            self.qt[t] = lam * lam * self.qt[t] + w * w;
        }
    }
}

impl OnlineModel for Marginal {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let out = self.predict(x, d_clock);
        self.learn(x, y, self.cfg.decay.factor(d_clock), weight);
        out
    }

    /// No prediction slots: the step reports `n_eff` and nothing else.
    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        Step {
            pred: Vec::new(),
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(crate::ModelState::Marginal(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Marginal(m) => Ok((**m).clone()),
            other => Err(StateError::WrongModel {
                expected: "marginal",
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

    /// Zero: the model predicts nothing per row. Its outputs are read from
    /// the state with [`Marginal::pair`].
    fn n_outputs(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EwCov, EwCovCfg, EwCovModel, EwCovStat};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(p: usize, t: usize) -> MarginalCfg {
        MarginalCfg {
            n_features: p,
            n_targets: t,
            decay: Decay::Halflife(20.0),
            min_periods: vec![0.0; t],
        }
    }

    /// Weighted, decayed moments of one pair written out longhand as sums
    /// over the rows the target was present: the oracle the recursion is
    /// held to.
    struct Longhand {
        xs: Vec<f64>,
        ys: Vec<f64>,
        ws: Vec<f64>,
        lams: Vec<f64>,
    }

    impl Longhand {
        /// The decay each row has suffered by the end: the product of the
        /// factors of every later row (present or not -- time passes for
        /// them all).
        fn weights(&self) -> Vec<f64> {
            let n = self.ws.len();
            (0..n)
                .map(|i| self.ws[i] * self.lams[i + 1..].iter().product::<f64>())
                .collect()
        }

        fn moments(&self) -> (f64, f64, f64, f64, f64, f64, f64) {
            let w = self.weights();
            let sw: f64 = w.iter().sum();
            let sq: f64 = w.iter().map(|v| v * v).sum();
            let mx = w.iter().zip(&self.xs).map(|(w, x)| w * x).sum::<f64>() / sw;
            let my = w.iter().zip(&self.ys).map(|(w, y)| w * y).sum::<f64>() / sw;
            let sxx = w
                .iter()
                .zip(&self.xs)
                .map(|(w, x)| w * (x - mx) * (x - mx))
                .sum::<f64>()
                / sw;
            let syy = w
                .iter()
                .zip(&self.ys)
                .map(|(w, y)| w * (y - my) * (y - my))
                .sum::<f64>()
                / sw;
            let sxy = w
                .iter()
                .zip(&self.xs)
                .zip(&self.ys)
                .map(|((w, x), y)| w * (x - mx) * (y - my))
                .sum::<f64>()
                / sw;
            (sw, sq, mx, my, sxx, syy, sxy)
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn validation_rejects_each_bad_field() {
        let err = |c: MarginalCfg, what: &str| {
            let e = Marginal::new(c).unwrap_err();
            assert!(e.contains(what), "{e}");
        };
        err(cfg(0, 1), "at least one feature");
        err(cfg(1, 0), "at least one target");
        let mut c = cfg(1, 1);
        c.min_periods = vec![-1.0];
        err(c.clone(), "min_periods");
        c.min_periods = vec![f64::INFINITY];
        err(c.clone(), "min_periods");
        c.min_periods = vec![f64::NAN];
        err(c.clone(), "min_periods");
        c.min_periods = vec![1.0, 2.0];
        err(c.clone(), "2 entries for 1 targets");
        c.min_periods = vec![];
        err(c, "0 entries for 1 targets");
        Marginal::new(cfg(3, 2)).unwrap();
    }

    #[test]
    fn matches_the_longhand_moments_with_weights_gaps_and_a_missing_target() {
        // Two targets, the second missing on every third row, uneven weights
        // (some zero), and clock gaps: each target's pair moments must equal
        // the decayed weighted sums over the rows where that target was
        // present, with the decay of every row -- present or not -- applied.
        let mut m = Marginal::new(cfg(3, 2)).unwrap();
        let mut s = 7u64;
        let mut long: Vec<Vec<Longhand>> = (0..2)
            .map(|_| {
                (0..3)
                    .map(|_| Longhand {
                        xs: vec![],
                        ys: vec![],
                        ws: vec![],
                        lams: vec![],
                    })
                    .collect()
            })
            .collect();
        for i in 0..300 {
            let x: Vec<f64> = (0..3).map(|_| 2.0 * lcg(&mut s) + 100.0).collect();
            let y0 = x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
            let y1 = -x[2] + 0.1 * lcg(&mut s);
            let y = [Some(y0), (i % 3 != 2).then_some(y1)];
            let w = match i % 7 {
                0 => 0.0,
                1 => 2.5,
                _ => 1.0,
            };
            let d = if i == 0 {
                0.0
            } else if i % 50 == 0 {
                15.0
            } else {
                1.0
            };
            let lam = m.cfg().decay.factor(d);
            for (t, yt) in y.iter().enumerate() {
                for j in 0..3 {
                    let l = &mut long[t][j];
                    // Every row ages the target's history; only a present
                    // target adds a row.
                    // A missing target is a zero-weight row in the longhand:
                    // it adds nothing to the sums and `weights()` still
                    // applies its decay factor to every earlier row.
                    let (xv, yv, wv) = match yt {
                        Some(yt) => (x[j], *yt, w),
                        None => (0.0, 0.0, 0.0),
                    };
                    l.xs.push(xv);
                    l.ys.push(yv);
                    l.ws.push(wv);
                    l.lams.push(lam);
                }
            }
            let step = m.step(&x, &y, d, w);
            assert!(step.pred.is_empty());
            assert!(step.n_eff.is_finite());
        }
        for (t, row) in long.iter().enumerate() {
            for (j, l) in row.iter().enumerate() {
                let (sw, sq, mx, my, sxx, syy, sxy) = l.moments();
                let p = m.pair(t, j);
                assert!(close(p.n_eff, sw, 1e-12), "W_{t}: {} vs {sw}", p.n_eff);
                assert!(
                    close(p.n_kish, sw * sw / sq, 1e-12),
                    "kish_{t}: {} vs {}",
                    p.n_kish,
                    sw * sw / sq
                );
                assert!(
                    close(p.mean_x, mx, 1e-12),
                    "mx[{t},{j}] {} vs {mx}",
                    p.mean_x
                );
                assert!(close(p.mean_y, my, 1e-12), "my[{t}] {} vs {my}", p.mean_y);
                // Centred second moments around an offset of 100: the
                // Welford form keeps them to ~1e-12 relative; a raw
                // `E[x²] − m²` would have lost them.
                assert!(
                    close(p.var_x, sxx, 1e-9),
                    "sxx[{t},{j}] {} vs {sxx}",
                    p.var_x
                );
                assert!(close(p.var_y, syy, 1e-9), "syy[{t}] {} vs {syy}", p.var_y);
                assert!(close(p.cov, sxy, 1e-9), "sxy[{t},{j}] {} vs {sxy}", p.cov);
                let corr = sxy / (sxx * syy).sqrt();
                assert!(
                    close(p.corr, corr, 1e-9),
                    "corr[{t},{j}] {} vs {corr}",
                    p.corr
                );
                assert!(close(p.beta, sxy / sxx, 1e-9), "beta[{t},{j}]");
                let n = sw * sw / sq;
                let tt = corr * ((n - 2.0) / (1.0 - corr * corr)).sqrt();
                assert!(close(p.t, tt, 1e-9), "t[{t},{j}] {} vs {tt}", p.t);
            }
        }
        // The model-level weight counts every row, including the ones where
        // a target was missing: it is `W_0` here, since target 0 was always
        // present.
        assert_eq!(m.n_eff().to_bits(), m.target_weight(0).to_bits());
        assert!(m.target_weight(1) < m.target_weight(0));
    }

    #[test]
    fn a_pair_is_the_ew_cov_of_the_two_columns_to_the_bit() {
        // `ew_cov` over `[x_j, y]` and the pair `(j, 0)` here, fed the same
        // rows: the same recursion in the same order gives the same bits --
        // moments, and the correlation.
        let mut m = Marginal::new(cfg(2, 1)).unwrap();
        let mut covs = [EwCov::new(2), EwCov::new(2)];
        let mut full = EwCovModel::new(EwCovCfg {
            n_features: 2,
            decay: Decay::Halflife(20.0),
            stats: vec![EwCovStat::Corr],
            min_periods: 0.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 1,
        })
        .unwrap();
        let mut s = 99u64;
        for i in 0..200 {
            let x = [lcg(&mut s) * 3.0, lcg(&mut s) + 5.0];
            let y = x[0] + 0.7 * x[1] + 0.2 * lcg(&mut s);
            let w = if i % 5 == 0 {
                0.0
            } else {
                1.0 + 0.5 * (i % 4) as f64
            };
            let d = if i == 0 { 0.0 } else { 0.5 + (i % 3) as f64 };
            let lam = m.cfg().decay.factor(d);
            // The ew_cov reference is read *before* the row too.
            let corr_ref = full.step(&[x[0], y], &[], d, w).pred[0];
            let before = m.pair(0, 0);
            if i > 0 {
                assert_eq!(before.corr.to_bits(), corr_ref.to_bits(), "row {i}");
            }
            m.step(&x, &[Some(y)], d, w);
            for (j, c) in covs.iter_mut().enumerate() {
                c.update(&[x[j], y], lam, w);
            }
        }
        for (j, c) in covs.iter().enumerate() {
            let p = m.pair(0, j);
            assert_eq!(p.n_eff.to_bits(), c.n_eff().to_bits());
            assert_eq!(p.mean_x.to_bits(), c.mean(0).to_bits());
            assert_eq!(p.mean_y.to_bits(), c.mean(1).to_bits());
            assert_eq!(p.var_x.to_bits(), c.var(0).to_bits());
            assert_eq!(p.var_y.to_bits(), c.var(1).to_bits());
            assert_eq!(p.cov.to_bits(), c.cov(0, 1).to_bits());
        }
    }

    #[test]
    fn n_eff_is_the_weight_before_the_row_and_min_periods_gates_the_derived_values() {
        let mut c = cfg(1, 1);
        c.min_periods = vec![3.0];
        let mut m = Marginal::new(c).unwrap();
        let lam = 0.5f64.powf(1.0 / 20.0);
        let mut expect = 0.0;
        for i in 0..6 {
            let x = [i as f64];
            let step = m.step(
                &x,
                &[Some(2.0 * i as f64 + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            assert_eq!(step.n_eff, expect, "row {i}");
            expect = if i == 0 { 1.0 } else { lam * expect + 1.0 };
            let p = m.pair(0, 0);
            // The moments are there from the first row; the derived values
            // wait for the weight.
            assert!(p.mean_x.is_finite());
            if p.n_eff < 3.0 {
                assert!(
                    p.corr.is_nan() && p.beta.is_nan() && p.t.is_nan(),
                    "row {i}"
                );
            } else {
                assert!(
                    (p.corr - 1.0).abs() < 1e-12,
                    "y = 2x + 1: corr 1, got {}",
                    p.corr
                );
                assert!((p.beta - 2.0).abs() < 1e-9, "beta 2, got {}", p.beta);
                // `corr` is 1 to rounding, so `1 − corr²` is tiny or zero
                // and t is enormous or +inf: either is the honest value.
                assert!(p.t > 1e3, "perfect fit: t huge, got {}", p.t);
            }
        }
    }

    #[test]
    fn a_zero_weight_first_row_and_a_constant_column_stay_finite() {
        let mut m = Marginal::new(cfg(2, 1)).unwrap();
        // Weight 0 on the very first row: nothing to average, no 0/0.
        let s0 = m.step(&[1.0, 2.0], &[Some(3.0)], 0.0, 0.0);
        assert_eq!(s0.n_eff, 0.0);
        let p = m.pair(0, 0);
        assert_eq!(p.n_eff, 0.0);
        assert!(p.n_kish.is_nan(), "0/0 before any weight");
        assert_eq!(p.mean_x, 0.0);
        assert!(p.corr.is_nan());
        // Then a constant feature (column 1): var_x = 0 exactly, corr and
        // beta NaN rather than infinite, everything else finite.
        for i in 0..20 {
            let xi = (i as f64).sin();
            m.step(&[xi, 7.0], &[Some(2.0 * xi)], 1.0, 1.0);
        }
        let p = m.pair(0, 1);
        assert_eq!(p.var_x, 0.0);
        assert_eq!(p.cov, 0.0);
        assert!(p.corr.is_nan() && p.beta.is_nan());
        assert!(p.mean_y.is_finite() && p.var_y > 0.0);
        let p = m.pair(0, 0);
        assert!((p.corr - 1.0).abs() < 1e-12);
        assert!((p.beta - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kish_size_of_unit_weights_tends_to_one_plus_lam_over_one_minus_lam() {
        let mut m = Marginal::new(cfg(1, 1)).unwrap();
        let mut s = 3u64;
        for i in 0..5000 {
            let x = lcg(&mut s);
            m.step(
                &[x],
                &[Some(x + lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let lam = 0.5f64.powf(1.0 / 20.0);
        let p = m.pair(0, 0);
        assert!(close(p.n_eff, 1.0 / (1.0 - lam), 1e-9));
        assert!(
            close(p.n_kish, (1.0 + lam) / (1.0 - lam), 1e-9),
            "{}",
            p.n_kish
        );
        // Unequal weights lower it: a single heavy row dominates.
        let mut m2 = Marginal::new(cfg(1, 1)).unwrap();
        for i in 0..50 {
            let w = if i == 49 { 1000.0 } else { 1.0 };
            m2.step(
                &[i as f64],
                &[Some(i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                w,
            );
        }
        let p2 = m2.pair(0, 0);
        assert!(p2.n_kish < 1.1, "one row carries the weight: {}", p2.n_kish);
        assert!(p2.n_eff > 1000.0);
    }

    #[test]
    fn state_round_trips_and_continues_identically() {
        let mut m = Marginal::new(cfg(3, 2)).unwrap();
        let mut s = 11u64;
        let row = |s: &mut u64| {
            let x: Vec<f64> = (0..3).map(|_| lcg(s)).collect();
            let y = [Some(x[0] + lcg(s)), (lcg(s) > 0.0).then(|| x[1] - lcg(s))];
            (x, y)
        };
        for i in 0..50 {
            let (x, y) = row(&mut s);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let mut r = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(r, m);
        for _ in 0..50 {
            let (x, y) = row(&mut s);
            let a = m.step(&x, &y, 1.0, 1.5);
            let b = r.step(&x, &y, 1.0, 1.5);
            assert_eq!(a, b);
        }
        assert_eq!(r, m);
        let wrong = Marginal::restore(
            &EwCovModel::new(EwCovCfg {
                n_features: 1,
                decay: Decay::Halflife(1.0),
                stats: vec![],
                min_periods: 0.0,
                precision_prior: None,
                mahal_quantiles: vec![],
                pca: 0,
                pca_every: 1,
            })
            .unwrap()
            .state(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            wrong.contains("marginal") && wrong.contains("ew_cov"),
            "{wrong}"
        );
    }

    #[test]
    fn shape_accessors() {
        let m = Marginal::new(cfg(4, 3)).unwrap();
        assert_eq!(m.n_features(), 4);
        assert_eq!(m.n_targets(), 3);
        assert_eq!(m.n_outputs(), 0);
        assert_eq!(m.n_eff(), 0.0);
        assert_eq!(m.state().model.kind(), "marginal");
        let p = m.predict(&[0.0; 4], 1.0);
        assert!(p.pred.is_empty() && p.n_eff == 0.0 && p.extra.is_none());
    }
}
