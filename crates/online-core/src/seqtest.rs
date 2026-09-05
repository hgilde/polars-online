//! A sequential test on the sign of a stream (docs/ENHANCEMENTS.md E42).
//!
//! Per target, the row's value `y` is reduced to its sign `s ∈ {−1, 0, +1}`
//! and two betting processes are run on it, one per direction. With `n⁺`
//! and `n⁻` the counts of positive and negative rows *before* this one and
//! `n = n⁺ + n⁻`:
//!
//! ```text
//! λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))        λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
//! E⁺ ← E⁺ · (1 + λ⁺ s)                    E⁻ ← E⁻ · (1 − λ⁻ s)
//! log_e_pos = ln E⁺                       log_e_neg = ln E⁻
//! ```
//!
//! `(n⁺ − n⁻)/(n + 1)` is `2p̂ − 1` for the Krichevsky–Trofimov estimate
//! `p̂ = (n⁺ + ½)/(n + 1)` of `P(s = +1)`: the stake a gambler with a
//! `Beta(½, ½)` prior puts on the next sign. Clipped at zero, each side bets
//! only on the direction it tests. Both stakes are *predictable* -- computed
//! from the rows before -- and below 1, which is what makes each `E` a
//! non-negative supermartingale under its null: `E[1 + λ⁺s | past] =
//! 1 + λ⁺ E[s | past] ≤ 1` whenever `E[s | past] ≤ 0`, that is whenever,
//! given everything before it, a row is at least as likely to be negative
//! as positive. Ville's inequality then gives `P(sup_t E⁺_t ≥ 1/α) ≤ α`:
//! `log_e_pos ≥ ln(1/α)` on *any* row rejects "no more positives than
//! negatives" at level `α`, so the test can be read at every row and stopped
//! the moment it crosses (Waudby-Smith & Ramdas 2020). `(E⁺ + E⁻)/2` is the
//! two-sided e-value.
//!
//! The null is conditional on the past and about the sign alone, so it needs
//! no independence, no bound and no moments of `y` -- a squared loss with
//! infinite variance is as good a stream as any. What it does not test is the
//! *size* of the values: a stream that is up by a hair 60% of the time and
//! down by a mile the other 40% rejects.
//!
//! Ties (`s = 0`) bet nothing and count nothing. There is no decay: a process
//! that forgot its losses would not be an e-process. A weight of 0 skips the
//! row; any other weight is one trial, and counts itself toward `n_eff`.
//! With the unclipped stake the wealth has the closed form
//! `E_n = 2^n · B(n⁺ + ½, n⁻ + ½) / B(½, ½)`, which the tests use as the
//! oracle wherever the clip does not bind.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// Output slots per target: `log_e_pos, log_e_neg, n_pos, n_neg`.
pub const SLOTS: usize = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqTestCfg {
    pub n_targets: usize,
    /// Rows (weight) seen before the outputs are emitted.
    pub min_periods: f64,
}

impl SeqTestCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err("seqtest: n_targets must be >= 1".into());
        }
        if self.min_periods.is_nan() || self.min_periods < 0.0 {
            return Err("seqtest: min_periods must be >= 0".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeqTest {
    cfg: SeqTestCfg,
    n_pos: Vec<f64>,
    n_neg: Vec<f64>,
    log_e_pos: Vec<f64>,
    log_e_neg: Vec<f64>,
    w_sum: f64,
}

impl SeqTest {
    pub fn new(cfg: SeqTestCfg) -> Result<Self, String> {
        cfg.validate()?;
        let m = cfg.n_targets;
        Ok(Self {
            n_pos: vec![0.0; m],
            n_neg: vec![0.0; m],
            log_e_pos: vec![0.0; m],
            log_e_neg: vec![0.0; m],
            w_sum: 0.0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &SeqTestCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// `ln E⁺` per target: the evidence for "more positives than negatives".
    pub fn log_e_pos(&self) -> &[f64] {
        &self.log_e_pos
    }

    /// `ln E⁻` per target: the evidence for "more negatives than positives".
    pub fn log_e_neg(&self) -> &[f64] {
        &self.log_e_neg
    }

    /// Positive rows learned from, per target.
    pub fn n_pos(&self) -> &[f64] {
        &self.n_pos
    }

    /// Negative rows learned from, per target.
    pub fn n_neg(&self) -> &[f64] {
        &self.n_neg
    }

    /// The stakes the next row would be bet at, `(λ⁺, λ⁻)`, for target `j`.
    pub fn stakes(&self, j: usize) -> (f64, f64) {
        let (np, nn) = (self.n_pos[j], self.n_neg[j]);
        let n1 = np + nn + 1.0;
        (((np - nn) / n1).max(0.0), ((nn - np) / n1).max(0.0))
    }

    fn outputs(&self) -> Vec<f64> {
        let m = self.cfg.n_targets;
        if self.w_sum < self.cfg.min_periods {
            return vec![f64::NAN; SLOTS * m];
        }
        let mut pred = Vec::with_capacity(SLOTS * m);
        for j in 0..m {
            pred.extend_from_slice(&[
                self.log_e_pos[j],
                self.log_e_neg[j],
                self.n_pos[j],
                self.n_neg[j],
            ]);
        }
        pred
    }
}

impl OnlineModel for SeqTest {
    fn step(&mut self, _x: &[f64], y: &[Option<f64>], _d_clock: f64, weight: f64) -> Step {
        let out = self.predict(_x, _d_clock);
        if weight > 0.0 {
            for (j, yj) in y.iter().enumerate().take(self.cfg.n_targets) {
                let Some(yj) = *yj else { continue };
                // Callers hand in finite values; a NaN would compare false
                // both ways and fall through as a tie, which is the right
                // reading of "no sign" anyway.
                let s = if yj > 0.0 {
                    1.0
                } else if yj < 0.0 {
                    -1.0
                } else {
                    continue;
                };
                let (lam_pos, lam_neg) = self.stakes(j);
                // `ln_1p` for the tiny stakes early on; `1 − λ ≥ 1/(n + 1)`,
                // so a losing bet never takes the wealth to zero.
                self.log_e_pos[j] += (lam_pos * s).ln_1p();
                self.log_e_neg[j] += (-lam_neg * s).ln_1p();
                if s > 0.0 {
                    self.n_pos[j] += 1.0;
                } else {
                    self.n_neg[j] += 1.0;
                }
            }
            self.w_sum += weight;
        }
        out
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        Step {
            pred: self.outputs(),
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::SeqTest(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::SeqTest(m) => Ok((**m).clone()),
            other => Err(StateError::WrongModel {
                expected: "seqtest",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// A test reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        SLOTS * self.cfg.n_targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(m: usize) -> SeqTestCfg {
        SeqTestCfg {
            n_targets: m,
            min_periods: 0.0,
        }
    }

    fn run(signs: &[f64]) -> SeqTest {
        let mut m = SeqTest::new(cfg(1)).unwrap();
        for (i, s) in signs.iter().enumerate() {
            m.step(&[], &[Some(*s)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m
    }

    /// `ln k!`.
    fn ln_fact(k: usize) -> f64 {
        (1..=k).map(|i| (i as f64).ln()).sum()
    }

    /// `ln Γ(k + ½) = ln (2k)! − k ln 4 − ln k! + ½ ln π`.
    fn ln_gamma_half(k: usize) -> f64 {
        ln_fact(2 * k) - (k as f64) * 4f64.ln() - ln_fact(k) + 0.5 * std::f64::consts::PI.ln()
    }

    /// `ln(2^n · B(n⁺ + ½, n⁻ + ½) / B(½, ½))`, the Krichevsky–Trofimov
    /// mixture wealth: `B(½, ½) = π`.
    fn kt_closed_form(n_pos: usize, n_neg: usize) -> f64 {
        let n = n_pos + n_neg;
        (n as f64) * 2f64.ln() + ln_gamma_half(n_pos) + ln_gamma_half(n_neg)
            - ln_fact(n)
            - std::f64::consts::PI.ln()
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn cfg_validation() {
        assert!(SeqTest::new(cfg(0)).is_err());
        for mp in [-1.0, f64::NAN] {
            let bad = SeqTestCfg {
                n_targets: 1,
                min_periods: mp,
            };
            assert_eq!(
                SeqTest::new(bad).unwrap_err(),
                "seqtest: min_periods must be >= 0"
            );
        }
        assert!(SeqTest::new(cfg(3)).is_ok());
    }

    #[test]
    fn outputs_are_read_before_the_row() {
        let mut m = SeqTest::new(cfg(1)).unwrap();
        let s0 = m.step(&[], &[Some(2.5)], 0.0, 1.0);
        assert_eq!(s0.pred, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(s0.n_eff, 0.0);
        // The first row is bet at stake 0 -- nothing is known -- so the
        // wealth is still 1 on the second row; only the count moved.
        let s1 = m.step(&[], &[Some(-1.0)], 1.0, 1.0);
        assert_eq!(s1.pred, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(s1.n_eff, 1.0);
        // Third row: the lead was +1 before row 2 and is 0 now, so row 2 was
        // bet on at λ⁺ = 1/2 and lost: E⁺ = 1/2.
        let s2 = m.step(&[], &[Some(1.0)], 1.0, 1.0);
        assert!((s2.pred[0] - 0.5f64.ln()).abs() < 1e-15);
        assert_eq!(s2.pred[1], 0.0, "λ⁻ was clipped at 0");
        assert_eq!(&s2.pred[2..], &[1.0, 1.0]);
        assert_eq!(s2.n_eff, 2.0);
    }

    #[test]
    fn matches_the_kt_closed_form_where_the_clip_does_not_bind() {
        // Alternating from +: the lead before each row is 0 or 1, never
        // negative, so λ⁺ is the unclipped KT stake throughout and E⁺ is the
        // KT mixture; λ⁻ is clipped to 0 on every row.
        for n in [1usize, 2, 3, 10, 51, 400] {
            let signs: Vec<f64> = (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let m = run(&signs);
            let (np, nn) = (n.div_ceil(2), n / 2);
            let want = kt_closed_form(np, nn);
            assert!(
                (m.log_e_pos()[0] - want).abs() < 1e-12 * (1.0 + want.abs()),
                "n = {n}: got {} want {want}",
                m.log_e_pos()[0]
            );
            assert_eq!(m.log_e_neg()[0], 0.0, "n = {n}");
            assert_eq!((m.n_pos()[0], m.n_neg()[0]), (np as f64, nn as f64));
        }
        // A run of positives then as many negatives: the lead climbs and
        // comes back to 0, never below.
        let mut signs = vec![1.0; 30];
        signs.extend(vec![-1.0; 30]);
        let m = run(&signs);
        let want = kt_closed_form(30, 30);
        assert!((m.log_e_pos()[0] - want).abs() < 1e-12 * (1.0 + want.abs()));
        assert_eq!(m.log_e_neg()[0], 0.0);
        // The mirror image runs the other process.
        let mirrored: Vec<f64> = signs.iter().map(|s| -s).collect();
        let m = run(&mirrored);
        assert!((m.log_e_neg()[0] - want).abs() < 1e-12 * (1.0 + want.abs()));
        assert_eq!(m.log_e_pos()[0], 0.0);
    }

    #[test]
    fn the_evidence_grows_with_the_imbalance() {
        // 60 of 100 positive beats 55 of 100, which beats 50 of 100 (= 0).
        let e = |np: usize| {
            let mut signs = vec![1.0; np];
            signs.extend(vec![-1.0; 100 - np]);
            run(&signs).log_e_pos()[0]
        };
        assert!(e(60) > e(55) && e(55) > e(50));
        assert!((e(50) - kt_closed_form(50, 50)).abs() < 1e-10);
        assert!(e(50) < 0.0, "a balanced stream loses a little: {}", e(50));
    }

    #[test]
    fn a_losing_streak_never_bankrupts() {
        // λ⁺ ≤ n/(n + 1) < 1: after 200 positives the stake is about 0.995,
        // and 200 negatives in a row then cost nearly all of the wealth --
        // from e^135 down to the balanced mixture's e^-3 -- but never all
        // of it. The lead stays >= 0 throughout, so the clip never binds and
        // the end is the closed form.
        let mut m = run(&vec![1.0; 200]);
        let peak = m.log_e_pos()[0];
        assert!((peak - kt_closed_form(200, 0)).abs() < 1e-10 * peak);
        assert!(peak > 130.0);
        for _ in 0..200 {
            m.step(&[], &[Some(-1.0)], 1.0, 1.0);
            assert!(m.log_e_pos()[0].is_finite());
        }
        let end = m.log_e_pos()[0];
        assert!((end - kt_closed_form(200, 200)).abs() < 1e-10 * (1.0 + end.abs()));
        assert!(end < -3.0 && end > -3.5, "{end}");
        assert_eq!(m.log_e_neg()[0], 0.0, "the lead was never negative");
    }

    #[test]
    fn ties_nulls_and_zero_weights_bet_nothing() {
        let m0 = run(&[1.0, 1.0, -1.0, 1.0]);
        let mut m = m0.clone();
        let before = m.state();
        // A tie: counted nowhere, bet on nowhere; the row still counts as
        // seen.
        let s = m.step(&[], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(s.pred, m0.predict(&[], 1.0).pred);
        assert_eq!(m.n_pos(), m0.n_pos());
        assert_eq!(m.log_e_pos(), m0.log_e_pos());
        assert_eq!(m.n_eff(), m0.n_eff() + 1.0);
        // A null target: the same.
        let mut m = m0.clone();
        m.step(&[], &[None], 1.0, 1.0);
        assert_eq!(m.log_e_pos(), m0.log_e_pos());
        assert_eq!(m.n_neg(), m0.n_neg());
        assert_eq!(m.n_eff(), m0.n_eff() + 1.0);
        // A zero weight: nothing at all, not even the count of rows.
        let mut m = m0.clone();
        let s = m.step(&[], &[Some(1e9)], 1.0, 0.0);
        assert_eq!(m.state(), before);
        assert_eq!(s.n_eff, m0.n_eff());
        // A zero weight as the very first row.
        let mut m = SeqTest::new(cfg(1)).unwrap();
        let s = m.step(&[], &[Some(1.0)], 0.0, 0.0);
        assert_eq!(s.pred, vec![0.0; 4]);
        assert_eq!(m.n_eff(), 0.0);
        let s = m.step(&[], &[Some(1.0)], 1.0, 1.0);
        assert_eq!(s.pred, vec![0.0; 4]);
        assert!(m.log_e_pos()[0] == 0.0 && m.n_pos()[0] == 1.0);
    }

    #[test]
    fn a_weight_is_one_trial_that_counts_itself() {
        let a = run(&[1.0, 1.0, -1.0]);
        let mut b = SeqTest::new(cfg(1)).unwrap();
        for (i, (s, w)) in [(1.0, 0.5), (1.0, 3.0), (-1.0, 0.1)].iter().enumerate() {
            b.step(&[], &[Some(*s)], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        assert_eq!(a.log_e_pos(), b.log_e_pos());
        assert_eq!(a.log_e_neg(), b.log_e_neg());
        assert_eq!(a.n_pos(), b.n_pos());
        assert_eq!(a.n_eff(), 3.0);
        assert!((b.n_eff() - 3.6).abs() < 1e-15);
    }

    #[test]
    fn only_the_sign_matters() {
        let a = run(&[1.0, 2.0, -3.0, 0.5, -0.25]);
        let b = run(&[1e-300, 1e300, -1e-9, 7.0, -1e100]);
        assert_eq!(a, b);
    }

    #[test]
    fn targets_are_independent_tests() {
        let mut m = SeqTest::new(cfg(2)).unwrap();
        let mut a = SeqTest::new(cfg(1)).unwrap();
        let mut b = SeqTest::new(cfg(1)).unwrap();
        let mut s = 7u64;
        for i in 0..1000 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let ya = if lcg(&mut s) < 0.7 {
                Some(1.0)
            } else {
                Some(-1.0)
            };
            let yb = if lcg(&mut s) < 0.3 {
                None
            } else if lcg(&mut s) < 0.4 {
                Some(1.0)
            } else {
                Some(-1.0)
            };
            let joint = m.step(&[], &[ya, yb], d, 1.0);
            let sa = a.step(&[], &[ya], d, 1.0);
            let sb = b.step(&[], &[yb], d, 1.0);
            assert_eq!(&joint.pred[..4], &sa.pred[..]);
            assert_eq!(&joint.pred[4..], &sb.pred[..]);
            assert_eq!(joint.n_eff, sa.n_eff);
        }
        assert!(
            m.log_e_pos()[0] > 50.0,
            "70% positive is overwhelming by 1000"
        );
        assert!(m.log_e_neg()[1] > 3.0, "40% positive: the other side gains");
        assert!(
            m.log_e_pos()[1] < 0.0,
            "and this side lost what it bet early"
        );
    }

    #[test]
    fn min_periods_withholds_the_outputs_and_nothing_else() {
        let mut m = SeqTest::new(SeqTestCfg {
            n_targets: 1,
            min_periods: 3.0,
        })
        .unwrap();
        let mut twin = SeqTest::new(cfg(1)).unwrap();
        for i in 0..6 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let s = m.step(&[], &[Some(1.0)], d, 1.0);
            let t = twin.step(&[], &[Some(1.0)], d, 1.0);
            assert_eq!(s.n_eff, t.n_eff);
            if i < 3 {
                assert!(s.pred.iter().all(|p| p.is_nan()), "row {i}");
            } else {
                assert_eq!(s.pred, t.pred, "row {i}");
            }
        }
        assert_eq!(m.log_e_pos(), twin.log_e_pos());
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut m = SeqTest::new(cfg(2)).unwrap();
        let mut s = 3u64;
        for i in 0..100 {
            let y = [Some(lcg(&mut s) - 0.4), Some(lcg(&mut s) - 0.6)];
            let before = m.clone();
            let p = m.predict(&[], 1.0);
            assert_eq!(m, before, "predict moved the state at row {i}");
            let st = m.step(&[], &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
            assert_eq!(p, st, "row {i}");
        }
    }

    #[test]
    fn state_round_trips_and_continues_identically() {
        let mut m = run(&[1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let mut r = SeqTest::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(r, m);
        for s in [1.0, 1.0, -1.0, 1.0] {
            assert_eq!(
                r.step(&[], &[Some(s)], 1.0, 1.0),
                m.step(&[], &[Some(s)], 1.0, 1.0)
            );
        }
        assert_eq!(m.state().model.kind(), "seqtest");
        let holt = crate::Holt::new(crate::HoltCfg {
            n_targets: 1,
            level_halflife: 1.0,
            trend_halflife: 1.0,
            min_periods: 0.0,
        })
        .unwrap();
        assert!(matches!(
            SeqTest::restore(&holt.state()),
            Err(StateError::WrongModel {
                expected: "seqtest",
                found: "holt"
            })
        ));
    }

    /// Streams of `rows` fair or biased coin flips; returns, per stream, the
    /// largest `E⁺` seen and the final `E⁺`.
    fn simulate(streams: usize, rows: usize, p: f64, seed: u64) -> Vec<(f64, f64)> {
        let mut s = seed;
        (0..streams)
            .map(|_| {
                let mut m = SeqTest::new(cfg(1)).unwrap();
                let mut sup = f64::NEG_INFINITY;
                for i in 0..rows {
                    let y = if lcg(&mut s) < p { 1.0 } else { -1.0 };
                    m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                    sup = sup.max(m.log_e_pos()[0]);
                }
                (sup.exp(), m.log_e_pos()[0].exp())
            })
            .collect()
    }

    #[test]
    fn type_i_error_is_controlled_under_a_fair_coin() {
        // Ville: P(sup_t E⁺_t ≥ 1/α) ≤ α under the null, at every α at once.
        // 4000 streams of 1000 rows, read at every row; the crossing rate at
        // 1/α = 20 is about 1%, well inside the bound. (The mean of the
        // final wealth is not checked: under a fair coin E⁺ is a martingale
        // with E[E⁺_n] = 1 exactly, but that mean is carried by streams so
        // rare that 4000 of them rarely hold one -- the median is far
        // below 1, as a supermartingale's is.)
        let sims = simulate(4000, 1000, 0.5, 20260905);
        let crossing = |level: f64| {
            sims.iter().filter(|(sup, _)| *sup >= level).count() as f64 / sims.len() as f64
        };
        for alpha in [0.5, 0.2, 0.1, 0.05, 0.01] {
            let rate = crossing(1.0 / alpha);
            assert!(
                rate <= alpha,
                "P(sup E ≥ {}) = {rate} > {alpha}",
                1.0 / alpha
            );
        }
        assert!(
            crossing(20.0) <= 0.03,
            "{}: far above what KT gives",
            crossing(20.0)
        );
        // And it does bet: a process that never staked anything would pass
        // the bounds above with a crossing rate of 0 everywhere.
        assert!(crossing(2.0) >= 0.1, "{}", crossing(2.0));
    }

    #[test]
    fn power_grows_with_the_bias_and_the_rows() {
        let power = |rows: usize, p: f64| {
            let sims = simulate(1000, rows, p, 7);
            sims.iter().filter(|(sup, _)| *sup >= 20.0).count() as f64 / 1000.0
        };
        let p60 = power(1000, 0.6);
        let p55 = power(1000, 0.55);
        let p55_short = power(200, 0.55);
        assert!(p60 >= 0.99, "60% positive over 1000 rows: power {p60}");
        assert!(
            p55 > p55_short,
            "more rows, more power: {p55} vs {p55_short}"
        );
        assert!(
            (0.3..=0.9).contains(&p55),
            "55% over 1000 rows: power {p55}"
        );
    }
}
