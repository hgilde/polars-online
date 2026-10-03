//! Page-Hinkley drift detection (docs/ENHANCEMENTS.md E20), on the model's
//! clock (docs/PLAN.md task 146).
//!
//! Decay and drift detection answer different questions. A half-life forgets
//! *smoothly and always*, which is right when the world moves gradually; it is
//! slow when the world breaks. A drift detector watches for a break and says so,
//! which lets a caller react at once — flag the row, or reset the state.
//!
//! Page-Hinkley on a stream of non-negative error values `e_t`, each arriving
//! `d_t` clock units after the last with decay `λ_t` over that step:
//!
//! ```text
//! W_t    = λ_t W_{t-1} + 1                       the mean's weight
//! mean_t = mean_{t-1} + (e_t − mean_{t-1}) / W_t the error's EW mean
//! m_t    = m_{t-1} + d_t (e_t − mean_t − delta)  cumulative signed excess
//! M_t    = min(M_{t-1}, m_t)                     the running low-water mark
//! detect when  m_t − M_t > threshold
//! ```
//!
//! `delta` is the size of change to tolerate before accumulating, so ordinary
//! noise does not drift the statistic upward; `threshold` is how much
//! accumulated excess counts as a break, in error units times clock units.
//! The excess is integrated over the clock, so the same burst counts the
//! same whether the rows during it are one or a thousand to a clock unit;
//! and the mean decays at the model's half-life, so it forgets on the clock
//! as the model does. At rows one unit apart with no decay this is the
//! classic test, a running mean and one excess a row. A step with no error
//! to score ages the mean and adds no excess. This detects error going
//! *up*, which is the direction that matters for a model that has stopped
//! fitting.

use serde::{Deserialize, Serialize};

/// Page-Hinkley state for one monitored signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageHinkley {
    /// Tolerated magnitude of change before excess accumulates.
    pub delta: f64,
    /// Accumulated excess that counts as drift, in error units times clock
    /// units.
    pub threshold: f64,
    n: f64,
    /// The weight behind `mean`, decayed on the clock.
    w: f64,
    mean: f64,
    cum: f64,
    min_cum: f64,
}

impl PageHinkley {
    pub fn new(delta: f64, threshold: f64) -> Self {
        Self {
            delta,
            threshold,
            n: 0.0,
            w: 0.0,
            mean: 0.0,
            cum: 0.0,
            min_cum: 0.0,
        }
    }

    /// Observations seen since the last reset.
    pub fn n(&self) -> f64 {
        self.n
    }

    /// How far the accumulated excess is above its low-water mark.
    pub fn statistic(&self) -> f64 {
        self.cum - self.min_cum
    }

    /// A step of the clock with no error to score: the mean's weight decays
    /// by `lam`, and nothing accumulates.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
    }

    /// Feed one error value `e`, `d` clock units after the last step, whose
    /// decay is `lam`. Returns true when drift is detected, and clears the
    /// state so detection restarts from the new regime. A non-finite `e`
    /// ages the mean as [`PageHinkley::age`] does and is otherwise not seen.
    pub fn update(&mut self, e: f64, d: f64, lam: f64) -> bool {
        if !e.is_finite() {
            self.age(lam);
            return false;
        }
        self.n += 1.0;
        self.w = lam * self.w + 1.0;
        self.mean += (e - self.mean) / self.w;
        self.cum += d * (e - self.mean - self.delta);
        if self.cum < self.min_cum {
            self.min_cum = self.cum;
        }
        if self.statistic() > self.threshold {
            self.reset();
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.n = 0.0;
        self.w = 0.0;
        self.mean = 0.0;
        self.cum = 0.0;
        self.min_cum = 0.0;
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

    #[test]
    fn the_statistic_is_the_excess_above_its_low_water_mark() {
        // Page-Hinkley written out: a running mean, a cumulative sum of
        // (e - mean - delta), and the distance of that sum above the lowest
        // value it has reached. Each piece is checked against a longhand
        // recomputation, and `n` counts the observations since the last reset.
        let (delta, threshold) = (0.05, 1e9); // never fires: measure, don't trip
        let mut ph = PageHinkley::new(delta, threshold);
        let (mut mean, mut cum, mut min_cum, mut n) = (0.0, 0.0, 0.0f64, 0.0);
        let mut s = 7u64;
        for i in 0..300 {
            let e = (1.0 + 0.5 * lcg(&mut s)).abs();
            assert!(!ph.update(e, 1.0, 1.0));
            n += 1.0;
            mean += (e - mean) / n;
            cum += e - mean - delta;
            min_cum = min_cum.min(cum);
            assert!((ph.n() - n).abs() < 1e-12, "row {i}: n");
            assert!(
                (ph.statistic() - (cum - min_cum)).abs() < 1e-9,
                "row {i}: {} vs {}",
                ph.statistic(),
                cum - min_cum
            );
            assert!(ph.statistic() >= 0.0, "row {i}: never negative");
        }
        assert!(
            n > 0.0 && min_cum < 0.0,
            "the low-water mark should have moved"
        );
    }

    #[test]
    fn a_non_finite_error_is_ignored_entirely() {
        let mut ph = PageHinkley::new(0.05, 5.0);
        let mut s = 11u64;
        for _ in 0..50 {
            ph.update((1.0 + 0.2 * lcg(&mut s)).abs(), 1.0, 1.0);
        }
        let (n, stat) = (ph.n(), ph.statistic());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                !ph.update(bad, 1.0, 1.0),
                "a non-finite error must not signal drift"
            );
        }
        assert_eq!(ph.n(), n, "and must not count as an observation");
        assert_eq!(ph.statistic(), stat);
    }

    #[test]
    fn delta_sets_the_size_of_change_that_is_ignored() {
        // The allowance subtracted from every observation: a drift smaller
        // than delta must never accumulate, and a larger one must.
        let run = |delta: f64, jump: f64| {
            let mut ph = PageHinkley::new(delta, 5.0);
            let mut s = 13u64;
            for _ in 0..500 {
                ph.update(1.0 + 0.05 * lcg(&mut s), 1.0, 1.0);
            }
            (0..3000).any(|_| ph.update(1.0 + jump + 0.05 * lcg(&mut s), 1.0, 1.0))
        };
        assert!(!run(0.5, 0.1), "a jump well under delta must be absorbed");
        assert!(run(0.01, 0.1), "a jump well over delta must be caught");
    }

    #[test]
    fn quiet_stream_does_not_drift() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        let mut s = 1u64;
        for _ in 0..20000 {
            let e = (1.0 + 0.2 * lcg(&mut s)).abs();
            assert!(
                !ph.update(e, 1.0, 1.0),
                "false positive on a stationary stream"
            );
        }
    }

    #[test]
    fn detects_a_step_change_in_error() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        let mut s = 2u64;
        for _ in 0..2000 {
            ph.update((1.0 + 0.2 * lcg(&mut s)).abs(), 1.0, 1.0);
        }
        let mut detected_at = None;
        for i in 0..2000 {
            if ph.update((4.0 + 0.2 * lcg(&mut s)).abs(), 1.0, 1.0) {
                detected_at = Some(i);
                break;
            }
        }
        let at = detected_at.expect("no drift detected after a 4x error jump");
        assert!(at < 50, "took {at} rows to notice a 4x jump");
    }

    #[test]
    fn resets_after_detecting() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        for _ in 0..100 {
            ph.update(1.0, 1.0, 1.0);
        }
        while !ph.update(10.0, 1.0, 1.0) {}
        assert_eq!(ph.n(), 0.0, "state should clear on detection");
        assert_eq!(ph.statistic(), 0.0);
    }

    #[test]
    fn a_bigger_threshold_is_slower() {
        let time_to_detect = |threshold: f64| {
            let mut ph = PageHinkley::new(0.01, threshold);
            for _ in 0..500 {
                ph.update(1.0, 1.0, 1.0);
            }
            (0..10000).position(|_| ph.update(3.0, 1.0, 1.0)).unwrap()
        };
        assert!(time_to_detect(50.0) > time_to_detect(5.0));
    }

    #[test]
    fn delta_absorbs_small_shifts() {
        // A shift smaller than delta must never accumulate into a detection.
        let mut ph = PageHinkley::new(1.0, 5.0);
        for _ in 0..500 {
            ph.update(1.0, 1.0, 1.0);
        }
        for _ in 0..20000 {
            assert!(
                !ph.update(1.5, 1.0, 1.0),
                "a shift below delta should be absorbed"
            );
        }
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        assert!(!ph.update(f64::NAN, 1.0, 1.0));
        assert!(!ph.update(f64::INFINITY, 1.0, 1.0));
        assert_eq!(ph.n(), 0.0);
    }

    /// Task 146 written out: the mean an EW mean whose weight decays by each
    /// step's `λ`, and the excess each error's distance above it less
    /// `delta`, times the clock since the last; checked against a longhand
    /// recomputation on irregular steps.
    #[test]
    fn the_statistic_is_the_clock_weighted_excess_against_a_decayed_mean() {
        let (delta, threshold, h) = (0.05, 1e9, 20.0);
        let mut ph = PageHinkley::new(delta, threshold);
        let (mut w, mut mean, mut cum, mut min_cum) = (0.0, 0.0, 0.0, 0.0f64);
        let mut s = 17u64;
        for i in 0..500 {
            let d = 1.5 * (lcg(&mut s) + 1.0);
            let lam = (-(d / h)).exp2();
            let e = (1.0 + 0.5 * lcg(&mut s)).abs();
            assert!(!ph.update(e, d, lam));
            w = lam * w + 1.0;
            mean += (e - mean) / w;
            cum += d * (e - mean - delta);
            min_cum = min_cum.min(cum);
            assert!(
                (ph.statistic() - (cum - min_cum)).abs() < 1e-9,
                "row {i}: {} vs {}",
                ph.statistic(),
                cum - min_cum
            );
        }
    }

    /// The same 30-unit burst, rows one and four to a clock unit: found at
    /// the same clock, to within a unit. Counting rows found it four times
    /// sooner at four to a unit.
    #[test]
    fn a_burst_is_found_at_the_same_clock_at_any_row_density() {
        let found = |per_unit: f64| {
            let d = 1.0 / per_unit;
            let lam = (-(d / 50.0)).exp2();
            let mut ph = PageHinkley::new(0.5, 20.0);
            let mut t = 0.0;
            while t < 100.0 {
                t += d;
                assert!(!ph.update(1.0, d, lam), "nothing to find before the burst");
            }
            while t < 130.0 {
                t += d;
                if ph.update(3.0, d, lam) {
                    return t - 100.0;
                }
            }
            f64::INFINITY
        };
        let (one, four) = (found(1.0), found(4.0));
        assert!(one < 30.0, "the burst was not found at one row a unit");
        assert!(
            (one - four).abs() <= 1.0,
            "{one} against {four} clock units"
        );
    }

    /// A step with no error to score ages the mean, so the clock reaches it
    /// on every row: two steps' decay before an error is one step's decay
    /// of their product.
    #[test]
    fn a_step_without_an_error_ages_the_mean() {
        let (mut aged, mut once) = (PageHinkley::new(0.1, 1e9), PageHinkley::new(0.1, 1e9));
        let mut s = 19u64;
        for i in 0..400 {
            let e = (1.0 + lcg(&mut s)).abs();
            if i % 4 == 0 {
                aged.age(0.9);
                aged.update(e, 1.0, 0.8);
                once.update(e, 1.0, 0.9 * 0.8);
            } else {
                aged.update(e, 1.0, 0.95);
                once.update(e, 1.0, 0.95);
            }
        }
        assert!((aged.mean - once.mean).abs() < 1e-12);
        assert!((aged.statistic() - once.statistic()).abs() < 1e-9);
        let mut nan = once.clone();
        nan.update(f64::NAN, 1.0, 0.5);
        once.age(0.5);
        assert_eq!(nan, once, "a non-finite error is a step with no error");
    }
}
