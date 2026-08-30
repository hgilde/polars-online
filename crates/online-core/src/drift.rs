//! Page-Hinkley drift detection (docs/ENHANCEMENTS.md E20).
//!
//! Decay and drift detection answer different questions. A halflife forgets
//! *smoothly and always*, which is right when the world moves gradually; it is
//! slow when the world breaks. A drift detector watches for a break and says so,
//! which lets a caller react at once — flag the row, or reset the state.
//!
//! Page-Hinkley on a stream of non-negative error values `e_t`:
//!
//! ```text
//! mean_t = running mean of e
//! m_t    = m_{t-1} + (e_t − mean_t − delta)     cumulative signed excess
//! M_t    = min(M_{t-1}, m_t)                    the running low-water mark
//! detect when  m_t − M_t > threshold
//! ```
//!
//! `delta` is the size of change to tolerate before accumulating, so ordinary
//! noise does not drift the statistic upward; `threshold` is how much
//! accumulated excess counts as a break. This detects error going *up*, which
//! is the direction that matters for a model that has stopped fitting.

use serde::{Deserialize, Serialize};

/// Page-Hinkley state for one monitored signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageHinkley {
    /// Tolerated magnitude of change before excess accumulates.
    pub delta: f64,
    /// Accumulated excess that counts as drift.
    pub threshold: f64,
    n: f64,
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

    /// Feed one error value. Returns true when drift is detected, and clears
    /// the state so detection restarts from the new regime.
    pub fn update(&mut self, e: f64) -> bool {
        if !e.is_finite() {
            return false;
        }
        self.n += 1.0;
        self.mean += (e - self.mean) / self.n;
        self.cum += e - self.mean - self.delta;
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
    fn quiet_stream_does_not_drift() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        let mut s = 1u64;
        for _ in 0..20000 {
            let e = (1.0 + 0.2 * lcg(&mut s)).abs();
            assert!(!ph.update(e), "false positive on a stationary stream");
        }
    }

    #[test]
    fn detects_a_step_change_in_error() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        let mut s = 2u64;
        for _ in 0..2000 {
            ph.update((1.0 + 0.2 * lcg(&mut s)).abs());
        }
        let mut detected_at = None;
        for i in 0..2000 {
            if ph.update((4.0 + 0.2 * lcg(&mut s)).abs()) {
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
            ph.update(1.0);
        }
        while !ph.update(10.0) {}
        assert_eq!(ph.n(), 0.0, "state should clear on detection");
        assert_eq!(ph.statistic(), 0.0);
    }

    #[test]
    fn a_bigger_threshold_is_slower() {
        let time_to_detect = |threshold: f64| {
            let mut ph = PageHinkley::new(0.01, threshold);
            for _ in 0..500 {
                ph.update(1.0);
            }
            (0..10000).position(|_| ph.update(3.0)).unwrap()
        };
        assert!(time_to_detect(50.0) > time_to_detect(5.0));
    }

    #[test]
    fn delta_absorbs_small_shifts() {
        // A shift smaller than delta must never accumulate into a detection.
        let mut ph = PageHinkley::new(1.0, 5.0);
        for _ in 0..500 {
            ph.update(1.0);
        }
        for _ in 0..20000 {
            assert!(!ph.update(1.5), "a shift below delta should be absorbed");
        }
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let mut ph = PageHinkley::new(0.01, 5.0);
        assert!(!ph.update(f64::NAN));
        assert!(!ph.update(f64::INFINITY));
        assert_eq!(ph.n(), 0.0);
    }
}
