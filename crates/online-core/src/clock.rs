//! Clock/decay semantics shared by every entry point (docs/PLAN.md §3).
//!
//! Rows arrive with a raw clock value (or none, meaning row count) and an optional
//! session id. [`ClockState::advance`] turns those into the capped, gap-adjusted
//! `d_clock` a model consumes, folding the deltas of skipped (feature-null) rows
//! into the next accepted row so that decay still covers skipped time.

use serde::{Deserialize, Serialize};

/// Per-row decay: `halflife` in clock units, or a fixed per-unit factor `lam`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Decay {
    /// `factor = 0.5^(d_clock / halflife)`
    Halflife(f64),
    /// `factor = lam^d_clock` (with a row-count clock, `lam` per row).
    Lam(f64),
}

impl Decay {
    pub fn factor(&self, d_clock: f64) -> f64 {
        match *self {
            Decay::Halflife(h) => {
                if h.is_infinite() {
                    1.0
                } else {
                    0.5f64.powf(d_clock / h)
                }
            }
            Decay::Lam(l) => l.powf(d_clock),
        }
    }
}

/// What to do when the raw clock delta is negative (docs/PLAN.md §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnClockReset {
    /// Treat the delta as `max_dclock` (default).
    #[default]
    Max,
    /// Treat the delta as zero.
    Zero,
    /// Reset the model state.
    ResetState,
}

/// Delta override applied on a session change.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", untagged)]
pub enum SessionGap {
    /// Use this delta (in clock units) instead of the raw one.
    Gap(f64),
    /// Reset the model state.
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockCfg {
    /// Ceiling on the clock delta. Required when a clock column is used;
    /// `f64::INFINITY` is valid for row-count clocks.
    pub max_dclock: f64,
    pub on_clock_reset: OnClockReset,
    pub session_gap: Option<SessionGap>,
}

impl Default for ClockCfg {
    fn default() -> Self {
        Self {
            max_dclock: f64::INFINITY,
            on_clock_reset: OnClockReset::default(),
            session_gap: None,
        }
    }
}

/// Result of advancing the clock by one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockAdvance {
    /// Capped, gap-adjusted delta including any pending skipped-row time.
    /// Only meaningful when `accepted`.
    pub d_clock: f64,
    /// The caller must reset the model state before using this row.
    pub reset: bool,
    /// Whether the row was accepted (mirrors the `accept` argument).
    pub accepted: bool,
}

/// Per-stream clock state. Serialized as part of a stream's saved state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClockState {
    prev_clock: Option<f64>,
    prev_session: Option<u64>,
    /// Deltas of skipped rows, folded into the next accepted row.
    pending: f64,
    /// Whether any row has been seen (drives the row-count clock's first delta).
    started: bool,
}

impl ClockState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance by one row. `clock = None` means a row-count clock (delta 1).
    /// `accept = false` marks a skipped (feature-null) row: its delta is folded
    /// into `pending` instead of being returned.
    pub fn advance(
        &mut self,
        cfg: &ClockCfg,
        clock: Option<f64>,
        session: Option<u64>,
        accept: bool,
    ) -> ClockAdvance {
        let raw = match (clock, self.prev_clock) {
            (Some(c), Some(p)) => Some(c - p),
            (Some(_), None) => None, // first row of the stream
            (None, _) => {
                if self.started {
                    Some(1.0)
                } else {
                    None
                }
            }
        };
        let session_changed = match (session, self.prev_session) {
            (Some(s), Some(p)) => s != p,
            _ => false,
        };

        let mut reset = false;
        let mut d = match raw {
            None => 0.0,
            Some(raw) => {
                if session_changed {
                    match cfg.session_gap {
                        Some(SessionGap::Reset) => {
                            reset = true;
                            0.0
                        }
                        Some(SessionGap::Gap(g)) => g.clamp(0.0, cfg.max_dclock),
                        None => raw.clamp(0.0, cfg.max_dclock),
                    }
                } else if raw < 0.0 {
                    match cfg.on_clock_reset {
                        OnClockReset::Max => cfg.max_dclock,
                        OnClockReset::Zero => 0.0,
                        OnClockReset::ResetState => {
                            reset = true;
                            0.0
                        }
                    }
                } else {
                    raw.min(cfg.max_dclock)
                }
            }
        };

        if reset {
            self.pending = 0.0;
            d = 0.0;
        }
        self.prev_clock = clock;
        self.prev_session = session;
        self.started = true;

        if accept {
            let total = self.pending + d;
            self.pending = 0.0;
            ClockAdvance {
                d_clock: total,
                reset,
                accepted: true,
            }
        } else {
            self.pending += d;
            ClockAdvance {
                d_clock: 0.0,
                reset,
                accepted: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max: f64) -> ClockCfg {
        ClockCfg {
            max_dclock: max,
            ..Default::default()
        }
    }

    #[test]
    fn row_count_clock() {
        let mut c = ClockState::new();
        let cfg = ClockCfg::default();
        assert_eq!(c.advance(&cfg, None, None, true).d_clock, 0.0);
        assert_eq!(c.advance(&cfg, None, None, true).d_clock, 1.0);
        assert_eq!(c.advance(&cfg, None, None, true).d_clock, 1.0);
    }

    #[test]
    fn caps_and_negative_deltas() {
        // Mirrors test_compute_dclock_semantics in tests/reference.py.
        let t = [0.0, 10.0, 5.0, 6.0, 200.0];
        let mut c = ClockState::new();
        let cfg = cfg(50.0);
        let got: Vec<f64> = t
            .iter()
            .map(|&ti| c.advance(&cfg, Some(ti), None, true).d_clock)
            .collect();
        assert_eq!(got, vec![0.0, 10.0, 50.0, 1.0, 50.0]);

        let mut c = ClockState::new();
        let zero = ClockCfg {
            on_clock_reset: OnClockReset::Zero,
            ..cfg
        };
        c.advance(&zero, Some(0.0), None, true);
        c.advance(&zero, Some(10.0), None, true);
        assert_eq!(c.advance(&zero, Some(5.0), None, true).d_clock, 0.0);

        let mut c = ClockState::new();
        let rst = ClockCfg {
            on_clock_reset: OnClockReset::ResetState,
            ..cfg
        };
        c.advance(&rst, Some(0.0), None, true);
        c.advance(&rst, Some(10.0), None, true);
        let a = c.advance(&rst, Some(5.0), None, true);
        assert!(a.reset && a.d_clock == 0.0);
    }

    #[test]
    fn session_gap_overrides_delta() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            max_dclock: 50.0,
            session_gap: Some(SessionGap::Gap(7.5)),
            ..Default::default()
        };
        c.advance(&cfg, Some(0.0), Some(0), true);
        c.advance(&cfg, Some(10.0), Some(0), true);
        // negative raw delta AND session change: session gap wins
        let a = c.advance(&cfg, Some(5.0), Some(1), true);
        assert_eq!(a.d_clock, 7.5);

        let mut c = ClockState::new();
        let cfg = ClockCfg {
            max_dclock: 50.0,
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        c.advance(&cfg, Some(0.0), Some(0), true);
        let a = c.advance(&cfg, Some(10.0), Some(1), true);
        assert!(a.reset);
    }

    #[test]
    fn skipped_rows_fold_into_pending() {
        let mut c = ClockState::new();
        let cfg = cfg(f64::INFINITY);
        c.advance(&cfg, Some(0.0), None, true);
        let s = c.advance(&cfg, Some(3.0), None, false);
        assert!(!s.accepted);
        let a = c.advance(&cfg, Some(5.0), None, true);
        assert_eq!(a.d_clock, 5.0); // 3 (pending) + 2
    }

    #[test]
    fn decay_factors() {
        assert!((Decay::Halflife(10.0).factor(10.0) - 0.5).abs() < 1e-15);
        assert_eq!(Decay::Halflife(f64::INFINITY).factor(123.0), 1.0);
        assert!((Decay::Lam(0.9).factor(2.0) - 0.81).abs() < 1e-15);
        assert_eq!(Decay::Halflife(10.0).factor(0.0), 1.0);
    }
}
