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
    ///
    /// `inf` is a documented setting -- it means no decay -- so this is one
    /// of the floats a human-readable encoding has to be told about, or JSON
    /// writes it as `null` (`crate::humanfloat`). The msgpack the state file
    /// uses is untouched.
    Halflife(#[serde(with = "crate::humanfloat::f64_or_tag")] f64),
    /// `factor = lam^d_clock` (with a row-count clock, `lam` per row).
    Lam(#[serde(with = "crate::humanfloat::f64_or_tag")] f64),
}

impl Decay {
    pub fn factor(&self, d_clock: f64) -> f64 {
        match *self {
            Decay::Halflife(h) => {
                if h.is_infinite() {
                    1.0
                } else {
                    // Spelled `exp2(-x)` rather than `0.5.powf(x)`: LLVM
                    // rewrites the latter into the former at any
                    // optimisation level above zero (`pow(2^n, x)` ->
                    // `exp2(n x)`, SimplifyLibCalls), and the two libm
                    // calls differ in the last bit. Writing it out makes a
                    // debug build agree with a release build, and lets a
                    // reference in another language (`math.exp2` in
                    // `tests/reference_cluster.py`) reproduce the factor
                    // bit for bit.
                    (-(d_clock / h)).exp2()
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
    /// Refuse the row. A backwards clock is usually a data bug — a mis-sorted
    /// chunk, or rows from two streams interleaved — and the other policies
    /// absorb it silently, producing plausible but wrong output. This makes it
    /// loud (PLAN §5: "the bank asserts monotonicity ... and errors loudly").
    Error,
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
    /// Ceiling on the clock delta a model sees between two rows it learns
    /// from: a row's own delta, and the total a run of skipped rows carries
    /// into the next accepted one (review 2026-09-12, S3). Required when a
    /// clock column is used; `f64::INFINITY` is valid for row-count clocks.
    pub max_dclock: f64,
    pub on_clock_reset: OnClockReset,
    pub session_gap: Option<SessionGap>,
    /// Two backwards clock jumps within a session closer than this, in clock
    /// units, are out-of-order data, not two session boundaries: the second
    /// is refused as [`Disorder::TooSoon`] whatever `on_clock_reset` says. A
    /// single jump that then holds is a boundary and takes the policy. `0`
    /// disables. The spec defaults it to `max_dclock`, the largest gap that
    /// still counts as adjacency -- a "session" shorter than one such gap is
    /// not a session -- so the scale comes from a value already chosen.
    pub min_session_clock: f64,
    /// A backwards jump no larger than this multiple of the typical forward
    /// step (an EW mean of the forward deltas, [`TYPICAL_LAM`]) is jitter --
    /// two sources never merged, a transposed pair, a row one tick late --
    /// not a boundary, and is refused as [`Disorder::Jitter`] on its first
    /// occurrence. A real boundary jumps back by a session's span, many
    /// steps. `0` disables; the spec defaults it to `1.0`. Silent until a
    /// forward step has been seen.
    pub backwards_jitter_ratio: f64,
}

impl Default for ClockCfg {
    /// Both disorder checks **off** here: this is the bare core default. The
    /// spec layer is where "on by default" lives (`min_session_clock =
    /// max_dclock`, `backwards_jitter_ratio = 1.0`), so a direct core caller
    /// opts in explicitly and the core's own tests keep their literal meaning.
    fn default() -> Self {
        Self {
            max_dclock: f64::INFINITY,
            on_clock_reset: OnClockReset::default(),
            session_gap: None,
            min_session_clock: 0.0,
            backwards_jitter_ratio: 0.0,
        }
    }
}

/// Per-row decay of the typical-forward-step estimate the jitter rule reads:
/// a row-halflife of about 70 rows. A literal rather than `2^(-1/h)`, so the
/// value persisted in the clock state carries no libm result and is the same
/// bits on every platform (the portability rule).
pub const TYPICAL_LAM: f64 = 0.99;

/// Why a backwards clock jump was refused as obviously out-of-order data,
/// carried on [`ClockAdvance::disorder`] for the caller's message. Each names
/// the numbers that decided it and the setting that disables the rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Disorder {
    /// The step back was no larger than `ratio` typical forward steps.
    Jitter { back: f64, typical: f64, ratio: f64 },
    /// The previous backwards jump was `span` clock units ago, under
    /// `min_session_clock`.
    TooSoon {
        back: f64,
        span: f64,
        min_session_clock: f64,
    },
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
    /// Set when the raw delta was negative and the row is refused: under
    /// `on_clock_reset` = [`OnClockReset::Error`], or -- whatever the policy
    /// -- when the jump is obviously out-of-order data (`disorder` says
    /// which rule). The caller must turn this into an error naming the row.
    /// Carries the offending raw delta.
    pub backwards: Option<f64>,
    /// With `backwards`: the disorder rule that refused the jump, `None` when
    /// it was the `error` policy alone. Carries the numbers for the message.
    pub disorder: Option<Disorder>,
    /// The session id differs from the previous row's. Reported separately
    /// from `reset` so a caller can do something gentler than starting over —
    /// see `session_shrink` (ENHANCEMENTS E6).
    pub session_changed: bool,
    /// The raw delta asked for more clock than `max_dclock` allows, so the
    /// delta handed to the models is the ceiling rather than the truth.
    ///
    /// Decay copes with that by construction — it only ever forgets more —
    /// but anything **lagged by rows** does not: the row `ℓ` back is no
    /// longer `ℓ` rows *ago* in any useful sense once a weekend has passed
    /// between them. A caller that keeps such a ring clears it here
    /// ([`crate::OnlineModel::clear_lags`], docs/PLAN.md task 47). Per row,
    /// not per accumulated gap: it is the one row's jump that breaks
    /// adjacency.
    pub capped: bool,
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
    /// The clock value the last accepted in-session backwards jump landed
    /// on: where the current inferred session began. `None` before any
    /// jump, and cleared by an explicit session change. The frequency rule
    /// measures the previous session's span from it (a clock value, so
    /// portable bits).
    #[serde(default)]
    session_start: Option<f64>,
    /// EW mean of the positive forward deltas, decayed by [`TYPICAL_LAM`]
    /// per row: the typical step the jitter rule reads. Zero weight before
    /// any forward step, when the rule is silent.
    #[serde(default)]
    typical: f64,
    #[serde(default)]
    typical_w: f64,
}

impl ClockState {
    pub fn new() -> Self {
        Self::default()
    }

    /// The last clock value seen, `None` before the first row or on a
    /// row-count clock. What a caller uses to tell a stale group from a live
    /// one.
    pub fn last_clock(&self) -> Option<f64> {
        self.prev_clock
    }

    /// The hash of the last row's session value, `None` before the first row
    /// or on a stream with no session column. A caller that has to know where
    /// a session boundary falls *before* the rows are processed -- the E54
    /// close-on-session split -- reads it here and walks the column itself,
    /// rather than advancing a copy of the clock and discarding everything
    /// else the advance decided.
    pub fn prev_session(&self) -> Option<u64> {
        self.prev_session
    }

    /// Advance by one row. `clock = None` means a row-count clock (delta 1).
    /// `accept = false` marks a skipped (feature-null) row: its delta is folded
    /// into `pending` instead of being returned.
    ///
    /// ```
    /// use online_core::{ClockCfg, ClockState, OnClockReset};
    ///
    /// let cfg = ClockCfg { max_dclock: 60.0, on_clock_reset: OnClockReset::Max, ..ClockCfg::default() };
    /// let mut clock = ClockState::new();
    /// // The first row of a stream has nothing to be a delta from.
    /// assert_eq!(clock.advance(&cfg, Some(1000.0), None, true).d_clock, 0.0);
    /// assert_eq!(clock.advance(&cfg, Some(1010.0), None, true).d_clock, 10.0);
    /// // A skipped row still moves the clock: its 5 units are carried into
    /// // the next accepted row's delta.
    /// assert!(!clock.advance(&cfg, Some(1015.0), None, false).accepted);
    /// assert_eq!(clock.advance(&cfg, Some(1020.0), None, true).d_clock, 10.0);
    /// // A gap is capped at `max_dclock`, so a weekend does not decay the
    /// // state to nothing.
    /// assert_eq!(clock.advance(&cfg, Some(1e6), None, true).d_clock, 60.0);
    /// ```
    ///
    /// `on_clock_reset` handles a *backwards* delta, and only within a
    /// session: on a session change the delta is `session_gap` if set,
    /// otherwise the raw delta clamped to `[0, max_dclock]` -- so a backwards
    /// raw delta whose row *also* changes session is clamped to 0 rather than
    /// routed through `on_clock_reset`. The bank never builds that
    /// combination (a spec with a `session` column requires `session_gap`
    /// unless `group_close = "session"`, which resegments the run at each
    /// boundary so the clock never compares across one; `spec.rs`), so it
    /// reaches only a direct `online-core` caller. Such a caller that wants a
    /// backwards clock refused across a session change must leave `session`
    /// unset for that check; here a session change with `session_gap = None`
    /// is deliberately an ordinary forward row (review 2026-09-18, B1).
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
        let mut backwards = None;
        let mut disorder = None;
        let mut capped = false;
        let mut d = match raw {
            None => 0.0,
            Some(raw) => {
                if session_changed {
                    // An explicit boundary: the inferred-session memory the
                    // frequency rule keeps starts over here.
                    self.session_start = None;
                    match cfg.session_gap {
                        Some(SessionGap::Reset) => {
                            reset = true;
                            0.0
                        }
                        Some(SessionGap::Gap(g)) => {
                            capped = g > cfg.max_dclock;
                            g.clamp(0.0, cfg.max_dclock)
                        }
                        None => {
                            capped = raw > cfg.max_dclock;
                            raw.clamp(0.0, cfg.max_dclock)
                        }
                    }
                } else if raw < 0.0 {
                    // Obvious disorder is refused whatever the absorbing
                    // policy says: a reset or a capped delta on shuffled or
                    // interleaved rows is worse than stopping. Two rules,
                    // each off at 0. A step back no larger than the typical
                    // forward step is jitter, caught on its first occurrence;
                    // a second jump within `min_session_clock` of the
                    // previous one is a "session" too short to be one. A
                    // single jump that then holds is a boundary and takes the
                    // policy. The `error` policy refuses every backwards jump
                    // already, and keeps its own message: the rules add
                    // nothing there.
                    let back = -raw;
                    let refuses_all = matches!(cfg.on_clock_reset, OnClockReset::Error);
                    // `<=`, not `<`: on an integer grid a row one tick late
                    // steps back by exactly one typical step, and that is the
                    // commonest accident, not a boundary.
                    let jitter = !refuses_all
                        && cfg.backwards_jitter_ratio > 0.0
                        && self.typical_w > 0.0
                        && back <= cfg.backwards_jitter_ratio * self.typical;
                    let span = self.prev_clock.zip(self.session_start).map(|(p, s)| p - s);
                    let too_soon = !refuses_all
                        && cfg.min_session_clock > 0.0
                        && span.is_some_and(|s| s < cfg.min_session_clock);
                    if jitter {
                        disorder = Some(Disorder::Jitter {
                            back,
                            typical: self.typical,
                            ratio: cfg.backwards_jitter_ratio,
                        });
                        backwards = Some(raw);
                        0.0
                    } else if too_soon {
                        disorder = Some(Disorder::TooSoon {
                            back,
                            span: span.unwrap_or(0.0),
                            min_session_clock: cfg.min_session_clock,
                        });
                        backwards = Some(raw);
                        0.0
                    } else {
                        match cfg.on_clock_reset {
                            OnClockReset::Max => {
                                // The policy says "as far apart as they can
                                // be", which is the ceiling: adjacency is
                                // gone. The inferred session begins here.
                                self.session_start = clock;
                                capped = true;
                                cfg.max_dclock
                            }
                            OnClockReset::Zero => {
                                self.session_start = clock;
                                0.0
                            }
                            OnClockReset::ResetState => {
                                self.session_start = clock;
                                reset = true;
                                0.0
                            }
                            OnClockReset::Error => {
                                backwards = Some(raw);
                                0.0
                            }
                        }
                    }
                } else {
                    // A forward step feeds the typical-step estimate the
                    // jitter rule reads; a repeated clock is not a step.
                    if raw > 0.0 {
                        self.typical_w = TYPICAL_LAM * self.typical_w + 1.0;
                        self.typical += (raw - self.typical) / self.typical_w;
                    }
                    capped = raw > cfg.max_dclock;
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
            // The skipped rows' time is carried into this row's, and the
            // ceiling holds for the total: `max_dclock` is the most a model
            // sees between two rows it learns from. Ten skipped rows 100
            // apart under a cap of 60 handed the next one 660 (review
            // 2026-09-12, S3). A total over the cap is a capped gap.
            let total = self.pending + d;
            self.pending = 0.0;
            ClockAdvance {
                d_clock: total.min(cfg.max_dclock),
                reset,
                accepted: true,
                backwards,
                disorder,
                session_changed,
                capped: capped || total > cfg.max_dclock,
            }
        } else {
            self.pending += d;
            ClockAdvance {
                d_clock: 0.0,
                reset,
                accepted: false,
                backwards,
                disorder,
                session_changed,
                capped,
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

    /// Task 47: `capped` says "this jump was bigger than the model is
    /// allowed to see", which is the signal anything lagged by *rows* needs.
    #[test]
    fn capped_marks_the_rows_whose_gap_hit_the_ceiling() {
        let cfg = cfg(60.0);
        let mut c = ClockState::new();
        // The first row has no delta at all.
        assert!(!c.advance(&cfg, Some(0.0), None, true).capped);
        assert!(!c.advance(&cfg, Some(10.0), None, true).capped);
        // Exactly at the ceiling is not over it.
        assert!(!c.advance(&cfg, Some(70.0), None, true).capped);
        let over = c.advance(&cfg, Some(1e6), None, true);
        assert!(over.capped && over.d_clock == 60.0);
        // A skipped row's gap breaks adjacency too, so the flag is set
        // whether or not the row is accepted.
        assert!(c.advance(&cfg, Some(2e6), None, false).capped);
        // An infinite ceiling never caps.
        let mut c = ClockState::new();
        let none = ClockCfg::default();
        c.advance(&none, Some(0.0), None, true);
        assert!(!c.advance(&none, Some(1e300), None, true).capped);
        // A row-count clock cannot jump.
        let mut c = ClockState::new();
        c.advance(&cfg, None, None, true);
        assert!(!c.advance(&cfg, None, None, true).capped);
    }

    #[test]
    fn capped_follows_the_backwards_policy() {
        // `max` says "as far apart as they can be", which is the ceiling:
        // adjacency is gone, and the ring must go with it.
        for (policy, want) in [
            (OnClockReset::Max, true),
            (OnClockReset::Zero, false),
            (OnClockReset::ResetState, false),
            (OnClockReset::Error, false),
        ] {
            let cfg = ClockCfg {
                max_dclock: 60.0,
                on_clock_reset: policy,
                ..Default::default()
            };
            let mut c = ClockState::new();
            c.advance(&cfg, Some(100.0), None, true);
            assert_eq!(
                c.advance(&cfg, Some(10.0), None, true).capped,
                want,
                "{policy:?}"
            );
        }
    }

    #[test]
    fn a_session_gap_caps_only_when_it_is_over_the_ceiling() {
        let with_gap = |g: f64| ClockCfg {
            max_dclock: 60.0,
            on_clock_reset: OnClockReset::Max,
            session_gap: Some(SessionGap::Gap(g)),
            ..Default::default()
        };
        for (gap, want) in [(30.0, false), (600.0, true)] {
            let cfg = with_gap(gap);
            let mut c = ClockState::new();
            c.advance(&cfg, Some(0.0), Some(1), true);
            let adv = c.advance(&cfg, Some(1.0), Some(2), true);
            assert!(adv.session_changed);
            assert_eq!(adv.capped, want, "gap {gap}");
        }
        // A session reset rebuilds the model, so it caps nothing.
        let cfg = ClockCfg {
            max_dclock: 60.0,
            on_clock_reset: OnClockReset::Max,
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, Some(0.0), Some(1), true);
        let adv = c.advance(&cfg, Some(1e6), Some(2), true);
        assert!(adv.reset && !adv.capped);
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

    /// A repeated clock value is a *zero* delta, not a backwards one: it must
    /// not be routed through `on_clock_reset`. (Found by `cargo mutants`:
    /// nothing here distinguished `raw < 0.0` from `raw <= 0.0`, and under the
    /// default `Max` policy that is a whole `max_dclock` of spurious decay.)
    #[test]
    fn duplicate_clock_values_are_zero_deltas() {
        for policy in [
            OnClockReset::Max,
            OnClockReset::Zero,
            OnClockReset::ResetState,
            OnClockReset::Error,
        ] {
            let mut c = ClockState::new();
            let cfg = ClockCfg {
                max_dclock: 10.0,
                on_clock_reset: policy,
                ..Default::default()
            };
            assert_eq!(c.advance(&cfg, Some(5.0), None, true).d_clock, 0.0);
            let a = c.advance(&cfg, Some(5.0), None, true);
            assert_eq!(
                a.d_clock, 0.0,
                "{policy:?}: repeated clock must give delta 0"
            );
            assert!(
                !a.reset,
                "{policy:?}: a repeated clock must not reset state"
            );
            // and a genuinely backwards clock still is handled by the policy
            let b = c.advance(&cfg, Some(4.0), None, true);
            match policy {
                OnClockReset::Max => assert_eq!(b.d_clock, 10.0),
                OnClockReset::Zero => assert_eq!(b.d_clock, 0.0),
                OnClockReset::ResetState => assert!(b.reset),
                OnClockReset::Error => assert_eq!(b.backwards, Some(-1.0)),
            }
        }
    }

    #[test]
    fn error_policy_reports_a_backwards_clock() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            max_dclock: 50.0,
            on_clock_reset: OnClockReset::Error,
            ..Default::default()
        };
        assert!(c.advance(&cfg, Some(0.0), None, true).backwards.is_none());
        assert!(c.advance(&cfg, Some(10.0), None, true).backwards.is_none());
        // a repeated value is a zero delta, not backwards
        assert!(c.advance(&cfg, Some(10.0), None, true).backwards.is_none());
        let a = c.advance(&cfg, Some(4.0), None, true);
        assert_eq!(a.backwards, Some(-6.0));
        assert!(!a.reset, "the error policy must not silently reset");
    }

    #[test]
    fn session_change_is_reported_separately_from_reset() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            max_dclock: 50.0,
            session_gap: Some(SessionGap::Gap(5.0)),
            ..Default::default()
        };
        assert!(!c.advance(&cfg, Some(0.0), Some(1), true).session_changed);
        assert!(!c.advance(&cfg, Some(1.0), Some(1), true).session_changed);
        let a = c.advance(&cfg, Some(2.0), Some(2), true);
        assert!(a.session_changed, "a new session id should be reported");
        assert!(!a.reset, "a gap is not a reset");
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

    /// The two disorder rules (design note of 2026-09-19): a step back
    /// smaller than the typical forward step is jitter, refused on its first
    /// occurrence; a single jump that then holds is a boundary and takes the
    /// policy.
    #[test]
    fn a_step_back_smaller_than_the_typical_step_is_jitter() {
        let cfg = ClockCfg {
            max_dclock: 1e9,
            backwards_jitter_ratio: 1.0,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0, 30.0, 40.0] {
            assert!(c.advance(&cfg, Some(t), None, true).backwards.is_none());
        }
        // Back by 3 on a typical step of 10: jitter, refused under the
        // default `max` policy that would otherwise have absorbed it.
        let a = c.advance(&cfg, Some(37.0), None, true);
        assert_eq!(a.backwards, Some(-3.0));
        match a.disorder {
            Some(Disorder::Jitter {
                back,
                typical,
                ratio,
            }) => {
                assert_eq!(back, 3.0);
                assert!((typical - 10.0).abs() < 1e-12, "{typical}");
                assert_eq!(ratio, 1.0);
            }
            other => panic!("{other:?}"),
        }
        // Back by 50 -- five typical steps -- is a boundary: the policy
        // applies and nothing is refused.
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0, 30.0, 40.0] {
            c.advance(&cfg, Some(t), None, true);
        }
        let b = c.advance(&cfg, Some(-10.0), None, true);
        assert!(b.backwards.is_none() && b.disorder.is_none());
        assert!(b.capped, "the `max` policy took the boundary");
    }

    /// A second backwards jump within `min_session_clock` of the previous
    /// one is a session too short to be one -- out-of-order data -- while
    /// the same jump after the session has run long enough is another
    /// boundary.
    #[test]
    fn a_second_backwards_jump_too_soon_is_out_of_order() {
        let cfg = ClockCfg {
            max_dclock: 1e9,
            min_session_clock: 60.0,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 100.0, 200.0, 300.0] {
            c.advance(&cfg, Some(t), None, true);
        }
        let first = c.advance(&cfg, Some(50.0), None, true);
        assert!(first.backwards.is_none() && first.capped);
        c.advance(&cfg, Some(60.0), None, true);
        c.advance(&cfg, Some(70.0), None, true);
        // 20 clock units into the new session, another jump back.
        let second = c.advance(&cfg, Some(10.0), None, true);
        assert_eq!(second.backwards, Some(-60.0));
        match second.disorder {
            Some(Disorder::TooSoon {
                back,
                span,
                min_session_clock,
            }) => {
                assert_eq!(back, 60.0);
                assert_eq!(span, 20.0);
                assert_eq!(min_session_clock, 60.0);
            }
            other => panic!("{other:?}"),
        }
        let mut c = ClockState::new();
        for t in [0.0, 100.0, 200.0, 300.0] {
            c.advance(&cfg, Some(t), None, true);
        }
        c.advance(&cfg, Some(50.0), None, true);
        for t in [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0] {
            c.advance(&cfg, Some(t), None, true);
        }
        let later = c.advance(&cfg, Some(10.0), None, true);
        assert!(later.backwards.is_none(), "{:?}", later.disorder);
    }

    /// Both rules at 0 -- the core default -- and every backwards jump takes
    /// the policy as before, jitter-sized or repeated.
    #[test]
    fn the_disorder_rules_off_leave_the_policy_alone() {
        let cfg = ClockCfg {
            max_dclock: 1e9,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0] {
            c.advance(&cfg, Some(t), None, true);
        }
        assert!(c.advance(&cfg, Some(19.0), None, true).backwards.is_none());
        assert!(c.advance(&cfg, Some(18.0), None, true).backwards.is_none());
    }

    /// A session column declares its boundaries, so a declared change starts
    /// the frequency rule's memory over: the first in-session jump of the new
    /// session is a boundary, not a second jump too soon after the old one.
    #[test]
    fn an_explicit_session_change_clears_the_inferred_session() {
        let cfg = ClockCfg {
            max_dclock: 1e9,
            min_session_clock: 60.0,
            session_gap: Some(SessionGap::Gap(1.0)),
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 100.0, 200.0] {
            c.advance(&cfg, Some(t), Some(1), true);
        }
        c.advance(&cfg, Some(50.0), Some(1), true);
        c.advance(&cfg, Some(60.0), Some(1), true);
        let changed = c.advance(&cfg, Some(0.0), Some(2), true);
        assert!(changed.session_changed && changed.backwards.is_none());
        c.advance(&cfg, Some(10.0), Some(2), true);
        let a = c.advance(&cfg, Some(5.0), Some(2), true);
        assert!(a.backwards.is_none(), "{:?}", a.disorder);
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

    /// A skipped row's delta is capped on its own, and so is the total the
    /// next accepted row is handed: ten skipped rows 100 apart under a cap of
    /// 60 handed it 660, eleven times the ceiling `max_dclock` promises a
    /// model sees (review 2026-09-12, S3). The fold is capped, and says so.
    #[test]
    fn a_skipped_run_hands_the_next_row_at_most_the_cap() {
        let mut c = ClockState::new();
        let cfg = cfg(60.0);
        c.advance(&cfg, Some(0.0), None, true);
        for i in 1..=10 {
            let s = c.advance(&cfg, Some(100.0 * f64::from(i)), None, false);
            assert!(!s.accepted);
        }
        let a = c.advance(&cfg, Some(1100.0), None, true);
        assert_eq!(a.d_clock, 60.0);
        assert!(a.capped, "a folded total past the cap is a capped gap");
    }

    #[test]
    fn decay_factors() {
        assert!((Decay::Halflife(10.0).factor(10.0) - 0.5).abs() < 1e-15);
        assert_eq!(Decay::Halflife(f64::INFINITY).factor(123.0), 1.0);
        assert!((Decay::Lam(0.9).factor(2.0) - 0.81).abs() < 1e-15);
        assert_eq!(Decay::Halflife(10.0).factor(0.0), 1.0);
    }
}
