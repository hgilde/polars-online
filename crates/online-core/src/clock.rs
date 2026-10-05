//! Clock/decay semantics shared by every entry point (docs/PLAN.md §3).
//!
//! Rows arrive with a raw clock value (or none, meaning row count) and an optional
//! session id. [`ClockState::advance`] turns those into the capped, gap-adjusted
//! `d_clock` a model consumes, folding the deltas of skipped (feature-null) rows
//! into the next accepted row so that decay still covers skipped time.

use serde::{Deserialize, Serialize};

/// Per-row decay: `half_life` in clock units, or a fixed per-unit factor `lam`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Decay {
    /// `factor = 0.5^(d_clock / half_life)`
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

/// What a clock that goes back within a session means (docs/PLAN.md §3 and
/// task 120). Two policies: `"max"`, which took `gap_cap` as the step,
/// and `"zero"`, which took none, were removed on 2026-09-28 -- a cap is not
/// a step, and both absorbed a data bug into plausible, wrong output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnClockReset {
    /// Refuse the row (the default). A backwards clock is usually a data
    /// bug -- a mis-sorted chunk, or rows from two streams interleaved --
    /// and absorbing it gives plausible but wrong output.
    #[default]
    Error,
    /// The clock resets on purpose: a step back by more than
    /// `min_backwards_jump` starts the model over, and one by no more than
    /// it is a late row, refused as [`Disorder`].
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
    /// Ceiling on the clock delta a model sees between two rows it learns
    /// from: a row's own delta, and the total a run of skipped rows carries
    /// into the next accepted one (review 2026-09-12, S3), and the gap past
    /// which two rows are no longer adjacent ([`ClockAdvance::capped`]). The
    /// spec requires it finite and positive with a clock column (task 120);
    /// `f64::INFINITY` is the row-count clock's, whose step is always 1.
    pub gap_cap: f64,
    pub on_clock_reset: OnClockReset,
    pub session_gap: Option<SessionGap>,
    /// Under [`OnClockReset::ResetState`], a step back by no more than this,
    /// in clock units, is a late row and is refused as [`Disorder`]; a
    /// larger one resets the model. `0` resets on every step back. Unread
    /// under [`OnClockReset::Error`], which refuses every step back; the spec
    /// requires it with `"reset_state"` and refuses it with `"error"`.
    pub min_backwards_jump: f64,
}

impl Default for ClockCfg {
    /// The bare core default: a row-count clock's infinite cap, and the
    /// default policy, which refuses a step back.
    fn default() -> Self {
        Self {
            gap_cap: f64::INFINITY,
            on_clock_reset: OnClockReset::default(),
            session_gap: None,
            min_backwards_jump: 0.0,
        }
    }
}

/// Why a backwards clock jump was refused as a late row under
/// `"reset_state"`, carried on [`ClockAdvance::disorder`] for the caller's
/// message: the step and the minimum it did not exceed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disorder {
    /// How far the clock stepped back, in clock units.
    pub back: f64,
    /// The `min_backwards_jump` in force.
    pub min_backwards_jump: f64,
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
    /// [`OnClockReset::Error`], or under [`OnClockReset::ResetState`] when
    /// the step back is no larger than `min_backwards_jump` (`disorder`
    /// carries the numbers). The caller must turn this into an error naming
    /// the row. Carries the offending raw delta.
    pub backwards: Option<f64>,
    /// With `backwards`: the step and the minimum it did not exceed, `None`
    /// when it was the `error` policy.
    pub disorder: Option<Disorder>,
    /// The session id differs from the previous row's. Reported separately
    /// from `reset` so a caller can do something gentler than starting over —
    /// see `session_shrink` (ENHANCEMENTS E6).
    pub session_changed: bool,
    /// The raw delta asked for more clock than `gap_cap` allows, so the
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
    /// The time that passed since the previous accepted row, uncapped: the
    /// clock column's forward steps, skipped rows' included, and at a
    /// session change that restarts the clock, the session's gap. What a
    /// delay counts (docs/PLAN.md task 153): `gap_cap` and `session_gap`
    /// say how much a model forgets across a break, not how long it lasted,
    /// so `d_clock` runs slower than the clock across a capped gap and
    /// faster across a session gap longer than its step. `0` on the first
    /// row and at a reset; only meaningful when `accepted`.
    pub elapsed: f64,
}

/// One row's clock value, in the form the source had it: a number, or a
/// temporal clock's nanoseconds since the Unix epoch. A stream keeps its
/// previous row's value in that form, and the delta between two `Ns` values
/// is taken in integers before it becomes seconds, so a nanosecond
/// timestamp's gaps are exact whatever the stream's age. Read as a double
/// of seconds from any origin, a clock resolves only 2^-52 of the time
/// since that origin: under a nanosecond for six weeks, four nanoseconds
/// after a year (docs/PLAN.md task 88).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClockValue {
    /// A numeric clock, in the column's own units.
    F64(f64),
    /// A temporal clock, in nanoseconds since the Unix epoch.
    Ns(i64),
}

impl ClockValue {
    /// The value as a number a caller can report: a numeric clock as it
    /// is, a temporal one as seconds since the Unix epoch, which a double
    /// resolves to about a quarter of a microsecond at today's dates. For
    /// reporting; the delta a model sees never goes through this.
    pub fn seconds(self) -> f64 {
        match self {
            Self::F64(v) => v,
            Self::Ns(ns) => seconds_of_ns(i128::from(ns)),
        }
    }

    /// `self` minus `prev`, in clock units: for two temporal values, the
    /// seconds between them, taken in integer nanoseconds and rounded once.
    fn delta(self, prev: Self) -> f64 {
        match (self, prev) {
            (Self::Ns(c), Self::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
            (Self::F64(c), Self::F64(p)) => c - p,
            // A stream's clock keeps one form for its life -- the bank
            // refuses a chunk in the other form -- so this arm is a direct
            // caller's, and it reads both as the numbers they report.
            (c, p) => c.seconds() - p.seconds(),
        }
    }

    /// Whether `self` is before `prev`: exact between two temporal values.
    pub fn is_before(self, prev: Self) -> bool {
        match (self, prev) {
            (Self::Ns(c), Self::Ns(p)) => c < p,
            (c, p) => c.seconds() < p.seconds(),
        }
    }
}

impl From<f64> for ClockValue {
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}

/// Nanoseconds as seconds: the whole seconds exact, the rest rounded once.
/// A difference that fits an `i64` -- any two stamps less than 292 years
/// apart -- takes the 64-bit road: the same two Euclidean operations on the
/// same integers, so the same bits, without the 128-bit division and
/// conversion that cost a window core a tenth of its row (task 143).
pub fn seconds_of_ns(ns: i128) -> f64 {
    const PER: i128 = 1_000_000_000;
    if ns < 0 {
        // On the magnitude, so a step back is its mirror: `−1 ms` was
        // `−1 + 0.999`, a last bit over `−0.001`, and a step back of exactly
        // `restart_after_step_back` restarted the model at the inclusive
        // edge (task 159, R2).
        return -seconds_of_ns(ns.saturating_neg());
    }
    if let Ok(d) = i64::try_from(ns) {
        const PER64: i64 = 1_000_000_000;
        return d.div_euclid(PER64) as f64 + d.rem_euclid(PER64) as f64 / 1e9;
    }
    ns.div_euclid(PER) as f64 + ns.rem_euclid(PER) as f64 / 1e9
}

/// Per-stream clock state. Serialized as part of a stream's saved state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClockState {
    prev_clock: Option<ClockValue>,
    prev_session: Option<u64>,
    /// Deltas of skipped rows, folded into the next accepted row.
    pending: f64,
    /// Whether any row has been seen (drives the row-count clock's first delta).
    started: bool,
    /// The time skipped rows covered, uncapped, folded into the next
    /// accepted row's [`ClockAdvance::elapsed`] (docs/PLAN.md task 153).
    /// Skipped when 0, the usual case, so a state writes the bytes it did.
    #[serde(default, skip_serializing_if = "is_zero")]
    skipped_elapsed: f64,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

impl ClockState {
    pub fn new() -> Self {
        Self::default()
    }

    /// The last clock value seen, `None` before the first row or on a
    /// row-count clock. What a caller uses to tell a stale group from a live
    /// one; [`ClockValue::seconds`] is the number to report.
    pub fn last_clock(&self) -> Option<ClockValue> {
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
    /// use online_core::{ClockCfg, ClockState, ClockValue, OnClockReset};
    ///
    /// let cfg = ClockCfg { gap_cap: 60.0, on_clock_reset: OnClockReset::Error, ..ClockCfg::default() };
    /// let mut clock = ClockState::new();
    /// // The first row of a stream has nothing to be a delta from.
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1000.0)), None, true).d_clock, 0.0);
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1010.0)), None, true).d_clock, 10.0);
    /// // A skipped row still moves the clock: its 5 units are carried into
    /// // the next accepted row's delta.
    /// assert!(!clock.advance(&cfg, Some(ClockValue::F64(1015.0)), None, false).accepted);
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1020.0)), None, true).d_clock, 10.0);
    /// // A gap is capped at `gap_cap`, so a weekend does not decay the
    /// // state to nothing.
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1e6)), None, true).d_clock, 60.0);
    /// ```
    ///
    /// `on_clock_reset` handles a *backwards* delta, and only within a
    /// session: on a session change the delta is `session_gap` if set,
    /// otherwise the raw delta clamped to `[0, gap_cap]` -- so a backwards
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
        clock: Option<ClockValue>,
        session: Option<u64>,
        accept: bool,
    ) -> ClockAdvance {
        self.step(cfg, clock, session, accept, false)
    }

    /// [`Self::advance`] for a row that is scored and not learned from: a
    /// row before the last learned clock is scored against the state as it
    /// stands -- a step of 0, never refused, never a reset -- whatever the
    /// policy, since scoring changes nothing that a late row could corrupt
    /// and re-scoring learned rows is ordinary (docs/PLAN.md task 120).
    pub fn advance_scoring(
        &mut self,
        cfg: &ClockCfg,
        clock: Option<ClockValue>,
        session: Option<u64>,
    ) -> ClockAdvance {
        self.step(cfg, clock, session, true, true)
    }

    fn step(
        &mut self,
        cfg: &ClockCfg,
        clock: Option<ClockValue>,
        session: Option<u64>,
        accept: bool,
        scoring: bool,
    ) -> ClockAdvance {
        let raw = match (clock, self.prev_clock) {
            (Some(c), Some(p)) => Some(c.delta(p)),
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

        // The time that passed: the column's forward step; across a session
        // change that restarts the clock, the session's gap, the only
        // measure of it there is; nothing for a step back within a session,
        // which is refused, scored as a step of 0, or a reset.
        let mut elapsed = match raw {
            None => 0.0,
            Some(raw) if raw >= 0.0 => raw,
            Some(_) if session_changed => match cfg.session_gap {
                Some(SessionGap::Gap(g)) => g.max(0.0),
                _ => 0.0,
            },
            Some(_) => 0.0,
        };
        let mut reset = false;
        let mut backwards = None;
        let mut disorder = None;
        let mut capped = false;
        let mut d = match raw {
            None => 0.0,
            Some(raw) => {
                if session_changed {
                    match cfg.session_gap {
                        Some(SessionGap::Reset) => {
                            reset = true;
                            0.0
                        }
                        Some(SessionGap::Gap(g)) => {
                            capped = g > cfg.gap_cap;
                            g.clamp(0.0, cfg.gap_cap)
                        }
                        None => {
                            capped = raw > cfg.gap_cap;
                            raw.clamp(0.0, cfg.gap_cap)
                        }
                    }
                } else if raw < 0.0 {
                    // A step back within a session. Scored, it is the state
                    // as it stands. Learned, `"error"` refuses it; under
                    // `"reset_state"` a step back by no more than
                    // `min_backwards_jump` is a late row and is refused too
                    // -- inclusive, so a `Date` clock's one-day step back is
                    // caught at a one-day minimum -- and a larger one starts
                    // the model over.
                    let back = -raw;
                    if scoring {
                        0.0
                    } else {
                        match cfg.on_clock_reset {
                            OnClockReset::Error => {
                                backwards = Some(raw);
                                0.0
                            }
                            OnClockReset::ResetState if back <= cfg.min_backwards_jump => {
                                disorder = Some(Disorder {
                                    back,
                                    min_backwards_jump: cfg.min_backwards_jump,
                                });
                                backwards = Some(raw);
                                0.0
                            }
                            OnClockReset::ResetState => {
                                reset = true;
                                0.0
                            }
                        }
                    }
                } else {
                    capped = raw > cfg.gap_cap;
                    raw.min(cfg.gap_cap)
                }
            }
        };

        if reset {
            self.pending = 0.0;
            self.skipped_elapsed = 0.0;
            d = 0.0;
            elapsed = 0.0;
        }
        self.prev_clock = clock;
        self.prev_session = session;
        self.started = true;

        if accept {
            // The skipped rows' time is carried into this row's, and the
            // ceiling holds for the total: `gap_cap` is the most a model
            // sees between two rows it learns from. Ten skipped rows 100
            // apart under a cap of 60 handed the next one 660 (review
            // 2026-09-12, S3). A total over the cap is a capped gap.
            let total = self.pending + d;
            self.pending = 0.0;
            let elapsed = self.skipped_elapsed + elapsed;
            self.skipped_elapsed = 0.0;
            ClockAdvance {
                d_clock: total.min(cfg.gap_cap),
                reset,
                accepted: true,
                backwards,
                disorder,
                session_changed,
                capped: capped || total > cfg.gap_cap,
                elapsed,
            }
        } else {
            self.pending += d;
            self.skipped_elapsed += elapsed;
            ClockAdvance {
                d_clock: 0.0,
                reset,
                accepted: false,
                backwards,
                disorder,
                session_changed,
                capped,
                elapsed: 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max: f64) -> ClockCfg {
        ClockCfg {
            gap_cap: max,
            ..Default::default()
        }
    }

    /// Task 153: `elapsed` is the time that passed, uncapped, beside the
    /// capped `d_clock`: across a capped gap it is the whole gap, a skipped
    /// row's time joins the next accepted row's, a session change counts the
    /// column's step when the clock runs on and the session's gap when it
    /// restarts, and a reset or the first row is 0.
    #[test]
    fn elapsed_is_the_time_that_passed_uncapped() {
        let v = |x: f64| Some(ClockValue::F64(x));
        let mut c = ClockState::new();
        let cfg = cfg(2.0);
        assert_eq!(c.advance(&cfg, v(0.0), None, true).elapsed, 0.0);
        let gap = c.advance(&cfg, v(5.0), None, true);
        assert_eq!((gap.d_clock, gap.elapsed, gap.capped), (2.0, 5.0, true));
        // Two skipped rows, each a step of 1.5, then an accepted one a step
        // of 1 later: 4 of time, capped to 2 for the models.
        assert_eq!(c.advance(&cfg, v(6.5), None, false).elapsed, 0.0);
        c.advance(&cfg, v(8.0), None, false);
        let after = c.advance(&cfg, v(9.0), None, true);
        assert_eq!((after.d_clock, after.elapsed), (2.0, 4.0));

        let sessions = ClockCfg {
            gap_cap: 2.0,
            session_gap: Some(SessionGap::Gap(30.0)),
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&sessions, v(10.0), Some(1), true);
        // The clock runs on: one unit passed, whatever the model forgets.
        let on = c.advance(&sessions, v(11.0), Some(2), true);
        assert_eq!(
            (on.d_clock, on.elapsed, on.session_changed),
            (2.0, 1.0, true)
        );
        // The clock restarts: the session's gap is the only measure there is.
        let restart = c.advance(&sessions, v(0.0), Some(3), true);
        assert_eq!(restart.elapsed, 30.0);

        let reset = ClockCfg {
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&reset, v(0.0), Some(1), false);
        let r = c.advance(&reset, v(4.0), Some(2), true);
        assert_eq!((r.reset, r.elapsed), (true, 0.0));
    }

    /// The gap between two temporal values is taken in integer nanoseconds,
    /// so a nanosecond tick four decades into a stream is the delta it is
    /// at the start. As doubles of seconds the two instants round to the
    /// double's resolution at that age -- 60 ns at a decade -- and the
    /// tick is lost.
    #[test]
    fn a_temporal_gap_is_exact_at_any_age() {
        let cfg = ClockCfg::default();
        let decade: i64 = 315_576_000 * 1_000_000_000;
        for start in [0, decade, 4 * decade] {
            let mut c = ClockState::new();
            c.advance(&cfg, Some(ClockValue::Ns(start)), None, true);
            let one = c.advance(&cfg, Some(ClockValue::Ns(start + 1)), None, true);
            assert_eq!(one.d_clock, 1e-9);
            let half = c.advance(
                &cfg,
                Some(ClockValue::Ns(start + 1_500_000_001)),
                None,
                true,
            );
            assert_eq!(half.d_clock, 1.5);
            let odd = c.advance(
                &cfg,
                Some(ClockValue::Ns(start + 3_500_000_002)),
                None,
                true,
            );
            assert!((odd.d_clock - 2.000000001).abs() < 1e-15);
        }
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(decade as f64 / 1e9)), None, true);
        let lost = c.advance(
            &cfg,
            Some(ClockValue::F64((decade + 1) as f64 / 1e9)),
            None,
            true,
        );
        assert_eq!(lost.d_clock, 0.0);
    }

    /// `is_before` compares two temporal values exactly, where their
    /// seconds since the epoch are the same double.
    #[test]
    fn before_is_exact_between_temporal_values() {
        let t = ClockValue::Ns(1_704_187_800 * 1_000_000_000);
        let later = ClockValue::Ns(1_704_187_800 * 1_000_000_000 + 1);
        assert!(t.is_before(later));
        assert!(!later.is_before(t));
        assert!(!t.is_before(t));
        assert_eq!(t.seconds(), later.seconds());
        assert_eq!(t.seconds(), 1_704_187_800.0);
        assert!(ClockValue::F64(1.0).is_before(ClockValue::F64(2.0)));
        // A numeric value is not before itself, nor before the temporal
        // value that reports the same number.
        assert!(!ClockValue::F64(1.0).is_before(ClockValue::F64(1.0)));
        assert!(!ClockValue::F64(12.0).is_before(ClockValue::Ns(12_000_000_000)));
    }

    /// A step back of a fraction of a second is the mirror of the step
    /// forward: `−1 ms` is `−0.001` exactly, as `1 ms` is `0.001`. It read
    /// `−1 + 0.999`, a last bit over, so a step back of exactly
    /// `restart_after_step_back` fell past the inclusive edge and restarted
    /// the model (task 159, R2).
    #[test]
    fn a_step_back_is_the_mirror_of_the_step_forward() {
        for ns in [
            1i128,
            999,
            1_000_000,
            123_456_789,
            1_500_000_000,
            10_000_000_000_500_000_000,
        ] {
            assert_eq!(seconds_of_ns(-ns), -seconds_of_ns(ns), "{ns}");
        }
        assert_eq!(seconds_of_ns(-1_000_000), -0.001);
        assert_eq!(ClockValue::Ns(0).delta(ClockValue::Ns(1_000_000)), -0.001);
        assert_eq!(seconds_of_ns(i128::MIN), -seconds_of_ns(i128::MAX));
    }

    /// A count of nanoseconds past an `i64` -- a difference of two stamps
    /// more than 292 years apart -- takes the 128-bit road, and is still the
    /// whole seconds exact and the rest rounded once: `10^19 + 5·10^8` ns is
    /// `10^10 + 0.5` s, and its negative is `−10^10 − 0.5`, whose Euclidean
    /// whole part is `−10^10 − 1` with `0.5` over.
    #[test]
    fn nanoseconds_past_an_i64_are_seconds_too() {
        let ns: i128 = 10_000_000_000_500_000_000;
        assert!(i64::try_from(ns).is_err(), "the 128-bit road");
        assert_eq!(seconds_of_ns(ns), 10_000_000_000.5);
        assert_eq!(seconds_of_ns(-ns), -10_000_000_000.5);
        // A whole number of seconds, and one that is all remainder.
        assert_eq!(seconds_of_ns(30_000_000_000_000_000_000), 3e10);
        assert_eq!(
            seconds_of_ns(-30_000_000_000_250_000_000),
            -30_000_000_000.25
        );
    }

    /// Task 153's `skipped_elapsed` is written only while it holds time, so
    /// a state with none is the bytes it was before the field existed; one
    /// with some writes it, and reads it back.
    #[test]
    fn a_state_writes_the_skipped_time_only_while_there_is_some() {
        let cfg = ClockCfg::default();
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true);
        let json = serde_json::to_value(&c).unwrap();
        assert!(json.get("skipped_elapsed").is_none(), "{json}");
        let before = rmp_serde::to_vec(&c).unwrap();
        // A skipped row four units on: its time waits for the next row.
        c.advance(&cfg, Some(ClockValue::F64(4.0)), None, false);
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["skipped_elapsed"], 4.0, "{json}");
        let bytes = rmp_serde::to_vec(&c).unwrap();
        assert!(bytes.len() > before.len());
        let mut back: ClockState = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, c);
        let next = back.advance(&cfg, Some(ClockValue::F64(5.0)), None, true);
        assert_eq!((next.d_clock, next.elapsed), (5.0, 5.0));
        // Folded into that row, it is gone from the state again.
        let json = serde_json::to_value(&back).unwrap();
        assert!(json.get("skipped_elapsed").is_none(), "{json}");
    }

    /// The previous session is the last row's, skipped or not, and none
    /// before the first row.
    #[test]
    fn the_previous_session_is_the_last_rows() {
        let cfg = ClockCfg {
            session_gap: Some(SessionGap::Gap(1.0)),
            ..Default::default()
        };
        let mut c = ClockState::new();
        assert_eq!(c.prev_session(), None);
        c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(42), true);
        assert_eq!(c.prev_session(), Some(42));
        c.advance(&cfg, Some(ClockValue::F64(1.0)), Some(7), false);
        assert_eq!(c.prev_session(), Some(7));
        c.advance(&cfg, Some(ClockValue::F64(2.0)), None, true);
        assert_eq!(c.prev_session(), None);
    }

    /// A session's gap is the time that passed across a session change that
    /// restarts the clock, and only there: a step back within one session
    /// passed no time, scored, refused as a late row or refused outright,
    /// whatever `session_gap` says.
    #[test]
    fn a_step_back_within_a_session_passes_no_time() {
        let v = |x: f64| Some(ClockValue::F64(x));
        for policy in [OnClockReset::Error, OnClockReset::ResetState] {
            let cfg = ClockCfg {
                gap_cap: 60.0,
                on_clock_reset: policy,
                session_gap: Some(SessionGap::Gap(30.0)),
                min_backwards_jump: 5.0,
            };
            for session in [None, Some(3)] {
                let mut c = ClockState::new();
                c.advance(&cfg, v(100.0), session, true);
                let scored = c.clone().advance_scoring(&cfg, v(90.0), session);
                assert_eq!(
                    (scored.d_clock, scored.elapsed, scored.session_changed),
                    (0.0, 0.0, false),
                    "{policy:?} {session:?}: scored"
                );
                let refused = c.clone().advance(&cfg, v(98.0), session, true);
                assert_eq!(refused.backwards, Some(-2.0), "{policy:?} {session:?}");
                assert_eq!(refused.elapsed, 0.0, "{policy:?} {session:?}: refused");
                // Skipped, it carries nothing into the next row either.
                let mut skipped = c.clone();
                skipped.advance(&cfg, v(98.0), session, false);
                let next = skipped.advance(&cfg, v(99.0), session, true);
                assert_eq!(next.elapsed, 1.0, "{policy:?} {session:?}: skipped");
            }
        }
    }

    /// A gap exactly at the ceiling is not over it, on either road a session
    /// change takes: the session's gap, or the clock's own step where
    /// `session_gap` is unset (a direct caller's). Each is judged on the row
    /// itself, accepted or skipped, where an accepted row's total is judged
    /// again.
    #[test]
    fn a_session_change_at_the_ceiling_is_not_capped() {
        let v = |x: f64| Some(ClockValue::F64(x));
        for accept in [true, false] {
            let at_cap = ClockCfg {
                gap_cap: 60.0,
                session_gap: Some(SessionGap::Gap(60.0)),
                ..Default::default()
            };
            let mut c = ClockState::new();
            c.advance(&at_cap, v(0.0), Some(1), true);
            let a = c.advance(&at_cap, v(1.0), Some(2), accept);
            assert!(a.session_changed && !a.capped, "session gap at the cap");

            let unset = ClockCfg {
                gap_cap: 60.0,
                ..Default::default()
            };
            for (step, want) in [(10.0, false), (60.0, false), (61.0, true)] {
                let mut c = ClockState::new();
                c.advance(&unset, v(0.0), Some(1), true);
                let a = c.advance(&unset, v(step), Some(2), accept);
                assert!(a.session_changed);
                assert_eq!(a.capped, want, "a step of {step}, accepted {accept}");
            }
        }
    }

    /// A stream's clock keeps one form for its life; a direct caller that
    /// mixes the two gets the difference of the numbers they report.
    #[test]
    fn mixed_forms_read_as_their_numbers() {
        let cfg = ClockCfg::default();
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true);
        let adv = c.advance(&cfg, Some(ClockValue::Ns(12_000_000_000)), None, true);
        assert_eq!(adv.d_clock, 2.0);
        assert_eq!(c.last_clock(), Some(ClockValue::Ns(12_000_000_000)));
    }

    /// Task 47: `capped` says "this jump was bigger than the model is
    /// allowed to see", which is the signal anything lagged by *rows* needs.
    #[test]
    fn capped_marks_the_rows_whose_gap_hit_the_ceiling() {
        let cfg = cfg(60.0);
        let mut c = ClockState::new();
        // The first row has no delta at all.
        assert!(
            !c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true)
                .capped
        );
        assert!(
            !c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true)
                .capped
        );
        // Exactly at the ceiling is not over it.
        assert!(
            !c.advance(&cfg, Some(ClockValue::F64(70.0)), None, true)
                .capped
        );
        let over = c.advance(&cfg, Some(ClockValue::F64(1e6)), None, true);
        assert!(over.capped && over.d_clock == 60.0);
        // A skipped row's gap breaks adjacency too, so the flag is set
        // whether or not the row is accepted.
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(2e6)), None, false)
                .capped
        );
        // An infinite ceiling never caps.
        let mut c = ClockState::new();
        let none = ClockCfg::default();
        c.advance(&none, Some(ClockValue::F64(0.0)), None, true);
        assert!(
            !c.advance(&none, Some(ClockValue::F64(1e300)), None, true)
                .capped
        );
        // A row-count clock cannot jump.
        let mut c = ClockState::new();
        c.advance(&cfg, None, None, true);
        assert!(!c.advance(&cfg, None, None, true).capped);
    }

    /// A step back is refused or starts the model over; neither is a capped
    /// gap (the removed `"max"` policy's step was the ceiling, and capped).
    #[test]
    fn a_step_back_caps_nothing() {
        for policy in [OnClockReset::ResetState, OnClockReset::Error] {
            let cfg = ClockCfg {
                gap_cap: 60.0,
                on_clock_reset: policy,
                ..Default::default()
            };
            let mut c = ClockState::new();
            c.advance(&cfg, Some(ClockValue::F64(100.0)), None, true);
            let a = c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true);
            assert!(!a.capped, "{policy:?}");
        }
    }

    #[test]
    fn a_session_gap_caps_only_when_it_is_over_the_ceiling() {
        let with_gap = |g: f64| ClockCfg {
            gap_cap: 60.0,
            session_gap: Some(SessionGap::Gap(g)),
            ..Default::default()
        };
        for (gap, want) in [(30.0, false), (600.0, true)] {
            let cfg = with_gap(gap);
            let mut c = ClockState::new();
            c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(1), true);
            let adv = c.advance(&cfg, Some(ClockValue::F64(1.0)), Some(2), true);
            assert!(adv.session_changed);
            assert_eq!(adv.capped, want, "gap {gap}");
        }
        // A session reset rebuilds the model, so it caps nothing.
        let cfg = ClockCfg {
            gap_cap: 60.0,
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(1), true);
        let adv = c.advance(&cfg, Some(ClockValue::F64(1e6)), Some(2), true);
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
        let cfg = cfg(50.0);
        // Scored: a step back is no step, and a gap past the cap is the cap.
        let mut c = ClockState::new();
        let got: Vec<f64> = t
            .iter()
            .map(|&ti| {
                c.advance_scoring(&cfg, Some(ClockValue::F64(ti)), None)
                    .d_clock
            })
            .collect();
        assert_eq!(got, vec![0.0, 10.0, 0.0, 1.0, 50.0]);
        // Learned, the default policy refuses the step back ...
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true);
        c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true);
        let a = c.advance(&cfg, Some(ClockValue::F64(5.0)), None, true);
        assert_eq!(a.backwards, Some(-5.0));
        assert!(!a.reset);
        // ... and `reset_state` starts over past its minimum.
        let rst = ClockCfg {
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 2.0,
            ..cfg
        };
        let mut c = ClockState::new();
        c.advance(&rst, Some(ClockValue::F64(0.0)), None, true);
        c.advance(&rst, Some(ClockValue::F64(10.0)), None, true);
        let a = c.advance(&rst, Some(ClockValue::F64(5.0)), None, true);
        assert!(a.reset && a.d_clock == 0.0 && a.backwards.is_none());
    }

    /// Scoring never refuses and never resets: a row before the last
    /// learned clock is scored against the state as it stands, under either
    /// policy, and a late row's minimum does not apply.
    #[test]
    fn a_scored_step_back_is_the_state_as_it_stands() {
        for policy in [OnClockReset::Error, OnClockReset::ResetState] {
            let cfg = ClockCfg {
                gap_cap: 50.0,
                on_clock_reset: policy,
                min_backwards_jump: 5.0,
                ..Default::default()
            };
            let mut c = ClockState::new();
            c.advance(&cfg, Some(ClockValue::F64(100.0)), None, true);
            for t in [99.0, 10.0] {
                let a = c
                    .clone()
                    .advance_scoring(&cfg, Some(ClockValue::F64(t)), None);
                assert_eq!(
                    (a.d_clock, a.reset, a.backwards),
                    (0.0, false, None),
                    "{policy:?}"
                );
            }
        }
    }

    /// A repeated clock value is a *zero* delta, not a backwards one: it must
    /// not be routed through `on_clock_reset`. (Found by `cargo mutants`:
    /// nothing here distinguished `raw < 0.0` from `raw <= 0.0`.)
    #[test]
    fn duplicate_clock_values_are_zero_deltas() {
        for policy in [OnClockReset::ResetState, OnClockReset::Error] {
            let mut c = ClockState::new();
            let cfg = ClockCfg {
                gap_cap: 10.0,
                on_clock_reset: policy,
                ..Default::default()
            };
            assert_eq!(
                c.advance(&cfg, Some(ClockValue::F64(5.0)), None, true)
                    .d_clock,
                0.0
            );
            let a = c.advance(&cfg, Some(ClockValue::F64(5.0)), None, true);
            assert_eq!(
                a.d_clock, 0.0,
                "{policy:?}: repeated clock must give delta 0"
            );
            assert!(
                !a.reset && a.backwards.is_none(),
                "{policy:?}: a repeated clock is neither a reset nor a step back"
            );
            // and a genuinely backwards clock is handled by the policy
            let b = c.advance(&cfg, Some(ClockValue::F64(4.0)), None, true);
            match policy {
                OnClockReset::ResetState => assert!(b.reset),
                OnClockReset::Error => assert_eq!(b.backwards, Some(-1.0)),
            }
        }
    }

    #[test]
    fn error_policy_reports_a_backwards_clock() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            gap_cap: 50.0,
            on_clock_reset: OnClockReset::Error,
            ..Default::default()
        };
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true)
                .backwards
                .is_none()
        );
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true)
                .backwards
                .is_none()
        );
        // a repeated value is a zero delta, not backwards
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true)
                .backwards
                .is_none()
        );
        let a = c.advance(&cfg, Some(ClockValue::F64(4.0)), None, true);
        assert_eq!(a.backwards, Some(-6.0));
        assert!(!a.reset, "the error policy must not silently reset");
    }

    #[test]
    fn session_change_is_reported_separately_from_reset() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            gap_cap: 50.0,
            session_gap: Some(SessionGap::Gap(5.0)),
            ..Default::default()
        };
        assert!(
            !c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(1), true)
                .session_changed
        );
        assert!(
            !c.advance(&cfg, Some(ClockValue::F64(1.0)), Some(1), true)
                .session_changed
        );
        let a = c.advance(&cfg, Some(ClockValue::F64(2.0)), Some(2), true);
        assert!(a.session_changed, "a new session id should be reported");
        assert!(!a.reset, "a gap is not a reset");
    }

    #[test]
    fn session_gap_overrides_delta() {
        let mut c = ClockState::new();
        let cfg = ClockCfg {
            gap_cap: 50.0,
            session_gap: Some(SessionGap::Gap(7.5)),
            ..Default::default()
        };
        c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(0), true);
        c.advance(&cfg, Some(ClockValue::F64(10.0)), Some(0), true);
        // negative raw delta AND session change: session gap wins
        let a = c.advance(&cfg, Some(ClockValue::F64(5.0)), Some(1), true);
        assert_eq!(a.d_clock, 7.5);

        let mut c = ClockState::new();
        let cfg = ClockCfg {
            gap_cap: 50.0,
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(0), true);
        let a = c.advance(&cfg, Some(ClockValue::F64(10.0)), Some(1), true);
        assert!(a.reset);
    }

    /// Under `"reset_state"` a step back by no more than
    /// `min_backwards_jump` is a late row and is refused; inclusive, so a
    /// step equal to the minimum is caught (task 120: a `Date` clock's
    /// one-day step back at a one-day minimum used to pass). A larger one
    /// starts the model over.
    #[test]
    fn a_step_back_no_larger_than_the_minimum_is_a_late_row() {
        let cfg = ClockCfg {
            gap_cap: 100.0,
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 100.0,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0] {
            assert!(
                c.advance(&cfg, Some(ClockValue::F64(t)), None, true)
                    .backwards
                    .is_none()
            );
        }
        let a = c.advance(&cfg, Some(ClockValue::F64(17.0)), None, true);
        assert_eq!(a.backwards, Some(-3.0));
        assert_eq!(
            a.disorder,
            Some(Disorder {
                back: 3.0,
                min_backwards_jump: 100.0
            })
        );
        assert_eq!(a.d_clock, 0.0);
        assert!(!a.reset);
        for (back, late) in [(100.0, true), (100.5, false)] {
            let mut c = ClockState::new();
            c.advance(&cfg, Some(ClockValue::F64(200.0)), None, true);
            let b = c.advance(&cfg, Some(ClockValue::F64(200.0 - back)), None, true);
            assert_eq!(b.disorder.is_some(), late, "a step back of {back}");
            assert_eq!(b.reset, !late, "a step back of {back}");
            assert!(!b.capped);
        }
    }

    /// The very first delta is judged like any other: no warmup, no
    /// exemption.
    #[test]
    fn the_first_delta_is_judged_like_any_other() {
        let cfg = ClockCfg {
            gap_cap: 100.0,
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 100.0,
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(5.0)), None, true);
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true)
                .disorder
                .is_some()
        );
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(150.0)), None, true);
        let a = c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true);
        assert!(a.disorder.is_none() && a.reset);
    }

    /// A minimum of 0 under `"reset_state"` says every step back is a reset,
    /// however small.
    #[test]
    fn a_minimum_of_zero_resets_on_every_step_back() {
        let cfg = ClockCfg {
            gap_cap: 1e9,
            on_clock_reset: OnClockReset::ResetState,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0] {
            c.advance(&cfg, Some(ClockValue::F64(t)), None, true);
        }
        for t in [19.0, 18.9] {
            let a = c.advance(&cfg, Some(ClockValue::F64(t)), None, true);
            assert!(a.backwards.is_none() && a.reset, "{t}");
        }
    }

    /// The `error` policy refuses every backwards jump and reads no minimum:
    /// `disorder` stays `None` there, whatever `min_backwards_jump` holds.
    #[test]
    fn the_error_policy_keeps_its_own_refusal() {
        let cfg = ClockCfg {
            gap_cap: 100.0,
            min_backwards_jump: 100.0,
            on_clock_reset: OnClockReset::Error,
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true);
        let a = c.advance(&cfg, Some(ClockValue::F64(7.0)), None, true);
        assert_eq!(a.backwards, Some(-3.0));
        assert!(a.disorder.is_none());
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(500.0)), None, true);
        let b = c.advance(&cfg, Some(ClockValue::F64(7.0)), None, true);
        assert_eq!(b.backwards, Some(-493.0));
        assert!(!b.reset);
    }

    #[test]
    fn skipped_rows_fold_into_pending() {
        let mut c = ClockState::new();
        let cfg = cfg(f64::INFINITY);
        c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true);
        let s = c.advance(&cfg, Some(ClockValue::F64(3.0)), None, false);
        assert!(!s.accepted);
        let a = c.advance(&cfg, Some(ClockValue::F64(5.0)), None, true);
        assert_eq!(a.d_clock, 5.0); // 3 (pending) + 2
    }

    /// A skipped row's delta is capped on its own, and so is the total the
    /// next accepted row is handed: ten skipped rows 100 apart under a cap of
    /// 60 handed it 660, eleven times the ceiling `gap_cap` promises a
    /// model sees (review 2026-09-12, S3). The fold is capped, and says so.
    #[test]
    fn a_skipped_run_hands_the_next_row_at_most_the_cap() {
        let mut c = ClockState::new();
        let cfg = cfg(60.0);
        c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true);
        for i in 1..=10 {
            let s = c.advance(
                &cfg,
                Some(ClockValue::F64(100.0 * f64::from(i))),
                None,
                false,
            );
            assert!(!s.accepted);
        }
        let a = c.advance(&cfg, Some(ClockValue::F64(1100.0)), None, true);
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

/// Task 120, decided 2026-09-28: no model ever receives a non-finite step.
/// Every step a model is handed comes from [`ClockState::advance`] or
/// [`ClockState::advance_scoring`], so the property is held here, over every
/// configuration the spec lets through -- a finite positive cap with a clock
/// and none without one, either policy with any finite minimum, a finite
/// session gap or `"reset"` -- and clocks that step back, repeat, jump by
/// `1e300` and change session anywhere. An infinite cap or session gap
/// handed a model `+inf` at a step back or a session change before the
/// decision (`holt` then predicted null for good).
#[cfg(test)]
mod finite_steps {
    use super::*;
    use proptest::prelude::*;

    fn cfg() -> impl Strategy<Value = (ClockCfg, bool)> {
        (
            prop_oneof![Just(None), (1e-9f64..1e12).prop_map(Some)],
            any::<bool>(),
            0.0f64..1e12,
            prop_oneof![
                Just(None),
                Just(Some(SessionGap::Reset)),
                (0.0f64..1e13).prop_map(|g| Some(SessionGap::Gap(g))),
            ],
        )
            .prop_map(|(cap, reset, min, session_gap)| {
                let temporal = cap.is_some();
                (
                    ClockCfg {
                        // A row-count clock has no cap (`Spec::clock_cfg`).
                        gap_cap: cap.unwrap_or(f64::INFINITY),
                        on_clock_reset: if reset {
                            OnClockReset::ResetState
                        } else {
                            OnClockReset::Error
                        },
                        session_gap,
                        min_backwards_jump: min,
                    },
                    temporal,
                )
            })
    }

    fn clock() -> impl Strategy<Value = f64> {
        prop_oneof![
            -1e300f64..1e300,
            -1e6f64..1e6,
            Just(0.0),
            Just(1e300),
            Just(-1e300),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]
        #[test]
        fn every_step_is_finite(
            (cfg, with_clock) in cfg(),
            rows in prop::collection::vec((clock(), 0u64..3, any::<bool>()), 1..40),
        ) {
            let mut learned = ClockState::new();
            for (t, session, accept) in rows {
                let c = with_clock.then_some(ClockValue::F64(t));
                let scored = learned.clone().advance_scoring(&cfg, c, Some(session));
                let a = learned.advance(&cfg, c, Some(session), accept);
                for (what, d) in [("learned", a.d_clock), ("scored", scored.d_clock)] {
                    prop_assert!(
                        d.is_finite() && d >= 0.0 && d <= cfg.gap_cap,
                        "{what} step {d} under {cfg:?}"
                    );
                }
            }
        }
    }
}
