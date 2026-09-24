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
    /// A backwards clock jump smaller than this, in clock units, is refused as
    /// [`Disorder`] whatever `on_clock_reset` says: `max_dclock` is the most
    /// two adjacent rows can be apart and a session is longer than that, so a
    /// jump back by less than it is a late row, not a boundary. A jump of at
    /// least this much takes the policy. `0` disables; the spec defaults it
    /// to `max_dclock`, and to 0 under an infinite cap, which gives the check
    /// nothing to compare against.
    pub min_backwards_jump: f64,
}

impl Default for ClockCfg {
    /// The disorder check **off** here: this is the bare core default. The
    /// spec layer is where "on by default" lives (`min_backwards_jump =
    /// max_dclock`), so a direct core caller opts in explicitly and the
    /// core's own tests keep their literal meaning.
    fn default() -> Self {
        Self {
            max_dclock: f64::INFINITY,
            on_clock_reset: OnClockReset::default(),
            session_gap: None,
            min_backwards_jump: 0.0,
        }
    }
}

/// Why a backwards clock jump was refused as obviously out-of-order data,
/// carried on [`ClockAdvance::disorder`] for the caller's message: the jump
/// and the minimum it fell short of.
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
    /// `on_clock_reset` = [`OnClockReset::Error`], or -- whatever the policy
    /// -- when the jump is smaller than `min_backwards_jump` (`disorder`
    /// carries the numbers). The caller must turn this into an error naming
    /// the row. Carries the offending raw delta.
    pub backwards: Option<f64>,
    /// With `backwards`: the jump and the minimum it fell short of, `None`
    /// when it was the `error` policy alone.
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
pub fn seconds_of_ns(ns: i128) -> f64 {
    const PER: i128 = 1_000_000_000;
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
    /// let cfg = ClockCfg { max_dclock: 60.0, on_clock_reset: OnClockReset::Max, ..ClockCfg::default() };
    /// let mut clock = ClockState::new();
    /// // The first row of a stream has nothing to be a delta from.
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1000.0)), None, true).d_clock, 0.0);
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1010.0)), None, true).d_clock, 10.0);
    /// // A skipped row still moves the clock: its 5 units are carried into
    /// // the next accepted row's delta.
    /// assert!(!clock.advance(&cfg, Some(ClockValue::F64(1015.0)), None, false).accepted);
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1020.0)), None, true).d_clock, 10.0);
    /// // A gap is capped at `max_dclock`, so a weekend does not decay the
    /// // state to nothing.
    /// assert_eq!(clock.advance(&cfg, Some(ClockValue::F64(1e6)), None, true).d_clock, 60.0);
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
        clock: Option<ClockValue>,
        session: Option<u64>,
        accept: bool,
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
                    // interleaved rows is worse than stopping. One rule, off
                    // at 0: `max_dclock` is the most two adjacent rows can be
                    // apart and a session is longer than that, so a jump back
                    // by less than `min_backwards_jump` (the spec defaults it
                    // to `max_dclock`) is a late row, not a boundary. A jump
                    // of at least that much takes the policy. Strict `<`: a
                    // jump equal to the minimum meets it. The `error` policy
                    // refuses every backwards jump already and keeps its own
                    // message: the rule adds nothing there.
                    let back = -raw;
                    let refuses_all = matches!(cfg.on_clock_reset, OnClockReset::Error);
                    let too_small = !refuses_all
                        && cfg.min_backwards_jump > 0.0
                        && back < cfg.min_backwards_jump;
                    if too_small {
                        disorder = Some(Disorder {
                            back,
                            min_backwards_jump: cfg.min_backwards_jump,
                        });
                        backwards = Some(raw);
                        0.0
                    } else {
                        match cfg.on_clock_reset {
                            OnClockReset::Max => {
                                // The policy says "as far apart as they can
                                // be", which is the ceiling: adjacency is
                                // gone.
                                capped = true;
                                cfg.max_dclock
                            }
                            OnClockReset::Zero => 0.0,
                            OnClockReset::ResetState => {
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
            c.advance(&cfg, Some(ClockValue::F64(100.0)), None, true);
            assert_eq!(
                c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true)
                    .capped,
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
            c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(1), true);
            let adv = c.advance(&cfg, Some(ClockValue::F64(1.0)), Some(2), true);
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
        let mut c = ClockState::new();
        let cfg = cfg(50.0);
        let got: Vec<f64> = t
            .iter()
            .map(|&ti| {
                c.advance(&cfg, Some(ClockValue::F64(ti)), None, true)
                    .d_clock
            })
            .collect();
        assert_eq!(got, vec![0.0, 10.0, 50.0, 1.0, 50.0]);

        let mut c = ClockState::new();
        let zero = ClockCfg {
            on_clock_reset: OnClockReset::Zero,
            ..cfg
        };
        c.advance(&zero, Some(ClockValue::F64(0.0)), None, true);
        c.advance(&zero, Some(ClockValue::F64(10.0)), None, true);
        assert_eq!(
            c.advance(&zero, Some(ClockValue::F64(5.0)), None, true)
                .d_clock,
            0.0
        );

        let mut c = ClockState::new();
        let rst = ClockCfg {
            on_clock_reset: OnClockReset::ResetState,
            ..cfg
        };
        c.advance(&rst, Some(ClockValue::F64(0.0)), None, true);
        c.advance(&rst, Some(ClockValue::F64(10.0)), None, true);
        let a = c.advance(&rst, Some(ClockValue::F64(5.0)), None, true);
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
                !a.reset,
                "{policy:?}: a repeated clock must not reset state"
            );
            // and a genuinely backwards clock still is handled by the policy
            let b = c.advance(&cfg, Some(ClockValue::F64(4.0)), None, true);
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
            max_dclock: 50.0,
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
            max_dclock: 50.0,
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
            max_dclock: 50.0,
            session_gap: Some(SessionGap::Reset),
            ..Default::default()
        };
        c.advance(&cfg, Some(ClockValue::F64(0.0)), Some(0), true);
        let a = c.advance(&cfg, Some(ClockValue::F64(10.0)), Some(1), true);
        assert!(a.reset);
    }

    /// A backwards jump smaller than `min_backwards_jump` is out-of-order
    /// data whatever the policy: `max_dclock` is the most two adjacent rows
    /// can be apart, so a jump back by less cannot be a session boundary.
    /// Strict `<`: a jump equal to the minimum meets it and is a boundary.
    #[test]
    fn a_backwards_jump_under_the_minimum_is_refused() {
        let cfg = ClockCfg {
            max_dclock: 100.0,
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
        // Equal to the minimum: a boundary, absorbed by the `max` policy.
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 200.0] {
            c.advance(&cfg, Some(ClockValue::F64(t)), None, true);
        }
        let b = c.advance(&cfg, Some(ClockValue::F64(100.0)), None, true);
        assert!(b.backwards.is_none() && b.disorder.is_none() && b.capped);
        assert_eq!(b.d_clock, 100.0);
    }

    /// The very first delta is judged like any other: no warmup, no
    /// exemption.
    #[test]
    fn the_first_delta_is_judged_like_any_other() {
        let cfg = ClockCfg {
            max_dclock: 100.0,
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
        c.advance(&cfg, Some(ClockValue::F64(100.0)), None, true);
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(0.0)), None, true)
                .disorder
                .is_none()
        );
    }

    /// At 0 -- the core default -- every backwards jump takes the policy as
    /// before, however small.
    #[test]
    fn the_check_off_leaves_the_policy_alone() {
        let cfg = ClockCfg {
            max_dclock: 1e9,
            ..Default::default()
        };
        let mut c = ClockState::new();
        for t in [0.0, 10.0, 20.0] {
            c.advance(&cfg, Some(ClockValue::F64(t)), None, true);
        }
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(19.0)), None, true)
                .backwards
                .is_none()
        );
        assert!(
            c.advance(&cfg, Some(ClockValue::F64(18.0)), None, true)
                .backwards
                .is_none()
        );
    }

    /// The `error` policy refuses every backwards jump already and keeps its
    /// own report: `disorder` stays `None` there.
    #[test]
    fn the_error_policy_keeps_its_own_refusal() {
        let cfg = ClockCfg {
            max_dclock: 100.0,
            min_backwards_jump: 100.0,
            on_clock_reset: OnClockReset::Error,
            ..Default::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, Some(ClockValue::F64(10.0)), None, true);
        let a = c.advance(&cfg, Some(ClockValue::F64(7.0)), None, true);
        assert_eq!(a.backwards, Some(-3.0));
        assert!(a.disorder.is_none());
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
    /// 60 handed it 660, eleven times the ceiling `max_dclock` promises a
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
