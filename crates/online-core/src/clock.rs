//! Clock/decay semantics shared by every entry point (docs/PLAN.md §3).
//!
//! Rows arrive with a raw clock value (or none, meaning row count) and an optional
//! session id. [`ClockState::advance`] turns those into the capped, gap-adjusted
//! `d_clock` a model consumes, folding the deltas of skipped (feature-null) rows
//! into the next accepted row so that decay still covers skipped time.
//! [`ClockState::advance_stamped`] also gives each accepted row its [`Stamp`]:
//! the same decayed clock held exactly, which a model's window decides its
//! edge from (docs/PLAN.md task 175); and its place on the elapsed clock held
//! exactly, which a delay decides its release from (task 176).

use std::cmp::Ordering;

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
    /// Whether a model can run on this decay: a half-life above 0 (`inf`
    /// is no decay), or a factor in `(0, 1]` (1 is no decay). Every model's
    /// `new` makes this check, where only the bank's spec made it
    /// (`Spec::decays`), so the Rust API built a model on any decay:
    /// `Halflife(0)` made the first row's factor NaN (`exp2(-(0/0))`), a
    /// factor below 0 made NaN, and one above 1 made the weights grow
    /// (review 2026-10-05, CF5). The message names the parameter; the
    /// caller adds the model's name.
    pub fn check(&self) -> Result<(), String> {
        match *self {
            Decay::Halflife(h) if h.is_nan() || h <= 0.0 => {
                Err(format!("half_life must be > 0 (got {h}); inf is no decay"))
            }
            Decay::Lam(l) if l.is_nan() || l <= 0.0 || l > 1.0 => {
                Err(format!("lam must be in (0, 1] (got {l}); 1 is no decay"))
            }
            _ => Ok(()),
        }
    }

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
            // A step of one clock unit is `lam` itself, and no step is 1,
            // by construction rather than by the platform's `pow`: every
            // libm met so far returns `x` for `pow(x, 1)` and none promises
            // it, and a stream pinned to the bit under `Lam` at unit steps
            // (`sgd.rs`'s libm-free digests) rested on it (review round 5,
            // A4). No bit moves where `pow` already returned `x`.
            Decay::Lam(l) if d_clock == 1.0 => l,
            Decay::Lam(_) if d_clock == 0.0 => 1.0,
            Decay::Lam(l) => l.powf(d_clock),
        }
    }
}

/// What a clock that goes back within a session means (docs/PLAN.md §3 and
/// task 120). Two policies: `"max"`, which took `gap_cap` as the step,
/// and `"zero"`, which took none, were removed on 2026-09-28 -- a cap is not
/// a step, and both absorbed a data bug into plausible, wrong output.
///
/// A spec does not name this or [`ClockCfg::min_backwards_jump`]: its one
/// knob, `restart_after_step_back`, builds both (task 144). Unset is
/// [`OnClockReset::Error`]; given, it is [`OnClockReset::ResetState`] with
/// `min_backwards_jump` at its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnClockReset {
    /// Refuse the row (the default, and a spec without
    /// `restart_after_step_back`). A backwards clock is usually a data bug
    /// -- a mis-sorted chunk, or rows from two streams interleaved -- and
    /// absorbing it gives plausible but wrong output.
    #[default]
    Error,
    /// The clock resets on purpose (a spec with `restart_after_step_back`):
    /// a step back by more than `min_backwards_jump` starts the model over,
    /// and one by no more than it is a late row, refused as [`Disorder`].
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
    /// under [`OnClockReset::Error`], which refuses every step back. A spec
    /// sets the two together through `restart_after_step_back`, this its
    /// value (task 144; [`OnClockReset`]).
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
    /// The row asked for more clock than `gap_cap` allows, so the delta
    /// handed to the models is the ceiling rather than the truth: its own
    /// raw delta did, or the total that the skipped rows before it fold into
    /// it did, even where no single step among them passed the cap
    /// ([`ClockState::advance`]; review 2026-09-12, S3).
    ///
    /// Decay copes with that by construction — it only ever forgets more —
    /// but anything **lagged by rows** does not: the row `ℓ` back is no
    /// longer `ℓ` rows *ago* in any useful sense once a weekend has passed
    /// between them. A caller that keeps such a ring clears it here
    /// ([`crate::OnlineModel::clear_lags`], docs/PLAN.md task 47). Per
    /// accepted row: the gap between it and the last accepted row is what
    /// breaks adjacency, however many skipped rows it spans.
    pub capped: bool,
    /// The time that passed since the previous accepted row, uncapped: the
    /// clock column's forward steps, skipped rows' included, and at a
    /// session change that restarts the clock, the session's gap. What a
    /// delay counts (docs/PLAN.md task 153): `gap_cap` and `session_gap`
    /// say how much a model forgets across a break, not how long it lasted,
    /// so `d_clock` runs slower than the clock across a capped gap and
    /// faster across a session gap longer than its step. `0` on the first
    /// row and at a reset; only meaningful when `accepted`. A double, for a
    /// caller to report: a delay is decided on [`Self::elapsed_stamp`],
    /// where these steps add up exactly.
    pub elapsed: f64,
    /// The row's place on the decayed clock, held exactly: `Some` for an
    /// accepted row of [`ClockState::advance_stamped`], `None` otherwise.
    pub stamp: Option<Stamp>,
    /// The row's place on the elapsed clock, held exactly (docs/PLAN.md task
    /// 176): `Some` for an accepted row of [`ClockState::advance_stamped`],
    /// `None` otherwise. The elapsed clock adds up every row's
    /// [`Self::elapsed`] step, a skipped row's included: in integer
    /// nanoseconds on a temporal clock ([`Stamp::Ns`]); on a number clock as
    /// the row's raw value beside the steps the elapsed clock does not count
    /// ([`Stamp::Raw`]) -- a session restart's step back, for which it counts
    /// the session's gap -- so two rows with no session restart between them
    /// differ by one subtraction of their raw values; without a clock column,
    /// the row's place in the stream. A reset starts it over.
    ///
    /// What a delay is decided on: a row held under `embargo` is learned
    /// once an accepted row's place is at least the embargo past its own
    /// ([`Stamp::cmp_span_ns`]). Counted down in doubles, the embargo drifted:
    /// two thousand steps of 1 ms left a sliver of `"2s"`, and every row was
    /// learned a row late.
    pub elapsed_stamp: Option<Stamp>,
}

/// An accepted row's place on the decayed clock, held exactly (docs/PLAN.md
/// task 175): what a model's window keys its snapshots by, and decides its
/// edge and its snapshot spacing from.
///
/// The decayed clock is the one a window measures a row's age on: each step
/// after `gap_cap` and any `session_gap`, with a skipped row's step folded
/// into the next accepted row (README, *A hard window*). Decay reads it as
/// `d_clock`, a double per row. Summed, those doubles drift: a thousand
/// steps of 1 ms come to 1.0000000000000007 s, so a row exactly one window
/// old fell outside the window on some rows and inside on others (review
/// CB1). A stamp holds the same clock with nothing summed in floating point
/// that a window's edge could hang on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Stamp {
    /// A temporal clock: the steps since the stream began, or last started
    /// over, each capped, summed in integer nanoseconds.
    Ns(i128),
    /// A float clock, or none: the row's raw clock value (its place in the
    /// stream, without a clock column), and the time the caps and session
    /// steps have removed before it. The decayed clock is the first less
    /// the second, and two rows with no capped gap or session change between
    /// them hold the same removed time, so they differ by one subtraction of
    /// their raw values: the subtraction the window operators decide an edge
    /// by (task 159, W3).
    Raw(
        #[serde(with = "crate::humanfloat::f64_or_tag")] f64,
        #[serde(with = "crate::humanfloat::f64_or_tag")] f64,
    ),
    /// An integer clock (docs/PLAN.md task 200): the decayed clock in the
    /// column's own units, as a whole number of them, exact, and the
    /// fraction beside it that a fractional `gap_cap` or `session_gap`
    /// left, a double kept between −1 and 1. The whole number is the row's
    /// raw value less the whole units the caps and session steps removed
    /// before it, so two rows with no capped gap or session change between
    /// them differ by one subtraction of their raw values, in integers, at
    /// any size: an epoch-nanosecond column near 1.8e18, where a double
    /// resolves 256, keeps its 1 ns steps.
    Int(i128, #[serde(with = "crate::humanfloat::f64_or_tag")] f64),
}

impl Stamp {
    /// How `self`'s decayed clock less `older`'s compares with `span` clock
    /// units, decided exactly:
    ///
    /// - two temporal stamps: their difference in integer nanoseconds,
    ///   through [`seconds_of_ns`], the two operations a duration's text
    ///   goes through. A span given as a duration is that duration's
    ///   seconds, so a difference of exactly one span compares equal to it,
    ///   and since [`seconds_of_ns`] never decreases, a longer difference
    ///   never compares below it. Up to `2^23` seconds, about 97 days, where
    ///   a double resolves a nanosecond, this is the integer comparison
    ///   against the span's nanoseconds;
    /// - two raw stamps: the difference of their raw values less the
    ///   difference of their removed times, which is exactly one subtraction
    ///   when the removed times are equal;
    /// - two integer stamps: the difference of their whole parts, in
    ///   integers, against the span less the difference of their fractions,
    ///   compared exactly ([`cmp_int_f64`]): where every `gap_cap` and
    ///   `session_gap` is whole the fractions are 0 and the integer
    ///   difference meets the span itself, so a difference of exactly the
    ///   span is equal to it at any size of clock (task 200). Across a break
    ///   a fractional parameter made, the fractions' difference is a double
    ///   below 2 in size, rounded there and never at the clock's magnitude.
    ///
    /// Stamps of two forms -- which no stream hands one model -- compare
    /// their decayed clocks as numbers. A comparison with no answer (a NaN
    /// from an overflow) is `Equal`, as the window operators read one.
    pub fn cmp_span(self, older: Stamp, span: f64) -> Ordering {
        self.cmp_span_ns(older, span, None)
    }

    /// [`Self::cmp_span`], for a caller that also holds the span's integer
    /// nanoseconds -- a window written as a duration, as the window
    /// operators keep one: two temporal stamps then compare as integers at
    /// any length, where through seconds a span past `2^23` seconds with a
    /// part finer than a double resolves there can tie with a difference a
    /// nanosecond longer. Every other pair, and a span without nanoseconds,
    /// is [`Self::cmp_span`]'s.
    pub fn cmp_span_ns(self, older: Stamp, span: f64, span_ns: Option<i128>) -> Ordering {
        let diff = match (self, older, span_ns) {
            (Stamp::Ns(a), Stamp::Ns(b), Some(w)) => return a.saturating_sub(b).cmp(&w),
            (Stamp::Ns(a), Stamp::Ns(b), None) => seconds_of_ns(a.saturating_sub(b)),
            (Stamp::Raw(a, ra), Stamp::Raw(b, rb), _) => (a - b) - (ra - rb),
            (Stamp::Int(a, fa), Stamp::Int(b, fb), _) => {
                return cmp_int_f64(a.saturating_sub(b), span - (fa - fb))
                    .unwrap_or(Ordering::Equal);
            }
            (a, b, _) => a.value() - b.value(),
        };
        diff.partial_cmp(&span).unwrap_or(Ordering::Equal)
    }

    /// The decayed clock as a number of clock units: seconds on a temporal
    /// clock. For a stamp of another form; an edge never reads it.
    fn value(self) -> f64 {
        match self {
            Stamp::Ns(n) => seconds_of_ns(n),
            Stamp::Raw(raw, removed) => raw - removed,
            Stamp::Int(whole, frac) => whole as f64 + frac,
        }
    }

    /// An integer stamp moved back by `d` clock units, a finite double:
    /// its whole part by `d`'s, exactly, and its fraction by the rest
    /// (`crate::since`'s start of a stream). `None` for any other stamp, or
    /// a `d` with no finite value.
    pub(crate) fn int_back_by(self, d: f64) -> Option<Stamp> {
        match (self, Ticks::of(d)) {
            (Stamp::Int(whole, frac), Some(back)) => {
                let t = Ticks::norm(whole, frac).minus(back);
                Some(Stamp::Int(t.whole, t.frac))
            }
            _ => None,
        }
    }
}

/// How an integer compares with a double, exactly: neither is rounded to
/// the other's type (docs/PLAN.md task 200). An integer clock's step is an
/// integer and its parameters are doubles, possibly fractional (`gap_cap =
/// 0.5`), and a step of `2^53 + 1` against a cap of `2^53` is past it,
/// where the step's double ties with the cap. `None` for a NaN, which
/// orders with nothing.
pub fn cmp_int_f64(i: i128, f: f64) -> Option<Ordering> {
    // 2^127: every double at or past it is past every i128, and every one
    // below −2^127 is below every i128.
    const EDGE: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    if f.is_nan() {
        return None;
    }
    if f >= EDGE {
        return Some(Ordering::Less);
    }
    if f < -EDGE {
        return Some(Ordering::Greater);
    }
    let floor = f.floor();
    // Exact: `floor` is a whole number inside i128's range.
    match i.cmp(&(floor as i128)) {
        Ordering::Equal if f > floor => Some(Ordering::Less),
        o => Some(o),
    }
}

/// An amount of an integer clock's time (docs/PLAN.md task 200): whole
/// units, exact, and the fraction a parameter with one added, a double
/// kept between −1 and 1 so that summing fractions rounds at their own
/// size, never at the clock's. Where every parameter is whole the fraction
/// is 0 and the amount is an integer, exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Ticks {
    whole: i128,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    frac: f64,
}

impl Ticks {
    const ZERO: Ticks = Ticks {
        whole: 0,
        frac: 0.0,
    };

    fn int(i: i128) -> Self {
        Ticks {
            whole: i,
            frac: 0.0,
        }
    }

    /// A finite double, exactly: its whole part and its fraction, which
    /// `f − trunc(f)` gives without rounding. `None` for an infinite or NaN
    /// value: no amount (an infinite cap is no cap).
    fn of(f: f64) -> Option<Self> {
        if !f.is_finite() {
            return None;
        }
        let whole = f.trunc();
        Some(Ticks {
            whole: whole as i128,
            frac: f - whole,
        })
    }

    /// `whole + frac` with the fraction's whole part moved into the whole.
    fn norm(whole: i128, frac: f64) -> Self {
        let carry = frac.trunc();
        Ticks {
            whole: whole.saturating_add(carry as i128),
            frac: frac - carry,
        }
    }

    fn plus(self, o: Self) -> Self {
        Self::norm(self.whole.saturating_add(o.whole), self.frac + o.frac)
    }

    fn minus(self, o: Self) -> Self {
        Self::norm(self.whole.saturating_sub(o.whole), self.frac - o.frac)
    }

    /// `self` against `o`: the whole parts' difference, in integers,
    /// against the fractions' ([`cmp_int_f64`]).
    fn cmp(self, o: Self) -> Ordering {
        cmp_int_f64(self.whole.saturating_sub(o.whole), o.frac - self.frac)
            .unwrap_or(Ordering::Equal)
    }

    /// The amount as a double, rounded once where the whole part is past
    /// what a double holds.
    fn to_f64(self) -> f64 {
        self.whole as f64 + self.frac
    }

    fn is_zero(&self) -> bool {
        self.whole == 0 && self.frac == 0.0
    }
}

impl From<f64> for Stamp {
    /// A clock value with nothing removed: a model's own clock, summed from
    /// the steps it is handed, is the stamp of a row whose caller hands it
    /// none (`crate::OnlineModel::stamp_next`), as every window was keyed
    /// before task 175.
    fn from(clock: f64) -> Self {
        Stamp::Raw(clock, 0.0)
    }
}

/// A temporal clock's `gap_cap` and `session_gap` in integer nanoseconds,
/// as a temporal spec gives them -- durations -- for
/// [`ClockState::advance_stamped`]. One left `None` is read from the
/// [`ClockCfg`]'s seconds ([`ns_of_seconds`]), which gives the duration
/// back exactly under about 97 days and at any whole number of seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExactCaps {
    pub gap_cap_ns: Option<i64>,
    pub session_gap_ns: Option<i64>,
}

/// Seconds as integer nanoseconds: the inverse of [`seconds_of_ns`] wherever
/// that is one-to-one -- below `2^23` seconds, about 97 days, where a double
/// resolves a nanosecond, and at any whole number of seconds -- and the
/// nearest nanosecond elsewhere. `None` for an infinite or NaN value: no
/// cap.
pub fn ns_of_seconds(s: f64) -> Option<i128> {
    if !s.is_finite() {
        return None;
    }
    let whole = s.trunc();
    // Exact: `s` and its whole part share their leading bits.
    let frac = s - whole;
    Some(
        (whole as i128)
            .saturating_mul(1_000_000_000)
            .saturating_add((frac * 1e9).round() as i128),
    )
}

/// The decayed clock held exactly between rows, beside the doubles
/// [`ClockState::advance`] hands out: what [`ClockState::advance_stamped`]
/// keeps (docs/PLAN.md task 175); and the elapsed clock held exactly beside
/// it (task 176). Each stamp form reads its own part.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Exact {
    /// A temporal clock: the decayed clock of the last accepted row, in
    /// integer nanoseconds.
    ns: i128,
    /// A temporal clock: skipped rows' steps, each capped, in integer
    /// nanoseconds, waiting for the next accepted row.
    skipped_ns: i128,
    /// A number clock or none: the time the caps and session steps have
    /// removed, through the last row.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    removed: f64,
    /// No clock column: the rows advanced, the next row's place.
    rows: u64,
    /// A temporal clock: the elapsed clock of the last row, skipped or not,
    /// in integer nanoseconds: every step uncapped.
    elapsed_ns: i128,
    /// A number clock or none: the raw steps the elapsed clock does not
    /// count, through the last row -- at a session change that restarts the
    /// clock, the step back less the session's gap, which it counts instead.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    elapsed_removed: f64,
    /// An integer clock: `removed` in the column's own units, exact where
    /// every parameter is whole (task 200). Written only while it holds
    /// time, so a float or temporal clock's state is the bytes it was.
    #[serde(default, skip_serializing_if = "Ticks::is_zero")]
    removed_int: Ticks,
    /// An integer clock: `elapsed_removed` in the column's own units.
    #[serde(default, skip_serializing_if = "Ticks::is_zero")]
    elapsed_removed_int: Ticks,
}

/// One row of an integer clock as [`ClockState`] decided it, exactly
/// (task 200): handed to [`Exact::step`] for the row's stamps.
#[derive(Debug, Clone, Copy)]
struct IntRow {
    /// The column's step, `None` on the first row.
    raw: Option<i128>,
    /// What the model sees of the step: the step itself, the cap, the
    /// session's gap or nothing.
    d: Ticks,
    /// The time that passed: the step forward, or the session's gap.
    elapsed: Ticks,
    /// The skipped rows' steps and this row's, before the cap: what an
    /// accepted row's fold is judged on.
    total: Ticks,
}

impl Exact {
    /// One row, after [`ClockState`] has decided its step `d` and its
    /// elapsed time `elapsed` (and `raw`, its clock's step, `None` on the
    /// first row): the same decisions, in integer nanoseconds on a temporal
    /// clock, and as removed time beside the raw value on a number clock or
    /// none. The stamp of an accepted row, its skipped predecessors' steps
    /// folded in under the cap as [`ClockState::advance`] folds them, and its
    /// place on the elapsed clock, every row's step counted whole; `None`
    /// for both on a skipped row. An integer clock's row (`int`) is the
    /// same decisions in the column's own units, exactly (task 200).
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        cfg: &ClockCfg,
        caps: &ExactCaps,
        now: Option<ClockValue>,
        prev: Option<ClockValue>,
        session_changed: bool,
        raw: Option<f64>,
        d: f64,
        elapsed: f64,
        pending: f64,
        int: Option<IntRow>,
        reset: bool,
        accept: bool,
    ) -> (Option<Stamp>, Option<Stamp>) {
        if reset {
            // The stamps start over with the model, and the elapsed clock
            // with them: a reset drops every row a delay holds.
            *self = Exact::default();
        }
        let place = self.rows as f64;
        self.rows = self.rows.saturating_add(1);
        if let (Some(ClockValue::I64(v)), Some(row)) = (now, int) {
            // An integer clock: as a number clock's, with the removed time
            // in the column's own units -- the step less what the model
            // sees of it, and the fold's total past the cap -- so a stretch
            // without a break removes exactly nothing and two of its rows
            // differ by one subtraction in integers.
            if let Some(r) = row.raw.filter(|_| !reset) {
                let r = Ticks::int(r);
                self.removed_int = self.removed_int.plus(r.minus(row.d));
                self.elapsed_removed_int = self.elapsed_removed_int.plus(r.minus(row.elapsed));
            }
            if !accept {
                return (None, None);
            }
            if let Some(cap) = Ticks::of(cfg.gap_cap)
                && row.total.cmp(cap) == Ordering::Greater
            {
                self.removed_int = self.removed_int.plus(row.total.minus(cap));
            }
            let at = Ticks::int(i128::from(v));
            let (s, e) = (
                at.minus(self.removed_int),
                at.minus(self.elapsed_removed_int),
            );
            return (
                Some(Stamp::Int(s.whole, s.frac)),
                Some(Stamp::Int(e.whole, e.frac)),
            );
        }
        if let Some(ClockValue::Ns(_)) = now {
            let cap = caps
                .gap_cap_ns
                .map(i128::from)
                .or_else(|| ns_of_seconds(cfg.gap_cap));
            let capped = |v: i128| cap.map_or(v, |c| v.min(c));
            let (d_ns, e_ns) = match (now, prev, raw) {
                _ if reset => (0, 0),
                (_, _, None) => (0, 0),
                (Some(ClockValue::Ns(c)), Some(ClockValue::Ns(p)), Some(_)) => {
                    let r = i128::from(c) - i128::from(p);
                    // The session's gap in nanoseconds, where a session
                    // change takes one.
                    let gap = match cfg.session_gap {
                        Some(SessionGap::Gap(g)) => Some(
                            caps.session_gap_ns
                                .map(i128::from)
                                .or_else(|| ns_of_seconds(g))
                                .unwrap_or(0)
                                .max(0),
                        ),
                        _ => None,
                    };
                    let d_ns = if session_changed {
                        match cfg.session_gap {
                            Some(SessionGap::Reset) => 0,
                            Some(SessionGap::Gap(_)) => capped(gap.unwrap_or(0)),
                            None => capped(r.max(0)),
                        }
                    } else if r < 0 {
                        0
                    } else {
                        capped(r)
                    };
                    // The time that passed, as `ClockState::step` counts it:
                    // the column's forward step; the session's gap across a
                    // session change that restarts the clock; nothing for a
                    // step back within a session.
                    let e_ns = if r >= 0 {
                        r
                    } else if session_changed {
                        gap.unwrap_or(0)
                    } else {
                        0
                    };
                    (d_ns, e_ns)
                }
                // A step from a value of the other form, which only a direct
                // caller hands: the steps decided in seconds.
                _ => (
                    ns_of_seconds(d).unwrap_or(0),
                    ns_of_seconds(elapsed).unwrap_or(0),
                ),
            };
            // Every row's time passes, a skipped row's too.
            self.elapsed_ns = self.elapsed_ns.saturating_add(e_ns);
            if !accept {
                self.skipped_ns = self.skipped_ns.saturating_add(d_ns);
                return (None, None);
            }
            let total = self.skipped_ns.saturating_add(d_ns);
            self.skipped_ns = 0;
            self.ns = self.ns.saturating_add(capped(total));
            return (Some(Stamp::Ns(self.ns)), Some(Stamp::Ns(self.elapsed_ns)));
        }
        // A number clock, or none: the row's step less what the model sees of
        // it is removed time, and nothing else is -- an uncapped step is its
        // own `d`, so a stretch without a break removes exactly nothing. The
        // elapsed clock removes only what it does not count of the step: a
        // forward step is its own `elapsed`, and removes nothing, one that
        // overflows to infinity included (`inf − inf` would be NaN).
        if let Some(raw) = raw.filter(|_| !reset) {
            self.removed += raw - d;
            if raw != elapsed {
                self.elapsed_removed += raw - elapsed;
            }
        }
        if !accept {
            return (None, None);
        }
        // The cap on the folded total, as `advance` applies it.
        let total = pending + d;
        if total > cfg.gap_cap {
            self.removed += total - cfg.gap_cap;
        }
        let value = match now {
            Some(c) => c.seconds(),
            None => place,
        };
        (
            Some(Stamp::Raw(value, self.removed)),
            Some(Stamp::Raw(value, self.elapsed_removed)),
        )
    }
}

/// One row's clock value, in the form the source had it: a float, an
/// integer, or a temporal clock's nanoseconds since the Unix epoch. A
/// stream keeps its previous row's value in that form, and the delta
/// between two `Ns` values, or two `I64` ones, is taken in integers before
/// it becomes a double, so a nanosecond timestamp's gaps are exact whatever
/// the stream's age. Read as a double of seconds from any origin, a clock
/// resolves only 2^-52 of the time since that origin: under a nanosecond
/// for six weeks, four nanoseconds after a year (docs/PLAN.md task 88); an
/// integer column of epoch nanoseconds read as a double resolves 256 of
/// them in 2026 (task 200).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClockValue {
    /// A float clock, in the column's own units.
    F64(f64),
    /// A temporal clock, in nanoseconds since the Unix epoch.
    Ns(i64),
    /// An integer clock, in the column's own units (task 200): any integer
    /// column up to 64 bits wide, read without passing through a double.
    I64(i64),
}

impl ClockValue {
    /// The value as a number a caller can report: a float or integer clock
    /// as it is (an integer past `2^53` rounded to a double), a temporal
    /// one as seconds since the Unix epoch, which a double resolves to
    /// about a quarter of a microsecond at today's dates. For reporting;
    /// the delta a model sees never goes through this.
    pub fn seconds(self) -> f64 {
        match self {
            Self::F64(v) => v,
            Self::Ns(ns) => seconds_of_ns(i128::from(ns)),
            Self::I64(v) => v as f64,
        }
    }

    /// `self` minus `prev`, in clock units: for two temporal values, the
    /// seconds between them, taken in integer nanoseconds and rounded once;
    /// for two integer ones, their difference in integers, rounded once.
    fn delta(self, prev: Self) -> f64 {
        match (self, prev) {
            (Self::Ns(c), Self::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
            (Self::F64(c), Self::F64(p)) => c - p,
            (Self::I64(c), Self::I64(p)) => (i128::from(c) - i128::from(p)) as f64,
            // A stream's clock keeps one form for its life -- the bank
            // refuses a chunk in another -- so this arm is a direct
            // caller's, and it reads both as the numbers they report.
            (c, p) => c.seconds() - p.seconds(),
        }
    }

    /// `self` minus `prev` in integers, for two integer values: what every
    /// decision on an integer clock's step is taken on (task 200). `None`
    /// for any other pair.
    pub fn int_step(self, prev: Self) -> Option<i128> {
        match (self, prev) {
            (Self::I64(c), Self::I64(p)) => Some(i128::from(c) - i128::from(p)),
            _ => None,
        }
    }

    /// Whether `self` is before `prev`: exact between two temporal values,
    /// two integer ones, and an integer and a float.
    pub fn is_before(self, prev: Self) -> bool {
        match (self, prev) {
            (Self::Ns(c), Self::Ns(p)) | (Self::I64(c), Self::I64(p)) => c < p,
            (Self::I64(c), Self::F64(p)) => cmp_int_f64(i128::from(c), p) == Some(Ordering::Less),
            (Self::F64(c), Self::I64(p)) => {
                cmp_int_f64(i128::from(p), c) == Some(Ordering::Greater)
            }
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
    /// The decayed clock held exactly, for a caller that stamps its rows
    /// ([`Self::advance_stamped`], docs/PLAN.md task 175). `None` until the
    /// first stamped row, and for good for a caller that never asks -- the
    /// window operators -- whose state writes the bytes it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exact: Option<Exact>,
    /// An integer clock's skipped rows' deltas, in the column's own units,
    /// exact where every parameter is whole: `pending` for an integer
    /// clock, whose fold is judged against `gap_cap` exactly (task 200).
    /// Written only while it holds time, so a float or temporal clock's
    /// state is the bytes it was.
    #[serde(default, skip_serializing_if = "Ticks::is_zero")]
    pending_int: Ticks,
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
        self.step(cfg, None, clock, session, accept, false)
    }

    /// [`Self::advance`], with each accepted row's [`Stamp`] in
    /// [`ClockAdvance::stamp`]: the decayed clock `d_clock` adds up to, held
    /// exactly (docs/PLAN.md task 175). On a temporal clock it is the sum of
    /// the steps in integer nanoseconds, each capped at `caps`' `gap_cap`
    /// and a session change's step `caps`' `session_gap`, a skipped row's
    /// step folded into the next accepted one under the cap. On a number
    /// clock, or none, it is the row's raw value, or its place in the
    /// stream, beside the time the caps and session steps removed. A reset
    /// starts it over. Beside it, [`ClockAdvance::elapsed_stamp`] holds the
    /// elapsed clock the same way, every step uncapped (docs/PLAN.md task
    /// 176).
    ///
    /// A stream stamps every row it advances, or none: a row advanced by
    /// [`Self::advance`] leaves the exact clock where it was, so stamps after
    /// it would miss its step.
    ///
    /// ```
    /// use online_core::{ClockCfg, ClockState, ClockValue, ExactCaps, Stamp};
    ///
    /// let cfg = ClockCfg { gap_cap: 10.0, ..ClockCfg::default() };
    /// let caps = ExactCaps { gap_cap_ns: Some(10_000_000_000), session_gap_ns: None };
    /// let mut clock = ClockState::new();
    /// let mut stamp = None;
    /// for ms in 0..=1000 {
    ///     let at = Some(ClockValue::Ns(1_704_067_200_000_000_000 + ms * 1_000_000));
    ///     stamp = clock.advance_stamped(&cfg, &caps, at, None, true).stamp;
    /// }
    /// // A thousand steps of a millisecond are a second, exactly: summed as
    /// // doubles they are 1.0000000000000007.
    /// assert_eq!(stamp, Some(Stamp::Ns(1_000_000_000)));
    /// ```
    pub fn advance_stamped(
        &mut self,
        cfg: &ClockCfg,
        caps: &ExactCaps,
        clock: Option<ClockValue>,
        session: Option<u64>,
        accept: bool,
    ) -> ClockAdvance {
        self.step(cfg, Some(caps), clock, session, accept, false)
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
        self.step(cfg, None, clock, session, true, true)
    }

    fn step(
        &mut self,
        cfg: &ClockCfg,
        caps: Option<&ExactCaps>,
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
        // An integer clock's step in integers (task 200): every decision on
        // the step below is taken on it, exactly, against parameters that
        // are doubles, where `raw` is its double, for the model. An epoch-
        // nanosecond column near 1.8e18 steps by 1 where its doubles tie.
        let int_raw = match (clock, self.prev_clock) {
            (Some(c), Some(p)) => c.int_step(p),
            _ => None,
        };
        let integer = matches!(clock, Some(ClockValue::I64(_)))
            && matches!(self.prev_clock, None | Some(ClockValue::I64(_)));
        // `raw > x`, on the integers where the clock is one.
        let past = |raw: f64, x: f64| match int_raw {
            Some(i) => cmp_int_f64(i, x) == Some(Ordering::Greater),
            None => raw > x,
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
        // Whether the row's `d` is its own step, whole: what an integer
        // clock holds as the integers' difference.
        let mut whole = false;
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
                            capped = past(raw, cfg.gap_cap);
                            whole = raw >= 0.0 && !capped;
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
                    let late = match int_raw {
                        Some(i) => matches!(
                            cmp_int_f64(-i, cfg.min_backwards_jump),
                            Some(Ordering::Less | Ordering::Equal)
                        ),
                        None => back <= cfg.min_backwards_jump,
                    };
                    if scoring {
                        0.0
                    } else {
                        match cfg.on_clock_reset {
                            OnClockReset::Error => {
                                backwards = Some(raw);
                                0.0
                            }
                            OnClockReset::ResetState if late => {
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
                    capped = past(raw, cfg.gap_cap);
                    whole = !capped;
                    raw.min(cfg.gap_cap)
                }
            }
        };

        if reset {
            self.pending = 0.0;
            self.pending_int = Ticks::ZERO;
            self.skipped_elapsed = 0.0;
            d = 0.0;
            elapsed = 0.0;
        }
        // The same row in the integer clock's own units, exactly: the step
        // the model sees -- the integers' difference where it is the step
        // whole, else the cap or the session's gap, whose doubles split
        // exactly into whole units and a fraction -- the time that passed,
        // and the fold's total, which an accepted row is judged on.
        let int_row = integer.then(|| {
            let d = match int_raw {
                Some(i) if whole && !reset => Ticks::int(i),
                _ => Ticks::of(d).unwrap_or(Ticks::ZERO),
            };
            IntRow {
                raw: int_raw,
                d,
                elapsed: match int_raw {
                    Some(i) if i >= 0 && !reset => Ticks::int(i),
                    _ => Ticks::of(elapsed).unwrap_or(Ticks::ZERO),
                },
                total: self
                    .pending_int
                    .plus(Ticks::of(self.pending).unwrap_or(Ticks::ZERO))
                    .plus(d),
            }
        });
        // The same step held exactly, for a caller that stamps its rows
        // (task 175), and the time that passed held exactly beside it (task
        // 176): read before this row's clock replaces the last.
        let (stamp, elapsed_stamp) = caps.map_or((None, None), |caps| {
            self.exact.get_or_insert_with(Exact::default).step(
                cfg,
                caps,
                clock,
                self.prev_clock,
                session_changed,
                raw,
                d,
                elapsed,
                self.pending,
                int_row,
                reset,
                accept,
            )
        });
        self.prev_clock = clock;
        self.prev_session = session;
        self.started = true;

        if accept {
            // The skipped rows' time is carried into this row's, and the
            // ceiling holds for the total: `gap_cap` is the most a model
            // sees between two rows it learns from. Ten skipped rows 100
            // apart under a cap of 60 handed the next one 660 (review
            // 2026-09-12, S3). A total over the cap is a capped gap. On an
            // integer clock the total is the integers' and judged on them.
            let (d_clock, over) = match int_row {
                Some(row) => {
                    let over = Ticks::of(cfg.gap_cap)
                        .is_some_and(|cap| row.total.cmp(cap) == Ordering::Greater);
                    let d = if over {
                        cfg.gap_cap
                    } else {
                        row.total.to_f64().min(cfg.gap_cap)
                    };
                    (d, over)
                }
                None => {
                    let mut total = self.pending + d;
                    // Integer steps a direct caller's stream of mixed forms
                    // left waiting; never in a bank's stream.
                    if !self.pending_int.is_zero() {
                        total += self.pending_int.to_f64();
                    }
                    (total.min(cfg.gap_cap), total > cfg.gap_cap)
                }
            };
            self.pending = 0.0;
            self.pending_int = Ticks::ZERO;
            let elapsed = self.skipped_elapsed + elapsed;
            self.skipped_elapsed = 0.0;
            ClockAdvance {
                d_clock,
                reset,
                accepted: true,
                backwards,
                disorder,
                session_changed,
                capped: capped || over,
                elapsed,
                stamp,
                elapsed_stamp,
            }
        } else {
            match int_row {
                Some(row) => self.pending_int = self.pending_int.plus(row.d),
                None => self.pending += d,
            }
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
                stamp,
                elapsed_stamp,
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

    const T0: i64 = 1_704_067_200_000_000_000;

    /// One stamped row of a temporal stream at `ns` past `T0`.
    fn stamp_at(
        c: &mut ClockState,
        cfg: &ClockCfg,
        caps: &ExactCaps,
        ns: i64,
        session: Option<u64>,
        accept: bool,
    ) -> ClockAdvance {
        c.advance_stamped(cfg, caps, Some(ClockValue::Ns(T0 + ns)), session, accept)
    }

    /// Task 175 (review CB1): on a temporal clock a row's stamp is its
    /// decayed clock in integer nanoseconds, the oracle being the steps
    /// written out in integers: a thousand steps of 1 ms are a second exactly
    /// where their doubles sum to 1.0000000000000007; a gap past `gap_cap`
    /// adds the cap's own nanoseconds, a session change `session_gap`'s, both
    /// given as durations a double cannot hold (a hundred days and one
    /// nanosecond); skipped rows' steps fold into the next accepted row
    /// under the cap; a reset starts the stamps over; and only an accepted
    /// row of a stamped advance has one.
    #[test]
    fn a_temporal_stamp_is_the_decayed_clock_in_integer_nanoseconds() {
        let cap_ns: i64 = 100 * 86_400 * 1_000_000_000 + 1;
        let gap_ns: i64 = 3 * 86_400 * 1_000_000_000 + 7;
        let cfg = ClockCfg {
            gap_cap: seconds_of_ns(i128::from(cap_ns)),
            session_gap: Some(SessionGap::Gap(seconds_of_ns(i128::from(gap_ns)))),
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 60.0,
        };
        assert_ne!(ns_of_seconds(cfg.gap_cap), Some(i128::from(cap_ns)));
        let caps = ExactCaps {
            gap_cap_ns: Some(cap_ns),
            session_gap_ns: Some(gap_ns),
        };
        let mut c = ClockState::new();
        let mut summed = 0.0;
        let mut adv = stamp_at(&mut c, &cfg, &caps, 0, Some(1), true);
        assert_eq!(adv.stamp, Some(Stamp::Ns(0)), "the first row is the start");
        for ms in 1..=1000 {
            adv = stamp_at(&mut c, &cfg, &caps, ms * 1_000_000, Some(1), true);
            summed += adv.d_clock;
        }
        assert_eq!(adv.stamp, Some(Stamp::Ns(1_000_000_000)));
        assert!(summed > 1.0, "the doubles drift: {summed}");
        // A gap of 200 days: the cap, to the nanosecond.
        let mut at = 1_000_000_000 + 200 * 86_400 * 1_000_000_000;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(1), true);
        let mut want = 1_000_000_000 + i128::from(cap_ns);
        assert_eq!(adv.stamp, Some(Stamp::Ns(want)));
        // A session change, a millisecond on: the session's gap.
        at += 1_000_000;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(2), true);
        want += i128::from(gap_ns);
        assert_eq!(adv.stamp, Some(Stamp::Ns(want)));
        // Three skipped rows 40 days apart, then an accepted row a
        // nanosecond on: no step is past the cap, their total is, and the
        // accepted row's stamp moves by the cap.
        for _ in 0..3 {
            at += 40 * 86_400 * 1_000_000_000;
            adv = stamp_at(&mut c, &cfg, &caps, at, Some(2), false);
            assert_eq!(adv.stamp, None, "a skipped row has no stamp");
        }
        at += 1;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(2), true);
        want += i128::from(cap_ns);
        assert_eq!(adv.stamp, Some(Stamp::Ns(want)));
        // Two skipped rows a millisecond apart fold into the next whole.
        for _ in 0..2 {
            at += 1_000_000;
            stamp_at(&mut c, &cfg, &caps, at, Some(2), false);
        }
        at += 1_000_000;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(2), true);
        want += 3_000_000;
        assert_eq!(adv.stamp, Some(Stamp::Ns(want)));
        // A step back past `min_backwards_jump` restarts: the stamps start
        // over with the model.
        adv = stamp_at(&mut c, &cfg, &caps, at - 3_600_000_000_000, Some(2), true);
        assert!(adv.reset);
        assert_eq!(adv.stamp, Some(Stamp::Ns(0)));
        adv = stamp_at(&mut c, &cfg, &caps, at - 3_599_999_999_999, Some(2), true);
        assert_eq!(adv.stamp, Some(Stamp::Ns(1)));
        // An unstamped advance hands out none.
        let plain = c.advance(&cfg, Some(ClockValue::Ns(T0 + at)), Some(2), true);
        assert_eq!(plain.stamp, None);
    }

    /// Without the caps in nanoseconds a temporal stamp reads them from the
    /// configuration's seconds, which give a duration back exactly under
    /// about 97 days and at any whole number of seconds.
    #[test]
    fn a_temporal_stamp_reads_a_cap_it_is_not_handed_from_the_seconds() {
        let cfg = ClockCfg {
            gap_cap: 1.5,
            ..ClockCfg::default()
        };
        let mut c = ClockState::new();
        let none = ExactCaps::default();
        stamp_at(&mut c, &cfg, &none, 0, None, true);
        let adv = stamp_at(&mut c, &cfg, &none, 3_600_000_000_000, None, true);
        assert_eq!(adv.stamp, Some(Stamp::Ns(1_500_000_000)));
        // No cap at all: the step whole.
        let mut c = ClockState::new();
        let free = ClockCfg::default();
        stamp_at(&mut c, &free, &none, 0, None, true);
        let adv = stamp_at(&mut c, &free, &none, 3_600_000_000_000, None, true);
        assert_eq!(adv.stamp, Some(Stamp::Ns(3_600_000_000_000)));
    }

    /// On a number clock a stamp is the row's raw value beside the time the
    /// caps and session steps removed: nothing within a stretch, so two rows
    /// of it differ by one subtraction of their raw values (`0.8 − 0.5` is
    /// `0.30000000000000004`, past a window of `0.3`, where the steps of a
    /// tenth summed say `0.3`); across a capped gap, the raw step less the
    /// cap; at a session change, the raw step less the session's gap; and
    /// for a run of skipped rows, the folded total less the cap.
    #[test]
    fn a_number_clocks_stamp_is_its_raw_value_and_the_time_removed() {
        let cfg = ClockCfg {
            gap_cap: 5.0,
            session_gap: Some(SessionGap::Gap(2.0)),
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row = |t: f64, s: u64, accept: bool| {
            c.advance_stamped(&cfg, &caps, Some(ClockValue::F64(t)), Some(s), accept)
                .stamp
        };
        let mut stamps = Vec::new();
        for i in 0..10 {
            stamps.push(row(f64::from(i) / 10.0, 1, true).unwrap());
        }
        assert!(
            stamps
                .iter()
                .all(|s| matches!(s, Stamp::Raw(_, r) if *r == 0.0))
        );
        assert_eq!(stamps[8], Stamp::Raw(0.8, 0.0));
        assert_eq!(stamps[8].cmp_span(stamps[5], 0.3), Ordering::Greater);
        assert_eq!(stamps[3].cmp_span(stamps[0], 0.3), Ordering::Equal);
        // A gap of 100 past the cap of 5: 95 removed, the decayed clock 5 on.
        let after_gap = row(100.9, 1, true).unwrap();
        assert_eq!(after_gap, Stamp::Raw(100.9, (100.9 - 0.9) - 5.0));
        assert_eq!(after_gap, Stamp::Raw(100.9, 95.0));
        // A session change 1 on: the session's gap of 2 is the step.
        let session = row(101.9, 2, true).unwrap();
        assert_eq!(session, Stamp::Raw(101.9, 95.0 + (1.0 - 2.0)));
        // Two skipped rows 3 apart, then one 1 on: 7 folded, under the cap
        // of 5.
        assert_eq!(row(104.9, 2, false), None);
        assert_eq!(row(107.9, 2, false), None);
        let folded = row(108.9, 2, true).unwrap();
        assert_eq!(folded, Stamp::Raw(108.9, 94.0 + 2.0));
        // The decayed clock between the first row and the last: 0.9, then 5,
        // 2 and 5 -- the raw difference less the removed time, which across
        // a capped gap is a difference of rounded sums, exact only within a
        // stretch.
        let span = 0.9 + 5.0 + 2.0 + 5.0;
        assert_eq!(folded.cmp_span(stamps[0], span - 1e-12), Ordering::Greater);
        assert_eq!(folded.cmp_span(stamps[0], span + 1e-12), Ordering::Less);
    }

    /// Without a clock column a stamp is the row's place in the stream:
    /// the row count, exact, skipped rows counted; a session's gap of half a
    /// row is time removed.
    #[test]
    fn a_row_count_stamp_is_the_rows_place() {
        let cfg = ClockCfg {
            session_gap: Some(SessionGap::Gap(0.5)),
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row = |s: u64, accept: bool| c.advance_stamped(&cfg, &caps, None, Some(s), accept);
        assert_eq!(row(1, true).stamp, Some(Stamp::Raw(0.0, 0.0)));
        assert_eq!(row(1, true).stamp, Some(Stamp::Raw(1.0, 0.0)));
        assert_eq!(row(1, false).stamp, None);
        let adv = row(1, true);
        assert_eq!((adv.d_clock, adv.stamp), (2.0, Some(Stamp::Raw(3.0, 0.0))));
        let adv = row(2, true);
        assert_eq!((adv.d_clock, adv.stamp), (0.5, Some(Stamp::Raw(4.0, 0.5))));
    }

    /// A session that starts the model over starts its stamps over too, on
    /// every form of clock.
    #[test]
    fn a_reset_starts_the_stamps_over() {
        let cfg = ClockCfg {
            session_gap: Some(SessionGap::Reset),
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        for clock in [
            [Some(ClockValue::Ns(T0)), Some(ClockValue::Ns(T0 + 5))],
            [Some(ClockValue::F64(3.0)), Some(ClockValue::F64(8.5))],
            [Some(ClockValue::I64(T0)), Some(ClockValue::I64(T0 + 1))],
            [None, None],
        ] {
            let mut c = ClockState::new();
            c.advance_stamped(&cfg, &caps, clock[0], Some(1), true);
            c.advance_stamped(&cfg, &caps, clock[0], Some(1), true);
            let adv = c.advance_stamped(&cfg, &caps, clock[1], Some(2), true);
            assert!(adv.reset);
            let start = match clock[1] {
                Some(ClockValue::Ns(_)) => Stamp::Ns(0),
                Some(ClockValue::F64(v)) => Stamp::Raw(v, 0.0),
                Some(ClockValue::I64(v)) => Stamp::Int(i128::from(v), 0.0),
                None => Stamp::Raw(0.0, 0.0),
            };
            assert_eq!(adv.stamp, Some(start), "{clock:?}");
            assert_eq!(adv.elapsed_stamp, Some(start), "{clock:?}: elapsed");
        }
    }

    /// A row's stamp is the last accepted row's moved on by exactly the
    /// row's own `d_clock`, on either form of clock: what a stamp is
    /// (task 175). That holds on a step back the policy refuses too -- any
    /// step back under `"error"`, a late row under `"reset_state"` --
    /// whose `d_clock` is 0, so its stamp is the last one, and the row after
    /// it moves on from there by its own step. A caller that names the
    /// refused row and goes on stepping the clock meets no stamp behind the
    /// ones before it. The oracle is the steps written out: seconds 0, 10,
    /// 5 (refused), 12 and 12 are stamps 0, 10, 10, 17 and 17.
    #[test]
    fn a_refused_step_back_leaves_the_stamp_where_it_was() {
        let at = [0i64, 10, 5, 12, 12];
        let want = [0i128, 10, 10, 17, 17];
        for policy in [OnClockReset::Error, OnClockReset::ResetState] {
            let cfg = ClockCfg {
                gap_cap: 60.0,
                on_clock_reset: policy,
                session_gap: None,
                min_backwards_jump: 30.0,
            };
            let caps = ExactCaps {
                gap_cap_ns: Some(60_000_000_000),
                session_gap_ns: None,
            };
            let mut ns = ClockState::new();
            let mut num = ClockState::new();
            let mut last: Option<(Stamp, Stamp)> = None;
            for (i, (&s, &w)) in at.iter().zip(&want).enumerate() {
                let case = format!("{policy:?}, row {i}");
                let a = stamp_at(&mut ns, &cfg, &caps, s * 1_000_000_000, None, true);
                let b =
                    num.advance_stamped(&cfg, &caps, Some(ClockValue::F64(s as f64)), None, true);
                assert_eq!(a.backwards.is_some(), i == 2, "{case}");
                assert_eq!(b.backwards.is_some(), i == 2, "{case}: number");
                assert_eq!(a.d_clock, b.d_clock, "{case}");
                let (sa, sb) = (a.stamp.unwrap(), b.stamp.unwrap());
                assert_eq!(sa, Stamp::Ns(w * 1_000_000_000), "{case}");
                if let Some((pa, pb)) = last {
                    assert_eq!(sa.cmp_span(pa, a.d_clock), Ordering::Equal, "{case}");
                    assert_eq!(sb.cmp_span(pb, b.d_clock), Ordering::Equal, "{case}");
                }
                assert_eq!(
                    sb.cmp_span(Stamp::Raw(0.0, 0.0), w as f64),
                    Ordering::Equal,
                    "{case}: number"
                );
                last = Some((sa, sb));
            }
        }
    }

    /// Task 176: on a temporal clock a row's place on the elapsed clock is
    /// every step since the stream began, uncapped, summed in integer
    /// nanoseconds, the oracle being the steps written out in integers: two
    /// thousand steps of 1 ms are two seconds exactly, where their doubles
    /// do not sum to 2; a gap past `gap_cap` counts whole, where the decayed
    /// stamp adds the cap; a session change with the clock running on
    /// counts its step, not `session_gap`; one with the clock an hour back
    /// counts `session_gap`'s own nanoseconds (a hundred days and one, which
    /// no double holds); skipped rows' steps count whole; a reset starts it
    /// over; and only an accepted row of a stamped advance has one.
    #[test]
    fn a_temporal_elapsed_stamp_counts_every_step_whole_in_integer_nanoseconds() {
        let gap_ns: i64 = 100 * 86_400 * 1_000_000_000 + 1;
        let cfg = ClockCfg {
            gap_cap: 1.0,
            session_gap: Some(SessionGap::Gap(seconds_of_ns(i128::from(gap_ns)))),
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 60.0,
        };
        assert_ne!(
            ns_of_seconds(seconds_of_ns(i128::from(gap_ns))),
            Some(i128::from(gap_ns))
        );
        let caps = ExactCaps {
            gap_cap_ns: Some(1_000_000_000),
            session_gap_ns: Some(gap_ns),
        };
        let mut c = ClockState::new();
        let mut adv = stamp_at(&mut c, &cfg, &caps, 0, Some(1), true);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(0)), "the first row");
        let mut summed = 0.0;
        for ms in 1..=2000 {
            adv = stamp_at(&mut c, &cfg, &caps, ms * 1_000_000, Some(1), true);
            summed += adv.elapsed;
        }
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(2_000_000_000)));
        assert_ne!(summed, 2.0, "the doubles drift");
        // A gap of 200 days: whole, where the decayed stamp adds the cap.
        let day: i64 = 86_400 * 1_000_000_000;
        let mut at = 2_000_000_000 + 200 * day;
        let before = adv.stamp.unwrap();
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(1), true);
        let mut want = i128::from(at);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(want)));
        assert_eq!(
            adv.stamp
                .unwrap()
                .cmp_span_ns(before, 1.0, Some(1_000_000_000)),
            Ordering::Equal
        );
        // A session change a millisecond on, the clock running on: the step.
        at += 1_000_000;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(2), true);
        want += 1_000_000;
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(want)));
        // One at the same instant: no time passed, whatever the session's
        // gap makes the model forget.
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(4), true);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(want)));
        // A session change with the clock an hour back: the session's gap.
        at -= 3_600_000_000_000;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(3), true);
        want += i128::from(gap_ns);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(want)));
        // Three skipped rows 40 days apart, then an accepted row a
        // nanosecond on: every step whole.
        for _ in 0..3 {
            at += 40 * day;
            adv = stamp_at(&mut c, &cfg, &caps, at, Some(3), false);
            assert_eq!(adv.elapsed_stamp, None, "a skipped row has no place");
        }
        at += 1;
        adv = stamp_at(&mut c, &cfg, &caps, at, Some(3), true);
        want += i128::from(120 * day + 1);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(want)));
        // A step back past `min_backwards_jump` restarts it.
        adv = stamp_at(&mut c, &cfg, &caps, at - 3_600_000_000_000, Some(3), true);
        assert!(adv.reset);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(0)));
        adv = stamp_at(&mut c, &cfg, &caps, at - 3_599_999_999_999, Some(3), true);
        assert_eq!(adv.elapsed_stamp, Some(Stamp::Ns(1)));
        // An unstamped advance hands out none.
        let plain = c.advance(&cfg, Some(ClockValue::Ns(T0 + at)), Some(3), true);
        assert_eq!(plain.elapsed_stamp, None);
    }

    /// On a number clock a row's place on the elapsed clock is its raw value
    /// beside the steps the elapsed clock does not count: none within a
    /// stretch, across a capped gap or at a session change with the clock
    /// running on, so any two such rows differ by one subtraction of their
    /// raw values; at a session change that restarts the clock, the step
    /// back less the session's gap, which it counts instead.
    #[test]
    fn a_number_clocks_elapsed_stamp_is_its_raw_value_and_what_it_does_not_count() {
        let cfg = ClockCfg {
            gap_cap: 5.0,
            session_gap: Some(SessionGap::Gap(2.0)),
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row = |t: f64, s: u64, accept: bool| {
            c.advance_stamped(&cfg, &caps, Some(ClockValue::F64(t)), Some(s), accept)
                .elapsed_stamp
        };
        let first = row(0.0, 1, true).unwrap();
        for i in 1..10 {
            assert_eq!(
                row(f64::from(i) / 10.0, 1, true),
                Some(Stamp::Raw(f64::from(i) / 10.0, 0.0))
            );
        }
        // A gap of 100 past the cap, a session change one on with the clock
        // running on, and two skipped rows: nothing the elapsed clock drops.
        assert_eq!(row(100.9, 1, true), Some(Stamp::Raw(100.9, 0.0)));
        assert_eq!(row(101.9, 2, true), Some(Stamp::Raw(101.9, 0.0)));
        assert_eq!(row(104.9, 2, false), None);
        assert_eq!(row(107.9, 2, false), None);
        let last = row(108.9, 2, true).unwrap();
        assert_eq!(last, Stamp::Raw(108.9, 0.0));
        assert_eq!(last.cmp_span(first, 108.9), Ordering::Equal);
        // The clock restarts at a session change: the session's gap counts.
        let restart = row(3.0, 3, true).unwrap();
        assert_eq!(restart, Stamp::Raw(3.0, (3.0 - 108.9) - 2.0));
        assert_eq!(restart.cmp_span(last, 2.0 - 1e-9), Ordering::Greater);
        assert_eq!(restart.cmp_span(last, 2.0 + 1e-9), Ordering::Less);
        // Within the new stretch, one subtraction again.
        let on = row(3.3, 3, true).unwrap();
        assert_eq!(on.cmp_span(restart, 3.3 - 3.0), Ordering::Equal);
        // A step that overflows to infinity passed more than any embargo,
        // and leaves the places after it whole.
        let mut c = ClockState::new();
        let mut row = |t: f64| {
            c.advance_stamped(&cfg, &caps, Some(ClockValue::F64(t)), None, true)
                .elapsed_stamp
                .unwrap()
        };
        let low = row(-1e308);
        let high = row(1e308);
        assert_eq!(high, Stamp::Raw(1e308, 0.0));
        assert_eq!(high.cmp_span(low, 1e300), Ordering::Greater);
        assert_eq!(row(1e308).cmp_span(high, 0.0), Ordering::Equal);
    }

    /// Without a clock column a row's place on the elapsed clock is its
    /// place in the stream, skipped rows counted, a session change included:
    /// a session's gap of half a row is what a model forgets, not time that
    /// passed.
    #[test]
    fn a_row_count_elapsed_stamp_is_the_rows_place() {
        let cfg = ClockCfg {
            session_gap: Some(SessionGap::Gap(0.5)),
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row = |s: u64, accept: bool| {
            c.advance_stamped(&cfg, &caps, None, Some(s), accept)
                .elapsed_stamp
        };
        assert_eq!(row(1, true), Some(Stamp::Raw(0.0, 0.0)));
        assert_eq!(row(1, true), Some(Stamp::Raw(1.0, 0.0)));
        assert_eq!(row(1, false), None);
        assert_eq!(row(1, true), Some(Stamp::Raw(3.0, 0.0)));
        assert_eq!(row(2, true), Some(Stamp::Raw(4.0, 0.0)));
    }

    /// The comparison of two stamps against a span: two temporal stamps by
    /// their nanoseconds, so a difference of exactly a span given as a
    /// duration is equal to it and one nanosecond more is past it; two raw
    /// stamps by one subtraction when nothing was removed between them;
    /// stamps of two forms by their decayed clocks as numbers, a raw stamp's
    /// being its raw value less the time removed before it.
    #[test]
    fn two_stamps_compare_exactly_against_a_span() {
        let second = seconds_of_ns(1_000_000_000);
        let (a, b) = (Stamp::Ns(7), Stamp::Ns(7 + 1_000_000_000));
        assert_eq!(b.cmp_span(a, second), Ordering::Equal);
        assert_eq!(
            Stamp::Ns(8 + 1_000_000_000).cmp_span(a, second),
            Ordering::Greater
        );
        assert_eq!(
            Stamp::Ns(6 + 1_000_000_000).cmp_span(a, second),
            Ordering::Less
        );
        // A span of 1.5 ms, which no double holds exactly, given as a
        // duration's seconds.
        let span = seconds_of_ns(1_500_000);
        assert_eq!(Stamp::Ns(1_500_007).cmp_span(a, span), Ordering::Equal);
        assert_eq!(Stamp::Ns(1_500_008).cmp_span(a, span), Ordering::Greater);
        // Raw stamps: one subtraction, and the removed time beside it.
        assert_eq!(
            Stamp::Raw(0.8, 1.0).cmp_span(Stamp::Raw(0.5, 1.0), 0.3),
            Ordering::Greater
        );
        assert_eq!(
            Stamp::Raw(10.0, 6.0).cmp_span(Stamp::Raw(1.0, 0.0), 3.0),
            Ordering::Equal
        );
        // Two forms: the decayed clocks as numbers.
        assert_eq!(
            Stamp::Raw(2.0, 0.0).cmp_span(Stamp::Ns(1_000_000_000), 1.0),
            Ordering::Equal
        );
        // A raw stamp with time removed: raw 10 less 3 removed is a decayed
        // clock of 7, on either side of the comparison.
        let (raw, six) = (Stamp::Raw(10.0, 3.0), Stamp::Ns(6_000_000_000));
        assert_eq!(raw.cmp_span(six, 1.0), Ordering::Equal);
        assert_eq!(raw.cmp_span(six, 0.5), Ordering::Greater);
        assert_eq!(raw.cmp_span(six, 1.5), Ordering::Less);
        assert_eq!(Stamp::Ns(9_000_000_000).cmp_span(raw, 2.0), Ordering::Equal);
        assert_eq!(
            Stamp::Ns(9_000_000_000).cmp_span_ns(raw, 2.0, Some(2_000_000_000)),
            Ordering::Equal
        );
        // A difference with no answer is equal, as the operators read one.
        assert_eq!(
            Stamp::Raw(f64::INFINITY, 0.0).cmp_span(Stamp::Raw(f64::INFINITY, 0.0), 1.0),
            Ordering::Equal
        );
        assert_eq!(Stamp::from(2.5), Stamp::Raw(2.5, 0.0));
    }

    /// With the span's nanoseconds, two temporal stamps compare as integers
    /// at any length: a window of a hundred days and a nanosecond, whose
    /// seconds a double holds only to about two nanoseconds, ties through
    /// seconds with a difference a nanosecond longer, and is shorter than it
    /// as integers. Every other pair compares as without them -- the window
    /// operators' `Gap::cmp_window`, branch by branch.
    #[test]
    fn a_span_in_nanoseconds_compares_two_temporal_stamps_as_integers() {
        let w: i128 = 100 * 86_400 * 1_000_000_000 + 1;
        let span = seconds_of_ns(w);
        assert_eq!(seconds_of_ns(w + 1), span, "a tie through seconds");
        let (a, b) = (Stamp::Ns(0), Stamp::Ns(w + 1));
        assert_eq!(b.cmp_span(a, span), Ordering::Equal);
        assert_eq!(b.cmp_span_ns(a, span, Some(w)), Ordering::Greater);
        assert_eq!(Stamp::Ns(w).cmp_span_ns(a, span, Some(w)), Ordering::Equal);
        assert_eq!(
            Stamp::Ns(w - 1).cmp_span_ns(a, span, Some(w)),
            Ordering::Less
        );
        // The nanoseconds are the temporal pair's alone.
        let (r0, r1) = (Stamp::Raw(0.5, 0.0), Stamp::Raw(0.8, 0.0));
        for span_ns in [None, Some(300_000_000)] {
            assert_eq!(r1.cmp_span_ns(r0, 0.3, span_ns), Ordering::Greater);
        }
        assert_eq!(b.cmp_span_ns(a, span, None), b.cmp_span(a, span));
    }

    /// `seconds_of_ns` gives a duration's seconds back as nanoseconds,
    /// exactly, under `2^23` seconds and at any whole number of seconds; and
    /// gives none for no finite value.
    #[test]
    fn nanoseconds_of_a_durations_seconds_are_the_duration() {
        for ns in [
            0i128,
            1,
            999_999_999,
            1_000_000_000,
            1_500_000,
            86_400_000_000_007,
            (1i128 << 23) * 1_000_000_000 - 1,
            9_000_000_000 * 1_000_000_000,
        ] {
            assert_eq!(ns_of_seconds(seconds_of_ns(ns)), Some(ns), "{ns}");
            assert_eq!(ns_of_seconds(seconds_of_ns(-ns)), Some(-ns), "-{ns}");
        }
        assert_eq!(ns_of_seconds(f64::INFINITY), None);
        assert_eq!(ns_of_seconds(f64::NAN), None);
    }

    /// What a temporal stamp's comparison rests on: `seconds_of_ns` never
    /// decreases, so a longer difference never compares below a shorter
    /// one, around every whole second, across the 64-bit road and the
    /// 128-bit one, and far past where a double resolves a nanosecond; and
    /// below `2^23` seconds it is one-to-one, so a difference compares with
    /// a duration's seconds as its nanoseconds do.
    #[test]
    fn seconds_of_nanoseconds_never_decrease() {
        let wide = i128::from(i64::MAX);
        let mut probes: Vec<i128> = Vec::new();
        for k in [
            0i128,
            1,
            2,
            7,
            1 << 22,
            1 << 23,
            1 << 30,
            1 << 52,
            1 << 53,
            1 << 60,
        ] {
            let base = k * 1_000_000_000;
            probes.extend([base - 2, base - 1, base, base + 1, base + 999_999_999]);
        }
        probes.extend([
            wide - 1,
            wide,
            wide + 1,
            wide + 1_000_000_000,
            i128::MAX / 2,
        ]);
        probes.sort_unstable();
        for w in probes.windows(2) {
            assert!(
                seconds_of_ns(w[0]) <= seconds_of_ns(w[1]),
                "{} then {}",
                w[0],
                w[1]
            );
            assert!(seconds_of_ns(-w[1]) <= seconds_of_ns(-w[0]));
        }
        let mut s = 0x1234_5678_9abc_def0u64;
        let limit = (1i128 << 23) * 1_000_000_000 - 1_001;
        for _ in 0..20_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = i128::from(s >> 11) % limit;
            let b = a + 1 + i128::from(s % 1_000);
            assert!(seconds_of_ns(a) < seconds_of_ns(b), "{a} {b}");
        }
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

    /// A decay is checked at its edges: a half-life above 0, the smallest
    /// positive one included and `inf` too; a factor in `(0, 1]`, the
    /// smallest positive one and 1 included and the next double above 1
    /// not (review 2026-10-05, CF5).
    #[test]
    fn a_decay_is_checked_at_its_edges() {
        for ok in [
            Decay::Halflife(f64::MIN_POSITIVE),
            Decay::Halflife(f64::INFINITY),
            Decay::Lam(f64::MIN_POSITIVE),
            Decay::Lam(1.0),
        ] {
            assert_eq!(ok.check(), Ok(()), "{ok:?}");
        }
        for bad in [
            Decay::Halflife(0.0),
            Decay::Halflife(-0.0),
            Decay::Halflife(f64::NEG_INFINITY),
            Decay::Halflife(f64::NAN),
        ] {
            let e = bad.check().unwrap_err();
            assert!(e.starts_with("half_life must be > 0"), "{bad:?}: {e}");
        }
        for bad in [
            Decay::Lam(0.0),
            Decay::Lam(1.0f64.next_up()),
            Decay::Lam(f64::INFINITY),
            Decay::Lam(f64::NAN),
        ] {
            let e = bad.check().unwrap_err();
            assert!(e.starts_with("lam must be in (0, 1]"), "{bad:?}: {e}");
        }
    }

    /// No time is `+0.0`: zero is not a step back, so it takes the forward
    /// road, and two equal stamps are a delta of `+0.0`. The mutation run of
    /// 2026-10-05 left `ns < 0` as `ns <= 0` alive, since nothing asked for
    /// zero; under it zero is its own mirror, and recursed until the stack
    /// overflowed.
    #[test]
    fn no_time_is_positive_zero() {
        let zero = seconds_of_ns(0);
        assert_eq!(zero, 0.0);
        assert!(zero.is_sign_positive(), "{zero:?}");
        let same = ClockValue::Ns(1_700_000_000_000_000_000);
        assert!(same.delta(same).is_sign_positive());
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

    /// Epoch nanoseconds in 2026, a multiple of 256: a double resolves 256
    /// here, so `T_INT + 1` to `T_INT + 127` are all `T_INT` as one.
    const T_INT: i64 = 1_790_000_000_000_000_000;

    fn int(v: i64) -> Option<ClockValue> {
        Some(ClockValue::I64(v))
    }

    /// Task 200: an integer clock's step is the integers' difference, at any
    /// size: ticks 1, 7 and 100 apart near 1.79e18 are steps of 1, 7 and 100,
    /// where the same values as doubles are one value and steps of 0.
    #[test]
    fn an_integer_clocks_step_is_exact_at_any_size() {
        let cfg = ClockCfg {
            gap_cap: 1e6,
            ..ClockCfg::default()
        };
        let mut c = ClockState::new();
        let mut f = ClockState::new();
        let mut at = T_INT;
        assert_eq!(c.advance(&cfg, int(at), None, true).d_clock, 0.0);
        f.advance(&cfg, Some(ClockValue::F64(at as f64)), None, true);
        for step in [1i64, 7, 100, 1, 1] {
            at += step;
            let a = c.advance(&cfg, int(at), None, true);
            assert_eq!((a.d_clock, a.elapsed), (step as f64, step as f64), "{step}");
            let lost = f.advance(&cfg, Some(ClockValue::F64(at as f64)), None, true);
            assert_eq!(lost.d_clock, 0.0, "a double loses the step of {step}");
        }
        assert_eq!(c.last_clock(), int(at));
        assert!(ClockValue::I64(T_INT).is_before(ClockValue::I64(T_INT + 1)));
        assert!(!ClockValue::F64(T_INT as f64).is_before(ClockValue::F64((T_INT + 1) as f64)));
    }

    /// Every decision on an integer clock's step is taken on the integers,
    /// against parameters that are doubles: a step of `2^53 + 1` is past a
    /// cap of `2^53`, whose double the step's ties with; a step back of
    /// `2^53 + 1` is past a late-row minimum of `2^53`, and starts the
    /// stream over; a step back of exactly the minimum is a late row; a step
    /// of exactly the cap is not capped; and a fractional parameter is
    /// compared as the number it is.
    #[test]
    fn an_integer_step_is_judged_on_the_integers() {
        let two53: i64 = 1 << 53;
        assert_eq!((two53 + 1) as f64, two53 as f64, "the doubles tie");
        let cfg = ClockCfg {
            gap_cap: two53 as f64,
            on_clock_reset: OnClockReset::ResetState,
            session_gap: None,
            min_backwards_jump: two53 as f64,
        };
        let from = |v: i64| {
            let mut c = ClockState::new();
            c.advance(&cfg, int(v), None, true);
            c
        };
        let a = from(0).advance(&cfg, int(two53 + 1), None, true);
        assert!(a.capped, "past the cap by one");
        assert_eq!(a.d_clock, two53 as f64);
        let a = from(0).advance(&cfg, int(two53), None, true);
        assert!(!a.capped, "exactly the cap");
        let base = 2 * two53 + 10;
        let a = from(base).advance(&cfg, int(base - two53 - 1), None, true);
        assert!(a.reset && a.disorder.is_none(), "past the minimum by one");
        let a = from(base).advance(&cfg, int(base - two53), None, true);
        assert_eq!(
            a.disorder,
            Some(Disorder {
                back: two53 as f64,
                min_backwards_jump: two53 as f64
            }),
            "exactly the minimum is a late row"
        );
        // Near 1.79e18, against small parameters.
        let small = ClockCfg {
            gap_cap: 100.0,
            on_clock_reset: OnClockReset::ResetState,
            session_gap: None,
            min_backwards_jump: 100.0,
        };
        for (step, capped) in [(100i64, false), (101, true)] {
            let mut c = ClockState::new();
            c.advance(&small, int(T_INT + 27), None, true);
            let a = c.advance(&small, int(T_INT + 27 + step), None, true);
            assert_eq!(a.capped, capped, "a step of {step}");
        }
        for (back, late) in [(100i64, true), (101, false)] {
            let mut c = ClockState::new();
            c.advance(&small, int(T_INT + 228), None, true);
            let a = c.advance(&small, int(T_INT + 228 - back), None, true);
            assert_eq!(a.disorder.is_some(), late, "a step back of {back}");
            assert_eq!(a.reset, !late, "a step back of {back}");
        }
        let mut c = ClockState::new();
        c.advance(&small, int(T_INT + 5), None, true);
        assert_eq!(
            c.advance(&small, int(T_INT + 4), None, true).backwards,
            Some(-1.0)
        );
        // A fractional cap: a step of 1 is past 0.5, and its model sees 0.5.
        let half = ClockCfg {
            gap_cap: 0.5,
            ..ClockCfg::default()
        };
        let mut c = ClockState::new();
        c.advance(&half, int(T_INT), None, true);
        let a = c.advance(&half, int(T_INT + 1), None, true);
        assert!(a.capped && a.d_clock == 0.5);
    }

    /// Skipped rows' steps fold into the next accepted row in integers, and
    /// the total is judged against the cap exactly: two skipped steps of
    /// `2^52` and an accepted one of 1 come to `2^53 + 1`, past a cap of
    /// `2^53`, which their doubles' sum ties with.
    #[test]
    fn an_integer_fold_is_judged_on_the_integers() {
        let two52: i64 = 1 << 52;
        let cfg = ClockCfg {
            gap_cap: (2 * two52) as f64,
            ..ClockCfg::default()
        };
        let mut c = ClockState::new();
        c.advance(&cfg, int(0), None, true);
        c.advance(&cfg, int(two52), None, false);
        c.advance(&cfg, int(2 * two52), None, false);
        let a = c.advance(&cfg, int(2 * two52 + 1), None, true);
        assert!(a.capped, "a total one past the cap");
        assert_eq!(a.d_clock, (2 * two52) as f64);
        let mut c = ClockState::new();
        c.advance(&cfg, int(0), None, true);
        c.advance(&cfg, int(two52), None, false);
        let a = c.advance(&cfg, int(2 * two52), None, true);
        assert!(!a.capped, "a total of exactly the cap");
        assert_eq!(a.d_clock, (2 * two52) as f64);
    }

    /// On an integer clock a row's stamp is its raw value less the time the
    /// caps and session steps removed, in integers: two rows of a stretch
    /// differ by one subtraction of their values, so a difference of exactly
    /// a span is equal to it near 1.79e18; across a capped gap the decayed
    /// clock moves by the cap exactly, and across a session change by the
    /// session's gap; skipped rows fold under the cap; a reset starts the
    /// stamps over at the row's own value. The elapsed clock counts every
    /// step whole. The oracle is the steps written out in integers.
    #[test]
    fn an_integer_stamp_is_the_decayed_clock_in_integers() {
        let cfg = ClockCfg {
            gap_cap: 100.0,
            session_gap: Some(SessionGap::Gap(30.0)),
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 1000.0,
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row =
            |v: i64, s: u64, accept: bool| c.advance_stamped(&cfg, &caps, int(v), Some(s), accept);
        let first = row(T_INT, 1, true);
        assert_eq!(first.stamp, Some(Stamp::Int(i128::from(T_INT), 0.0)));
        let (d0, e0) = (first.stamp.unwrap(), first.elapsed_stamp.unwrap());
        let (mut last, mut at) = (d0, T_INT);
        for step in [1i64, 7, 100, 7, 1, 0] {
            at += step;
            let a = row(at, 1, true);
            let s = a.stamp.unwrap();
            let span = (at - T_INT) as f64;
            assert_eq!(s.cmp_span(last, step as f64), Ordering::Equal, "{step}");
            assert_eq!(s.cmp_span(d0, span), Ordering::Equal, "{step}");
            assert_eq!(s.cmp_span(d0, span - 0.5), Ordering::Greater, "{step}");
            assert_eq!(s.cmp_span(d0, span + 0.5), Ordering::Less, "{step}");
            assert_eq!(a.elapsed_stamp.unwrap().cmp_span(e0, span), Ordering::Equal);
            last = s;
        }
        // As doubles every one of these rows was `T_INT`: the stamps of a
        // number clock compare them as one.
        let raw = |v: i64| Stamp::Raw(v as f64, 0.0);
        assert_eq!(raw(at).cmp_span(raw(T_INT), 0.0), Ordering::Equal);
    }

    /// The decisions a stamp keeps across breaks, on an integer clock with
    /// whole parameters: a gap of 10^15 past a cap of 100 moves the decayed
    /// clock by 100 exactly and the elapsed clock by the whole gap; a session
    /// change moves it by the session's gap; skipped rows fold under the
    /// cap; a reset starts it over.
    #[test]
    fn an_integer_stamp_keeps_whole_parameters_exact_across_breaks() {
        let cfg = ClockCfg {
            gap_cap: 100.0,
            session_gap: Some(SessionGap::Gap(30.0)),
            on_clock_reset: OnClockReset::ResetState,
            min_backwards_jump: 1000.0,
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let mut row =
            |v: i64, s: u64, accept: bool| c.advance_stamped(&cfg, &caps, int(v), Some(s), accept);
        let start = row(T_INT, 1, true);
        let (d0, e0) = (start.stamp.unwrap(), start.elapsed_stamp.unwrap());
        let mut at = T_INT + 1_000_000_000_000_000;
        let gap = row(at, 1, true);
        assert!(gap.capped);
        assert_eq!(gap.stamp.unwrap().cmp_span(d0, 100.0), Ordering::Equal);
        assert_eq!(
            gap.elapsed_stamp.unwrap().cmp_span(e0, 1e15),
            Ordering::Equal,
            "the elapsed clock counts the gap whole"
        );
        at += 1;
        let session = row(at, 2, true);
        assert_eq!(session.stamp.unwrap().cmp_span(d0, 130.0), Ordering::Equal);
        assert_eq!(
            session.elapsed_stamp.unwrap().cmp_span(e0, 1e15 + 1.0),
            Ordering::Equal
        );
        // Three skipped rows 60 apart, then one 1 on: 181 folded, under the
        // cap of 100.
        for _ in 0..3 {
            at += 60;
            assert_eq!(row(at, 2, false).stamp, None);
        }
        at += 1;
        let folded = row(at, 2, true);
        assert!(folded.capped);
        assert_eq!(folded.stamp.unwrap().cmp_span(d0, 230.0), Ordering::Equal);
        assert_eq!(
            folded.stamp.unwrap(),
            Stamp::Int(i128::from(T_INT) + 230, 0.0)
        );
        // A step back past the minimum starts the stamps over at the row.
        let reset = row(at - 5000, 2, true);
        assert!(reset.reset);
        assert_eq!(reset.stamp, Some(Stamp::Int(i128::from(at - 5000), 0.0)));
        assert_eq!(reset.elapsed_stamp, reset.stamp);
    }

    /// A fractional cap on an integer clock: each capped step moves the
    /// decayed clock by the cap, its fraction carried beside the whole
    /// units, so three steps capped at 0.5 come to 1.5 exactly; and an
    /// integer stamp moved back by a fractional amount keeps it.
    #[test]
    fn an_integer_stamp_carries_a_fractional_cap_beside_the_whole_units() {
        let cfg = ClockCfg {
            gap_cap: 0.5,
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        let first = c
            .advance_stamped(&cfg, &caps, int(T_INT), None, true)
            .stamp
            .unwrap();
        let mut s = first;
        for k in 1..=3 {
            s = c
                .advance_stamped(&cfg, &caps, int(T_INT + 10 * k), None, true)
                .stamp
                .unwrap();
        }
        assert_eq!(s.cmp_span(first, 1.5), Ordering::Equal);
        assert_eq!(s.cmp_span(first, 1.25), Ordering::Greater);
        let back = s.int_back_by(0.25).unwrap();
        assert_eq!(back.cmp_span(first, 1.25), Ordering::Equal);
        assert_eq!(Stamp::Raw(1.0, 0.0).int_back_by(0.25), None);
    }

    /// `cmp_int_f64` orders an integer and a double without rounding either:
    /// at `2^53 + 1` against `2^53`, around a fraction on either sign, at
    /// zero of either sign, past every i128 and at a NaN.
    #[test]
    fn an_integer_and_a_double_compare_exactly() {
        let two53 = 1i128 << 53;
        let cases = [
            (two53 + 1, two53 as f64, Some(Ordering::Greater)),
            (two53, two53 as f64, Some(Ordering::Equal)),
            (5, 5.5, Some(Ordering::Less)),
            (6, 5.5, Some(Ordering::Greater)),
            (-5, -5.5, Some(Ordering::Greater)),
            (-6, -5.5, Some(Ordering::Less)),
            (0, -0.0, Some(Ordering::Equal)),
            (-1, -0.5, Some(Ordering::Less)),
            (i128::MAX, f64::INFINITY, Some(Ordering::Less)),
            (i128::MIN, f64::NEG_INFINITY, Some(Ordering::Greater)),
            (i128::MAX, 1e300, Some(Ordering::Less)),
            (i128::MIN, -1e300, Some(Ordering::Greater)),
            (
                i128::MIN,
                -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
                Some(Ordering::Equal),
            ),
            (0, f64::NAN, None),
        ];
        for (i, f, want) in cases {
            assert_eq!(cmp_int_f64(i, f), want, "{i} against {f}");
        }
    }

    /// An integer clock's state reads back, and a float clock's writes none
    /// of the integer fields: its bytes are what they were.
    #[test]
    fn an_integer_clocks_state_reads_back() {
        let cfg = ClockCfg {
            gap_cap: 0.5,
            ..ClockCfg::default()
        };
        let caps = ExactCaps::default();
        let mut c = ClockState::new();
        c.advance_stamped(&cfg, &caps, int(T_INT), None, true);
        c.advance_stamped(&cfg, &caps, int(T_INT + 3), None, false);
        let json = serde_json::to_value(&c).unwrap();
        assert!(json.get("pending_int").is_some(), "{json}");
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let mut back: ClockState = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, c);
        let a = back.advance_stamped(&cfg, &caps, int(T_INT + 4), None, true);
        let b = c.advance_stamped(&cfg, &caps, int(T_INT + 4), None, true);
        assert_eq!(a, b);
        let mut f = ClockState::new();
        f.advance_stamped(&cfg, &caps, Some(ClockValue::F64(1.0)), None, true);
        f.advance_stamped(&cfg, &caps, Some(ClockValue::F64(4.0)), None, false);
        let json = serde_json::to_value(&f).unwrap();
        assert!(json.get("pending_int").is_none(), "{json}");
        assert!(json["exact"].get("removed_int").is_none(), "{json}");
    }

    #[test]
    fn decay_factors() {
        assert!((Decay::Halflife(10.0).factor(10.0) - 0.5).abs() < 1e-15);
        assert_eq!(Decay::Halflife(f64::INFINITY).factor(123.0), 1.0);
        assert!((Decay::Lam(0.9).factor(2.0) - 0.81).abs() < 1e-15);
        assert_eq!(Decay::Halflife(10.0).factor(0.0), 1.0);
    }

    /// A `lam` factor at a step of one clock unit is `lam` itself, and at
    /// no step 1, to the bit and by construction: a stream pinned to the
    /// bit under `Decay::Lam` at unit steps (`sgd.rs`'s libm-free digests)
    /// rested on the platform's `pow(x, 1)` returning `x`, which every libm
    /// met so far does and none promises (review round 5, A4).
    #[test]
    fn a_lam_factor_at_a_unit_step_is_lam_to_the_bit() {
        for lam in [
            0.9914,
            0.5,
            0.999_999_999,
            1.0,
            1e-300,
            0.123_456_789_012_345_6,
        ] {
            assert_eq!(
                Decay::Lam(lam).factor(1.0).to_bits(),
                lam.to_bits(),
                "{lam}"
            );
            assert_eq!(
                Decay::Lam(lam).factor(0.0).to_bits(),
                1.0f64.to_bits(),
                "{lam}"
            );
            assert_eq!(
                Decay::Lam(lam).factor(-0.0).to_bits(),
                1.0f64.to_bits(),
                "{lam}"
            );
        }
        // Every other step is still the power.
        assert!((Decay::Lam(0.9).factor(0.5) - 0.9f64.sqrt()).abs() < 1e-15);
        assert_eq!(Decay::Lam(0.9).factor(f64::INFINITY), 0.0);
        assert!(Decay::Lam(0.9).factor(f64::NAN).is_nan());
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

    // Eight times proptest's own count: 2,048 cases, or 256 under the
    // essentials gate's `PROPTEST_CASES=32` (docs/TESTING.md, "Two tiers").
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8 * ProptestConfig::default().cases))]
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

        /// The same over an integer clock (task 200), its values anywhere
        /// in an `i64`, near one another or at its two ends, stamped, so
        /// the exact fold and the removed time run too.
        #[test]
        fn every_integer_step_is_finite(
            (cfg, _) in cfg(),
            rows in prop::collection::vec(
                (
                    prop_oneof![any::<i64>(), -1000i64..1000, Just(i64::MAX), Just(i64::MIN)],
                    0u64..3,
                    any::<bool>(),
                ),
                1..40,
            ),
        ) {
            let cfg = ClockCfg { gap_cap: cfg.gap_cap.min(1e12), ..cfg };
            let caps = ExactCaps::default();
            let mut learned = ClockState::new();
            for (t, session, accept) in rows {
                let c = Some(ClockValue::I64(t));
                let scored = learned.clone().advance_scoring(&cfg, c, Some(session));
                let a = learned.advance_stamped(&cfg, &caps, c, Some(session), accept);
                for (what, d) in [("learned", a.d_clock), ("scored", scored.d_clock)] {
                    prop_assert!(
                        d.is_finite() && d >= 0.0 && d <= cfg.gap_cap,
                        "{what} step {d} under {cfg:?}"
                    );
                }
                if let Some(Stamp::Int(_, frac)) = a.stamp {
                    prop_assert!(frac.abs() < 1.0, "{frac}");
                }
            }
        }
    }
}
