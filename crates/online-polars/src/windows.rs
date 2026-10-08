//! The window core (docs/PLAN.md tasks 78, 143 and 144): exponentially
//! weighted operators over a stream, looking back and ahead, one pass,
//! O(window) memory, exact.
//!
//! Everything stateful is a *kernel*: a direction, a half-life, a window and
//! a `closed` rule on the policy clock. Any number of *operators* share a
//! kernel's queue -- a time-weighted mean, a decayed sum, a rate -- each
//! reading one input of the row. The formula around the operators is
//! Polars' (`formula.rs`), and an `increment` is the frame runner's; this
//! core sees neither.
//!
//! **Definitions**, each held to a brute-force loop in the tests:
//!
//! - `ewm_mean` / `rewm_mean`: the time-weighted mean, Polars' `ewm_mean_by`.
//!   Looking back, row `i`'s value is held over `(t_{i-1}, t_i]`, the first
//!   value of a stretch from before it (`-inf`, as `ewm_mean_by`'s `a_1 = 1`;
//!   from `t_1 - w` under a window of `w`); the mean over `(T - w, T]` weighs
//!   each held value by `∫ λ^(T - s) ds` over its interval inside the
//!   window. Looking ahead, value `j` is held over `[t_j, t_{j+1})`, until
//!   the next row, and weighed by `∫ λ^(s - t) ds` inside `(t, t + w]`. A
//!   row at a repeated stamp holds no interval, so it moves no mean -- as in
//!   Polars.
//! - `ewm_sum` / `rewm_sum`: `Σ λ^|t_j - t| x_j` over the rows of the window,
//!   each counted once at its own time (`ewm_sum_by` on distinct stamps; at
//!   a repeated stamp every row holds the stamp's whole window, so each
//!   carries the stamp's total where `ewm_sum_by`'s is a running sum).
//! - `ewm_rate` / `rewm_rate`: the sum over the decayed time the window
//!   covers, `∫_0^T λ^s ds = h/ln2 · (1 - 2^(-T/h))`, `T` the window's span
//!   inside the stretch.
//! - `ewm_var` / `ewm_std` (task 212), looking back only: the variance about
//!   `ewm_mean`, each value weighed by the mean's weight `m_i`, its held
//!   interval's decayed mass inside the window: `Σ m_i (x_i - μ)² / V1`
//!   with `μ = Σ m_i x_i / V1`, `V1 = Σ m_i`; unless `bias`, times
//!   `V1² / (V1² - V2)`, `V2 = Σ m_i²`, the correction for unequal weights
//!   Polars' `ewm_var` applies to its own, null where one row carries all
//!   the weight (`V1² = V2`). `ewm_std`
//!   is its root. On a clock that steps by 1 a row these are Polars'
//!   `ewm_var` and `ewm_std` under `adjust = False`, the form `ewm_mean_by`
//!   is: a stretch's first value is held from before it, so it weighs
//!   `(1 - α)^(n-1)` as `adjust = False`'s first row does. A variance's
//!   sums are merged about their means -- `(V1, μ, M2)` by Chan, Golub and
//!   LeVeque's pairwise update, `M2 = M2_a + M2_b + δ² V1_a V1_b / V1` --
//!   never as a mean square less a squared mean, so a window holding one
//!   value is 0 exactly, and the mean is a compensated pair
//!   (`online_core::comp`), so a variance at a level decays with its
//!   history rather than settling on the square of a stalled mean's gap.
//!
//! **Which rows a window holds** follows Polars' `rolling_*_by`: a window is
//! a set of timestamps. `closed = "right"`, the default, is `(T - w, T]`
//! looking back and `(t, t + w]` looking ahead; `"left"`, `"both"` and
//! `"none"` move the ends. Every row at one stamp gets the same backward
//! window, later rows at that stamp included, so under `"right"` and
//! `"both"` a backward output waits for the next distinct stamp; under
//! `"left"` and `"none"` the stamp's own rows are outside, and the output
//! is known as the stamp arrives. A forward window counts a row exactly `w`
//! later under `"right"` and `"both"`.
//!
//! **Clock events** are task 78's: a gap past `gap_cap` or a session change
//! ends every window open across it (partial, under each operator's
//! `partial`); a reset discards them (null, never dropped); a group that
//! falls silent is cut once the stream's clock is `gap_cap` past its last
//! row. A forward `"right"` or `"both"` window whose far edge is the last
//! row before such a cut or reset holds every timestamp it can, and is
//! whole, as its backward mirror is (tasks 160 and 173). Only a clock column
//! cuts a silent group, so the frame runner and the specs refuse a forward
//! window under groups without one (`spec.rs`, `clock_cfg_of`; task 173):
//! on a row-count clock a silent group's clock never moves, and its open
//! windows would hold every later row. Rows leave in input order, each once
//! every window over it has resolved.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use online_core::{ClockAdvance, ClockCfg, ClockState, ClockValue, Stamp, seconds_of_ns};
use serde::{Deserialize, Serialize};

use crate::stream::usable;

/// Which way a window looks from its row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// The rows at or before the row, less than `window_size` older.
    Backward,
    /// The rows after the row, at most `window_size` later.
    Forward,
}

/// Which ends of a window are in it, as Polars' `closed` says it. The
/// windowed models take it too, `Right` or `Both` (`crate::spec`,
/// docs/PLAN.md task 196).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Closed {
    /// `(T - w, T]` back, `(t, t + w]` ahead: Polars' default.
    #[default]
    Right,
    /// `[T - w, T)` back, `[t, t + w)` ahead.
    Left,
    /// `[T - w, T]` back, `[t, t + w]` ahead.
    Both,
    /// `(T - w, T)` back, `(t, t + w)` ahead.
    #[serde(rename = "none")]
    Neither,
}

impl Closed {
    pub fn name(self) -> &'static str {
        match self {
            Closed::Right => "right",
            Closed::Left => "left",
            Closed::Both => "both",
            Closed::Neither => "none",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "right" => Some(Closed::Right),
            "left" => Some(Closed::Left),
            "both" => Some(Closed::Both),
            "none" => Some(Closed::Neither),
            _ => None,
        }
    }

    /// Whether the end at the row's own stamp is in the window: `T` of a
    /// backward window, `t` of a forward one.
    fn near(self, direction: Direction) -> bool {
        match direction {
            Direction::Backward => matches!(self, Closed::Right | Closed::Both),
            Direction::Forward => matches!(self, Closed::Left | Closed::Both),
        }
    }

    /// Whether the end a window away is in it: `T - w` of a backward window,
    /// `t + w` of a forward one.
    fn far(self, direction: Direction) -> bool {
        match direction {
            Direction::Backward => matches!(self, Closed::Left | Closed::Both),
            Direction::Forward => matches!(self, Closed::Right | Closed::Both),
        }
    }
}

/// What a window cut short before its span passed gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partial {
    /// The value over what the window saw.
    Keep,
    /// Null.
    Null,
    /// The row leaves the output.
    Drop,
}

impl Partial {
    pub fn name(self) -> &'static str {
        match self {
            Partial::Keep => "keep",
            Partial::Null => "null",
            Partial::Drop => "drop",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "keep" => Some(Partial::Keep),
            "null" => Some(Partial::Null),
            "drop" => Some(Partial::Drop),
            _ => None,
        }
    }
}

/// What an operator computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stat {
    Mean,
    Sum,
    Rate,
    /// The variance about the mean, under the mean's weights (task 212).
    Var,
    /// The variance's root.
    Std,
}

impl Stat {
    /// Whether a value weighs its held interval's mass, as a mean's does.
    #[inline]
    pub fn weighs_held(self) -> bool {
        matches!(self, Stat::Mean | Stat::Var | Stat::Std)
    }

    /// Whether its sums are a variance's: merged about a compensated mean.
    #[inline]
    pub fn is_var(self) -> bool {
        matches!(self, Stat::Var | Stat::Std)
    }
}

/// An operator, as the formulas name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    EwmMean,
    RewmMean,
    EwmSum,
    RewmSum,
    EwmRate,
    RewmRate,
    EwmVar,
    EwmStd,
    Increment,
}

impl OpKind {
    pub fn name(self) -> &'static str {
        match self {
            OpKind::EwmMean => "ewm_mean",
            OpKind::RewmMean => "rewm_mean",
            OpKind::EwmSum => "ewm_sum",
            OpKind::RewmSum => "rewm_sum",
            OpKind::EwmRate => "ewm_rate",
            OpKind::RewmRate => "rewm_rate",
            OpKind::EwmVar => "ewm_var",
            OpKind::EwmStd => "ewm_std",
            OpKind::Increment => "increment",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ewm_mean" => Some(OpKind::EwmMean),
            "rewm_mean" => Some(OpKind::RewmMean),
            "ewm_sum" => Some(OpKind::EwmSum),
            "rewm_sum" => Some(OpKind::RewmSum),
            "ewm_rate" => Some(OpKind::EwmRate),
            "rewm_rate" => Some(OpKind::RewmRate),
            "ewm_var" => Some(OpKind::EwmVar),
            "ewm_std" => Some(OpKind::EwmStd),
            "increment" => Some(OpKind::Increment),
            _ => None,
        }
    }

    pub fn direction(self) -> Direction {
        match self {
            OpKind::RewmMean | OpKind::RewmSum | OpKind::RewmRate => Direction::Forward,
            _ => Direction::Backward,
        }
    }

    /// The statistic of a window operator; `None` for `increment`.
    pub fn stat(self) -> Option<Stat> {
        match self {
            OpKind::EwmMean | OpKind::RewmMean => Some(Stat::Mean),
            OpKind::EwmSum | OpKind::RewmSum => Some(Stat::Sum),
            OpKind::EwmRate | OpKind::RewmRate => Some(Stat::Rate),
            OpKind::EwmVar => Some(Stat::Var),
            OpKind::EwmStd => Some(Stat::Std),
            OpKind::Increment => None,
        }
    }
}

/// A kernel: the queue every operator with these settings shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelDef {
    pub direction: Direction,
    /// Policy-clock units, above 0; infinite weighs the window evenly.
    pub half_life: f64,
    /// Policy-clock units, finite and above 0; `None` is no cutoff, which
    /// only a backward kernel may have.
    pub window_size: Option<f64>,
    /// The window in integer nanoseconds where it was given as a duration:
    /// on a temporal clock an edge is then decided between integers
    /// (review R2, W1), with no conversion to seconds on the way.
    pub window_ns: Option<i64>,
    pub closed: Closed,
}

impl KernelDef {
    /// # Errors
    ///
    /// A half-life that is not above 0, a window that is not finite and
    /// above 0 or that a forward kernel lacks.
    pub fn check(&self) -> Result<(), String> {
        if self.half_life.is_nan() || self.half_life <= 0.0 {
            return Err(format!("half_life must be above 0, got {}", self.half_life));
        }
        if self.half_life < f64::MIN_POSITIVE {
            // A subnormal half-life: `mass·x` underflows to 0 and a rate
            // divides by that mass, so the mean came out 0 and the rate
            // infinite (task 159, W5).
            return Err(format!(
                "half_life must be a normal number, at least {:e}, got {:e}",
                f64::MIN_POSITIVE,
                self.half_life
            ));
        }
        match (self.direction, self.window_size) {
            (Direction::Forward, None) => {
                return Err("a forward operator needs a window_size".into());
            }
            (_, Some(w)) if !w.is_finite() || w <= 0.0 => {
                return Err(format!("window_size must be finite and above 0, got {w}"));
            }
            _ => {}
        }
        // The two forms of one window agree (review R4, A4): membership is
        // decided on the integer, the far edge and `complete` on the number.
        if let (Some(w), Some(ns)) = (self.window_size, self.window_ns)
            && seconds_of_ns(i128::from(ns)) != w
        {
            return Err(format!(
                "window_ns {ns} is not window_size {w} in nanoseconds"
            ));
        }
        Ok(())
    }

    /// `λ^d`, the discount across `d` units of policy time.
    #[inline]
    fn discount(&self, d: f64) -> f64 {
        if self.half_life.is_infinite() {
            1.0
        } else {
            // `exp2(-x)`, as `Decay::factor` spells it, for the same reason.
            (-(d / self.half_life)).exp2()
        }
    }

    /// `∫_0^T λ^s ds`: the decayed mass of `T` units of time.
    #[inline]
    fn mass(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if self.half_life.is_infinite() {
            t
        } else {
            // `1 − 2^(−t/h)` cancels where `t ≪ h`: 6e-5 of the mass at
            // `h/t = 1e12`, all of it from 1e17, every mean null (task 159,
            // W2). `exp_m1` keeps the small difference whole; `discount`
            // itself has nothing to lose.
            let h = self.half_life;
            -(h / std::f64::consts::LN_2) * (-(t * std::f64::consts::LN_2 / h)).exp_m1()
        }
    }
}

/// One window operator: its kernel, statistic and input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpDef {
    /// Index into the kernels.
    pub kernel: usize,
    pub stat: Stat,
    /// Index into a row's values.
    pub input: usize,
    /// Null with fewer rows carrying a value in the window.
    pub min_samples: u32,
    pub partial: Partial,
    /// A variance's: the variance of the weights as they are, uncorrected
    /// for their count (Polars' `bias`); read by no other statistic.
    pub bias: bool,
}

/// One row as the core reads it: its group, clock and session, every
/// operator input, and whether a `like=` spec would learn from it.
#[derive(Debug)]
pub struct RowIn<'a> {
    /// From [`Windows::group`].
    pub group: usize,
    /// `None` on a row-count clock.
    pub clock: Option<ClockValue>,
    pub session: Option<u64>,
    /// Per operator input; not [`usable`] is missing.
    pub values: &'a [f64],
    /// Whether the spec behind `like=` would learn from this row. Its
    /// forward windows are null when it would not: a model never learns the
    /// target of a row it skips (docs/PLAN.md task 104). The row still
    /// steps the clocks and still counts in other rows' windows.
    pub accept: bool,
}

/// What the clocks make of a row before it is taken ([`Windows::peek`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Peek {
    /// A new start by the policy's word: a step back past
    /// `restart_after_step_back`, or a new session, on the stream's clock
    /// or the group's.
    NewStart,
    /// A step forward on the stream's clock, or its first row.
    Forward,
    /// The same stamp as the last row the stream read.
    SameStamp,
}

/// Why a row was refused. The caller names the row in the input.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The clock stepped back `back` clock units, from `prev` to `now`,
    /// where the policy refuses it: any step with no `restart_after_step_back`,
    /// and one no larger than it (carried here) with one. The two values let
    /// a caller state a temporal step exactly, in nanoseconds.
    Backwards {
        seq: u64,
        back: f64,
        prev: Option<ClockValue>,
        now: Option<ClockValue>,
        min_backwards_jump: Option<f64>,
    },
}

/// Per operator, a segment's sums: the mass-weighted value, the mass and
/// the rows with a value. For a sum or a rate `s` is the decayed sum and
/// `w` unused. For a variance (task 212) `s` is the mean itself, `hi` of a
/// compensated pair with `lo` (`online_core::comp`), `w` the mass `V1`,
/// `q` the mass-weighted squared deviations from the mean `M2`, and `e`
/// the masses' products in pairs, `Σ_{i<j} m_i m_j = (V1² − V2) / 2`, kept
/// as a sum of its own so the count correction cancels nothing, and is 0,
/// not a rounding of 0, while one row carries the window; the three are zero
/// for every other statistic.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Acc {
    s: f64,
    w: f64,
    n: f64,
    lo: f64,
    q: f64,
    e: f64,
}

impl Acc {
    /// A held value `x` of mass `m`: its sums as one row.
    #[inline]
    fn held(x: f64, m: f64, var: bool) -> Acc {
        if var {
            Acc {
                s: x,
                w: m,
                n: 1.0,
                lo: 0.0,
                q: 0.0,
                e: 0.0,
            }
        } else {
            Acc {
                s: m * x,
                w: m,
                n: 1.0,
                ..Acc::default()
            }
        }
    }

    /// The sums seen `f` of a discount later: a variance's mean stays where
    /// it is, its masses scale, and their squares with `f²`.
    #[inline]
    fn scaled(self, f: f64, var: bool) -> Acc {
        if var {
            Acc {
                s: self.s,
                lo: self.lo,
                w: f * self.w,
                n: self.n,
                q: f * self.q,
                e: f * (f * self.e),
            }
        } else {
            Acc {
                s: f * self.s,
                w: f * self.w,
                n: self.n,
                ..Acc::default()
            }
        }
    }

    /// Both segments' sums, seen from one time.
    #[inline]
    fn plus(self, o: Acc, var: bool) -> Acc {
        if var {
            return merge(self, o);
        }
        Acc {
            s: self.s + o.s,
            w: self.w + o.w,
            n: self.n + o.n,
            ..Acc::default()
        }
    }

    /// Read from a stack's arena: three fields, or a variance-wide queue's
    /// six.
    #[inline]
    fn read(v: &[f64], wide: bool) -> Acc {
        if wide {
            Acc {
                s: v[0],
                w: v[1],
                n: v[2],
                lo: v[3],
                q: v[4],
                e: v[5],
            }
        } else {
            Acc {
                s: v[0],
                w: v[1],
                n: v[2],
                ..Acc::default()
            }
        }
    }

    /// Written to a stack's arena, as [`Acc::read`] reads it.
    #[inline]
    fn write(self, out: &mut [f64], wide: bool) {
        out[0] = self.s;
        out[1] = self.w;
        out[2] = self.n;
        if wide {
            out[3] = self.lo;
            out[4] = self.q;
            out[5] = self.e;
        }
    }
}

/// Two variance segments' sums as one, about their pooled mean: Chan, Golub
/// and LeVeque's pairwise update, `V1 = V1_a + V1_b`, `μ = μ_a + δ V1_b / V1`
/// and `M2 = M2_a + M2_b + δ² V1_a V1_b / V1` with `δ = μ_b − μ_a`, and the
/// pairs' products `e = e_a + e_b + V1_a V1_b`. No term
/// is a difference of two large sums, so a window of one value has `M2 = 0`
/// exactly at any level. The mean moves by its step as a compensated pair
/// (`online_core::comp`): a plain one stops `1/(2 V1_b/V1)` rounding steps
/// short of a value held row after row, and `M2` then settles on that gap's
/// square instead of decaying with its history. A segment of no mass moves
/// nothing but the row count, so it takes no step (`comp::add` says why);
/// an empty one is `Acc::default()`.
#[inline]
fn merge(a: Acc, b: Acc) -> Acc {
    let n = a.n + b.n;
    let q = a.q + b.q;
    let e = a.e + b.e;
    if b.w <= 0.0 {
        return Acc { n, q, e, ..a };
    }
    if a.w <= 0.0 {
        return Acc { n, q, e, ..b };
    }
    let w = a.w + b.w;
    let d = online_core::comp::dev(b.s, a.s, a.lo) + b.lo;
    let (mut hi, mut lo) = (a.s, a.lo);
    online_core::comp::add(&mut hi, &mut lo, d * (b.w / w));
    Acc {
        s: hi,
        lo,
        w,
        n,
        q: q + d * d * (a.w / w * b.w),
        e: e + a.w * b.w,
    }
}

/// `a` then `b`, `a` the earlier, each anchored at its own time: the sums
/// over both, anchored at the newer looking back and at the older looking
/// ahead, so a window's sums are always anchored at the row they are read
/// from or discounted to it once.
#[inline]
fn then(k: &KernelDef, a: Acc, a_at: f64, b: Acc, b_at: f64, var: bool) -> (Acc, f64) {
    let f = k.discount(b_at - a_at);
    match k.direction {
        Direction::Backward => (a.scaled(f, var).plus(b, var), b_at),
        Direction::Forward => (a.plus(b.scaled(f, var), var), a_at),
    }
}

/// Per row, per operator, in a stack's `vals`: where the held interval
/// starts, the value held (NaN for none), and the row's own sums.
const ROW: usize = 5;
/// Per row, per operator, in a stack's `sums`: the partial sums.
const SUM: usize = 3;
/// [`ROW`] and [`SUM`] in a queue holding a variance: its sums are six wide
/// ([`Acc`]), and every operator of the kernel takes the one width.
const ROW_VAR: usize = 8;
const SUM_VAR: usize = 6;

/// Rows with their values and partial sums in flat arenas, so a row costs
/// no allocation: one value and one accumulator per operator, side by side.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Stack {
    n: usize,
    /// Whether the kernel holds a variance: its arenas' strides are
    /// [`ROW_VAR`] and [`SUM_VAR`], not [`ROW`] and [`SUM`].
    wide: bool,
    seq: Vec<u64>,
    tau: Vec<f64>,
    /// The row's raw clock in integer nanoseconds, 0 without a temporal
    /// clock: what a window's edge is decided from (review R2, W1).
    off: Vec<i64>,
    end: Vec<f64>,
    /// Stride `row_w() * n`.
    vals: Vec<f64>,
    /// Stride `sum_w() * n`, anchored at `at`.
    sums: Vec<f64>,
    at: Vec<f64>,
}

impl Stack {
    fn new(n: usize, wide: bool) -> Self {
        Stack {
            n,
            wide,
            ..Default::default()
        }
    }

    #[inline]
    fn row_w(&self) -> usize {
        if self.wide { ROW_VAR } else { ROW }
    }

    #[inline]
    fn sum_w(&self) -> usize {
        if self.wide { SUM_VAR } else { SUM }
    }

    #[inline]
    fn len(&self) -> usize {
        self.seq.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    fn clear(&mut self) {
        self.seq.clear();
        self.tau.clear();
        self.off.clear();
        self.end.clear();
        self.vals.clear();
        self.sums.clear();
        self.at.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, seq: u64, tau: f64, off: i64, end: f64, row: &[f64], sums: &[f64], at: f64) {
        self.seq.push(seq);
        self.tau.push(tau);
        self.off.push(off);
        self.end.push(end);
        self.vals.extend_from_slice(row);
        self.sums.extend_from_slice(sums);
        self.at.push(at);
    }

    fn pop(&mut self) {
        let i = self.len() - 1;
        self.seq.truncate(i);
        self.tau.truncate(i);
        self.off.truncate(i);
        self.end.truncate(i);
        self.vals.truncate(i * self.row_w() * self.n);
        self.sums.truncate(i * self.sum_w() * self.n);
        self.at.truncate(i);
    }

    #[inline]
    fn start(&self, i: usize, j: usize) -> f64 {
        self.vals[(i * self.n + j) * self.row_w()]
    }

    #[inline]
    fn x(&self, i: usize, j: usize) -> f64 {
        self.vals[(i * self.n + j) * self.row_w() + 1]
    }

    #[inline]
    fn own(&self, i: usize, j: usize) -> Acc {
        let b = (i * self.n + j) * self.row_w() + 2;
        Acc::read(&self.vals[b..], self.wide)
    }

    #[inline]
    fn sum(&self, i: usize, j: usize) -> Acc {
        let b = (i * self.n + j) * self.sum_w();
        Acc::read(&self.sums[b..], self.wide)
    }

    /// Whether every arena is as long as the rows say, at `n` operators of
    /// this width: what the accessors above index by ([`Windows::check`]).
    fn has_width(&self, n: usize, wide: bool) -> bool {
        let len = self.seq.len();
        self.n == n
            && self.wide == wide
            && [
                self.tau.len(),
                self.off.len(),
                self.end.len(),
                self.at.len(),
            ]
            .iter()
            .all(|&l| l == len)
            && len.checked_mul(self.row_w() * n) == Some(self.vals.len())
            && len.checked_mul(self.sum_w() * n) == Some(self.sums.len())
    }
}

/// Write `a` into `out`'s slot for operator `j`, at a queue's width.
#[inline]
fn put(out: &mut [f64], j: usize, a: Acc, wide: bool) {
    let w = if wide { SUM_VAR } else { SUM };
    a.write(&mut out[j * w..], wide);
}

/// The two-stack queue. `front` holds the older rows, the oldest on top,
/// each beside the sums from itself through the newest row in `front`;
/// `back` holds the newer rows in order, each beside the sums from the
/// oldest row in `back` through itself. The window's sums are `front`'s
/// top then `back`'s last. Reaching the oldest row with `front` empty first
/// turns `back` over into it, so each row is moved once: O(1) a row on
/// average, and exact. The partial sums also give the window less its
/// oldest rows without a subtraction, which is what lets a held value be
/// cut at the window's edge without cancellation.
///
/// A backward kernel with no window evicts nothing, so it keeps only
/// `total`, the running sums.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Queue {
    n: usize,
    /// Per operator: whether its sums are a variance's, merged rather than
    /// added ([`merge`]). A queue holding one is wide.
    var: Vec<bool>,
    front: Stack,
    back: Stack,
    /// Stride `sum_w() * n`; empty until the first row.
    total: Vec<f64>,
    total_at: f64,
}

impl Queue {
    fn new(var: Vec<bool>) -> Self {
        let (n, wide) = (var.len(), var.contains(&true));
        Queue {
            n,
            var,
            front: Stack::new(n, wide),
            back: Stack::new(n, wide),
            total: Vec::new(),
            total_at: 0.0,
        }
    }

    #[inline]
    fn wide(&self) -> bool {
        self.front.wide
    }

    /// A row's width per operator in this queue's arenas.
    #[inline]
    fn row_w(&self) -> usize {
        self.front.row_w()
    }

    #[inline]
    fn sum_w(&self) -> usize {
        self.front.sum_w()
    }

    /// Push a row: `row` is its `row_w() * n` values, `scratch` a buffer of
    /// `sum_w() * n` for the sums.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        k: &KernelDef,
        seq: u64,
        tau: f64,
        off: i64,
        end: f64,
        row: &[f64],
        scratch: &mut [f64],
    ) {
        let (n, wide, rw) = (self.n, self.wide(), self.row_w());
        // The buffers are sized for the widest kernel; this one's part.
        let scratch = &mut scratch[..self.sum_w() * n];
        let own = |j: usize| Acc::read(&row[j * rw + 2..], wide);
        if k.window_size.is_some() {
            let mut at = tau;
            match self.back.len().checked_sub(1) {
                None => {
                    for j in 0..n {
                        put(scratch, j, own(j), wide);
                    }
                }
                Some(last) => {
                    let last_at = self.back.at[last];
                    for j in 0..n {
                        let (a, a_at) =
                            then(k, self.back.sum(last, j), last_at, own(j), tau, self.var[j]);
                        put(scratch, j, a, wide);
                        at = a_at;
                    }
                }
            }
            self.back.push(seq, tau, off, end, row, scratch, at);
        } else if self.total.is_empty() {
            self.total.resize(self.sum_w() * n, 0.0);
            for j in 0..n {
                put(&mut self.total, j, own(j), wide);
            }
            self.total_at = tau;
        } else {
            let mut at = tau;
            let sw = self.sum_w();
            for j in 0..n {
                let t = Acc::read(&self.total[j * sw..], wide);
                let (a, a_at) = then(k, t, self.total_at, own(j), tau, self.var[j]);
                put(&mut self.total, j, a, wide);
                at = a_at;
            }
            self.total_at = at;
        }
    }

    /// Turn `back` over into `front` when `front` is empty, so the oldest
    /// row is `front`'s top.
    fn settle(&mut self, k: &KernelDef, scratch: &mut [f64]) {
        if !self.front.is_empty() {
            return;
        }
        let (n, wide, rw) = (self.n, self.wide(), self.row_w());
        let scratch = &mut scratch[..self.sum_w() * n];
        while let Some(i) = self.back.len().checked_sub(1) {
            let (seq, tau, off, end) = (
                self.back.seq[i],
                self.back.tau[i],
                self.back.off[i],
                self.back.end[i],
            );
            let row_base = i * rw * n;
            let mut at = tau;
            match self.front.len().checked_sub(1) {
                None => {
                    for j in 0..n {
                        put(scratch, j, self.back.own(i, j), wide);
                    }
                }
                Some(newer) => {
                    let newer_at = self.front.at[newer];
                    for j in 0..n {
                        let (a, a_at) = then(
                            k,
                            self.back.own(i, j),
                            tau,
                            self.front.sum(newer, j),
                            newer_at,
                            self.var[j],
                        );
                        put(scratch, j, a, wide);
                        at = a_at;
                    }
                }
            }
            let (front, back) = (&mut self.front, &mut self.back);
            front.push(
                seq,
                tau,
                off,
                end,
                &back.vals[row_base..row_base + rw * n],
                scratch,
                at,
            );
            back.pop();
        }
    }

    /// Pop the rows `stale` says have left the window, oldest first.
    fn evict(&mut self, k: &KernelDef, scratch: &mut [f64], stale: impl Fn(u64, f64, i64) -> bool) {
        loop {
            self.settle(k, scratch);
            let Some(top) = self.front.len().checked_sub(1) else {
                return;
            };
            if !stale(
                self.front.seq[top],
                self.front.tau[top],
                self.front.off[top],
            ) {
                return;
            }
            self.front.pop();
        }
    }

    /// The window's sums for operator `j`, with their anchor.
    fn sum_op(&self, k: &KernelDef, j: usize) -> Option<(Acc, f64)> {
        if k.window_size.is_none() {
            if self.total.is_empty() {
                return None;
            }
            return Some((
                Acc::read(&self.total[j * self.sum_w()..], self.wide()),
                self.total_at,
            ));
        }
        let f = self
            .front
            .len()
            .checked_sub(1)
            .map(|t| (self.front.sum(t, j), self.front.at[t]));
        let b = self
            .back
            .len()
            .checked_sub(1)
            .map(|l| (self.back.sum(l, j), self.back.at[l]));
        match (f, b) {
            (Some((a, a_at)), Some((b, b_at))) => Some(then(k, a, a_at, b, b_at, self.var[j])),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    /// The window's sums for operator `j` less its `p + 1` oldest rows.
    /// Where those reach into `back`, whose sums are prefixes, the rest is
    /// summed row by row: rare, as the rows before an operator's oldest
    /// valued one carry nothing of it.
    fn sum_after_op(&self, k: &KernelDef, p: usize, j: usize) -> Option<(Acc, f64)> {
        let var = self.var[j];
        let n = self.front.len();
        if p + 1 < n {
            let i = n - 2 - p;
            let a = (self.front.sum(i, j), self.front.at[i]);
            return Some(match self.back.len().checked_sub(1) {
                Some(l) => then(k, a.0, a.1, self.back.sum(l, j), self.back.at[l], var),
                None => a,
            });
        }
        let from = p + 1 - n;
        let mut sum: Option<(Acc, f64)> = None;
        for i in from..self.back.len() {
            let own = (self.back.own(i, j), self.back.tau[i]);
            sum = Some(match sum {
                None => own,
                Some((a, a_at)) => then(k, a, a_at, own.0, own.1, var),
            });
        }
        sum
    }

    /// The first row from the oldest that `pred` takes, as its position
    /// from the oldest, the stack it is in and its index there.
    fn find_oldest(&self, pred: impl Fn(&Stack, usize) -> bool) -> Option<(usize, &Stack, usize)> {
        let n = self.front.len();
        for p in 0..n {
            let i = n - 1 - p;
            if pred(&self.front, i) {
                return Some((p, &self.front, i));
            }
        }
        for i in 0..self.back.len() {
            if pred(&self.back, i) {
                return Some((n + i, &self.back, i));
            }
        }
        None
    }

    fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
        self.total.clear();
    }

    fn len(&self) -> usize {
        self.front.len() + self.back.len()
    }

    /// Whether both stacks and the running sums are at these operators:
    /// their count, which are variances, and the width that gives.
    fn has_width(&self, var: &[bool]) -> bool {
        let (n, wide) = (var.len(), var.contains(&true));
        self.n == n
            && self.var == var
            && self.front.has_width(n, wide)
            && self.back.has_width(n, wide)
            && (self.total.is_empty() || self.total.len() == self.sum_w() * n)
    }
}

/// A row waiting for its group's forward kernels to close.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Wait {
    seq: u64,
    tau: f64,
    /// The row's raw clock in integer nanoseconds on a temporal clock; the
    /// bits of its value on a number clock, and of the policy time without
    /// a clock column (task 159, W3).
    off: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Group {
    clock: ClockState,
    /// Policy time of the group's last row, from its last restart.
    tau: f64,
    /// The raw clock the policy time is measured from: the first row of
    /// the stretch (review R1, C2).
    origin: Option<ClockValue>,
    /// The last row's raw clock in integer nanoseconds (a temporal clock),
    /// as its integer (an integer clock, docs/PLAN.md task 200), else the
    /// bits of the clock's value (of the policy time without a clock
    /// column), `form` saying which: a window's edge between two rows is
    /// decided from the difference of their raw clocks, never from two
    /// rounded policy times (review R2, W1; task 159, W3).
    off: i64,
    form: OffForm,
    /// The next row starts the windows over: the group has no row yet, or
    /// an event since its last row ended them.
    restart: bool,
    /// Per kernel.
    queues: Vec<Queue>,
    /// Per forward kernel: the last row, whose interval ends at the next
    /// row, as its place and time; its values are in `open_x` and the
    /// values held over its interval in `open_held`, `n` each.
    open: Vec<Option<(u64, f64, i64)>>,
    open_x: Vec<Vec<f64>>,
    open_held: Vec<Vec<f64>>,
    /// Rows whose forward windows are not all closed, oldest first.
    waiting: VecDeque<Wait>,
    /// Per kernel: how many of `waiting`, from the front, it has closed.
    /// Always 0 for a backward kernel.
    closed: Vec<usize>,
    /// Per operator of a backward kernel: the policy time of its last row
    /// with a value, NaN for none since the restart. A value is held from
    /// there, as `ewm_mean_by` skips a null.
    last_valued: Vec<f64>,
    /// Per operator of a forward kernel: the last value, NaN for none since
    /// the restart, held until the next.
    held: Vec<f64>,
    /// Rows at the current stamp whose backward windows under `"right"` or
    /// `"both"` wait for the next distinct stamp.
    pending: Vec<u64>,
    /// The raw clock of the group's last row and of `pending`.
    stamp: Option<ClockValue>,
    /// Per backward kernel under `"left"` or `"none"`, the values of the
    /// current stamp's window, per operator of the kernel, read before the
    /// stamp's rows joined.
    stamp_values: Vec<Vec<f64>>,
    /// Raw clock of the group's last row, while it is in the silent list.
    last_raw: Option<ClockValue>,
    /// Neighbours in the silent list, the oldest `last_raw` at the head.
    prev_link: Option<usize>,
    next_link: Option<usize>,
    linked: bool,
}

/// How a window over a row ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// Its span passed.
    Complete,
    /// A capped gap or a session change cut it short: partial.
    Cut,
    /// A reset discarded it: null, and never dropped.
    Discard,
}

/// Per row not yet emitted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Meta {
    /// Kernels not yet resolved over the row.
    open: u32,
    accept: bool,
    drop: bool,
}

/// Rows out, in input order: `rows` rows from `first_seq` on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Emitted {
    pub first_seq: u64,
    pub rows: usize,
    /// Per operator, per row: the value, NaN where null.
    pub values: Vec<Vec<f64>>,
    /// Per row: whether a `"drop"` operator was partial on it.
    pub drop: Vec<bool>,
}

/// The window core: every group's kernels over one stream, fed rows in
/// input order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Windows {
    kernels: Vec<KernelDef>,
    ops: Vec<OpDef>,
    #[serde(with = "clock_form")]
    clock_cfg: ClockCfg,
    /// Per kernel: its operators, in output order.
    members: Vec<Vec<usize>>,
    n_outputs: usize,
    n_forward: u32,
    groups: Vec<Group>,
    /// Group keys, as text, to their index in `groups`; the null key apart.
    #[serde(with = "sorted_map")]
    index: HashMap<String, usize>,
    null_group: Option<usize>,
    /// The stream's own clock, in input order across groups.
    shared: ClockState,
    /// Whether the rows carry a group column: then a session is each
    /// group's and the stream's clock takes none (review R2, W3); without
    /// one the stream is the one group, and its session is the stream's.
    grouped: bool,
    /// Groups with rows waiting, in the order of their last row: the oldest
    /// at the head.
    head: Option<usize>,
    tail: Option<usize>,
    /// The rows fed and not yet emitted, from `held_first` on: per row,
    /// `n_outputs` values (NaN null) and one meta.
    held_first: u64,
    held_values: VecDeque<f64>,
    held_meta: VecDeque<Meta>,
    /// How many held rows, from the first, every kernel has resolved, and
    /// how many of those no operator drops: what [`Windows::drain`] emits.
    ready: usize,
    ready_kept: usize,
    /// The `seq` the next row fed takes.
    next_seq: u64,
    /// Buffers a row's work reuses, sized for the widest kernel.
    #[serde(skip)]
    scratch_row: Vec<f64>,
    #[serde(skip)]
    scratch_sum: Vec<f64>,
}

/// A map as a sorted list, so the state's bytes do not depend on the
/// hasher.
mod sorted_map {
    use std::collections::HashMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &HashMap<String, usize>, s: S) -> Result<S::Ok, S::Error> {
        let mut v: Vec<(&String, &usize)> = m.iter().collect();
        v.sort();
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<String, usize>, D::Error> {
        Ok(Vec::<(String, usize)>::deserialize(d)?
            .into_iter()
            .collect())
    }
}

/// `ClockCfg` with its session gap tagged. The core type's own derive
/// writes `Some(SessionGap::Reset)` as a bare unit, which msgpack reads back
/// as `None`: a saved `session_gap = "reset"` would resume as no session
/// gap at all. The bank never saves a `ClockCfg` -- it rebuilds one from
/// its specs -- so this state is the first to need the form.
mod clock_form {
    use online_core::{ClockCfg, OnClockReset, SessionGap};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Form {
        gap_cap: f64,
        on_clock_reset: OnClockReset,
        session_gap: GapForm,
        min_backwards_jump: f64,
    }

    #[derive(Serialize, Deserialize)]
    enum GapForm {
        Off,
        Gap(f64),
        Reset,
    }

    pub fn serialize<S: Serializer>(c: &ClockCfg, s: S) -> Result<S::Ok, S::Error> {
        Form {
            gap_cap: c.gap_cap,
            on_clock_reset: c.on_clock_reset,
            session_gap: match c.session_gap {
                None => GapForm::Off,
                Some(SessionGap::Gap(g)) => GapForm::Gap(g),
                Some(SessionGap::Reset) => GapForm::Reset,
            },
            min_backwards_jump: c.min_backwards_jump,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ClockCfg, D::Error> {
        let f = Form::deserialize(d)?;
        Ok(ClockCfg {
            gap_cap: f.gap_cap,
            on_clock_reset: f.on_clock_reset,
            session_gap: match f.session_gap {
                GapForm::Off => None,
                GapForm::Gap(g) => Some(SessionGap::Gap(g)),
                GapForm::Reset => Some(SessionGap::Reset),
            },
            min_backwards_jump: f.min_backwards_jump,
        })
    }
}

/// `now` minus `last` in clock units: between two temporal values, taken in
/// integer nanoseconds and rounded once, as the clock takes its steps.
/// The clock between two rows of one stretch, for a window's edge: the
/// exact difference of their raw clocks in integer nanoseconds on a
/// temporal clock (`exact`), else of the clocks carried as bits in `off`
/// (a number clock's own value; the policy time without a clock column),
/// never of two origin-subtracted policy times (task 159, W3).
/// Two rounded policy times differ by rounding, and a row exactly one
/// window after another landed on either side of the edge (review R2, W1:
/// 0.4 − 0.1 is not 0.3 in a double). Against a window given as a duration
/// the comparison is between integers ([`KernelDef::window_ns`]); against
/// one given as a number the gap is converted to seconds with the same two
/// operations a duration's text is (`seconds_of_ns`), so a gap of exactly
/// one window still compares equal to it.
#[derive(Clone, Copy, Debug)]
enum Gap {
    Ns(i64),
    Secs(f64),
    /// An integer clock's gap in the column's own units, exact (task 200).
    Int(i128),
}

/// What a row's `off` holds: a temporal clock's nanoseconds, an integer
/// clock's value (docs/PLAN.md task 200), or the bits of a double -- a float
/// clock's value, or the policy time without a clock column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum OffForm {
    Ns,
    Int,
    Bits,
}

#[inline]
fn gap(form: OffForm, from_off: i64, to_off: i64) -> Gap {
    match form {
        OffForm::Ns => Gap::Ns(to_off.saturating_sub(from_off)),
        OffForm::Int => Gap::Int(i128::from(to_off) - i128::from(from_off)),
        OffForm::Bits => Gap::Secs(f64::from_bits(to_off as u64) - f64::from_bits(from_off as u64)),
    }
}

impl Gap {
    /// The gap against the kernel's window, by the comparison a model's
    /// window decides its edge with ([`Stamp::cmp_span_ns`], task 175): in
    /// integer nanoseconds against a window written as a duration, through
    /// `seconds_of_ns` against one written as a number, and on a number
    /// clock the one subtraction of the raw values. No answer (a NaN) is
    /// `Equal`.
    #[inline]
    fn cmp_window(self, k: &KernelDef) -> Ordering {
        let w = k.window_size.expect("a windowed kernel");
        self.cmp_span(w, k.window_ns.map(i128::from))
    }

    /// The gap against zero, by the same comparison.
    #[inline]
    fn cmp_zero(self) -> Ordering {
        self.cmp_span(0.0, Some(0))
    }

    /// The gap as the newer of two stamps, the older at zero.
    #[inline]
    fn cmp_span(self, span: f64, span_ns: Option<i128>) -> Ordering {
        match self {
            Gap::Ns(n) => Stamp::Ns(i128::from(n)).cmp_span_ns(Stamp::Ns(0), span, span_ns),
            Gap::Secs(s) => Stamp::Raw(s, 0.0).cmp_span_ns(Stamp::Raw(0.0, 0.0), span, None),
            Gap::Int(n) => Stamp::Int(n, 0.0).cmp_span_ns(Stamp::Int(0, 0.0), span, None),
        }
    }
}

/// `now` minus `last` in clock units: between two temporal values, taken
/// in integer nanoseconds and rounded once; between two integer ones, in
/// integers and rounded once (task 200).
fn elapsed(now: ClockValue, last: ClockValue) -> f64 {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
        (c, p) => match c.int_step(p) {
            Some(step) => step as f64,
            None => c.seconds() - p.seconds(),
        },
    }
}

/// Whether `now` is more than `cap` clock units after `last`: on an
/// integer clock their difference in integers against the cap, exactly
/// (task 200), and as [`elapsed`] gives it otherwise.
fn past_cap(now: ClockValue, last: ClockValue, cap: f64) -> bool {
    match now.int_step(last) {
        Some(step) => online_core::cmp_int_f64(step, cap) == Some(Ordering::Greater),
        None => elapsed(now, last) > cap,
    }
}

fn refuse_backwards(
    adv: &ClockAdvance,
    seq: u64,
    prev: Option<ClockValue>,
    now: Option<ClockValue>,
) -> Result<(), Refusal> {
    match adv.backwards {
        Some(raw) => Err(Refusal::Backwards {
            seq,
            back: -raw,
            prev,
            now,
            min_backwards_jump: adv.disorder.map(|d| d.min_backwards_jump),
        }),
        None => Ok(()),
    }
}

/// `∫_a^b λ^(T - s) ds` for a backward kernel, `∫_a^b λ^(s - T) ds` for a
/// forward one: the decayed mass of `[a, b]` seen from `T`, `a <= b`.
#[inline]
fn mass_between(k: &KernelDef, t: f64, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    match k.direction {
        // ∫_a^b λ^(T - s) ds = M(T - a) - M(T - b)
        Direction::Backward => k.mass(t - a) - k.mass(t - b),
        // ∫_a^b λ^(s - T) ds = M(b - T) - M(a - T)
        Direction::Forward => k.mass(b - t) - k.mass(a - t),
    }
}

/// Whether operator `o` can run on its kernel: a mean or a variance with
/// no decay needs a window, and a variance looks back only (task 212).
fn check_op(o: usize, op: &OpDef, k: &KernelDef) -> Result<(), String> {
    if op.stat.weighs_held() && k.half_life.is_infinite() && k.window_size.is_none() {
        return Err(format!(
            "operator {o}: a mean or a variance with half_life = inf needs a window_size; over \
             an unbounded stretch every value weighs the same and the mean is not a number"
        ));
    }
    if op.stat.is_var() && k.direction == Direction::Forward {
        return Err(format!(
            "operator {o}: a variance looks back only; it has no forward kernel"
        ));
    }
    Ok(())
}

/// The rows not yet emitted, borrowed apart from the groups.
struct Held<'a> {
    first: u64,
    n_outputs: usize,
    values: &'a mut VecDeque<f64>,
    meta: &'a mut VecDeque<Meta>,
}

impl Windows {
    /// # Errors
    ///
    /// No operator, or a kernel this core cannot run.
    pub fn new(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        if ops.is_empty() {
            return Err("at least one operator is needed".into());
        }
        for k in &kernels {
            k.check()?;
        }
        let mut members = vec![Vec::new(); kernels.len()];
        for (o, op) in ops.iter().enumerate() {
            if op.kernel >= kernels.len() {
                return Err(format!(
                    "operator {o} names kernel {}, which there is not",
                    op.kernel
                ));
            }
            if op.min_samples == 0 {
                return Err("min_samples must be at least 1".into());
            }
            let k = &kernels[op.kernel];
            check_op(o, op, k)?;
            members[op.kernel].push(o);
        }
        if let Some(k) = members.iter().position(Vec::is_empty) {
            return Err(format!("kernel {k} has no operator"));
        }
        let widest = members.iter().map(Vec::len).max().unwrap_or(0);
        let n_forward = kernels
            .iter()
            .filter(|k| k.direction == Direction::Forward)
            .count();
        Ok(Self {
            n_outputs: ops.len(),
            n_forward: u32::try_from(n_forward).map_err(|_| "too many kernels")?,
            kernels,
            ops,
            clock_cfg,
            members,
            groups: Vec::new(),
            index: HashMap::new(),
            null_group: None,
            shared: ClockState::new(),
            grouped: false,
            head: None,
            tail: None,
            held_first: 0,
            held_values: VecDeque::new(),
            held_meta: VecDeque::new(),
            ready: 0,
            ready_kept: 0,
            next_seq: 0,
            scratch_row: vec![0.0; ROW_VAR * widest],
            scratch_sum: vec![0.0; SUM_VAR * widest],
        })
    }

    /// Per kernel, which of its operators are variances: what its queue is
    /// built with, and checked against.
    fn var_of(&self, ki: usize) -> Vec<bool> {
        self.members[ki]
            .iter()
            .map(|&o| self.ops[o].stat.is_var())
            .collect()
    }

    /// Say whether the rows carry a group column (review R2, W3): with
    /// one, a session is each group's and the stream's clock takes none;
    /// without, the stream is the one group, and its session change ends
    /// the windows as the one group's clock sees it.
    pub fn set_grouped(&mut self, grouped: bool) {
        self.grouped = grouped;
    }

    /// Whether a core read back from a state file holds what every method
    /// here indexes by (review 2026-10-06, PD1): the loader compared the
    /// kernels, the operators and the clock with its own call's and the
    /// held rows' count with the frame's, and took the rest on trust, so a
    /// damaged file that still decoded panicked at the next call. The rest:
    /// each kernel's operators and the output count are the operators',
    /// the held values `n_outputs` a held row, the rows ready among those
    /// held, each group at its kernels' widths with its waiting rows held
    /// and still open by exactly the windows left to close over them, the
    /// silent list a list of the groups, and the keys the groups one to
    /// one. `Err` names the first that fails.
    ///
    /// # Errors
    ///
    /// A core that is none this module could have built and fed.
    pub fn check(&self) -> Result<(), String> {
        for k in &self.kernels {
            k.check()?;
        }
        let (nk, no) = (self.kernels.len(), self.ops.len());
        let mut members = vec![Vec::new(); nk];
        for (o, op) in self.ops.iter().enumerate() {
            match members.get_mut(op.kernel) {
                Some(m) => m.push(o),
                None => return Err(format!("operator {o} names kernel {}", op.kernel)),
            }
            check_op(o, op, &self.kernels[op.kernel])?;
        }
        if no == 0 || members != self.members || members.iter().any(Vec::is_empty) {
            return Err("its kernels' operators are not its operators".into());
        }
        if self.n_outputs != no {
            return Err(format!("{} outputs for {no} operators", self.n_outputs));
        }
        let is_forward = |k: &KernelDef| k.direction == Direction::Forward;
        let forward = self.kernels.iter().filter(|k| is_forward(k)).count();
        if self.n_forward as usize != forward {
            return Err(format!("{} forward kernels of {forward}", self.n_forward));
        }
        let held = self.held_meta.len();
        if held.checked_mul(no) != Some(self.held_values.len()) {
            return Err(format!(
                "{} values held for {held} rows of {no} outputs",
                self.held_values.len()
            ));
        }
        let kept = self.held_meta.iter().take(self.ready).filter(|m| !m.drop);
        if self.ready > held || self.ready_kept != kept.count() {
            return Err(format!("{} rows ready of {held} held", self.ready));
        }
        if self.held_first.checked_add(held as u64) != Some(self.next_seq) {
            return Err(format!(
                "{held} rows held from row {} before row {}",
                self.held_first, self.next_seq
            ));
        }
        // Each group at its kernels' widths, and the windows still open over
        // each held row, which `close` and `flush_group` count down.
        let n_wait = u32::from(
            self.kernels
                .iter()
                .any(|k| k.direction == Direction::Backward && k.closed.near(Direction::Backward)),
        );
        let is_held = |seq: u64| seq >= self.held_first && seq < self.next_seq;
        let widths = |v: &[Vec<f64>]| {
            v.len() == nk && v.iter().zip(&self.members).all(|(x, m)| x.len() == m.len())
        };
        let mut open = vec![0u32; held];
        let var: Vec<Vec<bool>> = (0..nk).map(|ki| self.var_of(ki)).collect();
        for (gi, g) in self.groups.iter().enumerate() {
            let fits = g.queues.len() == nk
                && g.queues.iter().zip(&var).all(|(q, v)| q.has_width(v))
                && g.open.len() == nk
                && widths(&g.open_x)
                && widths(&g.open_held)
                && widths(&g.stamp_values)
                && g.closed.len() == nk
                && g.closed.iter().all(|&c| c <= g.waiting.len())
                && g.last_valued.len() == no
                && g.held.len() == no
                && (forward > 0 || g.waiting.is_empty())
                && (n_wait > 0 || g.pending.is_empty())
                && g.waiting.iter().all(|w| is_held(w.seq))
                && g.pending.iter().all(|&s| is_held(s))
                && (!g.linked || g.last_raw.is_some());
            if !fits {
                return Err(format!("group {gi} does not fit its kernels and held rows"));
            }
            for (i, w) in g.waiting.iter().enumerate() {
                let r = (w.seq - self.held_first) as usize;
                let still = self.kernels.iter().zip(&g.closed);
                open[r] += still.filter(|&(k, &c)| is_forward(k) && i >= c).count() as u32;
            }
            for &s in &g.pending {
                open[(s - self.held_first) as usize] += n_wait;
            }
        }
        if self.held_meta.iter().zip(&open).any(|(m, &o)| m.open != o) {
            return Err("a held row's open windows are not the ones still open over it".into());
        }
        self.check_links()?;
        // The keys name the groups one to one: `group` makes each group
        // under its key, and `push` reads a group by the index a key gives.
        let mut named = vec![false; self.groups.len()];
        for &gi in self.index.values().chain(&self.null_group) {
            match named.get_mut(gi) {
                Some(seen) if !*seen => *seen = true,
                _ => return Err("its keys do not name its groups one to one".into()),
            }
        }
        if named.contains(&false) {
            return Err("its keys do not name its groups one to one".into());
        }
        Ok(())
    }

    /// The silent list: links inside the groups, from the head through each
    /// linked group once, back links that agree, the tail last, and no
    /// unlinked group holding a link ([`Self::check`]).
    fn check_links(&self) -> Result<(), String> {
        let n = self.groups.len();
        let bad = || Err("its silent list is not a list of its groups".to_string());
        let inside = |l: Option<usize>| l.is_none_or(|i| i < n);
        let mut links = self.groups.iter().flat_map(|g| [g.prev_link, g.next_link]);
        if !inside(self.head) || !inside(self.tail) || !links.all(inside) {
            return bad();
        }
        let (mut at, mut prev, mut walked) = (self.head, None, 0);
        while let Some(i) = at {
            let g = &self.groups[i];
            if walked == n || !g.linked || g.prev_link != prev {
                return bad();
            }
            (prev, at, walked) = (Some(i), g.next_link, walked + 1);
        }
        let linked = self.groups.iter().filter(|g| g.linked).count();
        let stray = self
            .groups
            .iter()
            .any(|g| !g.linked && (g.prev_link.is_some() || g.next_link.is_some()));
        if prev != self.tail || walked != linked || stray {
            return bad();
        }
        Ok(())
    }

    /// One named damage, as a bit flip or a truncated array in a state file
    /// would leave the core: for the loader's test in `windows_frame.rs`,
    /// the fields being this module's.
    #[cfg(test)]
    pub(crate) fn damage(&mut self, what: &str) {
        match what {
            "held_values" => {
                self.held_values.pop_back();
            }
            "n_outputs" => self.n_outputs = 0,
            "queue" => self.groups[0].queues[0].n += 1,
            "link" => self.head = Some(self.groups.len()),
            "ready" => self.ready = self.held_meta.len() + 1,
            "waiting" => {
                let past = self.held_first + self.held_meta.len() as u64;
                self.groups[0].waiting[0].seq = past;
            }
            other => unreachable!("no damage {other:?}"),
        }
    }

    /// The scratch buffers, which a loaded state lacks.
    fn scratch(&mut self) {
        if self.scratch_sum.is_empty() {
            let widest = self.members.iter().map(Vec::len).max().unwrap_or(0);
            self.scratch_row = vec![0.0; ROW_VAR * widest];
            self.scratch_sum = vec![0.0; SUM_VAR * widest];
        }
    }

    pub fn kernels(&self) -> &[KernelDef] {
        &self.kernels
    }

    pub fn ops(&self) -> &[OpDef] {
        &self.ops
    }

    pub fn clock_cfg(&self) -> &ClockCfg {
        &self.clock_cfg
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// The stream's own clock: what the last row left it at.
    pub fn shared_clock(&self) -> &ClockState {
        &self.shared
    }

    /// Rows fed and not yet emitted.
    pub fn held(&self) -> usize {
        self.held_meta.len()
    }

    /// Contributing rows every queue holds, summed: what the windows cost.
    pub fn queued(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| g.queues.iter())
            .map(Queue::len)
            .sum()
    }

    /// The `seq` the next row fed takes: the rows fed so far, across a save
    /// and a load.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// The index of the group with this key, as text; a new key makes a new
    /// group. Only a group's first row allocates.
    pub fn group(&mut self, key: Option<&str>) -> usize {
        let found = match key {
            Some(k) => self.index.get(k).copied(),
            None => self.null_group,
        };
        if let Some(gi) = found {
            return gi;
        }
        let gi = self.groups.len();
        let queues = (0..self.kernels.len())
            .map(|ki| Queue::new(self.var_of(ki)))
            .collect();
        self.groups.push(Group {
            clock: ClockState::new(),
            tau: 0.0,
            origin: None,
            off: 0,
            form: OffForm::Bits,
            restart: true,
            queues,
            open: vec![None; self.kernels.len()],
            open_x: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            open_held: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            waiting: VecDeque::new(),
            closed: vec![0; self.kernels.len()],
            last_valued: vec![f64::NAN; self.ops.len()],
            held: vec![f64::NAN; self.ops.len()],
            pending: Vec::new(),
            stamp: None,
            stamp_values: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            last_raw: None,
            prev_link: None,
            next_link: None,
            linked: false,
        });
        match key {
            Some(k) => {
                self.index.insert(k.to_string(), gi);
            }
            None => self.null_group = Some(gi),
        }
        gi
    }

    /// Both clocks stepped on copies by a row's clock and session, for the
    /// caller to keep once neither refuses it. With groups a session is
    /// each group's: the stream's clock keeps the stream's order and its
    /// gaps and takes no session (review R2, W3: groups with sessions of
    /// their own restarted every group at every row); without, the stream
    /// is the one group. `gi` is `None` for a group with no row yet.
    fn step_clocks(
        &self,
        gi: Option<usize>,
        clock: Option<ClockValue>,
        session: Option<u64>,
    ) -> Result<(ClockState, ClockAdvance, ClockState, ClockAdvance), Refusal> {
        let seq = self.next_seq;
        let stream_session = if self.grouped { None } else { session };
        let mut shared = self.shared.clone();
        let adv = shared.advance(&self.clock_cfg, clock, stream_session, true);
        refuse_backwards(&adv, seq, self.shared.last_clock(), clock)?;
        // A restart on the stream's clock starts every group over, its clock
        // included: this row is the first on a fresh clock of its group, not
        // a step back on a stale one (task 159, W1: a group's next row within
        // `restart_after_step_back` of its last was refused as a late row
        // after the stream had restarted, where its windows had started over).
        let mut own_clock = if adv.reset {
            ClockState::new()
        } else {
            gi.map_or_else(ClockState::new, |g| self.groups[g].clock.clone())
        };
        let prev = own_clock.last_clock();
        let own = own_clock.advance(&self.clock_cfg, clock, session, true);
        refuse_backwards(&own, seq, prev, clock)?;
        Ok((shared, adv, own_clock, own))
    }

    /// What the clocks would make of a row with this group key, clock and
    /// session, without taking it: refused as a step back the policy
    /// refuses, or taken as a new start, a step forward or the same stamp.
    /// What a run resumed on an input that starts elsewhere asks of the
    /// input's first row (review R6, D3 and D4): a new start or a step
    /// forward is the next file.
    pub fn peek(
        &self,
        key: Option<&str>,
        clock: Option<ClockValue>,
        session: Option<u64>,
    ) -> Result<Peek, Refusal> {
        let gi = match key {
            Some(k) => self.index.get(k).copied(),
            None => self.null_group,
        };
        let (_, adv, _, own) = self.step_clocks(gi, clock, session)?;
        if adv.reset || adv.session_changed || own.reset || own.session_changed {
            return Ok(Peek::NewStart);
        }
        match (clock, self.shared.last_clock()) {
            (Some(now), Some(last)) if !now.is_before(last) && !last.is_before(now) => {
                Ok(Peek::SameStamp)
            }
            _ => Ok(Peek::Forward),
        }
    }

    /// Feed one row. Earlier rows may be resolved by it; take them with
    /// [`Self::drain`].
    ///
    /// # Errors
    ///
    /// A step back the clock policy refuses, on the stream's clock or on
    /// the group's. A refused row changes nothing.
    pub fn push(&mut self, row: &RowIn<'_>) -> Result<(), Refusal> {
        self.scratch();
        let seq = self.next_seq;
        let gi = row.group;
        // Both clocks step on copies, kept only once neither refuses the row.
        let (shared, adv, clock, own) = self.step_clocks(Some(gi), row.clock, row.session)?;
        self.shared = shared;
        self.next_seq += 1;
        // An event on the stream's clock reaches every group now; each
        // group's own clock would show it at that group's next row.
        if adv.reset {
            self.end_all(End::Discard);
            // Every group's clock too (task 159, W1): the row's own group
            // takes the fresh one stepped above.
            for g in &mut self.groups {
                g.clock = ClockState::new();
            }
        } else if adv.session_changed || adv.capped {
            self.end_all(End::Cut);
        }
        if let Some(now) = row.clock {
            self.end_silent(now);
        }
        if own.reset {
            self.end_group(gi, End::Discard);
        } else if own.session_changed || own.capped {
            self.end_group(gi, End::Cut);
        }

        // A new stamp resolves the stamps before it, in every group: the
        // stream is in clock order across groups.
        if let Some(now) = row.clock {
            self.flush_stamps_before(now);
        }

        self.held_values
            .extend(std::iter::repeat_n(f64::NAN, self.n_outputs));
        let has_clock = row.clock.is_some();
        // Under "right" and "both" a backward window waits for the next
        // distinct stamp; a row-count clock repeats no stamp, so the row's
        // own windows are read at the end of this push (review R1, C1: they
        // were never read without a clock, and every such output was null).
        let n_wait = u32::from(
            self.kernels
                .iter()
                .any(|k| k.direction == Direction::Backward && k.closed.near(Direction::Backward)),
        );
        self.held_meta.push_back(Meta {
            open: self.n_forward + n_wait,
            accept: row.accept,
            drop: false,
        });

        let g = &mut self.groups[gi];
        g.clock = clock;
        let restarted = own.reset || own.session_changed || own.capped || g.restart;
        if restarted {
            // Every window of the group was ended above or before: nothing
            // refers to the old policy time, so it starts over at 0.
            for q in &mut g.queues {
                q.clear();
            }
            g.open.iter_mut().for_each(|o| *o = None);
            g.last_valued.iter_mut().for_each(|v| *v = f64::NAN);
            g.held.iter_mut().for_each(|v| *v = f64::NAN);
            g.tau = 0.0;
            g.origin = row.clock;
            g.restart = false;
            g.stamp = None;
            g.pending.clear();
        } else if let (Some(now), Some(origin)) = (row.clock, g.origin) {
            // From the stretch's origin in one subtraction (exact in
            // integer nanoseconds), not a sum of rounded steps: summed, a
            // row exactly one window later landed on either side of the
            // edge by the sum's noise (review R1, C2).
            g.tau = elapsed(now, origin);
        } else {
            g.tau += own.d_clock;
            if g.origin.is_none() {
                g.origin = row.clock;
            }
        }
        // The raw clock a window's edge is decided from (review R2, W1): the
        // nanoseconds of a temporal clock, exact; the bits of a number
        // clock, so the edge is the difference of the two rows' clocks, as
        // Polars takes it, not of two origin-subtracted policy times, which
        // put a row exactly one window back outside the edge, `0.4 − 0.1`
        // not being `0.3` (task 159, W3); the policy time's bits without a
        // clock column, where the two are one.
        (g.off, g.form) = match row.clock {
            Some(ClockValue::Ns(c)) => (c, OffForm::Ns),
            Some(ClockValue::I64(c)) => (c, OffForm::Int),
            Some(ClockValue::F64(c)) => (c.to_bits() as i64, OffForm::Bits),
            None => (g.tau.to_bits() as i64, OffForm::Bits),
        };
        let (tau, off, form) = (g.tau, g.off, g.form);
        let new_stamp = g.stamp != row.clock || !has_clock;

        // Forward kernels: the windows this row reaches past close first --
        // it is not in them -- with the kernel's open row, the last one,
        // counted up to here; then that row joins the queue, its interval
        // ended by this one, and this row is the open one.
        for (ki, k) in self.kernels.iter().enumerate() {
            if k.direction != Direction::Forward {
                continue;
            }
            let w = k.window_size.expect("a forward kernel has a window");
            // A row exactly a window later is in it under "right" and
            // "both", so the window closes only past it.
            while g.closed[ki] < g.waiting.len() && {
                let t = g.waiting[g.closed[ki]];
                let d = gap(form, t.off, off).cmp_window(k);
                if k.closed.far(Direction::Forward) {
                    d == Ordering::Greater
                } else {
                    d != Ordering::Less
                }
            } {
                let t = g.waiting[g.closed[ki]];
                g.closed[ki] += 1;
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    values: &mut self.held_values,
                    meta: &mut self.held_meta,
                };
                close(
                    k,
                    &self.ops,
                    &self.members[ki],
                    &mut g.queues[ki],
                    &mut self.scratch_sum,
                    g.open[ki],
                    &g.open_x[ki],
                    &g.open_held[ki],
                    tau,
                    t,
                    End::Complete,
                    t.tau + w,
                    form,
                    held,
                );
            }
            if let Some((oseq, otau, ooff)) = g.open[ki].take() {
                let m = &self.members[ki];
                let rw = g.queues[ki].row_w();
                fill_forward(
                    k,
                    &self.ops,
                    m,
                    otau,
                    tau,
                    &g.open_x[ki],
                    &g.open_held[ki],
                    rw,
                    &mut self.scratch_row,
                );
                g.queues[ki].push(
                    k,
                    oseq,
                    otau,
                    ooff,
                    tau,
                    &self.scratch_row[..rw * m.len()],
                    &mut self.scratch_sum,
                );
            }
            // A value is held until the next: a row with none holds the last.
            for (j, &o) in self.members[ki].iter().enumerate() {
                let v = row.values[self.ops[o].input];
                let v = if usable(v) { v } else { f64::NAN };
                g.open_x[ki][j] = v;
                if !v.is_nan() {
                    g.held[o] = v;
                }
                g.open_held[ki][j] = g.held[o];
            }
            g.open[ki] = Some((seq, tau, off));
        }
        retire(&self.kernels, g);

        // Backward kernels: under "left" and "none" the stamp's window is
        // read before its rows join; then the row joins; under "right" and
        // "both" the stamp's rows wait for the next stamp.
        for (ki, k) in self.kernels.iter().enumerate() {
            if k.direction != Direction::Backward {
                continue;
            }
            let m = &self.members[ki];
            if !k.closed.near(Direction::Backward) && new_stamp {
                read_backward(
                    k,
                    &self.ops,
                    m,
                    &mut g.queues[ki],
                    &mut self.scratch_sum,
                    tau,
                    off,
                    form,
                    &mut g.stamp_values[ki],
                );
            }
            // A value is held from the operator's last valued row; the first
            // of a stretch from before it: from `-inf`, or from a window
            // back where one bounds the mass.
            let (rw, wide) = (g.queues[ki].row_w(), g.queues[ki].wide());
            for (j, &o) in m.iter().enumerate() {
                let v = row.values[self.ops[o].input];
                let v = if usable(v) { v } else { f64::NAN };
                let start = if g.last_valued[o].is_nan() {
                    k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w)
                } else {
                    g.last_valued[o]
                };
                let stat = self.ops[o].stat;
                let acc = if v.is_nan() {
                    Acc::default()
                } else if stat.weighs_held() {
                    Acc::held(v, mass_between(k, tau, start, tau), stat.is_var())
                } else {
                    Acc {
                        s: v,
                        n: 1.0,
                        ..Acc::default()
                    }
                };
                let b = j * rw;
                self.scratch_row[b] = start;
                self.scratch_row[b + 1] = v;
                acc.write(&mut self.scratch_row[b + 2..], wide);
                if !v.is_nan() {
                    g.last_valued[o] = tau;
                }
            }
            g.queues[ki].push(
                k,
                seq,
                tau,
                off,
                tau,
                &self.scratch_row[..rw * m.len()],
                &mut self.scratch_sum,
            );
            if !k.closed.near(Direction::Backward) {
                let complete = k.window_size.is_none_or(|w| tau >= w);
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    values: &mut self.held_values,
                    meta: &mut self.held_meta,
                };
                write_backward(&self.ops, m, seq, &g.stamp_values[ki], complete, held);
            }
        }
        if n_wait > 0 {
            g.pending.push(seq);
        }
        g.stamp = row.clock;

        if self.n_forward > 0 {
            g.waiting.push_back(Wait { seq, tau, off });
        }
        if let Some(now) = row.clock {
            g.last_raw = Some(now);
            self.link_back(gi);
        }
        if !has_clock {
            // A row-count clock repeats no stamp: the row's own backward
            // windows are known now.
            self.flush_group(gi);
        }
        self.advance_ready();
        Ok(())
    }

    /// Resolve the backward windows of every group's stamp before `now`.
    /// The silent list holds every group with a clock, oldest last row
    /// first, and the stream is in clock order across groups, so the
    /// groups behind `now` are its head.
    fn flush_stamps_before(&mut self, now: ClockValue) {
        let mut gi = self.head;
        while let Some(i) = gi {
            match self.groups[i].stamp {
                Some(s) if elapsed(now, s) > 0.0 => self.flush_group(i),
                _ => break,
            }
            gi = self.groups[i].next_link;
        }
    }

    /// Resolve the pending backward windows of group `gi` at its stamp.
    fn flush_group(&mut self, gi: usize) {
        let g = &mut self.groups[gi];
        if g.pending.is_empty() {
            return;
        }
        let (tau, off, form) = (g.tau, g.off, g.form);
        let pending = std::mem::take(&mut g.pending);
        for (ki, k) in self.kernels.iter().enumerate() {
            if k.direction != Direction::Backward || !k.closed.near(Direction::Backward) {
                continue;
            }
            let m = &self.members[ki];
            read_backward(
                k,
                &self.ops,
                m,
                &mut g.queues[ki],
                &mut self.scratch_sum,
                tau,
                off,
                form,
                &mut g.stamp_values[ki],
            );
            let complete = k.window_size.is_none_or(|w| tau >= w);
            for &seq in &pending {
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    values: &mut self.held_values,
                    meta: &mut self.held_meta,
                };
                write_backward(&self.ops, m, seq, &g.stamp_values[ki], complete, held);
            }
        }
        for &seq in &pending {
            let r = usize::try_from(seq - self.held_first).expect("a pending row is held");
            self.held_meta[r].open -= 1;
        }
    }

    /// End every group's open windows `how`; every group starts its windows
    /// over at its next row.
    fn end_all(&mut self, how: End) {
        for gi in 0..self.groups.len() {
            self.end_group(gi, how);
        }
    }

    /// Cut the open windows of every group whose last row is more than
    /// `gap_cap` behind `now`: the stream's clock has shown that its next
    /// row opens with a longer gap. Only a clock column has a cap; on a
    /// row-count clock nothing cuts a silent group, which is why a forward
    /// window under groups needs a clock column (task 173, PC1).
    fn end_silent(&mut self, now: ClockValue) {
        let cap = self.clock_cfg.gap_cap;
        if !cap.is_finite() {
            return;
        }
        while let Some(gi) = self.head {
            let last = self.groups[gi]
                .last_raw
                .expect("a listed group has a clock");
            if !past_cap(now, last, cap) {
                break;
            }
            self.end_group(gi, End::Cut);
        }
    }

    /// End a group's open windows `how`: its pending backward windows are
    /// read as they stand, its forward windows closed at the last row the
    /// group saw -- "the value over what it saw", with nothing held past
    /// that row (review R1, C4: held `gap_cap` past it, a kept rate's span
    /// grew with the cap) -- and its queues emptied; its next row starts
    /// every window over.
    fn end_group(&mut self, gi: usize, how: End) {
        self.flush_group(gi);
        let g = &mut self.groups[gi];
        g.restart = true;
        let end_tau = g.tau;
        for (ki, k) in self.kernels.iter().enumerate() {
            if k.direction != Direction::Forward {
                continue;
            }
            let w = k.window_size.expect("a forward kernel has a window");
            while g.closed[ki] < g.waiting.len() {
                let t = g.waiting[g.closed[ki]];
                g.closed[ki] += 1;
                // Under "right" and "both" a window whose far edge is the
                // stretch's last row holds every timestamp it can, and a cut
                // or a reset leaves none of them to come: it is whole, as its
                // backward mirror is, where a cut cut it short -- null, or
                // dropped (task 160, PC2) -- and a reset discarded it (task
                // 173, PC2). Under "left" and "none" a row on the edge closed
                // it already. A reset discards every other window.
                let whole = matches!(how, End::Cut | End::Discard)
                    && k.closed.far(Direction::Forward)
                    && gap(g.form, t.off, g.off).cmp_window(k) == Ordering::Equal;
                let (how, far) = if whole {
                    (End::Complete, t.tau + w)
                } else {
                    (how, end_tau)
                };
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    values: &mut self.held_values,
                    meta: &mut self.held_meta,
                };
                close(
                    k,
                    &self.ops,
                    &self.members[ki],
                    &mut g.queues[ki],
                    &mut self.scratch_sum,
                    g.open[ki],
                    &g.open_x[ki],
                    &g.open_held[ki],
                    end_tau,
                    t,
                    how,
                    far,
                    g.form,
                    held,
                );
            }
            g.open[ki] = None;
            g.queues[ki].clear();
        }
        retire(&self.kernels, g);
        self.unlink(gi);
    }

    fn unlink(&mut self, gi: usize) {
        let g = &mut self.groups[gi];
        if !g.linked {
            return;
        }
        let (prev, next) = (g.prev_link.take(), g.next_link.take());
        g.linked = false;
        match prev {
            Some(p) => self.groups[p].next_link = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.groups[n].prev_link = prev,
            None => self.tail = prev,
        }
    }

    /// Move a group to the tail of the silent list: its last row is the
    /// stream's latest.
    fn link_back(&mut self, gi: usize) {
        self.unlink(gi);
        let tail = self.tail;
        let g = &mut self.groups[gi];
        g.prev_link = tail;
        g.linked = true;
        match tail {
            Some(t) => self.groups[t].next_link = Some(gi),
            None => self.head = Some(gi),
        }
        self.tail = Some(gi);
    }

    fn advance_ready(&mut self) {
        while let Some(m) = self.held_meta.get(self.ready).filter(|m| m.open == 0) {
            self.ready_kept += usize::from(!m.drop);
            self.ready += 1;
        }
    }

    /// Resolved rows no operator drops, ready to go out: how far a
    /// [`Self::drain_kept`] could go now.
    pub fn ready_kept(&self) -> usize {
        self.ready_kept
    }

    /// The held rows every kernel has resolved, dropped ones included: what
    /// [`Self::drain`] emits.
    pub fn ready(&self) -> usize {
        self.ready
    }

    /// The rows every kernel has resolved, in input order.
    pub fn drain(&mut self) -> Emitted {
        self.emit(self.ready)
    }

    /// [`Self::drain`], stopping at the `limit`-th row no operator drops:
    /// the resolved rows after it stay held, for a later drain or a saved
    /// state.
    pub fn drain_kept(&mut self, limit: usize) -> Emitted {
        let mut kept = 0;
        let mut rows = 0;
        while rows < self.ready && kept < limit {
            kept += usize::from(!self.held_meta[rows].drop);
            rows += 1;
        }
        self.emit(rows)
    }

    /// The end of the input: every backward window still waiting is read
    /// as it stands, every forward window still open is unresolved -- null
    /// whatever `partial` says, never dropped -- and every row goes out.
    /// The core takes no more rows after this.
    pub fn finish(&mut self) -> Emitted {
        self.scratch();
        for gi in 0..self.groups.len() {
            self.flush_group(gi);
            let g = &mut self.groups[gi];
            g.waiting.clear();
            g.closed.iter_mut().for_each(|c| *c = 0);
            for (k, q) in self.kernels.iter().zip(g.queues.iter_mut()) {
                if k.direction == Direction::Forward {
                    q.clear();
                }
            }
            g.open.iter_mut().for_each(|o| *o = None);
            self.unlink(gi);
        }
        self.ready = self.held_meta.len();
        self.ready_kept = self.held_meta.iter().filter(|m| !m.drop).count();
        self.emit(self.ready)
    }

    fn emit(&mut self, rows: usize) -> Emitted {
        let n_out = self.n_outputs;
        let mut out = Emitted {
            first_seq: self.held_first,
            rows,
            values: vec![Vec::with_capacity(rows); n_out],
            drop: Vec::with_capacity(rows),
        };
        for (r, meta) in self.held_meta.drain(..rows).enumerate() {
            for (o, col) in out.values.iter_mut().enumerate() {
                col.push(self.held_values[r * n_out + o]);
            }
            out.drop.push(meta.drop);
            self.ready_kept -= usize::from(!meta.drop);
        }
        self.ready -= rows;
        self.held_values.drain(..rows * n_out);
        self.held_first += rows as u64;
        out
    }
}

/// A forward kernel's open row, once the next row at `end` has ended its
/// interval, written as a queue row into `row`: a mean weighs the held
/// value over `[tau, end)`, a sum counts the row's own value, and only a
/// row with its own value counts toward `min_samples`.
#[allow(clippy::too_many_arguments)]
fn fill_forward(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    tau: f64,
    end: f64,
    x: &[f64],
    held: &[f64],
    rw: usize,
    row: &mut [f64],
) {
    for (j, &o) in members.iter().enumerate() {
        let (v, h) = (x[j], held[j]);
        let n = f64::from(u8::from(!v.is_nan()));
        let acc = match ops[o].stat {
            Stat::Mean if !h.is_nan() => {
                let mass = mass_between(k, tau, tau, end);
                Acc {
                    s: mass * h,
                    w: mass,
                    n,
                    ..Acc::default()
                }
            }
            // A variance has no forward kernel (`check_op`).
            Stat::Mean | Stat::Var | Stat::Std => Acc::default(),
            Stat::Sum | Stat::Rate => Acc {
                s: if v.is_nan() { 0.0 } else { v },
                w: 0.0,
                n,
                ..Acc::default()
            },
        };
        let b = j * rw;
        row[b] = tau;
        row[b + 1] = h;
        acc.write(&mut row[b + 2..], rw == ROW_VAR);
    }
}

/// Each operator's value of a backward window at stamp time `tau`, the
/// queue evicted to the window first, into `out`; NaN where null.
#[allow(clippy::too_many_arguments)]
fn read_backward(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    q: &mut Queue,
    scratch: &mut [f64],
    tau: f64,
    off: i64,
    form: OffForm,
    out: &mut [f64],
) {
    if k.window_size.is_some() {
        // The row at the far edge is in under "left" and "both". Ages are
        // the exact gap from the row to this one (`gap`), one form
        // everywhere, so a row exactly a window old lands on the same side
        // every time.
        let far = k.closed.far(Direction::Backward);
        q.evict(k, scratch, |_, _, t_off| {
            let age = gap(form, t_off, off).cmp_window(k);
            if far {
                age == Ordering::Greater
            } else {
                age != Ordering::Less
            }
        });
    }
    let span = k.window_size.map_or(tau, |w| w.min(tau));
    let edge = k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w);
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        let var = op.stat.is_var();
        let Some((sum, at)) = q.sum_op(k, j) else {
            out[j] = f64::NAN;
            continue;
        };
        // The oldest held value's interval may start before the window:
        // only the part inside counts, so a mean (and a variance) takes the
        // window less the rows up to that one, then the row's mass inside,
        // with no subtraction to cancel. Each operator's oldest valued row
        // is its own.
        let cut = match (k.window_size, op.stat.weighs_held()) {
            (Some(_), true) => q
                .find_oldest(|st, i| !st.x(i, j).is_nan())
                .filter(|&(_, st, i)| st.start(i, j) < edge)
                .map(|(p, st, i)| (p, st.start(i, j), st.end[i], st.x(i, j))),
            _ => None,
        };
        let a = match cut {
            Some((p, start, end, x)) => {
                let mut a = q.sum_after_op(k, p, j).map_or(Acc::default(), |(r, r_at)| {
                    r.scaled(k.discount(tau - r_at), var)
                });
                let m = mass_between(k, tau, start.max(edge), end);
                a = a.plus(Acc::held(x, m, var), var);
                a
            }
            // The sums are anchored at the newest row; seen from `tau`.
            None => sum.scaled(k.discount(tau - at), var),
        };
        out[j] = value_of(k, op, a, span);
    }
}

/// An operator's value from its accumulator and the window's span `t`.
fn value_of(k: &KernelDef, op: &OpDef, a: Acc, t: f64) -> f64 {
    if a.n < f64::from(op.min_samples) {
        return f64::NAN;
    }
    match op.stat {
        Stat::Mean => {
            if a.w > 0.0 {
                a.s / a.w
            } else {
                f64::NAN
            }
        }
        Stat::Sum => a.s,
        Stat::Rate => {
            let m = k.mass(t);
            if m > 0.0 { a.s / m } else { f64::NAN }
        }
        Stat::Var | Stat::Std => {
            if a.w <= 0.0 {
                return f64::NAN;
            }
            let biased = a.q / a.w;
            // Polars' correction for unequal weights, `V1² / (V1² − V2)`
            // times the biased variance, its denominator as `2 e`; none where
            // one row carries the window's weight (`e = 0`).
            let var = if op.bias {
                biased
            } else if a.e > 0.0 {
                a.w * a.w / (2.0 * a.e) * biased
            } else {
                f64::NAN
            };
            if op.stat == Stat::Std {
                var.sqrt()
            } else {
                var
            }
        }
    }
}

/// Write a backward kernel's values for row `seq`, under each operator's
/// `partial` where the window reached before the stretch.
fn write_backward(
    ops: &[OpDef],
    members: &[usize],
    seq: u64,
    vals: &[f64],
    complete: bool,
    held: Held<'_>,
) {
    let r = usize::try_from(seq - held.first).expect("a row is held");
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        if !complete && op.partial == Partial::Drop {
            held.meta[r].drop = true;
        }
        if complete || op.partial == Partial::Keep {
            held.values[r * held.n_outputs + o] = vals[j];
        }
    }
}

/// Close forward kernel `k` over waiting row `t`: the rows its queue holds
/// after `t` are its window, every one of them held until a row inside it,
/// plus the kernel's open row, held from its time until `open_end` and
/// counted up to the window's far edge.
#[allow(clippy::too_many_arguments)]
fn close(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    q: &mut Queue,
    scratch: &mut [f64],
    open: Option<(u64, f64, i64)>,
    open_x: &[f64],
    open_held: &[f64],
    open_end: f64,
    t: Wait,
    how: End,
    end_tau: f64,
    form: OffForm,
    held: Held<'_>,
) {
    let w = k.window_size.expect("a forward kernel has a window");
    // A window is a set of timestamps: under "left" and "both" every row
    // at the row's own stamp is in it, the row itself included, as every
    // row at a stamp is in a backward "right" window (review R1, C3);
    // under "right" and "none" none at the stamp is. Each edge is the exact
    // span from the row (review R2, W1).
    let near_in = k.closed.near(Direction::Forward);
    q.evict(k, scratch, |_, _, j_off| {
        let ahead = gap(form, t.off, j_off).cmp_zero();
        if near_in {
            ahead == Ordering::Less
        } else {
            ahead != Ordering::Greater
        }
    });
    let r = usize::try_from(t.seq - held.first).expect("a waiting row is held");
    let meta = &mut held.meta[r];
    meta.open -= 1;
    let far = (t.tau + w).min(end_tau);
    let len = far - t.tau;
    // The open row is in the window when it is past the row (or at its
    // stamp, under "left" and "both") and not past the far edge.
    // The open row may be the row itself, when the next row closed the
    // window before the row joined the queue: in under "left" and "both".
    let open = open.filter(|&(_, _, o_off)| {
        let ahead = gap(form, t.off, o_off);
        let (near, far) = (ahead.cmp_zero(), ahead.cmp_window(k));
        (if near_in {
            near != Ordering::Less
        } else {
            near == Ordering::Greater
        }) && (if k.closed.far(Direction::Forward) {
            far != Ordering::Greater
        } else {
            far == Ordering::Less
        })
    });
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        if how == End::Cut && op.partial == Partial::Drop {
            meta.drop = true;
        }
        let keep = match how {
            End::Complete => true,
            End::Cut => op.partial == Partial::Keep,
            End::Discard => false,
        };
        // A variance has no forward kernel (`check_op`).
        if !(keep && meta.accept) || op.stat.is_var() {
            continue;
        }
        // The sums are anchored at the oldest row in the window; seen from
        // the row. A held value counts only where the row holding it is in
        // the window: a mean starts at the first row in the window with a
        // value of its own, since the rows before it hold a value from the
        // row itself or from one its stamp excludes.
        let seg = q.sum_op(k, j);
        let scaled = |(a, at): (Acc, f64)| a.scaled(k.discount(at - t.tau), false);
        let mut a = Acc::default();
        let mut any = seg.is_some();
        let mut first_own: Option<usize> = None;
        match op.stat {
            Stat::Mean => {
                first_own = q
                    .find_oldest(|st, i| st.own(i, j).n > 0.0)
                    .map(|(p, _, _)| p);
                let base = match first_own {
                    None => None,
                    Some(0) => seg,
                    Some(p) => q.sum_after_op(k, p - 1, j),
                };
                if let Some(b) = base {
                    a = scaled(b);
                }
            }
            Stat::Sum | Stat::Rate => {
                if let Some(s) = seg {
                    a = scaled(s);
                }
            }
            Stat::Var | Stat::Std => {}
        }
        if let Some((_, otau, _)) = open {
            let (x, h) = (open_x[j], open_held[j]);
            let own = !x.is_nan();
            let n = f64::from(u8::from(own));
            match op.stat {
                Stat::Mean if !h.is_nan() && (own || first_own.is_some()) => {
                    let m = mass_between(k, t.tau, otau, open_end.min(far));
                    a = a.plus(
                        Acc {
                            s: m * h,
                            w: m,
                            n,
                            ..Acc::default()
                        },
                        false,
                    );
                    any = true;
                }
                Stat::Sum | Stat::Rate if own => {
                    a = a.plus(
                        Acc {
                            s: k.discount(otau - t.tau) * x,
                            w: 0.0,
                            n,
                            ..Acc::default()
                        },
                        false,
                    );
                    any = true;
                }
                _ => {}
            }
        }
        if !any {
            continue;
        }
        held.values[r * held.n_outputs + o] = value_of(k, op, a, len);
    }
}

/// Drop the waiting rows every forward kernel has closed.
fn retire(kernels: &[KernelDef], g: &mut Group) {
    let done = kernels
        .iter()
        .zip(&g.closed)
        .filter(|(k, _)| k.direction == Direction::Forward)
        .map(|(_, &c)| c)
        .min()
        .unwrap_or(0);
    if done == 0 {
        return;
    }
    g.waiting.drain(..done);
    for (k, c) in kernels.iter().zip(g.closed.iter_mut()) {
        if k.direction == Direction::Forward {
            *c -= done;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The core against a brute-force loop that knows the whole stream: it
    //! steps the same `ClockState`s (the definition of the policy clock),
    //! cuts the stream into each group's stretches between events, and
    //! computes every operator from its definition, with no queue, no
    //! monoid and no closing order.
    use online_core::{OnClockReset, SessionGap};

    use super::*;

    #[derive(Debug, Clone)]
    struct Row {
        group: Option<String>,
        clock: Option<ClockValue>,
        session: Option<u64>,
        values: Vec<f64>,
        accept: bool,
    }

    /// Every output of every row, in input order.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct Table {
        values: Vec<Vec<f64>>,
        drop: Vec<bool>,
    }

    impl Table {
        fn append(&mut self, e: Emitted, next: &mut u64) {
            assert_eq!(e.first_seq, *next, "rows leave in order, each once");
            *next += e.rows as u64;
            if self.values.is_empty() {
                self.values = vec![Vec::new(); e.values.len()];
            }
            for (a, b) in self.values.iter_mut().zip(e.values) {
                a.extend(b);
            }
            self.drop.extend(e.drop);
        }

        /// Bit for bit, NaN equal to NaN.
        fn same_bits(&self, other: &Table) -> bool {
            self.drop == other.drop
                && self.values.len() == other.values.len()
                && self.values.iter().zip(&other.values).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                })
        }

        fn close_to(&self, other: &Table, tol: f64) -> Result<(), (usize, usize, String)> {
            if self.drop != other.drop {
                let i = self
                    .drop
                    .iter()
                    .zip(&other.drop)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return Err((usize::MAX, i, "drop flags differ".into()));
            }
            for (o, (a, b)) in self.values.iter().zip(&other.values).enumerate() {
                if a.len() != b.len() {
                    return Err((
                        o,
                        0,
                        format!("output {o}: {} rows against {}", a.len(), b.len()),
                    ));
                }
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    if x.is_nan() != y.is_nan()
                        || !x.is_nan() && (x - y).abs() > tol * (1.0 + x.abs().max(y.abs()))
                    {
                        return Err((o, i, format!("output {o} row {i}: {x} against {y}")));
                    }
                }
            }
            Ok(())
        }
    }

    fn push(core: &mut Windows, r: &Row) -> Result<(), Refusal> {
        let group = core.group(r.group.as_deref());
        core.push(&RowIn {
            group,
            clock: r.clock,
            session: r.session,
            values: &r.values,
            accept: r.accept,
        })
    }

    /// The core, draining every `every` rows.
    fn run(
        kernels: &[KernelDef],
        ops: &[OpDef],
        cfg: ClockCfg,
        rows: &[Row],
        every: usize,
    ) -> Result<Table, Refusal> {
        let mut core = Windows::new(kernels.to_vec(), ops.to_vec(), cfg).unwrap();
        core.set_grouped(rows.iter().any(|r| r.group.is_some()));
        let (mut t, mut next) = (Table::default(), 0);
        for (i, r) in rows.iter().enumerate() {
            push(&mut core, r)?;
            if (i + 1) % every == 0 {
                t.append(core.drain(), &mut next);
            }
            // Every state a stream passes through holds what a loader
            // checks (review 2026-10-06, PD1): `check` refuses no core this
            // module made, across every stream the brute force runs.
            core.check()
                .unwrap_or_else(|e| panic!("after row {i}: {e}"));
        }
        t.append(core.finish(), &mut next);
        core.check().unwrap_or_else(|e| panic!("finished: {e}"));
        assert_eq!(next, rows.len() as u64);
        Ok(t)
    }

    /// Each invariant [`Windows::check`] holds a loaded core to, broken
    /// alone on a core fed a stream with groups, sessions, gaps and every
    /// kernel -- so with rows held, waiting on forward windows and on the
    /// next stamp, and groups in the silent list -- is refused by name
    /// (review 2026-10-06, PD1).
    #[test]
    fn each_part_of_a_cores_invariants_is_checked_alone() {
        let (kernels, ops) = all_kernels();
        let rows = stream(9, 120, 3, true, true);
        let clock = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
        let mut core = Windows::new(kernels, ops, clock).unwrap();
        core.set_grouped(true);
        for r in &rows[..77] {
            push(&mut core, r).unwrap();
        }
        core.drain_kept(3);
        core.check().unwrap();
        let is_fwd = |k: &KernelDef| k.direction == Direction::Forward;
        let fwd = core.kernels.iter().position(is_fwd).unwrap();
        let held_in = |f: fn(&Group) -> bool| core.groups.iter().position(f);
        let waiting = held_in(|g| !g.waiting.is_empty()).expect("a waiting row");
        let pending = held_in(|g| !g.pending.is_empty()).expect("a pending row");
        let (head, tail) = (core.head.expect("a silent list"), core.tail.unwrap());
        assert_ne!(head, tail, "two groups listed");
        let fit = "does not fit";
        type Damage<'a> = (&'a str, &'a dyn Fn(&mut Windows));
        let damage: [Damage; 21] = [
            ("names kernel", &|c| c.ops[0].kernel = 99),
            ("operators are not", &|c| c.members.swap(0, 1)),
            ("outputs for", &|c| c.n_outputs += 1),
            ("forward kernels of", &|c| c.n_forward += 1),
            ("values held for", &|c| {
                c.held_values.pop_back();
            }),
            ("rows ready of", &|c| c.ready = c.held_meta.len() + 1),
            ("rows ready of", &|c| c.ready_kept += 1),
            ("before row", &|c| c.held_first += 1),
            (fit, &|c| c.groups[0].queues[0].n += 1),
            (fit, &|c| c.groups[0].queues[fwd].back.tau.push(0.0)),
            (fit, &|c| {
                c.groups[0].open_x[0].pop();
            }),
            (fit, &|c| {
                c.groups[waiting].closed[fwd] = c.groups[waiting].waiting.len() + 1;
            }),
            (fit, &|c| {
                c.groups[0].last_valued.pop();
            }),
            (fit, &|c| c.groups[waiting].waiting[0].seq = c.next_seq),
            (fit, &|c| {
                c.groups[pending].pending[0] = c.held_first.wrapping_sub(1);
            }),
            (fit, &|c| c.groups[head].last_raw = None),
            ("open windows", &|c| c.held_meta[c.ready].open += 1),
            ("silent list", &|c| c.head = Some(c.groups.len())),
            ("silent list", &|c| c.groups[tail].next_link = Some(head)),
            ("one to one", &|c| {
                c.index.insert("stray".into(), 0);
            }),
            ("one to one", &|c| c.null_group = Some(c.groups.len())),
        ];
        for (says, f) in damage {
            let mut broken = core.clone();
            f(&mut broken);
            let err = broken
                .check()
                .err()
                .unwrap_or_else(|| panic!("{says}: passed"));
            assert!(err.contains(says), "{says}: {err}");
        }
    }

    /// `mass` at a half-life far past the window is the even kernel's, to
    /// the series' next term: `1 − 2^(−t/h)` lost 6e-5 of it at `h/t = 1e12`
    /// and all of it from 1e17, every mean null (task 159, W2). Where the
    /// series is nowhere near, the closed form itself.
    #[test]
    fn mass_at_a_long_half_life_is_the_even_kernels() {
        let kernel = |h: f64| KernelDef {
            direction: Direction::Backward,
            half_life: h,
            window_size: Some(5.0),
            window_ns: None,
            closed: Closed::Right,
        };
        let ln2 = std::f64::consts::LN_2;
        for h in [1e9, 1e12, 1e17, 1e300] {
            let got = kernel(h).mass(5.0);
            let want = 5.0 - 12.5 * ln2 / h;
            assert!(
                (got - want).abs() <= 1e-12 * 5.0,
                "h = {h}: {got} vs {want}"
            );
        }
        let got = kernel(2.0).mass(3.0);
        let want = 2.0 / ln2 * (1.0 - 0.5f64.powf(1.5));
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }

    /// A subnormal half-life passed `check`: `mass·x` underflowed to 0 and
    /// a rate divided by that mass, so the mean came out 0 and the rate
    /// infinite (task 159, W5).
    #[test]
    fn a_subnormal_half_life_is_refused() {
        for h in [5e-324, 1e-310] {
            let k = KernelDef {
                direction: Direction::Backward,
                half_life: h,
                window_size: Some(2.0),
                window_ns: None,
                closed: Closed::Right,
            };
            let err = k.check().expect_err("refused");
            assert!(err.contains("normal number"), "{err}");
        }
    }

    /// `∫_a^b λ^(T - s) ds` / `∫_a^b λ^(s - T) ds`, written from the
    /// definition with the discount as a closed form.
    fn mass_bf(k: &KernelDef, t: f64, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let m = |d: f64| {
            if d <= 0.0 {
                0.0
            } else if k.half_life.is_infinite() {
                d
            } else {
                // The closed form by `exp_m1`, as `mass` takes it (task 159, W2).
                -(k.half_life / std::f64::consts::LN_2)
                    * (-(d * std::f64::consts::LN_2 / k.half_life)).exp_m1()
            }
        };
        match k.direction {
            Direction::Backward => m(t - a) - m(t - b),
            Direction::Forward => m(b - t) - m(a - t),
        }
    }

    fn elapsed_bf(now: ClockValue, last: ClockValue) -> f64 {
        match (now, last) {
            (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
            (ClockValue::F64(c), ClockValue::F64(p)) => c - p,
            (ClockValue::I64(c), ClockValue::I64(p)) => (i128::from(c) - i128::from(p)) as f64,
            _ => unreachable!("a test stream keeps one kind of clock"),
        }
    }

    /// A stretch: rows of one group between clock events, with their
    /// policy times from 0, and how it ended.
    struct Stretch {
        rows: Vec<usize>,
        tau: Vec<f64>,
        /// How it ended, and the policy time of the end, if it ended.
        end: Option<(End, f64)>,
    }

    /// The clock from stretch row `from` to stretch row `to`, as the core
    /// decides an edge: the raw clocks' exact difference on a temporal
    /// clock, the raw clocks' on a number clock (task 159, W3), else the
    /// policy times' (review R2, W1).
    fn gap_bf(rows: &[Row], s: &Stretch, from: usize, to: usize) -> Gap {
        match (rows[s.rows[from]].clock, rows[s.rows[to]].clock) {
            (Some(ClockValue::Ns(x)), Some(ClockValue::Ns(y))) => Gap::Ns(y - x),
            (Some(ClockValue::F64(x)), Some(ClockValue::F64(y))) => Gap::Secs(y - x),
            (Some(ClockValue::I64(x)), Some(ClockValue::I64(y))) => {
                Gap::Int(i128::from(y) - i128::from(x))
            }
            _ => Gap::Secs(s.tau[to] - s.tau[from]),
        }
    }

    fn brute(
        kernels: &[KernelDef],
        ops: &[OpDef],
        cfg: ClockCfg,
        rows: &[Row],
    ) -> Result<Table, Refusal> {
        struct G {
            clock: ClockState,
            tau: f64,
            origin: Option<ClockValue>,
            stretch: usize,
            restart: bool,
            last_raw: Option<ClockValue>,
        }
        // With groups a session is each group's (review R2, W3).
        let grouped = rows.iter().any(|r| r.group.is_some());
        let mut shared = ClockState::new();
        let mut groups: HashMap<Option<String>, G> = HashMap::new();
        let mut stretches: Vec<Stretch> = Vec::new();
        let mut live: Vec<bool> = Vec::new();
        let (mut stretch_of, mut tau_of) = (Vec::new(), Vec::new());
        for (i, r) in rows.iter().enumerate() {
            let seq = i as u64;
            let mut sc = shared.clone();
            let adv = sc.advance(&cfg, r.clock, if grouped { None } else { r.session }, true);
            refuse_backwards(&adv, seq, shared.last_clock(), r.clock)?;
            // A stream restart starts every group's clock over (task 159, W1).
            let mut gc = if adv.reset {
                ClockState::new()
            } else {
                groups
                    .get(&r.group)
                    .map_or_else(ClockState::new, |g| g.clock.clone())
            };
            let prev = gc.last_clock();
            let own = gc.advance(&cfg, r.clock, r.session, true);
            refuse_backwards(&own, seq, prev, r.clock)?;
            shared = sc;
            let end = |g: &mut G, how: End, stretches: &mut Vec<Stretch>, live: &mut Vec<bool>| {
                if live[g.stretch] {
                    live[g.stretch] = false;
                    // A cut stretch ends at the last row it saw (R1, C4).
                    stretches[g.stretch].end = Some((how, g.tau));
                }
                g.restart = true;
            };
            if adv.reset || adv.session_changed || adv.capped {
                let how = if adv.reset { End::Discard } else { End::Cut };
                for g in groups.values_mut() {
                    end(g, how, &mut stretches, &mut live);
                    if adv.reset {
                        g.clock = ClockState::new();
                    }
                }
            }
            if let (Some(now), true) = (r.clock, cfg.gap_cap.is_finite()) {
                for g in groups.values_mut() {
                    if let Some(last) = g.last_raw
                        && elapsed_bf(now, last) > cfg.gap_cap
                    {
                        end(g, End::Cut, &mut stretches, &mut live);
                    }
                }
            }
            let g = groups.entry(r.group.clone()).or_insert_with(|| G {
                clock: ClockState::new(),
                tau: 0.0,
                origin: None,
                stretch: usize::MAX,
                restart: true,
                last_raw: None,
            });
            if own.reset {
                end(g, End::Discard, &mut stretches, &mut live);
            } else if own.session_changed || own.capped {
                end(g, End::Cut, &mut stretches, &mut live);
            }
            g.clock = gc;
            if g.restart {
                g.tau = 0.0;
                g.origin = r.clock;
                g.restart = false;
                g.stretch = stretches.len();
                stretches.push(Stretch {
                    rows: Vec::new(),
                    tau: Vec::new(),
                    end: None,
                });
                live.push(true);
            } else if let (Some(now), Some(origin)) = (r.clock, g.origin) {
                g.tau = elapsed(now, origin);
            } else {
                g.tau += own.d_clock;
                if g.origin.is_none() {
                    g.origin = r.clock;
                }
            }
            stretches[g.stretch].rows.push(i);
            stretches[g.stretch].tau.push(g.tau);
            stretch_of.push(g.stretch);
            tau_of.push(g.tau);
            g.last_raw = r.clock;
        }
        let n_out = ops.len();
        let mut t = Table {
            values: vec![vec![f64::NAN; rows.len()]; n_out],
            drop: vec![false; rows.len()],
        };
        let value = |r: usize, input: usize| {
            let v = rows[r].values[input];
            if usable(v) { v } else { f64::NAN }
        };
        for s in &stretches {
            let n = s.rows.len();
            for (o, op) in ops.iter().enumerate() {
                let k = &kernels[op.kernel];
                for a in 0..n {
                    let (r, ta) = (s.rows[a], s.tau[a]);
                    let stamp = rows[r].clock;
                    let (members, complete, span): (Vec<usize>, bool, f64) = match k.direction {
                        Direction::Backward => {
                            let w = k.window_size;
                            // Rows at the stamp, per `closed`; the lower
                            // edge per `closed`.
                            let members: Vec<usize> = (0..n)
                                .filter(|&b| {
                                    let at_stamp =
                                        rows[s.rows[b]].clock == stamp && stamp.is_some();
                                    let before = b < a && !at_stamp
                                        || (at_stamp && b != a && stamp.is_some())
                                        || (b == a);
                                    let near_ok = if k.closed.near(Direction::Backward) {
                                        (b <= a) || at_stamp
                                    } else {
                                        b < a && !at_stamp
                                    };
                                    let far_ok = match w {
                                        None => true,
                                        Some(_) => {
                                            let age = gap_bf(rows, s, b, a).cmp_window(k);
                                            if k.closed.far(Direction::Backward) {
                                                age != Ordering::Greater
                                            } else {
                                                age == Ordering::Less
                                            }
                                        }
                                    };
                                    let _ = before;
                                    near_ok && far_ok
                                })
                                .collect();
                            (
                                members,
                                w.is_none_or(|w| ta >= w),
                                w.map_or(ta, |w| w.min(ta)),
                            )
                        }
                        Direction::Forward => {
                            let w = k.window_size.unwrap();
                            // A set of timestamps: under "left" and "both"
                            // every row at the row's stamp, itself included
                            // (R1, C3); under "right" and "none" none at it.
                            let later: Vec<usize> = (0..n)
                                .filter(|&b| {
                                    let d = gap_bf(rows, s, a, b);
                                    let (near, far) = (d.cmp_zero(), d.cmp_window(k));
                                    let near_ok = if k.closed.near(Direction::Forward) {
                                        near != Ordering::Less
                                    } else {
                                        near == Ordering::Greater
                                    };
                                    let far_ok = if k.closed.far(Direction::Forward) {
                                        far != Ordering::Greater
                                    } else {
                                        far == Ordering::Less
                                    };
                                    near_ok && far_ok
                                })
                                .collect();
                            // Whole once every timestamp of the stretch the
                            // window can hold is in, read from the stretch
                            // itself rather than from a closing rule (task
                            // 160, PC5). A stretch a break or a reset ended
                            // has every row it will have: the window is
                            // whole when the stretch reaches its far edge, so
                            // that its span lies inside the stretch (task
                            // 173, PC2, for a reset), and otherwise cut short
                            // by a break, or discarded by a reset. A stretch
                            // still open at the end of the input has shown
                            // its window whole only by a row stamped past
                            // every timestamp the window holds: past the far
                            // edge where the window holds it ("right",
                            // "both"), at it or past where it does not; a
                            // window it has not shown whole is unresolved,
                            // and null.
                            let reaches_edge =
                                gap_bf(rows, s, a, n - 1).cmp_window(k) != Ordering::Less;
                            let shown_whole = (a + 1..n).any(|b| {
                                let d = gap_bf(rows, s, a, b).cmp_window(k);
                                if k.closed.far(Direction::Forward) {
                                    d == Ordering::Greater
                                } else {
                                    d != Ordering::Less
                                }
                            });
                            match s.end {
                                Some((End::Cut | End::Discard, _)) if reaches_edge => {
                                    (later, true, w)
                                }
                                Some((End::Cut, end_tau)) => {
                                    if op.partial == Partial::Drop {
                                        t.drop[r] = true;
                                    }
                                    if op.partial != Partial::Keep || !rows[r].accept {
                                        continue;
                                    }
                                    (later, false, (ta + w).min(end_tau) - ta)
                                }
                                _ if shown_whole => (later, true, w),
                                _ => continue,
                            }
                        }
                    };
                    if k.direction == Direction::Backward {
                        if !complete && op.partial == Partial::Drop {
                            t.drop[r] = true;
                        }
                        if !complete && op.partial != Partial::Keep {
                            continue;
                        }
                    } else if !rows[r].accept {
                        continue;
                    }
                    // The value, from the definition.
                    let (mut sum, mut mass, mut count) = (0.0, 0.0, 0.0);
                    let mut held: Vec<(f64, f64)> = Vec::new();
                    for &b in &members {
                        let x = value(s.rows[b], op.input);
                        if x.is_nan() {
                            continue;
                        }
                        count += 1.0;
                        let tb = s.tau[b];
                        match op.stat {
                            Stat::Mean | Stat::Var | Stat::Std => {
                                let (lo, hi) = match k.direction {
                                    Direction::Backward => {
                                        // Held from the operator's last valued row.
                                        let prev = (0..b)
                                            .rev()
                                            .find(|&c| !value(s.rows[c], op.input).is_nan());
                                        let start = match prev {
                                            Some(c) => s.tau[c],
                                            None => k
                                                .window_size
                                                .map_or(f64::NEG_INFINITY, |w| s.tau[0] - w),
                                        };
                                        let edge =
                                            k.window_size.map_or(f64::NEG_INFINITY, |w| ta - w);
                                        (start.max(edge), tb)
                                    }
                                    Direction::Forward => {
                                        // Held until the operator's next valued row.
                                        let next = (b + 1..n)
                                            .find(|&c| !value(s.rows[c], op.input).is_nan());
                                        let end = match next {
                                            Some(c) => s.tau[c],
                                            None => s.end.map_or(f64::INFINITY, |(_, e)| e),
                                        };
                                        let far = ta + k.window_size.unwrap();
                                        (
                                            tb,
                                            end.min(far)
                                                .min(s.end.map_or(f64::INFINITY, |(_, e)| e)),
                                        )
                                    }
                                };
                                let m = mass_bf(k, ta, lo, hi);
                                sum += m * x;
                                mass += m;
                                held.push((m, x));
                            }
                            Stat::Sum | Stat::Rate => {
                                let d = (tb - ta).abs();
                                let f = if k.half_life.is_infinite() {
                                    1.0
                                } else {
                                    (-(d / k.half_life)).exp2()
                                };
                                sum += f * x;
                            }
                        }
                    }
                    if count < f64::from(op.min_samples) {
                        continue;
                    }
                    t.values[o][r] = match op.stat {
                        Stat::Mean => {
                            if mass > 0.0 {
                                sum / mass
                            } else {
                                f64::NAN
                            }
                        }
                        Stat::Sum => sum,
                        Stat::Var | Stat::Std => var_bf(&held, op.bias, op.stat == Stat::Std),
                        Stat::Rate => {
                            let m = mass_bf(k, 0.0, 0.0, span).abs();
                            let m = if k.direction == Direction::Backward {
                                mass_bf(k, span, 0.0, span)
                            } else {
                                m
                            };
                            if m > 0.0 { sum / m } else { f64::NAN }
                        }
                    };
                }
            }
        }
        let _ = (&stretch_of, &tau_of);
        Ok(t)
    }

    /// The variance of `held`'s `(weight, value)` pairs from the definition,
    /// in two passes: the weighted mean, then the weighted mean square
    /// deviation from it; unless `bias`, times `V1² / (V1² − V2)` (null
    /// where one weight is all of it); its root for a standard deviation.
    fn var_bf(held: &[(f64, f64)], bias: bool, std: bool) -> f64 {
        let v1: f64 = held.iter().map(|&(m, _)| m).sum();
        if v1 <= 0.0 {
            return f64::NAN;
        }
        let mu = held.iter().map(|&(m, x)| m * x).sum::<f64>() / v1;
        let mut var = held
            .iter()
            .map(|&(m, x)| m * (x - mu) * (x - mu))
            .sum::<f64>()
            / v1;
        if !bias {
            // `V1² − V2 = 2 Σ_{i<j} m_i m_j`, summed as the products of each
            // weight with the ones before it: every term at or above 0.
            let (mut before, mut pairs) = (0.0, 0.0);
            for &(m, _) in held {
                pairs += m * before;
                before += m;
            }
            if pairs <= 0.0 {
                return f64::NAN;
            }
            var *= v1 * v1 / (2.0 * pairs);
        }
        if std { var.sqrt() } else { var }
    }

    /// A test window's nanoseconds, as the frame reads them off a duration.
    fn ns_of(w: f64) -> i64 {
        (w * 1e9).round() as i64
    }

    fn cfg(cap: f64, session_gap: Option<SessionGap>, restart: Option<f64>) -> ClockCfg {
        ClockCfg {
            gap_cap: cap,
            on_clock_reset: if restart.is_some() {
                OnClockReset::ResetState
            } else {
                OnClockReset::Error
            },
            session_gap,
            min_backwards_jump: restart.unwrap_or(0.0),
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() * n as f64) as usize
        }
    }

    /// A stream with gaps, repeated stamps, session changes, missing
    /// values and several groups.
    fn stream(seed: u64, n: usize, groups: usize, temporal: bool, sessions: bool) -> Vec<Row> {
        let mut rng = Lcg(seed);
        let mut t = 0.0f64;
        let mut rows = Vec::with_capacity(n);
        let mut session = 0u64;
        for i in 0..n {
            let step = match rng.below(10) {
                0 => 0.0,
                1 | 2 => 0.5,
                3..=7 => 1.0 + rng.next(),
                8 => 4.0,
                _ => 12.0,
            };
            t += step;
            if sessions && i > 0 && rng.below(40) == 0 {
                session += 1;
            }
            let g = rng.below(groups);
            let value = |rng: &mut Lcg| {
                if rng.below(6) == 0 {
                    f64::NAN
                } else {
                    rng.next() * 10.0 - 3.0
                }
            };
            let v0 = value(&mut rng);
            let v1 = value(&mut rng);
            rows.push(Row {
                group: if groups > 1 {
                    Some(format!("g{g}"))
                } else {
                    None
                },
                clock: Some(if temporal {
                    ClockValue::Ns((t * 1e9).round() as i64)
                } else {
                    ClockValue::F64(t)
                }),
                session: sessions.then_some(session),
                values: vec![v0, v1],
                accept: rng.below(8) != 0,
            });
        }
        rows
    }

    /// [`stream`] on a half-unit grid: every step a multiple of 0.5, so a
    /// row lands exactly a window after another, and exactly on a window's
    /// far edge as the last row before a break, often (task 160, PC2;
    /// `stream`'s random steps land there only where its exact ones happen
    /// to add up). `repeats` lets a step of 0 repeat a stamp.
    fn grid_stream(seed: u64, n: usize, groups: usize, sessions: bool, repeats: bool) -> Vec<Row> {
        let mut rows = stream(seed, n, groups, false, sessions);
        let mut rng = Lcg(seed.wrapping_mul(31).wrapping_add(7));
        let mut t = 0.0f64;
        for r in &mut rows {
            t += match rng.below(9) {
                0 if repeats => 0.0,
                0 | 1 => 0.5,
                2 | 3 => 1.0,
                4 => 1.5,
                5 | 6 => 2.0,
                7 => 3.0,
                _ => 12.0,
            };
            r.clock = Some(ClockValue::F64(t));
        }
        rows
    }

    fn all_kernels() -> (Vec<KernelDef>, Vec<OpDef>) {
        let mut kernels = Vec::new();
        let mut ops = Vec::new();
        for (direction, half_life, window, closed) in [
            (Direction::Backward, 3.0, None, Closed::Right),
            (Direction::Backward, 3.0, Some(5.0), Closed::Right),
            (Direction::Backward, 2.0, Some(4.0), Closed::Left),
            (Direction::Backward, f64::INFINITY, Some(6.0), Closed::Both),
            (Direction::Backward, 5.0, Some(7.0), Closed::Neither),
            (Direction::Forward, 3.0, Some(5.0), Closed::Right),
            (Direction::Forward, 2.0, Some(4.0), Closed::Left),
            (Direction::Forward, f64::INFINITY, Some(6.0), Closed::Both),
            (Direction::Forward, 5.0, Some(3.0), Closed::Neither),
        ] {
            let ki = kernels.len();
            kernels.push(KernelDef {
                direction,
                half_life,
                window_size: window,
                window_ns: window.map(ns_of),
                closed,
            });
            for (stat, partial, min) in [
                (Stat::Mean, Partial::Keep, 1),
                (Stat::Sum, Partial::Null, 2),
                (Stat::Rate, Partial::Keep, 1),
                (Stat::Mean, Partial::Drop, 1),
            ] {
                // A drop operator on every kernel would drop most rows.
                if partial == Partial::Drop && ki != 5 {
                    continue;
                }
                ops.push(OpDef {
                    kernel: ki,
                    stat,
                    input: ops.len() % 2,
                    min_samples: min,
                    partial,
                    bias: false,
                });
            }
            // A variance and a standard deviation on every backward kernel,
            // corrected and not (task 212).
            if direction == Direction::Backward {
                for (stat, partial, min, bias) in [
                    (Stat::Var, Partial::Keep, 1, false),
                    (Stat::Std, Partial::Null, 2, true),
                ] {
                    ops.push(OpDef {
                        kernel: ki,
                        stat,
                        input: ops.len() % 2,
                        min_samples: min,
                        partial,
                        bias,
                    });
                }
            }
        }
        (kernels, ops)
    }

    /// Every operator on every kernel against the definition, on streams
    /// with gaps, repeated stamps, sessions, missing values and groups.
    #[test]
    fn every_operator_matches_its_definition() {
        let (kernels, ops) = all_kernels();
        // The last four on the half-unit grid, where windows end exactly on
        // the last row before a break (task 160, PC2).
        for (seed, groups, temporal, sessions, grid) in [
            (1, 1, false, false, false),
            (2, 1, true, false, false),
            (3, 3, false, true, false),
            (4, 2, true, true, false),
            (5, 4, true, false, false),
            (6, 1, false, false, true),
            (7, 1, false, true, true),
            (8, 3, false, true, true),
            (9, 2, false, false, true),
        ] {
            let rows = if grid {
                grid_stream(seed, 400, groups, sessions, true)
            } else {
                stream(seed, 400, groups, temporal, sessions)
            };
            let c = cfg(8.0, sessions.then_some(SessionGap::Gap(2.0)), None);
            assert_matches_brute(&kernels, &ops, c, &rows, seed);
        }
    }

    /// Task 200: on an integer clock of epoch nanoseconds near 1.79e18,
    /// where a double resolves 256, the core decides every edge, gap and
    /// silence from the integers: it matches the brute force, whose edges
    /// are the integers' differences, and gives what the same stream
    /// shifted to 0 gives as a float clock, every value of which a double
    /// holds -- to the bit. The streams are the half-unit grid doubled, so
    /// rows land exactly a window apart and exactly at the cap, often.
    #[test]
    fn an_integer_clock_decides_its_edges_in_integers() {
        const T0: i64 = 1_790_000_000_000_000_000;
        let (kernels, ops) = all_kernels();
        for (seed, groups, sessions) in [(21u64, 1usize, false), (22, 3, true), (23, 2, false)] {
            let grid = grid_stream(seed, 400, groups, sessions, true);
            let at = |r: &Row| match r.clock {
                Some(ClockValue::F64(t)) => (2.0 * t) as i64,
                _ => unreachable!("the grid is a float clock"),
            };
            let int: Vec<Row> = grid
                .iter()
                .map(|r| Row {
                    clock: Some(ClockValue::I64(T0 + at(r))),
                    ..r.clone()
                })
                .collect();
            let shifted: Vec<Row> = grid
                .iter()
                .map(|r| Row {
                    clock: Some(ClockValue::F64(at(r) as f64)),
                    ..r.clone()
                })
                .collect();
            let c = cfg(8.0, sessions.then_some(SessionGap::Gap(2.0)), None);
            assert_matches_brute(&kernels, &ops, c, &int, seed);
            let (a, b) = (
                run(&kernels, &ops, c, &int, 1).unwrap(),
                run(&kernels, &ops, c, &shifted, 1).unwrap(),
            );
            if let Err((_, _, e)) = a.close_to(&b, 0.0) {
                panic!("seed {seed}: {e}");
            }
        }
    }

    /// The core against the brute force on `rows`, to 1e-9, the rows around
    /// the first difference printed.
    fn assert_matches_brute(
        kernels: &[KernelDef],
        ops: &[OpDef],
        c: ClockCfg,
        rows: &[Row],
        seed: u64,
    ) {
        let want = brute(kernels, ops, c, rows).unwrap();
        let got = run(kernels, ops, c, rows, 1).unwrap();
        // A match where a variance is never valued holds nothing of it (a
        // sparse forward sum is valued on no row of one integer stream).
        for (o, col) in want.values.iter().enumerate() {
            let valued = col.iter().filter(|v| !v.is_nan()).count();
            assert!(
                valued > 0 || !ops[o].stat.is_var(),
                "seed {seed}: output {o} valued on {valued} rows"
            );
        }
        if let Err((o, i, e)) = got.close_to(&want, 1e-9) {
            for (r, row) in rows
                .iter()
                .enumerate()
                .take(i + 5)
                .skip(i.saturating_sub(10))
            {
                let (g, w) = if o < got.values.len() {
                    (got.values[o][r], want.values[o][r])
                } else {
                    (f64::NAN, f64::NAN)
                };
                eprintln!(
                    "row {r}: clock {:?} x {:?} accept {} got {g} want {w}",
                    row.clock, row.values, row.accept
                );
            }
            panic!("seed {seed}: {e}");
        }
    }

    /// Every operator on every kernel against the definition across resets
    /// (task 173, PC2): the half-unit grid's breaks made steps back past
    /// `restart_after_step_back` or new sessions under `session_gap =
    /// "reset"`, alone and with groups, so windows end exactly on the last
    /// row before a reset. Whole there, as before a cut; the rest of the
    /// windows a reset meets discarded, null under every `partial` and
    /// never dropped.
    #[test]
    fn every_operator_matches_its_definition_across_resets() {
        let (kernels, ops) = all_kernels();
        for (seed, groups) in [(21, 1), (22, 3), (23, 1), (24, 2)] {
            let (rows, c) = if seed < 23 {
                (
                    step_back_stream(seed, 400, groups, true),
                    cfg(8.0, None, Some(0.0)),
                )
            } else {
                (
                    session_reset_stream(seed, 400, groups, true),
                    cfg(8.0, Some(SessionGap::Reset), None),
                )
            };
            assert_matches_brute(&kernels, &ops, c, &rows, seed);
        }
    }

    /// Feeding a stream in chunks of any size gives the same bits.
    /// Review R1, C7: the core refused this only through the formula layer;
    /// a direct caller got a NaN mean.
    #[test]
    fn a_mean_with_no_decay_needs_a_window() {
        let err = Windows::new(
            vec![KernelDef {
                direction: Direction::Backward,
                half_life: f64::INFINITY,
                window_size: None,
                window_ns: None,
                closed: Closed::Right,
            }],
            vec![OpDef {
                kernel: 0,
                stat: Stat::Mean,
                input: 0,
                min_samples: 1,
                partial: Partial::Keep,
                bias: false,
            }],
            ClockCfg::default(),
        )
        .expect_err("refused");
        assert!(err.contains("needs a window_size"), "{err}");
    }

    #[test]
    fn chunking_changes_no_bit() {
        let (kernels, ops) = all_kernels();
        let rows = stream(7, 500, 3, true, true);
        let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
        let one = run(&kernels, &ops, c, &rows, usize::MAX).unwrap();
        for every in [1, 2, 3, 7, 50, 499] {
            let t = run(&kernels, &ops, c, &rows, every).unwrap();
            assert!(t.same_bits(&one), "every {every}");
        }
    }

    /// A test row's number clock.
    fn clock_of(r: &Row) -> f64 {
        match r.clock {
            Some(ClockValue::F64(x)) => x,
            _ => unreachable!("a number clock"),
        }
    }

    /// [`grid_stream`]'s clock with each of its steps past the cap turned
    /// into a step back of the same size, which `restart_after_step_back`
    /// restarts on: the same stretches, each ended by a reset rather than a
    /// cut (task 173, PC2).
    fn step_back_stream(seed: u64, n: usize, groups: usize, repeats: bool) -> Vec<Row> {
        let mut rows = grid_stream(seed, n, groups, false, repeats);
        let (mut prev, mut t) = (0.0, 0.0);
        for r in &mut rows {
            let step = clock_of(r) - prev;
            prev = clock_of(r);
            t += if step == 12.0 { -12.0 } else { step };
            r.clock = Some(ClockValue::F64(t));
        }
        rows
    }

    /// [`grid_stream`]'s clock with each of its steps past the cap turned
    /// into a step of 1 and a new session, which `session_gap = "reset"`
    /// restarts on: the same stretches again, each ended by the other kind
    /// of reset (task 173, PC2).
    fn session_reset_stream(seed: u64, n: usize, groups: usize, repeats: bool) -> Vec<Row> {
        let mut rows = grid_stream(seed, n, groups, false, repeats);
        let (mut prev, mut t, mut session) = (0.0, 0.0, 0u64);
        for r in &mut rows {
            let step = clock_of(r) - prev;
            prev = clock_of(r);
            if step == 12.0 {
                session += 1;
                t += 1.0;
            } else {
                t += step;
            }
            r.clock = Some(ClockValue::F64(t));
            r.session = Some(session);
        }
        rows
    }

    /// The time-reversal identity on `rows`, under `c`: row `i`'s forward
    /// window is the mirror's backward window at the mirrored row, where
    /// the mirror is the stream with its times negated and its rows
    /// reversed. `ends(j)` says a break follows row `j`, which ends its
    /// stretch: the mirror has the same break between the same two rows (a
    /// gap, a step back of the same size, the same new session). The last
    /// row is put after one, so the input ends on a stretch of one row and
    /// no forward window is left open by the end of the input where its
    /// mirror is whole (task 160, PC2).
    fn assert_mirror(rows: &[Row], c: ClockCfg, ends: impl Fn(&[Row], usize) -> bool) {
        let n = rows.len();
        let t: Vec<f64> = rows.iter().map(clock_of).collect();
        // The rows whose window of 5 ends exactly on the last row before a
        // break: the case the mirror catches.
        let on_edge = (0..n)
            .filter(|&i| {
                let end = (i..n).find(|&j| j + 1 == n || ends(rows, j)).unwrap();
                end + 1 < n && t[end] - t[i] == 5.0
            })
            .count();
        assert!(on_edge > 5, "{on_edge} rows end on the edge");
        let last = t[n - 1];
        let mut mirror: Vec<Row> = rows.iter().rev().cloned().collect();
        for r in &mut mirror {
            r.clock = Some(ClockValue::F64(last - clock_of(r)));
        }
        // The mirror: times negated and reversed, a backward kernel closed
        // on the far side -- "left" mirrors "right", "both" mirrors itself.
        for (closed, mirrored) in [(Closed::Right, Closed::Left), (Closed::Both, Closed::Both)] {
            let k = KernelDef {
                direction: Direction::Forward,
                half_life: 3.0,
                window_size: Some(5.0),
                window_ns: Some(ns_of(5.0)),
                closed,
            };
            let ops: Vec<OpDef> = [Stat::Mean, Stat::Sum, Stat::Rate]
                .into_iter()
                .map(|stat| OpDef {
                    kernel: 0,
                    stat,
                    input: 0,
                    min_samples: 1,
                    partial: Partial::Null,
                    bias: false,
                })
                .collect();
            let fwd = run(std::slice::from_ref(&k), &ops, c, rows, 1).unwrap();
            let kb = KernelDef {
                direction: Direction::Backward,
                closed: mirrored,
                ..k
            };
            let back = run(&[kb], &ops, c, &mirror, 1).unwrap();
            // Row i's forward window is the mirror's backward window at the
            // mirrored row i: whole where it is whole, cut short or discarded
            // where it does not reach its far edge (null both), and the same
            // number where it is whole.
            for o in 0..ops.len() {
                for (i, ti) in t.iter().enumerate() {
                    let (a, b) = (fwd.values[o][i], back.values[o][n - 1 - i]);
                    assert_eq!(
                        a.is_nan(),
                        b.is_nan(),
                        "{closed:?} op {o} row {i} at {ti}: {a} vs {b}"
                    );
                    if !a.is_nan() {
                        assert!(
                            (a - b).abs() <= 1e-9 * (1.0 + a.abs()),
                            "{closed:?} op {o} row {i}: {a} vs {b}"
                        );
                    }
                }
                assert!(fwd.values[o].iter().filter(|v| !v.is_nan()).count() > n / 3);
            }
        }
    }

    /// A forward window is a backward one over the reversed stream, less the
    /// row itself: the mirror of each kernel on the mirror of the stream.
    #[test]
    fn a_forward_window_is_a_backward_one_over_the_reversed_stream() {
        // Distinct stamps on a half-unit grid, so the row itself is the only
        // exclusion and a row lands exactly on a window's far edge, broken
        // by gaps past the cap.
        let mut rows = grid_stream(11, 300, 1, false, false);
        rows.iter_mut().for_each(|r| r.accept = true);
        let mut last_row = rows.last().unwrap().clone();
        last_row.clock = Some(ClockValue::F64(clock_of(&last_row) + 12.0));
        rows.push(last_row);
        assert_mirror(&rows, cfg(8.0, None, None), |rows, j| {
            clock_of(&rows[j + 1]) - clock_of(&rows[j]) > 8.0
        });
    }

    /// The identity across resets (task 173, PC2): a reset discards every
    /// window still open across it, but one whose far edge is the last row
    /// before it is whole, as at a cut -- every row it covers has arrived,
    /// and its mirror, a backward window reaching exactly back to its
    /// stretch's first row, is complete. Before, a reset discarded it, and
    /// the forward side was null where the mirror had its number.
    #[test]
    fn a_forward_window_is_a_backward_one_over_the_reversed_stream_across_resets() {
        // A step back past `restart_after_step_back` (0: every step back).
        let mut rows = step_back_stream(11, 300, 1, false);
        rows.iter_mut().for_each(|r| r.accept = true);
        let mut last_row = rows.last().unwrap().clone();
        last_row.clock = Some(ClockValue::F64(clock_of(&last_row) - 12.0));
        rows.push(last_row);
        assert_mirror(&rows, cfg(8.0, None, Some(0.0)), |rows, j| {
            clock_of(&rows[j + 1]) < clock_of(&rows[j])
        });
        // A new session under `session_gap = "reset"`.
        let mut rows = session_reset_stream(11, 300, 1, false);
        rows.iter_mut().for_each(|r| r.accept = true);
        let mut last_row = rows.last().unwrap().clone();
        last_row.clock = Some(ClockValue::F64(clock_of(&last_row) + 1.0));
        last_row.session = last_row.session.map(|s| s + 1);
        rows.push(last_row);
        assert_mirror(&rows, cfg(8.0, Some(SessionGap::Reset), None), |rows, j| {
            rows[j + 1].session != rows[j].session
        });
    }

    /// Every row at a repeated stamp gets the same backward window under
    /// "right", later rows at the stamp included; under "left" the stamp's
    /// rows are outside and the window is known as the stamp arrives.
    #[test]
    fn rows_at_one_stamp_share_a_backward_window() {
        let rows: Vec<Row> = [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (2.0, 4.0),
            (2.0, 5.0),
            (3.0, 6.0),
            (5.0, 7.0),
        ]
        .iter()
        .map(|&(t, x)| Row {
            group: None,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: vec![x],
            accept: true,
        })
        .collect();
        let c = cfg(100.0, None, None);
        let sums = |closed: Closed| {
            let k = KernelDef {
                direction: Direction::Backward,
                half_life: f64::INFINITY,
                window_size: Some(2.0),
                window_ns: Some(ns_of(2.0)),
                closed,
            };
            let op = OpDef {
                kernel: 0,
                stat: Stat::Sum,
                input: 0,
                min_samples: 1,
                partial: Partial::Keep,
                bias: false,
            };
            run(&[k], &[op], c, &rows, 1).unwrap().values[0].clone()
        };
        // Polars' rolling_sum_by, window 2, on these stamps (measured
        // 2026-10-02, docs/PLAN.md task 143).
        assert_eq!(
            sums(Closed::Right),
            vec![1.0, 3.0, 14.0, 14.0, 14.0, 18.0, 7.0]
        );
        assert_eq!(
            sums(Closed::Both),
            vec![1.0, 3.0, 15.0, 15.0, 15.0, 20.0, 13.0]
        );
        let left = sums(Closed::Left);
        assert!(left[0].is_nan() && left[1..] == [1.0, 3.0, 3.0, 3.0, 14.0, 6.0]);
        let none = sums(Closed::Neither);
        assert!(none[0].is_nan() && none[1..6] == [1.0, 2.0, 2.0, 2.0, 12.0] && none[6].is_nan());
    }

    /// The backward time-weighted mean is `ewm_mean_by`'s recursion with
    /// `a_1 = 1`, written out, and a repeated stamp moves it not at all.
    #[test]
    fn the_backward_mean_is_the_recursion() {
        let stamps = [0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.5, 9.0];
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows: Vec<Row> = stamps
            .iter()
            .zip(xs)
            .map(|(&t, x)| Row {
                group: None,
                clock: Some(ClockValue::F64(t)),
                session: None,
                values: vec![x],
                accept: true,
            })
            .collect();
        let h = 1.0;
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: h,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Mean,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let got = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
            .unwrap()
            .values[0]
            .clone();
        let mut y = xs[0];
        let mut want = vec![y];
        for i in 1..xs.len() {
            let a = 1.0 - (-(stamps[i] - stamps[i - 1]) / h).exp2();
            y = a * xs[i] + (1.0 - a) * y;
            want.push(y);
        }
        // Rows at one stamp share the window: the stamp's last recursion
        // value, which the zero-mass rows did not move.
        let want: Vec<f64> = (0..xs.len())
            .map(|i| {
                let last = (i..xs.len())
                    .take_while(|&j| stamps[j] == stamps[i])
                    .last()
                    .unwrap();
                want[last]
            })
            .collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-12, "row {i}: {g} vs {w}");
        }
    }

    /// One row's variance by Polars' own `ewm_var(adjust=False)` algorithm
    /// (pandas' `ewmcov`), written out: the old weight scaled by `1 − α`
    /// and the new row's by `α`, the weights renormalized to a sum of 1, the
    /// mean and the co-moment moved together, and unless `bias` the
    /// co-moment times `sum_wt² / (sum_wt² − sum_wt2)`.
    fn polars_ewm_var(xs: &[f64], h: f64, bias: bool) -> Vec<f64> {
        let alpha = 1.0 - (-1.0 / h).exp2();
        let (mut mean, mut cov) = (xs[0], 0.0);
        let (mut old_wt, mut sum_wt, mut sum_wt2) = (1.0f64, 1.0f64, 1.0f64);
        let mut out = vec![if bias { 0.0 } else { f64::NAN }];
        for &x in &xs[1..] {
            sum_wt *= 1.0 - alpha;
            sum_wt2 *= (1.0 - alpha) * (1.0 - alpha);
            old_wt *= 1.0 - alpha;
            let old_mean = mean;
            let wt_sum = old_wt + alpha;
            mean = (old_wt * old_mean + alpha * x) / wt_sum;
            cov = (old_wt * (cov + (old_mean - mean) * (old_mean - mean))
                + alpha * (x - mean) * (x - mean))
                / wt_sum;
            sum_wt += alpha;
            sum_wt2 += alpha * alpha;
            old_wt += alpha;
            sum_wt /= old_wt;
            sum_wt2 /= old_wt * old_wt;
            old_wt = 1.0;
            out.push(if bias {
                cov
            } else {
                let num = sum_wt * sum_wt;
                let den = num - sum_wt2;
                if den > 0.0 { num / den * cov } else { f64::NAN }
            });
        }
        out
    }

    /// On a clock that steps by 1, with `half_life` in rows, the variance is
    /// Polars' `ewm_var(adjust=False)`, whose algorithm is written out above:
    /// a stretch's first value held from before it weighs `(1 − α)^(n−1)`,
    /// as `adjust=False`'s first row does (task 212). The standard deviation
    /// is its root. The Python suite holds the same to Polars itself.
    #[test]
    fn the_variance_on_a_unit_clock_is_polars_adjust_false() {
        let mut rng = Lcg(77);
        let xs: Vec<f64> = (0..300).map(|_| rng.next() * 10.0 - 3.0).collect();
        let rows: Vec<Row> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| Row {
                group: None,
                clock: Some(ClockValue::F64(i as f64)),
                session: None,
                values: vec![x],
                accept: true,
            })
            .collect();
        for (h, bias) in [(1.0, false), (6.5, true), (6.5, false), (40.0, false)] {
            let k = KernelDef {
                direction: Direction::Backward,
                half_life: h,
                window_size: None,
                window_ns: None,
                closed: Closed::Right,
            };
            let op = |stat| OpDef {
                kernel: 0,
                stat,
                input: 0,
                min_samples: 1,
                partial: Partial::Keep,
                bias,
            };
            let got = run(
                &[k],
                &[op(Stat::Var), op(Stat::Std)],
                cfg(100.0, None, None),
                &rows,
                1,
            )
            .unwrap();
            let want = polars_ewm_var(&xs, h, bias);
            for (i, w) in want.iter().enumerate() {
                let (v, s) = (got.values[0][i], got.values[1][i]);
                assert_eq!(
                    v.is_nan(),
                    w.is_nan(),
                    "h {h} bias {bias} row {i}: {v} vs {w}"
                );
                if !w.is_nan() {
                    assert!(
                        (v - w).abs() <= 1e-12 * w.abs().max(1.0),
                        "h {h} row {i}: {v} vs {w}"
                    );
                    assert_eq!(s.to_bits(), v.sqrt().to_bits(), "row {i}");
                }
            }
        }
    }

    /// Every window holds one value however many rows it weighs, so its
    /// variance is 0 exactly, at any level and on every kernel: the merge
    /// of two segments moves the mean by their means' difference, 0, and
    /// adds nothing to `M2` (task 212). A mean square less a squared mean
    /// would cancel to rounding noise at a level.
    #[test]
    fn a_constant_input_has_a_variance_of_exactly_zero() {
        let (kernels, ops) = all_kernels();
        let var: Vec<usize> = (0..ops.len()).filter(|&o| ops[o].stat.is_var()).collect();
        for level in [0.1, 1e8 + 0.1, -3e12, 7.0] {
            let rows: Vec<Row> = stream(5, 400, 2, true, true)
                .into_iter()
                .map(|mut r| {
                    for v in &mut r.values {
                        if !v.is_nan() {
                            *v = level;
                        }
                    }
                    r
                })
                .collect();
            let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
            let t = run(&kernels, &ops, c, &rows, 1).unwrap();
            for &o in &var {
                let valued: Vec<f64> = t.values[o]
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .collect();
                assert!(
                    valued.len() > 20,
                    "level {level} output {o}: {}",
                    valued.len()
                );
                assert!(
                    valued.iter().all(|&v| v == 0.0),
                    "level {level} output {o}: {:?}",
                    valued.iter().find(|&&v| v != 0.0)
                );
            }
        }
    }

    /// A variance's mean is a compensated pair (`online_core::comp`): a
    /// value held row after row is approached as exact arithmetic approaches
    /// it. At 1e8 a plain mean stalls a few rounding steps short of a new
    /// level, and the variance about it settles on that gap's square, about
    /// 1e-15; the definition decays as `p (1 − p)`, `p = 2^(−T/h)` the weight
    /// left on the old level `T` units after its last row. The core's
    /// discount is an `exp2`, so the bound is relative, not bitwise.
    #[test]
    fn a_variance_at_a_level_decays_with_its_history() {
        let (a, n_a, n, h) = (1e8, 50usize, 500usize, 5.0);
        let rows: Vec<Row> = (0..n)
            .map(|i| Row {
                group: None,
                clock: Some(ClockValue::F64(i as f64)),
                session: None,
                values: vec![if i < n_a { a } else { a + 1.0 }],
                accept: true,
            })
            .collect();
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: h,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Var,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: true,
        };
        let v = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
            .unwrap()
            .values[0]
            .clone();
        for t in [n_a + 20, n_a + 100, n_a + 200, n - 1] {
            let p = (-((t - (n_a - 1)) as f64) / h).exp2();
            let want = p * (1.0 - p);
            assert!(
                (v[t] - want).abs() <= 1e-9 * want,
                "row {t}: {} vs {want}",
                v[t]
            );
        }
    }

    /// A variance looks back only, and needs a window without decay, as a
    /// mean does: refused when the core is built, and in a core read back
    /// from a damaged state.
    #[test]
    fn a_variance_looks_back_only() {
        let k = |direction| KernelDef {
            direction,
            half_life: 2.0,
            window_size: Some(4.0),
            window_ns: None,
            closed: Closed::Right,
        };
        let op = |kernel, stat| OpDef {
            kernel,
            stat,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let err = Windows::new(
            vec![k(Direction::Forward)],
            vec![op(0, Stat::Std)],
            ClockCfg::default(),
        )
        .expect_err("refused");
        assert!(err.contains("looks back only"), "{err}");
        let err = Windows::new(
            vec![KernelDef {
                half_life: f64::INFINITY,
                window_size: None,
                ..k(Direction::Backward)
            }],
            vec![op(0, Stat::Var)],
            ClockCfg::default(),
        )
        .expect_err("refused");
        assert!(err.contains("needs a window_size"), "{err}");
        let mut core = Windows::new(
            vec![k(Direction::Backward), k(Direction::Forward)],
            vec![op(0, Stat::Var), op(1, Stat::Sum)],
            ClockCfg::default(),
        )
        .unwrap();
        core.check().unwrap();
        core.ops[0].kernel = 1;
        let err = core.check().expect_err("refused");
        assert!(err.contains("looks back only"), "{err}");
    }

    /// A reset discards the windows open across it, whatever `partial`
    /// says; a cut keeps, nulls or drops them as it says. A window whose far
    /// edge is the last row before either is whole.
    #[test]
    fn a_reset_discards_and_a_cut_follows_partial() {
        let k = KernelDef {
            direction: Direction::Forward,
            half_life: 2.0,
            window_size: Some(5.0),
            window_ns: Some(ns_of(5.0)),
            closed: Closed::Right,
        };
        let ops = vec![
            OpDef {
                kernel: 0,
                stat: Stat::Sum,
                input: 0,
                min_samples: 1,
                partial: Partial::Keep,
                bias: false,
            },
            OpDef {
                kernel: 0,
                stat: Stat::Sum,
                input: 0,
                min_samples: 1,
                partial: Partial::Null,
                bias: false,
            },
            OpDef {
                kernel: 0,
                stat: Stat::Sum,
                input: 0,
                min_samples: 1,
                partial: Partial::Drop,
                bias: false,
            },
        ];
        let mk = |ts: &[f64]| -> Vec<Row> {
            ts.iter()
                .map(|&t| Row {
                    group: None,
                    clock: Some(ClockValue::F64(t)),
                    session: None,
                    values: vec![1.0],
                    accept: true,
                })
                .collect()
        };
        // A gap past the cap at row 3 cuts rows 0..3's windows.
        let cut = run(
            std::slice::from_ref(&k),
            &ops,
            cfg(4.0, None, None),
            &mk(&[0.0, 1.0, 2.0, 20.0, 21.0]),
            1,
        )
        .unwrap();
        assert!(cut.values[0][0] > 0.0 && cut.values[1][0].is_nan() && cut.drop[0]);
        assert!(!cut.drop[3]);
        // A step back restarts: discarded, null, never dropped.
        let reset = run(
            std::slice::from_ref(&k),
            &ops,
            cfg(4.0, None, Some(0.0)),
            &mk(&[0.0, 1.0, 2.0, 1.5, 2.5]),
            1,
        )
        .unwrap();
        assert!(reset.values[0][0].is_nan() && reset.values[1][0].is_nan() && !reset.drop[0]);
        // Task 173, PC2: row 0's window (0, 5] ends exactly on the last row
        // before the step back, so every row it covers has arrived: whole,
        // under every `partial`, as before a cut. Its sum from the
        // definition, `Σ 2^(−d/2)` over the rows 1, 2 and 5 later. The
        // windows that reach past that row are discarded, never dropped.
        let reset = run(
            &[k],
            &ops,
            cfg(100.0, None, Some(0.0)),
            &mk(&[0.0, 1.0, 2.0, 5.0, 3.0]),
            1,
        )
        .unwrap();
        let whole = 2f64.powf(-0.5) + 2f64.powf(-1.0) + 2f64.powf(-2.5);
        for o in 0..3 {
            assert!(
                (reset.values[o][0] - whole).abs() < 1e-15,
                "op {o}: {} vs {whole}",
                reset.values[o][0]
            );
            assert!(reset.values[o][1..].iter().all(|v| v.is_nan()), "op {o}");
        }
        assert!(reset.drop.iter().all(|d| !d), "{:?}", reset.drop);
    }

    /// `min_samples` nulls a window with fewer rows carrying a value.
    #[test]
    fn min_samples_counts_rows_with_a_value() {
        let rows: Vec<Row> = [1.0, f64::NAN, 2.0, 3.0]
            .iter()
            .enumerate()
            .map(|(i, &x)| Row {
                group: None,
                clock: Some(ClockValue::F64(i as f64)),
                session: None,
                values: vec![x],
                accept: true,
            })
            .collect();
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: f64::INFINITY,
            window_size: Some(3.0),
            window_ns: Some(ns_of(3.0)),
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 2,
            partial: Partial::Keep,
            bias: false,
        };
        let got = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
            .unwrap()
            .values[0]
            .clone();
        assert!(got[0].is_nan() && got[1].is_nan() && got[2] == 3.0 && got[3] == 5.0);
    }

    /// The state survives msgpack at every row and goes on identically.
    #[test]
    fn a_saved_state_resumes_at_any_row() {
        let (kernels, ops) = all_kernels();
        let rows = stream(9, 120, 2, true, true);
        let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
        let one = run(&kernels, &ops, c, &rows, 1).unwrap();
        for at in [1, 2, 17, 60, 119] {
            let mut core = Windows::new(kernels.clone(), ops.clone(), c).unwrap();
            core.set_grouped(true);
            let (mut t, mut next) = (Table::default(), 0);
            for r in &rows[..at] {
                push(&mut core, r).unwrap();
            }
            t.append(core.drain(), &mut next);
            let bytes = rmp_serde::to_vec_named(&core).unwrap();
            let mut core: Windows = rmp_serde::from_slice(&bytes).unwrap();
            core.check().unwrap();
            for r in &rows[at..] {
                push(&mut core, r).unwrap();
            }
            t.append(core.finish(), &mut next);
            assert!(t.same_bits(&one), "resumed at {at}");
        }
    }

    /// A step back is refused by name with the step, and nothing moves.
    #[test]
    fn a_step_back_is_refused_and_changes_nothing() {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: 2.0,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let mut core = Windows::new(vec![k], vec![op], cfg(10.0, None, None)).unwrap();
        let row = |t: f64| Row {
            group: None,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: vec![1.0],
            accept: true,
        };
        push(&mut core, &row(0.0)).unwrap();
        push(&mut core, &row(2.0)).unwrap();
        let before = rmp_serde::to_vec_named(&core).unwrap();
        let err = push(&mut core, &row(1.0)).unwrap_err();
        assert!(matches!(err, Refusal::Backwards { seq: 2, back, .. } if back == 1.0));
        assert_eq!(rmp_serde::to_vec_named(&core).unwrap(), before);
    }

    /// Review R2, W1: an edge was decided from two rounded policy times, so
    /// a row exactly one window from another landed on either side of it
    /// (0.4 - 0.1 is not 0.3 in a double); it is decided from the two rows'
    /// clocks, exact in nanoseconds, where Polars puts it, at any age.
    #[test]
    fn an_edge_is_decided_from_the_two_rows_clocks() {
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let same = |got: &[f64], want: &[f64]| {
            got.len() == want.len()
                && got
                    .iter()
                    .zip(want)
                    .all(|(x, y)| x.is_nan() && y.is_nan() || x == y)
        };
        const EPOCH_2024_MS: i64 = 1_704_067_200_000;
        for base in [0, EPOCH_2024_MS] {
            for (direction, closed, ms, want) in [
                // Backward "left": the row 300 ms older is at the far edge, in.
                (
                    Direction::Backward,
                    Closed::Left,
                    [0, 100, 400, 800],
                    [f64::NAN, 1.0, 1.0, f64::NAN],
                ),
                // Backward "right": the row 300 ms older is at the far edge, out.
                (
                    Direction::Backward,
                    Closed::Right,
                    [0, 400, 700, 1100],
                    [1.0, 1.0, 1.0, 1.0],
                ),
                // Forward "right": the row 300 ms later is at the far edge, in. A
                // forward window closes only at a stamp past its far edge, so the
                // fourth row is there to close row 1's; row 2's closes empty, and
                // row 3's is unresolved at the end.
                (
                    Direction::Forward,
                    Closed::Right,
                    [0, 100, 400, 800],
                    [1.0, 1.0, f64::NAN, f64::NAN],
                ),
            ] {
                let k = KernelDef {
                    direction,
                    half_life: f64::INFINITY,
                    window_size: Some(0.3),
                    window_ns: Some(ns_of(0.3)),
                    closed,
                };
                let rows: Vec<Row> = ms
                    .iter()
                    .map(|&m| Row {
                        group: None,
                        clock: Some(ClockValue::Ns((base + m) * 1_000_000)),
                        session: None,
                        values: vec![1.0],
                        accept: true,
                    })
                    .collect();
                let c = || cfg(100.0, None, None);
                let got = run(
                    std::slice::from_ref(&k),
                    std::slice::from_ref(&op),
                    c(),
                    &rows,
                    1,
                )
                .unwrap();
                assert!(
                    same(&got.values[0], &want),
                    "{direction:?} {closed:?} from {base}: {:?}",
                    got.values[0]
                );
                let bf = brute(
                    std::slice::from_ref(&k),
                    std::slice::from_ref(&op),
                    c(),
                    &rows,
                )
                .unwrap();
                assert!(got.same_bits(&bf), "the brute force decides the same edges");
            }
        }
    }

    /// Review R2, W3: the stream's clock took every row's session, so groups
    /// with sessions of their own saw a change at every row, and every
    /// group's windows started over at every row. With groups a session is
    /// each group's; without, the stream is the one group and its session
    /// change still ends the windows.
    #[test]
    fn with_groups_a_session_is_each_groups() {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: f64::INFINITY,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let row = |t: f64, g: Option<&str>, s: u64, x: f64| Row {
            group: g.map(str::to_string),
            clock: Some(ClockValue::F64(t)),
            session: Some(s),
            values: vec![x],
            accept: true,
        };
        let sums = |rows: &[Row]| {
            run(
                std::slice::from_ref(&k),
                std::slice::from_ref(&op),
                cfg(10.0, Some(SessionGap::Gap(1.0)), None),
                rows,
                1,
            )
            .unwrap()
            .values[0]
                .clone()
        };
        let (a, b) = (Some("a"), Some("b"));
        let rows = [
            row(0.0, a, 1, 1.0),
            row(1.0, b, 2, 10.0),
            row(2.0, a, 1, 3.0),
            row(3.0, b, 2, 30.0),
        ];
        assert_eq!(sums(&rows), [1.0, 10.0, 4.0, 40.0]);
        // A group's own session change ends its windows, and no other's.
        let rows = [
            row(0.0, a, 1, 1.0),
            row(1.0, b, 2, 10.0),
            row(2.0, a, 3, 3.0),
            row(3.0, b, 2, 30.0),
        ];
        assert_eq!(sums(&rows), [1.0, 10.0, 3.0, 40.0]);
        // One group: the stream's session, a change at rows 1, 2 and 3.
        let rows = [
            row(0.0, None, 1, 1.0),
            row(1.0, None, 2, 10.0),
            row(2.0, None, 3, 3.0),
            row(3.0, None, 2, 30.0),
        ];
        assert_eq!(sums(&rows), [1.0, 10.0, 3.0, 30.0]);
    }

    /// The seconds path, for a window given as a number on a temporal
    /// clock (no `window_ns`): the gap converts with the same operations a
    /// duration's text does, so every edge falls where the integer path
    /// puts it.
    #[test]
    fn a_numeric_window_on_a_temporal_clock_decides_the_same_edges() {
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let rows: Vec<Row> = [0, 100, 400, 800]
            .iter()
            .map(|&m| Row {
                group: None,
                clock: Some(ClockValue::Ns(m * 1_000_000)),
                session: None,
                values: vec![1.0],
                accept: true,
            })
            .collect();
        for (direction, closed) in [
            (Direction::Backward, Closed::Left),
            (Direction::Backward, Closed::Right),
            (Direction::Forward, Closed::Right),
            (Direction::Forward, Closed::Left),
        ] {
            let integer = KernelDef {
                direction,
                half_life: f64::INFINITY,
                window_size: Some(0.3),
                window_ns: Some(300_000_000),
                closed,
            };
            let seconds = KernelDef {
                window_ns: None,
                ..integer.clone()
            };
            let ops = std::slice::from_ref(&op);
            let a = run(
                std::slice::from_ref(&integer),
                ops,
                cfg(100.0, None, None),
                &rows,
                1,
            )
            .unwrap();
            let b = run(
                std::slice::from_ref(&seconds),
                ops,
                cfg(100.0, None, None),
                &rows,
                1,
            )
            .unwrap();
            assert!(
                a.same_bits(&b),
                "{direction:?} {closed:?}: {a:?} against {b:?}"
            );
            assert!(
                a.values[0].iter().filter(|v| !v.is_nan()).count() >= 2,
                "edges exercised"
            );
        }
    }

    /// Review R4, A1, recorded and not changed: under groups the stream's
    /// clock cuts a group silent past `gap_cap` when another group's row
    /// shows the silence, before the group's next row can say its session
    /// changed; a `session_gap = "reset"` then discards only what is still
    /// open. The same rows without a group column discard, since the one
    /// clock sees the gap and the session change at one row, and the reset
    /// comes first.
    #[test]
    fn a_silent_groups_reset_after_a_capped_gap_is_a_cut() {
        let k = KernelDef {
            direction: Direction::Forward,
            half_life: f64::INFINITY,
            window_size: Some(10.0),
            window_ns: None,
            closed: Closed::Right,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let row = |t: f64, g: Option<&str>, s: u64, x: f64| Row {
            group: g.map(str::to_string),
            clock: Some(ClockValue::F64(t)),
            session: Some(s),
            values: vec![x],
            accept: true,
        };
        let (a, b) = (Some("a"), Some("b"));
        let rows = [
            row(0.0, a, 1, 1.0),
            row(1.0, b, 1, 10.0),
            row(2.0, b, 1, 20.0),
            row(4.0, a, 1, 1.0),
            row(8.0, a, 1, 1.0),
            row(9.0, b, 2, 30.0),
        ];
        let c = || cfg(5.0, Some(SessionGap::Reset), None);
        let grouped = run(
            std::slice::from_ref(&k),
            std::slice::from_ref(&op),
            c(),
            &rows,
            1,
        )
        .unwrap();
        assert_eq!(
            grouped.values[0][1], 20.0,
            "cut at t = 8: the window over what b saw"
        );
        let alone: Vec<Row> = rows
            .iter()
            .filter(|r| r.group.as_deref() == b)
            .map(|r| Row {
                group: None,
                ..r.clone()
            })
            .collect();
        let one = run(
            std::slice::from_ref(&k),
            std::slice::from_ref(&op),
            c(),
            &alone,
            1,
        )
        .unwrap();
        assert!(
            one.values[0][0].is_nan(),
            "discarded at t = 9: {:?}",
            one.values[0]
        );
    }

    /// Review R4, A4: a kernel's two forms of its window agree, or the
    /// kernel is refused.
    #[test]
    fn a_kernels_window_ns_must_be_its_window_size() {
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
        };
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: 1.0,
            window_size: Some(0.3),
            window_ns: Some(300_000_001),
            closed: Closed::Right,
        };
        let err = Windows::new(vec![k], vec![op], cfg(10.0, None, None)).expect_err("refused");
        assert!(
            err.contains("window_ns 300000001 is not window_size 0.3"),
            "{err}"
        );
    }
}
