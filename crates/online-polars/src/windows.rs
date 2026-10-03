//! Exponentially weighted means with a hard cutoff, backward and forward
//! (docs/PLAN.md task 78, E69): the window core under `po.stream.with_windows`.
//!
//! For row `t` at policy time `τ_t`, over the rows `j` in its window, taken
//! in stream order:
//!
//! ```text
//! y_t = Σ w_j λ^|τ_j − τ_a| v_j / Σ w_j λ^|τ_j − τ_a|,   λ = 2^(−1/half_life)
//! ```
//!
//! *Backward* (`ewm`): the rows at or before `t` less than `horizon` older,
//! `(τ_t − horizon, τ_t]` -- an ordinary EWMA with a hard cutoff, over the
//! interval a time-indexed `rolling` takes by default. *Forward*
//! (`lookahead_rewm`): the rows after `t` less than `horizon` later,
//! `(τ_t, τ_t + horizon)`, so the weight falls away from the next row. A
//! weighted mean does not depend on the anchor `τ_a`: moving it multiplies
//! every weight by one constant. The anchor sits at the near end of every
//! segment only so that every factor is at most 1.
//!
//! **The mechanism.** Each contributing row is a segment `(S, W)` of one;
//! two segments combine associatively, the later one's sums discounted to
//! the earlier one's anchor going forward, the earlier one's to the later's
//! going backward. A two-stack queue over that monoid (Tangwongsan et al.'s
//! sliding-window aggregation) keeps a window's sums in O(1) work a row on
//! average, with no subtraction anywhere: prefix sums would cancel going
//! forward and overflow when flipped. Only contributing rows -- a usable
//! value and a usable, non-zero weight -- enter a queue, so memory is one
//! window of contributing rows per group, per output.
//!
//! **The clock is the model's.** Each group steps its own [`ClockState`], so
//! the horizon is measured on the policy clock: after `gap_cap` caps a
//! step and after `session_gap` replaces one. A capped gap and a session
//! change end every open window, as the `embargo` buffer releases every
//! waiting row on either (`stream.rs`, `apply_label_delay`): those windows
//! are partial. A reset discards them. After any of the three a group's
//! windows start over, and its policy time with them: `τ` is the time since
//! the group's last such event, so a double resolves it as finely as the
//! stretch between events allows, whatever the stream's age. The rows also
//! step one shared clock, in input order across groups, under the same
//! policy (the user, 2026-09-30: task 120's rules on the stream's clock,
//! sessions included): a step back the policy refuses is refused naming the
//! row, and a reset, a session change or a capped gap there ends every
//! group's windows.
//!
//! **A silent group holds the output for at most `gap_cap`** of the
//! stream's time. Rows leave in input order, so one group whose forward
//! windows are open holds back every later row of every group. Once the
//! stream's clock is more than `gap_cap` past a group's last row, that
//! group's next row is certain to open with a gap longer than the cap, so
//! its open windows are cut now rather than at that row. A row-count clock
//! steps each group only on its own rows, so it has no such bound.

use std::collections::{HashMap, VecDeque};

use online_core::{ClockAdvance, ClockCfg, ClockState, ClockValue, seconds_of_ns};
use serde::{Deserialize, Serialize};

use crate::stream::usable;

/// Which way a window looks from its row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// `ewm`: the rows at or before the row, less than `horizon` older.
    Backward,
    /// `lookahead_rewm`: the rows after the row, less than `horizon` later.
    Forward,
}

/// Whether later rows at a forward window's own policy time are "the next
/// rows". A zero step on the policy clock, not an equal stamp, is what makes
/// two rows share a clock (docs/PLAN.md task 78, *One clock, both forms*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SameClock {
    /// Stream order: a later row at the same clock is in the window.
    Include,
    /// What a time-indexed `rolling` does: only a positive step counts.
    Exclude,
}

/// What a window cut short before its horizon passed gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partial {
    /// The mean of what the window saw.
    Keep,
    /// Null.
    Null,
    /// The row leaves the output.
    Drop,
}

/// What a contributing row whose split value is not listed does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unlisted {
    /// Refuse it, naming the row.
    Error,
    /// Count it in the total only.
    Total,
    /// Count it nowhere.
    Ignore,
}

/// A split: one output per listed category of a column, and the total.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitDef {
    /// Index into a row's split codes.
    pub column: usize,
    /// How many categories are listed; a listed row's code is below this.
    pub categories: usize,
    /// Whether the total over every contributing row is an output too.
    pub total: bool,
    pub unlisted: Unlisted,
}

/// One window. A description with several columns, half-lives or horizons is
/// several of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowDef {
    pub direction: Direction,
    /// Index into a row's values.
    pub value: usize,
    /// Index into a row's weights; `None` weighs every row 1.
    pub weight: Option<usize>,
    /// Policy-clock units, above 0; infinite weighs the window evenly.
    pub half_life: f64,
    /// Policy-clock units, finite and above 0; `None` is no cutoff, which
    /// only a backward window may have.
    pub horizon: Option<f64>,
    pub split: Option<SplitDef>,
    /// Read by forward windows only.
    pub same_clock: SameClock,
    pub partial: Partial,
}

impl WindowDef {
    /// The outputs this window writes: the total first when there is one,
    /// then each listed category in order; one without a split.
    pub fn outputs(&self) -> usize {
        match &self.split {
            None => 1,
            Some(s) => s.categories + usize::from(s.total),
        }
    }

    /// # Errors
    ///
    /// A half-life that is not above 0, a horizon that is not finite and
    /// above 0 or that a forward window lacks, and a split that lists
    /// nothing or counts in a total it does not have.
    pub fn check(&self) -> Result<(), String> {
        if self.half_life.is_nan() || self.half_life <= 0.0 {
            return Err(format!("half_life must be above 0, got {}", self.half_life));
        }
        match (self.direction, self.horizon) {
            (Direction::Forward, None) => {
                return Err("a forward window needs a horizon".into());
            }
            (_, Some(h)) if !h.is_finite() || h <= 0.0 => {
                return Err(format!("horizon must be finite and above 0, got {h}"));
            }
            _ => {}
        }
        if let Some(s) = &self.split {
            if s.categories == 0 {
                return Err("a split lists at least one category".into());
            }
            if s.unlisted == Unlisted::Total && !s.total {
                return Err(
                    "unlisted=\"total\" counts a row in the total, and total=False has none".into(),
                );
            }
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

    /// The row's value and weight, if it counts: a usable value, and a
    /// usable weight other than zero.
    #[inline]
    fn contribution(&self, row: &RowIn<'_>) -> Option<(f64, f64)> {
        let v = row.values[self.value];
        let w = self.weight.map_or(1.0, |wi| row.weights[wi]);
        (usable(v) && usable(w) && w != 0.0).then_some((v, w))
    }
}

/// A row's value of a split column, as the caller read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitValue {
    /// The index of a listed category.
    Listed(u32),
    /// Present, and not in the list.
    Unlisted,
    Null,
}

/// One row as the core reads it: its group, clock and session, and every
/// column the windows read, already cast.
#[derive(Debug)]
pub struct RowIn<'a> {
    /// From [`Windows::group`].
    pub group: usize,
    /// `None` on a row-count clock.
    pub clock: Option<ClockValue>,
    pub session: Option<u64>,
    /// Per value column; not [`usable`] is missing.
    pub values: &'a [f64],
    /// Per weight column; not [`usable`] is missing.
    pub weights: &'a [f64],
    pub splits: &'a [SplitValue],
    /// Whether the spec behind `like=` would learn from this row. Its
    /// forward windows are null when it would not: a model never learns the
    /// target of a row it skips (docs/PLAN.md task 104, *Parity*, point 3).
    /// The row still steps the clocks and still counts in other rows'
    /// windows.
    pub accept: bool,
}

/// Why a row was refused. The caller names the row in the input.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The clock stepped back `back` clock units, from `prev` to `now`,
    /// where the policy refuses it: any step under `"error"`, and one no
    /// larger than `min_backwards_jump` (carried here) under
    /// `"reset_state"`. The two values let a caller state a temporal step
    /// exactly, in nanoseconds.
    Backwards {
        seq: u64,
        back: f64,
        prev: Option<ClockValue>,
        now: Option<ClockValue>,
        min_backwards_jump: Option<f64>,
    },
    /// A contributing row's split value is not listed, under `"error"`.
    Unlisted { seq: u64, window: usize },
    /// A weight below zero.
    NegativeWeight {
        seq: u64,
        window: usize,
        weight: f64,
    },
}

/// A segment of contributing rows: `S = Σ w λ^d v` and `W = Σ w λ^d`, each
/// row's `d` its distance from the segment's anchor, which is its earliest
/// row going forward and its latest going backward.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Seg {
    s: f64,
    w: f64,
    /// Policy time of the anchor.
    at: f64,
}

impl Seg {
    /// `a` then `b`, `a` the earlier: the factor is `λ` to a non-negative
    /// power, at most 1.
    #[inline]
    fn then(a: Seg, b: Seg, def: &WindowDef) -> Seg {
        let f = def.discount(b.at - a.at);
        match def.direction {
            Direction::Forward => Seg {
                s: a.s + f * b.s,
                w: a.w + f * b.w,
                at: a.at,
            },
            Direction::Backward => Seg {
                s: f * a.s + b.s,
                w: f * a.w + b.w,
                at: b.at,
            },
        }
    }

    /// The weighted mean; `None` when the weights sum to zero, the empty
    /// window hard rule 9 asks for, never 0/0.
    #[inline]
    fn mean(self) -> Option<f64> {
        (self.w > 0.0).then(|| self.s / self.w)
    }
}

/// A contributing row in a queue: its place in the input and its policy
/// time, which eviction reads, and its segment of one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Item {
    seq: u64,
    tau: f64,
    s: f64,
    w: f64,
}

impl Item {
    #[inline]
    fn seg(self) -> Seg {
        Seg {
            s: self.s,
            w: self.w,
            at: self.tau,
        }
    }
}

/// The two-stack queue. `front` holds the older rows, the oldest on top,
/// each beside the sum from itself through the newest row in `front`;
/// `back` holds the newer rows in order, and `back_sum` their sum. The
/// window's sum is `front`'s top sum then `back_sum`. Reaching the oldest
/// row with `front` empty first turns `back` over into it, so each row is
/// moved once: O(1) a row on average, and exact.
///
/// A backward window with no horizon evicts nothing, so it keeps only
/// `back_sum`, the running EWMA, and no rows.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Queue {
    front: Vec<(Item, Seg)>,
    back: Vec<Item>,
    back_sum: Option<Seg>,
}

impl Queue {
    fn push(&mut self, item: Item, def: &WindowDef) {
        self.back_sum = Some(match self.back_sum {
            None => item.seg(),
            Some(sum) => Seg::then(sum, item.seg(), def),
        });
        if def.horizon.is_some() {
            self.back.push(item);
        }
    }

    fn oldest(&mut self, def: &WindowDef) -> Option<Item> {
        if self.front.is_empty() {
            let mut newer: Option<Seg> = None;
            while let Some(item) = self.back.pop() {
                let sum = match newer {
                    None => item.seg(),
                    Some(n) => Seg::then(item.seg(), n, def),
                };
                newer = Some(sum);
                self.front.push((item, sum));
            }
            self.back_sum = None;
        }
        self.front.last().map(|&(item, _)| item)
    }

    /// Pop the rows `stale` says have left the window, oldest first.
    fn evict(&mut self, def: &WindowDef, stale: impl Fn(&Item) -> bool) {
        while let Some(item) = self.oldest(def) {
            if !stale(&item) {
                break;
            }
            self.front.pop();
        }
    }

    fn sum(&self, def: &WindowDef) -> Option<Seg> {
        match (self.front.last(), self.back_sum) {
            (Some(&(_, a)), Some(b)) => Some(Seg::then(a, b, def)),
            (Some(&(_, a)), None) => Some(a),
            (None, b) => b,
        }
    }

    fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
        self.back_sum = None;
    }

    fn len(&self) -> usize {
        self.front.len() + self.back.len()
    }
}

/// A row waiting for its group's forward windows to close.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Wait {
    seq: u64,
    tau: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Group {
    clock: ClockState,
    /// Policy time of the group's last row, from its last restart.
    tau: f64,
    /// The next row starts the windows over: the group has no row yet, or
    /// an event since its last row ended them.
    restart: bool,
    /// Per window, per output. A backward window's queues hold its last
    /// horizon of contributing rows; a forward one's, the contributing rows
    /// after the oldest row it has not closed.
    queues: Vec<Vec<Queue>>,
    /// Rows whose forward windows are not all closed, oldest first.
    waiting: VecDeque<Wait>,
    /// Per window: how many of `waiting`, from the front, it has closed.
    /// Always 0 for a backward window.
    closed: Vec<usize>,
    /// Raw clock of the group's last row, while it is in the silent list.
    last_raw: Option<ClockValue>,
    /// Neighbours in the silent list, the oldest `last_raw` at the head.
    prev: Option<usize>,
    next: Option<usize>,
    linked: bool,
}

/// How a forward window over a row ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// Its horizon passed.
    Complete,
    /// A capped gap or a session change cut it short: partial.
    Cut,
    /// A reset discarded it: null, and never dropped.
    Discard,
}

/// Per row not yet emitted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Meta {
    /// Forward windows not yet closed over the row.
    open: u32,
    accept: bool,
    drop: bool,
}

/// Rows out, in input order: `rows` rows from `first_seq` on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Emitted {
    pub first_seq: u64,
    pub rows: usize,
    /// Per output, per row: the mean, NaN where null.
    pub values: Vec<Vec<f64>>,
    /// Per window, per row: whether the window over the row was complete.
    pub complete: Vec<Vec<bool>>,
    /// Per row: whether a `"drop"` window was partial on it.
    pub drop: Vec<bool>,
}

/// The window core: every group's windows over one stream, fed rows in
/// input order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Windows {
    defs: Vec<WindowDef>,
    #[serde(with = "clock_form")]
    clock_cfg: ClockCfg,
    /// Where each window's first output sits among a row's outputs.
    offsets: Vec<usize>,
    n_outputs: usize,
    n_forward: u32,
    groups: Vec<Group>,
    /// Group keys, as text, to their index in `groups`; the null key apart.
    #[serde(with = "sorted_map")]
    index: HashMap<String, usize>,
    null_group: Option<usize>,
    /// The stream's own clock, in input order across groups.
    shared: ClockState,
    /// Groups with rows waiting, in the order of their last row: the oldest
    /// at the head.
    head: Option<usize>,
    tail: Option<usize>,
    /// The rows fed and not yet emitted, from `held_first` on: per row,
    /// `n_outputs` values (NaN null), `defs.len()` completes and one meta.
    held_first: u64,
    held_values: VecDeque<f64>,
    held_complete: VecDeque<bool>,
    held_meta: VecDeque<Meta>,
    /// How many held rows, from the first, every window has resolved, and
    /// how many of those no window drops: what [`Windows::drain`] emits.
    ready: usize,
    ready_kept: usize,
    /// The `seq` the next row fed takes.
    next_seq: u64,
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
fn elapsed(now: ClockValue, last: ClockValue) -> f64 {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
        (c, p) => c.seconds() - p.seconds(),
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

impl Windows {
    /// # Errors
    ///
    /// No windows, or a window whose half-life, horizon or split is not one
    /// this core runs.
    pub fn new(defs: Vec<WindowDef>, clock_cfg: ClockCfg) -> Result<Self, String> {
        if defs.is_empty() {
            return Err("at least one window is needed".into());
        }
        for d in &defs {
            d.check()?;
        }
        let mut offsets = Vec::with_capacity(defs.len());
        let mut n = 0;
        for d in &defs {
            offsets.push(n);
            n += d.outputs();
        }
        let n_forward = defs
            .iter()
            .filter(|d| d.direction == Direction::Forward)
            .count();
        Ok(Self {
            defs,
            clock_cfg,
            offsets,
            n_outputs: n,
            n_forward: u32::try_from(n_forward).map_err(|_| "too many windows")?,
            groups: Vec::new(),
            index: HashMap::new(),
            null_group: None,
            shared: ClockState::new(),
            head: None,
            tail: None,
            held_first: 0,
            held_values: VecDeque::new(),
            held_complete: VecDeque::new(),
            held_meta: VecDeque::new(),
            ready: 0,
            ready_kept: 0,
            next_seq: 0,
        })
    }

    pub fn defs(&self) -> &[WindowDef] {
        &self.defs
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
            .flat_map(|g| g.queues.iter().flatten())
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
        self.groups.push(Group {
            clock: ClockState::new(),
            tau: 0.0,
            restart: true,
            queues: self
                .defs
                .iter()
                .map(|d| vec![Queue::default(); d.outputs()])
                .collect(),
            waiting: VecDeque::new(),
            closed: vec![0; self.defs.len()],
            last_raw: None,
            prev: None,
            next: None,
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

    /// Feed one row. Earlier rows may be resolved by it; take them with
    /// [`Self::drain`].
    ///
    /// # Errors
    ///
    /// A step back the clock policy refuses, on the stream's clock or on
    /// the group's; a contributing row with an unlisted split value under
    /// `"error"`; a usable weight below zero. A refused row changes nothing.
    pub fn push(&mut self, row: &RowIn<'_>) -> Result<(), Refusal> {
        let seq = self.next_seq;
        let gi = row.group;
        self.check_row(row, seq)?;
        // Both clocks step on copies, kept only once neither refuses the row.
        let mut shared = self.shared.clone();
        let adv = shared.advance(&self.clock_cfg, row.clock, row.session, true);
        refuse_backwards(&adv, seq, self.shared.last_clock(), row.clock)?;
        let mut clock = self.groups[gi].clock.clone();
        let own = clock.advance(&self.clock_cfg, row.clock, row.session, true);
        refuse_backwards(&own, seq, self.groups[gi].clock.last_clock(), row.clock)?;
        self.shared = shared;
        self.next_seq += 1;

        // An event on the stream's clock reaches every group now; each
        // group's own clock would show it at that group's next row.
        if adv.reset {
            self.end_all(End::Discard);
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

        self.held_values
            .extend(std::iter::repeat_n(f64::NAN, self.n_outputs));
        self.held_complete
            .extend(std::iter::repeat_n(false, self.defs.len()));
        self.held_meta.push_back(Meta {
            open: self.n_forward,
            accept: row.accept,
            drop: false,
        });
        let at = self.held_meta.len() - 1;

        let g = &mut self.groups[gi];
        g.clock = clock;
        if own.reset || own.session_changed || own.capped || g.restart {
            // Every window of the group was ended above or before: nothing
            // refers to the old policy time, so it starts over at 0.
            for (d, qs) in self.defs.iter().zip(g.queues.iter_mut()) {
                if d.direction == Direction::Backward {
                    qs.iter_mut().for_each(Queue::clear);
                }
            }
            g.tau = 0.0;
            g.restart = false;
        } else {
            g.tau += own.d_clock;
        }
        let tau = g.tau;

        // The forward windows this row reaches past close first: it is not
        // in them.
        for (w, d) in self.defs.iter().enumerate() {
            let Some(h) = d.horizon.filter(|_| d.direction == Direction::Forward) else {
                continue;
            };
            while g.closed[w] < g.waiting.len() && tau - g.waiting[g.closed[w]].tau >= h {
                let t = g.waiting[g.closed[w]];
                g.closed[w] += 1;
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    n_windows: self.defs.len(),
                    values: &mut self.held_values,
                    complete: &mut self.held_complete,
                    meta: &mut self.held_meta,
                };
                close(
                    d,
                    w,
                    self.offsets[w],
                    &mut g.queues[w],
                    t,
                    End::Complete,
                    held,
                );
            }
        }
        retire(&self.defs, g);

        // Then the row joins every queue it counts in.
        for (w, d) in self.defs.iter().enumerate() {
            let Some((v, wt)) = d.contribution(row) else {
                continue;
            };
            let item = Item {
                seq,
                tau,
                s: wt * v,
                w: wt,
            };
            let qs = &mut g.queues[w];
            match &d.split {
                None => qs[0].push(item, d),
                Some(s) => {
                    let code = row.splits[s.column];
                    let listed = matches!(code, SplitValue::Listed(_));
                    if s.total && (listed || s.unlisted == Unlisted::Total) {
                        qs[0].push(item, d);
                    }
                    if let SplitValue::Listed(k) = code {
                        qs[usize::from(s.total) + k as usize].push(item, d);
                    }
                }
            }
        }

        // The backward windows are the row's own now.
        for (w, d) in self.defs.iter().enumerate() {
            if d.direction != Direction::Backward {
                continue;
            }
            if let Some(h) = d.horizon {
                for q in &mut g.queues[w] {
                    q.evict(d, |it| tau - it.tau >= h);
                }
            }
            let complete = d.horizon.is_none_or(|h| tau >= h);
            self.held_complete[at * self.defs.len() + w] = complete;
            if !complete && d.partial == Partial::Drop {
                self.held_meta[at].drop = true;
            }
            if complete || d.partial == Partial::Keep {
                for (o, q) in g.queues[w].iter().enumerate() {
                    if let Some(m) = q.sum(d).and_then(Seg::mean) {
                        self.held_values[at * self.n_outputs + self.offsets[w] + o] = m;
                    }
                }
            }
        }

        if self.n_forward > 0 {
            g.waiting.push_back(Wait { seq, tau });
            if let Some(now) = row.clock {
                g.last_raw = Some(now);
                self.link_back(gi);
            }
        }
        while let Some(m) = self.held_meta.get(self.ready).filter(|m| m.open == 0) {
            self.ready_kept += usize::from(!m.drop);
            self.ready += 1;
        }
        Ok(())
    }

    /// Refuse a row before anything moves.
    fn check_row(&self, row: &RowIn<'_>, seq: u64) -> Result<(), Refusal> {
        for (w, d) in self.defs.iter().enumerate() {
            if let Some(wi) = d.weight {
                let weight = row.weights[wi];
                if usable(weight) && weight < 0.0 {
                    return Err(Refusal::NegativeWeight {
                        seq,
                        window: w,
                        weight,
                    });
                }
            }
            if let Some(s) = &d.split {
                if s.unlisted == Unlisted::Error
                    && !matches!(row.splits[s.column], SplitValue::Listed(_))
                    && d.contribution(row).is_some()
                {
                    return Err(Refusal::Unlisted { seq, window: w });
                }
            }
        }
        Ok(())
    }

    /// End every group's open forward windows `how`; every group starts its
    /// windows over at its next row.
    fn end_all(&mut self, how: End) {
        for gi in 0..self.groups.len() {
            self.end_group(gi, how);
        }
    }

    /// Cut the open windows of every group whose last row is more than
    /// `gap_cap` behind `now`: the stream's clock has shown that its next
    /// row opens with a longer gap.
    fn end_silent(&mut self, now: ClockValue) {
        let cap = self.clock_cfg.gap_cap;
        if !cap.is_finite() {
            return;
        }
        while let Some(gi) = self.head {
            let last = self.groups[gi]
                .last_raw
                .expect("a listed group has a clock");
            if elapsed(now, last) <= cap {
                break;
            }
            self.end_group(gi, End::Cut);
        }
    }

    /// End a group's open forward windows `how` and empty its forward
    /// queues; its next row starts every window over.
    fn end_group(&mut self, gi: usize, how: End) {
        let g = &mut self.groups[gi];
        g.restart = true;
        for (w, d) in self.defs.iter().enumerate() {
            if d.direction != Direction::Forward {
                continue;
            }
            while g.closed[w] < g.waiting.len() {
                let t = g.waiting[g.closed[w]];
                g.closed[w] += 1;
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    n_windows: self.defs.len(),
                    values: &mut self.held_values,
                    complete: &mut self.held_complete,
                    meta: &mut self.held_meta,
                };
                close(d, w, self.offsets[w], &mut g.queues[w], t, how, held);
            }
            g.queues[w].iter_mut().for_each(Queue::clear);
        }
        retire(&self.defs, g);
        self.unlink(gi);
    }

    fn unlink(&mut self, gi: usize) {
        let g = &mut self.groups[gi];
        if !g.linked {
            return;
        }
        let (prev, next) = (g.prev.take(), g.next.take());
        g.linked = false;
        match prev {
            Some(p) => self.groups[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.groups[n].prev = prev,
            None => self.tail = prev,
        }
    }

    /// Move a group to the tail of the silent list: its last row is the
    /// stream's latest.
    fn link_back(&mut self, gi: usize) {
        self.unlink(gi);
        let tail = self.tail;
        let g = &mut self.groups[gi];
        g.prev = tail;
        g.linked = true;
        match tail {
            Some(t) => self.groups[t].next = Some(gi),
            None => self.head = Some(gi),
        }
        self.tail = Some(gi);
    }

    /// Resolved rows no window drops, ready to go out: how far a
    /// [`Self::drain_kept`] could go now.
    pub fn ready_kept(&self) -> usize {
        self.ready_kept
    }

    /// The rows every window has resolved, in input order.
    pub fn drain(&mut self) -> Emitted {
        self.emit(self.ready)
    }

    /// [`Self::drain`], stopping at the `limit`-th row no window drops: the
    /// resolved rows after it stay held, for a later drain or a saved
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

    /// The end of the input: every forward window still open is unresolved
    /// -- null whatever `partial` says, `complete` false, never dropped --
    /// and every row goes out. The core takes no more rows after this.
    pub fn finish(&mut self) -> Emitted {
        for gi in 0..self.groups.len() {
            let g = &mut self.groups[gi];
            g.waiting.clear();
            g.closed.iter_mut().for_each(|c| *c = 0);
            for (d, qs) in self.defs.iter().zip(g.queues.iter_mut()) {
                if d.direction == Direction::Forward {
                    qs.iter_mut().for_each(Queue::clear);
                }
            }
            self.unlink(gi);
        }
        self.ready = self.held_meta.len();
        self.ready_kept = self.held_meta.iter().filter(|m| !m.drop).count();
        self.emit(self.ready)
    }

    fn emit(&mut self, rows: usize) -> Emitted {
        let (n_out, n_win) = (self.n_outputs, self.defs.len());
        let mut out = Emitted {
            first_seq: self.held_first,
            rows,
            values: vec![Vec::with_capacity(rows); n_out],
            complete: vec![Vec::with_capacity(rows); n_win],
            drop: Vec::with_capacity(rows),
        };
        for (r, meta) in self.held_meta.drain(..rows).enumerate() {
            for (o, col) in out.values.iter_mut().enumerate() {
                col.push(self.held_values[r * n_out + o]);
            }
            for (w, col) in out.complete.iter_mut().enumerate() {
                col.push(self.held_complete[r * n_win + w]);
            }
            out.drop.push(meta.drop);
            self.ready_kept -= usize::from(!meta.drop);
        }
        self.ready -= rows;
        self.held_values.drain(..rows * n_out);
        self.held_complete.drain(..rows * n_win);
        self.held_first += rows as u64;
        out
    }
}

/// The rows not yet emitted, borrowed apart from the groups.
struct Held<'a> {
    first: u64,
    n_outputs: usize,
    n_windows: usize,
    values: &'a mut VecDeque<f64>,
    complete: &'a mut VecDeque<bool>,
    meta: &'a mut VecDeque<Meta>,
}

/// Close forward window `w` over waiting row `t`: the rows its queues hold
/// after `t` are its window.
fn close(
    d: &WindowDef,
    w: usize,
    offset: usize,
    queues: &mut [Queue],
    t: Wait,
    how: End,
    held: Held<'_>,
) {
    let exclude = d.same_clock == SameClock::Exclude;
    for q in queues.iter_mut() {
        q.evict(d, |it| it.seq <= t.seq || exclude && it.tau <= t.tau);
    }
    let r = usize::try_from(t.seq - held.first).expect("a waiting row is held");
    let meta = &mut held.meta[r];
    meta.open -= 1;
    held.complete[r * held.n_windows + w] = how == End::Complete;
    if how == End::Cut && d.partial == Partial::Drop {
        meta.drop = true;
    }
    let keep = match how {
        End::Complete => true,
        End::Cut => d.partial == Partial::Keep,
        End::Discard => false,
    };
    if keep && meta.accept {
        for (o, q) in queues.iter().enumerate() {
            if let Some(m) = q.sum(d).and_then(Seg::mean) {
                held.values[r * held.n_outputs + offset + o] = m;
            }
        }
    }
}

/// Drop the waiting rows every forward window has closed.
fn retire(defs: &[WindowDef], g: &mut Group) {
    let done = defs
        .iter()
        .zip(&g.closed)
        .filter(|(d, _)| d.direction == Direction::Forward)
        .map(|(_, &c)| c)
        .min()
        .unwrap_or(0);
    if done == 0 {
        return;
    }
    g.waiting.drain(..done);
    for (d, c) in defs.iter().zip(g.closed.iter_mut()) {
        if d.direction == Direction::Forward {
            *c -= done;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The core against a brute-force loop that knows the whole stream: it
    //! steps the same `ClockState`s (the definition of the policy clock),
    //! cuts the stream into each group's stretches between events, and
    //! computes every window from its definition, with no queue, no monoid
    //! and no closing order.

    use online_core::{OnClockReset, SessionGap};

    use super::*;

    #[derive(Debug, Clone)]
    struct Row {
        group: Option<String>,
        clock: Option<ClockValue>,
        session: Option<u64>,
        values: Vec<f64>,
        weights: Vec<f64>,
        splits: Vec<SplitValue>,
        accept: bool,
    }

    /// Every output of every row, in input order.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct Table {
        values: Vec<Vec<f64>>,
        complete: Vec<Vec<bool>>,
        drop: Vec<bool>,
    }

    impl Table {
        fn append(&mut self, e: Emitted, next: &mut u64) {
            assert_eq!(e.first_seq, *next, "rows leave in order, each once");
            *next += e.rows as u64;
            if self.values.is_empty() {
                self.values = vec![Vec::new(); e.values.len()];
                self.complete = vec![Vec::new(); e.complete.len()];
            }
            for (a, b) in self.values.iter_mut().zip(e.values) {
                a.extend(b);
            }
            for (a, b) in self.complete.iter_mut().zip(e.complete) {
                a.extend(b);
            }
            self.drop.extend(e.drop);
        }

        /// Bit for bit, NaN equal to NaN.
        fn same_bits(&self, other: &Table) -> bool {
            self.complete == other.complete
                && self.drop == other.drop
                && self.values.len() == other.values.len()
                && self.values.iter().zip(&other.values).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                })
        }
    }

    fn push(core: &mut Windows, r: &Row) -> Result<(), Refusal> {
        let group = core.group(r.group.as_deref());
        core.push(&RowIn {
            group,
            clock: r.clock,
            session: r.session,
            values: &r.values,
            weights: &r.weights,
            splits: &r.splits,
            accept: r.accept,
        })
    }

    /// The core, draining every `every` rows.
    fn run(
        defs: &[WindowDef],
        cfg: ClockCfg,
        rows: &[Row],
        every: usize,
    ) -> Result<Table, Refusal> {
        let mut core = Windows::new(defs.to_vec(), cfg).unwrap();
        let (mut t, mut next) = (Table::default(), 0);
        for (i, r) in rows.iter().enumerate() {
            push(&mut core, r)?;
            if (i + 1) % every == 0 {
                t.append(core.drain(), &mut next);
            }
        }
        t.append(core.finish(), &mut next);
        assert_eq!(next, rows.len() as u64);
        Ok(t)
    }

    fn contributes(d: &WindowDef, r: &Row, o: usize) -> Option<(f64, f64)> {
        let v = r.values[d.value];
        let w = d.weight.map_or(1.0, |wi| r.weights[wi]);
        if !(usable(v) && usable(w) && w != 0.0) {
            return None;
        }
        let ok = match &d.split {
            None => true,
            Some(s) => {
                let code = r.splits[s.column];
                if s.total && o == 0 {
                    matches!(code, SplitValue::Listed(_)) || s.unlisted == Unlisted::Total
                } else {
                    code == SplitValue::Listed((o - usize::from(s.total)) as u32)
                }
            }
        };
        ok.then_some((v, w))
    }

    /// `Σ w λ^|τ − τ_a| v / Σ w λ^|τ − τ_a|` over `(τ, v, w)`, anchored at
    /// `anchor`; NaN over nothing.
    fn mean(d: &WindowDef, members: &[(f64, f64, f64)], anchor: f64) -> f64 {
        if members.is_empty() {
            return f64::NAN;
        }
        let lam = |x: f64| {
            if d.half_life.is_infinite() {
                1.0
            } else {
                (-(x / d.half_life)).exp2()
            }
        };
        let (mut s, mut w) = (0.0, 0.0);
        for &(tau, v, wt) in members {
            let f = lam((tau - anchor).abs());
            s += f * wt * v;
            w += f * wt;
        }
        if w > 0.0 { s / w } else { f64::NAN }
    }

    fn elapsed_bf(now: ClockValue, last: ClockValue) -> f64 {
        match (now, last) {
            (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
            (ClockValue::F64(c), ClockValue::F64(p)) => c - p,
            _ => unreachable!("a test stream keeps one kind of clock"),
        }
    }

    fn brute(defs: &[WindowDef], cfg: ClockCfg, rows: &[Row]) -> Result<Table, Refusal> {
        struct G {
            clock: ClockState,
            tau: f64,
            epoch: usize,
            restart: bool,
            last_raw: Option<ClockValue>,
        }
        let n_forward = defs
            .iter()
            .filter(|d| d.direction == Direction::Forward)
            .count();
        let mut shared = ClockState::new();
        let mut groups: HashMap<Option<String>, G> = HashMap::new();
        // Per epoch: how it ended, if it did.
        let mut ends: Vec<Option<End>> = Vec::new();
        let mut live: Vec<bool> = Vec::new();
        let (mut epoch_of, mut tau_of) = (Vec::new(), Vec::new());
        for (i, r) in rows.iter().enumerate() {
            let seq = i as u64;
            for (w, d) in defs.iter().enumerate() {
                if let Some(wi) = d.weight {
                    let weight = r.weights[wi];
                    if usable(weight) && weight < 0.0 {
                        return Err(Refusal::NegativeWeight {
                            seq,
                            window: w,
                            weight,
                        });
                    }
                }
                if let Some(s) = &d.split {
                    if s.unlisted == Unlisted::Error
                        && !matches!(r.splits[s.column], SplitValue::Listed(_))
                        && contributes(
                            &WindowDef {
                                split: None,
                                ..d.clone()
                            },
                            r,
                            0,
                        )
                        .is_some()
                    {
                        return Err(Refusal::Unlisted { seq, window: w });
                    }
                }
            }
            let mut sc = shared.clone();
            let adv = sc.advance(&cfg, r.clock, r.session, true);
            refuse_backwards(&adv, seq, shared.last_clock(), r.clock)?;
            let mut gc = groups
                .get(&r.group)
                .map_or_else(ClockState::new, |g| g.clock.clone());
            let prev = gc.last_clock();
            let own = gc.advance(&cfg, r.clock, r.session, true);
            refuse_backwards(&own, seq, prev, r.clock)?;
            shared = sc;

            let end = |g: &mut G, how: End, ends: &mut Vec<Option<End>>, live: &mut Vec<bool>| {
                if live[g.epoch] {
                    live[g.epoch] = false;
                    ends[g.epoch] = Some(how);
                }
                g.restart = true;
            };
            if adv.reset || adv.session_changed || adv.capped {
                let how = if adv.reset { End::Discard } else { End::Cut };
                for g in groups.values_mut() {
                    end(g, how, &mut ends, &mut live);
                }
            }
            if let (Some(now), true) = (r.clock, cfg.gap_cap.is_finite() && n_forward > 0) {
                for g in groups.values_mut() {
                    if live[g.epoch] && elapsed_bf(now, g.last_raw.unwrap()) > cfg.gap_cap {
                        end(g, End::Cut, &mut ends, &mut live);
                    }
                }
            }
            let g = groups.entry(r.group.clone()).or_insert_with(|| G {
                clock: ClockState::new(),
                tau: 0.0,
                epoch: usize::MAX,
                restart: true,
                last_raw: None,
            });
            if g.epoch != usize::MAX && (own.reset || own.session_changed || own.capped) {
                let how = if own.reset { End::Discard } else { End::Cut };
                end(g, how, &mut ends, &mut live);
            }
            g.clock = gc;
            if g.restart {
                g.epoch = ends.len();
                ends.push(None);
                live.push(true);
                g.tau = 0.0;
                g.restart = false;
            } else {
                g.tau += own.d_clock;
            }
            g.last_raw = r.clock;
            epoch_of.push(g.epoch);
            tau_of.push(g.tau);
        }

        let n_out: usize = defs.iter().map(WindowDef::outputs).sum();
        let n = rows.len();
        let mut t = Table {
            values: vec![vec![f64::NAN; n]; n_out],
            complete: vec![vec![false; n]; defs.len()],
            drop: vec![false; n],
        };
        for i in 0..n {
            let (e, tau) = (epoch_of[i], tau_of[i]);
            let mut offset = 0;
            for (w, d) in defs.iter().enumerate() {
                let same = |j: usize| epoch_of[j] == e;
                match d.direction {
                    Direction::Backward => {
                        let complete = d.horizon.is_none_or(|h| tau >= h);
                        t.complete[w][i] = complete;
                        t.drop[i] |= !complete && d.partial == Partial::Drop;
                        if complete || d.partial == Partial::Keep {
                            for o in 0..d.outputs() {
                                let members: Vec<(f64, f64, f64)> = (0..=i)
                                    .filter(|&j| {
                                        same(j) && d.horizon.is_none_or(|h| tau - tau_of[j] < h)
                                    })
                                    .filter_map(|j| {
                                        contributes(d, &rows[j], o)
                                            .map(|(v, wt)| (tau_of[j], v, wt))
                                    })
                                    .collect();
                                let anchor = members.last().map_or(0.0, |m| m.0);
                                t.values[offset + o][i] = mean(d, &members, anchor);
                            }
                        }
                    }
                    Direction::Forward => {
                        let h = d.horizon.unwrap();
                        let later = || (i + 1..n).filter(|&j| same(j));
                        let how = if later().any(|j| tau_of[j] - tau >= h) {
                            Some(End::Complete)
                        } else {
                            ends[e]
                        };
                        t.complete[w][i] = how == Some(End::Complete);
                        t.drop[i] |= how == Some(End::Cut) && d.partial == Partial::Drop;
                        let keep = match how {
                            Some(End::Complete) => true,
                            Some(End::Cut) => d.partial == Partial::Keep,
                            Some(End::Discard) | None => false,
                        };
                        if keep && rows[i].accept {
                            for o in 0..d.outputs() {
                                let members: Vec<(f64, f64, f64)> = later()
                                    .filter(|&j| tau_of[j] - tau < h)
                                    .filter(|&j| {
                                        d.same_clock == SameClock::Include || tau_of[j] > tau
                                    })
                                    .filter_map(|j| {
                                        contributes(d, &rows[j], o)
                                            .map(|(v, wt)| (tau_of[j], v, wt))
                                    })
                                    .collect();
                                let anchor = members.first().map_or(0.0, |m| m.0);
                                t.values[offset + o][i] = mean(d, &members, anchor);
                            }
                        }
                    }
                }
                offset += d.outputs();
            }
        }
        Ok(t)
    }

    /// Agreement to rounding: the core sums by segments, the loop by rows.
    fn assert_close(core: &Table, bf: &Table, what: &str) {
        assert_eq!(core.complete, bf.complete, "{what}: complete");
        assert_eq!(core.drop, bf.drop, "{what}: drop");
        for (o, (a, b)) in core.values.iter().zip(&bf.values).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let ok = if x.is_nan() || y.is_nan() {
                    x.is_nan() && y.is_nan()
                } else {
                    (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
                };
                assert!(ok, "{what}: output {o}, row {i}: core {x}, brute force {y}");
            }
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
            xs[((self.next() * xs.len() as f64) as usize).min(xs.len() - 1)]
        }
    }

    fn def(direction: Direction, value: usize, half_life: f64, horizon: Option<f64>) -> WindowDef {
        WindowDef {
            direction,
            value,
            weight: None,
            half_life,
            horizon,
            split: None,
            same_clock: SameClock::Include,
            partial: if direction == Direction::Backward {
                Partial::Keep
            } else {
                Partial::Null
            },
        }
    }

    fn split(total: bool, unlisted: Unlisted) -> Option<SplitDef> {
        Some(SplitDef {
            column: 0,
            categories: 2,
            total,
            unlisted,
        })
    }

    /// Every kind of window at once: both directions, weights, splits under
    /// each `unlisted` and `total`, both `same_clock`, every `partial`.
    fn mixed_defs() -> Vec<WindowDef> {
        use Direction::*;
        vec![
            def(Backward, 0, 3.0, Some(8.0)),
            WindowDef {
                weight: Some(0),
                partial: Partial::Null,
                ..def(Backward, 1, 5.0, None)
            },
            WindowDef {
                weight: Some(1),
                split: split(true, Unlisted::Total),
                partial: Partial::Drop,
                ..def(Backward, 0, 2.0, Some(5.0))
            },
            def(Forward, 0, 3.0, Some(6.0)),
            WindowDef {
                weight: Some(1),
                split: split(true, Unlisted::Ignore),
                same_clock: SameClock::Exclude,
                partial: Partial::Keep,
                ..def(Forward, 0, 4.0, Some(9.0))
            },
            WindowDef {
                weight: Some(0),
                split: split(false, Unlisted::Ignore),
                partial: Partial::Drop,
                ..def(Forward, 1, f64::INFINITY, Some(4.0))
            },
        ]
    }

    #[derive(Clone, Copy)]
    enum Clock {
        Numbers,
        Nanos,
        Rows,
    }

    /// A stream in clock order across groups, with repeated stamps, gaps
    /// past the cap, session changes and -- under `"reset_state"` -- clock
    /// restarts far enough back that every group's own clock resets too.
    fn stream(seed: u64, n: usize, groups: usize, clock: Clock, restarts: bool) -> Vec<Row> {
        let mut rng = Lcg(seed);
        let mut t = 1e6;
        let mut session = 0u64;
        let mut era = 0.0;
        (0..n)
            .map(|_| {
                t += rng.pick(&[0.0, 0.0, 1.0, 1.0, 2.0, 0.5, 7.0, 40.0]);
                if rng.next() < 0.02 {
                    session += 1;
                }
                if restarts && rng.next() < 0.01 {
                    era -= 1e5;
                    t = 1e6;
                }
                let raw = era + t;
                let value = |rng: &mut Lcg| {
                    if rng.next() < 0.15 {
                        f64::NAN
                    } else {
                        rng.next() * 20.0 - 10.0
                    }
                };
                let weight = |rng: &mut Lcg| match rng.next() {
                    x if x < 0.15 => f64::NAN,
                    x if x < 0.30 => 0.0,
                    _ => rng.next() * 3.0 + 1e-3,
                };
                Row {
                    group: Some(format!("g{}", (rng.next() * groups as f64) as usize)),
                    clock: match clock {
                        Clock::Numbers => Some(ClockValue::F64(raw)),
                        Clock::Nanos => Some(ClockValue::Ns(
                            (raw * 1e9) as i64 + 1_700_000_000_000_000_000,
                        )),
                        Clock::Rows => None,
                    },
                    session: Some(session),
                    values: vec![value(&mut rng), value(&mut rng)],
                    weights: vec![weight(&mut rng), weight(&mut rng)],
                    splits: vec![rng.pick(&[
                        SplitValue::Listed(0),
                        SplitValue::Listed(1),
                        SplitValue::Unlisted,
                        SplitValue::Null,
                    ])],
                    accept: rng.next() < 0.9,
                }
            })
            .collect()
    }

    fn cfg(clock: Clock, on_clock_reset: OnClockReset, session_gap: SessionGap) -> ClockCfg {
        ClockCfg {
            gap_cap: match clock {
                Clock::Rows => f64::INFINITY,
                _ => 10.0,
            },
            on_clock_reset,
            session_gap: Some(session_gap),
            min_backwards_jump: if on_clock_reset == OnClockReset::ResetState {
                100.0
            } else {
                0.0
            },
        }
    }

    /// Every clock kind and policy, one to three groups, many seeds.
    #[test]
    fn every_window_matches_the_brute_force_loop() {
        let defs = mixed_defs();
        let mut checked = [0usize; 3];
        for seed in 0..40u64 {
            for (ci, clock) in [Clock::Numbers, Clock::Nanos, Clock::Rows]
                .into_iter()
                .enumerate()
            {
                for policy in [OnClockReset::Error, OnClockReset::ResetState] {
                    for gap in [SessionGap::Gap(3.0), SessionGap::Reset] {
                        let restarts =
                            policy == OnClockReset::ResetState && !matches!(clock, Clock::Rows);
                        let rows = stream(seed, 150, 1 + (seed % 3) as usize, clock, restarts);
                        let c = cfg(clock, policy, gap);
                        let core = run(&defs, c, &rows, 1).expect("the core accepts the stream");
                        let bf = brute(&defs, c, &rows).expect("so does the loop");
                        assert_close(&core, &bf, &format!("seed {seed}"));
                        checked[ci] += 1;
                    }
                }
            }
        }
        assert!(checked.iter().all(|&c| c == 160), "{checked:?}");
    }

    /// The loop is only evidence if the streams reach every case: each end
    /// of a forward window, a partial and a complete backward one, a drop,
    /// and a skipped row.
    #[test]
    fn the_streams_reach_every_case() {
        let defs = mixed_defs();
        let (mut complete, mut cut, mut discard, mut open, mut dropped) = (0, 0, 0, 0, 0);
        for seed in 0..40u64 {
            let rows = stream(seed, 150, 2, Clock::Numbers, true);
            let c = cfg(Clock::Numbers, OnClockReset::ResetState, SessionGap::Reset);
            let bf = brute(&defs, c, &rows).unwrap();
            let core = run(&defs, c, &rows, 1).unwrap();
            // Window 3 (forward, partial null): complete or not.
            for &c in &bf.complete[3] {
                if c {
                    complete += 1;
                } else {
                    open += 1;
                }
            }
            // Window 4 keeps a cut window's mean: a value with complete
            // false is a cut, a null one a discard or the end.
            let o4 = offset_of(&defs, 4);
            for (i, r) in rows.iter().enumerate() {
                if !bf.complete[4][i] && r.accept {
                    if bf.values[o4][i].is_nan() {
                        discard += 1;
                    } else {
                        cut += 1;
                    }
                }
            }
            dropped += core.drop.iter().filter(|&&d| d).count();
        }
        for (what, n) in [
            ("complete", complete),
            ("incomplete", open),
            ("cut", cut),
            ("discarded", discard),
            ("dropped", dropped),
        ] {
            assert!(n > 20, "{what}: {n}");
        }
    }

    fn offset_of(defs: &[WindowDef], w: usize) -> usize {
        defs[..w].iter().map(WindowDef::outputs).sum()
    }

    /// Draining after every row, every seventh or only at the end gives the
    /// same rows, bit for bit: nothing depends on when output is taken.
    #[test]
    fn when_output_is_taken_changes_nothing() {
        let defs = mixed_defs();
        for seed in 0..10u64 {
            let rows = stream(seed, 200, 3, Clock::Numbers, true);
            let c = cfg(
                Clock::Numbers,
                OnClockReset::ResetState,
                SessionGap::Gap(3.0),
            );
            let one = run(&defs, c, &rows, 1).unwrap();
            for every in [7, 64, 1000] {
                assert!(
                    one.same_bits(&run(&defs, c, &rows, every).unwrap()),
                    "seed {seed}, every {every}"
                );
            }
        }
    }

    /// A save and a load at any row, the rows before it already drained,
    /// continue exactly as one run.
    #[test]
    fn a_save_and_load_at_every_row_is_one_run() {
        let defs = mixed_defs();
        let rows = stream(3, 120, 2, Clock::Nanos, true);
        let c = cfg(Clock::Nanos, OnClockReset::ResetState, SessionGap::Reset);
        let one = run(&defs, c, &rows, 1).unwrap();
        for k in 0..=rows.len() {
            let mut core = Windows::new(defs.clone(), c).unwrap();
            let (mut t, mut next) = (Table::default(), 0);
            for r in &rows[..k] {
                push(&mut core, r).unwrap();
                t.append(core.drain(), &mut next);
            }
            let bytes = rmp_serde::to_vec(&core).unwrap();
            let mut core: Windows = rmp_serde::from_slice(&bytes).unwrap();
            for r in &rows[k..] {
                push(&mut core, r).unwrap();
                t.append(core.drain(), &mut next);
            }
            t.append(core.finish(), &mut next);
            assert!(t.same_bits(&one), "split at row {k}");
        }
    }

    fn row(group: &str, clock: f64, value: f64) -> Row {
        Row {
            group: Some(group.into()),
            clock: Some(ClockValue::F64(clock)),
            session: None,
            values: vec![value],
            weights: vec![1.0],
            splits: vec![SplitValue::Listed(0)],
            accept: true,
        }
    }

    fn numbers_cfg(gap_cap: f64) -> ClockCfg {
        ClockCfg {
            gap_cap,
            ..ClockCfg::default()
        }
    }

    /// A refused row changes nothing: the state's bytes before and after
    /// are equal, and the stream goes on as if it never came.
    #[test]
    fn a_refused_row_changes_nothing() {
        let defs = vec![
            WindowDef {
                weight: Some(0),
                ..def(Direction::Forward, 0, 2.0, Some(5.0))
            },
            WindowDef {
                split: Some(SplitDef {
                    column: 0,
                    categories: 1,
                    total: true,
                    unlisted: Unlisted::Error,
                }),
                ..def(Direction::Backward, 0, 2.0, Some(5.0))
            },
        ];
        let reset = ClockCfg {
            gap_cap: 10.0,
            on_clock_reset: OnClockReset::ResetState,
            session_gap: None,
            min_backwards_jump: 3.0,
        };
        type Case = (ClockCfg, Row, fn(&Refusal) -> bool);
        let cases: [Case; 4] = [
            (
                numbers_cfg(10.0),
                Row {
                    weights: vec![-1.0],
                    ..row("a", 3.0, 1.0)
                },
                |r| matches!(r, Refusal::NegativeWeight { seq: 2, window: 0, weight } if *weight == -1.0),
            ),
            (
                numbers_cfg(10.0),
                Row {
                    splits: vec![SplitValue::Null],
                    ..row("a", 3.0, 1.0)
                },
                |r| matches!(r, Refusal::Unlisted { seq: 2, window: 1 }),
            ),
            (
                numbers_cfg(10.0),
                row("b", 1.5, 1.0),
                |r| matches!(r, Refusal::Backwards { seq: 2, back, min_backwards_jump: None, .. } if *back == 0.5),
            ),
            (
                reset,
                row("b", 0.0, 1.0),
                |r| matches!(r, Refusal::Backwards { seq: 2, back, min_backwards_jump: Some(m), .. } if *back == 2.0 && *m == 3.0),
            ),
        ];
        for (c, bad, expect) in cases {
            let mut core = Windows::new(defs.clone(), c).unwrap();
            push(&mut core, &row("a", 1.0, 1.0)).unwrap();
            push(&mut core, &row("b", 2.0, 2.0)).unwrap();
            let before = rmp_serde::to_vec(&core).unwrap();
            let e = push(&mut core, &bad).unwrap_err();
            assert!(expect(&e), "{e:?}");
            assert_eq!(rmp_serde::to_vec(&core).unwrap(), before, "{e:?}");
        }
        // A row that would not count is never unlisted: a null value with a
        // null split passes under "error".
        let mut core = Windows::new(defs.clone(), numbers_cfg(10.0)).unwrap();
        push(
            &mut core,
            &Row {
                values: vec![f64::NAN],
                splits: vec![SplitValue::Null],
                ..row("a", 1.0, 0.0)
            },
        )
        .unwrap();
    }

    /// The forward window is `(τ_t, τ_t + horizon)`: a row exactly a horizon
    /// later closes it and is not in it; the backward one is
    /// `(τ_t − horizon, τ_t]`. Both anchored at the near end, so the next
    /// row weighs 1.
    #[test]
    fn the_ends_of_each_window() {
        let defs = vec![
            def(Direction::Forward, 0, 1.0, Some(2.0)),
            def(Direction::Backward, 0, 1.0, Some(2.0)),
        ];
        let rows: Vec<Row> = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (3.0, 8.0)]
            .iter()
            .map(|&(t, v)| row("a", t, v))
            .collect();
        let t = run(&defs, numbers_cfg(10.0), &rows, 1).unwrap();
        // Row 0: the rows at 1 (weight 1) only; the row at 2 closed it.
        assert_eq!(t.values[0][0], 2.0);
        assert_eq!(t.complete[0], [true, true, false, false]);
        // Row 2: the row at 3; unresolved at the end, so null.
        assert!(t.values[0][2].is_nan() && t.values[0][3].is_nan());
        // Backward at 3: the rows at 2 (weight 1/2) and 3 (weight 1).
        assert_eq!(t.values[1][3], (0.5 * 4.0 + 8.0) / 1.5);
        assert_eq!(t.complete[1], [false, false, true, true]);
    }

    /// `same_clock`: a later row at the row's own clock is in its forward
    /// window under "include", which is stream order, and not under
    /// "exclude", which is what a time-indexed `rolling` does.
    #[test]
    fn same_clock_is_about_steps() {
        let rows: Vec<Row> = [(0.0, 1.0), (0.0, 2.0), (1.0, 4.0), (9.0, 0.0)]
            .iter()
            .map(|&(t, v)| row("a", t, v))
            .collect();
        for (same_clock, want) in [
            (SameClock::Include, (2.0 + 0.5 * 4.0) / 1.5),
            (SameClock::Exclude, 4.0),
        ] {
            let defs = vec![WindowDef {
                same_clock,
                ..def(Direction::Forward, 0, 1.0, Some(5.0))
            }];
            let t = run(&defs, numbers_cfg(10.0), &rows, 1).unwrap();
            assert_eq!(t.values[0][0], want);
        }
    }

    /// A window whose weights sum to zero is empty: null, and complete.
    #[test]
    fn an_empty_window_is_null_and_complete() {
        let defs = vec![WindowDef {
            weight: Some(0),
            ..def(Direction::Forward, 0, 1.0, Some(2.0))
        }];
        let rows = vec![
            row("a", 0.0, 1.0),
            Row {
                weights: vec![0.0],
                ..row("a", 1.0, 5.0)
            },
            Row {
                weights: vec![f64::NAN],
                ..row("a", 1.5, 5.0)
            },
            row("a", 2.0, 3.0),
        ];
        let t = run(&defs, numbers_cfg(10.0), &rows, 1).unwrap();
        assert!(t.values[0][0].is_nan());
        assert!(t.complete[0][0]);
    }

    /// A constant column gives that constant to rounding -- a weighted mean
    /// of `n` rows is within about `(n + 2) ε` of it, and these windows hold
    /// at most 51 -- whatever the weights; and a
    /// half-life so short that every later factor underflows to 0 gives the
    /// near end's value: the anchor there keeps the sum from being empty.
    #[test]
    fn a_constant_and_an_underflow() {
        let mut rng = Lcg(9);
        let rows: Vec<Row> = (0..300)
            .map(|i| Row {
                weights: vec![rng.next() * 5.0 + 0.1],
                ..row("a", i as f64, 0.1)
            })
            .collect();
        for d in [Direction::Forward, Direction::Backward] {
            let defs = vec![WindowDef {
                weight: Some(0),
                ..def(d, 0, 7.0, Some(50.0))
            }];
            let t = run(&defs, numbers_cfg(10.0), &rows, 1).unwrap();
            for v in t.values[0].iter().filter(|v| !v.is_nan()) {
                assert!((v - 0.1).abs() <= 64.0 * f64::EPSILON * 0.1, "{v}");
            }
        }
        let rows: Vec<Row> = (0..50)
            .map(|i| row("a", i as f64 * 100.0, i as f64))
            .collect();
        let defs = vec![
            def(Direction::Forward, 0, 0.01, Some(1e4)),
            def(Direction::Backward, 0, 0.01, Some(1e4)),
        ];
        let t = run(&defs, numbers_cfg(1e3), &rows, 1).unwrap();
        for i in 0..49 {
            if t.complete[0][i] {
                assert_eq!(t.values[0][i], (i + 1) as f64);
            }
            assert_eq!(t.values[1][i], i as f64);
        }
    }

    /// Memory is one window of contributing rows: flat as the stream grows,
    /// and a backward window with no horizon keeps no rows at all.
    #[test]
    fn memory_is_one_window() {
        let defs = vec![
            def(Direction::Forward, 0, 5.0, Some(20.0)),
            def(Direction::Backward, 0, 5.0, Some(20.0)),
            def(Direction::Backward, 0, 5.0, None),
        ];
        let mut core = Windows::new(defs, numbers_cfg(10.0)).unwrap();
        let mut most = (0, 0);
        for i in 0..100_000 {
            // Ten rows a clock unit, one in ten contributing.
            let v = if i % 10 == 0 { 1.0 } else { f64::NAN };
            push(&mut core, &row("a", (i / 10) as f64, v)).unwrap();
            core.drain();
            most = (most.0.max(core.queued()), most.1.max(core.held()));
        }
        // Each queue holds at most one horizon of contributing rows (20 +
        // the one being pushed); output waits one horizon of all rows.
        assert!(most.0 <= 2 * 21, "{most:?}");
        assert!(most.1 <= 20 * 10 + 10, "{most:?}");
    }

    /// With only backward windows every row leaves as it arrives.
    #[test]
    fn backward_windows_hold_nothing() {
        let defs = vec![def(Direction::Backward, 0, 5.0, Some(20.0))];
        let mut core = Windows::new(defs, numbers_cfg(10.0)).unwrap();
        for i in 0..100 {
            push(
                &mut core,
                &row(if i % 2 == 0 { "a" } else { "b" }, i as f64, 1.0),
            )
            .unwrap();
            assert_eq!(core.drain().rows, 1);
        }
    }

    /// A group that falls silent holds the output for no more than
    /// `gap_cap` of the stream's time, and its windows are cut then, as
    /// its next row would cut them.
    #[test]
    fn a_silent_group_holds_output_for_at_most_the_cap() {
        let defs = vec![WindowDef {
            partial: Partial::Keep,
            ..def(Direction::Forward, 0, 5.0, Some(4.0))
        }];
        let mut core = Windows::new(defs.clone(), numbers_cfg(10.0)).unwrap();
        let mut rows = vec![row("quiet", 0.0, 1.0), row("quiet", 1.0, 2.0)];
        rows.extend((2..40).map(|t| row("busy", t as f64, 3.0)));
        let mut held = Vec::new();
        for r in &rows {
            push(&mut core, r).unwrap();
            core.drain();
            held.push(core.held());
        }
        // The quiet group's row at 1 holds everything until the stream's
        // clock passes 1 + 10, at the row at 12 (the row at 11 is exactly
        // the cap, which a step may take uncapped).
        assert!(held[..12].windows(2).all(|w| w[1] == w[0] + 1), "{held:?}");
        assert!(held[12] <= 5, "{held:?}");
        let bf = brute(&defs, numbers_cfg(10.0), &rows).unwrap();
        assert_close(
            &run(&defs, numbers_cfg(10.0), &rows, 1).unwrap(),
            &bf,
            "silent",
        );
        // Cut, and kept: the row at 0 saw the row at 1.
        assert_eq!(bf.values[0][0], 2.0);
        assert!(!bf.complete[0][0]);
    }

    #[test]
    fn a_description_it_cannot_run_is_refused() {
        let cases = [
            (def(Direction::Forward, 0, 1.0, None), "needs a horizon"),
            (def(Direction::Backward, 0, 0.0, None), "half_life"),
            (def(Direction::Backward, 0, f64::NAN, None), "half_life"),
            (
                def(Direction::Backward, 0, 1.0, Some(f64::INFINITY)),
                "horizon",
            ),
            (def(Direction::Backward, 0, 1.0, Some(0.0)), "horizon"),
            (
                WindowDef {
                    split: split(false, Unlisted::Total),
                    ..def(Direction::Backward, 0, 1.0, None)
                },
                "total=False",
            ),
            (
                WindowDef {
                    split: Some(SplitDef {
                        column: 0,
                        categories: 0,
                        total: true,
                        unlisted: Unlisted::Error,
                    }),
                    ..def(Direction::Backward, 0, 1.0, None)
                },
                "at least one",
            ),
        ];
        for (d, want) in cases {
            let e = Windows::new(vec![d], ClockCfg::default()).unwrap_err();
            assert!(e.contains(want), "{e}");
        }
        assert!(Windows::new(vec![], ClockCfg::default()).is_err());
    }
}
