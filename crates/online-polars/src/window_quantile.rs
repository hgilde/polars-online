//! The sketches behind `ewm_quantile` (docs/PLAN.md task 272 (d)): the
//! exponentially weighted quantile of a window, on `EwQuantile`'s buckets.
//!
//! **What it computes.** Each row with a value counts once, at its own time,
//! with the decayed weight `λ^|t_j − τ|` an `ewm_sum` gives it, and the
//! quantile at `p` is the smallest `x` with `W(≤ x) ≥ p W`: numpy's
//! `inverted_cdf` over the window's values and weights. The values are
//! bucketed as `online_core::EwQuantile` buckets them, `⌊64 (e + m − 1)⌋` of
//! `|x| = m · 2^e`, below zero by the magnitude's bucket, so the answer is
//! the representative of the bucket the exact quantile is in: within
//! `EW_QUANTILE_ALPHA = tanh(1/128)`, 0.78%, of it, relatively. A magnitude
//! below the smallest normal float counts as 0.
//!
//! **Three sketches serve the three shapes of a queue.** A backward queue
//! with no window keeps one running sketch (`back`): each value adds `1/s` to
//! its bucket, `s` the decay since the sketch's last fold, kept as one
//! factor beside the buckets and folded into them every 64 halvings, when an
//! end octave holding at most `1e-12` of the weight is dropped, as
//! `EwQuantile` does. A windowed queue is the two-stack queue of
//! `windows.rs`: its `back` stack's rows are a running sketch of the same
//! kind, and its `front` stack's rows, built once when `back` turns over
//! into it, a sketch of their own whose rows leave one at a time, oldest
//! first. A row leaving must take its weight out of its bucket without a
//! subtraction, which would leave a rounding behind in a bucket that should
//! be empty and let it build up row after row. So each `front` row keeps
//! the weight its bucket, its octave and the sketch's total held over the
//! rows newer than it, when `front` was built newest first, and a row
//! leaving puts those three back: the sketch is then exactly the one built
//! from the rows that are left. That is three doubles a row held, against a
//! sketch a row (a prefix merge per row, `2 · 64 · octaves` doubles) for the
//! textbook two-stack queue of mergeable sketches.
//!
//! **A read** walks the buckets of the sketches it reads, `front` and `back`
//! each weighed by its decay to the read and, looking ahead, the open row,
//! in value order, an octave at a time and then the octave's buckets: about
//! as many steps as the window's values span octaves, plus 64. A quantile
//! at or under one half is walked up from the smallest value, one above it
//! down from the largest.

use online_core::EwQuantile;
use serde::{Deserialize, Serialize};

use crate::windows::{Direction, KernelDef};

/// Buckets per octave, as `EwQuantile`'s.
const BINS: i32 = 64;
/// The decay a running sketch takes before it is folded into its buckets,
/// `2^-64`, as `EwQuantile`'s.
const RENORM: f64 = 5.421_010_862_427_522e-20;
/// An end octave at or under this share of the weight is dropped when an
/// unbounded sketch is folded, as `EwQuantile` drops an end bucket.
const PRUNE: f64 = 1e-12;

/// A value's place in a sketch's order: below zero by its magnitude's
/// bucket, zero, above zero by its bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Key {
    Neg(i32),
    Zero,
    Pos(i32),
}

impl Key {
    /// The key of a finite value.
    pub(crate) fn of(v: f64) -> Key {
        let a = v.abs();
        if a < f64::MIN_POSITIVE {
            Key::Zero
        } else if v > 0.0 {
            Key::Pos(EwQuantile::bucket_of(a))
        } else {
            Key::Neg(EwQuantile::bucket_of(a))
        }
    }

    /// The value a sketch reports for this key.
    pub(crate) fn value(self) -> f64 {
        match self {
            Key::Neg(k) => -EwQuantile::bucket_value(k),
            Key::Zero => 0.0,
            Key::Pos(k) => EwQuantile::bucket_value(k),
        }
    }
}

/// The octave a bucket is in.
#[inline]
fn octave(k: i32) -> i32 {
    k.div_euclid(BINS)
}

/// One sign's buckets: whole octaves from `lo`, a multiple of 64, each
/// octave's sum beside its buckets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Arm {
    lo: i32,
    w: Vec<f64>,
    oct: Vec<f64>,
}

impl Arm {
    /// The first and last octave held; `None` when empty.
    fn octaves(&self) -> Option<(i32, i32)> {
        (!self.oct.is_empty()).then(|| {
            let first = octave(self.lo);
            (first, first + self.oct.len() as i32 - 1)
        })
    }

    /// The slot of bucket `k`, the octaves grown to hold it.
    fn slot(&mut self, k: i32) -> usize {
        let o = octave(k);
        match self.octaves() {
            None => {
                self.lo = o * BINS;
                self.w = vec![0.0; BINS as usize];
                self.oct = vec![0.0];
            }
            Some((first, _)) if o < first => {
                let grow = (first - o) as usize;
                self.w
                    .splice(0..0, std::iter::repeat_n(0.0, grow * BINS as usize));
                self.oct.splice(0..0, std::iter::repeat_n(0.0, grow));
                self.lo = o * BINS;
            }
            Some((_, last)) if o > last => {
                let grow = (o - last) as usize;
                self.w.resize(self.w.len() + grow * BINS as usize, 0.0);
                self.oct.resize(self.oct.len() + grow, 0.0);
            }
            _ => {}
        }
        (k - self.lo) as usize
    }

    /// Add `x` to bucket `k`; the bucket's and its octave's weight before.
    fn add(&mut self, k: i32, x: f64) -> [f64; 2] {
        let i = self.slot(k);
        let o = i / BINS as usize;
        let prev = [self.w[i], self.oct[o]];
        self.w[i] += x;
        self.oct[o] += x;
        prev
    }

    /// Put back the bucket's and its octave's weight [`Arm::add`] returned.
    fn restore(&mut self, k: i32, prev: [f64; 2]) {
        let i = (k - self.lo) as usize;
        self.w[i] = prev[0];
        self.oct[i / BINS as usize] = prev[1];
    }

    #[inline]
    fn bucket(&self, k: i32) -> f64 {
        let i = i64::from(k) - i64::from(self.lo);
        if i < 0 {
            return 0.0;
        }
        self.w.get(i as usize).copied().unwrap_or(0.0)
    }

    #[inline]
    fn octave_weight(&self, o: i32) -> f64 {
        let i = i64::from(o) - i64::from(octave(self.lo));
        if i < 0 {
            return 0.0;
        }
        self.oct.get(i as usize).copied().unwrap_or(0.0)
    }

    fn scale(&mut self, f: f64) {
        self.w.iter_mut().for_each(|x| *x *= f);
        self.oct.iter_mut().for_each(|x| *x *= f);
    }

    fn sum(&self) -> f64 {
        self.oct.iter().sum()
    }

    /// Drop the end octaves holding at most `floor`.
    fn prune(&mut self, floor: f64) {
        let Some(first) = self.oct.iter().position(|&x| x > floor) else {
            *self = Arm::default();
            return;
        };
        let last = self.oct.iter().rposition(|&x| x > floor).unwrap_or(first);
        self.oct.truncate(last + 1);
        self.w.truncate((last + 1) * BINS as usize);
        self.oct.drain(..first);
        self.w.drain(..first * BINS as usize);
        self.lo += first as i32 * BINS;
    }

    fn clear(&mut self) {
        self.w.clear();
        self.oct.clear();
        self.lo = 0;
    }

    /// Whole octaves inside the buckets a value can have, every weight
    /// finite and at least 0: what the reads index by.
    fn has_shape(&self) -> bool {
        let indices = i64::from(EwQuantile::bucket_of(f64::MIN_POSITIVE))
            ..=i64::from(EwQuantile::bucket_of(f64::MAX));
        let ok = |x: &f64| x.is_finite() && *x >= 0.0;
        if self.oct.is_empty() {
            return self.w.is_empty();
        }
        let first = i64::from(self.lo);
        let last = first + self.w.len() as i64 - 1;
        self.lo.rem_euclid(BINS) == 0
            && Some(self.w.len()) == self.oct.len().checked_mul(BINS as usize)
            && indices.contains(&(first + i64::from(BINS) - 1))
            && indices.contains(&(last - i64::from(BINS) + 1))
            && self.w.iter().all(ok)
            && self.oct.iter().all(ok)
    }
}

/// Bucket weights of both signs and zero, each over `scale`, anchored at
/// `anchor` (NaN before the first value): a stored weight times `scale` is
/// the weight seen from `anchor`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Sketch {
    neg: Arm,
    zero: f64,
    pos: Arm,
    total: f64,
    anchor: f64,
    scale: f64,
}

impl Default for Sketch {
    fn default() -> Self {
        Sketch {
            neg: Arm::default(),
            zero: 0.0,
            pos: Arm::default(),
            total: 0.0,
            anchor: f64::NAN,
            scale: 1.0,
        }
    }
}

impl Sketch {
    fn clear(&mut self) {
        self.neg.clear();
        self.pos.clear();
        self.zero = 0.0;
        self.total = 0.0;
        self.anchor = f64::NAN;
        self.scale = 1.0;
    }

    /// Add stored weight `x` at `key`; the bucket's, its octave's and the
    /// total's stored weight before.
    fn add(&mut self, key: Key, x: f64) -> [f64; 3] {
        let total = self.total;
        self.total += x;
        match key {
            Key::Zero => {
                let z = self.zero;
                self.zero += x;
                [z, 0.0, total]
            }
            Key::Neg(k) => {
                let [b, o] = self.neg.add(k, x);
                [b, o, total]
            }
            Key::Pos(k) => {
                let [b, o] = self.pos.add(k, x);
                [b, o, total]
            }
        }
    }

    fn restore(&mut self, key: Key, prev: [f64; 3]) {
        match key {
            Key::Zero => self.zero = prev[0],
            Key::Neg(k) => self.neg.restore(k, [prev[0], prev[1]]),
            Key::Pos(k) => self.pos.restore(k, [prev[0], prev[1]]),
        }
        self.total = prev[2];
    }

    /// A value at `tau`, at or after every value before it, seen from the
    /// newest: the decay since the anchor kept in `scale` and folded into
    /// the buckets every 64 halvings, the end octaves of negligible weight
    /// dropped then where `prune`.
    fn add_newest(&mut self, k: &KernelDef, key: Key, tau: f64, prune: bool) {
        if self.anchor.is_nan() {
            self.scale = 1.0;
        } else {
            self.scale *= k.discount(tau - self.anchor);
            if self.scale < RENORM {
                self.fold(prune);
            }
        }
        self.anchor = tau;
        self.add(key, 1.0 / self.scale);
    }

    /// A value at `tau`, at or after the anchor, seen from the anchor: a
    /// forward window's rows, weighed from the oldest.
    fn add_from_anchor(&mut self, k: &KernelDef, key: Key, tau: f64) {
        if self.anchor.is_nan() {
            self.anchor = tau;
        }
        self.add(key, k.discount(tau - self.anchor));
    }

    /// Fold `scale` into the buckets.
    fn fold(&mut self, prune: bool) {
        let s = self.scale;
        self.neg.scale(s);
        self.pos.scale(s);
        self.zero *= s;
        self.scale = 1.0;
        self.total = self.zero + self.neg.sum() + self.pos.sum();
        if prune {
            let floor = PRUNE * self.total;
            self.neg.prune(floor);
            self.pos.prune(floor);
            if self.zero <= floor {
                self.zero = 0.0;
            }
            self.total = self.zero + self.neg.sum() + self.pos.sum();
        }
    }

    fn has_weight(&self) -> bool {
        !self.anchor.is_nan() && self.total > 0.0
    }

    fn has_shape(&self) -> bool {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        self.neg.has_shape()
            && self.pos.has_shape()
            && ok(self.zero)
            && ok(self.total)
            && !self.anchor.is_infinite()
            && self.scale > 0.0
            && self.scale <= 1.0
    }

    /// The buckets held: what the sketch costs.
    fn buckets(&self) -> usize {
        self.neg.w.len() + self.pos.w.len() + 1
    }
}

/// One quantile operator's sketches in a queue ([module docs](self)).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct QSide {
    /// A windowed queue's `front` rows, from the anchor: the newest looking
    /// back, the oldest looking ahead.
    front: Sketch,
    /// Per `front` row, in its order, three: the stored weight of its
    /// bucket, its octave and the total over the rows newer than it; NaN for
    /// a row with no value.
    prev: Vec<f64>,
    /// A windowed queue's `back` rows, or every row of an unbounded one.
    back: Sketch,
}

impl QSide {
    /// A row's value `x` (not NaN) at `tau` joins the queue.
    pub(crate) fn push(&mut self, k: &KernelDef, tau: f64, x: f64) {
        let key = Key::of(x);
        match (k.window_size, k.direction) {
            (None, _) => self.back.add_newest(k, key, tau, true),
            (Some(_), Direction::Backward) => self.back.add_newest(k, key, tau, false),
            (Some(_), Direction::Forward) => self.back.add_from_anchor(k, key, tau),
        }
    }

    /// `back`'s rows, their times `tau` and values `x(i)` (NaN for none),
    /// oldest first, turn over into `front`, which is empty: `front` is built
    /// newest first, as the queue pushes them, each row keeping what its
    /// bucket, octave and the total held before it.
    pub(crate) fn turn_over(&mut self, k: &KernelDef, tau: &[f64], x: impl Fn(usize) -> f64) {
        self.front.clear();
        self.prev.clear();
        if let (Some(&first), Some(&last)) = (tau.first(), tau.last()) {
            let anchor = match k.direction {
                Direction::Backward => last,
                Direction::Forward => first,
            };
            self.front.anchor = anchor;
            for i in (0..tau.len()).rev() {
                let v = x(i);
                if v.is_nan() {
                    self.prev.extend([f64::NAN; 3]);
                    continue;
                }
                let w = match k.direction {
                    Direction::Backward => k.discount(anchor - tau[i]),
                    Direction::Forward => k.discount(tau[i] - anchor),
                };
                let prev = self.front.add(Key::of(v), w);
                self.prev.extend(prev);
            }
        }
        self.back.clear();
    }

    /// `front`'s oldest row, at index `top` with value `x`, leaves: its
    /// bucket, octave and the total are what they were before it.
    pub(crate) fn pop(&mut self, top: usize, x: f64) {
        if !x.is_nan() {
            let p = &self.prev[3 * top..3 * top + 3];
            self.front.restore(Key::of(x), [p[0], p[1], p[2]]);
        }
        self.prev.truncate(3 * top);
        if top == 0 {
            self.front.clear();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.front.clear();
        self.prev.clear();
        self.back.clear();
    }

    /// Whether the sketches can be read and added to, `front` holding a row
    /// of `prev` for each of its `front_rows`.
    pub(crate) fn has_shape(&self, front_rows: usize) -> bool {
        Some(self.prev.len()) == front_rows.checked_mul(3)
            && self.front.has_shape()
            && self.back.has_shape()
    }

    /// The buckets the sketches hold.
    pub(crate) fn buckets(&self) -> usize {
        self.front.buckets() + self.back.buckets()
    }

    /// The quantile at `p` of the window the sketches hold, with `open`, a
    /// forward window's open row as its time and value, counted too; `None`
    /// with no weight.
    pub(crate) fn read(&self, k: &KernelDef, p: f64, open: Option<(f64, f64)>) -> Option<f64> {
        let held = [&self.front, &self.back].map(|s| s.has_weight().then_some(s));
        let anchors = held
            .iter()
            .flatten()
            .map(|s| s.anchor)
            .chain(open.map(|o| o.0));
        // Each part seen from the anchor nearest the window's row: the newest
        // looking back, the oldest looking ahead, so no factor is above 1.
        let forward = k.direction == Direction::Forward;
        let reference = if forward {
            anchors.fold(f64::INFINITY, f64::min)
        } else {
            anchors.fold(f64::NEG_INFINITY, f64::max)
        };
        let seen = |a: f64| {
            if forward {
                k.discount(a - reference)
            } else {
                k.discount(reference - a)
            }
        };
        let mut parts = [(&self.front, 0.0); 2];
        let mut n = 0;
        for s in held.into_iter().flatten() {
            parts[n] = (s, s.scale * seen(s.anchor));
            n += 1;
        }
        let extra = open.map(|(t, v)| (Key::of(v), seen(t)));
        quantile(&parts[..n], extra, p)
    }
}

/// The quantile at `p` of the sketches `parts`, each weighed by its factor,
/// and `extra`, one value at its weight: the representative of the first
/// bucket, in value order, at which the weight reaches `p` of the total.
/// `None` with no weight.
pub(crate) fn quantile(parts: &[(&Sketch, f64)], extra: Option<(Key, f64)>, p: f64) -> Option<f64> {
    let extra_w = extra.map_or(0.0, |(_, w)| w);
    let total: f64 = parts.iter().map(|(s, c)| c * s.total).sum::<f64>() + extra_w;
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let up = p <= 0.5;
    let mut walk = Walk {
        parts,
        extra,
        target: if up { p * total } else { (1.0 - p) * total },
        up,
        cum: 0.0,
        last: None,
    };
    // In value order: the negative arm from its largest magnitude, zero, the
    // positive arm from its smallest; or the reverse, from the top.
    let found = if up {
        walk.arm(true, true)
            .or_else(|| walk.zero())
            .or_else(|| walk.arm(false, false))
    } else {
        walk.arm(false, true)
            .or_else(|| walk.zero())
            .or_else(|| walk.arm(true, false))
    };
    // A rounding can leave the sums a hair short of the target at the far
    // end: the last bucket with weight is then the one.
    found.or_else(|| walk.last()).map(Key::value)
}

/// Where a walk last saw weight: a key, or an octave it passed whole.
#[derive(Clone, Copy)]
enum Last {
    Key(Key),
    Octave { neg: bool, o: i32, down: bool },
}

/// A walk over the buckets of several sketches in one order.
struct Walk<'a> {
    parts: &'a [(&'a Sketch, f64)],
    extra: Option<(Key, f64)>,
    target: f64,
    /// Up from the smallest value, reaching the target (`W(≤ x) ≥ p W`);
    /// else down from the largest, passing it (`W(≥ x) > (1 − p) W`).
    up: bool,
    cum: f64,
    last: Option<Last>,
}

impl Walk<'_> {
    fn reached(&self, cum: f64) -> bool {
        if self.up {
            cum >= self.target
        } else {
            cum > self.target
        }
    }

    fn zero(&mut self) -> Option<Key> {
        let w: f64 = self.parts.iter().map(|(s, c)| c * s.zero).sum::<f64>()
            + self
                .extra
                .filter(|(k, _)| *k == Key::Zero)
                .map_or(0.0, |(_, w)| w);
        if w > 0.0 {
            self.cum += w;
            self.last = Some(Last::Key(Key::Zero));
            if self.reached(self.cum) {
                return Some(Key::Zero);
            }
        }
        None
    }

    fn side(neg: bool, s: &Sketch) -> &Arm {
        if neg { &s.neg } else { &s.pos }
    }

    /// The extra value's bucket, where it is on this arm.
    fn extra_on(&self, neg: bool) -> Option<(i32, f64)> {
        self.extra.and_then(|(k, w)| match (k, neg) {
            (Key::Neg(b), true) | (Key::Pos(b), false) => Some((b, w)),
            _ => None,
        })
    }

    /// The octaves the parts and the extra value hold on one arm.
    fn octaves(&self, neg: bool) -> Option<(i32, i32)> {
        let held = self
            .parts
            .iter()
            .filter_map(|(s, _)| Self::side(neg, s).octaves());
        let ex = self.extra_on(neg).map(|(b, _)| (octave(b), octave(b)));
        held.chain(ex).reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
    }

    fn octave_weight(&self, neg: bool, o: i32) -> f64 {
        self.parts
            .iter()
            .map(|(s, c)| c * Self::side(neg, s).octave_weight(o))
            .sum::<f64>()
            + self
                .extra_on(neg)
                .filter(|(b, _)| octave(*b) == o)
                .map_or(0.0, |(_, w)| w)
    }

    fn bucket(&self, neg: bool, b: i32) -> f64 {
        self.parts
            .iter()
            .map(|(s, c)| c * Self::side(neg, s).bucket(b))
            .sum::<f64>()
            + self
                .extra_on(neg)
                .filter(|(e, _)| *e == b)
                .map_or(0.0, |(_, w)| w)
    }

    /// The `i`-th bucket of octave `o` in walk order.
    fn nth(o: i32, i: i32, down: bool) -> i32 {
        if down {
            o * BINS + BINS - 1 - i
        } else {
            o * BINS + i
        }
    }

    /// One arm, the negative where `neg`, its magnitudes from the largest
    /// down where `down`: the negative arm walks its magnitudes down going up
    /// in value, the positive arm going down.
    fn arm(&mut self, neg: bool, down: bool) -> Option<Key> {
        let (lo, hi) = self.octaves(neg)?;
        let key = |b: i32| if neg { Key::Neg(b) } else { Key::Pos(b) };
        for i in 0..=(hi - lo) {
            let o = if down { hi - i } else { lo + i };
            let ow = self.octave_weight(neg, o);
            if ow.is_nan() || ow <= 0.0 {
                continue;
            }
            if !self.reached(self.cum + ow) {
                self.cum += ow;
                self.last = Some(Last::Octave { neg, o, down });
                continue;
            }
            // The target is in this octave: its buckets, in order.
            let mut last = None;
            for j in 0..BINS {
                let b = Self::nth(o, j, down);
                let w = self.bucket(neg, b);
                if w > 0.0 {
                    self.cum += w;
                    last = Some(b);
                    if self.reached(self.cum) {
                        return Some(key(b));
                    }
                }
            }
            // The octave's sum reached it and its buckets' a rounding short.
            if let Some(b) = last {
                return Some(key(b));
            }
        }
        None
    }

    /// The last bucket the walk saw weight in.
    fn last(&self) -> Option<Key> {
        match self.last? {
            Last::Key(k) => Some(k),
            Last::Octave { neg, o, down } => (0..BINS)
                .rev()
                .map(|j| Self::nth(o, j, down))
                .find(|&b| self.bucket(neg, b) > 0.0)
                .map(|b| if neg { Key::Neg(b) } else { Key::Pos(b) }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windows::Closed;

    fn kernel(direction: Direction, half_life: f64, window: Option<f64>) -> KernelDef {
        KernelDef {
            direction,
            half_life,
            window_size: window,
            window_ns: None,
            closed: Closed::Right,
        }
    }

    /// The exact quantile at `p` of values and weights: the smallest `x`
    /// with `W(≤ x) ≥ p W`, numpy's `inverted_cdf`.
    fn exact(mut vw: Vec<(f64, f64)>, p: f64) -> f64 {
        vw.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = vw.iter().map(|x| x.1).sum();
        let mut cum = 0.0;
        for (v, w) in &vw {
            cum += w;
            if cum >= p * total {
                return *v;
            }
        }
        vw.last().unwrap().0
    }

    fn within(got: f64, want: f64) -> bool {
        (got - want).abs() <= online_core::EW_QUANTILE_ALPHA * want.abs() * (1.0 + 1e-12)
            || want.abs() < f64::MIN_POSITIVE && got == 0.0
    }

    /// A seeded stream of signed values over ten decades, with zeros.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                let e = ((s >> 3) % 11) as i32 - 5;
                match s % 13 {
                    0 => 0.0,
                    1..=5 => -u * 10f64.powi(e),
                    _ => u * 10f64.powi(e),
                }
            })
            .collect()
    }

    /// An unbounded backward sketch reads within the sketch's accuracy of
    /// the exact decayed quantile at every level, after every row, on
    /// irregular steps across many folds.
    #[test]
    fn a_running_sketch_is_within_its_accuracy_of_the_exact_quantile() {
        let k = kernel(Direction::Backward, 3.0, None);
        let xs = values(3000, 1);
        let mut q = QSide::default();
        let mut taus = Vec::new();
        let mut t = 0.0;
        for (i, &x) in xs.iter().enumerate() {
            t += [0.0, 0.5, 1.0, 2.5][i % 4];
            taus.push(t);
            q.push(&k, t, x);
            if i % 37 != 0 {
                continue;
            }
            for p in [0.01, 0.25, 0.5, 0.75, 0.99] {
                let vw: Vec<(f64, f64)> = xs[..=i]
                    .iter()
                    .zip(&taus)
                    .map(|(&v, &s)| (v, (-(t - s) / 3.0).exp2()))
                    .filter(|&(_, w)| w > 1e-11)
                    .collect();
                let want = exact(vw, p);
                let got = q.read(&k, p, None).unwrap();
                assert!(within(got, want), "row {i} p {p}: {got} against {want}");
            }
        }
        // Folded many times: 7,500 units at a half-life of 3.
        assert!(q.back.anchor > 2000.0 && q.back.scale > RENORM);
    }

    /// A row leaving `front` puts back its bucket's, octave's and the
    /// total's weight to the bit: the sketch after any number of rows leave
    /// is the one built from the rows that are left.
    #[test]
    fn a_row_leaving_front_leaves_the_sketch_of_the_rest_to_the_bit() {
        let k = kernel(Direction::Backward, 5.0, Some(100.0));
        let xs = values(200, 2);
        let tau: Vec<f64> = (0..xs.len()).map(|i| i as f64 * 0.5).collect();
        let mut a = QSide::default();
        a.turn_over(&k, &tau, |i| if i % 9 == 4 { f64::NAN } else { xs[i] });
        for top in (0..xs.len()).rev() {
            // `front` holds rows 0..=top in reverse: front index `f` is row
            // `len - 1 - f`, the oldest on top.
            let rest = xs.len() - top;
            let mut b = QSide::default();
            b.turn_over(&k, &tau[rest - 1..], |i| {
                let r = rest - 1 + i;
                if r % 9 == 4 { f64::NAN } else { xs[r] }
            });
            assert_eq!(a.front.total.to_bits(), b.front.total.to_bits(), "{top}");
            for p in [0.1, 0.5, 0.9] {
                assert_eq!(a.read(&k, p, None), b.read(&k, p, None), "{top} {p}");
            }
            let r = xs.len() - 1 - top;
            a.pop(top, if r % 9 == 4 { f64::NAN } else { xs[r] });
        }
        assert!(a.prev.is_empty() && !a.front.has_weight());
    }

    /// One value at a time: the representative of its bucket, its sign kept;
    /// a subnormal magnitude reads 0.
    #[test]
    fn one_value_reads_its_buckets_representative() {
        let k = kernel(Direction::Backward, 2.0, None);
        for x in [1.0, -1.0, 3.7e-200, -6.1e250, 1e-310, -0.0, 0.0] {
            let mut q = QSide::default();
            q.push(&k, 0.0, x);
            let got = q.read(&k, 0.5, None).unwrap();
            assert!(
                within(got, if x.abs() < f64::MIN_POSITIVE { 0.0 } else { x }),
                "{x}: {got}"
            );
            assert_eq!(
                got.is_sign_negative() && got != 0.0,
                x < 0.0 && x.abs() >= f64::MIN_POSITIVE
            );
        }
    }

    /// An outlier far past the rest is dropped when the sketch folds, once
    /// its weight is negligible: the buckets go back to the rest's span.
    #[test]
    fn an_unbounded_sketch_drops_an_outlier_once_it_is_negligible() {
        let k = kernel(Direction::Backward, 1.0, None);
        let mut q = QSide::default();
        q.push(&k, 0.0, 1e200);
        for i in 1..200 {
            q.push(&k, f64::from(i), 1.0 + f64::from(i % 7) / 10.0);
        }
        assert!(q.buckets() < 200, "{} buckets", q.buckets());
    }

    #[test]
    fn a_damaged_sketch_has_no_shape() {
        let k = kernel(Direction::Backward, 2.0, None);
        let mut q = QSide::default();
        for (i, x) in values(50, 3).into_iter().enumerate() {
            q.push(&k, i as f64, x);
        }
        assert!(q.has_shape(0));
        assert!(!q.has_shape(1));
        let mut bad = q.clone();
        bad.back.pos.lo += 1;
        assert!(!bad.has_shape(0));
        let mut bad = q.clone();
        bad.back.neg.oct.pop();
        assert!(!bad.has_shape(0));
        let mut bad = q.clone();
        bad.back.scale = 0.0;
        assert!(!bad.has_shape(0));
        let mut bad = q;
        bad.back.total = f64::NAN;
        assert!(!bad.has_shape(0));
    }
}
