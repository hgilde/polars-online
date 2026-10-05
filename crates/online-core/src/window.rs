//! A hard cutoff on an exponentially weighted accumulator (docs/PLAN.md §13).
//!
//! An EW mean never forgets: a half-life of `h` still leaves 12.5% of the
//! weight on data older than `3h`. Some questions need the other thing —
//! *nothing* from before a point, as a guarantee. The identity that makes it
//! cheap is that an EW sum contains its own past:
//!
//! ```text
//! A(t) = lam^(t-u) * A(u)  +  (everything after u)
//! ```
//!
//! so the part to discard is the accumulator's own earlier value, decayed
//! forward, and truncation is a subtraction rather than a recomputation.
//! What a model has to keep is therefore not the rows but a ring of past
//! *snapshots*, one per row (or one per `every` rows), which is what
//! this module holds.
//!
//! **The boundary is chosen conservatively, and the direction matters.** A
//! snapshot taken before the row at `t_j` carries everything strictly older
//! than `t_j`; subtracting it retains rows at `t_j` and after. Honouring
//! "nothing older than `window`" therefore needs `t_j >= t - window`, so the
//! boundary is the *oldest snapshot still inside the window* and a coarse
//! `every` discards slightly more than asked, never less. Rounding the other
//! way would keep rows the window promised to exclude, which is the one
//! failure this design refuses to have.
//!
//! That needs a snapshot inside the window. With a coarse `every` there was
//! none after a clock gap longer than the window, or wherever `every` rows
//! span more clock than it: the newest snapshot was older than the window,
//! it stayed the boundary, and rows the window excludes stayed in the fit.
//! So a row that finds the newest snapshot outside the window is
//! snapshotted whatever the cadence (review 2026-09-12, S6).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::EwCov;

/// What a window does when its snapshots pass a memory budget, and the
/// budget in MiB (review 2026-09-12, P4; the user's decision of
/// 2026-09-15). Past it the ring either thins -- every other snapshot
/// dropped and the spacing doubled, as often as it takes, so the boundary
/// grows coarser and never keeps an older row -- or stops: the snapshot that
/// would cross is not kept, none is made after it, and the overrun is
/// recorded for the caller to refuse the run on, naming the size and
/// `window_every`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowBudget {
    Thin(f64),
    Refuse(f64),
}

impl WindowBudget {
    /// Refuse past 256 MiB: the figure `gram_block_rows` refuses at, and
    /// what a spec that names no budget gets.
    pub const DEFAULT: WindowBudget = WindowBudget::Refuse(256.0);

    /// The budget in MiB.
    pub fn mib(&self) -> f64 {
        match *self {
            WindowBudget::Thin(m) | WindowBudget::Refuse(m) => m,
        }
    }

    fn bytes(&self) -> f64 {
        self.mib() * 1024.0 * 1024.0
    }
}

/// The heap a snapshot holds, in bytes: what a window's budget counts.
pub trait Footprint {
    fn footprint(&self) -> usize;
}

/// A snapshot reduced to its bytes: what a ring's shadow holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bytes(pub usize);

impl Footprint for Bytes {
    fn footprint(&self) -> usize {
        self.0
    }
}

/// A window's ring without its snapshots: each one's bytes where the ring
/// holds the snapshot, the model's clock, and the bytes the model's next
/// snapshot would hold. [`Self::learn`] is what a learned row does to the
/// ring -- the ring's own [`Snapshots::offer`] and [`Snapshots::trim`], run on
/// the shadow -- so a caller can find, before it runs a chunk's rows, whether
/// they would take the ring past a refusing budget, and refuse the chunk
/// with nothing learned (docs/PLAN.md task 115 (d)). A snapshot is taken at
/// the size the model's would be now; a model whose snapshot grows within
/// the chunk is seen short, and the ring's own refusal still stops it.
#[derive(Debug, Clone)]
pub struct WindowShadow {
    clock: f64,
    ring: Snapshots<Bytes>,
    snapshot: usize,
}

impl WindowShadow {
    /// The shadow of `ring`, whose model's clock is `clock`. Its snapshots
    /// are taken at the newest one's size, or, in an empty ring, at the size
    /// of the one `make` forms: the closure the model hands `offer`.
    pub fn new<S: Footprint>(clock: f64, ring: &Snapshots<S>, make: impl FnOnce() -> S) -> Self {
        let snapshot = match ring.ring.back() {
            Some((_, s)) => s.footprint(),
            None => make().footprint(),
        };
        Self {
            clock,
            ring: Snapshots {
                window: ring.window,
                every: ring.every,
                since: ring.since,
                ring: ring
                    .ring
                    .iter()
                    .map(|(t, s)| (*t, Bytes(s.footprint())))
                    .collect(),
                limit: ring.limit.clone(),
            },
            snapshot,
        }
    }

    /// A learned row `d_clock` after the last: the ring's offer and trim, as
    /// the model's step makes them.
    pub fn learn(&mut self, d_clock: f64) {
        let t = self.clock + d_clock;
        let snapshot = self.snapshot;
        self.ring.offer(t, || Bytes(snapshot));
        self.clock = t;
        self.ring.trim(t);
    }

    /// What the ring would report: [`Snapshots::over_budget`].
    pub fn over_budget(&self) -> Option<(usize, usize)> {
        self.ring.over_budget()
    }

    /// Whether `rows` learned rows could take the ring past a refusing
    /// budget at all: a row adds at most one snapshot, so a ring that fits
    /// that many more needs no replay.
    pub fn could_refuse(&self, rows: usize) -> bool {
        match self.ring.limit.budget {
            Some(b @ WindowBudget::Refuse(_)) => {
                self.ring.limit.over.is_some()
                    || (self.ring.limit.held + rows.saturating_mul(self.snapshot)) as f64
                        > b.bytes()
            }
            _ => false,
        }
    }
}

/// Bytes in a slice of floats, for a [`Footprint`].
pub(crate) fn floats(v: &[f64]) -> usize {
    std::mem::size_of_val(v)
}

/// Every value under a key named `key`, anywhere in a state's JSON, handed
/// to `edit`: for a test that plays a state written by another build.
#[cfg(test)]
pub(crate) fn json_edit(
    v: &mut serde_json::Value,
    key: &str,
    edit: &mut dyn FnMut(&mut serde_json::Value),
) {
    match v {
        serde_json::Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                if k == key {
                    edit(x);
                } else {
                    json_edit(x, key, edit);
                }
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(|x| json_edit(x, key, edit)),
        _ => {}
    }
}

/// Holds a snapshot type's footprint to what it holds, from two snapshots
/// of it, `small` and `large`, of the same shape with every vector longer
/// in `large`. The footprint must count at least the vectors, and at most
/// the vectors and the numbers held outside any; and what it counts beyond
/// the vectors must be the same in both, so a vector it leaves out shows as
/// a difference that grows with the vectors, however many scalars it
/// happens to count. The two are walked in step, and a vector that does not
/// grow is refused: from one snapshot, or from two that differ only in some
/// vectors, a left-out vector can hide behind the scalars (docs/PLAN.md task
/// 130; review 2026-09-26, B4).
#[cfg(test)]
pub(crate) fn assert_footprint_counts_every_vector<S: Footprint + Serialize>(
    small: &S,
    large: &S,
    what: &str,
) {
    use serde_json::Value;
    fn numeric(a: &[Value]) -> bool {
        a.iter().all(|x| x.is_number() || x.is_null())
    }
    /// The vector bytes of a part the smaller snapshot lacks, which must
    /// hold vectors and nothing else.
    fn vectors_only(v: &Value, at: &str, vl: &mut usize) {
        match v {
            Value::Array(a) if numeric(a) => *vl += a.len() * std::mem::size_of::<f64>(),
            Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    vectors_only(x, &format!("{at}[{i}]"), vl);
                }
            }
            Value::Object(o) => {
                for (k, x) in o {
                    vectors_only(x, &format!("{at}.{k}"), vl);
                }
            }
            _ => panic!("{at}: a number outside any vector in a part the smaller lacks"),
        }
    }
    /// The vector bytes of each and the scalars, walking both in step.
    fn walk(s: &Value, l: &Value, at: &str, vs: &mut usize, vl: &mut usize, scalars: &mut usize) {
        match (s, l) {
            (Value::Array(a), Value::Array(b)) if numeric(a) && numeric(b) => {
                assert!(
                    b.len() > a.len() || (a.is_empty() && b.is_empty()),
                    "{at}: a vector of {} does not grow ({} in the larger)",
                    a.len(),
                    b.len()
                );
                *vs += a.len() * std::mem::size_of::<f64>();
                *vl += b.len() * std::mem::size_of::<f64>();
            }
            (Value::Array(a), Value::Array(b)) => {
                // A list of parts may gain parts in the larger (one vector
                // per target, say); the parts both have are walked in step,
                // and the extra ones must be vectors alone, since a number
                // among them would move what the footprint counts beyond
                // the vectors.
                assert!(b.len() >= a.len(), "{at}: fewer parts in the larger");
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    walk(x, y, &format!("{at}[{i}]"), vs, vl, scalars);
                }
                for (i, y) in b.iter().enumerate().skip(a.len()) {
                    vectors_only(y, &format!("{at}[{i}]"), vl);
                }
            }
            (Value::Object(o), Value::Object(q)) => {
                let (ko, kq): (Vec<_>, Vec<_>) = (o.keys().collect(), q.keys().collect());
                assert_eq!(ko, kq, "{at}: the same fields in both");
                for (k, x) in o {
                    walk(x, &q[k], &format!("{at}.{k}"), vs, vl, scalars);
                }
            }
            (Value::Number(_) | Value::Null, Value::Number(_) | Value::Null) => *scalars += 1,
            (Value::Bool(_), Value::Bool(_)) | (Value::String(_), Value::String(_)) => {}
            _ => panic!("{at}: the two snapshots differ in shape"),
        }
    }
    let (mut vs, mut vl, mut scalars) = (0, 0, 0);
    walk(
        &serde_json::to_value(small).unwrap(),
        &serde_json::to_value(large).unwrap(),
        what,
        &mut vs,
        &mut vl,
        &mut scalars,
    );
    assert!(
        vl > vs && vs > 0,
        "{what}: the vectors must grow ({vs} to {vl} bytes)"
    );
    let (fs, fl) = (small.footprint(), large.footprint());
    for (f, v) in [(fs, vs), (fl, vl)] {
        assert!(
            f >= v && f <= v + scalars * std::mem::size_of::<f64>(),
            "{what}: the footprint counts {f} bytes; the vectors hold {v}, and {scalars} numbers \
             besides"
        );
    }
    assert_eq!(
        fs as i64 - vs as i64,
        fl as i64 - vl as i64,
        "{what}: what the footprint counts beyond the vectors moved with them: a vector is \
         left out"
    );
}

/// Snapshots of an accumulator, oldest first, spanning at most one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshots<S> {
    /// Clock units of history the window keeps.
    window: f64,
    /// Snapshot every `every` rows; `1` snapshots them all and makes the
    /// boundary as tight as the data allows. A row whose newest snapshot has
    /// left the window is snapshotted whatever the count.
    every: usize,
    /// Rows since the last snapshot, so the cadence is counted in the
    /// *stream* and never in the chunk (hard rule 3).
    since: usize,
    /// `(clock of the row the snapshot precedes, snapshot)`.
    ring: VecDeque<(f64, S)>,
    /// The budget, and a refusing budget's overrun: configuration the caller
    /// sets after building or restoring the model, so not part of the
    /// state, and two rings that differ only here are equal.
    #[serde(skip)]
    limit: Limit,
}

#[derive(Debug, Clone, Default)]
struct Limit {
    budget: Option<WindowBudget>,
    over: Option<usize>,
    /// The bytes the ring holds, kept as snapshots come and go so a push
    /// costs no walk of the ring; counted afresh whenever a budget is set.
    held: usize,
}

impl PartialEq for Limit {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<S> Snapshots<S> {
    /// `window` in clock units, `every` rows between snapshots.
    pub fn new(window: f64, every: usize) -> Result<Self, String> {
        if !window.is_finite() || window <= 0.0 {
            return Err(format!("window_size must be > 0 (got {window})"));
        }
        if every == 0 {
            return Err("window_every must be >= 1".into());
        }
        Ok(Self {
            window,
            every,
            since: usize::MAX, // the first row always snapshots
            ring: VecDeque::new(),
            limit: Limit::default(),
        })
    }

    pub fn window(&self) -> f64 {
        self.window
    }

    /// Every snapshot held, oldest first, for a caller that must rewrite them
    /// in place: a state converted as it is read.
    /// The snapshots, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &S> {
        self.ring.iter().map(|(_, s)| s)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.ring.iter_mut().map(|(_, s)| s)
    }

    /// The snapshot to subtract, and the clock it is referenced at.
    pub fn boundary(&self) -> Option<&(f64, S)> {
        self.ring.front()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

impl<S: Footprint> Snapshots<S> {
    /// Drop what can never be the boundary again: everything strictly older
    /// than `now - window`. What is left at the front is the oldest snapshot
    /// inside the window, which is the boundary; [`Self::offer`] at `now`
    /// has seen to it that there is one. (A stale front did not "subtract
    /// the whole accumulator", as the comment here said: it subtracts the
    /// state before the row it precedes, which keeps that row -- review
    /// 2026-09-12, S6.)
    pub fn trim(&mut self, now: f64) {
        let oldest = now - self.window;
        while self.ring.len() > 1 && self.ring[0].0 < oldest {
            if let Some((_, s)) = self.ring.pop_front() {
                self.limit.held = self.limit.held.saturating_sub(s.footprint());
            }
        }
    }

    /// Whether [`Self::offer`] at `clock` would form a snapshot: for a
    /// caller that holds work back and must finish it before the snapshot
    /// copies the state (`Marginal::step_sharded`). A refusing budget may
    /// then keep none of what it formed, and the work finished early was
    /// harmless.
    pub fn takes(&self, clock: f64) -> bool {
        if self.limit.over.is_some() {
            return false;
        }
        let stale = self
            .ring
            .back()
            .is_none_or(|&(t, _)| t < clock - self.window);
        self.since.saturating_add(1) >= self.every || stale
    }

    /// Offer a snapshot of the state *as it stands before* the row at
    /// `clock`, already decayed to that row. Taken when the cadence is due,
    /// and whatever the cadence when the newest snapshot is older than the
    /// window, so the boundary is always inside it (the module docs); `make`
    /// is not called otherwise. Past a thinning budget the ring then thins;
    /// a refusing one keeps no snapshot that would cross it, and makes none
    /// after ([`WindowBudget`]).
    pub fn offer(&mut self, clock: f64, make: impl FnOnce() -> S) {
        self.since = self.since.saturating_add(1);
        if self.limit.over.is_some() {
            return;
        }
        let stale = self
            .ring
            .back()
            .is_none_or(|&(t, _)| t < clock - self.window);
        if self.since >= self.every || stale {
            self.since = 0;
            let snap = make();
            let held = self.limit.held + snap.footprint();
            // The ring stops short of a refusing budget: the caller refuses
            // the run, and until it looks, the memory stays bounded.
            let refuse = matches!(
                self.limit.budget,
                Some(b @ WindowBudget::Refuse(_)) if held as f64 > b.bytes()
            );
            if refuse {
                self.limit.over = Some(held);
                return;
            }
            self.limit.held = held;
            self.ring.push_back((clock, snap));
            self.enforce();
        }
    }

    /// Bound the ring ([`WindowBudget`]), `None` for no bound. A ring already
    /// past the new budget thins, or records its overrun, at once.
    pub fn set_budget(&mut self, budget: Option<WindowBudget>) {
        self.limit = Limit {
            budget,
            over: None,
            held: self.bytes(),
        };
        self.enforce();
    }

    /// A refusing budget's overrun: the bytes the ring reached, and the
    /// spacing it reached them at (`window_every`, doubled by any thinning).
    pub fn over_budget(&self) -> Option<(usize, usize)> {
        self.limit.over.map(|bytes| (bytes, self.every))
    }

    /// The bytes the ring's snapshots hold.
    pub fn bytes(&self) -> usize {
        self.ring.iter().map(|(_, s)| s.footprint()).sum()
    }

    fn enforce(&mut self) {
        let Some(budget) = self.limit.budget else {
            return;
        };
        let limit = budget.bytes();
        if self.limit.held as f64 <= limit {
            return;
        }
        match budget {
            // Only a ring given a budget it is already past gets here: a
            // snapshot that would cross one is turned away in `offer`.
            WindowBudget::Refuse(_) => self.limit.over = Some(self.limit.held),
            WindowBudget::Thin(_) => {
                // Keep the newest and every second one before it, and take
                // them half as often, until the ring fits. The front snapshot
                // is the boundary, so losing it moves the boundary later: the
                // window drops more rows, and never keeps an older one.
                while self.limit.held as f64 > limit && self.ring.len() > 1 {
                    let n = self.ring.len();
                    self.ring = std::mem::take(&mut self.ring)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| (n - 1 - i) % 2 == 0)
                        .map(|(_, e)| e)
                        .collect();
                    self.every = self.every.saturating_mul(2);
                    self.limit.held = self.bytes();
                }
            }
        }
    }
}

/// The smallest remainder, relative to the weight it was subtracted from, that
/// is still something. The window's weight is `W - f·W_u`, a difference of two
/// positives that are equal in exact arithmetic when every row of the history
/// has aged out -- and in `f64` then comes out at about `±W·1e-16`, not 0. A
/// positive crumb that size would pass as "a window with rows in it" and turn
/// every mean divided by it into noise, so anything below this fraction is an
/// empty window (review 2026-09-12, C2: "clamp the truncated weight at 0
/// before the test"). A genuine window never gets near it: its newest row
/// alone carries weight 1 against a history of at most `1/(1 - lam^d)` for
/// rows `d` apart, about `1.44 · half_life / d`, which reaches `1e12` only
/// with a trillion unit rows to a half-life.
pub const EMPTY_FRACTION: f64 = 1e-12;

/// An [`EwCov`]'s data, decayed to the clock of the row it precedes. Means and
/// centred co-moments do not move under decay; only the two weight sums do,
/// which is why a snapshot is this and not a whole accumulator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub w: f64,
    pub m: Vec<f64>,
    pub c: Vec<f64>,
    /// The accumulator's count of learned rows when the snapshot was taken:
    /// the first learned row inside the window is the next one, which a
    /// held slot's run must have started at or before (`crate::Runs`).
    /// `None` in a snapshot written before it, which then holds nothing.
    #[serde(default)]
    pub rows: Option<u64>,
    /// The Kish sum, where the accumulator keeps one. Last, and the only
    /// skipped field: the compact encoding is positional, and a snapshot
    /// without it decoded `m` into this slot (review 2026-09-18, B7; no
    /// model writes one today, `EwCov::new` starts `q_sum` at `Some(0)`,
    /// so this is the rule kept rather than a failure seen; the byte change
    /// rides on `SCHEMA_VERSION` 11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q: Option<f64>,
}

impl Moments {
    /// The accumulator as it stands, decayed by `lam` -- what the snapshot
    /// before the row at `t` must hold for `t` to be retained when it is
    /// subtracted.
    pub fn of(cov: &EwCov, lam: f64) -> Self {
        Self {
            w: cov.n_eff() * lam,
            q: cov.q_sum().map(|q| q * lam * lam),
            m: cov.means().to_vec(),
            c: cov.comoments().to_vec(),
            rows: Some(cov.rows_learned()),
        }
    }
}

impl Footprint for Moments {
    fn footprint(&self) -> usize {
        // The weight, the row count and the second weight sum, and the two
        // vectors.
        3 * std::mem::size_of::<f64>() + floats(&self.m) + floats(&self.c)
    }
}

/// `cov` with everything before the row the snapshot precedes removed:
/// `A(t) - f·A(u)`, in the mean form the accumulator stores. That row, at
/// the snapshot's clock `u`, and every row after it are kept (the module
/// docs say why the boundary is drawn there).
///
/// `None` when the window holds no weight -- a clock gap longer than it, or a
/// boundary that is the whole accumulator. The caller reports nothing rather
/// than dividing by zero (hard rule 9).
///
/// **Centred, never through raw moments.** With `W` the live weight, `W_u` the
/// snapshot's weight decayed forward and `W_R = W - W_u` what the window
/// keeps, the pooling identity for two sets of weighted moments,
///
/// ```text
/// W·C = W_R·C_R + W_u·C_u + (W_u·W / W_R)·(m_u - m)(m_u - m)ᵀ
/// ```
///
/// solved for the remainder gives `C_R`, and `m_R = m - (W_u/W_R)·(m_u - m)`.
/// Every term is a centred moment or a difference of two means, so nothing is
/// ever the size of `m²`: the window keeps its precision at any offset. The
/// earlier form went back through `E[x x'] = C + m m'`, subtracted, and
/// re-centred, which at a level of `1e8` and unit variance left the windowed
/// variance with a resolution of about 2 -- nothing (review 2026-09-12, C17;
/// `tests/test_second_opinion.py` holds it to `numpy.cov` at that offset).
///
/// What precision remains to lose is the fraction discarded: the result is a
/// difference of positives of size `C`, so it is negligible at `window =
/// 3·half-life`, where the correction is an eighth, and worse as the window
/// shortens toward the half-life.
pub fn truncated(cov: &EwCov, old: &Moments, f: f64) -> Option<EwCov> {
    let k = cov.k();
    let w_now = cov.n_eff();
    let w_old = f * old.w;
    let w = w_now - w_old;
    if w <= EMPTY_FRACTION * w_now || !w.is_finite() {
        return None;
    }
    // `ratio = W_u / W_R` and `g = W / W_R`, so `C_R = g·C - ratio·C_u -
    // ratio·g·d dᵀ` with `d = m_u - m`.
    let (ratio, g) = (w_old / w, w_now / w);
    let d: Vec<f64> = (0..k).map(|i| old.m[i] - cov.mean(i)).collect();
    let mut mean: Vec<f64> = (0..k).map(|i| cov.mean(i) - ratio * d[i]).collect();
    let c_now = cov.comoments();
    let mut cen = vec![0.0; k * k];
    let mut unresolved = vec![false; k];
    for i in 0..k {
        for j in 0..k {
            let ij = i * k + j;
            cen[ij] = g * c_now[ij] - ratio * old.c[ij] - ratio * g * d[i] * d[j];
        }
        // What is left to lose is a difference of positives; a variance it
        // takes a hair below zero is zero (review V3).
        let ii = i * k + i;
        cen[ii] = cen[ii].max(0.0);
        // And a variance no larger than the rounding of the terms it
        // is formed from carries no digit of the window's spread: the live
        // accumulator itself resolves the window's rows only to `ε` of its
        // co-moment, so when rows far outside the window dominate it -- a
        // feature at `1e100` before the window, a row of weight `1e100`
        // inside it -- the difference is that rounding, `1e84` against a
        // spread of `1e-99` (review 2026-09-27, G5). Read as no spread, as a
        // held feature is below. On a window over ordinary rows the variance
        // is many orders above this. The third term, `ratio·g·d²`, needs no
        // place of its own: the variance kept is `g·C − ratio·C_u −
        // ratio·g·d² ≥ 0`, so that term is at most `g·C`, and leaving it out
        // moves the bound by a factor of two at most, inside the 64 (and a
        // bound no test could hold that term to, the mutants of 2026-09-27
        // showed).
        let terms = g * c_now[ii] + ratio * old.c[ii];
        unresolved[i] = cen[ii] <= 64.0 * f64::EPSILON * terms;
    }
    for i in (0..k).filter(|&i| unresolved[i]) {
        for j in 0..k {
            cen[i * k + j] = 0.0;
            cen[j * k + i] = 0.0;
        }
    }
    // A feature that held one value over every row inside the window has no
    // spread there, and the subtraction cannot say so: it leaves a remainder
    // that grows with the level and with the rows since the boundary, up to
    // half the terms that cancelled (docs/PLAN.md task 94). Its run can,
    // when it started at or before the first learned row inside the window,
    // the one after the snapshot's count. (The run's decayed weight against
    // the window's, equal in exact arithmetic, drifted past a tolerance of
    // 1e-12 under a long window and a long half-life: review 2026-09-25.)
    if let Some(rows) = old.rows {
        for i in 0..k {
            if let Some(value) = cov.held_from(i, rows + 1) {
                mean[i] = value;
                for j in 0..k {
                    cen[i * k + j] = 0.0;
                    cen[j * k + i] = 0.0;
                }
            }
        }
    }
    let q = match (cov.q_sum(), old.q) {
        (Some(q_now), Some(q_then)) => Some((q_now - f * f * q_then).max(0.0)),
        _ => None,
    };
    let mut out = cov.clone();
    out.set_moments(&mean, &cen, w, q);
    Some(out)
}

/// One mean-form accumulator's worth of subtraction: `(w, mean)` truncated
/// against a snapshot, for the per-target sums that sit beside a Gram.
/// `None` when nothing is left inside the window.
pub fn truncated_mean(
    w_now: f64,
    mean: &[f64],
    w_old: f64,
    mean_old: &[f64],
    f: f64,
) -> Option<(f64, Vec<f64>)> {
    let w = w_now - f * w_old;
    if w <= EMPTY_FRACTION * w_now || !w.is_finite() {
        return None;
    }
    let out = mean
        .iter()
        .zip(mean_old)
        .map(|(m, m0)| (w_now * m - f * w_old * m0) / w)
        .collect();
    Some((w, out))
}

/// [`truncated_mean`] for one scalar accumulator, with nothing allocated:
/// what a per-slot spread reads on every row (review 2026-09-12, S1).
pub fn truncated_scalar(
    w_now: f64,
    mean: f64,
    w_old: f64,
    mean_old: f64,
    f: f64,
) -> Option<(f64, f64)> {
    let w = w_now - f * w_old;
    if w <= EMPTY_FRACTION * w_now || !w.is_finite() {
        return None;
    }
    Some((w, (w_now * mean - f * w_old * mean_old) / w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decay;

    /// Eight bytes a snapshot, for rings of counters.
    impl Footprint for usize {
        fn footprint(&self) -> usize {
            8
        }
    }

    /// The scalar truncation is the slice one's arithmetic, to the bit, and
    /// empties where it does.
    #[test]
    fn the_scalar_truncation_is_the_slice_one() {
        for (w_now, m, w_old, m_old, f) in [
            (10.0, 2.5, 4.0, 1.5, 0.9),
            (1.0, 0.1, 0.999_999_999_999_9, 0.1, 1.0),
            (5.0, 3.0, 0.0, 0.0, 0.5),
            (3.0, 1e8 + 0.25, 2.0, 1e8, 0.75),
        ] {
            let slice = truncated_mean(w_now, &[m], w_old, &[m_old], f).map(|(w, v)| (w, v[0]));
            assert_eq!(truncated_scalar(w_now, m, w_old, m_old, f), slice);
        }
    }

    /// Past a thinning budget the ring keeps every other snapshot and
    /// doubles its spacing until it fits, and its boundary stays inside the
    /// window: it drops rows, never keeps an older one (review 2026-09-12,
    /// P4).
    #[test]
    fn past_a_thinning_budget_the_ring_fits_and_its_boundary_stays_inside() {
        let window = 100.0;
        let mut snaps = Snapshots::new(window, 1).unwrap();
        // Ten snapshots' worth.
        snaps.set_budget(Some(WindowBudget::Thin(80.0 / (1024.0 * 1024.0))));
        for i in 0..1000u32 {
            let t = f64::from(i);
            snaps.offer(t, || i as usize);
            snaps.trim(t);
            assert!(snaps.bytes() <= 80, "row {i}: {} bytes", snaps.bytes());
            assert_eq!(snaps.limit.held, snaps.bytes(), "row {i}: the count");
            let &(b, _) = snaps.boundary().unwrap();
            assert!(b >= t - window, "row {i}: bounded at {b}");
        }
        assert!(snaps.every > 1, "and it takes its snapshots less often");
        assert_eq!(
            snaps.over_budget(),
            None,
            "a thinning budget refuses nothing"
        );
    }

    /// Past a refusing budget the ring records the overrun, with its
    /// spacing, for the caller to refuse the run on -- and stops there: the
    /// snapshot that crossed is not kept and none is made after it, so the
    /// memory the refusal bounds stays bounded until the caller checks.
    #[test]
    fn past_a_refusing_budget_the_ring_records_the_overrun() {
        let mut snaps = Snapshots::new(100.0, 1).unwrap();
        snaps.set_budget(Some(WindowBudget::Refuse(80.0 / (1024.0 * 1024.0))));
        for i in 0..10u32 {
            snaps.offer(f64::from(i), || i as usize);
            assert_eq!(snaps.over_budget(), None, "row {i}: ten fit");
        }
        snaps.offer(10.0, || 10);
        assert_eq!(snaps.over_budget(), Some((88, 1)));
        assert_eq!(snaps.bytes(), 80, "the snapshot that crossed is not kept");
        let mut made = 0;
        for i in 11..1000u32 {
            snaps.offer(f64::from(i), || {
                made += 1;
                i as usize
            });
            snaps.trim(f64::from(i));
            assert!(snaps.bytes() <= 80, "row {i}: {} bytes", snaps.bytes());
            assert_eq!(snaps.limit.held, snaps.bytes(), "row {i}: the count");
        }
        assert_eq!(made, 0, "no snapshot is made past a refusal");
        assert_eq!(snaps.over_budget(), Some((88, 1)), "the first overrun");
    }

    /// The module doc's boundary, which `trim`'s comment had the other way
    /// round (review 2026-09-12, S6): a snapshot is the state before the
    /// row it precedes, so subtracting it keeps that row. Across a gap of
    /// twice the window the one snapshot left is the one taken before the
    /// row after the gap, and what the subtraction leaves is that row.
    #[test]
    fn the_boundary_keeps_the_row_it_precedes() {
        let (window, lam) = (5.0, 0.9f64);
        let mut cov = EwCov::new(1);
        let mut snaps = Snapshots::new(window, 1).unwrap();
        let mut t = 0.0;
        for (i, d) in [0.0, 1.0, 1.0, 1.0, 2.0 * window].into_iter().enumerate() {
            t += d;
            let l = lam.powf(d);
            snaps.offer(t, || Moments::of(&cov, l));
            snaps.trim(t);
            cov.update(&[i as f64], l, 1.0);
        }
        assert_eq!(
            snaps.len(),
            1,
            "one snapshot is left after a gap of twice the window"
        );
        let (clock, old) = snaps.boundary().unwrap();
        assert_eq!(*clock, t);
        let kept = truncated(&cov, old, 1.0).expect("the row after the gap is in the window");
        assert!(
            (kept.n_eff() - 1.0).abs() < 1e-12,
            "exactly that row's weight: {}",
            kept.n_eff()
        );
        assert!(
            (kept.mean(0) - 4.0).abs() < 1e-12,
            "and its value: {}",
            kept.mean(0)
        );
    }

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// One row of [`windowed`]'s stream: what the accumulator is given.
    #[derive(Clone, Copy, PartialEq)]
    enum Row {
        /// Learned at this weight.
        Learn(f64),
        /// Not learned: aged over, as a Gram ages over a row none of its
        /// targets has (`EwCov::skip`).
        Skip,
        /// Not learned: aged by `EwCov::decay`.
        Decay,
    }

    /// PLAN task 94's stream, one row per clock unit, snapshotted every row:
    /// features 0 and 2 move on every row, feature 1 moves until row `from`
    /// and holds `c` from there on, except where `jumps` gives it another
    /// value.
    /// The accumulator truncated to the last `window` clock units, `c`, and
    /// feature 1's values and weights on the rows inside the window.
    #[allow(clippy::too_many_arguments)]
    fn windowed(
        n: usize,
        window: f64,
        h: f64,
        level: f64,
        scale: f64,
        from: usize,
        rows: &dyn Fn(usize) -> Row,
        jumps: &[(usize, f64)],
        seed: u64,
    ) -> (EwCov, f64, Vec<(f64, f64)>) {
        let mut s = 0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(7919);
        let lam = if h.is_infinite() {
            1.0
        } else {
            (-(1.0 / h)).exp2()
        };
        let mut cov = EwCov::new(3);
        let mut snaps = Snapshots::new(window, 1).unwrap();
        let c = level + scale * lcg(&mut s);
        let mut inside = Vec::new();
        for i in 0..n {
            let t = i as f64;
            let l = if i == 0 { 1.0 } else { lam };
            snaps.offer(t, || Moments::of(&cov, l));
            snaps.trim(t);
            let moving = level + scale * lcg(&mut s);
            let mut x1 = if i >= from { c } else { moving };
            if let Some(&(_, v)) = jumps.iter().find(|&&(j, _)| j == i) {
                x1 = v;
            }
            let x = [scale * lcg(&mut s), x1, level - scale * lcg(&mut s)];
            let w = match rows(i) {
                Row::Learn(w) => {
                    cov.update(&x, l, w);
                    w
                }
                Row::Skip => {
                    cov.skip(&x, l);
                    0.0
                }
                Row::Decay => {
                    cov.decay(l);
                    0.0
                }
            };
            if t >= (n - 1) as f64 - window {
                inside.push((x1, w * lam.powi((n - 1 - i) as i32)));
            }
        }
        let t = (n - 1) as f64;
        let (u, old) = snaps.boundary().unwrap();
        let f = Decay::Halflife(h).factor(t - u);
        let out = truncated(&cov, old, f).expect("the window holds rows");
        (out, c, inside)
    }

    /// Feature rows fed through an accumulator and its snapshot ring, one row
    /// per clock unit, and the accumulator truncated to the last `window`
    /// units, beside the longhand variance of each feature over the rows
    /// inside, each weighted by its decay: what the guard in [`truncated`] is
    /// held to (review 2026-09-27, G5).
    fn truncated_beside_longhand<const K: usize>(
        rows: &[[f64; K]],
        window: f64,
        h: f64,
    ) -> (EwCov, [f64; K]) {
        let lam = (-(1.0 / h)).exp2();
        let mut cov = EwCov::new(K);
        let mut snaps = Snapshots::new(window, 1).unwrap();
        for (i, x) in rows.iter().enumerate() {
            let (t, l) = (i as f64, if i == 0 { 1.0 } else { lam });
            snaps.offer(t, || Moments::of(&cov, l));
            snaps.trim(t);
            cov.update(x, l, 1.0);
        }
        let now = (rows.len() - 1) as f64;
        let (u, old) = snaps.boundary().unwrap();
        let out = truncated(&cov, old, lam.powf(now - u)).expect("rows inside");
        let inside: Vec<(f64, [f64; K])> = rows
            .iter()
            .enumerate()
            .filter(|&(i, _)| now - i as f64 <= window)
            .map(|(i, x)| (lam.powf(now - i as f64), *x))
            .collect();
        let w: f64 = inside.iter().map(|(w, _)| w).sum();
        let mut var = [0.0; K];
        for (j, v) in var.iter_mut().enumerate() {
            let m = inside.iter().map(|(w, x)| w * x[j]).sum::<f64>() / w;
            *v = inside
                .iter()
                .map(|(w, x)| w * (x[j] - m).powi(2))
                .sum::<f64>()
                / w;
        }
        (out, var)
    }

    /// A window over ordinary rows at a tiny scale keeps its spread: the
    /// guard reads a variance as no spread only where it is below the
    /// rounding of the terms it is formed from, and a variance of 1e-18 over
    /// rows of 1e-9 is many orders above theirs (G5). A guard that added an
    /// O(1) term to that rounding bound read it as nothing.
    #[test]
    fn a_window_at_a_tiny_scale_keeps_its_spread() {
        let mut s = 31u64;
        let rows: Vec<[f64; 2]> = (0..120)
            .map(|_| [1e-9 * lcg(&mut s), 1e-9 * lcg(&mut s)])
            .collect();
        let (out, want) = truncated_beside_longhand(&rows, 30.0, 10.0);
        for (j, want) in want.iter().enumerate() {
            let got = out.var(j);
            assert!(got > 0.0, "feature {j}: read as no spread");
            assert!(
                (got - want).abs() <= 1e-8 * want,
                "feature {j}: {got:e} against {want:e}"
            );
        }
    }

    /// One feature the live state cannot resolve inside the window -- rows
    /// far past the window's own before it, ordinary ones inside -- beside
    /// two it can: the first reads as no spread, its covariance with each
    /// other feature is nothing in both halves of the matrix, and the others
    /// keep the spread they have (G5). At `1e100` and at `1e-40` against
    /// `1e-60`: the guard is a ratio, and holds at any scale.
    #[test]
    fn an_unresolved_feature_beside_resolved_ones() {
        for (far, near) in [(1e100, 1.0), (1e-40, 1e-60)] {
            let mut s = 37u64;
            let rows: Vec<[f64; 3]> = (0..120)
                .map(|i| {
                    let x2 = if i < 60 { far } else { near } * lcg(&mut s);
                    [near * lcg(&mut s), near * lcg(&mut s), x2]
                })
                .collect();
            let (out, want) = truncated_beside_longhand(&rows, 30.0, 10.0);
            assert_eq!(out.var(2), 0.0, "{far:e}: the unresolved feature");
            for (j, &w) in want.iter().enumerate().take(2) {
                assert_eq!(out.cov(j, 2), 0.0, "{far:e}: its column, row {j}");
                assert_eq!(out.cov(2, j), 0.0, "{far:e}: its row, column {j}");
                let got = out.var(j);
                assert!(
                    (got - w).abs() <= 1e-9 * w,
                    "{far:e}, feature {j}: {got:e} against {w:e}"
                );
            }
        }
    }

    /// A window the snapshot cannot be told apart from: the rows before it
    /// carry nearly all the spread the live state holds, so its co-moment
    /// and the snapshot's cancel to their rounding, and the window's own
    /// spread is below it. The difference of the two terms is not a bound
    /// on that rounding; their sum is (G5).
    #[test]
    fn a_window_cancelling_to_its_rounding_reads_as_no_spread() {
        let mut s = 41u64;
        // A long half-life keeps the old rows' share of the live state near
        // one, so `g·C` and `ratio·C_u` are nearly equal.
        let rows: Vec<[f64; 3]> = (0..400)
            .map(|i| {
                let x2 = if i < 370 { 1e60 } else { 1.0 } * lcg(&mut s);
                [lcg(&mut s), lcg(&mut s), x2]
            })
            .collect();
        let (out, _) = truncated_beside_longhand(&rows, 20.0, 5e3);
        assert_eq!(out.var(2), 0.0);
        assert!(out.var(0) > 0.0 && out.var(1) > 0.0);
    }

    /// Weights that move, as a bank's `weight` column does.
    fn weights(i: usize) -> Row {
        Row::Learn(0.5 + ((i * 7919) % 101) as f64 / 100.0)
    }

    /// PLAN task 94. A feature that holds one value over every row inside a
    /// window has no spread there, and the window's Gram says so exactly:
    /// zero variance, zero covariance with every other feature, and that
    /// value as its mean. It used to come back as whatever the subtraction
    /// left, measured up to 4.6e-1 of the terms that cancelled at a level of
    /// 1e8 and a spread of 1e-3, so the feature was kept and standardized by
    /// noise: a windowed lasso at a zero penalty predicted 2.8e55 where the
    /// fit is -1.0. No threshold separates that from a real spread, since the
    /// leftover grows with the level and with the rows since the boundary;
    /// the accumulator knows instead which features have held their value,
    /// and on what weight. The run starts at the boundary row or before it,
    /// across the levels, spreads, half-lives and window lengths measured,
    /// the last a window of a thousand half-lives, where the history's
    /// spread has decayed to its last ulp.
    /// The run's weight and the window's are equal in exact arithmetic;
    /// in doubles the window's is `w_now − f·old.w` with `f` one `exp2` over
    /// the window's clock and the run's a product of per-row factors, and
    /// the two drift apart by about the rows in the window times a rounding
    /// step, scaled by the history's weight. A long window under a long
    /// half-life -- a regime dummy held over a day of second bars under a
    /// month's half-life -- is where that drift is largest (review
    /// 2026-09-25, tasks 94-97, finding 1): fifty thousand rows of history,
    /// a window of as many, half-lives up to 1e7, unit and random weights.
    #[test]
    fn a_long_window_under_a_long_halflife_still_reads_a_held_feature() {
        for (h, window, seed) in [
            (1e6, 49_999.0, 0),
            (1e6, 49_999.0, 1),
            (3e5, 99_999.0, 2),
            (1e7, 199_999.0, 3),
        ] {
            let n = 50_000 + window as usize + 1;
            let from = n - 1 - window as usize;
            let ones = |_: usize| Row::Learn(1.0);
            for rows in [&ones as &dyn Fn(usize) -> Row, &weights] {
                let (w, c, _) = windowed(n, window, h, 0.0, 1.0, from, rows, &[], seed);
                let case = format!("h {h}, window {window}, seed {seed}");
                assert_eq!(w.var(1), 0.0, "{case}: variance");
                assert_eq!(w.cov(0, 1), 0.0, "{case}: covariance");
                assert_eq!(w.mean(1), c, "{case}: the value held is the mean");
            }
        }
    }

    #[test]
    fn a_feature_constant_inside_the_window_has_no_spread_there() {
        let shapes = [
            (0.0, 1.0),
            (0.0, 1e-3),
            (1e3, 1.0),
            (1e6, 1.0),
            (1e6, 1e-3),
            (-1e8, 1.0),
            (-1e8, 1e-3),
        ];
        // Half a clock unit is the last row alone.
        let windows = [
            (0.5, 1.0),
            (0.5, 40.0),
            (1.0, 3.0),
            (9.0, 10.0),
            (9.0, 30.0),
            (199.0, 70.0),
            (1999.0, 2.0),
        ];
        for &(level, scale) in &shapes {
            for &(window, h) in &windows {
                for before in [0usize, 3] {
                    for seed in 0..3 {
                        let n = 400 + window as usize;
                        let from = n - 1 - window as usize - before;
                        let (w, c, _) =
                            windowed(n, window, h, level, scale, from, &weights, &[], seed);
                        let case = format!(
                            "level {level}, scale {scale}, window {window}, h {h}, from {before} before, seed {seed}"
                        );
                        assert_eq!(w.var(1), 0.0, "{case}: variance");
                        for j in [0, 2] {
                            assert_eq!(w.cov(j, 1), 0.0, "{case}: covariance with {j}");
                            assert_eq!(w.cov(1, j), 0.0, "{case}: covariance with {j}, transposed");
                        }
                        assert_eq!(w.mean(1), c, "{case}: the value held is the mean");
                        if window >= 1.0 {
                            assert!(
                                w.var(0) > 0.0,
                                "{case}: the feature that moves keeps its spread"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The other side of the run: a feature whose value changes once inside
    /// the window, or whose boundary row has another value, keeps its
    /// spread, the variance of the rows inside it. Rows the accumulator does
    /// not learn age the run as they age everything else, so a run that
    /// skipped rows is not credited with weight it no longer carries.
    #[test]
    fn a_feature_that_moves_inside_the_window_keeps_its_spread() {
        let learn_or_skip = |i: usize| match i % 5 {
            1 => Row::Skip,
            3 => Row::Decay,
            _ => weights(i),
        };
        for (window, h, rows) in [
            (9.0, 10.0, &weights as &dyn Fn(usize) -> Row),
            (30.0, 8.0, &learn_or_skip),
            (7.0, 2.0, &learn_or_skip),
        ] {
            let n = 300;
            let boundary = n - 1 - window as usize;
            assert!(
                matches!(rows(boundary), Row::Learn(_)),
                "the boundary row must be learned"
            );
            // The run starts one row after the boundary, whose row carries
            // another value; then a run from the boundary with one jump.
            for (from, jumps) in [
                (boundary + 1, vec![(boundary, 5.0)]),
                (boundary, vec![(n - 3, 7.0)]),
            ] {
                let (w, _, inside) = windowed(n, window, h, 0.0, 1.0, from, rows, &jumps, 1);
                let total: f64 = inside.iter().map(|&(_, w)| w).sum();
                let mean = inside.iter().map(|&(x, w)| w * x).sum::<f64>() / total;
                let exact = inside
                    .iter()
                    .map(|&(x, w)| w * (x - mean).powi(2))
                    .sum::<f64>()
                    / total;
                assert!(exact > 1e-6, "window {window}: the case needs a spread");
                let got = w.var(1);
                assert!(
                    (got - exact).abs() <= 1e-9 * exact,
                    "window {window}, h {h}, from {from}: {got} against {exact}"
                );
            }
        }
    }

    /// A row of weight 0 is legal and learns nothing (CLAUDE.md hard rule
    /// 9), so its value is none of the window's spread: a zero-weight row
    /// carrying another value inside the window does not end the run.
    #[test]
    fn a_row_of_no_weight_does_not_end_a_run() {
        let (window, n) = (9.0, 300);
        let odd = n - 4;
        let rows = |i: usize| {
            if i == odd {
                Row::Learn(0.0)
            } else {
                weights(i)
            }
        };
        let (w, c, _) = windowed(n, window, 10.0, 1e3, 1.0, n - 20, &rows, &[(odd, -5.0)], 2);
        assert_eq!(w.var(1), 0.0);
        assert_eq!(w.mean(1), c);
        // Nor does a row whose weight is nothing next to the window's: the
        // boundary row, at a weight far below `EMPTY_FRACTION` of the live
        // weight, carrying another value before the run begins. What the run
        // misses of the window's weight is within what the window's own
        // weight counts as nothing, so the feature has no spread there.
        let boundary = n - 1 - window as usize;
        let rows = |i: usize| {
            if i == boundary {
                Row::Learn(1e-12)
            } else {
                weights(i)
            }
        };
        let (w, c, _) = windowed(
            n,
            window,
            10.0,
            1e3,
            1.0,
            boundary + 1,
            &rows,
            &[(boundary, -5.0)],
            3,
        );
        assert_eq!(w.var(1), 0.0, "a boundary row of no account");
        assert_eq!(w.mean(1), c);
    }

    /// The module doc's promise -- a coarse `every` discards more than
    /// asked, never less -- held only while a snapshot stayed inside the
    /// window. After a clock gap longer than the window, or where `every`
    /// rows span more than the window, the newest snapshot was older than
    /// the window, `trim` kept it as the boundary, and rows the window
    /// excludes stayed in the fit (found testing review 2026-09-12, S6).
    #[test]
    fn the_boundary_is_never_older_than_the_window() {
        let ramp = |n: u32| (0..n).map(f64::from).collect::<Vec<_>>();
        let gap = |mut v: Vec<f64>| {
            let end = v[v.len() - 1];
            v.extend([end + 100.0, end + 101.0, end + 102.0]);
            v
        };
        for (every, window, clocks) in [
            // A gap longer than the window.
            (5, 20.0, gap(ramp(101))),
            // No gap: five rows between snapshots span more than the window.
            (5, 2.5, ramp(30)),
            // The control: every row is snapshotted.
            (1, 20.0, gap(ramp(101))),
        ] {
            let mut snaps = Snapshots::new(window, every).unwrap();
            for (i, &t) in clocks.iter().enumerate() {
                snaps.offer(t, || i);
                snaps.trim(t);
                let &(b, _) = snaps.boundary().unwrap();
                assert!(
                    b >= t - window,
                    "every {every}, window {window}: the row at {t} is bounded at {b}, \
                     outside the window"
                );
            }
        }
    }

    /// A toy snapshot whose footprint leaves out a vector that is as long
    /// in both shapes: the check must still see it (review 2026-09-26, B4).
    #[test]
    #[should_panic(expected = "does not grow")]
    fn the_footprint_check_sees_a_vector_that_does_not_grow() {
        #[derive(Serialize)]
        struct Toy {
            a: Vec<f64>,
            b: Vec<f64>,
            w: f64,
            q: f64,
            r: f64,
        }
        impl Footprint for Toy {
            fn footprint(&self) -> usize {
                3 * std::mem::size_of::<f64>() + floats(&self.a)
            }
        }
        let toy = |n: usize| Toy {
            a: vec![1.0; n],
            b: vec![1.0; 2],
            w: 0.0,
            q: 0.0,
            r: 0.0,
        };
        assert_footprint_counts_every_vector(&toy(2), &toy(6), "toy");
    }

    /// `takes` says exactly when `offer` forms a snapshot: under every
    /// cadence, across clock gaps, with a thinning budget and a refusing
    /// one, before and after each trips (review 2026-09-26, B missing 4).
    #[test]
    fn takes_says_when_offer_forms_a_snapshot() {
        /// A snapshot of one MiB, so a budget of a few MiB trips in a few rows.
        struct Big;
        impl Footprint for Big {
            fn footprint(&self) -> usize {
                1 << 20
            }
        }
        let mut seed = 5u64;
        let mut lcg = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut tripped = 0;
        for every in [1usize, 2, 3, 5] {
            for budget in [
                None,
                Some(WindowBudget::Thin(4.0)),
                Some(WindowBudget::Refuse(3.0)),
            ] {
                let mut s: Snapshots<Big> = Snapshots::new(6.0, every).unwrap();
                s.set_budget(budget);
                let (mut clock, mut formed, mut refused) = (0.0, 0, false);
                for _ in 0..200 {
                    clock += if lcg() > 0.6 { 4.0 } else { 0.5 };
                    let said = s.takes(clock);
                    let mut made = false;
                    s.offer(clock, || {
                        made = true;
                        Big
                    });
                    assert_eq!(said, made, "every {every}, {budget:?}, at {clock}");
                    formed += usize::from(made);
                    refused |= s.over_budget().is_some();
                    s.trim(clock);
                }
                // A refusing budget that trips forms nothing from then on:
                // at least the budget's worth first (how many more depends
                // on what the clock's gaps trimmed meanwhile), and no more;
                // one that never trips forms as the cadence says, like the
                // others.
                if refused {
                    assert!(matches!(budget, Some(WindowBudget::Refuse(_))));
                    assert!(formed >= 3, "every {every}: {formed} then refused");
                    tripped += 1;
                } else {
                    assert!(formed > 10, "every {every}, {budget:?}: {formed} snapshots");
                }
            }
        }
        assert!(
            tripped >= 1,
            "the refusing budget tripped at the shortest cadence"
        );
    }

    /// Ten of the test rings' eight-byte snapshots, in MiB.
    const TEN: f64 = 80.0 / (1024.0 * 1024.0);

    /// A ring of `n` snapshots a clock unit apart, then given `budget`.
    fn ring_of(n: usize, window: f64, budget: Option<WindowBudget>) -> Snapshots<usize> {
        let mut r = Snapshots::new(window, 1).unwrap();
        for i in 0..n {
            r.offer(i as f64, || i);
            r.trim(i as f64);
        }
        r.set_budget(budget);
        r
    }

    /// A ring's shadow, learning the rows the ring takes, reports the
    /// overrun the ring reaches, when it reaches it, and none where the ring
    /// stays inside its budget: a window of three clock units holds four
    /// snapshots at most, one of thirty reaches ten at the eleventh row --
    /// from a first row at clock 10, and at clock 0 (docs/PLAN.md task 115
    /// (d)).
    #[test]
    fn a_shadow_reports_the_overrun_its_ring_reaches() {
        for (window, start, trips_at) in [
            (3.0, 10.0, None),
            (3.0, 0.0, None),
            (30.0, 10.0, Some(10)),
            (30.0, 0.0, Some(10)),
        ] {
            let mut ring: Snapshots<usize> = Snapshots::new(window, 1).unwrap();
            ring.set_budget(Some(WindowBudget::Refuse(TEN)));
            ring.offer(start, || 0);
            ring.trim(start);
            let mut shadow = WindowShadow::new(start, &ring, || 0);
            let mut clock = start;
            let mut tripped = None;
            for i in 1..40 {
                clock += 1.0;
                ring.offer(clock, || i);
                ring.trim(clock);
                shadow.learn(1.0);
                let case = format!("window {window}, from {start}, row {i}");
                assert_eq!(shadow.over_budget(), ring.over_budget(), "{case}");
                if tripped.is_none() && ring.over_budget().is_some() {
                    tripped = Some(i);
                }
            }
            assert_eq!(tripped, trips_at, "window {window}, from {start}");
            if trips_at.is_some() {
                assert_eq!(shadow.over_budget(), Some((88, 1)), "eleven snapshots");
            }
        }
    }

    /// `could_refuse` says whether that many rows could take the ring past a
    /// refusing budget, a snapshot a row at most: a budget reached exactly
    /// still fits, one row more does not, a ring already past it could
    /// refuse with no row at all, and a thinning budget or none refuses
    /// nothing.
    #[test]
    fn could_refuse_says_whether_rows_could_cross_a_refusing_budget() {
        let refuse = Some(WindowBudget::Refuse(TEN));
        // Four snapshots of eight bytes held, room for six more.
        let shadow = WindowShadow::new(3.0, &ring_of(4, 1e9, refuse), || 0);
        assert!(!shadow.could_refuse(0));
        assert!(!shadow.could_refuse(6), "eighty bytes: the budget exactly");
        assert!(shadow.could_refuse(7), "eighty-eight");
        for budget in [None, Some(WindowBudget::Thin(TEN))] {
            let shadow = WindowShadow::new(3.0, &ring_of(4, 1e9, budget), || 0);
            assert!(!shadow.could_refuse(1000), "{budget:?}");
        }
        let over = ring_of(12, 1e9, refuse);
        assert_eq!(over.over_budget(), Some((96, 1)), "the fixture");
        assert!(WindowShadow::new(11.0, &over, || 0).could_refuse(0));
    }

    /// The accessors: the window as given, the snapshots oldest first, and
    /// whether there are any.
    #[test]
    fn a_ring_reports_its_window_and_its_snapshots_oldest_first() {
        let mut r: Snapshots<usize> = Snapshots::new(7.5, 1).unwrap();
        assert_eq!(r.window(), 7.5);
        assert!(r.is_empty());
        assert_eq!(r.iter().count(), 0);
        for i in 0..4 {
            r.offer(i as f64, || 10 + i);
            r.trim(i as f64);
        }
        assert!(!r.is_empty());
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![10, 11, 12, 13]);
    }

    /// A ring stopped by a refusing budget makes no snapshot, so the
    /// clock can leave its newest behind: `trim` keeps that one, the
    /// boundary, however old it is.
    #[test]
    fn trim_keeps_the_newest_snapshot_however_old() {
        let mut r: Snapshots<usize> = Snapshots::new(5.0, 1).unwrap();
        r.set_budget(Some(WindowBudget::Refuse(8.0 / (1024.0 * 1024.0))));
        r.offer(0.0, || 0);
        r.trim(0.0);
        r.offer(100.0, || 1);
        assert_eq!(r.over_budget(), Some((16, 1)), "the second is refused");
        r.trim(100.0);
        assert_eq!(r.len(), 1);
        assert_eq!(r.boundary().map(|b| b.0), Some(0.0));
    }

    /// A snapshot exactly a window old is inside the window (the module
    /// docs: `t_j >= t - window`), so it does not force one off the
    /// cadence; a hair older does.
    #[test]
    fn a_snapshot_at_the_window_edge_is_inside_it() {
        let mut r: Snapshots<usize> = Snapshots::new(5.0, 10).unwrap();
        r.offer(0.0, || 0);
        assert!(!r.takes(5.0), "exactly a window old");
        r.offer(5.0, || 1);
        assert_eq!(r.len(), 1, "and the cadence is not due");
        assert!(r.takes(5.5));
        r.offer(5.5, || 2);
        assert_eq!(r.len(), 2);
    }

    /// Thinning keeps the newest snapshot and every second one before it,
    /// and stops as soon as the ring fits: six snapshots under a budget of
    /// three keep the second, fourth and sixth, at twice the spacing.
    #[test]
    fn thinning_keeps_the_newest_and_every_second_before_it() {
        let r = ring_of(6, 100.0, Some(WindowBudget::Thin(24.0 / (1024.0 * 1024.0))));
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![1, 3, 5]);
        assert_eq!(r.every, 2);
        assert_eq!(r.bytes(), 24, "the budget exactly, which fits");
    }

    /// One snapshot larger than a thinning budget is kept: there is nothing
    /// to thin, and thinning a ring of one would never end.
    #[test]
    fn a_ring_of_one_snapshot_past_a_thinning_budget_keeps_it() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut r: Snapshots<usize> = Snapshots::new(100.0, 1).unwrap();
            r.set_budget(Some(WindowBudget::Thin(4.0 / (1024.0 * 1024.0))));
            r.offer(0.0, || 7);
            let _ = tx.send(r.iter().copied().collect::<Vec<_>>());
        });
        let kept = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("thinning a ring of one snapshot did not end");
        assert_eq!(kept, vec![7]);
    }

    /// What a `Moments` snapshot counts toward a budget: eight bytes for
    /// every number it holds, with every field present.
    #[test]
    fn a_moments_snapshot_counts_eight_bytes_for_every_number_it_holds() {
        fn numbers(v: &serde_json::Value) -> usize {
            match v {
                serde_json::Value::Number(_) => 1,
                serde_json::Value::Array(a) => a.iter().map(numbers).sum(),
                serde_json::Value::Object(o) => o.values().map(numbers).sum(),
                _ => 0,
            }
        }
        let snap = Moments {
            w: 3.0,
            m: vec![1.0, -2.0],
            c: vec![4.0, 0.5, 0.5, 9.0],
            rows: Some(7),
            q: Some(2.5),
        };
        let held = numbers(&serde_json::to_value(&snap).unwrap());
        assert_eq!(held, 9, "the fixture");
        assert_eq!(snap.footprint(), 8 * held);
    }

    /// A window holding less than `EMPTY_FRACTION` of the live weight is
    /// empty, at a live weight of a million and of a millionth: the
    /// fraction is relative, so a remainder ten times it is a window at
    /// either scale, and one a tenth of it is none. So is a remainder of
    /// nothing at all.
    #[test]
    fn a_window_is_empty_below_a_fraction_of_the_live_weight_at_any_scale() {
        for w_now in [1e6, 1e-6] {
            let mut cov = EwCov::new(1);
            cov.update(&[2.0], 1.0, w_now);
            assert_eq!(cov.n_eff(), w_now, "the fixture");
            for (rest, empty) in [
                (0.0, true),
                (0.1 * EMPTY_FRACTION * w_now, true),
                (10.0 * EMPTY_FRACTION * w_now, false),
            ] {
                let w_old = w_now - rest;
                let case = format!("live {w_now:e}, remainder {rest:e}");
                assert_eq!(
                    truncated_scalar(w_now, 2.0, w_old, 2.0, 1.0).is_none(),
                    empty,
                    "{case}: scalar"
                );
                assert_eq!(
                    truncated_mean(w_now, &[2.0], w_old, &[2.0], 1.0).is_none(),
                    empty,
                    "{case}: mean"
                );
                let mut old = Moments::of(&cov, 1.0);
                old.w = w_old;
                assert_eq!(
                    truncated(&cov, &old, 1.0).is_none(),
                    empty,
                    "{case}: moments"
                );
            }
        }
    }
}
