//! The plain run's row loop (docs/PLAN.md task 429).
//!
//! [`Windows::push`] decides, for every row, what the clocks make of it, whether a
//! window ended, which stamp it belongs to and which group's queues it
//! joins, though a stream in one group whose clock steps forward within the
//! cap and its session has none of those to decide, whatever its inputs hold.
//! This is that stream's loop: a block of up to [`BLOCK`] such rows is found
//! in one pass over its clock, and every backward kernel without a window is
//! then folded over the block one operator at a time, the distance to the
//! row before and the two decays it asks for taken once for the kernel, not
//! once for each of its operators. A row at the stamp of the one before it
//! (a tie) takes no read of its own, as in `push`: the stamp's rows share
//! the read made as the next stamp arrives, or before the stamp's first row
//! joins under `"left"` and `"none"`. A null input adds nothing to its
//! operator and moves nothing it holds. Every number is the one `push` makes,
//! by the same functions ([`Acc::scaled`], [`Acc::plus`], [`value_of`],
//! [`mass_held_to`], [`KernelDef::discount_in`]) applied in the same order to
//! the same doubles, so a run of the plain loop and a run of `push` are equal
//! to the bit in every output and every byte of the saved state; the block
//! ends at the first row that is not plain, which `push` takes, and the loop
//! goes on after it. `crates/online-polars/tests/window_plain.rs` holds the
//! two to it.

use super::*;

/// Rows a block holds: the buffers it keeps (a few doubles a row) stay in
/// the first level of the cache.
const BLOCK: usize = 1024;

/// What the plain loop keeps between blocks, none of it state.
#[derive(Debug, Clone, Default)]
pub(super) struct Bufs {
    /// The block's policy times.
    tau: Vec<f64>,
    /// A kernel's decay across each row's step.
    f: Vec<f64>,
    /// A kernel's mass of each row's step.
    m: Vec<f64>,
    /// A rate's mass over the span each step's read has.
    rate: Vec<f64>,
    /// Per row, whether its stamp is a new one: not the stamp of the row before.
    new: Vec<bool>,
    /// Per row, its raw clock as a window's edge is decided from it.
    off: Vec<i64>,
}

/// What a core's definitions say once, for the rows that follow: derived
/// when the core is built or read back, never saved.
#[derive(Debug, Clone, Default)]
pub(super) struct Derived {
    /// Whether the rest is worked out.
    pub(super) ready: bool,
    /// Backward windows that wait for the next distinct stamp ([`Closed::near`]):
    /// 1 if any kernel's does, else 0.
    pub(super) n_wait: u32,
    /// Whether every kernel looks back: a core with a forward kernel takes
    /// `push` for every row.
    plain: bool,
    /// The plain loop is switched off (`Windows::set_plain_run`).
    pub(super) off: bool,
    /// Rows the plain loop has taken since the core was built or read back.
    taken: u64,
    bufs: Bufs,
}

impl PartialEq for Derived {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Derived {
    pub(super) fn of(kernels: &[KernelDef], n_forward: u32, off: bool) -> Self {
        let n_wait = u32::from(
            kernels
                .iter()
                .any(|k| k.direction == Direction::Backward && k.closed.near(Direction::Backward)),
        );
        let plain = n_forward == 0 && kernels.iter().all(|k| k.direction == Direction::Backward);
        Derived {
            ready: true,
            n_wait,
            plain,
            off,
            taken: 0,
            bufs: Bufs::default(),
        }
    }
}

/// The rows [`Windows::push_plain`] is offered: a chunk's columns, read from
/// `start` on.
pub(crate) struct PlainRows<'a> {
    /// The group every row of the chunk is in (from [`Windows::group`]).
    pub group: usize,
    pub start: usize,
    pub clock: &'a [ClockValue],
    pub session: Option<&'a [u64]>,
    /// Per operator input, every row of the chunk (NaN for none).
    pub values: &'a [Vec<f64>],
    /// Per `like=` column, as `values`: a row is accepted where all are usable.
    pub accepts: &'a [Vec<f64>],
    /// The rows' group keys, where there is a group column: a block ends at a
    /// row of another key.
    pub keys: Option<&'a [Option<&'a str>]>,
}

/// One operator's fold over a block. `NEAR`: its kernel's windows wait for
/// the next distinct stamp. `TIES`: some row of the block repeats the stamp
/// before it. `NULLS`: some input of the block is not usable. See
/// [`Windows::plain_block`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn fold_op<const NEAR: bool, const TIES: bool, const NULLS: bool>(
    k: &KernelDef,
    op: &OpDef,
    var: bool,
    wide: bool,
    total: &mut [f64],
    x: &[f64],
    steps: Steps<'_>,
    decay: &mut DecayMemo,
    out: &mut [f64],
    o: usize,
    n_out: usize,
    rows_at: Rows<'_>,
    stamp: &mut f64,
    last_valued: &mut f64,
) {
    let mut acc = Acc::read(total, wide);
    let weighs = op.stat.weighs_held();
    let rate = op.stat == Stat::Rate;
    let read = |a: Acc, r: usize| -> f64 {
        if rate {
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                rate_of(a, steps.rate[r])
            }
        } else {
            value_of(k, op, a, 0.0)
        }
    };
    // Where the stamp the next read is of begins, among the block's rows: -1
    // for the rows pending from before the block, which it includes.
    let mut began: isize = -1;
    let mut lv = *last_valued;
    let mut cur = *stamp;
    for (r, &xr) in x.iter().enumerate() {
        let f = steps.f[r];
        let scaled = acc.scaled(f, var);
        let is_new = !TIES || steps.new[r];
        // Under "left" and "none" the stamp's window is read before its first
        // row joins, from the sums a step older than the row; its other rows
        // are written the same. Under "right" and "both" the previous stamp's
        // window is read as the new one arrives, from the sums as they stand
        // (a distance of 0 discounts nothing), and written to its rows.
        if NEAR {
            if is_new {
                cur = read(acc, r);
                if began < 0 {
                    for &at in rows_at.pending {
                        out[at * n_out + o] = cur;
                    }
                }
                let from = usize::try_from(began).unwrap_or(0);
                for q in from..r {
                    out[(rows_at.first + q) * n_out + o] = cur;
                }
                began = r as isize;
            }
        } else {
            if is_new {
                cur = read(scaled, r);
            }
            out[(rows_at.first + r) * n_out + o] = cur;
        }
        // The row's own sums, as `push` makes them: a value held over its
        // interval for a mean or a variance, the value itself for a sum, and
        // nothing for a null.
        let own = if NULLS && !usable(xr) {
            Acc::default()
        } else if weighs {
            let m = if NULLS {
                // From the operator's last valued row: the row before, as for
                // every operator that has had a value at each row, where the
                // kernel's mass is the same call's; else its own.
                if r > 0 && lv == steps.tau[r - 1] {
                    steps.m[r]
                } else {
                    let start = if lv.is_nan() { f64::NEG_INFINITY } else { lv };
                    mass_held_to(k, steps.tau[r], start, decay)
                }
            } else if r == 0 {
                steps.m0
            } else {
                steps.m[r]
            };
            Acc::held(xr, m, op.stat.is_var())
        } else {
            Acc {
                s: xr,
                n: 1.0,
                ..Acc::default()
            }
        };
        if NULLS && usable(xr) {
            lv = steps.tau[r];
        }
        acc = scaled.plus(own, var);
    }
    acc.write(total, wide);
    *stamp = cur;
    *last_valued = if NULLS { lv } else { steps.tau[x.len() - 1] };
}

/// A kernel's per-row factors over a block.
#[derive(Clone, Copy)]
struct Steps<'a> {
    tau: &'a [f64],
    f: &'a [f64],
    m: &'a [f64],
    rate: &'a [f64],
    new: &'a [bool],
    /// The first row's mass, which this operator's own last valued row sets.
    m0: f64,
}

/// Where a block's rows are in the held values: the first one's index, and
/// the indices of the rows pending from before.
#[derive(Clone, Copy)]
struct Rows<'a> {
    first: usize,
    pending: &'a [usize],
}

/// Whether a kernel is folded a block at a time by [`fold_op`]: its sums
/// are the running total, with no window to evict from and no sketch to feed.
fn fast_kernel(k: &KernelDef, members: &[usize], ops: &[OpDef]) -> bool {
    k.window_size.is_none() && members.iter().all(|&o| ops[o].stat != Stat::Quantile)
}

impl Windows {
    /// Switch the plain loop off or on again: for a test that holds it to
    /// `push`, and a timing of what it saves. Not state.
    #[doc(hidden)]
    pub fn set_plain_run(&mut self, on: bool) {
        self.scratch();
        self.derived.off = !on;
    }

    /// Rows the plain loop has taken: for a test to see that it ran.
    #[doc(hidden)]
    pub fn plain_rows(&self) -> u64 {
        self.derived.taken
    }

    /// Whether [`Windows::push_plain`] could take a row now: a plain core
    /// with one group (or one not yet worked out), the plain loop on. A
    /// caller asks this before it builds the rows, so a stream of many
    /// groups pays one test a row for the loop it cannot use.
    #[inline]
    pub(crate) fn plain_may_run(&self) -> bool {
        !self.derived.ready || (self.derived.plain && !self.derived.off && self.groups.len() == 1)
    }

    /// Take as many of `rows`, from its start, as the plain loop runs: each a
    /// step forward in time within the cap and its session, in the one group
    /// of a core whose kernels look back with no window, every input of it
    /// usable. Returns how many; the rest is for [`Windows::push`], and so is
    /// the first row of a stream and any row after a cut. The state after the
    /// rows is the state `push` would have left.
    pub(crate) fn push_plain(&mut self, rows: &PlainRows<'_>) -> usize {
        self.scratch();
        if !self.derived.plain || self.derived.off {
            return 0;
        }
        let mut done = 0;
        while rows.start + done < rows.clock.len() {
            let got = self.plain_block(rows, rows.start + done);
            done += got;
            if got < BLOCK {
                break;
            }
        }
        self.derived.taken += done as u64;
        done
    }

    /// One block of up to [`BLOCK`] plain rows from `at`.
    #[allow(clippy::too_many_lines)]
    fn plain_block(&mut self, rows: &PlainRows<'_>, at: usize) -> usize {
        if self.groups.len() != 1 || rows.group != 0 {
            return 0;
        }
        let n_rows = (rows.clock.len() - at).min(BLOCK);
        let cap = self.clock_cfg.gap_cap;
        let grouped = self.grouped;
        let n_wait = self.derived.n_wait;
        // Where the stream stands: the last row's clock, the stretch's origin,
        // and the group's own state, which a plain row continues.
        let (prev, origin, g_tau, g_off, form, prev_session) = {
            let g = &self.groups[0];
            let (Some(prev), Some(origin)) = (g.stamp, g.origin) else {
                return 0;
            };
            if g.restart
                || !g.linked
                || g.last_raw != Some(prev)
                || g.clock.last_clock() != Some(prev)
                || self.shared.last_clock() != Some(prev)
                || (n_wait > 0) == g.pending.is_empty()
                || !(grouped || self.shared.prev_session() == g.clock.prev_session())
            {
                return 0;
            }
            // What a read at the last row's own time reads: the sums as they
            // stand, so each queue is at it.
            let ready = self
                .kernels
                .iter()
                .zip(&g.queues)
                .zip(self.members.iter())
                .all(|((k, q), m)| {
                    !fast_kernel(k, m, &self.ops)
                        || (q.total_at == g.tau && q.total.len() == q.sum_w() * q.n)
                });
            if !ready {
                return 0;
            }
            (prev, origin, g.tau, g.off, g.form, g.clock.prev_session())
        };

        // The block's rows: one pass over the clock and the sessions.
        let mut bufs = std::mem::take(&mut self.derived.bufs);
        bufs.tau.clear();
        bufs.new.clear();
        bufs.off.clear();
        let mut last = prev;
        let mut last_session = prev_session;
        let (mut ties, mut news) = (false, 0usize);
        for j in 0..n_rows {
            let now = rows.clock[at + j];
            // A step forward within the cap, or no step: the same stamp.
            let step_ok = match (now, last) {
                (ClockValue::Ns(c), ClockValue::Ns(p)) => {
                    c == p || (c > p && !(cap.is_finite() && past_cap(now, last, cap)))
                }
                (ClockValue::F64(c), ClockValue::F64(p)) => {
                    let raw = c - p;
                    (0.0..=1e308).contains(&raw) && raw <= cap
                }
                _ => false,
            };
            let session = rows.session.map(|s| s[at + j]);
            let session_ok = match (session, last_session) {
                (Some(s), Some(q)) => s == q,
                _ => true,
            };
            let keys_ok = rows.keys.is_none_or(|k| k[at + j] == k[at]);
            if !(step_ok && session_ok && keys_ok) {
                break;
            }
            let new = now != last;
            ties |= !new;
            news += usize::from(new);
            bufs.new.push(new);
            bufs.tau.push(elapsed(now, origin));
            bufs.off.push(match now {
                ClockValue::Ns(c) | ClockValue::I64(c) => c,
                ClockValue::F64(c) => c.to_bits() as i64,
            });
            last = now;
            last_session = session;
        }
        let len = bufs.tau.len();
        if len == 0 {
            self.derived.bufs = bufs;
            return 0;
        }
        // The stamp the block ends in begins at its last new stamp, or runs
        // on from the rows pending before it.
        let final_from = if news == 0 {
            0
        } else {
            bufs.new.iter().rposition(|&n| n).unwrap_or(0)
        };

        let Windows {
            kernels,
            ops,
            members,
            groups,
            held_values,
            held_meta,
            held_first,
            n_outputs,
            next_seq,
            scratch_row,
            scratch_sum,
            ..
        } = &mut *self;
        let n_out = *n_outputs;
        let g = &mut groups[0];
        let seq0 = *next_seq;
        let first = usize::try_from(seq0 - *held_first).expect("held rows precede the next");
        // The rows pending from before, as held indices; a new stamp's read
        // goes to them. A kernel without a wait has none.
        let pending: Vec<usize> = g
            .pending
            .iter()
            .map(|&s| usize::try_from(s - *held_first).expect("a pending row is held"))
            .collect();
        if news > 0 {
            for &p in &pending {
                held_meta[p].open -= 1;
            }
        }
        held_values.extend(std::iter::repeat_n(f64::NAN, len * n_out));
        for r in 0..len {
            held_meta.push_back(Meta {
                open: if r >= final_from { n_wait } else { 0 },
                accept: rows.accepts.iter().all(|c| usable(c[at + r])),
                drop: false,
            });
        }
        // A kernel with a window, or a quantile, is the queue's: the same reads,
        // row values and pushes `push` makes, a row at a time, without what
        // `push` decides around them. Each kernel's queue is its own, so a
        // kernel is taken over the whole block before the next.
        for (ki, k) in kernels.iter().enumerate() {
            let m = &members[ki];
            if fast_kernel(k, m, ops) {
                continue;
            }
            let q = &mut g.queues[ki];
            let near = k.closed.near(Direction::Backward);
            let (rw, wide) = (q.row_w(), q.wide());
            // Where the stamp the next read is of begins, among the block's
            // rows: -1 for the rows pending from before it.
            let mut began: isize = -1;
            for r in 0..len {
                let (tau, off, new) = (bufs.tau[r], bufs.off[r], bufs.new[r]);
                let seq = seq0 + r as u64;
                if new && near {
                    // The stamp before this one is complete: its rows are
                    // read at its own time.
                    let (ptau, poff) = if r == 0 {
                        (g_tau, g_off)
                    } else {
                        (bufs.tau[r - 1], bufs.off[r - 1])
                    };
                    by_width!(
                        wide,
                        read_backward(
                            k,
                            ops,
                            m,
                            q,
                            scratch_sum,
                            ptau,
                            poff,
                            form,
                            &mut g.stamp_values[ki],
                        )
                    );
                    let complete = k.window_size.is_none_or(|w| ptau >= w);
                    let from = usize::try_from(began).unwrap_or(0);
                    let nothing: &[u64] = &[];
                    let earlier = if began < 0 {
                        g.pending.as_slice()
                    } else {
                        nothing
                    };
                    let theirs = (from..r).map(|q| seq0 + q as u64);
                    for pending in earlier.iter().copied().chain(theirs) {
                        let held = Held {
                            first: *held_first,
                            n_outputs: n_out,
                            values: &mut *held_values,
                            meta: &mut *held_meta,
                        };
                        write_backward(ops, m, pending, &g.stamp_values[ki], complete, held);
                    }
                    began = r as isize;
                }
                if new && !near {
                    by_width!(
                        wide,
                        read_backward(
                            k,
                            ops,
                            m,
                            q,
                            scratch_sum,
                            tau,
                            off,
                            form,
                            &mut g.stamp_values[ki],
                        )
                    );
                }
                // A value is held from the operator's last valued row; the
                // first of a stretch from before it: from `-inf`, or from a
                // window back where one bounds the mass.
                for (j, &o) in m.iter().enumerate() {
                    let v = rows.values[ops[o].input][at + r];
                    let v = if usable(v) { v } else { f64::NAN };
                    let start = if g.last_valued[o].is_nan() {
                        k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w)
                    } else {
                        g.last_valued[o]
                    };
                    let stat = ops[o].stat;
                    let acc = if v.is_nan() {
                        Acc::default()
                    } else if stat.weighs_held() {
                        Acc::held(v, mass_held_to(k, tau, start, &mut q.decay), stat.is_var())
                    } else {
                        Acc {
                            s: v,
                            n: 1.0,
                            ..Acc::default()
                        }
                    };
                    let b = j * rw;
                    scratch_row[b] = start;
                    scratch_row[b + 1] = v;
                    acc.write(&mut scratch_row[b + 2..], wide);
                    if !v.is_nan() {
                        g.last_valued[o] = tau;
                    }
                }
                q.push_row(
                    k,
                    seq,
                    tau,
                    off,
                    tau,
                    &scratch_row[..rw * m.len()],
                    scratch_sum,
                );
                if !near {
                    let complete = k.window_size.is_none_or(|w| tau >= w);
                    let held = Held {
                        first: *held_first,
                        n_outputs: n_out,
                        values: &mut *held_values,
                        meta: &mut *held_meta,
                    };
                    write_backward(ops, m, seq, &g.stamp_values[ki], complete, held);
                }
            }
        }
        let out = held_values.make_contiguous();
        let rows_at = Rows {
            first,
            pending: &pending,
        };

        for (ki, k) in kernels.iter().enumerate() {
            if !fast_kernel(k, &members[ki], ops) {
                continue;
            }
            let q = &mut g.queues[ki];
            let (wide, sw) = (q.wide(), q.sum_w());
            let near = k.closed.near(Direction::Backward);
            let has_rate = members[ki].iter().any(|&o| ops[o].stat == Stat::Rate);
            let needs_mass = members[ki].iter().any(|&o| ops[o].stat.weighs_held());
            bufs.f.clear();
            bufs.m.clear();
            bufs.rate.clear();
            let mut before = q.total_at;
            for r in 0..len {
                let t = bufs.tau[r];
                bufs.f.push(k.discount_in(t - before, &mut q.decay));
                bufs.m.push(if r == 0 || !needs_mass {
                    0.0
                } else {
                    mass_held_to(k, t, bufs.tau[r - 1], &mut q.decay)
                });
                if has_rate {
                    // The read at this step: the previous stamp's under "right"
                    // and "both", the row's own before it joins otherwise.
                    let span = match (near, r) {
                        (true, 0) => g_tau,
                        (true, _) => bufs.tau[r - 1],
                        (false, _) => t,
                    };
                    bufs.rate.push(k.mass(span));
                }
                before = t;
            }
            for (j, &o) in members[ki].iter().enumerate() {
                let op = &ops[o];
                let var = wide && q.var[j];
                let x = &rows.values[op.input][at..at + len];
                let nulls = !x.iter().fold(true, |ok, &v| ok & usable(v));
                // The first row's mass is from the operator's own last valued
                // row, which a null input moved for this operator alone.
                let start = if g.last_valued[o].is_nan() {
                    f64::NEG_INFINITY
                } else {
                    g.last_valued[o]
                };
                let m0 = if op.stat.weighs_held() && !nulls {
                    mass_held_to(k, bufs.tau[0], start, &mut q.decay)
                } else {
                    0.0
                };
                let steps = Steps {
                    tau: &bufs.tau,
                    f: &bufs.f,
                    m: &bufs.m,
                    rate: &bufs.rate,
                    new: &bufs.new,
                    m0,
                };
                macro_rules! fold {
                    ($near:literal, $ties:literal, $nulls:literal) => {
                        fold_op::<$near, $ties, $nulls>(
                            k,
                            op,
                            var,
                            wide,
                            &mut q.total[j * sw..],
                            x,
                            steps,
                            &mut q.decay,
                            out,
                            o,
                            n_out,
                            rows_at,
                            &mut g.stamp_values[ki][j],
                            &mut g.last_valued[o],
                        )
                    };
                }
                match (near, ties, nulls) {
                    (true, false, false) => fold!(true, false, false),
                    (true, false, true) => fold!(true, false, true),
                    (true, true, false) => fold!(true, true, false),
                    (true, true, true) => fold!(true, true, true),
                    (false, false, false) => fold!(false, false, false),
                    (false, false, true) => fold!(false, false, true),
                    (false, true, false) => fold!(false, true, false),
                    (false, true, true) => fold!(false, true, true),
                }
            }
            q.total_at = bufs.tau[len - 1];
        }

        // The group and the stream where the last row leaves them.
        let (tau, now) = (bufs.tau[len - 1], rows.clock[at + len - 1]);
        let session = rows.session.map(|s| s[at + len - 1]);
        g.tau = tau;
        (g.off, g.form) = match now {
            ClockValue::Ns(c) => (c, OffForm::Ns),
            ClockValue::I64(c) => (c, OffForm::Int),
            ClockValue::F64(c) => (c.to_bits() as i64, OffForm::Bits),
        };
        g.stamp = Some(now);
        g.last_raw = Some(now);
        if n_wait > 0 {
            if news > 0 {
                g.pending.clear();
            }
            g.pending.extend((final_from..len).map(|r| seq0 + r as u64));
        }
        g.clock.take_plain_run(Some(now), session);
        self.shared
            .take_plain_run(Some(now), if grouped { None } else { session });
        self.next_seq += len as u64;
        self.derived.bufs = bufs;
        self.link_back(0);
        self.advance_ready();
        len
    }
}
