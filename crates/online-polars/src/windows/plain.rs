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
//! goes on after it. A forward kernel (task 441) is taken the same way: a
//! row of a block closes the windows it reaches past and joins the kernel's
//! queue by [`forward_step`], the function `push` calls for it, a row at a
//! time and a kernel over the whole block (the block's rows wait first, none
//! closing a window of its own or a later row's), so the two cannot differ. `crates/online-polars/tests/window_plain.rs` holds the
//! two to it.
//!
//! With several groups (task 431) a pass finds the plain rows in any of them,
//! in row order: the stream's clock steps forward, and each row is a step
//! forward within the cap and the session of its own group, which already has
//! a row. A group's rows are its own whatever the rows of the other groups
//! between them do, so the pass then takes each group over all its rows in it,
//! as the block does one group's, and leaves the stream as the rows in order
//! would have left it. Three things `push` does at each row are done once, at
//! the pass's end, because nothing a row of another group does can change
//! them: a group's pending windows are read at its stamp when a later stamp
//! has passed it (a read of a group nothing touches in between is the same
//! whenever it is made), the groups' places in the silent list follow the
//! order of their last rows, and the groups the last row's clock shows silent
//! past the cap are cut (cut sooner, a group's windows would be read and
//! ended the same, and a row of its own after the cut would have been past
//! the cap and not in the pass).

use super::*;

/// Rows a block holds: the buffers it keeps (a few doubles a row) stay in
/// the first level of the cache.
const BLOCK: usize = 1024;

/// Rows a pass over several groups holds: its buffers are a few doubles a
/// row, and a group's rows in it are the more the longer it is; this keeps
/// the scratch of a core about 2 MB (3 MB at 10,000 groups), where 2^18 rows
/// held 7 to 20 MB, and the groups' rows are run in segments of a block.
const PASS: usize = 1 << 16;

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
    /// Per row, its `seq`.
    seq: Vec<u64>,
    /// The held indices of the group's rows pending from before a run.
    pending: Vec<usize>,
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
    /// The indices of the forward kernels and of the backward ones.
    pub(super) forward: Vec<u32>,
    pub(super) backward: Vec<u32>,
    /// The plain loop is switched off (`Windows::set_plain_run`).
    pub(super) off: bool,
    /// Rows the plain loop has taken since the core was built or read back.
    taken: u64,
    bufs: Bufs,
    multi: Multi,
}

impl PartialEq for Derived {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Derived {
    pub(super) fn of(kernels: &[KernelDef], off: bool) -> Self {
        let n_wait = u32::from(
            kernels
                .iter()
                .any(|k| k.direction == Direction::Backward && k.closed.near(Direction::Backward)),
        );
        let of = |d: Direction| {
            (0..kernels.len())
                .filter(|&ki| kernels[ki].direction == d)
                .map(|ki| ki as u32)
                .collect::<Vec<_>>()
        };
        Derived {
            ready: true,
            forward: of(Direction::Forward),
            backward: of(Direction::Backward),
            n_wait,
            off,
            taken: 0,
            bufs: Bufs::default(),
            multi: Multi::default(),
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
    /// The rows' group keys, where there is a group column: a block of one
    /// group ends at a row of another key, and a pass over several groups at
    /// a row of a group the core has not made.
    pub keys: Option<&'a ChunkKeys>,
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
                    out[rows_at.at(q) * n_out + o] = cur;
                }
                began = r as isize;
            }
        } else {
            if is_new {
                cur = read(scaled, r);
            }
            out[rows_at.at(r) * n_out + o] = cur;
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
    held_first: u64,
    /// Each row's `seq`.
    seq: &'a [u64],
    pending: &'a [usize],
}

impl Rows<'_> {
    /// The held index of the block's row `r`.
    #[inline(always)]
    fn at(&self, r: usize) -> usize {
        (self.seq[r] - self.held_first) as usize
    }
}

/// Whether a kernel is folded a block at a time by [`fold_op`]: its sums
/// are the running total, with no window to evict from and no sketch to feed.
fn fast_kernel(k: &KernelDef, members: &[usize], ops: &[OpDef]) -> bool {
    k.window_size.is_none() && members.iter().all(|&o| ops[o].stat != Stat::Quantile)
}

/// Whether a row at `now` is a step forward from `last` within `cap`, or no
/// step: what `push` takes for a plain row on a clock.
#[inline]
fn step_ok(now: ClockValue, last: ClockValue, cap: f64) -> bool {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => {
            c == p || (c > p && !(cap.is_finite() && past_cap(now, last, cap)))
        }
        (ClockValue::F64(c), ClockValue::F64(p)) => {
            let raw = c - p;
            (0.0..=1e308).contains(&raw) && raw <= cap
        }
        _ => false,
    }
}

/// Whether `now` is at or after `last`: no step back.
#[inline]
fn not_before(now: ClockValue, last: ClockValue) -> bool {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => c >= p,
        (ClockValue::F64(c), ClockValue::F64(p)) => (0.0..=1e308).contains(&(c - p)),
        _ => false,
    }
}

/// A group's rows of one run, as the fold reads their inputs: each input's
/// values for the group's rows, in row order.
#[derive(Clone, Copy)]
struct Cols<'a> {
    whole: &'a [Vec<f64>],
    from: usize,
    len: usize,
}

impl Cols<'_> {
    #[inline(always)]
    fn x(&self, input: usize) -> &[f64] {
        &self.whole[input][self.from..self.from + self.len]
    }
}

/// What a group's run reads of its rows beside their inputs.
struct Shape {
    len: usize,
    /// Rows that begin a stamp: not the stamp of the row before.
    news: usize,
    /// Where the stamp the run ends in begins, among its rows.
    final_from: usize,
    /// The group's policy time and raw clock before the run.
    g_tau: f64,
    g_off: i64,
    form: OffForm,
    /// The run's last row.
    last_now: ClockValue,
    last_session: Option<u64>,
}

/// What a run borrows of the core beside the group.
struct Parts<'a> {
    kernels: &'a [KernelDef],
    ops: &'a [OpDef],
    members: &'a [Vec<usize>],
    held_values: &'a mut VecDeque<f64>,
    held_meta: &'a mut VecDeque<Meta>,
    held_first: u64,
    n_out: usize,
    scratch_row: &'a mut Vec<f64>,
    scratch_sum: &'a mut Vec<f64>,
    n_wait: u32,
    /// Forward kernels: each row of a run waits on them too.
    n_forward: u32,
}

/// A group a pass over several has met.
#[derive(Debug, Clone)]
struct Touch {
    gi: usize,
    /// The group's last clock, and its session, as far as the pass has read.
    last: ClockValue,
    last_session: Option<u64>,
    /// The raw clock its policy times are measured from.
    origin: ClockValue,
    /// Rows of the pass in the group, and the position of the last.
    count: u32,
    last_row: u32,
    /// Rows pending from before, none or some.
    waiting: bool,
}

/// A row of a pass: its group's place in `touched`, whether it begins a
/// stamp, its policy time and its raw clock.
#[derive(Debug, Clone, Copy)]
struct RowRec {
    tau: f64,
    off: i64,
    slot: u32,
    new: bool,
}

/// What a pass over several groups keeps: none of it state.
#[derive(Debug, Clone, Default)]
pub(super) struct Multi {
    /// Per group, whether the pass has met it (`mark == epoch`) and its place
    /// in `touched`.
    epoch: u32,
    mark: Vec<u32>,
    slot: Vec<u32>,
    touched: Vec<Touch>,
    /// Per row of the pass.
    rows: Vec<RowRec>,
    /// The pass's rows by group: positions in the pass, group after group.
    order: Vec<u32>,
    starts: Vec<u32>,
    /// The inputs of the group being run.
    xg: Vec<Vec<f64>>,
    fill: Vec<u32>,
}

impl Multi {
    fn begin(&mut self, groups: usize) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.epoch = 1;
        }
        if self.mark.len() < groups {
            self.mark.resize(groups, 0);
            self.slot.resize(groups, 0);
        }
        self.touched.clear();
        self.rows.clear();
    }
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

    /// Bytes the plain loop's buffers hold: what the loop costs in memory.
    #[doc(hidden)]
    pub fn plain_scratch_bytes(&self) -> usize {
        let d = &self.derived;
        let b = &d.bufs;
        let m = &d.multi;
        let f = std::mem::size_of::<f64>();
        (b.tau.capacity() + b.f.capacity() + b.m.capacity() + b.rate.capacity() + b.off.capacity())
            * f
            + b.new.capacity()
            + b.seq.capacity() * 8
            + b.pending.capacity() * std::mem::size_of::<usize>()
            + m.rows.capacity() * std::mem::size_of::<RowRec>()
            + (m.order.capacity()
                + m.starts.capacity()
                + m.fill.capacity()
                + m.mark.capacity()
                + m.slot.capacity())
                * 4
            + m.touched.capacity() * std::mem::size_of::<Touch>()
            + m.xg.iter().map(|v| v.capacity() * f).sum::<usize>()
    }

    /// Whether [`Windows::push_plain`] could take a row now: a plain core
    /// (or one not yet worked out) with the plain loop on. A caller asks this
    /// before it builds the rows.
    #[inline]
    pub(crate) fn plain_may_run(&self) -> bool {
        !self.derived.ready || !self.derived.off
    }

    /// Take as many of `rows`, from its start, as the plain loop runs: each a
    /// step forward in time within the cap and its session, in a core whose
    /// kernels look back, every input of it usable. Returns how many; the
    /// rest is for [`Windows::push`], and so is the first row of a stream, of
    /// a group, and any row after a cut. The state after the rows is the
    /// state `push` would have left.
    pub(crate) fn push_plain(&mut self, rows: &PlainRows<'_>) -> usize {
        self.scratch();
        if self.derived.off {
            return 0;
        }
        let several = self.groups.len() > 1 && rows.keys.is_some();
        let mut done = 0;
        while rows.start + done < rows.clock.len() {
            let got = if several {
                self.plain_pass(rows, rows.start + done)
            } else {
                self.plain_block(rows, rows.start + done)
            };
            done += got;
            if got < if several { PASS } else { BLOCK } {
                break;
            }
        }
        self.derived.taken += done as u64;
        done
    }

    /// One block of up to [`BLOCK`] plain rows from `at`, in the one group.
    fn plain_block(&mut self, rows: &PlainRows<'_>, at: usize) -> usize {
        if self.groups.len() != 1 || rows.group != 0 {
            return 0;
        }
        // The group is the first row's: a later block begins where its key may
        // be another's.
        if rows.keys.is_some_and(|k| k.code(at) != k.code(rows.start)) {
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
                || !self.queues_at_rest(g)
            {
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
        let mut news = 0usize;
        for j in 0..n_rows {
            let now = rows.clock[at + j];
            // A step forward within the cap, or no step: the same stamp.
            let step_ok = step_ok(now, last, cap);
            let session = rows.session.map(|s| s[at + j]);
            let session_ok = match (session, last_session) {
                (Some(s), Some(q)) => s == q,
                _ => true,
            };
            let keys_ok = rows.keys.is_none_or(|k| k.code(at + j) == k.code(at));
            if !(step_ok && session_ok && keys_ok) {
                break;
            }
            let new = now != last;
            news += usize::from(new);
            bufs.new.push(new);
            bufs.tau.push(elapsed(now, origin));
            bufs.off.push(offset_of(now));
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
            n_forward,
            ..
        } = &mut *self;
        let seq0 = *next_seq;
        bufs.seq.clear();
        bufs.seq.extend((0..len).map(|r| seq0 + r as u64));
        held_values.extend(std::iter::repeat_n(f64::NAN, len * *n_outputs));
        for r in 0..len {
            held_meta.push_back(Meta {
                open: *n_forward + if r >= final_from { n_wait } else { 0 },
                accept: rows.accepts.iter().all(|c| usable(c[at + r])),
                drop: false,
            });
        }
        let mut parts = Parts {
            kernels: kernels.as_slice(),
            ops: ops.as_slice(),
            members: members.as_slice(),
            held_values,
            held_meta,
            held_first: *held_first,
            n_out: *n_outputs,
            scratch_row,
            scratch_sum,
            n_wait,
            n_forward: *n_forward,
        };
        let shape = Shape {
            len,
            news,
            final_from,
            g_tau,
            g_off,
            form,
            last_now: rows.clock[at + len - 1],
            last_session: rows.session.map(|s| s[at + len - 1]),
        };
        let cols = Cols {
            whole: rows.values,
            from: at,
            len,
        };
        run_group(&mut parts, &mut groups[0], cols, &mut bufs, &shape);

        // The stream where the last row leaves it.
        self.shared.take_plain_run(
            Some(shape.last_now),
            if grouped { None } else { shape.last_session },
        );
        self.next_seq += len as u64;
        self.derived.bufs = bufs;
        self.link_back(0);
        self.advance_ready();
        len
    }

    /// Whether every fast kernel of the group is at the group's own last row:
    /// what a read at its time reads is the sums as they stand.
    #[inline]
    fn queues_at_rest(&self, g: &Group) -> bool {
        self.kernels
            .iter()
            .zip(&g.queues)
            .zip(self.members.iter())
            .all(|((k, q), m)| {
                !fast_kernel(k, m, &self.ops)
                    || (q.total_at == g.tau && q.total.len() == q.sum_w() * q.n)
            })
    }

    /// A group of the pass's first row of it, if it can continue a plain run:
    /// what `plain_block` asks of its one group, but for the stream's clock,
    /// which stands wherever the stream's last row left it.
    fn touch(&self, gi: usize, n_wait: u32) -> Option<Touch> {
        let g = &self.groups[gi];
        let (prev, origin) = (g.stamp?, g.origin?);
        if g.restart
            || !g.linked
            || g.last_raw != Some(prev)
            || g.clock.last_clock() != Some(prev)
            || !self.queues_at_rest(g)
        {
            return None;
        }
        Some(Touch {
            gi,
            last: prev,
            last_session: g.clock.prev_session(),
            origin,
            count: 0,
            last_row: 0,
            // A group whose pending rows a later stamp has resolved has none.
            waiting: n_wait > 0 && !g.pending.is_empty(),
        })
    }

    /// One pass of up to [`PASS`] plain rows from `at`, in any of the core's
    /// groups: found in row order, then run a group at a time. A group's rows
    /// are its own whatever the other groups' rows between them do, so each
    /// group is taken over all its rows of the pass, and the stream's end is
    /// put as the rows in order would have put it: the groups' places in the
    /// silent list, the windows the new stamps resolve, and the groups whose
    /// silence the last row's clock shows.
    #[allow(clippy::too_many_lines)]
    fn plain_pass(&mut self, rows: &PlainRows<'_>, at: usize) -> usize {
        let Some(keys) = rows.keys else { return 0 };
        if !self.grouped {
            return 0;
        }
        let Some(first_stream) = self.shared.last_clock() else {
            return 0;
        };
        let n_rows = (rows.clock.len() - at).min(PASS);
        let cap = self.clock_cfg.gap_cap;
        let n_wait = self.derived.n_wait;
        let mut m = std::mem::take(&mut self.derived.multi);
        m.begin(self.groups.len());

        // The rows: one pass over the clock, the sessions and the keys.
        let mut stream = first_stream;
        for j in 0..n_rows {
            let pos = at + j;
            let Some(gi) = keys.known(pos) else { break };
            let now = rows.clock[pos];
            // The stream steps forward or stays: within the cap it is by the
            // group's own step below, the group's last row being at or before
            // the stream's.
            if !not_before(now, stream) {
                break;
            }
            let slot = if m.mark[gi] == m.epoch {
                m.slot[gi] as usize
            } else {
                let Some(t) = self.touch(gi, n_wait) else {
                    break;
                };
                m.mark[gi] = m.epoch;
                m.slot[gi] = m.touched.len() as u32;
                m.touched.push(t);
                m.touched.len() - 1
            };
            let t = &mut m.touched[slot];
            let session = rows.session.map(|s| s[pos]);
            let new = now != t.last;
            if !step_ok(now, t.last, cap)
                || matches!((session, t.last_session), (Some(s), Some(q)) if s != q)
                // A group whose pending rows a later stamp resolved does not
                // begin on a tie of its own stamp.
                || (t.count == 0 && !new && n_wait > 0 && !t.waiting)
            {
                break;
            }
            m.rows.push(RowRec {
                tau: elapsed(now, t.origin),
                off: offset_of(now),
                slot: slot as u32,
                new,
            });
            t.last = now;
            t.last_session = session;
            t.count += 1;
            t.last_row = j as u32;
            stream = now;
        }
        // A group met by the row that stopped the pass has none in it.
        while m.touched.last().is_some_and(|t| t.count == 0) {
            let t = m.touched.pop().expect("one");
            m.mark[t.gi] = 0;
        }
        let len = m.rows.len();
        if len == 0 {
            self.derived.multi = m;
            return 0;
        }

        // The groups whose rows were pending at the stream's stamp and are
        // not in the pass: a later stamp resolves them, where they stand.
        let last_now = rows.clock[at + len - 1];
        if n_wait > 0 && is_after(last_now, first_stream) {
            let mut at_tail = self.tail;
            while let Some(gi) = at_tail {
                if self.groups[gi].pending.is_empty() {
                    break;
                }
                at_tail = self.groups[gi].prev_link;
                if m.mark[gi] != m.epoch {
                    self.flush_group(gi);
                }
            }
        }

        // The pass's rows by group.
        let touched = m.touched.len();
        m.starts.clear();
        m.starts.resize(touched + 1, 0);
        for r in &m.rows {
            m.starts[r.slot as usize + 1] += 1;
        }
        for k in 0..touched {
            m.starts[k + 1] += m.starts[k];
        }
        m.order.clear();
        m.order.resize(len, 0);
        m.fill.clear();
        m.fill.extend_from_slice(&m.starts);
        for (j, r) in m.rows.iter().enumerate() {
            let f = &mut m.fill[r.slot as usize];
            m.order[*f as usize] = j as u32;
            *f += 1;
        }

        let mut bufs = std::mem::take(&mut self.derived.bufs);
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
            n_forward,
            ..
        } = &mut *self;
        let seq0 = *next_seq;
        held_values.extend(std::iter::repeat_n(f64::NAN, len * *n_outputs));
        for j in 0..len {
            held_meta.push_back(Meta {
                open: *n_forward,
                accept: rows.accepts.iter().all(|c| usable(c[at + j])),
                drop: false,
            });
        }
        let mut parts = Parts {
            kernels: kernels.as_slice(),
            ops: ops.as_slice(),
            members: members.as_slice(),
            held_values,
            held_meta,
            held_first: *held_first,
            n_out: *n_outputs,
            scratch_row,
            scratch_sum,
            n_wait,
            n_forward: *n_forward,
        };
        let n_inputs = rows.values.len();
        let mut xg = std::mem::take(&mut m.xg);
        xg.resize_with(n_inputs, Vec::new);
        for t_slot in 0..touched {
            let (lo, hi) = (m.starts[t_slot] as usize, m.starts[t_slot + 1] as usize);
            let g = &mut groups[m.touched[t_slot].gi];
            // A group's rows in segments, as the one-group loop takes its blocks:
            // its buffers stay a block's.
            for idx in m.order[lo..hi].chunks(BLOCK) {
                let len_g = idx.len();
                bufs.tau.clear();
                bufs.new.clear();
                bufs.off.clear();
                bufs.seq.clear();
                bufs.tau.extend(idx.iter().map(|&j| m.rows[j as usize].tau));
                bufs.new.extend(idx.iter().map(|&j| m.rows[j as usize].new));
                bufs.off.extend(idx.iter().map(|&j| m.rows[j as usize].off));
                bufs.seq.extend(idx.iter().map(|&j| seq0 + u64::from(j)));
                let news = bufs.new.iter().filter(|&&n| n).count();
                for (i, col) in xg.iter_mut().enumerate() {
                    col.clear();
                    col.extend(idx.iter().map(|&j| rows.values[i][at + j as usize]));
                }
                let final_from = if news == 0 {
                    0
                } else {
                    bufs.new.iter().rposition(|&n| n).unwrap_or(0)
                };
                // The rows of the stamp the group's run ends in wait for the next.
                if n_wait > 0 {
                    for r in final_from..len_g {
                        parts.held_meta[(bufs.seq[r] - parts.held_first) as usize].open += n_wait;
                    }
                }
                let last = idx[len_g - 1] as usize;
                let shape = Shape {
                    len: len_g,
                    news,
                    final_from,
                    g_tau: g.tau,
                    g_off: g.off,
                    form: g.form,
                    last_now: rows.clock[at + last],
                    last_session: rows.session.map(|s| s[at + last]),
                };
                let cols = Cols {
                    whole: &xg,
                    from: 0,
                    len: len_g,
                };
                run_group(&mut parts, g, cols, &mut bufs, &shape);
            }
        }
        m.xg = xg;

        // The stream, and the groups' ends: the rows' last clock, the groups
        // that stamp left behind resolved, the silent list as the rows in order
        // leave it, the groups the clock shows silent.
        self.shared.take_plain_run(Some(last_now), None);
        self.next_seq += len as u64;
        self.derived.bufs = bufs;
        if n_wait > 0 {
            for t in &m.touched {
                if is_after(
                    last_now,
                    self.groups[t.gi].stamp.expect("a group with a row"),
                ) {
                    self.flush_group(t.gi);
                }
            }
        }
        // The groups' places in the silent list: each group goes to the tail
        // at its last row, in the order of the rows.
        for (j, r) in m.rows.iter().enumerate() {
            let t = &m.touched[r.slot as usize];
            if t.last_row as usize == j {
                self.link_back(t.gi);
            }
        }
        self.end_silent(last_now);
        self.advance_ready();
        self.derived.multi = m;
        len
    }
}

/// The raw clock a window's edge is decided from.
#[inline]
fn offset_of(now: ClockValue) -> i64 {
    match now {
        ClockValue::Ns(c) | ClockValue::I64(c) => c,
        ClockValue::F64(c) => c.to_bits() as i64,
    }
}

/// One group's run of rows, in their own order: the windows the rows resolve
/// are written, every kernel is taken over the whole run, and the group is left
/// as `push` would leave it. The caller has made the rows' places in the held
/// values and decided that each is a plain step forward in the group.
#[allow(clippy::too_many_lines)]
fn run_group(p: &mut Parts<'_>, g: &mut Group, cols: Cols<'_>, bufs: &mut Bufs, sh: &Shape) {
    let Shape {
        len,
        news,
        final_from,
        g_tau,
        g_off,
        form,
        last_now,
        last_session,
    } = *sh;
    let n_out = p.n_out;
    let n_wait = p.n_wait;
    let held_first = p.held_first;
    // The rows pending from before, as held indices; a new stamp's read goes to
    // them. A kernel without a wait has none.
    let mut pending = std::mem::take(&mut bufs.pending);
    pending.clear();
    pending.extend(
        g.pending
            .iter()
            .map(|&s| usize::try_from(s - held_first).expect("a pending row is held")),
    );
    if news > 0 {
        for &q in &pending {
            p.held_meta[q].open -= 1;
        }
    }
    // A kernel with a window, or a quantile, is the queue's: the same reads,
    // row values and pushes `push` makes, a row at a time, without what
    // `push` decides around them. Each kernel's queue is its own, so a
    // kernel is taken over the whole block before the next.
    for (ki, k) in p.kernels.iter().enumerate() {
        let m = &p.members[ki];
        if k.direction == Direction::Forward || fast_kernel(k, m, p.ops) {
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
            let seq = bufs.seq[r];
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
                        p.ops,
                        m,
                        q,
                        p.scratch_sum,
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
                let theirs = (from..r).map(|q| bufs.seq[q]);
                for waiting in earlier.iter().copied().chain(theirs) {
                    let held = Held {
                        first: held_first,
                        n_outputs: n_out,
                        values: &mut *p.held_values,
                        meta: &mut *p.held_meta,
                    };
                    write_backward(p.ops, m, waiting, &g.stamp_values[ki], complete, held);
                }
                began = r as isize;
            }
            if new && !near {
                by_width!(
                    wide,
                    read_backward(
                        k,
                        p.ops,
                        m,
                        q,
                        p.scratch_sum,
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
                let v = cols.x(p.ops[o].input)[r];
                let v = if usable(v) { v } else { f64::NAN };
                let start = if g.last_valued[o].is_nan() {
                    k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w)
                } else {
                    g.last_valued[o]
                };
                let stat = p.ops[o].stat;
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
                p.scratch_row[b] = start;
                p.scratch_row[b + 1] = v;
                acc.write(&mut p.scratch_row[b + 2..], wide);
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
                &p.scratch_row[..rw * m.len()],
                p.scratch_sum,
            );
            if !near {
                let complete = k.window_size.is_none_or(|w| tau >= w);
                let held = Held {
                    first: held_first,
                    n_outputs: n_out,
                    values: &mut *p.held_values,
                    meta: &mut *p.held_meta,
                };
                write_backward(p.ops, m, seq, &g.stamp_values[ki], complete, held);
            }
        }
    }
    // A forward kernel is the queue's too: each row closes the windows it
    // reaches past and joins the queue as `push` does it, a row at a time, a
    // kernel over the whole block. The rows wait first, all of them: no row
    // closes a window of its own or a later row's (a window of a width above
    // 0 holds the rows after its row, and a row at the stamp is not past it),
    // and the kernels' queues share nothing.
    if p.n_forward > 0 {
        for r in 0..len {
            g.waiting.push_back(Wait {
                seq: bufs.seq[r],
                tau: bufs.tau[r],
                off: bufs.off[r],
            });
        }
        let mut held = Held {
            first: held_first,
            n_outputs: n_out,
            values: &mut *p.held_values,
            meta: &mut *p.held_meta,
        };
        for (ki, k) in p.kernels.iter().enumerate() {
            if k.direction != Direction::Forward {
                continue;
            }
            for r in 0..len {
                forward_step(
                    k,
                    p.ops,
                    &p.members[ki],
                    ki,
                    g,
                    p.scratch_row,
                    p.scratch_sum,
                    FwdRow {
                        seq: bufs.seq[r],
                        tau: bufs.tau[r],
                        off: bufs.off[r],
                        form,
                    },
                    |input| cols.x(input)[r],
                    &mut held,
                );
            }
        }
        retire(p.kernels, g);
    }
    let out = p.held_values.make_contiguous();
    let rows_at = Rows {
        held_first,
        seq: &bufs.seq,
        pending: &pending,
    };

    for (ki, k) in p.kernels.iter().enumerate() {
        if !fast_kernel(k, &p.members[ki], p.ops) {
            continue;
        }
        let q = &mut g.queues[ki];
        let (wide, sw) = (q.wide(), q.sum_w());
        let near = k.closed.near(Direction::Backward);
        let has_rate = p.members[ki].iter().any(|&o| p.ops[o].stat == Stat::Rate);
        let needs_mass = p.members[ki].iter().any(|&o| p.ops[o].stat.weighs_held());
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
        let ties = news < len;
        for (j, &o) in p.members[ki].iter().enumerate() {
            let op = &p.ops[o];
            let var = wide && q.var[j];
            let x = cols.x(op.input);
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

    // The group where the last row leaves it.
    g.tau = bufs.tau[len - 1];
    (g.off, g.form) = match last_now {
        ClockValue::Ns(c) => (c, OffForm::Ns),
        ClockValue::I64(c) => (c, OffForm::Int),
        ClockValue::F64(c) => (c.to_bits() as i64, OffForm::Bits),
    };
    g.stamp = Some(last_now);
    g.last_raw = Some(last_now);
    if n_wait > 0 {
        if news > 0 {
            g.pending.clear();
        }
        g.pending.extend((final_from..len).map(|r| bufs.seq[r]));
    }
    g.clock.take_plain_run(Some(last_now), last_session);
    bufs.pending = pending;
}
