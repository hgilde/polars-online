//! A window's snapshots laid out flat: the numbers of every snapshot but the
//! boundary in a few large blocks of words, and the boundary alone held as the
//! struct the model reads.
//!
//! **Why.** A snapshot as a struct of its own vectors cost several times the
//! numbers it holds: an `ewridge` snapshot of three features and one target
//! holds 47 numbers, 376 bytes, which is what `window_budget` counts, in 18
//! vectors, each allocated on its own and rounded up to the allocator's size
//! class, beside 424 bytes of the struct itself inside the ring and the ring's
//! spare room -- about 2.6 KB a snapshot in a running process, so a ring the
//! budget let reach 256 MiB held about 1.8 GB. Only the boundary, the oldest
//! snapshot, is ever read on a row; every other one waits to become it. So
//! every snapshot after the boundary is written as its numbers alone
//! ([`Pack`]), one after another into blocks of words, and its structure --
//! each vector's length, whether each optional part is there -- once for a
//! run of snapshots that share it ([`Shape`]). A snapshot then costs its
//! numbers, its place in the window (its clock, its stamp and its running
//! products, six words or so), and its share of the blocks' spare room, at
//! most about a sixteenth. A co-moment matrix is written as its lower
//! triangle, the only part any reader reads, so a wide snapshot holds about
//! half of what the budget counts of it (the budget counts the whole `k²`
//! storage, as it always has).
//!
//! **A model offers its snapshot as a view** ([`Offer`]) where copying it
//! first would cost more than writing it: `ew_cov`'s co-moments, `marginal`'s
//! pairs and the bank's residual spread are written straight from the
//! model's accumulators into the block, with no vector allocated, and only a
//! ring's first snapshot, its boundary, is formed as the struct.
//!
//! **Nothing a model reads changes.** The boundary is held decoded, as the
//! struct it was, so [`super::Snapshots::cut`] hands out a reference, as it
//! did; when the boundary leaves the window the next snapshot is decoded into
//! its place, reusing its vectors. A snapshot decodes to the bits it was
//! written with, and the budget counts each one as it always did, by the
//! snapshot's own [`super::Footprint`] (a view by the same count), taken
//! before it was written, so a budget thins or refuses at the same row. The
//! saved state is the ring written snapshot by snapshot, as before.

use std::cell::RefCell;
use std::collections::VecDeque;

use super::{Entry, Footprint, Product, Products};
use crate::Stamp;

/// Where [`Pack::pack`] writes a snapshot: its numbers as words, and its
/// structure -- lengths and which optional parts are there -- as dimensions.
#[derive(Debug, Default)]
pub struct Packer {
    words: Vec<u64>,
    dims: Vec<usize>,
}

impl Packer {
    /// One number, as its bits.
    #[inline]
    pub fn f64(&mut self, v: f64) {
        self.words.push(v.to_bits());
    }

    /// One count, or any other 64-bit word.
    #[inline]
    pub fn word(&mut self, w: u64) {
        self.words.push(w);
    }

    /// Numbers, as their bits.
    #[inline]
    pub fn f64s(&mut self, v: &[f64]) {
        self.words.extend(v.iter().map(|x| x.to_bits()));
    }

    /// A length, or which form a part takes.
    #[inline]
    pub fn dim(&mut self, d: usize) {
        self.dims.push(d);
    }
}

/// What [`Pack::unpack`] reads a snapshot from, in the order it was written.
#[derive(Debug)]
pub struct Unpacker<'a> {
    words: &'a [u64],
    dims: &'a [usize],
}

impl<'a> Unpacker<'a> {
    /// The next word.
    #[inline]
    pub fn word(&mut self) -> u64 {
        let (w, rest) = self
            .words
            .split_first()
            .expect("a snapshot reads the words it was written with");
        self.words = rest;
        *w
    }

    /// The next number.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        f64::from_bits(self.word())
    }

    /// The next `n` words.
    #[inline]
    pub fn words(&mut self, n: usize) -> &'a [u64] {
        let (w, rest) = self.words.split_at(n);
        self.words = rest;
        w
    }

    /// The next dimension.
    #[inline]
    pub fn dim(&mut self) -> usize {
        let (d, rest) = self
            .dims
            .split_first()
            .expect("a snapshot reads the dimensions it was written with");
        self.dims = rest;
        *d
    }
}

/// A snapshot written as words and read back to the same bits: what a window's
/// ring holds of every snapshot but its boundary (the module docs). `unpack`
/// reads exactly what `pack` wrote, in the same order; `unpack_into` reads it
/// into a value already there, reusing its vectors. A struct whose fields all
/// pack is written field by field with [`crate::pack_fields`].
pub trait Pack: Sized {
    fn pack(&self, out: &mut Packer);

    fn unpack(from: &mut Unpacker<'_>) -> Self;

    #[inline]
    fn unpack_into(&mut self, from: &mut Unpacker<'_>) {
        *self = Self::unpack(from);
    }

    /// Every item of a slice, after its length: one at a time, unless the
    /// item writes a slice faster whole.
    #[doc(hidden)]
    #[inline]
    fn pack_all(items: &[Self], out: &mut Packer) {
        for x in items {
            x.pack(out);
        }
    }

    /// `n` items into `items`, reusing the ones there.
    #[doc(hidden)]
    #[inline]
    fn unpack_all_into(items: &mut Vec<Self>, n: usize, from: &mut Unpacker<'_>) {
        items.truncate(n);
        for x in items.iter_mut() {
            x.unpack_into(from);
        }
        while items.len() < n {
            items.push(Self::unpack(from));
        }
    }
}

/// A snapshot as a model offers it to its ring: the snapshot itself, or a
/// view that writes the same words straight from the model's accumulators,
/// so the copy the ring keeps is the only one made. `bytes` is what the
/// budget counts of the snapshot it forms, and `write` writes what
/// [`Pack::pack`] of it would; `into_snapshot` forms it, for a ring that
/// holds none, whose first snapshot is its boundary. Debug builds check
/// every view's words, shape and bytes against the snapshot it forms.
pub trait Offer<S>: Clone {
    /// Whether this is a view, which debug builds check.
    const VIEW: bool = true;

    fn bytes(&self) -> usize;

    fn write(&self, out: &mut Packer);

    fn into_snapshot(self) -> S;
}

impl<S: Pack + super::Footprint + Clone> Offer<S> for S {
    const VIEW: bool = false;

    #[inline]
    fn bytes(&self) -> usize {
        self.footprint()
    }

    #[inline]
    fn write(&self, out: &mut Packer) {
        self.pack(out);
    }

    #[inline]
    fn into_snapshot(self) -> S {
        self
    }
}

/// [`Pack`] for a struct, field by field in the order named: every field must
/// be named, since the struct is taken apart and built again whole.
#[macro_export]
macro_rules! pack_fields {
    ($t:ident { $($f:ident),* $(,)? }) => {
        impl $crate::Pack for $t {
            #[inline]
            fn pack(&self, out: &mut $crate::Packer) {
                let $t { $($f),* } = self;
                $( $crate::Pack::pack($f, out); )*
            }

            #[inline]
            fn unpack(from: &mut $crate::Unpacker<'_>) -> Self {
                $( let $f = $crate::Pack::unpack(from); )*
                $t { $($f),* }
            }

            #[inline]
            fn unpack_into(&mut self, from: &mut $crate::Unpacker<'_>) {
                let $t { $($f),* } = self;
                $( $crate::Pack::unpack_into($f, from); )*
            }
        }
    };
}

impl Pack for f64 {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.f64(*self);
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        from.f64()
    }

    #[inline]
    fn pack_all(items: &[Self], out: &mut Packer) {
        out.f64s(items);
    }

    #[inline]
    fn unpack_all_into(items: &mut Vec<Self>, n: usize, from: &mut Unpacker<'_>) {
        let words = from.words(n);
        if items.len() == n {
            for (x, w) in items.iter_mut().zip(words) {
                *x = f64::from_bits(*w);
            }
        } else {
            items.clear();
            items.extend(words.iter().map(|w| f64::from_bits(*w)));
        }
    }
}

impl Pack for u64 {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.word(*self);
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        from.word()
    }

    #[inline]
    fn pack_all(items: &[Self], out: &mut Packer) {
        out.words.extend_from_slice(items);
    }

    #[inline]
    fn unpack_all_into(items: &mut Vec<Self>, n: usize, from: &mut Unpacker<'_>) {
        let words = from.words(n);
        if items.len() == n {
            items.copy_from_slice(words);
        } else {
            items.clear();
            items.extend_from_slice(words);
        }
    }
}

impl Pack for usize {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.word(*self as u64);
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        from.word() as usize
    }
}

impl<T: Pack> Pack for Vec<T> {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.dim(self.len());
        T::pack_all(self, out);
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        let n = from.dim();
        let mut v = Vec::with_capacity(n);
        T::unpack_all_into(&mut v, n, from);
        v
    }

    #[inline]
    fn unpack_into(&mut self, from: &mut Unpacker<'_>) {
        let n = from.dim();
        T::unpack_all_into(self, n, from);
    }
}

impl<T: Pack> Pack for Option<T> {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        match self {
            None => out.dim(0),
            Some(v) => {
                out.dim(1);
                v.pack(out);
            }
        }
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        (from.dim() == 1).then(|| T::unpack(from))
    }

    #[inline]
    fn unpack_into(&mut self, from: &mut Unpacker<'_>) {
        match (self.as_mut(), from.dim()) {
            (Some(v), 1) => v.unpack_into(from),
            (_, 1) => *self = Some(T::unpack(from)),
            _ => *self = None,
        }
    }
}

impl Pack for Stamp {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        match *self {
            Stamp::Ns(ns) => {
                out.dim(0);
                out.word(ns as u64);
                out.word((ns >> 64) as u64);
            }
            Stamp::Raw(raw, removed) => {
                out.dim(1);
                out.f64(raw);
                out.f64(removed);
            }
            Stamp::Int(whole, frac) => {
                out.dim(2);
                out.word(whole as u64);
                out.word((whole >> 64) as u64);
                out.f64(frac);
            }
        }
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        let i128_of = |from: &mut Unpacker<'_>| {
            let lo = from.word();
            let hi = from.word();
            (u128::from(lo) | (u128::from(hi) << 64)) as i128
        };
        match from.dim() {
            0 => Stamp::Ns(i128_of(from)),
            1 => Stamp::Raw(from.f64(), from.f64()),
            _ => Stamp::Int(i128_of(from), from.f64()),
        }
    }
}

impl Pack for Product {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.f64(self.0);
        out.word(self.1 as u64);
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        Product(from.f64(), from.word() as i64)
    }
}

impl Pack for Products {
    #[inline]
    fn pack(&self, out: &mut Packer) {
        out.dim(self.len());
        for p in &self.p[..self.len()] {
            p.pack(out);
        }
    }

    #[inline]
    fn unpack(from: &mut Unpacker<'_>) -> Self {
        let n = from.dim();
        let mut out = Products::ones(n);
        for p in &mut out.p[..n] {
            *p = Product::unpack(from);
        }
        out
    }
}

/// A snapshot with its place: the model's clock, the stamp, the running
/// products, then the snapshot.
#[cfg(test)]
fn pack_entry<S: Pack>(e: &Entry<S>, out: &mut Packer) {
    out.f64(e.clock);
    e.stamp.pack(out);
    e.decay.pack(out);
    e.snap.pack(out);
}

#[inline]
fn unpack_entry<S: Pack>(from: &mut Unpacker<'_>) -> Entry<S> {
    let clock = from.f64();
    let stamp = Stamp::unpack(from);
    let decay = Products::unpack(from);
    Entry {
        clock,
        stamp,
        snap: S::unpack(from),
        decay,
    }
}

#[inline]
fn unpack_entry_into<S: Pack>(e: &mut Entry<S>, from: &mut Unpacker<'_>) {
    e.clock = from.f64();
    e.stamp = Stamp::unpack(from);
    e.decay = Products::unpack(from);
    e.snap.unpack_into(from);
}

/// The structure a run of written snapshots shares, and what each one of
/// them holds: its words, and the bytes the budget counts of it.
#[derive(Debug, Clone, PartialEq)]
struct Shape {
    dims: Box<[usize]>,
    words: usize,
    footprint: usize,
}

/// Snapshots in a row that share one shape.
#[derive(Debug, Clone)]
struct Run {
    shape: Shape,
    count: usize,
}

thread_local! {
    /// Where a snapshot's dimensions are written before they are matched
    /// to its run's: one a thread, so a ring holds none.
    static SCRATCH: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// The fewest words a block is made with.
const MIN_BLOCK: usize = 32;

/// A window's snapshots, oldest first: the oldest held as the struct a model
/// reads, every other one as words (the module docs).
#[derive(Clone)]
pub(super) struct Ring<S> {
    /// The oldest snapshot, the window's boundary; `None` only in an empty
    /// ring.
    front: Option<Entry<S>>,
    /// The later snapshots' shapes, oldest first, a run of snapshots of one
    /// shape at a time.
    runs: VecDeque<Run>,
    /// The later snapshots.
    later: usize,
    /// The later snapshots' words, oldest first, one after another. A
    /// snapshot never spans two blocks, and a block is never grown: one that
    /// lacks the room for the next is followed by a new one.
    blocks: VecDeque<Vec<u64>>,
    /// The words of the first block already read back.
    head: usize,
    /// The words the later snapshots hold, which sizes a new block.
    words: usize,
}

/// The later snapshots' shapes and words, oldest first.
fn records<'a>(
    runs: &'a VecDeque<Run>,
    blocks: &'a VecDeque<Vec<u64>>,
    head: usize,
) -> impl Iterator<Item = (&'a Shape, &'a [u64])> + 'a {
    let (mut b, mut at) = (0, head);
    runs.iter()
        .flat_map(|run| std::iter::repeat_n(&run.shape, run.count))
        .map(move |shape| {
            while at == blocks[b].len() {
                b += 1;
                at = 0;
            }
            let rec = &blocks[b][at..at + shape.words];
            at += shape.words;
            (shape, rec)
        })
}

impl<S> Ring<S> {
    pub(super) fn new() -> Self {
        Self {
            front: None,
            runs: VecDeque::new(),
            later: 0,
            blocks: VecDeque::new(),
            head: 0,
            words: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        usize::from(self.front.is_some()) + self.later
    }

    pub(super) fn is_empty(&self) -> bool {
        self.front.is_none()
    }

    /// The oldest snapshot.
    pub(super) fn front(&self) -> Option<&Entry<S>> {
        self.front.as_ref()
    }

    /// A newest block with room for `n` more words: the one there, or a
    /// new one after it.
    fn room(&mut self, n: usize) {
        let fits = self
            .blocks
            .back()
            .is_some_and(|b| b.capacity() - b.len() >= n);
        if !fits {
            let cap = (4 * n).max(self.words / 16).max(MIN_BLOCK);
            self.blocks.push_back(Vec::with_capacity(cap));
        }
    }

    /// Copy one written snapshot, of `dims` and counted at `footprint`, to
    /// the end of the blocks.
    fn put(&mut self, dims: &[usize], footprint: usize, rec: &[u64]) {
        let n = rec.len();
        self.room(n);
        if let Some(b) = self.blocks.back_mut() {
            b.extend_from_slice(rec);
        }
        self.record(dims, footprint, n);
    }

    /// Count a snapshot of `n` words just written at the end of the blocks.
    fn record(&mut self, dims: &[usize], footprint: usize, n: usize) {
        self.words += n;
        self.later += 1;
        match self.runs.back_mut() {
            Some(run) if run.shape.footprint == footprint && *run.shape.dims == *dims => {
                run.count += 1;
            }
            _ => self.runs.push_back(Run {
                shape: Shape {
                    dims: dims.into(),
                    words: n,
                    footprint,
                },
                count: 1,
            }),
        }
    }
}

impl<S: Pack + Footprint + Clone> Ring<S> {
    /// Append a snapshot: the boundary itself in an empty ring, written as
    /// words otherwise.
    pub(super) fn push_back(&mut self, e: Entry<S>) {
        self.push_offer(e.clock, e.stamp, e.decay, e.snap);
    }

    /// Append the snapshot `snap` forms, at its place: the boundary itself
    /// in an empty ring, written as words otherwise.
    pub(super) fn push_offer<V: Offer<S>>(
        &mut self,
        clock: f64,
        stamp: Stamp,
        decay: Products,
        snap: V,
    ) {
        if self.front.is_none() {
            self.front = Some(Entry {
                clock,
                stamp,
                snap: snap.into_snapshot(),
                decay,
            });
            return;
        }
        // Written straight into the newest block where the newest run's
        // size fits, as it does while the shape holds; a snapshot that
        // turns out longer than the room is moved to a block of its own.
        let expect = self.runs.back().map_or(0, |r| r.shape.words);
        self.room(expect);
        let footprint = snap.bytes();
        SCRATCH.with_borrow_mut(|dims| {
            dims.clear();
            let block = self
                .blocks
                .back_mut()
                .map(std::mem::take)
                .unwrap_or_default();
            let (at, cap) = (block.len(), block.capacity());
            let mut p = Packer {
                words: block,
                dims: std::mem::take(dims),
            };
            p.f64(clock);
            stamp.pack(&mut p);
            decay.pack(&mut p);
            snap.write(&mut p);
            let Packer {
                words: mut block,
                dims: written,
            } = p;
            *dims = written;
            if V::VIEW && cfg!(debug_assertions) {
                // A view writes what the snapshot it forms writes, and is
                // counted as it is: checked wherever the tests run.
                let mut own = Packer::default();
                own.f64(clock);
                stamp.pack(&mut own);
                decay.pack(&mut own);
                let owned = snap.clone().into_snapshot();
                owned.pack(&mut own);
                assert_eq!(own.words, block[at..], "a view writes its snapshot's words");
                assert_eq!(own.dims, *dims, "a view writes its snapshot's shape");
                assert_eq!(
                    owned.footprint(),
                    footprint,
                    "a view counts its snapshot's bytes"
                );
            }
            let n = block.len() - at;
            if block.len() > cap {
                let rec = block.split_off(at);
                block.shrink_to(cap);
                if let Some(b) = self.blocks.back_mut() {
                    *b = block;
                }
                self.put(dims, footprint, &rec);
            } else {
                if let Some(b) = self.blocks.back_mut() {
                    *b = block;
                }
                self.record(dims, footprint, n);
            }
        });
    }

    /// Append a written snapshot, decoded if it is to be the boundary.
    fn push_record(&mut self, shape: &Shape, rec: &[u64]) {
        if self.front.is_none() {
            self.front = Some(unpack_entry(&mut Unpacker {
                words: rec,
                dims: &shape.dims,
            }));
        } else {
            self.put(&shape.dims, shape.footprint, rec);
        }
    }

    /// Drop the boundary, and decode the next snapshot into its place:
    /// the bytes the budget counted of the one dropped.
    pub(super) fn pop_front(&mut self) -> Option<usize> {
        let front = self.front.as_mut()?;
        let footprint = front.snap.footprint();
        let Some(run) = self.runs.front_mut() else {
            self.front = None;
            return Some(footprint);
        };
        while self.head == self.blocks[0].len() {
            self.blocks.pop_front();
            self.head = 0;
        }
        let n = run.shape.words;
        unpack_entry_into(
            front,
            &mut Unpacker {
                words: &self.blocks[0][self.head..self.head + n],
                dims: &run.shape.dims,
            },
        );
        run.count -= 1;
        if run.count == 0 {
            self.runs.pop_front();
        }
        self.later -= 1;
        self.head += n;
        self.words -= n;
        if self.later == 0 {
            // Keep the newest block, empty, for the next snapshot.
            while self.blocks.len() > 1 {
                self.blocks.pop_front();
            }
            self.blocks[0].clear();
            self.head = 0;
        }
        Some(footprint)
    }

    /// The newest snapshot's clock and stamp.
    pub(super) fn back_place(&self) -> Option<(f64, Stamp)> {
        match (self.runs.back(), self.blocks.back()) {
            (Some(run), Some(tail)) => {
                let mut from = Unpacker {
                    words: &tail[tail.len() - run.shape.words..],
                    dims: &run.shape.dims,
                };
                let clock = from.f64();
                Some((clock, Stamp::unpack(&mut from)))
            }
            _ => self.front.as_ref().map(|e| (e.clock, e.stamp)),
        }
    }

    /// The bytes the budget counts of the newest snapshot.
    pub(super) fn back_footprint(&self) -> Option<usize> {
        match self.runs.back() {
            Some(run) => Some(run.shape.footprint),
            None => self.front.as_ref().map(|e| e.snap.footprint()),
        }
    }

    /// The bytes the budget counts of every snapshot.
    pub(super) fn footprint(&self) -> usize {
        self.front.as_ref().map_or(0, |e| e.snap.footprint())
            + self
                .runs
                .iter()
                .map(|r| r.shape.footprint * r.count)
                .sum::<usize>()
    }

    /// Each snapshot's clock, stamp and counted bytes, oldest first, none
    /// of them decoded.
    pub(super) fn places(&self) -> impl Iterator<Item = (f64, Stamp, usize)> + '_ {
        let front = self
            .front
            .as_ref()
            .map(|e| (e.clock, e.stamp, e.snap.footprint()));
        front.into_iter().chain(
            records(&self.runs, &self.blocks, self.head).map(|(shape, rec)| {
                let mut from = Unpacker {
                    words: rec,
                    dims: &shape.dims,
                };
                let clock = from.f64();
                (clock, Stamp::unpack(&mut from), shape.footprint)
            }),
        )
    }

    /// Keep the newest snapshot and every second one before it.
    pub(super) fn keep_alternate(&mut self) {
        let n = self.len();
        let keep = |i: usize| (n - 1 - i).is_multiple_of(2);
        let old = std::mem::replace(self, Ring::new());
        if keep(0) {
            self.front = old.front;
        }
        for (i, (shape, rec)) in records(&old.runs, &old.blocks, old.head).enumerate() {
            if keep(i + 1) {
                self.push_record(shape, rec);
            }
        }
    }
}

impl<S: Pack + Footprint + Clone> Ring<S> {
    /// Every snapshot with its place, decoded, oldest first.
    pub(super) fn entries(&self) -> impl Iterator<Item = Entry<S>> + '_ {
        self.front
            .iter()
            .cloned()
            .chain(
                records(&self.runs, &self.blocks, self.head).map(|(shape, rec)| {
                    unpack_entry(&mut Unpacker {
                        words: rec,
                        dims: &shape.dims,
                    })
                }),
            )
    }
}

impl<S: Pack + Footprint + Clone + PartialEq> PartialEq for Ring<S> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.entries().zip(other.entries()).all(|(a, b)| a == b)
    }
}

impl<S: Pack + Footprint + Clone + std::fmt::Debug> std::fmt::Debug for Ring<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.entries()).finish()
    }
}

impl<S: Pack + Footprint + Clone + serde::Serialize> serde::Serialize for Ring<S> {
    /// The snapshots in order, each as `(clock, stamp, snapshot, products)`:
    /// the sequence the ring wrote when it held them as structs.
    fn serialize<Z: serde::Serializer>(&self, z: Z) -> Result<Z::Ok, Z::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = z.serialize_seq(Some(self.len()))?;
        if let Some(front) = &self.front {
            seq.serialize_element(front)?;
        }
        for (shape, rec) in records(&self.runs, &self.blocks, self.head) {
            let e: Entry<S> = unpack_entry(&mut Unpacker {
                words: rec,
                dims: &shape.dims,
            });
            seq.serialize_element(&e)?;
        }
        seq.end()
    }
}

impl<'de, S: Pack + Footprint + Clone + serde::Deserialize<'de>> serde::Deserialize<'de>
    for Ring<S>
{
    /// The sequence [`Ring`]'s `Serialize` writes, each snapshot written as
    /// words as it is read.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visit<S>(std::marker::PhantomData<S>);
        impl<'de, S: Pack + Footprint + Clone + serde::Deserialize<'de>> serde::de::Visitor<'de>
            for Visit<S>
        {
            type Value = Ring<S>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of snapshots")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Ring<S>, A::Error> {
                let mut ring = Ring::new();
                while let Some(e) = a.next_element::<Entry<S>>()? {
                    ring.push_back(e);
                }
                Ok(ring)
            }
        }
        d.deserialize_seq(Visit(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LowerTriangle;
    use serde::{Deserialize, Serialize};

    /// A snapshot of every form a part can take: numbers, counts, an
    /// optional part there or not, nested vectors of changing lengths, and a
    /// lower triangle.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Snap {
        w: f64,
        v: Vec<f64>,
        n: Option<Vec<u64>>,
        vv: Vec<Vec<f64>>,
        c: LowerTriangle,
        i: usize,
    }

    crate::pack_fields!(Snap { w, v, n, vv, c, i });

    impl Footprint for Snap {
        fn footprint(&self) -> usize {
            8 * (2 + self.v.len() + self.n.as_ref().map_or(0, Vec::len) + self.c.len())
                + self.vv.iter().map(|v| 8 * v.len()).sum::<usize>()
        }
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Bits that `==` would not tell apart, or would call unequal: both
    /// zeros, NaNs with payloads, the infinities, a subnormal.
    const ODD: [u64; 6] = [
        0,
        1 << 63,
        0x7ff8_0000_0000_0001,
        0xfff4_0000_0000_00ff,
        0x7ff0_0000_0000_0000,
        1,
    ];

    fn number(s: &mut u64) -> f64 {
        let r = lcg(s);
        if r.is_multiple_of(7) {
            f64::from_bits(ODD[(r / 7) as usize % ODD.len()])
        } else {
            (r as f64 / (1u64 << 31) as f64 - 0.5) * 1e3
        }
    }

    fn snap(s: &mut u64) -> Snap {
        // A shape that holds for a run of snapshots, then changes.
        let shape = lcg(s) % 9;
        let k = (shape % 4) as usize;
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                c[i * k + j] = number(s);
                c[j * k + i] = c[i * k + j];
            }
        }
        Snap {
            w: number(s),
            v: (0..shape as usize % 3).map(|_| number(s)).collect(),
            n: shape
                .is_multiple_of(2)
                .then(|| (0..k).map(|_| lcg(s) << 31).collect()),
            vv: (0..k)
                .map(|j| (0..j + 1).map(|_| number(s)).collect())
                .collect(),
            c: LowerTriangle::from(c),
            i: lcg(s) as usize,
        }
    }

    fn stamp(s: &mut u64, i: i128) -> Stamp {
        match lcg(s) % 3 {
            0 => Stamp::Ns(-(i << 70) + i),
            1 => Stamp::Raw(number(s), number(s)),
            _ => Stamp::Int(i << 66, number(s)),
        }
    }

    /// Two snapshots alike to the bit, through their written form.
    fn bits(e: &Entry<Snap>) -> Vec<u64> {
        let mut p = Packer::default();
        pack_entry(e, &mut p);
        p.words.extend(p.dims.iter().map(|&d| d as u64));
        p.words
    }

    fn assert_same(ring: &Ring<Snap>, want: &VecDeque<Entry<Snap>>, at: &str) {
        assert_eq!(ring.len(), want.len(), "{at}: the length");
        let got: Vec<_> = ring.entries().collect();
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(bits(g), bits(w), "{at}: snapshot {i}");
        }
        let fp: usize = want.iter().map(|e| e.snap.footprint()).sum();
        assert_eq!(ring.footprint(), fp, "{at}: the counted bytes");
        assert_eq!(
            ring.back_footprint(),
            want.back().map(|e| e.snap.footprint()),
            "{at}"
        );
        let place = |clock: f64, stamp: Stamp| {
            let mut p = Packer::default();
            p.f64(clock);
            stamp.pack(&mut p);
            (p.words, p.dims)
        };
        assert_eq!(
            ring.back_place().map(|(c, s)| place(c, s)),
            want.back().map(|e| place(e.clock, e.stamp)),
            "{at}: the newest place"
        );
        let places: Vec<_> = ring.places().map(|(c, s, b)| (place(c, s), b)).collect();
        let wanted: Vec<_> = want
            .iter()
            .map(|e| (place(e.clock, e.stamp), e.snap.footprint()))
            .collect();
        assert_eq!(places, wanted, "{at}: places");
    }

    /// The flat ring holds what a queue of the structs holds, to the bit,
    /// under every operation and every shape a snapshot takes: pushes, pops
    /// to empty and back, thinning, a clone, and a round trip through the
    /// state's encoding, which writes the same bytes as the queue.
    #[test]
    fn the_flat_ring_holds_what_a_queue_of_structs_holds_to_the_bit() {
        let mut s = 349u64;
        let mut ring: Ring<Snap> = Ring::new();
        let mut want: VecDeque<Entry<Snap>> = VecDeque::new();
        let mut thinned = 0;
        for step in 0..4_000i128 {
            let at = format!("step {step}");
            match lcg(&mut s) % 16 {
                0..=8 => {
                    let mut decay = Products::ones((lcg(&mut s) % 3) as usize);
                    for p in &mut decay.p {
                        *p = Product(number(&mut s), lcg(&mut s) as i64 - (1 << 30));
                    }
                    let e = Entry {
                        clock: number(&mut s),
                        stamp: stamp(&mut s, step),
                        snap: snap(&mut s),
                        decay,
                    };
                    ring.push_back(e.clone());
                    want.push_back(e);
                }
                9..=13 => {
                    let popped = ring.pop_front();
                    assert_eq!(
                        popped,
                        want.pop_front().map(|e| e.snap.footprint()),
                        "{at}: the bytes popped"
                    );
                }
                14 if want.len() > 1 => {
                    ring.keep_alternate();
                    let n = want.len();
                    want = std::mem::take(&mut want)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| (n - 1 - i).is_multiple_of(2))
                        .map(|(_, e)| e)
                        .collect();
                    thinned += 1;
                }
                _ => {
                    let bytes = rmp_serde::to_vec_named(&ring).unwrap();
                    assert_eq!(bytes, rmp_serde::to_vec_named(&want).unwrap(), "{at}");
                    ring = rmp_serde::from_slice(&bytes).unwrap();
                    let json = serde_json::to_string(&ring.clone()).unwrap();
                    assert_eq!(json, serde_json::to_string(&want).unwrap(), "{at}");
                }
            }
            assert_same(&ring, &want, &at);
            if step % 97 == 0 {
                assert_same(&ring.clone(), &want, &format!("{at}, cloned"));
            }
        }
        assert!(thinned > 50, "the stream thinned the ring {thinned} times");
    }

    /// A snapshot decoded into the boundary's place, over a snapshot of
    /// another shape, reads what one decoded fresh reads, to the bit, its
    /// lower triangle among it.
    #[test]
    fn a_snapshot_decoded_into_the_boundary_reads_the_same_as_one_decoded_fresh() {
        let mut s = 7u64;
        for _ in 0..500 {
            let (a, b) = (snap(&mut s), snap(&mut s));
            let mut p = Packer::default();
            b.pack(&mut p);
            let mut into = a.clone();
            into.unpack_into(&mut Unpacker {
                words: &p.words,
                dims: &p.dims,
            });
            let fresh = Snap::unpack(&mut Unpacker {
                words: &p.words,
                dims: &p.dims,
            });
            let all = |x: &Snap| {
                let mut p = Packer::default();
                x.pack(&mut p);
                (p.words, p.dims)
            };
            assert_eq!(all(&into), all(&b));
            assert_eq!(all(&fresh), all(&b));
            let k = b.c.len().isqrt();
            let lower = |c: &LowerTriangle| -> Vec<u64> {
                (0..k)
                    .flat_map(|i| (0..=i).map(move |j| (i, j)))
                    .map(|(i, j)| c.get(k, i, j).to_bits())
                    .collect()
            };
            assert_eq!(lower(&into.c), lower(&b.c), "the lower triangle");
            assert_eq!(lower(&fresh.c), lower(&b.c), "the lower triangle");
        }
    }

    /// A lower triangle decoded over one of any other side reads the one
    /// written, at its own side: a side whose square is twice the new side
    /// (4 for 8) among them.
    #[test]
    fn a_lower_triangle_decoded_over_another_side_reads_its_own() {
        let mut s = 236u64;
        let tri = |k: usize, s: &mut u64| {
            let mut c = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..=i {
                    c[i * k + j] = number(s);
                    c[j * k + i] = c[i * k + j];
                }
            }
            LowerTriangle::from(c)
        };
        for a in 0..10 {
            for b in 0..10 {
                let (mut into, want) = (tri(a, &mut s), tri(b, &mut s));
                let mut p = Packer::default();
                want.pack(&mut p);
                into.unpack_into(&mut Unpacker {
                    words: &p.words,
                    dims: &p.dims,
                });
                assert_eq!(into.len(), b * b, "side {a} read as {b}");
                for i in 0..b {
                    for j in 0..=i {
                        assert_eq!(
                            into.get(b, i, j).to_bits(),
                            want.get(b, i, j).to_bits(),
                            "side {a} read as {b}: ({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    /// A snapshot of one shape: `m` numbers, and an optional part of `m`
    /// counts.
    fn fixed(s: &mut u64, m: usize, step: i128) -> Entry<Snap> {
        Entry {
            clock: number(s),
            stamp: Stamp::Ns(step),
            snap: Snap {
                w: number(s),
                v: (0..m).map(|_| number(s)).collect(),
                n: Some((0..m).map(|_| lcg(s)).collect()),
                vv: vec![],
                c: LowerTriangle::from(vec![]),
                i: 3,
            },
            decay: Products::ones(0),
        }
    }

    /// Each block's capacity, oldest first.
    fn capacities(ring: &Ring<Snap>) -> Vec<usize> {
        ring.blocks.iter().map(Vec::capacity).collect()
    }

    /// What the layout keeps true whatever happens to the ring: the words
    /// it says the later snapshots hold are theirs, which sizes each new
    /// block; neighbouring runs differ in shape (a snapshot of the newest
    /// run's shape joins it), and an empty ring of later snapshots keeps one
    /// block at most, empty.
    fn assert_layout(ring: &Ring<Snap>, at: &str) {
        let held: usize = ring.runs.iter().map(|r| r.shape.words * r.count).sum();
        assert_eq!(ring.words, held, "{at}: the words held");
        let counted: usize = ring.runs.iter().map(|r| r.count).sum();
        assert_eq!(ring.later, counted, "{at}: the later snapshots");
        for pair in ring.runs.iter().collect::<Vec<_>>().windows(2) {
            assert_ne!(pair[0].shape, pair[1].shape, "{at}: two runs of a shape");
        }
        if ring.later == 0 {
            assert!(ring.blocks.len() <= 1, "{at}: {} blocks", ring.blocks.len());
            assert!(ring.blocks.iter().all(Vec::is_empty), "{at}: a block left");
        }
        let used: usize = ring.blocks.iter().map(Vec::len).sum();
        assert_eq!(used - ring.head, ring.words, "{at}: the words written");
    }

    /// A ring's blocks are never grown: a push leaves every block there as
    /// it was made and adds one at the end where the newest lacks the room;
    /// a pop takes blocks from the front alone; and the layout holds under
    /// every operation of the reference stream.
    #[test]
    fn a_ring_never_grows_a_block_and_keeps_its_layout() {
        let mut s = 353u64;
        let mut ring: Ring<Snap> = Ring::new();
        let mut want: VecDeque<Entry<Snap>> = VecDeque::new();
        let (mut grew, mut emptied) = (0, 0);
        for step in 0..4_000i128 {
            let at = format!("step {step}");
            let before = capacities(&ring);
            match lcg(&mut s) % 16 {
                0..=8 => {
                    let e = Entry {
                        clock: number(&mut s),
                        stamp: stamp(&mut s, step),
                        snap: snap(&mut s),
                        decay: Products::ones(0),
                    };
                    ring.push_back(e.clone());
                    want.push_back(e);
                    let after = capacities(&ring);
                    assert_eq!(after[..before.len()], before, "{at}: a block grown");
                    grew += usize::from(after.len() > before.len());
                }
                9..=14 => {
                    let had = ring.later;
                    ring.pop_front();
                    want.pop_front();
                    let after = capacities(&ring);
                    assert_eq!(after, before[before.len() - after.len()..], "{at}");
                    emptied += usize::from(had > 0 && ring.later == 0 && before.len() > 1);
                }
                _ if !want.is_empty() => {
                    ring.keep_alternate();
                    let n = want.len();
                    want = std::mem::take(&mut want)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| (n - 1 - i).is_multiple_of(2))
                        .map(|(_, e)| e)
                        .collect();
                }
                _ => {}
            }
            assert_layout(&ring, &at);
            assert_same(&ring, &want, &at);
        }
        assert!(grew > 20, "a push made a new block {grew} times");
        assert!(
            emptied > 5,
            "a ring of several blocks emptied {emptied} times"
        );
    }

    /// A ring of one shape keeps four of its snapshots to a block while it
    /// is small, and later a sixteenth of what it holds: about 16 ln N
    /// blocks for N snapshots, with spare room of about a sixteenth. Popped
    /// to its boundary, it keeps one block, empty, and reads what is pushed
    /// next.
    #[test]
    fn a_ring_s_blocks_hold_four_snapshots_then_a_sixteenth_of_the_ring() {
        let mut s = 548u64;
        let mut ring: Ring<Snap> = Ring::new();
        let mut want: VecDeque<Entry<Snap>> = VecDeque::new();
        for step in 0..5 {
            let e = fixed(&mut s, 40, step);
            ring.push_back(e.clone());
            want.push_back(e);
        }
        let n = ring.runs[0].shape.words;
        assert!(n > MIN_BLOCK, "a snapshot of {n} words");
        assert_eq!(
            ring.blocks.iter().filter(|b| !b.is_empty()).count(),
            1,
            "four snapshots of {n} words in one block: {:?}",
            capacities(&ring)
        );
        for step in 5..10_001 {
            let e = fixed(&mut s, 40, step);
            ring.push_back(e.clone());
            want.push_back(e);
        }
        assert_layout(&ring, "pushed");
        assert_same(&ring, &want, "pushed");
        let blocks = ring.blocks.len();
        let bound = 16.0 * (ring.words as f64 / (4 * n) as f64).ln() + 20.0;
        assert!(
            (blocks as f64) < bound,
            "{blocks} blocks for {} snapshots",
            ring.later
        );
        let spare: usize = ring.blocks.iter().map(|b| b.capacity() - b.len()).sum();
        assert!(
            spare <= ring.words / 16 + 4 * n + blocks * n,
            "{spare} words spare of {}",
            ring.words
        );
        while ring.later > 0 {
            ring.pop_front();
            want.pop_front();
        }
        assert_layout(&ring, "popped");
        assert_eq!(ring.blocks.len(), 1, "one block kept");
        for step in 0..3 {
            let e = fixed(&mut s, 40, step);
            ring.push_back(e.clone());
            want.push_back(e);
        }
        assert_layout(&ring, "pushed again");
        assert_same(&ring, &want, "pushed again");
    }

    /// Decoding the next snapshot into the boundary's place reuses its
    /// vectors, the one inside an optional part among them, where they have
    /// the room.
    #[test]
    fn the_boundary_is_decoded_into_its_own_vectors() {
        let mut s = 339u64;
        let mut ring: Ring<Snap> = Ring::new();
        for step in 0..3 {
            ring.push_back(fixed(&mut s, 6, step));
        }
        let ptrs = |ring: &Ring<Snap>| {
            let snap = &ring.front().unwrap().snap;
            (snap.v.as_ptr(), snap.n.as_ref().unwrap().as_ptr())
        };
        let before = ptrs(&ring);
        ring.pop_front();
        assert_eq!(ptrs(&ring), before, "the boundary's vectors reused");
    }

    /// Two rings are equal where they hold equal snapshots, as many: a
    /// ring and a longer one that starts with it are not. Its debug form
    /// lists its snapshots, and its encoding refuses what is no sequence by
    /// saying what it expected.
    #[test]
    fn a_ring_compares_prints_and_refuses_by_its_snapshots() {
        let mut s = 808u64;
        // Numbers `==` holds equal to themselves: no NaN.
        let fixed = |s: &mut u64, step| {
            let mut e = fixed(s, 3, step);
            e.clock = step as f64;
            e.snap.w = 0.5;
            e.snap.v = vec![1.0, -2.0, step as f64];
            e
        };
        let mut ring: Ring<Snap> = Ring::new();
        let mut want: Vec<Entry<Snap>> = vec![];
        for step in 0..6 {
            let e = fixed(&mut s, step);
            ring.push_back(e.clone());
            want.push(e);
        }
        assert!(ring == ring.clone());
        let mut longer = ring.clone();
        longer.push_back(fixed(&mut s, 6));
        assert!(ring != longer && longer != ring, "a longer ring");
        let mut other: Ring<Snap> = Ring::new();
        for (i, e) in want.iter().enumerate() {
            let mut e = e.clone();
            if i == 4 {
                e.snap.w += 1.0;
            }
            other.push_back(e);
        }
        assert!(ring != other, "a snapshot differs");
        assert_eq!(format!("{ring:?}"), format!("{want:?}"));
        let err = serde_json::from_str::<Ring<Snap>>("3").unwrap_err();
        assert!(err.to_string().contains("a sequence of snapshots"), "{err}");
    }
}
