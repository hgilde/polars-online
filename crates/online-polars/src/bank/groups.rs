//! A chunk's groups of one spec, as the bank partitions the chunk by group
//! key: one flat list of the rows, grouped, and where each group's rows end,
//! with the keys as the chunk's own values -- not a `Vec` of row numbers and
//! a `String` for each group of each chunk. A bank of many small groups made
//! two allocations for each group of each chunk here, and the key's text was
//! formatted and hashed for a lookup in the streams' map that could as well
//! read it from a stack buffer.

use std::hash::Hash;

use equivalent::Equivalent;
use polars::prelude::*;
use polars_utils::aliases::PlHashMap;

use super::{GroupKey, session_hash};
use crate::arrow::{ArrowChunk, ArrowCol};
use crate::spec::Spec;

/// One group's key as the chunk holds it: an integer's value, or the text
/// borrowed from the chunk's column. Its text -- an integer's decimal -- is
/// formed only for a lookup (`with_ref`, onto the stack) or a new group
/// (`to_key`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum RawKey<'c> {
    Null,
    Signed(i64),
    Unsigned(u64),
    Text(&'c str),
}

/// A [`GroupKey`] by reference: hashes as one and compares equal to the one
/// with the same text, so a map keyed by `GroupKey` is looked up without
/// building the owned key.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct KeyRef<'a>(pub Option<&'a str>);

impl Equivalent<GroupKey> for KeyRef<'_> {
    fn equivalent(&self, key: &GroupKey) -> bool {
        self.0 == key.as_str()
    }
}

/// An integer's decimal digits, written at the end of `buf`.
fn decimal(mut v: u64, negative: bool, buf: &mut [u8; 20]) -> &str {
    let mut at = buf.len();
    loop {
        at -= 1;
        buf[at] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    if negative {
        at -= 1;
        buf[at] = b'-';
    }
    std::str::from_utf8(&buf[at..]).expect("ASCII digits")
}

impl RawKey<'_> {
    /// `f` over the key by reference, an integer's text on the stack.
    pub(crate) fn with_ref<R>(self, f: impl FnOnce(KeyRef<'_>) -> R) -> R {
        let mut buf = [0u8; 20];
        match self {
            RawKey::Null => f(KeyRef(None)),
            RawKey::Text(s) => f(KeyRef(Some(s))),
            RawKey::Signed(v) => f(KeyRef(Some(decimal(v.unsigned_abs(), v < 0, &mut buf)))),
            RawKey::Unsigned(v) => f(KeyRef(Some(decimal(v, false, &mut buf)))),
        }
    }

    /// The owned key, for a group the bank has not seen, or a reader that
    /// wants a `GroupKey`.
    pub(crate) fn to_key(self) -> GroupKey {
        self.with_ref(|r| GroupKey(r.0.map(str::to_string)))
    }
}

/// The groups of one chunk and spec, in the order their rows first arrive:
/// every group's rows are in `rows`, one after another, each group's in
/// ascending row order, and `ends[g]` is where group `g`'s end.
pub(crate) struct Groups<'c> {
    keys: Vec<RawKey<'c>>,
    rows: Vec<usize>,
    ends: Vec<usize>,
}

impl<'c> Groups<'c> {
    /// Every row in one group.
    fn one(key: RawKey<'c>, n: usize) -> Self {
        Self {
            keys: vec![key],
            rows: (0..n).collect(),
            ends: vec![n],
        }
    }

    /// The rows' groups in two passes: each row's group, counted, then the
    /// rows placed. `of` gives a row's key and its identity (equal
    /// identities are one group); a key is kept for a group's first row.
    fn partition(n: usize, mut of: impl FnMut(usize) -> (RawKey<'c>, u64)) -> Self {
        let mut slot_of: PlHashMap<u64, usize> = PlHashMap::default();
        let mut keys: Vec<RawKey<'c>> = Vec::new();
        let mut ends: Vec<usize> = Vec::new();
        let mut slots: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let (key, id) = of(i);
            let slot = *slot_of.entry(id).or_insert_with(|| {
                keys.push(key);
                ends.push(0);
                keys.len() - 1
            });
            ends[slot] += 1;
            slots.push(slot);
        }
        Self::placed(keys, ends, slots)
    }

    /// An integer column's groups: `identity` maps a value to its 64-bit
    /// identity (sign extension is a bijection, so `as u64` serves every
    /// signed width), `key` to the key. A null is its own group.
    fn of_integers<T: Copy>(
        values: impl Iterator<Item = Option<T>>,
        n: usize,
        identity: impl Fn(T) -> u64,
        key: impl Fn(T) -> RawKey<'c>,
    ) -> Self {
        // A null's identity cannot be an integer's: every 64-bit value is
        // one, so the null is told apart by a flag taken as a slot of its own.
        let mut null_slot: Option<usize> = None;
        let mut slot_of: PlHashMap<u64, usize> = PlHashMap::default();
        let mut keys: Vec<RawKey<'c>> = Vec::new();
        let mut ends: Vec<usize> = Vec::new();
        let mut slots: Vec<usize> = Vec::with_capacity(n);
        for v in values {
            let slot = match v {
                None => *null_slot.get_or_insert_with(|| {
                    keys.push(RawKey::Null);
                    ends.push(0);
                    keys.len() - 1
                }),
                Some(v) => *slot_of.entry(identity(v)).or_insert_with(|| {
                    keys.push(key(v));
                    ends.push(0);
                    keys.len() - 1
                }),
            };
            ends[slot] += 1;
            slots.push(slot);
        }
        Self::placed(keys, ends, slots)
    }

    /// The groups once each row's group is known: `ends` the counts.
    fn placed(keys: Vec<RawKey<'c>>, mut ends: Vec<usize>, slots: Vec<usize>) -> Self {
        let mut at = 0;
        for e in &mut ends {
            let count = *e;
            *e = at;
            at += count;
        }
        let mut rows = vec![0usize; slots.len()];
        for (i, &slot) in slots.iter().enumerate() {
            rows[ends[slot]] = i;
            ends[slot] += 1;
        }
        Self { keys, rows, ends }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Group `g`'s rows, ascending.
    pub(crate) fn rows(&self, g: usize) -> &[usize] {
        let from = if g == 0 { 0 } else { self.ends[g - 1] };
        &self.rows[from..self.ends[g]]
    }

    pub(crate) fn key(&self, g: usize) -> RawKey<'c> {
        self.keys[g]
    }

    /// Each group's key and rows, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (RawKey<'c>, &[usize])> + '_ {
        (0..self.len()).map(|g| (self.keys[g], self.rows(g)))
    }

    /// The owned keys, for the readers that compare or keep them.
    pub(crate) fn owned_keys(&self) -> Vec<GroupKey> {
        self.keys.iter().map(|k| k.to_key()).collect()
    }

    /// Each group's owned key and its rows copied, for a reader that holds
    /// them as pairs (the formula targets' resolvers).
    pub(crate) fn pairs(&self) -> Vec<(GroupKey, Vec<usize>)> {
        self.iter().map(|(k, r)| (k.to_key(), r.to_vec())).collect()
    }

    /// The layout that puts each group's rows one after another, in
    /// first-seen order, or `None` when the chunk is already in that order
    /// (ungrouped, one group, or groups that arrive as blocks). A stream then
    /// reads its rows as one contiguous run instead of gathering every
    /// column at a stride, twice per row; with many groups interleaved row by
    /// row that stride is a cache line per value, and cost 14% of `process` on
    /// one thread and 38% on fourteen at k=20 (docs/records/PERFORMANCE-LOG.md
    /// P9). The gather is one pass per column, and runs per column in parallel.
    ///
    /// The flat list itself, not a copy of it.
    pub(crate) fn layout(&self) -> Option<&[usize]> {
        let mut base = 0;
        let blocked = self.iter().all(|(_, idx)| {
            // Indices are distinct and increasing, so first and last say
            // whether the group is one run, and `base` whether the runs are
            // in order.
            let run = idx
                .first()
                .is_none_or(|&f| f == base && idx[idx.len() - 1] == base + idx.len() - 1);
            base += idx.len();
            run
        });
        (!blocked).then_some(self.rows.as_slice())
    }
}

/// Row-index partition by group key, in row order.
///
/// Keyed on a 64-bit hash of the string value rather than on the string
/// itself, so the per-row cost is a hash rather than a `String` allocation and
/// two more clones (docs/records/PERFORMANCE-LOG.md P3). The key is read from
/// the chunk, and its `GroupKey` built only for a group the bank has not
/// seen -- it is still the *value* that is stored and serialized, so state
/// files are unaffected. A 64-bit collision would merge two groups; the same
/// 2^-64 exposure the session hash already documents.
pub(crate) fn group_indices<'c>(chunk: &'c ArrowChunk, spec: &Spec) -> PolarsResult<Groups<'c>> {
    let n = chunk.height();
    match &spec.group {
        None => Ok(Groups::one(RawKey::Text(""), n)),
        Some(g) => {
            // An integer key's text is its decimal, which is exactly what the
            // String cast would produce (`integer_group_keys_match_the_
            // string_cast` in tests/bank.rs pins the two paths to the same
            // keys and output), so the value itself is the bucket: no cast, no
            // hash, and no collision to document (docs/records/PERFORMANCE-LOG.md P11).
            // Which form a key arrives in is the chunk's choice, not this
            // function's (`crate::arrow`).
            match chunk.key(spec, "group", g)? {
                ArrowCol::U64(a) => Ok(Groups::of_integers(
                    a.iter().map(|v| v.copied()),
                    n,
                    |v| v,
                    RawKey::Unsigned,
                )),
                ArrowCol::I64(a) => Ok(Groups::of_integers(
                    a.iter().map(|v| v.copied()),
                    n,
                    |v| v as u64,
                    RawKey::Signed,
                )),
                ArrowCol::Str(a) => {
                    let mut values = a.iter();
                    Ok(Groups::partition(n, |_| {
                        let v = values.next().expect("one value a row");
                        // `session_hash`, not a copy of it (review 2026-09-12, D7).
                        (v.map_or(RawKey::Null, RawKey::Text), session_hash(v))
                    }))
                }
                // `ArrowChunk::key` never returns the numeric form, a clock
                // or a boolean.
                ArrowCol::F64(_) | ArrowCol::Num(_) | ArrowCol::Nanos(_) | ArrowCol::Bool(_) => {
                    unreachable!("a group key is never read as a number, a clock or a boolean")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars_utils::aliases::PlIndexMap;
    use proptest::prelude::*;

    /// The groups as the earlier `Vec` of pairs built them: one `Vec` of row
    /// numbers a key, the key's text from `to_string`.
    fn naive(ids: &[Option<i64>]) -> Vec<(Option<String>, Vec<usize>)> {
        let mut out: Vec<(Option<String>, Vec<usize>)> = Vec::new();
        for (i, v) in ids.iter().enumerate() {
            let key = v.map(|v| v.to_string());
            match out.iter_mut().find(|(k, _)| *k == key) {
                Some((_, rows)) => rows.push(i),
                None => out.push((key, vec![i])),
            }
        }
        out
    }

    #[test]
    fn an_integers_text_is_its_decimal() {
        for v in [0i64, 1, -1, 9, 10, -10, 12345, i64::MIN, i64::MAX] {
            RawKey::Signed(v).with_ref(|r| assert_eq!(r.0, Some(v.to_string().as_str())));
            assert_eq!(RawKey::Signed(v).to_key().0, Some(v.to_string()));
        }
        for v in [0u64, 7, u64::MAX, i64::MAX as u64 + 1] {
            RawKey::Unsigned(v).with_ref(|r| assert_eq!(r.0, Some(v.to_string().as_str())));
        }
        RawKey::Null.with_ref(|r| assert_eq!(r.0, None));
        assert_eq!(RawKey::Text("a b").to_key(), GroupKey(Some("a b".into())));
    }

    /// The streams' map is keyed by `GroupKey` and looked up by `KeyRef`: the
    /// two must hash alike, or a lookup misses every stream it holds.
    #[test]
    fn a_key_found_by_reference_is_the_one_the_map_holds() {
        let keys = ["", "0", "-5", "10", "é", "a\u{0}b", "<null>"];
        let mut map: PlIndexMap<GroupKey, usize> = PlIndexMap::default();
        for (i, k) in keys.iter().enumerate() {
            map.insert(GroupKey(Some((*k).to_string())), i);
        }
        map.insert(GroupKey(None), keys.len());
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(map.get_index_of(&KeyRef(Some(k))), Some(i));
            assert_eq!(RawKey::Text(k).with_ref(|r| map.get_index_of(&r)), Some(i));
        }
        assert_eq!(map.get_index_of(&KeyRef(None)), Some(keys.len()));
        assert_eq!(map.get_index_of(&KeyRef(Some("1"))), None);
        // An integer finds the key with its decimal's text.
        assert_eq!(
            RawKey::Signed(-5).with_ref(|r| map.get_index_of(&r)),
            Some(2)
        );
        assert_eq!(
            RawKey::Unsigned(10).with_ref(|r| map.get_index_of(&r)),
            Some(3)
        );
        assert_eq!(RawKey::Signed(11).with_ref(|r| map.get_index_of(&r)), None);
    }

    proptest! {
        /// The flat list is the earlier lists of rows, group by group, in
        /// first-seen order, with the keys' text the same.
        #[test]
        fn the_flat_groups_are_the_lists_they_replace(
            ids in prop::collection::vec(prop::option::weighted(0.8, -3i64..5), 0..60),
        ) {
            let want = naive(&ids);
            let got = Groups::of_integers(
                ids.iter().copied(),
                ids.len(),
                |v| v as u64,
                RawKey::Signed,
            );
            prop_assert_eq!(got.len(), want.len());
            for (g, (key, rows)) in want.iter().enumerate() {
                prop_assert_eq!(&got.key(g).to_key().0, key);
                prop_assert_eq!(got.rows(g), rows.as_slice());
            }
            // The layout is the rows of every group, one after another, or
            // none where the chunk is in that order already.
            let all: Vec<usize> = want.iter().flat_map(|(_, r)| r.iter().copied()).collect();
            let blocked = all.iter().copied().eq(0..ids.len());
            match got.layout() {
                None => prop_assert!(blocked),
                Some(l) => {
                    prop_assert!(!blocked);
                    prop_assert_eq!(l, all.as_slice());
                }
            }
            prop_assert_eq!(got.pairs().len(), want.len());
        }

        #[test]
        fn text_groups_are_the_same_partition(
            ids in prop::collection::vec(prop::option::weighted(0.8, 0u8..4), 0..60),
        ) {
            let texts: Vec<Option<String>> = ids.iter().map(|v| v.map(|v| format!("k{v}"))).collect();
            let got = Groups::partition(texts.len(), |i| match &texts[i] {
                Some(t) => (RawKey::Text(t), crate::bank::session_hash(Some(t))),
                None => (RawKey::Null, crate::bank::session_hash(None)),
            });
            let want = naive(&ids.iter().map(|v| v.map(i64::from)).collect::<Vec<_>>());
            prop_assert_eq!(got.len(), want.len());
            for (g, (_, rows)) in want.iter().enumerate() {
                prop_assert_eq!(got.rows(g), rows.as_slice());
                let first = rows[0];
                prop_assert_eq!(&got.key(g).to_key().0, &texts[first]);
            }
        }
    }

    #[test]
    fn a_chunk_of_no_rows_has_no_groups() {
        let g = Groups::of_integers(
            std::iter::empty::<Option<i64>>(),
            0,
            |v| v as u64,
            RawKey::Signed,
        );
        assert!(g.is_empty());
        assert!(g.layout().is_none());
    }
}
