//! A chunk's group keys and session hashes, read from the column's own
//! values (docs/PLAN.md task 431).
//!
//! A group is named by its key's text: the state file keeps the text, and so
//! does the core's index. Reading the text of every row's key -- a cast of the
//! column to String every chunk, then a hash of each string -- was a fifth of
//! a grouped row. The values themselves say the same: an integer is its own
//! key, a categorical column's key is its category's, a string is its text.
//! [`ChunkKeys`] gives each row a code, the index of its group where the core
//! has one, and, where it has none, a number among the keys the chunk adds,
//! whose groups are made at the first row that reaches them, as `push` always
//! made them: the groups stand in the order their keys are first seen, which
//! the outputs depend on. Two rows of a chunk have one code exactly when they
//! have one key.
//!
//! The session column's values are read the same way: [`session_hashes`] hashes
//! the text of each value as [`crate::bank::session_hash`] does (the hash is
//! saved in the state), from the integer's digits, the string, or the
//! category's string, never from a cast column.

use std::collections::HashMap;

use polars::prelude::*;
use polars_utils::aliases::PlFixedStateQuality;

use super::Windows;
use crate::arrow::key_text;
use crate::bank::{session_hash, session_hash_int};

/// A code at or past this is a key the chunk adds: `NEW + n` for its n-th.
const NEW: u32 = 1 << 31;

type Fixed<K, V> = HashMap<K, V, PlFixedStateQuality>;

/// A key no group of the core has yet.
#[derive(Debug, Clone, PartialEq)]
enum Key {
    Null,
    Int(i64),
    Text(String),
}

/// A key the chunk adds, and its group once a row has reached it.
#[derive(Debug)]
struct Fresh {
    key: Key,
    group: Option<usize>,
}

/// The persistent lookup from an integer key to its group's index, kept beside
/// the core's index of texts. Not state: a loaded core starts it empty, and an
/// integer not in it is looked up by its text. Two cores are equal whatever it
/// holds.
#[derive(Debug, Clone, Default)]
pub(super) struct IntGroups(Fixed<i64, u32>);

impl PartialEq for IntGroups {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The rows' group keys of one chunk, as codes ([module](self)).
#[derive(Debug, Default)]
pub(crate) struct ChunkKeys {
    /// Per row; empty when the chunk has no group column.
    codes: Vec<u32>,
    fresh: Vec<Fresh>,
    /// No group column: every key is the empty text.
    ungrouped: bool,
    one: Option<usize>,
}

impl Windows {
    /// The group of an integer key, as its text would find it.
    fn find_int(&mut self, v: i64) -> Option<usize> {
        if let Some(&g) = self.int_groups.0.get(&v) {
            return Some(g as usize);
        }
        let g = self.index.get(v.to_string().as_str()).copied()?;
        self.int_groups.0.insert(v, u32::try_from(g).ok()?);
        Some(g)
    }

    /// The group of an integer key, made if there is none.
    fn make_int(&mut self, v: i64) -> usize {
        let g = self.group(Some(&v.to_string()));
        if let Ok(g32) = u32::try_from(g) {
            self.int_groups.0.insert(v, g32);
        }
        g
    }

    /// Whether a code can name this core's groups: its index fits below
    /// [`NEW`].
    fn codes_fit(&self) -> bool {
        self.groups.len() < NEW as usize - 1
    }
}

impl ChunkKeys {
    /// A chunk with no group column.
    pub(crate) fn ungrouped() -> Self {
        Self {
            ungrouped: true,
            ..Self::default()
        }
    }

    /// The codes of a group column's rows. A dtype with no cheaper form is read
    /// as the text [`key_text`] gives.
    ///
    /// # Errors
    ///
    /// A column that cannot be read as text, as [`key_text`] refuses it.
    pub(crate) fn of(core: &mut Windows, s: &Series) -> PolarsResult<Self> {
        let mut b = Builder::new(core, s.len());
        match s.dtype() {
            DataType::Int64 => b.ints(s.i64()?.iter()),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => {
                let wide = s.cast(&DataType::Int64)?;
                b.ints(wide.i64()?.iter());
            }
            DataType::String => b.texts(s.str()?.iter()),
            DataType::Categorical(..) | DataType::Enum(..) => {
                if let Ok(ca) = s.cat8() {
                    let m = ca.get_mapping();
                    b.cats(ca.physical().iter().map(|c| c.map(u32::from)), |c| {
                        m.cat_to_str(c)
                    });
                } else if let Ok(ca) = s.cat16() {
                    let m = ca.get_mapping();
                    b.cats(ca.physical().iter().map(|c| c.map(u32::from)), |c| {
                        m.cat_to_str(c)
                    });
                } else {
                    let ca = s.cat32()?;
                    let m = ca.get_mapping();
                    b.cats(ca.physical().iter(), |c| m.cat_to_str(c));
                }
            }
            _ => {
                let text = key_text(s)?;
                b.texts(text.str()?.iter());
            }
        }
        Ok(b.done())
    }

    /// Whether the chunk has a group column.
    #[inline]
    pub(crate) fn is_grouped(&self) -> bool {
        !self.ungrouped
    }

    /// Row `i`'s code: two rows have one exactly when they have one key.
    #[inline]
    pub(crate) fn code(&self, i: usize) -> u32 {
        self.codes[i]
    }

    /// The group of row `i` if the core has made it.
    #[inline]
    pub(crate) fn known(&self, i: usize) -> Option<usize> {
        let c = self.codes[i];
        if c < NEW {
            Some(c as usize)
        } else {
            self.fresh[(c - NEW) as usize].group
        }
    }

    /// The group of row `i`: made, if the core has none yet.
    #[inline]
    pub(crate) fn group(&mut self, core: &mut Windows, i: usize) -> usize {
        if self.ungrouped {
            return *self.one.get_or_insert_with(|| core.group(Some("")));
        }
        let c = self.codes[i];
        if c < NEW {
            c as usize
        } else {
            self.made(core, (c - NEW) as usize)
        }
    }

    #[cold]
    #[inline(never)]
    fn made(&mut self, core: &mut Windows, n: usize) -> usize {
        let f = &mut self.fresh[n];
        if let Some(g) = f.group {
            return g;
        }
        let g = match &f.key {
            Key::Null => core.group(None),
            Key::Int(v) => core.make_int(*v),
            Key::Text(t) => core.group(Some(t)),
        };
        f.group = Some(g);
        g
    }
}

/// [`ChunkKeys::of`]'s pass over a column.
struct Builder<'a> {
    core: &'a mut Windows,
    codes: Vec<u32>,
    fresh: Vec<Fresh>,
    null: Option<u32>,
}

impl<'a> Builder<'a> {
    fn new(core: &'a mut Windows, n: usize) -> Self {
        Self {
            core,
            codes: Vec::with_capacity(n),
            fresh: Vec::new(),
            null: None,
        }
    }

    fn done(self) -> ChunkKeys {
        ChunkKeys {
            codes: self.codes,
            fresh: self.fresh,
            ungrouped: false,
            one: None,
        }
    }

    /// A key the chunk adds.
    fn add(&mut self, key: Key) -> u32 {
        let n = u32::try_from(self.fresh.len()).expect("a chunk's keys fit") + NEW;
        self.fresh.push(Fresh { key, group: None });
        n
    }

    fn null_code(&mut self) -> u32 {
        if let Some(c) = self.null {
            return c;
        }
        let c = match self.core.find_null() {
            Some(g) => g as u32,
            None => self.add(Key::Null),
        };
        self.null = Some(c);
        c
    }

    fn ints(&mut self, values: impl Iterator<Item = Option<i64>>) {
        let ok = self.core.codes_fit();
        let mut added: Fixed<i64, u32> = Fixed::default();
        let mut last: Option<(i64, u32)> = None;
        for v in values {
            let code = match v {
                None => self.null_code(),
                Some(v) => match last {
                    Some((l, c)) if l == v => c,
                    _ => {
                        let c = match self.core.int_groups.0.get(&v) {
                            Some(&g) => g,
                            None => self.int_slow(v, ok, &mut added),
                        };
                        last = Some((v, c));
                        c
                    }
                },
            };
            self.codes.push(code);
        }
    }

    #[cold]
    #[inline(never)]
    fn int_slow(&mut self, v: i64, ok: bool, added: &mut Fixed<i64, u32>) -> u32 {
        if let Some(&c) = added.get(&v) {
            return c;
        }
        let c = match self.core.find_int(v) {
            Some(g) if ok => g as u32,
            _ => self.add(Key::Int(v)),
        };
        added.insert(v, c);
        c
    }

    fn texts<'s>(&mut self, values: impl Iterator<Item = Option<&'s str>>) {
        let ok = self.core.codes_fit();
        let mut added: Fixed<&str, u32> = Fixed::default();
        let mut last: Option<(&str, u32)> = None;
        for v in values {
            let code = match v {
                None => self.null_code(),
                Some(t) => match last {
                    Some((l, c)) if l == t => c,
                    _ => {
                        let c = match self.core.find_text(t) {
                            Some(g) if ok => g as u32,
                            _ => match added.get(t) {
                                Some(&c) => c,
                                None => {
                                    let c = self.add(Key::Text(t.to_string()));
                                    added.insert(t, c);
                                    c
                                }
                            },
                        };
                        last = Some((t, c));
                        c
                    }
                },
            };
            self.codes.push(code);
        }
    }

    /// A categorical column: its categories' ids, and the text of an id.
    fn cats<'s>(
        &mut self,
        values: impl Iterator<Item = Option<u32>>,
        text_of: impl Fn(u32) -> Option<&'s str>,
    ) {
        let ok = self.core.codes_fit();
        let mut seen: Fixed<u32, u32> = Fixed::default();
        let mut last: Option<(u32, u32)> = None;
        for v in values {
            let code = match v {
                None => self.null_code(),
                Some(id) => match last {
                    Some((l, c)) if l == id => c,
                    _ => {
                        let c = if let Some(&c) = seen.get(&id) {
                            c
                        } else {
                            let t = text_of(id).unwrap_or("");
                            let c = match self.core.find_text(t) {
                                Some(g) if ok => g as u32,
                                _ => self.add(Key::Text(t.to_string())),
                            };
                            seen.insert(id, c);
                            c
                        };
                        last = Some((id, c));
                        c
                    }
                },
            };
            self.codes.push(code);
        }
    }
}

impl Windows {
    pub(crate) fn find_text(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    pub(crate) fn find_null(&self) -> Option<usize> {
        self.null_group
    }
}

/// The session column's values, each as the hash the core reads: that of the
/// value's text ([`session_hash`]), from the value itself where it is an
/// integer, a string or a category.
///
/// # Errors
///
/// A column that cannot be read as text, as [`key_text`] refuses it.
pub(crate) fn session_hashes(s: &Series) -> PolarsResult<Vec<u64>> {
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    match s.dtype() {
        DataType::Int64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => {
            let wide = s.cast(&DataType::Int64)?;
            let mut last: Option<(i64, u64)> = None;
            for v in wide.i64()?.iter() {
                out.push(match v {
                    None => session_hash(None),
                    Some(v) => match last {
                        Some((l, h)) if l == v => h,
                        _ => {
                            let h = session_hash_int(v);
                            last = Some((v, h));
                            h
                        }
                    },
                });
            }
        }
        DataType::String => text_hashes(s.str()?.iter(), &mut out),
        DataType::Categorical(..) | DataType::Enum(..) => {
            if let Ok(ca) = s.cat8() {
                let m = ca.get_mapping();
                cat_hashes(
                    ca.physical().iter().map(|c| c.map(u32::from)),
                    &mut out,
                    |c| m.cat_to_str(c),
                );
            } else if let Ok(ca) = s.cat16() {
                let m = ca.get_mapping();
                cat_hashes(
                    ca.physical().iter().map(|c| c.map(u32::from)),
                    &mut out,
                    |c| m.cat_to_str(c),
                );
            } else {
                let ca = s.cat32()?;
                let m = ca.get_mapping();
                cat_hashes(ca.physical().iter(), &mut out, |c| m.cat_to_str(c));
            }
        }
        _ => {
            let text = key_text(s)?;
            text_hashes(text.str()?.iter(), &mut out);
        }
    }
    Ok(out)
}

fn text_hashes<'s>(values: impl Iterator<Item = Option<&'s str>>, out: &mut Vec<u64>) {
    let mut last: Option<(&str, u64)> = None;
    for v in values {
        out.push(match v {
            None => session_hash(None),
            Some(t) => match last {
                Some((l, h)) if l == t => h,
                _ => {
                    let h = session_hash(Some(t));
                    last = Some((t, h));
                    h
                }
            },
        });
    }
}

fn cat_hashes<'s>(
    values: impl Iterator<Item = Option<u32>>,
    out: &mut Vec<u64>,
    text_of: impl Fn(u32) -> Option<&'s str>,
) {
    let mut seen: Fixed<u32, u64> = Fixed::default();
    let mut last: Option<(u32, u64)> = None;
    for v in values {
        out.push(match v {
            None => session_hash(None),
            Some(id) => match last {
                Some((l, h)) if l == id => h,
                _ => {
                    let h = *seen
                        .entry(id)
                        .or_insert_with(|| session_hash(Some(text_of(id).unwrap_or(""))));
                    last = Some((id, h));
                    h
                }
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the session column's hash was before it was read from the values:
    /// the hash of each row's text, as a cast to String gives it.
    fn by_text(s: &Series) -> Vec<u64> {
        key_text(s)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(session_hash)
            .collect()
    }

    #[test]
    fn an_integers_hash_is_its_texts() {
        let edges = [
            0i64,
            1,
            -1,
            9,
            10,
            -10,
            99,
            100,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            i64::MAX - 1,
            1_000_000_007,
            -1_000_000_007,
        ];
        for v in edges.into_iter().chain((-3000..3000).map(|k| k * 7919)) {
            assert_eq!(
                session_hash_int(v),
                session_hash(Some(&v.to_string())),
                "{v}"
            );
        }
    }

    #[test]
    fn every_dtype_hashes_as_its_text_does() {
        let ints: Vec<Option<i64>> = vec![
            Some(0),
            Some(-7),
            None,
            Some(300),
            Some(300),
            Some(i64::MAX),
        ];
        let small: Vec<Option<i64>> = vec![Some(0), Some(7), None, Some(100), Some(100), Some(1)];
        let words: Vec<Option<&str>> = vec![
            Some("a"),
            Some(""),
            None,
            Some("\0<null>"),
            Some("a"),
            Some("é"),
            Some("12"),
        ];
        let mut series: Vec<Series> = vec![
            Series::new("x".into(), &ints),
            Series::new("x".into(), &words),
            Series::new("x".into(), &[Some(1.5f64), None, Some(-0.0), Some(1e300)]),
            Series::new("x".into(), &[Some(true), None, Some(false)]),
            Series::new("x".into(), &[Some(19_000i32), None, Some(-5)])
                .cast(&DataType::Date)
                .unwrap(),
        ];
        for dt in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            series.push(Series::new("x".into(), &small).cast(&dt).unwrap());
        }
        series.push(
            Series::new("x".into(), &words)
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap(),
        );
        let names = ["a", "", "\0<null>", "é", "12"];
        let frozen = FrozenCategories::new(names).unwrap();
        series.push(
            Series::new("x".into(), &words)
                .cast(&DataType::from_frozen_categories(frozen))
                .unwrap(),
        );
        // A chunked column: two chunks of the same dtype.
        let mut two = Series::new("x".into(), &ints[..3]);
        two.append(&Series::new("x".into(), &ints[3..])).unwrap();
        series.push(two);
        for s in &series {
            assert_eq!(session_hashes(s).unwrap(), by_text(s), "{:?}", s.dtype());
        }
    }
}
