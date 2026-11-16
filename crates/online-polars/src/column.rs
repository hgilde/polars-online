//! One output column under construction, as the bank assembles its
//! struct columns: the values and their validity bits, handed on without a
//! second copy (docs/records/PERFORMANCE-LOG.md §13).
//!
//! The buffers are Arrow's already -- a values `Vec` and a validity `Bitmap`
//! are exactly what a primitive array is -- so every finisher builds an Arrow
//! array and nothing here touches polars (docs/PLAN.md task 86). The one
//! hand-off to polars is in the bank, where the struct array is named.

use polars::prelude::PlSmallStr;
use polars_arrow::array::{
    BooleanArray, DictionaryArray, Float64Array, Int64Array, MutableBinaryViewArray,
    PrimitiveArray, Utf8ViewArray,
};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::datatypes::{ArrowDataType, DTYPE_ENUM_VALUES_NEW, IntegerType, Metadata};

/// One output column under construction: a values buffer, NaN where no
/// finite value has been set, and its validity bits. The finishers hand both
/// to an Arrow array as they are: no `Vec<Option<f64>>` and no second copy.
///
/// The bits are packed a byte at a time as `scatter`'s run path copies a
/// chunk, while its values are still in cache, and are trusted at `finish`
/// only if every chunk went that way (`packed`); otherwise validity is
/// `is_finite` over `values` in one pass at `finish`. Neither sets a bit per
/// value as it lands: that read-modify-write was a third of assembling a
/// 230-statistic `ew_cov`, and the separate pass re-read every value from
/// memory (docs/records/PERFORMANCE-LOG.md §13). `n_eff` is the one column reported as it
/// is, finite or not, so it alone keeps a bit set per row (`set`).
pub(crate) struct F64Column {
    /// NaN where no finite value has been set, for the first `filled` rows;
    /// always `n` long. The rows past `filled` are zeros, never read: a
    /// column made by [`Self::appending`] is allocated already zeroed (the
    /// allocator hands zeroed memory over without writing it), its rows are
    /// written as the runs arrive, and [`Self::complete`] fills the rest
    /// with NaN.
    values: Vec<f64>,
    /// The column's length in rows.
    n: usize,
    /// The rows from 0 that are written: `n` for a column made by
    /// [`Self::new`], and for an appending one where its last run ended.
    filled: usize,
    /// `values.len()` bits, packed little-endian as polars keeps them.
    bits: Vec<u8>,
    /// Every set value's bit is in `bits`.
    packed: bool,
    /// Scratch for `run`: one flag byte per row of the run.
    flags: Vec<u8>,
}

impl F64Column {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            values: vec![f64::NAN; n],
            n,
            filled: n,
            bits: vec![0u8; n.div_ceil(8)],
            packed: true,
            flags: Vec::new(),
        }
    }

    /// A column of `n` rows for runs that arrive in order from row 0, as a
    /// single stream's do: each run is written where the last ended, with
    /// no NaN fill first (a fill the runs overwrite in full, a write of the
    /// whole column an extra time). A run that does not start where the
    /// last ended, or a read of the rows not yet reached, finds the rest
    /// filled with NaN, as [`Self::new`]'s are ([`Self::complete`]).
    pub(crate) fn appending(n: usize) -> Self {
        Self {
            values: vec![0.0; n],
            n,
            filled: 0,
            bits: vec![0u8; n.div_ceil(8)],
            packed: true,
            flags: Vec::new(),
        }
    }

    /// Every row of the column present: the rows no run reached are NaN.
    pub(crate) fn complete(&mut self) {
        if self.filled < self.n {
            self.values[self.filled..].fill(f64::NAN);
            self.filled = self.n;
        }
    }

    /// The contract is finite-or-null. NaN is the models' own null encoding,
    /// but a diverged model can also reach exact +/-inf, and `is_nan` alone
    /// would hand that to the user.
    #[inline]
    pub(crate) fn set_if_finite(&mut self, i: usize, v: f64) {
        debug_assert_eq!(self.filled, self.n, "`complete` before setting a row");
        if v.is_finite() {
            self.values[i] = v;
            self.packed = false;
        }
    }

    /// Valid whatever the value: `n_eff`, which is reported as it is.
    #[inline]
    pub(crate) fn set(&mut self, i: usize, v: f64) {
        debug_assert_eq!(self.filled, self.n, "`complete` before setting a row");
        self.values[i] = v;
        self.bits[i / 8] |= 1 << (i % 8);
    }

    /// The run `vals` of a chunk whose rows are `base..base + vals.len()`,
    /// with `processed` alongside: a processed, finite value lands with its
    /// bit, anything else leaves NaN and a clear bit -- exactly what
    /// `set_if_finite` over the same rows gives.
    ///
    /// A run of [`SHORT_RUN`] rows or more of which every row is processed
    /// and finite -- nearly every run of a model that has warmed up -- is one
    /// slice copy and a range of set bits: the values are checked in a pass
    /// that is one vector add a pair of values ([`all_finite`]), and the copy
    /// and the bits are then written without a decision a value. A shorter
    /// run, which a grouped bank's runs mostly are, is written a row at a
    /// time, since the passes cost more than its rows.
    ///
    /// Any other run takes two passes the compiler vectorizes -- the select
    /// into `values` with a byte flag per row, then the flags packed eight at
    /// a time by the multiply that gathers the low bit of each byte into one
    /// byte -- rather than one pass that shifts each flag into place, which
    /// it compiles a lane at a time: 0.5 ns a value against 0.3, measured in
    /// isolation. The partial bytes at each end go bit by bit.
    ///
    /// A run that starts where the written rows end is written in place, and
    /// any other first fills the rows no run has reached with NaN.
    pub(crate) fn run(&mut self, base: usize, vals: &[f64], processed: &[bool]) {
        let n = vals.len();
        if base == self.filled && base + n <= self.n {
            self.filled = base + n;
        } else {
            self.complete();
        }
        let dst = &mut self.values[base..base + n];
        if n < SHORT_RUN {
            // A grouped bank's runs are a few rows, from one up, and the
            // passes below cost more than the rows: each row on its own.
            let rows = dst.iter_mut().zip(vals).zip(processed).enumerate();
            for (k, ((d, &v), &p)) in rows {
                let ok = p & v.is_finite();
                *d = if ok { v } else { f64::NAN };
                let i = base + k;
                self.bits[i / 8] |= u8::from(ok) << (i % 8);
            }
            return;
        }
        // `&` over the bytes, not `all`, which stops at the first false and
        // so is not a vector loop.
        if processed.iter().fold(true, |all, &p| all & p) && all_finite(vals) {
            dst.copy_from_slice(vals);
            set_bits(&mut self.bits, base, n);
            return;
        }
        // Only the first `n` flags are read, and every one of them is
        // written below: the buffer is grown, not cleared.
        if self.flags.len() < n {
            self.flags.resize(n, 0);
        }
        let flags = &mut self.flags[..n];
        let rows = dst
            .iter_mut()
            .zip(vals)
            .zip(processed)
            .zip(flags.iter_mut());
        for (((d, &v), &p), f) in rows {
            let ok = p & v.is_finite();
            *d = if ok { v } else { f64::NAN };
            *f = ok as u8;
        }
        let head = ((8 - base % 8) % 8).min(n);
        let body = head + (n - head) / 8 * 8;
        for k in (0..head).chain(body..n) {
            if flags[k] != 0 {
                self.bits[(base + k) / 8] |= 1 << ((base + k) % 8);
            }
        }
        let bytes = &mut self.bits[(base + head) / 8..];
        // `body - head` is a whole number of eights, so nothing remains.
        let (eights, _) = flags[head..body].as_chunks::<8>();
        for (f, byte) in eights.iter().zip(bytes) {
            let x = u64::from_le_bytes(*f);
            *byte = (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        }
    }

    /// True where the value is set: finite, or -- with `all` -- written.
    #[inline]
    fn is_valid(&self, i: usize) -> bool {
        if self.packed {
            self.bits[i / 8] >> (i % 8) & 1 == 1
        } else {
            self.values[i].is_finite()
        }
    }

    /// The validity as polars wants it: `None` when every row is valid.
    ///
    /// Borrows `bits` rather than taking them, so a finisher may read
    /// `is_valid` before or after this and get the same answer. The order used
    /// to be load-bearing and enforced by a comment (review 2026-09-17, A3);
    /// the cost of it not being is one copy of `n / 8` bytes for a packed
    /// column, beside the `8 n` bytes of values it travels with.
    fn validity(&self) -> Option<Bitmap> {
        let n = self.values.len();
        let bits = if self.packed {
            MutableBitmap::from_vec(self.bits.clone(), n)
        } else {
            MutableBitmap::from_trusted_len_iter(self.values.iter().map(|v| v.is_finite()))
        };
        (bits.unset_bits() > 0).then(|| bits.into())
    }

    /// The values and their validity as an Arrow array: no copy, since a
    /// primitive array *is* a values buffer and a validity bitmap.
    pub(crate) fn finish_array(mut self) -> Float64Array {
        self.complete();
        let validity = self.validity();
        PrimitiveArray::new(ArrowDataType::Float64, self.values.into(), validity)
    }

    /// The same column as `i64`, for every integer a model emits (a cluster
    /// label, an id, a count, a state). Every set value is an integer by
    /// construction; the null rows carry NaN and are masked, not cast.
    pub(crate) fn finish_i64_array(mut self) -> Int64Array {
        self.complete();
        let validity = self.validity();
        let values: Vec<i64> = self
            .values
            .iter()
            .map(|&v| if v.is_finite() { v as i64 } else { 0 })
            .collect();
        PrimitiveArray::new(ArrowDataType::Int64, values.into(), validity)
    }

    /// The same column as `Boolean`, for a `1.0` / `0.0` flag.
    ///
    /// A boolean array is two bitmaps, values and validity. The values bitmap
    /// is built from every row, masked or not: a masked row's value is
    /// arbitrary, so `v == 1.0` over every row is right and the mask decides
    /// what is seen. `validity` borrows `bits`, so the two may be built in
    /// either order.
    pub(crate) fn finish_bool_array(mut self) -> BooleanArray {
        self.complete();
        let values: Bitmap =
            MutableBitmap::from_trusted_len_iter(self.values.iter().map(|&v| v == 1.0)).into();
        let validity = self.validity();
        BooleanArray::new(ArrowDataType::Boolean, values, validity)
    }

    /// The same column as the class names, for an `ew_class` prediction:
    /// every set value is a position in `classes` by construction (the model
    /// emits the argmax over its own classes), and the null rows stay null.
    /// A dictionary over the classes in their order, which [`enum_metadata`]
    /// on its field makes polars read as an `Enum` of them (task 244).
    pub(crate) fn finish_label_array(mut self, classes: &[String]) -> DictionaryArray<u32> {
        self.complete();
        let keys = self.values.iter().enumerate().map(|(i, &v)| {
            (self.is_valid(i) && v >= 0.0 && (v as usize) < classes.len()).then_some(v as u32)
        });
        let names: Vec<&str> = classes.iter().map(String::as_str).collect();
        dict_array(keys, &names)
    }
}

/// Rows below which [`F64Column::run`] takes each row on its own, with no
/// pass over the run.
const SHORT_RUN: usize = 16;

/// Whether every value is finite, or, where a finite sum overflowed, not
/// known to be: `false` is never wrong about a non-finite value and may be
/// wrong about a finite run, which then takes the exact path. A NaN or an
/// infinity anywhere makes the sum NaN or infinite (they do not cancel: an
/// infinity and its negative sum to NaN), and the sum is a vector add a
/// pair of values where a test of each is several instructions each.
///
/// Sixteen partial sums, which the compiler holds in vector registers and
/// adds a lane at a time, never in an order that matters: only whether the
/// total is finite is read.
#[inline]
fn all_finite(vals: &[f64]) -> bool {
    let mut partial = [0.0f64; 16];
    let (blocks, tail) = vals.as_chunks::<16>();
    for block in blocks {
        for (p, v) in partial.iter_mut().zip(block) {
            *p += v;
        }
    }
    // Apart from `partial`, so that the loop above stays a vector loop.
    let rest: f64 = tail.iter().sum();
    (partial.iter().sum::<f64>() + rest).is_finite()
}

/// Bits `base..base + n` of `bits` set, the others as they were.
fn set_bits(bits: &mut [u8], base: usize, n: usize) {
    let Some(end) = (base + n).checked_sub(1).filter(|_| n > 0) else {
        return;
    };
    let (first_byte, last_byte) = (base / 8, end / 8);
    let head = 0xFFu8 << (base % 8);
    let tail = 0xFFu8 >> (7 - end % 8);
    if first_byte == last_byte {
        bits[first_byte] |= head & tail;
    } else {
        bits[first_byte] |= head;
        bits[first_byte + 1..last_byte].fill(0xFF);
        bits[last_byte] |= tail;
    }
}

/// A dictionary-encoded string array over `names`, one key a row, null
/// where a row has none: what an `Enum` field is made of, with
/// [`enum_metadata`] on its field. A key past `names` is null.
pub(crate) fn dict_array(
    keys: impl Iterator<Item = Option<u32>>,
    names: &[&str],
) -> DictionaryArray<u32> {
    let keys: Vec<Option<u32>> = keys
        .map(|k| k.filter(|&k| (k as usize) < names.len()))
        .collect();
    let validity: Bitmap =
        MutableBitmap::from_trusted_len_iter(keys.iter().map(Option::is_some)).into();
    let keys = PrimitiveArray::new(
        ArrowDataType::UInt32,
        keys.iter()
            .map(|k| k.unwrap_or(0))
            .collect::<Vec<u32>>()
            .into(),
        (validity.unset_bits() > 0).then_some(validity),
    );
    let mut values = MutableBinaryViewArray::<str>::with_capacity(names.len());
    for n in names {
        values.push(Some(*n));
    }
    let values: Utf8ViewArray = values.into();
    let dtype = ArrowDataType::Dictionary(
        IntegerType::UInt32,
        Box::new(ArrowDataType::Utf8View),
        false,
    );
    DictionaryArray::try_new(dtype, keys, Box::new(values)).expect("every key is a name")
}

/// A column of small codes as a dictionary-encoded string array, `0` null:
/// what `withheld_reason` is (docs/WARMUP-AND-CONVERGENCE.md §3). A
/// dictionary array is a key a row plus the few names once, where a string
/// column costs sixteen bytes a row even when every value is null. With
/// [`enum_metadata`] on its field polars reads it as an `Enum` over exactly
/// these names -- one dictionary for every batch, which the IPC writer
/// requires and a per-chunk categorical does not give. The keys are `u32`,
/// the width polars has always read a string dictionary in.
pub(crate) fn code_array(codes: &[u8], names: &[&str]) -> DictionaryArray<u32> {
    // A code with no name -- a bit flipped in a state file's last row -- is
    // null, not a panic (`tests/summary.rs`, the bit-flip test).
    let known = |c: u8| c != 0 && usize::from(c) <= names.len();
    let validity: Bitmap =
        MutableBitmap::from_trusted_len_iter(codes.iter().map(|&c| known(c))).into();
    let keys: Vec<u32> = codes
        .iter()
        .map(|&c| if known(c) { u32::from(c - 1) } else { 0 })
        .collect();
    let keys = PrimitiveArray::new(
        ArrowDataType::UInt32,
        keys.into(),
        (validity.unset_bits() > 0).then_some(validity),
    );
    let mut values = MutableBinaryViewArray::<str>::with_capacity(names.len());
    for n in names {
        values.push(Some(*n));
    }
    let values: Utf8ViewArray = values.into();
    let dtype = ArrowDataType::Dictionary(
        IntegerType::UInt32,
        Box::new(ArrowDataType::Utf8View),
        false,
    );
    DictionaryArray::try_new(dtype, keys, Box::new(values)).expect("every key is a name")
}

/// The field metadata that makes polars read a string dictionary as an
/// `Enum` over `names`, in polars' own encoding: each name as its length, a
/// semicolon and the name, concatenated (`DTYPE_ENUM_VALUES_NEW`).
pub(crate) fn enum_metadata(names: &[&str]) -> Metadata {
    let mut encoded = String::new();
    for n in names {
        encoded.push_str(&n.len().to_string());
        encoded.push(';');
        encoded.push_str(n);
    }
    Metadata::from([(
        PlSmallStr::from_static(DTYPE_ENUM_VALUES_NEW),
        PlSmallStr::from_string(encoded),
    )])
}

impl F64Column {
    /// The boxed forms, for a caller assembling a struct from mixed arrays:
    /// every field a sink finishes is a `Box<dyn Array>`, so the boxing lives
    /// here rather than at fourteen call sites.
    pub(crate) fn finish_array_boxed(self) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_array())
    }

    pub(crate) fn finish_i64_array_boxed(self) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_i64_array())
    }

    pub(crate) fn finish_bool_array_boxed(self) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_bool_array())
    }

    pub(crate) fn finish_label_array_boxed(
        self,
        classes: &[String],
    ) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_label_array(classes))
    }
}

#[cfg(test)]
mod column_tests {
    use super::F64Column;

    /// A deterministic mix of finite, NaN and infinite values.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match (x >> 33) % 7 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => i as f64 * 0.5 - 3.0,
                }
            })
            .collect()
    }

    /// The packed run path is `set_if_finite` over the same rows, to the bit:
    /// the same values, the same validity, whatever the alignment of the run
    /// within the column and the mix of skipped and non-finite rows.
    #[test]
    fn a_packed_run_matches_the_scatter_bit_for_bit() {
        for base in 0..17 {
            for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100] {
                let n = base + len + 5;
                let vals = values(len, base as u64 * 1000 + len as u64);
                let processed: Vec<bool> = (0..len).map(|k| (k * 7 + base) % 5 != 0).collect();

                let mut fast = F64Column::new(n);
                fast.run(base, &vals, &processed);
                let mut slow = F64Column::new(n);
                for (k, (&v, &p)) in vals.iter().zip(&processed).enumerate() {
                    if p {
                        slow.set_if_finite(base + k, v);
                    }
                }

                assert!(fast.packed, "base {base} len {len}");
                for i in 0..n {
                    assert_eq!(
                        fast.values[i].to_bits(),
                        slow.values[i].to_bits(),
                        "value at {i}: base {base} len {len}"
                    );
                    assert_eq!(
                        fast.is_valid(i),
                        slow.is_valid(i),
                        "bit at {i}: base {base} len {len}"
                    );
                    assert_eq!(fast.is_valid(i), fast.values[i].is_finite());
                }
                let (fv, sv) = (fast.validity(), slow.validity());
                assert_eq!(fv, sv, "validity: base {base} len {len}");
            }
        }
    }

    /// A run of which every row is processed and finite is one copy and a
    /// range of bits; one flaw anywhere in it -- a NaN, either infinity, a
    /// row not processed -- sends it down the exact path, and a finite run
    /// whose sum overflows is not mistaken for a flawed one's opposite. Each
    /// is the column `set_if_finite` over the same rows gives, to the bit,
    /// whatever the run's length (the sum goes sixteen values at a time),
    /// its place in the column and where the flaw falls.
    #[test]
    fn a_clean_run_and_a_run_with_one_flaw_match_the_scatter() {
        enum Flaw {
            None,
            Nan,
            Inf,
            NegInf,
            Unprocessed,
            // Finite values whose sum is infinite.
            Overflow,
            // Negative zero and a subnormal: finite, kept as they are.
            Odd,
        }
        for base in [0usize, 1, 5, 8, 13] {
            for len in [1usize, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 100, 257] {
                for at in [0, len / 2, len - 1] {
                    for flaw in [
                        Flaw::None,
                        Flaw::Nan,
                        Flaw::Inf,
                        Flaw::NegInf,
                        Flaw::Unprocessed,
                        Flaw::Overflow,
                        Flaw::Odd,
                    ] {
                        let n = base + len + 3;
                        let mut vals: Vec<f64> = (0..len).map(|k| k as f64 * 0.25 - 7.0).collect();
                        let mut processed = vec![true; len];
                        match flaw {
                            Flaw::None => {}
                            Flaw::Nan => vals[at] = f64::NAN,
                            Flaw::Inf => vals[at] = f64::INFINITY,
                            Flaw::NegInf => vals[at] = f64::NEG_INFINITY,
                            Flaw::Unprocessed => processed[at] = false,
                            Flaw::Overflow => {
                                for v in vals.iter_mut() {
                                    *v = f64::MAX;
                                }
                            }
                            Flaw::Odd => {
                                vals[at] = -0.0;
                                vals[len - 1 - at] = f64::MIN_POSITIVE / 4.0;
                            }
                        }
                        let mut fast = F64Column::appending(n);
                        fast.run(base, &vals, &processed);
                        fast.complete();
                        let mut slow = F64Column::new(n);
                        for (k, (&v, &p)) in vals.iter().zip(&processed).enumerate() {
                            if p {
                                slow.set_if_finite(base + k, v);
                            }
                        }
                        let ctx = format!("base {base} len {len} at {at}");
                        assert!(fast.packed, "{ctx}");
                        for i in 0..n {
                            assert_eq!(
                                fast.values[i].to_bits(),
                                slow.values[i].to_bits(),
                                "value at {i}: {ctx}"
                            );
                            assert_eq!(fast.is_valid(i), slow.is_valid(i), "bit at {i}: {ctx}");
                        }
                        assert_eq!(fast.validity(), slow.validity(), "{ctx}");
                    }
                }
            }
        }
    }

    /// `set_bits` sets exactly the bits it is given, leaving the rest.
    #[test]
    fn set_bits_sets_its_range_and_no_other() {
        for base in 0..20usize {
            for n in 0..40usize {
                let mut bits = vec![0b0101_0101u8; 8];
                super::set_bits(&mut bits, base, n);
                for i in 0..64 {
                    let before = i % 2 == 0;
                    let inside = (base..base + n).contains(&i);
                    assert_eq!(
                        bits[i / 8] >> (i % 8) & 1 == 1,
                        before || inside,
                        "bit {i}: base {base} n {n}"
                    );
                }
            }
        }
    }

    /// Once any row went through `set_if_finite`, the bits are not trusted:
    /// validity comes from the values, so a mixed column is still right.
    #[test]
    fn a_mixed_column_falls_back_to_the_values() {
        let n = 40;
        let vals = values(16, 7);
        let mut col = F64Column::new(n);
        col.run(8, &vals, &[true; 16]);
        col.set_if_finite(3, 1.5);
        col.set_if_finite(30, f64::NAN);
        assert!(!col.packed);
        for i in 0..n {
            assert_eq!(col.is_valid(i), col.values[i].is_finite());
        }
        let v = col.validity().expect("some rows are null");
        for (i, value) in col.values.iter().enumerate().take(n) {
            assert_eq!(v.get_bit(i), value.is_finite());
        }
    }

    /// A column that takes its runs in order, with no NaN fill first, is the
    /// filled column those runs make, to the bit -- whatever the runs'
    /// sizes and alignment, with a gap before one (the rest filled), a run
    /// that starts before the end of what is written, and rows no run
    /// reached.
    #[test]
    fn an_appending_column_matches_a_filled_one() {
        for n in [0usize, 1, 7, 8, 9, 64, 100, 131] {
            for cuts in [
                vec![],
                vec![3],
                vec![8, 16],
                vec![1, 2, 3, 70],
                vec![64],
                vec![50, 99],
            ] {
                // The runs' starts, then every row from the last start.
                let mut bounds: Vec<usize> = cuts.iter().copied().filter(|&c| c < n).collect();
                bounds.insert(0, 0);
                for stop_early in [false, true] {
                    for skip in [0usize, 1] {
                        let vals = values(n, n as u64 * 31 + 5);
                        let processed: Vec<bool> = (0..n).map(|k| k % 4 != 3).collect();
                        let mut filled = F64Column::new(n);
                        let mut appending = F64Column::appending(n);
                        for (w, &from) in bounds.iter().enumerate() {
                            let to = bounds.get(w + 1).copied().unwrap_or(n);
                            // `skip`: leave a gap before every second run.
                            let from = if skip == 1 && w % 2 == 1 {
                                (from + 1).min(to)
                            } else {
                                from
                            };
                            if stop_early && to == n && w > 0 {
                                continue;
                            }
                            for col in [&mut filled, &mut appending] {
                                col.run(from, &vals[from..to], &processed[from..to]);
                            }
                        }
                        // A write by place, after runs.
                        if n > 2 {
                            for col in [&mut filled, &mut appending] {
                                col.complete();
                                col.set_if_finite(n - 2, 2.5);
                            }
                        }
                        let (a, b) = (filled.finish_array(), appending.finish_array());
                        assert_eq!(a.len(), n);
                        assert_eq!(a.validity(), b.validity(), "n {n} {bounds:?}");
                        assert_eq!(
                            a.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "n {n} {bounds:?}"
                        );
                    }
                }
            }
        }
    }

    /// `set` is for `n_eff`: valid whatever the value, null where never set.
    #[test]
    fn set_is_valid_whatever_the_value() {
        let mut col = F64Column::new(10);
        col.set(2, f64::NAN);
        col.set(9, 0.0);
        assert!(col.packed);
        let v = col.validity().expect("rows 0, 1, 3..9 are null");
        assert_eq!(v.set_bits(), 2);
        assert!(v.get_bit(2) && v.get_bit(9));
    }

    /// A fully valid column reports no validity at all, as polars expects.
    #[test]
    fn a_full_column_has_no_validity() {
        let mut col = F64Column::new(24);
        let vals: Vec<f64> = (0..24).map(|i| i as f64).collect();
        col.run(0, &vals, &[true; 24]);
        assert!(col.validity().is_none());
        let mut col = F64Column::new(24);
        for (i, &v) in vals.iter().enumerate() {
            col.set_if_finite(i, v);
        }
        assert!(col.validity().is_none());
    }
}
