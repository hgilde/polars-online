//! One output column under construction, as the bank assembles its
//! struct columns: the values and their validity bits, handed on without a
//! second copy (docs/PERFORMANCE.md §13).
//!
//! The buffers are Arrow's already -- a values `Vec` and a validity `Bitmap`
//! are exactly what a primitive array is -- so every finisher builds an Arrow
//! array and nothing here touches polars (docs/PLAN.md task 86). The one
//! hand-off to polars is in `assemble`, where the struct array is named.

use polars::prelude::PlSmallStr;
use polars_arrow::array::{
    BooleanArray, DictionaryArray, Float64Array, Int32Array, Int64Array, MutableBinaryViewArray,
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
/// memory (docs/PERFORMANCE.md §13). `n_eff` is the one column reported as it
/// is, finite or not, so it alone keeps a bit set per row (`set`).
pub(crate) struct F64Column {
    values: Vec<f64>,
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
            bits: vec![0u8; n.div_ceil(8)],
            packed: true,
            flags: Vec::new(),
        }
    }

    /// The contract is finite-or-null. NaN is the models' own null encoding,
    /// but a diverged model can also reach exact +/-inf, and `is_nan` alone
    /// would hand that to the user.
    #[inline]
    pub(crate) fn set_if_finite(&mut self, i: usize, v: f64) {
        if v.is_finite() {
            self.values[i] = v;
            self.packed = false;
        }
    }

    /// Valid whatever the value: `n_eff`, which is reported as it is.
    #[inline]
    pub(crate) fn set(&mut self, i: usize, v: f64) {
        self.values[i] = v;
        self.bits[i / 8] |= 1 << (i % 8);
    }

    /// The run `vals` of a chunk whose rows are `base..base + vals.len()`,
    /// with `processed` alongside: a processed, finite value lands with its
    /// bit, anything else leaves NaN and a clear bit -- exactly what
    /// `set_if_finite` over the same rows gives.
    ///
    /// Two passes the compiler vectorizes -- the select into `values` with a
    /// byte flag per row, then the flags packed eight at a time by the
    /// multiply that gathers the low bit of each byte into one byte -- rather
    /// than one pass that shifts each flag into place, which it compiles a
    /// lane at a time: 0.5 ns a value against 0.3, measured in isolation.
    /// The partial bytes at each end go bit by bit.
    pub(crate) fn run(&mut self, base: usize, vals: &[f64], processed: &[bool]) {
        let n = vals.len();
        let dst = &mut self.values[base..base + n];
        self.flags.clear();
        self.flags.resize(n, 0);
        let rows = dst
            .iter_mut()
            .zip(vals)
            .zip(processed)
            .zip(self.flags.iter_mut());
        for (((d, &v), &p), f) in rows {
            let ok = p & v.is_finite();
            *d = if ok { v } else { f64::NAN };
            *f = ok as u8;
        }
        let head = ((8 - base % 8) % 8).min(n);
        let body = head + (n - head) / 8 * 8;
        for k in (0..head).chain(body..n) {
            if self.flags[k] != 0 {
                self.bits[(base + k) / 8] |= 1 << ((base + k) % 8);
            }
        }
        let bytes = &mut self.bits[(base + head) / 8..];
        for (f, byte) in self.flags[head..body].chunks_exact(8).zip(bytes) {
            let x = u64::from_le_bytes(f.try_into().expect("eight flags"));
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
    pub(crate) fn finish_array(self) -> Float64Array {
        let validity = self.validity();
        PrimitiveArray::new(ArrowDataType::Float64, self.values.into(), validity)
    }

    /// The same column as `i32`, for a value that is an index or a count (a
    /// `kmeans` assignment, a `micro` count). Every set value is a small
    /// non-negative integer by construction; the null rows carry NaN and are
    /// masked, not cast.
    pub(crate) fn finish_i32_array(self) -> Int32Array {
        let validity = self.validity();
        let values: Vec<i32> = self
            .values
            .iter()
            .map(|&v| if v.is_finite() { v as i32 } else { 0 })
            .collect();
        PrimitiveArray::new(ArrowDataType::Int32, values.into(), validity)
    }

    /// The same column as `i64`, for an id that only ever grows (a `micro`
    /// id or label).
    pub(crate) fn finish_i64_array(self) -> Int64Array {
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
    pub(crate) fn finish_bool_array(self) -> BooleanArray {
        let values: Bitmap =
            MutableBitmap::from_trusted_len_iter(self.values.iter().map(|&v| v == 1.0)).into();
        let validity = self.validity();
        BooleanArray::new(ArrowDataType::Boolean, values, validity)
    }

    /// The same column as the class names, for an `ew_class` prediction:
    /// every set value is a position in `classes` by construction (the model
    /// emits the argmax over its own classes), and the null rows stay null.
    pub(crate) fn finish_label_array(self, classes: &[String]) -> Utf8ViewArray {
        let mut b = MutableBinaryViewArray::<str>::with_capacity(self.values.len());
        for (i, &v) in self.values.iter().enumerate() {
            b.push(
                self.is_valid(i)
                    .then(|| classes.get(v as usize).map(String::as_str))
                    .flatten(),
            );
        }
        b.into()
    }
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
    /// every branch of `assemble` yields `Box<dyn Array>`, so the boxing lives
    /// here rather than at fourteen call sites.
    pub(crate) fn finish_array_boxed(self) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_array())
    }

    pub(crate) fn finish_i32_array_boxed(self) -> Box<dyn polars_arrow::array::Array> {
        Box::new(self.finish_i32_array())
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
        for i in 0..n {
            assert_eq!(v.get_bit(i), col.values[i].is_finite());
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
