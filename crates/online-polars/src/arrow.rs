//! A chunk as Arrow, and the polars adapter that makes one (docs/PLAN.md
//! task 86).
//!
//! The bank reads a chunk in exactly two forms: numbers as `f64`, and keys as
//! text. [`ArrowChunk`] is that, and nothing more -- a caller holding Arrow
//! arrays can build one and feed the bank without polars in the process. A
//! feature is the one exception: a column only features read may stay in
//! its own numeric type ([`NumArray`]), widened to `f64` as each run of rows
//! is copied, so a chunk of float32 features is never cast whole.
//!
//! Everything *polars-shaped* about reading a frame lives here rather than in
//! the bank: finding a column by name, refusing a dtype that cannot be read,
//! casting to `Float64` or to text, and deciding whether a group key is an
//! integer. Everything *model-shaped* -- a clock that must be finite, a weight
//! that must not be negative, a label that must be one of the declared classes
//! -- stays in the bank, because it is true whoever supplies the data.
//!
//! A dtype the role cannot take is refused as `InvalidOperation`, the kind
//! [`crate::is_wrong_type`] reads (task 355), and every other refusal here
//! as `ComputeError`; the text alone is what either displays.

use polars::prelude::*;
// The trait is brought in unnamed: `ArrowArray` below is the C Data Interface
// struct, not `array::Array`, and the two must not collide.
use polars_arrow::array::Array as _;
use polars_arrow::array::{
    BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, Utf8ViewArray,
};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{ArrowArray, ArrowSchema, export_array_to_c, export_field_to_c};

use online_core::ClockValue;
use polars_utils::aliases::{PlHashMap, PlHashSet};
use serde::{Deserialize, Serialize};

use crate::rows::FeatureCol;
use crate::spec::{ClockScale, ModelKind, Spec};

/// One column of an [`ArrowChunk`], in the form the bank reads it.
#[derive(Clone, Debug)]
pub enum ArrowCol {
    /// A number: null is a NaN to every consumer, so the validity is applied
    /// when the values are read rather than carried alongside.
    F64(Float64Array),
    /// A number in the type the source holds it in, read only as a feature
    /// ([`NumArray`]): widened to `f64` as each run of rows is copied, so
    /// the chunk is never cast whole. Any other role reads [`Self::F64`].
    Num(NumArray),
    /// A key or a label, as text.
    Str(Utf8ViewArray),
    /// A signed integer: a group key, kept as an integer so the bank can
    /// bucket on the value itself -- no cast, no hash, no collision to
    /// document (docs/records/PERFORMANCE-LOG.md P11) -- or an integer clock, kept as
    /// an integer so its steps are taken in integers (docs/PLAN.md task
    /// 200).
    I64(Int64Array),
    /// The same, unsigned, for a `u64` column whose values do not fit an
    /// `i64`.
    U64(UInt64Array),
    /// A temporal clock, as nanoseconds since the Unix epoch whatever the
    /// source column's unit. Kept as an integer so the gap between two rows
    /// is taken in integers and a nanosecond timestamp's gaps stay exact
    /// for the stream's life ([`online_core::ClockValue`], docs/PLAN.md
    /// task 88). Read only as a clock: a temporal column in any other role
    /// is refused before the chunk is built.
    Nanos(Int64Array),
    /// A boolean, read only by a formula target's expression (docs/PLAN.md
    /// task 104; review R1, D2): every other role takes a boolean as a
    /// number.
    Bool(BooleanArray),
}

impl ArrowCol {
    pub fn len(&self) -> usize {
        match self {
            Self::F64(a) => a.len(),
            Self::Num(a) => a.len(),
            Self::Str(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::U64(a) => a.len(),
            Self::Nanos(a) => a.len(),
            Self::Bool(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True for the two integer forms, which is what `group_close =
    /// "monotone"` orders numerically rather than lexicographically.
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::I64(_) | Self::U64(_))
    }

    /// Whether the column is held in `form`.
    pub fn is(&self, form: Form) -> bool {
        matches!(
            (self, form),
            (Self::F64(_), Form::Number)
                | (Self::Str(_), Form::Text)
                | (Self::I64(_) | Self::U64(_), Form::Key)
                | (Self::Nanos(_), Form::Clock)
                | (Self::Bool(_), Form::Bool)
        )
    }

    /// The form, for a message: what a caller gave, against what a role reads.
    pub fn form(&self) -> &'static str {
        match self {
            Self::F64(_) | Self::Num(_) => "a number",
            Self::Str(_) => "text",
            Self::I64(_) | Self::U64(_) => "an integer key",
            Self::Nanos(_) => "a temporal clock",
            Self::Bool(_) => "a boolean",
        }
    }
}

/// A feature column in the type its source holds it in: `Float32`, or an
/// integer of 8 to 64 bits. The polars adapter keeps a column in this form
/// when only features read it, and the copy into rows widens each value as
/// it writes it, `as f64` -- the conversion polars' own cast to `Float64`
/// makes -- with a null read as [`online_core::NULL`]. Cast when the chunk
/// was read, a float32 column was a whole-chunk copy of its own: 160 MB a
/// chunk at 20,000 rows of 1,000 features.
#[derive(Clone, Debug)]
pub enum NumArray {
    F32(Float32Array),
    I8(Int8Array),
    I16(Int16Array),
    I32(Int32Array),
    I64(Int64Array),
    U8(UInt8Array),
    U16(UInt16Array),
    U32(UInt32Array),
    U64(UInt64Array),
}

impl NumArray {
    /// The array, as the trait object polars builds a `Series` from.
    fn boxed(&self) -> Box<dyn polars_arrow::array::Array> {
        match self {
            Self::F32(a) => a.clone().boxed(),
            Self::I8(a) => a.clone().boxed(),
            Self::I16(a) => a.clone().boxed(),
            Self::I32(a) => a.clone().boxed(),
            Self::I64(a) => a.clone().boxed(),
            Self::U8(a) => a.clone().boxed(),
            Self::U16(a) => a.clone().boxed(),
            Self::U32(a) => a.clone().boxed(),
            Self::U64(a) => a.clone().boxed(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(a) => a.len(),
            Self::I8(a) => a.len(),
            Self::I16(a) => a.len(),
            Self::I32(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::U8(a) => a.len(),
            Self::U16(a) => a.len(),
            Self::U32(a) => a.len(),
            Self::U64(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The type, as polars names it, for a message.
    pub fn dtype_name(&self) -> &'static str {
        match self {
            Self::F32(_) => "Float32",
            Self::I8(_) => "Int8",
            Self::I16(_) => "Int16",
            Self::I32(_) => "Int32",
            Self::I64(_) => "Int64",
            Self::U8(_) => "UInt8",
            Self::U16(_) => "UInt16",
            Self::U32(_) => "UInt32",
            Self::U64(_) => "UInt64",
        }
    }

    /// `s` in this form, when its dtype is one of these; `None` for any
    /// other, which is cast to `Float64` instead.
    pub(crate) fn of(s: &Series) -> PolarsResult<Option<Self>> {
        fn one<T: PolarsNumericType>(
            ca: &ChunkedArray<T>,
        ) -> polars_arrow::array::PrimitiveArray<T::Native> {
            ca.downcast_iter()
                .next()
                .cloned()
                .unwrap_or_else(|| polars_arrow::array::PrimitiveArray::from_vec(Vec::new()))
        }
        // Rechunked only once the dtype is known to be kept: a `Float64`
        // column read from several row groups would be joined here for
        // nothing, and again by its cast.
        let kept = matches!(
            s.dtype(),
            DataType::Float32
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        );
        if !kept {
            return Ok(None);
        }
        let r = s.rechunk();
        Ok(Some(match s.dtype() {
            DataType::Float32 => Self::F32(one(r.f32()?)),
            DataType::Int8 => Self::I8(one(r.i8()?)),
            DataType::Int16 => Self::I16(one(r.i16()?)),
            DataType::Int32 => Self::I32(one(r.i32()?)),
            DataType::Int64 => Self::I64(one(r.i64()?)),
            DataType::UInt8 => Self::U8(one(r.u8()?)),
            DataType::UInt16 => Self::U16(one(r.u16()?)),
            DataType::UInt32 => Self::U32(one(r.u32()?)),
            DataType::UInt64 => Self::U64(one(r.u64()?)),
            _ => unreachable!("a dtype kept as is"),
        }))
    }
}

/// What a group or session key's text depends on besides its value
/// (review round 4, N22): two columns of one form give one key for one
/// value, whatever their width -- an `Int32` `1` and an `Int64` `1` are
/// `"1"` -- and two forms two keys, an `Int64` `1` being `"1"` and a
/// `Float64` one `"1.0"`. A temporal key's text has its unit in it, and a
/// zoned one is its instant, so such a column's form is its whole dtype, as
/// is any other dtype's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyForm {
    Integer,
    Float,
    Text,
    Boolean,
    /// This dtype, as polars names it.
    Dtype(String),
}

/// A key column's form, and the dtype it was first seen as, which the
/// refusal of another form names (review round 4, N22).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyDtype {
    pub form: KeyForm,
    pub dtype: String,
}

impl KeyDtype {
    /// The form of a polars column: `None` for a column of nulls with no
    /// type of its own (`Null`), which is the null key in every form.
    pub fn of(dtype: &DataType) -> Option<Self> {
        let form = match dtype {
            DataType::Null => return None,
            d if d.is_integer() => KeyForm::Integer,
            d if d.is_float() => KeyForm::Float,
            DataType::String | DataType::Categorical(..) | DataType::Enum(..) => KeyForm::Text,
            DataType::Boolean => KeyForm::Boolean,
            d => KeyForm::Dtype(d.to_string()),
        };
        Some(Self {
            form,
            dtype: dtype.to_string(),
        })
    }
}

/// The forms a chunk holds a column in ([`ArrowChunk::series`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Number,
    Text,
    Key,
    Clock,
    Bool,
}

/// A chunk's clock column in the form the source had it: a float clock's
/// numbers, an integer clock's integers (docs/PLAN.md task 200), or a
/// temporal clock's nanoseconds since the Unix epoch.
#[derive(Clone, Copy, Debug)]
pub enum ClockArray<'a> {
    F64(&'a Float64Array),
    Nanos(&'a Int64Array),
    I64(&'a Int64Array),
    /// An unsigned 64-bit clock, whose values past `i64::MAX` the bank
    /// refuses by row.
    U64(&'a UInt64Array),
}

/// A stream's clock column as the bank hands it to a stream: a
/// [`ClockArray`] read in the stream's row order, with no nulls left in it.
#[derive(Clone, Debug)]
pub enum ClockCol {
    F64(Vec<f64>),
    Ns(Vec<i64>),
    /// An integer clock, in the column's own units (task 200).
    I64(Vec<i64>),
}

impl ClockCol {
    /// Row `i`'s value, in the form the source had it.
    #[inline]
    pub fn at(&self, i: usize) -> ClockValue {
        match self {
            Self::F64(v) => ClockValue::F64(v[i]),
            Self::Ns(v) => ClockValue::Ns(v[i]),
            Self::I64(v) => ClockValue::I64(v[i]),
        }
    }
}

/// A stream's `label_clock` column as the bank hands it to a stream (task
/// 341): each row's label time in the clock's own form, and which rows have
/// one. A null, or a float that is not finite, is a label that never
/// arrives.
#[derive(Debug, Clone)]
pub struct LabelTimes {
    pub values: ClockCol,
    /// `false` where the row's label time is null; `None` when none is.
    pub valid: Option<Vec<bool>>,
}

impl LabelTimes {
    /// Row `i`'s label time, `None` where it has none.
    #[inline]
    pub fn at(&self, i: usize) -> Option<ClockValue> {
        if self.valid.as_ref().is_some_and(|v| !v[i]) {
            return None;
        }
        Some(self.values.at(i)).filter(|v| !matches!(v, ClockValue::F64(f) if !f.is_finite()))
    }
}

/// The columns a bank reads for one chunk, already cast.
///
/// A name may appear twice with different forms: one spec may read a column as
/// a feature and another as a group key, and the two want different arrays.
/// Lookup is therefore by name *and* by the form the caller wants.
#[derive(Debug)]
pub struct ArrowChunk {
    height: usize,
    cols: Vec<(PlSmallStr, ArrowCol)>,
    /// Every column name the source had, for the "not found" message and for
    /// the spec-name clash check -- including the ones no spec reads.
    names: Vec<PlSmallStr>,
    /// Where each name's columns sit in `cols`, in `cols` order: one form,
    /// or two when one spec's feature is another's group key. A lookup is a
    /// hash, not a scan of every column: at 10,000 features the scans were
    /// most of a call (docs/records/PERFORMANCE-LOG.md §24).
    index: PlHashMap<PlSmallStr, Vec<usize>>,
    /// `names` as a set, for [`Self::has`].
    name_set: PlHashSet<PlSmallStr>,
    /// The input row this chunk's first row is: an error names `row_base +
    /// i`, so a surface that feeds an input in chunks names the input's row,
    /// not the chunk's (task 120). 0 unless [`Self::with_row_base`] says.
    row_base: usize,
    /// Each temporal or integer clock column's dtype as it arrived, for the
    /// clock fields to echo it (docs/PLAN.md tasks 152 and 200).
    clock_dtypes: Vec<(PlSmallStr, DataType)>,
    /// The group key columns held as text that are integers, too wide for
    /// a 64-bit form (task 160, PA3): `group_close = "monotone"` orders
    /// them as numbers, as it does the integer forms.
    integer_text: PlHashSet<PlSmallStr>,
    /// Each group and session column's dtype as it arrived (review round 4,
    /// N22), for the bank to keep the form of from the first chunk. An Arrow
    /// caller's chunk has none: its form is its array's
    /// ([`Self::key_form`]).
    key_dtypes: Vec<(PlSmallStr, DataType)>,
}

impl ArrowChunk {
    /// A chunk from columns the caller has already cast: numbers as
    /// `Float64Array`, keys and labels as `Utf8ViewArray`, integer group keys
    /// as `Int64Array` or `UInt64Array`. A column only features read may
    /// instead be [`ArrowCol::Num`], in its own type.
    ///
    /// `names` is every column the source had, which may be more than the
    /// columns given here but never fewer: it is what an error message lists,
    /// what the spec-name clash check reads, and what decides whether a column
    /// a scoring call may leave out is absent. Names are anything that
    /// converts to a `PlSmallStr`, so a caller may pass `&str`.
    ///
    /// Refused: a column whose length is not `height`; a column given but not
    /// listed in `names`, which would otherwise be invisible to that absence
    /// check and silently score as if missing; and a name given twice in the
    /// same form, where the first would silently win. A name *may* appear in
    /// two forms -- one spec's feature is another's group key.
    pub fn new<N: Into<PlSmallStr>, M: Into<PlSmallStr>>(
        height: usize,
        cols: Vec<(N, ArrowCol)>,
        names: Vec<M>,
    ) -> PolarsResult<Self> {
        let cols: Vec<(PlSmallStr, ArrowCol)> =
            cols.into_iter().map(|(n, c)| (n.into(), c)).collect();
        let names: Vec<PlSmallStr> = names.into_iter().map(Into::into).collect();
        if let Some((name, col)) = cols.iter().find(|(_, c)| c.len() != height) {
            polars_bail!(ShapeMismatch:
                "chunk: column {:?} has {} rows, but the chunk has {}. Give every column the \
                 chunk's number of rows.",
                name.as_str(), col.len(), height
            );
        }
        let name_set: PlHashSet<PlSmallStr> = names.iter().cloned().collect();
        if let Some((name, _)) = cols.iter().find(|(n, _)| !name_set.contains(n)) {
            polars_bail!(ColumnNotFound:
                "chunk: column {:?} is given but not listed in names, which decides whether a \
                 column a predict call may leave out is absent. List every column given in \
                 names.",
                name.as_str()
            );
        }
        let mut seen: std::collections::HashSet<(&str, &'static str)> =
            std::collections::HashSet::with_capacity(cols.len());
        for (name, col) in &cols {
            if !seen.insert((name.as_str(), col.form())) {
                polars_bail!(Duplicate:
                    "chunk: column {:?} is given twice as {}. Give each column once in each \
                     form.",
                    name.as_str(), col.form()
                );
            }
        }
        let mut index: PlHashMap<PlSmallStr, Vec<usize>> = PlHashMap::default();
        for (i, (name, _)) in cols.iter().enumerate() {
            index.entry(name.clone()).or_default().push(i);
        }
        Ok(Self {
            height,
            cols,
            names,
            index,
            name_set,
            row_base: 0,
            clock_dtypes: Vec::new(),
            integer_text: PlHashSet::default(),
            key_dtypes: Vec::new(),
        })
    }

    /// The chunk, with these text columns marked as integer group keys:
    /// decimal text of integers too wide for a 64-bit form, which
    /// `group_close = "monotone"` orders as numbers (task 160, PA3).
    #[must_use]
    pub fn with_integer_text_keys(mut self, names: impl IntoIterator<Item = PlSmallStr>) -> Self {
        self.integer_text.extend(names);
        self
    }

    /// Whether `name` is a group key held as the text of integers
    /// ([`Self::with_integer_text_keys`]).
    pub fn is_integer_text(&self, name: &str) -> bool {
        self.integer_text.contains(name)
    }

    /// The dtype a temporal or integer clock column arrived with (tasks 152
    /// and 200); for an Arrow caller's chunk, which names no dtype, the
    /// integer form it holds the column in, an `Int64` or a `UInt64`.
    /// `None` for a float clock, or a column that is not a clock here.
    pub fn clock_dtype(&self, name: &str) -> Option<DataType> {
        if let Some((_, d)) = self.clock_dtypes.iter().find(|(n, _)| n.as_str() == name) {
            return Some(d.clone());
        }
        match self.find(name, |c| matches!(c, ArrowCol::I64(_) | ArrowCol::U64(_))) {
            Some(ArrowCol::I64(_)) => Some(DataType::Int64),
            Some(ArrowCol::U64(_)) => Some(DataType::UInt64),
            _ => None,
        }
    }

    #[must_use]
    pub fn with_clock_dtypes(mut self, dtypes: Vec<(PlSmallStr, DataType)>) -> Self {
        self.clock_dtypes = dtypes;
        self
    }

    /// The chunk, with each key column's source dtype (review round 4, N22).
    #[must_use]
    pub fn with_key_dtypes(mut self, dtypes: Vec<(PlSmallStr, DataType)>) -> Self {
        self.key_dtypes = dtypes;
        self
    }

    /// The form of key column `name` in this chunk (review round 4, N22):
    /// from the dtype the source gave it, or else from the array the chunk
    /// holds it in -- an integer one, or text. `None` for a column the chunk
    /// has not got, and for a column of nulls with no type of its own.
    pub fn key_form(&self, name: &str) -> Option<KeyDtype> {
        if let Some((_, dtype)) = self.key_dtypes.iter().find(|(n, _)| n.as_str() == name) {
            return KeyDtype::of(dtype);
        }
        let held = self
            .find(name, |c| matches!(c, ArrowCol::I64(_) | ArrowCol::U64(_)))
            .or_else(|| self.find(name, |c| matches!(c, ArrowCol::Str(_))))?;
        KeyDtype::of(&match held {
            ArrowCol::I64(_) => DataType::Int64,
            ArrowCol::U64(_) => DataType::UInt64,
            // Text the caller marked as integers too wide for 64 bits.
            _ if self.is_integer_text(name) => DataType::Int128,
            _ => DataType::String,
        })
    }

    /// The chunk, as rows `row_base..` of a longer input: what an error
    /// counts its row from.
    #[must_use]
    pub fn with_row_base(mut self, row_base: usize) -> Self {
        self.row_base = row_base;
        self
    }

    /// The input row the chunk's first row is ([`Self::with_row_base`]).
    pub fn row_base(&self) -> usize {
        self.row_base
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn names(&self) -> &[PlSmallStr] {
        &self.names
    }

    pub fn has(&self, name: &str) -> bool {
        self.name_set.contains(name)
    }

    /// A column as a polars `Series`, in the first of the given forms the
    /// chunk holds it in: for the frame a spec's window core reads
    /// (docs/PLAN.md task 104). A temporal clock comes back as
    /// `Datetime(ns)`, which keeps its nanoseconds exact.
    pub fn series(&self, name: &str, forms: &[Form]) -> Option<Series> {
        let col = forms.iter().find_map(|f| self.find(name, |c| c.is(*f)))?;
        let name: PlSmallStr = name.into();
        let series = match col {
            ArrowCol::F64(a) => Series::from_arrow(name, Box::new(a.clone())),
            ArrowCol::Num(a) => Series::from_arrow(name, a.boxed()),
            ArrowCol::Str(a) => Series::from_arrow(name, Box::new(a.clone())),
            ArrowCol::I64(a) => Series::from_arrow(name, Box::new(a.clone())),
            ArrowCol::U64(a) => Series::from_arrow(name, Box::new(a.clone())),
            ArrowCol::Bool(a) => Series::from_arrow(name, Box::new(a.clone())),
            ArrowCol::Nanos(a) => Series::from_arrow(name, Box::new(a.clone())).and_then(|s| {
                s.cast(&DataType::Datetime(
                    polars::prelude::TimeUnit::Nanoseconds,
                    None,
                ))
            }),
        };
        series.ok()
    }

    /// The numeric form of a column, or the error naming the spec and role.
    pub fn f64(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&Float64Array> {
        match self.find(name, |c| matches!(c, ArrowCol::F64(_))) {
            Some(ArrowCol::F64(a)) => Ok(a),
            _ => Err(self.missing(spec, role, name, "a number")),
        }
    }

    /// A feature column as the copy into rows reads it: a number in the type
    /// the chunk holds it in, `Float64` or one of [`NumArray`]'s, or the
    /// error naming the spec.
    pub fn feature(&self, spec: &Spec, name: &str) -> PolarsResult<FeatureCol> {
        match self.find(name, |c| matches!(c, ArrowCol::F64(_) | ArrowCol::Num(_))) {
            Some(ArrowCol::F64(a)) => Ok(FeatureCol::f64(a)),
            Some(ArrowCol::Num(a)) => Ok(FeatureCol::num(a)),
            _ => Err(self.missing(spec, "feature", name, "a number")),
        }
    }

    /// The clock column in the form the source had it -- a temporal clock's
    /// nanoseconds or an integer clock's integers where the adapter read
    /// one (task 200), else the float form -- or the error naming the spec.
    /// The integer form comes first: one column may be held both ways, a
    /// float for a spec that reads it as a feature.
    pub fn clock(&self, spec: &Spec, name: &str) -> PolarsResult<ClockArray<'_>> {
        let exact = self.find(name, |c| {
            matches!(c, ArrowCol::Nanos(_) | ArrowCol::I64(_) | ArrowCol::U64(_))
        });
        match exact.or_else(|| self.find(name, |c| matches!(c, ArrowCol::F64(_)))) {
            Some(ArrowCol::Nanos(a)) => Ok(ClockArray::Nanos(a)),
            Some(ArrowCol::I64(a)) => Ok(ClockArray::I64(a)),
            Some(ArrowCol::U64(a)) => Ok(ClockArray::U64(a)),
            Some(ArrowCol::F64(a)) => Ok(ClockArray::F64(a)),
            _ => Err(self.missing(
                spec,
                "clock",
                name,
                "a number, an integer or a temporal clock",
            )),
        }
    }

    /// The text form of a column, or the error naming the spec and role.
    pub fn str(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&Utf8ViewArray> {
        match self.find(name, |c| matches!(c, ArrowCol::Str(_))) {
            Some(ArrowCol::Str(a)) => Ok(a),
            _ => Err(self.missing(spec, role, name, "text")),
        }
    }

    /// The group key column in whatever form the adapter chose for it.
    pub fn key(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&ArrowCol> {
        // Prefer an integer form. The same column can be present in two forms
        // -- one used as both `session` (cast to text) and `group` (a key)
        // pushes a `Str` and an integer under the same name -- and picking the
        // text one made `group_close = "monotone"` order it lexically, so a
        // numerically sorted integer key was refused at "10 after 9" (review
        // 2026-09-18, V12). The integer form is the one `monotone` reads as a
        // number.
        //
        // Else the text, and only the text: a key is pushed as one of these
        // three forms (`cast_to`, `Want::Key`), and the fallback that took
        // any form but a number or a clock took the boolean a formula target
        // reads from the same column, which `group_indices` met at its
        // `unreachable!` (review round 4, PA1).
        self.find(name, |c| matches!(c, ArrowCol::I64(_) | ArrowCol::U64(_)))
            .or_else(|| self.find(name, |c| matches!(c, ArrowCol::Str(_))))
            .ok_or_else(|| self.missing(spec, role, name, "text or an integer key"))
    }

    /// The first column named `name`, in `cols` order, that `want` accepts.
    fn find(&self, name: &str, want: impl Fn(&ArrowCol) -> bool) -> Option<&ArrowCol> {
        self.index
            .get(name)?
            .iter()
            .map(|&i| &self.cols[i].1)
            .find(|c| want(c))
    }

    /// A column lookup that says which spec asked and what it asked for.
    /// Polars' own `not found: "x"` names neither, and in a bank of ten specs
    /// that is the difference between a fix and a search.
    ///
    /// A column that *is* here, in a form the role does not read -- a clock
    /// given as an integer array -- is named as such, rather than reported
    /// "not found" beside a list that includes it. The polars adapter casts to
    /// the wanted form, so that branch is only ever an Arrow caller's.
    fn missing(&self, spec: &Spec, role: &str, name: &str, wanted: &str) -> PolarsError {
        if let Some((_, c)) = self.cols.iter().find(|(n, _)| n == name) {
            // A number in its own type is read only as a feature; every
            // other role reads a number as `Float64`.
            if let ArrowCol::Num(a) = c {
                return polars_err!(InvalidOperation:
                    "spec {:?}: {} column {:?} is given as {} numbers, which only a feature \
                     reads in their own type. Give a {} column as {}.",
                    spec.name, role, name, a.dtype_name(), role, wanted
                );
            }
            return polars_err!(InvalidOperation:
                "spec {:?}: {} column {:?} is given as {}, but a {} is read as {}. Give the \
                 column as {}.",
                spec.name, role, name, c.form(), role, wanted, wanted
            );
        }
        let have: Vec<&str> = self.names.iter().map(|n| n.as_str()).collect();
        polars_err!(ColumnNotFound:
            "spec {:?}: {} column {:?} is not in the input, whose columns are {:?}. Name a \
             column the input has, or add it to the input.",
            spec.name, role, name, have
        )
    }
}

/// The values of a numeric column, null as NaN, without a copy where there is
/// nothing to fill in.
///
/// Plain `f64` with NaN for null, not `Option<f64>` (docs/records/PERFORMANCE-LOG.md P3):
/// every consumer already collapses the two, since a feature or weight is
/// taken only when finite and a target only when finite. The NaN is
/// [`online_core::NULL`], whose payload tells it from a NaN the data holds:
/// an `audit` counts the two apart (docs/PLAN.md task 223 (b)), and every
/// other consumer reads either as the NaN it is.
pub fn f64_values(a: &Float64Array) -> std::borrow::Cow<'_, [f64]> {
    match a.validity() {
        Some(v) if v.unset_bits() > 0 => std::borrow::Cow::Owned(
            a.values()
                .iter()
                .zip(v.iter())
                .map(|(&x, ok)| if ok { x } else { online_core::NULL })
                .collect(),
        ),
        _ => std::borrow::Cow::Borrowed(a.values().as_slice()),
    }
}

/// One spec's output struct as the two C Data Interface structs a consumer
/// imports: the field, which carries the name and the struct's schema, and
/// the array.
///
/// This is the whole of what a binding layer needs to hand a spec's output to
/// another Arrow implementation -- pyarrow, duckdb, or py-polars'
/// `Series.from_arrow_c_array` -- over the public, standardised interface
/// rather than a private one (docs/PLAN.md task 86).
///
/// Ownership passes to the caller. Each struct carries a `release` callback
/// into *this* binary, so the consumer frees what this binary allocated, with
/// this binary's allocator; dropping one a consumer never took calls that
/// callback, and dropping one it did take is a no-op, because taking it nulls
/// the pointer. That is what makes the hand-off safe across two copies of a
/// library, and it is the same mechanism the pyo3-polars boundary uses -- the
/// difference being that this interface is public and versioned and that one
/// is not.
pub fn export_struct_to_c(name: &str, st: StructArray) -> (ArrowSchema, ArrowArray) {
    let field = ArrowField::new(name.into(), st.dtype().clone(), true);
    (
        export_field_to_c(&field),
        export_array_to_c(Box::new(st) as Box<dyn polars_arrow::array::Array>),
    )
}

/// What role a spec reads a column in, which decides the form it is cast to.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Want {
    Number,
    /// A feature: a number, kept in its own type where it is a `Float32` or
    /// an integer of up to 64 bits ([`NumArray`]) and no other role reads
    /// the column as a number; cast to `Float64` otherwise.
    Feature,
    Text,
    /// A group key: an integer stays an integer, anything else becomes text.
    Key,
    /// A clock (docs/PLAN.md task 200): a temporal column as its
    /// nanoseconds, an integer one as its integers, as a key keeps them,
    /// and anything else as a number.
    Clock,
    /// A column of a formula target (docs/PLAN.md task 104): a number as a
    /// number, text as text, so the formula reads what the frame has. A
    /// temporal column is refused, as in every role but the clock.
    Value,
}

/// Every column the specs read, in the form each reads it.
fn wanted(specs: &[Spec]) -> Vec<(PlSmallStr, Want)> {
    let mut out: Vec<(PlSmallStr, Want)> = Vec::new();
    // First-seen order in `out`, membership in `seen`: a scan of `out` per
    // column was quadratic in the columns (docs/records/PERFORMANCE-LOG.md §24).
    let mut seen: PlHashSet<(PlSmallStr, Want)> = PlHashSet::default();
    let mut push = |name: &str, want: Want| {
        let key: PlSmallStr = name.into();
        if seen.insert((key.clone(), want)) {
            out.push((key, want));
        }
    };
    for s in specs {
        for f in &s.features {
            push(f, Want::Feature);
        }
        // `ew_class` reads its target as a label; every other model as a number.
        let target_want = match &s.model {
            ModelKind::EwClass { .. } => Want::Text,
            _ => Want::Number,
        };
        // A target's own column, read at its row; a formula target's
        // columns, read by the window core.
        if s.model.compares().is_none() {
            for t in s.targets.defs() {
                if t.is_formula() {
                    for c in t.columns() {
                        push(&c, Want::Value);
                    }
                    continue;
                }
                push(&t.column, target_want);
            }
        }
        if let Some(c) = &s.clock {
            push(c, Want::Clock);
        }
        // Each row's label time, read as the clock is: a temporal one in
        // nanoseconds, an integer one in its integers (task 341).
        if let Some(l) = &s.label_clock {
            push(l.column(), Want::Clock);
        }
        if let Some(w) = &s.weight {
            push(w, Want::Number);
        }
        if let Some(c) = &s.session {
            push(c, Want::Text);
        }
        if let Some(g) = &s.group {
            push(g, Want::Key);
        }
    }
    out
}

/// Each column's first reader in each form [`wanted`] reads it in: the spec
/// and the role a failed cast names, so a column one spec reads as a group
/// key and another as a feature is named, when its number cast fails, by
/// the spec that reads it as a number.
fn want_readers(specs: &[Spec]) -> PlHashMap<(&str, Want), (&str, &'static str)> {
    let mut out: PlHashMap<(&str, Want), (&str, &'static str)> = PlHashMap::default();
    for s in specs {
        let name = s.name.as_str();
        let (target_want, target_role) = match &s.model {
            ModelKind::EwClass { .. } => (Want::Text, "label"),
            _ => (Want::Number, "target"),
        };
        let mut reads: Vec<(&str, Want, &'static str)> = s
            .features
            .iter()
            .map(|f| (f.as_str(), Want::Feature, "feature"))
            .collect();
        if s.model.compares().is_none() {
            for t in s.targets.defs() {
                if let Some(tree) = &t.formula {
                    for c in tree.columns_ref() {
                        reads.push((c, Want::Value, "formula target"));
                    }
                    continue;
                }
                reads.push((t.column.as_str(), target_want, target_role));
            }
        }
        reads.extend(s.clock.as_deref().map(|c| (c, Want::Clock, "clock")));
        reads.extend(
            s.label_clock
                .as_ref()
                .map(|l| (l.column(), Want::Clock, "label_clock")),
        );
        reads.extend(s.weight.as_deref().map(|c| (c, Want::Number, "weight")));
        reads.extend(s.session.as_deref().map(|c| (c, Want::Text, "session")));
        reads.extend(s.group.as_deref().map(|c| (c, Want::Key, "group")));
        for (column, want, role) in reads {
            out.entry((column, want)).or_insert((name, role));
        }
    }
    out
}

/// The form [`cast_to`] produces for a role on a dtype, known before the cast:
/// a duplicate can then be skipped without paying for the cast it would throw
/// away. Must agree with [`ArrowCol::form`].
fn form_of(want: Want, dtype: &DataType) -> &'static str {
    match want {
        Want::Number | Want::Feature => "a number",
        Want::Text => "text",
        Want::Key | Want::Clock if fits_64(dtype) => "an integer key",
        Want::Key => "text",
        Want::Clock if dtype.is_temporal() => "a temporal clock",
        Want::Clock => "a number",
        Want::Value if value_is_number(dtype) => "a number",
        Want::Value if *dtype == DataType::Boolean => "a boolean",
        Want::Value => "text",
    }
}

/// Whether a formula target reads a column as a number (a boolean stays a
/// boolean, anything else is text).
fn value_is_number(dtype: &DataType) -> bool {
    dtype.is_numeric() || matches!(dtype, DataType::Null)
}

/// One `Series` as the Arrow array of a given form.
fn cast_to(
    s: &Series,
    want: Want,
    spec_name: &str,
    role: &str,
    name: &str,
) -> PolarsResult<ArrowCol> {
    match want {
        Want::Number => {
            let dtype = s.dtype();
            if !(dtype.is_numeric() || matches!(dtype, DataType::Boolean | DataType::Null)) {
                polars_bail!(InvalidOperation:
                    "spec {:?}: {} column {:?} must be a number, but it is {}. Cast it, such \
                     as with pl.col({:?}).cast(pl.Float64).",
                    spec_name, role, name, dtype_name(dtype), name
                );
            }
            let f = s.cast(&DataType::Float64)?.rechunk();
            let ca = f.f64()?;
            let arr = ca
                .downcast_iter()
                .next()
                .cloned()
                .unwrap_or_else(|| Float64Array::new_empty(ArrowDataType::Float64));
            Ok(ArrowCol::F64(arr))
        }
        Want::Feature => match NumArray::of(s)? {
            Some(a) => Ok(ArrowCol::Num(a)),
            None => cast_to(s, Want::Number, spec_name, role, name),
        },
        Want::Text => Ok(ArrowCol::Str(text_array(s, spec_name, role, name)?)),
        // A temporal clock is read in nanoseconds before a cast is asked
        // for (`chunk_from_frame_at`).
        Want::Clock if fits_64(s.dtype()) => cast_to(s, Want::Key, spec_name, role, name),
        Want::Clock => cast_to(s, Want::Number, spec_name, role, name),
        Want::Value => {
            let dtype = s.dtype();
            if dtype.is_temporal() {
                polars_bail!(InvalidOperation:
                    "spec {:?}: {} column {:?} is {}, but a formula target reads numbers, \
                     booleans and text, and a temporal column can only be a clock. Cast it to a number, such \
                     as with pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64), or \
                     pl.col({:?}).dt.total_seconds() for a Duration.",
                    spec_name, role, name, dtype_name(dtype), name, name
                );
            }
            if value_is_number(dtype) {
                return cast_to(s, Want::Number, spec_name, role, name);
            }
            if *dtype == DataType::Boolean {
                let r = s.rechunk();
                let ca = r.bool()?;
                return Ok(ArrowCol::Bool(
                    ca.downcast_iter()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| BooleanArray::new_empty(ArrowDataType::Boolean)),
                ));
            }
            Ok(ArrowCol::Str(text_array(s, spec_name, role, name)?))
        }
        Want::Key => {
            // An integer wider than 64 bits is its text, as a Decimal is: a
            // non-strict cast to Int64 made a value past i64 a null key,
            // merged with the real nulls in silence (task 160, PA3).
            if fits_64(s.dtype()) {
                return Ok(if *s.dtype() == DataType::UInt64 {
                    let r = s.rechunk();
                    let ca = r.u64()?;
                    ArrowCol::U64(
                        ca.downcast_iter()
                            .next()
                            .cloned()
                            .unwrap_or_else(|| UInt64Array::new_empty(ArrowDataType::UInt64)),
                    )
                } else {
                    let r = s.cast(&DataType::Int64)?.rechunk();
                    let ca = r.i64()?;
                    ArrowCol::I64(
                        ca.downcast_iter()
                            .next()
                            .cloned()
                            .unwrap_or_else(|| Int64Array::new_empty(ArrowDataType::Int64)),
                    )
                });
            }
            Ok(ArrowCol::Str(text_array(s, spec_name, role, name)?))
        }
    }
}

/// An integer dtype a group key keeps as an integer: one of the 64-bit
/// forms holds every value of it. A wider one (`Int128`) is read as its
/// text, which holds every value too, and is still ordered as a number
/// ([`ArrowChunk::is_integer_text`]).
pub(crate) fn fits_64(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// A dtype as a reader writes it in Python: Polars' class names, `String`,
/// `Float64`, `Datetime("us")`, `List(Float64)`, where the Rust display is
/// `str`, `f64`, `datetime[μs]` and `list[f64]`. Every message that prints a
/// dtype says it this way (WRITING §11: the reader's own words).
pub(crate) fn dtype_name(dtype: &DataType) -> String {
    let unit = |tu: &TimeUnit| match tu {
        TimeUnit::Milliseconds => "ms",
        TimeUnit::Microseconds => "us",
        TimeUnit::Nanoseconds => "ns",
    };
    match dtype {
        DataType::Boolean => "Boolean".into(),
        DataType::UInt8 => "UInt8".into(),
        DataType::UInt16 => "UInt16".into(),
        DataType::UInt32 => "UInt32".into(),
        DataType::UInt64 => "UInt64".into(),
        DataType::UInt128 => "UInt128".into(),
        DataType::Int8 => "Int8".into(),
        DataType::Int16 => "Int16".into(),
        DataType::Int32 => "Int32".into(),
        DataType::Int64 => "Int64".into(),
        DataType::Int128 => "Int128".into(),
        DataType::Float32 => "Float32".into(),
        DataType::Float64 => "Float64".into(),
        DataType::String => "String".into(),
        DataType::Binary => "Binary".into(),
        DataType::Date => "Date".into(),
        DataType::Time => "Time".into(),
        DataType::Null => "Null".into(),
        DataType::Datetime(tu, None) => format!("Datetime(\"{}\")", unit(tu)),
        DataType::Datetime(tu, Some(tz)) => format!("Datetime(\"{}\", \"{tz}\")", unit(tu)),
        DataType::Duration(tu) => format!("Duration(\"{}\")", unit(tu)),
        DataType::List(inner) => format!("List({})", dtype_name(inner)),
        DataType::Categorical(..) => "Categorical".into(),
        DataType::Enum(..) => "Enum".into(),
        DataType::Struct(_) => "Struct".into(),
        DataType::Decimal(..) => "Decimal".into(),
        other => other.to_string(),
    }
}

/// The refusal of an integer clock wider than 64 bits, `who` the caller as
/// its own errors name it (review round 5, B3). An integer clock is held as
/// an `i64` (task 200); an `Int128` was read as a double instead, which
/// rounds its steps of 1 away past 2^53 -- a step of 0, a window holding a
/// row it should not -- and said nothing. The advice is the one a `UInt64`
/// past the largest `Int64` gets.
pub(crate) fn wide_integer_clock(who: &str, clock: &str, dtype: &DataType) -> String {
    let name = dtype_name(dtype);
    format!(
        "{who}: clock column {clock:?} is {name}, wider than the Int64 an integer clock is \
         held in. Cast it to Int64 if its values fit, or after subtracting an origin."
    )
}

/// A key column as the text its keys are: each value's string form, which
/// is what a group, a session and a label are read as. A zoned Datetime is
/// its instant, written as the UTC wall time without the zone,
/// `2024-10-27 00:30:00.000000`: this build formats no time zone, and the
/// cast to text failed on one with polars' own message. So two zones
/// showing one instant give one key, and two instants one zone shows at
/// the same wall time, as a clock goes back an hour, give two (task 160,
/// PA4 and PA4b). The same text as Polars' `cast(pl.Datetime(unit))` then
/// `cast(pl.String)`, which keep the instant.
pub(crate) fn key_text(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(unit, Some(_)) => s
            .cast(&DataType::Datetime(*unit, None))?
            .cast(&DataType::String),
        _ => s.cast(&DataType::String),
    }
}

/// A key or a label as text ([`key_text`]). Any dtype with a string form is
/// a key (ints, dates, zoned Datetimes and categoricals included); a nested
/// one is refused by name.
fn text_array(s: &Series, spec_name: &str, role: &str, name: &str) -> PolarsResult<Utf8ViewArray> {
    let cast = key_text(s).map_err(|e| {
        polars_err!(InvalidOperation:
            "spec {:?}: {} column {:?} is {}, which cannot be read as text ({}). Give a column \
             with one value per row, such as String or an integer; join a list's items into \
             one, such as with pl.col({:?}).cast(pl.List(pl.String)).list.join(\",\").",
            spec_name, role, name, dtype_name(s.dtype()), e, name
        )
    })?;
    let r = cast.rechunk();
    let ca = r.str()?;
    Ok(ca
        .downcast_iter()
        .next()
        .cloned()
        .unwrap_or_else(|| Utf8ViewArray::new_empty(ArrowDataType::Utf8View)))
}

/// A frame as an [`ArrowChunk`], reading only what the specs name.
///
/// This is the polars adapter: every dtype decision the bank used to make is
/// made here, so the bank itself sees numbers and text and nothing else.
///
pub fn chunk_from_frame(df: &DataFrame, specs: &[Spec]) -> PolarsResult<ArrowChunk> {
    chunk_from_frame_at(df, specs, 0)
}

/// [`chunk_from_frame`] for rows `row_base..` of a longer input, so that an
/// error names the input's row ([`ArrowChunk::with_row_base`]).
pub fn chunk_from_frame_at(
    df: &DataFrame,
    specs: &[Spec],
    row_base: usize,
) -> PolarsResult<ArrowChunk> {
    let names: Vec<PlSmallStr> = df.get_column_names().iter().map(|n| (*n).clone()).collect();
    // `group_close = "monotone"` reads keys in the *column's* order -- an
    // integer column numerically, a text one bytewise -- so a dtype with
    // neither order is refused here. It would otherwise be cast to text and
    // compared as its string form, where 9.0 sorts after 10.0 and a closed
    // group could reopen.
    for spec in specs.iter().filter(|s| s.closes_monotone()) {
        let Some(g) = &spec.group else { continue };
        // A column the frame has not got is `group_indices`' error to name.
        let Ok(col) = df.column(g.as_str()) else {
            continue;
        };
        let dt = col.dtype();
        if !(dt.is_integer() || matches!(dt, DataType::String | DataType::Categorical(..))) {
            polars_bail!(InvalidOperation:
                "spec {:?}: group_close = \"monotone\" needs a group column it can order, but \
                 {:?} is {}. Cast it to an integer, String or Categorical column.",
                spec.name, g, dtype_name(dt)
            );
        }
    }
    check_clocks(df, specs)?;
    let readers = first_readers(specs);
    let by_want = want_readers(specs);
    let mut cols: Vec<(PlSmallStr, ArrowCol)> = Vec::new();
    let mut clock_dtypes: Vec<(PlSmallStr, DataType)> = Vec::new();
    let mut integer_text: Vec<PlSmallStr> = Vec::new();
    // Every key column's dtype as the frame has it, before any cast: what
    // decides its keys' text, which the bank keeps the form of (N22).
    let mut key_dtypes: Vec<(PlSmallStr, DataType)> = Vec::new();
    for c in specs
        .iter()
        .flat_map(|s| [s.group.as_deref(), s.session.as_deref()])
        .flatten()
    {
        if let Ok(col) = df.column(c)
            && !key_dtypes.iter().any(|(n, _)| n.as_str() == c)
        {
            key_dtypes.push((c.into(), col.dtype().clone()));
        }
    }
    let wanted = wanted(specs);
    // The columns some role other than a feature reads as a number: a
    // feature on one of them shares that role's `Float64` array, rather
    // than holding a second form of the same column.
    let numbers: PlHashSet<&str> = wanted
        .iter()
        .filter(|(name, want)| {
            *want != Want::Feature
                && df
                    .column(name.as_str())
                    .is_ok_and(|c| form_of(*want, c.dtype()) == "a number")
        })
        .map(|(name, _)| name.as_str())
        .collect();
    let mut have: PlHashSet<(PlSmallStr, &'static str)> = PlHashSet::default();
    for (name, asked) in wanted.iter().cloned() {
        let want = match asked {
            Want::Feature if numbers.contains(name.as_str()) => Want::Number,
            w => w,
        };
        // A column a scoring chunk may leave out is not an error here: the
        // bank decides, because whether it is optional depends on the call.
        let Some(col) = df.column(name.as_str()).ok() else {
            continue;
        };
        // The spec and role of this read, for the errors its cast raises.
        let (spec, role) = by_want
            .get(&(name.as_str(), asked))
            .or_else(|| readers.get(name.as_str()))
            .copied()
            .unwrap_or(("", "column"));
        let s = col.as_materialized_series();
        // A temporal clock is read in its own nanoseconds whatever the
        // column's unit, so the gap between two rows is taken in integers
        // and a nanosecond timestamp's gaps stay exact for the stream's
        // life (`online_core::ClockValue`). Casting it to a number instead
        // would expose its *internal representation*, so the same 60
        // seconds would be 60_000 / 60_000_000 / 60_000_000_000 clock units
        // for Datetime(ms/us/ns) (docs/records/TESTING-LOG.md T-E10). `check_clocks` has
        // already refused a spec that gives this clock plain numbers.
        if want == Want::Clock && s.dtype().is_temporal() {
            let who = format!("spec {spec:?}");
            let col = ArrowCol::Nanos(nanos_array(s, row_base, NanosRole::Clock, &who)?);
            clock_dtypes.push((name.clone(), s.dtype().clone()));
            have.insert((name.clone(), col.form()));
            cols.push((name.clone(), col));
            continue;
        }
        // An integer clock is read as its integers, the form a key keeps
        // them in, so its steps are taken in integers (task 200): cast to a
        // double, an epoch-nanosecond column resolved 256 of them. Its dtype
        // is kept for the clock fields, whether or not a key of the same
        // column has made the array already.
        if want == Want::Clock && fits_64(s.dtype()) {
            clock_dtypes.push((name.clone(), s.dtype().clone()));
        }
        // Marked whether or not the cast below runs: a session read from the
        // same column may have made the text already.
        if want == Want::Key && s.dtype().is_integer() && !fits_64(s.dtype()) {
            integer_text.push(name.clone());
        }
        // One column read in two roles that cast to the same form -- a
        // column that is a group key for one spec and a session for another,
        // both text -- would cast identically twice. `ArrowChunk::new` refuses
        // a repeated `(name, form)` because a hand-built one hides a caller's
        // mistake; here the two are the same array from the same source, so
        // the second role is skipped -- before its cast, since the form is
        // known from the role and the dtype. A key and a feature on the same
        // column cast to *different* forms and both stay.
        if have.contains(&(name.clone(), form_of(want, s.dtype()))) {
            continue;
        }
        let col = cast_to(s, want, spec, role, name.as_str())?;
        have.insert((name.clone(), col.form()));
        cols.push((name.clone(), col));
    }
    Ok(ArrowChunk::new(df.height(), cols, names)?
        .with_row_base(row_base)
        .with_clock_dtypes(clock_dtypes)
        .with_integer_text_keys(integer_text)
        .with_key_dtypes(key_dtypes))
}

/// Each column's first reader, for the errors a cast can raise: the first
/// spec, in bank order, that reads the column in any role, and the first of
/// its roles in the order features, targets (each target's column, or its
/// formula's columns), clock, weight, session, group.
/// Built once, where a scan of every spec's lists per column was quadratic
/// in the columns (docs/records/PERFORMANCE-LOG.md §24).
fn first_readers(specs: &[Spec]) -> PlHashMap<&str, (&str, &'static str)> {
    let mut out: PlHashMap<&str, (&str, &'static str)> = PlHashMap::default();
    for s in specs {
        let name = s.name.as_str();
        let target_role = match &s.model {
            ModelKind::EwClass { .. } => "label",
            _ => "target",
        };
        let roles = s
            .features
            .iter()
            .map(|c| (c.as_str(), "feature"))
            .chain(s.targets.defs().iter().flat_map(|t| {
                let formula: Vec<(&str, &'static str)> = match &t.formula {
                    Some(tree) => tree
                        .columns_ref()
                        .into_iter()
                        .map(|c| (c, "formula target"))
                        .collect(),
                    None => Vec::new(),
                };
                t.value_column()
                    .map(|c| (c, target_role))
                    .into_iter()
                    .chain(formula)
            }))
            .chain(s.clock.as_deref().map(|c| (c, "clock")))
            .chain(s.label_clock.as_ref().map(|l| (l.column(), "label_clock")))
            .chain(s.weight.as_deref().map(|c| (c, "weight")))
            .chain(s.session.as_deref().map(|c| (c, "session")))
            .chain(s.group.as_deref().map(|c| (c, "group")));
        for (column, role) in roles {
            out.entry(column).or_insert((name, role));
        }
    }
    out
}

/// Each clock column against the specs that read it (docs/PLAN.md task 88).
/// A temporal clock measures durations and a numeric one plain numbers of
/// its own units; a spec that gives a clock the other kind is refused,
/// naming the column, the parameter and the fix, before anything is cast.
fn check_clocks(df: &DataFrame, specs: &[Spec]) -> PolarsResult<()> {
    for spec in specs {
        let Some(clock) = spec.clock.as_deref() else {
            continue;
        };
        // A column the frame has not got is another check's error to name.
        let Ok(col) = df.column(clock) else {
            continue;
        };
        let dtype = col.dtype();
        check_label_clock(df, spec, clock, dtype)?;
        let scale = spec
            .clock_scale()
            .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        if matches!(dtype, DataType::Time) {
            polars_bail!(InvalidOperation:
                "spec {:?}: clock column {:?} is a time of day, which starts again at \
                 midnight, so it cannot be a clock. Combine it with its date into a Datetime, \
                 such as with pl.col(\"date\").dt.combine(pl.col({:?})).",
                spec.name, clock, clock
            );
        }
        if dtype.is_integer() && !fits_64(dtype) {
            polars_bail!(InvalidOperation: "{}",
                wide_integer_clock(&format!("spec {:?}", spec.name), clock, dtype));
        }
        if dtype.is_temporal() {
            if let ClockScale::Numbers(param @ ("lam" | "q")) = scale {
                let (instead, what) = if param == "lam" {
                    ("half_life", "lam is a decay per clock unit")
                } else {
                    ("coef_half_life", "q is the noise per clock unit")
                };
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} is {}, a temporal clock, but {}, which has \
                     no duration form. Leave out {} and \
                     give {} as a duration, such as \"10m\" or pl.duration(minutes=10), or cast \
                     the clock to a number of the units you mean, such as with \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64) (pl.col({:?}).dt.total_seconds() \
                     for a Duration).",
                    spec.name, clock, dtype_name(dtype), what, param, instead, clock, clock
                );
            }
            if let ClockScale::Numbers(param) = scale {
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} is {}, a temporal clock, but {} is a plain \
                     number. A number has no unit, so the unit the column is stored in, such as \
                     microseconds since 1970, would silently become its unit. Give {} and the other clock \
                     parameters as durations, such as \"10m\" or pl.duration(minutes=10), or \
                     cast the clock to a number of the units you mean, such as with \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64) (pl.col({:?}).dt.total_seconds() \
                     for a Duration).",
                    spec.name, clock, dtype_name(dtype), param, param, clock, clock
                );
            }
            // A cap or a late-row threshold finer than the clock's own step
            // cannot act on the data: every step would be cut to the cap, so
            // the clock would count rows, and no step back could be as small
            // as the threshold, so every one would start the model over. A
            // threshold of exactly one step does act: the comparison is
            // inclusive (task 120).
            let tick: f64 = match dtype {
                DataType::Date => 86_400.0,
                DataType::Datetime(tu, _) | DataType::Duration(tu) => match tu {
                    TimeUnit::Milliseconds => 1e-3,
                    TimeUnit::Microseconds => 1e-6,
                    TimeUnit::Nanoseconds => 1e-9,
                },
                _ => 0.0,
            };
            let step = match dtype {
                DataType::Date => "a day",
                DataType::Datetime(TimeUnit::Milliseconds, _)
                | DataType::Duration(TimeUnit::Milliseconds) => "a millisecond",
                DataType::Datetime(TimeUnit::Microseconds, _)
                | DataType::Duration(TimeUnit::Microseconds) => "a microsecond",
                _ => "a nanosecond",
            };
            for (param, span) in spec.clock_spans() {
                let v = span.value();
                if !(span.is_duration() && v > 0.0 && v < tick) {
                    continue;
                }
                let (why, or_zero) = match param {
                    "gap_cap" => (
                        "every step would be cut to it and the clock would count rows rather \
                         than measure time",
                        "",
                    ),
                    "restart_after_step_back" => (
                        "no step back could be as small and every one would reset the model",
                        ", or to 0 to reset at every step back",
                    ),
                    _ => continue,
                };
                polars_bail!(ComputeError:
                    "spec {:?}: {} is {}, less than {}, the smallest step clock column {:?} \
                     ({}) can take, so {}. Set {} to at least {}{}.",
                    spec.name, param, span, step, clock, dtype_name(dtype), why, param, step, or_zero
                );
            }
            // Read in seconds, a temporal column would feed any other
            // numeric role a number the spec never asked for.
            for other in specs {
                // By column, not name (review 2026-09-26, D6).
                let defs = other.targets.defs();
                let (role, must) = if other.features.iter().any(|f| f == clock) {
                    ("a feature", "a number")
                } else if other.weight.as_deref() == Some(clock) {
                    ("a weight", "a number")
                } else if defs.iter().any(|t| t.column == *clock)
                    && other.model.compares().is_none()
                    && !matches!(other.model, ModelKind::EwClass { .. })
                {
                    ("a target", "a number")
                } else if defs
                    .iter()
                    .any(|t| t.is_formula() && t.columns().iter().any(|c| c == clock))
                {
                    (
                        "a column of a formula target",
                        "a number, a boolean or text",
                    )
                } else {
                    continue;
                };
                polars_bail!(InvalidOperation:
                    "spec {:?}: it reads column {:?} as {}, which must be {must}, but the \
                     column is {}, and a temporal column can only be a clock. Add a number \
                     column for that use, such as \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64).alias(\"{}_s\") \
                     (pl.col({:?}).dt.total_seconds() for a Duration), and read {}_s as {}.",
                    other.name, clock, role, dtype_name(dtype), clock, clock, clock, clock, role
                );
            }
        } else if let ClockScale::Durations(param) = scale {
            polars_bail!(ComputeError:
                "spec {:?}: {} is a duration, but clock column {:?} is {}, which has no unit \
                 to measure it in. Use a temporal clock, a Datetime, Date or Duration column \
                 such as pl.from_epoch({:?}, time_unit=\"s\"), or give {} as a number of the \
                 clock's own units.",
                spec.name, param, clock, dtype_name(dtype), clock, param
            );
        }
    }
    Ok(())
}

/// A spec's `label_clock` column against its clock (task 341): a label
/// time is a place on the clock, so a temporal clock takes a temporal label
/// column and a number clock a numeric one. A column the frame has not got
/// is the bank's missing-column error to name.
fn check_label_clock(
    df: &DataFrame,
    spec: &Spec,
    clock: &str,
    dtype: &DataType,
) -> PolarsResult<()> {
    let Some(label) = spec.label_clock.as_ref().map(|l| l.column()) else {
        return Ok(());
    };
    let Ok(col) = df.column(label) else {
        return Ok(());
    };
    let ldt = col.dtype();
    let fits = if dtype.is_temporal() {
        ldt.is_temporal() && !matches!(ldt, DataType::Time)
    } else {
        ldt.is_numeric() && !(ldt.is_integer() && !fits_64(ldt))
    };
    if !fits {
        let want = if dtype.is_temporal() {
            "a Datetime, Date or Duration column"
        } else {
            "a number column of at most 64 bits, in the clock's own units"
        };
        polars_bail!(InvalidOperation:
            "spec {:?}: label_clock column {:?} is {}, and clock column {:?} is {}, so its \
             values are not times on that clock. Give label_clock {want}.",
            spec.name, label, dtype_name(ldt), clock, dtype_name(dtype)
        );
    }
    Ok(())
}

/// What a temporal column is read in nanoseconds for, which decides the
/// dtypes it may have and what a refusal names (review round 4, PD5: one
/// message about a clock served every caller, an increment's input
/// included).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NanosRole {
    /// A clock, whose differences are time elapsed: an instant or a span.
    Clock,
    /// A window formula's `increment` input, whose differences are steps: a
    /// time of day as well.
    Increment,
}

impl NanosRole {
    fn what(self) -> &'static str {
        match self {
            NanosRole::Clock => "a clock",
            NanosRole::Increment => "an increment's input",
        }
    }
}

/// Nanoseconds in one unit of a temporal column: a `Date` counts days, and a
/// `Time`, read as an increment's input only, nanoseconds since midnight, so
/// a step across midnight is negative, as Polars' `diff` on a `Time` gives
/// it. A `Time` clock is refused: it starts again every midnight.
fn nanos_per_unit(s: &Series, role: NanosRole, who: &str) -> PolarsResult<i64> {
    let dtype = s.dtype();
    Ok(match dtype {
        DataType::Datetime(TimeUnit::Milliseconds, _)
        | DataType::Duration(TimeUnit::Milliseconds) => 1_000_000,
        DataType::Datetime(TimeUnit::Microseconds, _)
        | DataType::Duration(TimeUnit::Microseconds) => 1_000,
        DataType::Datetime(TimeUnit::Nanoseconds, _)
        | DataType::Duration(TimeUnit::Nanoseconds) => 1,
        DataType::Date => 86_400 * 1_000_000_000,
        DataType::Time if role == NanosRole::Increment => 1,
        DataType::Time => polars_bail!(InvalidOperation:
            "{who}: clock column {:?} is a time of day, which starts again at midnight, so it \
             cannot be a clock. Combine it with its date into a Datetime, such as with \
             pl.col(\"date\").dt.combine(pl.col({:?})).",
            s.name(), s.name()),
        dt => polars_bail!(InvalidOperation:
            "{who}: column {:?} is {}, which cannot be read as {}. Use a Datetime, Date or \
             Duration column.",
            s.name(), dtype_name(dt), role.what()),
    })
}

/// A temporal column as nanoseconds since the Unix epoch, null where it is
/// null. The unit is scaled away in integers, so one instant is the same
/// value whether it was stored in milliseconds, microseconds or
/// nanoseconds, and a timezone changes nothing: a `Datetime` is stored in
/// UTC, so a change of clocks for summer time neither stretches nor folds
/// the clock. A `Date` or a coarse `Datetime` can reach past what
/// nanoseconds in an `i64` hold, and such a value is refused by row. A
/// nanosecond column is taken as it is, without a pass over it. `role` is
/// what the column is read for ([`NanosRole`]).
pub(crate) fn nanos_array(
    s: &Series,
    row_base: usize,
    role: NanosRole,
    who: &str,
) -> PolarsResult<Int64Array> {
    let per = nanos_per_unit(s, role, who)?;
    let phys = s.to_physical_repr();
    let too_far = |i: usize| {
        if matches!(s.dtype(), DataType::Duration(_)) {
            polars_err!(ComputeError:
                "{who}: column {:?} holds a duration at row {} (counting from 0) longer than \
                 about 292 years, which nanoseconds cannot hold. Drop or correct the rows with \
                 such a duration.",
                s.name(), row_base + i
            )
        } else {
            polars_err!(ComputeError:
                "{who}: column {:?} holds an instant at row {} (counting from 0) before 21 \
                 September 1677 or after 11 April 2262, which nanoseconds cannot hold. Drop or \
                 correct the rows outside those dates.",
                s.name(), row_base + i
            )
        }
    };
    let scaled = |i: usize, v: i64| v.checked_mul(per).ok_or_else(|| too_far(i));
    let ns: Int64Chunked = match s.dtype() {
        DataType::Date => phys
            .i32()?
            .iter()
            .enumerate()
            .map(|(i, d)| d.map(|d| scaled(i, i64::from(d))).transpose())
            .collect::<PolarsResult<_>>()?,
        _ if per == 1 => phys.i64()?.clone(),
        _ => phys
            .i64()?
            .iter()
            .enumerate()
            .map(|(i, v)| v.map(|v| scaled(i, v)).transpose())
            .collect::<PolarsResult<_>>()?,
    };
    let ns = ns.rechunk();
    Ok(ns
        .downcast_iter()
        .next()
        .cloned()
        .unwrap_or_else(|| Int64Array::new_empty(ArrowDataType::Int64)))
}

/// The scans [`first_readers`] replaced, kept as its oracle.
#[cfg(test)]
fn reads(s: &Spec, name: &str) -> bool {
    s.features.iter().any(|f| f == name)
        || s.targets
            .defs()
            .iter()
            .any(|t| t.columns().iter().any(|c| c == name))
        || s.clock.as_deref() == Some(name)
        || s.weight.as_deref() == Some(name)
        || s.session.as_deref() == Some(name)
        || s.group.as_deref() == Some(name)
}

/// The role to name in an error, from the first spec that reads the column.
#[cfg(test)]
fn role_of(specs: &[Spec], name: &str) -> &'static str {
    for s in specs {
        if s.features.iter().any(|f| f == name) {
            return "feature";
        }
        for t in s.targets.defs() {
            if t.value_column() == Some(name) {
                return match &s.model {
                    ModelKind::EwClass { .. } => "label",
                    _ => "target",
                };
            }
            if t.is_formula() && t.columns().iter().any(|c| c == name) {
                return "formula target";
            }
        }
        if s.clock.as_deref() == Some(name) {
            return "clock";
        }
        if s.weight.as_deref() == Some(name) {
            return "weight";
        }
        if s.session.as_deref() == Some(name) {
            return "session";
        }
        if s.group.as_deref() == Some(name) {
            return "group";
        }
    }
    "column"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(json: &str) -> Spec {
        serde_json::from_str(json).unwrap()
    }

    /// Task 160, PA4b: a zoned Datetime is keyed by its instant, as the UTC
    /// wall time with no zone. Two zones showing one instant give one key,
    /// and the two instants Amsterdam shows at 02:30 as its clock goes back
    /// give two. The cast to text failed: this build formats no zone.
    #[test]
    fn a_zoned_datetime_key_is_its_instant() {
        // 2024-10-27 00:30 and 01:30 UTC, in microseconds.
        let instants = Int64Chunked::new(
            "g".into(),
            [1_729_989_000_000_000i64, 1_729_992_600_000_000],
        );
        let naive = instants
            .clone()
            .into_datetime(TimeUnit::Microseconds, None)
            .into_series();
        assert!(naive.cast(&DataType::String).is_ok());
        let want = key_text(&naive).unwrap();
        let text: Vec<Option<&str>> = want.str().unwrap().iter().collect();
        assert_eq!(
            text,
            vec![
                Some("2024-10-27 00:30:00.000000"),
                Some("2024-10-27 01:30:00.000000")
            ]
        );
        for tz in ["Europe/Amsterdam", "America/New_York", "UTC"] {
            let zoned = instants
                .clone()
                .into_datetime(
                    TimeUnit::Microseconds,
                    TimeZone::opt_try_new(Some(tz)).unwrap(),
                )
                .into_series();
            assert!(
                zoned.cast(&DataType::String).is_err(),
                "{tz}: still unformattable"
            );
            assert!(key_text(&zoned).unwrap().equals(&want), "{tz}");
        }
    }

    /// Review round 4, PD5: a time of day is read as an increment's input,
    /// in nanoseconds since midnight, and never as a clock; and a refusal
    /// names the role the column was read for, where one message about a
    /// clock served every caller.
    #[test]
    fn a_time_of_day_is_an_increments_input_and_never_a_clock() {
        let since_midnight = [Some(32_400_000_000_000i64), None, Some(3_600_000_000_500)];
        let tod = Int64Chunked::new("tod".into(), since_midnight)
            .into_time()
            .into_series();
        let ns = nanos_array(&tod, 0, NanosRole::Increment, "w").unwrap();
        let read: Vec<Option<i64>> = (0..ns.len()).map(|i| ns.get(i)).collect();
        assert_eq!(read, since_midnight);
        let err = nanos_array(&tod, 0, NanosRole::Clock, "w").unwrap_err();
        assert!(
            err.to_string().contains(
                "w: clock column \"tod\" is a time of day, which starts again at midnight"
            ),
            "{err}"
        );
        let text = Series::new("s".into(), ["a"]);
        let err = nanos_array(&text, 0, NanosRole::Increment, "w").unwrap_err();
        assert!(
            err.to_string().contains(
                "w: column \"s\" is String, which cannot be read as an increment's input"
            ),
            "{err}"
        );
        // The other temporal dtypes read alike in both roles.
        let date = Int32Chunked::new("d".into(), [1i32])
            .into_date()
            .into_series();
        for role in [NanosRole::Clock, NanosRole::Increment] {
            assert_eq!(
                nanos_array(&date, 0, role, "w").unwrap().value(0),
                86_400_000_000_000
            );
        }
    }

    /// `first_readers` names, for every column, the spec and the role the
    /// two scans it replaced named: the first spec in bank order that reads
    /// the column, and the first of that spec's roles. Specs that share
    /// columns across roles, a label, and a column no spec reads.
    #[test]
    fn the_first_reader_is_the_one_the_scans_found() {
        let specs = vec![
            spec(
                r#"{"name": "a", "model": {"type": "ewridge"}, "targets": ["y", "g"],
                    "features": ["x0", "x1"], "clock": "t", "weight": "w", "group": "g"}"#,
            ),
            spec(
                r#"{"name": "b", "model": {"type": "ew_class", "classes": ["u", "v"],
                    "precision_prior": 1.0},
                    "targets": ["lab"], "features": ["x1", "t", "s"], "session": "s"}"#,
            ),
            spec(
                r#"{"name": "c", "model": {"type": "ewridge"}, "targets": ["x0"],
                    "features": ["w", "lab"], "group": "k"}"#,
            ),
            // A renamed target: its column is a target, its name no column
            // at all, unless an earlier spec read the column first.
            spec(
                r#"{"name": "d", "model": {"type": "ewridge"},
                    "targets": [{"column": "q", "name": "r"},
                                {"column": "x1", "name": "t2"}],
                    "features": ["w"]}"#,
            ),
        ];
        let readers = first_readers(&specs);
        let columns = [
            "x0", "x1", "y", "g", "t", "w", "lab", "s", "k", "q", "r", "t2", "absent",
        ];
        for name in columns {
            let want_spec = specs
                .iter()
                .find(|s| reads(s, name))
                .map_or("", |s| s.name.as_str());
            let want = (want_spec, role_of(&specs, name));
            let got = readers.get(name).copied().unwrap_or(("", "column"));
            assert_eq!(got, want, "column {name}");
        }
        assert_eq!(
            readers.get("g"),
            Some(&("a", "target")),
            "a target before a group"
        );
        assert_eq!(readers.get("lab"), Some(&("b", "label")));
        assert_eq!(readers.get("k"), Some(&("c", "group")));
        assert_eq!(readers.get("q"), Some(&("d", "target")));
        assert!(!readers.contains_key("r") && !readers.contains_key("t2"));
        assert_eq!(
            readers.get("x1"),
            Some(&("a", "feature")),
            "an earlier reader wins"
        );
        assert!(!readers.contains_key("absent"));
    }
}
