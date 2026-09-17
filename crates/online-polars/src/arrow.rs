//! A chunk as Arrow, and the polars adapter that makes one (docs/PLAN.md
//! task 86).
//!
//! The bank reads a chunk in exactly two forms: numbers as `f64`, and keys as
//! text. [`ArrowChunk`] is that, and nothing more -- a caller holding Arrow
//! arrays can build one and feed the bank without polars in the process.
//!
//! Everything *polars-shaped* about reading a frame lives here rather than in
//! the bank: finding a column by name, refusing a dtype that cannot be read,
//! casting to `Float64` or to text, and deciding whether a group key is an
//! integer. Everything *model-shaped* -- a clock that must be finite, a weight
//! that must not be negative, a label that must be one of the declared classes
//! -- stays in the bank, because it is true whoever supplies the data.

use polars::prelude::*;
// The trait is brought in unnamed: `ArrowArray` below is the C Data Interface
// struct, not `array::Array`, and the two must not collide.
use polars_arrow::array::Array as _;
use polars_arrow::array::{Float64Array, Int64Array, StructArray, UInt64Array, Utf8ViewArray};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{ArrowArray, ArrowSchema, export_array_to_c, export_field_to_c};

use crate::spec::{ModelKind, Spec};

/// One column of an [`ArrowChunk`], in the form the bank reads it.
#[derive(Clone, Debug)]
pub enum ArrowCol {
    /// A number: null is a NaN to every consumer, so the validity is applied
    /// when the values are read rather than carried alongside.
    F64(Float64Array),
    /// A key or a label, as text.
    Str(Utf8ViewArray),
    /// A signed integer group key. Kept as an integer so the bank can bucket
    /// on the value itself -- no cast, no hash, no collision to document
    /// (docs/PERFORMANCE.md P11).
    I64(Int64Array),
    /// The same, unsigned, for a `u64` key whose values do not fit an `i64`.
    U64(UInt64Array),
}

impl ArrowCol {
    pub fn len(&self) -> usize {
        match self {
            Self::F64(a) => a.len(),
            Self::Str(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::U64(a) => a.len(),
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
}

impl ArrowChunk {
    /// A chunk from columns the caller has already cast: numbers as
    /// `Float64Array`, keys and labels as `Utf8ViewArray`, integer group keys
    /// as `Int64Array` or `UInt64Array`.
    ///
    /// `names` is every column the source had, which need not be every column
    /// given here: it is what an error message lists and what the spec-name
    /// clash check reads.
    pub fn new(
        height: usize,
        cols: Vec<(PlSmallStr, ArrowCol)>,
        names: Vec<PlSmallStr>,
    ) -> PolarsResult<Self> {
        if let Some((name, col)) = cols.iter().find(|(_, c)| c.len() != height) {
            polars_bail!(ShapeMismatch:
                "column {:?} has {} rows, the chunk has {}",
                name.as_str(), col.len(), height
            );
        }
        Ok(Self {
            height,
            cols,
            names,
        })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn names(&self) -> &[PlSmallStr] {
        &self.names
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// The numeric form of a column, or the error naming the spec and role.
    pub fn f64(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&Float64Array> {
        match self.find(name, |c| matches!(c, ArrowCol::F64(_))) {
            Some(ArrowCol::F64(a)) => Ok(a),
            _ => Err(self.missing(spec, role, name)),
        }
    }

    /// The text form of a column, or the error naming the spec and role.
    pub fn str(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&Utf8ViewArray> {
        match self.find(name, |c| matches!(c, ArrowCol::Str(_))) {
            Some(ArrowCol::Str(a)) => Ok(a),
            _ => Err(self.missing(spec, role, name)),
        }
    }

    /// The group key column in whatever form the adapter chose for it.
    pub fn key(&self, spec: &Spec, role: &str, name: &str) -> PolarsResult<&ArrowCol> {
        self.find(name, |c| !matches!(c, ArrowCol::F64(_)))
            .ok_or_else(|| self.missing(spec, role, name))
    }

    fn find(&self, name: &str, want: impl Fn(&ArrowCol) -> bool) -> Option<&ArrowCol> {
        self.cols
            .iter()
            .find(|(n, c)| n == name && want(c))
            .map(|(_, c)| c)
    }

    /// A column lookup that says which spec asked and what it asked for.
    /// Polars' own `not found: "x"` names neither, and in a bank of ten specs
    /// that is the difference between a fix and a search.
    fn missing(&self, spec: &Spec, role: &str, name: &str) -> PolarsError {
        let have: Vec<&str> = self.names.iter().map(|n| n.as_str()).collect();
        polars_err!(ColumnNotFound:
            "spec {:?}: {} column {:?} not found; the frame has columns {:?}",
            spec.name, role, name, have
        )
    }
}

/// The values of a numeric column, null as NaN, without a copy where there is
/// nothing to fill in.
///
/// Plain `f64` with NaN for null, not `Option<f64>` (docs/PERFORMANCE.md P3):
/// every consumer already collapses the two, since a feature or weight is
/// taken only when finite and a target only when finite.
pub fn f64_values(a: &Float64Array) -> std::borrow::Cow<'_, [f64]> {
    match a.validity() {
        Some(v) if v.unset_bits() > 0 => std::borrow::Cow::Owned(
            a.values()
                .iter()
                .zip(v.iter())
                .map(|(&x, ok)| if ok { x } else { f64::NAN })
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    Number,
    Text,
    /// A group key: an integer stays an integer, anything else becomes text.
    Key,
}

/// Every column the specs read, in the form each reads it.
fn wanted(specs: &[Spec]) -> Vec<(PlSmallStr, Want)> {
    let mut out: Vec<(PlSmallStr, Want)> = Vec::new();
    let mut push = |name: &str, want: Want| {
        let key: PlSmallStr = name.into();
        if !out.iter().any(|(n, w)| *n == key && *w == want) {
            out.push((key, want));
        }
    };
    for s in specs {
        for f in &s.features {
            push(f, Want::Number);
        }
        // `ew_class` reads its target as a label; every other model as a number.
        let target_want = match &s.model {
            ModelKind::EwClass { .. } => Want::Text,
            _ => Want::Number,
        };
        if s.model.compares().is_none() {
            for t in &s.targets {
                push(t, target_want);
            }
        }
        if let Some(c) = &s.clock {
            push(c, Want::Number);
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
                polars_bail!(ComputeError:
                    "spec {:?}: {} column {:?} has dtype {}; it must be numeric \
                     (cast it, e.g. pl.col({:?}).cast(pl.Float64))",
                    spec_name, role, name, dtype, name
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
        Want::Text => Ok(ArrowCol::Str(text_array(s, spec_name, role, name)?)),
        Want::Key => {
            if s.dtype().is_integer() {
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

/// A key or a label as text. Any dtype with a string form is a key (ints,
/// dates and categoricals included); a nested one is refused by name.
fn text_array(s: &Series, spec_name: &str, role: &str, name: &str) -> PolarsResult<Utf8ViewArray> {
    let cast = s.cast(&DataType::String).map_err(|e| {
        polars_err!(ComputeError:
            "spec {:?}: {} column {:?} has dtype {}, which cannot be used as a key: {}",
            spec_name, role, name, s.dtype(), e
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
pub fn chunk_from_frame(df: &DataFrame, specs: &[Spec]) -> PolarsResult<ArrowChunk> {
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
            polars_bail!(ComputeError:
                "spec {:?}: group_close = \"monotone\" needs a group column it can order, and \
                 {:?} is {}; use an integer, String or Categorical key (a float or a temporal \
                 column can be cast to one)",
                spec.name, g, dt
            );
        }
    }
    let mut cols: Vec<(PlSmallStr, ArrowCol)> = Vec::new();
    for (name, want) in wanted(specs) {
        // A column a scoring chunk may leave out is not an error here: the
        // bank decides, because whether it is optional depends on the call.
        let Some(col) = df.column(name.as_str()).ok() else {
            continue;
        };
        let spec = specs
            .iter()
            .find(|s| reads(s, name.as_str()))
            .map_or("", |s| s.name.as_str());
        let role = role_of(specs, name.as_str());
        let s = col.as_materialized_series();
        // A temporal clock column is refused rather than cast. Casting one to
        // f64 exposes its *internal representation*, so the same 60 seconds
        // becomes 60_000 / 60_000_000 / 60_000_000_000 clock units depending
        // only on whether the column is Datetime(ms/us/ns), and a Date becomes
        // 1 unit per day. `halflife`, `max_dclock` and `session_gap` all live
        // in those units, so `halflife = 600` on a microsecond column silently
        // means 600 microseconds (docs/TESTING.md T-E10).
        if s.dtype().is_temporal() {
            if let Some(owner) = specs
                .iter()
                .find(|sp| sp.clock.as_deref() == Some(name.as_str()))
            {
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} has dtype {}; a temporal clock would be \
                     read as its internal representation (e.g. epoch microseconds), so \
                     halflife/max_dclock/session_gap would silently be in those units. \
                     Cast it to the scale you mean, e.g. \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64), and use that column.",
                    owner.name, name.as_str(), s.dtype(), name.as_str()
                );
            }
        }
        cols.push((name.clone(), cast_to(s, want, spec, role, name.as_str())?));
    }
    ArrowChunk::new(df.height(), cols, names)
}

fn reads(s: &Spec, name: &str) -> bool {
    s.features.iter().any(|f| f == name)
        || s.targets.iter().any(|t| t == name)
        || s.clock.as_deref() == Some(name)
        || s.weight.as_deref() == Some(name)
        || s.session.as_deref() == Some(name)
        || s.group.as_deref() == Some(name)
}

/// The role to name in an error, from the first spec that reads the column.
fn role_of(specs: &[Spec], name: &str) -> &'static str {
    for s in specs {
        if s.features.iter().any(|f| f == name) {
            return "feature";
        }
        if s.targets.iter().any(|t| t == name) {
            return match &s.model {
                ModelKind::EwClass { .. } => "label",
                _ => "target",
            };
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
