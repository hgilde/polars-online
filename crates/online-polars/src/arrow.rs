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

use online_core::ClockValue;

use crate::spec::{ClockScale, ModelKind, Spec};

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
    /// A temporal clock, as nanoseconds since the Unix epoch whatever the
    /// source column's unit. Kept as an integer so the gap between two rows
    /// is taken in integers and a nanosecond timestamp's gaps stay exact
    /// for the stream's life ([`online_core::ClockValue`], docs/PLAN.md
    /// task 88). Read only as a clock: a temporal column in any other role
    /// is refused before the chunk is built.
    Nanos(Int64Array),
}

impl ArrowCol {
    pub fn len(&self) -> usize {
        match self {
            Self::F64(a) => a.len(),
            Self::Str(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::U64(a) => a.len(),
            Self::Nanos(a) => a.len(),
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

    /// The form, for a message: what a caller gave, against what a role reads.
    pub fn form(&self) -> &'static str {
        match self {
            Self::F64(_) => "a number",
            Self::Str(_) => "text",
            Self::I64(_) | Self::U64(_) => "an integer key",
            Self::Nanos(_) => "a temporal clock",
        }
    }
}

/// A chunk's clock column in the form the source had it: a numeric clock's
/// numbers, or a temporal clock's nanoseconds since the Unix epoch.
#[derive(Clone, Copy, Debug)]
pub enum ClockArray<'a> {
    F64(&'a Float64Array),
    Nanos(&'a Int64Array),
}

/// A stream's clock column as the bank hands it to a stream: a
/// [`ClockArray`] read in the stream's row order, with no nulls left in it.
#[derive(Clone, Debug)]
pub enum ClockCol {
    F64(Vec<f64>),
    Ns(Vec<i64>),
}

impl ClockCol {
    /// Row `i`'s value, in the form the source had it.
    #[inline]
    pub fn at(&self, i: usize) -> ClockValue {
        match self {
            Self::F64(v) => ClockValue::F64(v[i]),
            Self::Ns(v) => ClockValue::Ns(v[i]),
        }
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
                "column {:?} has {} rows, the chunk has {}",
                name.as_str(), col.len(), height
            );
        }
        if let Some((name, _)) = cols.iter().find(|(n, _)| !names.iter().any(|m| m == n)) {
            polars_bail!(ColumnNotFound:
                "column {:?} is given but not listed in `names`; every column given must be \
                 listed, since `names` decides whether a column a scoring call may leave out \
                 is absent",
                name.as_str()
            );
        }
        let mut seen: std::collections::HashSet<(&str, &'static str)> =
            std::collections::HashSet::with_capacity(cols.len());
        for (name, col) in &cols {
            if !seen.insert((name.as_str(), col.form())) {
                polars_bail!(Duplicate:
                    "column {:?} is given twice as {}", name.as_str(), col.form()
                );
            }
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
            _ => Err(self.missing(spec, role, name, "a number")),
        }
    }

    /// The clock column in the form the source had it -- a temporal clock's
    /// nanoseconds where the adapter read one, else the numeric form -- or
    /// the error naming the spec.
    pub fn clock(&self, spec: &Spec, name: &str) -> PolarsResult<ClockArray<'_>> {
        match self.find(name, |c| matches!(c, ArrowCol::Nanos(_) | ArrowCol::F64(_))) {
            Some(ArrowCol::Nanos(a)) => Ok(ClockArray::Nanos(a)),
            Some(ArrowCol::F64(a)) => Ok(ClockArray::F64(a)),
            _ => Err(self.missing(spec, "clock", name, "a number or a temporal clock")),
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
        self.find(name, |c| matches!(c, ArrowCol::I64(_) | ArrowCol::U64(_)))
            .or_else(|| {
                self.find(name, |c| {
                    !matches!(c, ArrowCol::F64(_) | ArrowCol::Nanos(_))
                })
            })
            .ok_or_else(|| self.missing(spec, role, name, "text or an integer key"))
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
    ///
    /// A column that *is* here, in a form the role does not read -- a clock
    /// given as an integer array -- is named as such, rather than reported
    /// "not found" beside a list that includes it. The polars adapter casts to
    /// the wanted form, so that branch is only ever an Arrow caller's.
    fn missing(&self, spec: &Spec, role: &str, name: &str, wanted: &str) -> PolarsError {
        if let Some((_, c)) = self.cols.iter().find(|(n, _)| n == name) {
            return polars_err!(SchemaMismatch:
                "spec {:?}: {} column {:?} is given as {}, but a {} is read as {}",
                spec.name, role, name, c.form(), role, wanted
            );
        }
        let have: Vec<&str> = self.names.iter().map(|n| n.as_str()).collect();
        polars_err!(ColumnNotFound:
            "spec {:?}: {} column {:?} not found; the input has columns {:?}",
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

/// The form [`cast_to`] produces for a role on a dtype, known before the cast:
/// a duplicate can then be skipped without paying for the cast it would throw
/// away. Must agree with [`ArrowCol::form`].
fn form_of(want: Want, dtype: &DataType) -> &'static str {
    match want {
        Want::Number => "a number",
        Want::Text => "text",
        Want::Key if dtype.is_integer() => "an integer key",
        Want::Key => "text",
    }
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
///
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
    check_clocks(df, specs)?;
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
        // A temporal clock is read in its own nanoseconds whatever the
        // column's unit, so the gap between two rows is taken in integers
        // and a nanosecond timestamp's gaps stay exact for the stream's
        // life (`online_core::ClockValue`). Casting it to a number instead
        // would expose its *internal representation*, so the same 60
        // seconds would be 60_000 / 60_000_000 / 60_000_000_000 clock units
        // for Datetime(ms/us/ns) (docs/TESTING.md T-E10). `check_clocks` has
        // already refused a spec that gives this clock plain numbers.
        if want == Want::Number
            && s.dtype().is_temporal()
            && specs
                .iter()
                .any(|sp| sp.clock.as_deref() == Some(name.as_str()))
        {
            cols.push((name.clone(), ArrowCol::Nanos(nanos_array(s)?)));
            continue;
        }
        // One column read in two roles that cast to the same form -- a
        // column that is a group key for one spec and a session for another,
        // both text -- would cast identically twice. `ArrowChunk::new` refuses
        // a repeated `(name, form)` because a hand-built one hides a caller's
        // mistake; here the two are the same array from the same source, so
        // the second role is skipped -- before its cast, since the form is
        // known from the role and the dtype. A key and a feature on the same
        // column cast to *different* forms and both stay.
        if cols
            .iter()
            .any(|(n, c)| n == &name && c.form() == form_of(want, s.dtype()))
        {
            continue;
        }
        cols.push((name.clone(), cast_to(s, want, spec, role, name.as_str())?));
    }
    ArrowChunk::new(df.height(), cols, names)
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
        let scale = spec
            .clock_scale()
            .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        if matches!(dtype, DataType::Time) {
            polars_bail!(ComputeError:
                "spec {:?}: clock column {:?} is a time of day, which starts again at \
                 midnight, so it cannot be a clock; combine it with its date into a \
                 Datetime, e.g. pl.col(\"date\").dt.combine(pl.col({:?}))",
                spec.name, clock, clock
            );
        }
        if dtype.is_temporal() {
            if let ClockScale::Numbers(param @ ("lam" | "q")) = scale {
                let instead = if param == "lam" {
                    "halflife"
                } else {
                    "coef_halflife"
                };
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} has dtype {}, a temporal clock, but {} is a \
                     rate per clock unit, which has no duration form; leave it out and give {} \
                     as a duration, e.g. pl.duration(minutes=10), timedelta(minutes=10) or \
                     \"10m\"; or cast the clock to the unit you mean, e.g. \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64), and use that column.",
                    spec.name, clock, dtype, param, instead, clock
                );
            }
            if let ClockScale::Numbers(param) = scale {
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} has dtype {}, a temporal clock, but {} is a \
                     plain number, which it cannot read: a number has no unit, and the \
                     column's own (e.g. epoch microseconds) would silently become it. Give {} \
                     and the other clock parameters as durations, e.g. \
                     pl.duration(minutes=10), timedelta(minutes=10) or \"10m\"; or cast the \
                     clock to the unit you mean, e.g. \
                     pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64), and use that column.",
                    spec.name, clock, dtype, param, param, clock
                );
            }
            // A cap or a disorder threshold finer than the clock's own step
            // cannot act on the data: every step would be cut to the cap, so
            // the clock would count rows, and no backwards jump could be
            // smaller than the threshold, so the check would never fire.
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
                let why = match param {
                    "max_dclock" => {
                        "every step would be capped to it, so the clock would count rows \
                         rather than measure time"
                    }
                    "min_backwards_jump" => {
                        "no backwards jump could be smaller, so the check would never fire \
                         (0 switches it off)"
                    }
                    _ => continue,
                };
                polars_bail!(ComputeError:
                    "spec {:?}: {} is {}, less than {}, the smallest step clock column {:?} \
                     ({}) can take: {}",
                    spec.name, param, span, step, clock, dtype, why
                );
            }
            // Read in seconds, a temporal column would feed any other
            // numeric role a number the spec never asked for.
            for other in specs {
                let role = if other.features.iter().any(|f| f == clock) {
                    "a feature"
                } else if other.weight.as_deref() == Some(clock) {
                    "a weight"
                } else if other.targets.iter().any(|t| t == clock)
                    && other.model.compares().is_none()
                    && !matches!(other.model, ModelKind::EwClass { .. })
                {
                    "a target"
                } else {
                    continue;
                };
                polars_bail!(ComputeError:
                    "spec {:?}: column {:?} has dtype {}, and {} reads it as {}, which must be \
                     numeric; a temporal column can only be a clock (cast it for the other \
                     role, e.g. pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64))",
                    other.name, clock, dtype, other.name, role, clock
                );
            }
        } else if let ClockScale::Durations(param) = scale {
            polars_bail!(ComputeError:
                "spec {:?}: {} is a duration, but clock column {:?} has dtype {}, which has no \
                 unit to measure it in. Use a temporal clock -- a Datetime, Date or Duration \
                 column, e.g. pl.from_epoch({:?}, time_unit=\"s\") -- or give {} as a number \
                 of the clock's own units.",
                spec.name, param, clock, dtype, clock, param
            );
        }
    }
    Ok(())
}

/// Nanoseconds in one unit of a temporal column: a `Date` counts days.
fn nanos_per_unit(dtype: &DataType) -> PolarsResult<i64> {
    Ok(match dtype {
        DataType::Datetime(TimeUnit::Milliseconds, _)
        | DataType::Duration(TimeUnit::Milliseconds) => 1_000_000,
        DataType::Datetime(TimeUnit::Microseconds, _)
        | DataType::Duration(TimeUnit::Microseconds) => 1_000,
        DataType::Datetime(TimeUnit::Nanoseconds, _)
        | DataType::Duration(TimeUnit::Nanoseconds) => 1,
        DataType::Date => 86_400 * 1_000_000_000,
        dt => polars_bail!(ComputeError: "a {} column cannot be read as a clock", dt),
    })
}

/// A temporal column as nanoseconds since the Unix epoch, null where it is
/// null. The unit is scaled away in integers, so one instant is the same
/// value whether it was stored in milliseconds, microseconds or
/// nanoseconds, and a timezone changes nothing: a `Datetime` is stored in
/// UTC, so a change of clocks for summer time neither stretches nor folds
/// the clock. A `Date` or a coarse `Datetime` can reach past what
/// nanoseconds in an `i64` hold, and such a value is refused by row. A
/// nanosecond column is taken as it is, without a pass over it.
fn nanos_array(s: &Series) -> PolarsResult<Int64Array> {
    let per = nanos_per_unit(s.dtype())?;
    let phys = s.to_physical_repr();
    let too_far = |i: usize| {
        polars_err!(ComputeError:
            "clock column {:?} has an instant at row {} that nanoseconds cannot hold (before \
             1677 or after 2262)",
            s.name(), i
        )
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
