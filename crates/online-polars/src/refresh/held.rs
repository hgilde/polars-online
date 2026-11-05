//! The points of one tick that a limited feed holds back.
//!
//! With `pairs`, one row can complete several pairs' points at once. A feed
//! limited to `n` points returns exactly `n`, so a limit that falls among
//! that row's points leaves the rest of them to the next feed, which returns
//! them first -- in the same run, or in a run resumed from the state saved
//! there. So a slice of the output and a run resumed after it give the one
//! run's grid, however the slice falls.
//!
//! A held point is what the scan computed (its pair, the ticks each series
//! contributed, each series' last value) beside the completing row's values
//! of the columns a point carries from the input: the group, the clock and
//! the `keep` columns. Every held point is of the one row, so the row is
//! kept once. Its values are kept as plain msgpack values ([`Cell`]): a
//! number, a string, a temporal value as its integer and unit. Any other
//! dtype (a list, a struct, a categorical, a decimal, ...) is kept as that
//! one value in Arrow IPC, written as a msgpack binary. A point of the
//! common columns costs a few dozen bytes in the state file, where the
//! whole row as Arrow IPC cost a couple of thousand.
//!
//! The row's values are cast to the dtypes of the input that the feed
//! returning them is given, which the output schema is declared from; they
//! are the same dtypes within a run, and a resumed run on a temporal clock
//! may give another unit.

use std::io::Cursor;

use polars::prelude::*;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Bytes written as one msgpack binary rather than an array of integers,
/// with `serde`'s own `serialize_bytes`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Bin(pub(super) Vec<u8>);

impl Serialize for Bin {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for Bin {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct BinVisitor;
        impl<'de> Visitor<'de> for BinVisitor {
            type Value = Bin;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Bin, E> {
                Ok(Bin(v.to_vec()))
            }
            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Bin, E> {
                Ok(Bin(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Bin, A::Error> {
                let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(b) = seq.next_element()? {
                    v.push(b);
                }
                Ok(Bin(v))
            }
        }
        d.deserialize_byte_buf(BinVisitor)
    }
}

/// A temporal value's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Unit {
    Ns,
    Us,
    Ms,
}

impl From<TimeUnit> for Unit {
    fn from(u: TimeUnit) -> Self {
        match u {
            TimeUnit::Nanoseconds => Self::Ns,
            TimeUnit::Microseconds => Self::Us,
            TimeUnit::Milliseconds => Self::Ms,
        }
    }
}

impl From<Unit> for TimeUnit {
    fn from(u: Unit) -> Self {
        match u {
            Unit::Ns => Self::Nanoseconds,
            Unit::Us => Self::Microseconds,
            Unit::Ms => Self::Milliseconds,
        }
    }
}

/// One value of the completing row. The variant names are the file's tags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum Cell {
    Null,
    Bool(bool),
    /// Any signed integer up to 64 bits.
    Int(i64),
    /// Any unsigned integer up to 64 bits.
    UInt(u64),
    /// A `Float64`, or a `Float32` widened exactly.
    Float(f64),
    Str(String),
    /// Days since the epoch.
    Date(i32),
    /// The integer in its unit since the epoch, and the time zone.
    Datetime(i64, Unit, Option<String>),
    Duration(i64, Unit),
    /// Nanoseconds since midnight.
    Time(i64),
    /// Any other dtype: the one value as an Arrow IPC file.
    Arrow(Bin),
}

impl Cell {
    /// The value of a one-row column.
    pub(super) fn of(col: &Column) -> PolarsResult<Self> {
        let s = col.as_materialized_series();
        Ok(match s.get(0)? {
            AnyValue::Null => Self::Null,
            AnyValue::Boolean(b) => Self::Bool(b),
            AnyValue::Int8(v) => Self::Int(v.into()),
            AnyValue::Int16(v) => Self::Int(v.into()),
            AnyValue::Int32(v) => Self::Int(v.into()),
            AnyValue::Int64(v) => Self::Int(v),
            AnyValue::UInt8(v) => Self::UInt(v.into()),
            AnyValue::UInt16(v) => Self::UInt(v.into()),
            AnyValue::UInt32(v) => Self::UInt(v.into()),
            AnyValue::UInt64(v) => Self::UInt(v),
            AnyValue::Float32(v) => Self::Float(v.into()),
            AnyValue::Float64(v) => Self::Float(v),
            AnyValue::String(v) => Self::Str(v.to_owned()),
            AnyValue::StringOwned(v) => Self::Str(v.to_string()),
            AnyValue::Date(v) => Self::Date(v),
            AnyValue::Datetime(v, u, tz) => {
                Self::Datetime(v, u.into(), tz.map(|z| z.as_str().to_owned()))
            }
            AnyValue::DatetimeOwned(v, u, tz) => {
                Self::Datetime(v, u.into(), tz.map(|z| z.as_str().to_owned()))
            }
            AnyValue::Duration(v, u) => Self::Duration(v, u.into()),
            AnyValue::Time(v) => Self::Time(v),
            other => {
                // Built again from the value, so the bytes are the value's
                // alone: a slice of a column can still point into the whole
                // column's buffers, which the IPC writer would write out.
                let one = Series::from_any_values_and_dtype(
                    s.name().clone(),
                    &[other.into_static()],
                    s.dtype(),
                    true,
                )?;
                let mut frame = DataFrame::new(1, vec![one.into_column()])?;
                let mut bytes = Vec::new();
                IpcWriter::new(&mut bytes).finish(&mut frame)?;
                Self::Arrow(Bin(bytes))
            }
        })
    }

    /// The value as a one-row series named `name`, in the dtype it was
    /// read from (an integer as 64 bits, a float as a `Float64`).
    pub(super) fn series(&self, name: &str) -> PolarsResult<Series> {
        let n: PlSmallStr = name.into();
        Ok(match self {
            Self::Null => Series::full_null(n, 1, &DataType::Null),
            Self::Bool(b) => Series::new(n, [*b]),
            Self::Int(v) => Series::new(n, [*v]),
            Self::UInt(v) => Series::new(n, [*v]),
            Self::Float(v) => Series::new(n, [*v]),
            Self::Str(v) => Series::new(n, [v.as_str()]),
            Self::Date(v) => Series::new(n, [*v]).into_date(),
            Self::Datetime(v, u, tz) => Series::new(n, [*v])
                .into_datetime((*u).into(), TimeZone::opt_try_new(tz.as_deref())?),
            Self::Duration(v, u) => Series::new(n, [*v]).into_duration((*u).into()),
            Self::Time(v) => Series::new(n, [*v]).into_time(),
            Self::Arrow(b) => {
                let frame = IpcReader::new(Cursor::new(&b.0)).finish()?;
                let col = frame.columns().first().ok_or_else(|| {
                    polars_err!(ComputeError: "refresh_time: the state is damaged: a held value has no column")
                })?;
                if col.len() != 1 {
                    polars_bail!(ComputeError:
                        "refresh_time: the state is damaged: a held value has {} rows", col.len());
                }
                col.as_materialized_series().clone().with_name(n)
            }
        })
    }
}

/// One held point: its pair, the ticks each series contributed since the
/// pair's previous point, and each series' last value.
pub(super) type HeldPoint = (usize, Vec<u64>, Vec<f64>);

/// The points of one row a limited feed held back, and that row's values
/// of the columns a point carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Held {
    /// The group, clock and `keep` columns, by name, at the completing row.
    pub(super) row: Vec<(String, Cell)>,
    pub(super) points: Vec<HeldPoint>,
}

impl Held {
    /// The row of `df` at `at`, in the columns `names`.
    pub(super) fn new(
        df: &DataFrame,
        names: &[&str],
        at: usize,
        points: Vec<HeldPoint>,
    ) -> PolarsResult<Self> {
        let row = names
            .iter()
            .map(|n| {
                let one = df.column(n)?.slice(at as i64, 1);
                Ok(((*n).to_owned(), Cell::of(&one)?))
            })
            .collect::<PolarsResult<_>>()?;
        Ok(Self { row, points })
    }

    /// The held row as a one-row frame in `names`, each column in `df`'s
    /// dtype, which the output schema is declared from.
    pub(super) fn frame(&self, df: &DataFrame, names: &[&str]) -> PolarsResult<DataFrame> {
        let mut cols = Vec::with_capacity(names.len());
        for n in names {
            let Some((_, cell)) = self.row.iter().find(|(k, _)| k == n) else {
                polars_bail!(ComputeError:
                    "refresh_time: the state holds points of a tick saved without a {:?} \
                     column; resume with the group and keep columns it was saved with ({:?})",
                    n, self.row.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
                );
            };
            let s = cell.series(n)?;
            let want = df.column(n)?.dtype();
            let s = if s.dtype() == want {
                s
            } else {
                let cast = s.strict_cast(want).map_err(|_| {
                    polars_err!(ComputeError:
                        "refresh_time: the state holds points of a tick whose {:?} is a {}, \
                         which this input's {} cannot hold; resume on the columns the state \
                         was saved from",
                        n, s.dtype(), want
                    )
                })?;
                // A cast to a zoned `Datetime` can come back naive: the zone
                // is the input's, set on the instant the cast kept.
                match want {
                    DataType::Datetime(u, tz) if cast.dtype() != want => {
                        cast.into_datetime(*u, tz.clone())
                    }
                    _ => cast,
                }
            };
            cols.push(s.into_column());
        }
        DataFrame::new(1, cols)
    }
}
