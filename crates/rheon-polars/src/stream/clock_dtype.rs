//! The clock column's type, kept per spec ([`ClockDtype`]), moved out of
//! `stream.rs`, which had passed the size a source file is held to (task
//! 278, batch 17).

use super::*;

/// The clock column's type, kept per spec so that `scored_clock` and
/// `learned_clock` (docs/PLAN.md task 152) come out as the column came in:
/// a `Datetime` in its own unit and zone, a `Date`, a `Duration`, an
/// integer in its own width (task 200), a float for any other number clock,
/// and the group's row index as an integer with none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClockDtype {
    Rows,
    Numeric,
    Date,
    Datetime {
        unit: ClockUnit,
        tz: Option<String>,
    },
    Duration {
        unit: ClockUnit,
    },
    /// An integer clock up to 64 bits wide, held in integers (task 200).
    Int(IntWidth),
}

/// An integer clock column's dtype (docs/PLAN.md task 200).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntWidth {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClockUnit {
    Ms,
    Us,
    Ns,
}

impl ClockDtype {
    /// From the spec's clock column as the chunk holds it: `dtype` is the
    /// column's when it is temporal, `None` otherwise.
    pub fn of(has_clock: bool, dtype: Option<&polars::prelude::DataType>) -> Self {
        use polars::prelude::{DataType as D, TimeUnit as T};
        let unit = |u: &T| match u {
            T::Milliseconds => ClockUnit::Ms,
            T::Microseconds => ClockUnit::Us,
            T::Nanoseconds => ClockUnit::Ns,
        };
        let width = |d: &D| {
            Some(match d {
                D::Int8 => IntWidth::I8,
                D::Int16 => IntWidth::I16,
                D::Int32 => IntWidth::I32,
                D::Int64 => IntWidth::I64,
                D::UInt8 => IntWidth::U8,
                D::UInt16 => IntWidth::U16,
                D::UInt32 => IntWidth::U32,
                D::UInt64 => IntWidth::U64,
                _ => return None,
            })
        };
        match (has_clock, dtype) {
            (false, _) => Self::Rows,
            (true, Some(d)) if width(d).is_some() => Self::Int(width(d).expect("matched")),
            (true, Some(D::Date)) => Self::Date,
            (true, Some(D::Datetime(u, tz))) => Self::Datetime {
                unit: unit(u),
                tz: tz.as_ref().map(|z| z.to_string()),
            },
            (true, Some(D::Duration(u))) => Self::Duration { unit: unit(u) },
            (true, _) => Self::Numeric,
        }
    }

    /// Nanoseconds in one unit of the column.
    fn ns_per_unit(&self) -> i64 {
        match self {
            Self::Date => 86_400 * 1_000_000_000,
            Self::Datetime { unit, .. } | Self::Duration { unit } => match unit {
                ClockUnit::Ms => 1_000_000,
                ClockUnit::Us => 1_000,
                ClockUnit::Ns => 1,
            },
            Self::Rows | Self::Numeric | Self::Int(_) => 1,
        }
    }

    /// Whether a chunk whose clock column is `now` reads its steps another
    /// way than this dtype, which the bank kept from its first chunk (task
    /// 200): an integer clock's steps are taken in integers and a float
    /// clock's in doubles, so a stream reads one or the other for its life;
    /// and an integer clock's fields come out in its width, which a wider
    /// column would not fit. A temporal clock is read in nanoseconds
    /// whatever its unit, and a spec's clock parameters keep it temporal.
    pub fn conflicts(&self, now: &ClockDtype) -> bool {
        match (self, now) {
            (Self::Int(a), Self::Int(b)) => a != b,
            (Self::Int(_), Self::Numeric) | (Self::Numeric, Self::Int(_)) => true,
            _ => false,
        }
    }

    /// The column's type as polars names it: what [`Self::array`] makes.
    pub fn dtype(&self) -> polars::prelude::DataType {
        use polars::prelude::{DataType as D, TimeUnit as T};
        let unit = |u: &ClockUnit| match u {
            ClockUnit::Ms => T::Milliseconds,
            ClockUnit::Us => T::Microseconds,
            ClockUnit::Ns => T::Nanoseconds,
        };
        match self {
            Self::Rows => D::Int64,
            Self::Numeric => D::Float64,
            Self::Int(w) => match w {
                IntWidth::I8 => D::Int8,
                IntWidth::I16 => D::Int16,
                IntWidth::I32 => D::Int32,
                IntWidth::I64 => D::Int64,
                IntWidth::U8 => D::UInt8,
                IntWidth::U16 => D::UInt16,
                IntWidth::U32 => D::UInt32,
                IntWidth::U64 => D::UInt64,
            },
            Self::Date => D::Date,
            // A zone polars gave the column is one it reads back.
            Self::Datetime { unit: u, tz } => D::Datetime(
                unit(u),
                polars::prelude::TimeZone::opt_try_new(tz.clone())
                    .ok()
                    .flatten(),
            ),
            Self::Duration { unit: u } => D::Duration(unit(u)),
        }
    }

    /// A frame's clock column of `values` (review round 4, N18): in the
    /// clock column's own dtype, exactly, as `scored_clock` is -- a
    /// `Datetime` in its unit and zone, a `Date`, a `Duration`, an integer
    /// in its own width (task 200), a `Float64` for any other number clock
    /// -- and a `Float64` of nulls where there is no clock column (`Rows`)
    /// or no chunk has said what it is (`None`).
    pub fn column(
        dtype: Option<&ClockDtype>,
        name: &str,
        values: &[Option<ClockValue>],
    ) -> polars::prelude::PolarsResult<polars::prelude::Column> {
        use polars::prelude::{Column, Series};
        match dtype {
            Some(d) if *d != Self::Rows => Ok(Column::from(Series::from_arrow(
                name.into(),
                d.array(values),
            )?)),
            _ => Ok(Column::new(
                name.into(),
                values
                    .iter()
                    .map(|c| c.map(ClockValue::seconds))
                    .collect::<Vec<Option<f64>>>(),
            )),
        }
    }

    /// The values as an Arrow array of the column's type, null where `None`.
    pub fn array(&self, values: &[Option<ClockValue>]) -> Box<dyn polars_arrow::array::Array> {
        use polars_arrow::array::PrimitiveArray;
        use polars_arrow::datatypes::{ArrowDataType, TimeUnit as AT};
        let au = |u: &ClockUnit| match u {
            ClockUnit::Ms => AT::Millisecond,
            ClockUnit::Us => AT::Microsecond,
            ClockUnit::Ns => AT::Nanosecond,
        };
        let per = self.ns_per_unit();
        let ns = |c: Option<ClockValue>| match c {
            Some(ClockValue::Ns(n)) => Some(n / per),
            _ => None,
        };
        // An integer clock's values in its width: each came from a column
        // of it, the bank refusing another width (`Self::conflicts`), so
        // none is out of its range; one that were would be null, not wrapped.
        fn ints<T>(values: &[Option<ClockValue>]) -> Box<dyn polars_arrow::array::Array>
        where
            T: polars_arrow::types::NativeType + TryFrom<i64>,
        {
            let v: Vec<Option<T>> = values
                .iter()
                .map(|c| match c {
                    Some(ClockValue::I64(x)) => T::try_from(*x).ok(),
                    _ => None,
                })
                .collect();
            Box::new(PrimitiveArray::<T>::from(v))
        }
        match self {
            Self::Numeric => {
                let v: Vec<Option<f64>> = values
                    .iter()
                    .map(|c| match c {
                        Some(ClockValue::F64(x)) => Some(*x),
                        _ => None,
                    })
                    .collect();
                Box::new(PrimitiveArray::<f64>::from(v))
            }
            Self::Int(w) => match w {
                IntWidth::I8 => ints::<i8>(values),
                IntWidth::I16 => ints::<i16>(values),
                IntWidth::I32 => ints::<i32>(values),
                IntWidth::I64 => ints::<i64>(values),
                IntWidth::U8 => ints::<u8>(values),
                IntWidth::U16 => ints::<u16>(values),
                IntWidth::U32 => ints::<u32>(values),
                IntWidth::U64 => ints::<u64>(values),
            },
            Self::Rows => {
                let v: Vec<Option<i64>> = values
                    .iter()
                    .map(|c| match c {
                        Some(ClockValue::F64(x)) => Some(*x as i64),
                        _ => None,
                    })
                    .collect();
                Box::new(PrimitiveArray::<i64>::from(v))
            }
            Self::Date => {
                let v: Vec<Option<i32>> = values.iter().map(|c| ns(*c).map(|d| d as i32)).collect();
                Box::new(PrimitiveArray::<i32>::from(v).to(ArrowDataType::Date32))
            }
            Self::Datetime { unit, tz } => {
                let v: Vec<Option<i64>> = values.iter().map(|c| ns(*c)).collect();
                let dt = ArrowDataType::Timestamp(au(unit), tz.clone().map(Into::into));
                Box::new(PrimitiveArray::<i64>::from(v).to(dt))
            }
            Self::Duration { unit } => {
                let v: Vec<Option<i64>> = values.iter().map(|c| ns(*c)).collect();
                Box::new(PrimitiveArray::<i64>::from(v).to(ArrowDataType::Duration(au(unit))))
            }
        }
    }
}
