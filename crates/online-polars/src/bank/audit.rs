//! An `audit` spec's counts, as frames (docs/PLAN.md task 223 (b)):
//! `ModelBank.audit` in Python.

use online_core::{Audit, AuditColumn};
use polars::prelude::*;

use super::{Bank, GroupKey};
use crate::spec::ModelKind;
use crate::stream::{AnyModel, ClockDtype, ClockUnit};
use crate::summary::counts;

/// Which of an audit's frames to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditTable {
    /// One row per column.
    Columns,
    /// One row per pair of columns, when the spec keeps pairs.
    Pairs,
    /// One row: the clock's steps.
    Clock,
}

impl AuditTable {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "columns" => Ok(Self::Columns),
            "pairs" => Ok(Self::Pairs),
            "clock" => Ok(Self::Clock),
            other => Err(format!(
                "table must be \"columns\", \"pairs\" or \"clock\", got {other:?}"
            )),
        }
    }
}

/// NaN, the core's "undefined", as null.
fn opt(v: f64) -> Option<f64> {
    (!v.is_nan()).then_some(v)
}

impl Bank {
    /// An `audit` spec's counts as a frame ([`AuditTable`]), led by `spec`,
    /// as every reader of the bank is: per group in key order
    /// ([`Self::sorted_keys`]), or, `pooled`, one audit of every group the
    /// frame covers ([`Audit::merge`]), with no `group` column. Every count
    /// is an `Int64`, as in every frame (task 244).
    ///
    /// `Columns`: `spec`, `group`, `column`, `rows_fed`, `null_count`,
    /// `nan_count`, `pos_inf_count`, `neg_inf_count`, `beyond_bound_count`,
    /// `count` (usable
    /// values), `mean`, `std`, `skew`, `kurtosis`, `min`, `max`, `median`,
    /// `mad`, `robust_z`, `n_unique` (null past the cap), `top_value`,
    /// `top_count`, `second_count`, `count_error`, `longest_run`,
    /// `equal_prev`, `adjacent`, `equal_by_chance`, `min_change`,
    /// `min_change_count`, `autocorr`, `unit_root_t`. `Pairs`: `spec`,
    /// `group`, `column_a`, `column_b`, `count`, `corr`, `equal`. `Clock`:
    /// `spec`, `group`, `steps`, `duplicates`, `gaps` (null without
    /// `gap_cap`), `regular`, `step_mean`, `step_std`, `step_cv`,
    /// `step_max`; no rows for a stream without a clock column. Each step
    /// is the time that passed, uncapped (review 6, C-5); on a temporal
    /// clock `step_mean`, `step_std` and `step_max` are `Duration`s in the
    /// column's unit (milliseconds for a `Date`), and plain numbers on a
    /// number clock or before a chunk has said which.
    ///
    /// # Errors
    ///
    /// `spec` out of range, not an `audit` spec, or `Pairs` of a spec that
    /// keeps none.
    pub fn audit(
        &self,
        spec: usize,
        group: Option<&[GroupKey]>,
        table: AuditTable,
        pooled: bool,
    ) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        let ModelKind::Audit { pairs, .. } = &s.model else {
            return Err(format!(
                "spec {:?} has model type {:?}, not \"audit\"; only an audit keeps these counts",
                s.name,
                s.model.kind_name()
            ));
        };
        if table == AuditTable::Pairs && !pairs.unwrap_or(false) {
            return Err(format!(
                "spec {:?} keeps no pairs: build it with pairs=True",
                s.name
            ));
        }
        let mut audits: Vec<(Option<&str>, Audit)> = Vec::new();
        for key in keys {
            for (_, model) in &states[key].models {
                let AnyModel::Audit(m) = model else {
                    unreachable!("an audit spec builds audit models");
                };
                audits.push((key.as_str(), (**m).clone()));
            }
        }
        if pooled {
            let mut it = audits.into_iter().map(|(_, a)| a);
            audits = match it.next() {
                Some(mut first) => {
                    for a in it {
                        first.merge(&a)?;
                    }
                    vec![(None, first)]
                }
                None => Vec::new(),
            };
        }
        let mut cols = match table {
            AuditTable::Columns => columns_frame(&audits, &s.features),
            AuditTable::Pairs => pairs_frame(&audits, &s.features),
            AuditTable::Clock => clock_frame(&audits, self.clock_dtypes[spec].as_ref()),
        };
        if pooled {
            cols.remove(0);
        }
        let height = cols.first().map_or(0, Column::len);
        cols.insert(0, Column::new("spec".into(), vec![s.name.as_str(); height]));
        DataFrame::new(height, cols).map_err(|e| e.to_string())
    }
}

fn columns_frame(audits: &[(Option<&str>, Audit)], names: &[String]) -> Vec<Column> {
    let mut group: Vec<Option<&str>> = Vec::new();
    let mut column: Vec<&str> = Vec::new();
    let mut reps: Vec<AuditColumn> = Vec::new();
    for (g, a) in audits {
        for (j, name) in names.iter().enumerate() {
            group.push(*g);
            column.push(name);
            reps.push(a.column(j));
        }
    }
    let int = |name: &str, f: fn(&AuditColumn) -> u64| -> Column {
        counts(name, reps.iter().map(|r| Some(f(r))))
    };
    let num = |f: fn(&AuditColumn) -> f64| -> Vec<Option<f64>> {
        reps.iter().map(|r| opt(f(r))).collect()
    };
    vec![
        Column::new("group".into(), group),
        Column::new("column".into(), column),
        // Polars' names for the same counts (`null_count`, `n_unique`), and
        // the rows fed as every frame says it (task 244).
        int("rows_fed", |r| r.rows),
        int("null_count", |r| r.null),
        int("nan_count", |r| r.nan),
        int("pos_inf_count", |r| r.pos_inf),
        int("neg_inf_count", |r| r.neg_inf),
        int("beyond_bound_count", |r| r.beyond),
        int("count", |r| r.count),
        Column::new("mean".into(), num(|r| r.mean)),
        Column::new("std".into(), num(|r| r.std)),
        Column::new("skew".into(), num(|r| r.skew)),
        Column::new("kurtosis".into(), num(|r| r.kurtosis)),
        Column::new("min".into(), num(|r| r.min)),
        Column::new("max".into(), num(|r| r.max)),
        Column::new("median".into(), num(|r| r.median)),
        Column::new("mad".into(), num(|r| r.mad)),
        Column::new("robust_z".into(), num(|r| r.robust_z)),
        counts("n_unique", reps.iter().map(|r| r.distinct)),
        Column::new("top_value".into(), num(|r| r.top_value)),
        int("top_count", |r| r.top_count),
        int("second_count", |r| r.second_count),
        int("count_error", |r| r.count_error),
        int("longest_run", |r| r.longest_run),
        int("equal_prev", |r| r.equal_prev),
        int("adjacent", |r| r.adjacent),
        Column::new("equal_by_chance".into(), num(|r| r.equal_by_chance)),
        Column::new("min_change".into(), num(|r| r.min_change)),
        int("min_change_count", |r| r.min_change_count),
        Column::new("autocorr".into(), num(|r| r.autocorr)),
        Column::new("unit_root_t".into(), num(|r| r.unit_root_t)),
    ]
}

fn pairs_frame(audits: &[(Option<&str>, Audit)], names: &[String]) -> Vec<Column> {
    let mut group: Vec<Option<&str>> = Vec::new();
    let (mut a_col, mut b_col): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
    let (mut count, mut corr, mut equal) = (Vec::new(), Vec::new(), Vec::new());
    for (g, a) in audits {
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let p = a.pair(i, j).expect("an audit with pairs keeps every one");
                group.push(*g);
                a_col.push(&names[i]);
                b_col.push(&names[j]);
                count.push(Some(p.count));
                corr.push(opt(p.corr));
                equal.push(Some(p.equal));
            }
        }
    }
    vec![
        Column::new("group".into(), group),
        Column::new("column_a".into(), a_col),
        Column::new("column_b".into(), b_col),
        counts("count", count),
        Column::new("corr".into(), corr),
        counts("equal", equal),
    ]
}

/// Nanoseconds in one unit of a `Duration` column for a temporal clock's
/// steps, and that column's unit: the clock column's own, milliseconds for a
/// `Date` (polars' `Date - Date`). `None` for a clock that is a number.
fn duration_unit(dtype: Option<&ClockDtype>) -> Option<(f64, TimeUnit)> {
    let unit = |u: &ClockUnit| match u {
        ClockUnit::Ms => (1e6, TimeUnit::Milliseconds),
        ClockUnit::Us => (1e3, TimeUnit::Microseconds),
        ClockUnit::Ns => (1.0, TimeUnit::Nanoseconds),
    };
    match dtype? {
        ClockDtype::Date => Some(unit(&ClockUnit::Ms)),
        ClockDtype::Datetime { unit: u, .. } | ClockDtype::Duration { unit: u } => Some(unit(u)),
        ClockDtype::Rows | ClockDtype::Numeric | ClockDtype::Int(_) => None,
    }
}

fn clock_frame(audits: &[(Option<&str>, Audit)], dtype: Option<&ClockDtype>) -> Vec<Column> {
    let mut group: Vec<Option<&str>> = Vec::new();
    let mut reps = Vec::new();
    for (g, a) in audits {
        if let Some(c) = a.clock() {
            group.push(*g);
            reps.push(c);
        }
    }
    let int = |name: &str, f: fn(&online_core::AuditClock) -> u64| -> Column {
        counts(name, reps.iter().map(|r| Some(f(r))))
    };
    let num = |f: fn(&online_core::AuditClock) -> f64| -> Vec<Option<f64>> {
        reps.iter().map(|r| opt(f(r))).collect()
    };
    // A step in the clock's own terms: a temporal clock's steps are kept in
    // seconds, and read back as a `Duration` of its unit, rounded once to
    // it; a number clock's as the number (review 6, C-5).
    let span = |name: &str, f: fn(&online_core::AuditClock) -> f64| -> Column {
        match duration_unit(dtype) {
            None => Column::new(name.into(), num(f)),
            Some((ns, unit)) => {
                let v: Vec<Option<i64>> = reps
                    .iter()
                    .map(|r| opt(f(r)).map(|s| (s * 1e9 / ns).round() as i64))
                    .collect();
                Column::new(name.into(), v)
                    .cast(&DataType::Duration(unit))
                    .expect("an i64 column casts to a Duration")
            }
        }
    };
    vec![
        Column::new("group".into(), group),
        int("steps", |r| r.steps),
        int("duplicates", |r| r.duplicates),
        counts("gaps", reps.iter().map(|r| r.gaps)),
        int("regular", |r| r.regular),
        span("step_mean", |r| r.step_mean),
        span("step_std", |r| r.step_std),
        Column::new("step_cv".into(), num(|r| r.step_cv)),
        // `<quantity>_<stat>`, as `step_mean` (task 244).
        span("step_max", |r| r.max_step),
    ]
}
