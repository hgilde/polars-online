//! An `audit` spec's counts, as frames (docs/PLAN.md task 223 (b)):
//! `ModelBank.audit` in Python.

use online_core::{Audit, AuditColumn};
use polars::prelude::*;

use super::{Bank, GroupKey};
use crate::spec::ModelKind;
use crate::stream::AnyModel;

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
    /// An `audit` spec's counts as a frame ([`AuditTable`]): per group in
    /// key order ([`Self::sorted_keys`]), or, `pooled`, one audit of every
    /// group the frame covers ([`Audit::merge`]), with no `group` column.
    ///
    /// `Columns`: `group`, `column`, `rows`, `null`, `nan`, `pos_inf`,
    /// `neg_inf`, `beyond_bound`, `count` (usable values), `mean`, `std`,
    /// `skew`, `kurtosis`, `min`, `max`, `median`, `mad`, `robust_z`,
    /// `distinct` (null past the cap), `top_value`, `top_count`,
    /// `second_count`, `count_error`, `longest_run`, `equal_prev`,
    /// `adjacent`, `equal_by_chance`, `autocorr`, `unit_root_t`. `Pairs`: `group`, `column_a`,
    /// `column_b`, `count`, `corr`, `equal`. `Clock`: `group`, `steps`,
    /// `duplicates`, `gaps` (null without `gap_cap`), `regular`,
    /// `step_mean`, `step_std`, `step_cv`, `max_step`; no rows for a stream
    /// without a clock column.
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
            AuditTable::Clock => clock_frame(&audits),
        };
        if pooled {
            cols.remove(0);
        }
        let height = cols.first().map_or(0, Column::len);
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
    let int = |f: fn(&AuditColumn) -> u64| -> Vec<u64> { reps.iter().map(f).collect() };
    let num = |f: fn(&AuditColumn) -> f64| -> Vec<Option<f64>> {
        reps.iter().map(|r| opt(f(r))).collect()
    };
    vec![
        Column::new("group".into(), group),
        Column::new("column".into(), column),
        Column::new("rows".into(), int(|r| r.rows)),
        Column::new("null".into(), int(|r| r.null)),
        Column::new("nan".into(), int(|r| r.nan)),
        Column::new("pos_inf".into(), int(|r| r.pos_inf)),
        Column::new("neg_inf".into(), int(|r| r.neg_inf)),
        Column::new("beyond_bound".into(), int(|r| r.beyond)),
        Column::new("count".into(), int(|r| r.count)),
        Column::new("mean".into(), num(|r| r.mean)),
        Column::new("std".into(), num(|r| r.std)),
        Column::new("skew".into(), num(|r| r.skew)),
        Column::new("kurtosis".into(), num(|r| r.kurtosis)),
        Column::new("min".into(), num(|r| r.min)),
        Column::new("max".into(), num(|r| r.max)),
        Column::new("median".into(), num(|r| r.median)),
        Column::new("mad".into(), num(|r| r.mad)),
        Column::new("robust_z".into(), num(|r| r.robust_z)),
        Column::new(
            "distinct".into(),
            reps.iter()
                .map(|r| r.distinct)
                .collect::<Vec<Option<u64>>>(),
        ),
        Column::new("top_value".into(), num(|r| r.top_value)),
        Column::new("top_count".into(), int(|r| r.top_count)),
        Column::new("second_count".into(), int(|r| r.second_count)),
        Column::new("count_error".into(), int(|r| r.count_error)),
        Column::new("longest_run".into(), int(|r| r.longest_run)),
        Column::new("equal_prev".into(), int(|r| r.equal_prev)),
        Column::new("adjacent".into(), int(|r| r.adjacent)),
        Column::new("equal_by_chance".into(), num(|r| r.equal_by_chance)),
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
                count.push(p.count);
                corr.push(opt(p.corr));
                equal.push(p.equal);
            }
        }
    }
    vec![
        Column::new("group".into(), group),
        Column::new("column_a".into(), a_col),
        Column::new("column_b".into(), b_col),
        Column::new("count".into(), count),
        Column::new("corr".into(), corr),
        Column::new("equal".into(), equal),
    ]
}

fn clock_frame(audits: &[(Option<&str>, Audit)]) -> Vec<Column> {
    let mut group: Vec<Option<&str>> = Vec::new();
    let mut reps = Vec::new();
    for (g, a) in audits {
        if let Some(c) = a.clock() {
            group.push(*g);
            reps.push(c);
        }
    }
    let int = |f: fn(&online_core::AuditClock) -> u64| -> Vec<u64> { reps.iter().map(f).collect() };
    let num = |f: fn(&online_core::AuditClock) -> f64| -> Vec<Option<f64>> {
        reps.iter().map(|r| opt(f(r))).collect()
    };
    vec![
        Column::new("group".into(), group),
        Column::new("steps".into(), int(|r| r.steps)),
        Column::new("duplicates".into(), int(|r| r.duplicates)),
        Column::new(
            "gaps".into(),
            reps.iter().map(|r| r.gaps).collect::<Vec<Option<u64>>>(),
        ),
        Column::new("regular".into(), int(|r| r.regular)),
        Column::new("step_mean".into(), num(|r| r.step_mean)),
        Column::new("step_std".into(), num(|r| r.step_std)),
        Column::new("step_cv".into(), num(|r| r.step_cv)),
        Column::new("max_step".into(), num(|r| r.max_step)),
    ]
}
