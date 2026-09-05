//! What a stream has been fed (docs/PLAN.md task 35): row counts, the clock
//! range, the weight, and per-column statistics over every row routed to
//! the stream, skipped or not. Saved with the state, so a bank loaded from a
//! file says what data each model was trained on without the data.
//!
//! Everything here is *undecayed* and accumulated in row order: the counts
//! are counts, the moments are Welford's recursion over the usable values
//! of each column, and the clock range is a running min/max. Row order is
//! the only order the recursion has, so a stream fed as one chunk or a
//! thousand -- or in runs of [`crate::ChunkOut::run_rows`] -- accumulates
//! the same bits (hard rule 3). Nothing here is read by a model: the models'
//! own view of the data is exponentially weighted and lives in their state.

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spec::Spec;
use crate::stream::usable;

/// Count, nulls and moments of one input column over the rows fed.
///
/// `mean` and `m2` (the sum of squared deviations) are Welford's; `min` and
/// `max` start at the identities so the update has no first-row branch and
/// are reported as null while `count` is 0. A value is counted when
/// [`usable`] -- finite and within the input bound -- and a null otherwise:
/// polars nulls arrive as NaN, so "null" here is null, NaN, an infinity or
/// a magnitude the models would not accept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnStats {
    pub count: u64,
    pub nulls: u64,
    pub mean: f64,
    pub m2: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for ColumnStats {
    fn default() -> Self {
        Self {
            count: 0,
            nulls: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

impl ColumnStats {
    /// One value. `n_row` is the ordinal of the row being fed and `inv_row`
    /// its reciprocal, shared by every column that has never had a null (the
    /// usual case), so a row costs one division rather than one per column;
    /// a column with nulls behind it divides by its own count, which is the
    /// same number `inv_row` would be for it, so the bits do not depend on
    /// which path a column took.
    #[inline]
    fn push(&mut self, x: f64, n_row: u64, inv_row: f64) {
        if usable(x) {
            self.count += 1;
            let inv = if self.count == n_row {
                inv_row
            } else {
                1.0 / self.count as f64
            };
            let d = x - self.mean;
            self.mean += d * inv;
            self.m2 += d * (x - self.mean);
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        } else {
            self.nulls += 1;
        }
    }

    /// Sample standard deviation (`ddof = 1`, as polars' `std` and
    /// `describe`), `None` below two values. `m2` is non-negative in exact
    /// arithmetic; rounding can put it a hair below zero, which reads as 0.
    pub fn std(&self) -> Option<f64> {
        (self.count >= 2).then(|| (self.m2.max(0.0) / (self.count - 1) as f64).sqrt())
    }

    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    pub fn min(&self) -> Option<f64> {
        (self.count > 0).then_some(self.min)
    }

    pub fn max(&self) -> Option<f64> {
        (self.count > 0).then_some(self.max)
    }

    /// Whether the numbers could have come from [`Self::push`] over `fed`
    /// rows: the counts add up, the moments are finite, the range is a
    /// range. What [`DataSummary::validate`] asks of a loaded file.
    fn is_consistent(&self, fed: u64) -> bool {
        if self.count + self.nulls != fed || !self.mean.is_finite() || !self.m2.is_finite() {
            return false;
        }
        if self.count == 0 {
            return self.mean == 0.0 && self.m2 == 0.0;
        }
        self.min.is_finite()
            && self.max.is_finite()
            && self.min <= self.mean
            && self.mean <= self.max
            && (self.count > 1 || (self.m2 == 0.0 && self.min == self.max))
    }
}

/// Which spec column a [`ColumnStats`] describes. The order in
/// [`DataSummary::columns`] is the features, then the targets, then the
/// weight column when the spec has one -- the order [`DataSummary::new`]
/// lays out and [`DataSummary::feed_row`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Feature,
    Target,
    Weight,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Feature => "feature",
            Role::Target => "target",
            Role::Weight => "weight",
        }
    }
}

/// The training data of one stream, as fed. See the module docs.
///
/// Row counts partition the rows fed: a row is *skipped* when a feature or
/// the weight is not usable (the models never see it; the clock still
/// moves), else *processed*; a processed row has *zero weight* (advances
/// the clock, teaches nothing -- hard rule 9), or is *learned* (weight
/// above zero and, for a model with targets, at least one usable target),
/// or is predict-only (every target null). `rows_processed` itself is the
/// stream's `rows_seen`, kept beside this since before it existed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSummary {
    /// Rows routed to the stream, skipped or not.
    pub rows_fed: u64,
    /// Processed rows the models updated from.
    pub rows_learned: u64,
    /// Processed rows with weight 0.
    pub rows_zero_weight: u64,
    /// Sum of the processed rows' weights (1 each without a weight column):
    /// the undecayed total the models were handed.
    pub weight_sum: f64,
    /// The clock range over the rows fed; `None` on a row-count clock or
    /// before the first row.
    pub clock_min: Option<f64>,
    pub clock_max: Option<f64>,
    /// Rows whose session id differed from the previous row's.
    pub session_changes: u64,
    /// Rows whose clock was below the previous row's within a session --
    /// what `on_clock_reset` had to decide about.
    pub clock_backwards: u64,
    /// Rows at which the session or clock policy restarted the stream's
    /// state (`session_gap = "reset"`, `on_clock_reset = "reset_state"`).
    pub resets: u64,
    /// Features, then targets, then the weight column when there is one.
    pub columns: Vec<ColumnStats>,
}

impl DataSummary {
    /// Empty, laid out for `spec`'s columns.
    pub fn new(spec: &Spec) -> Self {
        let n = spec.features.len() + spec.targets.len() + usize::from(spec.weight.is_some());
        Self {
            rows_fed: 0,
            rows_learned: 0,
            rows_zero_weight: 0,
            weight_sum: 0.0,
            clock_min: None,
            clock_max: None,
            session_changes: 0,
            clock_backwards: 0,
            resets: 0,
            columns: vec![ColumnStats::default(); n],
        }
    }

    /// The columns' names and roles in `columns` order, for `spec`.
    pub fn layout(spec: &Spec) -> Vec<(&str, Role)> {
        spec.features
            .iter()
            .map(|c| (c.as_str(), Role::Feature))
            .chain(spec.targets.iter().map(|c| (c.as_str(), Role::Target)))
            .chain(spec.weight.iter().map(|c| (c.as_str(), Role::Weight)))
            .collect()
    }

    /// Row `i` of the stream's input columns: one more row fed, its values
    /// into the column statistics, its clock into the range. `targets` has
    /// one vector per spec target (empty until the bank fills a
    /// comparison's), `weight` the weight column's value when the spec has
    /// one; `clock` is `None` on a row-count clock.
    #[inline]
    pub fn feed_row(
        &mut self,
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
        weight: Option<f64>,
        clock: Option<f64>,
        i: usize,
    ) {
        self.rows_fed += 1;
        let n_row = self.rows_fed;
        let inv_row = 1.0 / n_row as f64;
        let nf = features.len();
        // The layout's target slots, whether or not the caller filled them:
        // a target the row has no vector for is a null, so the counts still
        // add up to the rows fed.
        let n_cols = self.columns.len();
        let nt = n_cols.saturating_sub(nf + usize::from(weight.is_some()));
        for (c, f) in self.columns.iter_mut().zip(features) {
            c.push(f[i], n_row, inv_row);
        }
        for (k, c) in self.columns[nf.min(n_cols)..]
            .iter_mut()
            .take(nt)
            .enumerate()
        {
            match targets.get(k) {
                Some(t) => c.push(t[i], n_row, inv_row),
                None => c.nulls += 1,
            }
        }
        if let (Some(w), Some(c)) = (weight, self.columns.get_mut(nf + nt)) {
            c.push(w, n_row, inv_row);
        }
        if let Some(c) = clock {
            self.clock_min = Some(self.clock_min.map_or(c, |m| m.min(c)));
            self.clock_max = Some(self.clock_max.map_or(c, |m| m.max(c)));
        }
    }

    /// What the clock schedule found at the row just fed.
    #[inline]
    pub fn events(&mut self, session_changed: bool, backwards: bool, reset: bool) {
        self.session_changes += u64::from(session_changed);
        self.clock_backwards += u64::from(backwards);
        self.resets += u64::from(reset);
    }

    /// The row just fed was accepted: its weight, and whether the models
    /// were handed a target to learn from (a model without targets always
    /// is).
    #[inline]
    pub fn accepted(&mut self, w: f64, has_target: bool) {
        self.weight_sum += w;
        if w == 0.0 {
            self.rows_zero_weight += 1;
        } else if has_target {
            self.rows_learned += 1;
        }
    }

    /// Whether the numbers could have come from feeding `rows_seen`
    /// processed rows of `spec` through this. A loaded file that fails this
    /// is refused rather than reported.
    pub fn validate(&self, spec: &Spec, rows_seen: u64) -> Result<(), String> {
        let fail = |what: &str| {
            Err(format!(
                "saved data summary of spec {:?} is not its spec's ({what})",
                spec.name
            ))
        };
        let want = spec.features.len() + spec.targets.len() + usize::from(spec.weight.is_some());
        if self.columns.len() != want {
            return fail(&format!(
                "{} columns where the spec has {want}",
                self.columns.len()
            ));
        }
        if self.rows_fed < rows_seen {
            return fail("fewer rows fed than processed");
        }
        if self.rows_learned + self.rows_zero_weight > rows_seen {
            return fail("more rows learned than processed");
        }
        if !(self.weight_sum.is_finite() && self.weight_sum >= 0.0) {
            return fail("weight sum is not a finite non-negative number");
        }
        match (self.clock_min, self.clock_max) {
            (None, None) => {}
            (Some(lo), Some(hi)) if lo.is_finite() && hi.is_finite() && lo <= hi => {}
            _ => return fail("clock range is not a range"),
        }
        if self.session_changes > self.rows_fed
            || self.clock_backwards > self.rows_fed
            || self.resets > self.rows_fed
        {
            return fail("more events than rows");
        }
        if let Some(bad) = self
            .columns
            .iter()
            .position(|c| !c.is_consistent(self.rows_fed))
        {
            return fail(&format!("column {bad}'s statistics do not add up"));
        }
        Ok(())
    }
}

/// A row of [`crate::Bank::summary`]: one stream's numbers, or nulls for a
/// stream restored from a file written before the summary existed.
pub struct SummaryRow<'a> {
    pub group: Option<&'a str>,
    pub rows_processed: u64,
    pub last_clock: Option<f64>,
    pub summary: Option<&'a DataSummary>,
}

/// The `summary` frame for these rows: `group`, `rows_fed`,
/// `rows_processed`, `rows_skipped`, `rows_learned`, `rows_zero_weight`,
/// `weight_sum`, `clock_min`, `clock_max`, `last_clock`, `session_changes`,
/// `clock_backwards`, `resets`.
pub fn summary_frame(rows: &[SummaryRow<'_>]) -> PolarsResult<DataFrame> {
    let s = |f: fn(&DataSummary) -> u64| -> Vec<Option<u64>> {
        rows.iter().map(|r| r.summary.map(f)).collect()
    };
    let cols = vec![
        Column::new(
            "group".into(),
            rows.iter().map(|r| r.group).collect::<Vec<_>>(),
        ),
        Column::new("rows_fed".into(), s(|d| d.rows_fed)),
        Column::new(
            "rows_processed".into(),
            rows.iter().map(|r| r.rows_processed).collect::<Vec<u64>>(),
        ),
        Column::new(
            "rows_skipped".into(),
            rows.iter()
                .map(|r| r.summary.map(|d| d.rows_fed - r.rows_processed))
                .collect::<Vec<Option<u64>>>(),
        ),
        Column::new("rows_learned".into(), s(|d| d.rows_learned)),
        Column::new("rows_zero_weight".into(), s(|d| d.rows_zero_weight)),
        Column::new(
            "weight_sum".into(),
            rows.iter()
                .map(|r| r.summary.map(|d| d.weight_sum))
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "clock_min".into(),
            rows.iter()
                .map(|r| r.summary.and_then(|d| d.clock_min))
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "clock_max".into(),
            rows.iter()
                .map(|r| r.summary.and_then(|d| d.clock_max))
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "last_clock".into(),
            rows.iter()
                .map(|r| r.last_clock)
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new("session_changes".into(), s(|d| d.session_changes)),
        Column::new("clock_backwards".into(), s(|d| d.clock_backwards)),
        Column::new("resets".into(), s(|d| d.resets)),
    ];
    DataFrame::new(rows.len(), cols)
}

/// The `describe` frame for these streams: one row per (group, column) in
/// `layout` order -- `group`, `column`, `role`, `count`, `null_count`,
/// `mean`, `std`, `min`, `max`. `keep` says which layout entries to report
/// (a label target has counts only, an unsupervised model's mirrored
/// target none): `Some(true)` reports the column, `Some(false)` its counts
/// alone, `None` leaves it out. A stream without a summary reports its
/// columns with every number null.
pub fn describe_frame(
    layout: &[(&str, Role)],
    keep: &dyn Fn(usize, Role) -> Option<bool>,
    streams: &[(Option<&str>, Option<&DataSummary>)],
) -> PolarsResult<DataFrame> {
    let mut group: Vec<Option<&str>> = Vec::new();
    let mut column: Vec<&str> = Vec::new();
    let mut role: Vec<&str> = Vec::new();
    let mut count: Vec<Option<u64>> = Vec::new();
    let mut nulls: Vec<Option<u64>> = Vec::new();
    let (mut mean, mut std, mut min, mut max): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
        Default::default();
    for (g, d) in streams {
        for (ci, (name, r)) in layout.iter().enumerate() {
            let Some(numbers) = keep(ci, *r) else {
                continue;
            };
            let c = d.map(|d| &d.columns[ci]);
            group.push(*g);
            column.push(name);
            role.push(r.as_str());
            count.push(c.map(|c| c.count));
            nulls.push(c.map(|c| c.nulls));
            let c = c.filter(|_| numbers);
            mean.push(c.and_then(ColumnStats::mean));
            std.push(c.and_then(ColumnStats::std));
            min.push(c.and_then(ColumnStats::min));
            max.push(c.and_then(ColumnStats::max));
        }
    }
    let height = group.len();
    DataFrame::new(
        height,
        vec![
            Column::new("group".into(), group),
            Column::new("column".into(), column),
            Column::new("role".into(), role),
            Column::new("count".into(), count),
            Column::new("null_count".into(), nulls),
            Column::new("mean".into(), mean),
            Column::new("std".into(), std),
            Column::new("min".into(), min),
            Column::new("max".into(), max),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-pass reference over the usable values.
    fn reference(xs: &[f64]) -> (u64, u64, f64, f64, f64, f64) {
        let ok: Vec<f64> = xs.iter().copied().filter(|x| usable(*x)).collect();
        let n = ok.len() as f64;
        let mean = ok.iter().sum::<f64>() / n;
        let m2 = ok.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
        let min = ok.iter().copied().fold(f64::INFINITY, f64::min);
        let max = ok.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (
            ok.len() as u64,
            (xs.len() - ok.len()) as u64,
            mean,
            m2,
            min,
            max,
        )
    }

    #[test]
    fn welford_matches_a_two_pass_reference_with_nulls_in_the_way() {
        let mut s = 7u64;
        let xs: Vec<f64> = (0..5000)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                match i % 97 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => 1e300,                 // beyond the input bound
                    _ => 1e6 + 3.0 * (u - 0.5), // a large offset against a small spread
                }
            })
            .collect();
        let mut c = ColumnStats::default();
        for (i, &x) in xs.iter().enumerate() {
            let n = i as u64 + 1;
            c.push(x, n, 1.0 / n as f64);
        }
        let (count, nulls, mean, m2, min, max) = reference(&xs);
        assert_eq!((c.count, c.nulls), (count, nulls));
        assert_eq!((c.min, c.max), (min, max));
        assert!(
            (c.mean - mean).abs() <= 1e-9 * mean.abs(),
            "{} vs {mean}",
            c.mean
        );
        assert!((c.m2 - m2).abs() <= 1e-9 * m2, "{} vs {m2}", c.m2);
        assert!(c.is_consistent(xs.len() as u64));
        let std = (m2 / (count as f64 - 1.0)).sqrt();
        assert!((c.std().unwrap() - std).abs() <= 1e-9 * std);
    }

    #[test]
    fn the_shared_reciprocal_gives_the_bits_of_a_private_division() {
        // A column that had a null takes the `1.0 / count` path from then
        // on; a column that did not shares the row's reciprocal. Fed the
        // same values afterwards, both must be exactly the numbers a plain
        // Welford recursion gives.
        let mut shared = ColumnStats::default();
        let mut private = ColumnStats::default();
        let mut plain = ColumnStats::default();
        private.push(f64::NAN, 1, 1.0);
        for i in 0..1000u64 {
            let x = (i as f64 * 0.37).sin() * 1e3 + 1e5;
            shared.push(x, i + 1, 1.0 / (i + 1) as f64);
            private.push(x, i + 2, 1.0 / (i + 2) as f64);
            // Plain Welford: `count` divides, every time.
            plain.count += 1;
            let d = x - plain.mean;
            plain.mean += d / plain.count as f64;
            plain.m2 += d * (x - plain.mean);
        }
        assert_eq!(shared.mean.to_bits(), plain.mean.to_bits());
        assert_eq!(shared.m2.to_bits(), plain.m2.to_bits());
        assert_eq!(private.mean.to_bits(), plain.mean.to_bits());
        assert_eq!(private.m2.to_bits(), plain.m2.to_bits());
    }

    #[test]
    fn empty_and_single_columns_report_nulls_where_they_must() {
        let mut c = ColumnStats::default();
        assert_eq!(
            (c.mean(), c.std(), c.min(), c.max()),
            (None, None, None, None)
        );
        assert!(c.is_consistent(0));
        c.push(2.5, 1, 1.0);
        assert_eq!(
            (c.mean(), c.min(), c.max()),
            (Some(2.5), Some(2.5), Some(2.5))
        );
        assert_eq!(c.std(), None, "one value has no sample deviation");
        assert!(c.is_consistent(1));
        assert!(!c.is_consistent(2), "a row unaccounted for");
    }

    #[test]
    fn inconsistent_statistics_are_seen() {
        let mut c = ColumnStats::default();
        for i in 0..5u64 {
            c.push(i as f64, i + 1, 1.0 / (i + 1) as f64);
        }
        assert!(c.is_consistent(5));
        let mut bad = c.clone();
        bad.min = 10.0; // above the mean
        assert!(!bad.is_consistent(5));
        let mut bad = c.clone();
        bad.m2 = f64::NAN;
        assert!(!bad.is_consistent(5));
        let mut bad = c.clone();
        bad.nulls = 1;
        assert!(!bad.is_consistent(5));
        assert!(bad.is_consistent(6));
    }
}
