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

use online_core::ClockValue;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arrow::ClockCol;
use crate::rows::FeatureRows;
use crate::spec::Spec;
use crate::stream::{ClockDtype, all_usable, usable};

/// Count, nulls and moments of one input column over the rows fed.
///
/// `mean` and `m2` (the sum of squared deviations) are Welford's; `min` and
/// `max` start at the identities so the update has no first-row branch and
/// are reported as null while `count` is 0. A value is counted when
/// [`usable`] -- finite and within the input bound -- and a null otherwise:
/// polars nulls arrive as NaN, so "null" here is null, NaN, an infinity or
/// a magnitude the models would not accept.
///
/// The identities are infinite, and a column no row gave a value keeps
/// them, so `min` and `max` are written as tags in JSON, as every other
/// non-finite float is: without it `ModelBank.to_json` refused such a bank
/// as a dropped value (found 2026-09-29). The binary state is unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnStats {
    pub count: u64,
    pub nulls: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(with = "online_core::humanfloat::f64_or_tag")]
    pub min: f64,
    #[serde(with = "online_core::humanfloat::f64_or_tag")]
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
        // Every count here saturates: a state file can carry one at the top
        // of its range, and a wrapped count reads as a fresh one (task 160,
        // PA2).
        if usable(x) {
            self.count = self.count.saturating_add(1);
            let inv = if self.count == n_row {
                inv_row
            } else {
                1.0 / self.count as f64
            };
            self.add(x, inv);
        } else {
            self.nulls = self.nulls.saturating_add(1);
        }
    }

    /// Welford's step for a value just counted, `inv` the reciprocal of the
    /// new count.
    #[inline]
    fn add(&mut self, x: f64, inv: f64) {
        let d = x - self.mean;
        self.mean += d * inv;
        self.m2 += d * (x - self.mean);
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }

    /// A value its row was fed without, known now: counted a null when the
    /// row was fed, it moves from the nulls to the moments when usable, so
    /// the counts still partition the rows fed. An unusable one stays the
    /// null it was counted as.
    #[inline]
    fn resolve(&mut self, x: f64) {
        if usable(x) && self.nulls > 0 {
            self.nulls -= 1;
            self.count = self.count.saturating_add(1);
            self.add(x, 1.0 / self.count as f64);
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
        // Checked: a file's counts can sum past the top of the range, and a
        // wrapped sum matched a small `fed` (task 160, PA2).
        if self.count.checked_add(self.nulls) != Some(fed)
            || !self.mean.is_finite()
            || !self.m2.is_finite()
        {
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

/// Feature columns' statistics, one array per statistic, for
/// [`DataSummary::feed_rows`]: the same numbers as the [`ColumnStats`] they
/// are copied from and back to, laid out so one row's step across the
/// columns is a vector loop.
struct Columns {
    mean: Vec<f64>,
    m2: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

impl Columns {
    fn of(cols: &[ColumnStats]) -> Self {
        Self {
            mean: cols.iter().map(|c| c.mean).collect(),
            m2: cols.iter().map(|c| c.m2).collect(),
            min: cols.iter().map(|c| c.min).collect(),
            max: cols.iter().map(|c| c.max).collect(),
        }
    }

    /// Back into `cols`, which [`Self::of`] read, each count now `count`:
    /// every row on the path stepped each column's count to the row's
    /// ordinal, so the last row's is every column's, and with no row it is
    /// the count each column had, the path being taken only when every
    /// column's equals the rows fed. `nulls` never moved.
    fn write(&self, cols: &mut [ColumnStats], count: u64) {
        for (j, c) in cols.iter_mut().enumerate() {
            c.count = count;
            c.mean = self.mean[j];
            c.m2 = self.m2[j];
            c.min = self.min[j];
            c.max = self.max[j];
        }
    }

    /// One row of usable values into columns that have counted every row
    /// before it: [`ColumnStats::push`]'s usable branch, where each count
    /// steps to the row's ordinal (written once, by [`Self::write`]) and so
    /// takes the shared reciprocal `inv`. `ColumnStats::add` written across
    /// the columns, operation for operation.
    #[inline]
    fn add_all(&mut self, x: &[f64], inv: f64) {
        let n = x.len();
        let (mean, m2, min, max) = (
            &mut self.mean[..n],
            &mut self.m2[..n],
            &mut self.min[..n],
            &mut self.max[..n],
        );
        for j in 0..n {
            let xj = x[j];
            let d = xj - mean[j];
            mean[j] += d * inv;
            m2[j] += d * (xj - mean[j]);
            min[j] = min[j].min(xj);
            max[j] = max[j].max(xj);
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Rows [`DataSummary::feed_rows`] checked for usable features itself.
    static CHECKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One input row of a run, as the clock schedule decided it: what
/// [`DataSummary::feed_rows`] needs beyond the columns.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Fed {
    /// The row's index in the input columns.
    pub i: usize,
    /// Every feature of the row is known to be usable: the schedule
    /// accepted it, which for every model but an `audit` it does only then.
    /// The path across the columns skips its own check for such a row.
    pub usable: bool,
    /// What [`DataSummary::events`] counts.
    pub session_changed: bool,
    pub backwards: bool,
    pub reset: bool,
    /// The row was accepted, at weight `w` ([`DataSummary::accepted`]).
    pub accept: bool,
    pub w: f64,
}

/// [`DataSummary::events`] and [`DataSummary::accepted`] over a run, kept
/// in locals and added to the summary once at its end. Each count only
/// grows by one a row, so adding the run's total with one saturating add
/// leaves what a saturating add a row would; the weight is summed in row
/// order from the summary's own, so its bits are the per-row sum's.
struct RunCounts {
    session_changes: u64,
    clock_backwards: u64,
    resets: u64,
    zero_weight: u64,
    learned: u64,
    weight_sum: f64,
}

impl RunCounts {
    fn new(weight_sum: f64) -> Self {
        Self {
            session_changes: 0,
            clock_backwards: 0,
            resets: 0,
            zero_weight: 0,
            learned: 0,
            weight_sum,
        }
    }

    /// Row `r`: a plain target counts at its row (a formula target counts
    /// when the row is released, [`DataSummary::learned_late`]).
    #[inline]
    fn add(&mut self, r: &Fed, targets: &[Vec<f64>]) {
        self.session_changes += u64::from(r.session_changed);
        self.clock_backwards += u64::from(r.backwards);
        self.resets += u64::from(r.reset);
        if r.accept {
            self.weight_sum += r.w;
            if r.w == 0.0 {
                self.zero_weight += 1;
            } else if targets.is_empty() || targets.iter().any(|t| usable(t[r.i])) {
                self.learned += 1;
            }
        }
    }

    fn write(self, s: &mut DataSummary) {
        s.session_changes = s.session_changes.saturating_add(self.session_changes);
        s.clock_backwards = s.clock_backwards.saturating_add(self.clock_backwards);
        s.resets = s.resets.saturating_add(self.resets);
        s.rows_zero_weight = s.rows_zero_weight.saturating_add(self.zero_weight);
        s.rows_learned = s.rows_learned.saturating_add(self.learned);
        s.weight_sum = self.weight_sum;
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
    /// The clock range over the rows fed, as the stream holds its clock: a
    /// temporal one in integer nanoseconds, so the frames give it back
    /// exactly in the column's own dtype, where a double of seconds was 64 ns
    /// off at 2024 (review round 4, N18). `None` on a row-count clock or
    /// before the first row.
    pub clock_min: Option<ClockValue>,
    pub clock_max: Option<ClockValue>,
    /// Rows whose session id differed from the previous row's.
    pub session_changes: u64,
    /// Rows whose clock was below the previous row's within a session --
    /// what `restart_after_step_back` had to decide about.
    pub clock_backwards: u64,
    /// Rows at which the session or clock policy restarted the stream's
    /// state (`session_gap = "reset"`, a step back of at least
    /// `restart_after_step_back`).
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
    /// into the column statistics, its clock into the range. `features` is
    /// the row's feature values, `targets` has one vector per spec target
    /// (empty until the bank fills a comparison's), `weight` the weight
    /// column's value when the spec has one; `clock` is `None` on a
    /// row-count clock.
    #[inline]
    pub fn feed_row(
        &mut self,
        features: &[f64],
        targets: &[Vec<f64>],
        weight: Option<f64>,
        clock: Option<ClockValue>,
        i: usize,
    ) {
        self.rows_fed = self.rows_fed.saturating_add(1);
        let n_row = self.rows_fed;
        let inv_row = 1.0 / n_row as f64;
        let nf = features.len();
        // The layout's target slots, whether or not the caller filled them:
        // a target the row has no vector for is a null, so the counts still
        // add up to the rows fed.
        let n_cols = self.columns.len();
        let nt = n_cols.saturating_sub(nf + usize::from(weight.is_some()));
        for (c, &f) in self.columns.iter_mut().zip(features) {
            c.push(f, n_row, inv_row);
        }
        for (k, c) in self.columns[nf.min(n_cols)..]
            .iter_mut()
            .take(nt)
            .enumerate()
        {
            match targets.get(k) {
                Some(t) => c.push(t[i], n_row, inv_row),
                None => c.nulls = c.nulls.saturating_add(1),
            }
        }
        if let (Some(w), Some(c)) = (weight, self.columns.get_mut(nf + nt)) {
            c.push(w, n_row, inv_row);
        }
        if let Some(c) = clock {
            self.clock_range(c);
        }
    }

    /// The clock range with one more row's clock.
    #[inline]
    fn clock_range(&mut self, c: ClockValue) {
        (self.clock_min, self.clock_max) = Self::range_with(self.clock_min, self.clock_max, c);
    }

    /// The range `(lo, hi)` with one more clock `c`: the earliest kept on a
    /// tie, as the latest is.
    #[inline]
    fn range_with(
        lo: Option<ClockValue>,
        hi: Option<ClockValue>,
        c: ClockValue,
    ) -> (Option<ClockValue>, Option<ClockValue>) {
        (
            Some(lo.map_or(c, |m| if c.is_before(m) { c } else { m })),
            Some(hi.map_or(c, |m| if m.is_before(c) { c } else { m })),
        )
    }

    /// The input rows `rows` of a run, in order: for each, what
    /// [`Self::feed_row`], [`Self::events`] and, for an accepted row,
    /// [`Self::accepted`] do, to the bit, in one pass over the rows and
    /// faster where it can be.
    ///
    /// While every feature column has counted every row fed -- no feature
    /// has been null in this stream -- and the run's rows have every feature
    /// usable, each feature column's step is the same: one more value, the
    /// row's shared reciprocal, Welford's update. Those columns are copied
    /// out into one array per statistic for the run, where the step runs
    /// across the columns as a loop the compiler vectorises, and copied
    /// back at the end. From the first row with a feature that is not
    /// usable, and in a stream where one has been, the rows go through
    /// [`Self::feed_row`] one by one. Each column sees the same values in
    /// the same order with the same reciprocal either way, so the bits do
    /// not depend on the path. Measured in a microbenchmark of this code:
    /// 18.4 to 13.4 ns a row at 20 features, 10.1 to 8.4 at 8, 42.5 to 28.7
    /// at 50; at 4 the copy cost more than the loop saved, 7.4 against 8.3,
    /// so fewer than [`Self::ACROSS_COLUMNS_FROM`] features take the
    /// row-by-row path.
    ///
    /// A row the schedule accepted ([`Fed::usable`]) is not checked again;
    /// the clock range is kept in locals; the counts are summed over the run
    /// ([`RunCounts`]). One pass where there were two, over the rows the
    /// models read next: switching the summary off had been worth about 20
    /// ns of a 130 ns `sgd` row.
    ///
    /// Returns how many rows took the path across the columns, which only
    /// a test reads.
    pub(crate) fn feed_rows(
        &mut self,
        rows: impl IntoIterator<Item = Fed>,
        features: &FeatureRows,
        targets: &[Vec<f64>],
        weight: Option<&[f64]>,
        clock: Option<&ClockCol>,
    ) -> usize {
        let mut rows = rows.into_iter();
        let nf = features.k();
        let n_cols = self.columns.len();
        let has_w = usize::from(weight.is_some());
        let mut counts = RunCounts::new(self.weight_sum);
        let mut across = 0;
        let mut first_slow = None;
        if nf >= Self::ACROSS_COLUMNS_FROM
            && nf + has_w <= n_cols
            && self.columns[..nf].iter().all(|c| c.count == self.rows_fed)
        {
            let nt = n_cols - nf - has_w;
            let mut soa = Columns::of(&self.columns[..nf]);
            let (mut lo, mut hi) = (self.clock_min, self.clock_max);
            for r in rows.by_ref() {
                let i = r.i;
                let x = features.row(i);
                if !(r.usable || Self::check_row(x)) {
                    first_slow = Some(r);
                    break;
                }
                self.rows_fed = self.rows_fed.saturating_add(1);
                let n_row = self.rows_fed;
                let inv_row = 1.0 / n_row as f64;
                debug_assert!(all_usable(x), "row {i} was passed as usable");
                soa.add_all(x, inv_row);
                for (k, c) in self.columns[nf..].iter_mut().take(nt).enumerate() {
                    match targets.get(k) {
                        Some(t) => c.push(t[i], n_row, inv_row),
                        None => c.nulls = c.nulls.saturating_add(1),
                    }
                }
                if let Some(w) = weight {
                    self.columns[nf + nt].push(w[i], n_row, inv_row);
                }
                if let Some(c) = clock {
                    (lo, hi) = Self::range_with(lo, hi, c.at(i));
                }
                counts.add(&r, targets);
                across += 1;
            }
            (self.clock_min, self.clock_max) = (lo, hi);
            soa.write(&mut self.columns[..nf], self.rows_fed);
        }
        for r in first_slow.into_iter().chain(rows) {
            let i = r.i;
            self.feed_row(
                features.row(i),
                targets,
                weight.map(|w| w[i]),
                clock.map(|c| c.at(i)),
                i,
            );
            counts.add(&r, targets);
        }
        counts.write(self);
        across
    }

    /// The path's own check of a row not known to be usable; the tests
    /// count the rows it was asked about.
    #[inline]
    fn check_row(x: &[f64]) -> bool {
        #[cfg(test)]
        CHECKED.with(|c| c.set(c.get() + 1));
        all_usable(x)
    }

    /// The fewest features [`Self::feed_rows`] steps across the columns.
    pub(crate) const ACROSS_COLUMNS_FROM: usize = 8;

    /// What the clock schedule found at the row just fed.
    #[inline]
    pub fn events(&mut self, session_changed: bool, backwards: bool, reset: bool) {
        self.session_changes = self
            .session_changes
            .saturating_add(u64::from(session_changed));
        self.clock_backwards = self.clock_backwards.saturating_add(u64::from(backwards));
        self.resets = self.resets.saturating_add(u64::from(reset));
    }

    /// A row accepted earlier, counted without a target, is learned from
    /// now: a formula target resolved with a value (docs/PLAN.md task 104,
    /// review R1 D3).
    #[inline]
    pub fn learned_late(&mut self) {
        self.rows_learned = self.rows_learned.saturating_add(1);
    }

    /// A row held for its formula target is released to the models with
    /// `y` as target `target` (an index into the spec's targets). The row
    /// was fed before its window closed, and its target counted a null
    /// then; the value joins the target's statistics now, in release order
    /// -- row order, whatever the chunking -- so `describe` reads what the
    /// models were handed, as `rows_learned` counts it (review round 4,
    /// PA9: every such value went uncounted). A row still held, or one
    /// whose window was cut, stays a null.
    #[inline]
    pub fn resolved(&mut self, n_features: usize, target: usize, y: f64) {
        if let Some(c) = self.columns.get_mut(n_features + target) {
            c.resolve(y);
        }
    }

    /// The row just fed was accepted: its weight, and whether the models
    /// were handed a target to learn from (a model without targets always
    /// is).
    #[inline]
    pub fn accepted(&mut self, w: f64, has_target: bool) {
        self.weight_sum += w;
        if w == 0.0 {
            self.rows_zero_weight = self.rows_zero_weight.saturating_add(1);
        } else if has_target {
            self.rows_learned = self.rows_learned.saturating_add(1);
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
        // Saturating: `u64::MAX + 1` wrapped to 0 learned rows and loaded
        // (task 160, PA2).
        if self.rows_learned.saturating_add(self.rows_zero_weight) > rows_seen {
            return fail("more rows learned than processed");
        }
        if !(self.weight_sum.is_finite() && self.weight_sum >= 0.0) {
            return fail("weight sum is not a finite non-negative number");
        }
        let finite = |c: ClockValue| match c {
            ClockValue::F64(v) => v.is_finite(),
            ClockValue::Ns(_) | ClockValue::I64(_) => true,
        };
        match (self.clock_min, self.clock_max) {
            (None, None) => {}
            (Some(lo), Some(hi)) if finite(lo) && finite(hi) && !hi.is_before(lo) => {}
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
    pub last_clock: Option<ClockValue>,
    pub summary: Option<&'a DataSummary>,
    /// The solves that needed jitter or kept the previous fit, over the
    /// stream's models (`Stream::solve_failures`).
    pub solve_failures: u64,
    /// Where the stream stands on the readiness statistics
    /// (docs/WARMUP-AND-CONVERGENCE.md §3); `None` where a caller has no
    /// stream to read them from.
    pub readiness: Option<crate::stream::Readiness>,
    /// Whether the spec fits a model. An `audit` does not: it reads every
    /// row and learns from none, so its `rows_learned` and `n_coef` are
    /// null, where they counted a row with its first column usable and the
    /// columns plus one (review 6, C-7).
    pub fits: bool,
}

/// A count as the frames give every count: an `Int64`, polars' default
/// integer, one dtype in every frame and field of the library (task 244).
/// A count at the top of a `u64` -- a state file can carry one, since every
/// count saturates -- reads `i64::MAX`, never a negative number.
pub fn count(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// A column of counts, [`count`] each, null where a row has none.
pub fn counts(name: &str, values: impl IntoIterator<Item = Option<u64>>) -> Column {
    Column::new(
        name.into(),
        values
            .into_iter()
            .map(|v| v.map(count))
            .collect::<Vec<Option<i64>>>(),
    )
}

/// The `summary` frame for these rows, 20 columns: `group`, `rows_fed`,
/// `rows_processed`, `rows_skipped`, `rows_learned`, `rows_zero_weight`,
/// `weight_sum`, `clock_min`, `clock_max`, `clock_last`, `session_changes`,
/// `clock_backwards`, `resets`, `solve_failures`, and the readiness
/// statistics (docs/WARMUP-AND-CONVERGENCE.md §3): `settled_frac`,
/// `weight_sum_settled`, `error_inflation`, `support_coef_min`,
/// `support_coef_min_feature` and `n_coef`. Every count is an `Int64`
/// ([`count`]); the three clock columns are in `clock`'s dtype
/// ([`ClockDtype::column`]).
pub fn summary_frame(
    rows: &[SummaryRow<'_>],
    clock: Option<&ClockDtype>,
) -> PolarsResult<DataFrame> {
    let clocks = |name: &str, f: &dyn Fn(&SummaryRow<'_>) -> Option<ClockValue>| {
        ClockDtype::column(clock, name, &rows.iter().map(f).collect::<Vec<_>>())
    };
    let s = |name: &str, f: fn(&DataSummary) -> u64| -> Column {
        counts(name, rows.iter().map(|r| r.summary.map(f)))
    };
    let cols = vec![
        Column::new(
            "group".into(),
            rows.iter().map(|r| r.group).collect::<Vec<_>>(),
        ),
        s("rows_fed", |d| d.rows_fed),
        counts(
            "rows_processed",
            rows.iter().map(|r| Some(r.rows_processed)),
        ),
        counts(
            "rows_skipped",
            rows.iter()
                .map(|r| r.summary.map(|d| d.rows_fed - r.rows_processed)),
        ),
        counts(
            "rows_learned",
            rows.iter()
                .map(|r| r.summary.filter(|_| r.fits).map(|d| d.rows_learned)),
        ),
        s("rows_zero_weight", |d| d.rows_zero_weight),
        Column::new(
            "weight_sum".into(),
            rows.iter()
                .map(|r| r.summary.map(|d| d.weight_sum))
                .collect::<Vec<Option<f64>>>(),
        ),
        clocks("clock_min", &|r| r.summary.and_then(|d| d.clock_min))?,
        clocks("clock_max", &|r| r.summary.and_then(|d| d.clock_max))?,
        clocks("clock_last", &|r| r.last_clock)?,
        s("session_changes", |d| d.session_changes),
        s("clock_backwards", |d| d.clock_backwards),
        s("resets", |d| d.resets),
        counts(
            "solve_failures",
            rows.iter().map(|r| Some(r.solve_failures)),
        ),
        // The readiness statistics, NaN as null: a fraction that does not
        // exist (no decay), a ratio the model has not got, a share before
        // the first solve.
        Column::new(
            "settled_frac".into(),
            rows.iter()
                .map(|r| {
                    r.readiness
                        .as_ref()
                        .map(|x| x.settled_frac)
                        .filter(|v| v.is_finite())
                })
                .collect::<Vec<Option<f64>>>(),
        ),
        // The weight the stream settles at (`crate::stream::settled_weight`),
        // null where there is none: no decay, a window, a single row.
        Column::new(
            "weight_sum_settled".into(),
            rows.iter()
                .map(|r| {
                    r.readiness
                        .as_ref()
                        .map(|x| x.weight_sum_settled)
                        .filter(|v| v.is_finite())
                })
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "error_inflation".into(),
            rows.iter()
                .map(|r| {
                    r.readiness
                        .as_ref()
                        .map(|x| x.error_inflation)
                        .filter(|v| v.is_finite())
                })
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "support_coef_min".into(),
            rows.iter()
                .map(|r| {
                    r.readiness
                        .as_ref()
                        .map(|x| x.min_support_coef)
                        .filter(|v| v.is_finite())
                })
                .collect::<Vec<Option<f64>>>(),
        ),
        Column::new(
            "support_coef_min_feature".into(),
            rows.iter()
                .map(|r| {
                    r.readiness
                        .as_ref()
                        .and_then(|x| x.min_support_coef_feature.as_deref())
                })
                .collect::<Vec<Option<&str>>>(),
        ),
        counts(
            "n_coef",
            rows.iter()
                .map(|r| r.readiness.as_ref().filter(|_| r.fits).map(|x| x.n_coef)),
        ),
    ];
    DataFrame::new(rows.len(), cols)
}

/// The `describe` frame for these streams: one row per (group, column) in
/// `layout` order -- `group`, `column`, `role`, `count`, `unusable_count`,
/// `mean`, `std`, `min`, `max`. `unusable_count` is not polars'
/// `null_count`: it counts every value a model cannot learn from, a NaN, an
/// infinity and a magnitude past the bound with the nulls, so `count` and
/// it partition the rows. `keep` says which layout entries to report
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
    let mut usable: Vec<Option<u64>> = Vec::new();
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
            usable.push(c.map(|c| c.count));
            nulls.push(c.map(|c| c.nulls));
            let c = c.filter(|_| numbers);
            mean.push(c.and_then(ColumnStats::mean));
            std.push(c.and_then(ColumnStats::std));
            min.push(c.and_then(ColumnStats::min));
            max.push(c.and_then(ColumnStats::max));
        }
    }
    let height = group.len();
    // An `Enum` over the three roles, in their order (task 244).
    let roles = [Role::Feature, Role::Target, Role::Weight].map(Role::as_str);
    let role = Column::new("role".into(), role).cast(&DataType::from_frozen_categories(
        FrozenCategories::new(roles).expect("distinct names"),
    ))?;
    DataFrame::new(
        height,
        vec![
            Column::new("group".into(), group),
            Column::new("column".into(), column),
            role,
            counts("count", usable),
            counts("unusable_count", nulls),
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

    /// `n` rows of `k` features, one target and a weight, with the values
    /// `odd` puts in place of the generated ones: a large offset against a
    /// small spread, signed zeros at rows 3 and 4 of every column (`min` and
    /// `max` decide between two equal values there), and a clock.
    fn rows(n: usize, k: usize, odd: &dyn Fn(usize, usize) -> Option<f64>) -> Rows {
        let mut s = 11u64;
        let mut r = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..n)
                    .map(|i| match (odd(i, j), i) {
                        (Some(v), _) => v,
                        (None, 3) => -0.0,
                        (None, 4) => 0.0,
                        _ => 1e6 + 3.0 * r(),
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        Rows {
            features: FeatureRows::from_columns(&refs, n, None, false),
            targets: vec![
                (0..n)
                    .map(|i| if i % 9 == 2 { f64::NAN } else { r() })
                    .collect(),
            ],
            weight: (0..n)
                .map(|i| if i % 7 == 1 { 0.0 } else { 0.5 + r().abs() })
                .collect(),
            clock: ClockCol::F64((0..n).map(|i| (i as f64 * 0.3).sin() + i as f64).collect()),
        }
    }

    struct Rows {
        features: FeatureRows,
        targets: Vec<Vec<f64>>,
        weight: Vec<f64>,
        clock: ClockCol,
    }

    /// A summary laid out for `k` features, one target and a weight.
    fn fresh(k: usize) -> DataSummary {
        let mut s = DataSummary {
            rows_fed: 0,
            rows_learned: 0,
            rows_zero_weight: 0,
            weight_sum: 0.0,
            clock_min: None,
            clock_max: None,
            session_changes: 0,
            clock_backwards: 0,
            resets: 0,
            columns: Vec::new(),
        };
        s.columns = vec![ColumnStats::default(); k + 2];
        s
    }

    /// `rows` through [`DataSummary::feed_row`], one at a time.
    fn by_row(s: &mut DataSummary, r: &Rows, rows: std::ops::Range<usize>) {
        for i in rows {
            s.feed_row(
                r.features.row(i),
                &r.targets,
                Some(r.weight[i]),
                Some(r.clock.at(i)),
                i,
            );
        }
    }

    /// `rows` through [`DataSummary::feed_rows`] in runs of `run` rows: the
    /// rows each run took across the columns.
    fn by_runs(
        s: &mut DataSummary,
        r: &Rows,
        rows: std::ops::Range<usize>,
        run: usize,
    ) -> Vec<usize> {
        let idx: Vec<usize> = rows.collect();
        idx.chunks(run)
            .map(|c| {
                s.feed_rows(
                    c.iter().map(|&i| Fed {
                        i,
                        ..Fed::default()
                    }),
                    &r.features,
                    &r.targets,
                    Some(&r.weight),
                    Some(&r.clock),
                )
            })
            .collect()
    }

    /// Every number of two summaries, to the bit.
    fn assert_bits(a: &DataSummary, b: &DataSummary, what: &str) {
        assert_eq!(a.rows_fed, b.rows_fed, "{what}: rows_fed");
        assert_eq!(a.clock_min, b.clock_min, "{what}: clock_min");
        assert_eq!(a.clock_max, b.clock_max, "{what}: clock_max");
        assert_eq!(a.columns.len(), b.columns.len());
        for (j, (p, q)) in a.columns.iter().zip(&b.columns).enumerate() {
            let bits = |c: &ColumnStats| {
                (
                    c.count,
                    c.nulls,
                    c.mean.to_bits(),
                    c.m2.to_bits(),
                    c.min.to_bits(),
                    c.max.to_bits(),
                )
            };
            assert_eq!(bits(p), bits(q), "{what}: column {j}");
        }
    }

    /// The path across the columns (task 250): taken for every row of a
    /// stream whose features are all usable, at the width it starts at and
    /// above, and the same bits as the rows fed one at a time -- the
    /// signed zeros' `min` and `max` included -- whatever the runs.
    #[test]
    fn feed_rows_across_the_columns_is_feed_row_to_the_bit() {
        let n = 500;
        for k in [DataSummary::ACROSS_COLUMNS_FROM, 20, 33] {
            let r = rows(n, k, &|_, _| None);
            let mut want = fresh(k);
            by_row(&mut want, &r, 0..n);
            assert_eq!(want.columns[0].min.to_bits(), (-0.0f64).to_bits());
            assert_eq!(want.columns[0].max.to_bits() >> 63, 0);
            for run in [1, 7, 64, n] {
                let mut got = fresh(k);
                let across = by_runs(&mut got, &r, 0..n, run);
                assert_eq!(across.iter().sum::<usize>(), n, "k={k}, runs of {run}");
                assert_bits(&want, &got, &format!("k={k}, runs of {run}"));
            }
        }
    }

    /// Below that width every row goes one at a time, and the bits are the
    /// same.
    #[test]
    fn feed_rows_below_the_width_goes_row_by_row() {
        let (n, k) = (300, DataSummary::ACROSS_COLUMNS_FROM - 1);
        let r = rows(n, k, &|_, _| None);
        let mut want = fresh(k);
        by_row(&mut want, &r, 0..n);
        let mut got = fresh(k);
        assert_eq!(by_runs(&mut got, &r, 0..n, 64), vec![0; 5]);
        assert_bits(&want, &got, "narrow");
    }

    /// A feature that is not usable ends the path at its row, mid-run: the
    /// rows before it went across the columns, it and the rest of the run
    /// one at a time, and every later run too, since that column has a null
    /// behind it now and takes its own reciprocal (`1.0 / count`) where the
    /// others share the row's. A null, NaN, an infinity and a value past
    /// the bound each end it. The bits are the rows fed one at a time.
    #[test]
    fn feed_rows_leaves_the_path_at_the_first_unusable_feature() {
        let (n, k, bad_row) = (400, 12, 150);
        for bad in [f64::NAN, f64::INFINITY, -1e300] {
            let r = rows(n, k, &|i, j| (i == bad_row && j == 5).then_some(bad));
            let mut want = fresh(k);
            by_row(&mut want, &r, 0..n);
            let mut got = fresh(k);
            let across = by_runs(&mut got, &r, 0..n, 100);
            assert_eq!(across, vec![100, 50, 0, 0], "{bad}");
            assert_eq!(got.columns[5].nulls, 1);
            assert_bits(&want, &got, &format!("{bad} at row {bad_row}"));
        }
    }

    /// A stream whose summary was saved with a null feature behind it, and
    /// loaded, takes no row across the columns: the counts say so, not
    /// anything kept beside them.
    #[test]
    fn feed_rows_reads_the_path_off_the_counts_after_a_load() {
        let (n, k) = (300, 10);
        let r = rows(n, k, &|i, j| (i == 20 && j == 0).then_some(f64::NAN));
        let mut head = fresh(k);
        by_row(&mut head, &r, 0..100);
        let bytes = rmp_serde::to_vec_named(&head).unwrap();
        let mut loaded: DataSummary = rmp_serde::from_slice(&bytes).unwrap();
        let mut want = head.clone();
        by_row(&mut want, &r, 100..n);
        assert_eq!(by_runs(&mut loaded, &r, 100..n, 50), vec![0; 4]);
        assert_bits(&want, &loaded, "with a null behind it");
        // And a stream with none behind it takes every row across, loaded.
        let r = rows(n, k, &|_, _| None);
        let mut head = fresh(k);
        by_row(&mut head, &r, 0..100);
        let bytes = rmp_serde::to_vec_named(&head).unwrap();
        let mut loaded: DataSummary = rmp_serde::from_slice(&bytes).unwrap();
        let mut want = head.clone();
        by_row(&mut want, &r, 100..n);
        assert_eq!(by_runs(&mut loaded, &r, 100..n, 50), vec![50; 4]);
        assert_bits(&want, &loaded, "with no null behind it");
    }

    /// Counts at the top of their range, as a state file can carry them:
    /// `rows_fed` saturates, every column's count with it, and the path
    /// across the columns steps them as `ColumnStats::push` does -- to the
    /// same `u64::MAX`, with the same shared reciprocal.
    #[test]
    fn feed_rows_at_saturated_counts_is_feed_row_to_the_bit() {
        let (n, k) = (50, 9);
        let r = rows(n, k, &|_, _| None);
        let mut start = fresh(k);
        start.rows_fed = u64::MAX;
        for c in &mut start.columns {
            c.count = u64::MAX;
            c.mean = 1e6;
            c.min = 1e6;
            c.max = 1e6;
        }
        let mut want = start.clone();
        by_row(&mut want, &r, 0..n);
        let mut got = start;
        assert_eq!(by_runs(&mut got, &r, 0..n, n), vec![n]);
        assert_bits(&want, &got, "saturated");
    }

    /// The facts the schedule hands each row of `r`: a session change, a
    /// step back and a reset now and then, every fifth row not accepted,
    /// the weight column's weight, and the features known usable on an
    /// accepted row whose features are.
    fn facts(r: &Rows, i: usize) -> Fed {
        let accept = i % 5 != 4;
        Fed {
            i,
            usable: accept && all_usable(r.features.row(i)),
            session_changed: i.is_multiple_of(11),
            backwards: i % 13 == 5,
            reset: i % 17 == 3,
            accept,
            w: r.weight[i],
        }
    }

    /// `rows` as the stream fed them before the one pass: the columns a
    /// row at a time, then `events` and `accepted` a row at a time.
    fn by_row_counted(s: &mut DataSummary, r: &Rows, rows: std::ops::Range<usize>) {
        by_row(s, r, rows.clone());
        for i in rows {
            let f = facts(r, i);
            s.events(f.session_changed, f.backwards, f.reset);
            if f.accept {
                let has_target = r.targets.iter().any(|t| usable(t[i]));
                s.accepted(f.w, has_target);
            }
        }
    }

    fn counts(s: &DataSummary) -> [u64; 6] {
        [
            s.session_changes,
            s.clock_backwards,
            s.resets,
            s.rows_zero_weight,
            s.rows_learned,
            s.weight_sum.to_bits(),
        ]
    }

    /// The one pass counts what `events` and `accepted` counted a row at a
    /// time, the weight's sum to the bit, whatever the runs, on the path
    /// across the columns, below its width, from a row with a feature that
    /// is not usable on, and at counts at the top of their range; and the
    /// columns are the same bits as ever.
    #[test]
    fn feed_rows_counts_what_events_and_accepted_counted() {
        let n = 400;
        let cases: [(usize, Option<usize>, bool); 4] = [
            (20, None, false),
            (DataSummary::ACROSS_COLUMNS_FROM - 1, None, false),
            (12, Some(150), false),
            (12, None, true),
        ];
        for (k, bad_row, saturated) in cases {
            let r = rows(n, k, &|i, j| {
                (Some(i) == bad_row && j == 5).then_some(f64::NAN)
            });
            let mut start = fresh(k);
            if saturated {
                start.session_changes = u64::MAX - 2;
                start.clock_backwards = u64::MAX;
                start.resets = u64::MAX - 1;
                start.rows_zero_weight = u64::MAX - 3;
                start.rows_learned = u64::MAX - 5;
                start.weight_sum = 1e300;
            }
            let mut want = start.clone();
            by_row_counted(&mut want, &r, 0..n);
            assert!(want.rows_learned > 0 && want.rows_zero_weight > 0 && want.resets > 0);
            for run in [1, 7, 64, n] {
                let mut got = start.clone();
                let idx: Vec<usize> = (0..n).collect();
                let across: usize = idx
                    .chunks(run)
                    .map(|c| {
                        got.feed_rows(
                            c.iter().map(|&i| facts(&r, i)),
                            &r.features,
                            &r.targets,
                            Some(&r.weight),
                            Some(&r.clock),
                        )
                    })
                    .sum();
                let what = format!("k={k} bad={bad_row:?} saturated={saturated} runs of {run}");
                assert_eq!(counts(&got), counts(&want), "{what}");
                assert_bits(&want, &got, &what);
                let want_across = match (k >= DataSummary::ACROSS_COLUMNS_FROM, bad_row) {
                    (false, _) => 0,
                    (true, None) => n,
                    (true, Some(b)) => b,
                };
                assert_eq!(across, want_across, "{what}");
            }
        }
    }

    /// A row the schedule passed as usable is not checked again on the
    /// path across the columns, and one it did not pass is; the bits are
    /// the same either way.
    #[test]
    fn a_row_passed_as_usable_is_not_checked_again() {
        let (n, k) = (300, 16);
        let r = rows(n, k, &|_, _| None);
        let mut want = fresh(k);
        by_row(&mut want, &r, 0..n);
        for passed in [true, false] {
            let mut got = fresh(k);
            CHECKED.with(|c| c.set(0));
            let across = got.feed_rows(
                (0..n).map(|i| Fed {
                    i,
                    usable: passed,
                    ..Fed::default()
                }),
                &r.features,
                &r.targets,
                Some(&r.weight),
                Some(&r.clock),
            );
            assert_eq!(across, n);
            let checked = CHECKED.with(|c| c.get());
            assert_eq!(checked, if passed { 0 } else { n }, "passed={passed}");
            assert_bits(&want, &got, &format!("passed={passed}"));
        }
    }
}
