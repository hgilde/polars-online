//! Refresh-time sampling (docs/ENHANCEMENTS.md E58): a common grid for
//! series that tick at their own times.
//!
//! Barndorff-Nielsen, Hansen, Lunde & Shephard's rule, their Definition 1.
//! Given `m` series observed at their own irregular times, a grid point is
//! placed at the first instant by which **every** series has ticked at least
//! once since the previous point:
//!
//! ```text
//! tau_0     = max_i (first tick of series i)
//! tau_{j+1} = max_i (first tick of series i strictly after tau_j)
//! ```
//!
//! and each series contributes its last value at that instant. Two properties
//! make it the right sampler for a correlation over asynchronous data.
//! Nothing is interpolated -- every value in the output was observed -- and
//! the grid adapts: a quiet series slows the whole grid down rather than
//! being carried forward across a stale interval.
//!
//! What it costs is rows. The number of grid points is at most the tick count
//! of the slowest series, so a fast series' ticks are mostly dropped;
//! `retained_fraction` reports how many survived, which is the number to look
//! at before trusting a correlation computed on the result.
//!
//! **Ties are broken by row order.** "Strictly after `tau_j`" is read
//! against the row sequence, not the timestamp alone: a tick that carries
//! the same timestamp as the one that just closed a grid point, but arrives
//! later in the frame, belongs to the next interval. Tick data comes in
//! sequence and a timestamp is rarely finer than the sequence, so row order
//! is the tiebreak the stream actually has -- and it is what makes the
//! sampler chunk-invariant, since a point can be emitted the moment its last
//! series ticks rather than held back until a strictly greater timestamp
//! arrives (docs/REVIEW-E54-E64.md RT4). Sort the input by time *and* by the
//! order you want within a timestamp.
//!
//! **The staleness caveat** (their §2.1) is worth stating, because the output
//! looks synchronous and is not: a refresh vector is *treated* as observed at
//! `time_refresh`, but each series' value is up to one of its own inter-tick
//! intervals old. `n_obs_<s>` counts a series' ticks between two grid
//! points, of which the grid keeps the last, so a large count is ticks the
//! grid dropped. The series holding the grid up is the one whose tick
//! completes each point: its count is near 1, and its value is the freshest,
//! observed at `time_refresh` itself.
//!
//! # Why a Rust operator
//!
//! The recursion is a sequential scan: whether row `t` closes a grid point
//! depends on every row before it. No window expression writes that, and a
//! per-row Python loop over a tick stream is exactly the cost this library
//! exists to avoid. So the scan is here, per `by` key, `O(m)` a row (or
//! `O(m²)` with `pairs`), and `polars_online.stream.refresh_time` wraps it as a
//! lazy source the way the bank is wrapped.

use std::collections::HashMap;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bank::GroupKey;
use crate::stream::usable;

/// One series' state within a group: its last value, whether it has ticked
/// since the last grid point, and how many times.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SeriesState {
    last: f64,
    seen: bool,
    ticks: u64,
}

/// One group's (or one pair's) refresh state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridState {
    series: Vec<SeriesState>,
    /// Series still to tick before the next grid point; `m` after a point.
    pending: usize,
}

impl GridState {
    fn new(m: usize) -> Self {
        Self {
            series: vec![SeriesState::default(); m],
            pending: m,
        }
    }

    /// One tick. Returns true when it completed the set, i.e. this row is a
    /// grid point.
    fn tick(&mut self, si: usize, value: f64) -> bool {
        let s = &mut self.series[si];
        s.last = value;
        // Saturating: a state file can carry a count at the top of its
        // range (task 160, PA5).
        s.ticks = s.ticks.saturating_add(1);
        if !s.seen {
            s.seen = true;
            self.pending -= 1;
        }
        self.pending == 0
    }

    /// Whether a grid read from a file is one this sampler could have left:
    /// `width` series, and `pending` the count of those still unseen, at
    /// least one -- a point closes the moment the last of them ticks. A
    /// narrower grid panicked at the next tick of a series past its width;
    /// a `pending` below the unseen count wrapped below zero, and one above
    /// it never reached zero, so the grid never completed again (task 160,
    /// PA5).
    fn fits(&self, width: usize) -> bool {
        let unseen = self.series.iter().filter(|s| !s.seen).count();
        self.series.len() == width && self.pending >= 1 && self.pending == unseen
    }

    /// Start the next interval: every series unseen, every count back to
    /// zero. Returns the counts the point that just closed was made of.
    fn close(&mut self, m: usize) -> Vec<u64> {
        let ticks: Vec<u64> = self.series.iter().map(|s| s.ticks).collect();
        for s in self.series.iter_mut() {
            s.seen = false;
            s.ticks = 0;
        }
        self.pending = m;
        ticks
    }
}

/// The sampler: one [`GridState`] per `by` key (or per (key, pair)), fed
/// frames in stream order.
///
/// State lives across `feed` calls, so a stream can arrive in any chunking
/// and give the same grid -- a grid point is a property of the ticks up to
/// it, which is what makes that true.
#[derive(Debug)]
pub struct RefreshTime {
    /// Series names, in the order the output columns take.
    names: Vec<String>,
    /// The pairs to run independently, as index pairs into `names`; empty
    /// for the joint grid over every series.
    pairs: Vec<(usize, usize)>,
    /// Per group key, the joint state or one state per pair.
    states: HashMap<GroupKey, Vec<GridState>>,
    /// The last `time` seen per group, so a backwards clock is refused.
    last_time: HashMap<GroupKey, Instant>,
    /// Whether the input has a `group` column, once a chunk was fed; a
    /// state resumes only under the grouping it was saved with.
    grouped: Option<bool>,
    /// The kind of clock, once a row was fed; the other kind is refused on
    /// every row, not only where a group has a prior.
    kind: Option<ClockKind>,
    /// Rows fed before this chunk, so an error names the input's row, not
    /// the chunk's (task 120).
    rows_fed: usize,
}

/// A clock value as the order check compares it: a number as itself, a
/// temporal value in integer nanoseconds since the epoch, exactly, whatever
/// the column's unit. Read as a double, a `Datetime` in nanoseconds resolves
/// 256 ns at today's dates, and a step back smaller than that was a tie
/// (task 120). In nanoseconds, a state saved from one unit resumes on
/// another (task 105).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Instant {
    Number(f64),
    Nanos(i64),
}

impl Instant {
    /// The value as the column shows it: a temporal one in its own kind.
    fn show(self, dtype: &DataType) -> String {
        const DAY: i64 = 86_400 * 1_000_000_000;
        match (self, dtype) {
            (Self::Number(v), _) => format!("{v}"),
            (Self::Nanos(v), DataType::Datetime(_, tz)) => {
                format!(
                    "{}",
                    AnyValue::Datetime(v, TimeUnit::Nanoseconds, tz.as_ref())
                )
            }
            (Self::Nanos(v), DataType::Date) => match i32::try_from(v.div_euclid(DAY)) {
                Ok(d) => format!("{}", AnyValue::Date(d)),
                Err(_) => format!("{v} ns"),
            },
            (Self::Nanos(v), DataType::Duration(_)) => {
                format!("{}", AnyValue::Duration(v, TimeUnit::Nanoseconds))
            }
            (Self::Nanos(v), _) => format!("{v} ns"),
        }
    }

    fn kind(self) -> ClockKind {
        match self {
            Self::Number(_) => ClockKind::Numeric,
            Self::Nanos(_) => ClockKind::Temporal,
        }
    }
}

/// What [`RefreshTime::save_bytes`] writes: the sampler's state, and what
/// it was built from, which a load must match.
#[derive(Serialize, Deserialize)]
struct RefreshFile {
    magic: String,
    version: u32,
    names: Vec<String>,
    pairs: bool,
    /// Whether the input had a `group` column: an ungrouped state holds the
    /// one key `""`, which a grouped input never checks against.
    grouped: bool,
    /// The kind of clock the state was fed, once a row was: the other kind
    /// cannot be ordered against it.
    kind: Option<ClockKind>,
    /// Sorted by key, so one state is one file, byte for byte.
    states: Vec<(GroupKey, Vec<GridState>)>,
    last_time: Vec<(GroupKey, Instant)>,
}

/// Which kind of clock a sampler has been fed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ClockKind {
    Numeric,
    Temporal,
}

impl ClockKind {
    fn name(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Temporal => "temporal",
        }
    }
}

const REFRESH_MAGIC: &str = "polars-online refresh_time";
const REFRESH_VERSION: u32 = 1;

/// One completed grid point, as the scan collects it: the group it belongs
/// to, which pair (0 for the joint grid), the row of the completing tick,
/// the ticks each series contributed since the previous point, and each
/// series' last value.
type Point = (GroupKey, usize, usize, Vec<u64>, Vec<f64>);

/// A group column as the text its keys are, as the bank keys a group
/// ([`crate::arrow::key_text`]: a zoned Datetime by its instant, task 160,
/// PA4). The text is the group's identity alone: the output carries the
/// input's own values.
fn key_text(col: &Column) -> PolarsResult<Column> {
    crate::arrow::key_text(col.as_materialized_series()).map(Series::into_column)
}

/// The columns [`RefreshTime::feed`] reads.
pub struct RefreshCols<'a> {
    pub series: &'a str,
    pub clock: &'a str,
    pub value: &'a str,
    pub group: Option<&'a str>,
    /// Columns carried through at their value on the completing tick.
    pub keep: &'a [String],
}

impl RefreshTime {
    /// `names` is the series, in output order; `pairs` runs one independent
    /// two-series grid per unordered pair instead of one joint grid.
    ///
    /// # Errors
    ///
    /// Fewer than two names, or a duplicate.
    pub fn new(names: Vec<String>, pairs: bool) -> Result<Self, String> {
        if names.len() < 2 {
            return Err(format!(
                "refresh_time: at least two series are needed (got {}); a grid over one series \
                 is that series' own ticks",
                names.len()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
            return Err(format!("refresh_time: names lists {dup:?} more than once"));
        }
        let pairs = if pairs {
            (0..names.len())
                .flat_map(|i| ((i + 1)..names.len()).map(move |j| (i, j)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            names,
            pairs,
            states: HashMap::new(),
            last_time: HashMap::new(),
            grouped: None,
            kind: None,
            rows_fed: 0,
        })
    }

    /// The sampler's state as bytes: every group's grid, part-way through
    /// an interval or not, and its last clock, with the `names` and `pairs`
    /// it was built from, which a load must match. Versioned msgpack, one
    /// state one file (task 105: a stateful transform resumes).
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let mut states: Vec<_> = self
            .states
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        states.sort_by(|a, b| a.0.cmp(&b.0));
        let mut last_time: Vec<_> = self
            .last_time
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        last_time.sort_by(|a, b| a.0.cmp(&b.0));
        let file = RefreshFile {
            magic: REFRESH_MAGIC.into(),
            version: REFRESH_VERSION,
            names: self.names.clone(),
            pairs: !self.pairs.is_empty(),
            grouped: self.grouped.unwrap_or(false),
            kind: self.kind,
            states,
            last_time,
        };
        rmp_serde::to_vec_named(&file).map_err(|e| e.to_string())
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.save_bytes().map_err(std::io::Error::other)?;
        crate::atomic::write(path, &bytes)
    }

    /// A sampler that goes on from a saved state. `names` and `pairs` must
    /// be the ones it was saved with: they decide the output's columns and
    /// what each grid holds. Rows are counted from the start of the new
    /// input.
    ///
    /// # Errors
    ///
    /// Bytes that are not a refresh_time state, a version this build does
    /// not read, other `names` or `pairs`, or a grid that does not fit them
    /// (a damaged state).
    pub fn load_bytes(
        bytes: &[u8],
        names: Vec<String>,
        pairs: bool,
        grouped: bool,
    ) -> Result<Self, String> {
        let file: RefreshFile =
            rmp_serde::from_slice(bytes).map_err(|e| format!("not a refresh_time state ({e})"))?;
        if file.magic != REFRESH_MAGIC {
            return Err("not a refresh_time state".into());
        }
        if file.version != REFRESH_VERSION {
            return Err(format!(
                "refresh_time state version {} not supported (this build reads {REFRESH_VERSION})",
                file.version
            ));
        }
        if file.names != names || file.pairs != pairs {
            return Err(format!(
                "refresh_time: the state was saved with names = {:?} and pairs = {}, not {:?} \
                 and {}; a state resumes only the grid it holds",
                file.names, file.pairs, names, pairs
            ));
        }
        if file.grouped != grouped {
            return Err(format!(
                "refresh_time: the state was saved {} a group column and is resumed {} one; \
                 a grouped state holds a grid per key, an ungrouped one a single grid",
                if file.grouped { "with" } else { "without" },
                if grouped { "with" } else { "without" }
            ));
        }
        let mut rt = Self::new(names, pairs)?;
        // One joint grid of every series per group, or one two-series grid
        // per pair, each in a state the scan could have left it in.
        let (n_grids, width) = if rt.pairs.is_empty() {
            (1, rt.names.len())
        } else {
            (rt.pairs.len(), 2)
        };
        if let Some((key, _)) = file
            .states
            .iter()
            .find(|(_, grids)| grids.len() != n_grids || !grids.iter().all(|g| g.fits(width)))
        {
            return Err(format!(
                "refresh_time: the state is damaged: group {key}'s grids are not {n_grids} of \
                 {width} series, each waiting on the series it has not seen"
            ));
        }
        rt.states = file.states.into_iter().collect();
        rt.last_time = file.last_time.into_iter().collect();
        rt.grouped = Some(grouped);
        rt.kind = file.kind;
        Ok(rt)
    }

    /// The output schema, which a lazy source has to declare before a row is
    /// read -- the reason `names` is required rather than discovered.
    pub fn schema(&self, cols: &RefreshCols<'_>, input: &Schema) -> Schema {
        let mut out = Schema::default();
        if let Some(by) = cols.group {
            out.insert(
                by.into(),
                input.get(by).cloned().unwrap_or(DataType::String),
            );
        }
        // The completing tick's clock, in the clock column's own dtype
        // (task 160, PA6).
        let time = input.get(cols.clock).cloned().unwrap_or(DataType::Float64);
        if self.pairs.is_empty() {
            out.insert("time_refresh".into(), time);
            for n in &self.names {
                out.insert(format!("{n}_value").into(), DataType::Float64);
            }
            for n in &self.names {
                out.insert(format!("n_obs_{n}").into(), DataType::Int64);
            }
            out.insert("retained_fraction".into(), DataType::Float64);
        } else {
            out.insert("pair".into(), DataType::String);
            out.insert("time_refresh".into(), time);
            out.insert("a_value".into(), DataType::Float64);
            out.insert("b_value".into(), DataType::Float64);
            out.insert("n_obs_a".into(), DataType::Int64);
            out.insert("n_obs_b".into(), DataType::Int64);
            out.insert("retained_fraction".into(), DataType::Float64);
        }
        for k in cols.keep {
            out.insert(
                k.as_str().into(),
                input.get(k.as_str()).cloned().unwrap_or(DataType::Null),
            );
        }
        out
    }

    /// One chunk of the long input, in `time` order within each `by` key.
    /// A `value` that is null, NaN, infinite or past the input bound is a
    /// tick that observed nothing: the series has not moved.
    ///
    /// # Errors
    ///
    /// `ColumnNotFound` for a column the frame has not got; `ComputeError`
    /// for a `value` column that is not numeric (a boolean and a column of
    /// nulls are), naming it, and for a `series` value not in `names` (a
    /// dropped row would hide a misspelling), a null or non-finite `time`,
    /// or a `time` below the previous row's within a group -- each naming
    /// the row.
    pub fn feed(&mut self, df: &DataFrame, cols: &RefreshCols<'_>) -> PolarsResult<DataFrame> {
        self.feed_limited(df, cols, None)
    }

    /// [`Self::feed`], stopping once `limit` grid points are out: the rows
    /// after the one that completed the last of them are not read, so the
    /// state is the state after that row, wherever the chunk ends. That is
    /// what a slice of the output saves (task 105): the state after the
    /// input behind the rows returned, the same whatever the chunk size.
    /// One row can complete several pairs' points at once; it is taken
    /// whole, so the points may pass `limit` by the rest of that row's.
    pub fn feed_limited(
        &mut self,
        df: &DataFrame,
        cols: &RefreshCols<'_>,
        limit: Option<usize>,
    ) -> PolarsResult<DataFrame> {
        let n = df.height();
        let series = df.column(cols.series)?.cast(&DataType::String)?;
        let series = series.str()?;
        let time_col = df.column(cols.clock)?;
        let time_dtype = time_col.dtype().clone();
        let base = self.rows_fed;
        let grouped = cols.group.is_some();
        match self.grouped {
            Some(g) if g != grouped => polars_bail!(ComputeError:
                "refresh_time: fed {} a group column after being fed {} one",
                if grouped { "with" } else { "without" },
                if g { "with" } else { "without" }
            ),
            _ => self.grouped = Some(grouped),
        }
        let (numbers, nanos) = if time_dtype.is_temporal() {
            let ns = crate::arrow::nanos_array(
                time_col.as_materialized_series(),
                base,
                crate::arrow::NanosRole::Clock,
            )?;
            (None, Some(ns))
        } else {
            (Some(time_col.cast(&DataType::Float64)?), None)
        };
        let numbers = numbers.as_ref().map(|c| c.f64()).transpose()?;
        let instant = |row: usize| -> Option<Instant> {
            match (numbers, nanos.as_ref()) {
                (Some(n), _) => n.get(row).filter(|t| t.is_finite()).map(Instant::Number),
                (_, Some(p)) => p.get(row).map(Instant::Nanos),
                _ => None,
            }
        };
        // A number, as a spec's feature is: a boolean or a column of nulls
        // too, and nothing else (review round 4, PC7). Cast without the
        // check, text that is not a number came out all null -- an empty
        // grid, nothing said -- and a `Date` came out as its day count.
        let value = df.column(cols.value)?;
        let dtype = value.dtype();
        if !(dtype.is_numeric() || matches!(dtype, DataType::Boolean | DataType::Null)) {
            polars_bail!(ComputeError:
                "refresh_time: value column {:?} has dtype {}; it must be numeric \
                 (cast it, e.g. pl.col({:?}).cast(pl.Float64))",
                cols.value, dtype, cols.value
            );
        }
        let value = value.cast(&DataType::Float64)?;
        let value = value.f64()?;
        let by = match cols.group {
            Some(b) => Some(key_text(df.column(b)?)?),
            None => None,
        };
        let by = by.as_ref().map(|s| s.str()).transpose()?;
        let index: HashMap<&str, usize> = self
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let m = self.names.len();
        // Rows of the output, as (group, pair index, row index, ticks).
        let mut points: Vec<Point> = Vec::new();
        let mut consumed = n;
        for row in 0..n {
            if limit.is_some_and(|l| points.len() >= l) {
                consumed = row;
                break;
            }
            let at = base + row;
            let Some(name) = series.get(row) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {at} has a null {:?}", cols.series);
            };
            let Some(&si) = index.get(name) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {at} names series {name:?}, which is not in `names` \
                     ({:?}); a row for an unknown series is a misspelling, not something to drop",
                    self.names
                );
            };
            let Some(t) = instant(row) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {at} has a null or non-finite {:?}", cols.clock);
            };
            // The other kind of clock cannot be ordered against a saved one,
            // on any row, a new group's included.
            match self.kind {
                Some(k) if k != t.kind() => polars_bail!(ComputeError:
                    "refresh_time: clock column {:?} is {}, and the state it resumes is \
                     {}; resume a state on the kind of clock that wrote it",
                    cols.clock, t.kind().name(), k.name()
                ),
                _ => self.kind = Some(t.kind()),
            }
            let key = GroupKey(
                by.map_or_else(|| Some(String::new()), |b| b.get(row).map(str::to_string)),
            );
            // Update the last-seen time in place; the key is cloned into the
            // map only the first time its group is seen, not on every row
            // (review 2026-09-18, minor).
            match self.last_time.get_mut(&key) {
                Some(prev) => {
                    if t < *prev {
                        polars_bail!(ComputeError:
                            "refresh_time: row {at} has {:?} = {}, below the previous row's \
                             {} in the same group; the input must be in clock order",
                            cols.clock,
                            t.show(&time_dtype),
                            prev.show(&time_dtype)
                        );
                    }
                    *prev = t;
                }
                None => {
                    self.last_time.insert(key.clone(), t);
                }
            }
            // A null value is not an update: the tick happened, but nothing
            // was observed, so the series has not moved. Nor is a value the
            // bank reads as missing -- NaN, an infinity, a magnitude past
            // the input bound -- by the rule every spec column follows
            // (`usable`); they were values (review round 4, PC7).
            let Some(v) = value.get(row).filter(|v| usable(*v)) else {
                continue;
            };

            let n_states = if self.pairs.is_empty() {
                1
            } else {
                self.pairs.len()
            };
            // Likewise: build and insert only when the group is new, then take
            // it by reference, rather than cloning the key into `entry` every
            // row (review 2026-09-18, minor).
            if !self.states.contains_key(&key) {
                let init: Vec<GridState> = (0..n_states)
                    .map(|_| GridState::new(if self.pairs.is_empty() { m } else { 2 }))
                    .collect();
                self.states.insert(key.clone(), init);
            }
            let states = self.states.get_mut(&key).expect("inserted above");
            if self.pairs.is_empty() {
                if states[0].tick(si, v) {
                    let last: Vec<f64> = states[0].series.iter().map(|s| s.last).collect();
                    let ticks = states[0].close(m);
                    points.push((key, 0, row, ticks, last));
                }
            } else {
                for (pi, &(a, b)) in self.pairs.iter().enumerate() {
                    let slot = if si == a {
                        0
                    } else if si == b {
                        1
                    } else {
                        continue;
                    };
                    if states[pi].tick(slot, v) {
                        let last: Vec<f64> = states[pi].series.iter().map(|s| s.last).collect();
                        let ticks = states[pi].close(2);
                        points.push((key.clone(), pi, row, ticks, last));
                    }
                }
            }
        }
        self.rows_fed += consumed;
        self.frame(df, cols, points)
    }

    /// The grid points as a frame, in the order they were completed.
    fn frame(
        &self,
        df: &DataFrame,
        cols: &RefreshCols<'_>,
        points: Vec<Point>,
    ) -> PolarsResult<DataFrame> {
        let h = points.len();
        // The row of each point's completing tick, which every column taken
        // from the input is read at.
        let idx: Vec<IdxSize> = points
            .iter()
            .map(|(_, _, row, ..)| *row as IdxSize)
            .collect();
        let mut out: Vec<Column> = Vec::new();
        if let Some(b) = cols.group {
            // The keys are held as text, because that is what a group key
            // is here, but the column goes out as the input's own values at
            // the completing ticks, in the dtype they came in as, so the
            // result joins to the input it came from (docs/REVIEW-E54-E64.md
            // RT1). The text cast back made a Datetime(us), a Time or a
            // Struct key null, and failed for a Boolean or a zoned Datetime
            // (task 160, PA4).
            out.push(df.column(b)?.take_slice(&idx)?.with_name(b.into()));
        }
        let pair_of = |pi: usize| {
            let (a, b) = self.pairs[pi];
            format!("{}|{}", self.names[a], self.names[b])
        };
        if !self.pairs.is_empty() {
            out.push(Column::new(
                "pair".into(),
                points
                    .iter()
                    .map(|(_, pi, ..)| pair_of(*pi))
                    .collect::<Vec<_>>(),
            ));
        }
        // The completing tick's clock as the input has it: a nanosecond clock
        // read as a Float64 lost its last digits, and a Datetime came out as
        // a bare number (task 160, PA6).
        out.push(
            df.column(cols.clock)?
                .take_slice(&idx)?
                .with_name("time_refresh".into()),
        );
        let value_names: Vec<String> = if self.pairs.is_empty() {
            self.names.iter().map(|n| format!("{n}_value")).collect()
        } else {
            vec!["a_value".into(), "b_value".into()]
        };
        for (si, name) in value_names.iter().enumerate() {
            out.push(Column::new(
                name.as_str().into(),
                points
                    .iter()
                    .map(|(_, _, _, _, last)| last[si])
                    .collect::<Vec<_>>(),
            ));
        }
        let tick_names: Vec<String> = if self.pairs.is_empty() {
            self.names.iter().map(|n| format!("n_obs_{n}")).collect()
        } else {
            vec!["n_obs_a".into(), "n_obs_b".into()]
        };
        for (si, name) in tick_names.iter().enumerate() {
            out.push(Column::new(
                name.as_str().into(),
                points
                    .iter()
                    .map(|(_, _, _, ticks, _)| ticks[si] as i64)
                    .collect::<Vec<_>>(),
            ));
        }
        // `m / sum(ticks)`: one row of the grid was made from this many
        // ticks, and kept `m` of them.
        out.push(Column::new(
            "retained_fraction".into(),
            points
                .iter()
                .map(|(_, _, _, ticks, _)| {
                    let total: u64 = ticks.iter().fold(0, |a, &t| a.saturating_add(t));
                    if total == 0 {
                        f64::NAN
                    } else {
                        ticks.len() as f64 / total as f64
                    }
                })
                .collect::<Vec<_>>(),
        ));
        for k in cols.keep {
            let taken = df.column(k.as_str())?.take_slice(&idx)?;
            out.push(taken.with_name(k.as_str().into()));
        }
        DataFrame::new(h, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long(rows: &[(&str, f64, f64)]) -> DataFrame {
        df!(
            "series" => rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            "t" => rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            "v" => rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn cols<'a>(keep: &'a [String]) -> RefreshCols<'a> {
        RefreshCols {
            series: "series",
            clock: "t",
            value: "v",
            group: None,
            keep,
        }
    }

    /// BNHLS's own picture: three series with 8, 9 and 10 ticks give seven
    /// refresh points and keep 21 of the 27 ticks. Their tick times are only
    /// a figure in the paper, so the stream here is constructed to those
    /// counts and the reduction is what is checked.
    #[test]
    fn the_three_series_example_gives_seven_points_and_21_of_27() {
        // Seven intervals, each closing on the tick that completes the set;
        // the repeats inside an interval are the ticks the grid drops.
        let intervals: [&[&str]; 7] = [
            &["c", "c", "c", "a", "b"],
            &["b", "b", "a", "c"],
            &["b", "b", "a", "c"],
            &["a", "a", "b", "c"],
            &["c", "c", "a", "b"],
            &["a", "b", "c"],
            &["a", "b", "c"],
        ];
        let mut rows: Vec<(&str, f64, f64)> = Vec::new();
        for iv in intervals {
            for s in iv {
                let t = rows.len() as f64 + 1.0;
                rows.push((s, t, t));
            }
        }
        let counts = ["a", "b", "c"].map(|s| rows.iter().filter(|r| r.0 == s).count());
        assert_eq!(counts, [8, 9, 10], "the paper's tick counts");
        assert_eq!(rows.len(), 27);

        let names = ["a", "b", "c"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, false).unwrap();
        let out = rt.feed(&long(&rows), &cols(&[])).unwrap();
        assert_eq!(out.height(), 7, "N = 7");
        // 21 of the 27 ticks are on the grid: three per point.
        let total_ticks: i64 = ["a", "b", "c"]
            .iter()
            .map(|s| {
                out.column(&format!("n_obs_{s}"))
                    .unwrap()
                    .i64()
                    .unwrap()
                    .sum()
                    .unwrap()
            })
            .sum();
        assert_eq!(total_ticks, 27, "every tick is counted somewhere");
        assert_eq!(out.height() * 3, 21, "and three of each interval survive");
        // The per-point fraction is 3 over that interval's ticks.
        let want = [3.0 / 5.0, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0];
        let got = out.column("retained_fraction").unwrap().f64().unwrap();
        for (i, w) in want.iter().enumerate() {
            assert!((got.get(i).unwrap() - w).abs() < 1e-12, "point {i}");
        }
        // Overall, which is the number the paper quotes: three values kept
        // per point out of every tick fed.
        let overall = (3 * out.height()) as f64 / total_ticks as f64;
        assert!((overall - 21.0 / 27.0).abs() < 1e-12, "{overall}");
    }

    #[test]
    fn a_synchronous_input_comes_back_unchanged() {
        let rows: Vec<(&str, f64, f64)> = (0..10)
            .flat_map(|i| [("a", i as f64, i as f64), ("b", i as f64, -(i as f64))])
            .collect();
        let names = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, false).unwrap();
        let out = rt.feed(&long(&rows), &cols(&[])).unwrap();
        assert_eq!(out.height(), 10);
        for s in ["a", "b"] {
            let ticks = out.column(&format!("n_obs_{s}")).unwrap().i64().unwrap();
            assert!(ticks.into_no_null_iter().all(|t| t == 1));
        }
        let r = out.column("retained_fraction").unwrap().f64().unwrap();
        assert!(r.into_no_null_iter().all(|v| v == 1.0));
    }

    #[test]
    fn the_grid_is_the_same_however_the_chunks_fall() {
        let rows: Vec<(&str, f64, f64)> = (0..60)
            .map(|i| {
                let s = ["a", "b", "c"][(i * 7 % 3) as usize];
                (s, i as f64, (i * i % 17) as f64)
            })
            .collect();
        let names = ["a", "b", "c"].map(str::to_string).to_vec();
        let whole = RefreshTime::new(names.clone(), false)
            .unwrap()
            .feed(&long(&rows), &cols(&[]))
            .unwrap();
        for size in [1usize, 2, 7, 59] {
            let mut rt = RefreshTime::new(names.clone(), false).unwrap();
            let df = long(&rows);
            let parts: Vec<DataFrame> = (0..df.height())
                .step_by(size)
                .map(|i| rt.feed(&df.slice(i as i64, size), &cols(&[])).unwrap())
                .collect();
            let joined = accumulate(parts);
            assert!(whole.equals(&joined), "chunk size {size}");
        }
    }

    fn accumulate(parts: Vec<DataFrame>) -> DataFrame {
        let mut it = parts.into_iter();
        let mut first = it.next().expect("at least one");
        for p in it {
            first.vstack_mut(&p).unwrap();
        }
        first.align_chunks_par();
        first
    }

    #[test]
    fn pairs_run_independent_grids() {
        // a and b tick together; c is slow. The (a, b) pair keeps every
        // tick, while every pair with c is held to c's pace.
        let mut rows: Vec<(&str, f64, f64)> = Vec::new();
        for i in 0..12 {
            rows.push(("a", i as f64, i as f64));
            rows.push(("b", i as f64, i as f64));
            if i % 4 == 0 {
                rows.push(("c", i as f64, i as f64));
            }
        }
        let names = ["a", "b", "c"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, true).unwrap();
        let out = rt.feed(&long(&rows), &cols(&[])).unwrap();
        let pairs = out.column("pair").unwrap().str().unwrap();
        let counts = |p: &str| pairs.iter().flatten().filter(|v| *v == p).count();
        assert_eq!(counts("a|b"), 12);
        assert_eq!(counts("a|c"), 3);
        assert_eq!(counts("b|c"), 3);
    }

    #[test]
    fn a_null_value_is_not_an_update() {
        let df = df!(
            "series" => ["a", "b", "a", "b"],
            "t" => [1.0, 2.0, 3.0, 4.0],
            "v" => [Some(1.0), None, Some(3.0), Some(4.0)],
        )
        .unwrap();
        let names = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, false).unwrap();
        let out = rt.feed(&df, &cols(&[])).unwrap();
        // b's first tick carried nothing, so the point waits for row 3.
        assert_eq!(out.height(), 1);
        assert_eq!(
            out.column("time_refresh").unwrap().f64().unwrap().get(0),
            Some(4.0)
        );
        assert_eq!(
            out.column("a_value").unwrap().f64().unwrap().get(0),
            Some(3.0)
        );
    }

    /// A value the bank reads as missing is a tick that observed nothing, as
    /// a null is (review round 4, PC7): NaN, an infinity and a magnitude past
    /// the input bound were folded in as values, so a point completed on
    /// such a tick reported it -- `b_value` NaN at `time_refresh` 2.
    #[test]
    fn a_value_that_is_not_usable_is_not_an_update() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e101, -1e101] {
            let df = df!(
                "series" => ["a", "b", "a", "b"],
                "t" => [1.0, 2.0, 3.0, 4.0],
                "v" => [1.0, bad, 3.0, 4.0],
            )
            .unwrap();
            let names = ["a", "b"].map(str::to_string).to_vec();
            let mut rt = RefreshTime::new(names, false).unwrap();
            let out = rt.feed(&df, &cols(&[])).unwrap();
            // b's first tick carried nothing, so the point waits for row 3.
            assert_eq!(out.height(), 1, "{bad}");
            let at = |c: &str| out.column(c).unwrap().f64().unwrap().get(0);
            assert_eq!(at("time_refresh"), Some(4.0), "{bad}");
            assert_eq!(at("a_value"), Some(3.0), "{bad}");
            assert_eq!(at("b_value"), Some(4.0), "{bad}");
        }
        // At the bound itself a value is a value.
        let df = df!(
            "series" => ["a", "b"],
            "t" => [1.0, 2.0],
            "v" => [1.0, -1e100],
        )
        .unwrap();
        let names = ["a", "b"].map(str::to_string).to_vec();
        let out = RefreshTime::new(names, false)
            .unwrap()
            .feed(&df, &cols(&[]))
            .unwrap();
        assert_eq!(
            out.column("b_value").unwrap().f64().unwrap().get(0),
            Some(-1e100)
        );
    }

    /// A value column that is not numeric is refused by name, as a spec's
    /// feature is (review round 4, PC7). It was read through a non-strict
    /// cast: text that is not a number came out all null, so the grid was
    /// empty and nothing said why, and a `Date` was read as its day count.
    /// The frame of no rows a plan is built from is refused too, so the plan
    /// says so before it runs. Integers, booleans and a column of nulls are
    /// numbers, as they are to a spec.
    #[test]
    fn a_value_column_that_is_not_numeric_is_refused_by_name() {
        let names = ["a", "b"].map(str::to_string).to_vec();
        let with_value = |v: Column| {
            let mut df = df!("series" => ["a", "b"], "t" => [1.0, 2.0]).unwrap();
            df.with_column(v.with_name("v".into())).unwrap();
            df
        };
        let date = Column::new("v".into(), [19_000i32, 19_001])
            .cast(&DataType::Date)
            .unwrap();
        for (what, v) in [
            ("text", Column::new("v".into(), ["abc", "def"])),
            ("numbers as text", Column::new("v".into(), ["1.5", "2.5"])),
            ("a date", date),
        ] {
            for df in [with_value(v.clone()), with_value(v.clone()).clear()] {
                let e = RefreshTime::new(names.clone(), false)
                    .unwrap()
                    .feed(&df, &cols(&[]))
                    .unwrap_err()
                    .to_string();
                assert!(
                    e.contains("value column \"v\"") && e.contains("must be numeric"),
                    "{what}, {} rows: {e}",
                    df.height()
                );
            }
        }
        for (what, v) in [
            ("integers", Column::new("v".into(), [1i64, 2])),
            ("booleans", Column::new("v".into(), [true, false])),
            ("float32", Column::new("v".into(), [1.0f32, 2.0])),
            ("nulls", Column::full_null("v".into(), 2, &DataType::Null)),
        ] {
            RefreshTime::new(names.clone(), false)
                .unwrap()
                .feed(&with_value(v), &cols(&[]))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }

    #[test]
    fn keep_columns_take_their_value_on_the_completing_tick() {
        let mut df = long(&[
            ("a", 1.0, 1.0),
            ("b", 2.0, 2.0),
            ("a", 3.0, 3.0),
            ("b", 4.0, 4.0),
        ]);
        df.with_column(Column::new("tag".into(), ["p", "q", "r", "s"]))
            .unwrap();
        let names = ["a", "b"].map(str::to_string).to_vec();
        let keep = vec!["tag".to_string()];
        let mut rt = RefreshTime::new(names, false).unwrap();
        let out = rt.feed(&df, &cols(&keep)).unwrap();
        assert_eq!(
            out.column("tag")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>(),
            ["q", "s"]
        );
    }

    #[test]
    fn a_bad_input_is_refused_naming_the_row() {
        let names = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        let e = rt
            .feed(&long(&[("a", 1.0, 1.0), ("z", 2.0, 2.0)]), &cols(&[]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("row 1") && e.contains("\"z\""), "{e}");

        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        let e = rt
            .feed(&long(&[("a", 5.0, 1.0), ("b", 2.0, 2.0)]), &cols(&[]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("row 1") && e.contains("clock order"), "{e}");

        assert!(
            RefreshTime::new(vec!["a".into()], false)
                .unwrap_err()
                .contains("at least two series")
        );
        assert!(
            RefreshTime::new(vec!["a".into(), "a".into()], false)
                .unwrap_err()
                .contains("more than once")
        );
    }

    #[test]
    fn groups_keep_their_own_grids() {
        let df = df!(
            "series" => ["a", "a", "b", "b"],
            "t" => [1.0, 1.0, 2.0, 2.0],
            "v" => [1.0, 2.0, 3.0, 4.0],
            "g" => ["x", "y", "x", "y"],
        )
        .unwrap();
        let names = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, false).unwrap();
        let keep: Vec<String> = Vec::new();
        let out = rt
            .feed(
                &df,
                &RefreshCols {
                    series: "series",
                    clock: "t",
                    value: "v",
                    group: Some("g"),
                    keep: &keep,
                },
            )
            .unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(
            out.column("g")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>(),
            ["x", "y"]
        );
        assert_eq!(
            out.column("a_value")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [1.0, 2.0]
        );
    }

    /// Task 105, rule 5: a stateful transform resumes. Saved at every split
    /// of a stream -- grids part-way through an interval included -- and
    /// loaded, the sampler gives the grid one uninterrupted run gives, and
    /// the bytes are the same whichever split they were taken at.
    #[test]
    fn a_saved_state_resumes_where_it_stopped() {
        let names: Vec<String> = ["a", "b", "c"].map(str::to_string).to_vec();
        let rows: Vec<(&str, f64, f64)> = (0..40)
            .map(|i| {
                let s = ["a", "b", "a", "c", "b", "a", "c"][i % 7];
                (s, i as f64, (i * i % 11) as f64)
            })
            .collect();
        let whole = RefreshTime::new(names.clone(), false)
            .unwrap()
            .feed(&long(&rows), &cols(&[]))
            .unwrap();
        for pairs in [false, true] {
            let one = RefreshTime::new(names.clone(), pairs)
                .unwrap()
                .feed(&long(&rows), &cols(&[]))
                .unwrap();
            if !pairs {
                assert!(one.equals(&whole));
            }
            for split in [1, 5, 17, 39] {
                let mut first = RefreshTime::new(names.clone(), pairs).unwrap();
                let a = first.feed(&long(&rows[..split]), &cols(&[])).unwrap();
                let bytes = first.save_bytes().unwrap();
                let mut resumed =
                    RefreshTime::load_bytes(&bytes, names.clone(), pairs, false).unwrap();
                assert_eq!(resumed.save_bytes().unwrap(), bytes, "split {split}");
                let b = resumed.feed(&long(&rows[split..]), &cols(&[])).unwrap();
                let mut both = a.clone();
                both.vstack_mut(&b).unwrap();
                assert!(both.equals(&one), "pairs {pairs}, split {split}");
            }
        }
    }

    #[test]
    fn a_state_loads_only_into_the_grid_it_holds() {
        let names: Vec<String> = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        rt.feed(&long(&[("a", 1.0, 1.0)]), &cols(&[])).unwrap();
        let bytes = rt.save_bytes().unwrap();
        let other = vec!["a".to_string(), "c".to_string()];
        assert!(
            RefreshTime::load_bytes(&bytes, other, false, false)
                .unwrap_err()
                .contains("names")
        );
        assert!(
            RefreshTime::load_bytes(&bytes, names.clone(), true, false)
                .unwrap_err()
                .contains("pairs")
        );
        assert!(
            RefreshTime::load_bytes(b"nope", names.clone(), false, false)
                .unwrap_err()
                .contains("not a refresh_time state")
        );
        // Resumed on the other kind of clock, or before its last clock.
        let mut resumed = RefreshTime::load_bytes(&bytes, names.clone(), false, false).unwrap();
        let temporal = df!(
            "series" => ["b"],
            "t" => Series::new("t".into(), [5i64]).cast(&DataType::Datetime(TimeUnit::Nanoseconds, None)).unwrap(),
            "v" => [1.0],
        )
        .unwrap();
        let e = resumed.feed(&temporal, &cols(&[])).unwrap_err().to_string();
        assert!(e.contains("temporal") && e.contains("numeric"), "{e}");
        let mut resumed = RefreshTime::load_bytes(&bytes, names, false, false).unwrap();
        let e = resumed
            .feed(&long(&[("b", 0.5, 1.0)]), &cols(&[]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("row 0") && e.contains("clock order"), "{e}");
    }

    /// Kept in nanoseconds, a temporal clock resumes on another unit: saved
    /// from microseconds, one nanosecond after the last instant is after it.
    #[test]
    fn a_temporal_state_resumes_on_another_unit() {
        let names: Vec<String> = ["a", "b"].map(str::to_string).to_vec();
        let frame = |ns: &[i64], unit: TimeUnit, series: &[&str]| {
            let per = match unit {
                TimeUnit::Nanoseconds => 1,
                TimeUnit::Microseconds => 1_000,
                TimeUnit::Milliseconds => 1_000_000,
            };
            let t: Vec<i64> = ns.iter().map(|v| v / per).collect();
            df!(
                "series" => series,
                "t" => Series::new("t".into(), t).cast(&DataType::Datetime(unit, None)).unwrap(),
                "v" => vec![1.0; ns.len()],
            )
            .unwrap()
        };
        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        rt.feed(
            &frame(&[1_000, 2_000], TimeUnit::Microseconds, &["a", "b"]),
            &cols(&[]),
        )
        .unwrap();
        let mut resumed =
            RefreshTime::load_bytes(&rt.save_bytes().unwrap(), names.clone(), false, false)
                .unwrap();
        resumed
            .feed(
                &frame(&[2_001, 2_002], TimeUnit::Nanoseconds, &["b", "a"]),
                &cols(&[]),
            )
            .unwrap();
        let mut resumed =
            RefreshTime::load_bytes(&rt.save_bytes().unwrap(), names, false, false).unwrap();
        let e = resumed
            .feed(&frame(&[1_999], TimeUnit::Nanoseconds, &["b"]), &cols(&[]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("clock order"), "{e}");
    }

    /// Review 2026-09-28: the file records the grouping and the clock kind,
    /// and both are held on load and on every row, a new group's included.
    #[test]
    fn a_state_holds_its_grouping_and_its_clock_kind() {
        let names: Vec<String> = ["a", "b"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        rt.feed(&long(&[("a", 1.0, 1.0)]), &cols(&[])).unwrap();
        let bytes = rt.save_bytes().unwrap();
        let e = RefreshTime::load_bytes(&bytes, names.clone(), false, true).unwrap_err();
        assert!(
            e.contains("without a group column") && e.contains("resumed with"),
            "{e}"
        );
        // Fed a new group only, the other kind of clock is still refused.
        let grouped = df!("series" => ["a"], "t" => [1.0], "v" => [1.0], "g" => ["x"]).unwrap();
        let gcols = RefreshCols {
            series: "series",
            clock: "t",
            value: "v",
            group: Some("g"),
            keep: &[],
        };
        let mut rt = RefreshTime::new(names.clone(), false).unwrap();
        rt.feed(&grouped, &gcols).unwrap();
        let mut resumed =
            RefreshTime::load_bytes(&rt.save_bytes().unwrap(), names, false, true).unwrap();
        let other = df!(
            "series" => ["a"],
            "t" => Series::new("t".into(), [5i64]).cast(&DataType::Datetime(TimeUnit::Nanoseconds, None)).unwrap(),
            "v" => [1.0],
            "g" => ["y"],
        )
        .unwrap();
        let e = resumed.feed(&other, &gcols).unwrap_err().to_string();
        assert!(e.contains("temporal") && e.contains("numeric"), "{e}");
    }

    /// A limited feed stops after the row that completed its last point: the
    /// state is the state after exactly those rows, and the rows after it are
    /// neither read nor counted, a bad one included.
    #[test]
    fn a_limited_feed_stops_after_the_row_of_its_last_point() {
        let names: Vec<String> = ["a", "b"].map(str::to_string).to_vec();
        // Points complete at rows 1, 3 and 5; row 6 names an unknown series.
        let rows = [
            ("a", 1.0, 1.0),
            ("b", 2.0, 2.0),
            ("a", 3.0, 3.0),
            ("b", 4.0, 4.0),
            ("b", 5.0, 5.0),
            ("a", 6.0, 6.0),
            ("z", 7.0, 7.0),
        ];
        for (limit, through) in [(0, 0), (1, 2), (2, 4)] {
            let mut limited = RefreshTime::new(names.clone(), false).unwrap();
            let out = limited
                .feed_limited(&long(&rows), &cols(&[]), Some(limit))
                .unwrap();
            assert_eq!(out.height(), limit);
            let mut exact = RefreshTime::new(names.clone(), false).unwrap();
            exact.feed(&long(&rows[..through]), &cols(&[])).unwrap();
            assert_eq!(
                limited.save_bytes().unwrap(),
                exact.save_bytes().unwrap(),
                "{limit}"
            );
            assert_eq!(limited.rows_fed, through);
        }
        let mut unlimited = RefreshTime::new(names, false).unwrap();
        assert!(unlimited.feed(&long(&rows), &cols(&[])).is_err());
    }

    /// Task 160, PA4: a zoned Datetime group was refused, its cast to text
    /// failing -- this build formats no zone. It is keyed by its instant and
    /// comes back as the input had it, zone and all.
    #[test]
    fn a_zoned_datetime_group_comes_back_as_it_went_in() {
        let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
        let g = Int64Chunked::new("g".into(), [0i64, 0, 0, 3_600_000, 3_600_000, 3_600_000])
            .into_datetime(TimeUnit::Milliseconds, tz)
            .into_series();
        let dtype = g.dtype().clone();
        assert!(
            matches!(dtype, DataType::Datetime(_, Some(_))),
            "a zoned column: {dtype}"
        );
        let df = df!(
            "series" => ["a", "b", "c", "a", "b", "c"],
            "t" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "v" => [1.0; 6],
            "g" => g.clone(),
        )
        .unwrap();
        let names: Vec<String> = ["a", "b", "c"].map(str::to_string).to_vec();
        let mut rt = RefreshTime::new(names, false).unwrap();
        let cols = RefreshCols {
            series: "series",
            clock: "t",
            value: "v",
            group: Some("g"),
            keep: &[],
        };
        let out = rt.feed(&df, &cols).unwrap();
        let got = out.column("g").unwrap();
        assert_eq!(got.dtype(), &dtype);
        let want = g.take_slice(&[2, 5]).unwrap();
        assert!(got.as_materialized_series().equals(&want), "{got:?}");
    }

    /// Task 160, PA5: a state whose grids do not fit its series is refused at
    /// load. A grid narrower than the series panicked at the next tick of a
    /// series past its width; a `pending` count other than the series still
    /// unseen wrapped below zero, or never reached it, so the grid never
    /// completed a point again.
    #[test]
    fn a_state_whose_grids_do_not_fit_its_series_is_refused() {
        type Case<'a> = (&'a str, &'a dyn Fn(&mut RefreshFile));
        let names: Vec<String> = ["a", "b", "c"].map(str::to_string).to_vec();
        let reencoded = |pairs: bool, edit: &dyn Fn(&mut RefreshFile)| {
            let mut rt = RefreshTime::new(names.clone(), pairs).unwrap();
            rt.feed(&long(&[("a", 1.0, 1.0), ("b", 2.0, 2.0)]), &cols(&[]))
                .unwrap();
            let mut file: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
            edit(&mut file);
            rmp_serde::to_vec_named(&file).unwrap()
        };
        let joint: Vec<Case> = vec![
            ("a grid one series wide", &|f| {
                f.states[0].1[0].series.truncate(1)
            }),
            ("a grid four series wide", &|f| {
                f.states[0].1[0].series.push(SeriesState::default());
            }),
            ("pending 0, a series unseen", &|f| {
                f.states[0].1[0].pending = 0
            }),
            ("pending 7 of three series", &|f| {
                f.states[0].1[0].pending = 7
            }),
            ("a second joint grid", &|f| {
                let g = f.states[0].1[0].clone();
                f.states[0].1.push(g);
            }),
            ("no grid at all", &|f| f.states[0].1.clear()),
        ];
        for (what, edit) in joint {
            let bytes = reencoded(false, edit);
            match RefreshTime::load_bytes(&bytes, names.clone(), false, false) {
                Err(e) => assert!(e.contains("damaged"), "{what}: {e}"),
                Ok(_) => panic!("{what}: loaded"),
            }
        }
        // Under pairs, one two-series grid per pair.
        let paired: Vec<Case> = vec![
            ("a pair's grid three wide", &|f| {
                f.states[0].1[1].series.push(SeriesState::default());
            }),
            ("a pair short", &|f| {
                f.states[0].1.pop();
            }),
        ];
        for (what, edit) in paired {
            let bytes = reencoded(true, edit);
            match RefreshTime::load_bytes(&bytes, names.clone(), true, false) {
                Err(e) => assert!(e.contains("damaged"), "{what}: {e}"),
                Ok(_) => panic!("{what}: loaded"),
            }
        }
        // And the untouched states load.
        for pairs in [false, true] {
            let bytes = reencoded(pairs, &|_| {});
            assert!(RefreshTime::load_bytes(&bytes, names.clone(), pairs, false).is_ok());
        }
    }
}
