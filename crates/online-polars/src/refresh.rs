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
//! **The staleness caveat** (their §2.1) is worth stating, because the output
//! looks synchronous and is not: a refresh vector is *treated* as observed at
//! `time_refresh`, but each series' value is up to one of its own inter-tick
//! intervals old. `n_ticks_<s>` is that staleness made visible -- a series
//! with a large count between two grid points is the one holding the grid up,
//! and the one whose value is freshest; a series with a count of 1 has not
//! moved since it last did.
//!
//! # Why a Rust operator
//!
//! The recursion is a sequential scan: whether row `t` closes a grid point
//! depends on every row before it. No window expression writes that, and a
//! per-row Python loop over a tick stream is exactly the cost this library
//! exists to avoid. So the scan is here, per `by` key, `O(m)` a row (or
//! `O(m²)` with `pairs`), and `polars_online.prep.refresh_time` wraps it as a
//! lazy source the way the bank is wrapped.

use std::collections::HashMap;

use polars::prelude::*;

use crate::bank::GroupKey;

/// One series' state within a group: its last value, whether it has ticked
/// since the last grid point, and how many times.
#[derive(Debug, Clone, Default)]
struct SeriesState {
    last: f64,
    seen: bool,
    ticks: u64,
}

/// One group's (or one pair's) refresh state.
#[derive(Debug, Clone)]
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
        s.ticks += 1;
        if !s.seen {
            s.seen = true;
            self.pending -= 1;
        }
        self.pending == 0
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
    last_time: HashMap<GroupKey, f64>,
}

/// One completed grid point, as the scan collects it: the group it belongs
/// to, which pair (0 for the joint grid), the row of the completing tick,
/// the ticks each series contributed since the previous point, and each
/// series' last value.
type Point = (GroupKey, usize, usize, Vec<u64>, Vec<f64>);

/// The columns [`RefreshTime::feed`] reads.
pub struct RefreshCols<'a> {
    pub series: &'a str,
    pub time: &'a str,
    pub value: &'a str,
    pub by: Option<&'a str>,
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
        })
    }

    /// The output schema, which a lazy source has to declare before a row is
    /// read -- the reason `names` is required rather than discovered.
    pub fn schema(&self, cols: &RefreshCols<'_>, input: &Schema) -> Schema {
        let mut out = Schema::default();
        if let Some(by) = cols.by {
            out.insert(
                by.into(),
                input.get(by).cloned().unwrap_or(DataType::String),
            );
        }
        if self.pairs.is_empty() {
            out.insert("time_refresh".into(), DataType::Float64);
            for n in &self.names {
                out.insert(format!("{n}_value").into(), DataType::Float64);
            }
            for n in &self.names {
                out.insert(format!("n_ticks_{n}").into(), DataType::Int64);
            }
            out.insert("retained_fraction".into(), DataType::Float64);
        } else {
            out.insert("pair".into(), DataType::String);
            out.insert("time_refresh".into(), DataType::Float64);
            out.insert("a_value".into(), DataType::Float64);
            out.insert("b_value".into(), DataType::Float64);
            out.insert("n_ticks_a".into(), DataType::Int64);
            out.insert("n_ticks_b".into(), DataType::Int64);
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
    ///
    /// # Errors
    ///
    /// `ColumnNotFound` for a column the frame has not got; `ComputeError`
    /// for a `series` value not in `names` (a dropped row would hide a
    /// misspelling), a null or non-finite `time`, or a `time` below the
    /// previous row's within a group -- each naming the row.
    pub fn feed(&mut self, df: &DataFrame, cols: &RefreshCols<'_>) -> PolarsResult<DataFrame> {
        let n = df.height();
        let series = df.column(cols.series)?.cast(&DataType::String)?;
        let series = series.str()?;
        let time = df.column(cols.time)?.cast(&DataType::Float64)?;
        let time = time.f64()?;
        let value = df.column(cols.value)?.cast(&DataType::Float64)?;
        let value = value.f64()?;
        let by = match cols.by {
            Some(b) => Some(df.column(b)?.cast(&DataType::String)?),
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
        for row in 0..n {
            let Some(name) = series.get(row) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {row} has a null {:?}", cols.series);
            };
            let Some(&si) = index.get(name) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {row} names series {name:?}, which is not in `names` \
                     ({:?}); a row for an unknown series is a misspelling, not something to drop",
                    self.names
                );
            };
            let Some(t) = time.get(row).filter(|t| t.is_finite()) else {
                polars_bail!(ComputeError:
                    "refresh_time: row {row} has a null or non-finite {:?}", cols.time);
            };
            let key = GroupKey(
                by.map_or_else(|| Some(String::new()), |b| b.get(row).map(str::to_string)),
            );
            match self.last_time.get(&key) {
                Some(&prev) if t < prev => {
                    polars_bail!(ComputeError:
                        "refresh_time: row {row} has {:?} = {t}, below the previous row's {prev} \
                         in the same group; the input must be in time order",
                        cols.time
                    );
                }
                _ => {
                    self.last_time.insert(key.clone(), t);
                }
            }
            // A null value is not an update: the tick happened, but nothing
            // was observed, so the series has not moved.
            let Some(v) = value.get(row) else { continue };

            let n_states = if self.pairs.is_empty() {
                1
            } else {
                self.pairs.len()
            };
            let states = self.states.entry(key.clone()).or_insert_with(|| {
                (0..n_states)
                    .map(|_| GridState::new(if self.pairs.is_empty() { m } else { 2 }))
                    .collect()
            });
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
        let time = df.column(cols.time)?.cast(&DataType::Float64)?;
        let time = time.f64()?;
        let mut out: Vec<Column> = Vec::new();
        if let Some(b) = cols.by {
            out.push(Column::new(
                b.into(),
                points
                    .iter()
                    .map(|(k, ..)| k.as_str().map(str::to_string))
                    .collect::<Vec<_>>(),
            ));
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
        out.push(Column::new(
            "time_refresh".into(),
            points
                .iter()
                .map(|(_, _, row, ..)| time.get(*row))
                .collect::<Vec<_>>(),
        ));
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
            self.names.iter().map(|n| format!("n_ticks_{n}")).collect()
        } else {
            vec!["n_ticks_a".into(), "n_ticks_b".into()]
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
                    let total: u64 = ticks.iter().sum();
                    if total == 0 {
                        f64::NAN
                    } else {
                        ticks.len() as f64 / total as f64
                    }
                })
                .collect::<Vec<_>>(),
        ));
        for k in cols.keep {
            let col = df.column(k.as_str())?;
            let idx: Vec<IdxSize> = points
                .iter()
                .map(|(_, _, row, ..)| *row as IdxSize)
                .collect();
            let taken = col.take_slice(&idx)?;
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
            time: "t",
            value: "v",
            by: None,
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
                out.column(&format!("n_ticks_{s}"))
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
            let ticks = out.column(&format!("n_ticks_{s}")).unwrap().i64().unwrap();
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
        assert!(e.contains("row 1") && e.contains("time order"), "{e}");

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
                    time: "t",
                    value: "v",
                    by: Some("g"),
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
}
