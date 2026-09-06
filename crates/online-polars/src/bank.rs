//! The chunk-fed model bank (docs/PLAN.md §5): column extraction, per-group
//! state, fan-out over (spec x group) on the bank's pool (pool.rs), versioned
//! msgpack save/load.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use online_core::ClockCfg;
use polars::prelude::*;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_utils::aliases::PlHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spec::{ModelKind, Spec};
use crate::stream::{AnyModel, ChunkOut, Stream, StreamState, combo_labels};
use crate::summary::{DataSummary, Role, SummaryRow, describe_frame, summary_frame};

/// One stream's group key. A null group value is its own key, distinct from any
/// string a user might have in the column — notably the literal `"<null>"`,
/// which the earlier string sentinel collided with.
///
/// Serialized transparently as its inner `Option<String>` (msgpack `nil` or a
/// string), so bank state files written before this type existed (plain string
/// keys) still load, as `Some(..)`. The one thing such a file cannot express is
/// the difference between a null group and a group literally named `"<null>"`;
/// it is read as the literal, which is the likelier intent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupKey(pub Option<String>);

impl GroupKey {
    /// The single key used when a spec has no `group` column.
    pub fn ungrouped() -> Self {
        GroupKey(Some(String::new()))
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl std::fmt::Display for GroupKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(s) => write!(f, "{s}"),
            None => write!(f, "<null group>"),
        }
    }
}

/// Bank state-file layout version, independent of `online_core::SCHEMA_VERSION`
/// (which versions the *model* state). Version 2 made group keys nullable;
/// version 1 files still load (see [`GroupKey`]).
const BANK_FORMAT_VERSION: u32 = 2;

fn default_format_version() -> u32 {
    1
}

/// Stable, platform-independent 64-bit hash for session-change detection.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The hash a null session value carries. Kept as the hash of the historical
/// sentinel string so that saved states resume with no spurious session change
/// -- `prev_session` stores these hashes, and a resumed stream compares its
/// next row against the stored value.
fn null_session_hash() -> u64 {
    fnv1a(b"\0<null>")
}

/// Session values are compared by 64-bit hash (`ClockState.prev_session`). A
/// null is its own session, and must be distinct from *every* string -- the
/// T-E2 bug, one layer down: hashing null as a sentinel string made a session
/// literally named `"\0<null>"` indistinguishable from null, silently sharing
/// one session with it. Any string that lands on the null hash (the sentinel
/// itself, or a 2^-64 accident) is nudged to a neighbouring value instead.
fn session_hash(v: Option<&str>) -> u64 {
    match v {
        None => null_session_hash(),
        Some(s) => {
            let h = fnv1a(s.as_bytes());
            if h == null_session_hash() { h ^ 1 } else { h }
        }
    }
}

/// Columns extracted once per (spec, chunk).
///
/// Plain `f64` with **NaN for null**, not `Option<f64>` (docs/PERFORMANCE.md
/// P3): half the bytes, no per-value branch, and a memcpy instead of an
/// element-wise walk for a null-free column. Sound because every consumer
/// already collapses the two — a feature or weight is accepted only when it
/// `is_finite()`, so null and NaN both skip the row, and a target is taken
/// only when finite, so null and NaN are both "no target". The clock is the
/// one column where null is an *error*, and that is checked at extraction.
struct SpecColumns {
    features: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
    clock: Option<Vec<f64>>,
    session: Option<Vec<u64>>,
    weight: Option<Vec<f64>>,
    /// The session column's *values*, in the frame's own row order, for a
    /// spec that closes a group on a session change (E54): the closed row
    /// reports the span's session, and `session` above is a hash. Read at
    /// one row per closed span, so it is the series rather than a gathered
    /// `Vec<Option<String>>` -- which would allocate a string per row of the
    /// chunk to use one of them.
    session_str: Option<Series>,
}

/// A column lookup that says which spec asked and what it asked for. Polars'
/// own `not found: "x"` names neither, and in a bank of ten specs that is the
/// difference between a fix and a search.
fn column<'a>(df: &'a DataFrame, spec: &Spec, role: &str, name: &str) -> PolarsResult<&'a Column> {
    df.column(name).map_err(|_| {
        let have: Vec<&str> = df.get_column_names().iter().map(|n| n.as_str()).collect();
        polars_err!(ColumnNotFound:
            "spec {:?}: {} column {:?} not found; the frame has columns {:?}",
            spec.name, role, name, have
        )
    })
}

/// Values as `f64`, null as NaN, in the order `layout` gives (see
/// [`Layout`]). Zero-copy-ish for the common case: a null-free Float64
/// column is a `memcpy` per arrow chunk when the layout is the identity and
/// one gather otherwise. (A batch from `collect_batches` that spans two
/// parquet row groups arrives as two chunks, so `cont_slice` alone is not
/// the common case; docs/PERFORMANCE.md P11.)
///
/// Only numeric, Boolean and Null columns are accepted. Anything else would be
/// cast non-strictly, and a String column of numbers-as-text (or of anything)
/// becomes all-null: every prediction null and no error to say why.
fn f64_column(
    df: &DataFrame,
    spec: &Spec,
    role: &str,
    name: &str,
    layout: Layout<'_>,
) -> PolarsResult<Vec<f64>> {
    let col = column(df, spec, role, name)?;
    let dtype = col.dtype();
    if !(dtype.is_numeric() || matches!(dtype, DataType::Boolean | DataType::Null)) {
        polars_bail!(ComputeError:
            "spec {:?}: {} column {:?} has dtype {}; it must be numeric \
             (cast it, e.g. pl.col({:?}).cast(pl.Float64))",
            spec.name, role, name, dtype, name
        );
    }
    let s = col.as_materialized_series().cast(&DataType::Float64)?;
    let ca = s.f64()?;
    if let (Some(perm), Ok(slice)) = (layout, ca.cont_slice()) {
        return Ok(perm.iter().map(|&i| slice[i]).collect());
    }
    let mut v: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        match arr.validity() {
            Some(valid) if valid.unset_bits() > 0 => {
                v.extend(arr.iter().map(|x| x.copied().unwrap_or(f64::NAN)));
            }
            _ => v.extend_from_slice(arr.values().as_slice()),
        }
    }
    Ok(gathered(v, layout))
}

/// The order a spec's columns are extracted in: `None` is the frame's own
/// order; `Some(perm)` puts row `perm[j]` at position `j`, which the bank uses
/// to lay a chunk out group after group (docs/PERFORMANCE.md P9). Only ever
/// a permutation of `0..height`, so every row appears exactly once.
type Layout<'a> = Option<&'a [usize]>;

/// `v` in layout order.
fn gathered<T: Copy>(v: Vec<T>, layout: Layout<'_>) -> Vec<T> {
    match layout {
        None => v,
        Some(perm) => perm.iter().map(|&i| v[i]).collect(),
    }
}

/// Where a position in layout order came from, for naming a row in an error.
fn source_row(layout: Layout<'_>, j: usize) -> usize {
    layout.map_or(j, |perm| perm[j])
}

/// The layout that puts each spec's groups one after another, in first-seen
/// order, or `None` when the chunk is already in that order (ungrouped, one
/// group, or groups that arrive as blocks). A stream then reads its rows as
/// one contiguous run instead of gathering every column at a stride, twice
/// per row; with many groups interleaved row by row that stride is a cache
/// line per value, and cost 14% of `process` on one thread and 38% on
/// fourteen at k=20 (docs/PERFORMANCE.md P9). The gather here is one pass
/// per column, and runs per column in parallel.
fn layout_of(groups: &[(GroupKey, Vec<usize>)], n: usize) -> Option<Vec<usize>> {
    let mut base = 0;
    let blocked = groups.iter().all(|(_, idx)| {
        // Indices are distinct and increasing, so first and last say whether
        // the group is one run, and `base` whether the runs are in order.
        let run = idx
            .first()
            .is_none_or(|&f| f == base && idx[idx.len() - 1] == base + idx.len() - 1);
        base += idx.len();
        run
    });
    if blocked {
        return None;
    }
    let mut perm = Vec::with_capacity(n);
    for (_, idx) in groups {
        perm.extend_from_slice(idx);
    }
    debug_assert_eq!(perm.len(), n);
    Some(perm)
}

/// A session or group key as strings. Any dtype with a string form is a key
/// (ints, dates and categoricals included); a nested one is refused by name.
fn key_column(df: &DataFrame, spec: &Spec, role: &str, name: &str) -> PolarsResult<Series> {
    let col = column(df, spec, role, name)?;
    col.as_materialized_series()
        .cast(&DataType::String)
        .map_err(|e| {
            polars_err!(ComputeError:
                "spec {:?}: {} column {:?} has dtype {}, which cannot be used as a key: {}",
                spec.name, role, name, col.dtype(), e
            )
        })
}

/// An `ew_class` label column as class indices: each value's position in
/// `classes` as `f64`, null as NaN (a row scored but not learned from). Any
/// dtype with a string form is a label, as for a key; a non-null value the
/// spec does not list is an error naming the row, the value and the classes,
/// rather than a row silently not learned from.
fn label_column(
    df: &DataFrame,
    spec: &Spec,
    name: &str,
    classes: &[String],
    layout: Layout<'_>,
) -> PolarsResult<Vec<f64>> {
    let s = key_column(df, spec, "label", name)?;
    let ca = s.str()?;
    let mut v: Vec<f64> = Vec::with_capacity(ca.len());
    let mut last: Option<(&str, f64)> = None;
    for (j, value) in ca.iter().enumerate() {
        v.push(match value {
            None => f64::NAN,
            Some(val) => match last {
                // Labels run in streaks; the last hit answers most rows.
                Some((l, idx)) if l == val => idx,
                _ => match classes.iter().position(|c| c == val) {
                    Some(i) => {
                        last = Some((val, i as f64));
                        i as f64
                    }
                    None => polars_bail!(ComputeError:
                        "spec {:?}: label column {:?} has the value {:?} at row {}, which is \
                         not one of the classes {:?}; list every class the column can hold, \
                         or null the rows that should only be scored",
                        spec.name, name, val, j, classes
                    ),
                },
            },
        });
    }
    Ok(gathered(v, layout))
}

/// Below this many rows a chunk's columns are read, and a spec's fields
/// assembled, on the calling thread. A task at the floor is a 32 KB copy,
/// about what a rayon dispatch costs, so there is nothing to gain -- and
/// something to lose: under `.over()` the expression plugin hands the bank
/// groups of a few dozen rows, and fanning those out spread their
/// allocations over the pool's threads. Measured as a doubled RSS wobble in
/// `tests/test_ffi_memory.py` (±4.6 vs ±2 KB per call around a flat mean)
/// with no change in speed (docs/PERFORMANCE.md §12). Public so that
/// `tests/chunk_plan.rs` can run the same frames on both sides of it.
pub const PAR_MIN_ROWS: usize = 4096;

/// `items.par_iter().map(f)` when `par`, the same on this thread otherwise.
fn map_maybe_par<T, R, C>(items: &[T], par: bool, f: impl Fn(&T) -> R + Sync + Send) -> C
where
    T: Sync,
    R: Send,
    C: FromIterator<R> + FromParallelIterator<R>,
{
    if par {
        items.par_iter().map(f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

/// `rayon::join(a, b)` when `par`, `a` then `b` on this thread otherwise.
fn join_maybe_par<A, B>(
    par: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    if par { rayon::join(a, b) } else { (a(), b()) }
}

/// `scoring` is [`Bank::predict`]'s reading of the frame: the features and
/// the clock are required as ever, a target or session column is read when
/// present and taken as absent otherwise, and the weight column is not read
/// at all -- a scoring row has nothing to weigh.
///
/// Every column comes back in `layout` order. The columns are independent
/// passes over the frame, so they are read in parallel (from [`PAR_MIN_ROWS`]
/// up): with one spec in the bank this phase was a single thread copying
/// every column in turn.
fn extract(
    df: &DataFrame,
    spec: &Spec,
    scoring: bool,
    layout: Layout<'_>,
) -> PolarsResult<SpecColumns> {
    let optional = |name: &str| scoring && df.get_column_index(name).is_none();
    let par = df.height() >= PAR_MIN_ROWS;
    let features = || -> PolarsResult<Vec<Vec<f64>>> {
        map_maybe_par(&spec.features, par, |c| {
            f64_column(df, spec, "feature", c, layout)
        })
    };
    let targets = || -> PolarsResult<Vec<Vec<f64>>> {
        // A comparison's targets are residual fields of two other specs'
        // output, which the bank fills in once those have run
        // (`compare_targets`); the frame is not read for them.
        if spec.model.compares().is_some() {
            return Ok(Vec::new());
        }
        map_maybe_par(&spec.targets, par, |c| {
            if optional(c) {
                Ok(vec![f64::NAN; df.height()])
            } else if let ModelKind::EwClass { classes, .. } = &spec.model {
                label_column(df, spec, c, classes, layout)
            } else {
                f64_column(df, spec, "target", c, layout)
            }
        })
    };
    let clock = || -> PolarsResult<Option<Vec<f64>>> {
        match &spec.clock {
            Some(c) => {
                // A temporal clock column is refused rather than cast. Casting one
                // to f64 exposes its *internal representation*, so the same 60
                // seconds becomes 60_000 / 60_000_000 / 60_000_000_000 clock units
                // depending only on whether the column is Datetime(ms/us/ns), and a
                // Date becomes 1 unit per day. `halflife`, `max_dclock` and
                // `session_gap` all live in those units, so `halflife = 600` on a
                // microsecond column silently means 600 microseconds: every row
                // decays to nothing and the output is plausible-looking garbage
                // with no error. Making the user cast is one expression and makes
                // the intended scale explicit (docs/TESTING.md T-E10).
                let dtype = column(df, spec, "clock", c)?.dtype().clone();
                if dtype.is_temporal() {
                    polars_bail!(ComputeError:
                        "spec {:?}: clock column {:?} has dtype {}; a temporal clock would be \
                         read as its internal representation (e.g. epoch microseconds), so \
                         halflife/max_dclock/session_gap would silently be in those units. \
                         Cast it to the scale you mean, e.g. \
                         pl.col({:?}).dt.epoch(\"s\").cast(pl.Float64), and use that column.",
                        spec.name, c, dtype, c
                    );
                }
                let v = f64_column(df, spec, "clock", c, layout)?;
                // Nulls arrive as NaN, which this rejects along with inf: a clock
                // with no value has no defined delta either way.
                if let Some(j) = v.iter().position(|f| !f.is_finite()) {
                    polars_bail!(ComputeError:
                        "spec {:?}: clock column {:?} has a null/non-finite value at row {}",
                        spec.name, c, source_row(layout, j)
                    );
                }
                Ok(Some(v))
            }
            None => Ok(None),
        }
    };
    let session = || -> PolarsResult<(Option<Vec<u64>>, Option<Series>)> {
        match &spec.session {
            Some(c) if !optional(c) => {
                let s = key_column(df, spec, "session", c)?;
                let hashes = gathered(s.str()?.iter().map(session_hash).collect(), layout);
                let vals = spec.closes_on_session().then(|| s.clone());
                Ok((Some(hashes), vals))
            }
            _ => Ok((None, None)),
        }
    };
    let weight = || -> PolarsResult<Option<Vec<f64>>> {
        match &spec.weight {
            Some(_) if scoring => Ok(None),
            Some(c) => {
                let v = f64_column(df, spec, "weight", c, layout)?;
                // A negative weight is never meaningful for a weighted mean, and
                // silently letting one through corrupts the accumulators (the EW
                // count and the per-target cross moments disagree about whether the
                // row happened). Non-finite weights are a different case, handled
                // uniformly with non-finite features: they mean "no information for
                // this row" and skip it (docs/PLAN.md §3), so only a *finite*
                // negative weight is an error.
                if let Some(j) = v.iter().position(|f| f.is_finite() && *f < 0.0) {
                    polars_bail!(ComputeError:
                        "spec {:?}: weight column {:?} has a negative value ({}) at row {}; \
                         weights must be >= 0 (use null to skip a row)",
                        spec.name, c, v[j], source_row(layout, j)
                    );
                }
                // `rcov` sums returns over a block; a fractional weight has
                // no meaning there, so it takes a weight only as "this row
                // is a return" or "this row is not" (E57).
                if matches!(spec.model, ModelKind::Rcov { .. }) {
                    if let Some(j) = v
                        .iter()
                        .position(|f| f.is_finite() && *f != 0.0 && *f != 1.0)
                    {
                        polars_bail!(ComputeError:
                            "spec {:?}: weight column {:?} has {} at row {}; rcov takes 0 or 1 \
                             (a realised covariance is a sum over returns, not a weighted mean)",
                            spec.name, c, v[j], source_row(layout, j)
                        );
                    }
                }
                Ok(Some(v))
            }
            None => Ok(None),
        }
    };
    // Every column in one parallel batch: the session hash and the clock
    // check are passes of their own, and used to wait for the features.
    let ((features, targets), (clock, (session, weight))) = join_maybe_par(
        par,
        || join_maybe_par(par, features, targets),
        || join_maybe_par(par, clock, || join_maybe_par(par, session, weight)),
    );
    let (session, session_str) = session?;
    Ok(SpecColumns {
        features: features?,
        targets: targets?,
        clock: clock?,
        session,
        weight: weight?,
        session_str,
    })
}

/// One stream's flat output buffers for a chunk. Fallible because a strict
/// clock policy can refuse a row (`on_clock_reset = "error"`).
type StreamRows = PolarsResult<ChunkOut>;

/// The [`ClosedRow`]s of one stream at the moment it closes: one per decay
/// instance, the state exactly as it stands (docs/ENHANCEMENTS.md E54).
///
/// `at` is where the closing row sits in the chunk -- the first row of the
/// new session, or the first row of the greater key -- which is what orders
/// the bank's queue. The PCA block is left empty; the bank fills it as the
/// row is queued, so that the sign continuity depends on queue order alone
/// and not on which thread got there first.
fn close_rows(
    si: usize,
    spec: &Spec,
    key: &GroupKey,
    stream: &Stream,
    at: usize,
) -> Vec<ClosedRow> {
    let summary = stream.summary();
    let session = spec
        .closes_on_session()
        .then(|| stream.last_session.clone())
        .flatten();
    stream
        .models
        .iter()
        .map(|(label, model)| {
            let gram = gram_of(key, label, model);
            let pairs = match model {
                AnyModel::Marginal(m) => spec
                    .targets
                    .iter()
                    .enumerate()
                    .flat_map(|(t_i, t)| {
                        spec.features.iter().enumerate().map(move |(j, f)| {
                            let p = m.pair(t_i, j);
                            PairRow {
                                feature: f.clone(),
                                target: t.clone(),
                                n_eff: p.n_eff,
                                n_kish: p.n_kish,
                                mean_x: p.mean_x,
                                var_x: p.var_x,
                                mean_y: p.mean_y,
                                var_y: p.var_y,
                                cov: p.cov,
                                corr: p.corr,
                                beta: p.beta,
                                t: p.t,
                            }
                        })
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let rcov = match model {
                AnyModel::Rcov(m) => {
                    let e = m.estimate();
                    let k = spec.k();
                    Some(RcovRow {
                        rcov: e.rcov.as_ref().map(|c| vech(c, k)),
                        rcorr: e.rcorr.as_ref().map(|c| vech(c, k)),
                        n: e.n,
                        kind: e.kind.to_string(),
                        bandwidth_used: e.bandwidth_used,
                        omega2: e.omega2,
                        iv_sparse: e.iv_sparse,
                        iq: e.iq,
                        psd_repaired: e.psd_repaired,
                    })
                }
                _ => None,
            };
            ClosedRow {
                spec: si,
                group: key.clone(),
                instance: label.clone(),
                session: session.clone(),
                n_eff: model.n_eff(),
                n_kish: gram.as_ref().and_then(|g| g.n_kish),
                rows_fed: summary.map(|s| s.rows_fed),
                rows_learned: summary.map(|s| s.rows_learned),
                clock_min: summary.and_then(|s| s.clock_min),
                clock_max: summary.and_then(|s| s.clock_max),
                gram,
                coef: model
                    .coefficients()
                    .map(|c| c.into_iter().flatten().collect()),
                eig_vals: None,
                eig_vecs: None,
                pairs,
                rcov,
                at,
            }
        })
        .collect()
}

/// Refuse a chunk whose group keys are not in non-decreasing order, or
/// whose first key is below what this spec has already closed past
/// (docs/ENHANCEMENTS.md E54).
///
/// This is what makes interleaving an *error* rather than a wrong answer:
/// [`group_indices`] partitions by first appearance, so the rows of `A, B,
/// A` would be gathered into one run per key and the close would emit `A`
/// while its later rows were still to come. The check runs before any model
/// is touched, beside `check_clock`, so a refused chunk leaves the bank as
/// it was.
///
/// `slot` is scratch: the group each row belongs to, so the walk compares
/// two `GroupKey`s rather than re-reading the column.
fn check_monotone(
    spec: &Spec,
    groups: &[(GroupKey, Vec<usize>)],
    n: usize,
    integer: bool,
    high_water: Option<&GroupKey>,
) -> PolarsResult<()> {
    use std::cmp::Ordering::*;
    if groups.is_empty() {
        return Ok(());
    }
    if let Some((_, idx)) = groups.iter().find(|(k, _)| k.as_str().is_none()) {
        polars_bail!(ComputeError:
            "spec {:?}: row {} has a null group; group_close = \"monotone\" orders the keys, \
             and a null has no place in that order (a null key is an ordinary group under \
             group_close = \"session\")",
            spec.name, idx[0]
        );
    }
    let mut slot = vec![usize::MAX; n];
    for (gi, (_, idx)) in groups.iter().enumerate() {
        for &r in idx {
            slot[r] = gi;
        }
    }
    let mut prev = usize::MAX;
    for (r, &gi) in slot.iter().enumerate() {
        if gi == usize::MAX {
            continue;
        }
        if prev != usize::MAX
            && gi != prev
            && key_cmp(&groups[gi].0, &groups[prev].0, integer) == Less
        {
            polars_bail!(ComputeError:
                "spec {:?}: row {}: group {} after group {} -- group_close = \"monotone\" needs \
                 the keys in non-decreasing order (sort the input by {:?}, or use \
                 group_close = \"session\")",
                spec.name, r, groups[gi].0, groups[prev].0,
                spec.group.as_deref().unwrap_or("")
            );
        }
        prev = gi;
    }
    if let Some(hw) = high_water {
        // A mark stored under one dtype and read under another is a
        // different key: `"10"` orders after `"9"` as an integer and before
        // it as a string, so the mark would let through exactly the groups
        // it exists to refuse.
        if integer && hw.as_str().is_none_or(|k| k.parse::<i128>().is_err()) {
            polars_bail!(ComputeError:
                "spec {:?}: the saved high-water key {} is not an integer, and the group column \
                 {:?} is; the state was saved under a different key column",
                spec.name, hw, spec.group.as_deref().unwrap_or("")
            );
        }
        // The first key of the chunk is the smallest, the walk above having
        // passed; anything below the mark belongs to a group this bank has
        // already emitted and dropped.
        if key_cmp(&groups[0].0, hw, integer) == Less {
            polars_bail!(ComputeError:
                "spec {:?}: row {}: group {} is below {}, which this bank has already closed \
                 past; a closed group cannot be reopened",
                spec.name, groups[0].1[0], groups[0].0, hw
            );
        }
    }
    Ok(())
}

/// The rows of `idx` at which this stream's session value changes, under
/// `group_close = "session"` (E54). Each is the first row of a new span, so
/// the span before it closes there.
///
/// Read from the session hashes the columns already carry and the stream's
/// own last hash, in row order: a boundary is a property of two consecutive
/// rows, which is what makes the split chunk-invariant.
fn session_bounds(stream: &Stream, sess: &[u64], n: usize, base: usize) -> Vec<usize> {
    let mut prev = stream.clock.prev_session();
    let mut bounds = Vec::new();
    for ri in 0..n {
        let h = sess[base + ri];
        if prev.is_some_and(|p| p != h) {
            bounds.push(ri);
        }
        prev = Some(h);
    }
    bounds
}

/// Run one phase of a chunk through its streams, learning: every `(spec,
/// group)` task in parallel, each in runs of [`ChunkOut::run_rows`] rows
/// through its stream in order, one set of buffers per run. The runs are a
/// cache-sizing device (docs/PERFORMANCE.md §13); chunk invariance makes them
/// the same computation as one, and `last` keeps the coefficient report on
/// the chunk's final row.
fn process(
    work: Vec<(usize, &GroupKey, &Vec<usize>, usize, &mut Stream)>,
    specs: &[Spec],
    cfgs: &[ClockCfg],
    cols: &[SpecColumns],
) -> (Vec<(usize, StreamRows)>, Vec<ClosedRow>) {
    type TaskOut = (Vec<(usize, StreamRows)>, Vec<ClosedRow>);
    let per_task: Vec<TaskOut> = work
        .into_par_iter()
        .map(|(si, key, idx, base, stream)| {
            let spec = &specs[si];
            let sc = &cols[si];
            let mut outs: Vec<(usize, StreamRows)> = Vec::new();
            let mut closed: Vec<ClosedRow> = Vec::new();
            // A session close splits the run: the rows before the boundary
            // are the span, the stream emits its row and starts over, and
            // the rows after are a fresh stream's first rows (E54). Every
            // segment gets its own buffers, exactly as the cache-sized runs
            // below do, so nothing else in the chunk path changes.
            let bounds = match (spec.closes_on_session(), sc.session.as_deref()) {
                (true, Some(sess)) => session_bounds(stream, sess, idx.len(), base),
                _ => Vec::new(),
            };
            let mut starts: Vec<usize> = Vec::with_capacity(bounds.len() + 1);
            starts.push(0);
            starts.extend_from_slice(&bounds);
            let session_of = |row: usize| -> Option<String> {
                let s = sc.session_str.as_ref()?;
                s.str().ok()?.get(row).map(str::to_string)
            };
            'segments: for (seg, &start) in starts.iter().enumerate() {
                let end = starts.get(seg + 1).copied().unwrap_or(idx.len());
                if seg > 0 {
                    // The span before this row is finished.
                    closed.extend(close_rows(si, spec, key, stream, idx[start]));
                    *stream = Stream::new(spec).expect("the spec built this stream once");
                }
                let (n_models, n_slots) = (stream.n_models(), stream.n_slots());
                let run_rows = ChunkOut::run_rows(spec, n_models, n_slots);
                let mut off = start;
                for run in idx[start..end].chunks(run_rows) {
                    let last = off + run.len() == idx.len();
                    let mut out = ChunkOut::new(spec, n_models, n_slots, run.len());
                    let r = stream.process_chunk(
                        spec,
                        &cfgs[si],
                        &sc.features,
                        &sc.targets,
                        sc.clock.as_deref(),
                        sc.session.as_deref(),
                        sc.weight.as_deref(),
                        run,
                        base + off,
                        &mut out,
                        last,
                    );
                    off += run.len();
                    match r {
                        Ok(()) => {
                            stream.remember_last(&out);
                            outs.push((si, Ok(out)));
                        }
                        Err((raw, i)) => {
                            outs.push((si, Err(backwards_clock(spec, raw, i))));
                            break 'segments;
                        }
                    }
                }
                // The span's session value, for the row it will close with.
                if end > start && sc.session_str.is_some() {
                    stream.last_session = session_of(idx[end - 1]);
                }
            }
            (outs, closed)
        })
        .collect();
    let mut outs = Vec::new();
    let mut closed = Vec::new();
    for (o, c) in per_task {
        outs.extend(o);
        closed.extend(c);
    }
    (outs, closed)
}

/// [`process`] without the learning: [`Stream::predict_chunk`] per task.
fn score(
    work: Vec<(usize, &Vec<usize>, usize, &Stream)>,
    specs: &[Spec],
    cfgs: &[ClockCfg],
    cols: &[SpecColumns],
) -> Vec<(usize, StreamRows)> {
    work.into_par_iter()
        .map(|(si, idx, base, stream)| {
            let spec = &specs[si];
            let sc = &cols[si];
            let r = (|| {
                let mut out = ChunkOut::new(spec, stream.n_models(), stream.n_slots(), idx.len());
                stream
                    .predict_chunk(
                        spec,
                        &cfgs[si],
                        &sc.features,
                        &sc.targets,
                        sc.clock.as_deref(),
                        sc.session.as_deref(),
                        idx,
                        base,
                        &mut out,
                    )
                    .map_err(|(raw, i)| backwards_clock(spec, raw, i))?;
                Ok(out)
            })();
            (si, r)
        })
        .collect()
}

/// Assemble the structs of the specs `pick` selects, in parallel, into
/// `out` (docs/PERFORMANCE.md P4).
fn assemble_phase(
    specs: &[Spec],
    derived: &[SpecDerived],
    n: usize,
    rows: &[Vec<ChunkOut>],
    out: &mut [Option<Column>],
    pick: impl Fn(usize) -> bool + Sync,
) -> PolarsResult<()> {
    let built: Vec<(usize, Column)> = (0..specs.len())
        .into_par_iter()
        .filter(|si| pick(*si))
        .map(|si| Ok((si, assemble(&specs[si], &derived[si], n, &rows[si])?)))
        .collect::<PolarsResult<_>>()?;
    for (si, c) in built {
        out[si] = Some(c);
    }
    Ok(())
}

/// A comparison's targets once its two sides have run: per target `t`,
/// `|resid_b| - |resid_a|` from the two structs -- positive when `a` was
/// closer on the row -- in the spec's layout order, NaN where either side
/// is null (a row one side did not predict is no trial). Any loss that
/// grows with `|resid|` orders the two sides the same way, so the sign is
/// the same test for absolute, squared or Huber loss.
fn compare_targets(
    spec: &Spec,
    (a, b): (usize, usize),
    out: &[Option<Column>],
    layout: Layout<'_>,
) -> PolarsResult<Vec<Vec<f64>>> {
    let cmp = spec.model.compares().expect("resolved as a comparison");
    let side = |si: usize, field: &str| -> PolarsResult<Vec<f64>> {
        let col = out[si]
            .as_ref()
            .expect("the two sides of a comparison are assembled in the first phase");
        let field = col.struct_()?.field_by_name(field)?;
        Ok(field.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
    };
    spec.targets
        .iter()
        .map(|t| {
            let (fa, fb) = cmp.fields(t);
            let (ra, rb) = (side(a, &fa)?, side(b, &fb)?);
            let d = ra.iter().zip(&rb).map(|(x, y)| y.abs() - x.abs()).collect();
            Ok(gathered(d, layout))
        })
        .collect()
}

/// The error for a backwards clock under `on_clock_reset = "error"`: names the
/// spec, the column, the size of the step back, the row, and the way out.
fn backwards_clock(spec: &Spec, raw: f64, row: usize) -> PolarsError {
    polars_err!(ComputeError:
        "spec {:?}: clock column {:?} goes backwards by {} at row {} \
         (on_clock_reset = \"error\"); the bank was not updated. Sort each \
         group by the clock, or choose \"max\"/\"zero\"/\"reset_state\" to \
         define what a backwards clock means.",
        spec.name,
        spec.clock.as_deref().unwrap_or("<row count>"),
        -raw, row
    )
}

/// Drop the streams a refused chunk materialized, `(spec index, key)` each.
fn forget(states: &mut [HashMap<GroupKey, Stream>], fresh: &[(usize, GroupKey)]) {
    for (si, key) in fresh {
        states[*si].remove(key);
    }
}

/// Row-index partition by group key, in row order.
///
/// Keyed on a 64-bit hash of the string value rather than on the string
/// itself, so the per-row cost is a hash rather than a `String` allocation and
/// two more clones (docs/PERFORMANCE.md P3). The `GroupKey` is materialized
/// once per distinct group, not once per row -- and it is still the *value*
/// that is stored and serialized, so state files are unaffected. A 64-bit
/// collision would merge two groups; the same 2^-64 exposure the session hash
/// already documents.
fn group_indices(df: &DataFrame, spec: &Spec) -> PolarsResult<Vec<(GroupKey, Vec<usize>)>> {
    match &spec.group {
        None => Ok(vec![(GroupKey::ungrouped(), (0..df.height()).collect())]),
        Some(g) => {
            // An integer key's text is its decimal, which is exactly what the
            // String cast below would produce (`integer_group_keys_match_the_
            // string_cast` in tests/bank.rs pins the two paths to the same
            // keys and output), so the value itself is the bucket: no cast, no
            // hash, and no collision to document (docs/PERFORMANCE.md P11).
            let col = column(df, spec, "group", g)?;
            if col.dtype().is_integer() {
                let s = col.as_materialized_series();
                return Ok(if *s.dtype() == DataType::UInt64 {
                    integer_groups(s.u64()?, |v| v)
                } else {
                    integer_groups(s.cast(&DataType::Int64)?.i64()?, |v| v as u64)
                });
            }
            let s = key_column(df, spec, "group", g)?;
            let mut order: Vec<(GroupKey, Vec<usize>)> = Vec::new();
            let mut slot_of: PlHashMap<u64, usize> = PlHashMap::default();
            for (i, v) in s.str()?.iter().enumerate() {
                let h = match v {
                    None => null_session_hash(),
                    Some(v) => {
                        let h = fnv1a(v.as_bytes());
                        if h == null_session_hash() { h ^ 1 } else { h }
                    }
                };
                match slot_of.get(&h) {
                    Some(&slot) => order[slot].1.push(i),
                    None => {
                        slot_of.insert(h, order.len());
                        order.push((GroupKey(v.map(str::to_string)), vec![i]));
                    }
                }
            }
            Ok(order)
        }
    }
}

/// The order `group_close = "monotone"` reads keys in.
///
/// [`GroupKey`] holds every key as a string and its derived `Ord` is
/// therefore lexicographic, which puts `"10"` before `"9"` -- a monotone
/// close on that order would close group 10 before group 9 had arrived. So
/// the comparison is the *column's*: integer keys compare as integers (the
/// text is what [`integer_groups`] formatted, so it parses), and everything
/// else -- String and Categorical, both rendered as strings by [`extract`]
/// -- compares bytewise, which is what a string column's own sort does.
///
/// A null key has no place in either order; `"monotone"` refuses one at the
/// pre-check, so `None` here only ever meets `None`.
pub(crate) fn key_cmp(a: &GroupKey, b: &GroupKey, integer: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.as_str(), b.as_str()) {
        (Some(a), Some(b)) if integer => match (a.parse::<i128>(), b.parse::<i128>()) {
            (Ok(a), Ok(b)) => a.cmp(&b),
            // Unreachable through `group_indices`, which formats these keys
            // itself; a state file whose key text is not a number falls back
            // to the bytes rather than panicking.
            _ => a.cmp(b),
        },
        (Some(a), Some(b)) => a.as_bytes().cmp(b.as_bytes()),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

/// Is this spec's group column an integer one, so that [`key_cmp`] compares
/// its keys as numbers? `Err` names a dtype `"monotone"` cannot order.
fn group_key_is_integer(df: &DataFrame, spec: &Spec) -> PolarsResult<bool> {
    let Some(g) = &spec.group else {
        return Ok(false);
    };
    let dt = column(df, spec, "group", g)?.dtype().clone();
    if dt.is_integer() {
        return Ok(true);
    }
    if matches!(dt, DataType::String | DataType::Categorical(..)) {
        return Ok(false);
    }
    polars_bail!(ComputeError:
        "spec {:?}: group_close = \"monotone\" needs a group column it can order, and {:?} is \
         {}; use an integer, String or Categorical key (a float or a temporal column can be \
         cast to one)",
        spec.name, g, dt
    )
}

/// [`group_indices`] over an integer column: `bucket` maps a value to its
/// 64-bit identity (sign extension is a bijection, so `as u64` serves every
/// signed width), and the key text is formatted once per distinct group.
fn integer_groups<T>(
    ca: &ChunkedArray<T>,
    bucket: impl Fn(T::Native) -> u64,
) -> Vec<(GroupKey, Vec<usize>)>
where
    T: PolarsNumericType,
    T::Native: std::fmt::Display,
{
    let mut order: Vec<(GroupKey, Vec<usize>)> = Vec::new();
    let mut slot_of: PlHashMap<u64, usize> = PlHashMap::default();
    let mut null_slot: Option<usize> = None;
    for (i, v) in ca.iter().enumerate() {
        let slot = match v {
            None => *null_slot.get_or_insert_with(|| {
                order.push((GroupKey(None), Vec::new()));
                order.len() - 1
            }),
            Some(v) => *slot_of.entry(bucket(v)).or_insert_with(|| {
                order.push((GroupKey(Some(v.to_string())), Vec::new()));
                order.len() - 1
            }),
        };
        order[slot].1.push(i);
    }
    order
}

/// The axes a spec's [`Gram`] is indexed by: its columns (the features,
/// with the intercept in front when the spec has one) and its targets. The
/// Python `ModelBank.gram` builds the same two lists from the spec dict, and
/// `tests/test_closed_groups.py` holds the two to each other.
pub(crate) fn gram_axes(spec: &Spec) -> (Vec<String>, Vec<String>) {
    // `ew_cov` accumulates over the features alone -- no target, and so no
    // constant column to regress one on.
    let unsupervised = matches!(spec.model, ModelKind::EwCov { .. });
    let mut columns = spec.features.clone();
    if spec.add_intercept && !unsupervised {
        columns.insert(0, "intercept".to_string());
    }
    let targets = if unsupervised {
        Vec::new()
    } else {
        spec.targets.clone()
    };
    (columns, targets)
}

/// The upper triangle of a row-major `k * k` symmetric matrix, with the
/// diagonal, row by row: `k(k+1)/2` numbers instead of `k^2`, which is what
/// the closed row carries (`po.gram.from_row` expands it). A co-moment
/// matrix is symmetric to 1e-16 and not to the bit (E48), so the half that
/// is kept is named: the upper one.
pub(crate) fn vech(m: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k * (k + 1) / 2);
    for i in 0..k {
        for j in i..k {
            out.push(m[i * k + j]);
        }
    }
    out
}

/// One model instance's [`Gram`], or `None` for a kind that keeps no
/// co-moment matrix (`rls` and `kalman` track an inverse; the gradient
/// models keep no second moment at all).
///
/// The one builder behind both [`Bank::gram`] and a [`ClosedRow`]'s Gram
/// block, so "a closed row equals `gram()` read at the same point" holds by
/// construction and the test is the tripwire that keeps it so.
pub(crate) fn gram_of(key: &GroupKey, label: &str, model: &AnyModel) -> Option<Gram> {
    let lag = match model {
        AnyModel::EwCov(m) => m.lag(),
        _ => None,
    };
    let (cov, cross, weights, tm, targetless) = match model {
        AnyModel::EwRidge(m) => (
            m.cov(),
            m.cross_moments().to_vec(),
            m.target_weights().to_vec(),
            m.target_moments(),
            false,
        ),
        AnyModel::Lasso(m) => (
            m.cov(),
            m.cross_moments().to_vec(),
            m.target_weights().to_vec(),
            m.target_moments(),
            false,
        ),
        // No targets, so no cross-moments: the matrix is the whole output.
        AnyModel::EwCov(m) => (m.cov(), Vec::new(), Vec::new(), None, true),
        _ => return None,
    };
    // Empty says "this model has no targets"; `None` says "this state was
    // written before task 38 and cannot say". They are different answers, so
    // `ew_cov` reports empty, not `None`.
    let (target_means, target_vars, target_n_kish) = if targetless {
        (Some(Vec::new()), Some(Vec::new()), Some(Vec::new()))
    } else {
        (
            tm.map(|t| t.means().to_vec()),
            tm.map(|t| t.vars().to_vec()),
            tm.map(|t| t.n_kish(&weights)),
        )
    };
    Some(Gram {
        group: key.clone(),
        instance: label.to_string(),
        k: cov.k(),
        n_eff: cov.n_eff(),
        n_kish: cov.n_kish(),
        means: cov.means().to_vec(),
        comoments: cov.comoments().to_vec(),
        cross_moments: cross,
        target_means,
        target_vars,
        target_n_kish,
        target_weights: weights,
        lags: lag.map(|l| l.lags().to_vec()),
        lag_comoments: lag.map(|l| l.comoments().to_vec()),
    })
}

/// One instance's EW accumulators, as returned by [`Bank::gram`].
///
/// Values are in the features' original units. `comoments` is **centered**
/// (E11b, which is what makes it accurate at large offsets); `cross_moments`
/// is **uncentered**, because that is the form the solve consumes. The two
/// are bridged by one identity, and getting it wrong is a silent wrong
/// answer rather than an error:
///
/// ```text
/// raw[i][j] = comoments[i*k+j] + means[i]*means[j]
/// raw · beta[t] = cross_moments[t]        (up to the ridge term)
/// ```
///
/// The intercept, when the spec has one, is column 0: a constant 1, so it has
/// zero variance in `comoments` and `raw[0][j] == means[j]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gram {
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for a single instance).
    pub instance: String,
    /// Columns, including the intercept when the spec has one.
    pub k: usize,
    /// Accumulated weight behind these moments.
    pub n_eff: f64,
    /// Kish's effective sample size over the feature rows, `n_eff^2 / Sum w^2`
    /// (docs/ENHANCEMENTS.md E45): the number of equally weighted rows these
    /// moments are worth, which is what a standard error divides by. `None`
    /// before the first row and for a state written before task 38.
    pub n_kish: Option<f64>,
    /// EW column means, length `k`.
    pub means: Vec<f64>,
    /// Centered co-moments, row-major `k*k`.
    pub comoments: Vec<f64>,
    /// Per-target uncentered cross-moments, each `k` long. Empty for
    /// `ew_cov`.
    pub cross_moments: Vec<Vec<f64>>,
    /// Per-target accumulated weight. Empty for `ew_cov`.
    pub target_weights: Vec<f64>,
    /// Per-target EW mean of the target. Empty for `ew_cov` (no targets);
    /// `None` for a state written before task 38.
    pub target_means: Option<Vec<f64>>,
    /// Per-target EW centered variance of the target, alongside
    /// [`Self::target_means`].
    pub target_vars: Option<Vec<f64>>,
    /// Per-target Kish effective sample size, `W_t^2 / Q_t`; an entry is
    /// `None` for a target that has not seen a weighted row.
    pub target_n_kish: Option<Vec<Option<f64>>>,
    /// The lags an `ew_cov(lags = ...)` accumulates at, in output order, and
    /// their cross-moments as `L * k * k` row-major within a lag
    /// (docs/ENHANCEMENTS.md E56). `None` for a spec without lags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lags: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_comoments: Option<Vec<f64>>,
}

/// One (feature, target) pair of a closed `marginal` group, the columns
/// [`Bank::marginal`] reports for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRow {
    pub feature: String,
    pub target: String,
    pub n_eff: f64,
    pub n_kish: f64,
    pub mean_x: f64,
    pub var_x: f64,
    pub mean_y: f64,
    pub var_y: f64,
    pub cov: f64,
    pub corr: f64,
    pub beta: f64,
    pub t: f64,
}

/// One group's accumulators at the moment the bank could prove that no
/// further row would join it (docs/ENHANCEMENTS.md E54), one per decay
/// instance.
///
/// The row is the state *as it stood*, so it equals the [`Bank::gram`] a
/// driver would have read at the same point, field for field and bit for
/// bit -- both are built by [`gram_of`], which is what keeps that true. The
/// stream is dropped once the row is made, which is the point: a bank over
/// an unbounded key space stays bounded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedRow {
    /// Index of the spec in the bank; the frame carries its name.
    pub spec: usize,
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for one instance).
    pub instance: String,
    /// The session value of the span just closed, under `group_close =
    /// "session"`; `None` under `"monotone"`, and for a null session value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    pub n_eff: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_kish: Option<f64>,
    /// Rows fed and rows learned from over the span, from the stream's own
    /// [`DataSummary`]; `None` only for a stream restored from a file
    /// written before task 35's summary existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_fed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_learned: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_max: Option<f64>,
    /// The accumulators, for a kind that keeps them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gram: Option<Gram>,
    /// The flat `coef` list as of the last solve, for a kind that reports
    /// one; the exact solve on the closed Gram is
    /// `po.gram.solve(po.gram.from_row(row))`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coef: Option<Vec<f64>>,
    /// `ew_cov(pca = r)`: the eigenvalues of this row's own `comoments`,
    /// descending, and the loadings row-major `r * k`. Filled by the bank
    /// as the row is queued, so the sign continuity is a function of queue
    /// order alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eig_vals: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eig_vecs: Option<Vec<f64>>,
    /// A `marginal` spec's pairs, in [`Bank::marginal`] order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pairs: Vec<PairRow>,
    /// An `rcov` spec's block (docs/ENHANCEMENTS.md E57): the realised
    /// covariance and correlation as `vech` of the upper triangle, the
    /// effective return count, the estimator, and the noise diagnostics
    /// behind an automatic bandwidth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rcov: Option<RcovRow>,
    /// Position of the closing row within the chunk that closed it: the
    /// first row of the new session, or the first row of the greater key.
    /// The queue's order, and not part of the row itself.
    #[serde(skip)]
    pub at: usize,
}

/// A `list[f64]` column: one list per row, `None` where the row has none,
/// and null inside for a value that is not finite -- `coef`'s rule, and
/// `marginal()`'s (a constant column's `corr` is NaN there and null here).
fn list_f64(name: &str, rows: &[ClosedRow], f: impl Fn(&ClosedRow) -> Option<Vec<f64>>) -> Column {
    let mut b = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        name.into(),
        rows.len(),
        8,
        DataType::Float64,
    );
    for r in rows {
        match f(r) {
            Some(v) => b.append_iter(v.iter().map(|x| x.is_finite().then_some(*x))),
            None => b.append_null(),
        }
    }
    b.finish().into_series().into()
}

/// A `list[str]` column, as [`list_f64`].
fn list_str(
    name: &str,
    rows: &[ClosedRow],
    f: impl Fn(&ClosedRow) -> Option<Vec<String>>,
) -> Column {
    let mut b = ListStringChunkedBuilder::new(name.into(), rows.len(), 8);
    for r in rows {
        match f(r) {
            Some(v) => b.append_values_iter(v.iter().map(String::as_str)),
            None => b.append_null(),
        }
    }
    b.finish().into_series().into()
}

/// The frame [`Bank::closed_groups`] returns: the common columns always,
/// and a kind's block when **any** spec of the bank closes groups and is of
/// that kind, null on the rows of other kinds (docs/ENHANCEMENTS.md E54).
///
/// One schema per bank, not per call: a driver concatenating a run's drains
/// needs every frame to have the same columns, whatever happened to close in
/// that chunk.
fn closed_frame(specs: &[Spec], rows: &[ClosedRow]) -> PolarsResult<DataFrame> {
    let closing: Vec<&Spec> = specs.iter().filter(|s| s.group_close.is_some()).collect();
    let any = |f: fn(&Spec) -> bool| closing.iter().any(|s| f(s));
    let has_gram = any(|s| {
        matches!(
            s.model,
            ModelKind::EwRidge { .. } | ModelKind::Lasso { .. } | ModelKind::EwCov { .. }
        )
    });
    let has_pca = any(|s| matches!(s.model, ModelKind::EwCov { pca: Some(_), .. }));
    let has_pairs = any(|s| matches!(s.model, ModelKind::Marginal {}));
    let has_rcov = any(|s| matches!(s.model, ModelKind::Rcov { .. }));
    let opt = |v: f64| v.is_finite().then_some(v);

    let mut cols: Vec<Column> = vec![
        Column::new(
            "spec".into(),
            rows.iter()
                .map(|r| specs[r.spec].name.as_str())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "group".into(),
            rows.iter().map(|r| r.group.as_str()).collect::<Vec<_>>(),
        ),
        Column::new(
            "instance".into(),
            rows.iter().map(|r| r.instance.as_str()).collect::<Vec<_>>(),
        ),
        Column::new(
            "session".into(),
            rows.iter()
                .map(|r| r.session.as_deref())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "n_eff".into(),
            rows.iter().map(|r| opt(r.n_eff)).collect::<Vec<_>>(),
        ),
        Column::new(
            "n_kish".into(),
            rows.iter()
                .map(|r| r.n_kish.and_then(opt))
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "rows_fed".into(),
            rows.iter()
                .map(|r| r.rows_fed.map(|v| v as i64))
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "rows_learned".into(),
            rows.iter()
                .map(|r| r.rows_learned.map(|v| v as i64))
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "clock_min".into(),
            rows.iter()
                .map(|r| r.clock_min.and_then(opt))
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "clock_max".into(),
            rows.iter()
                .map(|r| r.clock_max.and_then(opt))
                .collect::<Vec<_>>(),
        ),
    ];
    if has_gram {
        let axes = |r: &ClosedRow| gram_axes(&specs[r.spec]);
        cols.push(list_str("columns", rows, |r| {
            r.gram.as_ref().map(|_| axes(r).0)
        }));
        cols.push(list_f64("means", rows, |r| {
            r.gram.as_ref().map(|g| g.means.clone())
        }));
        cols.push(list_f64("comoments", rows, |r| {
            r.gram.as_ref().map(|g| vech(&g.comoments, g.k))
        }));
        cols.push(list_str("targets", rows, |r| {
            r.gram.as_ref().map(|_| axes(r).1)
        }));
        cols.push(list_f64("target_means", rows, |r| {
            r.gram.as_ref().and_then(|g| g.target_means.clone())
        }));
        cols.push(list_f64("target_vars", rows, |r| {
            r.gram.as_ref().and_then(|g| g.target_vars.clone())
        }));
        cols.push(list_f64("target_weights", rows, |r| {
            r.gram.as_ref().map(|g| g.target_weights.clone())
        }));
        cols.push(list_f64("target_n_kish", rows, |r| {
            r.gram.as_ref().and_then(|g| {
                g.target_n_kish
                    .as_ref()
                    .map(|v| v.iter().map(|x| x.unwrap_or(f64::NAN)).collect())
            })
        }));
        cols.push(list_f64("cross_moments", rows, |r| {
            r.gram
                .as_ref()
                .map(|g| g.cross_moments.iter().flatten().copied().collect())
        }));
    }
    if any(|s| matches!(s.model, ModelKind::EwCov { lags: Some(_), .. })) {
        let mut b = ListPrimitiveChunkedBuilder::<Int64Type>::new(
            "lags".into(),
            rows.len(),
            4,
            DataType::Int64,
        );
        for r in rows {
            match r.gram.as_ref().and_then(|g| g.lags.as_ref()) {
                Some(v) => b.append_values_iter(v.iter().map(|&l| l as i64)),
                None => b.append_null(),
            }
        }
        cols.push(b.finish().into_series().into());
        cols.push(list_f64("lag_comoments", rows, |r| {
            r.gram.as_ref().and_then(|g| g.lag_comoments.clone())
        }));
    }
    cols.push(list_f64("coef", rows, |r| r.coef.clone()));
    if has_pca {
        cols.push(list_f64("eig_vals", rows, |r| r.eig_vals.clone()));
        cols.push(list_f64("eig_vecs", rows, |r| r.eig_vecs.clone()));
    }
    if has_pairs {
        let empty = |r: &ClosedRow| r.pairs.is_empty();
        cols.push(list_str("pair_feature", rows, |r| {
            (!empty(r)).then(|| r.pairs.iter().map(|p| p.feature.clone()).collect())
        }));
        cols.push(list_str("pair_target", rows, |r| {
            (!empty(r)).then(|| r.pairs.iter().map(|p| p.target.clone()).collect())
        }));
        for (name, f) in [
            ("pair_n_eff", (|p: &PairRow| p.n_eff) as fn(&PairRow) -> f64),
            ("pair_n_kish", |p| p.n_kish),
            ("pair_mean_x", |p| p.mean_x),
            ("pair_var_x", |p| p.var_x),
            ("pair_mean_y", |p| p.mean_y),
            ("pair_var_y", |p| p.var_y),
            ("pair_cov", |p| p.cov),
            ("pair_corr", |p| p.corr),
            ("pair_beta", |p| p.beta),
            ("pair_t", |p| p.t),
        ] {
            cols.push(list_f64(name, rows, move |r| {
                (!empty(r)).then(|| r.pairs.iter().map(f).collect())
            }));
        }
    }
    if has_rcov {
        let block =
            |f: fn(&RcovRow) -> Option<Vec<f64>>| move |r: &ClosedRow| r.rcov.as_ref().and_then(f);
        cols.push(list_f64("rcov", rows, block(|b| b.rcov.clone())));
        cols.push(list_f64("rcorr", rows, block(|b| b.rcorr.clone())));
        cols.push(Column::new(
            "rcov_n".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().map(|b| b.n))
                .collect::<Vec<_>>(),
        ));
        cols.push(Column::new(
            "rcov_kind".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().map(|b| b.kind.as_str()))
                .collect::<Vec<_>>(),
        ));
        cols.push(Column::new(
            "bandwidth_used".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().and_then(|b| b.bandwidth_used))
                .collect::<Vec<_>>(),
        ));
        cols.push(list_f64("omega2", rows, block(|b| b.omega2.clone())));
        cols.push(list_f64("iv_sparse", rows, block(|b| b.iv_sparse.clone())));
        cols.push(list_f64("iq", rows, block(|b| b.iq.clone())));
        cols.push(Column::new(
            "psd_repaired".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().map(|b| b.psd_repaired))
                .collect::<Vec<_>>(),
        ));
    }
    DataFrame::new(rows.len(), cols)
}

/// A closed `rcov` group's block, as the row carries it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RcovRow {
    /// `vech` of the upper triangle with the diagonal, as `comoments` is.
    pub rcov: Option<Vec<f64>>,
    pub rcorr: Option<Vec<f64>>,
    pub n: i64,
    pub kind: String,
    pub bandwidth_used: Option<i64>,
    pub omega2: Option<Vec<f64>>,
    pub iv_sparse: Option<Vec<f64>>,
    pub iq: Option<Vec<f64>>,
    pub psd_repaired: bool,
}

/// One decay instance's coefficients, from [`Bank::coef`].
#[derive(Debug, Clone, PartialEq)]
pub struct Coef {
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for a single instance).
    pub instance: String,
    /// The accumulated weight behind the fit -- the next row's `n_eff`. The
    /// solve schedule, not `min_periods`, decides when `coef` first appears,
    /// so this is how a caller tells a warm fit from one over fewer rows
    /// than `min_periods` asks for (`pred` waits for that; `coef` does not).
    pub n_eff: f64,
    /// The flat list the output's `coef` field reports, in
    /// `polars_online.spec.coef_index` order; `None` before the first solve.
    pub coef: Option<Vec<f64>>,
}

/// One coefficient of a spec's output, from [`coef_fields`]: which `coef`
/// list it sits in and where, and the flat column name it gets when the
/// struct is unnested with the coefficients named
/// (`polars_online.spec.coef_fields`, `lf.online.unnest`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CoefField {
    /// The `coef` field holding the list (`coef`, or `coef@h500` per
    /// instance).
    pub field: String,
    /// Position in that list.
    pub position: usize,
    /// `coef_{target}_{term}{combo}{instance}` -- the field grammar with the
    /// term after the target, so `coef_y_x1__r0.5@h500` sits beside
    /// `pred_y__r0.5@h500`. Prefixed, because a bare `x1` would collide with
    /// the feature column of that name in the same frame.
    pub name: String,
    pub target: String,
    pub halflife: Option<f64>,
    pub lam: Option<f64>,
    pub ridge: Option<f64>,
    pub feature_set: Option<String>,
    /// Lasso path point.
    pub lambda: Option<f64>,
    /// `intercept`, a feature name, or `level` / `trend` for `holt`.
    pub term: String,
}

const BANK_MAGIC: &str = "polars-online-bank";

/// The envelope of a [`BankFile`], read on its own first: a file from a newer
/// build may carry keys this build's [`Spec`] refuses, and the version check
/// has to run before that refusal can be mistaken for "not a bank file".
#[derive(Deserialize)]
struct BankHeader {
    magic: String,
    #[serde(default = "default_format_version")]
    format_version: u32,
    schema_version: u32,
}

#[derive(Serialize, Deserialize)]
struct BankFile {
    magic: String,
    /// Bank file layout (see [`BANK_FORMAT_VERSION`]). Absent in v1 files.
    #[serde(default = "default_format_version")]
    format_version: u32,
    /// Model state layout (`online_core::SCHEMA_VERSION`).
    schema_version: u32,
    package_version: String,
    specs: Vec<Spec>,
    /// Per spec: (group key, stream state) pairs.
    states: Vec<Vec<(GroupKey, StreamState)>>,
    /// Rows fed over the bank's life. An optional field (the file is a map),
    /// so a file without it still loads and reports the streams' own count.
    #[serde(default)]
    rows_fed: u64,
    /// Closed rows nobody has drained yet (E54). A map key with a default,
    /// so a file written without one still loads and a bank that closes
    /// nothing writes the same bytes it always did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    closed: Vec<ClosedRow>,
    /// `(spec index, high-water key)` for the specs that have one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    high_water: Vec<(usize, GroupKey)>,
    /// `(spec index, instance, components)` for the PCA sign continuity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pca_prev: Vec<(usize, String, online_core::Pca)>,
}

pub struct Bank {
    specs: Vec<Spec>,
    clock_cfgs: Vec<ClockCfg>,
    derived: Vec<SpecDerived>,
    states: Vec<HashMap<GroupKey, Stream>>,
    /// Rows fed so far. Kept at the bank level because a stream's `rows_seen`
    /// goes with it when its group is dropped.
    rows_fed: u64,
    /// Rows of groups that have closed and not yet been read, oldest first
    /// (docs/ENHANCEMENTS.md E54). Saved with the bank: a driver that saves
    /// between chunks without draining would otherwise lose them silently.
    closed: Vec<ClosedRow>,
    /// Per spec under `group_close = "monotone"`: the largest key seen, past
    /// which no group may reopen.
    high_water: Vec<Option<GroupKey>>,
    /// The last closed row's components per (spec, decay instance), so an
    /// `ew_cov(pca = r)` row's loadings keep their sign across a group and
    /// across a save/load.
    pca_prev: HashMap<(usize, String), online_core::Pca>,
    /// Whether each spec's group column is an integer one, so that
    /// [`key_cmp`] orders its keys as numbers. Learned from the first chunk
    /// a `"monotone"` spec sees and false until then; the queue's key order
    /// reads it.
    key_integer: Vec<bool>,
}

/// Everything `assemble` needs that follows from the `Spec` alone.
///
/// These used to be recomputed per chunk — `decays()` four times, `combos()`
/// three, each re-running the validation that already passed at construction
/// and each allocating (docs/SIMPLIFICATION.md S4). At the default 100k-row
/// chunk that was noise; a caller feeding 1k-row chunks paid it 100x more
/// often. Computed once here, it is not paid at all.
pub struct SpecDerived {
    /// Every output field, in struct order, with the buffer each reads from.
    schema: Vec<FieldMeta>,
    /// Grid-slot labels, for `emit_selected`'s `selected_<t>` column.
    slot_labels: Vec<String>,
    n_models: usize,
    /// Combos per target.
    nc: usize,
    /// Targets.
    m: usize,
    /// Slots one instance owns: `m * nc`, or the statistic count for `ew_cov`.
    per_model: usize,
    /// For a `seqtest` that compares two specs: their indices in the bank,
    /// `(a, b)`. Resolved once in [`Bank::new`], which also checks that the
    /// residual fields the targets name exist on both.
    compare: Option<(usize, usize)>,
}

impl SpecDerived {
    fn new(spec: &Spec) -> Self {
        let schema = output_index(spec);
        let n_models = spec.decays().expect("validated").len();
        let nc = crate::stream::combos(spec).len();
        let m = spec.m();
        // The models with no target prediction have as slots whatever rides
        // in the `pred` buffer (statistics, an assignment and its distances,
        // a class and its posteriors), which the schema says with
        // `Source::Stat` / `Cluster` / `Id` / `Flag` / `Label`.
        let per_model = if spec.model.predicts_no_target() {
            schema
                .iter()
                .filter(|f| {
                    matches!(
                        f.src,
                        Source::Stat(_)
                            | Source::Cluster(_)
                            | Source::Id(_)
                            | Source::Flag(_)
                            | Source::Label(_)
                    )
                })
                .count()
                / n_models
        } else {
            m * nc
        };
        Self {
            schema,
            slot_labels: slot_labels(spec),
            n_models,
            nc,
            m,
            per_model,
            compare: None,
        }
    }
}

/// The residual fields of a spec's output, for a comparison to name.
fn resid_fields(spec: &Spec) -> Vec<String> {
    output_fields(spec)
        .into_iter()
        .filter(|f| f.starts_with("resid_"))
        .collect()
}

/// Resolve a `seqtest` comparison against the bank's specs: `a` and `b`
/// must be specs of the bank, and every target `t` must name a residual
/// field of both (`resid_<t>`, plus the side's grid suffix). The error names
/// what is there instead.
fn resolve_compare(specs: &[Spec], spec: &Spec) -> Result<Option<(usize, usize)>, String> {
    let Some(cmp) = spec.model.compares() else {
        return Ok(None);
    };
    let mut idx = [0; 2];
    for (k, (side, name)) in [("a", cmp.a), ("b", cmp.b)].into_iter().enumerate() {
        let Some(si) = specs.iter().position(|s| s.name == name) else {
            let have: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
            return Err(format!(
                "spec {:?}: seqtest {side} = {name:?} is not a spec of this bank (the bank \
                 has {have:?})",
                spec.name
            ));
        };
        if matches!(specs[si].model, ModelKind::SeqTest { .. }) {
            return Err(format!(
                "spec {:?}: seqtest {side} = {name:?} is itself a seqtest and has no \
                 residuals; compare two specs that predict",
                spec.name
            ));
        }
        let resid = resid_fields(&specs[si]);
        for t in &spec.targets {
            let fields = cmp.fields(t);
            let want = if k == 0 { fields.0 } else { fields.1 };
            if !resid.contains(&want) {
                return Err(format!(
                    "spec {:?}: seqtest target {t:?} names no residual of {side} = {name:?}: \
                     it has no field {want:?} (its residual fields are {resid:?}); a target \
                     is the part after \"resid_\", and {side}_suffix the grid part after \
                     that",
                    spec.name
                ));
            }
        }
        idx[k] = si;
    }
    Ok(Some((idx[0], idx[1])))
}

impl Bank {
    /// A bank over `specs`, each validated and its models built eagerly, so
    /// that every parameter problem is reported here and not on the first
    /// chunk.
    ///
    /// # Errors
    ///
    /// No specs; two with the same name; a spec [`Spec::validate`] refuses
    /// or whose model cannot be built from its parameters -- the message
    /// names the spec and the parameter, as the Python builders' do.
    pub fn new(mut specs: Vec<Spec>) -> Result<Self, String> {
        if specs.is_empty() {
            return Err("at least one spec is required".into());
        }
        // What a spec may leave out, filled before anything reads it, so the
        // bank -- and the state file it writes -- carries the same spec
        // whichever surface wrote it (docs/ENHANCEMENTS.md E53).
        for s in specs.iter_mut() {
            s.fill_defaults();
        }
        let mut names = std::collections::HashSet::new();
        for s in &specs {
            s.validate()?;
            // validate model construction eagerly too
            crate::stream::build_models(s)?;
            if !names.insert(s.name.clone()) {
                return Err(format!("duplicate spec name {:?}", s.name));
            }
            // The rendered field names are the user's handle on every output;
            // a duplicate inside one struct would otherwise surface much later
            // as a confusing polars error. Cannot happen with today's grammar
            // (targets are deduplicated and every suffix applies uniformly),
            // so this is a tripwire for future grammar changes, not a code
            // path with a known trigger.
            let fields = output_fields(s);
            let mut seen_fields = HashSet::with_capacity(fields.len());
            if let Some(dup) = fields.into_iter().find(|f| !seen_fields.insert(f.clone())) {
                return Err(format!(
                    "spec {:?}: two outputs render to the same field name {dup:?}; \
                     rename a target or grid label to disambiguate",
                    s.name
                ));
            }
        }
        let clock_cfgs = specs
            .iter()
            .map(|s| s.clock_cfg())
            .collect::<Result<Vec<_>, _>>()?;
        let derived = specs
            .iter()
            .map(|s| {
                let mut d = SpecDerived::new(s);
                d.compare = resolve_compare(&specs, s)?;
                Ok(d)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let states = specs.iter().map(|_| HashMap::new()).collect();
        let high_water = specs.iter().map(|_| None).collect();
        let key_integer = vec![false; specs.len()];
        Ok(Self {
            specs,
            clock_cfgs,
            derived,
            states,
            rows_fed: 0,
            closed: Vec::new(),
            high_water,
            pca_prev: HashMap::new(),
            key_integer,
        })
    }

    pub fn specs(&self) -> &[Spec] {
        &self.specs
    }

    /// The groups each spec holds state for, sorted by key: `(key, rows
    /// processed, last clock value)`. Rows processed are the group's rows the
    /// null policy did not skip -- the count the stream's own `coef_every`
    /// cadence runs on. A group's state lives until [`Self::drop_groups`]
    /// removes it, so this is how a long-running bank finds the ones that have
    /// gone quiet (docs/IMPROVEMENTS.md U3).
    pub fn groups(&self) -> Vec<Vec<(GroupKey, u64, Option<f64>)>> {
        self.states
            .iter()
            .map(|hm| {
                let mut v: Vec<_> = hm
                    .iter()
                    .map(|(k, s)| (k.clone(), s.rows_seen, s.clock.last_clock()))
                    .collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            })
            .collect()
    }

    /// Forget the state of these groups -- in every spec, or in one -- and
    /// return how many streams were dropped. A dropped group starts cold if it
    /// appears again, exactly as a never-seen one would. `spec` is an index
    /// into [`Self::specs`]; the caller keeps it in range.
    pub fn drop_groups(&mut self, keys: &[GroupKey], spec: Option<usize>) -> Result<usize, String> {
        if let Some(si) = spec {
            if si >= self.states.len() {
                return Err(format!("spec index {si} out of range"));
            }
        }
        let mut dropped = 0;
        for (si, hm) in self.states.iter_mut().enumerate() {
            if spec.is_some_and(|s| s != si) {
                continue;
            }
            for key in keys {
                dropped += usize::from(hm.remove(key).is_some());
            }
        }
        Ok(dropped)
    }

    /// Rows fed so far, over every chunk and group: skipped rows and dropped
    /// groups included, which is why it is not the sum of [`Self::groups`].
    pub fn rows_seen(&self) -> u64 {
        self.rows_fed
    }

    /// Jittered/failed factorizations so far, per spec, as `(group, count)`
    /// pairs sorted by group (docs/PLAN.md §7: a solve never returns NaN
    /// silently, so this is the only way to notice degenerate inputs).
    pub fn solve_failures(&self) -> Vec<Vec<(GroupKey, u64)>> {
        self.states
            .iter()
            .map(|hm| {
                let mut v: Vec<(GroupKey, u64)> = hm
                    .iter()
                    .map(|(k, s)| (k.clone(), s.solve_failures()))
                    .collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            })
            .collect()
    }

    /// The EW accumulators behind a spec's fit, per group and decay instance
    /// (docs/ENHANCEMENTS.md E30).
    ///
    /// Returns one [`Gram`] per (group, instance), sorted by group then by the
    /// spec's halflife order. `spec` is an index into [`Self::specs`].
    ///
    /// The point is that these are the *same* matrices the deployed model
    /// solves against, at any point in the stream, from a single pass over data
    /// that is never materialized — so an analysis built on them cannot
    /// silently disagree with the model it is analysing. What they are not is a
    /// speed claim: for one batch Gram over materialized data, BLAS `dgemm`
    /// wins comfortably.
    ///
    /// Only the models that keep a co-moment matrix report: `ewridge`, `lasso`
    /// and `ew_cov`. Everything else yields an empty vector, because there is
    /// no such matrix to hand back — `rls` and `kalman` track an inverse, and
    /// the gradient models track no second moment at all.
    pub fn gram(&self, spec: usize, group: Option<&str>) -> Result<Vec<Gram>, String> {
        let states = self
            .states
            .get(spec)
            .ok_or_else(|| format!("spec index {spec} out of range"))?;
        let mut keys: Vec<&GroupKey> = match group {
            Some(g) => states.keys().filter(|k| k.as_str() == Some(g)).collect(),
            None => states.keys().collect(),
        };
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            for (label, model) in &states[key].models {
                out.extend(gram_of(key, label, model));
            }
        }
        Ok(out)
    }

    /// The pairs of a `marginal` spec (docs/ENHANCEMENTS.md E44), as a long
    /// frame: one row per (group, decay instance, feature, target), sorted
    /// by group, in spec order within one, with `group`, `instance` (the
    /// halflife-grid suffix, `""` for a single instance), `feature`,
    /// `target`, `n_eff` (the weight behind the target's pairs), `n_kish`
    /// (Kish's effective sample size, `(Σw)²/Σw²`), `mean_x`, `var_x`,
    /// `mean_y`, `var_y`, `cov`, `corr`, `beta` (the slope of the target on
    /// the feature) and `t` (the t-statistic of the correlation at Kish's
    /// `n`), as [`online_core::Marginal::pair`] reads them from the state
    /// as it stands, with the core's NaN -- `corr`, `beta` and `t` below the
    /// target's `min_periods` or undefined, `n_kish` before its first row --
    /// as null.
    ///
    /// `group` narrows the frame to one group; a group the bank has never
    /// seen gives an empty frame, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range, or not a `marginal` spec.
    pub fn marginal(&self, spec: usize, group: Option<&str>) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        if !matches!(s.model, ModelKind::Marginal {}) {
            return Err(format!(
                "spec {:?} has model type {:?}, not \"marginal\"; its pairs are not kept (an \
                 ew_cov's Gram is read with gram())",
                s.name,
                s.model.kind_name()
            ));
        }
        let mut group_col: Vec<Option<&str>> = Vec::new();
        let mut instance: Vec<&str> = Vec::new();
        let mut feature: Vec<&str> = Vec::new();
        let mut target: Vec<&str> = Vec::new();
        let mut pairs: Vec<online_core::MarginalPair> = Vec::new();
        for key in keys {
            for (label, model) in &states[key].models {
                let AnyModel::Marginal(m) = model else {
                    unreachable!("a marginal spec builds marginal models");
                };
                for (t_i, t) in s.targets.iter().enumerate() {
                    for (j, f) in s.features.iter().enumerate() {
                        group_col.push(key.as_str());
                        instance.push(label.as_str());
                        feature.push(f.as_str());
                        target.push(t.as_str());
                        pairs.push(m.pair(t_i, j));
                    }
                }
            }
        }
        // NaN is the core's "undefined" (a constant column, too few rows);
        // the frame says null, as the output structs do.
        let num = |f: fn(&online_core::MarginalPair) -> f64| -> Vec<Option<f64>> {
            pairs
                .iter()
                .map(f)
                .map(|v| (!v.is_nan()).then_some(v))
                .collect()
        };
        let cols = vec![
            Column::new("group".into(), group_col),
            Column::new("instance".into(), instance),
            Column::new("feature".into(), feature),
            Column::new("target".into(), target),
            Column::new("n_eff".into(), num(|p| p.n_eff)),
            Column::new("n_kish".into(), num(|p| p.n_kish)),
            Column::new("mean_x".into(), num(|p| p.mean_x)),
            Column::new("var_x".into(), num(|p| p.var_x)),
            Column::new("mean_y".into(), num(|p| p.mean_y)),
            Column::new("var_y".into(), num(|p| p.var_y)),
            Column::new("cov".into(), num(|p| p.cov)),
            Column::new("corr".into(), num(|p| p.corr)),
            Column::new("beta".into(), num(|p| p.beta)),
            Column::new("t".into(), num(|p| p.t)),
        ];
        DataFrame::new(pairs.len(), cols).map_err(|e| e.to_string())
    }

    /// The coefficients behind a spec's fit, per group and decay instance:
    /// the flat list the output's `coef` field reports, as of the last row
    /// each stream learned from -- `coef` on that row said the same, and
    /// the next row's `pred` is computed from it. The layout is
    /// `polars_online.spec.coef_index`'s: (target x combo) slots, each with
    /// its terms in order. `None` before a stream's first solve, and for a
    /// model without coefficients (`ew_cov`, `seqtest`). The first solve is the solve
    /// schedule's to decide (`solve_every`, `max_rows_between_solves`), not
    /// `min_periods`: `pred` waits for `min_periods`, `coef` does not, so a
    /// row's `n_eff` says how much weight is behind it.
    ///
    /// `group` narrows the list to one group; a group the bank has never
    /// seen gives an empty vector, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range.
    pub fn coef(&self, spec: usize, group: Option<&str>) -> Result<Vec<Coef>, String> {
        let states = self
            .states
            .get(spec)
            .ok_or_else(|| format!("spec index {spec} out of range"))?;
        let mut keys: Vec<&GroupKey> = match group {
            Some(g) => states.keys().filter(|k| k.as_str() == Some(g)).collect(),
            None => states.keys().collect(),
        };
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            for (label, model) in &states[key].models {
                out.push(Coef {
                    group: key.clone(),
                    instance: label.clone(),
                    n_eff: model.n_eff(),
                    coef: model
                        .coefficients()
                        .map(|c| c.into_iter().flatten().collect()),
                });
            }
        }
        Ok(out)
    }

    /// The output struct as it stood on the last row each stream of a spec
    /// learned from, per group (docs/PLAN.md task 34): the sorted group keys
    /// and a struct column with one row for each, in that order -- the row
    /// [`Bank::fit_predict`] reported for that row, field for field,
    /// including `coef` when the row carried it. It travels with the state,
    /// so a bank loaded from a file answers "how was this model doing?"
    /// without the output frame, and a directory of files compares without
    /// keeping the last row of each output. A group with no learned row
    /// yet, or restored from a file written before this existed, is a row
    /// of nulls. [`Bank::predict`] does not move it.
    ///
    /// `group` narrows the list to one group; a group the bank has never
    /// seen gives an empty column, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range.
    pub fn last_row(
        &self,
        spec: usize,
        group: Option<&str>,
    ) -> Result<(Vec<GroupKey>, Column), String> {
        let states = self
            .states
            .get(spec)
            .ok_or_else(|| format!("spec index {spec} out of range"))?;
        let mut keys: Vec<&GroupKey> = match group {
            Some(g) => states.keys().filter(|k| k.as_str() == Some(g)).collect(),
            None => states.keys().collect(),
        };
        keys.sort();
        let (s, d) = (&self.specs[spec], &self.derived[spec]);
        let chunks: Vec<ChunkOut> = keys
            .iter()
            .enumerate()
            .map(|(row, key)| {
                let stream = &states[*key];
                match stream.last_row() {
                    Some(last) => last.to_chunk(s, stream.n_models(), stream.n_slots(), row),
                    // One unprocessed row: every field null.
                    None => {
                        let mut out = ChunkOut::new(s, stream.n_models(), stream.n_slots(), 1);
                        out.rows.push(row);
                        Ok(out)
                    }
                }
            })
            .collect::<Result<_, String>>()?;
        let col = assemble(s, d, keys.len(), &chunks).map_err(|e| e.to_string())?;
        Ok((keys.into_iter().cloned().collect(), col))
    }

    /// What each group of `spec` has been fed (docs/PLAN.md task 35), one
    /// row per group sorted by key: `group`, `rows_fed`, `rows_processed`,
    /// `rows_skipped`, `rows_learned`, `rows_zero_weight`, `weight_sum`,
    /// `clock_min`, `clock_max`, `last_clock`, `session_changes`,
    /// `clock_backwards`, `resets`. The counts are over every row routed to
    /// the group since its state began, undecayed, and survive
    /// [`Bank::save_bytes`] / [`Bank::load_bytes`]; a group restored from a
    /// file written before the summary existed has `group`, `rows_processed`
    /// and `last_clock` and nulls elsewhere. `clock_min`/`clock_max` are
    /// null on a row-count clock. [`Bank::predict`] moves none of it.
    ///
    /// `group` narrows the frame to one group; a group the bank has never
    /// seen gives an empty frame, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range.
    pub fn summary(&self, spec: usize, group: Option<&str>) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let states = &self.states[spec];
        let rows: Vec<SummaryRow<'_>> = keys
            .iter()
            .map(|k| {
                let stream = &states[*k];
                SummaryRow {
                    group: k.as_str(),
                    rows_processed: stream.rows_seen,
                    last_clock: stream.clock.last_clock(),
                    summary: stream.summary(),
                }
            })
            .collect();
        summary_frame(&rows).map_err(|e| e.to_string())
    }

    /// Per-column statistics of what each group of `spec` has been fed
    /// (docs/PLAN.md task 35): one row per (group, input column) in spec
    /// order -- features, then targets, then the weight column -- with
    /// `group`, `column`, `role` (`"feature"`, `"target"`, `"weight"`),
    /// `count`, `null_count`, `mean`, `std` (sample, `ddof = 1`; null below
    /// two values), `min`, `max`. A value counts when finite and within the
    /// input bound, as the models take it, and is a null otherwise. An
    /// unsupervised model's targets are not listed (it has none; the spec's
    /// mirror a feature), an `ew_class` label column has its counts only, a
    /// comparison's targets are named as the spec names them. A group
    /// restored from a file written before the summary existed lists its
    /// columns with every number null.
    ///
    /// `group` narrows the frame to one group; a group the bank has never
    /// seen gives an empty frame, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range.
    pub fn describe(&self, spec: usize, group: Option<&str>) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        let layout = DataSummary::layout(s);
        let keep = |_ci: usize, role: Role| -> Option<bool> {
            match role {
                Role::Target if s.model.is_unsupervised() => None,
                Role::Target if matches!(s.model, ModelKind::EwClass { .. }) => Some(false),
                _ => Some(true),
            }
        };
        let streams: Vec<(Option<&str>, Option<&DataSummary>)> = keys
            .iter()
            .map(|k| (k.as_str(), states[*k].summary()))
            .collect();
        describe_frame(&layout, &keep, &streams).map_err(|e| e.to_string())
    }

    /// The keys of `spec`'s groups, sorted, narrowed to `group` when given.
    fn sorted_keys(&self, spec: usize, group: Option<&str>) -> Result<Vec<&GroupKey>, String> {
        let states = self
            .states
            .get(spec)
            .ok_or_else(|| format!("spec index {spec} out of range"))?;
        let mut keys: Vec<&GroupKey> = match group {
            Some(g) => states.keys().filter(|k| k.as_str() == Some(g)).collect(),
            None => states.keys().collect(),
        };
        keys.sort();
        Ok(keys)
    }

    /// Outputs are attached with `with_column`, which replaces a column of
    /// the same name, so a spec named like an input would silently eat the
    /// input (the Python bank and the CLI runner both attach that way).
    fn refuse_name_clash(&self, df: &DataFrame) -> PolarsResult<()> {
        if let Some(spec) = self
            .specs
            .iter()
            .find(|s| df.get_column_index(&s.name).is_some())
        {
            polars_bail!(Duplicate:
                "spec {:?} has the same name as an input column; the output struct \
                 would replace it. Rename the spec.",
                spec.name
            );
        }
        Ok(())
    }

    /// Run every spec over one chunk; returns one struct column per spec.
    /// Chunks must arrive in stream order within each group.
    ///
    /// # Errors
    ///
    /// Each names the spec and the column: `ColumnNotFound` for a column a
    /// spec reads that the frame has not got (and what it has);
    /// `ComputeError` for a target, feature, clock or weight column that is
    /// not numeric (a temporal clock is refused rather than read as its
    /// epoch integer), a clock with a null or non-finite value, a finite
    /// negative weight, or a group's clock running backwards under
    /// `on_clock_reset = "error"`; `Duplicate` for a spec named like an
    /// input column, which its struct would replace. A refused chunk leaves
    /// the bank exactly as it was -- no state is updated, no new group is
    /// kept -- so the corrected chunk can be fed. And, on the first call
    /// only, a `POLARS_ONLINE_MAX_THREADS` that is not a count of threads.
    pub fn fit_predict(&mut self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        // Everything parallel below -- the `par_iter`s here and the
        // per-instance ones in `Stream` -- runs on the bank's own pool
        // (pool.rs), never on rayon's global one, whichever thread calls.
        crate::pool::pool()?.install(|| self.fit_predict_on_pool(df))
    }

    fn fit_predict_on_pool(&mut self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        // Section timings to stderr when ONLINE_TIMING is set; costs one env
        // read per chunk. This is how docs/PERFORMANCE.md's numbers are made.
        let timing = std::env::var_os("ONLINE_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let n = df.height();
        self.refuse_name_clash(df)?;
        // Independent per spec, and each is a full pass over its columns, so
        // they run in parallel with each other (docs/PERFORMANCE.md P3). The
        // groups come first because they decide the layout the columns are
        // extracted in (P9).
        let groups: Vec<Vec<(GroupKey, Vec<usize>)>> = self
            .specs
            .par_iter()
            .map(|s| group_indices(df, s))
            .collect::<PolarsResult<_>>()?;
        // `group_close = "monotone"` reads the keys in the column's own
        // order, not `GroupKey`'s lexicographic one, and refuses a chunk
        // whose keys are out of order -- before any model is touched, beside
        // the clock check below (E54).
        let mut integer_keys = vec![false; self.specs.len()];
        for (si, spec) in self.specs.iter().enumerate() {
            if spec.closes_monotone() {
                integer_keys[si] = group_key_is_integer(df, spec)?;
                self.key_integer[si] = integer_keys[si];
                check_monotone(
                    spec,
                    &groups[si],
                    n,
                    integer_keys[si],
                    self.high_water[si].as_ref(),
                )?;
            }
        }
        let layouts: Vec<Option<Vec<usize>>> = groups.iter().map(|g| layout_of(g, n)).collect();
        let t_group = t0.elapsed();
        let t1 = std::time::Instant::now();
        let mut cols: Vec<SpecColumns> = self
            .specs
            .par_iter()
            .zip(layouts.par_iter())
            .map(|(s, l)| extract(df, s, false, l.as_deref()))
            .collect::<PolarsResult<_>>()?;
        let t_extract = t1.elapsed();
        let t2 = std::time::Instant::now();

        // Materialize missing streams, then fan out over (spec x group).
        // `fresh` remembers which, so that a refused chunk can take them
        // away again.
        let mut fresh: Vec<(usize, GroupKey)> = Vec::new();
        for (si, spec) in self.specs.iter().enumerate() {
            for (key, _) in &groups[si] {
                if !self.states[si].contains_key(key) {
                    self.states[si].insert(
                        key.clone(),
                        Stream::new(spec).map_err(|e| polars_err!(ComputeError: "{}", e))?,
                    );
                    fresh.push((si, key.clone()));
                }
            }
        }

        // One flat task pool over (spec x group), not a loop of per-spec pools
        // (docs/PERFORMANCE.md P2). A bank of N single-group specs used to run
        // on one core at a time; now every stream in the bank is one task.
        let specs = &self.specs;
        let derived = &self.derived;
        let cfgs = &self.clock_cfgs;
        // Each task owns a disjoint `&mut Stream`, so the borrow checker needs
        // them pulled out of the maps up front. `base` is where the group's
        // run starts in the laid-out columns.
        let mut work: Vec<(usize, &GroupKey, &Vec<usize>, usize, &mut Stream)> = Vec::new();
        for (si, hm) in self.states.iter_mut().enumerate() {
            let mut taken: HashMap<&GroupKey, &mut Stream> = hm.iter_mut().collect();
            let mut base = 0;
            for (key, idx) in &groups[si] {
                let stream = taken.remove(key).expect("stream materialized above");
                work.push((si, key, idx, base, stream));
                base += idx.len();
            }
        }
        // Longest stream first: with a few big groups and many small ones,
        // starting the big ones last leaves cores idle at the tail.
        work.sort_by_key(|(_, _, idx, _, _)| std::cmp::Reverse(idx.len()));

        // Under `on_clock_reset = "error"` the chunk is refused as a whole:
        // every stream checks its clock schedule on a copy before any model is
        // touched (docs/IMPROVEMENTS.md C3), so the bank is left exactly as it
        // was -- the streams the chunk would have created included, so that
        // `groups()` lists what the bank has learned from -- and the corrected
        // chunk can be fed. A no-op under every other policy.
        let checked = work.par_iter().try_for_each(|(si, _, idx, base, stream)| {
            let sc = &cols[*si];
            stream
                .check_clock(
                    &cfgs[*si],
                    sc.clock.as_deref(),
                    sc.session.as_deref(),
                    idx,
                    *base,
                )
                .map_err(|(raw, i)| backwards_clock(&specs[*si], raw, i))
        });
        if let Err(e) = checked {
            drop(work);
            forget(&mut self.states, &fresh);
            return Err(e);
        }

        // Two phases: a `seqtest` that compares two specs reads the residuals
        // they report for this chunk, so those run and assemble first, and
        // the comparisons after (docs/ENHANCEMENTS.md E42). The clock check
        // above covered both, so nothing in either phase can refuse the
        // chunk; the `forget` paths below are the tripwire that keeps that
        // true if a stream ever grows another way to fail.
        let (work2, work1): (Vec<_>, Vec<_>) = work
            .into_iter()
            .partition(|(si, ..)| derived[*si].compare.is_some());
        let mut out: Vec<Option<Column>> = specs.iter().map(|_| None).collect();
        let mut per_spec_rows: Vec<Vec<ChunkOut>> = (0..specs.len()).map(|_| Vec::new()).collect();
        let mut closed: Vec<ClosedRow> = Vec::new();
        let (rows1, closed1) = process(work1, specs, cfgs, &cols);
        closed.extend(closed1);
        for (si, r) in rows1 {
            match r {
                Ok(o) => per_spec_rows[si].push(o),
                Err(e) => {
                    drop(work2);
                    forget(&mut self.states, &fresh);
                    return Err(e);
                }
            }
        }
        let mut t_process = t2.elapsed();
        let t3 = std::time::Instant::now();
        // Specs assemble independently (docs/PERFORMANCE.md P4).
        assemble_phase(specs, derived, n, &per_spec_rows, &mut out, |si| {
            derived[si].compare.is_none()
        })?;
        let mut t_assemble = t3.elapsed();
        if !work2.is_empty() {
            let t4 = std::time::Instant::now();
            for (si, d) in derived.iter().enumerate() {
                if let Some(ab) = d.compare {
                    cols[si].targets =
                        compare_targets(&specs[si], ab, &out, layouts[si].as_deref())?;
                }
            }
            let (rows2, closed2) = process(work2, specs, cfgs, &cols);
            closed.extend(closed2);
            for (si, r) in rows2 {
                match r {
                    Ok(o) => per_spec_rows[si].push(o),
                    Err(e) => {
                        forget(&mut self.states, &fresh);
                        return Err(e);
                    }
                }
            }
            t_process += t4.elapsed();
            let t5 = std::time::Instant::now();
            assemble_phase(specs, derived, n, &per_spec_rows, &mut out, |si| {
                derived[si].compare.is_some()
            })?;
            t_assemble += t5.elapsed();
        }
        // ---- the monotone close batch (E54) ----
        // After both phases and in key order: every group below the chunk's
        // largest key is finished, whatever the chunking, because every row
        // of it precedes the first row of a greater key.
        for si in 0..self.specs.len() {
            if !self.specs[si].closes_monotone() || groups[si].is_empty() {
                continue;
            }
            let integer = integer_keys[si];
            let starts: Vec<(GroupKey, usize)> = groups[si]
                .iter()
                .map(|(k, idx)| (k.clone(), idx[0]))
                .collect();
            let max_key = starts.last().expect("the chunk has a group").0.clone();
            let mut closing: Vec<GroupKey> = self.states[si]
                .keys()
                .filter(|k| key_cmp(k, &max_key, integer) == std::cmp::Ordering::Less)
                .cloned()
                .collect();
            closing.sort_by(|a, b| key_cmp(a, b, integer));
            for key in closing {
                // The closing row is the first row of the first greater key
                // in this chunk; a group closed without a row of its own here
                // closes at the chunk's first row.
                let at = starts.partition_point(|(k, _)| {
                    key_cmp(k, &key, integer) != std::cmp::Ordering::Greater
                });
                let at = starts.get(at).map_or(0, |(_, r)| *r);
                let stream = self.states[si].remove(&key).expect("key came from the map");
                closed.extend(close_rows(si, &self.specs[si], &key, &stream, at));
            }
            self.high_water[si] = Some(max_key);
        }
        self.queue_closed(closed);

        if timing {
            let total = t0.elapsed();
            eprintln!(
                "ONLINE_TIMING rows={n} extract={:.1}ms group={:.1}ms process={:.1}ms \
                 assemble={:.1}ms total={:.1}ms ({:.0} rows/s)",
                t_extract.as_secs_f64() * 1e3,
                t_group.as_secs_f64() * 1e3,
                t_process.as_secs_f64() * 1e3,
                t_assemble.as_secs_f64() * 1e3,
                total.as_secs_f64() * 1e3,
                n as f64 / total.as_secs_f64()
            );
        }
        self.rows_fed += n as u64;
        Ok(out
            .into_iter()
            .map(|c| c.expect("every spec is assembled in one of the two phases"))
            .collect())
    }

    /// Score one chunk against the bank as it stands, learning nothing
    /// (docs/ENHANCEMENTS.md E31): for every row, the struct
    /// [`Self::fit_predict`] would produce for it as the next row of its
    /// group's stream. The bank is not touched -- not the models, the clocks,
    /// the diagnostics or the row counts -- so the call is `&self` and any
    /// number of them can run at once, and row order does not matter.
    ///
    /// The frame needs each spec's feature and clock columns. A target
    /// column is optional and yields `resid` where present; the session
    /// column is optional and feeds `session_gap`; a weight column is not
    /// read. Rows of a group the bank has never seen are null throughout,
    /// like a skipped row. Per field, see [`Stream::predict_chunk`].
    ///
    /// # Errors
    ///
    /// As [`Self::fit_predict`]'s, less a missing target, which is not one.
    pub fn predict(&self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        crate::pool::pool()?.install(|| self.predict_on_pool(df))
    }

    fn predict_on_pool(&self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        let n = df.height();
        self.refuse_name_clash(df)?;
        let groups: Vec<Vec<(GroupKey, Vec<usize>)>> = self
            .specs
            .par_iter()
            .map(|s| group_indices(df, s))
            .collect::<PolarsResult<_>>()?;
        let layouts: Vec<Option<Vec<usize>>> = groups.iter().map(|g| layout_of(g, n)).collect();
        let mut cols: Vec<SpecColumns> = self
            .specs
            .par_iter()
            .zip(layouts.par_iter())
            .map(|(s, l)| extract(df, s, true, l.as_deref()))
            .collect::<PolarsResult<_>>()?;

        let specs = &self.specs;
        let derived = &self.derived;
        let cfgs = &self.clock_cfgs;
        let mut work: Vec<(usize, &Vec<usize>, usize, &Stream)> = Vec::new();
        for (si, hm) in self.states.iter().enumerate() {
            let mut base = 0;
            for (key, idx) in &groups[si] {
                if let Some(stream) = hm.get(key) {
                    work.push((si, idx, base, stream));
                }
                base += idx.len();
            }
        }
        work.sort_by_key(|(_, idx, _, _)| std::cmp::Reverse(idx.len()));
        // The same two phases as `fit_predict`: a comparison scores the
        // residuals its two sides report for this chunk.
        let (work2, work1): (Vec<_>, Vec<_>) = work
            .into_iter()
            .partition(|(si, ..)| derived[*si].compare.is_some());
        let mut out: Vec<Option<Column>> = specs.iter().map(|_| None).collect();
        let mut per_spec_rows: Vec<Vec<ChunkOut>> = (0..specs.len()).map(|_| Vec::new()).collect();
        for (si, r) in score(work1, specs, cfgs, &cols) {
            per_spec_rows[si].push(r?);
        }
        assemble_phase(specs, derived, n, &per_spec_rows, &mut out, |si| {
            derived[si].compare.is_none()
        })?;
        if !work2.is_empty() {
            for (si, d) in derived.iter().enumerate() {
                if let Some(ab) = d.compare {
                    cols[si].targets =
                        compare_targets(&specs[si], ab, &out, layouts[si].as_deref())?;
                }
            }
            for (si, r) in score(work2, specs, cfgs, &cols) {
                per_spec_rows[si].push(r?);
            }
            assemble_phase(specs, derived, n, &per_spec_rows, &mut out, |si| {
                derived[si].compare.is_some()
            })?;
        }
        Ok(out
            .into_iter()
            .map(|c| c.expect("every spec is assembled in one of the two phases"))
            .collect())
    }

    /// Append a chunk's closed rows to the queue in the one order row order
    /// implies: by the row that closed them, then by key, then by instance
    /// (docs/ENHANCEMENTS.md E54). Streams are processed in parallel, so
    /// this sort is what makes the queue a function of the data alone.
    ///
    /// The `ew_cov(pca = r)` block is computed here rather than at the
    /// close, for the same reason: the loadings are signed for continuity
    /// with the previous closed row of the same (spec, instance), and
    /// "previous" has to mean previous in the queue.
    fn queue_closed(&mut self, mut rows: Vec<ClosedRow>) {
        if rows.is_empty() {
            return;
        }
        rows.sort_by(|a, b| {
            // Only rows of one spec ever reach the key comparison, the tuple
            // above having separated the specs, so one spec's flag decides.
            let integer = self.key_integer[a.spec];
            (a.at, a.spec)
                .cmp(&(b.at, b.spec))
                .then_with(|| key_cmp(&a.group, &b.group, integer))
                .then_with(|| a.instance.cmp(&b.instance))
        });
        for row in rows.iter_mut() {
            let ModelKind::EwCov { pca: Some(r), .. } = self.specs[row.spec].model else {
                continue;
            };
            let Some(g) = &row.gram else { continue };
            let key = (row.spec, row.instance.clone());
            let prev = self.pca_prev.get(&key);
            if let Some(p) = online_core::Pca::of(&g.comoments, g.k, r, prev) {
                row.eig_vals = Some(p.eig.clone());
                row.eig_vecs = Some(p.loadings.clone());
                self.pca_prev.insert(key, p);
            }
        }
        self.closed.extend(rows);
    }

    /// The groups that have closed and not been read (docs/ENHANCEMENTS.md
    /// E54), oldest first, as one long frame; see [`ClosedRow`] and
    /// `docs/PLAN.md` §11a for the schema. `spec` narrows it to one spec's
    /// rows. `drop` removes what it returns from the queue -- the ordinary
    /// use, a drain per chunk -- and `false` peeks at it.
    ///
    /// A bank whose specs have no `group_close` never queues anything and
    /// always answers with an empty frame of the schema.
    ///
    /// # Errors
    ///
    /// `spec` out of range, or a frame the columns cannot make.
    pub fn closed_groups(&mut self, spec: Option<usize>, drop: bool) -> Result<DataFrame, String> {
        if let Some(si) = spec {
            if si >= self.specs.len() {
                return Err(format!("spec index {si} out of range"));
            }
        }
        let taken: Vec<ClosedRow> = if drop {
            let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.closed)
                .into_iter()
                .partition(|r| spec.is_none_or(|si| r.spec == si));
            self.closed = rest;
            mine
        } else {
            self.closed
                .iter()
                .filter(|r| spec.is_none_or(|si| r.spec == si))
                .cloned()
                .collect()
        };
        closed_frame(&self.specs, &taken).map_err(|e| e.to_string())
    }

    /// The bank as versioned msgpack: the specs, every group's state and the
    /// row count, behind a magic string and two version numbers
    /// (`BANK_FORMAT_VERSION` for this envelope, `online_core::SCHEMA_VERSION`
    /// for the states). Fails only if serialization does, which a bank
    /// built by this crate cannot make happen.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let file = BankFile {
            magic: BANK_MAGIC.to_string(),
            format_version: BANK_FORMAT_VERSION,
            rows_fed: self.rows_fed,
            schema_version: online_core::SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            specs: self.specs.clone(),
            states: self
                .states
                .iter()
                .map(|hm| {
                    let mut v: Vec<(GroupKey, StreamState)> =
                        hm.iter().map(|(k, s)| (k.clone(), s.save())).collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                })
                .collect(),
            closed: self.closed.clone(),
            high_water: self
                .high_water
                .iter()
                .enumerate()
                .filter_map(|(si, k)| k.clone().map(|k| (si, k)))
                .collect(),
            pca_prev: {
                let mut v: Vec<(usize, String, online_core::Pca)> = self
                    .pca_prev
                    .iter()
                    .map(|((si, inst), p)| (*si, inst.clone(), p.clone()))
                    .collect();
                v.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
                v
            },
        };
        rmp_serde::to_vec_named(&file).map_err(|e| e.to_string())
    }

    /// A bank from what [`Bank::save_bytes`] wrote, on this or any other OS.
    /// Refused, with the reason: bytes that are not a bank file (with
    /// `rmp_serde`'s message when they do not even parse), a file written by
    /// a newer build of this crate (envelope version) or of `online-core`
    /// (schema version), and, when `expected_specs` is given, specs that
    /// differ from the file's -- the state would not be the state of the bank
    /// the caller has in mind.
    pub fn load_bytes(bytes: &[u8], expected_specs: Option<&[Spec]>) -> Result<Self, String> {
        let header: BankHeader = rmp_serde::from_slice(bytes)
            .map_err(|e| format!("not a polars-online bank state file ({e})"))?;
        if header.magic != BANK_MAGIC {
            return Err("not a polars-online bank state file".into());
        }
        if header.format_version > BANK_FORMAT_VERSION {
            return Err(format!(
                "bank state file format version {} is newer than this build supports ({})",
                header.format_version, BANK_FORMAT_VERSION
            ));
        }
        if !(online_core::MIN_SCHEMA_VERSION..=online_core::SCHEMA_VERSION)
            .contains(&header.schema_version)
        {
            return Err(format!(
                "state schema version {} not supported (this build loads {}..={})",
                header.schema_version,
                online_core::MIN_SCHEMA_VERSION,
                online_core::SCHEMA_VERSION
            ));
        }
        let file: BankFile = rmp_serde::from_slice(bytes)
            .map_err(|e| format!("not a polars-online bank state file ({e})"))?;
        if let Some(exp) = expected_specs {
            // Both sides filled, since filling is what a bank does to its
            // specs before it runs them: a caller who wrote the spec in TOML
            // without a target for an unsupervised model, or without
            // `drift_action`, is handing over the same spec (E53) -- and a
            // file written before the filling existed carries the unfilled
            // form.
            let fill = |s: &Spec| {
                let mut s = s.clone();
                s.fill_defaults();
                s
            };
            let exp: Vec<Spec> = exp.iter().map(fill).collect();
            let saved: Vec<Spec> = file.specs.iter().map(fill).collect();
            if exp != saved {
                return Err("saved specs do not match the bank's specs; refusing to load".into());
            }
        }
        let mut bank = Bank::new(file.specs.clone())?;
        for (si, groups) in file.states.iter().enumerate() {
            for (key, st) in groups {
                let stream = Stream::restore(&file.specs[si], st)?;
                bank.states[si].insert(key.clone(), stream);
            }
        }
        // A file from before the counter existed: every spec sees every row,
        // so the first spec's streams have the count, less any dropped group
        // and less the rows the null policy skipped.
        bank.closed = file.closed.clone();
        for (si, key) in &file.high_water {
            // A key stored under a column dtype the bank no longer has is a
            // different key: `"10"` orders after `"9"` as an integer and
            // before it as a string, so the mark would let through exactly
            // the groups it exists to refuse.
            if let Some(slot) = bank.high_water.get_mut(*si) {
                *slot = Some(key.clone());
            }
        }
        for (si, inst, pca) in &file.pca_prev {
            bank.pca_prev.insert((*si, inst.clone()), pca.clone());
        }
        bank.rows_fed = if file.rows_fed > 0 {
            file.rows_fed
        } else {
            bank.states
                .first()
                .map_or(0, |hm| hm.values().map(|s| s.rows_seen).sum())
        };
        Ok(bank)
    }

    /// Write the state to `path`, atomically: a temporary sibling, then a
    /// rename over the destination (`crate::atomic`). An interrupted save
    /// used to leave a truncated file and take the last good state with it,
    /// which is a resume loop starting the stream over.
    ///
    /// The error is the filesystem's, without the path: the caller names it,
    /// and keeps the kind (the Python side raises `FileNotFoundError` for a
    /// directory that is not there, `PermissionError` for one it cannot
    /// write, ... from it).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = self.save_bytes().map_err(std::io::Error::other)?;
        crate::atomic::write(path, &bytes)
    }

    /// [`Bank::load_bytes`] over a file. The error is a message, whichever
    /// step failed; a caller who needs to tell the two apart -- a file that
    /// could not be read from a file that is not a bank -- does the read
    /// itself (`RunConfig::open_bank`, the Python `ModelBank.load`).
    pub fn load(path: &Path, expected_specs: Option<&[Spec]>) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::load_bytes(&bytes, expected_specs)
    }
}

/// `ew_cov` output: one f64 column per statistic slot, plus `n_eff`.
/// One output field with the machine values its name encodes.
///
/// This is the antidote to string formatting as API: a caller filters this
/// table for `kind == "pred" && target == "y" && ridge == Some(0.5)` instead
/// of constructing `"pred_y__r0.5@h500"` by hand — which would require
/// reimplementing `num_label`'s float rendering. Produced by the same code
/// that renders the names, so the two cannot drift.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldMeta {
    pub field: String,
    /// pred / resid / sigma / resid_z / ic / r2 / hit_rate / absresid_q /
    /// autocorr / drift / n_eff / coef / lam_selected / selected /
    /// pred_selected / pred_averaged — or an `ew_cov` statistic name.
    pub kind: String,
    pub target: Option<String>,
    /// The instance's decay, as configured (present even when the suffix is
    /// empty because there is a single instance).
    pub halflife: Option<f64>,
    pub lam: Option<f64>,
    pub ridge: Option<f64>,
    pub feature_set: Option<String>,
    /// Lasso path point.
    pub lambda: Option<f64>,
    /// Quantile level (`absresid_q*` fields).
    pub quantile: Option<f64>,
    /// Lag, in learned rows, of an `ew_cov` `lagcorr_*` field
    /// (docs/ENHANCEMENTS.md E56); `None` for every other field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
    /// Columns an `ew_cov` statistic is over.
    pub columns: Option<Vec<String>>,
    /// The polars dtype the field is materialized with, as its string form
    /// (`f64`, `bool`, `str`, `list[f64]`). Set from `src`, so it is the same
    /// table `assemble` fills the buffers from; the expression plugin declares
    /// its output struct from [`FieldMeta::dtype`] (docs/IMPROVEMENTS.md C1 —
    /// a name-prefix guess there once declared `drift_*` as `f64` while the
    /// bank produced `bool`, and polars refused the struct).
    pub dtype: String,
    /// Which assembled buffer, and where in it, this field's values come from.
    /// Private and not serialized: it is how `assemble` walks this schema
    /// instead of rebuilding the same nested loops with its own `format!`
    /// calls (docs/SIMPLIFICATION.md S1).
    #[serde(skip)]
    src: Source,
}

/// The buffer a field's values are scattered into, with the index into it.
/// One variant per buffer `assemble` allocates.
#[derive(Debug, Clone, Default, PartialEq)]
enum Source {
    /// Assigned before the field is pushed; never observed.
    #[default]
    Unset,
    Pred(usize),
    Resid(usize),
    Sigma(usize),
    ResidZ(usize),
    Drift(usize),
    /// `(which of ic/r2/hit_rate, index)`.
    Metric(usize, usize),
    /// `(which of lo/hi/coverage, index)`, laid out like `Metric`.
    Conformal(usize, usize),
    Quantile(usize),
    Autocorr(usize),
    NEff(usize),
    Coef(usize),
    LamSelected(usize),
    SelPred(usize),
    SelName(usize),
    AvgPred(usize),
    /// An `ew_cov` statistic, or a `kmeans` / `micro` distance, which rides
    /// in the `pred` buffer.
    Stat(usize),
    /// A `kmeans` assignment, or a `micro` count: the `pred` buffer holds a
    /// small non-negative integer as an f64 (NaN = null), materialized as
    /// `i32`.
    Cluster(usize),
    /// A `micro` id or label, or a `seqtest` count: monotone and never
    /// reused, so `i64`, the same way.
    Id(usize),
    /// A `micro` flag: `1.0` / `0.0` in the `pred` buffer (NaN = null),
    /// materialized as `Boolean`.
    Flag(usize),
    /// An `ew_class` prediction: the class's position in `classes` as an
    /// f64 in the `pred` buffer (NaN = null), materialized as the class name.
    Label(usize),
}

impl FieldMeta {
    fn new(field: String, kind: &str) -> Self {
        Self {
            field,
            kind: kind.to_string(),
            target: None,
            halflife: None,
            lam: None,
            ridge: None,
            feature_set: None,
            lambda: None,
            quantile: None,
            lag: None,
            columns: None,
            dtype: String::new(),
            src: Source::Unset,
        }
    }
    fn src(mut self, src: Source) -> Self {
        self.src = src;
        self.dtype = self.dtype().to_string();
        self
    }

    /// The dtype `assemble` materializes this field with.
    pub fn dtype(&self) -> DataType {
        match self.src {
            Source::Drift(_) => DataType::Boolean,
            Source::SelName(_) => DataType::String,
            Source::Coef(_) => DataType::List(Box::new(DataType::Float64)),
            Source::Cluster(_) => DataType::Int32,
            Source::Id(_) => DataType::Int64,
            Source::Flag(_) => DataType::Boolean,
            Source::Label(_) => DataType::String,
            Source::Unset => unreachable!("every field is given a source in output_index"),
            _ => DataType::Float64,
        }
    }
    fn decay(mut self, d: &online_core::Decay) -> Self {
        match d {
            online_core::Decay::Halflife(h) => self.halflife = Some(*h),
            online_core::Decay::Lam(l) => self.lam = Some(*l),
        }
        self
    }
    fn target(mut self, t: &str) -> Self {
        self.target = Some(t.to_string());
        self
    }
    fn combo(mut self, c: &crate::stream::Combo) -> Self {
        self.ridge = c.ridge;
        self.feature_set = c.feature_set.clone();
        self.lambda = c.lambda;
        self
    }
}

/// Every coefficient a spec reports, in `coef` list order, with the name
/// [`CoefField`] gives it. Empty for `ew_cov` and `seqtest`, which have
/// none, and for `micro`, whose `coef` is one `[id, label, n, radius, c_1 ..
/// c_p]` row per potential summary -- as many as there are, so no position
/// has a name.
///
/// The layout is the models' (`online_core`): per instance, one `coef` list
/// holding `(target, combo)` slots in the order the `pred` fields declare
/// them, each slot the full term vector -- `intercept` first when the spec
/// has one, then every feature, zeros for features a feature set leaves out
/// (`EwRidgeModel::solve` scatters each combo's solution into `k_total`
/// columns) -- or `level`, `trend` for `holt`. Rendered here, beside the
/// field names, from the same combos and suffixes, so the two cannot drift.
///
/// `kmeans` reports its centres here: `k` slots named `cluster{j}` in place
/// of the targets, each the centre's coordinate per feature, so
/// `coef_cluster0_x1` is centre 0's `x1` and `coef_index` lays the list out
/// as `(cluster, feature)`. `ew_class` does the same with its class means:
/// one slot per class, named by the class, so `coef_a_x1` is class `a`'s
/// mean of `x1`.
pub fn coef_fields(spec: &Spec) -> Vec<CoefField> {
    if matches!(
        spec.model,
        crate::ModelKind::EwCov { .. }
            | crate::ModelKind::Micro { .. }
            | crate::ModelKind::SeqTest { .. }
            | crate::ModelKind::Marginal {}
            | crate::ModelKind::Rcov { .. }
            | crate::ModelKind::CorrChange { .. }
            | crate::ModelKind::Bocpd { .. }
    ) {
        return Vec::new();
    }
    // deco's `coef` is the correlation values themselves, one slot each,
    // named for the block or the pair of blocks they belong to.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = online_core::Deco::labels(&names);
        let values: Vec<&String> = labels
            .iter()
            .filter(|l| l.as_str() == "rho" || l.starts_with("rho_"))
            .collect();
        let mut out = Vec::new();
        for (suffix, d) in spec.decays().expect("validated") {
            for (position, l) in values.iter().enumerate() {
                let slot = l.strip_prefix("rho_").unwrap_or("rho");
                out.push(CoefField {
                    field: format!("coef{suffix}"),
                    position,
                    name: format!("coef_{slot}{suffix}"),
                    target: slot.to_string(),
                    halflife: match d {
                        online_core::Decay::Halflife(h) => Some(h),
                        online_core::Decay::Lam(_) => None,
                    },
                    lam: match d {
                        online_core::Decay::Lam(l) => Some(l),
                        online_core::Decay::Halflife(_) => None,
                    },
                    ridge: None,
                    feature_set: None,
                    lambda: None,
                    term: "rho".to_string(),
                });
            }
        }
        return out;
    }
    let slots: Vec<String> = match &spec.model {
        crate::ModelKind::KMeans { k, .. } => (0..*k).map(|j| format!("cluster{j}")).collect(),
        // One slot per hidden state, named as `kmeans` names its centres.
        crate::ModelKind::Hmm { k, .. } => (0..*k).map(|j| format!("state{j}")).collect(),
        crate::ModelKind::EwClass { classes, .. } => classes.clone(),
        _ => spec.targets.clone(),
    };
    let terms: Vec<String> = if matches!(spec.model, crate::ModelKind::Holt { .. }) {
        vec!["level".into(), "trend".into()]
    } else if matches!(
        spec.model,
        crate::ModelKind::KMeans { .. }
            | crate::ModelKind::EwClass { .. }
            | crate::ModelKind::Hmm { .. }
    ) {
        spec.features.clone()
    } else {
        let mut t = Vec::with_capacity(spec.features.len() + 1);
        if spec.add_intercept {
            t.push("intercept".to_string());
        }
        t.extend(spec.features.iter().cloned());
        t
    };
    let combos = crate::stream::combos(spec);
    let decays = spec.decays().expect("validated");
    let mut out = Vec::new();
    for (suffix, d) in &decays {
        let (halflife, lam) = match d {
            online_core::Decay::Halflife(h) => (Some(*h), None),
            online_core::Decay::Lam(l) => (None, Some(*l)),
        };
        let mut position = 0;
        for t in &slots {
            for c in &combos {
                for term in &terms {
                    out.push(CoefField {
                        field: format!("coef{suffix}"),
                        position,
                        name: format!("coef_{t}_{term}{}{suffix}", c.label),
                        target: t.clone(),
                        halflife,
                        lam,
                        ridge: c.ridge,
                        feature_set: c.feature_set.clone(),
                        lambda: c.lambda,
                        term: term.clone(),
                    });
                    position += 1;
                }
            }
        }
    }
    out
}

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
    output_index(spec).into_iter().map(|m| m.field).collect()
}

/// Every output field with its metadata, in struct order.
pub fn output_index(spec: &Spec) -> Vec<FieldMeta> {
    let decays = spec.decays().expect("validated");
    // deco is not a regression either: its slots are the row's own
    // equicorrelation estimate, the level before the row and the row's
    // log-density under it -- one of each unblocked, one `u_*` and one
    // `rho_*` per value and a single `loglik` with blocks.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = online_core::Deco::labels(&names);
        let n_slots = labels.len();
        // The columns a slot is over: its block, its pair of blocks, or all
        // of them for `loglik` and the unblocked scalars.
        let blocks = crate::stream::deco_cfg(spec).expect("validated");
        let all = spec.features.clone();
        let cols_of = |slot: usize| -> Vec<String> {
            if names.is_empty() {
                return all.clone();
            }
            let k = names.len();
            let m = k + k * (k - 1) / 2;
            if slot >= 2 * m {
                return all.clone();
            }
            let v = slot % m;
            let of = |b: usize| -> Vec<String> {
                blocks.blocks[b].iter().map(|&i| all[i].clone()).collect()
            };
            if v < k {
                return of(v);
            }
            let mut p = k;
            for i in 0..k {
                for j in (i + 1)..k {
                    if p == v {
                        let mut cols = of(i);
                        cols.extend(of(j));
                        return cols;
                    }
                    p += 1;
                }
            }
            all.clone()
        };
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                // The kind is the label without its block suffix, so
                // `u_tech` and `u_banks` are both `u` to a reader of the
                // index.
                let kind = l.split('_').next().unwrap_or(l).to_string();
                let mut m = FieldMeta::new(format!("{l}{suffix}"), &kind)
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols_of(slot));
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // ew_cov is not a regression: its slots are named statistics, not
    // pred/resid pairs, and it has no targets or coefficients.
    if let crate::ModelKind::EwCov {
        stats,
        mahal_quantiles,
        pca,
        lags,
        ..
    } = &spec.model
    {
        let names = stats
            .clone()
            .unwrap_or_else(|| vec!["mean".into(), "std".into(), "corr".into()]);
        let kinds: Vec<online_core::EwCovStat> = names
            .iter()
            .map(|s| match s.as_str() {
                "mean" => online_core::EwCovStat::Mean,
                "var" => online_core::EwCovStat::Var,
                "std" => online_core::EwCovStat::Std,
                "cov" => online_core::EwCovStat::Cov,
                "partial_corr" => online_core::EwCovStat::PartialCorr,
                "mahal" => online_core::EwCovStat::Mahal,
                "lagcorr" => online_core::EwCovStat::LagCorr,
                _ => online_core::EwCovStat::Corr,
            })
            .collect();
        let levels: Vec<f64> = mahal_quantiles.clone().unwrap_or_default();
        let r = pca.unwrap_or(0);
        let lags: Vec<usize> = lags.clone().unwrap_or_default();
        let labels = online_core::EwCovModel::labels(&spec.features, &kinds, &levels, r, &lags);
        // Statistic kind, the columns it is over and its quantile level, in
        // label order: the same walk `labels` makes (per stat: each column,
        // each i<j pair, or all of them; then the levels; then `k + 3` per
        // component, all over every column).
        let all = spec.features.clone();
        // Statistic kind, the columns it is over, its quantile level and its
        // lag, in label order.
        type StatMeta = (String, Vec<String>, Option<f64>, Option<usize>);
        let mut meta: Vec<StatMeta> = Vec::new();
        for (name, kind) in names.iter().zip(&kinds) {
            match kind {
                online_core::EwCovStat::Mean
                | online_core::EwCovStat::Var
                | online_core::EwCovStat::Std => {
                    for col in &spec.features {
                        meta.push((name.clone(), vec![col.clone()], None, None));
                    }
                }
                online_core::EwCovStat::Mahal => meta.push((name.clone(), all.clone(), None, None)),
                // One entry per (lag, ordered pair), the auto terms
                // included: both orientations, since the lagged matrix is
                // not symmetric.
                online_core::EwCovStat::LagCorr => {
                    for &l in &lags {
                        for a in &spec.features {
                            for b in &spec.features {
                                meta.push((
                                    name.clone(),
                                    vec![a.clone(), b.clone()],
                                    None,
                                    Some(l),
                                ));
                            }
                        }
                    }
                }
                _ => {
                    for i in 0..spec.features.len() {
                        for j in (i + 1)..spec.features.len() {
                            meta.push((
                                name.clone(),
                                vec![spec.features[i].clone(), spec.features[j].clone()],
                                None,
                                None,
                            ));
                        }
                    }
                }
            }
        }
        for &q in &levels {
            meta.push(("mahal_q".into(), all.clone(), Some(q), None));
        }
        for _ in 0..r {
            meta.push(("pc_var".into(), all.clone(), None, None));
            meta.push(("pc_share".into(), all.clone(), None, None));
            for col in &spec.features {
                meta.push(("pc_loading".into(), vec![col.clone()], None, None));
            }
            meta.push(("pc_score".into(), all.clone(), None, None));
        }
        debug_assert_eq!(meta.len(), labels.len());
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, (l, (kind, cols, q, lag))) in labels.iter().zip(&meta).enumerate() {
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols.clone());
                m.quantile = *q;
                m.lag = *lag;
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // kmeans is not a regression either: per instance, the nearest centre
    // and two distances (read before the row is learned), `n_eff`, and the
    // centres as `coef`.
    if matches!(spec.model, crate::ModelKind::KMeans { .. }) {
        let n_slots = 3;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(d)
                    .src(Source::Cluster(mi * n_slots)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + 1)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist2{suffix}"), "dist2")
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + 2)),
            ));
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // micro: per instance, the nearest cluster's label and distance, the
    // micro-cluster id the row goes to, an outlier flag, two counts (all
    // read before the row is learned), `n_eff`, and the potential summaries
    // as `coef`.
    if matches!(spec.model, crate::ModelKind::Micro { .. }) {
        let n_slots = 6;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            let at = |slot: usize| mi * n_slots + slot;
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(d)
                    .src(Source::Id(at(0))),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(d)
                    .src(Source::Stat(at(1))),
            ));
            fields.push(over(
                FieldMeta::new(format!("micro{suffix}"), "micro")
                    .decay(d)
                    .src(Source::Id(at(2))),
            ));
            fields.push(over(
                FieldMeta::new(format!("outlier{suffix}"), "outlier")
                    .decay(d)
                    .src(Source::Flag(at(3))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_clusters{suffix}"), "n_clusters")
                    .decay(d)
                    .src(Source::Cluster(at(4))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_micro{suffix}"), "n_micro")
                    .decay(d)
                    .src(Source::Cluster(at(5))),
            ));
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // bocpd's value is a posterior over run lengths: per instance, the
    // changepoint mass, the run-length mode and mean, a predictive mean per
    // column and the row's log score.
    if matches!(spec.model, crate::ModelKind::Bocpd { .. }) {
        let labels = online_core::Bocpd::labels(&spec.features);
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = if l == "run_mode" {
                    Source::Id(at)
                } else {
                    Source::Stat(at)
                };
                let kind = l.split('_').next().unwrap_or(l);
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(d)
                    .src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // corrchange is a test, not a model of the data: per instance the
    // statistic, its critical value, the flag and the rows since the last
    // one. Nothing is reported except where a span closes.
    if matches!(spec.model, crate::ModelKind::CorrChange { .. }) {
        let labels = online_core::CorrChange::labels();
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = match l.as_str() {
                    "flag" => Source::Flag(at),
                    "since_flag" => Source::Id(at),
                    _ => Source::Stat(at),
                };
                let mut m = FieldMeta::new(format!("{l}{suffix}"), l).decay(d).src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // hmm's state is hidden: per instance, the filtered and predicted
    // posteriors, the state, the row's log-likelihood, `n_eff` and the
    // state means as `coef`.
    if let crate::ModelKind::Hmm { k, .. } = &spec.model {
        let labels = online_core::Hmm::labels(*k);
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                // `state` is a small count, so it rides in `pred` and comes
                // out as an `i32`.
                let src = if l == "state" {
                    Source::Cluster(at)
                } else {
                    Source::Stat(at)
                };
                let mut m =
                    FieldMeta::new(format!("{l}{suffix}"), l.split('_').next().unwrap_or(l))
                        .decay(d)
                        .src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // ew_class predicts a label, not a number: per instance, the class the
    // row is assigned to and one posterior per class (read before the row
    // is learned), `n_eff`, and the class means as `coef`.
    if let crate::ModelKind::EwClass { classes, .. } = &spec.model {
        let n_slots = 1 + classes.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("class{suffix}"), "class")
                    .decay(d)
                    .target(&spec.targets[0])
                    .src(Source::Label(mi * n_slots)),
            ));
            for (c, class) in classes.iter().enumerate() {
                fields.push(over(
                    FieldMeta::new(format!("p_{class}{suffix}"), "p")
                        .decay(d)
                        .target(&spec.targets[0])
                        .src(Source::Stat(mi * n_slots + 1 + c)),
                ));
            }
            fields.push(
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // seqtest predicts nothing: per target, the two log e-values and the two
    // sign counts they are staked on (read before the row is learned), then
    // `n_eff`. One undecayed instance, so no suffix and no decay on the
    // fields; no `coef`. A comparison names its fields by the two sides.
    if let crate::ModelKind::SeqTest { .. } = &spec.model {
        let n_slots = online_core::SEQTEST_SLOTS;
        let names: [&str; 4] = if spec.model.compares().is_some() {
            ["log_e_a", "log_e_b", "wins_a", "wins_b"]
        } else {
            ["log_e_pos", "log_e_neg", "n_pos", "n_neg"]
        };
        let mut fields = Vec::new();
        for (t_i, t) in spec.targets.iter().enumerate() {
            let at = |slot: usize| t_i * n_slots + slot;
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[0]), names[0])
                    .target(t)
                    .src(Source::Stat(at(0))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[1]), names[1])
                    .target(t)
                    .src(Source::Stat(at(1))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[2]), names[2])
                    .target(t)
                    .src(Source::Id(at(2))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[3]), names[3])
                    .target(t)
                    .src(Source::Id(at(3))),
            );
        }
        fields.push(FieldMeta::new("n_eff".into(), "n_eff").src(Source::NEff(0)));
        return fields;
    }
    // rcov and marginal emit nothing per row but `n_eff`, one per instance:
    // rcov's value is the block it emits at the group's close, marginal's
    // are the pairs `Bank::marginal` reads from the state.
    if matches!(
        spec.model,
        crate::ModelKind::Marginal {} | crate::ModelKind::Rcov { .. }
    ) {
        return decays
            .iter()
            .enumerate()
            .map(|(mi, (suffix, d))| {
                FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                    .decay(d)
                    .src(Source::NEff(mi))
            })
            .collect();
    }
    let combos = crate::stream::combos(spec);
    let (nc, m, n_models) = (combos.len(), spec.m(), decays.len());
    let mut fields = Vec::new();
    for (mi, (suffix, d)) in decays.iter().enumerate() {
        // `dst` is the flat (instance, target, combo) index every per-slot
        // buffer in `assemble` is laid out by. Computing it here, once, is
        // what lets `assemble` be a walk over this vector instead of the same
        // nested loops written a second time.
        let mk = |kind: &str, t: &str, c: &crate::stream::Combo, src: Source| {
            FieldMeta::new(format!("{kind}_{t}{}{suffix}", c.label), kind)
                .decay(d)
                .target(t)
                .combo(c)
                .src(src)
        };
        let dst = |t_i: usize, c_i: usize| mi * m * nc + t_i * nc + c_i;
        for (t_i, t) in spec.targets.iter().enumerate() {
            for (c_i, c) in combos.iter().enumerate() {
                fields.push(mk("pred", t, c, Source::Pred(dst(t_i, c_i))));
                fields.push(mk("resid", t, c, Source::Resid(dst(t_i, c_i))));
            }
        }
        if spec.emit_sigma {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("sigma", t, c, Source::Sigma(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_resid_z {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("resid_z", t, c, Source::ResidZ(dst(t_i, c_i))));
                }
            }
        }
        if spec.conformal.is_some() {
            // Not `pred_lo`: `pred_` is the prefix that marks a prediction,
            // for `eval.unpack` and for the README's grammar alike.
            for (k, name) in ["lo", "hi", "coverage"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Conformal(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if spec.emit_metrics {
            for (k, name) in ["ic", "r2", "hit_rate"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Metric(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if let Some(levels) = &spec.resid_quantiles {
            for (li, q) in levels.iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        let name = format!(
                            "absresid_q{}_{t}{}{suffix}",
                            crate::spec::num_label(*q),
                            c.label
                        );
                        let idx = (li * n_models + mi) * m * nc + t_i * nc + c_i;
                        let mut f = FieldMeta::new(name, "absresid_q")
                            .decay(d)
                            .target(t)
                            .combo(c)
                            .src(Source::Quantile(idx));
                        f.quantile = Some(*q);
                        fields.push(f);
                    }
                }
            }
        }
        if spec.emit_autocorr {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("autocorr", t, c, Source::Autocorr(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_drift {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("drift", t, c, Source::Drift(dst(t_i, c_i))));
                }
            }
        }
        fields.push(
            FieldMeta::new(format!("n_eff{suffix}"), "n_eff")
                .decay(d)
                .src(Source::NEff(mi)),
        );
        fields.push(
            FieldMeta::new(format!("coef{suffix}"), "coef")
                .decay(d)
                .src(Source::Coef(mi)),
        );
        if matches!(spec.model, crate::ModelKind::Lasso { .. }) {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(
                    FieldMeta::new(format!("lam_selected_{t}{suffix}"), "lam_selected")
                        .decay(d)
                        .target(t)
                        .src(Source::LamSelected(mi * m + t_i)),
                );
            }
        }
    }
    if spec.emit_selected {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__selected"), "pred_selected")
                    .target(t)
                    .src(Source::SelPred(t_i)),
            );
            fields.push(
                FieldMeta::new(format!("selected_{t}"), "selected")
                    .target(t)
                    .src(Source::SelName(t_i)),
            );
        }
    }
    if spec.emit_averaged {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__averaged"), "pred_averaged")
                    .target(t)
                    .src(Source::AvgPred(t_i)),
            );
        }
    }
    fields
}

/// Labels for every prediction slot of one target, across halflife instances
/// and combos, in the order the slots appear.
fn slot_labels(spec: &Spec) -> Vec<String> {
    let decays = spec.decays().expect("validated");
    let combos = combo_labels(spec);
    let mut out = Vec::new();
    for (suffix, _) in &decays {
        for c in &combos {
            let label = format!("{c}{suffix}");
            out.push(if label.is_empty() {
                "default".to_string()
            } else {
                label.trim_start_matches("__").to_string()
            });
        }
    }
    out
}

/// One output column under construction: a values buffer, NaN where no
/// finite value has been set, and its validity bits. `finish` hands both to
/// polars as they are: no `Vec<Option<f64>>` and no second copy into the
/// `Series`.
///
/// The bits are packed a byte at a time as `scatter`'s run path copies a
/// chunk, while its values are still in cache, and are trusted at `finish`
/// only if every chunk went that way (`packed`); otherwise validity is
/// `is_finite` over `values` in one pass at `finish`. Neither sets a bit per
/// value as it lands: that read-modify-write was a third of assembling a
/// 230-statistic `ew_cov`, and the separate pass re-read every value from
/// memory (docs/PERFORMANCE.md §13). `n_eff` is the one column reported as it
/// is, finite or not, so it alone keeps a bit set per row (`set`).
struct F64Column {
    values: Vec<f64>,
    /// `values.len()` bits, packed little-endian as polars keeps them.
    bits: Vec<u8>,
    /// Every set value's bit is in `bits`.
    packed: bool,
    /// Scratch for `run`: one flag byte per row of the run.
    flags: Vec<u8>,
}

impl F64Column {
    fn new(n: usize) -> Self {
        Self {
            values: vec![f64::NAN; n],
            bits: vec![0u8; n.div_ceil(8)],
            packed: true,
            flags: Vec::new(),
        }
    }

    /// The contract is finite-or-null. NaN is the models' own null encoding,
    /// but a diverged model can also reach exact +/-inf, and `is_nan` alone
    /// would hand that to the user.
    #[inline]
    fn set_if_finite(&mut self, i: usize, v: f64) {
        if v.is_finite() {
            self.values[i] = v;
            self.packed = false;
        }
    }

    /// Valid whatever the value: `n_eff`, which is reported as it is.
    #[inline]
    fn set(&mut self, i: usize, v: f64) {
        self.values[i] = v;
        self.bits[i / 8] |= 1 << (i % 8);
    }

    /// The run `vals` of a chunk whose rows are `base..base + vals.len()`,
    /// with `processed` alongside: a processed, finite value lands with its
    /// bit, anything else leaves NaN and a clear bit -- exactly what
    /// `set_if_finite` over the same rows gives.
    ///
    /// Two passes the compiler vectorizes -- the select into `values` with a
    /// byte flag per row, then the flags packed eight at a time by the
    /// multiply that gathers the low bit of each byte into one byte -- rather
    /// than one pass that shifts each flag into place, which it compiles a
    /// lane at a time: 0.5 ns a value against 0.3 in isolation
    /// (docs/PERFORMANCE.md §13). The partial bytes at each end go bit by bit.
    fn run(&mut self, base: usize, vals: &[f64], processed: &[bool]) {
        let n = vals.len();
        let dst = &mut self.values[base..base + n];
        self.flags.clear();
        self.flags.resize(n, 0);
        let rows = dst
            .iter_mut()
            .zip(vals)
            .zip(processed)
            .zip(self.flags.iter_mut());
        for (((d, &v), &p), f) in rows {
            let ok = p & v.is_finite();
            *d = if ok { v } else { f64::NAN };
            *f = ok as u8;
        }
        let head = ((8 - base % 8) % 8).min(n);
        let body = head + (n - head) / 8 * 8;
        for k in (0..head).chain(body..n) {
            if self.flags[k] != 0 {
                self.bits[(base + k) / 8] |= 1 << ((base + k) % 8);
            }
        }
        let bytes = &mut self.bits[(base + head) / 8..];
        for (f, byte) in self.flags[head..body].chunks_exact(8).zip(bytes) {
            let x = u64::from_le_bytes(f.try_into().expect("eight flags"));
            *byte = (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        }
    }

    /// True where the value is set: finite, or -- with `all` -- written.
    #[inline]
    fn is_valid(&self, i: usize) -> bool {
        if self.packed {
            self.bits[i / 8] >> (i % 8) & 1 == 1
        } else {
            self.values[i].is_finite()
        }
    }

    /// The validity as polars wants it: `None` when every row is valid.
    fn validity(&mut self) -> Option<Bitmap> {
        let n = self.values.len();
        let bits = if self.packed {
            MutableBitmap::from_vec(std::mem::take(&mut self.bits), n)
        } else {
            MutableBitmap::from_trusted_len_iter(self.values.iter().map(|v| v.is_finite()))
        };
        (bits.unset_bits() > 0).then(|| bits.into())
    }

    fn finish(mut self, name: PlSmallStr) -> Series {
        let validity = self.validity();
        Float64Chunked::from_vec_validity(name, self.values, validity).into_series()
    }

    /// The same column as `i32`, for a value that is an index or a count (a
    /// `kmeans` assignment, a `micro` count). Every set value is a small
    /// non-negative integer by construction; the null rows carry NaN and are
    /// masked, not cast.
    fn finish_i32(mut self, name: PlSmallStr) -> Series {
        let validity = self.validity();
        let values: Vec<i32> = self
            .values
            .iter()
            .map(|&v| if v.is_finite() { v as i32 } else { 0 })
            .collect();
        Int32Chunked::from_vec_validity(name, values, validity).into_series()
    }

    /// The same column as `i64`, for an id that only ever grows (a `micro`
    /// id or label).
    fn finish_i64(mut self, name: PlSmallStr) -> Series {
        let validity = self.validity();
        let values: Vec<i64> = self
            .values
            .iter()
            .map(|&v| if v.is_finite() { v as i64 } else { 0 })
            .collect();
        Int64Chunked::from_vec_validity(name, values, validity).into_series()
    }

    /// The same column as `Boolean`, for a `1.0` / `0.0` flag.
    fn finish_bool(self, name: PlSmallStr) -> Series {
        let values: Vec<Option<bool>> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| self.is_valid(i).then_some(v == 1.0))
            .collect();
        Series::new(name, values.as_slice())
    }

    /// The same column as the class names, for an `ew_class` prediction:
    /// every set value is a position in `classes` by construction (the model
    /// emits the argmax over its own classes), and the null rows stay null.
    fn finish_label(self, name: PlSmallStr, classes: &[String]) -> Series {
        let values: Vec<Option<&str>> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                self.is_valid(i)
                    .then(|| classes.get(v as usize).map(String::as_str))
                    .flatten()
            })
            .collect();
        Series::new(name, values.as_slice())
    }
}

/// Scatter one value per processed row of every chunk into a column:
/// `run(chunk, n_rows)` is the field's `n_rows` values for that chunk in row
/// order -- every `ChunkOut` buffer is slot-major, so that is one slice.
/// Non-finite values stay null unless `all` (for `n_eff`, which is always
/// finite and is reported as-is).
///
/// A chunk whose rows are one unbroken run -- a single group, or a group that
/// arrives in blocks -- is copied as a select over three slices with no index
/// indirection, its validity packed on the way (`F64Column::run`); the
/// general path scatters through `rows`. Same values either way: a processed,
/// finite value lands, anything else leaves the NaN prefill
/// (docs/PERFORMANCE.md §13).
fn scatter(
    n: usize,
    chunks: &[ChunkOut],
    all: bool,
    run: impl for<'a> Fn(&'a ChunkOut, usize) -> &'a [f64],
) -> F64Column {
    let mut col = F64Column::new(n);
    for ch in chunks {
        let nr = ch.rows.len();
        let vals = run(ch, nr);
        debug_assert_eq!(vals.len(), nr);
        match ch.contiguous() {
            Some(base) if !all => col.run(base, vals, &ch.processed),
            _ => {
                for (ri, &i) in ch.rows.iter().enumerate() {
                    if !ch.processed[ri] {
                        continue;
                    }
                    if all {
                        col.set(i, vals[ri]);
                    } else {
                        col.set_if_finite(i, vals[ri]);
                    }
                }
            }
        }
    }
    col
}

fn assemble(spec: &Spec, d: &SpecDerived, n: usize, chunks: &[ChunkOut]) -> PolarsResult<Column> {
    let SpecDerived {
        schema,
        slot_labels: labels,
        n_models,
        nc,
        m,
        per_model,
        compare: _,
    } = d;
    let (n_models, nc, m, per_model) = (*n_models, *nc, *m, *per_model);
    // `ew_cov` emits named statistics rather than pred/resid pairs, and has no
    // targets or coefficients — but its values ride in the same `pred` buffer
    // and the schema says so (`Source::Stat`), so one assembler covers both
    // (docs/SIMPLIFICATION.md S2). Only `pred` and `n_eff` are populated for
    // it; every other buffer is length zero and never indexed. `kmeans` is
    // the same shape with an `i32` slot (`Source::Cluster`) and `coef`.
    let n_levels = spec.resid_quantiles.as_ref().map_or(0, Vec::len);

    // Online selection across every slot of a target: pick the slot with the
    // lowest EW out-of-sample error so far (`sigma`, already tracked for E12),
    // and emit that slot's prediction plus its label. Same idea as the lasso's
    // `lam_selected`, generalized to ridge / feature-set / halflife grids.
    let mut sel_pred = vec![vec![None::<f64>; n]; if spec.emit_selected { m } else { 0 }];
    let mut sel_name = vec![vec![None::<&str>; n]; if spec.emit_selected { m } else { 0 }];
    let mut avg_pred = vec![vec![None::<f64>; n]; if spec.emit_averaged { m } else { 0 }];
    if spec.emit_averaged {
        // Exponentially weighted forecaster: weight each slot by
        // exp(-eta * EW squared error), normalized. `sigma` is that error's
        // square root, already tracked for E12, so this costs one pass.
        let eta = spec.average_eta.unwrap_or(1.0);
        for ch in chunks {
            let nr = ch.rows.len();
            for (ri, &i) in ch.rows.iter().enumerate() {
                if !ch.processed[ri] {
                    continue;
                }
                let sig =
                    |mi: usize, slot: usize| ch.sigma[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                let prd =
                    |mi: usize, slot: usize| ch.pred[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                for (t_i, avg_t) in avg_pred.iter_mut().enumerate() {
                    // Subtract the best loss before exponentiating, so the
                    // weights are identical but nothing overflows.
                    let mut best = f64::INFINITY;
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let s = sig(mi, t_i * nc + c_i);
                            if s.is_finite() && prd(mi, t_i * nc + c_i).is_finite() {
                                best = best.min(s * s);
                            }
                        }
                    }
                    if !best.is_finite() {
                        continue;
                    }
                    let (mut num, mut den) = (0.0, 0.0);
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let slot = t_i * nc + c_i;
                            let (s, p) = (sig(mi, slot), prd(mi, slot));
                            if s.is_finite() && p.is_finite() {
                                let wgt = (-eta * (s * s - best)).exp();
                                num += wgt * p;
                                den += wgt;
                            }
                        }
                    }
                    if den > 0.0 {
                        avg_t[i] = Some(num / den);
                    }
                }
            }
        }
    }
    if spec.emit_selected {
        for ch in chunks {
            let nr = ch.rows.len();
            for (ri, &i) in ch.rows.iter().enumerate() {
                if !ch.processed[ri] {
                    continue;
                }
                let sig =
                    |mi: usize, slot: usize| ch.sigma[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                let prd =
                    |mi: usize, slot: usize| ch.pred[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                for t_i in 0..m {
                    let mut best: Option<(f64, usize, usize)> = None;
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let s = sig(mi, t_i * nc + c_i);
                            if s.is_finite() && best.is_none_or(|(b, _, _)| s < b) {
                                best = Some((s, mi, c_i));
                            }
                        }
                    }
                    if let Some((_, mi, c_i)) = best {
                        let p = prd(mi, t_i * nc + c_i);
                        if p.is_finite() {
                            sel_pred[t_i][i] = Some(p);
                            sel_name[t_i][i] = Some(labels[mi * nc + c_i].as_str());
                        }
                    }
                }
            }
        }
    }

    // The schema is written once, in `output_index`, which also records where
    // each field's values live. Emission is a walk over it: no second copy of
    // the ordering, no `format!` here at all, and no way for the declared
    // schema and the realized struct to disagree (docs/SIMPLIFICATION.md S1 —
    // that divergence is exactly what defect E23 was).
    //
    // One job per field, each its own scatter over every stream's run, so
    // the fields of a spec are built in parallel from `PAR_MIN_ROWS` up
    // (docs/PERFORMANCE.md P10): one spec with a grid of instances used to
    // assemble on a single thread. A field's index `i` is
    // `mi * per_model + slot`, the order `output_index` numbers the
    // per-instance fields in.
    let block = |ch: &ChunkOut, nr: usize| ch.n_slots * nr;
    let fields: Vec<Series> =
        map_maybe_par::<_, _, PolarsResult<_>>(schema, n >= PAR_MIN_ROWS, |f| {
            let name: PlSmallStr = f.field.as_str().into();
            // Where field `i`'s run of `nr` values starts in a per-instance buffer.
            let at = |ch: &ChunkOut, nr: usize, i: usize| {
                ChunkOut::at(ch.n_slots, nr, i / per_model, i % per_model, 0)
            };
            Ok(match f.src {
                Source::Pred(i) | Source::Stat(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.pred[at(ch, nr, i)..][..nr]).finish(name)
                }
                Source::Cluster(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.pred[at(ch, nr, i)..][..nr])
                        .finish_i32(name)
                }
                Source::Id(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.pred[at(ch, nr, i)..][..nr])
                        .finish_i64(name)
                }
                Source::Flag(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.pred[at(ch, nr, i)..][..nr])
                        .finish_bool(name)
                }
                Source::Label(i) => {
                    let classes: &[String] = match &spec.model {
                        ModelKind::EwClass { classes, .. } => classes,
                        _ => unreachable!("only ew_class emits a label"),
                    };
                    scatter(n, chunks, false, |ch, nr| &ch.pred[at(ch, nr, i)..][..nr])
                        .finish_label(name, classes)
                }
                Source::Resid(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.resid[at(ch, nr, i)..][..nr])
                        .finish(name)
                }
                Source::Sigma(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.sigma[at(ch, nr, i)..][..nr])
                        .finish(name)
                }
                Source::ResidZ(i) => scatter(n, chunks, false, |ch, nr| {
                    &ch.resid_z[at(ch, nr, i)..][..nr]
                })
                .finish(name),
                Source::Autocorr(i) => scatter(n, chunks, false, |ch, nr| {
                    &ch.autocorr[at(ch, nr, i)..][..nr]
                })
                .finish(name),
                Source::Metric(k, i) => scatter(n, chunks, false, |ch, nr| {
                    // Model-major: instance mi owns 3 contiguous blocks.
                    let (mi, slot) = (i / per_model, i % per_model);
                    &ch.metrics[mi * 3 * block(ch, nr) + k * block(ch, nr) + slot * nr..][..nr]
                })
                .finish(name),
                Source::Conformal(k, i) => scatter(n, chunks, false, |ch, nr| {
                    let (mi, slot) = (i / per_model, i % per_model);
                    &ch.conformal[mi * 3 * block(ch, nr) + k * block(ch, nr) + slot * nr..][..nr]
                })
                .finish(name),
                Source::Quantile(i) => scatter(n, chunks, false, |ch, nr| {
                    // `(li * n_models + mi) * m * nc + slot`, as `output_index`
                    // numbers the quantile fields.
                    let slot = i % (m * nc);
                    let (li, mi) = ((i / (m * nc)) / n_models, (i / (m * nc)) % n_models);
                    &ch.resid_q[mi * n_levels * block(ch, nr) + li * block(ch, nr) + slot * nr..]
                        [..nr]
                })
                .finish(name),
                Source::NEff(mi) => {
                    scatter(n, chunks, true, |ch, nr| &ch.n_eff[mi * nr..][..nr]).finish(name)
                }
                Source::LamSelected(i) => {
                    scatter(n, chunks, false, |ch, nr| &ch.lam_selected[i * nr..][..nr])
                        .finish(name)
                }
                Source::Drift(i) => {
                    let mut drift = vec![None::<bool>; n];
                    for ch in chunks {
                        let nr = ch.rows.len();
                        for (ri, &row) in ch.rows.iter().enumerate() {
                            if ch.processed[ri] {
                                drift[row] = Some(ch.drift[at(ch, nr, i) + ri]);
                            }
                        }
                    }
                    Series::new(name, drift.as_slice())
                }
                Source::SelPred(i) => Series::new(name, sel_pred[i].as_slice()),
                Source::SelName(i) => Series::new(name, sel_name[i].as_slice()),
                Source::AvgPred(i) => Series::new(name, avg_pred[i].as_slice()),
                Source::Coef(mi) => {
                    let mut coef: Vec<Option<&Vec<f64>>> = vec![None; n];
                    for ch in chunks {
                        for (ri, &row) in ch.rows.iter().enumerate() {
                            if ch.processed[ri] {
                                if let Some(c) = &ch.coef[mi][ri] {
                                    coef[row] = Some(c);
                                }
                            }
                        }
                    }
                    let mut b = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        name,
                        n,
                        8,
                        DataType::Float64,
                    );
                    for v in &coef {
                        match v {
                            // Finite-or-null inside the list too: an
                            // `ew_class` class no row has carried yet has
                            // NaN means, and a null says so.
                            Some(flat) => {
                                b.append_iter(flat.iter().map(|c| c.is_finite().then_some(*c)))
                            }
                            None => b.append_null(),
                        }
                    }
                    b.finish().into_series()
                }
                Source::Unset => unreachable!("every field is given a source in output_index"),
            })
        })?;
    let st = StructChunked::from_series(spec.name.as_str().into(), n, fields.iter())?;
    Ok(st.into_series().into())
}

#[cfg(test)]
mod column_tests {
    use super::F64Column;

    /// A deterministic mix of finite, NaN and infinite values.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match (x >> 33) % 7 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => i as f64 * 0.5 - 3.0,
                }
            })
            .collect()
    }

    /// The packed run path is `set_if_finite` over the same rows, to the bit:
    /// the same values, the same validity, whatever the alignment of the run
    /// within the column and the mix of skipped and non-finite rows.
    #[test]
    fn a_packed_run_matches_the_scatter_bit_for_bit() {
        for base in 0..17 {
            for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100] {
                let n = base + len + 5;
                let vals = values(len, base as u64 * 1000 + len as u64);
                let processed: Vec<bool> = (0..len).map(|k| (k * 7 + base) % 5 != 0).collect();

                let mut fast = F64Column::new(n);
                fast.run(base, &vals, &processed);
                let mut slow = F64Column::new(n);
                for (k, (&v, &p)) in vals.iter().zip(&processed).enumerate() {
                    if p {
                        slow.set_if_finite(base + k, v);
                    }
                }

                assert!(fast.packed, "base {base} len {len}");
                for i in 0..n {
                    assert_eq!(
                        fast.values[i].to_bits(),
                        slow.values[i].to_bits(),
                        "value at {i}: base {base} len {len}"
                    );
                    assert_eq!(
                        fast.is_valid(i),
                        slow.is_valid(i),
                        "bit at {i}: base {base} len {len}"
                    );
                    assert_eq!(fast.is_valid(i), fast.values[i].is_finite());
                }
                let (fv, sv) = (fast.validity(), slow.validity());
                assert_eq!(fv, sv, "validity: base {base} len {len}");
            }
        }
    }

    /// Once any row went through `set_if_finite`, the bits are not trusted:
    /// validity comes from the values, so a mixed column is still right.
    #[test]
    fn a_mixed_column_falls_back_to_the_values() {
        let n = 40;
        let vals = values(16, 7);
        let mut col = F64Column::new(n);
        col.run(8, &vals, &[true; 16]);
        col.set_if_finite(3, 1.5);
        col.set_if_finite(30, f64::NAN);
        assert!(!col.packed);
        for i in 0..n {
            assert_eq!(col.is_valid(i), col.values[i].is_finite());
        }
        let v = col.validity().expect("some rows are null");
        for i in 0..n {
            assert_eq!(v.get_bit(i), col.values[i].is_finite());
        }
    }

    /// `set` is for `n_eff`: valid whatever the value, null where never set.
    #[test]
    fn set_is_valid_whatever_the_value() {
        let mut col = F64Column::new(10);
        col.set(2, f64::NAN);
        col.set(9, 0.0);
        assert!(col.packed);
        let v = col.validity().expect("rows 0, 1, 3..9 are null");
        assert_eq!(v.set_bits(), 2);
        assert!(v.get_bit(2) && v.get_bit(9));
    }

    /// A fully valid column reports no validity at all, as polars expects.
    #[test]
    fn a_full_column_has_no_validity() {
        let mut col = F64Column::new(24);
        let vals: Vec<f64> = (0..24).map(|i| i as f64).collect();
        col.run(0, &vals, &[true; 24]);
        assert!(col.validity().is_none());
        let mut col = F64Column::new(24);
        for (i, &v) in vals.iter().enumerate() {
            col.set_if_finite(i, v);
        }
        assert!(col.validity().is_none());
    }
}
