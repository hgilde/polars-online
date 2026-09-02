//! The chunk-fed model bank (docs/PLAN.md §5): column extraction, per-group
//! state, rayon fan-out over (spec x group), versioned msgpack save/load.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use online_core::ClockCfg;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spec::Spec;
use crate::stream::{AnyModel, ChunkOut, Stream, StreamState, combo_labels};

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

/// Values as `f64`, null as NaN. Zero-copy-ish for the common case: a
/// null-free contiguous Float64 column is a `memcpy`.
///
/// Only numeric, Boolean and Null columns are accepted. Anything else would be
/// cast non-strictly, and a String column of numbers-as-text (or of anything)
/// becomes all-null: every prediction null and no error to say why.
fn f64_column(df: &DataFrame, spec: &Spec, role: &str, name: &str) -> PolarsResult<Vec<f64>> {
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
    if ca.null_count() == 0 {
        if let Ok(slice) = ca.cont_slice() {
            return Ok(slice.to_vec());
        }
    }
    Ok(ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
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

fn extract(df: &DataFrame, spec: &Spec) -> PolarsResult<SpecColumns> {
    let features = spec
        .features
        .iter()
        .map(|c| f64_column(df, spec, "feature", c))
        .collect::<PolarsResult<Vec<_>>>()?;
    let targets = spec
        .targets
        .iter()
        .map(|c| f64_column(df, spec, "target", c))
        .collect::<PolarsResult<Vec<_>>>()?;
    let clock = match &spec.clock {
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
            let v = f64_column(df, spec, "clock", c)?;
            // Nulls arrive as NaN, which this rejects along with inf: a clock
            // with no value has no defined delta either way.
            if let Some(i) = v.iter().position(|f| !f.is_finite()) {
                polars_bail!(ComputeError:
                    "spec {:?}: clock column {:?} has a null/non-finite value at row {}",
                    spec.name, c, i
                );
            }
            Some(v)
        }
        None => None,
    };
    let session = match &spec.session {
        Some(c) => {
            let s = key_column(df, spec, "session", c)?;
            Some(s.str()?.iter().map(session_hash).collect())
        }
        None => None,
    };
    let weight = match &spec.weight {
        Some(c) => {
            let v = f64_column(df, spec, "weight", c)?;
            // A negative weight is never meaningful for a weighted mean, and
            // silently letting one through corrupts the accumulators (the EW
            // count and the per-target cross moments disagree about whether the
            // row happened). Non-finite weights are a different case, handled
            // uniformly with non-finite features: they mean "no information for
            // this row" and skip it (docs/PLAN.md §3), so only a *finite*
            // negative weight is an error.
            if let Some(i) = v.iter().position(|f| f.is_finite() && *f < 0.0) {
                polars_bail!(ComputeError:
                    "spec {:?}: weight column {:?} has a negative value ({}) at row {}; \
                     weights must be >= 0 (use null to skip a row)",
                    spec.name, c, v[i], i
                );
            }
            Some(v)
        }
        None => None,
    };
    Ok(SpecColumns {
        features,
        targets,
        clock,
        session,
        weight,
    })
}

/// One stream's flat output buffers for a chunk. Fallible because a strict
/// clock policy can refuse a row (`on_clock_reset = "error"`).
type StreamRows = PolarsResult<ChunkOut>;

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
            let s = key_column(df, spec, "group", g)?;
            let mut order: Vec<(GroupKey, Vec<usize>)> = Vec::new();
            let mut slot_of: HashMap<u64, usize> = HashMap::new();
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
#[derive(Debug, Clone)]
pub struct Gram {
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for a single instance).
    pub instance: String,
    /// Columns, including the intercept when the spec has one.
    pub k: usize,
    /// Accumulated weight behind these moments.
    pub n_eff: f64,
    /// EW column means, length `k`.
    pub means: Vec<f64>,
    /// Centered co-moments, row-major `k*k`.
    pub comoments: Vec<f64>,
    /// Per-target centered cross-moments, each `k` long. Empty for `ew_cov`.
    /// Per-target uncentered cross-moments, each `k` long. Empty for
    /// `ew_cov`.
    pub cross_moments: Vec<Vec<f64>>,
    /// Per-target accumulated weight. Empty for `ew_cov`.
    pub target_weights: Vec<f64>,
}

const BANK_MAGIC: &str = "polars-online-bank";

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
}

pub struct Bank {
    specs: Vec<Spec>,
    clock_cfgs: Vec<ClockCfg>,
    derived: Vec<SpecDerived>,
    states: Vec<HashMap<GroupKey, Stream>>,
    /// Rows fed so far. Kept at the bank level because a stream's `rows_seen`
    /// goes with it when its group is dropped.
    rows_fed: u64,
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
}

impl SpecDerived {
    fn new(spec: &Spec) -> Self {
        let schema = output_index(spec);
        let n_models = spec.decays().expect("validated").len();
        let nc = crate::stream::combos(spec).len();
        let m = spec.m();
        let per_model = if matches!(spec.model, crate::ModelKind::EwCov { .. }) {
            schema.iter().filter(|f| f.kind != "n_eff").count() / n_models
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
        }
    }
}

impl Bank {
    pub fn new(specs: Vec<Spec>) -> Result<Self, String> {
        if specs.is_empty() {
            return Err("at least one spec is required".into());
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
        let derived = specs.iter().map(SpecDerived::new).collect();
        let states = specs.iter().map(|_| HashMap::new()).collect();
        Ok(Self {
            specs,
            clock_cfgs,
            derived,
            states,
            rows_fed: 0,
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
    /// appears again, exactly as a never-seen one would.
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
            let stream = &states[key];
            for (label, model) in &stream.models {
                let (cov, cross, weights) = match model {
                    AnyModel::EwRidge(m) => (
                        m.cov(),
                        m.cross_moments().to_vec(),
                        m.target_weights().to_vec(),
                    ),
                    AnyModel::Lasso(m) => (
                        m.cov(),
                        m.cross_moments().to_vec(),
                        m.target_weights().to_vec(),
                    ),
                    // No targets, so no cross-moments: the matrix is the whole
                    // output.
                    AnyModel::EwCov(m) => (m.cov(), Vec::new(), Vec::new()),
                    _ => continue,
                };
                out.push(Gram {
                    group: key.clone(),
                    instance: label.clone(),
                    k: cov.k(),
                    n_eff: cov.n_eff(),
                    means: cov.means().to_vec(),
                    comoments: cov.comoments().to_vec(),
                    cross_moments: cross,
                    target_weights: weights,
                });
            }
        }
        Ok(out)
    }

    /// Run every spec over one chunk; returns one struct column per spec.
    /// Chunks must arrive in stream order within each group.
    pub fn fit_predict(&mut self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        // Section timings to stderr when ONLINE_TIMING is set; costs one env
        // read per chunk. This is how docs/PERFORMANCE.md's numbers are made.
        let timing = std::env::var_os("ONLINE_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let n = df.height();
        // Outputs are attached with `with_column`, which replaces a column of
        // the same name, so a spec named like an input would silently eat the
        // input (the Python bank and the CLI runner both attach that way).
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
        // Independent per spec, and each is a full pass over its columns, so
        // they run in parallel with each other (docs/PERFORMANCE.md P3).
        let cols: Vec<SpecColumns> = self
            .specs
            .par_iter()
            .map(|s| extract(df, s))
            .collect::<PolarsResult<_>>()?;
        let t_extract = t0.elapsed();
        let t1 = std::time::Instant::now();
        let groups: Vec<Vec<(GroupKey, Vec<usize>)>> = self
            .specs
            .par_iter()
            .map(|s| group_indices(df, s))
            .collect::<PolarsResult<_>>()?;
        let t_group = t1.elapsed();
        let t2 = std::time::Instant::now();

        // Materialize missing streams, then fan out over (spec x group).
        for (si, spec) in self.specs.iter().enumerate() {
            for (key, _) in &groups[si] {
                if !self.states[si].contains_key(key) {
                    self.states[si].insert(
                        key.clone(),
                        Stream::new(spec).map_err(|e| polars_err!(ComputeError: "{}", e))?,
                    );
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
        // them pulled out of the maps up front.
        let mut work: Vec<(usize, &Vec<usize>, &mut Stream)> = Vec::new();
        for (si, hm) in self.states.iter_mut().enumerate() {
            let mut taken: HashMap<&GroupKey, &mut Stream> = hm.iter_mut().collect();
            for (key, idx) in &groups[si] {
                let stream = taken.remove(key).expect("stream materialized above");
                work.push((si, idx, stream));
            }
        }
        // Longest stream first: with a few big groups and many small ones,
        // starting the big ones last leaves cores idle at the tail.
        work.sort_by_key(|(_, idx, _)| std::cmp::Reverse(idx.len()));

        // Under `on_clock_reset = "error"` the chunk is refused as a whole:
        // every stream checks its clock schedule on a copy before any model is
        // touched (docs/IMPROVEMENTS.md C3), so the bank is left exactly as it
        // was and the corrected chunk can be fed. A no-op under every other
        // policy.
        work.par_iter().try_for_each(|(si, idx, stream)| {
            let sc = &cols[*si];
            stream
                .check_clock(&cfgs[*si], sc.clock.as_deref(), sc.session.as_deref(), idx)
                .map_err(|(raw, i)| backwards_clock(&specs[*si], raw, i))
        })?;

        let done: Vec<(usize, StreamRows)> = work
            .into_par_iter()
            .map(|(si, idx, stream)| {
                let spec = &specs[si];
                let cfg = &cfgs[si];
                let sc = &cols[si];
                let r = (|| {
                    let mut out =
                        ChunkOut::new(spec, stream.n_models(), stream.n_slots(), idx.len());
                    stream
                        .process_chunk(
                            spec,
                            cfg,
                            &sc.features,
                            &sc.targets,
                            sc.clock.as_deref(),
                            sc.session.as_deref(),
                            sc.weight.as_deref(),
                            idx,
                            &mut out,
                        )
                        .map_err(|(raw, i)| backwards_clock(spec, raw, i))?;
                    Ok(out)
                })();
                (si, r)
            })
            .collect();

        let mut per_spec_rows: Vec<Vec<ChunkOut>> = (0..specs.len()).map(|_| Vec::new()).collect();
        for (si, r) in done {
            per_spec_rows[si].push(r?);
        }
        let t_process = t2.elapsed();
        let t3 = std::time::Instant::now();

        // Specs assemble independently (docs/PERFORMANCE.md P4).
        let out: Vec<Column> = specs
            .par_iter()
            .zip(derived.par_iter())
            .zip(per_spec_rows.par_iter())
            .map(|((spec, d), rows)| assemble(spec, d, n, rows))
            .collect::<PolarsResult<_>>()?;
        if timing {
            let t_assemble = t3.elapsed();
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
        Ok(out)
    }

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
        };
        rmp_serde::to_vec_named(&file).map_err(|e| e.to_string())
    }

    pub fn load_bytes(bytes: &[u8], expected_specs: Option<&[Spec]>) -> Result<Self, String> {
        let file: BankFile = rmp_serde::from_slice(bytes).map_err(|e| e.to_string())?;
        if file.magic != BANK_MAGIC {
            return Err("not a polars-online bank state file".into());
        }
        if file.format_version > BANK_FORMAT_VERSION {
            return Err(format!(
                "bank state file format version {} is newer than this build supports ({})",
                file.format_version, BANK_FORMAT_VERSION
            ));
        }
        if !(online_core::MIN_SCHEMA_VERSION..=online_core::SCHEMA_VERSION)
            .contains(&file.schema_version)
        {
            return Err(format!(
                "state schema version {} not supported (this build loads {}..={})",
                file.schema_version,
                online_core::MIN_SCHEMA_VERSION,
                online_core::SCHEMA_VERSION
            ));
        }
        if let Some(exp) = expected_specs {
            if exp != file.specs.as_slice() {
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
        bank.rows_fed = if file.rows_fed > 0 {
            file.rows_fed
        } else {
            bank.states
                .first()
                .map_or(0, |hm| hm.values().map(|s| s.rows_seen).sum())
        };
        Ok(bank)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = self.save_bytes()?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path, expected_specs: Option<&[Spec]>) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
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
    Quantile(usize),
    Autocorr(usize),
    NEff(usize),
    Coef(usize),
    LamSelected(usize),
    SelPred(usize),
    SelName(usize),
    AvgPred(usize),
    /// An `ew_cov` statistic, which rides in the `pred` buffer.
    Stat(usize),
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

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
    output_index(spec).into_iter().map(|m| m.field).collect()
}

/// Every output field with its metadata, in struct order.
pub fn output_index(spec: &Spec) -> Vec<FieldMeta> {
    let decays = spec.decays().expect("validated");
    // ew_cov is not a regression: its slots are named statistics, not
    // pred/resid pairs, and it has no targets or coefficients.
    if let crate::ModelKind::EwCov { stats, .. } = &spec.model {
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
                _ => online_core::EwCovStat::Corr,
            })
            .collect();
        let labels = online_core::EwCovModel::labels(&spec.features, &kinds);
        // Statistic kind and the columns it is over, in label order: the same
        // walk `labels` makes (per stat: each column, or each i<j pair).
        let mut meta: Vec<(String, Vec<String>)> = Vec::new();
        for (name, kind) in names.iter().zip(&kinds) {
            match kind {
                online_core::EwCovStat::Mean
                | online_core::EwCovStat::Var
                | online_core::EwCovStat::Std => {
                    for col in &spec.features {
                        meta.push((name.clone(), vec![col.clone()]));
                    }
                }
                _ => {
                    for i in 0..spec.features.len() {
                        for j in (i + 1)..spec.features.len() {
                            meta.push((
                                name.clone(),
                                vec![spec.features[i].clone(), spec.features[j].clone()],
                            ));
                        }
                    }
                }
            }
        }
        debug_assert_eq!(meta.len(), labels.len());
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, (l, (kind, cols))) in labels.iter().zip(&meta).enumerate() {
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols.clone());
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
    let combos = crate::stream::combos(spec);
    let (nc, m, n_models) = (combos.len(), spec.m(), decays.len());
    let n_levels = spec.resid_quantiles.as_ref().map_or(0, Vec::len);
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
    let _ = n_levels;
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

fn assemble(spec: &Spec, d: &SpecDerived, n: usize, chunks: &[ChunkOut]) -> PolarsResult<Column> {
    let SpecDerived {
        schema,
        slot_labels: labels,
        n_models,
        nc,
        m,
        per_model,
    } = d;
    let (n_models, nc, m, per_model) = (*n_models, *nc, *m, *per_model);
    // `ew_cov` emits named statistics rather than pred/resid pairs, and has no
    // targets or coefficients — but its values ride in the same `pred` buffer
    // and the schema says so (`Source::Stat`), so one assembler covers both
    // (docs/SIMPLIFICATION.md S2). Only `pred` and `n_eff` are populated for
    // it; every other buffer is length zero and never indexed.
    let is_ew_cov = matches!(spec.model, crate::ModelKind::EwCov { .. });
    let reg = if is_ew_cov { 0 } else { n_models * m * nc };

    let mut pred = vec![vec![None::<f64>; n]; n_models * per_model];
    let mut resid = vec![vec![None::<f64>; n]; reg];
    let n_extra = if spec.emit_sigma || spec.emit_resid_z {
        reg
    } else {
        0
    };
    let mut sigma = vec![vec![None::<f64>; n]; n_extra];
    let mut resid_z = vec![vec![None::<f64>; n]; n_extra];
    let n_drift = if spec.emit_drift { reg } else { 0 };
    let mut drift = vec![vec![None::<bool>; n]; n_drift];
    let n_met = if spec.emit_metrics { reg } else { 0 };
    let mut met = vec![vec![vec![None::<f64>; n]; n_met]; 3];
    let n_levels = spec.resid_quantiles.as_ref().map_or(0, Vec::len);
    let mut rq = vec![vec![None::<f64>; n]; n_levels * reg];
    let n_ac = if spec.emit_autocorr { reg } else { 0 };
    let mut ac = vec![vec![None::<f64>; n]; n_ac];
    let mut n_eff = vec![vec![None::<f64>; n]; n_models];
    let mut coef: Vec<Vec<Option<Vec<f64>>>> = vec![vec![None; n]; n_models];
    let is_lasso = matches!(spec.model, crate::ModelKind::Lasso { .. });
    let mut lam_sel = vec![vec![None::<f64>; n]; if is_lasso { n_models * m } else { 0 }];

    // Scatter the flat per-chunk buffers into per-column vectors. NaN is null
    // for every numeric output; `processed` is what distinguishes a skipped row
    // (all null, including the bool `drift`) from one that produced NaN.
    // The contract is finite-or-null. NaN is the models' own null encoding,
    // but a diverged model can also reach exact +/-inf, and `is_nan` alone
    // would hand that to the user.
    let some_if_finite = |v: f64| v.is_finite().then_some(v);
    for ch in chunks {
        let nr = ch.rows.len();
        let block = ch.n_slots * nr;
        for (ri, &i) in ch.rows.iter().enumerate() {
            if !ch.processed[ri] {
                continue;
            }
            for mi in 0..n_models {
                for slot in 0..per_model {
                    let at = ChunkOut::at(ch.n_slots, nr, mi, slot, ri);
                    let dst = mi * per_model + slot;
                    pred[dst][i] = some_if_finite(ch.pred[at]);
                    if is_ew_cov {
                        continue;
                    }
                    resid[dst][i] = some_if_finite(ch.resid[at]);
                    if n_drift > 0 {
                        drift[dst][i] = Some(ch.drift[at]);
                    }
                    if n_met > 0 {
                        // Model-major: instance mi owns 3 contiguous blocks.
                        let mbase = mi * 3 * block + slot * nr + ri;
                        for (k, met_k) in met.iter_mut().enumerate() {
                            met_k[dst][i] = some_if_finite(ch.metrics[mbase + k * block]);
                        }
                    }
                    for li in 0..n_levels {
                        let qbase = mi * n_levels * block + li * block + slot * nr + ri;
                        rq[(li * n_models + mi) * m * nc + slot][i] =
                            some_if_finite(ch.resid_q[qbase]);
                    }
                    if n_ac > 0 {
                        ac[dst][i] = some_if_finite(ch.autocorr[at]);
                    }
                    if n_extra > 0 {
                        sigma[dst][i] = some_if_finite(ch.sigma[at]);
                        resid_z[dst][i] = some_if_finite(ch.resid_z[at]);
                    }
                }
                n_eff[mi][i] = Some(ch.n_eff[mi * nr + ri]);
                if let Some(c) = &ch.coef[mi][ri] {
                    coef[mi][i] = Some(c.clone());
                }
                if is_lasso {
                    for t_i in 0..m {
                        lam_sel[mi * m + t_i][i] =
                            some_if_finite(ch.lam_selected[(mi * m + t_i) * nr + ri]);
                    }
                }
            }
        }
    }

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
    let mut fields: Vec<Series> = Vec::with_capacity(schema.len());
    for f in schema {
        let name: PlSmallStr = f.field.as_str().into();
        fields.push(match f.src {
            Source::Pred(i) | Source::Stat(i) => Series::new(name, pred[i].as_slice()),
            Source::Resid(i) => Series::new(name, resid[i].as_slice()),
            Source::Sigma(i) => Series::new(name, sigma[i].as_slice()),
            Source::ResidZ(i) => Series::new(name, resid_z[i].as_slice()),
            Source::Drift(i) => Series::new(name, drift[i].as_slice()),
            Source::Metric(k, i) => Series::new(name, met[k][i].as_slice()),
            Source::Quantile(i) => Series::new(name, rq[i].as_slice()),
            Source::Autocorr(i) => Series::new(name, ac[i].as_slice()),
            Source::NEff(i) => Series::new(name, n_eff[i].as_slice()),
            Source::LamSelected(i) => Series::new(name, lam_sel[i].as_slice()),
            Source::SelPred(i) => Series::new(name, sel_pred[i].as_slice()),
            Source::SelName(i) => Series::new(name, sel_name[i].as_slice()),
            Source::AvgPred(i) => Series::new(name, avg_pred[i].as_slice()),
            Source::Coef(i) => {
                let mut b =
                    ListPrimitiveChunkedBuilder::<Float64Type>::new(name, n, 8, DataType::Float64);
                for v in &coef[i] {
                    match v {
                        Some(flat) => b.append_slice(flat),
                        None => b.append_null(),
                    }
                }
                b.finish().into_series()
            }
            Source::Unset => unreachable!("every field is given a source in output_index"),
        });
    }
    let st = StructChunked::from_series(spec.name.as_str().into(), n, fields.iter())?;
    Ok(st.into_series().into())
}
