//! The chunk-fed model bank (docs/PLAN.md §5): column extraction, per-group
//! state, rayon fan-out over (spec x group), versioned msgpack save/load.

use std::collections::HashMap;
use std::path::Path;

use online_core::ClockCfg;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spec::Spec;
use crate::stream::{ChunkOut, Stream, StreamState, combo_labels};

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

/// Values as `f64`, null as NaN. Zero-copy-ish for the common case: a
/// null-free contiguous Float64 column is a `memcpy`.
fn f64_column(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let s = df
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::Float64)?;
    let ca = s.f64()?;
    if ca.null_count() == 0 {
        if let Ok(slice) = ca.cont_slice() {
            return Ok(slice.to_vec());
        }
    }
    Ok(ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn extract(df: &DataFrame, spec: &Spec) -> PolarsResult<SpecColumns> {
    let features = spec
        .features
        .iter()
        .map(|c| f64_column(df, c))
        .collect::<PolarsResult<Vec<_>>>()?;
    let targets = spec
        .targets
        .iter()
        .map(|c| f64_column(df, c))
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
            let dtype = df.column(c)?.dtype().clone();
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
            let v = f64_column(df, c)?;
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
            let s = df
                .column(c)?
                .as_materialized_series()
                .cast(&DataType::String)?;
            Some(s.str()?.iter().map(session_hash).collect())
        }
        None => None,
    };
    let weight = match &spec.weight {
        Some(c) => {
            let v = f64_column(df, c)?;
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

/// Row-index partition by group key, in row order.
///
/// Keyed on a 64-bit hash of the string value rather than on the string
/// itself, so the per-row cost is a hash rather than a `String` allocation and
/// two more clones (docs/PERFORMANCE.md P3). The `GroupKey` is materialized
/// once per distinct group, not once per row -- and it is still the *value*
/// that is stored and serialized, so state files are unaffected. A 64-bit
/// collision would merge two groups; the same 2^-64 exposure the session hash
/// already documents.
fn group_indices(
    df: &DataFrame,
    group: &Option<String>,
) -> PolarsResult<Vec<(GroupKey, Vec<usize>)>> {
    match group {
        None => Ok(vec![(GroupKey::ungrouped(), (0..df.height()).collect())]),
        Some(g) => {
            let s = df
                .column(g)?
                .as_materialized_series()
                .cast(&DataType::String)?;
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
}

pub struct Bank {
    specs: Vec<Spec>,
    clock_cfgs: Vec<ClockCfg>,
    states: Vec<HashMap<GroupKey, Stream>>,
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
        }
        let clock_cfgs = specs
            .iter()
            .map(|s| s.clock_cfg())
            .collect::<Result<Vec<_>, _>>()?;
        let states = specs.iter().map(|_| HashMap::new()).collect();
        Ok(Self {
            specs,
            clock_cfgs,
            states,
        })
    }

    pub fn specs(&self) -> &[Spec] {
        &self.specs
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

    /// Run every spec over one chunk; returns one struct column per spec.
    /// Chunks must arrive in stream order within each group.
    pub fn fit_predict(&mut self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        // Section timings to stderr when ONLINE_TIMING is set; costs one env
        // read per chunk. This is how docs/PERFORMANCE.md's numbers are made.
        let timing = std::env::var_os("ONLINE_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let n = df.height();
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
            .map(|s| group_indices(df, &s.group))
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

        // Simpler: process per spec, groups in parallel within the spec.
        let specs = &self.specs;
        let cfgs = &self.clock_cfgs;
        let mut per_spec_rows: Vec<Vec<ChunkOut>> = Vec::with_capacity(specs.len());
        for (si, hm) in self.states.iter_mut().enumerate() {
            let spec = &specs[si];
            let cfg = &cfgs[si];
            let sc = &cols[si];
            let spec_groups = &groups[si];
            // Pull each group's stream out so rayon tasks own disjoint &mut.
            let mut work: Vec<(&GroupKey, &Vec<usize>, &mut Stream)> = Vec::new();
            let mut taken: HashMap<&GroupKey, &mut Stream> = hm.iter_mut().collect();
            for (key, idx) in spec_groups {
                let stream = taken.remove(key).expect("stream materialized above");
                work.push((key, idx, stream));
            }
            let rows: Vec<StreamRows> = work
                .into_par_iter()
                .map(|(_key, idx, stream)| {
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
                        .map_err(|(raw, i)| {
                            polars_err!(ComputeError:
                                "spec {:?}: clock column {:?} goes backwards by {} at \
                                 row {} (on_clock_reset = \"error\"). Sort each group by \
                                 the clock, or choose \"max\"/\"zero\"/\"reset_state\" \
                                 to define what a backwards clock means.",
                                spec.name,
                                spec.clock.as_deref().unwrap_or("<row count>"),
                                -raw, i
                            )
                        })?;
                    Ok(out)
                })
                .collect();
            per_spec_rows.push(rows.into_iter().collect::<PolarsResult<Vec<_>>>()?);
        }
        let t_process = t2.elapsed();
        let t3 = std::time::Instant::now();

        let mut out = Vec::with_capacity(specs.len());
        for (si, spec) in specs.iter().enumerate() {
            out.push(assemble(spec, n, &per_spec_rows[si])?);
        }
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
        Ok(out)
    }

    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let file = BankFile {
            magic: BANK_MAGIC.to_string(),
            format_version: BANK_FORMAT_VERSION,
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
        if file.schema_version != online_core::SCHEMA_VERSION {
            return Err(format!(
                "state schema version {} not supported (current: {})",
                file.schema_version,
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
fn assemble_ew_cov(spec: &Spec, n: usize, chunks: &[ChunkOut]) -> PolarsResult<Column> {
    let names = output_fields(spec);
    let per_model: usize = names.len() / spec.decays().expect("validated").len();
    let n_slots = per_model - 1; // the trailing n_eff
    let n_models = spec.decays().expect("validated").len();

    let mut cols = vec![vec![None::<f64>; n]; n_models * n_slots];
    let mut n_eff = vec![vec![None::<f64>; n]; n_models];
    for ch in chunks {
        let nr = ch.rows.len();
        for (ri, &i) in ch.rows.iter().enumerate() {
            if !ch.processed[ri] {
                continue;
            }
            for mi in 0..n_models {
                for slot in 0..n_slots {
                    let v = ch.pred[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                    cols[mi * n_slots + slot][i] = if v.is_nan() { None } else { Some(v) };
                }
                n_eff[mi][i] = Some(ch.n_eff[mi * nr + ri]);
            }
        }
    }
    let mut fields: Vec<Series> = Vec::with_capacity(names.len());
    let mut name_iter = names.iter();
    for mi in 0..n_models {
        for slot in 0..n_slots {
            let name = name_iter.next().unwrap();
            fields.push(Series::new(
                name.as_str().into(),
                cols[mi * n_slots + slot].as_slice(),
            ));
        }
        let name = name_iter.next().unwrap();
        fields.push(Series::new(name.as_str().into(), n_eff[mi].as_slice()));
    }
    let st = StructChunked::from_series(spec.name.as_str().into(), n, fields.iter())?;
    Ok(st.into_series().into())
}

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
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
        let mut fields = Vec::new();
        for (suffix, _) in &decays {
            fields.extend(labels.iter().map(|l| format!("{l}{suffix}")));
            fields.push(format!("n_eff{suffix}"));
        }
        return fields;
    }
    let combos = combo_labels(spec);
    let mut fields = Vec::new();
    for (suffix, _) in &decays {
        for t in &spec.targets {
            for c in &combos {
                fields.push(format!("pred_{t}{c}{suffix}"));
                fields.push(format!("resid_{t}{c}{suffix}"));
            }
        }
        if spec.emit_sigma {
            for t in &spec.targets {
                for c in &combos {
                    fields.push(format!("sigma_{t}{c}{suffix}"));
                }
            }
        }
        if spec.emit_resid_z {
            for t in &spec.targets {
                for c in &combos {
                    fields.push(format!("resid_z_{t}{c}{suffix}"));
                }
            }
        }
        if spec.emit_metrics {
            for name in ["ic", "r2", "hit_rate"] {
                for t in &spec.targets {
                    for c in &combos {
                        fields.push(format!("{name}_{t}{c}{suffix}"));
                    }
                }
            }
        }
        if let Some(levels) = &spec.resid_quantiles {
            for q in levels {
                for t in &spec.targets {
                    for c in &combos {
                        fields.push(format!("absresid_q{q}_{t}{c}{suffix}"));
                    }
                }
            }
        }
        if spec.emit_autocorr {
            for t in &spec.targets {
                for c in &combos {
                    fields.push(format!("autocorr_{t}{c}{suffix}"));
                }
            }
        }
        if spec.emit_drift {
            for t in &spec.targets {
                for c in &combos {
                    fields.push(format!("drift_{t}{c}{suffix}"));
                }
            }
        }
        fields.push(format!("n_eff{suffix}"));
        fields.push(format!("coef{suffix}"));
        if matches!(spec.model, crate::ModelKind::Lasso { .. }) {
            for t in &spec.targets {
                fields.push(format!("lam_selected_{t}{suffix}"));
            }
        }
    }
    if spec.emit_selected {
        for t in &spec.targets {
            fields.push(format!("pred_{t}__selected"));
            fields.push(format!("selected_{t}"));
        }
    }
    if spec.emit_averaged {
        for t in &spec.targets {
            fields.push(format!("pred_{t}__averaged"));
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

fn assemble(spec: &Spec, n: usize, chunks: &[ChunkOut]) -> PolarsResult<Column> {
    if matches!(spec.model, crate::ModelKind::EwCov { .. }) {
        return assemble_ew_cov(spec, n, chunks);
    }
    let decays = spec.decays().expect("validated");
    let n_models = decays.len();
    let combos = combo_labels(spec);
    let nc = combos.len();
    let m = spec.m();

    let mut pred = vec![vec![None::<f64>; n]; n_models * m * nc];
    let mut resid = vec![vec![None::<f64>; n]; n_models * m * nc];
    let n_extra = if spec.emit_sigma || spec.emit_resid_z {
        n_models * m * nc
    } else {
        0
    };
    let mut sigma = vec![vec![None::<f64>; n]; n_extra];
    let mut resid_z = vec![vec![None::<f64>; n]; n_extra];
    let n_drift = if spec.emit_drift {
        n_models * m * nc
    } else {
        0
    };
    let mut drift = vec![vec![None::<bool>; n]; n_drift];
    let n_met = if spec.emit_metrics {
        n_models * m * nc
    } else {
        0
    };
    let mut met = vec![vec![vec![None::<f64>; n]; n_met]; 3];
    let n_levels = spec.resid_quantiles.as_ref().map_or(0, Vec::len);
    let mut rq = vec![vec![None::<f64>; n]; n_levels * n_models * m * nc];
    let n_ac = if spec.emit_autocorr {
        n_models * m * nc
    } else {
        0
    };
    let mut ac = vec![vec![None::<f64>; n]; n_ac];
    let mut n_eff = vec![vec![None::<f64>; n]; n_models];
    let mut coef: Vec<Vec<Option<Vec<f64>>>> = vec![vec![None; n]; n_models];
    let is_lasso = matches!(spec.model, crate::ModelKind::Lasso { .. });
    let mut lam_sel = vec![vec![None::<f64>; n]; if is_lasso { n_models * m } else { 0 }];

    // Scatter the flat per-chunk buffers into per-column vectors. NaN is null
    // for every numeric output; `processed` is what distinguishes a skipped row
    // (all null, including the bool `drift`) from one that produced NaN.
    let some_if_finite = |v: f64| if v.is_nan() { None } else { Some(v) };
    for ch in chunks {
        let nr = ch.rows.len();
        let per = ch.n_models * ch.n_slots * nr;
        for (ri, &i) in ch.rows.iter().enumerate() {
            if !ch.processed[ri] {
                continue;
            }
            for mi in 0..n_models {
                for slot in 0..m * nc {
                    let at = ChunkOut::at(ch.n_slots, nr, mi, slot, ri);
                    let dst = mi * m * nc + slot;
                    pred[dst][i] = some_if_finite(ch.pred[at]);
                    resid[dst][i] = some_if_finite(ch.resid[at]);
                    if n_drift > 0 {
                        drift[dst][i] = Some(ch.drift[at]);
                    }
                    if n_met > 0 {
                        for (k, met_k) in met.iter_mut().enumerate() {
                            met_k[dst][i] = some_if_finite(ch.metrics[k * per + at]);
                        }
                    }
                    for li in 0..n_levels {
                        rq[(li * n_models + mi) * m * nc + slot][i] =
                            some_if_finite(ch.resid_q[li * per + at]);
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
    let labels = slot_labels(spec);
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

    let mut fields: Vec<Series> = Vec::new();
    for (mi, (suffix, _)) in decays.iter().enumerate() {
        for (t_i, t) in spec.targets.iter().enumerate() {
            for (c_i, c) in combos.iter().enumerate() {
                let slot = t_i * nc + c_i;
                fields.push(Series::new(
                    format!("pred_{t}{c}{suffix}").into(),
                    pred[mi * m * nc + slot].as_slice(),
                ));
                fields.push(Series::new(
                    format!("resid_{t}{c}{suffix}").into(),
                    resid[mi * m * nc + slot].as_slice(),
                ));
            }
        }
        for (name, src) in [
            ("sigma", (spec.emit_sigma, &sigma)),
            ("resid_z", (spec.emit_resid_z, &resid_z)),
        ] {
            let (enabled, data) = src;
            if !enabled {
                continue;
            }
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    let slot = t_i * nc + c_i;
                    fields.push(Series::new(
                        format!("{name}_{t}{c}{suffix}").into(),
                        data[mi * m * nc + slot].as_slice(),
                    ));
                }
            }
        }
        if spec.emit_metrics {
            for (k, name) in ["ic", "r2", "hit_rate"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        let slot = t_i * nc + c_i;
                        fields.push(Series::new(
                            format!("{name}_{t}{c}{suffix}").into(),
                            met[k][mi * m * nc + slot].as_slice(),
                        ));
                    }
                }
            }
        }
        if let Some(levels) = &spec.resid_quantiles {
            for (li, q) in levels.iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        let slot = t_i * nc + c_i;
                        let idx = (li * n_models + mi) * m * nc + slot;
                        fields.push(Series::new(
                            format!("absresid_q{q}_{t}{c}{suffix}").into(),
                            rq[idx].as_slice(),
                        ));
                    }
                }
            }
        }
        if spec.emit_autocorr {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    let slot = t_i * nc + c_i;
                    fields.push(Series::new(
                        format!("autocorr_{t}{c}{suffix}").into(),
                        ac[mi * m * nc + slot].as_slice(),
                    ));
                }
            }
        }
        if spec.emit_drift {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    let slot = t_i * nc + c_i;
                    fields.push(Series::new(
                        format!("drift_{t}{c}{suffix}").into(),
                        drift[mi * m * nc + slot].as_slice(),
                    ));
                }
            }
        }
        fields.push(Series::new(
            format!("n_eff{suffix}").into(),
            n_eff[mi].as_slice(),
        ));
        let mut b = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            format!("coef{suffix}").into(),
            n,
            8,
            DataType::Float64,
        );
        for v in &coef[mi] {
            match v {
                Some(flat) => b.append_slice(flat),
                None => b.append_null(),
            }
        }
        fields.push(b.finish().into_series());
        if is_lasso {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(Series::new(
                    format!("lam_selected_{t}{suffix}").into(),
                    lam_sel[mi * m + t_i].as_slice(),
                ));
            }
        }
    }
    if spec.emit_selected {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(Series::new(
                format!("pred_{t}__selected").into(),
                sel_pred[t_i].as_slice(),
            ));
            fields.push(Series::new(
                format!("selected_{t}").into(),
                sel_name[t_i].as_slice(),
            ));
        }
    }
    if spec.emit_averaged {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(Series::new(
                format!("pred_{t}__averaged").into(),
                avg_pred[t_i].as_slice(),
            ));
        }
    }
    let st = StructChunked::from_series(spec.name.as_str().into(), n, fields.iter())?;
    Ok(st.into_series().into())
}
