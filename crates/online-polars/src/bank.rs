//! The chunk-fed model bank (docs/PLAN.md §5): column extraction, per-group
//! state, rayon fan-out over (spec x group), versioned msgpack save/load.

use std::collections::HashMap;
use std::path::Path;

use online_core::ClockCfg;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spec::Spec;
use crate::stream::{RowOut, Stream, StreamState, combo_labels};

/// Stable, platform-independent 64-bit hash for session-change detection.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Columns extracted once per (spec, chunk).
struct SpecColumns {
    features: Vec<Vec<Option<f64>>>,
    targets: Vec<Vec<Option<f64>>>,
    clock: Option<Vec<Option<f64>>>,
    session: Option<Vec<u64>>,
    weight: Option<Vec<Option<f64>>>,
}

fn f64_column(df: &DataFrame, name: &str) -> PolarsResult<Vec<Option<f64>>> {
    let s = df
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().collect())
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
            let v = f64_column(df, c)?;
            if let Some(i) = v.iter().position(|x| x.is_none_or(|f| !f.is_finite())) {
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
            Some(
                s.str()?
                    .iter()
                    .map(|v| fnv1a(v.unwrap_or("\0<null>").as_bytes()))
                    .collect(),
            )
        }
        None => None,
    };
    let weight = match &spec.weight {
        Some(c) => Some(f64_column(df, c)?),
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

/// Row-index partition by group key, in row order.
fn group_indices(
    df: &DataFrame,
    group: &Option<String>,
) -> PolarsResult<Vec<(String, Vec<usize>)>> {
    match group {
        None => Ok(vec![(String::new(), (0..df.height()).collect())]),
        Some(g) => {
            let s = df
                .column(g)?
                .as_materialized_series()
                .cast(&DataType::String)?;
            let mut order: Vec<String> = Vec::new();
            let mut map: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, v) in s.str()?.iter().enumerate() {
                let key = v.unwrap_or("<null>").to_string();
                map.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    Vec::new()
                });
                map.get_mut(&key).unwrap().push(i);
            }
            Ok(order
                .into_iter()
                .map(|k| {
                    let idx = map.remove(&k).unwrap();
                    (k, idx)
                })
                .collect())
        }
    }
}

const BANK_MAGIC: &str = "polars-online-bank";

#[derive(Serialize, Deserialize)]
struct BankFile {
    magic: String,
    schema_version: u32,
    package_version: String,
    specs: Vec<Spec>,
    /// Per spec: (group key, stream state) pairs.
    states: Vec<Vec<(String, StreamState)>>,
}

pub struct Bank {
    specs: Vec<Spec>,
    clock_cfgs: Vec<ClockCfg>,
    states: Vec<HashMap<String, Stream>>,
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

    /// Run every spec over one chunk; returns one struct column per spec.
    /// Chunks must arrive in stream order within each group.
    pub fn fit_predict(&mut self, df: &DataFrame) -> PolarsResult<Vec<Column>> {
        let n = df.height();
        let cols: Vec<SpecColumns> = self
            .specs
            .iter()
            .map(|s| extract(df, s))
            .collect::<PolarsResult<_>>()?;
        let groups: Vec<Vec<(String, Vec<usize>)>> = self
            .specs
            .iter()
            .map(|s| group_indices(df, &s.group))
            .collect::<PolarsResult<_>>()?;

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
        let mut per_spec_rows: Vec<Vec<(usize, Option<RowOut>)>> = Vec::with_capacity(specs.len());
        for (si, hm) in self.states.iter_mut().enumerate() {
            let spec = &specs[si];
            let cfg = &cfgs[si];
            let sc = &cols[si];
            let spec_groups = &groups[si];
            // Pull each group's stream out so rayon tasks own disjoint &mut.
            let mut work: Vec<(&String, &Vec<usize>, &mut Stream)> = Vec::new();
            let mut taken: HashMap<&String, &mut Stream> = hm.iter_mut().collect();
            for (key, idx) in spec_groups {
                let stream = taken.remove(key).expect("stream materialized above");
                work.push((key, idx, stream));
            }
            let rows: Vec<Vec<(usize, Option<RowOut>)>> = work
                .into_par_iter()
                .map(|(_key, idx, stream)| {
                    let last = *idx.last().unwrap_or(&usize::MAX);
                    idx.iter()
                        .map(|&i| {
                            let x: Vec<Option<f64>> = sc.features.iter().map(|f| f[i]).collect();
                            let y: Vec<Option<f64>> = sc.targets.iter().map(|t| t[i]).collect();
                            let out = stream.process_row(
                                spec,
                                cfg,
                                &x,
                                &y,
                                sc.clock.as_ref().map(|c| c[i].unwrap()),
                                sc.session.as_ref().map(|s| s[i]),
                                sc.weight.as_ref().map(|w| w[i].unwrap_or(f64::NAN)),
                                i == last,
                            );
                            (i, out)
                        })
                        .collect()
                })
                .collect();
            per_spec_rows.push(rows.into_iter().flatten().collect());
        }

        let mut out = Vec::with_capacity(specs.len());
        for (si, spec) in specs.iter().enumerate() {
            out.push(assemble(spec, n, &per_spec_rows[si])?);
        }
        Ok(out)
    }

    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let file = BankFile {
            magic: BANK_MAGIC.to_string(),
            schema_version: online_core::SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            specs: self.specs.clone(),
            states: self
                .states
                .iter()
                .map(|hm| {
                    let mut v: Vec<(String, StreamState)> =
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

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
    let decays = spec.decays().expect("validated");
    let combos = combo_labels(spec);
    let mut fields = Vec::new();
    for (suffix, _) in &decays {
        for t in &spec.targets {
            for c in &combos {
                fields.push(format!("pred_{t}{c}{suffix}"));
                fields.push(format!("resid_{t}{c}{suffix}"));
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
    fields
}

fn assemble(spec: &Spec, n: usize, rows: &[(usize, Option<RowOut>)]) -> PolarsResult<Column> {
    let decays = spec.decays().expect("validated");
    let n_models = decays.len();
    let combos = combo_labels(spec);
    let nc = combos.len();
    let m = spec.m();

    let mut pred = vec![vec![None::<f64>; n]; n_models * m * nc];
    let mut resid = vec![vec![None::<f64>; n]; n_models * m * nc];
    let mut n_eff = vec![vec![None::<f64>; n]; n_models];
    let mut coef: Vec<Vec<Option<Vec<f64>>>> = vec![vec![None; n]; n_models];
    let is_lasso = matches!(spec.model, crate::ModelKind::Lasso { .. });
    let mut lam_sel = vec![vec![None::<f64>; n]; if is_lasso { n_models * m } else { 0 }];

    for (i, out) in rows {
        let Some(out) = out else { continue };
        for mi in 0..n_models {
            for slot in 0..m * nc {
                let v = out.pred[mi][slot];
                pred[mi * m * nc + slot][*i] = if v.is_nan() { None } else { Some(v) };
                let r = out.resid[mi][slot];
                resid[mi * m * nc + slot][*i] = if r.is_nan() { None } else { Some(r) };
            }
            n_eff[mi][*i] = Some(out.n_eff[mi]);
            if let Some(c) = &out.coef {
                let flat: Vec<f64> = c[mi].iter().flatten().copied().collect();
                coef[mi][*i] = Some(flat);
            }
            if is_lasso {
                if let Some(online_core::Extra::Lasso { lam_selected }) = &out.extra[mi] {
                    for (t_i, l) in lam_selected.iter().enumerate() {
                        lam_sel[mi * m + t_i][*i] = Some(*l);
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
    let st = StructChunked::from_series(spec.name.as_str().into(), n, fields.iter())?;
    Ok(st.into_series().into())
}
