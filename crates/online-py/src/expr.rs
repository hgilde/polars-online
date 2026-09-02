//! Expression plugin (docs/PLAN.md §6): one function `online_run` driven by a
//! full spec in kwargs. It runs a single spec over the whole column it receives,
//! so `.over(group)` gives per-group streams. The implementation is the bank
//! itself (group = None), which makes expression ≡ bank true by construction.
//! All input columns arrive packed in one struct (see [`online_run`]).

use online_polars::{Bank, Spec, output_index};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cell::RefCell;

#[derive(Deserialize)]
struct OnlineKwargs {
    spec_json: String,
}

thread_local! {
    /// The last spec this thread parsed, keyed by the JSON it came from.
    ///
    /// Under `.over(group)` polars calls the plugin once per group with byte-
    /// identical kwargs, so without this a thousand groups meant a thousand
    /// JSON parses and a thousand validations of the same spec
    /// (docs/PERFORMANCE.md P5). Thread-local rather than a shared map: no
    /// lock, and polars spreads the groups across its own threads.
    static SPEC_CACHE: RefCell<Option<(String, Spec)>> = const { RefCell::new(None) };
}

fn parse_spec(kwargs: &OnlineKwargs) -> PolarsResult<Spec> {
    if let Some(hit) = SPEC_CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(json, _)| json == &kwargs.spec_json)
            .map(|(_, spec)| spec.clone())
    }) {
        return Ok(hit);
    }
    let mut spec: Spec =
        crate::from_json(&kwargs.spec_json).map_err(|e| polars_err!(ComputeError: "{}", e))?;
    // The expression API always streams over the column it receives; grouping
    // is polars' job via `.over()`.
    spec.group = None;
    spec.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    SPEC_CACHE.with(|c| *c.borrow_mut() = Some((kwargs.spec_json.clone(), spec.clone())));
    Ok(spec)
}

/// The declared output struct. Polars checks it against what the bank
/// realizes, so both come from the same descriptor: `FieldMeta::dtype`.
fn online_output(_input_fields: &[Field], kwargs: OnlineKwargs) -> PolarsResult<Field> {
    let spec = parse_spec(&kwargs)?;
    let fields: Vec<Field> = output_index(&spec)
        .iter()
        .map(|f| Field::new(f.field.as_str().into(), f.dtype()))
        .collect();
    Ok(Field::new(
        spec.name.as_str().into(),
        DataType::Struct(fields),
    ))
}

/// Column names the packed input's fields carry, in the order
/// `python/polars_online/_expr.py` packs them: targets, features, then any of
/// clock / session / weight that the spec uses. Polars strips input names, so
/// they are reattached here.
fn input_names(spec: &Spec) -> Vec<&str> {
    // ew_cov has no target column: its features are the whole input.
    let mut names: Vec<&str> = if matches!(spec.model, online_polars::ModelKind::EwCov { .. }) {
        Vec::new()
    } else {
        spec.targets.iter().map(String::as_str).collect()
    };
    names.extend(spec.features.iter().map(String::as_str));
    names.extend(
        [&spec.clock, &spec.session, &spec.weight]
            .into_iter()
            .flatten()
            .map(String::as_str),
    );
    names
}

/// The plugin takes **one** input: a struct whose fields are the spec's
/// columns in [`input_names`] order. One packed input rather than one input
/// per column because of how polars evaluates a group-aware function under
/// `.over(group)`: the multi-input path (`apply_multiple_group_aware`) walks
/// the groups one after another on a single thread, while the single-input
/// path runs them through rayon. Packing turned 1000 groups from 4.2 M rows/s
/// into 21 M rows/s (docs/IMPROVEMENTS.md P1).
#[polars_expr(output_type_func_with_kwargs=online_output)]
fn online_run(inputs: &[Series], kwargs: OnlineKwargs) -> PolarsResult<Series> {
    let spec = parse_spec(&kwargs)?;
    let names = input_names(&spec);
    let [packed] = inputs else {
        polars_bail!(ComputeError:
            "online: expected one struct input (the packed columns), got {} inputs",
            inputs.len()
        );
    };
    let fields = packed.struct_()?.fields_as_series();
    if fields.len() != names.len() {
        polars_bail!(ComputeError:
            "online: expected {} packed input columns for this spec, got {}",
            names.len(), fields.len()
        );
    }
    let height = packed.len();
    // One column can serve two roles (a feature that is also the weight, say);
    // the bank reads by name, so the frame needs each name once.
    let mut columns: Vec<Column> = Vec::with_capacity(names.len());
    for (s, name) in fields.into_iter().zip(&names) {
        if columns.iter().all(|c| c.name() != name) {
            columns.push(s.with_name((*name).into()).into());
        }
    }
    let df = DataFrame::new(height, columns)?;
    let mut bank = Bank::new(vec![spec]).map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let mut cols = bank.fit_predict(&df)?;
    Ok(cols.pop().unwrap().take_materialized_series())
}
