//! Expression plugin (docs/PLAN.md §6): one function `online_run` driven by a
//! full spec in kwargs. It runs a single spec over the whole column it receives,
//! so `.over(group)` gives per-group streams. The implementation is the bank
//! itself (group = None), which makes expression ≡ bank true by construction.

use online_polars::{Bank, Spec, output_fields};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct OnlineKwargs {
    spec_json: String,
}

fn parse_spec(kwargs: &OnlineKwargs) -> PolarsResult<Spec> {
    let mut spec: Spec = serde_json::from_str(&kwargs.spec_json)
        .map_err(|e| polars_err!(ComputeError: "invalid online spec: {}", e))?;
    // The expression API always streams over the column it receives; grouping
    // is polars' job via `.over()`.
    spec.group = None;
    spec.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    Ok(spec)
}

fn online_output(_input_fields: &[Field], kwargs: OnlineKwargs) -> PolarsResult<Field> {
    let spec = parse_spec(&kwargs)?;
    let fields: Vec<Field> = output_fields(&spec)
        .into_iter()
        .map(|name| {
            let dtype = if name.starts_with("coef") {
                DataType::List(Box::new(DataType::Float64))
            } else {
                DataType::Float64
            };
            Field::new(name.into(), dtype)
        })
        .collect();
    Ok(Field::new(
        spec.name.as_str().into(),
        DataType::Struct(fields),
    ))
}

/// Column names the plugin's positional inputs carry, in the order
/// `python/polars_online/_expr.py` passes them: targets, features, then any of
/// clock / session / weight that the spec uses. Polars strips input names, so
/// they are reattached here.
fn input_names(spec: &Spec) -> Vec<&str> {
    let mut names: Vec<&str> = spec.targets.iter().map(String::as_str).collect();
    names.extend(spec.features.iter().map(String::as_str));
    names.extend(
        [&spec.clock, &spec.session, &spec.weight]
            .into_iter()
            .flatten()
            .map(String::as_str),
    );
    names
}

#[polars_expr(output_type_func_with_kwargs=online_output)]
fn online_run(inputs: &[Series], kwargs: OnlineKwargs) -> PolarsResult<Series> {
    let spec = parse_spec(&kwargs)?;
    let names = input_names(&spec);
    if inputs.len() != names.len() {
        polars_bail!(ComputeError:
            "online: expected {} input columns for this spec, got {}",
            names.len(), inputs.len()
        );
    }
    let height = inputs.first().map(|s| s.len()).unwrap_or(0);
    let columns: Vec<Column> = inputs
        .iter()
        .zip(&names)
        .map(|(s, name)| s.clone().with_name((*name).into()).into())
        .collect();
    let df = DataFrame::new(height, columns)?;
    let mut bank = Bank::new(vec![spec]).map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let mut cols = bank.fit_predict(&df)?;
    Ok(cols.pop().unwrap().take_materialized_series())
}
