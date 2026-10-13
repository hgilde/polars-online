//! The two ways a spec's runs reach its output struct (task 255 (a)): one
//! run at a time as its stream runs, for a spec one stream serves, and every
//! run at once after the streams are done. A test-only switch sends the
//! specs named `sink-off-*` the second way, and a counter records the runs
//! the first way takes for the specs named `sink-on-*`, so the other tests
//! of this binary, running beside these, are untouched.
use std::collections::HashMap;
use std::sync::Mutex;

use polars::prelude::*;
use polars_arrow::array::{Array, Float64Array, StructArray};

use crate::{Bank, Spec};

static ABSORBED: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// Whether `spec` is sent the way every run goes at once.
pub(in crate::bank) fn sink_off(spec: &str) -> bool {
    spec.starts_with("sink-off-")
}

/// Count a run a sink took as its stream ran, for the specs probed here.
pub(super) fn absorbed(spec: &str) {
    if spec.starts_with("sink-on-") {
        *ABSORBED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .entry(spec.to_string())
            .or_default() += 1;
    }
}

fn runs_absorbed(spec: &str) -> usize {
    ABSORBED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(|m| m.get(spec).copied())
        .unwrap_or(0)
}

/// Every output a regression of a grid can emit, `coef` on every row:
/// `ewridge` the uncertainty fields, `lasso` the selected slot.
fn specs(name: &str) -> Vec<Spec> {
    let common = r#""targets": ["y", "z"], "features": ["x0", "x1", "x2"], "clock": "t",
        "gap_cap": 6.0, "weight": "w", "emit_clocks": true, "coef_every": 0,
        "emit_sigma": true, "emit_zscore": true, "emit_averaged": true, "emit_metrics": true,
        "conformal": 0.9, "resid_quantiles": [0.1, 0.9], "emit_autocorr": true,
        "emit_drift": true, "drift_threshold": 20.0, "emit_calibration": true,
        "emit_breaks": true, "emit_specification": true, "emit_tails": true,
        "emit_feature_health": true"#;
    let ewridge = format!(
        r#"{{"name": "{name}-ewridge", "model": {{"type": "ewridge", "ridge": [1e-6, 1.0]}},
            "half_life": [25.0, 100.0], "emit_error_inflation": true, "emit_se_coef": true,
            "emit_robust_se": true, "emit_influence": true, {common}}}"#
    );
    let lasso = format!(
        r#"{{"name": "{name}-lasso", "model": {{"type": "lasso", "penalties": [0.2, 0.0]}},
            "half_life": 25.0, "emit_selected": true, {common}}}"#
    );
    [ewridge, lasso]
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

/// An irregular clock with a gap past the cap, null features (skipped
/// rows), null targets, zero weights: rows of every kind in every run.
fn frame(n: usize) -> DataFrame {
    let mut t = 0.0;
    let mut clock = Vec::with_capacity(n);
    for i in 0..n {
        t += [0.5, 1.0, 1.5][i % 3] + if i == n / 3 { 40.0 } else { 0.0 };
        clock.push(t);
    }
    let x = |a: usize, b: usize| -> Vec<Option<f64>> {
        (0..n)
            .map(|i| (i % 97 != a).then(|| ((i * b) % 23) as f64 / 7.0 - 1.5))
            .collect()
    };
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 89 != 5).then(|| ((i * 7) % 19) as f64 / 5.0))
        .collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 11) % 17) as f64 / 3.0).collect();
    let w: Vec<f64> = (0..n)
        .map(|i| if i % 31 == 0 { 0.0 } else { 1.0 })
        .collect();
    df!("t" => clock, "x0" => x(3, 5), "x1" => x(50, 3), "x2" => x(98, 13), "y" => y,
        "z" => z, "w" => w)
    .unwrap()
}

/// Two struct arrays equal field for field: the same dtype and validity,
/// and for a float the same bits in every slot, a null's included.
fn assert_same(a: &StructArray, b: &StructArray, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    assert_eq!(a.dtype(), b.dtype(), "{what}");
    for (i, (x, y)) in a.values().iter().zip(b.values()).enumerate() {
        assert_eq!(x.validity(), y.validity(), "{what}: field {i}");
        match (
            x.as_any().downcast_ref::<Float64Array>(),
            y.as_any().downcast_ref::<Float64Array>(),
        ) {
            (Some(x), Some(y)) => {
                let bits =
                    |v: &Float64Array| v.values().iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(x), bits(y), "{what}: field {i}");
            }
            _ => assert!(x.as_ref() == y.as_ref(), "{what}: field {i}"),
        }
    }
}

fn structs(cols: &[Column]) -> Vec<StructArray> {
    cols.iter()
        .map(|c| {
            let s = c.as_materialized_series().rechunk();
            s.chunks()[0]
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone()
        })
        .collect()
}

/// A spec one stream serves takes its runs one at a time, and its output is
/// the output of the same runs taken at once, to the bit, over one chunk and
/// over three.
#[test]
fn a_run_at_a_time_is_every_run_at_once() {
    let df = frame(2400);
    for (k, parts) in [1usize, 3].into_iter().enumerate() {
        let on = format!("sink-on-{k}");
        let mut a = Bank::new(specs(&on)).unwrap();
        let mut b = Bank::new(specs(&format!("sink-off-{k}"))).unwrap();
        let step = df.height().div_ceil(parts);
        for p in 0..parts {
            let chunk = df.slice((p * step) as i64, step);
            let (sa, sb) = (
                structs(&a.fit_predict(&chunk).unwrap()),
                structs(&b.fit_predict(&chunk).unwrap()),
            );
            for (x, y) in sa.iter().zip(&sb) {
                assert_same(x, y, &format!("{parts} chunks, chunk {p}"));
            }
        }
        // The states differ only by the specs' names, which they carry;
        // the stream's last row is kept before a run goes to its sink.
        // Both specs took the first way, and `ewridge`, the wider, split
        // the one chunk into runs.
        let runs = |s: &str| runs_absorbed(&format!("{on}-{s}"));
        assert!(runs("ewridge") >= parts.max(2), "{} runs", runs("ewridge"));
        assert!(runs("lasso") >= parts, "{} runs", runs("lasso"));
        assert_eq!(runs_absorbed(&format!("sink-off-{k}-ewridge")), 0);
    }
}
