//! The output index: one row per field, and no two rows the same apart
//! from the field's name. `ew_cov`'s `lag_corr` at different lags,
//! `hmm`'s per-state posteriors, `ew_class`'s per-class ones and `bocpd`'s
//! per-feature predictions each shared every value but the name, so a
//! caller filtering the index could not reach one of them without
//! parsing it.

use online_polars::{FieldMeta, Spec, output_index};

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

/// Every value of a row but its name, as the index serializes it.
fn rest(m: &FieldMeta) -> serde_json::Value {
    let mut v = serde_json::to_value(m).unwrap();
    v.as_object_mut().unwrap().remove("field");
    v
}

fn assert_unique(s: &Spec) {
    let idx = output_index(s);
    for (i, a) in idx.iter().enumerate() {
        for b in &idx[i + 1..] {
            assert_ne!(
                rest(a),
                rest(b),
                "{} and {} differ only in name",
                a.field,
                b.field
            );
        }
    }
}

fn by_name<'a>(idx: &'a [FieldMeta], name: &str) -> &'a FieldMeta {
    idx.iter().find(|m| m.field == name).unwrap()
}

#[test]
fn ew_cov_lag_corr_carries_its_lag() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ew_cov", "stats": ["lag_corr", "corr"],
            "lags": [1, 3]}, "targets": ["a"], "features": ["a", "b"], "half_life": 10}"#,
    );
    assert_unique(&s);
    let idx = output_index(&s);
    assert_eq!(by_name(&idx, "lag_corr_a_b_l3").lag, Some(3));
    assert_eq!(by_name(&idx, "lag_corr_b_a_l1").lag, Some(1));
    assert_eq!(by_name(&idx, "corr_a_b").lag, None);
    // Serialized whatever its value, so the Python table has the column.
    assert!(rest(by_name(&idx, "corr_a_b")).get("lag").is_some());
}

#[test]
fn hmm_posteriors_carry_their_state() {
    let s = spec(
        r#"{"name": "h", "model": {"type": "hmm", "n_components": 3, "precision_prior": 1.0},
            "targets": ["a"], "features": ["a", "b"], "half_life": 10}"#,
    );
    assert_unique(&s);
    let idx = output_index(&s);
    for st in 0..3 {
        assert_eq!(by_name(&idx, &format!("filtered_{st}")).state, Some(st));
        assert_eq!(by_name(&idx, &format!("predicted_{st}")).state, Some(st));
    }
    assert_eq!(by_name(&idx, "state").state, None);
    assert_eq!(by_name(&idx, "loglik").state, None);
}

#[test]
fn ew_class_posteriors_carry_their_class() {
    let s = spec(
        r#"{"name": "c", "model": {"type": "ew_class", "classes": ["u", "v_w"],
            "precision_prior": 1.0}, "targets": ["y"], "features": ["a"], "half_life": 10}"#,
    );
    assert_unique(&s);
    let idx = output_index(&s);
    assert_eq!(by_name(&idx, "p_u").class.as_deref(), Some("u"));
    assert_eq!(by_name(&idx, "p_v_w").class.as_deref(), Some("v_w"));
    assert_eq!(by_name(&idx, "class").class, None);
}

/// The index names each `p_<class>` and the bank fills it from that
/// class's posterior: on three well-separated classes the posteriors sum
/// to 1 on every row, and the class the row is assigned is the largest.
#[test]
fn ew_class_fills_each_posterior_from_its_own_class() {
    use polars::prelude::*;
    // Two instances, so each instance's block of slots is read too: at
    // half-lives this far apart their posteriors differ.
    let s = spec(
        r#"{"name": "c", "model": {"type": "ew_class", "classes": ["u", "v", "w"],
            "precision_prior": 1.0}, "targets": ["y"], "features": ["a"],
            "half_life": [5, 3000], "min_weight": 3}"#,
    );
    let suffixes: Vec<String> = output_index(&s)
        .iter()
        .filter(|m| m.kind == "class")
        .map(|m| m.field["class".len()..].to_string())
        .collect();
    assert_eq!(suffixes.len(), 2, "{suffixes:?}");
    let n = 300;
    let label = |i: usize| ["u", "v", "w"][i % 3];
    let a: Vec<f64> = (0..n)
        .map(|i| 10.0 * (i % 3) as f64 + 0.1 * ((i * 7919) % 11) as f64)
        .collect();
    let y: Vec<&str> = (0..n).map(label).collect();
    let df = df!("a" => a, "y" => y).unwrap();
    let mut bank = online_polars::Bank::new(vec![s]).unwrap();
    let out = DataFrame::new(n, bank.fit_predict(&df).unwrap()).unwrap();
    let fields = out
        .column("c")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series();
    let get = |name: &str| {
        fields
            .iter()
            .find(|f| f.name().as_str() == name)
            .unwrap_or_else(|| panic!("no field {name}"))
    };
    let mut each = Vec::new();
    for suffix in &suffixes {
        let cls: Vec<Option<String>> = get(&format!("class{suffix}"))
            .cast(&DataType::String)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(String::from))
            .collect();
        let p: Vec<Vec<Option<f64>>> = ["u", "v", "w"]
            .iter()
            .map(|c| {
                get(&format!("p_{c}{suffix}"))
                    .f64()
                    .unwrap()
                    .iter()
                    .collect()
            })
            .collect();
        let mut read = 0;
        for i in 0..n {
            let Some(c) = &cls[i] else { continue };
            let row: Vec<f64> = p.iter().map(|col| col[i].unwrap()).collect();
            assert!(
                (row.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "{suffix} row {i}: {row:?}"
            );
            let best = (0..3).max_by(|&x, &y| row[x].total_cmp(&row[y])).unwrap();
            assert_eq!(c, ["u", "v", "w"][best], "{suffix} row {i}: {row:?}");
            assert_eq!(c, label(i), "{suffix} row {i}: the classes are apart");
            read += 1;
        }
        assert!(read > 250, "{suffix}: {read} rows read");
        each.push(p);
    }
    assert_ne!(each[0], each[1], "each instance reads its own posteriors");
}

#[test]
fn bocpd_predictions_carry_their_column() {
    let s = spec(
        r#"{"name": "b", "model": {"type": "bocpd", "hazard": 50.0}, "targets": ["a"],
            "features": ["a", "b", "c"]}"#,
    );
    assert_unique(&s);
    let idx = output_index(&s);
    let all = ["a", "b", "c"].map(String::from).to_vec();
    for f in ["a", "b", "c"] {
        assert_eq!(
            by_name(&idx, &format!("pred_{f}")).columns,
            Some(vec![f.to_string()])
        );
    }
    for f in ["p_change", "run_mode", "run_mean", "loglik"] {
        assert_eq!(by_name(&idx, f).columns.as_ref(), Some(&all), "{f}");
    }
}

#[test]
fn a_grid_regression_is_unique_apart_from_names() {
    assert_unique(&spec(
        r#"{"name": "r", "model": {"type": "ewridge", "ridge": [0.1, 1.0],
            "feature_sets": [["one", ["a"]], ["two", ["a", "b"]]]}, "targets": ["y", "z"],
            "features": ["a", "b"], "half_life": [10, 100], "emit_sigma": true,
            "resid_quantiles": [0.1, 0.9]}"#,
    ));
}
