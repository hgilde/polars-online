//! Task 460, B2 and B3: a state file whose rings are not the shape the code
//! that reads them assumes is refused at load, never a panic at the next row
//! and never output that differs from a run that was not saved. A file of
//! their own, as `damaged_file_tests` keeps bank.rs under the source cap.
use super::BankFile;
use crate::{Bank, Spec};
use polars::prelude::*;

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

fn frame(n: usize, seed: usize) -> DataFrame {
    let x0: Vec<f64> = (0..n).map(|i| ((i * 7 + seed) % 11) as f64 * 0.3).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 3 + seed) % 5) as f64 - 2.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * x0[i] - x1[i] + ((i * 13 + seed) % 7) as f64 * 0.05)
        .collect();
    let g = vec![1i64; n];
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn reencoded(bytes: &[u8], edit: impl FnOnce(&mut BankFile)) -> Vec<u8> {
    let mut file: BankFile = rmp_serde::from_slice(bytes).unwrap();
    edit(&mut file);
    rmp_serde::to_vec_named(&file).unwrap()
}

/// A named change to a ring held as JSON.
type Edit = (&'static str, fn(&mut serde_json::Value));

/// The value of the first field named `key` anywhere under `v`.
fn find<'a>(v: &'a mut serde_json::Value, key: &str) -> Option<&'a mut serde_json::Value> {
    match v {
        serde_json::Value::Object(m) => {
            if m.contains_key(key) {
                return m.get_mut(key);
            }
            m.values_mut().find_map(|c| find(c, key))
        }
        serde_json::Value::Array(a) => a.iter_mut().find_map(|c| find(c, key)),
        _ => None,
    }
}

/// B2: the robust standard errors' sandwich decays only the lags `gammas`
/// holds, which is every score its ring holds while the ring is no longer
/// than the lags. A file whose ring is longer, or whose rows are not `z`'s
/// width, is refused at load, so the output after a load is the output of a
/// run that was never saved (the second half is pinned for the file as saved).
#[test]
fn a_sandwich_ring_longer_than_its_lags_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": 1e-6}, "targets": ["y"],
            "features": ["x0", "x1"], "group": "g", "half_life": 5.0,
            "emit_robust_se": true, "robust_se_lags": 2}"#,
    );
    let specs = std::slice::from_ref(&s);
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&frame(200, 0)).unwrap();
    let bytes = bank.save_bytes().unwrap();

    // As saved: loads, and continues to the bit.
    let mut never_saved = Bank::load_bytes(&bank.save_bytes().unwrap(), Some(specs)).unwrap();
    let a = bank.fit_predict(&frame(200, 1)).unwrap();
    let b = never_saved.fit_predict(&frame(200, 1)).unwrap();
    assert_eq!(a, b, "a load of a saved state continues to the bit");

    let edits: [Edit; 3] = [
        ("one more row in the ring than lags", |ring| {
            let row = ring[0].clone();
            ring.as_array_mut().unwrap().push(row);
        }),
        ("a ring row of another width", |ring| {
            ring[0].as_array_mut().unwrap().push(0.0.into());
        }),
        ("a ring of many rows", |ring| {
            let row = ring[0].clone();
            for _ in 0..50 {
                ring.as_array_mut().unwrap().push(row.clone());
            }
        }),
    ];
    for (what, edit) in edits {
        let damaged = reencoded(&bytes, |f| {
            let sw = &mut f.states[0][0].1.persisted.checks[0].sandwich[0];
            let mut v = serde_json::to_value(&*sw).unwrap();
            let ring = find(&mut v, "ring").unwrap();
            assert_eq!(ring.as_array().unwrap().len(), 2, "the ring is full");
            edit(ring);
            *sw = serde_json::from_value(v).unwrap();
        });
        match Bank::load_bytes(&damaged, Some(specs)) {
            Err(err) => assert!(err.contains("damaged"), "{what}: {err}"),
            Ok(mut back) => {
                let next = back.fit_predict(&frame(200, 1)).unwrap();
                panic!("{what}: loaded, and went on to {next:?}");
            }
        }
    }
}

/// B3: `rcov`'s unrolled lagged products read each ring row as `k` wide and
/// panic on one that is not. A file with a ring row of another width, a
/// short one or a long one, is refused at load -- through the bank, as a
/// user's file arrives -- and never reaches the next row.
#[test]
fn an_rcov_ring_row_that_is_not_k_wide_is_refused_at_load() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "rcov", "kind": "kernel", "block_rows": 100},
            "features": ["x0", "x1"], "group": "g", "group_close": "monotone"}"#,
    );
    let specs = std::slice::from_ref(&s);
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&frame(30, 0)).unwrap();
    let bytes = bank.save_bytes().unwrap();
    assert!(Bank::load_bytes(&bytes, Some(specs)).is_ok(), "as saved");
    let edits: [Edit; 3] = [
        ("a ring row a value short", |fin| {
            fin[0].as_array_mut().unwrap().pop();
        }),
        ("a ring row a value long", |fin| {
            fin[0].as_array_mut().unwrap().push(0.0.into());
        }),
        ("a ring row of none", |fin| {
            fin[0].as_array_mut().unwrap().clear();
        }),
    ];
    for (what, edit) in edits {
        let damaged = reencoded(&bytes, |f| {
            let m = &mut f.states[0][0].1.models[0];
            let mut v = serde_json::to_value(&*m).unwrap();
            let fin = find(&mut v, "fin").unwrap();
            assert!(!fin.as_array().unwrap().is_empty(), "the ring holds rows");
            edit(fin);
            *m = serde_json::from_value(v).unwrap();
        });
        match Bank::load_bytes(&damaged, Some(specs)) {
            Err(err) => assert!(err.contains("damaged"), "{what}: {err}"),
            Ok(mut back) => {
                let next = back.fit_predict(&frame(30, 1));
                panic!("{what}: loaded, and went on to {next:?}");
            }
        }
    }
}
