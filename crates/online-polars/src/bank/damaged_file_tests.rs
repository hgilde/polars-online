//! A state file re-encoded with one field changed, everything else what
//! a real save wrote (task 160, PA1 and PA2): refused at load, never a
//! panic at load or at the first read after it. Beside them, what a file
//! keeps that it should not: a dropped group's PCA continuity (PA7) and the
//! key order of a refused chunk (PA9). A file of their own, keeping bank.rs
//! under `tests/test_repo_hygiene.py`'s 250 KB cap for a source file.
use super::{BankFile, GroupKey};
use crate::{Bank, Spec};
use polars::prelude::*;

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

fn ridge() -> Spec {
    spec(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0}"#,
    )
}

fn frame() -> DataFrame {
    df!(
        "g" => ["a", "b", "a", "b", "a", "b"],
        "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y" => [2.0, 4.1, 6.0, 8.2, 10.0, 12.1]
    )
    .unwrap()
}

/// The file of a bank whose key 1 closed when key 2 arrived, its row
/// still waiting in the queue.
fn closing() -> (Spec, Vec<u8>) {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0,
            "group_close": "monotone"}"#,
    );
    let df = df!(
        "g" => [1i64, 1, 1, 1, 2, 2, 2, 2],
        "x" => [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "y" => [2.0, 4.1, 6.0, 8.2, 2.0, 4.0, 6.1, 8.0]
    )
    .unwrap();
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&df).unwrap();
    (s, bank.save_bytes().unwrap())
}

fn reencoded(bytes: &[u8], edit: impl FnOnce(&mut BankFile)) -> Vec<u8> {
    let mut file: BankFile = rmp_serde::from_slice(bytes).unwrap();
    edit(&mut file);
    rmp_serde::to_vec_named(&file).unwrap()
}

/// PA1: `states` holds one list per spec. One more panicked at load,
/// with the caller's specs and without; one fewer loaded and forgot
/// every group of the spec it lost.
#[test]
fn a_file_whose_states_do_not_match_its_specs_is_refused() {
    let specs = vec![ridge()];
    let mut bank = Bank::new(specs.clone()).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let more = reencoded(&bytes, |f| f.states.push(f.states[0].clone()));
    let fewer = reencoded(&bytes, |f| f.states.clear());
    for (what, b) in [("one more", &more), ("one fewer", &fewer)] {
        for expected in [None, Some(&specs[..])] {
            let err = Bank::load_bytes(b, expected)
                .err()
                .unwrap_or_else(|| panic!("{what}: loaded"));
            assert!(
                err.contains("damaged") && err.contains("states"),
                "{what}: {err}"
            );
        }
    }
    assert!(Bank::load_bytes(&bytes, Some(&specs)).is_ok());
}

/// Task 176: a row held under `embargo` keeps its place on the elapsed
/// clock, which its release is measured from. A file whose held row has
/// none is refused at load, rather than left to hold the row for good.
#[test]
fn a_held_row_without_its_place_on_the_elapsed_clock_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0, "embargo": 2}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let damaged = reencoded(&bytes, |f| {
        let held = &mut f.states[0][0].1.pending;
        assert!(!held.is_empty(), "the group holds rows");
        held[0].arrived = None;
    });
    let err = Bank::load_bytes(&damaged, None).err().expect("loaded");
    assert!(err.contains("pending rows do not fit"), "{err}");
    assert!(Bank::load_bytes(&bytes, None).is_ok());
}

/// A row held under `embargo` is learned at its release with the values the
/// file holds, where none of the bank's checks on a row ran: a feature or a
/// target that is not a usable number, a weight that is not one or is below
/// 0, a step that is not finite, below 0 or past `gap_cap` loaded, and
/// reached the models as no live row can -- `clock.rs`'s "no model ever
/// receives a non-finite step" with it (review 2026-10-06, PB4).
#[test]
fn a_held_row_whose_values_the_bank_never_sends_is_refused() {
    use crate::stream::PendingRow;
    let s = spec(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "clock": "t", "gap_cap": 10.0,
            "half_life": 5.0, "embargo": 2.0}"#,
    );
    let df = df!(
        "g" => ["a", "b", "a", "b", "a", "b"],
        "t" => [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y" => [2.0, 4.1, 6.0, 8.2, 10.0, 12.1]
    )
    .unwrap();
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&df).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let specs = std::slice::from_ref(&s);
    assert!(Bank::load_bytes(&bytes, Some(specs)).is_ok(), "as saved");
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut PendingRow));
    let damage: [Damage; 10] = [
        ("a NaN feature", &|r| r.xs[0] = f64::NAN),
        ("an infinite feature", &|r| r.xs[0] = f64::INFINITY),
        ("a feature past the input bound", &|r| r.xs[0] = 1e101),
        ("a NaN target", &|r| r.ys[0] = Some(f64::NAN)),
        ("a NaN weight", &|r| r.w = f64::NAN),
        ("a negative weight", &|r| r.w = -1.0),
        ("a negative step", &|r| r.d_clock = -3.0),
        ("an infinite step", &|r| r.d_clock = f64::INFINITY),
        ("a NaN step", &|r| r.d_clock = f64::NAN),
        ("a step past gap_cap", &|r| r.d_clock = 11.0),
    ];
    for (what, f) in damage {
        let b = reencoded(&bytes, |file| {
            let held = &mut file.states[0][0].1.pending;
            assert!(!held.is_empty(), "the group holds rows");
            f(&mut held[0]);
        });
        let err = Bank::load_bytes(&b, Some(specs))
            .err()
            .unwrap_or_else(|| panic!("{what}: loaded"));
        assert!(
            err.contains("held row") && err.contains("damaged"),
            "{what}: {err}"
        );
    }
}

/// The stream's residual quantile sketches are held to their own shape as
/// `ew_cov`'s is: a level pointer outside the buckets loaded, and the next
/// residual's `add` indexed past them (CB2's class in the stream, review
/// 2026-10-06; `EwQuantile::reset` then needs no `expect`).
#[test]
fn a_residual_sketch_of_the_wrong_shape_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0,
            "resid_quantiles": [0.5, 0.9]}"#,
    );
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let specs = std::slice::from_ref(&s);
    assert!(Bank::load_bytes(&bytes, Some(specs)).is_ok(), "as saved");
    let damaged = reencoded(&bytes, |file| {
        let q = &mut file.states[0][0].1.resid_q[0][0];
        let mut v = serde_json::to_value(&*q).unwrap();
        v["at"] = serde_json::json!([-60_000, -60_000]);
        *q = serde_json::from_value(v).unwrap();
    });
    match Bank::load_bytes(&damaged, Some(specs)) {
        Err(err) => assert!(err.contains("residual quantiles"), "{err}"),
        Ok(mut back) => {
            let read = back.fit_predict(&frame());
            panic!("a sketch pointing outside its buckets loaded: {read:?}");
        }
    }
}

/// PA8: every per-spec entry of the envelope names a spec the bank has.
/// One past the specs was dropped in silence (`high_water`,
/// `clock_dtypes`, `key_integer`, `resolvers`) or kept and saved again for
/// ever (`pca_prev`, `pca_prev_by_group`), where a `states` list or a
/// closed row of the same shape is refused (task 160, PA1/PA2).
#[test]
fn an_envelope_entry_for_a_spec_the_bank_has_not_got_is_refused() {
    let specs = vec![ridge()];
    let mut bank = Bank::new(specs.clone()).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let pca = || online_core::Pca {
        eig: vec![1.0],
        trace: 1.0,
        loadings: vec![1.0],
    };
    let z = || GroupKey(Some("z".into()));
    type Edit<'a> = (&'a str, &'a dyn Fn(&mut BankFile));
    let edits: [Edit; 6] = [
        ("high_water", &|f| f.high_water.push((7, z()))),
        ("clock_dtypes", &|f| {
            f.clock_dtypes.push((7, crate::stream::ClockDtype::Numeric));
        }),
        ("key_integer", &|f| f.key_integer.push((7, true))),
        ("resolvers", &|f| {
            f.resolvers.push((7, vec![(z(), vec![1, 2])]))
        }),
        ("pca_prev", &|f| f.pca_prev.push((7, String::new(), pca()))),
        ("pca_prev_by_group", &|f| {
            f.pca_prev_by_group.push((7, z(), String::new(), pca()));
        }),
    ];
    for (what, edit) in edits {
        let b = reencoded(&bytes, edit);
        for expected in [None, Some(&specs[..])] {
            match Bank::load_bytes(&b, expected) {
                Err(err) => assert!(
                    err.contains("damaged") && err.contains(what) && err.contains("spec 7"),
                    "{what}: {err}"
                ),
                Ok(loaded) => {
                    let again: BankFile =
                        rmp_serde::from_slice(&loaded.save_bytes().unwrap()).unwrap();
                    panic!(
                        "{what}: an entry for spec 7 of 1 loaded; saved again, the file \
                         holds {} PCA entries",
                        again.pca_prev.len() + again.pca_prev_by_group.len()
                    );
                }
            }
        }
    }
    assert!(Bank::load_bytes(&bytes, Some(&specs)).is_ok());
}

/// PA2: a file above this build's schema was told it was "saved before
/// schema 37" and to refit, where the fix is to upgrade; one below keeps
/// today's message.
#[test]
fn a_schema_refusal_says_which_side_of_the_range_the_file_is_on() {
    let specs = vec![ridge()];
    let mut bank = Bank::new(specs.clone()).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let at = |v: u32| reencoded(&bytes, |f| f.schema_version = v);
    let current = online_core::SCHEMA_VERSION;
    let newer = Bank::load_bytes(&at(current + 1), Some(&specs))
        .err()
        .expect("a newer file loaded");
    assert!(
        newer.contains("written by a newer version")
            && newer.contains(&format!("schema {}", current + 1))
            && newer.contains(&format!("this build reads up to {current}"))
            && newer.contains("upgrade polars-online")
            && !newer.contains("refit"),
        "{newer}"
    );
    let older = Bank::load_bytes(&at(super::MIN_BANK_SCHEMA_VERSION - 1), Some(&specs))
        .err()
        .expect("an older file loaded");
    assert!(
        older.contains("not supported")
            && older.contains("refit it from its input")
            && !older.contains("newer"),
        "{older}"
    );
}

/// PA2: a closed row whose spec index, targets or Gram do not fit its
/// spec loaded, saved again, and panicked in `closed_groups`.
#[test]
fn a_closed_row_that_does_not_fit_its_spec_is_refused() {
    let (s, bytes) = closing();
    let mut good = Bank::load_bytes(&bytes, Some(std::slice::from_ref(&s))).unwrap();
    assert_eq!(good.closed_groups(None, false).unwrap().height(), 1);
    fn gram(f: &mut BankFile) -> &mut super::Gram {
        f.closed[0].gram.as_mut().expect("ew_ridge keeps a Gram")
    }
    type Edit = fn(&mut BankFile);
    let cases: Vec<(&str, Edit)> = vec![
        ("spec index 5", |f| f.closed[0].spec = 5),
        ("no comoments", |f| gram(f).comoments.clear()),
        ("target 9", |f| gram(f).targets = vec![9]),
        ("k = 1000", |f| gram(f).k = 1000),
        ("one mean short", |f| {
            gram(f).means.pop();
        }),
    ];
    for (what, edit) in cases {
        let b = reencoded(&bytes, edit);
        for expected in [None, Some(std::slice::from_ref(&s))] {
            match Bank::load_bytes(&b, expected) {
                Err(err) => assert!(
                    err.contains("damaged") && err.contains("closed"),
                    "{what}: {err}"
                ),
                Ok(mut bank) => {
                    // What loaded must at least read: the old build
                    // panicked here.
                    let read = bank.closed_groups(None, false);
                    panic!("{what}: loaded, and closed_groups gave {read:?}");
                }
            }
        }
    }
}

/// The bank's row count at the top of its range: saturated by the next
/// chunk, where a release build wrapped it to the chunk's size and a
/// debug one panicked.
#[test]
fn a_row_count_at_the_top_of_its_range_saturates() {
    let specs = vec![ridge()];
    let mut bank = Bank::new(specs.clone()).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = reencoded(&bank.save_bytes().unwrap(), |f| f.rows_fed = u64::MAX);
    let mut loaded = Bank::load_bytes(&bytes, Some(&specs)).unwrap();
    loaded.fit_predict(&frame()).unwrap();
    assert_eq!(loaded.rows_seen(), u64::MAX);
}

/// A summary whose learned rows wrap past zero when added to its
/// zero-weight rows: `u64::MAX + 1` read as 0 rows learned and loaded
/// in a release build, and panicked in a debug one.
#[test]
fn a_summary_count_that_wraps_is_refused() {
    let specs = vec![ridge()];
    let mut bank = Bank::new(specs.clone()).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = reencoded(&bank.save_bytes().unwrap(), |f| {
        let summary = f.states[0][0].1.summary.as_mut().expect("a summary");
        summary.rows_learned = u64::MAX;
        summary.rows_zero_weight = 1;
    });
    let err = Bank::load_bytes(&bytes, Some(&specs))
        .err()
        .expect("refused");
    assert!(err.contains("more rows learned than processed"), "{err}");
}

/// PA9: a chunk `"monotone"` refuses for its key order leaves the bank as
/// it was, the order its keys are read in included: that was kept
/// before the check ran.
#[test]
fn a_refused_monotone_chunk_keeps_no_key_order() {
    let (s, _) = closing();
    let mut bank = Bank::new(vec![s]).unwrap();
    let backwards = df!(
        "g" => [2i64, 1],
        "x" => [1.0, 2.0],
        "y" => [2.0, 4.0]
    )
    .unwrap();
    let err = bank.fit_predict(&backwards).unwrap_err().to_string();
    assert!(err.contains("non-decreasing order"), "{err}");
    let file: BankFile = rmp_serde::from_slice(&bank.save_bytes().unwrap()).unwrap();
    assert!(file.key_integer.is_empty(), "{:?}", file.key_integer);
    // Taken, the chunk keeps it.
    let forwards = df!("g" => [1i64, 2], "x" => [1.0, 2.0], "y" => [2.0, 4.0]).unwrap();
    bank.fit_predict(&forwards).unwrap();
    let file: BankFile = rmp_serde::from_slice(&bank.save_bytes().unwrap()).unwrap();
    assert_eq!(file.key_integer, vec![(0, true)]);
}

/// PA7: under `group_close = "session"` the PCA sign continuity is kept
/// per group; `drop_groups` drops it with the group, so a bank over an
/// unbounded key space stays bounded, file and all.
#[test]
fn drop_groups_drops_the_groups_pca_continuity() {
    let s = spec(
        r#"{"name": "cov", "model": {"type": "ew_cov", "pca": 2},
            "features": ["a", "b", "c"], "group": "g", "session": "s",
            "half_life": 5.0, "group_close": "session"}"#,
    );
    let keys: Vec<String> = (0..5).map(|k| format!("k{k}")).collect();
    let chunk = |session: &str| {
        let (mut g, mut ss, mut a, mut b, mut c) = (vec![], vec![], vec![], vec![], vec![]);
        for k in &keys {
            for i in 0..6 {
                g.push(k.clone());
                ss.push(session.to_string());
                a.push(1.0 + i as f64);
                b.push(2.0 - 0.5 * i as f64);
                c.push(((i * 3) % 5) as f64);
            }
        }
        df!("g" => g, "s" => ss, "a" => a, "b" => b, "c" => c).unwrap()
    };
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&chunk("s1")).unwrap();
    bank.fit_predict(&chunk("s2")).unwrap();
    bank.closed_groups(None, true).unwrap();
    let held = |bank: &Bank| -> Vec<GroupKey> {
        let file: BankFile = rmp_serde::from_slice(&bank.save_bytes().unwrap()).unwrap();
        file.pca_prev_by_group
            .into_iter()
            .map(|(_, g, _, _)| g)
            .collect()
    };
    assert_eq!(held(&bank).len(), keys.len(), "each key closed once");
    let gone: Vec<GroupKey> = keys.iter().map(|k| GroupKey(Some(k.clone()))).collect();
    bank.drop_groups(&gone[..2], None).unwrap();
    assert_eq!(
        held(&bank),
        gone[2..].to_vec(),
        "the dropped keys' entries went"
    );
    bank.drop_groups(&gone[2..], Some(0)).unwrap();
    assert!(held(&bank).is_empty());
}
