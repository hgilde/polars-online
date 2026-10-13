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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0, "embargo": 2}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let damaged = reencoded(&bytes, |f| {
        let held = &mut f.states[0][0].1.persisted.pending;
        assert!(!held.is_empty(), "the group holds rows");
        held[0].arrived = None;
    });
    let err = Bank::load_bytes(&damaged, None).err().expect("loaded");
    assert!(err.contains("pending rows do not fit"), "{err}");
    assert!(Bank::load_bytes(&bytes, None).is_ok());
}

/// A held-rows clock is kept, one per model instance, by a stream with an
/// `embargo`, and by no other. A schema-14 file had none, and its loader
/// rebuilt it from the rows still held, until the minimum schema passed it
/// (docs/PLAN.md tasks 198 and 194-202): one of the wrong length is
/// refused, as is one beside a spec that keeps none.
#[test]
fn a_held_rows_clock_of_the_wrong_length_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0, "embargo": 2}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    for (what, edit) in [
        ("none", (|c: &mut Vec<f64>| c.clear()) as fn(&mut Vec<f64>)),
        ("one too many", |c: &mut Vec<f64>| c.push(0.0)),
    ] {
        let damaged = reencoded(&bytes, |f| {
            edit(&mut f.states[0][0].1.persisted.pending_clock)
        });
        let err = Bank::load_bytes(&damaged, None).err().expect(what);
        assert!(err.contains("held-rows clock"), "{what}: {err}");
    }
    let mut plain = Bank::new(vec![ridge()]).unwrap();
    plain.fit_predict(&frame()).unwrap();
    let damaged = reencoded(&plain.save_bytes().unwrap(), |f| {
        f.states[0][0].1.persisted.pending_clock = vec![0.0];
    });
    let err = Bank::load_bytes(&damaged, None).err().expect("no embargo");
    assert!(err.contains("held-rows clock"), "{err}");
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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
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
            let held = &mut file.states[0][0].1.persisted.pending;
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
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0,
            "resid_quantiles": [0.5, 0.9]}"#,
    );
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let specs = std::slice::from_ref(&s);
    assert!(Bank::load_bytes(&bytes, Some(specs)).is_ok(), "as saved");
    let damaged = reencoded(&bytes, |file| {
        let q = &mut file.states[0][0].1.persisted.resid_q[0][0];
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

/// The stream's `emit_metrics` accumulators were taken on their count
/// alone, so a file whose joint moments were a value short loaded, and the
/// next scored row indexed past them in `EwCov::update` (review round 5,
/// C2). Held to their shape, as the sketches are.
#[test]
fn a_metrics_accumulator_of_the_wrong_shape_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0,
            "emit_metrics": true}"#,
    );
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let specs = std::slice::from_ref(&s);
    assert!(Bank::load_bytes(&bytes, Some(specs)).is_ok(), "as saved");
    for field in ["m", "c", "m_lo"] {
        let damaged = reencoded(&bytes, |file| {
            let m = &mut file.states[0][0].1.persisted.metrics[0][0];
            let mut v = serde_json::to_value(&*m).unwrap();
            let short = v["joint"][field].as_array_mut().unwrap();
            assert!(short.pop().is_some(), "{field} holds a value");
            *m = serde_json::from_value(v).unwrap();
        });
        match Bank::load_bytes(&damaged, Some(specs)) {
            Err(err) => assert!(err.contains("metrics"), "{field}: {err}"),
            Ok(mut back) => {
                let read = back.fit_predict(&frame());
                panic!("{field}: a metrics accumulator a value short loaded: {read:?}");
            }
        }
    }
}

/// The decay time is kept per model instance by every file this build
/// loads; one of the wrong length is damage, and was replaced with zeros,
/// which restarted `settled_frac` and `weight_sum_settled` from the load
/// (review round 5, C7). Refused by name, as the held-rows clock is.
#[test]
fn a_decay_time_of_the_wrong_length_is_refused() {
    let mut bank = Bank::new(vec![ridge()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    for (what, edit) in [
        ("none", (|c: &mut Vec<f64>| c.clear()) as fn(&mut Vec<f64>)),
        ("one too many", |c: &mut Vec<f64>| c.push(0.0)),
    ] {
        let damaged = reencoded(&bytes, |f| edit(&mut f.states[0][0].1.persisted.decay_time));
        let err = Bank::load_bytes(&damaged, None).err().expect(what);
        assert!(err.contains("decay time"), "{what}: {err}");
    }
    assert!(Bank::load_bytes(&bytes, None).is_ok());
}

/// PA8: every per-spec entry of the envelope names a spec the bank has.
/// One past the specs was dropped in silence (`high_water`,
/// `clock_dtypes`, `key_dtypes`, `resolvers`) or kept and saved again for
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
        ("key_dtypes", &|f| f.key_dtypes.push((7, [None, None]))),
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
        f.closed[0].gram.as_mut().expect("ewridge keeps a Gram")
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
    assert_eq!(loaded.rows_fed(), u64::MAX);
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
        let summary = f.states[0][0]
            .1
            .persisted
            .summary
            .as_mut()
            .expect("a summary");
        summary.rows_learned = u64::MAX;
        summary.rows_zero_weight = 1;
    });
    let err = Bank::load_bytes(&bytes, Some(&specs))
        .err()
        .expect("refused");
    assert!(err.contains("more rows learned than processed"), "{err}");
}

/// PA9: a chunk `"monotone"` refuses for its key order leaves the bank as
/// it was, the order its keys are read in included -- the group column's
/// form since review round 4, N22 -- which was kept before the check ran.
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
    assert!(file.key_dtypes.is_empty(), "{:?}", file.key_dtypes);
    // Taken, the chunk keeps it.
    let forwards = df!("g" => [1i64, 2], "x" => [1.0, 2.0], "y" => [2.0, 4.0]).unwrap();
    bank.fit_predict(&forwards).unwrap();
    let file: BankFile = rmp_serde::from_slice(&bank.save_bytes().unwrap()).unwrap();
    let integer = crate::arrow::KeyDtype::of(&DataType::Int64);
    assert_eq!(file.key_dtypes, vec![(0, [integer, None])]);
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

// ---- Task 252: damage no test had reached. ----

/// A model state of a kind the bank never builds -- `ew_cov`'s bare
/// accumulator, which only the core uses -- is refused at load, by kind.
#[test]
fn a_model_state_the_bank_never_builds_is_refused_by_kind() {
    let mut bank = Bank::new(vec![ridge()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let b = reencoded(&bank.save_bytes().unwrap(), |f| {
        f.states[0][0].1.models[0].model =
            online_core::ModelState::EwCov(Box::new(online_core::EwCov::new(1)));
    });
    let err = Bank::load_bytes(&b, None).err().expect("loaded");
    assert!(
        err.contains("state is for a different model: expected a bank-supported model"),
        "{err}"
    );
}

/// A diagnostic kept under the spec's switches but shaped for another lag
/// or horizon than the file's own spec now says is damage, refused at load,
/// where a fresh start would hide it.
#[test]
fn a_diagnostic_shaped_for_another_lag_or_horizon_is_refused() {
    let at = |extra: &str| {
        spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x"], "group": "g", "half_life": 10.0{extra}}}"#
        ))
    };
    type Edit = fn(&mut Spec);
    let cases: [(&str, &str, Edit, &str); 2] = [
        (
            "autocorrelation",
            r#", "emit_autocorr": true, "resid_autocorr_lag": 1"#,
            |s| s.resid_autocorr_lag = Some(3),
            "saved state's autocorrelations do not fit this spec",
        ),
        (
            "horizon",
            r#", "emit_calibration": true, "horizon_rows": 1"#,
            |s| s.horizon_rows = Some(4),
            "saved state's residual diagnostics",
        ),
    ];
    for (what, extra, edit, want) in cases {
        let mut bank = Bank::new(vec![at(extra)]).unwrap();
        bank.fit_predict(&frame()).unwrap();
        let bytes = bank.save_bytes().unwrap();
        let b = reencoded(&bytes, |f| edit(&mut f.specs[0]));
        let err = Bank::load_bytes(&b, None).err().expect(what);
        assert!(err.contains(want), "{what}: {err}");
        assert!(Bank::load_bytes(&bytes, None).is_ok());
    }
}

/// A formula target's window core whose saved bytes do not load is refused
/// at the chunk that resumes it, naming the spec and the formula target,
/// and the refused chunk teaches nothing.
#[test]
fn a_formula_targets_window_core_that_does_not_load_is_refused_at_its_chunk() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"},
            "targets": [{"name": "fwd", "formula": ["rewm_mean", ["col", "y"], {"half_life": 2.0, "window_size": 3.0}]}],
            "features": ["x"], "clock": "t", "gap_cap": 5.0, "half_life": 10.0,
            "embargo": 3.0}"#,
    );
    let rows = |from: i32| {
        df!(
            "t" => (from..from + 6).map(f64::from).collect::<Vec<_>>(),
            "x" => (from..from + 6).map(|i| f64::from(i % 4)).collect::<Vec<_>>(),
            "y" => (from..from + 6).map(|i| f64::from(i % 3)).collect::<Vec<_>>()
        )
        .unwrap()
    };
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&rows(0)).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let b = reencoded(&bytes, |f| {
        assert_eq!(f.resolvers.len(), 1, "the core is saved");
        for (_, groups) in &mut f.resolvers {
            for (_, saved) in groups {
                *saved = vec![0xc1, 0xc1];
            }
        }
    });
    let mut damaged = Bank::load_bytes(&b, None).unwrap();
    let fed = damaged.rows_fed();
    let err = damaged.fit_predict(&rows(6)).unwrap_err().to_string();
    assert!(err.starts_with("spec \"m\": formula target: "), "{err}");
    assert_eq!(damaged.rows_fed(), fed, "the refused chunk taught nothing");
    let mut good = Bank::load_bytes(&bytes, None).unwrap();
    good.fit_predict(&rows(6)).unwrap();
}

/// A stream whose file keeps no data summary loads and goes on without
/// one, for good: counting from the load would report a number that looks
/// like the whole history and is not.
#[test]
fn a_stream_without_a_summary_goes_on_without_one() {
    let mut bank = Bank::new(vec![ridge()]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let b = reencoded(&bank.save_bytes().unwrap(), |f| {
        for (_, st) in &mut f.states[0] {
            st.persisted.summary = None;
        }
    });
    let mut loaded = Bank::load_bytes(&b, None).unwrap();
    let out = loaded.fit_predict(&frame()).unwrap();
    assert_eq!(out[0].len(), 6);
    let again: BankFile = rmp_serde::from_slice(&loaded.save_bytes().unwrap()).unwrap();
    assert!(
        again.states[0]
            .iter()
            .all(|(_, s)| s.persisted.summary.is_none()),
        "a summary was started after the load"
    );
}

/// Rows held under `embargo` whose score-time predictions the file does not
/// keep fold nothing into the residual spread when they mature: there is no
/// honest prediction to fold. The model learns them as it would.
#[test]
fn a_held_row_without_its_score_folds_nothing_when_it_matures() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "half_life": 10.0, "embargo": 2, "emit_sigma": true}"#,
    );
    let first = df!("x" => [1.0, 2.0, 3.0, 4.0, 5.0], "y" => [2.1, 3.9, 6.2, 7.8, 10.1]).unwrap();
    let rest = df!("x" => [6.0, 7.0, 8.0, 9.0], "y" => [12.2, 13.8, 16.1, 18.0]).unwrap();
    let mut whole = Bank::new(vec![s]).unwrap();
    whole.fit_predict(&first).unwrap();
    let bytes = whole.save_bytes().unwrap();
    let b = reencoded(&bytes, |f| {
        let p = &mut f.states[0][0].1.persisted;
        assert!(!p.pending.is_empty(), "rows are held");
        p.score_pred
            .iter_mut()
            .for_each(std::collections::VecDeque::clear);
    });
    let mut emptied = Bank::load_bytes(&b, None).unwrap();
    let a = whole.fit_predict(&rest).unwrap();
    let e = emptied.fit_predict(&rest).unwrap();
    let field = |out: &[Column], name: &str| {
        out[0]
            .struct_()
            .unwrap()
            .field_by_name(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(
        field(&a, "pred_y"),
        field(&e, "pred_y"),
        "the model learned the same rows"
    );
    let resid_w = |bank: &Bank| {
        let f: BankFile = rmp_serde::from_slice(&bank.save_bytes().unwrap()).unwrap();
        f.states[0][0].1.persisted.resid_w[0][0]
    };
    assert!(
        resid_w(&emptied) < resid_w(&whole),
        "the held rows folded: {} against {}",
        resid_w(&emptied),
        resid_w(&whole)
    );
}

/// A closed row the rest of the file contradicts: its spec closes no
/// group, it carries a Gram its spec's model keeps none of, its per-target
/// lists are of another length than the targets it names, or its lags have
/// no co-moments beside them. Each is refused at load, naming the damage.
#[test]
fn a_closed_row_its_spec_or_its_own_gram_contradicts_is_refused() {
    let (s, bytes) = closing();
    fn gram(f: &mut BankFile) -> &mut super::Gram {
        f.closed[0].gram.as_mut().expect("ewridge keeps a Gram")
    }
    type Edit = fn(&mut BankFile);
    let cases: Vec<(&str, Edit)> = vec![
        ("which closes no group", |f| f.specs[0].group_close = None),
        ("carries per-target lists of another length", |f| {
            gram(f).target_weights.push(1.0)
        }),
        (
            "carries lag co-moments that are not one 2 x 2 block per lag",
            |f| {
                gram(f).lags = Some(vec![1]);
            },
        ),
    ];
    for (want, edit) in cases {
        let b = reencoded(&bytes, edit);
        let err = Bank::load_bytes(&b, None).err().expect(want);
        assert!(
            err.starts_with("bank state file is damaged: a closed row of group 1")
                && err.contains(want),
            "{want}: {err}"
        );
    }
    // A null group is named as one.
    let b = reencoded(&bytes, |f| {
        f.specs[0].group_close = None;
        f.closed[0].group = GroupKey(None);
    });
    let err = Bank::load_bytes(&b, None)
        .err()
        .expect("a null group's row");
    assert!(
        err.contains("a closed row of group <null group> belongs to spec \"m\""),
        "{err}"
    );
    // A Gram on the closed row of a spec whose model keeps none.
    let sgd = spec(
        r#"{"name": "m", "model": {"type": "sgd"}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0,
            "group_close": "monotone"}"#,
    );
    let mut bank = Bank::new(vec![sgd]).unwrap();
    bank.fit_predict(
        &df!("g" => [1i64, 1, 2, 2], "x" => [1.0, 2.0, 1.0, 2.0], "y" => [2.0, 4.0, 2.0, 4.0])
            .unwrap(),
    )
    .unwrap();
    let ridge: BankFile = rmp_serde::from_slice(&bytes).unwrap();
    let g = ridge.closed[0].gram.clone();
    let b = reencoded(&bank.save_bytes().unwrap(), move |f| {
        assert!(f.closed[0].gram.is_none(), "sgd keeps no Gram");
        f.closed[0].gram = g;
    });
    let err = Bank::load_bytes(&b, None)
        .err()
        .expect("a Gram on sgd's row");
    assert!(
        err.contains("carries a Gram, and spec \"m\" keeps none"),
        "{err}"
    );
    assert!(Bank::load_bytes(&bytes, Some(std::slice::from_ref(&s))).is_ok());
}

/// An envelope newer than this build's is told to upgrade, by number.
#[test]
fn a_file_of_a_newer_format_is_refused_by_number() {
    let (_, bytes) = closing();
    let newer = super::BANK_FORMAT_VERSION + 1;
    let b = reencoded(&bytes, |f| f.format_version = newer);
    let err = Bank::load_bytes(&b, None).err().expect("loaded");
    assert_eq!(
        err,
        format!(
            "bank state file format version {newer} is newer than this build supports ({})",
            super::BANK_FORMAT_VERSION
        )
    );
}

/// A high-water mark that is not an integer, under an integer group
/// column, is a mark saved under another key column: the next chunk is
/// refused, naming the mark, rather than read against a key it does not
/// order with.
#[test]
fn a_high_water_mark_that_is_not_an_integer_is_refused_under_an_integer_key() {
    let (_, bytes) = closing();
    let b = reencoded(&bytes, |f| {
        assert_eq!(f.high_water.len(), 1, "group 1 closed");
        f.high_water[0].1 = GroupKey(Some("one".into()));
    });
    let mut bank = Bank::load_bytes(&b, None).unwrap();
    let next = df!("g" => [3i64], "x" => [1.0], "y" => [2.0]).unwrap();
    let err = bank.fit_predict(&next).unwrap_err().to_string();
    assert!(
        err.contains(
            "the saved high-water key one is not an integer, and the group column \"g\" is"
        ),
        "{err}"
    );
}

/// The JSON export is checked against the state it came from: a text that
/// drops a value fails to read back, and the failure shows the lines around
/// the place it failed, the failing one marked; a text that reads back as
/// another state is refused as not faithful.
#[test]
fn a_json_export_that_is_not_the_state_is_refused_with_its_place() {
    let (_, bytes) = closing();
    let file: BankFile = rmp_serde::from_slice(&bytes).unwrap();
    let text = serde_json::to_string_pretty(&file).unwrap();
    super::json_is_faithful(&file, &text).unwrap();
    // The package version, a string, written as null: the read fails there.
    let line = text
        .lines()
        .position(|l| l.contains("\"package_version\""))
        .expect("the export names the package version")
        + 1;
    let broken = text
        .lines()
        .map(|l| {
            if l.contains("\"package_version\"") {
                "  \"package_version\": null,".to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let err = super::json_is_faithful(&file, &broken).unwrap_err();
    assert!(
        err.starts_with("the JSON export dropped a value the state holds")
            && err.contains(", at:\n")
            && err.contains("\n  >   \"package_version\": null,")
            && err.contains("without it serde_json writes NaN and +/-inf as null"),
        "{err}"
    );
    assert_eq!(super::near(&broken, 0), "", "no line, no context");
    let shown = super::near(&broken, line);
    assert_eq!(shown.lines().count(), 1 + line.min(4), "{shown}");
    // Another state's text reads back, and is not this state.
    let mut other = rmp_serde::from_slice::<BankFile>(&bytes).unwrap();
    other.rows_fed += 1;
    let err = super::json_is_faithful(&file, &serde_json::to_string(&other).unwrap()).unwrap_err();
    assert!(
        err.starts_with("the JSON export did not read back"),
        "{err}"
    );
}

/// A windowed spread's ring kept for another number of slots is damage:
/// refused at load, where it would index past this spec's slots at the
/// next row.
#[test]
fn a_windowed_spreads_ring_of_another_width_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "window_size": 4.0}, "targets": ["y"],
            "features": ["x"], "group": "g", "half_life": 10.0, "emit_sigma": true}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&frame()).unwrap();
    let bytes = bank.save_bytes().unwrap();
    // Through msgpack's own tree: the ring's fields are its module's.
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    fn key<'a>(v: &'a mut rmpv::Value, k: &str) -> &'a mut rmpv::Value {
        let rmpv::Value::Map(m) = v else {
            panic!("not a map at {k}")
        };
        &mut m
            .iter_mut()
            .find(|(name, _)| name.as_str() == Some(k))
            .unwrap_or_else(|| panic!("no {k}"))
            .1
    }
    fn at(v: &mut rmpv::Value, i: usize) -> &mut rmpv::Value {
        let rmpv::Value::Array(a) = v else {
            panic!("not a list at {i}")
        };
        &mut a[i]
    }
    let state = at(at(at(key(&mut v, "states"), 0), 0), 1);
    let ring = at(key(key(state, "persisted"), "resid_win"), 0);
    let rmpv::Value::Array(rows) = key(ring, "rows") else {
        panic!("rows is a list")
    };
    assert_eq!(rows.len(), 1, "one slot");
    rows.push(rmpv::Value::from(0u64));
    let mut b = Vec::new();
    rmpv::encode::write_value(&mut b, &v).unwrap();
    let err = Bank::load_bytes(&b, None).err().expect("loaded");
    assert!(
        err.contains("saved state's residual window does not fit this spec"),
        "{err}"
    );
    assert!(Bank::load_bytes(&bytes, None).is_ok());
}
