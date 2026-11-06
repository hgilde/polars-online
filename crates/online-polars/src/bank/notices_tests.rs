//! A call that learned rows and then returned an error leaves its streams'
//! notices in them; the next call that completes says them, once (task
//! 332), and so does a path that removes such a stream before that call
//! (task 333): the monotone close of an earlier group, and `drop_groups`.
//! No input reaches such a return -- it is an output the bank could not
//! assemble -- so the test makes one, by the bank's own hook
//! (`Bank::fail_next_call_after_learning`), and since schema 67 the
//! notices it leaves are saved with their streams (task 340).
use crate::{Bank, GroupKey, Spec};
use polars::prelude::*;

/// The bank's next learning call fails once its streams have learned.
fn arm(bank: &mut Bank) {
    bank.fail_next_call_after_learning();
}

/// Whether the failure armed was reached (and so disarmed).
fn reached(bank: &Bank) -> bool {
    !bank.fail_after_learning
}

fn spec(name: &str, extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "{name}", "model": {{"type": "ewridge", "ridge": 0.5}},
            "targets": ["y"], "features": ["x0", "x1", "x2"],
            "half_life": 4.0, "coef_every": 1, "group": "g"{extra}}}"#
    ))
    .unwrap()
}

/// Rows `from..from + n` of group `g`: under a ridge of 0.5 a duplicated
/// feature reads a support below 0.5, and at a half-life of 4 rows the
/// group is 95% settled from its row 19 (0-based), where the support
/// notice is said.
fn rows(g: i64, from: usize, n: usize) -> DataFrame {
    let x0: Vec<f64> = (from..from + n)
        .map(|i| ((i * 7) % 11) as f64 - 5.0)
        .collect();
    let x1: Vec<f64> = (from..from + n)
        .map(|i| ((i * 5) % 9) as f64 - 4.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| x0[i] - x1[i] + (((from + i) * 3) % 7) as f64 * 0.1)
        .collect();
    df!("g" => vec![g; n], "x0" => &x0, "x1" => &x1, "x2" => &x0, "y" => &y).unwrap()
}

#[test]
fn a_notice_left_by_a_call_that_failed_after_learning_is_said_once() {
    let mut bank = Bank::new(vec![spec("fails-after-learning-a", "")]).unwrap();
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    bank.fit_predict(&rows(2, 0, 2)).unwrap();
    assert!(bank.take_notices().is_empty());
    // Group 1's row 19 is learned, and the call fails after it.
    arm(&mut bank);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(&bank), "the call reached the failure");
    assert!(bank.take_notices().is_empty());
    // A call to another group says it.
    bank.fit_predict(&rows(2, 2, 1)).unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    assert!(
        said[0].starts_with("spec \"fails-after-learning-a\" group \"1\":")
            && said[0].contains("support_coef < 0.5"),
        "{said:#?}"
    );
    // Once.
    bank.fit_predict(&rows(1, 20, 2)).unwrap();
    bank.fit_predict(&rows(2, 3, 1)).unwrap();
    assert!(bank.take_notices().is_empty());
}

/// Under `group_close = "monotone"`, a group whose notice a failed call
/// left in its stream, closed by the next call's greater key: the next call
/// says it, once, before the close removes the stream.
#[test]
fn a_notice_left_in_a_group_the_next_call_closes_is_said_once() {
    let name = "fails-after-learning-monotone";
    let mut bank = Bank::new(vec![spec(name, r#", "group_close": "monotone""#)]).unwrap();
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    assert!(bank.take_notices().is_empty());
    arm(&mut bank);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(&bank), "the call reached the failure");
    assert!(bank.take_notices().is_empty());
    // Group 2's first row closes group 1.
    bank.fit_predict(&rows(2, 0, 1)).unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    assert!(
        said[0].starts_with(&format!("spec {name:?} group \"1\":"))
            && said[0].contains("support_coef < 0.5"),
        "{said:#?}"
    );
    assert_eq!(bank.closed_groups(None, true).unwrap().height(), 1);
    bank.fit_predict(&rows(2, 1, 2)).unwrap();
    assert!(bank.take_notices().is_empty());
}

/// A group whose notice a failed call left in its stream, dropped before
/// any call completes: the drop says it, once, and the dropped group,
/// seen again, starts cold and says its own at its own row 19.
#[test]
fn a_notice_left_in_a_dropped_group_is_said_once() {
    let name = "fails-after-learning-drop";
    let mut bank = Bank::new(vec![spec(name, "")]).unwrap();
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    arm(&mut bank);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(&bank), "the call reached the failure");
    let one = GroupKey(Some("1".to_string()));
    assert_eq!(bank.drop_groups(&[one], None), Ok(1));
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    assert!(
        said[0].starts_with(&format!("spec {name:?} group \"1\":")),
        "{said:#?}"
    );
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    assert!(bank.take_notices().is_empty());
    bank.fit_predict(&rows(1, 19, 1)).unwrap();
    assert_eq!(bank.take_notices().len(), 1);
}

/// A failed call's notices outlive a save (schema 67, task 340): the
/// notices it left are saved with their streams (`Notified::pending`), and
/// a loaded bank's first completing call takes every stream's, as the next
/// call after the failure does in the bank that failed. Before 67 only the
/// flag that a notice was raised was saved, its text not, so a bank saved
/// and loaded after such a call never said it. Both banks say both groups'
/// at their next completing call, whichever group it feeds, in the groups'
/// order, and once.
#[test]
fn a_failed_calls_notices_outlive_a_save() {
    let name = "fails-after-learning-saved";
    let mut bank = Bank::new(vec![spec(name, "")]).unwrap();
    let both = |from: usize, n: usize| {
        let mut df = rows(1, from, n);
        df.vstack_mut(&rows(2, from, n)).unwrap();
        df
    };
    bank.fit_predict(&both(0, 19)).unwrap();
    assert!(bank.take_notices().is_empty());
    // Both groups' row 19 learned, and the call fails after it.
    arm(&mut bank);
    assert!(bank.fit_predict(&both(19, 1)).is_err());
    assert!(reached(&bank), "the call reached the failure");
    let held = |bank: &Bank| -> usize {
        bank.states
            .iter()
            .flat_map(|hm| hm.values())
            .flat_map(|s| &s.persisted.notified)
            .map(|n| n.pending.len())
            .sum()
    };
    assert_eq!(held(&bank), 2, "each group's stream holds its notice");
    let bytes = bank.save_bytes().unwrap();
    assert!(
        String::from_utf8_lossy(&bytes).contains("support_coef < 0.5"),
        "the notice's text is in the file"
    );

    let mut loaded = Bank::load_bytes(&bytes, None).unwrap();
    assert_eq!(held(&loaded), 2, "the notices survive the save");
    assert!(loaded.notices_left, "its first call takes every stream's");
    let expect_both = |said: &[String]| {
        assert_eq!(said.len(), 2, "{said:#?}");
        assert!(said[0].starts_with(&format!("spec {name:?} group \"1\":")));
        assert!(said[1].starts_with(&format!("spec {name:?} group \"2\":")));
        assert!(said.iter().all(|s| s.contains("support_coef < 0.5")));
    };
    // The loaded bank's first call feeds group 1 alone and says both.
    loaded.fit_predict(&rows(1, 20, 1)).unwrap();
    expect_both(&loaded.take_notices());
    assert!(!loaded.notices_left);
    assert_eq!(held(&loaded), 0);
    // Once: the next calls, to either group, say nothing more.
    loaded.fit_predict(&rows(2, 20, 1)).unwrap();
    loaded.fit_predict(&rows(1, 21, 1)).unwrap();
    assert!(loaded.take_notices().is_empty());
    // A save after they were said holds no notice.
    let again = loaded.save_bytes().unwrap();
    assert!(!String::from_utf8_lossy(&again).contains("support_coef < 0.5"));
    assert_eq!(held(&Bank::load_bytes(&again, None).unwrap()), 0);

    // The bank that failed says both at its next call, which feeds group 1
    // alone, as before.
    bank.fit_predict(&rows(1, 20, 1)).unwrap();
    expect_both(&bank.take_notices());
}

/// A stream saved with a pending notice by a build of schema 66 or
/// earlier has no `pending` list -- the notices were held in memory alone
/// -- and loads with none: the flags that they were raised are saved, so
/// such a bank says nothing for them, as it did. The file is the current
/// one with its `pending` lists removed (the clock's `pending`, a number,
/// stays) and every schema at 66, the header's and the model state's, the
/// one layout difference.
#[test]
fn a_bank_of_schema_66_loads_its_streams_with_no_pending_notice() {
    let name = "fails-after-learning-66";
    let mut bank = Bank::new(vec![spec(name, "")]).unwrap();
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    arm(&mut bank);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(&bank));
    let bytes = bank.save_bytes().unwrap();
    fn strip(v: &mut rmpv::Value, pending: &mut usize, schema: &mut usize) {
        match v {
            rmpv::Value::Map(m) => {
                let before = m.len();
                m.retain(|(k, v)| !(k.as_str() == Some("pending") && v.is_array()));
                *pending += before - m.len();
                for (k, v) in m.iter_mut() {
                    if k.as_str() == Some("schema_version") && v.as_u64() == Some(67) {
                        *v = rmpv::Value::from(66);
                        *schema += 1;
                    }
                    strip(v, pending, schema);
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(|v| strip(v, pending, schema)),
            _ => {}
        }
    }
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let (mut pending, mut schema) = (0, 0);
    strip(&mut v, &mut pending, &mut schema);
    assert_eq!(
        (pending, schema),
        (1, 2),
        "one pending list; the header's schema and the one model state's"
    );
    let mut old = Vec::new();
    rmpv::encode::write_value(&mut old, &v).unwrap();
    let mut loaded = Bank::load_bytes(&old, None).unwrap();
    let held: usize = loaded
        .states
        .iter()
        .flat_map(|hm| hm.values())
        .flat_map(|s| &s.persisted.notified)
        .map(|n| n.pending.len())
        .sum();
    assert_eq!(held, 0, "a 66 stream holds no notice");
    assert!(
        loaded.states[0]
            .values()
            .all(|s| s.persisted.notified[0].support),
        "and keeps that it was raised"
    );
    loaded.fit_predict(&rows(1, 20, 2)).unwrap();
    assert!(loaded.take_notices().is_empty(), "so it is never said");
}
