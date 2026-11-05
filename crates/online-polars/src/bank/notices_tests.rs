//! A call that learned rows and then returned an error leaves its streams'
//! notices in them; the next call that completes says them, once (task
//! 332), and so does a path that removes such a stream before that call
//! (task 333): the monotone close of an earlier group, and `drop_groups`.
//! No input reaches such a return -- it is an output the bank could not
//! assemble -- so the test makes one, by a spec's name, once armed.
use std::sync::Mutex;

use crate::{Bank, GroupKey, Spec};
use polars::prelude::*;

/// The names of the specs whose bank's next call fails after learning:
/// one name a test, so tests run side by side do not take each other's.
static ARMED: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn armed() -> std::sync::MutexGuard<'static, Vec<String>> {
    ARMED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether this call returns an error once its streams have learned: once,
/// when armed, for a bank with a spec named for it.
pub(super) fn fail_after_learning(specs: &[Spec]) -> bool {
    let mut armed = armed();
    let before = armed.len();
    armed.retain(|name| !specs.iter().any(|s| &s.name == name));
    armed.len() < before
}

fn arm(name: &str) {
    armed().push(name.to_string());
}

/// Whether the failure armed for `name` was reached (and so disarmed).
fn reached(name: &str) -> bool {
    !armed().iter().any(|n| n == name)
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
    arm("fails-after-learning-a");
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(
        reached("fails-after-learning-a"),
        "the call reached the failure"
    );
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
    arm(name);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(name), "the call reached the failure");
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
    arm(name);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(reached(name), "the call reached the failure");
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
