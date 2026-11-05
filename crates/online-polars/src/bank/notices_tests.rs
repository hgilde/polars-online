//! A call that learned rows and then returned an error leaves its streams'
//! notices in them; the next call that completes says them, once (task
//! 332). No input reaches such a return -- it is an output the bank could
//! not assemble -- so the test makes one, by a spec's name, once armed.
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Bank, Spec};
use polars::prelude::*;

static ARMED: AtomicBool = AtomicBool::new(false);

/// Whether this call returns an error once its streams have learned: once,
/// when armed, for a bank with a spec named for it.
pub(super) fn fail_after_learning(specs: &[Spec]) -> bool {
    specs
        .iter()
        .any(|s| s.name.starts_with("fails-after-learning-"))
        && ARMED.swap(false, Ordering::SeqCst)
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
    let spec: Spec = serde_json::from_str(
        r#"{"name": "fails-after-learning-a", "model": {"type": "ewridge", "ridge": 0.5},
            "targets": ["y"], "features": ["x0", "x1", "x2"],
            "half_life": 4.0, "coef_every": 1, "group": "g"}"#,
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.fit_predict(&rows(1, 0, 19)).unwrap();
    bank.fit_predict(&rows(2, 0, 2)).unwrap();
    assert!(bank.take_notices().is_empty());
    // Group 1's row 19 is learned, and the call fails after it.
    ARMED.store(true, Ordering::SeqCst);
    assert!(bank.fit_predict(&rows(1, 19, 1)).is_err());
    assert!(
        !ARMED.load(Ordering::SeqCst),
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
