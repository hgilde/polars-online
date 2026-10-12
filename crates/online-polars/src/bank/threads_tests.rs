//! A bank's `threads` (task 253): the pool a bank runs on, seen from the
//! inside. A test-only probe in `process` and `score` records the thread
//! each (spec, group) task runs on, for the specs named here only, so the
//! other tests of this binary, running beside these, add nothing to it.
use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::{Bank, Spec, Threads};
use polars::prelude::*;

/// Per probed spec name, the threads its tasks ran on.
static SEEN: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// The spec names this module probes; every other spec is ignored.
const PROBED: [&str; 2] = ["threads-probe-one", "threads-probe-two"];

/// Record the current thread for `spec`, when it is one this module probes.
pub(super) fn probe(spec: &str) {
    if PROBED.contains(&spec) {
        let name = std::thread::current().name().unwrap_or("").to_string();
        SEEN.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((spec.to_string(), name));
    }
}

fn seen(spec: &str) -> BTreeSet<String> {
    SEEN.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(s, _)| s == spec)
        .map(|(_, n)| n.clone())
        .collect()
}

fn spec(name: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "{name}", "model": {{"type": "ewridge", "gram_threads": "auto"}},
            "targets": ["y"], "features": ["x0", "x1"], "group": "g", "half_life": 10.0}}"#
    ))
    .unwrap()
}

/// 64 groups, so that a pool of more than one thread would spread them.
fn frame() -> DataFrame {
    let n = 2048usize;
    let g: Vec<i64> = (0..n).map(|i| (i % 64) as i64).collect();
    let x0: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 3) % 11) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// A bank of one thread runs every task, learning and scoring, on that one
/// thread, never on rayon's global pool or the process default.
#[test]
fn a_bank_of_one_thread_runs_every_task_on_its_one_thread() {
    let name = PROBED[0];
    let mut bank = Bank::new(vec![spec(name)]).unwrap();
    bank.set_threads(Some(1)).unwrap();
    let df = frame();
    for chunk in [df.slice(0, 1000), df.slice(1000, 1048)] {
        bank.fit_predict(&chunk).unwrap();
    }
    bank.predict(&df).unwrap();
    let names = seen(name);
    assert_eq!(names.len(), 1, "{names:?}");
    let only = names.iter().next().unwrap();
    // A pool of its own (`polars-online-1t-0`), or the process default
    // when that has one thread (`polars-online-0`).
    assert!(
        only == "polars-online-1t-0" || only == "polars-online-0",
        "{only}"
    );
}

/// A bank of two runs on two threads of a pool of its size, and on nothing
/// else.
#[test]
fn a_bank_of_two_threads_runs_on_those_two() {
    let name = PROBED[1];
    let mut bank = Bank::new(vec![spec(name)]).unwrap();
    bank.set_threads(Some(2)).unwrap();
    assert_eq!(bank.threads().asked(), Some(2));
    assert_eq!(bank.threads().pool().unwrap().current_num_threads(), 2);
    bank.fit_predict(&frame()).unwrap();
    let names = seen(name);
    assert!(!names.is_empty() && names.len() <= 2, "{names:?}");
    // A pool of its own, unless the process default has two threads.
    let default = std::ptr::eq(bank.threads().pool().unwrap(), crate::pool::pool().unwrap());
    let want = if default {
        ["polars-online-0", "polars-online-1"]
    } else {
        ["polars-online-2t-0", "polars-online-2t-1"]
    };
    for n in &names {
        assert!(
            want.contains(&n.as_str()),
            "{n}: a task ran off the bank's pool"
        );
    }
}

/// Two banks asking for one count share one pool; a bank keeps its pool
/// until it is dropped, and the pool goes with the last bank holding it.
#[test]
fn banks_of_one_count_share_a_pool_that_goes_with_the_last() {
    // A count no other test here asks for, and not the default's size.
    let n = if crate::pool::pool().unwrap().current_num_threads() == 11 {
        13
    } else {
        11
    };
    let mut a = Bank::new(vec![spec("m")]).unwrap();
    let mut b = Bank::new(vec![spec("m")]).unwrap();
    a.set_threads(Some(n)).unwrap();
    b.set_threads(Some(n)).unwrap();
    assert!(std::ptr::eq(
        a.threads().pool().unwrap(),
        b.threads().pool().unwrap()
    ));
    a.fit_predict(&frame()).unwrap();
    drop(a);
    assert!(Threads::registry_holds(n), "b still holds it");
    drop(b);
    assert!(!Threads::registry_holds(n), "the last bank took it");
}

/// `threads = 0` is refused naming the argument, and the bank keeps the
/// pool it had.
#[test]
fn zero_threads_is_refused_and_the_bank_keeps_its_pool() {
    let mut bank = Bank::new(vec![spec("m")]).unwrap();
    bank.set_threads(Some(3)).unwrap();
    let err = bank.set_threads(Some(0)).unwrap_err().to_string();
    assert!(
        err.contains("threads=0 is not a number of threads"),
        "{err}"
    );
    assert_eq!(bank.threads().asked(), Some(3));
    bank.set_threads(None).unwrap();
    assert_eq!(bank.threads().asked(), None);
}
