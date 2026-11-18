//! Task 460, B1: a call that runs inline on the caller's thread
//! (`bank/inline.rs`) must not reach rayon's global pool, whatever the number
//! of right-hand sides the model solves at once. `solve_in_place`
//! (`online-core/src/solve.rs`) spreads 256 or more right-hand sides over
//! `rayon::current_num_threads()` workers with `rayon::scope`; on a thread
//! that is on no pool, both name rayon's global pool, not the bank's.
//!
//! The test builds the global pool itself with two threads and keeps
//! both busy, in this process alone (every integration test file is its
//! own process). A call that hands work to the global pool then cannot
//! finish: the test gives it a deadline and reports which pool it was on.

use std::sync::mpsc;
use std::time::Duration;

use online_polars::Bank;
use polars::prelude::*;

const TARGETS: usize = 300;
/// Threads of the global pool: two, so that a spread has a second worker.
const GLOBAL: usize = 2;

fn bank_and_frame(threads: usize) -> (Bank, DataFrame) {
    let ys: Vec<String> = (0..TARGETS).map(|i| format!("\"y{i}\"")).collect();
    let spec = serde_json::from_str(&format!(
        r#"{{"name": "wide", "model": {{"type": "ewridge", "ridge": 1e-6,
            "max_rows_between_solves": 1}}, "targets": [{}], "features": ["x0"],
            "half_life": 10.0}}"#,
        ys.join(", ")
    ))
    .unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.set_threads(Some(threads)).unwrap();
    let n = 4usize;
    let mut cols = vec![Column::new(
        "x0".into(),
        (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
    )];
    for j in 0..TARGETS {
        cols.push(Column::new(
            format!("y{j}").into(),
            (0..n)
                .map(|i| ((i * (j + 1)) % 7) as f64)
                .collect::<Vec<_>>(),
        ));
    }
    (bank, DataFrame::new(n, cols).unwrap())
}

#[test]
fn a_wide_inline_call_stays_off_the_global_pool() {
    // The global pool: every thread occupied until released.
    rayon::ThreadPoolBuilder::new()
        .num_threads(GLOBAL)
        .thread_name(|i| format!("global-{i}"))
        .build_global()
        .unwrap();
    let mut holds = Vec::new();
    for _ in 0..GLOBAL {
        let (hold, release) = mpsc::channel::<()>();
        rayon::spawn(move || {
            let _ = release.recv();
        });
        holds.push(hold);
    }

    let (mut bank, frame) = bank_and_frame(1);
    let (done, finished) = mpsc::channel();
    let runner = std::thread::Builder::new()
        .name("caller".into())
        .spawn(move || {
            // The call is small: one ungrouped model, four rows.
            let out = bank.fit_predict(&frame).map(|_| ());
            let _ = done.send(out.is_ok());
        })
        .unwrap();
    let on_time = finished.recv_timeout(Duration::from_secs(20));
    // Let the global pool go whatever happened, so the runner can end.
    for hold in holds {
        let _ = hold.send(());
    }
    runner.join().unwrap();
    assert_eq!(
        on_time,
        Ok(true),
        "the call with {TARGETS} right-hand sides and threads=1 waited on rayon's global pool"
    );
}
