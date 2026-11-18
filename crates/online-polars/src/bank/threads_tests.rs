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
const PROBED: [&str; 3] = [
    "threads-probe-one",
    "threads-probe-two",
    "threads-probe-small",
];

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
        err.contains("threads: 0 is not a number of threads"),
        "{err}"
    );
    assert_eq!(bank.threads().asked(), Some(3));
    bank.set_threads(None).unwrap();
    assert_eq!(bank.threads().asked(), None);
}

/// A `threads` above the bound one pool may have is refused before a pool
/// is built, and the bank keeps the pool it had (task 356).
#[test]
fn a_threads_above_the_bound_is_refused_and_the_bank_keeps_its_pool() {
    let mut bank = Bank::new(vec![spec("m")]).unwrap();
    bank.set_threads(Some(3)).unwrap();
    let err = bank.set_threads(Some(1_000_000)).unwrap_err().to_string();
    assert!(
        err.contains("threads: 1000000 is more threads than one pool may have"),
        "{err}"
    );
    assert!(!Threads::registry_holds(1_000_000));
    assert_eq!(bank.threads().asked(), Some(3));
}

// ---- `"auto"`'s share of the pool (task 253, part 2) ----

/// Per probed spec name, the `gram_threads` each task's streams were set to.
static GRAM: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

/// The spec names whose Gram threads are recorded.
const GRAM_PROBED: [&str; 3] = ["gram-share-auto", "gram-share-count", "gram-share-none"];

/// Record the Gram threads `process` set for `spec`'s task, when probed.
pub(super) fn probe_gram(spec: &str, threads: usize) {
    if GRAM_PROBED.contains(&spec) {
        GRAM.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((spec.to_string(), threads));
    }
}

/// Take what was recorded for `spec`, emptying it.
fn gram_seen(spec: &str) -> Vec<usize> {
    let mut all = GRAM
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let got = all
        .iter()
        .filter(|(s, _)| s == spec)
        .map(|(_, t)| *t)
        .collect();
    all.retain(|(s, _)| s != spec);
    got
}

fn spec_gram(name: &str, gram: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "{name}", "model": {{"type": "ewridge"{gram}}},
            "targets": ["y"], "features": ["x0", "x1"], "group": "g", "half_life": 10.0}}"#
    ))
    .unwrap()
}

fn frame_of(groups: usize) -> DataFrame {
    let n = 4 * 64usize;
    let g: Vec<i64> = (0..n).map(|i| (i % groups) as i64).collect();
    let x0: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 3) % 11) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

#[test]
fn the_share_is_one_until_half_the_pool_is_idle_then_twice_the_even_share() {
    use crate::pool::gram_share;
    assert_eq!(gram_share(1, 14), 14);
    assert_eq!(gram_share(2, 14), 14);
    assert_eq!(gram_share(3, 14), 9);
    assert_eq!(gram_share(4, 14), 7);
    assert_eq!(gram_share(7, 14), 4);
    assert_eq!(gram_share(8, 14), 1);
    assert_eq!(gram_share(13, 14), 1);
    assert_eq!(gram_share(14, 14), 1);
    assert_eq!(gram_share(28, 14), 1);
    assert_eq!(gram_share(64, 14), 1);
    assert_eq!(gram_share(4, 8), 4);
    assert_eq!(gram_share(5, 8), 1);
    assert_eq!(gram_share(1, 2), 2);
    assert_eq!(gram_share(2, 2), 1);
    assert_eq!(gram_share(0, 8), 1);
    assert_eq!(gram_share(1, 1), 1);
    assert_eq!(gram_share(5, 1), 1);
}

/// `"auto"` splits a group's update only when half the pool would be idle:
/// one group on 14 threads takes all 14, four take seven each, and 14, 28
/// or 64 groups take one each, from the rule and not from timing. A count
/// is the caller's whatever the share, and none is `"auto"`, the default.
#[test]
fn auto_takes_the_pool_share_of_each_task() {
    for (groups, want) in [(1usize, 14usize), (4, 7), (14, 1), (28, 1), (64, 1)] {
        let mut bank = Bank::new(vec![
            spec_gram("gram-share-auto", r#", "gram_threads": "auto""#),
            spec_gram("gram-share-count", r#", "gram_threads": 5"#),
            spec_gram("gram-share-none", ""),
        ])
        .unwrap();
        bank.set_threads(Some(14)).unwrap();
        bank.fit_predict(&frame_of(groups)).unwrap();
        // Three specs of `groups` tasks each share the pool.
        let share = crate::pool::gram_share(3 * groups, 14);
        let auto = gram_seen("gram-share-auto");
        assert_eq!(auto, vec![share; groups], "{groups} groups");
        assert_eq!(gram_seen("gram-share-count"), vec![5; groups]);
        assert_eq!(gram_seen("gram-share-none"), vec![share; groups]);
        // Alone in the bank, the spec has the share `want` names.
        let mut alone = Bank::new(vec![spec_gram(
            "gram-share-auto",
            r#", "gram_threads": "auto""#,
        )])
        .unwrap();
        alone.set_threads(Some(14)).unwrap();
        alone.fit_predict(&frame_of(groups)).unwrap();
        assert_eq!(
            gram_seen("gram-share-auto"),
            vec![want; groups],
            "{groups} groups alone"
        );
    }
}

/// A chunk's transpose into rows spreads over the pool for a lone spec
/// with features, and runs on each spec's own thread where several are
/// extracted at once (task 257): a spec without features does not count.
#[test]
fn the_transpose_spreads_only_for_a_lone_spec_with_features() {
    use super::transposes_in_parallel;
    let one = spec("t-one");
    let mut bare = spec("t-bare");
    bare.features.clear();
    assert!(transposes_in_parallel(std::slice::from_ref(&one)));
    assert!(transposes_in_parallel(&[one.clone(), bare.clone()]));
    assert!(!transposes_in_parallel(&[one.clone(), spec("t-two")]));
    assert!(transposes_in_parallel(&[]));
}

/// A spec of two features over four groups, small enough to run inline.
fn small_spec(name: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "{name}", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x0", "x1"], "group": "g", "half_life": 10.0}}"#
    ))
    .unwrap()
}

fn small_frame(seed: usize) -> DataFrame {
    let n = 12usize;
    let g: Vec<i64> = (0..n).map(|i| (i % 4) as i64).collect();
    let x0: Vec<f64> = (0..n).map(|i| ((i + seed) % 7) as f64 + 0.5).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 3 + seed) % 11) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i + 2 * seed) % 5) as f64).collect();
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// A call too small for the pool runs every task on the calling thread, and
/// one at the cut or above it is handed to the pool.
#[test]
fn a_small_call_runs_on_the_calling_thread_and_a_large_one_on_the_pool() {
    let name = PROBED[2];
    let mut bank = Bank::new(vec![small_spec(name)]).unwrap();
    bank.set_threads(Some(2)).unwrap();
    assert!(bank.runs_inline(12));
    bank.fit_predict(&small_frame(0)).unwrap();
    bank.predict(&small_frame(1)).unwrap();
    let caller = std::thread::current().name().unwrap_or("").to_string();
    assert_eq!(seen(name), BTreeSet::from([caller]));
    // Past the work cut (here 4 groups of a 2-feature spec: 9 a row), and
    // from the rows at which a chunk is spread.
    assert!(bank.runs_inline(100));
    assert!(!bank.runs_inline(200));
    assert!(!bank.runs_inline(crate::PAR_MIN_ROWS));
    // On a pool's own thread the call stays on that pool.
    let pool = bank.threads().pool().unwrap();
    assert!(!pool.install(|| bank.runs_inline(12)));
}

/// One ungrouped single-instance spec has one task whatever the chunk: it
/// runs inline at every size below the rows at which a chunk is spread. A
/// grid of instances, a count of Gram threads and a wide spec do not.
#[test]
fn what_runs_inline_is_decided_by_tasks_work_and_width() {
    let one = |extra: &str, features: &str| -> Bank {
        let s: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"{extra}}}, "targets": ["y"],
                "features": [{features}], "half_life": 10.0}}"#
        ))
        .unwrap();
        Bank::new(vec![s]).unwrap()
    };
    assert!(one("", r#""x0", "x1""#).runs_inline(crate::PAR_MIN_ROWS - 1));
    assert!(!one("", r#""x0", "x1""#).runs_inline(crate::PAR_MIN_ROWS));
    let wide: Vec<String> = (0..=super::inline::INLINE_MAX_FEATURES)
        .map(|i| format!("\"x{i}\""))
        .collect();
    assert!(!one("", &wide.join(", ")).runs_inline(1));
    assert!(!one(r#", "gram_threads": 2"#, r#""x0""#).runs_inline(1));
    assert!(one(r#", "gram_threads": 1"#, r#""x0""#).runs_inline(1));
    let grid: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0", "x1"], "half_life": [10.0, 20.0, 40.0]}"#,
    )
    .unwrap();
    let grid = Bank::new(vec![grid]).unwrap();
    assert!(grid.runs_inline(10) && !grid.runs_inline(1000));
}

/// The same chunks give the same bits run inline and on the pool, in the
/// outputs of learning and of scoring.
#[test]
fn a_small_call_gives_the_bits_the_pool_gives() {
    let specs = || vec![small_spec("a"), small_spec("b")];
    let (mut inline, mut pooled) = (Bank::new(specs()).unwrap(), Bank::new(specs()).unwrap());
    pooled.set_threads(Some(2)).unwrap();
    for seed in 0..6 {
        let chunk = small_frame(seed);
        assert!(inline.runs_inline(chunk.height()));
        let a = inline.fit_predict(&chunk).unwrap();
        let pool = pooled.threads().clone();
        let b = pool
            .install(|| {
                assert!(!pooled.runs_inline(chunk.height()));
                pooled.fit_predict(&chunk)
            })
            .unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "chunk {seed}");
        let probe = small_frame(seed + 20);
        let a = inline.predict(&probe).unwrap();
        let b = pool.install(|| pooled.predict(&probe)).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "predict {seed}");
    }
}

/// Task 458: a chunk is read in parallel from `PAR_MIN_ROWS` rows, or from
/// as many feature values as `PAR_MIN_CELLS`, which a wide chunk of a few
/// hundred rows reaches first. A chunk one value short of it and one at it,
/// of a bank on the pool: the bits of a bank with one thread.
#[test]
fn a_wide_chunk_is_read_the_same_each_side_of_the_cell_cut() {
    let width = 80usize;
    let features: Vec<String> = (0..width).map(|i| format!("\"x{i}\"")).collect();
    let spec = || -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "wide", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": [{}], "half_life": 50.0}}"#,
            features.join(", ")
        ))
        .unwrap()
    };
    let frame = |n: usize, seed: u64| {
        let mut st = seed;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut cols: Vec<Column> = (0..width)
            .map(|i| {
                Column::new(
                    format!("x{i}").into(),
                    (0..n).map(|_| next()).collect::<Vec<f64>>(),
                )
            })
            .collect();
        cols.push(Column::new(
            "y".into(),
            (0..n).map(|_| next()).collect::<Vec<f64>>(),
        ));
        DataFrame::new(n, cols).unwrap()
    };
    // `PAR_MIN_CELLS` is not a multiple of the width: the rows either side
    // of it hold one value short of it and 80 over.
    let below = (super::PAR_MIN_CELLS - 1) / width;
    assert!(below * width < super::PAR_MIN_CELLS && (below + 1) * width >= super::PAR_MIN_CELLS);
    assert!(below + 1 < crate::PAR_MIN_ROWS);
    for n in [below, below + 1] {
        let (mut one, mut many) = (
            Bank::new(vec![spec()]).unwrap(),
            Bank::new(vec![spec()]).unwrap(),
        );
        one.set_threads(Some(1)).unwrap();
        many.set_threads(Some(3)).unwrap();
        for seed in [1u64, 2] {
            let chunk = frame(n, seed + n as u64);
            let a = one.fit_predict(&chunk).unwrap();
            let pool = many.threads().clone();
            let b = pool.install(|| many.fit_predict(&chunk)).unwrap();
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "{n} rows, chunk {seed}");
        }
    }
}

/// A small call runs on the caller's own stack, which a thread of another
/// program may have made small: it needs little.
#[test]
fn a_small_call_fits_a_half_megabyte_stack() {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let mut bank = Bank::new(vec![small_spec("s")]).unwrap();
            for seed in 0..3 {
                bank.fit_predict(&small_frame(seed)).unwrap();
            }
            bank.predict(&small_frame(9)).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
