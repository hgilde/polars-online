//! Every cell of the scaling table (`online_bench::scaling`), measured at its
//! two sizes and held to the growth its algorithm declares: a row's work by
//! its time and its allocations, the state by its saved bytes and the heap
//! it holds (docs/PERFORMANCE.md, "How cost grows").
//!
//! The counts are the allocator's, kept process-wide, so the cells take
//! turns under one lock: the tests of this binary may still be run side by
//! side, and nextest runs each in a process of its own.
//!
//! `SCALING_REPORT=1` prints every cell's ratios, the audit's table.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use online_bench::scaling::{
    CELLS, Cell, ScaleRun, polars_online_budget_scale_rows, scale_setup_blocks,
};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    let now = LIVE.fetch_add(by as isize, Ordering::Relaxed) + by as isize;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grew(new_size);
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static TURN: Mutex<()> = Mutex::new(());

/// Blocks timed at each size; the least of them is the cell's time.
const BLOCKS: usize = 3;

/// What one size of a cell measured.
#[derive(Debug, Clone, Copy)]
struct Measured {
    /// The least time of a block.
    time: Duration,
    /// Allocations over every block.
    calls: usize,
    /// The saved state's bytes, and the heap the state holds.
    saved: usize,
    heap: usize,
    /// The most a block held above what was live when it started.
    peak: usize,
    /// Rows a block steps.
    rows: usize,
}

/// The heap a run holds once its rows are dropped: what dropping it frees.
fn heap_of(mut run: ScaleRun) -> usize {
    run.drop_data();
    let held = LIVE.load(Ordering::Relaxed);
    drop(run);
    (held - LIVE.load(Ordering::Relaxed)).max(0) as usize
}

fn block(run: &mut ScaleRun) -> (Duration, usize, usize) {
    let pool = run
        .on_the_pool()
        .then(|| online_polars::pool().expect("the bank's pool"));
    let calls = CALLS.load(Ordering::Relaxed);
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    let at = Instant::now();
    let sum = match pool {
        Some(p) => p.install(|| polars_online_budget_scale_rows(run)),
        None => polars_online_budget_scale_rows(run),
    };
    let time = at.elapsed();
    std::hint::black_box(sum);
    let peak = (PEAK.load(Ordering::Relaxed) - live).max(0) as usize;
    (time, CALLS.load(Ordering::Relaxed) - calls, peak)
}

/// Both sizes of a cell, their blocks interleaved, so a change in the
/// machine's load falls on both alike.
fn measure(c: &Cell) -> (Measured, Measured) {
    let _turn = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut small = scale_setup_blocks(c.id, c.small, BLOCKS);
    let mut large = scale_setup_blocks(c.id, c.large(), BLOCKS);
    let mut out = [&small, &large].map(|run| Measured {
        time: Duration::MAX,
        calls: 0,
        saved: 0,
        heap: 0,
        peak: 0,
        rows: run.block_rows,
    });
    for _ in 0..BLOCKS {
        for (m, run) in out.iter_mut().zip([&mut small, &mut large]) {
            let (t, calls, peak) = block(run);
            m.time = m.time.min(t);
            m.calls += calls;
            m.peak = m.peak.max(peak);
        }
    }
    let is_run = small.is_run();
    // Each size is on the path its cell is for: a refit still waiting, a
    // cap filled, a window holding its rows.
    for run in [&small, &large] {
        run.on_its_path().unwrap_or_else(|e| panic!("{e}"));
    }
    out[0].saved = small.state_bytes();
    out[1].saved = large.state_bytes();
    out[1].heap = heap_of(large);
    out[0].heap = heap_of(small);
    // A run's memory is what it holds at its peak, not what is left.
    if is_run {
        for m in &mut out {
            m.heap = m.peak;
        }
    }
    (out[0], out[1])
}

fn ratio(a: f64, b: f64) -> f64 {
    b / a.max(f64::MIN_POSITIVE)
}

/// How far past its declared growth a cell's time may read: a factor of
/// `step^TIME_SLACK`. Time is the coarse backstop (the instruction counts on
/// CI are the fine one), and a wrong growth is a whole power of the step
/// past the declared one: 8x at the usual step of 8, where this allows 2.8x.
const TIME_SLACK: f64 = 0.5;

/// How far the deterministic counts -- allocations, saved bytes, heap -- may
/// read past the declared growth, as a share: what a vector's doubling and
/// a hash table's growth add at one size and not the other.
const COUNT_TOL: f64 = 0.25;

struct Verdict {
    line: String,
    failures: Vec<String>,
}

fn judge(c: &Cell) -> Verdict {
    let (s, l) = measure(c);
    let work = c.work.bound(c.small, c.large());
    let state = c.state.bound(c.small, c.large());
    let time_bound = work * (c.step as f64).powf(TIME_SLACK);
    // A row's work: a block's, over its rows (the same at both sizes but
    // for a run whose size is its rows).
    let per = |m: &Measured, v: f64| v / m.rows as f64;
    let t = ratio(per(&s, s.time.as_secs_f64()), per(&l, l.time.as_secs_f64()));
    // An allocation a block or two is nothing to scale: the floor is five
    // a block.
    let floor = (5 * BLOCKS) as f64;
    let a = ratio(
        per(&s, s.calls as f64 + floor),
        per(&l, l.calls as f64 + floor),
    );
    let saved = ratio(s.saved as f64, l.saved as f64);
    let heap = ratio(s.heap as f64 + 1024.0, l.heap as f64 + 1024.0);
    let mut failures = Vec::new();
    let mut check = |what: &str, got: f64, bound: f64| {
        if got > bound {
            failures.push(format!(
                "{}: {what} grew {got:.2}x from {} to {}, past the {:.2}x its declared growth allows",
                c.id,
                c.small,
                c.large(),
                bound
            ));
        }
    };
    check("a block's time", t, time_bound);
    check("a block's allocations", a, work * (1.0 + COUNT_TOL));
    check("the saved state", saved, state * (1.0 + COUNT_TOL));
    check("the state's heap", heap, state * (1.0 + COUNT_TOL));
    let line = format!(
        "{:<28} work {:<10} state {:<10} | time {:>7.2}x (<= {:>7.2}) {:>9.3} ms -> {:>9.3} ms | allocs {:>7.2}x ({} -> {}) | saved {:>6.2}x ({} -> {}) | heap {:>6.2}x ({} -> {}) {}",
        c.id,
        c.work.show("n"),
        c.state.show("n"),
        t,
        time_bound,
        s.time.as_secs_f64() * 1e3,
        l.time.as_secs_f64() * 1e3,
        a,
        s.calls,
        l.calls,
        saved,
        s.saved,
        l.saved,
        heap,
        s.heap,
        l.heap,
        if failures.is_empty() { "" } else { "FAIL" },
    );
    Verdict { line, failures }
}

/// Every cell whose id starts with one of `prefixes`, but those of an open
/// finding, each held to its declaration; and the findings' own tests, below.
fn hold_surfaces(prefixes: &[&str]) {
    let ids: Vec<&str> = CELLS
        .iter()
        .filter(|c| c.finding.is_none() && prefixes.iter().any(|p| c.id.starts_with(p)))
        .map(|c| c.id)
        .collect();
    assert!(!ids.is_empty(), "no cell starts with {prefixes:?}");
    hold(&ids);
}

fn hold(ids: &[&str]) {
    let mut failures = Vec::new();
    for id in ids {
        let v = judge(online_bench::scaling::cell_of(id));
        if std::env::var("SCALING_REPORT").as_deref() == Ok("1") {
            eprintln!("{}", v.line);
        }
        failures.extend(v.failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The surfaces each test below holds, by the prefix of their cells' ids.
const SURFACES: &[&[&str]] = &[
    &["ewridge.", "rls.", "lasso.", "kalman."],
    &["huber.", "quantile.", "sgd.", "pa.", "ftrl.", "holt."],
    &["ew_cov.", "marginal.", "deco.", "rcov.", "corrchange."],
    &[
        "hmm.",
        "kmeans.",
        "micro.",
        "ew_class.",
        "seqtest.",
        "bocpd.",
        "audit.",
    ],
    &["bank."],
    &["runner.", "windows.", "refresh.", "gbdt."],
];

#[test]
fn every_cell_is_held_by_one_test() {
    let mut ids: Vec<&str> = CELLS.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    let n = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), n, "a cell id twice");
    for c in CELLS {
        let held = SURFACES
            .iter()
            .filter(|ps| ps.iter().any(|p| c.id.starts_with(p)))
            .count();
        assert_eq!(held, 1, "{}: held by {held} tests", c.id);
        assert!(c.small >= 1 && c.step >= 2, "{}", c.id);
    }
}

/// Every model kind has its cells: one that holds a row's work and the state
/// to O(1) in the rows the stream has seen, and one for a size it has. A
/// kind added without them is in no row of the table (docs/EXTENDING.md).
/// `rcov` alone has no history cell: its block is closed by the bank, so a
/// model fed rows with no close holds every row by design, and its cells are
/// the block's length and the width.
#[test]
fn every_kind_has_its_cells() {
    for kind in online_polars::ModelKind::KINDS {
        let cells: Vec<&str> = CELLS
            .iter()
            .filter(|c| c.id.split_once('.').map(|(k, _)| k) == Some(*kind))
            .map(|c| c.id)
            .collect();
        let history = format!("{kind}.history");
        assert!(
            *kind == "rcov" || cells.contains(&history.as_str()),
            "{kind}: no {history} cell"
        );
        assert!(
            cells.iter().any(|c| !c.ends_with(".history")),
            "{kind}: no cell for a size the model has"
        );
    }
}

#[test]
fn the_linear_models_grow_as_declared() {
    hold_surfaces(SURFACES[0]);
}

#[test]
fn the_robust_and_gradient_models_grow_as_declared() {
    hold_surfaces(SURFACES[1]);
}

#[test]
fn the_moment_models_grow_as_declared() {
    hold_surfaces(SURFACES[2]);
}

#[test]
fn the_state_and_cluster_models_grow_as_declared() {
    hold_surfaces(SURFACES[3]);
}

#[test]
fn the_bank_grows_as_declared() {
    hold_surfaces(SURFACES[4]);
}

#[test]
fn the_runner_windows_and_samplers_grow_as_declared() {
    hold_surfaces(SURFACES[5]);
}

/// A finding's cells, held to their declaration: each fails until its
/// finding is fixed, and then gates like any other cell.
fn hold_finding(id: &str) {
    let ids: Vec<&str> = CELLS
        .iter()
        .filter(|c| c.finding == Some(id))
        .map(|c| c.id)
        .collect();
    assert!(!ids.is_empty(), "no cell of finding {id}");
    hold(&ids);
}

/// `corrchange(kind="window")` computes both windows' correlations afresh
/// on every row: `O(window_rows · k²)` a row.
#[test]
#[ignore = "extended: scaling finding S3, corrchange's window costs O(window_rows) a row"]
fn finding_s3_corrchange_window_costs_o_1_a_row_in_its_window() {
    hold_finding("S3");
}
