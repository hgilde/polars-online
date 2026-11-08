//! Every cell of the scaling table (`online_bench::scaling`), measured at its
//! two sizes and held to the growth its algorithm declares: a row's work by
//! its time and its allocations, the state by its saved bytes and the heap
//! it holds (docs/PERFORMANCE.md, "How cost grows").
//!
//! The counts are the allocator's, kept process-wide, so the cells take
//! turns under one lock: the tests of this binary may still be run side by
//! side, and nextest runs each in a process of its own.
//!
//! A block's allocations are counted on every thread but the harness's.
//! The measured work runs on the measuring thread and on threads it does
//! not name -- the bank's pool, Polars' pool, the runner's reader thread --
//! so it cannot be counted by thread from the inside. The harness's threads
//! can: libtest's main thread, which reports each finished test and starts
//! the next while another test measures, and each test's own thread outside
//! the blocks it measures (`harness_thread`). Counted with the rest, they
//! read 0 -> 4 allocations over its blocks for a cell that allocates
//! nothing, and 2 -> 18 for another, when the tests ran side by side on
//! three threads with their output captured, as `cargo test` runs them on
//! CI: past the ratio a growth of O(1) allows.
//!
//! `SCALING_REPORT=1` prints every cell's ratios, the audit's table.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use online_bench::scaling::{
    CELLS, Cell, ScaleRun, polars_online_budget_scale_rows, scale_setup_blocks,
};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
/// Allocations on every thread but the harness's: what a block counts.
static CALLS: AtomicUsize = AtomicUsize::new(0);
/// Allocations on the harness's threads, reported beside a cell's count.
static HARNESS_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Whether the process has allocated: its first allocation is on its main
/// thread, libtest's, before libtest has started any other.
static STARTED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Whether this thread is the harness's: libtest's main thread, or a
    /// test's thread outside the blocks it measures. A `const` `Cell` of a
    /// `bool` has no destructor and its first read allocates nothing, so
    /// the allocator may read it on any thread at any time.
    static HARNESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn grew(by: usize) {
    if !STARTED.load(Ordering::Relaxed) && !STARTED.swap(true, Ordering::Relaxed) {
        HARNESS.set(true);
    }
    if HARNESS.get() {
        HARNESS_CALLS.fetch_add(1, Ordering::Relaxed);
    } else {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
    let now = LIVE.fetch_add(by as isize, Ordering::Relaxed) + by as isize;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

/// Marks this test's thread as the harness's: its allocations are a cell's
/// only inside the blocks it measures. The first line of every test.
fn harness_thread() {
    HARNESS.set(true);
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
    /// The least CPU time of a block, where it is read.
    cpu: Option<Duration>,
    /// Allocations over every block.
    calls: usize,
    /// The harness's allocations meanwhile, which `calls` leaves out.
    harness: usize,
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

/// The CPU time every thread of this process has run, which a block's time
/// is judged by where it is read: a thread waiting for a core adds none.
///
/// The wall clock reads the machine's load. With more runnable threads than
/// cores, a thread runs a slice and then waits its turn, so a block longer
/// than a slice is stretched by about the load over the cores, while one
/// shorter may run whole in one slice and is not: the least of several
/// blocks then reads a short block at its own time and a long one
/// stretched. A thread woken to run a block (a pool's) may also wait a
/// whole slice first. So a ratio of two sizes' wall times grows with the
/// load whenever the small size's block is short. At a 1-minute load of
/// about 50 on 14 cores, four runs of this binary judged by the wall clock
/// each failed three or four cells (`gbdt.history`, a cell of declared O(1)
/// growth, read 25-33x against its 2.83x: 0.3 ms -> 9.2 ms), and measuring a
/// failing cell four times more, keeping each size's least block, still
/// failed them; judged by CPU time, no cell failed in thirteen runs at
/// loads of 25-99, the highest reading 0.89 of its bound, and idle the two
/// agree (no cell above 0.58). A row's work on a pool's threads counts, as
/// the work the harness cannot name by thread does for the allocations.
///
/// `None` where it is not read (`clock_gettime` is POSIX): there the wall
/// clock is judged.
#[cfg(unix)]
fn cpu_time() -> Option<Duration> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec` for the call's duration.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
}

#[cfg(not(unix))]
fn cpu_time() -> Option<Duration> {
    None
}

/// One block of a run: its time, its allocations, and the most it held above
/// the live heap. Its allocations are those on every thread but the
/// harness's, so this thread counts while the block runs; the harness's
/// threads' allocations meanwhile are the third.
fn block(run: &mut ScaleRun) -> (Duration, Option<Duration>, usize, usize, usize) {
    let pool = run
        .on_the_pool()
        .then(|| online_polars::pool().expect("the bank's pool"));
    let calls = CALLS.load(Ordering::Relaxed);
    let harness = HARNESS_CALLS.load(Ordering::Relaxed);
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    HARNESS.set(false);
    let cpu_at = cpu_time();
    let at = Instant::now();
    let sum = match pool {
        Some(p) => p.install(|| polars_online_budget_scale_rows(run)),
        None => polars_online_budget_scale_rows(run),
    };
    let time = at.elapsed();
    let cpu = cpu_time().zip(cpu_at).map(|(b, a)| b.saturating_sub(a));
    HARNESS.set(true);
    std::hint::black_box(sum);
    let peak = (PEAK.load(Ordering::Relaxed) - live).max(0) as usize;
    (
        time,
        cpu,
        CALLS.load(Ordering::Relaxed) - calls,
        peak,
        HARNESS_CALLS.load(Ordering::Relaxed) - harness,
    )
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
        cpu: cpu_time().map(|_| Duration::MAX),
        calls: 0,
        harness: 0,
        saved: 0,
        heap: 0,
        peak: 0,
        rows: run.block_rows,
    });
    for _ in 0..BLOCKS {
        for (m, run) in out.iter_mut().zip([&mut small, &mut large]) {
            let (t, cpu, calls, peak, harness) = block(run);
            m.time = m.time.min(t);
            m.cpu = m.cpu.zip(cpu).map(|(a, b)| a.min(b));
            m.calls += calls;
            m.harness += harness;
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

/// Allocations a block may make past its declared growth whatever the
/// ratio reads, so a cell that allocates a handful a block is not judged
/// by a ratio of two handfuls. What reaches a count beside the measured work
/// is a handful: libtest allocates on a test's thread before the test's
/// first line can mark it as the harness's, so a test that starts while
/// another measures adds a few, and so does a thread of a pool a finished
/// workload dropped, which exits on its own time. Over ten runs of this
/// binary on three threads, a small count moved by 2 at most between runs
/// once the harness's threads were left out. The growth the check is for
/// adds an allocation a row, and a block is at least 200 rows.
const ALLOC_SLACK: usize = 4;

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
    let wall = ratio(per(&s, s.time.as_secs_f64()), per(&l, l.time.as_secs_f64()));
    let cpu = s
        .cpu
        .zip(l.cpu)
        .map(|(a, b)| ratio(per(&s, a.as_secs_f64()), per(&l, b.as_secs_f64())));
    // The time judged is the CPU time where it is read, the wall clock's
    // elsewhere (`cpu_time`).
    let t = cpu.unwrap_or(wall);
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
    let what = if cpu.is_some() {
        "a block's CPU time"
    } else {
        "a block's time"
    };
    check(what, t, time_bound);
    // Within the declared ratio, or within the slack of the count the
    // declared growth gives the large size from the small one's.
    let expected = s.calls as f64 * work * l.rows as f64 / s.rows as f64;
    if l.calls as f64 > expected + (ALLOC_SLACK * BLOCKS) as f64 {
        check(
            &format!(
                "a block's allocations ({} -> {}, more than {ALLOC_SLACK} a block past the declared growth)",
                s.calls, l.calls
            ),
            a,
            work * (1.0 + COUNT_TOL),
        );
    }
    check("the saved state", saved, state * (1.0 + COUNT_TOL));
    check("the state's heap", heap, state * (1.0 + COUNT_TOL));
    let line = format!(
        "{:<28} work {:<10} state {:<10} | cpu {:>7.2}x (<= {:>7.2}) {:>9.3} ms -> {:>9.3} ms | wall {:>7.2}x {:>9.3} ms -> {:>9.3} ms | allocs {:>7.2}x ({} -> {}, harness {} -> {}) | saved {:>6.2}x ({} -> {}) | heap {:>6.2}x ({} -> {}) {}",
        c.id,
        c.work.show("n"),
        c.state.show("n"),
        cpu.unwrap_or(f64::NAN),
        time_bound,
        s.cpu.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3),
        l.cpu.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3),
        wall,
        s.time.as_secs_f64() * 1e3,
        l.time.as_secs_f64() * 1e3,
        a,
        s.calls,
        l.calls,
        s.harness,
        l.harness,
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
    harness_thread();
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
    harness_thread();
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
    harness_thread();
    hold_surfaces(SURFACES[0]);
}

#[test]
fn the_robust_and_gradient_models_grow_as_declared() {
    harness_thread();
    hold_surfaces(SURFACES[1]);
}

#[test]
fn the_moment_models_grow_as_declared() {
    harness_thread();
    hold_surfaces(SURFACES[2]);
}

#[test]
fn the_state_and_cluster_models_grow_as_declared() {
    harness_thread();
    hold_surfaces(SURFACES[3]);
}

#[test]
fn the_bank_grows_as_declared() {
    harness_thread();
    hold_surfaces(SURFACES[4]);
}

#[test]
fn the_runner_windows_and_samplers_grow_as_declared() {
    harness_thread();
    hold_surfaces(SURFACES[5]);
}

/// A finding's cells, held to their declaration: each fails until its
/// finding is fixed, and then gates like any other cell. No finding is open
/// (S1 to S3 are fixed); the next one's `#[ignore = "extended: scaling
/// finding <id>"]` test calls this.
#[allow(dead_code)]
fn hold_finding(id: &str) {
    let ids: Vec<&str> = CELLS
        .iter()
        .filter(|c| c.finding == Some(id))
        .map(|c| c.id)
        .collect();
    assert!(!ids.is_empty(), "no cell of finding {id}");
    hold(&ids);
}
