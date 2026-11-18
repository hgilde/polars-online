//! The thread pools of the tests that run one computation in pools of
//! several sizes, built so that the threads they hold stay within what a
//! Windows test runner can commit.
//!
//! On x86-64 the linear-algebra crate's matrix product (`private-gemm-x86`)
//! gives every thread
//! that runs a product past 16 x 16 x 16 a packing buffer of twice the L3
//! cache its CPUID reports, held until the thread exits. Linux and macOS
//! back only the pages touched, but Windows commits the whole reservation
//! against the page file. A GitHub Windows runner with an Intel Xeon 6973P
//! reports 480 MiB of L3, so 960 MiB a thread: the pools of 1, 3, 8 and 14
//! threads one test held at once, beside another test's and the global
//! pool's, ran the unit tests out of commit ("memory allocation of
//! 1006632960 bytes failed", CI on 2fcf8d4a, 2026-11-08, both Windows
//! legs; task 362). Runners whose L3 reads 260 MiB or less passed.
//!
//! So, on Windows x86-64 alone, a test pool has at most [`MOST`] threads,
//! the runner's four cores, and the test pools of the whole process hold at
//! most [`MOST`] threads between them: [`pool`] waits until the threads of
//! the pools dropped before it have exited (each releases its place from
//! rayon's exit handler). With the global pool's four threads and the two
//! test threads CI runs, at most ten threads hold a buffer, 9.4 GiB at
//! 960 MiB. That is a platform guard: the larger pools still run on Linux
//! x86-64, through the same kernel, and on macOS. Elsewhere [`pool`] neither
//! caps nor waits. On every platform a thread holds one test pool at a
//! time, so that the wait cannot deadlock on Windows; a test that holds a
//! second panics everywhere, not only there.

use std::cell::Cell;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The most threads a test pool has, and all test pools hold together.
#[cfg(all(windows, target_arch = "x86_64"))]
pub(crate) const MOST: Option<usize> = Some(4);
/// The most threads a test pool has, and all test pools hold together:
/// no limit.
#[cfg(not(all(windows, target_arch = "x86_64")))]
pub(crate) const MOST: Option<usize> = None;

/// The threads of the test pools not yet exited.
static LIVE: Mutex<usize> = Mutex::new(0);
static EXITED: Condvar = Condvar::new();

thread_local! {
    static HOLDING: Cell<bool> = const { Cell::new(false) };
}

/// A test pool: the rayon pool, and this thread's one place to hold it.
pub(crate) struct TestPool {
    pool: rayon::ThreadPool,
}

impl Deref for TestPool {
    type Target = rayon::ThreadPool;
    fn deref(&self) -> &rayon::ThreadPool {
        &self.pool
    }
}

impl Drop for TestPool {
    fn drop(&mut self) {
        HOLDING.with(|h| h.set(false));
    }
}

/// The size a test pool asked for `n` threads has: `n`, but at most
/// [`MOST`].
pub(crate) fn size(n: usize) -> usize {
    MOST.map_or(n, |most| n.min(most))
}

/// A pool of [`size`]`(n)` threads, once the test pools' threads leave room
/// for them.
pub(crate) fn pool(n: usize) -> TestPool {
    pool_gated(n, None)
}

/// [`pool`], its threads gated by `gate` where there is one ([`before_item`]).
fn pool_gated(n: usize, gate: Option<Arc<Gate>>) -> TestPool {
    assert!(
        !HOLDING.with(Cell::get),
        "a thread holds one test pool at a time: drop the first before building the next"
    );
    let n = size(n);
    {
        let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        while MOST.is_some_and(|most| *live + n > most) {
            live = EXITED.wait(live).unwrap_or_else(|e| e.into_inner());
        }
        *live += n;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(move |_| GATE.set(gate.clone()))
        .exit_handler(|_| {
            *LIVE.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
            EXITED.notify_all();
        })
        .build()
        .expect("a test pool");
    HOLDING.with(|h| h.set(true));
    TestPool { pool }
}

thread_local! {
    /// The gate of the pool this thread belongs to, if it has one.
    static GATE: Cell<Option<Arc<Gate>>> = const { Cell::new(None) };
}

/// What [`read_met_by_a_read`] holds two threads of a pool with: the one
/// that runs the first read (the owner) waits, at its first piece of a
/// parallel section, until the pool's other thread has taken a piece of that
/// section too; that thread stops there, holding it, until the second read
/// is done. So the owner cannot finish the section: it waits for the piece
/// the other holds, and takes other work meanwhile.
#[derive(Default)]
struct Gate {
    owner: AtomicUsize,
    arrived: AtomicBool,
    release: AtomicBool,
    /// The owner gave up waiting for the other thread.
    timed_out: AtomicBool,
}

/// Called by `pieces::spread` before each thread takes an item: a no-op
/// except on the threads of a gated pool.
pub(crate) fn before_item() {
    let held = GATE.take();
    GATE.set(held.clone());
    let Some(gate) = held else {
        return;
    };
    let (me, owner) = (rayon::current_thread_index(), gate.owner.load(SeqCst));
    if gate.release.load(SeqCst) || owner == usize::MAX {
        return;
    }
    if me == Some(owner) {
        let start = Instant::now();
        while !gate.arrived.load(SeqCst) {
            if start.elapsed() > Duration::from_secs(20) {
                gate.timed_out.store(true, SeqCst);
                gate.release.store(true, SeqCst);
                return;
            }
            std::thread::yield_now();
        }
    } else {
        gate.arrived.store(true, SeqCst);
        while !gate.release.load(SeqCst) {
            std::thread::yield_now();
        }
    }
}

/// `read` run twice on a pool of two threads, the second time on the thread
/// that is inside the first, waiting in a rayon scope the first opened, and
/// the two results: what a thread-local buffer held across such a wait
/// meets, a panic ("RefCell already borrowed") or a shared buffer.
///
/// Deterministic, not a race for a steal. The first read's first parallel
/// section is held open by the pool's other thread ([`Gate`]); the second
/// read is queued then, on the pool's global queue, where the only thread
/// free to take it is the first's, waiting in the section. A `read` that
/// opens no parallel section on two threads (below the sizes at which the
/// solves split) makes the owner wait out the time-out and fail.
pub(crate) fn read_met_by_a_read<T: Send>(read: impl Fn() -> T + Sync) -> (T, T) {
    /// Lets the held thread go, however the read ends.
    struct Release<'a>(&'a Gate);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.release.store(true, SeqCst);
        }
    }
    let gate = Arc::new(Gate {
        owner: AtomicUsize::new(usize::MAX),
        ..Gate::default()
    });
    let pool = pool_gated(2, Some(gate.clone()));
    let (first, second) = (Mutex::new(None), Mutex::new(None));
    pool.in_place_scope(|s| {
        s.spawn(|_| {
            let _release = Release(&gate);
            gate.owner.store(
                rayon::current_thread_index().expect("a pool thread"),
                SeqCst,
            );
            *first.lock().unwrap_or_else(|e| e.into_inner()) = Some(read());
        });
        while !gate.arrived.load(SeqCst) && !gate.release.load(SeqCst) {
            std::thread::yield_now();
        }
        s.spawn(|_| {
            let _release = Release(&gate);
            *second.lock().unwrap_or_else(|e| e.into_inner()) = Some(read());
        });
    });
    assert!(
        !gate.timed_out.load(SeqCst),
        "the first read opened no parallel section the pool's other thread took part in"
    );
    let take = |m: Mutex<Option<T>>| m.into_inner().unwrap_or_else(|e| e.into_inner());
    (
        take(first).expect("the first read"),
        take(second).expect("the second read"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool has the size asked, at most [`MOST`], and its threads are
    /// counted while they live and released when they exit.
    #[test]
    fn a_pool_is_counted_while_its_threads_live() {
        for n in [1usize, 3, 8] {
            let p = pool(n);
            assert_eq!(p.current_num_threads(), size(n));
            assert_eq!(p.install(rayon::current_num_threads), size(n));
            drop(p);
        }
        assert_eq!(size(14), MOST.map_or(14, |most| most.min(14)));
    }

    /// A second pool on the same thread is refused before it waits.
    #[test]
    #[should_panic(expected = "one test pool at a time")]
    fn a_second_pool_on_one_thread_is_refused() {
        let _first = pool(1);
        let _second = pool(1);
    }
}
