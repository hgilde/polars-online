//! A grouped bank allocates for its chunk, not for each group of it (task
//! 434). Each group of each chunk cost the bank five allocations beside its
//! models' own: the group's list of rows and the text of its key in the
//! chunk's partition, and the lists a run is made of (the plans, the
//! instance, the task's result, the feature rows -- task 421's). The
//! partition is now one flat list of rows with the keys as the chunk's own
//! values, found in the streams' map by their text on the stack, so a chunk
//! of 1,000 groups makes a few allocations in all.
//!
//! Counted by a global allocator of the test binary's own, over every
//! thread: the bank runs its groups on a pool thread, which a per-thread
//! count does not see. The file holds one test, so nothing else allocates
//! while it counts.

use rheon_bench::{ROWS, bank_setup, rheon_budget_bank_rows};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Relaxed);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Allocations a row of each grouped bank case's measured block (1,000
/// groups, a block of [`ROWS`] rows): the bound is what the change leaves,
/// the count before it beside.
#[test]
fn a_grouped_bank_allocates_for_its_chunk_not_for_each_group() {
    for (id, bound, before) in [
        ("bank_seqtest_groups", 0.3, 2.24),
        ("bank_ewridge_k5_groups", 3.7, 5.35),
    ] {
        let mut run = bank_setup(id);
        let start = CALLS.load(Relaxed);
        let out = std::hint::black_box(rheon_budget_bank_rows(&mut run));
        let got = (CALLS.load(Relaxed) - start) as f64 / ROWS as f64;
        drop(out);
        eprintln!("{id}: {got:.3} allocations a row (bound {bound}, {before} before)");
        assert!(got <= bound, "{id}: {got} allocations a row, past {bound}");
    }
}
