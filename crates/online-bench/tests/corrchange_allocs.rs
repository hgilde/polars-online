//! `corrchange`'s window kind allocates a bounded number of times a row
//! (docs/PLAN.md task 270, D-2): it cloned its whole ring of two windows on
//! every row, 204 allocations a row at windows of 100 rows, and every
//! pooled row again on each permutation draw. Counted here by a global
//! allocator of the test binary's own, around the `corrchange_window_w100`
//! case's measured block, which redraws its permutation critical value
//! four times.

use online_bench::{ROWS, model_setup, polars_online_budget_model_rows};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[test]
fn the_window_kind_allocates_a_few_times_a_row() {
    let mut run = model_setup("corrchange_window_w100");
    let before = COUNT.with(Cell::get);
    let block = polars_online_budget_model_rows(&mut run);
    let per_row = (COUNT.with(Cell::get) - before) as f64 / ROWS as f64;
    assert_eq!(block.predicted, ROWS);
    // The row's report and its copy into the ring, each window's one buffer
    // of sums, and each draw's two (four redraws of 100 draws in the
    // block): 5.8 a row when this was written, 286 before.
    assert!(per_row <= 8.0, "{per_row} allocations a row");
}
