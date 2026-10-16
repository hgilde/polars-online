//! Which threads of the bank's pool run a wide `ewridge`'s matrix products
//! (docs/PLAN.md task 269, review 8 F-1), seen through a global allocator
//! that keeps each thread's live bytes and their peak: a thread that ran a
//! product allocated the `gemm` crate's slab or a piece's scratch.
//!
//! The two halves of an `ewridge` fit read two different counts.
//! `gram_threads` caps the update of the `k x k` matrix alone
//! (`EwCov::set_threads`); the solve's factorization and its triangular
//! solves spread over the whole pool (`solve.rs` reads
//! `rayon::current_num_threads()`). So only the pool's size, the bank's
//! `threads=`, bounds how many threads multiply, and with it the packing
//! buffer of twice the L3 cache that faer's x86 kernel keeps on each of
//! them, which Windows commits whole (the CHANGELOG's *Known issues* and
//! docs/PERFORMANCE.md §43 say so).
//!
//! Only where the `gemm` crate runs, as `product_memory.rs`: x86-64 builds
//! take faer's x86 kernel, whose buffer is reserved per thread at its
//! first product whatever it multiplies. One test in this binary, so no
//! other test's allocations reach the count.

#![cfg(not(target_arch = "x86_64"))]

use online_polars::online_core::{EwCov, SpdFactor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn add(bytes: isize) {
    let live = LIVE.with(|l| {
        l.set(l.get() + bytes);
        l.get()
    });
    PEAK.with(|p| p.set(p.get().max(live)));
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, and only counts besides.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as isize);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as isize);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        add(-(layout.size() as isize));
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        add(new_size as isize - layout.size() as isize);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// A thread that ran a product allocated more than this above its start.
const RAN: usize = 1 << 19;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Run `f` once inside a pool of `threads` made for it, so every thread
/// makes its first product inside the count; return how many of its
/// threads allocated more than [`RAN`] above their start.
fn threads_that_ran(case: &str, threads: usize, f: impl Fn() + Sync) -> usize {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    let start: Vec<isize> = pool.broadcast(|_| {
        let live = LIVE.with(Cell::get);
        PEAK.with(|p| p.set(live));
        live
    });
    pool.install(&f);
    let peaks: Vec<usize> =
        pool.broadcast(|b| (PEAK.with(Cell::get) - start[b.index()]).max(0) as usize);
    drop(pool);
    let ran = peaks.iter().filter(|&&p| p > RAN).count();
    println!(
        "{case}: per-thread peaks {:?} MiB, {ran} of {threads} threads ran a product",
        peaks
            .iter()
            .map(|p| format!("{:.2}", *p as f64 / (1 << 20) as f64))
            .collect::<Vec<_>>()
    );
    ran
}

#[test]
fn only_the_pool_bounds_the_threads_that_multiply() {
    let pool = 4;

    // The update of the matrix on `gram_threads = 1`, blocked, so its merge
    // is a product: one thread of the pool runs it.
    let k = 700;
    let mut s = 7u64;
    let xs: Vec<f64> = (0..512 * k).map(|_| lcg(&mut s)).collect();
    let update = threads_that_ran("EwCov k=700, gram_threads 1", pool, || {
        let mut c = EwCov::new(k);
        c.set_block_rows(256);
        c.set_threads(1);
        for r in xs.chunks_exact(k) {
            c.update(r, 0.999, 1.0);
        }
        c.flush();
        assert!(c.comoments()[k * k - 1] > 0.0);
    });
    assert_eq!(
        update, 1,
        "gram_threads = 1 ran the update on {update} threads"
    );

    // The solve that reads such a matrix, which no `gram_threads` reaches:
    // a factorization at k = 1,500 and one solve. Symmetric, its diagonal
    // dominant, so positive definite.
    let k = 1500;
    let a: Vec<f64> = (0..k * k)
        .map(|ij| {
            let (i, j) = (ij / k, ij % k);
            let (lo, hi) = (i.min(j), i.max(j));
            if i == j {
                k as f64
            } else {
                ((lo * 31 + hi * 17) % 97) as f64 / 97.0 - 0.5
            }
        })
        .collect();
    let b: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
    let solve = || {
        let f = SpdFactor::of(&a, k).expect("positive definite");
        let x = f.solve(&b, k, 1);
        assert!(x.iter().all(|v| v.is_finite()));
    };
    // Every thread of the pool may take a piece; one that wakes after the
    // queue is empty takes none, so the bound is more than the one thread
    // `gram_threads = 1` allows, not all of them (8 of 8 in a pool of 8 on
    // an idle Apple M4 Pro).
    let wide = threads_that_ran("SpdFactor k=1500 and one solve", pool, solve);
    assert!(
        wide > 1,
        "the solve ran on {wide} of {pool} threads: it no longer spreads over the pool"
    );
    let one = threads_that_ran("SpdFactor k=1500 and one solve", 1, solve);
    assert_eq!(one, 1, "a pool of one ran the solve on {one} threads");
}
