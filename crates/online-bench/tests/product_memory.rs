//! What a matrix product allocates on each thread that runs it (docs/PLAN.md
//! task 258), counted by a global allocator that keeps each thread's live
//! bytes and their peak.
//!
//! faer 0.24 with its `std` feature sends every product past 16 x 16 x 16
//! on x86-64 to `private-gemm-x86`, which keeps a packing buffer of twice
//! the L3 cache on every thread that has ever multiplied, made at the
//! thread's first product and kept until it exits: 960 MiB a thread on a
//! GitHub Windows runner, which Windows commits whole, so a wide Gram on
//! many threads ran out of commit. Without that feature every product takes
//! the `gemm` crate, as aarch64 always did: a thread keeps one slab the size
//! of its L2 cache, and a product's packing is sized to the product and
//! freed when it returns.
//!
//! Only where the `gemm` crate runs: every target that is not x86-64. Every
//! x86-64 build has faer's `std` and its x86 kernel (tasks 260 and 264), the
//! faster product, whose per-thread buffer this bound would refuse; the
//! measured alternatives for Windows, where that buffer is committed, are
//! PLAN task 263's.
//!
//! Each case runs on a pool made for it, so every one of its threads makes
//! its first product inside the count, where a buffer kept per thread is
//! made. The bound is per thread, [`PER_THREAD`]: the `gemm` crate's slab,
//! the thread's share of its L2 cache (3.2 MiB on an Apple M4 Pro, its
//! 16 MiB over the five cores that share it; 0.5 to 2 MiB on an x86-64
//! core), the packing of a product of these shapes, about 2 MiB, and the
//! scratch and state of the factorization, the eigensolver and the Gram
//! beside them: at most 9.6 MiB a thread measured on that M4 Pro. Twice an
//! L3 is 64 MiB on a 32 MiB L3 and was 960 MiB on GitHub's Windows runners,
//! so the bound fails the old build on x86-64.
//!
//! One test in this binary, so no other test's allocations reach the count.

#![cfg(not(target_arch = "x86_64"))]

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::{Accum, Mat, Par};
use online_polars::online_core::{EwCov, Pca, SpdFactor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// The live bytes each thread holds, its peak since the last reset, and its
/// largest single allocation since then. An allocation freed on another
/// thread than made it would move both threads' counts; the cases here free
/// their scratch where they made it.
struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn add(bytes: isize) {
    // `const` thread-locals without a destructor: never torn down, and no
    // allocation to register one.
    let live = LIVE.with(|l| {
        l.set(l.get() + bytes);
        l.get()
    });
    PEAK.with(|p| p.set(p.get().max(live)));
    if bytes > 0 {
        LARGEST.with(|m| m.set(m.get().max(bytes as usize)));
    }
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

/// The most a thread may allocate above what it held when its case began.
const PER_THREAD: usize = 24 << 20;

const MIB: f64 = (1 << 20) as f64;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn data(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..len).map(|_| lcg(&mut s)).collect()
}

/// Run `f` on a pool of `threads` made for it, `f` itself on every thread
/// at once when `each` is set (`f(i)` for thread `i`), else once inside the
/// pool; return each thread's peak above its live bytes at the start, and
/// its largest allocation.
fn per_thread(threads: usize, each: bool, f: impl Fn(usize) + Sync) -> Vec<(usize, usize)> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    let start: Vec<isize> = pool.broadcast(|_| {
        let live = LIVE.with(Cell::get);
        PEAK.with(|p| p.set(live));
        LARGEST.with(|m| m.set(0));
        live
    });
    if each {
        pool.broadcast(|b| f(b.index()));
    } else {
        pool.install(|| f(0));
    }
    let out = pool.broadcast(|b| {
        let peak = PEAK.with(Cell::get) - start[b.index()];
        (peak.max(0) as usize, LARGEST.with(Cell::get))
    });
    drop(pool);
    out
}

fn check(case: &str, got: &[(usize, usize)]) {
    let worst = got.iter().map(|g| g.0).max().unwrap_or(0);
    let largest = got.iter().map(|g| g.1).max().unwrap_or(0);
    println!(
        "{case}: per-thread peaks {:?} MiB, largest allocation {:.2} MiB",
        got.iter()
            .map(|g| format!("{:.2}", g.0 as f64 / MIB))
            .collect::<Vec<_>>(),
        largest as f64 / MIB
    );
    assert!(
        worst <= PER_THREAD,
        "{case}: a thread allocated {:.1} MiB, over the {:.0} MiB bound",
        worst as f64 / MIB,
        PER_THREAD as f64 / MIB
    );
}

#[test]
fn a_product_allocates_a_bounded_amount_on_each_thread() {
    // The calls the Gram's pieces make (`pieces.rs`, `ewcov/par.rs`): a tile
    // of a rectangle, `Dᵀ D` over a block of rows past the kernels' 512-row
    // panel, and a diagonal block's lower triangle; on four threads at once,
    // each its own.
    let (n, k) = (600usize, 255usize);
    let d = Mat::from_fn(n, k, |i, j| ((i * 31 + j * 17) % 97) as f64 / 97.0 - 0.5);
    let mut outs: Vec<std::sync::Mutex<Mat<f64>>> = (0..4)
        .map(|_| std::sync::Mutex::new(Mat::zeros(k, k)))
        .collect();
    let got = per_thread(4, true, |i| {
        let mut c = outs[i].lock().expect("unpoisoned");
        matmul(
            c.as_mut(),
            Accum::Add,
            d.transpose(),
            d.as_ref(),
            1.0,
            Par::Seq,
        );
        tri(
            c.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            d.transpose(),
            BlockStructure::Rectangular,
            d.as_ref(),
            BlockStructure::Rectangular,
            -1.0,
            Par::Seq,
        );
    });
    check("tile and triangle", &got);
    // The products did run: the rectangle less the triangle leaves the
    // upper part of `DᵀD`, and the lower part to rounding.
    let c = outs[0].get_mut().expect("unpoisoned");
    assert!(c[(0, k - 1)].abs() > 1.0 && c[(k - 1, k - 1)].abs() < 1e-9);

    // A wide Gram as the bank runs it: `gram_block_rows = 256`, its merges
    // cut into pieces on `gram_threads = 4`, a block past one leaf.
    let k = 700;
    let xs = data(512 * k, 7);
    let got = per_thread(4, false, |_| {
        let mut c = EwCov::new(k);
        c.set_block_rows(256);
        c.set_threads(4);
        for r in xs.chunks_exact(k) {
            c.update(r, 0.999, 1.0);
        }
        c.flush();
        assert!(c.comoments()[k * k - 1] > 0.0);
    });
    check("EwCov k=700, gram_block_rows 256, gram_threads 4", &got);

    // Its solve: faer's blocked Cholesky steps, the panel's triangular
    // solves and the trailing products on the pool's threads, then the two
    // triangular solves of the right-hand sides.
    let k = 700;
    let x = data((k + 9) * k, 11);
    let mut a = vec![0.0; k * k];
    for r in x.chunks_exact(k) {
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] += r[i] * r[j];
            }
        }
    }
    let b = data(k * 3, 13);
    let got = per_thread(4, false, |_| {
        let f = SpdFactor::of(&a, k).expect("positive definite");
        let s = f.solve(&b, k, 3);
        assert!(s.iter().all(|v| v.is_finite()));
    });
    check("SpdFactor k=700 and a solve, pool of 4", &got);

    // The eigensolver, on its fixed degree of eight threads.
    let k = 300;
    let a: Vec<f64> = (0..k * k)
        .map(|ij| {
            let (i, j): (usize, usize) = (ij / k, ij % k);
            if i == j {
                2.0
            } else {
                1.0 / (1 + i.abs_diff(j)) as f64
            }
        })
        .collect();
    let got = per_thread(8, false, |_| {
        let p = Pca::of(&a, k, 2, None).expect("an eigendecomposition");
        assert!(p.eig[0].is_finite());
    });
    check("Pca k=300, pool of 8", &got);
}
