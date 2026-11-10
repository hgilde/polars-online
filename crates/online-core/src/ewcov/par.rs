//! The co-moment matrix's `k²` work on several threads, to the same bits as
//! on one (`gram_threads`, docs/PLAN.md task 225).
//!
//! Two kinds of work touch every entry of `C`. The per-row rank-1 update is
//! one expression per entry, so a split of the rows across threads cannot
//! move a bit: [`rows`]. A block's merge ([`super::EwCov::flush`]) is a
//! product, `C += a·DᵀD`, where an entry is a sum over the block's rows, and
//! a parallel product whose split depends on the thread count sums an entry
//! in another order on another count. [`merge_lower`] cuts it into
//! [`crate::pieces`], fixed by `k` alone, each one sequential faer call on
//! one thread: on aarch64 the bits a build before task 225 wrote, at every
//! thread count.
//!
//! The rest of a merge is per entry -- the history's scale before the
//! product, the two rank-1 corrections after it -- and runs inside the same
//! piece, while its tile is in cache: one pass over `C` where there were
//! four. Each entry still takes the same operations in the same order. Only
//! the lower triangle is written: the matrix is kept as it
//! ([`super::LowerTriangle`]), so the mirror into the upper one each merge
//! once ended with is gone (task 294).

use super::row_at;
use crate::pieces::{LEAF, Place, TILE, places, spread};
use faer::linalg::matmul::matmul as gemm;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::reborrow::ReborrowMut;
use faer::{Accum, MatMut, MatRef, Par};
#[cfg(test)]
use std::sync::Mutex;

/// Below this many entries of `C` a row's rank-1 update stays on the
/// calling thread at any thread count, `2^19`, so from `k = 725`. Measured
/// (`examples/gram_threads_bench.rs`, 2026-10-08, 14 cores at a load of 7):
/// at `k = 256` two to fourteen threads took 1.3 to 3.7 times as long as
/// one, at 512 they were 1.1 to 1.4 times as fast, at 724 1.5 to 2.2, and at
/// 1,000 1.5 to 3.2. A row's dispatch costs microseconds, and the update is
/// bound by memory, so it pays only on a matrix of megabytes.
pub(crate) const PAR_MIN_ENTRIES: usize = 1 << 19;

/// What a merge does to each lower entry `C[i][j]` around the product:
/// `C·scale`, then the product's `+ alpha·Σ dᵢdⱼ`, then
/// `+ (bi[i]·big[j] − wi[i]·delta[j])`. The upper triangle is not written.
pub(crate) struct Merge<'a> {
    pub(crate) scale: f64,
    pub(crate) alpha: f64,
    /// `between · big`, per row.
    pub(crate) bi: &'a [f64],
    /// `within · delta`, per row.
    pub(crate) wi: &'a [f64],
    pub(crate) big: &'a [f64],
    pub(crate) delta: &'a [f64],
}

/// The merge of a block `D` (`n × k`, row-major, rows already centred and
/// weighted) into the packed lower triangle `c` over `k` features, on
/// `threads` threads. Each piece is a sequential faer call on a row-major
/// view: the piece lays its part of the triangle into the view, scaled, as
/// its first pass and takes it back, corrected, as its last ([`run`]), on
/// the thread that runs it. A matrix of one piece, no wider than a leaf,
/// runs in `square`, the caller's to keep, sized here only where it is not
/// `k²` (its strict upper triangle is never written, so it stays as sized).
/// Each piece of a wider one runs in its thread's own buffer ([`TILE_BUF`]).
///
/// Laid into a kept square and taken back whole on the calling thread, the
/// square zeroed first, the copies cost a 1,000-feature fit blocked at 256
/// rows, solved every 256, a third of its speed on arm64 (16,800 rows a
/// second against 24,900 while the square was the matrix, 14 threads) and
/// a 500-feature one a sixth: two serial passes of 4 MB while the pieces'
/// threads waited. Moved into the pieces, each still laying its part into
/// the shared square, the 1,000-feature fit ran at 20,900: a piece wrote
/// lines of the square that the core which ran it the merge before held. A
/// thread's own buffer is in its own cache: 24,950, and 94,500 at 500
/// features (95,000 before).
pub(crate) fn merge_lower(
    c: &mut [f64],
    square: &mut Vec<f64>,
    k: usize,
    d: &[f64],
    n: usize,
    m: &Merge<'_>,
    threads: usize,
) {
    merge_lower_cut(c, square, k, d, n, m, threads, (LEAF, TILE));
}

/// [`merge_lower`] with its cut given: `(leaf, tile)`. The tests cut small
/// matrices finely, so that a cheap case has many pieces.
#[allow(clippy::too_many_arguments)]
fn merge_lower_cut(
    c: &mut [f64],
    square: &mut Vec<f64>,
    k: usize,
    d: &[f64],
    n: usize,
    m: &Merge<'_>,
    threads: usize,
    cut: (usize, usize),
) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    if k <= cut.0 {
        // One diagonal block at 0, whose rows are the packed rows in turn:
        // no list of them is made, and nothing is allocated.
        if square.len() != k * k {
            square.clear();
            square.resize(k * k, 0.0);
        }
        let view = MatMut::from_row_major_slice_mut(square, k, k);
        let at = Place {
            diag: true,
            r0: 0,
            c0: 0,
            rows: k,
            cols: k,
        };
        return run(at, view, Runs::Whole(c), dm, m);
    }
    let at = places(k, cut);
    let runs = runs_of(c, k, &at);
    let items: Vec<_> = at.into_iter().zip(runs).collect();
    spread(items, threads, |(at, runs)| {
        TILE_BUF.with_borrow_mut(|buf| {
            let len = at.rows * at.cols;
            if buf.len() < len {
                buf.resize(len, 0.0);
            }
            let view = MatMut::from_row_major_slice_mut(&mut buf[..len], at.rows, at.cols);
            run(at, view, runs, dm, m);
        });
    });
}

thread_local! {
    /// A thread's working copy of the piece of a merge it runs
    /// ([`merge_lower`]): kept, so it is in this thread's cache, and grown
    /// to the largest piece it has run, at most a leaf's square, 512 KiB.
    /// Every entry a piece reads is laid in first, so what it held before
    /// does not matter. Not state.
    static TILE_BUF: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A piece's entries in the packed triangle: one run a row of the piece, in
/// its rows' order.
enum Runs<'a> {
    /// The whole triangle, the one piece of a matrix no wider than a leaf:
    /// row `i` is the next `i + 1` numbers.
    Whole(&'a mut [f64]),
    /// Each row's run, handed out by [`runs_of`].
    Listed(Vec<&'a mut [f64]>),
}

impl Runs<'_> {
    /// `f(r, run)` for each of the piece's rows `r`, in order.
    #[inline]
    fn each(&mut self, mut f: impl FnMut(usize, &mut [f64])) {
        match self {
            Runs::Whole(c) => {
                let mut rest: &mut [f64] = c;
                let mut i = 0;
                while !rest.is_empty() {
                    let (run, tail) = std::mem::take(&mut rest).split_at_mut(i + 1);
                    f(i, run);
                    rest = tail;
                    i += 1;
                }
            }
            Runs::Listed(runs) => {
                for (r, run) in runs.iter_mut().enumerate() {
                    f(r, run);
                }
            }
        }
    }
}

/// Where each piece's rows lie in the packed triangle `c`: per piece, in
/// the pieces' order, its rows' runs, split off `c` row by row. A diagonal
/// block at `r0` takes columns `r0 ..= i` of its row `i`, a tile its `cols`
/// columns from `c0`; the pieces cover the lower triangle once
/// (`every_lower_entry_is_in_one_piece`), so the runs are disjoint and
/// cover it.
fn runs_of<'a>(c: &'a mut [f64], k: usize, at: &[Place]) -> Vec<Runs<'a>> {
    // The pieces by their first column: along a row, those that hold it
    // come in this order.
    let mut by_col: Vec<usize> = (0..at.len()).collect();
    by_col.sort_unstable_by_key(|&p| at[p].c0);
    let mut out: Vec<Vec<&'a mut [f64]>> = at.iter().map(|a| Vec::with_capacity(a.rows)).collect();
    let mut rows: &'a mut [f64] = c;
    for i in 0..k {
        let (mut row, tail) = std::mem::take(&mut rows).split_at_mut(i + 1);
        rows = tail;
        for &p in &by_col {
            let a = at[p];
            if !(a.r0..a.r0 + a.rows).contains(&i) {
                continue;
            }
            debug_assert_eq!(i + 1 - row.len(), a.c0, "row {i}: the pieces leave a gap");
            let len = if a.diag { i - a.r0 + 1 } else { a.cols };
            let (run, rest) = std::mem::take(&mut row).split_at_mut(len);
            out[p].push(run);
            row = rest;
        }
        debug_assert!(row.is_empty(), "row {i}: the pieces leave its end");
    }
    out.into_iter().map(Runs::Listed).collect()
}

/// Row `i` of a row-major view, as a slice.
fn row<'a>(m: MatMut<'a, f64>, i: usize) -> &'a mut [f64] {
    m.row_mut(i)
        .try_as_row_major_mut()
        .expect("a row-major view has unit column stride")
        .as_slice_mut()
}

/// One piece of the merge at `at`, in the row-major view `v` of its shape:
/// its part of the triangle laid in scaled (`c·scale`), the product added
/// there, and taken back with the corrections (`+ (bi·big − wi·delta)`).
/// Each entry's operations, in their order, as four passes over the square
/// in place once gave them, and the product the same faer call on a view of
/// the same shape.
fn run(at: Place, mut v: MatMut<'_, f64>, mut runs: Runs<'_>, d: MatRef<'_, f64>, m: &Merge<'_>) {
    let Place {
        diag,
        r0,
        c0,
        rows,
        cols,
    } = at;
    // A diagonal block's row `i` is its first `i + 1` columns.
    let width = |i: usize| if diag { i + 1 } else { cols };
    runs.each(|i, src| {
        for (cj, &x) in row(v.rb_mut(), i)[..width(i)].iter_mut().zip(&*src) {
            *cj = x * m.scale;
        }
    });
    if diag {
        tri(
            v.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            d.subcols(r0, rows).transpose(),
            BlockStructure::Rectangular,
            d.subcols(r0, rows),
            BlockStructure::Rectangular,
            m.alpha,
            Par::Seq,
        );
    } else {
        gemm(
            v.rb_mut(),
            Accum::Add,
            d.subcols(r0, rows).transpose(),
            d.subcols(c0, cols),
            m.alpha,
            Par::Seq,
        );
    }
    runs.each(|i, dst| {
        let (bi, wi) = (m.bi[r0 + i], m.wi[r0 + i]);
        let w = width(i);
        let big = &m.big[c0..c0 + w];
        let delta = &m.delta[c0..c0 + w];
        let prod = &row(v.rb_mut(), i)[..w];
        for (((out, &cj), &gj), &dj) in dst.iter_mut().zip(prod).zip(big).zip(delta) {
            *out = cj + (bi * gj - wi * dj);
        }
    });
}

/// `f(i, row)` for each row of the packed lower triangle `c` over `k`
/// features, row `i` its `i + 1` entries, on `threads` threads: each takes
/// a band of whole rows, so an entry's arithmetic is the same whichever
/// thread runs it.
pub(crate) fn rows(c: &mut [f64], k: usize, threads: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    if k == 0 {
        return;
    }
    let threads = threads.max(1);
    let mut bands: Vec<(usize, &mut [f64])> = Vec::with_capacity(threads);
    let mut rest = c;
    let mut at = 0;
    for b in 1..=threads {
        let end = band_end(k, threads, b).clamp(at, k);
        let (head, tail) = rest.split_at_mut(row_at(end) - row_at(at));
        if !head.is_empty() {
            bands.push((at, head));
        }
        rest = tail;
        at = end;
    }
    spread(bands, threads, |(start, mut band)| {
        let mut i = start;
        while !band.is_empty() {
            let (row, tail) = std::mem::take(&mut band).split_at_mut(i + 1);
            f(i, row);
            band = tail;
            i += 1;
        }
    });
}

/// Where band `b` of `threads` (from 1) ends among `n_rows` rows, for
/// [`rows`]. Row `i` writes its lower `i + 1` entries, so the first `r`
/// rows hold about `r²/2` of them, and bands cut at `n·√(b/t)` take equal
/// work; bands of equal rows gave the last one 7/16 of it at four threads.
/// The last band ends at the last row. Which band holds a row says only
/// which thread runs it: an entry is the same expression on any.
fn band_end(n_rows: usize, threads: usize, b: usize) -> usize {
    if b >= threads {
        return n_rows;
    }
    let share = (b as f64 / threads as f64).sqrt();
    (n_rows as f64 * share).round() as usize
}

#[cfg(test)]
mod tests;
