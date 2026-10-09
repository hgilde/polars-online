//! Matrix work on several threads, cut into pieces fixed by the matrix's
//! shape alone, so the bits never depend on the thread count (docs/PLAN.md
//! tasks 225 and 231).
//!
//! A parallel product whose split follows the thread count sums an entry in
//! another order on another count, and faer's own factorizations and solves
//! run at its global parallelism, which is the size of the current rayon
//! pool. Here the work is cut by the shape: each piece is one sequential
//! faer call on one thread, every entry is computed by one call in one
//! order, and the thread count decides only which thread makes the call.
//!
//! The lower triangle of a symmetric product is cut as faer's own
//! sequential triangular product recurses (halve; the bottom-left block is a
//! rectangle; recurse on the two diagonal blocks), stopped at diagonal blocks
//! of [`LEAF`] columns, each rectangle cut into tiles of [`TILE`] to
//! `2·TILE − 1` on a side ([`pieces`]). The pieces are the calls faer's
//! sequential product makes, and its kernel sums a tile of a rectangle as it
//! sums the whole rectangle, so on aarch64 the result is that product's to
//! the bit. On x86-64 faer hands the whole triangle to one kernel whose
//! blocking the pieces do not reproduce, so there the bits are the pieces'
//! at every thread count, and may differ in the last place from the single
//! call's.

use faer::linalg::matmul::matmul as gemm;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::{Accum, MatMut, MatRef, Par};
use std::sync::{Mutex, PoisonError};

/// A diagonal block this wide or narrower is one triangular product, so a
/// matrix of 256 columns or fewer is one piece. Measured with [`TILE`]
/// (`examples/gram_kernel_probe.rs`, `examples/gram_threads_bench.rs`,
/// 2026-10-08): one thread costs what the single call did (24.4 to 24.8 ps
/// per entry and row at `k = 1,000` to 4,000, against 24.6 to 25.7 before,
/// the two builds run in turn), and `k = 1,000` is 15 pieces, enough to
/// keep fourteen threads busy. Changing either constant can move the last
/// bits on x86-64 (the module doc), never on aarch64.
pub(crate) const LEAF: usize = 256;

/// The least side of a tile a rectangle is cut into ([`tiles`]), and of a
/// band of right-hand sides ([`col_parts`]).
pub(crate) const TILE: usize = 128;

pub(crate) enum Piece<'a> {
    /// A diagonal block at `o`, its lower triangle with the diagonal.
    Diag(MatMut<'a, f64>, usize),
    /// A tile of a bottom-left rectangle at rows `r0`, columns `c0`, and
    /// the tile of the upper triangle it is mirrored into.
    Tile(MatMut<'a, f64>, MatMut<'a, f64>, usize, usize),
}

/// The pieces of the square `c`, cut by `(leaf, tile)`.
pub(crate) fn pieces(c: MatMut<'_, f64>, cut: (usize, usize)) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    plan(c, 0, cut, &mut out);
    out
}

fn plan<'a>(dst: MatMut<'a, f64>, o: usize, cut: (usize, usize), out: &mut Vec<Piece<'a>>) {
    let s = dst.nrows();
    if s <= cut.0 {
        out.push(Piece::Diag(dst, o));
        return;
    }
    // faer's split: the first half is `s / 2` rows.
    let h = s / 2;
    let (tl, tr, bl, br) = dst.split_at_mut(h, h);
    tiles(bl, tr, o + h, o, cut.1, out);
    plan(tl, o, cut, out);
    plan(br, o + h, cut, out);
}

/// `len / tile` near-equal parts, never narrower than `tile` (one part when
/// `len < 2·tile`): a sliver -- one row, or a tile of 64 or fewer on a side
/// -- would take another of faer's kernels (a matrix-vector product, a
/// small-matrix kernel with its own blocking of the rows), which sums an
/// entry in another order than the whole call does.
fn first_part(len: usize, tile: usize) -> Option<usize> {
    let parts = (len / tile).max(1);
    (parts > 1).then(|| len.div_ceil(parts))
}

/// `low` is the rectangle at rows `r0`, columns `c0`; `up` is its mirror,
/// at rows `c0`, columns `r0`. Cut both alike ([`first_part`]).
fn tiles<'a>(
    low: MatMut<'a, f64>,
    up: MatMut<'a, f64>,
    r0: usize,
    c0: usize,
    tile: usize,
    out: &mut Vec<Piece<'a>>,
) {
    let (rows, cols) = (low.nrows(), low.ncols());
    if let Some(first) = first_part(rows, tile) {
        let (la, lb) = low.split_at_row_mut(first);
        let (ua, ub) = up.split_at_col_mut(first);
        tiles(la, ua, r0, c0, tile, out);
        tiles(lb, ub, r0 + first, c0, tile, out);
    } else if let Some(first) = first_part(cols, tile) {
        let (la, lb) = low.split_at_col_mut(first);
        let (ua, ub) = up.split_at_row_mut(first);
        tiles(la, ua, r0, c0, tile, out);
        tiles(lb, ub, r0, c0 + first, tile, out);
    } else {
        out.push(Piece::Tile(low, up, r0, c0));
    }
}

/// The columns of `m` in bands of [`first_part`]'s rule: right-hand sides
/// that one sequential solve each takes, fixed by their count alone.
pub(crate) fn col_parts(m: MatMut<'_, f64>, tile: usize) -> Vec<MatMut<'_, f64>> {
    let mut out = Vec::new();
    let mut rest = m;
    while let Some(first) = first_part(rest.ncols(), tile) {
        let (a, b) = rest.split_at_col_mut(first);
        out.push(a);
        rest = b;
    }
    out.push(rest);
    out
}

/// `C += alpha · Dᵀ D` on the lower triangle of the square `c`, with `D`
/// (`n × k`) read in place, in [`pieces`] cut by `cut` on `threads`
/// threads: what faer's sequential triangular product computes, to the bit
/// on aarch64. The upper triangle is not touched.
pub(crate) fn syrk_lower_cut(
    c: MatMut<'_, f64>,
    d: MatRef<'_, f64>,
    alpha: f64,
    threads: usize,
    cut: (usize, usize),
) {
    spread(pieces(c, cut), threads, |p| match p {
        Piece::Diag(blk, o) => {
            let s = blk.nrows();
            tri(
                blk,
                BlockStructure::TriangularLower,
                Accum::Add,
                d.subcols(o, s).transpose(),
                BlockStructure::Rectangular,
                d.subcols(o, s),
                BlockStructure::Rectangular,
                alpha,
                Par::Seq,
            );
        }
        Piece::Tile(low, _, r0, c0) => {
            let (rows, cols) = (low.nrows(), low.ncols());
            gemm(
                low,
                Accum::Add,
                d.subcols(r0, rows).transpose(),
                d.subcols(c0, cols),
                alpha,
                Par::Seq,
            );
        }
    });
}

/// `f` on every item, in order on this thread when `threads <= 1`, else on
/// up to `threads` workers of the current rayon pool -- the bank's, inside a
/// bank -- each taking the next item until none is left. Which worker takes
/// which item does not matter: the items are disjoint and each is done
/// whole by one call to `f`.
pub(crate) fn spread<T: Send>(items: Vec<T>, threads: usize, f: impl Fn(T) + Sync) {
    let workers = threads.min(items.len());
    if workers <= 1 {
        items.into_iter().for_each(f);
        return;
    }
    let mut items = items;
    items.reverse();
    let queue = Mutex::new(items);
    let work = || {
        loop {
            let next = queue.lock().unwrap_or_else(PoisonError::into_inner).pop();
            match next {
                Some(item) => f(item),
                None => break,
            }
        }
    };
    rayon::scope(|s| {
        for _ in 1..workers {
            s.spawn(|_| work());
        }
        work();
    });
}
