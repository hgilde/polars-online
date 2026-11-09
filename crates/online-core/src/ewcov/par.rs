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
use crate::pieces::{LEAF, Piece, TILE, pieces, spread};
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
/// `threads` threads. The pieces are views of a row-major square, so the
/// triangle is laid into one and taken back after: two copies of its
/// `k(k+1)/2` numbers a block, where the merge does `n` times that work,
/// into `square`, the caller's to keep.
pub(crate) fn merge_lower(
    c: &mut [f64],
    square: &mut Vec<f64>,
    k: usize,
    d: &[f64],
    n: usize,
    m: &Merge<'_>,
    threads: usize,
) {
    square.clear();
    square.resize(k * k, 0.0);
    for (i, dst) in square.chunks_exact_mut(k.max(1)).take(k).enumerate() {
        dst[..=i].copy_from_slice(&c[row_at(i)..=row_at(i) + i]);
    }
    merge_lower_cut(square, k, d, n, m, threads, (LEAF, TILE));
    for (i, src) in square.chunks_exact(k.max(1)).take(k).enumerate() {
        c[row_at(i)..=row_at(i) + i].copy_from_slice(&src[..=i]);
    }
}

/// [`merge_lower`] with its cut given: `(leaf, tile)`. The tests cut small
/// matrices finely, so that a cheap case has many pieces.
fn merge_lower_cut(
    c: &mut [f64],
    k: usize,
    d: &[f64],
    n: usize,
    m: &Merge<'_>,
    threads: usize,
    cut: (usize, usize),
) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    let pieces = pieces(MatMut::from_row_major_slice_mut(c, k, k), cut);
    spread(pieces, threads, |p| run(p, dm, m));
}

/// Row `i` of a row-major view, as a slice.
fn row<'a>(m: MatMut<'a, f64>, i: usize) -> &'a mut [f64] {
    m.row_mut(i)
        .try_as_row_major_mut()
        .expect("a row-major view has unit column stride")
        .as_slice_mut()
}

fn run(p: Piece<'_>, d: MatRef<'_, f64>, m: &Merge<'_>) {
    match p {
        Piece::Diag(mut blk, o) => {
            let s = blk.nrows();
            for i in 0..s {
                for cj in &mut row(blk.rb_mut(), i)[..=i] {
                    *cj *= m.scale;
                }
            }
            tri(
                blk.rb_mut(),
                BlockStructure::TriangularLower,
                Accum::Add,
                d.subcols(o, s).transpose(),
                BlockStructure::Rectangular,
                d.subcols(o, s),
                BlockStructure::Rectangular,
                m.alpha,
                Par::Seq,
            );
            for i in 0..s {
                let (bi, wi) = (m.bi[o + i], m.wi[o + i]);
                let big = &m.big[o..=o + i];
                let delta = &m.delta[o..=o + i];
                for ((cj, &gj), &dj) in row(blk.rb_mut(), i)[..=i].iter_mut().zip(big).zip(delta) {
                    *cj += bi * gj - wi * dj;
                }
            }
        }
        Piece::Tile(mut low, r0, c0) => {
            let (rows, cols) = (low.nrows(), low.ncols());
            for i in 0..rows {
                for cj in row(low.rb_mut(), i).iter_mut() {
                    *cj *= m.scale;
                }
            }
            gemm(
                low.rb_mut(),
                Accum::Add,
                d.subcols(r0, rows).transpose(),
                d.subcols(c0, cols),
                m.alpha,
                Par::Seq,
            );
            let big = &m.big[c0..c0 + cols];
            let delta = &m.delta[c0..c0 + cols];
            for i in 0..rows {
                let (bi, wi) = (m.bi[r0 + i], m.wi[r0 + i]);
                for ((cj, &gj), &dj) in row(low.rb_mut(), i).iter_mut().zip(big).zip(delta) {
                    *cj += bi * gj - wi * dj;
                }
            }
        }
    }
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
