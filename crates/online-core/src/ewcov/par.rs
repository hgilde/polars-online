//! The co-moment matrix's `k²` work on several threads, to the same bits as
//! on one (`gram_threads`, docs/PLAN.md task 225).
//!
//! Two kinds of work touch every entry of `C`. The per-row rank-1 update is
//! one expression per entry, so a split of the rows across threads cannot
//! move a bit: [`rows`]. A block's merge ([`super::EwCov::flush`]) is a
//! product, `C += a·DᵀD`, where an entry is a sum over the block's rows, and
//! a parallel product whose split depends on the thread count sums an entry
//! in another order on another count. [`merge_lower`] fixes the split by `k`
//! alone: it walks the recursion faer's own triangular product walks
//! (halve; the bottom-left block is a rectangle; recurse on the two diagonal
//! blocks), stops at diagonal blocks of [`LEAF`] columns, and cuts each
//! rectangle into tiles of [`TILE`] to `2·TILE − 1` on a side. Each piece is
//! one sequential faer call on one thread, so every entry is summed by one
//! call in one order, and the thread count decides only which thread makes
//! the call. The pieces are the calls faer's sequential product makes, and
//! its kernel sums a tile of a rectangle as it sums the whole rectangle, so
//! on aarch64 the result is that product's to the bit, which the tests
//! check: the bits a build before task 225 wrote. On x86-64 faer hands the
//! whole triangle to one kernel, whose blocking the pieces do not
//! reproduce, so there the bits are the pieces' at every thread count, and
//! may differ in the last place from the single call's.
//!
//! The rest of a merge is per entry -- the history's scale before the
//! product, the two rank-1 corrections after it, the mirror into the upper
//! triangle -- and runs inside the same piece, while its tile is in cache:
//! one pass over `C` where there were four. Each entry still takes the
//! same operations in the same order.

use faer::linalg::matmul::matmul as gemm;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};
use std::sync::{Mutex, PoisonError};

/// A diagonal block this wide or narrower is one triangular product, so a
/// matrix of 256 columns or fewer is one piece. Measured with [`TILE`]
/// (`examples/gram_kernel_probe.rs`, `examples/gram_threads_bench.rs`,
/// 2026-10-08): one thread costs what the single call did (24.4 to 24.8 ps
/// per entry and row at `k = 1,000` to 4,000, against 24.6 to 25.7 before,
/// the two builds run in turn), and `k = 1,000` is 15 pieces, enough to
/// keep fourteen threads busy.
/// Changing either constant can move the last bits on x86-64 (the module
/// doc), never on aarch64.
pub(crate) const LEAF: usize = 256;

/// The least side of a tile a rectangle is cut into ([`tiles`]).
pub(crate) const TILE: usize = 128;

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
/// `+ (bi[i]·big[j] − wi[i]·delta[j])`, then the mirror `C[j][i] = C[i][j]`.
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

enum Piece<'a> {
    /// A diagonal block at `o`, its lower triangle with the diagonal.
    Diag(MatMut<'a, f64>, usize),
    /// A tile of a bottom-left rectangle at rows `r0`, columns `c0`, and
    /// the tile of the upper triangle it is mirrored into.
    Tile(MatMut<'a, f64>, MatMut<'a, f64>, usize, usize),
}

/// The merge of a block `D` (`n × k`, row-major, rows already centred and
/// weighted) into the row-major `k × k` matrix `c`, on `threads` threads.
pub(crate) fn merge_lower(
    c: &mut [f64],
    k: usize,
    d: &[f64],
    n: usize,
    m: &Merge<'_>,
    threads: usize,
) {
    merge_lower_cut(c, k, d, n, m, threads, (LEAF, TILE));
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

fn pieces(c: MatMut<'_, f64>, cut: (usize, usize)) -> Vec<Piece<'_>> {
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

/// `low` is the rectangle at rows `r0`, columns `c0`; `up` is its mirror,
/// at rows `c0`, columns `r0`. Cut both alike, each side into `len / tile`
/// near-equal parts, so a part is never narrower than `tile`: a sliver --
/// one row, or a tile of 64 or fewer on a side -- would take another of
/// faer's kernels (a matrix-vector product, a small-matrix kernel with its
/// own blocking of the rows), which sums an entry in another order than
/// the rectangle's single call does.
fn tiles<'a>(
    low: MatMut<'a, f64>,
    up: MatMut<'a, f64>,
    r0: usize,
    c0: usize,
    tile: usize,
    out: &mut Vec<Piece<'a>>,
) {
    let (rows, cols) = (low.nrows(), low.ncols());
    let parts = |len: usize| (len / tile).max(1);
    if parts(rows) > 1 {
        let first = rows.div_ceil(parts(rows));
        let (la, lb) = low.split_at_row_mut(first);
        let (ua, ub) = up.split_at_col_mut(first);
        tiles(la, ua, r0, c0, tile, out);
        tiles(lb, ub, r0 + first, c0, tile, out);
    } else if parts(cols) > 1 {
        let first = cols.div_ceil(parts(cols));
        let (la, lb) = low.split_at_col_mut(first);
        let (ua, ub) = up.split_at_row_mut(first);
        tiles(la, ua, r0, c0, tile, out);
        tiles(lb, ub, r0, c0 + first, tile, out);
    } else {
        out.push(Piece::Tile(low, up, r0, c0));
    }
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
            for i in 1..s {
                for j in 0..i {
                    let v = blk[(i, j)];
                    blk[(j, i)] = v;
                }
            }
        }
        Piece::Tile(mut low, mut up, r0, c0) => {
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
            up.copy_from(low.rb().transpose());
        }
    }
}

/// `f(i, row)` for each row of the row-major `k`-wide matrix `c`, on
/// `threads` threads: each takes whole bands of rows, so an entry's
/// arithmetic is the same whichever thread runs it.
pub(crate) fn rows(c: &mut [f64], k: usize, threads: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    if k == 0 {
        return;
    }
    let n_rows = c.len() / k;
    let band = n_rows.div_ceil(threads.max(1)).max(1);
    let bands: Vec<(usize, &mut [f64])> = c.chunks_mut(band * k).enumerate().collect();
    spread(bands, threads, |(b, rows)| {
        for (r, row) in rows.chunks_mut(k).enumerate() {
            f(b * band + r, row);
        }
    });
}

/// `f` on every item, in order on this thread when `threads <= 1`, else on
/// up to `threads` workers of the current rayon pool -- the bank's, inside a
/// bank -- each taking the next item until none is left. Which worker takes
/// which item does not matter: the items are disjoint and each is done
/// whole by one call to `f`.
fn spread<T: Send>(items: Vec<T>, threads: usize, f: impl Fn(T) + Sync) {
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

#[cfg(test)]
mod tests;
