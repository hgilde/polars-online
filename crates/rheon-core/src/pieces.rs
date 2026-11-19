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
//! the bit. On x86-64 the same calls reach the same crate (`gemm`; the
//! workspace Cargo.toml says why faer's own x86 kernel is left out, task
//! 258), with that architecture's kernels, which have not been measured
//! here to sum a tile as the whole: so there the bits are the pieces' at
//! every thread count, and may differ in the last place from the single
//! call's. On both, every entry is held to an exactly rounded oracle within
//! its rounding bound (`oracle_tests`).

use faer::linalg::matmul::matmul as gemm;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::{Accum, MatMut, MatRef, Par};
use std::cell::Cell;
use std::sync::{Mutex, PoisonError};

thread_local! {
    static SEQUENTIAL: Cell<bool> = const { Cell::new(false) };
}

/// While the guard lives, the matrix work started on this thread runs on
/// this thread, in order, whatever the rayon pool the thread belongs to or
/// does not: a call that is small enough not to be handed to a pool
/// (`rheon-polars`' inline call) must not reach rayon's global pool by
/// way of a wide solve, where it would ignore the `threads` the bank was
/// given and wake threads the caller never asked for. The bands and pieces
/// are the same, cut by shape alone, so the bits do not change; only the
/// thread that makes each call.
pub struct Sequential(bool);

impl Sequential {
    /// Marks this thread; the guard puts back what it found.
    #[must_use]
    pub fn enter() -> Self {
        Self(SEQUENTIAL.with(|c| c.replace(true)))
    }
}

impl Drop for Sequential {
    fn drop(&mut self) {
        SEQUENTIAL.with(|c| c.set(self.0));
    }
}

/// The parallel degree of matrix work started here: the current pool's
/// size, or one under a [`Sequential`] guard.
pub(crate) fn degree() -> usize {
    if SEQUENTIAL.with(Cell::get) {
        1
    } else {
        rayon::current_num_threads()
    }
}

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
    /// A tile of a bottom-left rectangle at rows `r0`, columns `c0`.
    Tile(MatMut<'a, f64>, usize, usize),
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
    let (tl, _, bl, br) = dst.split_at_mut(h, h);
    tiles(bl, o + h, o, cut.1, out);
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

/// `low` is the rectangle at rows `r0`, columns `c0`, cut by
/// [`first_part`].
fn tiles<'a>(low: MatMut<'a, f64>, r0: usize, c0: usize, tile: usize, out: &mut Vec<Piece<'a>>) {
    let (rows, cols) = (low.nrows(), low.ncols());
    if let Some(first) = first_part(rows, tile) {
        let (la, lb) = low.split_at_row_mut(first);
        tiles(la, r0, c0, tile, out);
        tiles(lb, r0 + first, c0, tile, out);
    } else if let Some(first) = first_part(cols, tile) {
        let (la, lb) = low.split_at_col_mut(first);
        tiles(la, r0, c0, tile, out);
        tiles(lb, r0, c0 + first, tile, out);
    } else {
        out.push(Piece::Tile(low, r0, c0));
    }
}

/// Where one of [`pieces`]' pieces lies: rows `r0..r0 + rows`, columns
/// `c0..c0 + cols`; a diagonal block's (`diag`) is its lower triangle with
/// the diagonal, `r0 == c0` and `rows == cols`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Place {
    pub(crate) diag: bool,
    pub(crate) r0: usize,
    pub(crate) c0: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

/// The places of the pieces [`pieces`] cuts a `k × k` square into, in its
/// order, with no square to cut: for work that lays each piece out where
/// its thread keeps it. The same recursion, on sizes alone
/// (`places_are_where_the_pieces_lie`).
pub(crate) fn places(k: usize, cut: (usize, usize)) -> Vec<Place> {
    let mut out = Vec::new();
    plan_places(0, k, cut, &mut out);
    out
}

fn plan_places(o: usize, s: usize, cut: (usize, usize), out: &mut Vec<Place>) {
    if s <= cut.0 {
        out.push(Place {
            diag: true,
            r0: o,
            c0: o,
            rows: s,
            cols: s,
        });
        return;
    }
    let h = s / 2;
    tile_places(o + h, o, s - h, h, cut.1, out);
    plan_places(o, h, cut, out);
    plan_places(o + h, s - h, cut, out);
}

fn tile_places(r0: usize, c0: usize, rows: usize, cols: usize, tile: usize, out: &mut Vec<Place>) {
    if let Some(first) = first_part(rows, tile) {
        tile_places(r0, c0, first, cols, tile, out);
        tile_places(r0 + first, c0, rows - first, cols, tile, out);
    } else if let Some(first) = first_part(cols, tile) {
        tile_places(r0, c0, rows, first, tile, out);
        tile_places(r0, c0 + first, rows, cols - first, tile, out);
    } else {
        out.push(Place {
            diag: false,
            r0,
            c0,
            rows,
            cols,
        });
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
        Piece::Tile(low, r0, c0) => {
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
    crate::scratch::assert_none_borrowed();
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
            #[cfg(test)]
            crate::test_pool::before_item();
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
mod oracle_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The degree is the pool's size, one under a [`Sequential`] guard on
    /// any thread of any pool, and the guard puts back what it found.
    #[test]
    fn a_sequential_guard_takes_the_degree_to_one_and_back() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(degree(), 3);
            let outer = Sequential::enter();
            assert_eq!(degree(), 1);
            let inner = Sequential::enter();
            drop(inner);
            assert_eq!(degree(), 1, "the inner guard leaves the outer's");
            drop(outer);
            assert_eq!(degree(), 3);
        });
    }

    /// Bands of right-hand sides are solved the same, bit for bit, under the
    /// guard as spread over a pool.
    #[test]
    fn a_wide_solve_is_the_same_to_the_bit_under_the_guard() {
        let k = 6;
        let a: Vec<f64> = (0..k * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                if r == c {
                    4.0 + r as f64
                } else {
                    1.0 / (1.0 + (r + c) as f64)
                }
            })
            .collect();
        let m = 300;
        let b: Vec<f64> = (0..k * m).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let spread = pool.install(|| crate::solve_spd(&a, &b, k, m).unwrap().0);
        let seq = pool.install(|| {
            let _g = Sequential::enter();
            crate::solve_spd(&a, &b, k, m).unwrap().0
        });
        assert!(
            spread
                .iter()
                .zip(&seq)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    /// [`places`] are where [`pieces`] cuts, piece for piece, in order, at
    /// every side up to past four leaves and at a fine cut.
    #[test]
    fn places_are_where_the_pieces_lie() {
        for cut in [(LEAF, TILE), (16, 8), (128, 65)] {
            for k in (0..=40).chain([255, 256, 257, 300, 511, 512, 513, 700, 1000, 1100]) {
                let mut m = vec![0.0; k * k];
                let want: Vec<Place> = pieces(MatMut::from_row_major_slice_mut(&mut m, k, k), cut)
                    .into_iter()
                    .map(|p| match p {
                        Piece::Diag(b, o) => Place {
                            diag: true,
                            r0: o,
                            c0: o,
                            rows: b.nrows(),
                            cols: b.ncols(),
                        },
                        Piece::Tile(b, r0, c0) => Place {
                            diag: false,
                            r0,
                            c0,
                            rows: b.nrows(),
                            cols: b.ncols(),
                        },
                    })
                    .collect();
                assert_eq!(places(k, cut), want, "k = {k}, cut {cut:?}");
            }
        }
    }

    /// The cut the docs state: `k = 1,000` is 15 pieces at the shipped cut
    /// ([`LEAF`]) -- four diagonal blocks of 250, a 250 × 250 tile in each
    /// half, and nine tiles of about 167 on a side for the 500 × 500
    /// rectangle between them; a rectangle is cut into `len / tile` parts,
    /// none narrower than `tile`, and one part below `2·tile`; bands of
    /// right-hand sides follow the same rule.
    #[test]
    fn the_cut_is_the_documented_one() {
        let sides = |k: usize, cut: (usize, usize)| -> Vec<(usize, usize, bool)> {
            let mut m = vec![0.0; k * k];
            pieces(MatMut::from_row_major_slice_mut(&mut m, k, k), cut)
                .into_iter()
                .map(|p| match p {
                    Piece::Diag(b, _) => (b.nrows(), b.ncols(), true),
                    Piece::Tile(b, _, _) => (b.nrows(), b.ncols(), false),
                })
                .collect()
        };
        let cut = sides(1000, (LEAF, TILE));
        assert_eq!(cut.len(), 15);
        assert_eq!(cut.iter().filter(|p| p.2).count(), 4);
        let tiles: Vec<(usize, usize)> = cut.iter().filter(|p| !p.2).map(|p| (p.0, p.1)).collect();
        assert!(
            tiles.contains(&(167, 167)) && tiles.contains(&(250, 250)),
            "{tiles:?}"
        );
        assert!(
            tiles.iter().all(|&(r, c)| r >= TILE && c >= TILE),
            "{tiles:?}"
        );
        assert_eq!(sides(256, (LEAF, TILE)).len(), 1);
        for (len, want) in [
            (1, None),
            (255, None),
            (256, Some(128)),
            (300, Some(150)),
            (383, Some(192)),
            (384, Some(128)),
        ] {
            assert_eq!(first_part(len, TILE), want, "len {len}");
        }
        let widths = |n: usize| -> Vec<usize> {
            let mut m = vec![0.0; 2 * n];
            col_parts(MatMut::from_row_major_slice_mut(&mut m, 2, n), TILE)
                .iter()
                .map(|b| b.ncols())
                .collect()
        };
        assert_eq!(widths(100), vec![100]);
        assert_eq!(widths(300), vec![150, 150]);
        assert_eq!(widths(1000), vec![143, 143, 143, 143, 143, 143, 142]);
    }
}
