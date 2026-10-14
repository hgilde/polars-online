//! A chunk's features, read a run of rows at a time (docs/PERFORMANCE.md
//! §20, §42).
//!
//! The bank reads a frame the way polars stores it, a column at a time, but
//! everything downstream reads it a row at a time: the accept test looks at
//! every feature of the row, and a model's `step` takes the row as one
//! slice. Kept as columns, each of those reads was a walk across `k`
//! separate allocations. At `k = 20` the walk is twenty cache lines that
//! stay hot for the next seven rows; at `k = 10,000` it is ten thousand
//! pages per row, and the walk is bound by the TLB rather than by anything
//! it computes -- it was most of the 78-130 µs a wide `sgd` row cost
//! through the bank, against 22.5 µs for the model's own step as it then
//! was. So the rows are transposed, in tiles, and everything after reads
//! contiguous rows.
//!
//! The transpose used to write the whole chunk out in rows before any
//! stream read it: 160 MB a chunk at 20,000 rows of 1,000 features,
//! written to memory and read back, which made it bound by memory and cost
//! about 0.9 ns per feature and row on one thread. Now the chunk keeps its
//! columns ([`FeatureCols`]), and each stream copies the rows of the run it
//! is about to learn into a buffer of its own ([`FeatureRows`]), reused run
//! after run. A run is sized to stay in cache with its outputs
//! (`ChunkOut::run_rows`), so the rows are read from cache right after they
//! were written, and the whole chunk is never held in rows.

use std::ops::Deref;

use polars_arrow::array::{Array, Float64Array};
use rayon::prelude::*;

use online_core::usable;

/// One feature column: `f64`, NaN for null, with no validity to consult.
/// The chunk's own array where it has no nulls (its buffer shared, not
/// copied), the filled copy otherwise.
#[derive(Debug, Clone)]
pub struct FeatureCol(Float64Array);

impl FeatureCol {
    /// `values` as a column; a validity it carries is not read, so the
    /// caller fills nulls in first (`crate::arrow::f64_filled`).
    pub fn new(values: Float64Array) -> Self {
        debug_assert!(values.null_count() == 0 || values.validity().is_none());
        Self(values)
    }
}

impl From<Vec<f64>> for FeatureCol {
    fn from(v: Vec<f64>) -> Self {
        Self(Float64Array::from_vec(v))
    }
}

impl Deref for FeatureCol {
    type Target = [f64];
    #[inline]
    fn deref(&self) -> &[f64] {
        self.0.values().as_slice()
    }
}

/// A spec's feature columns for one chunk, as the chunk holds them: `f64`,
/// NaN for null, no copy where a column has no nulls to fill in. Read as
/// rows through [`FeatureCols::rows_into`], a run at a time.
#[derive(Debug, Default)]
pub struct FeatureCols {
    cols: Vec<FeatureCol>,
    /// The bank's group-blocked order (docs/PERFORMANCE.md P9): position
    /// `j` is source row `layout[j]`. `None` is the chunk's own order.
    layout: Option<Vec<usize>>,
    n: usize,
    /// Whether a copy of many rows at once may spread over the pool: for a
    /// lone spec with features, whose copy would otherwise be one thread's
    /// while the rest wait.
    par: bool,
}

impl FeatureCols {
    /// Every column holds `n` values, and `layout`, when given, is a
    /// permutation of `0..n`.
    pub fn new(cols: Vec<FeatureCol>, n: usize, layout: Option<Vec<usize>>, par: bool) -> Self {
        debug_assert!(cols.iter().all(|c| c.len() == n));
        debug_assert!(layout.as_ref().is_none_or(|p| p.len() == n));
        Self {
            cols,
            layout,
            n,
            par,
        }
    }

    /// Features per row.
    #[inline]
    pub fn k(&self) -> usize {
        self.cols.len()
    }

    /// Rows in the chunk.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether every feature at position `i` is [`usable`]: the same test as
    /// [`online_core::all_usable`] over the row, read from the columns.
    #[inline]
    pub fn usable_at(&self, i: usize) -> bool {
        let src = self.layout.as_ref().map_or(i, |p| p[i]);
        self.cols.iter().fold(true, |ok, c| ok & usable(c[src]))
    }

    /// Positions `start..start + len` into `rows`, reusing its buffer. On the
    /// caller's thread: a stream copies its own rows.
    pub fn rows_into(&self, rows: &mut FeatureRows, start: usize, len: usize) {
        debug_assert!(start + len <= self.n);
        rows.fill(&self.cols, start, len, self.layout.as_deref(), false);
    }

    /// Positions `start..start + len` as rows of their own: for a pass that
    /// reads many rows at once ([`Self::rows_into`] for a run), spread over
    /// the pool for a lone spec.
    pub fn rows(&self, start: usize, len: usize) -> FeatureRows {
        debug_assert!(start + len <= self.n);
        let mut rows = FeatureRows::default();
        rows.fill(&self.cols, start, len, self.layout.as_deref(), self.par);
        rows
    }
}

/// Feature values, row after row, for the positions `start..start + n` of a
/// chunk: position `i` is the `k` contiguous values from `(i - start) * k`.
/// Filled by [`FeatureCols::rows_into`] or [`FeatureCols::rows`], and read by
/// [`FeatureRows::row`] with the position in the chunk.
#[derive(Debug, Default)]
pub struct FeatureRows {
    data: Vec<f64>,
    k: usize,
    start: usize,
    n: usize,
}

/// The transpose's tile: `ROW_TILE` rows by `COL_TILE` columns. Inside a
/// tile the reads stay within `COL_TILE` columns -- one cache line of each
/// serves eight consecutive rows -- and the writes are contiguous runs of
/// `COL_TILE` values; both working sets fit L1 at any `k`. Row tiles are
/// the unit of parallelism.
const ROW_TILE: usize = 128;
const COL_TILE: usize = 64;

impl FeatureRows {
    /// Row `j`, column `c` is `cols[c][layout[j]]`, or `cols[c][j]` without
    /// a layout, for `j` in `0..n`. `par` spreads the row tiles over the
    /// rayon pool.
    pub fn from_columns(cols: &[&[f64]], n: usize, layout: Option<&[usize]>, par: bool) -> Self {
        debug_assert!(cols.iter().all(|c| c.len() == n));
        debug_assert!(layout.is_none_or(|p| p.len() == n));
        let mut rows = Self::default();
        rows.fill(cols, 0, n, layout, par);
        rows
    }

    /// Positions `start..start + len` of `cols`, the gather of `layout` folded
    /// into the transpose. Every value of the buffer is written, so a buffer
    /// reused at the same size is not cleared first.
    fn fill<C>(&mut self, cols: &[C], start: usize, len: usize, layout: Option<&[usize]>, par: bool)
    where
        C: Deref<Target = [f64]> + Sync,
    {
        let k = cols.len();
        self.k = k;
        self.start = start;
        self.n = len;
        self.data.resize(k * len, 0.0);
        if k == 0 {
            return;
        }
        // The columns as plain slices, once, so the tile's inner loop reads a
        // pointer and a length per column rather than each column's handle.
        let cols: Vec<&[f64]> = cols.iter().map(|c| &c[..]).collect();
        let cols = &cols[..];
        let fill = |(t, tile): (usize, &mut [f64])| {
            let j0 = start + t * ROW_TILE;
            for c0 in (0..k).step_by(COL_TILE) {
                let c1 = (c0 + COL_TILE).min(k);
                for (r, row) in tile.chunks_exact_mut(k).enumerate() {
                    let src = layout.map_or(j0 + r, |p| p[j0 + r]);
                    for (dst, col) in row[c0..c1].iter_mut().zip(&cols[c0..c1]) {
                        *dst = col[src];
                    }
                }
            }
        };
        if par {
            self.data
                .par_chunks_mut(ROW_TILE * k)
                .enumerate()
                .for_each(fill);
        } else {
            self.data
                .chunks_mut(ROW_TILE * k)
                .enumerate()
                .for_each(fill);
        }
    }

    /// Features per row.
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Rows held.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The `k` features at position `i` of the chunk, contiguous; `i` is one
    /// of the positions held. Empty for a spec without features.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let j = i - self.start;
        &self.data[j * self.k..(j + 1) * self.k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(k: usize, n: usize) -> Vec<Vec<f64>> {
        (0..k)
            .map(|c| (0..n).map(|j| (c * 100_000 + j) as f64).collect())
            .collect()
    }

    fn check(k: usize, n: usize, layout: Option<&[usize]>, par: bool) {
        let cols = columns(k, n);
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let rows = FeatureRows::from_columns(&refs, n, layout, par);
        assert_eq!((rows.k(), rows.n()), (k, n));
        for j in 0..n {
            let src = layout.map_or(j, |p| p[j]);
            let want: Vec<f64> = cols.iter().map(|c| c[src]).collect();
            assert_eq!(rows.row(j), &want[..], "k={k} n={n} row {j}");
        }
    }

    /// Every tile shape: exact tiles, ragged edges in both directions, one
    /// row, one column, and shapes below a single tile.
    #[test]
    fn transpose_is_exact_at_every_shape() {
        for &(k, n) in &[
            (1, 1),
            (1, 300),
            (3, 1),
            (20, 4097),
            (64, 128),
            (65, 129),
            (130, 257),
            (200, 5),
        ] {
            check(k, n, None, false);
            check(k, n, None, true);
        }
    }

    /// The layout is applied as a gather of source rows, per row.
    #[test]
    fn layout_gathers_rows() {
        let n = 301;
        let perm: Vec<usize> = (0..n).map(|j| (j * 7) % n).collect();
        check(70, n, Some(&perm), false);
        check(70, n, Some(&perm), true);
    }

    /// No features: no data, and every row is the empty slice.
    #[test]
    fn no_features() {
        let rows = FeatureRows::from_columns(&[], 5, None, false);
        assert_eq!((rows.k(), rows.n()), (0, 5));
        assert!(rows.row(4).is_empty());
        let rows = FeatureRows::from_columns(&[&[1.0, 2.0]], 2, None, true);
        assert_eq!(rows.row(1), &[2.0]);
        let cols = FeatureCols::new(Vec::new(), 9, None, false);
        let mut rows = FeatureRows::default();
        cols.rows_into(&mut rows, 3, 4);
        assert_eq!((rows.k(), rows.n()), (0, 4));
        assert!(rows.row(3).is_empty() && rows.row(6).is_empty());
        assert!(cols.usable_at(8));
    }

    fn feature_cols(k: usize, n: usize, layout: Option<Vec<usize>>, par: bool) -> FeatureCols {
        let cols = columns(k, n).into_iter().map(FeatureCol::from).collect();
        FeatureCols::new(cols, n, layout, par)
    }

    /// A run's rows, read by their position in the chunk, are the chunk's
    /// rows at those positions: at every offset and length, through the
    /// layout, into one buffer reused across runs of different sizes (a
    /// shorter run after a longer one leaves nothing of it behind), and the
    /// same whether the copy spreads over the pool.
    #[test]
    fn runs_are_the_chunks_rows_at_their_positions() {
        let n = 1_000;
        let perm: Vec<usize> = (0..n).map(|j| (j * 13) % n).collect();
        for (k, layout) in [(1, None), (65, None), (130, Some(perm.clone()))] {
            let cols = columns(k, n);
            let fc = feature_cols(k, n, layout.clone(), false);
            let par = feature_cols(k, n, layout.clone(), true);
            assert_eq!((fc.k(), fc.n()), (k, n));
            let mut rows = FeatureRows::default();
            for (start, len) in [
                (0, 300),
                (300, 129),
                (429, 1),
                (430, 570),
                (999, 1),
                (0, 1000),
            ] {
                fc.rows_into(&mut rows, start, len);
                let whole = par.rows(start, len);
                assert_eq!((rows.k(), rows.n()), (k, len));
                for i in start..start + len {
                    let src = layout.as_ref().map_or(i, |p| p[i]);
                    let want: Vec<f64> = cols.iter().map(|c| c[src]).collect();
                    assert_eq!(rows.row(i), &want[..], "k={k} start={start} row {i}");
                    assert_eq!(whole.row(i), &want[..], "k={k} start={start} row {i}");
                }
            }
        }
    }

    /// `usable_at` is `all_usable` of the row the run would hold, for a
    /// null (NaN), either infinity, a value past the bound and a clean row,
    /// through the layout.
    #[test]
    fn usable_at_is_all_usable_of_the_row() {
        let n = 6;
        let a = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, f64::INFINITY, 4.0, 1e300, f64::NEG_INFINITY];
        let perm = vec![5, 4, 3, 2, 1, 0];
        for layout in [None, Some(perm)] {
            let fc = FeatureCols::new(
                vec![FeatureCol::from(a.clone()), FeatureCol::from(b.clone())],
                n,
                layout.clone(),
                false,
            );
            let rows = fc.rows(0, n);
            for i in 0..n {
                assert_eq!(
                    fc.usable_at(i),
                    online_core::all_usable(rows.row(i)),
                    "row {i}"
                );
            }
            let src = |i: usize| layout.as_ref().map_or(i, |p| p[i]);
            assert!(fc.usable_at((0..n).find(|&i| src(i) == 3).unwrap()));
            assert!(!fc.usable_at((0..n).find(|&i| src(i) == 1).unwrap()));
        }
    }
}
