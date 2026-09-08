//! A chunk's features laid out row by row (docs/PERFORMANCE.md §20).
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
//! was. One tiled transpose per chunk here, and everything after it reads
//! contiguous rows.

use rayon::prelude::*;

/// A chunk's feature values, row after row: row `i` is the `k` contiguous
/// values from `i * k`. Built once per chunk by [`FeatureRows::from_columns`]
/// and read by [`FeatureRows::row`].
#[derive(Debug, Default)]
pub struct FeatureRows {
    data: Vec<f64>,
    k: usize,
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
    /// a layout. Every column holds `n` values and `layout`, when given, is
    /// a permutation of `0..n` (the bank's group-blocked order, docs/
    /// PERFORMANCE.md P9), so the gather that used to be a pass per column
    /// is folded into the transpose. `par` spreads the row tiles over the
    /// rayon pool.
    pub fn from_columns(cols: &[&[f64]], n: usize, layout: Option<&[usize]>, par: bool) -> Self {
        let k = cols.len();
        debug_assert!(cols.iter().all(|c| c.len() == n));
        debug_assert!(layout.is_none_or(|p| p.len() == n));
        let mut data = vec![0.0; k * n];
        if k > 0 {
            let fill = |(t, tile): (usize, &mut [f64])| {
                let j0 = t * ROW_TILE;
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
                data.par_chunks_mut(ROW_TILE * k).enumerate().for_each(fill);
            } else {
                data.chunks_mut(ROW_TILE * k).enumerate().for_each(fill);
            }
        }
        Self { data, k, n }
    }

    /// Features per row.
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Rows in the chunk.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Row `i`'s `k` features, contiguous. Empty for a spec without features.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.k..(i + 1) * self.k]
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
    }
}
