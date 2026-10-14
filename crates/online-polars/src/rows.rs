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
//!
//! The columns are the source's own, in its own type: a `Float32` or an
//! integer column is widened to `f64`, and a null read as
//! [`online_core::NULL`], as the copy writes each value. Cast and filled
//! when the chunk was read, a float32 column was a second whole-chunk copy
//! of the features (160 MB a chunk for the same 1,000 columns), the one
//! left after the rows went.

use polars_arrow::array::{Float64Array, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::bitmap::utils::BitmapIter;
use rayon::prelude::*;

use online_core::{NULL, usable};

use crate::arrow::NumArray;

/// A feature column's values in the type the source holds them in.
#[derive(Debug, Clone)]
enum Values {
    F64(PrimitiveArray<f64>),
    F32(PrimitiveArray<f32>),
    I8(PrimitiveArray<i8>),
    I16(PrimitiveArray<i16>),
    I32(PrimitiveArray<i32>),
    I64(PrimitiveArray<i64>),
    U8(PrimitiveArray<u8>),
    U16(PrimitiveArray<u16>),
    U32(PrimitiveArray<u32>),
    U64(PrimitiveArray<u64>),
}

/// `$body` for the values `$v` of every type, as a slice. Each arm is its
/// own code, so `$v[i] as f64` is the conversion of that type: the one a
/// polars cast to `Float64` makes (`AsPrimitive::as_`, which is `as`).
macro_rules! each_type {
    ($values:expr, $v:ident => $body:expr) => {
        match $values {
            Values::F64(b) => {
                let $v: &[f64] = b.values();
                $body
            }
            Values::F32(b) => {
                let $v: &[f32] = b.values();
                $body
            }
            Values::I8(b) => {
                let $v: &[i8] = b.values();
                $body
            }
            Values::I16(b) => {
                let $v: &[i16] = b.values();
                $body
            }
            Values::I32(b) => {
                let $v: &[i32] = b.values();
                $body
            }
            Values::I64(b) => {
                let $v: &[i64] = b.values();
                $body
            }
            Values::U8(b) => {
                let $v: &[u8] = b.values();
                $body
            }
            Values::U16(b) => {
                let $v: &[u16] = b.values();
                $body
            }
            Values::U32(b) => {
                let $v: &[u32] = b.values();
                $body
            }
            Values::U64(b) => {
                let $v: &[u64] = b.values();
                $body
            }
        }
    };
}

/// One feature column as the chunk holds it, read as `f64` with null as
/// [`NULL`]: its buffer and its validity shared with the chunk, nothing
/// copied. The validity is kept only where it has a null to mark.
#[derive(Debug, Clone)]
pub struct FeatureCol {
    values: Values,
    nulls: Option<Bitmap>,
}

/// A validity worth reading: one that marks at least one null.
fn nulls_of(validity: Option<&Bitmap>) -> Option<Bitmap> {
    validity.filter(|v| v.unset_bits() > 0).cloned()
}

impl FeatureCol {
    /// A `Float64` column.
    pub fn f64(a: &Float64Array) -> Self {
        Self {
            values: Values::F64(a.clone()),
            nulls: nulls_of(a.validity()),
        }
    }

    /// A column in the type its source holds it in ([`NumArray`]).
    pub fn num(a: &NumArray) -> Self {
        let (values, validity) = match a {
            NumArray::F32(a) => (Values::F32(a.clone()), a.validity()),
            NumArray::I8(a) => (Values::I8(a.clone()), a.validity()),
            NumArray::I16(a) => (Values::I16(a.clone()), a.validity()),
            NumArray::I32(a) => (Values::I32(a.clone()), a.validity()),
            NumArray::I64(a) => (Values::I64(a.clone()), a.validity()),
            NumArray::U8(a) => (Values::U8(a.clone()), a.validity()),
            NumArray::U16(a) => (Values::U16(a.clone()), a.validity()),
            NumArray::U32(a) => (Values::U32(a.clone()), a.validity()),
            NumArray::U64(a) => (Values::U64(a.clone()), a.validity()),
        };
        Self {
            values,
            nulls: nulls_of(validity),
        }
    }

    /// Values in the column.
    pub fn len(&self) -> usize {
        each_type!(&self.values, v => v.len())
    }

    /// Whether the column has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value at source row `i`, as `f64`, null as [`NULL`].
    #[inline]
    #[allow(
        clippy::unnecessary_cast,
        reason = "one arm of `each_type!` is f64 already"
    )]
    pub fn at(&self, i: usize) -> f64 {
        if let Some(b) = &self.nulls
            && !b.get_bit(i)
        {
            return NULL;
        }
        each_type!(&self.values, v => v[i] as f64)
    }
}

impl From<Vec<f64>> for FeatureCol {
    fn from(v: Vec<f64>) -> Self {
        Self::f64(&Float64Array::from_vec(v))
    }
}

/// The columns' group: one cache line of `f64` in a row. A tile is written
/// a group of columns at a time, through a scratch of [`ROW_TILE`] rows of
/// one group.
const GROUP: usize = 8;

/// The tile's scratch: a row tile's values for one group of columns.
type Scratch = [[f64; GROUP]; ROW_TILE];

/// A column the transpose reads: [`FeatureCol`], or a plain slice of `f64`.
trait Source: Sync {
    /// Source rows `j0..j0 + out.len()` -- or `layout[j]` for each such
    /// position `j` -- as `f64`, into slot `g` of each scratch row.
    fn put(&self, out: &mut [[f64; GROUP]], g: usize, j0: usize, layout: Option<&[usize]>);
}

/// [`Source::put`] over values `vals`, widened by `as f64`, null as
/// [`NULL`] where `nulls` marks it.
#[inline]
fn put_values<T: Copy>(
    vals: &[T],
    widen: impl Fn(T) -> f64,
    nulls: Option<&Bitmap>,
    out: &mut [[f64; GROUP]],
    g: usize,
    j0: usize,
    layout: Option<&[usize]>,
) {
    let n = out.len();
    match (layout, nulls) {
        (None, None) => {
            for (d, &v) in out.iter_mut().zip(&vals[j0..j0 + n]) {
                d[g] = widen(v);
            }
        }
        (None, Some(b)) => {
            let (bytes, offset, _) = b.as_slice();
            let valid = BitmapIter::new(bytes, offset + j0, n);
            for ((d, &v), ok) in out.iter_mut().zip(&vals[j0..j0 + n]).zip(valid) {
                d[g] = if ok { widen(v) } else { NULL };
            }
        }
        (Some(p), None) => {
            for (d, &s) in out.iter_mut().zip(&p[j0..j0 + n]) {
                d[g] = widen(vals[s]);
            }
        }
        (Some(p), Some(b)) => {
            for (d, &s) in out.iter_mut().zip(&p[j0..j0 + n]) {
                d[g] = if b.get_bit(s) { widen(vals[s]) } else { NULL };
            }
        }
    }
}

impl Source for FeatureCol {
    #[allow(
        clippy::unnecessary_cast,
        reason = "one arm of `each_type!` is f64 already"
    )]
    fn put(&self, out: &mut [[f64; GROUP]], g: usize, j0: usize, layout: Option<&[usize]>) {
        let nulls = self.nulls.as_ref();
        each_type!(&self.values, v => put_values(v, |x| x as f64, nulls, out, g, j0, layout))
    }
}

impl Source for &[f64] {
    fn put(&self, out: &mut [[f64; GROUP]], g: usize, j0: usize, layout: Option<&[usize]>) {
        put_values(self, |x| x, None, out, g, j0, layout)
    }
}

/// A spec's feature columns for one chunk, as the chunk holds them: each
/// its source's buffer and validity, nothing copied. Read as rows through
/// [`FeatureCols::rows_into`], a run at a time.
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
        self.cols.iter().fold(true, |ok, c| ok & usable(c.at(src)))
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

/// The transpose's row tile, the unit of parallelism. Inside a tile each
/// column is read for [`ROW_TILE`] rows at once (one type, one branch on its
/// validity), widened into a scratch of one [`GROUP`] of columns that stays
/// in L1, and the scratch is then written out a cache line of each row at a
/// time. The scratch keeps the writes of one column, a row apart, out of
/// the tile itself, where at a width of a power of two they would all land
/// in one set of the cache.
const ROW_TILE: usize = 128;

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
    fn fill<C: Source>(
        &mut self,
        cols: &[C],
        start: usize,
        len: usize,
        layout: Option<&[usize]>,
        par: bool,
    ) {
        let k = cols.len();
        self.k = k;
        self.start = start;
        self.n = len;
        self.data.resize(k * len, 0.0);
        if k == 0 {
            return;
        }
        let fill = |(t, tile): (usize, &mut [f64])| {
            let j0 = start + t * ROW_TILE;
            let rows = tile.len() / k;
            let mut scratch: Scratch = [[0.0; GROUP]; ROW_TILE];
            let scratch = &mut scratch[..rows];
            for c0 in (0..k).step_by(GROUP) {
                let c1 = (c0 + GROUP).min(k);
                for (g, col) in cols[c0..c1].iter().enumerate() {
                    col.put(scratch, g, j0, layout);
                }
                for (row, s) in tile.chunks_exact_mut(k).zip(scratch.iter()) {
                    row[c0..c1].copy_from_slice(&s[..c1 - c0]);
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
    use polars::prelude::NamedFrom;

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
            (1024, 130),
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

    /// A column of each type, with a null every seventh row and the values
    /// a widening can get wrong: NaNs of several payloads (a signalling one
    /// and a negative one among them), both infinities, the extremes, a
    /// subnormal, a negative zero; integers past 2^53, where `as f64`
    /// rounds.
    fn typed_series(n: usize) -> Vec<polars::prelude::Series> {
        use polars::prelude::*;
        fn with_nulls<T: Copy>(n: usize, vals: &[T]) -> Vec<Option<T>> {
            (0..n)
                .map(|i| (i % 7 != 3).then(|| vals[(i * 5 + i / 3) % vals.len()]))
                .collect()
        }
        let f32s = [
            1.5f32,
            -0.0,
            f32::NAN,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc1_2345),
            f32::from_bits(0x7f80_0001),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            1e-7,
            0.1,
            3.4e38,
        ];
        let i64s = [
            i64::MIN,
            i64::MAX,
            (1 << 53) + 1,
            -(1 << 53) - 3,
            0,
            -1,
            123_456_789_012_345_679,
        ];
        let u64s = [u64::MAX, (1 << 63) + 1025, (1 << 53) + 1, 0, 7];
        vec![
            Series::new("f32".into(), with_nulls(n, &f32s)),
            Series::new("i8".into(), with_nulls(n, &[i8::MIN, i8::MAX, 0, -1])),
            Series::new("i16".into(), with_nulls(n, &[i16::MIN, i16::MAX, 3])),
            Series::new("i32".into(), with_nulls(n, &[i32::MIN, i32::MAX, -7])),
            Series::new("i64".into(), with_nulls(n, &i64s)),
            Series::new("u8".into(), with_nulls(n, &[u8::MAX, 0, 9])),
            Series::new("u16".into(), with_nulls(n, &[u16::MAX, 0, 11])),
            Series::new("u32".into(), with_nulls(n, &[u32::MAX, 0, 13])),
            Series::new("u64".into(), with_nulls(n, &u64s)),
        ]
    }

    /// Polars' own cast of `s` to `Float64`, null as `NULL`: what the chunk
    /// held before it kept a column in its own type.
    fn cast(s: &polars::prelude::Series) -> Vec<f64> {
        use polars::prelude::*;
        let f = s.cast(&DataType::Float64).unwrap().rechunk();
        let a = f.f64().unwrap().downcast_iter().next().unwrap().clone();
        crate::arrow::f64_values(&a).into_owned()
    }

    /// A column kept in its own type reads, value by value and through the
    /// copy into rows (in order and through a layout, at every offset), the
    /// bits polars' cast to `Float64` and the fill of its nulls gave.
    #[test]
    fn a_column_in_its_own_type_reads_as_its_cast() {
        let n = 301;
        let series = typed_series(n);
        let wants: Vec<Vec<f64>> = series.iter().map(cast).collect();
        let mut cols: Vec<FeatureCol> = series
            .iter()
            .map(|s| {
                let a = crate::arrow::NumArray::of(s).unwrap().expect("kept as is");
                FeatureCol::num(&a)
            })
            .collect();
        // A float64 column with nulls, and one without.
        let f64s: Vec<Option<f64>> = (0..n)
            .map(|i| (i % 5 != 1).then_some(i as f64 * 0.25 - 7.0))
            .collect();
        let s = polars::prelude::Series::new("f64".into(), f64s);
        let want_f64 = cast(&s);
        let a = s.f64().unwrap().downcast_iter().next().unwrap().clone();
        assert!(a.validity().is_some_and(|v| v.unset_bits() > 0));
        cols.push(FeatureCol::f64(&a));
        let mut wants = wants;
        wants.push(want_f64);
        wants.push((0..n).map(|i| i as f64).collect());
        cols.push(FeatureCol::from(wants.last().unwrap().clone()));
        for (c, want) in cols.iter().zip(&wants) {
            for (i, w) in want.iter().enumerate() {
                assert_eq!(c.at(i).to_bits(), w.to_bits(), "column {c:?} row {i}");
            }
        }
        // Repeated past a group of eight, so a tile holds several groups.
        let k = 3 * cols.len();
        let cols: Vec<FeatureCol> = (0..k).map(|c| cols[c % cols.len()].clone()).collect();
        let perm: Vec<usize> = (0..n).map(|j| (j * 13) % n).collect();
        for layout in [None, Some(perm)] {
            let fc = FeatureCols::new(cols.clone(), n, layout.clone(), false);
            let mut rows = FeatureRows::default();
            for (start, len) in [(0, n), (0, 129), (129, 1), (130, 171)] {
                fc.rows_into(&mut rows, start, len);
                let whole =
                    FeatureCols::new(cols.clone(), n, layout.clone(), true).rows(start, len);
                for i in start..start + len {
                    let src = layout.as_ref().map_or(i, |p| p[i]);
                    for (c, (got, par)) in rows.row(i).iter().zip(whole.row(i)).enumerate() {
                        let w = wants[c % wants.len()][src];
                        assert_eq!(got.to_bits(), w.to_bits(), "row {i} column {c}");
                        assert_eq!(par.to_bits(), w.to_bits(), "row {i} column {c}");
                    }
                    assert_eq!(fc.usable_at(i), online_core::all_usable(rows.row(i)));
                }
            }
        }
    }

    /// The null marker is `NULL` itself, told apart from a NaN the data
    /// holds, in every type and through the tile.
    #[test]
    fn a_null_is_null_and_a_nan_is_not() {
        let s = polars::prelude::Series::new("x".into(), vec![Some(f32::NAN), None, Some(1.0f32)]);
        let a = crate::arrow::NumArray::of(&s).unwrap().unwrap();
        let fc = FeatureCols::new(vec![FeatureCol::num(&a)], 3, None, false);
        let rows = fc.rows(0, 3);
        assert!(!online_core::is_null(rows.row(0)[0]) && rows.row(0)[0].is_nan());
        assert!(online_core::is_null(rows.row(1)[0]));
        assert_eq!(rows.row(2), &[1.0]);
    }
}
