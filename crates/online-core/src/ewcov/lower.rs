//! [`LowerTriangle`]: a symmetric matrix kept as its lower triangle.

use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A symmetric `k × k` matrix kept as its lower triangle alone -- the
/// entries `(i, j)` with `j ≤ i` -- packed row by row: row `i` is `i + 1`
/// numbers from `i(i + 1)/2`. The co-moments of an [`super::EwCov`], and a
/// window's snapshot of them.
///
/// The rank-one update writes `c_ij = a·c_ij + (a·b·d_i)·d_j`, and its
/// mirror would be `(a·b·d_j)·d_i`: the same number in exact arithmetic,
/// but rounded apart, so a matrix updated whole was not symmetric (after
/// 3,000 rows at `k = 20`, 156 of its 190 pairs differed, by up to 9.1e-14
/// of the entry). Every factorization reads the lower triangle alone, so
/// the upper one was work the solves never used. Kept as its lower
/// triangle, the matrix is symmetric by construction and the update does
/// `k(k+1)/2` entries where it did `k²`.
///
/// **Packed, not inside a square.** A window copies the triangle into its
/// ring on every snapshot it takes and back out of it when the snapshot
/// becomes the window's edge. Inside a `k × k` square the triangle is `k`
/// runs, one a row, and each was a call to the library's copy; packed it is
/// one. Measured on arm64 (a standalone loop, 2026-11-09): writing the
/// triangle out took 10-11 ns at `k = 4` and 81 at `k = 30` by rows, 3.2
/// and 34 packed. The update's rows are the same rows, one after another,
/// and run at the same speed (5.9 against 5.9 ns at `k = 4`, 82 against 85
/// at `k = 30`).
///
/// Every read goes through the lower triangle: [`Self::get`] takes `(i, j)`
/// either way round, [`Self::to_full`] mirrors it into a whole matrix, and
/// [`Self::packed`] hands out the packed rows. Equality compares the lower
/// triangles; `Debug` and the state file show the whole matrix, mirrored,
/// so a file still holds `k²` numbers, symmetric, and a file written by a
/// build that rounded the upper triangle on its own loads as its lower
/// triangle mirrored. A file whose matrix is no square, a damaged state's,
/// is kept as it was written, for its owner's shape check to refuse.
#[derive(Clone, Default)]
pub struct LowerTriangle {
    /// The side, `k`; 0 in a damaged matrix.
    k: usize,
    /// The whole matrix's entries, `k²`, or a damaged one's numbers.
    n: usize,
    /// The lower triangle's rows, packed, `k(k + 1)/2`; or, where `n` is no
    /// square, the `n` numbers as written.
    v: Vec<f64>,
}

/// Where row `i` of a packed lower triangle starts.
#[inline]
pub(crate) fn row_at(i: usize) -> usize {
    i * (i + 1) / 2
}

impl LowerTriangle {
    /// The `k × k` zero matrix.
    pub(crate) fn zeros(k: usize) -> Self {
        Self {
            k,
            n: k * k,
            v: vec![0.0; row_at(k)],
        }
    }

    /// `n` numbers that are no square matrix, kept as they are.
    fn damaged(v: Vec<f64>) -> Self {
        Self {
            k: 0,
            n: v.len(),
            v,
        }
    }

    /// `k`, where the matrix is `k × k`; `None` for a damaged state whose
    /// length is no square, which every owner refuses by its shape.
    fn side(&self) -> Option<usize> {
        (self.k * self.k == self.n).then_some(self.k)
    }

    /// The whole matrix's entries, `k²`: what an owner checks the shape
    /// by, and what a window's budget counts of it.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the matrix has no entry (`k = 0`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Entry `(i, j)` of the `k × k` matrix, read from the lower triangle
    /// whichever way round it is asked for.
    #[inline]
    pub fn get(&self, _k: usize, i: usize, j: usize) -> f64 {
        self.v[row_at(i.max(j)) + i.min(j)]
    }

    /// The lower triangle's rows, packed: entry `(i, j)`, `j ≤ i`, at
    /// `i(i + 1)/2 + j`.
    #[inline]
    pub fn packed(&self) -> &[f64] {
        &self.v
    }

    /// The packed rows, to update in place.
    #[inline]
    pub(crate) fn packed_mut(&mut self) -> &mut [f64] {
        &mut self.v
    }

    /// The diagonal, `(i, i)` for each `i` in turn, each row's start carried
    /// to the next rather than found again ([`Self::get`] finds it).
    #[inline]
    pub(crate) fn diagonal(&self) -> impl Iterator<Item = f64> + '_ {
        let mut at = 0;
        (0..self.k).map(move |i| {
            let v = self.v[at + i];
            at += i + 1;
            v
        })
    }

    /// The whole matrix, row-major, its upper triangle mirrored from the
    /// lower: for a reader that takes a full symmetric matrix.
    pub fn to_full(&self) -> Vec<f64> {
        let mut v = Vec::new();
        self.full_into(&mut v);
        v
    }

    /// [`Self::to_full`] into `out`, whatever it held.
    pub fn full_into(&self, out: &mut Vec<f64>) {
        out.clear();
        let Some(k) = self.side() else {
            out.extend_from_slice(&self.v);
            return;
        };
        out.resize(k * k, 0.0);
        self.lower_into(out);
        for i in 1..k {
            for j in 0..i {
                out[j * k + i] = out[i * k + j];
            }
        }
    }

    /// The lower triangle into the row-major `k × k` `out`, row by row; its
    /// strict upper triangle is not written. What a factorization that reads
    /// the lower triangle of a square takes.
    pub(crate) fn lower_into(&self, out: &mut [f64]) {
        let k = self.k;
        let mut rows = self.v.as_slice();
        for (i, dst) in out.chunks_exact_mut(k.max(1)).take(k).enumerate() {
            let (row, rest) = rows.split_at(i + 1);
            rows = rest;
            dst[..=i].copy_from_slice(row);
        }
    }

    /// `C + ridge·I` into the row-major `k × k` `m`, its lower triangle
    /// alone, for a factorization that reads the lower triangle and the
    /// diagonal and nothing else (`SpdFactor::of`, `solve_spd`). `m` is
    /// sized only where its length is not `k²`, so a kept one is not zeroed
    /// again: its strict upper triangle holds whatever it held -- zeros, if
    /// it was sized here -- and is not written. The rows are copied as they
    /// lie, one after the next. Zeroed whole on every call, as
    /// [`Self::lower_into`]'s callers did, the `k²` fill was 3.2M of the
    /// 9.2% more instructions the packed triangle first cost `ew_class`'s
    /// block on x86 (task 374).
    pub(crate) fn ridged_lower_into(&self, m: &mut Vec<f64>, ridge: f64) {
        let k = self.k;
        if m.len() != k * k {
            m.clear();
            m.resize(k * k, 0.0);
        }
        let mut rows = self.v.as_slice();
        for (i, dst) in m.chunks_exact_mut(k.max(1)).take(k).enumerate() {
            let (row, rest) = rows.split_at(i + 1);
            rows = rest;
            for (d, &c) in dst.iter_mut().zip(row) {
                *d = c;
            }
            dst[i] += ridge;
        }
    }

    /// The strict upper triangle, row by row -- pair `(i, j)`, `j > i`, in
    /// the order `(0, 1), (0, 2), .., (1, 2), ..` -- each entry `v` written as
    /// `f(i, j, v)` into `dst`, which holds `k(k-1)/2`. Pair `(i, j)` is the lower entry
    /// `(j, i)`, so an output row is a column of the lower triangle; the
    /// columns are read four at a time, so each row of the storage is read
    /// in runs of four and each of the four output rows is written in
    /// order. A column read alone, one entry a row apart at a time, cost 3.8
    /// instructions a pair more than the contiguous row it replaced.
    pub(crate) fn upper_rows_map(
        &self,
        k: usize,
        mut dst: &mut [f64],
        f: impl Fn(usize, usize, f64) -> f64,
    ) {
        let c = &self.v;
        let at = |i: usize, j: usize| row_at(i) + j;
        let mut i = 0;
        while i + 4 <= k {
            let (r0, rest) = std::mem::take(&mut dst).split_at_mut(k - 1 - i);
            let (r1, rest) = rest.split_at_mut(k - 2 - i);
            let (r2, rest) = rest.split_at_mut(k - 3 - i);
            let (r3, rest) = rest.split_at_mut(k - 4 - i);
            dst = rest;
            // The corner, rows `i + 1 ..= i + 3`, where the four columns
            // have not all begun.
            r0[0] = f(i, i + 1, c[at(i + 1, i)]);
            r0[1] = f(i, i + 2, c[at(i + 2, i)]);
            r1[0] = f(i + 1, i + 2, c[at(i + 2, i + 1)]);
            r0[2] = f(i, i + 3, c[at(i + 3, i)]);
            r1[1] = f(i + 1, i + 3, c[at(i + 3, i + 1)]);
            r2[0] = f(i + 2, i + 3, c[at(i + 3, i + 2)]);
            let outs = r0[3..]
                .iter_mut()
                .zip(r1[2..].iter_mut())
                .zip(r2[1..].iter_mut())
                .zip(r3.iter_mut());
            // Row `j`'s entry `(j, i)`, carried from row to row (`+ j + 1`)
            // rather than found again: `row_at(j)` on every row of the
            // column block was the most of the 706k more instructions the
            // packed rows first cost `ew_cov`'s block on x86 (task 374).
            // Each row split off whole instead, so that the run needed no
            // check of its own, read 2.5% more instructions a row at 20
            // features on arm64, by how the correlations' loop compiled.
            let mut ji = at(i + 4, i);
            for (j, (((o0, o1), o2), o3)) in (i + 4..).zip(outs) {
                let q: &[f64; 4] = c[ji..]
                    .first_chunk()
                    .expect("four columns of a row below them");
                *o0 = f(i, j, q[0]);
                *o1 = f(i + 1, j, q[1]);
                *o2 = f(i + 2, j, q[2]);
                *o3 = f(i + 3, j, q[3]);
                ji += j + 1;
            }
            i += 4;
        }
        while i + 1 < k {
            let (row, rest) = std::mem::take(&mut dst).split_at_mut(k - 1 - i);
            dst = rest;
            let mut ji = at(i + 1, i);
            for (j, o) in (i + 1..).zip(row.iter_mut()) {
                *o = f(i, j, c[ji]);
                ji += j + 1;
            }
            i += 1;
        }
    }

    /// Take a whole row-major matrix of the same size: its lower triangle
    /// is kept, its upper one is not read.
    pub(crate) fn copy_from(&mut self, full: &[f64]) {
        let k = self.k;
        debug_assert_eq!(full.len(), k * k);
        let mut rows = self.v.as_mut_slice();
        for (i, src) in full.chunks_exact(k.max(1)).take(k).enumerate() {
            let (row, rest) = std::mem::take(&mut rows).split_at_mut(i + 1);
            rows = rest;
            row.copy_from_slice(&src[..=i]);
        }
    }

    /// Take a lower triangle of the same size, packed.
    pub(crate) fn copy_from_packed(&mut self, packed: &[f64]) {
        self.v.copy_from_slice(packed);
    }

    /// The matrix with one stored number dropped: a damaged state's shape,
    /// for the tests that refuse one.
    #[cfg(test)]
    pub(crate) fn pop(&mut self) -> Option<f64> {
        let mut full = self.to_full();
        let last = full.pop();
        *self = Self::damaged(full);
        last
    }
}

impl From<Vec<f64>> for LowerTriangle {
    /// A whole row-major matrix, as its lower triangle; a length that is no
    /// square is kept as it is.
    fn from(v: Vec<f64>) -> Self {
        let n = v.len();
        let k = n.isqrt();
        if k * k != n {
            return Self::damaged(v);
        }
        let mut out = Self::zeros(k);
        out.copy_from(&v);
        out
    }
}

impl PartialEq for LowerTriangle {
    /// The lower triangles, entry by entry, as `==` compares doubles; a
    /// matrix that is no square is compared whole.
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.k == other.k && self.v == other.v
    }
}

impl crate::Pack for LowerTriangle {
    /// The lower triangle alone, row by row, after `k`, in one run: what a
    /// window's ring holds of a snapshot's co-moments, about half the
    /// whole matrix. A matrix that is no square, a damaged state's, is
    /// written whole.
    #[inline]
    fn pack(&self, out: &mut crate::Packer) {
        match self.side() {
            Some(k) => {
                out.dim(0);
                out.dim(k);
            }
            None => {
                out.dim(1);
                out.dim(self.n);
            }
        }
        out.f64s(&self.v);
    }

    fn unpack(from: &mut crate::Unpacker<'_>) -> Self {
        let mut out = Self::default();
        out.unpack_into(from);
        out
    }

    #[inline]
    fn unpack_into(&mut self, from: &mut crate::Unpacker<'_>) {
        let whole = from.dim() == 1;
        let n = from.dim();
        let (k, n, len) = if whole {
            (0, n, n)
        } else {
            (n, n * n, row_at(n))
        };
        let words = from.words(len);
        if self.v.len() == len {
            self.v.copy_from_slice(words);
        } else {
            self.v.clear();
            self.v.extend_from_slice(words);
        }
        self.k = k;
        self.n = n;
    }

    #[inline]
    fn refill(&mut self, from: &mut crate::Unpacker<'_>) {
        let words = from.words(self.v.len());
        self.v.copy_from_slice(words);
    }
}

impl std::fmt::Debug for LowerTriangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_full().fmt(f)
    }
}

impl Serialize for LowerTriangle {
    /// The whole matrix, mirrored, as the sequence a `Vec<f64>` writes: the
    /// same bytes a state of the same lower triangle held before the upper
    /// one stopped being kept, wherever the two triangles agreed.
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.n))?;
        match self.side() {
            Some(k) => {
                for i in 0..k {
                    for j in 0..k {
                        seq.serialize_element(&self.get(k, i, j))?;
                    }
                }
            }
            None => {
                for v in &self.v {
                    seq.serialize_element(v)?;
                }
            }
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for LowerTriangle {
    /// A whole matrix, as its lower triangle: one written while the upper
    /// triangle was rounded on its own loads as the lower one, mirrored.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Vec::<f64>::deserialize(d).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_read_is_the_lower_triangle() {
        // An upper triangle that disagrees with the lower one.
        let full = vec![1.0, 9.0, 9.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0];
        let mut m = LowerTriangle::zeros(3);
        m.copy_from(&full);
        for (i, j, v) in [(1, 0, 2.0), (2, 0, 4.0), (2, 1, 5.0)] {
            assert_eq!(m.get(3, i, j), v);
            assert_eq!(m.get(3, j, i), v);
        }
        assert_eq!(m.packed(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.len(), 9);
        let mirrored = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0];
        assert_eq!(m.to_full(), mirrored);
        let mut out = vec![7.0];
        m.full_into(&mut out);
        assert_eq!(out, mirrored);
        let mut square = vec![-1.0; 9];
        m.lower_into(&mut square);
        assert_eq!(square, [1.0, -1.0, -1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0]);
        assert_eq!(format!("{m:?}"), format!("{mirrored:?}"));
        assert_eq!(m, LowerTriangle::from(mirrored.clone()));
        assert_eq!(m, LowerTriangle::from(full.clone()));
        let mut other = m.clone();
        other.packed_mut()[1] = 2.5;
        assert_ne!(m, other, "a lower entry is compared");
        let mut again = LowerTriangle::zeros(3);
        again.copy_from_packed(m.packed());
        assert_eq!(again, m);
    }

    /// The strict upper triangle row by row, mapped, at every `k` the
    /// four-column blocks and the columns left over meet: `0` to `9`.
    #[test]
    fn the_upper_rows_are_the_mirrored_pairs_in_order() {
        for k in 0..10usize {
            let full: Vec<f64> = (0..k * k).map(|v| v as f64 + 0.25).collect();
            let mut m = LowerTriangle::zeros(k);
            m.copy_from(&full);
            let want: Vec<f64> = (0..k)
                .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
                .map(|(i, j)| 3.0 * full[j * k + i])
                .collect();
            let mut got = vec![f64::NAN; want.len()];
            m.upper_rows_map(k, &mut got, |i, j, v| {
                assert!(i < j && j < k);
                3.0 * v
            });
            assert_eq!(got, want, "k = {k}");
        }
    }

    #[test]
    fn the_file_holds_the_whole_matrix_mirrored() {
        let full = vec![1.0, 9.0, 2.0, 3.0];
        let mut m = LowerTriangle::zeros(2);
        m.copy_from(&full);
        let bytes = rmp_serde::to_vec_named(&m).unwrap();
        assert_eq!(
            bytes,
            rmp_serde::to_vec_named(&vec![1.0, 2.0, 2.0, 3.0]).unwrap()
        );
        // A matrix whose upper triangle was rounded on its own loads as its
        // lower triangle, mirrored.
        let old = rmp_serde::to_vec_named(&full).unwrap();
        let back: LowerTriangle = rmp_serde::from_slice(&old).unwrap();
        assert_eq!(back.to_full(), [1.0, 2.0, 2.0, 3.0]);
        assert_eq!(back.packed(), &[1.0, 2.0, 3.0]);
        // A length that is no square is kept as it is, for its owner's
        // shape check to refuse.
        let odd: LowerTriangle =
            rmp_serde::from_slice(&rmp_serde::to_vec(&vec![1.0, 2.0]).unwrap()).unwrap();
        assert_eq!(odd.packed(), &[1.0, 2.0]);
        assert_eq!(odd.len(), 2);
        assert_eq!(
            rmp_serde::to_vec(&odd).unwrap(),
            rmp_serde::to_vec(&vec![1.0, 2.0]).unwrap()
        );
        assert_ne!(odd, LowerTriangle::from(vec![1.0, 3.0]));
    }
}
