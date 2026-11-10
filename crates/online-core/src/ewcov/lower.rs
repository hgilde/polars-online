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

    /// The diagonal, `(i, i)` for each `i` in turn: each row split off the
    /// packed rows in turn, and its last entry read, rather than its place
    /// found again ([`Self::get`] finds it).
    #[inline]
    pub(crate) fn diagonal(&self) -> impl Iterator<Item = f64> + '_ {
        let mut rows = self.v.as_slice();
        (0..self.k).map_while(move |i| {
            let (row, rest) = rows.split_at_checked(i + 1)?;
            rows = rest;
            row.last().copied()
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
    /// `f(v, w[i], w[j])` into `dst`, which holds `k(k-1)/2`; `w` holds `k`
    /// numbers, which only `f` reads. Pair `(i, j)` is the lower entry
    /// `(j, i)`, so an output row is a column of the lower triangle.
    ///
    /// The triangle is read in the order it is stored, four lower rows at a
    /// time: rows `j ..= j + 3` side by side, along `i < j`, give pairs
    /// `(i, j ..= j + 3)`, four neighbours in output row `i`, written as one
    /// run; the run's place moves down the output column by a count, `k − i
    /// − 2` more each step, so no place is found by a multiply and no read
    /// is strided. The pairs among the four rows themselves, six, follow.
    /// The rows the fours leave over are read first, one at a time. The same `f` of the same
    /// numbers into the same slots; only the order of the writes, to
    /// distinct slots, differs. `w`'s entries reach `f` with the rows they
    /// go with, not indexed a pair at a time.
    ///
    /// The packed rows first cost `ew_cov`'s bench block 706k more
    /// instructions than the square rows on x86 read column by column, four
    /// at a time, each run's place found again by `row_at`, and 378k with
    /// the places carried (tasks 371 and 374). Walked one stored row at a
    /// time, each pair written to its own place, every write checked, it
    /// cost 24% more a row at 20 features on arm64.
    pub(crate) fn upper_rows_map(
        &self,
        k: usize,
        dst: &mut [f64],
        w: &[f64],
        f: impl Fn(f64, f64, f64) -> f64,
    ) {
        let c = &self.v;
        let w = &w[..k];
        // Row `j` of the packed rows starts at `start`, `j + 1` long. The
        // rows the fours leave over, one at a time, are the first, the
        // shortest: at 20 features, rows 1 to 3 rather than 17 to 19.
        let (mut j, mut start) = (1, row_at(1));
        let single = 1 + k.saturating_sub(1) % 4;
        while j < single {
            let row = &c[start..start + j];
            let wj = w[j];
            let (mut at, mut step) = (j - 1, k - 2);
            for (&a, &wi) in row.iter().zip(&w[..j]) {
                dst[at] = f(a, wi, wj);
                at += step;
                // Past the row's last pair at `k = j + 1`, never read.
                step = step.wrapping_sub(1);
            }
            start += j + 1;
            j += 1;
        }
        while j + 4 <= k {
            let rows = &c[start..];
            let (r0, rows) = rows.split_at(j + 1);
            let (r1, rows) = rows.split_at(j + 2);
            let (r2, rows) = rows.split_at(j + 3);
            let r3 = &rows[..j + 4];
            let [w0, w1, w2, w3] = *w[j..].first_chunk::<4>().expect("four rows of the block");
            // `(i, j)` at `j − 1` for `i = 0`, then `k − i − 2` on.
            let (mut at, mut step) = (j - 1, k - 2);
            let cols = r0[..j]
                .iter()
                .zip(&r1[..j])
                .zip(&r2[..j])
                .zip(&r3[..j])
                .zip(&w[..j]);
            for ((((&a0, &a1), &a2), &a3), &wi) in cols {
                let run: &mut [f64; 4] = dst[at..]
                    .first_chunk_mut()
                    .expect("four pairs of an output row");
                run[0] = f(a0, wi, w0);
                run[1] = f(a1, wi, w1);
                run[2] = f(a2, wi, w2);
                run[3] = f(a3, wi, w3);
                at += step;
                step -= 1;
            }
            // `at` is now where output row `j` starts, less one.
            let o0 = at + 1;
            let o1 = o0 + (k - j - 1);
            let o2 = o1 + (k - j - 2);
            dst[o0] = f(r1[j], w0, w1);
            dst[o0 + 1] = f(r2[j], w0, w2);
            dst[o0 + 2] = f(r3[j], w0, w3);
            dst[o1] = f(r2[j + 1], w1, w2);
            dst[o1 + 1] = f(r3[j + 1], w1, w3);
            dst[o2] = f(r3[j + 2], w2, w3);
            start += 4 * j + 10;
            j += 4;
        }
        debug_assert!(j >= k, "every row read");
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

    /// `C + ridge·I` is written into the lower triangle and diagonal of a
    /// square of the side, and a square kept at that size has its strict
    /// upper triangle left as it held it, not zeroed again; one of another
    /// size is sized, zeros above the diagonal.
    #[test]
    fn the_ridged_lower_triangle_leaves_a_kept_square_s_upper_triangle() {
        let full = vec![1.0, 9.0, 9.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0];
        let mut m = LowerTriangle::zeros(3);
        m.copy_from(&full);
        let mut square = vec![-1.0; 9];
        m.ridged_lower_into(&mut square, 0.5);
        assert_eq!(square, [1.5, -1.0, -1.0, 2.0, 3.5, -1.0, 4.0, 5.0, 6.5]);
        let mut sized = vec![-1.0; 4];
        m.ridged_lower_into(&mut sized, 0.5);
        assert_eq!(sized, [1.5, 0.0, 0.0, 2.0, 3.5, 0.0, 4.0, 5.0, 6.5]);
    }

    /// Only the matrix of side 0 is empty; a damaged one with numbers is not.
    #[test]
    fn only_the_matrix_of_no_side_is_empty() {
        assert!(LowerTriangle::zeros(0).is_empty());
        assert!(LowerTriangle::default().is_empty());
        assert!(!LowerTriangle::zeros(1).is_empty());
        assert!(!LowerTriangle::zeros(3).is_empty());
        assert!(!LowerTriangle::from(vec![1.0, 2.0]).is_empty());
    }

    /// The strict upper triangle row by row, mapped, at every `k` the
    /// four-column blocks and the columns left over meet: `0` to `9`.
    #[test]
    fn the_upper_rows_are_the_mirrored_pairs_in_order() {
        for k in 0..14usize {
            let full: Vec<f64> = (0..k * k).map(|v| v as f64 + 0.25).collect();
            let mut m = LowerTriangle::zeros(k);
            m.copy_from(&full);
            let want: Vec<f64> = (0..k)
                .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
                .map(|(i, j)| 3.0 * full[j * k + i])
                .collect();
            let mut got = vec![f64::NAN; want.len()];
            let w: Vec<f64> = (0..k).map(|i| i as f64).collect();
            m.upper_rows_map(k, &mut got, &w, |v, wi, wj| {
                assert!(wi < wj && wj < k as f64);
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
