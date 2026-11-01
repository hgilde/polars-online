//! [`LowerTriangle`]: a symmetric matrix kept as its lower triangle.

use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A symmetric `k × k` matrix, row-major, of which only the lower triangle
/// -- the entries `(i, j)` with `j ≤ i` -- is kept current: the co-moments
/// of an [`super::EwCov`], and a window's snapshot of them.
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
/// Every read goes through the lower triangle: [`Self::get`] takes `(i, j)`
/// either way round, [`Self::to_full`] mirrors it into a whole matrix, and
/// [`Self::lower`] hands out the storage, whose strict upper triangle holds
/// no meaning. Equality compares the lower triangles; `Debug` and the state
/// file show the whole matrix, mirrored, so a file still holds `k²` numbers,
/// now symmetric, and a file written by a build that rounded the upper
/// triangle on its own loads as its lower triangle mirrored.
#[derive(Clone, Default)]
pub struct LowerTriangle(Vec<f64>);

impl LowerTriangle {
    /// The `k × k` zero matrix.
    pub(crate) fn zeros(k: usize) -> Self {
        Self(vec![0.0; k * k])
    }

    /// `k` where the storage holds `k²` numbers; `None` for a damaged state
    /// whose length is no square, which every owner refuses by its shape.
    fn side(&self) -> Option<usize> {
        let n = self.0.len();
        let k = n.isqrt();
        (k * k == n).then_some(k)
    }

    /// The numbers stored, `k²`.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether nothing is stored (`k = 0`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entry `(i, j)` of the `k × k` matrix, read from the lower triangle
    /// whichever way round it is asked for.
    #[inline]
    pub fn get(&self, k: usize, i: usize, j: usize) -> f64 {
        self.0[i.max(j) * k + i.min(j)]
    }

    /// The storage, row-major `k × k`: entry `(i, j)` with `j ≤ i` is at
    /// `i·k + j`, and the strict upper triangle is not kept and holds no
    /// meaning. What a factorization that reads the lower triangle takes.
    #[inline]
    pub fn lower(&self) -> &[f64] {
        &self.0
    }

    /// The storage to update in place; only its lower triangle is read.
    #[inline]
    pub(crate) fn lower_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }

    /// The whole matrix, row-major, its upper triangle mirrored from the
    /// lower: for a reader that takes a full symmetric matrix.
    pub fn to_full(&self) -> Vec<f64> {
        let mut v = self.0.clone();
        mirror(&mut v);
        v
    }

    /// [`Self::to_full`] into `out`, whatever it held.
    pub fn full_into(&self, out: &mut Vec<f64>) {
        out.clear();
        out.extend_from_slice(&self.0);
        mirror(out);
    }

    /// The strict upper triangle, row by row -- pair `(i, j)`, `j > i`, in
    /// the order `(0, 1), (0, 2), .., (1, 2), ..` -- each entry `v` written as
    /// `f(i, j, v)` into `dst`, which holds `k(k-1)/2`. Pair `(i, j)` is the lower entry
    /// `(j, i)`, so an output row is a column of the lower triangle; the
    /// columns are read four at a time, so each row of the storage is read
    /// in runs of four and each of the four output rows is written in
    /// order. A column read alone, one entry `k` apart at a time, cost 3.8
    /// instructions a pair more than the contiguous row it replaced.
    pub(crate) fn upper_rows_map(
        &self,
        k: usize,
        mut dst: &mut [f64],
        f: impl Fn(usize, usize, f64) -> f64,
    ) {
        let c = &self.0;
        let mut i = 0;
        while i + 4 <= k {
            let (r0, rest) = std::mem::take(&mut dst).split_at_mut(k - 1 - i);
            let (r1, rest) = rest.split_at_mut(k - 2 - i);
            let (r2, rest) = rest.split_at_mut(k - 3 - i);
            let (r3, rest) = rest.split_at_mut(k - 4 - i);
            dst = rest;
            // The corner, rows `i + 1 ..= i + 3`, where the four columns
            // have not all begun.
            r0[0] = f(i, i + 1, c[(i + 1) * k + i]);
            r0[1] = f(i, i + 2, c[(i + 2) * k + i]);
            r1[0] = f(i + 1, i + 2, c[(i + 2) * k + i + 1]);
            r0[2] = f(i, i + 3, c[(i + 3) * k + i]);
            r1[1] = f(i + 1, i + 3, c[(i + 3) * k + i + 1]);
            r2[0] = f(i + 2, i + 3, c[(i + 3) * k + i + 2]);
            let outs = r0[3..]
                .iter_mut()
                .zip(r1[2..].iter_mut())
                .zip(r2[1..].iter_mut())
                .zip(r3.iter_mut());
            let rows = c[(i + 4) * k..].chunks_exact(k);
            for (j, ((((o0, o1), o2), o3), row)) in (i + 4..).zip(outs.zip(rows)) {
                let q = &row[i..i + 4];
                *o0 = f(i, j, q[0]);
                *o1 = f(i + 1, j, q[1]);
                *o2 = f(i + 2, j, q[2]);
                *o3 = f(i + 3, j, q[3]);
            }
            i += 4;
        }
        while i + 1 < k {
            let (row, rest) = std::mem::take(&mut dst).split_at_mut(k - 1 - i);
            dst = rest;
            let rows = c[(i + 1) * k..].chunks_exact(k);
            for (j, (o, r)) in (i + 1..).zip(row.iter_mut().zip(rows)) {
                *o = f(i, j, r[i]);
            }
            i += 1;
        }
    }

    /// Take a whole row-major matrix of the same size: its lower triangle
    /// is kept, its upper one is not read.
    pub(crate) fn copy_from(&mut self, full: &[f64]) {
        self.0.copy_from_slice(full);
    }

    /// The matrix with one stored number dropped: a damaged state's shape,
    /// for the tests that refuse one.
    #[cfg(test)]
    pub(crate) fn pop(&mut self) -> Option<f64> {
        self.0.pop()
    }
}

/// Copy the lower triangle of the row-major square `v` into its upper one;
/// a length that is no square is left as it is.
fn mirror(v: &mut [f64]) {
    let n = v.len();
    let k = n.isqrt();
    if k * k != n {
        return;
    }
    for i in 1..k {
        for j in 0..i {
            v[j * k + i] = v[i * k + j];
        }
    }
}

impl From<Vec<f64>> for LowerTriangle {
    /// A whole row-major matrix, as its lower triangle.
    fn from(mut v: Vec<f64>) -> Self {
        mirror(&mut v);
        Self(v)
    }
}

impl PartialEq for LowerTriangle {
    /// The lower triangles, entry by entry, as `==` compares doubles; a
    /// storage that is no square is compared whole.
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        let Some(k) = self.side() else {
            return self.0 == other.0;
        };
        (0..k).all(|i| self.0[i * k..=i * k + i] == other.0[i * k..=i * k + i])
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
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        match self.side() {
            Some(k) => {
                for i in 0..k {
                    for j in 0..k {
                        seq.serialize_element(&self.get(k, i, j))?;
                    }
                }
            }
            None => {
                for v in &self.0 {
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
        let mirrored = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0];
        assert_eq!(m.to_full(), mirrored);
        let mut out = vec![7.0];
        m.full_into(&mut out);
        assert_eq!(out, mirrored);
        assert_eq!(format!("{m:?}"), format!("{mirrored:?}"));
        assert_eq!(m, LowerTriangle::from(mirrored.clone()));
        assert_eq!(LowerTriangle::from(full.clone()).lower(), &mirrored[..]);
        let mut other = m.clone();
        other.lower_mut()[3] = 2.5;
        assert_ne!(m, other, "a lower entry is compared");
        other.lower_mut()[3] = 2.0;
        other.lower_mut()[1] = -1.0;
        assert_eq!(m, other, "an upper entry is not");
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
        assert_eq!(back.lower(), &[1.0, 2.0, 2.0, 3.0]);
        // A length that is no square is kept as it is, for its owner's
        // shape check to refuse.
        let odd: LowerTriangle =
            rmp_serde::from_slice(&rmp_serde::to_vec(&vec![1.0, 2.0]).unwrap()).unwrap();
        assert_eq!(odd.lower(), &[1.0, 2.0]);
        assert_eq!(
            rmp_serde::to_vec(&odd).unwrap(),
            rmp_serde::to_vec(&vec![1.0, 2.0]).unwrap()
        );
        assert_ne!(odd, LowerTriangle::from(vec![1.0, 3.0]));
    }
}
