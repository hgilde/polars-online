//! The co-moment matrix kept as its lower triangle (docs/PLAN.md task 294):
//! symmetric as read after many rows, and its lower triangle the
//! whole-matrix update's to the bit, the triangle every build computed
//! alike.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// The update every build before task 294 ran: the whole `k × k` matrix,
/// each entry `(a·b·d_i)·d_j` and its mirror `(a·b·d_j)·d_i` rounded on
/// their own, with the same means, weights and deviations as
/// [`EwCov::update`] (a stream of positive weights, so no row is the
/// decay alone).
struct Whole {
    k: usize,
    w: f64,
    m: Vec<f64>,
    lo: Vec<f64>,
    c: Vec<f64>,
}

impl Whole {
    fn new(k: usize) -> Self {
        Self {
            k,
            w: 0.0,
            m: vec![0.0; k],
            lo: vec![0.0; k],
            c: vec![0.0; k * k],
        }
    }

    fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        let w_new = lam * self.w + w;
        let (a, b) = (lam * self.w / w_new, w / w_new);
        let d: Vec<f64> = (0..self.k)
            .map(|i| crate::comp::dev(x[i], self.m[i], self.lo[i]))
            .collect();
        for (row, &di) in self.c.chunks_exact_mut(self.k).zip(&d) {
            let ab_di = a * b * di;
            for (cij, &dj) in row.iter_mut().zip(&d) {
                *cij = a * *cij + ab_di * dj;
            }
        }
        for ((m, lo), &di) in self.m.iter_mut().zip(self.lo.iter_mut()).zip(&d) {
            crate::comp::add(m, lo, b * di);
        }
        self.w = w_new;
    }
}

/// After 3,000 rows at `k = 1, 2, 20`, with a decay and without, row by row
/// and in blocks: every `cov(i, j)` is `cov(j, i)` to the bit, the whole
/// matrix handed out is symmetric, the state file holds that matrix, and
/// the row-by-row lower triangle is the whole-matrix update's to the bit.
/// At `k = 20` the whole-matrix update's own upper triangle parts from its
/// lower one, which is the asymmetry the lower triangle removes; reading
/// the upper entry would have read it.
#[test]
fn the_matrix_is_symmetric_as_read_and_its_lower_triangle_is_unchanged() {
    for k in [1usize, 2, 20] {
        for lam in [1.0, 0.97] {
            for block in [0usize, 7] {
                let what = format!("k = {k}, lam = {lam}, block {block}");
                let mut s = 17 + k as u64;
                let mut cov = EwCov::new(k);
                cov.set_block_rows(block);
                let mut whole = Whole::new(k);
                for _ in 0..3_000 {
                    let x: Vec<f64> = (0..k).map(|j| 3.0 * j as f64 + lcg(&mut s)).collect();
                    let w = 1.0 + 0.5 * lcg(&mut s);
                    cov.update(&x, lam, w);
                    whole.update(&x, lam, w);
                }
                cov.flush();
                let full = cov.comoments().to_full();
                let mut parted = 0;
                for i in 0..k {
                    for j in 0..k {
                        let got = cov.cov(i, j);
                        assert_eq!(got.to_bits(), cov.cov(j, i).to_bits(), "{what}: ({i}, {j})");
                        assert_eq!(got.to_bits(), full[i * k + j].to_bits(), "{what}: whole");
                        if block == 0 && j <= i {
                            let want = whole.c[i * k + j];
                            assert_eq!(got.to_bits(), want.to_bits(), "{what}: lower ({i}, {j})");
                        }
                        if j < i && whole.c[j * k + i] != whole.c[i * k + j] {
                            parted += 1;
                        }
                    }
                }
                if k == 20 {
                    assert!(parted > 0, "{what}: the whole update was symmetric");
                }
                let bytes = rmp_serde::to_vec_named(&cov).unwrap();
                let back: rmpv::Value = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
                let c = back
                    .as_map()
                    .unwrap()
                    .iter()
                    .find(|(key, _)| key.as_str() == Some("c"))
                    .map(|(_, v)| v.as_array().unwrap())
                    .unwrap();
                let saved: Vec<f64> = c.iter().map(|v| v.as_f64().unwrap()).collect();
                assert_eq!(saved, full, "{what}: the state file");
                let loaded: EwCov = rmp_serde::from_slice(&bytes).unwrap();
                assert_eq!(loaded, cov, "{what}: round trip");
                assert_eq!(loaded.comoments().to_full(), full, "{what}: loaded");
            }
        }
    }
}

/// A state written by a build that updated the whole matrix loads as its
/// lower triangle mirrored: every read, and the state written again, is
/// the matrix the lower triangle makes, and the next row moves the lower
/// triangle as the whole-matrix update moves its own.
#[test]
fn an_old_state_loads_as_its_lower_triangle_mirrored() {
    let k = 20;
    let mut s = 5u64;
    let mut cov = EwCov::new(k);
    let mut whole = Whole::new(k);
    let rows: Vec<Vec<f64>> = (0..3_001)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    for x in &rows[..3_000] {
        cov.update(x, 0.97, 1.0);
        whole.update(x, 0.97, 1.0);
    }
    // The old file: the same state, its matrix the whole update's.
    let mut v = serde_json::to_value(&cov).unwrap();
    v["c"] = serde_json::json!(whole.c);
    let old: EwCov = serde_json::from_value(v).unwrap();
    assert_eq!(old, cov, "the lower triangles agree");
    let mirrored = cov.comoments().to_full();
    assert_ne!(whole.c, mirrored, "the old matrix is not symmetric");
    assert_eq!(old.comoments().to_full(), mirrored);
    for i in 0..k {
        for j in 0..k {
            assert_eq!(old.cov(i, j).to_bits(), mirrored[i * k + j].to_bits());
        }
    }
    assert_eq!(
        rmp_serde::to_vec_named(&old).unwrap(),
        rmp_serde::to_vec_named(&cov).unwrap(),
        "re-saved as the current state"
    );
    let mut next = old.clone();
    next.update(&rows[3_000], 0.97, 1.0);
    whole.update(&rows[3_000], 0.97, 1.0);
    for i in 0..k {
        for j in 0..=i {
            assert_eq!(next.cov(i, j).to_bits(), whole.c[i * k + j].to_bits());
        }
    }
}

/// The whole-matrix update was symmetric to the eye but **not** to the
/// bit: `c[i][j]` was `((a*b)*d_i)*d_j` and `c[j][i]` `((a*b)*d_j)*d_i`,
/// which round differently. E48 proposed mirroring one triangle onto the
/// other after each row and called it bit-identical; it was neither
/// that nor faster (docs/records/PERFORMANCE-LOG.md section 14: 49% to 107% slower,
/// the mirror store walking a new cache line per element; measured again
/// for task 294, +20% at `k = 20`). The matrix is kept as its lower
/// triangle instead ([`LowerTriangle`]): nothing is mirrored per row,
/// and every read of `(i, j)` and `(j, i)` is the one entry, to the bit.
#[test]
fn the_two_triangles_read_as_one() {
    let mut ew = EwCov::new(6);
    let mut s = 11u64;
    for _ in 0..200 {
        let x: Vec<f64> = (0..6)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                100.0 + ((s >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect();
        ew.update(&x, 0.99, 1.0);
    }
    for i in 0..6 {
        for j in (i + 1)..6 {
            assert_eq!(ew.cov(i, j).to_bits(), ew.cov(j, i).to_bits(), "({i}, {j})");
            assert_eq!(
                ew.raw(i, j).to_bits(),
                ew.raw(j, i).to_bits(),
                "raw ({i}, {j})"
            );
        }
    }
}

/// The serial step's two forms (task 371): a side fixed when compiled at 3
/// to 12 features, a side read at run time elsewhere. At every width from 0
/// to 14, over one stream, `EwCov::update`, the dispatch and the run-time
/// rows run directly leave the same packed triangle to the bit, equal
/// matrices and the same state bytes.
#[test]
fn fixed_and_run_time_rows_leave_the_same_state() {
    for k in 0..=14usize {
        let mut s = 29 + k as u64;
        let mut cov = EwCov::new(k);
        let n = k * (k + 1) / 2;
        let (mut fixed, mut lower) = (vec![0.0; n], vec![0.0; n]);
        for _ in 0..2_000 {
            let x: Vec<f64> = (0..k).map(|j| j as f64 + lcg(&mut s)).collect();
            let w = 1.0 + 0.5 * lcg(&mut s);
            let lam = 0.98;
            // The same `a`, `b` and deviations `update` forms.
            let w_new = lam * cov.n_eff() + w;
            let (a, b) = (lam * cov.n_eff() / w_new, w / w_new);
            let d: Vec<f64> = (0..k).map(|i| cov.deviation(i, x[i])).collect();
            rank_one_rows(&mut fixed, k, &d, a, b);
            rank_one_lower_rows(&mut lower, k, &d, a, b);
            cov.update(&x, lam, w);
        }
        let bits = |c: &[f64]| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&fixed), bits(&lower), "k = {k}");
        assert_eq!(
            bits(cov.comoments().packed()),
            bits(&lower),
            "k = {k}: update"
        );
        if k > 0 {
            assert!(
                lower.iter().all(|&v| v != 0.0),
                "k = {k}: every entry moved"
            );
        }
        let mut twin = cov.clone();
        twin.c.copy_from_packed(&lower);
        assert_eq!(twin, cov, "k = {k}");
        assert_eq!(
            rmp_serde::to_vec_named(&twin).unwrap(),
            rmp_serde::to_vec_named(&cov).unwrap(),
            "k = {k}: the state bytes"
        );
    }
}

/// The run-time rows at no feature and at one.
#[test]
fn run_time_rows_at_no_feature_and_one() {
    rank_one_lower_rows(&mut [], 0, &[], 0.5, 0.5);
    let mut lower = [0.75f64];
    rank_one_lower_rows(&mut lower, 1, &[0.3], 0.9, 0.1);
    assert_eq!(lower[0], 0.9 * 0.75 + 0.9 * 0.1 * 0.3 * 0.3);
}
