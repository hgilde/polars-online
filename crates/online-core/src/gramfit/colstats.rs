//! The ridge fits' two quadratic forms per target, for every target at
//! once, each target's sums in the order a target alone sums them
//! (docs/PLAN.md task 256).
//!
//! A target's `b' C b` and `m' C⁻¹ m` are sums of `k²` and `k²/2` terms,
//! each summed in index order: a single chain of dependent additions, which
//! runs at one addition per add latency. Tiles of rows and targets carried
//! side by side in registers ([`super::trisolve::tiles`]) run many such
//! chains at once and share each load of the matrix between the targets,
//! without moving one addition: every sum here is the left fold from `-0.0`
//! that `Iterator::sum` is, over the same terms in the same order, so a
//! target's bits are those it has alone, and those of the loop over
//! targets this replaces.

use faer::Mat;

use super::trisolve::tiles;

/// `b_c' C b_c` for each column `c` of `B`, row-major `n × t`, `C`
/// row-major `n × n`: `Σ_p b_p · (Σ_q C_pq · b_q)`, both sums in index order.
pub(super) fn sym_quad(c: &[f64], n: usize, b: &[f64], t: usize) -> Vec<f64> {
    assert_eq!(c.len(), n * n, "an {n} x {n} matrix");
    assert_eq!(b.len(), n * t, "an {n} x {t} right-hand side");
    let mut out = vec![-0.0; t];
    tiles(t, n, |rb, r, c0, w| {
        macro_rules! shape {
            ($($rb:literal x $w:literal),*) => {
                match (rb, w) {
                    $(($rb, $w) => sym_tile::<$rb, $w>(c, n, b, t, r, c0, &mut out),)*
                    _ => unreachable!("a {rb} x {w} tile"),
                }
            };
        }
        shape!(4 x 8, 4 x 4, 1 x 8, 1 x 4, 8 x 4, 16 x 1, 4 x 1, 1 x 1)
    });
    out
}

fn sym_tile<const R: usize, const W: usize>(
    c: &[f64],
    n: usize,
    b: &[f64],
    t: usize,
    r: usize,
    c0: usize,
    out: &mut [f64],
) {
    let rows: [&[f64]; R] = std::array::from_fn(|q| &c[(r + q) * n..][..n]);
    let mut acc = [[-0.0; W]; R];
    for (j, brow) in b.chunks_exact(t).enumerate() {
        let lj: [f64; R] = std::array::from_fn(|q| rows[q][j]);
        let xj: &[f64; W] = brow[c0..c0 + W].try_into().unwrap();
        for (a, &l) in acc.iter_mut().zip(&lj) {
            for (s, &v) in a.iter_mut().zip(xj) {
                *s += l * v;
            }
        }
    }
    for (q, a) in acc.iter().enumerate() {
        let bp = &b[(r + q) * t + c0..][..W];
        for ((o, &bv), &v) in out[c0..c0 + W].iter_mut().zip(bp).zip(a) {
            *o += bv * v;
        }
    }
}

/// `‖L⁻¹ m_c‖²` for each column `c` of `M`, row-major `n × t`, from `L⁻¹`,
/// lower triangular: `Σ_j (Σ_{i ≤ j} L⁻¹_ji · m_i)²`, both sums in index
/// order.
pub(super) fn lower_quad(inv: &Mat<f64>, m: &[f64], t: usize) -> Vec<f64> {
    let n = inv.nrows();
    assert_eq!(m.len(), n * t, "an {n} x {t} right-hand side");
    let mut out = vec![-0.0; t];
    tiles(t, n, |rb, r, c0, w| {
        macro_rules! shape {
            ($($rb:literal x $w:literal),*) => {
                match (rb, w) {
                    $(($rb, $w) => lower_tile::<$rb, $w>(inv, m, t, r, c0, &mut out),)*
                    _ => unreachable!("a {rb} x {w} tile"),
                }
            };
        }
        shape!(4 x 8, 4 x 4, 1 x 8, 1 x 4, 8 x 4, 16 x 1, 4 x 1, 1 x 1)
    });
    out
}

fn lower_tile<const R: usize, const W: usize>(
    inv: &Mat<f64>,
    m: &[f64],
    t: usize,
    r: usize,
    c0: usize,
    out: &mut [f64],
) {
    let mut acc = [[-0.0; W]; R];
    for (i, mrow) in m[..r * t].chunks_exact(t).enumerate() {
        let lj: &[f64; R] = inv.col_as_slice(i)[r..r + R].try_into().unwrap();
        let xj: &[f64; W] = mrow[c0..c0 + W].try_into().unwrap();
        for (a, &l) in acc.iter_mut().zip(lj) {
            for (s, &v) in a.iter_mut().zip(xj) {
                *s += l * v;
            }
        }
    }
    // The tile's own triangle: row `r + q` sums on to `i = r + q`.
    for (q, a) in acc.iter_mut().enumerate() {
        for i in r..=r + q {
            let l = inv[(r + q, i)];
            let xi = &m[i * t + c0..][..W];
            for (s, &v) in a.iter_mut().zip(xi) {
                *s += l * v;
            }
        }
    }
    for a in &acc {
        for (o, &v) in out[c0..c0 + W].iter_mut().zip(a) {
            *o += v.powi(2);
        }
    }
}
