//! `X ← A⁻¹ X` from `A`'s lower Cholesky factor `L`, for several right-hand
//! sides at once, each column's arithmetic its own (docs/PLAN.md task 256).
//!
//! Each column is solved by the textbook recurrences, forward then back,
//!
//! ```text
//! y_i = (((b_i − L_i0 y_0) − L_i1 y_1) − … − L_i,i−1 y_i−1) / L_ii      j ascending
//! x_i = (((y_i − L_n−1,i x_n−1) − L_n−2,i x_n−2) − … − L_i+1,i x_i+1) / L_ii   j descending
//! ```
//!
//! one rounded product and one rounded difference per term, in that order,
//! and nothing else: no fused multiply-add (Rust never contracts `a − l·x`
//! into one), no split of a sum into partial sums. The blocking only
//! decides which of these operations run side by side. The rows go in
//! panels of [`PANEL`]: first every term from outside the panel, a tile of
//! rows and columns at a time carried in registers through the `j` loop,
//! so the tile's columns share each load of `L` and its rows each load of
//! `X`; then the panel's own triangle, row by row. A column's operations,
//! and so its bits, are the same whether it is solved alone or beside any
//! other columns, in any order, and whatever the tile shapes, which may
//! therefore differ by platform. faer's blocked solve, which this replaces
//! in the ridge fits, sums a column's terms in a split that depends on the
//! number of columns beside it, so a target's last bits moved with its
//! neighbours.
//!
//! `L` is read where it lies, column-major: the forward tiles read a few
//! rows of one column per `j`, the back tiles one entry of each of their
//! rows' columns per `j`, each a contiguous stream. Sequential, as the
//! fits around it are: the caller's pool runs whole fits side by side.

use faer::Mat;

/// Rows of a panel: the tallest tile.
const PANEL: usize = 16;

/// `X ← (L Lᵀ)⁻¹ X` for `X` row-major `n × t`, `n` the order of `L`, whose
/// strictly upper triangle is not read.
pub(crate) fn solve_llt(l: &Mat<f64>, x: &mut [f64], t: usize) {
    forward(l, x, t);
    backward(l, x, t);
}

/// `X ← L⁻¹ X`, panels top to bottom.
pub(crate) fn forward(l: &Mat<f64>, x: &mut [f64], t: usize) {
    let n = l.nrows();
    assert_eq!(x.len(), n * t, "an {n} x {t} right-hand side");
    if t == 0 {
        return;
    }
    let mut i0 = 0;
    while i0 < n {
        let p = PANEL.min(n - i0);
        let (done, rest) = x.split_at_mut(i0 * t);
        let panel = &mut rest[..p * t];
        tiles(t, p, |rb, r, c0, w| {
            rect(Pass::Forward, l, done, panel, t, i0, r, rb, c0, w)
        });
        for q in 0..p {
            let (above, row) = panel.split_at_mut(q * t);
            let row = &mut row[..t];
            for (pq, xp) in above.chunks_exact(t).enumerate() {
                axpy(row, l[(i0 + q, i0 + pq)], xp);
            }
            divide(row, l[(i0 + q, i0 + q)]);
        }
        i0 += p;
    }
}

/// `X ← L⁻ᵀ X`, panels bottom to top.
pub(crate) fn backward(l: &Mat<f64>, x: &mut [f64], t: usize) {
    let n = l.nrows();
    assert_eq!(x.len(), n * t, "an {n} x {t} right-hand side");
    if t == 0 {
        return;
    }
    let mut end = n;
    while end > 0 {
        let p = PANEL.min(end);
        let i0 = end - p;
        let (head, done) = x.split_at_mut(end * t);
        let panel = &mut head[i0 * t..];
        tiles(t, p, |rb, r, c0, w| {
            rect(Pass::Backward, l, done, panel, t, i0, r, rb, c0, w)
        });
        for q in (0..p).rev() {
            let (row, below) = panel.split_at_mut((q + 1) * t);
            let row = &mut row[q * t..];
            for (pq, xp) in below.chunks_exact(t).enumerate().rev() {
                axpy(row, l[(i0 + q + 1 + pq, i0 + q)], xp);
            }
            divide(row, l[(i0 + q, i0 + q)]);
        }
        end = i0;
    }
}

/// `row ← row − l·x`, term by term.
fn axpy(row: &mut [f64], l: f64, x: &[f64]) {
    for (s, &v) in row.iter_mut().zip(x) {
        *s -= l * v;
    }
}

fn divide(row: &mut [f64], d: f64) {
    for s in row.iter_mut() {
        *s /= d;
    }
}

#[derive(Clone, Copy)]
enum Pass {
    Forward,
    Backward,
}

/// The tiles of a panel of `p` rows and `t` columns, as `(rows, first row,
/// first column, columns)`, each tile's rows after the rows above it in its
/// columns: wide tiles of a few rows where the columns
/// allow, the rest a column at a time down the whole panel. NEON has 32
/// vector registers of two lanes and SSE2 16: a 4 × 8 or an 8 × 4 tile of
/// accumulators fills 16 of them, 4 × 4 eight.
pub(super) fn tiles(t: usize, p: usize, mut tile: impl FnMut(usize, usize, usize, usize)) {
    #[cfg(target_arch = "aarch64")]
    const SHAPES: [(usize, usize); 3] = [(8, 4), (4, 8), (1, PANEL)];
    #[cfg(not(target_arch = "aarch64"))]
    const SHAPES: [(usize, usize); 2] = [(4, 4), (1, PANEL)];
    let mut c0 = 0;
    for (w, rb) in SHAPES {
        while c0 + w <= t {
            let mut r = 0;
            for rb in [rb, 4, 1] {
                while r + rb <= p {
                    tile(rb, r, c0, w);
                    r += rb;
                }
            }
            c0 += w;
        }
    }
}

/// Rows `r..r + rb` of a panel at `i0`, columns `c0..c0 + w`: every term
/// from outside the panel, `done` the rows solved before it.
#[allow(clippy::too_many_arguments)]
fn rect(
    pass: Pass,
    l: &Mat<f64>,
    done: &[f64],
    panel: &mut [f64],
    t: usize,
    i0: usize,
    r: usize,
    rb: usize,
    c0: usize,
    w: usize,
) {
    macro_rules! shape {
        ($($rb:literal x $w:literal),*) => {
            match (rb, w) {
                $(($rb, $w) => match pass {
                    Pass::Forward => fwd::<$rb, $w>(l, done, panel, t, i0, r, c0),
                    Pass::Backward => bwd::<$rb, $w>(l, done, panel, t, i0, r, c0),
                },)*
                _ => unreachable!("a {rb} x {w} tile"),
            }
        };
    }
    shape!(4 x 8, 4 x 4, 1 x 8, 1 x 4, 8 x 4, 16 x 1, 4 x 1, 1 x 1)
}

fn load<const R: usize, const W: usize>(
    panel: &[f64],
    t: usize,
    r: usize,
    c0: usize,
) -> [[f64; W]; R] {
    let mut acc = [[0.0; W]; R];
    for (q, a) in acc.iter_mut().enumerate() {
        a.copy_from_slice(&panel[(r + q) * t + c0..][..W]);
    }
    acc
}

fn store<const R: usize, const W: usize>(
    acc: &[[f64; W]; R],
    panel: &mut [f64],
    t: usize,
    r: usize,
    c0: usize,
) {
    for (q, a) in acc.iter().enumerate() {
        panel[(r + q) * t + c0..][..W].copy_from_slice(a);
    }
}

/// `acc[q][c] ← acc[q][c] − lj[q]·xj[c]`, each in its own lane.
#[inline(always)]
fn step<const R: usize, const W: usize>(acc: &mut [[f64; W]; R], lj: &[f64; R], xj: &[f64; W]) {
    for (a, &lq) in acc.iter_mut().zip(lj) {
        for (s, &v) in a.iter_mut().zip(xj) {
            *s -= lq * v;
        }
    }
}

/// The forward tile: `j` ascending over the rows above the panel, `L[i, j]`
/// for the tile's rows a contiguous piece of column `j`.
fn fwd<const R: usize, const W: usize>(
    l: &Mat<f64>,
    done: &[f64],
    panel: &mut [f64],
    t: usize,
    i0: usize,
    r: usize,
    c0: usize,
) {
    let mut acc = load::<R, W>(panel, t, r, c0);
    let rows = i0 + r;
    for (j, xrow) in done.chunks_exact(t).enumerate() {
        let lj: &[f64; R] = l.col_as_slice(j)[rows..rows + R].try_into().unwrap();
        let xj: &[f64; W] = xrow[c0..c0 + W].try_into().unwrap();
        step(&mut acc, lj, xj);
    }
    store(&acc, panel, t, r, c0);
}

/// The back tile: `j` descending over the rows below the panel, `L[j, i]`
/// for the tile's rows one entry of each row's column.
fn bwd<const R: usize, const W: usize>(
    l: &Mat<f64>,
    done: &[f64],
    panel: &mut [f64],
    t: usize,
    i0: usize,
    r: usize,
    c0: usize,
) {
    let mut acc = load::<R, W>(panel, t, r, c0);
    let below = done.len() / t;
    let from = l.nrows() - below;
    let cols: [&[f64]; R] =
        std::array::from_fn(|q| &l.col_as_slice(i0 + r + q)[from..from + below]);
    for (jj, xrow) in done.chunks_exact(t).enumerate().rev() {
        let lj: [f64; R] = std::array::from_fn(|q| cols[q][jj]);
        let xj: &[f64; W] = xrow[c0..c0 + W].try_into().unwrap();
        step(&mut acc, &lj, xj);
    }
    store(&acc, panel, t, r, c0);
}
