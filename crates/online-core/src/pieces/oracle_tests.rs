//! The Gram's product against an exactly rounded oracle (docs/PLAN.md task
//! 258): [`syrk_lower_cut`]'s every entry of `C + α·DᵀD` held to
//! [`oracle::dot3`]'s, which rounds once and shares no arithmetic with
//! faer's kernels.
//!
//! The bound is the classical one for a sum evaluated in any order
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! §3.1): an entry the kernel forms as `C + α·Σᵣ dᵣᵢdᵣⱼ` takes, along any
//! one term's path, one rounding for `dᵣᵢdᵣⱼ` (none where it is fused into
//! the add), at most `n − 1` within its panel's sum, one for the scale `α`
//! and one to add the panel into `C`, with at most `n` panels on the path
//! (one per panel of the depth that a blocked kernel adds in turn), so at
//! most `2n + 2` roundings. With `γₘ = m·u/(1 − m·u)` and `u = 2⁻⁵³`,
//!
//! `|ĉᵢⱼ − cᵢⱼ| ≤ γ₂ₙ₊₂ · (|Cᵢⱼ| + |α|·Σᵣ |dᵣᵢdᵣⱼ|)`,
//!
//! and the oracle is `cᵢⱼ` within half an ulp, `u·|cᵢⱼ|`. A wrong row, a
//! tile at the wrong offset or a lost `α` moves an entry by about one term,
//! `Σ/n`, which is `1/(2n²·u)` of the bound: past 10⁹ of it at `n = 1,000`.

use super::*;
use crate::oracle;

const U: f64 = f64::EPSILON / 2.0;

fn gamma(m: usize) -> f64 {
    let mu = m as f64 * U;
    mu / (1.0 - mu)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `C` (`k × k`, row-major, symmetric) and `D` (`n × k`, row-major), each
/// entry of column `j` scaled by `scale[j]` (`C`'s `(i, j)` by both), and
/// every row `r` with `zero(r)` set to zeros: a row of zero weight.
struct Gram {
    k: usize,
    n: usize,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl Gram {
    fn new(k: usize, n: usize, seed: u64, scale: &dyn Fn(usize) -> f64) -> Self {
        let mut s = seed;
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                let v = lcg(&mut s) * scale(i) * scale(j);
                c[i * k + j] = v;
                c[j * k + i] = v;
            }
        }
        let d = (0..n * k)
            .map(|rj| {
                let v = lcg(&mut s) * scale(rj % k);
                if (rj / k) % 5 == 3 { 0.0 } else { v }
            })
            .collect();
        Self { k, n, c, d }
    }

    /// [`syrk_lower_cut`] on a copy of `C`, inside a pool of `threads`.
    fn cut(&self, alpha: f64, threads: usize, cut: (usize, usize)) -> Vec<f64> {
        let mut c = self.c.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a test pool");
        pool.install(|| {
            syrk_lower_cut(
                MatMut::from_row_major_slice_mut(&mut c, self.k, self.k),
                MatRef::from_row_major_slice(&self.d, self.n, self.k),
                alpha,
                threads,
                cut,
            );
        });
        c
    }

    /// Each lower entry's exactly rounded value, and its bound.
    fn oracle(&self, alpha: f64) -> Vec<(f64, f64)> {
        let (k, n) = (self.k, self.n);
        let gamma = gamma(2 * n + 2);
        let mut out = Vec::with_capacity(k * (k + 1) / 2);
        for i in 0..k {
            for j in 0..=i {
                let c = self.c[i * k + j];
                let rows = (0..n).map(|r| (alpha, self.d[r * k + i], self.d[r * k + j]));
                let want = oracle::dot3(std::iter::once((1.0, c, 1.0)).chain(rows));
                let size: f64 = c.abs()
                    + alpha.abs()
                        * (0..n)
                            .map(|r| (self.d[r * k + i] * self.d[r * k + j]).abs())
                            .sum::<f64>();
                // `size` is a sum of terms of one sign, within `γₙ` of its
                // own value; `1 + 1e-9` covers that and the bound's `γ`.
                out.push((want, (gamma * size + U * want.abs()) * (1.0 + 1e-9)));
            }
        }
        out
    }

    /// Every lower entry of `got` within its bound of the oracle's, and
    /// every upper one `C`'s own bits: the product leaves it alone.
    fn check(&self, got: &[f64], oracle: &[(f64, f64)], what: &str) {
        let k = self.k;
        let mut lower = oracle.iter();
        for i in 0..k {
            for j in 0..k {
                let g = got[i * k + j];
                if j > i {
                    assert_eq!(
                        g.to_bits(),
                        self.c[i * k + j].to_bits(),
                        "{what}: ({i}, {j}) is above the diagonal and moved"
                    );
                    continue;
                }
                let &(want, bound) = lower.next().expect("an oracle entry");
                assert!(
                    (g - want).abs() <= bound,
                    "{what}: ({i}, {j}) = {g:e}, the oracle {want:e}, off by {:e} past {bound:e}",
                    (g - want).abs()
                );
            }
        }
    }
}

/// A cut fine enough that a small matrix is many pieces of both kinds.
const FINE: (usize, usize) = (16, 8);

fn unit(_: usize) -> f64 {
    1.0
}

/// Every width up to 20, past each register block a kernel might have on
/// either architecture (2 to 16 rows by 4 to 8 columns), and depths from
/// none to past 16: no rows (the product adds nothing), one row (faer's
/// rank-one update), and the small kernel below 16³ entries and the blocked
/// one above it. Both cuts, one thread and four, and three scales: `α = −1`
/// as the Cholesky's trailing update has it, `1`, and one with a rounding.
#[test]
fn every_small_shape_is_the_oracles_within_its_bound() {
    for k in 1..=20 {
        for n in [0usize, 1, 2, 3, 5, 9, 17] {
            let g = Gram::new(k, n, 3 + (k * 31 + n) as u64, &unit);
            for alpha in [-1.0, 1.0, 1.0 / 3.0] {
                let oracle = g.oracle(alpha);
                for (threads, cut) in [(1, (LEAF, TILE)), (4, FINE)] {
                    let what = format!("k = {k}, n = {n}, alpha = {alpha}, {threads} threads");
                    g.check(&g.cut(alpha, threads, cut), &oracle, &what);
                }
            }
        }
    }
}

/// The widths either side of each cut -- a fine cut's leaf (16) and tile
/// (8, so tiles of 8 to 15), the shipped tile (128) -- and depths either
/// side of the kernels' 64-row blocks, on one, two, four and eight threads.
#[test]
fn the_cuts_edges_are_the_oracles_within_its_bound() {
    for k in [31usize, 32, 33, 63, 64, 65, 129] {
        for n in [7usize, 64, 65] {
            let g = Gram::new(k, n, 5 + (k * 31 + n) as u64, &unit);
            let oracle = g.oracle(-1.0);
            for threads in [1usize, 2, 4, 8] {
                for cut in [FINE, (LEAF, TILE)] {
                    let what = format!("k = {k}, n = {n}, {threads} threads, cut {cut:?}");
                    g.check(&g.cut(-1.0, threads, cut), &oracle, &what);
                }
            }
        }
    }
}

/// Magnitudes from 10⁻¹²⁰ to 10¹²⁰ in one matrix, a column's scale drawn
/// log-uniform, so its entries span 10²⁴⁰ and each is held to its own
/// size; and a matrix all near 10¹⁴⁰, one all near 10⁻¹⁴⁰ (products near
/// 10±²⁸⁰, inside the range where the oracle's splits are exact).
#[test]
fn tiny_and_huge_entries_are_the_oracles_within_their_bounds() {
    let mixed = |j: usize| {
        let mut s = 99 + j as u64;
        10f64.powf(120.0 * lcg(&mut s))
    };
    let huge = |_: usize| 1e140;
    let tiny = |_: usize| 1e-140;
    let scales: [(&str, &dyn Fn(usize) -> f64); 3] =
        [("mixed", &mixed), ("huge", &huge), ("tiny", &tiny)];
    for (name, scale) in scales {
        for (k, n) in [(40usize, 50usize), (130, 9)] {
            let g = Gram::new(k, n, 17 + k as u64, scale);
            for alpha in [1.0, -0.75] {
                let oracle = g.oracle(alpha);
                for (threads, cut) in [(1, (LEAF, TILE)), (4, FINE)] {
                    let what =
                        format!("{name}: k = {k}, n = {n}, alpha = {alpha}, {threads} threads");
                    g.check(&g.cut(alpha, threads, cut), &oracle, &what);
                }
            }
        }
    }
}

/// The shipped cut's own edges (a leaf of 256, tiles of 128 to 255, the
/// rectangle cut in three past 384) at the depths a bank's blocks have:
/// `gram_block_rows` either side of 256, a partial last block, and depths
/// either side of the kernels' 512-row panel, on one to eight threads.
#[test]
#[ignore = "extended: an exact oracle over 1.5 · 10⁸ products in a debug build"]
fn the_shipped_cuts_edges_are_the_oracles_within_its_bound() {
    for (k, n) in [(255usize, 513usize), (256, 256), (257, 1025), (385, 255)] {
        let g = Gram::new(k, n, 7 + (k * 31 + n) as u64, &unit);
        let oracle = g.oracle(-1.0);
        for threads in [1usize, 2, 4, 8] {
            let what = format!("k = {k}, n = {n}, {threads} threads");
            g.check(&g.cut(-1.0, threads, (LEAF, TILE)), &oracle, &what);
        }
    }
}
