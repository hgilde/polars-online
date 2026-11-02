//! [`HeldSteps`]: the co-moments' rank-one steps of a run of rows, held and
//! taken together in register blocks, to the bit of taking them row by row.

use super::{CoStep, Threads};

/// Rows a [`HeldSteps`] holds before it takes their steps.
const HELD_ROWS: usize = 64;

/// Whether a tile spans two column blocks, 4 × 8, where two fit: 32 sums,
/// sixteen of aarch64's 32 vector registers, enough independent sums to
/// keep its pipes busy where a 4 × 4 tile's 16 waited on their own
/// products. x86-64's baseline has 16 vector registers of two lanes, which
/// 32 sums would fill before the row's operands, so there the tiles stay
/// 4 × 4.
#[cfg(not(target_arch = "x86_64"))]
const WIDE: bool = true;
#[cfg(target_arch = "x86_64")]
const WIDE: bool = false;

/// The co-moments' rank-one steps of the rows an owner replays, held and
/// taken [`HELD_ROWS`] at a time ([`super::EwCov::update_held`]).
///
/// Row `t`'s step is `c_ij = a_t·c_ij + (a_t·b_t·d_ti)·d_tj` on every lower
/// entry, `j ≤ i`. Taken as the row arrives, it loads and stores every
/// entry of the triangle for every row: at 21 coefficients, 231 entries a
/// row. Held, each row keeps its `a_t`, its deviations `d_t` and the
/// products `(a_t·b_t)·d_ti`, and the triangle is walked in tiles of four
/// of its rows (fewer in the last) by four or eight columns, on or below
/// the diagonal, each tile's entries kept in registers while it runs down
/// the held rows, and stored once. Every entry takes each row's step in
/// the rows' order, with the same expression -- one product
/// for `a_t·c_ij`, one for the update, one sum, no fused multiply-add --
/// so each entry is the number the row-by-row steps give, to the bit; only
/// the order of the work across entries differs. The steps are written on
/// a copy of the triangle padded to a multiple of 4 columns, whose padding
/// takes zeros and stays zero, and the lower triangle is copied back by
/// [`Self::merge_into`]. Not state: an owner keeps one as working space.
///
/// Measured on an M-series core, one pass of 2,500 rows timed alone: at 21
/// coefficients 129 µs row by row, 106 in 4 × 4 tiles over every padded
/// row, 81 as here (the last block's padding rows left out, 4 × 8 tiles);
/// 50% less at 16 and 20, 28% at 5. A `huber` refit of 2,500 rows at 20
/// features ran 13-17% faster, a quantile fit's 4-5%.
#[derive(Clone, Default)]
pub(crate) struct HeldSteps {
    /// The features the buffers are shaped for, and that padded to a
    /// multiple of 4.
    k: usize,
    ks: usize,
    /// Rows held, whose steps are not yet in `c`.
    rows: usize,
    /// Whether `c` holds the co-moments: from the first step held to the
    /// merge.
    open: bool,
    /// One allocation, in four parts ([`Self::parts`]): the co-moments,
    /// padded `ks × ks`, row-major, while open; per held row, `a_t`; per
    /// held row, `d_t`, padded to `ks` with zeros; and per held row,
    /// `(a_t·b_t)·d_ti`, padded the same way.
    buf: Vec<f64>,
}

impl HeldSteps {
    /// Copy the lower triangle of `c`, `k × k`, into the padded copy, every
    /// other entry 0, shaping the buffers for `k` first.
    fn open(&mut self, c: &[f64], k: usize) {
        if self.k != k || self.buf.is_empty() {
            let ks = k.div_ceil(4) * 4;
            self.k = k;
            self.ks = ks;
            self.buf = vec![0.0; ks * ks + HELD_ROWS * (1 + 2 * ks)];
        }
        let ks = self.ks;
        let (cp, ..) = self.parts();
        cp.fill(0.0);
        for i in 0..k {
            cp[i * ks..=i * ks + i].copy_from_slice(&c[i * k..=i * k + i]);
        }
        self.rows = 0;
        self.open = true;
    }

    /// The buffer's four parts: the padded co-moments, and per held row
    /// `a_t`, `d_t` and `(a_t·b_t)·d_t`.
    #[inline(always)]
    fn parts(&mut self) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        let ks = self.ks;
        let (c, rest) = self.buf.split_at_mut(ks * ks);
        let (a, rest) = rest.split_at_mut(HELD_ROWS);
        let (d, abd) = rest.split_at_mut(HELD_ROWS * ks);
        (c, a, d, abd)
    }

    /// Take the held rows' steps on the padded copy, row block by row
    /// block: four rows of the triangle at a time, and the last block's
    /// rows past `k` left out; across a block, two column blocks at a time
    /// ([`WIDE`]) while two fit at or below the diagonal block, then one.
    fn take(&mut self) {
        let k = self.k;
        for pb in 0..self.ks / 4 {
            match k - pb * 4 {
                1 => self.row_block::<1>(pb),
                2 => self.row_block::<2>(pb),
                3 => self.row_block::<3>(pb),
                _ => self.row_block::<4>(pb),
            }
        }
        self.rows = 0;
    }

    /// Row block `pb`'s tiles, its `R` rows each across the columns up to
    /// and including the diagonal block.
    #[inline(always)]
    fn row_block<const R: usize>(&mut self, pb: usize) {
        let mut qb = 0;
        while qb <= pb {
            if WIDE && qb < pb {
                self.tile::<R, 8>(pb, qb);
                qb += 2;
            } else {
                self.tile::<R, 4>(pb, qb);
                qb += 1;
            }
        }
    }

    /// The `R × C` tile at rows `4·pb..`, columns `4·qb..`, its entries in
    /// registers down the held rows in their order.
    #[inline(always)]
    fn tile<const R: usize, const C: usize>(&mut self, pb: usize, qb: usize) {
        let (ks, n) = (self.ks, self.rows);
        let (p0, q0) = (pb * 4, qb * 4);
        let (c, a, d, abd) = self.parts();
        let mut acc = [[0.0_f64; C]; R];
        for (r, acc_r) in acc.iter_mut().enumerate() {
            acc_r.copy_from_slice(&c[(p0 + r) * ks + q0..][..C]);
        }
        for ((&at, abd), d) in a[..n]
            .iter()
            .zip(abd.chunks_exact(ks))
            .zip(d.chunks_exact(ks))
        {
            let abd = &abd[p0..p0 + R];
            let d = &d[q0..q0 + C];
            for (acc_r, &e) in acc.iter_mut().zip(abd) {
                for (s, &dl) in acc_r.iter_mut().zip(d) {
                    *s = at * *s + e * dl;
                }
            }
        }
        for (r, acc_r) in acc.iter().enumerate() {
            c[(p0 + r) * ks + q0..][..C].copy_from_slice(acc_r);
        }
    }

    /// Take every held step and write the lower triangle back into `c`,
    /// `k × k`; nothing when no step is held.
    pub(crate) fn merge_into(&mut self, c: &mut [f64], k: usize) {
        if !self.open {
            return;
        }
        self.take();
        let ks = self.ks;
        let (cp, ..) = self.parts();
        for i in 0..k {
            c[i * k..=i * k + i].copy_from_slice(&cp[i * ks..=i * ks + i]);
        }
        self.open = false;
    }
}

impl CoStep for HeldSteps {
    #[inline]
    fn rank_one(&mut self, c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64, _: Threads) {
        if !self.open {
            self.open(c, k);
        }
        let (r, ks) = (self.rows, self.ks);
        let (_, at, dt, abd) = self.parts();
        at[r] = a;
        let ab = a * b;
        let row = r * ks..r * ks + k;
        for ((dt, e), &di) in dt[row.clone()].iter_mut().zip(&mut abd[row]).zip(d) {
            *dt = di;
            *e = ab * di;
        }
        self.rows += 1;
        if self.rows == HELD_ROWS {
            self.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::EwCov;
    use super::*;

    /// A stream of `n` rows at `k` features with weights drawn among 0,
    /// tiny and ordinary ones, and decays among 1, 0.97 and 0.5.
    fn stream(k: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64, f64)> {
        let mut st = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        (0..n)
            .map(|_| {
                let x: Vec<f64> = (0..k).map(|j| 3.0 * next() - 1.0 + j as f64).collect();
                let w = match (next() * 6.0) as u32 {
                    0 => 0.0,
                    1 => 1e-9,
                    _ => 0.5 + next(),
                };
                let lam = match (next() * 3.0) as u32 {
                    0 => 1.0,
                    1 => 0.97,
                    _ => 0.5,
                };
                (x, lam, w)
            })
            .collect()
    }

    /// Bits of the lower triangle, the means and the weights.
    fn bits(c: &EwCov) -> Vec<u64> {
        let k = c.k();
        let mut out = vec![c.n_eff().to_bits()];
        for i in 0..k {
            out.push(c.mean(i).to_bits());
            for j in 0..=i {
                out.push(c.comoments().get(k, i, j).to_bits());
            }
        }
        out
    }

    #[test]
    fn held_steps_are_the_row_by_row_steps_to_the_bit() {
        // Widths on and off a multiple of 4, runs shorter and longer than
        // one hold, zero-weight rows (a decay alone) and zero-weight heads.
        let mut checked = 0;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 21, 22, 27,
        ] {
            for n in [1, 5, 63, 64, 65, 200] {
                for seed in 0..3 {
                    let rows = stream(k, n, seed * 1000 + k as u64 * 7 + n as u64);
                    let mut each = EwCov::new(k).without_runs();
                    let mut held_cov = EwCov::new(k).without_runs();
                    let mut held = HeldSteps::default();
                    for (t, (x, lam, w)) in rows.iter().enumerate() {
                        // A pure decay now and then, as a replay's ageing
                        // runs and zero Huber weights are.
                        if t % 11 == 5 {
                            each.decay(*lam);
                            held_cov.decay(*lam);
                            continue;
                        }
                        each.update(x, *lam, *w);
                        held_cov.update_held(x, *lam, *w, &mut held);
                        // The means and weights move as the row arrives.
                        assert_eq!(each.n_eff().to_bits(), held_cov.n_eff().to_bits());
                        assert_eq!(each.mean(k - 1).to_bits(), held_cov.mean(k - 1).to_bits());
                    }
                    held_cov.merge_held(&mut held);
                    assert_eq!(bits(&each), bits(&held_cov), "k {k}, n {n}, seed {seed}");
                    assert_eq!(each, held_cov);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 18 * 6 * 3);
    }

    #[test]
    fn one_hold_serves_many_replays_and_widths() {
        // A kept hold reshapes for a new width and starts over at each
        // replay, from the co-moments as they stand.
        let mut held = HeldSteps::default();
        for (k, seed) in [(5, 1), (5, 2), (9, 3), (3, 4), (9, 5)] {
            let rows = stream(k, 150, seed);
            let (head, tail) = rows.split_at(40);
            let mut each = EwCov::new(k).without_runs();
            for (x, lam, w) in head {
                each.update(x, *lam, *w);
            }
            let mut held_cov = each.clone();
            for (x, lam, w) in tail {
                each.update(x, *lam, *w);
                held_cov.update_held(x, *lam, *w, &mut held);
            }
            held_cov.merge_held(&mut held);
            assert_eq!(bits(&each), bits(&held_cov), "k {k}");
            // A merge with nothing held moves nothing.
            held_cov.merge_held(&mut held);
            assert_eq!(bits(&each), bits(&held_cov), "k {k}");
        }
    }

    #[test]
    fn a_blocked_accumulator_takes_its_rows_as_updates() {
        let k = 6;
        let rows = stream(k, 120, 9);
        let mut each = EwCov::new(k);
        each.set_block_rows(16);
        let mut held_cov = each.clone();
        let mut held = HeldSteps::default();
        for (x, lam, w) in &rows {
            each.update(x, *lam, *w);
            held_cov.update_held(x, *lam, *w, &mut held);
        }
        held_cov.merge_held(&mut held);
        each.flush();
        held_cov.flush();
        assert_eq!(bits(&each), bits(&held_cov));
    }
}
