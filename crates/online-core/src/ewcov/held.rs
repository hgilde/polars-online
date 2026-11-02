//! [`HeldSteps`]: the co-moments' rank-one steps of a run of rows, held and
//! taken together in register blocks, to the bit of taking them row by row.

use super::{CoStep, Threads};

/// Rows a [`HeldSteps`] holds before it takes their steps.
const HELD_ROWS: usize = 64;

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
/// the held rows, and stored once (in AVX2's lanes on x86-64, and row by
/// row on an x86-64 CPU without them: [`Self::take`]). Every entry takes
/// each row's step in the rows' order, with the same expression -- one
/// product for `a_t·c_ij`, one for the update, one sum, no fused
/// multiply-add --
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
/// features ran 12-16% faster, a quantile fit's 3.5-5%.
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

    /// Take the held rows' steps on the padded copy. In tiles
    /// ([`Self::tile`]) where the CPU's vector registers hold a tile:
    /// aarch64's 32 always, x86-64's 16 where it has AVX2's four lanes
    /// ([`HeldLanes`]); on an x86-64 CPU without them, whose 16 registers of
    /// two lanes a tile's sums and operands overflow, row by row
    /// ([`Self::take_rows`]).
    fn take(&mut self) {
        #[cfg(target_arch = "x86_64")]
        match crate::sgd::x86_v3() {
            Some(simd) => {
                let (k, ks, n) = (self.k, self.ks, self.rows);
                let (c, a, d, abd) = self.parts();
                pulp::Simd::vectorize(
                    simd,
                    HeldLanes {
                        c,
                        a: &a[..n],
                        d,
                        abd,
                        k,
                        ks,
                    },
                );
            }
            None => self.take_rows(),
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.take_tiles();
        self.rows = 0;
    }

    /// The held rows' steps in tiles, row block by row block
    /// ([`Self::row_block`]).
    #[cfg(any(not(target_arch = "x86_64"), test))]
    fn take_tiles(&mut self) {
        for pb in 0..self.ks / 4 {
            match self.k - pb * 4 {
                1 => self.row_block::<1>(pb),
                2 => self.row_block::<2>(pb),
                3 => self.row_block::<3>(pb),
                _ => self.row_block::<4>(pb),
            }
        }
    }

    /// Row block `pb`'s tiles: its `R` rows (four, or the last block's rows
    /// below `k`), two column blocks at a time while two fit at or below
    /// the diagonal block, then one. A 4 × 8 tile's 32 sums are 16 of
    /// aarch64's 32 vector registers, enough independent sums to keep its
    /// pipes busy where a 4 × 4 tile's 16 waited on their own products.
    #[cfg(any(not(target_arch = "x86_64"), test))]
    #[inline(always)]
    fn row_block<const R: usize>(&mut self, pb: usize) {
        let mut qb = 0;
        while qb <= pb {
            if qb < pb {
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
    #[cfg(any(not(target_arch = "x86_64"), test))]
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

    /// The held rows' steps row by row, each lower entry's expression as
    /// [`super::rank_one_rows`] takes it: x86-64 without AVX2.
    #[cfg(any(target_arch = "x86_64", test))]
    fn take_rows(&mut self) {
        let (k, ks, n) = (self.k, self.ks, self.rows);
        let (c, a, d, abd) = self.parts();
        for ((&at, d), abd) in a[..n]
            .iter()
            .zip(d.chunks_exact(ks))
            .zip(abd.chunks_exact(ks))
        {
            for (i, &e) in abd[..k].iter().enumerate() {
                for (cj, &dj) in c[i * ks..=i * ks + i].iter_mut().zip(d) {
                    *cj = at * *cj + e * dj;
                }
            }
        }
    }

    /// Whether a step is held that [`Self::merge_into`] would write.
    pub(crate) fn is_open(&self) -> bool {
        self.open
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

/// [`HeldSteps::take`] in AVX2's four lanes: [`HeldSteps::row_block`]'s
/// tiles, each tile row's four or eight sums in one or two vectors (eight
/// of the 16 registers at most), a held row's `a_t` and `(a_t·b_t)·d_ti`
/// broadcast, and each lane `a_t·s + e·d`, one product, the other, one
/// sum, no fused multiply-add: the tiles' arithmetic to the bit. Through
/// pulp's checked dispatch, with no `unsafe` here.
#[cfg(target_arch = "x86_64")]
struct HeldLanes<'a> {
    c: &'a mut [f64],
    a: &'a [f64],
    d: &'a [f64],
    abd: &'a [f64],
    k: usize,
    ks: usize,
}

#[cfg(target_arch = "x86_64")]
impl pulp::WithSimd for HeldLanes<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        // Only AVX2's four lanes reach here (`crate::sgd::x86_v3`).
        debug_assert_eq!(S::F64_LANES, 4);
        let Self {
            c,
            a,
            d,
            abd,
            k,
            ks,
        } = self;
        for pb in 0..ks / 4 {
            let mut qb = 0;
            while qb <= pb {
                let wide = qb < pb;
                let t = Tile { pb, qb, ks };
                match (k - pb * 4, wide) {
                    (1, true) => lanes_tile::<S, 1, 2>(simd, c, a, d, abd, t),
                    (2, true) => lanes_tile::<S, 2, 2>(simd, c, a, d, abd, t),
                    (3, true) => lanes_tile::<S, 3, 2>(simd, c, a, d, abd, t),
                    (_, true) => lanes_tile::<S, 4, 2>(simd, c, a, d, abd, t),
                    (1, false) => lanes_tile::<S, 1, 1>(simd, c, a, d, abd, t),
                    (2, false) => lanes_tile::<S, 2, 1>(simd, c, a, d, abd, t),
                    (3, false) => lanes_tile::<S, 3, 1>(simd, c, a, d, abd, t),
                    (_, false) => lanes_tile::<S, 4, 1>(simd, c, a, d, abd, t),
                }
                qb += if wide { 2 } else { 1 };
            }
        }
    }
}

/// Where a tile sits: row block `pb`, column block `qb`, in a padded
/// matrix `ks` wide.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Tile {
    pb: usize,
    qb: usize,
    ks: usize,
}

/// One `R × 4V` tile of [`HeldLanes`], `V` vectors a tile row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn lanes_tile<S: pulp::Simd, const R: usize, const V: usize>(
    simd: S,
    c: &mut [f64],
    a: &[f64],
    d: &[f64],
    abd: &[f64],
    Tile { pb, qb, ks }: Tile,
) {
    let (p0, q0) = (pb * 4, qb * 4);
    let load = |v: &[f64]| S::as_simd_f64s(&v[..4]).0[0];
    let mut acc = [[simd.splat_f64s(0.0); V]; R];
    for (r, acc_r) in acc.iter_mut().enumerate() {
        for (v, s) in acc_r.iter_mut().enumerate() {
            *s = load(&c[(p0 + r) * ks + q0 + 4 * v..]);
        }
    }
    for ((&at, abd), d) in a.iter().zip(abd.chunks_exact(ks)).zip(d.chunks_exact(ks)) {
        let at = simd.splat_f64s(at);
        let mut dv = [simd.splat_f64s(0.0); V];
        for (v, x) in dv.iter_mut().enumerate() {
            *x = load(&d[q0 + 4 * v..]);
        }
        for (acc_r, &e) in acc.iter_mut().zip(&abd[p0..p0 + R]) {
            let e = simd.splat_f64s(e);
            for (s, &x) in acc_r.iter_mut().zip(&dv) {
                *s = simd.add_f64s(simd.mul_f64s(at, *s), simd.mul_f64s(e, x));
            }
        }
    }
    for (r, acc_r) in acc.iter().enumerate() {
        for (v, s) in acc_r.iter().enumerate() {
            let at = (p0 + r) * ks + q0 + 4 * v;
            S::as_mut_simd_f64s(&mut c[at..at + 4]).0[0] = *s;
        }
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

    /// The tiles, the row-by-row steps and whichever this CPU takes (AVX2's
    /// lanes on x86-64 where it has them) are each the row-by-row update's
    /// numbers, at widths on and off a multiple of 4 and holds of a few
    /// rows to a full one, from co-moments a head of rows has moved: from
    /// an empty accumulator every tile starts from zeros, wherever it is
    /// read from, and on x86-64 nothing else takes the tiles (task 307).
    #[test]
    fn every_way_of_taking_the_steps_is_the_same() {
        let mut checked = 0;
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 21, 24, 27] {
            for n in [1, 3, 30, 63] {
                let rows = stream(k, n + 7, k as u64 * 31 + n as u64);
                let (head, rows) = rows.split_at(7);
                let mut each = EwCov::new(k).without_runs();
                for (x, lam, w) in head {
                    each.update(x, *lam, *w);
                }
                assert!(
                    (0..k).all(|i| each.comoments().get(k, i, 0) != 0.0),
                    "k {k}, n {n}: the head moved every row of the co-moments"
                );
                let mut cov = each.clone();
                let mut held = HeldSteps::default();
                for (x, lam, w) in rows {
                    each.update(x, *lam, *w);
                    cov.update_held(x, *lam, *w, &mut held);
                }
                let ways: [fn(&mut HeldSteps); 3] =
                    [HeldSteps::take, HeldSteps::take_tiles, HeldSteps::take_rows];
                for take in ways {
                    let mut h = held.clone();
                    let mut c = cov.clone();
                    if h.open {
                        take(&mut h);
                        let ks = h.ks;
                        let (cp, ..) = h.parts();
                        let lower = c.c.lower_mut();
                        for i in 0..k {
                            lower[i * k..=i * k + i].copy_from_slice(&cp[i * ks..=i * ks + i]);
                        }
                    }
                    assert_eq!(bits(&each), bits(&c), "k {k}, n {n}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 16 * 4 * 3);
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

    /// The steps are held, not taken: after each row the co-moments stand
    /// where the hold opened, the means and weights have moved, and the
    /// hold counts the row, until the `HELD_ROWS`-th takes them all into
    /// the padded copy; the merge brings the co-moments to the row-by-row
    /// numbers (task 307: the bits alone cannot tell a hold that takes its
    /// steps a row at a time, or never holds them, from one that holds).
    #[test]
    fn the_steps_are_held_until_a_full_hold_or_the_merge() {
        let k = 5;
        let rows = stream(k, 3 * HELD_ROWS / 2, 41);
        let (head, rows) = rows.split_at(HELD_ROWS / 4);
        let mut each = EwCov::new(k).without_runs();
        for (x, lam, w) in head {
            each.update(x, *lam, *w);
        }
        let mut cov = each.clone();
        let opened = bits(&cov)[1..].to_vec();
        let comoments = |c: &EwCov| -> Vec<u64> {
            (0..k * k)
                .map(|ij| c.comoments().get(k, ij / k, ij % k).to_bits())
                .collect()
        };
        let at_open = comoments(&cov);
        let mut held = HeldSteps::default();
        for (t, (x, lam, w)) in rows.iter().enumerate() {
            each.update(x, *lam, *w);
            cov.update_held(x, *lam, *w, &mut held);
            assert_eq!(
                comoments(&cov),
                at_open,
                "row {t}: the co-moments are behind"
            );
            assert_eq!(cov.n_eff().to_bits(), each.n_eff().to_bits(), "row {t}");
            assert!(held.open, "row {t}");
            assert_eq!(held.rows, (t + 1) % HELD_ROWS, "row {t}: rows held");
        }
        assert_ne!(bits(&cov)[1..], opened[..], "the means moved");
        cov.merge_held(&mut held);
        assert!(!held.open);
        assert_eq!(bits(&each), bits(&cov));
    }

    /// Steps held on an accumulator that then blocks its rows are merged
    /// before the first row the block takes, as `update_held` says: the
    /// block's merge reads the co-moments, and held steps under it would be
    /// left out of it, then written back over it by their own merge (task
    /// 307).
    #[test]
    fn steps_held_before_the_accumulator_blocks_are_merged_first() {
        let k = 6;
        let rows = stream(k, 150, 13);
        let mut each = EwCov::new(k).without_runs();
        let mut held_cov = each.clone();
        let mut held = HeldSteps::default();
        for (t, (x, lam, w)) in rows.iter().enumerate() {
            if t == 40 {
                each.set_block_rows(16);
                held_cov.set_block_rows(16);
                assert!(held.open && held.rows > 0, "steps are held as it blocks");
            }
            each.update(x, *lam, *w);
            held_cov.update_held(x, *lam, *w, &mut held);
            if t >= 40 {
                assert!(!held.open, "row {t}: merged, and nothing held since");
            }
        }
        held_cov.merge_held(&mut held);
        each.flush();
        held_cov.flush();
        assert_eq!(bits(&each), bits(&held_cov));
    }
}
