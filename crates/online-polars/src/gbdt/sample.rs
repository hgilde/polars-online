//! A sample of the rows, drawn in one pass, and each feature's bin edges
//! placed from it: the cuts that replace XGBoost's sketch pass
//! (docs/GBDT-ADAPTER.md §3.1, §4.1).
//!
//! **The sample.** Row `i` (its index in the stream) is given the hash
//! `h(i)`, SplitMix64 of the index and a seed, and with it
//! `u_i = (2 h(i) + 1) / 2^65`, uniform on (0, 1). Row `i` of weight `w_i`
//! (1 without a weight column) has the priority `w_i / u_i`, and the `k`
//! rows of largest priority are kept (an earlier row before a later one of
//! equal priority): priority sampling (Duffield, Lund & Thorup 2007).
//! Without weights that is the `k` rows of smallest hash, a bottom-`k`
//! sketch: a uniform sample of `k` rows without replacement. Either way it
//! is all of them when the stream is shorter, the same rows however the
//! stream is cut into frames and whichever engine reads it, and no row count
//! needed in advance. A kept row stores each feature as XGBoost reads it,
//! float32, NaN for a null. Two priorities are compared exactly, as the
//! integers `w_a (2 h_b + 1)` and `w_b (2 h_a + 1)`, so that equal weights
//! keep exactly the rows of smallest hash.
//!
//! **What a kept row stands for.** With `tau` the largest priority not kept
//! (0 when every row is), a kept row stands for `max(w_i, tau)` of weight:
//! a row of weight above `tau` is always kept and counts its own weight, and
//! the rest are kept with probability about `w_i / tau` and each count
//! `tau`. Summed over any set of rows, these weights are an unbiased
//! estimate of the set's weight (the priority-sampling estimator), so the
//! sample's weighted quantiles estimate the stream's. A row of weight 0 has
//! priority 0, is kept only when the stream is shorter than `k`, and stands
//! for nothing.
//!
//! **The edges.** XGBoost bins a value `v` into the bin of the first cut
//! above it, its cuts being `[-inf, c_1, .., c_m, c_last]`. Planted as a
//! `ref` column of `d <= max_bin` distinct values `u_0 < u_1 < .. <
//! u_{d-1}`, its cuts come back as `[-inf, u_1, .., u_{d-1}, u_{d-1} +
//! |u_{d-1}| + 1e-5]` (measured on XGBoost 3.4.1), so the planted values
//! are the edges. Only kept rows of positive weight and a finite value
//! count. A feature whose sample holds at most `max_bin` distinct values
//! plants them all, so that each value it saw gets a bin of its own, as
//! XGBoost's own sketch gives an exact column. Otherwise, with `s` the `m`
//! sample values sorted, it plants `s_0` and, for `j = 1 .. max_bin - 1`,
//! the first `s_i` whose cumulative weight exceeds `j / max_bin` of the
//! total, each distinct: `max_bin` bins of equal weight, the target of
//! XGBoost's weighted sketch. When every kept row stands for the same
//! weight, as without a weight column, that is `s_{floor(j m / max_bin)}`,
//! computed so, which equal weights give to the bit. A categorical feature
//! plants its codes `0 .. levels`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::frames::{Segment, checked_f64};
use super::{Kind, Layout};

/// What a `ref` matrix plants for one feature.
#[derive(Debug, Clone, PartialEq)]
pub enum Planted {
    /// Distinct finite values, ascending: the lowest sets no cut, each other
    /// one is a cut.
    Values(Vec<f32>),
    /// A categorical feature's codes `0..levels`.
    Levels(u32),
}

impl Planted {
    fn column(&self) -> Vec<f32> {
        match self {
            Planted::Values(v) => v.clone(),
            Planted::Levels(n) => (0..(*n).max(1)).map(|c| c as f32).collect(),
        }
    }
}

/// `2^-65`, exactly.
const TWO_POW_M65: f64 = 1.0 / 36_893_488_147_419_103_232.0;

/// A row's priority `w / u`, `u = (2 h + 1) / 2^65`, compared exactly.
#[derive(Debug, Clone, Copy)]
struct Priority {
    w: f64,
    h: u64,
}

/// A positive finite `x` as `m 2^e`, `m` an integer below `2^53`.
fn split(x: f64) -> (u128, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = u128::from(bits & ((1u64 << 52) - 1));
    if exp == 0 {
        (frac, -1074)
    } else {
        // `+`, not `|`: the bit is clear, and `+` has no equivalent mutant.
        (frac + (1u128 << 52), exp - 1075)
    }
}

impl Ord for Priority {
    fn cmp(&self, o: &Self) -> Ordering {
        match (self.w == 0.0, o.w == 0.0) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }
        if self.w == o.w {
            // The same weight: the smaller hash, the larger priority.
            return o.h.cmp(&self.h);
        }
        // w_a / u_a against w_b / u_b is w_a (2 h_b + 1) against
        // w_b (2 h_a + 1): at most 53 + 65 bits each, so exact in u128.
        let ((ma, ea), (mb, eb)) = (split(self.w), split(o.w));
        let pa = ma * (2 * u128::from(o.h) + 1);
        let pb = mb * (2 * u128::from(self.h) + 1);
        let la = (128 - pa.leading_zeros()) as i32 + ea;
        let lb = (128 - pb.leading_zeros()) as i32 + eb;
        if la != lb {
            return la.cmp(&lb);
        }
        // The same bit length: shifting the one of larger exponent down to
        // the other's keeps it within that length.
        if ea >= eb {
            (pa << (ea - eb)).cmp(&pb)
        } else {
            pa.cmp(&(pb << (eb - ea)))
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl PartialEq for Priority {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Priority {}

impl Priority {
    /// `w / u` divided by `scale`, as a float: at most `2^65 w / scale`.
    fn value(self, scale: f64) -> f64 {
        (self.w / scale) / ((2.0 * self.h as f64 + 1.0) * TWO_POW_M65)
    }
}

/// A priority sample of the rows (module docs): without weights, the `k`
/// rows of smallest hash of their index.
pub struct Sampler {
    k: usize,
    kinds: Vec<Kind>,
    seed: u64,
    weighted: bool,
    /// `(priority, slot)` of every kept row, the smallest priority on top.
    heap: BinaryHeap<(Reverse<Priority>, usize)>,
    /// Feature `j` of the row in slot `s` is `store[j][s]`; empty for a
    /// categorical feature, whose codes are planted from its levels.
    store: Vec<Vec<f32>>,
    /// The weight of the row in slot `s`, when `weighted`.
    weights: Vec<f64>,
    /// The largest priority not kept.
    tau: Option<Priority>,
    filled: usize,
    seen: u64,
}

/// SplitMix64 (Steele, Lea & Flood 2014) of row `i` under `seed`.
pub(crate) fn hash(i: u64, seed: u64) -> u64 {
    let mut z = (i ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Sampler {
    /// A sampler of `k` rows over features of `kinds`, by the frames'
    /// weight column when `weighted` (the layout's last column).
    pub fn new(kinds: Vec<Kind>, k: usize, seed: u64, weighted: bool) -> Self {
        let store = kinds.iter().map(|_| Vec::new()).collect();
        Self {
            k: k.max(1),
            kinds,
            seed,
            weighted,
            heap: BinaryHeap::new(),
            store,
            weights: Vec::new(),
            tau: None,
            filled: 0,
            seen: 0,
        }
    }

    /// Rows offered so far.
    pub fn rows_seen(&self) -> u64 {
        self.seen
    }

    /// The weights of the kept rows, by slot (empty without weights).
    #[cfg(test)]
    pub(crate) fn kept_weights(&self) -> &[f64] {
        &self.weights[..self.filled.min(self.weights.len())]
    }

    /// Rows kept so far.
    pub fn rows_kept(&self) -> usize {
        self.filled
    }

    pub(crate) fn push(
        &mut self,
        seg: &Segment,
        layout: &Layout,
        pool: &rayon::ThreadPool,
    ) -> Result<(), super::GbdtError> {
        let w = if self.weighted {
            let j = layout.names.len() + 1;
            Some(checked_f64(seg, j, layout, self.seen)?)
        } else {
            None
        };
        let mut picks: Vec<(usize, usize)> = Vec::new();
        for r in 0..seg.n {
            let p = Priority {
                w: w.map_or(1.0, |w| w[r]),
                h: hash(self.seen + r as u64, self.seed),
            };
            if self.filled < self.k {
                self.heap.push((Reverse(p), self.filled));
                picks.push((r, self.filled));
                self.filled += 1;
            } else if let Some(&(Reverse(top), slot)) = self.heap.peek() {
                let out = if p > top {
                    self.heap.pop();
                    self.heap.push((Reverse(p), slot));
                    picks.push((r, slot));
                    top
                } else {
                    p
                };
                if self.weighted && self.tau.is_none_or(|t| out > t) {
                    self.tau = Some(out);
                }
            }
        }
        self.seen += seg.n as u64;
        if picks.is_empty() {
            return Ok(());
        }
        if let Some(w) = w {
            for &(r, slot) in &picks {
                if slot < self.weights.len() {
                    self.weights[slot] = w[r];
                } else {
                    self.weights.push(w[r]);
                }
            }
        }
        let kinds = &self.kinds;
        pool.install(|| {
            self.store
                .par_iter_mut()
                .enumerate()
                .filter(|(j, _)| !matches!(kinds[*j], Kind::Category { .. }))
                .for_each(|(j, col)| {
                    let num = seg.num(j);
                    // In order: a slot taken twice in one segment keeps the
                    // later row, as the heap does.
                    for &(r, slot) in &picks {
                        let v = num.f32_at(r);
                        if slot < col.len() {
                            col[slot] = v;
                        } else {
                            col.push(v);
                        }
                    }
                });
        });
        Ok(())
    }

    /// What each kept row stands for, `max(w_i, tau)` over the largest
    /// weight kept (module docs); `None` when every kept row stands for the
    /// same positive weight, as without weights, whose quantiles are then
    /// the unweighted ones.
    pub(crate) fn stands_for(&self) -> Option<Vec<f64>> {
        if !self.weighted {
            return None;
        }
        let w = &self.weights[..self.filled];
        let scale = w.iter().copied().fold(0.0, f64::max);
        if scale == 0.0 {
            // No weight at all: nothing seen.
            return Some(vec![0.0; w.len()]);
        }
        let tau = self.tau.map_or(0.0, |t| t.value(scale));
        let est: Vec<f64> = w.iter().map(|&x| (x / scale).max(tau)).collect();
        if est.iter().all(|&e| e == est[0]) {
            None
        } else {
            Some(est)
        }
    }

    /// Each feature's planted values for `max_bin` bins (module docs).
    pub fn planted(&self, max_bin: usize, pool: &rayon::ThreadPool) -> Vec<Planted> {
        let max_bin = max_bin.max(2);
        let est = self.stands_for();
        pool.install(|| {
            self.kinds
                .par_iter()
                .zip(self.store.par_iter())
                .map(|(kind, col)| match kind {
                    Kind::Category { levels } => Planted::Levels(*levels),
                    _ => Planted::Values(match &est {
                        None => planted_values(&col[..self.filled], max_bin),
                        Some(est) => weighted_values(&col[..self.filled], est, max_bin),
                    }),
                })
                .collect()
        })
    }

    /// The `ref` matrix XGBoost places its cuts from: row-major float32, a
    /// column per feature holding its planted values, a shorter column
    /// padded by repeating its last value. Returns the values and the row
    /// count; both engines build their `ref` from this one matrix.
    pub fn ref_matrix(&self, max_bin: usize, pool: &rayon::ThreadPool) -> (Vec<f32>, usize) {
        let cols: Vec<Vec<f32>> = self
            .planted(max_bin, pool)
            .iter()
            .map(Planted::column)
            .collect();
        let p = cols.len();
        let m = cols.iter().map(Vec::len).max().unwrap_or(1);
        let mut out = vec![0f32; m * p];
        for (j, col) in cols.iter().enumerate() {
            for r in 0..m {
                out[r * p + j] = col[r.min(col.len() - 1)];
            }
        }
        (out, m)
    }

    /// The frames of `layout` offered in order (the features only are read).
    pub fn push_frame(
        &mut self,
        df: &polars::prelude::DataFrame,
        layout: &Layout,
        pool: &rayon::ThreadPool,
    ) -> Result<(), super::GbdtError> {
        for seg in super::frames::segments(df, layout)? {
            self.push(&seg, layout, pool)?;
        }
        Ok(())
    }

    pub(crate) fn push_any(
        &mut self,
        frame: super::frames::Frame,
        layout: &Layout,
        pool: &rayon::ThreadPool,
    ) -> Result<(), super::GbdtError> {
        for seg in super::frames::segments_of(frame, layout)? {
            self.push(&seg, layout, pool)?;
        }
        Ok(())
    }
}

fn planted_values(vals: &[f32], max_bin: usize) -> Vec<f32> {
    let mut s: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if s.is_empty() {
        // Nothing seen: one value, so one bin; every value lands in it.
        return vec![0.0];
    }
    s.sort_unstable_by(f32::total_cmp);
    let mut distinct = s.clone();
    // `==` holds -0.0 and 0.0 equal, as XGBoost's comparisons do.
    distinct.dedup();
    if distinct.len() <= max_bin {
        return distinct;
    }
    let m = s.len();
    let mut out = vec![s[0]];
    for j in 1..max_bin {
        let v = s[j * m / max_bin];
        if v > *out.last().expect("one value at least") {
            out.push(v);
        }
    }
    out
}

/// The planted values of a feature whose kept rows stand for `est` (module
/// docs): rows of weight 0 and missing values count for nothing.
fn weighted_values(vals: &[f32], est: &[f64], max_bin: usize) -> Vec<f32> {
    let mut s: Vec<(f32, f64)> = vals
        .iter()
        .zip(est)
        .filter(|(v, e)| v.is_finite() && **e > 0.0)
        .map(|(v, e)| (*v, *e))
        .collect();
    if s.is_empty() {
        return vec![0.0];
    }
    // By value, then by weight, so that the running sum is one order
    // whichever slots the rows sit in; -0.0 and 0.0 are one value.
    s.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("finite")
            .then(a.1.total_cmp(&b.1))
    });
    let mut distinct: Vec<f32> = s.iter().map(|p| p.0).collect();
    distinct.dedup();
    if distinct.len() <= max_bin {
        return distinct;
    }
    let total: f64 = s.iter().map(|p| p.1).sum();
    let mut out = vec![s[0].0];
    // The running sum ends at `total`, summed in the same order, and every
    // target is below it: `j / max_bin <= 1 - 1 / max_bin` is far more than
    // a rounding from 1, and `total` is at least 1, the largest weight over
    // itself. So the loop stops at a row.
    let (mut i, mut acc) = (0, s[0].1);
    for j in 1..max_bin {
        let target = total * j as f64 / max_bin as f64;
        while acc <= target {
            i += 1;
            acc += s[i].1;
        }
        if s[i].0 > *out.last().expect("one value at least") {
            out.push(s[i].0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_of_few_values_plants_them_all() {
        let v = [3.0, 1.0, 2.0, 1.0, f32::NAN, 3.0, -0.0, 0.0];
        assert_eq!(planted_values(&v, 256), vec![-0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_column_of_many_values_plants_max_bin_quantiles() {
        let v: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        let p = planted_values(&v, 256);
        assert_eq!(p.len(), 256);
        assert_eq!(p[0], 0.0);
        // s_{floor(j m / max_bin)}: j = 1 is row 39 of 10,000.
        assert_eq!(p[1], 39.0);
        assert_eq!(p[255], (255 * 10_000 / 256) as f32);
        assert!(p.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn an_unseen_column_plants_one_value() {
        assert_eq!(planted_values(&[f32::NAN, f32::INFINITY], 256), vec![0.0]);
    }

    fn prio(w: f64, h: u64) -> Priority {
        Priority { w, h }
    }

    #[test]
    fn priorities_compare_exactly() {
        use Ordering::*;
        // u = (2h + 1) / 2^65: h = 1 is three times h = 0, so weight 3 at
        // h = 1 ties weight 1 at h = 0, to the last bit.
        assert_eq!(prio(3.0, 1).cmp(&prio(1.0, 0)), Equal);
        assert_eq!(prio(3.0f64.next_up(), 1).cmp(&prio(1.0, 0)), Greater);
        assert_eq!(prio(3.0f64.next_down(), 1).cmp(&prio(1.0, 0)), Less);
        // The same weight: the smaller hash ranks higher.
        assert_eq!(prio(2.5, 10).cmp(&prio(2.5, 11)), Greater);
        assert_eq!(prio(2.5, 11).cmp(&prio(2.5, 10)), Less);
        // Weight 0 ranks below any other, whatever its hash.
        assert_eq!(prio(0.0, 0).cmp(&prio(f64::MIN_POSITIVE, u64::MAX)), Less);
        assert_eq!(prio(5e-324, u64::MAX).cmp(&prio(0.0, 0)), Greater);
        assert_eq!(prio(0.0, 1).cmp(&prio(-0.0, 7)), Equal);
        assert!(prio(3.0, 1) == prio(1.0, 0) && prio(3.0, 1) != prio(1.0, 1));
        // As a float: w / u over the scale, u = (2h + 1) / 2^65.
        let two65 = 36_893_488_147_419_103_232.0;
        assert_eq!(prio(1.0, 0).value(1.0), two65);
        assert_eq!(prio(3.0, 1).value(1.0), two65);
        assert_eq!(prio(6.0, 1).value(2.0), two65);
        assert_eq!(prio(5.0, 2).value(1.0), two65);
        // Across the whole range, against the exact rational order.
        // Subnormal and the smallest normal too: 2^-1074 at a large hash
        // ratio outranks 2^-1022 at a small one.
        let ws = [
            5e-324,
            1e-310,
            f64::MIN_POSITIVE,
            1e-300,
            0.1,
            0.5,
            1.0,
            2.0,
            3.0,
            7.25,
            1e300,
            f64::MAX,
        ];
        let hs = [0, 1, 2, 3, 1 << 40, u64::MAX / 3, u64::MAX - 1, u64::MAX];
        for &wa in &ws {
            for &ha in &hs {
                for &wb in &ws {
                    for &hb in &hs {
                        let want = exact(wa, hb).cmp(&exact(wb, ha));
                        assert_eq!(prio(wa, ha).cmp(&prio(wb, hb)), want, "{wa} {ha} {wb} {hb}");
                    }
                }
            }
        }
    }

    /// `w (2 h + 1)` as an exact big integer scaled by `2^1074`: a vector
    /// of base-2^32 limbs, most significant first. `w` is taken apart by
    /// doubling, which is exact, rather than by its bits as `split` does.
    fn exact(w: f64, h: u64) -> Vec<u64> {
        let (mut v, mut e) = (w, 0i32);
        while v.fract() != 0.0 {
            v *= 2.0;
            e -= 1;
        }
        while v >= 9_007_199_254_740_992.0 {
            v /= 2.0;
            e += 1;
        }
        let m = v as u128;
        let mut limbs = vec![0u64; 72];
        let prod = m * (2 * u128::from(h) + 1);
        let shift = (e + 1074) as usize;
        for bit in 0..128 {
            if prod >> bit & 1 == 1 {
                let at = bit + shift;
                limbs[at / 32] |= 1 << (at % 32);
            }
        }
        limbs.reverse();
        limbs
    }

    #[test]
    fn weighted_values_put_equal_weight_in_each_bin() {
        // 0..1000, the upper half of weight 9: nine tenths of the weight
        // above 500, so 9 of 10 bins' edges.
        let v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let w: Vec<f64> = (0..1000)
            .map(|i| if i >= 500 { 9.0 } else { 1.0 })
            .collect();
        let p = weighted_values(&v, &w, 10);
        // Total 5000; edge j is the first value whose running weight
        // exceeds 500 j: 500 for j = 1, then every 500 / 9 rows above it.
        assert_eq!(p[0], 0.0);
        assert_eq!(p[1], 500.0);
        assert_eq!(p[2], 555.0);
        assert_eq!(p.len(), 10);
        assert_eq!(p[9], 944.0);
    }

    #[test]
    fn a_row_of_weight_zero_or_a_missing_value_places_nothing() {
        let v = [-50.0, 1.0, 2.0, f32::NAN, 3.0, 99.0];
        let w = [0.0, 1.0, 1.0, 5.0, 1.0, 0.0];
        assert_eq!(weighted_values(&v, &w, 256), vec![1.0, 2.0, 3.0]);
        let many: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut w = vec![1.0; 100];
        w[0] = 0.0;
        w[99] = 0.0;
        let p = weighted_values(&many, &w, 4);
        assert_eq!(p, vec![1.0, 25.0, 50.0, 74.0]);
        assert_eq!(weighted_values(&v, &[0.0; 6], 256), vec![0.0]);
    }

    #[test]
    fn one_row_of_nearly_all_the_weight_takes_every_edge() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut w = vec![1.0; 1000];
        w[321] = 1e12;
        assert_eq!(weighted_values(&v, &w, 256), vec![0.0, 321.0]);
    }

    #[test]
    fn equal_weights_place_the_unweighted_edges() {
        let v: Vec<f32> = (0..10_007)
            .map(|i| ((i * 7919) % 10_007) as f32 * 0.1)
            .collect();
        for x in [1.0, 0.1, 3.7, 1e-300] {
            assert_eq!(
                weighted_values(&v, &vec![x; v.len()], 256),
                planted_values(&v, 256),
                "{x}"
            );
        }
    }

    #[test]
    fn the_hash_spreads_consecutive_rows() {
        let below: usize = (0..100_000u64)
            .filter(|&i| hash(i, 0) < u64::MAX / 10)
            .count();
        assert!((9_000..11_000).contains(&below), "{below}");
        assert_ne!(hash(1, 0), hash(1, 1));
    }
}
