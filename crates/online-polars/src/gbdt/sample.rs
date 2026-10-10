//! A uniform sample of the rows, drawn in one pass, and each feature's bin
//! edges placed from it: the cuts that replace XGBoost's sketch pass
//! (docs/GBDT-ADAPTER.md §3.1, §4.1).
//!
//! **The sample.** Row `i` (its index in the stream) is given the hash
//! `h(i)`, SplitMix64 of the index and a seed, and the `k` rows with the
//! smallest hashes are kept, a bottom-`k` sketch: a uniform sample of `k`
//! rows without replacement (all of them when the stream is shorter), the
//! same rows however the stream is cut into frames and whichever engine
//! reads it, and no row count needed in advance. A kept row stores each
//! feature as XGBoost reads it, float32, NaN for a null.
//!
//! **The edges.** XGBoost bins a value `v` into the bin of the first cut
//! above it, its cuts being `[-inf, c_1, .., c_m, c_last]`. Planted as a
//! `ref` column of `d <= max_bin` distinct values `u_0 < u_1 < .. <
//! u_{d-1}`, its cuts come back as `[-inf, u_1, .., u_{d-1}, u_{d-1} +
//! |u_{d-1}| + 1e-5]` (measured on XGBoost 3.4.1), so the planted values
//! are the edges. A feature whose sample holds at most `max_bin` distinct
//! values plants them all, so that each value it saw gets a bin of its own,
//! as XGBoost's own sketch gives an exact column. Otherwise, with `s` the
//! sorted `m` sample values, it plants `s_0` and the quantiles
//! `s_{floor(j m / max_bin)}` for `j = 1 .. max_bin - 1`, each distinct:
//! `max_bin` bins at equal sample frequency, the target of XGBoost's
//! sketch (a rank error of `1 / max_bin`). A categorical feature plants its
//! codes `0 .. levels`.

use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::frames::Segment;
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

/// A bottom-`k` sample of the rows by the hash of their index.
pub struct Sampler {
    k: usize,
    kinds: Vec<Kind>,
    seed: u64,
    /// `(hash, slot)` of every kept row, the largest hash on top.
    heap: BinaryHeap<(u64, usize)>,
    /// Feature `j` of the row in slot `s` is `store[j][s]`; empty for a
    /// categorical feature, whose codes are planted from its levels.
    store: Vec<Vec<f32>>,
    filled: usize,
    seen: u64,
}

/// SplitMix64 (Steele, Lea & Flood 2014) of row `i` under `seed`.
fn hash(i: u64, seed: u64) -> u64 {
    let mut z = (i ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Sampler {
    /// A sampler of `k` rows over features of `kinds`.
    pub fn new(kinds: Vec<Kind>, k: usize, seed: u64) -> Self {
        let store = kinds.iter().map(|_| Vec::new()).collect();
        Self {
            k: k.max(1),
            kinds,
            seed,
            heap: BinaryHeap::new(),
            store,
            filled: 0,
            seen: 0,
        }
    }

    /// Rows offered so far.
    pub fn rows_seen(&self) -> u64 {
        self.seen
    }

    /// Rows kept so far.
    pub fn rows_kept(&self) -> usize {
        self.filled
    }

    pub(crate) fn push(&mut self, seg: &Segment, pool: &rayon::ThreadPool) {
        let mut picks: Vec<(usize, usize)> = Vec::new();
        for r in 0..seg.n {
            let h = hash(self.seen + r as u64, self.seed);
            if self.filled < self.k {
                self.heap.push((h, self.filled));
                picks.push((r, self.filled));
                self.filled += 1;
            } else if let Some(&(top, slot)) = self.heap.peek()
                && h < top
            {
                self.heap.pop();
                self.heap.push((h, slot));
                picks.push((r, slot));
            }
        }
        self.seen += seg.n as u64;
        if picks.is_empty() {
            return;
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
    }

    /// Each feature's planted values for `max_bin` bins (module docs).
    pub fn planted(&self, max_bin: usize, pool: &rayon::ThreadPool) -> Vec<Planted> {
        let max_bin = max_bin.max(2);
        pool.install(|| {
            self.kinds
                .par_iter()
                .zip(self.store.par_iter())
                .map(|(kind, col)| match kind {
                    Kind::Category { levels } => Planted::Levels(*levels),
                    _ => Planted::Values(planted_values(&col[..self.filled], max_bin)),
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
            self.push(&seg, pool);
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
            self.push(&seg, pool);
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

    #[test]
    fn the_hash_spreads_consecutive_rows() {
        let below: usize = (0..100_000u64)
            .filter(|&i| hash(i, 0) < u64::MAX / 10)
            .count();
        assert!((9_000..11_000).contains(&below), "{below}");
        assert_ne!(hash(1, 0), hash(1, 1));
    }
}
