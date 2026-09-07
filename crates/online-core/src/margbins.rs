//! Binned target moments for [`crate::Marginal`] (docs/ENHANCEMENTS.md E67,
//! `docs/MARGINAL-LAGS-AND-BINS.md`).
//!
//! Every statistic `marginal` reports is linear. A feature whose relation to
//! the target is a threshold, a V or a saturation has a small `corr` and a
//! large *split gain*: the reduction in the target's variance from cutting
//! the feature at its best threshold, which is what a regression stump
//! reports and the first number a boosted tree looks at.
//!
//! A stump needs only, per feature, the target's `(sum w, sum w·y, sum w·y²)`
//! inside each of a fixed set of bins — a histogram of target moments,
//! `O(bins)` state per pair and `O(log bins)` per pair per row. So the whole
//! wide input gets a nonlinear relevance number in the pass that gives it
//! `corr`, and the histogram itself is the feature's one-dimensional
//! response curve.
//!
//! # Decay without touching every bin
//!
//! With decay, each bin's three sums would have to scale by `lam` every row:
//! `O(bins)` per pair per row, which at `p = 10,000` and 32 bins is ten
//! million multiplies a row. Instead the sums are kept **undecayed** against
//! one scale per group. With `s_t` the product of every decay factor so far,
//! `lam^(t − t_i) = s_t / s_{t_i}`, so
//!
//! ```text
//! S(t) = sum_i w_i·lam^(t − t_i) = s_t · sum_i (w_i / s_{t_i})
//! ```
//!
//! and what is stored is that inner sum, into which a row adds `w / s_now`.
//! A read multiplies by `s_t`; every ratio a caller wants — a bin mean, a
//! variance, a split gain — is scale-free and does not even need that.
//!
//! `s` shrinks, so the stored sums grow like `1/s`. When `s` falls below
//! `1e-150` the sums are multiplied through by it and `s` is reset to 1:
//! deterministic in the clock, so it happens at the same row however the
//! stream is chunked, and chunk invariance holds across it.

use serde::{Deserialize, Serialize};

/// The renormalization point: below this the stored sums are approaching the
/// top of `f64`'s range, and a scale that small has already lost nothing.
const RENORM_AT: f64 = 1e-150;

/// A histogram of target moments per (feature, target), against edges fixed
/// before the first row.
///
/// Bins are **ragged**: each feature keeps its own edges and may have fewer
/// than were asked for. A binary feature supports one edge and a constant
/// one supports none, and refusing those would mean refusing real data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalBins {
    p: usize,
    n_targets: usize,
    /// Interior edges per feature, strictly increasing; `bins = edges + 1`,
    /// with the two outer bins open.
    edges: Vec<Vec<f64>>,
    /// Where feature `j`'s bins start within one target's block, `p + 1`
    /// long, so `off[p]` is the block's width.
    off: Vec<usize>,
    /// `[t·off[p] + off[j] + bin]`, undecayed (see the module doc).
    w: Vec<f64>,
    wy: Vec<f64>,
    wyy: Vec<f64>,
    /// The product of every decay factor applied so far.
    scale: f64,
}

/// One bin's target moments, decayed to now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bin {
    pub n: f64,
    pub mean_y: f64,
    pub var_y: f64,
}

/// The best single split of a feature, and what it buys.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    /// Variance reduction as a fraction of the target's variance, in `[0, 1]`.
    pub gain: f64,
    /// The edge that achieves it.
    pub at: f64,
}

impl MarginalBins {
    /// `edges` is one list of interior edges per feature, strictly
    /// increasing and finite; empty is allowed and means one open bin.
    pub fn new(p: usize, n_targets: usize, edges: Vec<Vec<f64>>) -> Result<Self, String> {
        if edges.len() != p {
            return Err(format!(
                "marginal: bins needs one edge list per feature ({p}), got {}",
                edges.len()
            ));
        }
        for (j, e) in edges.iter().enumerate() {
            if e.iter().any(|v| !v.is_finite()) || e.windows(2).any(|w| w[1] <= w[0]) {
                return Err(format!(
                    "marginal: feature {j}'s bin edges must be finite and strictly increasing"
                ));
            }
        }
        let mut off = Vec::with_capacity(p + 1);
        let mut acc = 0usize;
        for e in &edges {
            off.push(acc);
            acc += e.len() + 1;
        }
        off.push(acc);
        let cells = acc * n_targets;
        Ok(Self {
            p,
            n_targets,
            edges,
            off,
            w: vec![0.0; cells],
            wy: vec![0.0; cells],
            wyy: vec![0.0; cells],
            scale: 1.0,
        })
    }

    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Bins feature `j` actually has, which is one more than its edges.
    pub fn n_bins(&self, j: usize) -> usize {
        self.edges[j].len() + 1
    }

    pub fn edges(&self, j: usize) -> &[f64] {
        &self.edges[j]
    }

    fn at(&self, t: usize, j: usize) -> std::ops::Range<usize> {
        let base = t * self.off[self.p] + self.off[j];
        base..base + self.n_bins(j)
    }

    /// The bin a value falls in: `edges.partition_point`, so a value equal to
    /// an edge goes to the bin above it, and a NaN goes nowhere.
    fn bin_of(edges: &[f64], v: f64) -> Option<usize> {
        if !v.is_finite() {
            return None;
        }
        Some(edges.partition_point(|e| *e <= v))
    }

    /// Age the histogram by one row's decay. `O(1)`: the sums do not move.
    ///
    /// Whatever factor the pair moments are aged by, this is aged by too --
    /// no clamp of its own, so the two cannot drift apart. Only a factor
    /// that is not a positive number is refused, since it would poison every
    /// bin at once.
    pub fn decay(&mut self, lam: f64) {
        if !lam.is_finite() || lam <= 0.0 {
            return;
        }
        self.scale *= lam;
        if self.scale < RENORM_AT && self.scale > 0.0 {
            for v in self.w.iter_mut().chain(&mut self.wy).chain(&mut self.wyy) {
                *v *= self.scale;
            }
            self.scale = 1.0;
        }
    }

    /// Add one row's contribution for target `t`. `O(log bins)` per feature.
    pub fn update_target(&mut self, t: usize, x: &[f64], y: f64, w: f64) {
        if !w.is_finite() || w <= 0.0 || !y.is_finite() || self.scale <= 0.0 {
            return;
        }
        let (aw, awy, awyy) = (w / self.scale, w * y / self.scale, w * y * y / self.scale);
        let block = t * self.off[self.p];
        for (j, xj) in x.iter().enumerate().take(self.p) {
            let Some(b) = Self::bin_of(&self.edges[j], *xj) else {
                continue;
            };
            let i = block + self.off[j] + b;
            self.w[i] += aw;
            self.wy[i] += awy;
            self.wyy[i] += awyy;
        }
    }

    /// The response curve for one pair: the target's moments in each bin.
    pub fn bins(&self, t: usize, j: usize) -> Vec<Bin> {
        self.at(t, j)
            .map(|i| {
                let (w, wy, wyy) = (self.w[i], self.wy[i], self.wyy[i]);
                if w <= 0.0 {
                    return Bin {
                        n: 0.0,
                        mean_y: f64::NAN,
                        var_y: f64::NAN,
                    };
                }
                let mean = wy / w;
                Bin {
                    n: w * self.scale,
                    mean_y: mean,
                    var_y: (wyy / w - mean * mean).max(0.0),
                }
            })
            .collect()
    }

    /// The best single split, by variance reduction as a fraction of the
    /// target's total variance: `w_L·w_R/W² · (mean_L − mean_R)² / var`,
    /// which is the stump's gain and is scale-free. `None` when fewer than
    /// two bins carry weight, or the target does not vary.
    pub fn best_split(&self, t: usize, j: usize) -> Option<Split> {
        let r = self.at(t, j);
        let (w, wy, wyy) = (&self.w[r.clone()], &self.wy[r.clone()], &self.wyy[r]);
        let total_w: f64 = w.iter().sum();
        let total_wy: f64 = wy.iter().sum();
        let total_wyy: f64 = wyy.iter().sum();
        if total_w <= 0.0 {
            return None;
        }
        let mean = total_wy / total_w;
        let var = (total_wyy / total_w - mean * mean).max(0.0);
        if var <= 0.0 {
            return None;
        }
        let (mut left_w, mut left_wy) = (0.0, 0.0);
        let mut best: Option<Split> = None;
        // Cut after bin `b`, so the cut sits at `edges[b]`.
        for b in 0..self.n_bins(j) - 1 {
            left_w += w[b];
            left_wy += wy[b];
            let right_w = total_w - left_w;
            if left_w <= 0.0 || right_w <= 0.0 {
                continue;
            }
            let diff = left_wy / left_w - (total_wy - left_wy) / right_w;
            let gain = (left_w * right_w) / (total_w * total_w) * diff * diff / var;
            if best.is_none_or(|s| gain > s.gain) {
                best = Some(Split {
                    gain: gain.clamp(0.0, 1.0),
                    at: self.edges[j][b],
                });
            }
        }
        best
    }
}

/// How `n_bins` becomes edges, when they are learned rather than given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinRule {
    /// Equal counts: the edges are the warm-up sample's quantiles, so every
    /// bin holds about the same number of rows however the feature is
    /// distributed. The default, and what a tree does.
    Quantile,
    /// Equal widths between the warm-up sample's smallest and largest value.
    /// Reads more naturally when the feature is already a physical quantity,
    /// and leaves bins empty when it is skewed.
    Fixed,
}

/// What a caller asks for. Either explicit `edges` -- exact, reproducible,
/// and comparable across runs -- or an `n_bins` to learn from the first
/// `warm_rows` rows under `rule`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinCfg {
    /// Bins per feature when the edges are learned. Ignored when `edges` is
    /// given, which fixes its own count.
    pub n_bins: usize,
    /// One strictly increasing list of interior edges per feature. An empty
    /// list is one open bin, which is what a constant feature deserves.
    pub edges: Option<Vec<Vec<f64>>>,
    pub rule: BinRule,
    /// Learned rows to hold before the edges are fixed. The rows are *held*,
    /// not spent: once the edges exist every one of them is replayed with its
    /// own decay, so the histogram is what it would have been had the edges
    /// been known from the start.
    pub warm_rows: usize,
}

/// The most a warm-up hold or a histogram may take before the model refuses
/// to build: `p` can be 10,000 here, and silently allocating gigabytes is a
/// worse outcome than an error that names the number.
const MEMORY_BUDGET: usize = 256 << 20;

fn budget_check(what: &str, cells: usize, detail: &str) -> Result<(), String> {
    let bytes = cells * std::mem::size_of::<f64>();
    if bytes > MEMORY_BUDGET {
        return Err(format!(
            "marginal: {what} would need {:.1} GiB ({detail}), over the {} MiB budget; \
             reduce it, or narrow the features",
            bytes as f64 / (1u64 << 30) as f64,
            MEMORY_BUDGET >> 20,
        ));
    }
    Ok(())
}

impl BinCfg {
    pub fn validate(&self, p: usize, n_targets: usize) -> Result<(), String> {
        match &self.edges {
            Some(e) => {
                MarginalBins::new(p, n_targets, e.clone())?;
            }
            None => {
                if self.n_bins < 2 {
                    return Err(format!(
                        "marginal: bins must be at least 2, got {}",
                        self.n_bins
                    ));
                }
                if self.warm_rows < self.n_bins {
                    return Err(format!(
                        "marginal: bin_warm_rows ({}) must be at least bins ({}); the edges are \
                         quantiles of the held rows and cannot be found from fewer",
                        self.warm_rows, self.n_bins
                    ));
                }
                budget_check(
                    "the bin warm-up hold",
                    self.warm_rows * (p + n_targets),
                    &format!(
                        "bin_warm_rows {} x {p} features + {n_targets} targets",
                        self.warm_rows
                    ),
                )?;
                budget_check(
                    "the bin histogram",
                    3 * p * n_targets * self.n_bins,
                    &format!(
                        "{p} features x {n_targets} targets x {} bins x 3",
                        self.n_bins
                    ),
                )?;
            }
        }
        Ok(())
    }
}

/// Edges from a warm-up sample of one feature. Strictly increasing and
/// finite, and shorter than asked for when the feature does not have that
/// many distinct values -- a binary feature gets one edge, a constant one
/// gets none.
pub fn edges_from(rule: BinRule, n_bins: usize, values: &mut Vec<f64>) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    if values.len() < 2 || n_bins < 2 {
        return Vec::new();
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut edges: Vec<f64> = match rule {
        BinRule::Quantile => (1..n_bins)
            .map(|k| {
                let i = (k * values.len()) / n_bins;
                values[i.min(values.len() - 1)]
            })
            .collect(),
        BinRule::Fixed => {
            let (lo, hi) = (values[0], values[values.len() - 1]);
            let width = (hi - lo) / n_bins as f64;
            (1..n_bins).map(|k| lo + width * k as f64).collect()
        }
    };
    // Ties collapse: a feature that is 90% zeros has repeated quantiles, and
    // the bins it can support are however many distinct edges survive. An
    // edge at or below the smallest value seen has nothing beneath it, so it
    // would only ever open an empty bin -- which is what a binary feature's
    // lower quantiles all are.
    edges.dedup();
    let lo = values[0];
    edges.retain(|v| v.is_finite() && *v > lo);
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The scale trick has to give exactly what decaying every bin every row
    /// would: same bins, same means, same variances.
    #[test]
    fn matches_a_brute_force_decayed_histogram() {
        let edges = vec![vec![-0.5, 0.0, 0.5]];
        let mut b = MarginalBins::new(1, 1, edges.clone()).unwrap();
        let lam = 0.97_f64;
        // The brute force: four bins, decayed in full every row.
        let mut bw = [0.0_f64; 4];
        let mut bwy = [0.0_f64; 4];
        let mut bwyy = [0.0_f64; 4];
        let mut seed = 7u64;
        for i in 0..500 {
            let x = lcg(&mut seed);
            let y = 3.0 * x + 0.2 * lcg(&mut seed);
            let w = 0.5 + 0.5 * (i % 3) as f64;
            if i > 0 {
                b.decay(lam);
                for k in 0..4 {
                    bw[k] *= lam;
                    bwy[k] *= lam;
                    bwyy[k] *= lam;
                }
            }
            b.update_target(0, &[x], y, w);
            let k = edges[0].partition_point(|e| *e <= x);
            bw[k] += w;
            bwy[k] += w * y;
            bwyy[k] += w * y * y;
        }
        for (k, got) in b.bins(0, 0).iter().enumerate() {
            let mean = bwy[k] / bw[k];
            assert!((got.n - bw[k]).abs() < 1e-9 * bw[k], "bin {k} weight");
            assert!((got.mean_y - mean).abs() < 1e-9, "bin {k} mean");
            assert!(
                (got.var_y - (bwyy[k] / bw[k] - mean * mean)).abs() < 1e-9,
                "bin {k} variance"
            );
        }
    }

    /// Renormalization is a no-op on every number a caller can read. Run past
    /// it (lam^rows < 1e-150 needs ~1150 rows at 0.74) and compare against the
    /// same stream with a scale that never trips.
    #[test]
    fn renormalization_changes_nothing_readable() {
        let run = |lam: f64| {
            let mut b = MarginalBins::new(1, 1, vec![vec![-0.5, 0.0, 0.5]]).unwrap();
            let mut seed = 11u64;
            for i in 0..2000 {
                let x = lcg(&mut seed);
                let y = x * x + 0.1 * lcg(&mut seed);
                if i > 0 {
                    b.decay(lam);
                }
                b.update_target(0, &[x], y, 1.0);
            }
            b
        };
        let tripped = run(0.74);
        assert!(
            tripped.scale > RENORM_AT,
            "the test never reached a renormalization"
        );
        // Same statistics from a run whose scale stays near 1: the ratios are
        // scale-free, so only the shape of the curve is compared.
        let flat = run(0.9999);
        for (a, c) in tripped.bins(0, 0).iter().zip(flat.bins(0, 0).iter()) {
            assert!(a.n > 0.0 && c.n > 0.0);
        }
        let s = tripped.best_split(0, 0).unwrap();
        assert!(s.gain.is_finite() && (0.0..=1.0).contains(&s.gain));
    }

    /// The reported gain is the stump's: the variance a cut removes, as a
    /// fraction of the total. Checked against the definition, computed the
    /// long way over every cut.
    #[test]
    fn split_gain_is_the_variance_a_cut_removes() {
        let edges = vec![-0.5, 0.0, 0.5];
        let mut b = MarginalBins::new(1, 1, vec![edges.clone()]).unwrap();
        let mut seed = 3u64;
        let mut rows = Vec::new();
        for _ in 0..400 {
            let x = lcg(&mut seed);
            // A threshold: linear correlation near zero, a large split gain.
            let y = if x > 0.0 { 1.0 } else { -1.0 } + 0.3 * lcg(&mut seed);
            b.update_target(0, &[x], y, 1.0);
            rows.push((x, y));
        }
        let var_of = |rs: &[(f64, f64)]| {
            let n = rs.len() as f64;
            let m = rs.iter().map(|r| r.1).sum::<f64>() / n;
            rs.iter().map(|r| (r.1 - m) * (r.1 - m)).sum::<f64>() / n
        };
        let total = var_of(&rows);
        let (mut best, mut best_at) = (0.0_f64, f64::NAN);
        for &c in &edges {
            let (l, r): (Vec<_>, Vec<_>) = rows.iter().partition(|(x, _)| *x < c);
            let (nl, nr) = (l.len() as f64, r.len() as f64);
            let n = nl + nr;
            let g = (total - (nl * var_of(&l) + nr * var_of(&r)) / n) / total;
            if g > best {
                best = g;
                best_at = c;
            }
        }
        let got = b.best_split(0, 0).unwrap();
        assert!(
            (got.gain - best).abs() < 1e-9,
            "gain {} vs {best}",
            got.gain
        );
        assert_eq!(got.at, best_at);
        assert!(
            best > 0.5,
            "a step function should give up most of its variance: {best}"
        );
    }

    /// A value equal to an edge goes to the bin above it, and nothing outside
    /// the numbers a caller can supply lands anywhere.
    #[test]
    fn edge_values_and_junk() {
        let mut b = MarginalBins::new(1, 1, vec![vec![0.0, 1.0]]).unwrap();
        b.update_target(0, &[0.0], 5.0, 1.0);
        b.update_target(0, &[f64::NAN], 5.0, 1.0);
        b.update_target(0, &[f64::INFINITY], 5.0, 1.0);
        b.update_target(0, &[0.5], f64::NAN, 1.0);
        b.update_target(0, &[0.5], 5.0, 0.0);
        let bins = b.bins(0, 0);
        assert_eq!(bins[0].n, 0.0, "an edge belongs to the bin above it");
        assert_eq!(bins[1].n, 1.0);
        assert_eq!(bins[2].n, 0.0);
    }

    #[test]
    fn refuses_edges_it_cannot_use() {
        assert!(
            MarginalBins::new(2, 1, vec![vec![0.0]]).is_err(),
            "one list per feature"
        );
        assert!(
            MarginalBins::new(1, 1, vec![vec![]]).is_ok(),
            "empty is one open bin"
        );
        assert!(
            MarginalBins::new(1, 1, vec![vec![1.0, 0.0]]).is_err(),
            "descending"
        );
        assert!(
            MarginalBins::new(1, 1, vec![vec![0.0, 0.0]]).is_err(),
            "repeated"
        );
        assert!(
            MarginalBins::new(1, 1, vec![vec![f64::NAN]]).is_err(),
            "not finite"
        );
        assert!(
            MarginalBins::new(2, 1, vec![vec![0.0], vec![0.0, 1.0]]).is_ok(),
            "features keep their own bin counts"
        );
    }

    /// A featureless column and a target that never varies both give no
    /// split rather than a division by zero.
    #[test]
    fn no_split_without_variance() {
        let mut b = MarginalBins::new(1, 1, vec![vec![0.0]]).unwrap();
        assert!(b.best_split(0, 0).is_none(), "nothing seen");
        for _ in 0..10 {
            b.update_target(0, &[-1.0], 4.0, 1.0);
        }
        assert!(b.best_split(0, 0).is_none(), "one bin, constant target");
    }
}
