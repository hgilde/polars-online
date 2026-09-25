//! Binned target moments for [`crate::Marginal`] (docs/ENHANCEMENTS.md E67,
//! `docs/MARGINAL-LAGS-AND-BINS.md`).
//!
//! Every statistic `marginal` reports is linear. A feature whose relation to
//! the target is a threshold, a V or a saturation has a small `corr` and a
//! large *split gain*: the reduction in the target's variance from cutting
//! the feature at its best threshold, which is what a regression stump
//! reports and the first number a boosted tree looks at.
//!
//! A stump needs only, per feature, the target's weight, mean and centred
//! second moment inside each of a fixed set of bins -- a histogram of target
//! moments, `O(bins)` state per pair and `O(log bins)` per pair per row. So
//! the whole wide input gets a nonlinear relevance number in the pass that
//! gives it `corr`, and the histogram itself is the feature's
//! one-dimensional response curve.
//!
//! # Welford per bin
//!
//! Each bin keeps `(W, mean, M2)` -- its weight, the weighted mean of the
//! target and the weighted sum of squared deviations from that mean -- and
//! not the raw sums `(Σw, Σw·y, Σw·y²)`. The raw form's variance,
//! `Σw·y²/Σw − mean²`, is the difference of two numbers of size `mean²`,
//! and a target that sits at `10⁶` with a spread of `10⁻²` has nothing left
//! after the subtraction: `f64` carries sixteen digits and the two agree in
//! all of them. The pair moments are kept centred for that reason, and a
//! bin's variance is the same statistic over fewer rows, so it gets the same
//! form. The update is West's weighted recurrence, with `W' = W + w`:
//!
//! ```text
//! mean' = mean + (w/W')·(y − mean)
//! M2'   = M2 + w·(y − mean)·(y − mean')
//! ```
//!
//! exact for any sequence of positive weights.
//!
//! # Decay without touching every bin
//!
//! With decay, each bin's moments would have to scale by `lam` every row:
//! `O(bins)` per pair per row, which at `p = 10,000` and 32 bins is ten
//! million multiplies a row. Instead the weights are kept **undecayed**
//! against one scale per group. With `s_t` the product of every decay
//! factor so far, `lam^(t − t_i) = s_t / s_{t_i}`, so
//!
//! ```text
//! W(t) = Σ_i w_i·lam^(t − t_i) = s_t · Σ_i (w_i / s_{t_i})
//! ```
//!
//! and what is stored is that inner sum, into which a row adds `w / s_now`.
//! The mean and `M2` come out right because Welford's recurrence is
//! homogeneous in the weights: run on `w_i / s_{t_i}` it gives the mean of
//! the decayed weights exactly, and the decayed `M2` divided by `s_t`. A
//! read of the weight multiplies by `s_t`; every ratio a caller wants -- a
//! bin mean, a variance, a split gain -- is scale-free and does not even
//! need that.
//!
//! `s` shrinks, so the stored weights grow like `1/s`. When `s` would fall
//! below `1e-150` it is folded into `W` and `M2` and reset to 1:
//! deterministic in the clock, so it happens at the same row however the
//! stream is chunked, and chunk invariance holds across it. A factor of
//! zero -- a clock gap past `max_dclock` under the default `+inf` cap --
//! folds in the same way and leaves every bin empty, which is what the pair
//! moments do on that row.

use serde::{Deserialize, Serialize};

/// The renormalization point: below this the stored weights are approaching
/// the top of `f64`'s range, and a scale that small has already lost
/// nothing.
const RENORM_AT: f64 = 1e-150;

/// A feature whose value falls in no bin on this row (it is not finite).
pub(crate) const NO_BIN: usize = usize::MAX;

/// Where each feature's value fell on the row being learned, as a cell
/// offset within one target's block ([`MarginalBins::update_row`]). A
/// reusable buffer, not state: two histograms with the same cells are the
/// same histogram whatever is left here, and a state file carries none of it.
#[derive(Debug, Clone, Default)]
struct RowBins(Vec<usize>);

impl PartialEq for RowBins {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

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
    /// `[t·off[p] + off[j] + bin]`: the bin's weight, undecayed (see the
    /// module doc). Zero is an empty bin, whatever the other two hold.
    w: Vec<f64>,
    /// The bin's weighted mean of the target: a ratio of two decayed sums,
    /// so scale-free.
    mean: Vec<f64>,
    /// The bin's weighted sum of squared deviations from `mean`, undecayed
    /// like `w`.
    m2: Vec<f64>,
    /// The product of every decay factor applied since the last fold.
    scale: f64,
    /// No bin holds weight: at construction, and after a fold has taken
    /// every weight to zero. Decay is a no-op while this holds (see
    /// [`Self::decay`]), so `scale` is `1` whenever it is set.
    empty: bool,
    /// What each bin's `mean` leaves out: a pair no step is rounded off
    /// ([`crate::comp`]; docs/PLAN.md task 101). A plain mean given one
    /// target value row after row stopped short of it, and left `m2` fed the
    /// gap and the split gain on a ratio of rounding artefacts. Empty in a
    /// state written before it.
    #[serde(default)]
    mean_lo: Vec<f64>,
    /// Row scratch for [`Self::update_row`]. Not part of the state -- serde
    /// skips it and [`PartialEq`] ignores it.
    #[serde(skip)]
    row: RowBins,
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

/// The shape [`MarginalBins::new`] accepts, checked without allocating: one
/// list per feature, each finite and strictly increasing.
fn check_edges(p: usize, edges: &[Vec<f64>]) -> Result<(), String> {
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
    Ok(())
}

impl MarginalBins {
    /// `edges` is one list of interior edges per feature, strictly
    /// increasing and finite; empty is allowed and means one open bin.
    pub fn new(p: usize, n_targets: usize, edges: Vec<Vec<f64>>) -> Result<Self, String> {
        check_edges(p, &edges)?;
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
            mean: vec![0.0; cells],
            m2: vec![0.0; cells],
            scale: 1.0,
            empty: true,
            mean_lo: vec![0.0; cells],
            row: RowBins::default(),
        })
    }

    /// Whether the edges, the offsets and the three cell vectors are those
    /// of `p` features and `n_targets` targets: what a restored state must
    /// hold to be updated (review 2026-09-18, B3).
    pub fn has_shape(&self, p: usize, n_targets: usize) -> bool {
        let cells = self.off.last().map_or(0, |&o| o * n_targets);
        self.p == p
            && self.n_targets == n_targets
            && self.edges.len() == p
            && self.off.len() == p + 1
            && self.w.len() == cells
            && self.mean.len() == cells
            && self.m2.len() == cells
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

    /// Age the histogram by one row's decay. `O(1)`: the moments do not
    /// move.
    ///
    /// Whatever factor the pair moments are aged by, this is aged by too --
    /// no clamp of its own, so the two cannot drift apart. That includes a
    /// factor of **zero**, which the clock does produce (a gap past
    /// `max_dclock` under the default `+inf` cap): the pair moments forget
    /// everything on that row, and so does this. Only a factor that is not a
    /// number in `[0, ∞)` is refused, since it would poison every bin at
    /// once.
    ///
    /// An **empty** histogram is not aged: there is nothing to age, and a
    /// scale it picked up while empty would still divide every later row's
    /// weight and multiply every later read, moving each by a rounding.
    /// The pair moments carry no trace of a zero-weight prefix (their mix
    /// on such a row is `a = 1, b = 0`), and `label_delay`'s doubled stream
    /// begins with one -- so neither does this.
    pub fn decay(&mut self, lam: f64) {
        if self.empty || !lam.is_finite() || lam < 0.0 {
            return;
        }
        let s = self.scale * lam;
        if s >= RENORM_AT {
            self.scale = s;
            return;
        }
        // Fold the scale into the weights before it can underflow -- to a
        // subnormal, where the weights would lose digits, or to zero, where
        // the next row's `w / scale` would poison every bin at once. Two
        // factors rather than their product, since the product is what just
        // proved too small to keep.
        for v in self.w.iter_mut().chain(&mut self.m2) {
            *v = *v * self.scale * lam;
        }
        self.scale = 1.0;
        // A factor of zero wipes every bin; so, more rarely, does a fold
        // whose factor underflows a small weight. Either way the histogram
        // is back where it started, scale included.
        self.empty = self.w.iter().all(|v| *v == 0.0);
    }

    /// Add one row's contribution for target `t` alone, searching every
    /// feature's edges: the form the model used before [`Self::update_row`],
    /// kept for the tests, where it is the oracle the row update is held to.
    #[cfg(test)]
    pub fn update_target(&mut self, t: usize, x: &[f64], y: f64, w: f64) {
        if !w.is_finite() || w <= 0.0 || !y.is_finite() {
            return;
        }
        let u = w / self.scale;
        let block = t * self.off[self.p];
        for (j, xj) in x.iter().enumerate().take(self.p) {
            let Some(b) = Self::bin_of(&self.edges[j], *xj) else {
                continue;
            };
            self.update_cell(block + self.off[j] + b, u, y);
        }
    }

    /// One row of undecayed weight `u` and target value `y` into cell `i`:
    /// the cell update as it was written before [`update_cells`] took
    /// slices, kept for the tests as that kernel's oracle.
    #[cfg(test)]
    fn update_cell(&mut self, i: usize, u: f64, y: f64) {
        let cells = self.w.len();
        self.empty = false;
        if self.w[i] <= 0.0 {
            self.w[i] = u;
            self.mean[i] = y;
            self.m2[i] = 0.0;
            *crate::comp::lo_slot(&mut self.mean_lo, cells, i) = 0.0;
            return;
        }
        let wb = self.w[i] + u;
        let lo = crate::comp::lo_slot(&mut self.mean_lo, cells, i);
        let delta = crate::comp::dev(y, self.mean[i], *lo);
        crate::comp::add(&mut self.mean[i], lo, delta * (u / wb));
        self.m2[i] += u * delta * crate::comp::dev(y, self.mean[i], *lo);
        self.w[i] = wb;
    }

    /// Add one row's contribution for every target it carries: `y[t]` is
    /// target `t`'s value, `None` where the row does not carry it.
    ///
    /// Each feature's bin is found **once per row**, before the targets are
    /// visited: the edges are the feature's, the same for every target, so
    /// a search per (feature, target) found the same bin `T` times (E71,
    /// docs/PLAN.md task 122). The cells are then updated target by target
    /// and feature by feature, the order [`Self::update_target`] takes, so
    /// the histogram is the same to the bit. A row that carries no target
    /// forms no indices.
    pub fn update_row(&mut self, x: &[f64], y: &[Option<f64>], w: f64) {
        let Some(u) = self.row_weight(y, w) else {
            return;
        };
        let mut row = std::mem::take(&mut self.row.0);
        self.bin_offsets(0, x, &mut row);
        let (cells, width) = (self.w.len(), self.off[self.p]);
        // What each mean leaves out, at the length of the cells, which a
        // state written before it does not carry: sized where the row writes
        // a cell, as the cell update sized it before it took slices.
        if self.mean_lo.len() != cells && row.iter().any(|&o| o != NO_BIN) {
            self.mean_lo = vec![0.0; cells];
        }
        for (t, yt) in y.iter().enumerate().take(self.n_targets) {
            let Some(v) = yt.filter(|v| v.is_finite()) else {
                continue;
            };
            let r = t * width..(t + 1) * width;
            if update_cells(
                u,
                v,
                &row,
                &mut self.w[r.clone()],
                &mut self.mean[r.clone()],
                &mut self.m2[r.clone()],
                &mut self.mean_lo[r],
            ) {
                self.empty = false;
            }
        }
        self.row.0 = row;
    }

    /// The undecayed weight a row of weight `w` adds to each cell it
    /// writes, or `None` where it writes none: a weight that is not a
    /// positive number, or no target present.
    pub(crate) fn row_weight(&self, y: &[Option<f64>], w: f64) -> Option<f64> {
        if !w.is_finite() || w <= 0.0 {
            return None;
        }
        let present = |v: &Option<f64>| v.is_some_and(f64::is_finite);
        y.iter()
            .take(self.n_targets)
            .any(present)
            .then(|| w / self.scale)
    }

    /// Where each value of `x`, the row's features from `j0` on, falls: its
    /// bin as an offset into one target's cells counted from feature `j0`'s
    /// first, or [`NO_BIN`]. Each feature's edges are searched once a row,
    /// whatever the number of targets (E71).
    pub(crate) fn bin_offsets(&self, j0: usize, x: &[f64], out: &mut Vec<usize>) {
        bin_offsets_in(&self.edges, &self.off, j0, x, out);
    }

    /// Whether ageing by `lam` would fold the scale into every cell
    /// ([`Self::decay`]): what a caller holding rows for later must apply
    /// first, since they were weighed against the scale before the fold.
    pub(crate) fn folds_at(&self, lam: f64) -> bool {
        !self.empty && lam.is_finite() && lam >= 0.0 && self.scale * lam < RENORM_AT
    }

    /// No bin holds weight ([`Self::decay`] is then a no-op).
    pub(crate) fn is_empty(&self) -> bool {
        self.empty
    }

    /// A row's cells are about to be written by a caller holding them for
    /// later: from here the histogram is not empty, as the cell update
    /// itself would have said.
    pub(crate) fn mark_written(&mut self) {
        self.empty = false;
    }

    /// The cells and what finds them, split so a caller can hand disjoint
    /// ranges of the cells to several writers ([`crate::Marginal`]'s
    /// shards), with `mean_lo` at the cells' length.
    pub(crate) fn cell_parts(&mut self) -> CellParts<'_> {
        let cells = self.w.len();
        if self.mean_lo.len() != cells {
            self.mean_lo = vec![0.0; cells];
        }
        CellParts {
            w: &mut self.w,
            mean: &mut self.mean,
            m2: &mut self.m2,
            mean_lo: &mut self.mean_lo,
            edges: &self.edges,
            off: &self.off,
        }
    }
}

/// [`MarginalBins::cell_parts`]: the four cell vectors, laid out
/// `[t·off[p] + off[j] + bin]`, and the edges and feature offsets that
/// find a value's cell ([`bin_offsets_in`]).
pub(crate) struct CellParts<'a> {
    pub w: &'a mut [f64],
    pub mean: &'a mut [f64],
    pub m2: &'a mut [f64],
    pub mean_lo: &'a mut [f64],
    pub edges: &'a [Vec<f64>],
    pub off: &'a [usize],
}

/// [`MarginalBins::bin_offsets`] from the edges and offsets alone, for a
/// caller that holds the cells apart from them.
pub(crate) fn bin_offsets_in(
    edges: &[Vec<f64>],
    off: &[usize],
    j0: usize,
    x: &[f64],
    out: &mut Vec<usize>,
) {
    let base = off[j0];
    out.clear();
    out.extend(
        edges[j0..]
            .iter()
            .zip(&off[j0..])
            .zip(x)
            .map(|((e, &o), &xj)| MarginalBins::bin_of(e, xj).map_or(NO_BIN, |b| o - base + b)),
    );
}

/// One target's cells for a range of features: each offset in `at` (from
/// [`MarginalBins::bin_offsets`]) names the cell, within the four slices,
/// that takes the value `y` at undecayed weight `u`; [`NO_BIN`] takes
/// nothing. Whether any cell was written. West's weighted recurrence, the
/// module doc's, with the mean a compensated pair ([`crate::comp`]).
#[inline]
pub(crate) fn update_cells(
    u: f64,
    y: f64,
    at: &[usize],
    w: &mut [f64],
    mean: &mut [f64],
    m2: &mut [f64],
    mean_lo: &mut [f64],
) -> bool {
    let mut wrote = false;
    for &o in at {
        if o == NO_BIN {
            continue;
        }
        wrote = true;
        let (wi, mi, lo) = (&mut w[o], &mut mean[o], &mut mean_lo[o]);
        if *wi <= 0.0 {
            // The bin's first row, or its first after a wipe: the mean it
            // holds is nobody's, and `mean + (y − mean)` is not `y` to the
            // bit when the old mean is far away.
            *wi = u;
            *mi = y;
            m2[o] = 0.0;
            *lo = 0.0;
            continue;
        }
        let wb = *wi + u;
        let delta = crate::comp::dev(y, *mi, *lo);
        crate::comp::add(mi, lo, delta * (u / wb));
        m2[o] += u * delta * crate::comp::dev(y, *mi, *lo);
        *wi = wb;
    }
    wrote
}

impl MarginalBins {
    /// The response curve for one pair: the target's moments in each bin.
    pub fn bins(&self, t: usize, j: usize) -> Vec<Bin> {
        self.at(t, j)
            .map(|i| {
                let w = self.w[i];
                if w <= 0.0 {
                    return Bin {
                        n: 0.0,
                        mean_y: f64::NAN,
                        var_y: f64::NAN,
                    };
                }
                Bin {
                    n: w * self.scale,
                    mean_y: self.mean[i],
                    var_y: (self.m2[i] / w).max(0.0),
                }
            })
            .collect()
    }

    /// The best single split, by variance reduction as a fraction of the
    /// target's total variance: `w_L·w_R/W² · (mean_L − mean_R)² / var`,
    /// the stump's gain, computed in the centred form the bins are kept in.
    /// `None` when fewer than two bins carry weight, or the target does not
    /// vary.
    pub fn best_split(&self, t: usize, j: usize) -> Option<Split> {
        let r = self.at(t, j);
        let (w, mean, m2) = (&self.w[r.clone()], &self.mean[r.clone()], &self.m2[r]);
        let total_w: f64 = w.iter().sum();
        if total_w <= 0.0 {
            return None;
        }
        // The grand mean as an offset from the first occupied bin's, so a
        // target far from zero keeps its digits (the module doc's point),
        // then the total centred second moment by the parallel-variance
        // identity `M2 = Σ_b [M2_b + w_b·(mean_b − mean)²]`.
        let m0 = w
            .iter()
            .zip(mean)
            .find(|(wb, _)| **wb > 0.0)
            .map(|(_, m)| *m)?;
        let mean_all = m0
            + w.iter()
                .zip(mean)
                .map(|(wb, mb)| wb * (mb - m0))
                .sum::<f64>()
                / total_w;
        let total_m2: f64 = w
            .iter()
            .zip(mean)
            .zip(m2)
            .map(|((wb, mb), m2b)| {
                let d = mb - mean_all;
                m2b + wb * d * d
            })
            .sum();
        if total_m2 <= 0.0 {
            return None;
        }
        let var = total_m2 / total_w;
        // Cut after bin `c`, so the cut sits at `edges[c]`. The between-group
        // sum of squares of a two-way split is `d_L²·W / (w_L·w_R)`, with
        // `d_L = Σ_{b≤c} w_b·(mean_b − mean)` the left side's total
        // deviation -- the right side's is `−d_L`, since the two sum to
        // zero -- and the gain is that over the total, `W·var`. Taken as a
        // product of ratios: the undecayed weights run to `1e150` times the
        // real ones just before a fold, and three of them multiplied
        // together overflow.
        let (mut left_w, mut left_d) = (0.0, 0.0);
        let mut best: Option<Split> = None;
        for c in 0..self.n_bins(j) - 1 {
            left_w += w[c];
            left_d += w[c] * (mean[c] - mean_all);
            let right_w = total_w - left_w;
            if left_w <= 0.0 || right_w <= 0.0 {
                continue;
            }
            let gain = (left_d / left_w) * (left_d / right_w) / var;
            if best.is_none_or(|s| gain > s.gain) {
                best = Some(Split {
                    gain: gain.clamp(0.0, 1.0),
                    at: self.edges[j][c],
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
    /// Equal weights: the edges are weighted quantiles of the warm-up
    /// sample, so every bin holds about the same weight however the feature
    /// is distributed. The default, and what a tree does. A value the
    /// feature takes on most rows fills one bin on its own and the others
    /// share the rest (see [`edges_from`]).
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
    let bytes = cells.saturating_mul(std::mem::size_of::<f64>());
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
                check_edges(p, e)?;
                let bins: usize = e.iter().map(|f| f.len() + 1).sum();
                budget_check(
                    "the bin histogram",
                    3usize.saturating_mul(n_targets).saturating_mul(bins),
                    &format!("{bins} bins over {p} features x {n_targets} targets x 3"),
                )?;
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
                    self.warm_rows.saturating_mul(p + n_targets),
                    &format!(
                        "bin_warm_rows {} x {p} features + {n_targets} targets",
                        self.warm_rows
                    ),
                )?;
                budget_check(
                    "the bin histogram",
                    3usize
                        .saturating_mul(p)
                        .saturating_mul(n_targets)
                        .saturating_mul(self.n_bins),
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

/// Edges from a warm-up sample of one feature, as `(value, weight)` pairs.
/// Weighted by the row weights -- what a row counts for -- and not by their
/// decay, which says when a row arrived and nothing about what the feature
/// looks like. Strictly increasing and finite, and shorter than asked for
/// when the feature does not have that many distinct values: a binary
/// feature gets one edge, a constant one gets none.
///
/// Under [`BinRule::Quantile`] the edges are placed one at a time, each
/// closing a bin of the weight still unassigned divided among the bins still
/// to come. That is what makes a **point mass** survive: a feature that is
/// zero on 95% of rows has every plain quantile sitting on zero, which
/// collapses to no edge at all, and the 5% that carry the information vanish
/// into the same bin as the zeros. Here the mass fills one bin -- an edge
/// that would close an empty bin, because the value that crosses the share
/// is the first value still unassigned, is moved up to the first value above
/// it -- and the bins that remain share what is left.
pub fn edges_from(rule: BinRule, n_bins: usize, values: &mut Vec<(f64, f64)>) -> Vec<f64> {
    values.retain(|(v, w)| v.is_finite() && w.is_finite() && *w > 0.0);
    if values.len() < 2 || n_bins < 2 {
        return Vec::new();
    }
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let n = values.len();
    let (lo, hi) = (values[0].0, values[n - 1].0);
    let mut edges: Vec<f64> = match rule {
        BinRule::Quantile => {
            let cum: Vec<f64> = values
                .iter()
                .scan(0.0, |acc, (_, w)| {
                    *acc += w;
                    Some(*acc)
                })
                .collect();
            let total = cum[n - 1];
            let mut edges = Vec::with_capacity(n_bins - 1);
            // `start`: the first row not yet closed into a bin.
            let mut start = 0usize;
            for remaining in (2..=n_bins).rev() {
                let base = if start == 0 { 0.0 } else { cum[start - 1] };
                let share = (total - base) / remaining as f64;
                if !share.is_finite() || share <= 0.0 {
                    break;
                }
                // The first unassigned row whose weight crosses the share.
                let i = start + cum[start..].partition_point(|c| c - base <= share);
                if i >= n {
                    break;
                }
                let q = values[i].0;
                let edge = if q > values[start].0 {
                    q
                } else {
                    // A point mass at the front: an edge at its value would
                    // close an empty bin, so the bin takes the whole mass.
                    match values[i..].iter().find(|(v, _)| *v > q) {
                        Some((v, _)) => *v,
                        None => break,
                    }
                };
                edges.push(edge);
                start += values[start..].partition_point(|(v, _)| *v < edge);
            }
            edges
        }
        BinRule::Fixed => {
            let width = (hi - lo) / n_bins as f64;
            (1..n_bins).map(|k| lo + width * k as f64).collect()
        }
    };
    // Ties collapse under `Fixed` when the width rounds away against a
    // large `lo`, and an edge at or below the smallest value seen has nothing
    // beneath it, so it would only ever open an empty bin. The quantile rule
    // produces neither by construction; the sweep is what makes the result
    // acceptable to `MarginalBins::new` whatever the rule.
    edges.dedup();
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

    /// The brute force: every bin's raw sums, decayed in full every row.
    /// Centred at the end, which is exact enough at these magnitudes.
    fn brute(lam: f64, rows: usize, edges: &[f64]) -> (MarginalBins, Vec<[f64; 3]>) {
        let mut b = MarginalBins::new(1, 1, vec![edges.to_vec()]).unwrap();
        let mut sums = vec![[0.0_f64; 3]; edges.len() + 1];
        let mut seed = 7u64;
        for i in 0..rows {
            let x = lcg(&mut seed);
            let y = 3.0 * x + 0.2 * lcg(&mut seed);
            let w = 0.5 + 0.5 * (i % 3) as f64;
            if i > 0 {
                b.decay(lam);
                for s in sums.iter_mut() {
                    for v in s.iter_mut() {
                        *v *= lam;
                    }
                }
            }
            b.update_target(0, &[x], y, w);
            let k = edges.partition_point(|e| *e <= x);
            sums[k][0] += w;
            sums[k][1] += w * y;
            sums[k][2] += w * y * y;
        }
        (b, sums)
    }

    fn assert_matches(b: &MarginalBins, sums: &[[f64; 3]]) {
        for (k, got) in b.bins(0, 0).iter().enumerate() {
            let [w, wy, wyy] = sums[k];
            let mean = wy / w;
            assert!((got.n - w).abs() < 1e-9 * w, "bin {k} weight");
            assert!((got.mean_y - mean).abs() < 1e-9, "bin {k} mean");
            assert!(
                (got.var_y - (wyy / w - mean * mean)).abs() < 1e-9,
                "bin {k} variance"
            );
        }
    }

    /// The scale trick and the per-bin Welford have to give exactly what
    /// decaying every bin every row would: same bins, same means, same
    /// variances.
    #[test]
    fn matches_a_brute_force_decayed_histogram() {
        let (b, sums) = brute(0.97, 500, &[-0.5, 0.0, 0.5]);
        assert_matches(&b, &sums);
    }

    /// Renormalization is a no-op on every number a caller can read. Run
    /// past it -- `lam^rows < 1e-150` needs ~1150 rows at 0.74 -- and compare
    /// against the brute force, which never scales anything.
    #[test]
    fn renormalization_changes_nothing_readable() {
        let (b, sums) = brute(0.74, 2000, &[-0.5, 0.0, 0.5]);
        assert!(
            b.scale > RENORM_AT && b.scale < 1e-100,
            "the test never reached a renormalization, or reached it on the last row: {}",
            b.scale
        );
        assert_matches(&b, &sums);
        let s = b.best_split(0, 0).unwrap();
        assert!(s.gain.is_finite() && (0.0..=1.0).contains(&s.gain));
    }

    /// A factor of zero is a row the pair moments forget everything on, and
    /// the histogram must too: every bin empty, no split, and the next row
    /// starts each bin over. Before this was held to, `decay` refused the
    /// factor and the bins reported a regime the pairs had already dropped.
    #[test]
    fn a_total_gap_empties_the_histogram() {
        let (mut b, _) = brute(0.97, 200, &[0.0]);
        assert!(b.bins(0, 0).iter().all(|bin| bin.n > 0.0));
        b.decay(0.0);
        assert!(
            b.bins(0, 0).iter().all(|bin| bin.n == 0.0),
            "{:?}",
            b.bins(0, 0)
        );
        assert!(b.best_split(0, 0).is_none());
        b.update_target(0, &[0.5], 2.0, 3.0);
        let bins = b.bins(0, 0);
        assert_eq!(bins[0].n, 0.0);
        assert_eq!((bins[1].n, bins[1].mean_y, bins[1].var_y), (3.0, 2.0, 0.0));
    }

    /// The scale folds before it can underflow. Two factors of `1e-200` take
    /// the product of the old code's scale to zero, after which every row
    /// was refused for good; here the second fold takes the old weights to
    /// zero -- honestly, they *have* decayed to nothing -- and the rows that
    /// follow count in full.
    #[test]
    fn the_scale_cannot_underflow_to_zero() {
        let (mut b, _) = brute(0.97, 50, &[0.0]);
        b.decay(1e-200);
        b.decay(1e-200);
        assert_eq!(b.scale, 1.0);
        b.update_target(0, &[-0.5], 1.0, 2.0);
        b.update_target(0, &[-0.5], 3.0, 2.0);
        let bins = b.bins(0, 0);
        assert_eq!((bins[0].n, bins[0].mean_y, bins[0].var_y), (4.0, 2.0, 1.0));
        assert_eq!(bins[1].n, 0.0);
    }

    /// Decay leaves no trace on an empty histogram. The doubled stream
    /// `label_delay` is checked against opens with `delay` zero-weight rows,
    /// and the native path with none; a scale picked up there would round
    /// every later weight differently, and the two stopped agreeing in the
    /// last bit until this held. Also the reason the flag is a field:
    /// `decay` is `O(1)` and may not look at the bins to find out.
    #[test]
    fn decay_leaves_no_trace_on_an_empty_histogram() {
        let mut aged = MarginalBins::new(2, 1, vec![vec![-0.3, 0.3], vec![0.0]]).unwrap();
        let mut fresh = aged.clone();
        for _ in 0..7 {
            aged.decay(0.5);
        }
        assert_eq!(aged, fresh, "an empty histogram is not aged");
        let mut seed = 11u64;
        for _ in 0..300 {
            let x = [lcg(&mut seed), lcg(&mut seed)];
            let y = lcg(&mut seed);
            for b in [&mut aged, &mut fresh] {
                b.decay(0.93);
                b.update_target(0, &x, y, 1.0);
            }
        }
        assert_eq!(aged, fresh);
        // And back to empty through a wipe: the scale is `1` again and the
        // next decay is a no-op again.
        aged.decay(0.0);
        assert!(aged.empty && aged.scale == 1.0);
        aged.decay(0.5);
        assert_eq!(aged.scale, 1.0);
    }

    /// The reason the bins are Welford and not raw sums: a target at `10⁶`
    /// with a spread of `10⁻³`. The raw form has `Σw·y²/Σw − mean²` cancel
    /// to noise of order `1e-4`, a hundred times the true variance.
    #[test]
    fn a_far_offset_keeps_the_variance() {
        let mut b = MarginalBins::new(1, 1, vec![vec![0.0]]).unwrap();
        let mut seed = 5u64;
        let mut ys = Vec::new();
        for _ in 0..1000 {
            let y = 1e6 + 1e-3 * lcg(&mut seed);
            b.decay(0.999);
            b.update_target(0, &[-1.0], y, 1.0);
            ys.push(y);
        }
        // The undecayed reference is close enough at lam = 0.999 over 1000
        // rows to say whether the digits survived; the brute-force test
        // above says the decay is exact.
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / ys.len() as f64;
        let got = b.bins(0, 0)[0];
        assert!(
            (got.var_y / var - 1.0).abs() < 0.2,
            "{} vs {var}",
            got.var_y
        );
        assert!((got.mean_y - mean).abs() < 1e-3);
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
        b.decay(f64::NAN);
        b.decay(-1.0);
        b.decay(f64::INFINITY);
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

    /// The budget applies to explicit edges as it does to learned ones, and
    /// is checked before anything the size of the histogram is allocated.
    #[test]
    fn explicit_edges_are_budgeted_too() {
        let cfg = BinCfg {
            n_bins: 0,
            edges: Some(vec![vec![0.0, 1.0, 2.0]]),
            rule: BinRule::Quantile,
            warm_rows: 0,
        };
        assert!(cfg.validate(1, 1).is_ok());
        let err = cfg.validate(1, 100_000_000).unwrap_err();
        assert!(err.contains("budget"), "{err}");
        let err = cfg.validate(2, 1).unwrap_err();
        assert!(err.contains("one edge list per feature"), "{err}");
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

    fn sample(values: &[f64]) -> Vec<(f64, f64)> {
        values.iter().map(|v| (*v, 1.0)).collect()
    }

    /// The quantile rule on a sample without ties is the plain one: the
    /// `k·n/n_bins`-th sorted value, and a value on an edge goes above it.
    #[test]
    fn quantile_edges_without_ties() {
        let mut v = sample(&[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]);
        assert_eq!(
            edges_from(BinRule::Quantile, 4, &mut v),
            vec![3.0, 5.0, 7.0]
        );
        assert_eq!(edges_from(BinRule::Quantile, 2, &mut v), vec![5.0]);
        assert_eq!(
            edges_from(BinRule::Fixed, 4, &mut sample(&[0.0, 4.0])),
            vec![1.0, 2.0, 3.0]
        );
    }

    /// A point mass takes one bin and the rest share what is left. Under a
    /// plain quantile rule every edge of a 95%-zeros feature sits on zero,
    /// which collapses to no edge, and the 5% that carry the signal are
    /// binned with the zeros.
    #[test]
    fn quantile_edges_keep_a_point_mass_in_its_own_bin() {
        let mut rare: Vec<(f64, f64)> = (0..100).map(|i| (f64::from(i >= 95), 1.0)).collect();
        assert_eq!(edges_from(BinRule::Quantile, 8, &mut rare), vec![1.0]);
        let mut mid: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                (
                    if i < 25 {
                        -1.0
                    } else if i < 75 {
                        0.0
                    } else {
                        1.0
                    },
                    1.0,
                )
            })
            .collect();
        assert_eq!(edges_from(BinRule::Quantile, 4, &mut mid), vec![0.0, 1.0]);
        // Mass in the middle with distinct values on both sides: the bins
        // on either side re-equalize over what is left.
        let mut spread: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let v = if (40..60).contains(&i) {
                    0.0
                } else {
                    i as f64 - 50.0
                };
                (v, 1.0)
            })
            .collect();
        let e = edges_from(BinRule::Quantile, 5, &mut spread);
        assert!(
            e.contains(&0.0) && e.iter().any(|v| *v > 0.0) && e.iter().any(|v| *v < 0.0),
            "{e:?}"
        );
        assert!(e.windows(2).all(|w| w[1] > w[0]));
    }

    /// Row weights count: one row that weighs as much as the other seven
    /// pulls the median up to it.
    #[test]
    fn quantile_edges_are_weighted() {
        let mut v = sample(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(edges_from(BinRule::Quantile, 2, &mut v), vec![5.0]);
        v[7].1 = 7.0;
        assert_eq!(edges_from(BinRule::Quantile, 2, &mut v), vec![8.0]);
    }

    /// Fewer distinct values than bins gives fewer edges, never a repeated
    /// or a useless one: a binary feature has one edge and a constant
    /// feature none, under either rule.
    #[test]
    fn degenerate_samples_give_the_edges_they_can() {
        let mut binary = sample(&[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(edges_from(BinRule::Quantile, 8, &mut binary), vec![1.0]);
        assert_eq!(
            edges_from(BinRule::Fixed, 4, &mut binary),
            vec![0.25, 0.5, 0.75]
        );
        let mut constant = sample(&[7.0; 20]);
        assert_eq!(
            edges_from(BinRule::Quantile, 8, &mut constant),
            Vec::<f64>::new()
        );
        assert_eq!(
            edges_from(BinRule::Fixed, 8, &mut constant),
            Vec::<f64>::new()
        );
        let mut junk = vec![(f64::NAN, 1.0), (1.0, f64::NAN), (2.0, 0.0), (3.0, 1.0)];
        assert_eq!(
            edges_from(BinRule::Quantile, 2, &mut junk),
            Vec::<f64>::new()
        );
        assert_eq!(
            edges_from(BinRule::Quantile, 1, &mut sample(&[1.0, 2.0])),
            Vec::<f64>::new()
        );
    }

    /// Every number a histogram holds, as bits: a comparison to the bit, and
    /// one that a `-0.0` or a NaN cannot pass by accident.
    fn bits(b: &MarginalBins) -> Vec<u64> {
        b.w.iter()
            .chain(&b.mean)
            .chain(&b.m2)
            .chain(&b.mean_lo)
            .chain(std::iter::once(&b.scale))
            .map(|v| v.to_bits())
            .chain(std::iter::once(u64::from(b.empty)))
            .collect()
    }

    /// E71 (docs/PLAN.md task 122): a row that finds each feature's bin once
    /// and then visits its targets is, to the bit, the same row fed one
    /// target at a time with a search per feature. Ragged edges (one feature
    /// has none), targets absent on a random third of the rows or not
    /// finite, features that are not finite, rows of weight zero, and a
    /// decay of zero that wipes every bin, so each path a cell can take is
    /// taken by both.
    #[test]
    fn the_row_update_is_the_per_target_update_to_the_bit() {
        let edges = vec![
            vec![-0.5, 0.0, 0.5],
            vec![],
            vec![-0.9, -0.1, 0.2, 0.3, 0.8],
            vec![0.0],
        ];
        let (p, n_targets) = (edges.len(), 5);
        for absent in [0.0, 1.0 / 3.0] {
            let mut by_row = MarginalBins::new(p, n_targets, edges.clone()).unwrap();
            let mut by_target = by_row.clone();
            let mut seed = 21u64;
            let mut wiped = false;
            for i in 0..3000usize {
                let lam = match i {
                    0 => 1.0,
                    1500 => 0.0,
                    _ => 0.99,
                };
                let x: Vec<f64> = (0..p)
                    .map(|j| {
                        if i % 97 == j {
                            f64::NAN
                        } else {
                            lcg(&mut seed)
                        }
                    })
                    .collect();
                let y: Vec<Option<f64>> = (0..n_targets)
                    .map(|t| {
                        let v = x[0].abs() * t as f64 + 0.1 * lcg(&mut seed);
                        if (lcg(&mut seed) + 1.0) / 2.0 < absent {
                            None
                        } else if i % 211 == t {
                            Some(f64::INFINITY)
                        } else {
                            Some(v)
                        }
                    })
                    .collect();
                let w = if i % 13 == 0 {
                    0.0
                } else {
                    0.5 + (lcg(&mut seed) + 1.0) / 2.0
                };
                by_row.decay(lam);
                by_target.decay(lam);
                wiped |= by_row.empty && i > 0;
                by_row.update_row(&x, &y, w);
                for (t, yt) in y.iter().enumerate() {
                    if let Some(v) = yt {
                        by_target.update_target(t, &x, *v, w);
                    }
                }
                assert_eq!(bits(&by_row), bits(&by_target), "absent {absent}, row {i}");
            }
            assert!(wiped, "the decay of zero must have emptied the histogram");
            // Every target's every feature has cells with weight: the
            // comparison above was of histograms that learned something.
            for t in 0..n_targets {
                for j in 0..p {
                    let held = by_row.bins(t, j).iter().filter(|b| b.n > 0.0).count();
                    assert!(
                        held > 0,
                        "absent {absent}: target {t} feature {j} learned nothing"
                    );
                }
            }
        }
    }
}
