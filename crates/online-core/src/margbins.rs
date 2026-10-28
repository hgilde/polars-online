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
//! moments, `O(bins)` state per pair, one search of the edges per feature
//! per row and `O(1)` per pair. So
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
//! below `1e-50` it is folded into `W` and `M2` and reset to 1 -- early
//! enough that `M2`, which grows like the weights times the target's spread
//! squared, stays finite for a spread at the input bound (`RENORM_AT`):
//! deterministic in the clock, so it happens at the same row however the
//! stream is chunked, and chunk invariance holds across it. A factor of
//! zero -- a clock gap past `gap_cap` under the default `+inf` cap --
//! folds in the same way and leaves every bin empty, which is what the pair
//! moments do on that row.
//!
//! # A robust location per bin
//!
//! Under [`BinStat::Huber`] a bin's location is Huber's M-estimate of
//! location of its decayed sample, learned in one pass (docs/HUBER.md §8,
//! request 4; docs/PLAN.md task 272 (c)). With `σ_t` the target's robust
//! scale and `c = δ·σ_t` the cut, both read before the row, a row of weight
//! `w` and deviation `e = y − m` from the bin's location before it is
//! weighed as `huber`'s regression weighs a residual, and the location is
//! the weighted mean over those weights:
//!
//! ```text
//! ω  = w·min(1, c/|e|)            Ω' = λ·Ω + ω       W' = λ·W + w
//! m' = m + (ω/Ω')·e = m + (w/Ω')·σ_t·ψ_δ(e/σ_t)      ψ_δ(u) = max(−δ, min(δ, u))
//! M2' = λ·M2 + w·ψ²·σ_t²·(λ·Ω/Ω')       (Welford's `w·e·(y − m')` where unclipped)
//! ```
//!
//! Its fixed point is the root of `Σ ω_i·(y_i − m) = 0`, that is of
//! `Σ w_i·ψ_δ((y_i − m)/σ)`: the Huber M-location of the bin's decayed
//! sample. A row's step is bounded by `c·w/Ω'`, so one spike moves a thin
//! bin by at most `δ·σ_t` times its share where it moves the mean by its
//! whole size times its share. The step is divided by `Ω`, the weight the
//! rows count for, and not by `W`, the weight they carry: a location that
//! starts far from its bin's sample -- its first row a spike `E₀` off at
//! full weight `Ω₀`, which happens only before the target has a scale, as
//! a later first row is winsorized at `m_t ± c` -- then recovers
//! exponentially: while every row is clipped, `ω = w·c/E`, so `Ω·E` stays
//! `Ω₀E₀` and `E(n) = E₀·exp(−c·n/(Ω₀E₀))`, within one scale `σ` after about
//! `(Ω₀E₀/c)·ln(E₀/σ)` rows. A step over `W` moves at most `c·w/W` a row and
//! recovers only `c·ln(rows)`. Simulated (no decay, unit weights, `σ`
//! fixed; docs/PLAN.md task 272 (e), R6-2): spikes of 10, 30 and 100 scales
//! were within one scale after 27, 101 and 381 rows over `Ω`; over `W`,
//! 0.14, 17 and 87 scales off after 20,000 rows.
//!
//! A bin has no location of its own to judge its first row against, so it
//! judges it against the target's: `m = m_t + max(−c, min(c, y − m_t))`,
//! the row winsorized at the cut around the target's own location `m_t`,
//! at the weight `ω` it counts for there. Until the target has a scale above 0 --
//! its first two rows, and while at least half its weight sits on
//! deviations of exactly 0 -- nothing is clipped, and every number is the
//! mean's to the bit, as `huber` down-weights nothing before its scale
//! exists.
//!
//! **The scale is per target**: `σ_t = median|y − m_t| / Φ⁻¹(3/4)`, an
//! [`EwQuantile`] of the target's deviations from its own location (itself
//! the one-bin form of the recursion above), bounded as `huber`'s MAD is
//! and aged on the histogram's clock. Per feature it would be `p·T`
//! sketches and a sketch update per pair per row -- several MB and a
//! multiple of the cells' cost at 10⁴ features by 50 targets -- for a scale
//! the cells share: what a cell's cut is for is a row far from the
//! target's spread, which is the same row whichever feature bins it.
//!
//! `bin_n` reads `W`, the rows' weight, under either statistic; `bin_var_y`
//! reads `M2/W`, the mean square of the deviations clipped at the cut,
//! `σ_t²·E[ψ_δ²]` at the fixed point, equal to the variance where nothing is
//! clipped; `split_gain` is the stump's gain read from those, the
//! locations' spread between bins over their total, so a spike bounds its
//! pull on both terms alike.
//!
//! # The refit
//!
//! One pass is *captured* by a share of early wild rows, as `huber`'s is
//! ([`crate::huber_refit`]): the first locations are pulled, the target's
//! scale learns the pulled deviations, the cut is drawn wide, and the wild
//! rows go on entering at nearly full weight; without decay the early rows
//! keep their share for good. Measured on `tests/test_marginal_bin_huber.py`'s
//! stream (16 bins, spikes of about 20 scales, no decay), the worst of 50
//! seeds had a bin 1.85, 1.29 and 0.23 of the target's scale from
//! statsmodels' M-location after 1,000, 2,000 and 20,000 rows at 12%
//! spikes, and 6.5, 3.5 and 1.0 at 20% (docs/PLAN.md task 278).
//!
//! So under [`BinCfg::refit_rows`] the model holds its learned rows, and on
//! the row their count is reached, after that row's own update, refits from
//! them ([`MarginalBins::refit`]): each target's location and MAD scale
//! jointly, as statsmodels' `RLM` with a constant column fits them; the
//! target's sketch rebuilt from the converged deviations; and each cell's
//! M-location at the cut that sketch gives, solved exactly
//! ([`crate::huber_refit::huber_location`]), with `Ω` and `M2` its rows'
//! Huber weights and clipped squares at it. The cells' `W` -- `bin_n` --
//! do not move. With learned edges the warm-up's own hold is those rows, so
//! the refit runs as the edges are fixed when its count is within the
//! warm-up, and otherwise the hold goes on past the freeze to the count.
//! The rows are then let go and the stream goes on in one pass. With the
//! refit, the same worst seeds were 0.017, 0.015 and 0.004 at 12%, and 0.15,
//! 0.14 and 0.057 at 20%.
//!
//! The count wants about 30 rows a bin: a bin whose rows at the refit are
//! mostly wild has its M-location among them, rightly for those rows, and
//! the next rows, clipped at the cut, bring it back only slowly
//! ([`crate::huber_refit::BIN_REFIT_ROWS_PER_BIN`]). Where a cell's rows fit
//! every point of an interval alike -- a bin of a clean row and a wild one
//! -- the point nearest the target's own location is taken, as the stream
//! judges a bin's first row against it.

use serde::{Deserialize, Serialize};

use crate::EwQuantile;

/// The renormalization point: the stored weights run to `1/RENORM_AT` times
/// the real ones before a fold, and a cell's `m2` adds `u·δ²` at that
/// weight. At `1e-50` a target spread at the input bound -- deviations of
/// `2e100`, `4e200` squared -- keeps `m2` finite up to a cell weight of
/// `4e57`. The fold was at `1e-150`, where `m2` overflowed for a spread
/// above about `1e79` and the fold kept the `inf` (review 2026-10-05,
/// CE2). A fold is `O(cells)`, now once every 166 half-lives rather than
/// 498.
const RENORM_AT: f64 = 1e-50;

/// A feature whose value falls in no bin on this row (it is not finite).
pub(crate) const NO_BIN: usize = usize::MAX;

/// The values a cell keeps, each a double in a vector of its own: the bin's
/// weight, its mean, its spread, and what the mean's double leaves out.
/// [`MarginalBins::new`] takes its per-cell vectors from one array of this
/// length, and the budget counts this many, so a vector added to the one is
/// counted by the other. The count was kept by hand before, and stayed at
/// three when task 101 added the fourth (docs/PLAN.md task 129).
pub(crate) const CELL_VALUES: usize = 4;

/// The values a cell keeps under [`BinStat::Huber`]: [`CELL_VALUES`] and
/// the weight its rows count for, `Ω` (the module doc's robust location).
pub(crate) const HUBER_CELL_VALUES: usize = CELL_VALUES + 1;

/// What a bin's location is ([`BinCfg::stat`]; docs/PLAN.md task 272 (c)).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinStat {
    /// The weighted mean, West's recurrence (the module doc).
    #[default]
    Mean,
    /// Huber's M-estimate of location, one pass, with the cut at `delta`
    /// times the target's robust scale (the module doc's "A robust location
    /// per bin"). `delta` above 0; `inf` clips nothing, which is the mean
    /// to the bit.
    Huber { delta: f64 },
}

impl BinStat {
    /// The values each cell keeps.
    pub(crate) fn cell_values(self) -> usize {
        match self {
            Self::Mean => CELL_VALUES,
            Self::Huber { .. } => HUBER_CELL_VALUES,
        }
    }
}

/// What a row's cells of one target are judged against under
/// [`BinStat::Huber`], read before the row ([`BinScales::judge`]): the cut
/// `δ·σ_t`, `+inf` while the target has no scale, and the target's own
/// location, which a bin's first row is judged against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Judge {
    pub cut: f64,
    pub m: f64,
    pub m_lo: f64,
    /// The target has a location (`m` is one).
    pub has: bool,
}

impl Judge {
    /// Nothing to judge against: no cut, no location.
    pub(crate) const NONE: Judge = Judge {
        cut: f64::INFINITY,
        m: 0.0,
        m_lo: 0.0,
        has: false,
    };
}

/// Per target, under [`BinStat::Huber`]: the target's own robust location
/// and the sketch of its deviations from it that gives its scale (the
/// module doc). The location is the one-bin form of a cell's: a weight it
/// counts for, undecayed against the histogram's scale as a cell's is, and
/// a compensated mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct BinScales {
    delta: f64,
    w: Vec<f64>,
    m: Vec<f64>,
    m_lo: Vec<f64>,
    mad: Vec<EwQuantile>,
}

impl BinScales {
    fn new(delta: f64, anchored: bool, n_targets: usize) -> Self {
        Self {
            delta,
            w: vec![0.0; n_targets],
            m: vec![0.0; n_targets],
            m_lo: vec![0.0; n_targets],
            mad: crate::robust::mad_sketches(crate::robust::HuberScale::Mad, anchored, n_targets),
        }
    }

    /// Whether the sketches are anchored ([`BinCfg::mad_anchor`]): the first
    /// one's, which [`Self::has_shape`] holds every other to.
    fn anchored(&self) -> bool {
        self.mad.first().is_some_and(|q| q.anchor().is_some())
    }

    fn has_shape(&self, n_targets: usize) -> bool {
        let anchored = self.anchored();
        (self.delta > 0.0)
            && [&self.w, &self.m, &self.m_lo]
                .iter()
                .all(|v| v.len() == n_targets)
            && self.w.iter().all(|w| w.is_finite() && *w >= 0.0)
            && crate::robust::mad_sketches_fit(
                crate::robust::HuberScale::Mad,
                anchored,
                n_targets,
                &self.mad,
            )
    }

    /// Target `t`'s cut and location, before the row.
    pub(crate) fn judge(&self, t: usize) -> Judge {
        let cut = crate::robust::mad_scale(&self.mad[t]).map_or(f64::INFINITY, |s| self.delta * s);
        Judge {
            cut,
            m: self.m[t],
            m_lo: self.m_lo[t],
            has: self.w[t] > 0.0,
        }
    }

    /// Target `t`'s value `y` at undecayed weight `u` and weight `w`, judged
    /// by `j` (its [`Self::judge`] before the row): the deviation from the
    /// location into the sketch, then the location's step.
    pub(crate) fn learn(&mut self, t: usize, y: f64, u: f64, w: f64, j: Judge) {
        if !j.has {
            self.w[t] = u;
            self.m[t] = y;
            self.m_lo[t] = 0.0;
            return;
        }
        let e = crate::comp::dev(y, self.m[t], self.m_lo[t]);
        self.mad[t].add(e.abs(), w);
        let (wt, m, lo) = (&mut self.w[t], &mut self.m[t], &mut self.m_lo[t]);
        if e.abs() <= j.cut {
            let ob = *wt + u;
            crate::comp::add(m, lo, e * (u / ob));
            *wt = ob;
        } else {
            let ob = *wt + u * (j.cut / e.abs());
            if ob > 0.0 {
                crate::comp::add(m, lo, j.cut.copysign(e) * (u / ob));
                *wt = ob;
            }
        }
    }

    fn age(&mut self, lam: f64) {
        self.mad.iter_mut().for_each(|q| q.age(lam));
    }

    fn reset(&mut self) {
        self.w.iter_mut().for_each(|v| *v = 0.0);
        self.m.iter_mut().for_each(|v| *v = 0.0);
        self.m_lo.iter_mut().for_each(|v| *v = 0.0);
        self.mad.iter_mut().for_each(EwQuantile::reset);
    }
}

/// A held row's feature and target, as [`HeldRow`] stores them and as the
/// hold's budget counts them: a target is an `Option<f64>`, twice a
/// double's size.
type HeldFeature = f64;
type HeldTarget = Option<f64>;

/// One warm-up row, kept whole so the replay can be exact
/// ([`crate::Marginal`]'s learned edges).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct HeldRow {
    pub x: Vec<HeldFeature>,
    pub y: Vec<HeldTarget>,
    pub lam: f64,
    pub w: f64,
}

/// The bytes a full warm-up hold takes: `warm_rows` rows of `p` features and
/// `n_targets` targets, each row's vectors and the row itself, from the
/// types. The model grows the hold by doubling, capped at `warm_rows`, so
/// its capacity never passes what this counts.
pub(crate) fn hold_bytes(warm_rows: usize, p: usize, n_targets: usize) -> usize {
    let row = std::mem::size_of::<HeldRow>()
        .saturating_add(p.saturating_mul(std::mem::size_of::<HeldFeature>()))
        .saturating_add(n_targets.saturating_mul(std::mem::size_of::<HeldTarget>()));
    warm_rows.saturating_mul(row)
}

/// The bytes a histogram's buffers take, for `p` features whose bins add up
/// to `width` over `n_targets` targets, `values` doubles a cell
/// ([`BinStat::cell_values`]): the cells, the edges (one fewer than
/// the bins, per feature) and their lists, the feature offsets, and the
/// row's bin offsets, which take at least four slots, a vector's smallest
/// non-zero capacity.
pub(crate) fn histogram_bytes(p: usize, n_targets: usize, width: usize, values: usize) -> usize {
    let (f, u) = (std::mem::size_of::<f64>(), std::mem::size_of::<usize>());
    let cells = n_targets.saturating_mul(width);
    cells
        .saturating_mul(values.saturating_mul(f))
        .saturating_add(width.saturating_sub(p).saturating_mul(f))
        .saturating_add(p.saturating_mul(std::mem::size_of::<Vec<f64>>()))
        .saturating_add(p.saturating_add(1).saturating_mul(u))
        .saturating_add(p.max(4).saturating_mul(u))
        .saturating_add(if values > CELL_VALUES {
            n_targets.saturating_mul(SCALES_BYTES_PER_TARGET)
        } else {
            0
        })
}

/// The most a target's location and scale take under [`BinStat::Huber`],
/// as each vector reports its allocation: three doubles for the location,
/// and a sketch of one level -- its fields, its level and the weight below
/// it, the bucket the level is in, and the buckets its band spans
/// ([`EwQuantile::one_level_buckets`]). It held room for the fields where
/// the sketch's buckets could take more than their band spans; the sketch
/// now allocates no more than that, and the bound is what it allocates at
/// its most (docs/PLAN.md task 272 (m)).
const SCALES_BYTES_PER_TARGET: usize = std::mem::size_of::<EwQuantile>()
    + (EwQuantile::one_level_buckets(crate::robust::MAD_BAND_OCTAVES) + 2 + 3)
        * std::mem::size_of::<f64>()
    + std::mem::size_of::<Option<i32>>();

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
    /// module doc). Zero is an empty bin, whatever the other three hold.
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
    /// gap and the split gain on a ratio of rounding artefacts. One per
    /// cell.
    mean_lo: Vec<f64>,
    /// Under [`BinStat::Huber`], per cell the weight its rows count for,
    /// `Ω`, undecayed like `w` (the module doc); empty under the mean.
    /// `default`, and not skipped: only one field may skip in the compact
    /// encoding, and `BinCfg::budget_mib` is that one.
    #[serde(default)]
    om: Vec<f64>,
    /// Under [`BinStat::Huber`], each target's location and scale; `None`
    /// under the mean.
    #[serde(default)]
    scales: Option<Box<BinScales>>,
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
    /// increasing and finite; empty is allowed and means one open bin. Each
    /// bin's location is its mean ([`Self::new_with`] for another). The
    /// tests' form: the model builds through [`Self::new_with`].
    #[cfg(test)]
    pub fn new(p: usize, n_targets: usize, edges: Vec<Vec<f64>>) -> Result<Self, String> {
        Self::new_with(p, n_targets, edges, BinStat::Mean)
    }

    /// [`Self::new`], each bin's location read as `stat` says, its MAD
    /// sketches unanchored: the tests' form.
    #[cfg(test)]
    pub fn new_with(
        p: usize,
        n_targets: usize,
        edges: Vec<Vec<f64>>,
        stat: BinStat,
    ) -> Result<Self, String> {
        Self::new_with_anchor(p, n_targets, edges, stat, false)
    }

    /// [`Self::new_with`], each target's MAD sketch under [`BinStat::Huber`]
    /// anchored where `anchored` says ([`BinCfg::mad_anchor`]).
    pub fn new_with_anchor(
        p: usize,
        n_targets: usize,
        mut edges: Vec<Vec<f64>>,
        stat: BinStat,
        anchored: bool,
    ) -> Result<Self, String> {
        check_edges(p, &edges)?;
        // Learned edges were reserved for every edge asked for; a feature
        // that supports fewer keeps only what it has.
        edges.iter_mut().for_each(Vec::shrink_to_fit);
        let mut off = Vec::with_capacity(p + 1);
        let mut acc = 0usize;
        for e in &edges {
            off.push(acc);
            acc += e.len() + 1;
        }
        off.push(acc);
        let cells = acc * n_targets;
        // Every per-cell vector, from one array of `CELL_VALUES`: the count
        // the budget reads (`histogram_bytes`).
        let [w, mean, m2, mean_lo]: [Vec<f64>; CELL_VALUES] =
            std::array::from_fn(|_| vec![0.0; cells]);
        let (om, scales) = match stat {
            BinStat::Mean => (Vec::new(), None),
            BinStat::Huber { delta } => {
                if delta.is_nan() || delta <= 0.0 {
                    return Err(format!(
                        "marginal: bin_huber_delta must be > 0 (inf is the mean), got {delta}"
                    ));
                }
                (
                    vec![0.0; cells],
                    Some(Box::new(BinScales::new(delta, anchored, n_targets))),
                )
            }
        };
        Ok(Self {
            p,
            n_targets,
            edges,
            off,
            w,
            mean,
            m2,
            scale: 1.0,
            empty: true,
            mean_lo,
            om,
            scales,
            row: RowBins::default(),
        })
    }

    /// Whether each target's MAD sketch is anchored ([`BinCfg::mad_anchor`]);
    /// `false` under the mean, which keeps none.
    pub fn mad_anchor(&self) -> bool {
        self.scales.as_ref().is_some_and(|s| s.anchored())
    }

    /// What each bin's location is: the statistic it was built with.
    pub fn stat(&self) -> BinStat {
        self.scales
            .as_ref()
            .map_or(BinStat::Mean, |s| BinStat::Huber { delta: s.delta })
    }

    /// Whether the edges, the offsets and the cell vectors are those of `p`
    /// features and `n_targets` targets: the offsets the edges' own, and the
    /// means' low parts absent (a state written before them) or at the
    /// cells' length. What a restored state must hold to be updated (review
    /// 2026-09-18, B3; the offsets and the low parts since review
    /// 2026-09-26, B2).
    pub fn has_shape(&self, p: usize, n_targets: usize) -> bool {
        let cells = self.off.last().map_or(0, |&o| o * n_targets);
        self.p == p
            && self.n_targets == n_targets
            && self.edges.len() == p
            && self.off.len() == p + 1
            && self.off[0] == 0
            && self
                .edges
                .iter()
                .enumerate()
                .all(|(j, e)| self.off[j + 1] == self.off[j] + e.len() + 1)
            && self.w.len() == cells
            && self.mean.len() == cells
            && self.m2.len() == cells
            && self.mean_lo.len() == cells
            && match &self.scales {
                None => self.om.is_empty(),
                Some(s) => self.om.len() == cells && s.has_shape(n_targets),
            }
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
    pub(crate) fn bin_of(edges: &[f64], v: f64) -> Option<usize> {
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
    /// `gap_cap` under the default `+inf` cap): the pair moments forget
    /// everything on that row, and so does this. Only a factor that is not a
    /// number in `[0, ∞)` is refused, since it would poison every bin at
    /// once.
    ///
    /// An **empty** histogram is not aged: there is nothing to age, and a
    /// scale it picked up while empty would still divide every later row's
    /// weight and multiply every later read, moving each by a rounding.
    /// The pair moments carry no trace of a zero-weight prefix (their mix
    /// on such a row is `a = 1, b = 0`), and `embargo`'s doubled stream
    /// begins with one -- so neither does this.
    pub fn decay(&mut self, lam: f64) {
        if self.empty || !lam.is_finite() || lam < 0.0 {
            return;
        }
        // The scales' sketches age on the histogram's clock, as their
        // locations do through `scale`.
        if let Some(sc) = self.scales.as_deref_mut() {
            sc.age(lam);
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
        let targets = self.scales.as_deref_mut().map(|s| &mut s.w);
        for v in self
            .w
            .iter_mut()
            .chain(&mut self.m2)
            .chain(&mut self.om)
            .chain(targets.into_iter().flatten())
        {
            *v = *v * self.scale * lam;
        }
        self.scale = 1.0;
        // A factor of zero wipes every bin; so, more rarely, does a fold
        // whose factor underflows a small weight. Either way the histogram
        // is back where it started, scale included, and so are the targets'
        // locations and scales.
        self.empty = self.w.iter().all(|v| *v == 0.0)
            && self
                .scales
                .as_ref()
                .is_none_or(|s| s.w.iter().all(|v| *v == 0.0));
        if self.empty
            && let Some(sc) = self.scales.as_deref_mut()
        {
            sc.reset();
        }
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
        self.empty = false;
        if self.w[i] <= 0.0 {
            self.w[i] = u;
            self.mean[i] = y;
            self.m2[i] = 0.0;
            self.mean_lo[i] = 0.0;
            return;
        }
        let wb = self.w[i] + u;
        let lo = &mut self.mean_lo[i];
        let delta = crate::comp::dev(y, self.mean[i], *lo);
        crate::comp::add(&mut self.mean[i], lo, delta * (u / wb));
        self.m2[i] += u * delta * crate::comp::dev(y, self.mean[i], *lo);
        self.w[i] = wb;
    }

    /// Cell `(t, j, b)`'s weight `W`, location and robust weight `Ω`, and
    /// target `t`'s cut before the next row: what the refit's tests read.
    #[cfg(test)]
    pub(crate) fn cell_and_cut(&self, t: usize, j: usize, b: usize) -> (f64, f64, f64, f64) {
        let i = self.at(t, j).start + b;
        let cut = self
            .scales
            .as_ref()
            .map_or(f64::INFINITY, |s| s.judge(t).cut);
        (
            self.w[i] * self.scale,
            self.mean[i],
            self.om[i] * self.scale,
            cut,
        )
    }

    /// Cell `(t, j, b)`'s clipped second moment `M2`, and target `t`'s
    /// robust weight and location, in the stream's units: what the refit's
    /// tests read beside [`Self::cell_and_cut`].
    #[cfg(test)]
    pub(crate) fn m2_and_target(&self, t: usize, j: usize, b: usize) -> (f64, f64, f64) {
        let i = self.at(t, j).start + b;
        let sc = self.scales.as_ref().expect("a Huber histogram");
        (self.m2[i] * self.scale, sc.w[t] * self.scale, sc.m[t])
    }

    /// The bytes this histogram's buffers take, as each vector reports its
    /// allocation: what [`histogram_bytes`] must count.
    #[cfg(test)]
    pub(crate) fn heap_bytes(&self) -> usize {
        let (f, u) = (std::mem::size_of::<f64>(), std::mem::size_of::<usize>());
        [&self.w, &self.mean, &self.m2, &self.mean_lo, &self.om]
            .iter()
            .map(|v| v.capacity() * f)
            .sum::<usize>()
            + self.edges.capacity() * std::mem::size_of::<Vec<f64>>()
            + self.edges.iter().map(|e| e.capacity() * f).sum::<usize>()
            + self.off.capacity() * u
            + self.row.0.capacity() * u
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
        let width = self.off[self.p];
        for (t, yt) in y.iter().enumerate().take(self.n_targets) {
            let Some(v) = yt.filter(|v| v.is_finite()) else {
                continue;
            };
            let r = t * width..(t + 1) * width;
            let wrote = match self.scales.as_deref_mut() {
                None => update_cells(
                    u,
                    v,
                    &row,
                    &mut self.w[r.clone()],
                    &mut self.mean[r.clone()],
                    &mut self.m2[r.clone()],
                    &mut self.mean_lo[r],
                ),
                Some(sc) => {
                    // The cut and the target's location before the row,
                    // then the cells, then the target's own step.
                    let j = sc.judge(t);
                    update_cells_huber(
                        u,
                        v,
                        j,
                        &row,
                        HuberCells {
                            w: &mut self.w[r.clone()],
                            om: &mut self.om[r.clone()],
                            mean: &mut self.mean[r.clone()],
                            m2: &mut self.m2[r.clone()],
                            mean_lo: &mut self.mean_lo[r],
                        },
                    );
                    sc.learn(t, v, u, w, j);
                    // The target's location holds weight now, and ages.
                    true
                }
            };
            if wrote {
                self.empty = false;
            }
        }
        self.row.0 = row;
    }

    /// Under [`BinStat::Huber`], the row's judges, one per target, pushed
    /// onto `out` -- [`Judge::NONE`] where the target is absent -- and each
    /// present target's own step taken: what a caller holding the row's
    /// cells for later ([`crate::Marginal`]'s shards) does now, since the
    /// next row's judge reads it. `u` and `w` as [`Self::update_row`] has
    /// them. Nothing under the mean.
    pub(crate) fn judge_row(&mut self, y: &[Option<f64>], u: f64, w: f64, out: &mut Vec<Judge>) {
        let Some(sc) = self.scales.as_deref_mut() else {
            return;
        };
        for (t, yt) in y.iter().enumerate().take(self.n_targets) {
            let Some(v) = yt.filter(|v| v.is_finite()) else {
                out.push(Judge::NONE);
                continue;
            };
            let j = sc.judge(t);
            sc.learn(t, v, u, w, j);
            out.push(j);
            self.empty = false;
        }
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
    /// shards).
    pub(crate) fn cell_parts(&mut self) -> CellParts<'_> {
        CellParts {
            w: &mut self.w,
            mean: &mut self.mean,
            m2: &mut self.m2,
            mean_lo: &mut self.mean_lo,
            om: &mut self.om,
            edges: &self.edges,
            off: &self.off,
        }
    }
}

/// [`MarginalBins::cell_parts`]: the cell vectors, laid out
/// `[t·off[p] + off[j] + bin]` (`om` empty under the mean), and the edges
/// and feature offsets that find a value's cell ([`bin_offsets_in`]).
pub(crate) struct CellParts<'a> {
    pub w: &'a mut [f64],
    pub mean: &'a mut [f64],
    pub m2: &'a mut [f64],
    pub mean_lo: &'a mut [f64],
    pub om: &'a mut [f64],
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

/// One target's cells under [`BinStat::Huber`], as [`update_cells_huber`]
/// writes them.
pub(crate) struct HuberCells<'a> {
    pub w: &'a mut [f64],
    pub om: &'a mut [f64],
    pub mean: &'a mut [f64],
    pub m2: &'a mut [f64],
    pub mean_lo: &'a mut [f64],
}

/// [`update_cells`] under [`BinStat::Huber`]: each named cell takes `y` at
/// undecayed weight `u`, judged by `j` (the module doc's robust location).
/// A row inside the cut takes West's step over `Ω` in place of `W`, the
/// same operations, so a cell no row was ever clipped in -- every cell
/// before the target has a scale, and every cell under `δ = inf` -- holds
/// the mean's numbers to the bit. A row past it steps by the cut over `Ω'`
/// at the weight `u·c/|e|`. Whether any cell was written.
#[inline]
pub(crate) fn update_cells_huber(
    u: f64,
    y: f64,
    j: Judge,
    at: &[usize],
    c: HuberCells<'_>,
) -> bool {
    let HuberCells {
        w,
        om,
        mean,
        m2,
        mean_lo,
    } = c;
    let mut wrote = false;
    for &o in at {
        if o == NO_BIN {
            continue;
        }
        wrote = true;
        let (mi, lo) = (&mut mean[o], &mut mean_lo[o]);
        if w[o] <= 0.0 {
            // The bin's first row: judged against the target's location,
            // where it has one, winsorized at the cut around it.
            w[o] = u;
            m2[o] = 0.0;
            if j.has {
                let e = crate::comp::dev(y, j.m, j.m_lo);
                if e.abs() > j.cut {
                    *mi = j.m;
                    *lo = j.m_lo;
                    crate::comp::add(mi, lo, j.cut.copysign(e));
                    om[o] = u * (j.cut / e.abs());
                    continue;
                }
            }
            *mi = y;
            *lo = 0.0;
            om[o] = u;
            continue;
        }
        let delta = crate::comp::dev(y, *mi, *lo);
        if delta.abs() <= j.cut {
            let ob = om[o] + u;
            crate::comp::add(mi, lo, delta * (u / ob));
            m2[o] += u * delta * crate::comp::dev(y, *mi, *lo);
            om[o] = ob;
        } else {
            let before = om[o];
            let ob = before + u * (j.cut / delta.abs());
            // A weight that underflowed to nothing takes no step: the
            // location holds, and the row counts in `W` alone.
            if ob > 0.0 {
                let c = j.cut.copysign(delta);
                crate::comp::add(mi, lo, c * (u / ob));
                m2[o] += u * c * c * (before / ob);
                om[o] = ob;
            }
        }
        w[o] += u;
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
        // product of ratios: the undecayed weights run to `1e50` times the
        // real ones just before a fold (`RENORM_AT`), and three of them
        // multiplied together with a spread squared overflow.
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

impl MarginalBins {
    /// The Huber refit of the first rows (docs/PLAN.md task 278): every
    /// target's location and scale, and every cell's location, rebuilt from
    /// the held rows `rows` -- the rows the histogram has learned so far,
    /// each with the decay applied before it -- as the converged batch fit of
    /// those rows, in place of the one pass (the module doc's "The refit").
    /// Run right after the last of them is learned, so no decay has applied
    /// since. Each target's rows are those carrying it:
    ///
    /// 1. its location and MAD scale jointly ([`crate::huber_refit::location_refit`]);
    /// 2. its scale's sketch rebuilt from the converged deviations at the
    ///    rows' raw weights, aged as the stream aged it;
    /// 3. each cell's location at the cut that sketch gives, exactly
    ///    ([`crate::huber_refit::huber_location`]); its `Ω` and `M2` the
    ///    rows' Huber weights and clipped squares at it.
    ///
    /// The cells' weights `W` -- what `bin_n` reads, and what every pair
    /// moment shares -- do not move (CLAUDE.md rule 8). The passes each
    /// target's location took, 0 for a target no row carried. Nothing under
    /// the mean.
    pub(crate) fn refit(&mut self, rows: &[HeldRow]) -> Vec<u32> {
        let mut passes = vec![0u32; self.n_targets];
        let Some(delta) = self.scales.as_ref().map(|s| s.delta) else {
            return passes;
        };
        let n = rows.len();
        // Each row's raw weight decayed to now, `v_i = w_i · Π_{l > i} λ_l`,
        // and in the cells' undecayed units, `v_i / scale` (the module doc's
        // "Decay without touching every bin"), across any fold since.
        let mut v = vec![0.0; n];
        let mut age = 1.0;
        for (i, r) in rows.iter().enumerate().rev() {
            v[i] = r.w * age;
            age *= r.lam;
        }
        let units: Vec<f64> = v.iter().map(|vi| vi / self.scale).collect();
        let width = self.off[self.p];
        let mut ys = Vec::new();
        let mut vs = Vec::new();
        let mut order: Vec<usize> = Vec::new();
        let mut bin_at = Vec::new();
        let mut counts = Vec::new();
        let mut yb = Vec::new();
        let mut vb = Vec::new();
        let mut ub = Vec::new();
        for (t, took_t) in passes.iter_mut().enumerate() {
            order.clear();
            order.extend((0..n).filter(|&i| {
                v[i] > 0.0
                    && rows[i]
                        .y
                        .get(t)
                        .copied()
                        .flatten()
                        .is_some_and(f64::is_finite)
            }));
            if order.is_empty() {
                continue;
            }
            let y_of = |i: usize| rows[i].y[t].expect("kept above");
            order.sort_by(|&a, &b| y_of(a).total_cmp(&y_of(b)));
            ys.clear();
            ys.extend(order.iter().map(|&i| y_of(i)));
            vs.clear();
            vs.extend(order.iter().map(|&i| v[i]));
            let Some((m, took)) = crate::huber_refit::location_refit(&ys, &vs, &vs, delta) else {
                continue;
            };
            *took_t = took;
            // The scale's sketch from the converged deviations, in the rows'
            // order, at their raw weights, aged on the histogram's clock from
            // the target's first row.
            let anchored = self.scales.as_deref().is_some_and(BinScales::anchored);
            let mut q = crate::robust::mad_sketches(crate::robust::HuberScale::Mad, anchored, 1)
                .pop()
                .expect("one sketch under the MAD");
            // Anchored at the deviations' median size, not at the first
            // deviation, which is rounded at the target's level and may sit
            // at that rounding: `crate::huber_refit::rebuild_scale`'s rule
            // (task 278, qequiv).
            if anchored {
                let dev: Vec<f64> = ys.iter().map(|y| (y - m).abs()).collect();
                q.anchor_at(crate::huber_refit::weighted_median(&dev, &vs));
            }
            let mut started = false;
            for (i, r) in rows.iter().enumerate() {
                if started {
                    q.age(r.lam);
                }
                if let Some(yi) = r.y.get(t).copied().flatten().filter(|y| y.is_finite())
                    && v[i] > 0.0
                {
                    q.add((yi - m).abs(), r.w);
                    started = true;
                }
            }
            let cut = crate::robust::mad_scale(&q).map_or(f64::INFINITY, |s| delta * s);
            let weigh = |e: f64| if e.abs() > cut { cut / e.abs() } else { 1.0 };
            let sc = self.scales.as_deref_mut().expect("checked above");
            sc.m[t] = m;
            sc.m_lo[t] = 0.0;
            sc.w[t] = order.iter().map(|&i| units[i] * weigh(y_of(i) - m)).sum();
            sc.mad[t] = q;
            // Each feature's cells: the target's rows dealt to their bins in
            // the order of `y`, so each bin's rows arrive sorted.
            let block = t * width;
            for j in 0..self.p {
                let edges = &self.edges[j];
                let nb = edges.len() + 1;
                bin_at.clear();
                bin_at.extend(order.iter().map(|&i| {
                    Self::bin_of(edges, rows[i].x.get(j).copied().unwrap_or(f64::NAN))
                        .unwrap_or(NO_BIN)
                }));
                counts.clear();
                counts.resize(nb + 1, 0usize);
                for &b in &bin_at {
                    if b != NO_BIN {
                        counts[b + 1] += 1;
                    }
                }
                for b in 0..nb {
                    counts[b + 1] += counts[b];
                }
                yb.clear();
                yb.resize(counts[nb], 0.0);
                vb.clear();
                vb.resize(counts[nb], 0.0);
                ub.clear();
                ub.resize(counts[nb], 0.0);
                let mut fill = counts.clone();
                for (k, &b) in bin_at.iter().enumerate() {
                    if b == NO_BIN {
                        continue;
                    }
                    let at = fill[b];
                    fill[b] += 1;
                    yb[at] = ys[k];
                    vb[at] = vs[k];
                    ub[at] = units[order[k]];
                }
                for b in 0..nb {
                    let r = counts[b]..counts[b + 1];
                    let cell = block + self.off[j] + b;
                    // A cell no row reached has no location to give, which
                    // `huber_location` says by `None`.
                    if self.w[cell] <= 0.0 {
                        continue;
                    }
                    let Some(loc) =
                        crate::huber_refit::huber_location(&yb[r.clone()], &vb[r.clone()], cut, m)
                    else {
                        continue;
                    };
                    let (mut om, mut m2) = (0.0, 0.0);
                    for k in r {
                        let e = yb[k] - loc;
                        om += ub[k] * weigh(e);
                        let c = e.abs().min(cut);
                        m2 += ub[k] * c * c;
                    }
                    self.mean[cell] = loc;
                    self.mean_lo[cell] = 0.0;
                    self.om[cell] = om;
                    self.m2[cell] = m2;
                }
            }
        }
        passes
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
    /// What each bin's location is: its mean, or Huber's M-location
    /// ([`BinStat`]; docs/PLAN.md task 272 (c)). `default`, the mean, so a
    /// state written before it reads as it was fitted; not skipped, since
    /// `budget_mib` is the one field here that may.
    #[serde(default)]
    pub stat: BinStat,
    /// Under [`BinStat::Huber`], the learned rows after which every
    /// target's location and scale and every cell's location are refit to
    /// the converged batch fit of the rows learned so far, once (the module
    /// doc's "The refit"; docs/PLAN.md task 278); 0 is off. With learned
    /// edges the warm-up already holds its rows, so the refit runs at the
    /// later of this count and `warm_rows`. `default`, 0, so a state written
    /// before it -- fitted without the refit -- loads and goes on as it was;
    /// not skipped, since `budget_mib` is the one field here that may.
    #[serde(default)]
    pub refit_rows: u32,
    /// Under [`BinStat::Huber`], whether each target's MAD sketch buckets
    /// its deviations in units of the first it takes (the refit's rebuilt
    /// sketch in units of their median size), as
    /// [`crate::RobustCfg::mad_anchor`]. A spec sets it; a state written
    /// before schema 64 loads as `false` and goes on as it was.
    #[serde(default)]
    pub mad_anchor: bool,
    /// The MiB the warm-up hold and the histogram may each take before the
    /// model refuses to build; `None` is [`DEFAULT_BUDGET_MIB`], and
    /// infinity is no bound (docs/PLAN.md task 131). **Last, and the one
    /// field that skips**, so a state written without it reads in both
    /// encodings and a spec that does not set it keeps its bytes
    /// (`tests/state_encoding.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_mib: Option<f64>,
}

/// The most a warm-up hold or a histogram may take before the model refuses
/// to build, unless the spec's `bin_budget` says otherwise: `p` can be
/// 10,000 here, and silently allocating gigabytes is a worse outcome than
/// an error that names the number.
///
/// Each is held to it on its own, and per model: every group keeps its own
/// hold and histogram, as does every half-life of a grid. At the warm-up's
/// last row the two exist at once, while the held rows are replayed into
/// the new histogram, so that row's peak is their sum. A check on the sum
/// was considered and not taken: it would refuse a spec of 10,000 features,
/// 50 targets and 16 bins, whose histogram alone fits (docs/PLAN.md task
/// 129).
pub const DEFAULT_BUDGET_MIB: f64 = 256.0;

/// The default in bytes, for the tests that pin the boundary.
#[cfg(test)]
const MEMORY_BUDGET: usize = 256 << 20;

fn budget_check(what: &str, bytes: usize, budget_mib: f64, detail: &str) -> Result<(), String> {
    // In MiB, as the budget is given: a budget of `inf` holds every size.
    // Said in MiB too, with the digits to read above the budget: in GiB to
    // two decimals a small one read "0.00 GiB" (review 2026-10-06, CD16).
    let mib = bytes as f64 / (1u64 << 20) as f64;
    if mib > budget_mib {
        return Err(format!(
            "marginal: {what} would need {} MiB ({detail}), over the {budget_mib} MiB each \
             group's model is held to; raise bin_budget, or ask for fewer bins, targets or \
             features",
            crate::budget::mib_over(mib, budget_mib),
        ));
    }
    Ok(())
}

impl BinCfg {
    /// The budget in force: `budget_mib`, or [`DEFAULT_BUDGET_MIB`] where
    /// the spec names none. The one place the default is read, so the API
    /// snapshot's `[resolved defaults]` reads what the check holds a model to.
    pub fn budget_mib_or_default(&self) -> f64 {
        self.budget_mib.unwrap_or(DEFAULT_BUDGET_MIB)
    }

    pub fn validate(&self, p: usize, n_targets: usize) -> Result<(), String> {
        let budget = self.budget_mib_or_default();
        // NaN compares false both ways, so it is named here rather than
        // read as no bound.
        if budget.is_nan() || budget <= 0.0 {
            return Err(format!(
                "marginal: bin_budget must be a positive number of MiB, or inf for no bound; \
                 got {budget}"
            ));
        }
        if let BinStat::Huber { delta } = self.stat
            && (delta.is_nan() || delta <= 0.0)
        {
            return Err(format!(
                "marginal: bin_huber_delta must be > 0 (inf is the mean), got {delta}"
            ));
        }
        // The refit is of Huber locations: under the mean, or a cut that
        // clips nothing, the one pass is already the batch fit.
        if self.refit_rows > 0
            && !matches!(self.stat, BinStat::Huber { delta } if delta.is_finite())
        {
            return Err(
                "marginal: bin_refit_rows applies only with bin_stat = \"huber\" and a finite \
                 bin_huber_delta: under the mean, or a cut that clips nothing, the one pass is \
                 already the batch fit"
                    .into(),
            );
        }
        let values = self.stat.cell_values();
        match &self.edges {
            Some(e) => {
                check_edges(p, e)?;
                let bins: usize = e.iter().map(|f| f.len() + 1).sum();
                budget_check(
                    "the bin histogram",
                    histogram_bytes(p, n_targets, bins, values),
                    budget,
                    &format!("{bins} bins over {p} features x {n_targets} targets"),
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
                    hold_bytes(self.warm_rows, p, n_targets),
                    budget,
                    &format!(
                        "bin_warm_rows {} of {p} features and {n_targets} targets",
                        self.warm_rows
                    ),
                )?;
                // A feature keeps at most the bins asked for, so this is the
                // most a learned histogram can take.
                budget_check(
                    "the bin histogram",
                    histogram_bytes(p, n_targets, p.saturating_mul(self.n_bins), values),
                    budget,
                    &format!("{p} features x {n_targets} targets x {} bins", self.n_bins),
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
            // The production row update, not the test-only per-target one
            // (review 2026-09-26, B missing 2).
            b.update_row(&[x], &[Some(y)], w);
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
    /// past it -- `lam^rows < 1e-50` needs ~383 rows at 0.74, so 2000 rows
    /// fold five times -- and compare against the brute force, which never
    /// scales anything.
    #[test]
    fn renormalization_changes_nothing_readable() {
        let (b, sums) = brute(0.74, 2000, &[-0.5, 0.0, 0.5]);
        assert!(0.74f64.powi(2000) < RENORM_AT, "the stream passes a fold");
        assert!(
            b.scale >= RENORM_AT && b.scale < 1e-6,
            "the last fold was not on the last rows: {}",
            b.scale
        );
        assert_matches(&b, &sums);
        let s = b.best_split(0, 0).unwrap();
        assert!(s.gain.is_finite() && (0.0..=1.0).contains(&s.gain));
    }

    /// A target spread at `1e90`, inside the input bound, through the folds
    /// of a half-life of one row: the histogram is the one the same rows
    /// make at a spread of 1, scaled -- every bin's mean by `1e90` and
    /// variance by `1e180`, the same weights, and the same split. The stored
    /// weights ran to `1e150` times the real ones before a fold at `1e-150`,
    /// and `u·δ²` overflowed `m2` for a spread above about `1e79`, which the
    /// fold kept: `inf` in a bin's variance and a split gain of 0 for good
    /// (review 2026-10-05, CE2).
    #[test]
    fn a_target_spread_at_the_bound_survives_the_folds() {
        let run = |level: f64| {
            let mut b = MarginalBins::new(1, 1, vec![vec![0.0]]).unwrap();
            let mut seed = 3u64;
            let (mut seen, mut folds) = (Vec::new(), 0);
            for i in 0..1600 {
                let x = lcg(&mut seed);
                let y = level * (x + 0.3 * lcg(&mut seed));
                if i > 0 {
                    b.decay(0.5);
                    folds += usize::from(b.scale == 1.0);
                }
                b.update_row(&[x], &[Some(y)], 1.0);
                seen.push((b.bins(0, 0), b.best_split(0, 0)));
            }
            assert!(folds >= 3, "the case needs folds: {folds}");
            seen
        };
        let (unit, big) = (run(1.0), run(1e90));
        for (i, ((u_bins, u_split), (b_bins, b_split))) in unit.iter().zip(&big).enumerate() {
            for (u, b) in u_bins.iter().zip(b_bins) {
                assert_eq!(u.n.to_bits(), b.n.to_bits(), "row {i}: the weights");
                if u.n > 0.0 {
                    assert!(b.var_y.is_finite(), "row {i}: {b:?}");
                    let (m, v) = (1e90 * u.mean_y, 1e180 * u.var_y);
                    assert!((b.mean_y - m).abs() <= 1e-12 * 1e90, "row {i}: {b:?} {u:?}");
                    assert!((b.var_y - v).abs() <= 1e-9 * v, "row {i}: {b:?} {u:?}");
                }
            }
            let (Some(u), Some(b)) = (u_split, b_split) else {
                assert_eq!(u_split.is_some(), b_split.is_some(), "row {i}");
                continue;
            };
            assert!(
                b.gain > 0.0 && (b.gain - u.gain).abs() <= 1e-9,
                "row {i}: {b:?} {u:?}"
            );
            assert_eq!(b.at, u.at);
        }
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
        b.update_row(&[0.5], &[Some(2.0)], 3.0);
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
        b.update_row(&[-0.5], &[Some(1.0)], 2.0);
        b.update_row(&[-0.5], &[Some(3.0)], 2.0);
        let bins = b.bins(0, 0);
        assert_eq!((bins[0].n, bins[0].mean_y, bins[0].var_y), (4.0, 2.0, 1.0));
        assert_eq!(bins[1].n, 0.0);
    }

    /// Decay leaves no trace on an empty histogram. The doubled stream
    /// `embargo` is checked against opens with `delay` zero-weight rows,
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
                b.update_row(&x, &[Some(y)], 1.0);
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
            b.update_row(&[-1.0], &[Some(y)], 1.0);
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
            b.update_row(&[x], &[Some(y)], 1.0);
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
        b.update_row(&[0.0], &[Some(5.0)], 1.0);
        b.update_row(&[f64::NAN], &[Some(5.0)], 1.0);
        b.update_row(&[f64::INFINITY], &[Some(5.0)], 1.0);
        b.update_row(&[0.5], &[Some(f64::NAN)], 1.0);
        b.update_row(&[0.5], &[Some(5.0)], 0.0);
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

    /// The histogram's budget is what its buffers take, to the byte, as the
    /// vectors report their allocations: for given edges and for learned
    /// ones, at one target and several, once a row has been learned
    /// (docs/PLAN.md task 129).
    #[test]
    fn the_histogram_budget_is_what_its_buffers_take() {
        // Ragged given edges, zero to five a feature, and learned ones, which
        // were reserved for every edge asked for; one target and several.
        let p = 23;
        let given: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..j % 6).map(|k| k as f64).collect())
            .collect();
        let mut seed = 3u64;
        let learned: Vec<Vec<f64>> = (0..p)
            .map(|j| {
                let mut v: Vec<(f64, f64)> = (0..200)
                    .map(|_| ((lcg(&mut seed) * (j % 4 + 1) as f64).round(), 1.0))
                    .collect();
                edges_from(BinRule::Quantile, 8, &mut v)
            })
            .collect();
        assert!(
            learned.iter().any(|e| e.len() < 7),
            "some feature supports fewer edges than asked for"
        );
        let x: Vec<f64> = (0..p).map(|j| j as f64 * 0.37).collect();
        // The row's offsets take at least four slots however narrow the row
        // (review 2026-09-26, B5).
        for p in 1..4 {
            let mut b = MarginalBins::new(p, 1, vec![vec![0.0]; p]).unwrap();
            b.update_row(&vec![0.5; p], &[Some(1.0)], 1.0);
            assert_eq!(
                b.heap_bytes(),
                histogram_bytes(p, 1, 2 * p, CELL_VALUES),
                "{p} features"
            );
        }
        for t in [1, 3, 20] {
            let y: Vec<Option<f64>> = (0..t).map(|k| Some(k as f64)).collect();
            for edges in [&given, &learned] {
                let width: usize = edges.iter().map(|e| e.len() + 1).sum();
                let mut b = MarginalBins::new(p, t, edges.clone()).unwrap();
                // The row's offsets are allocated by the first row.
                b.update_row(&x, &y, 1.0);
                assert_eq!(
                    b.heap_bytes(),
                    histogram_bytes(p, t, width, CELL_VALUES),
                    "{t} targets"
                );
                // What a learned kind is refused by bounds what it takes.
                assert!(b.heap_bytes() <= histogram_bytes(p, t, p * 8, CELL_VALUES));
            }
        }
    }

    /// Every number a histogram's state holds is a cell's value, an edge, a
    /// feature offset or one of its three scalars. A vector added to the
    /// state and not counted by `CELL_VALUES` fails here, before it can
    /// outgrow the budget unseen, as the fourth did (docs/PLAN.md task 129).
    #[test]
    fn the_state_holds_nothing_the_budget_does_not_count() {
        fn numbers(v: &serde_json::Value) -> usize {
            match v {
                serde_json::Value::Number(_) => 1,
                serde_json::Value::Array(a) => a.iter().map(numbers).sum(),
                serde_json::Value::Object(o) => o.values().map(numbers).sum(),
                _ => 0,
            }
        }
        let (p, t) = (5, 2);
        let edges = vec![
            vec![0.0],
            vec![],
            vec![-1.0, 1.0],
            vec![2.0],
            vec![0.5, 1.5, 2.5],
        ];
        let mut b = MarginalBins::new(p, t, edges.clone()).unwrap();
        b.update_row(&[0.1, 0.2, 0.3, 0.4, 0.5], &[Some(1.0), Some(2.0)], 1.0);
        let n_edges: usize = edges.iter().map(Vec::len).sum();
        let cells = t * (n_edges + p);
        let state = serde_json::to_value(&b).unwrap();
        assert_eq!(numbers(&state), CELL_VALUES * cells + n_edges + (p + 1) + 3);
    }

    /// Refused where the buffers would pass 256 MiB, counted in full. The
    /// three-value count let a histogram take a third more, and the hold
    /// counted its targets at half their size (docs/PLAN.md task 129).
    #[test]
    fn the_budget_refuses_at_the_real_size() {
        let learned = |n_bins, warm_rows| BinCfg {
            n_bins,
            edges: None,
            rule: BinRule::Quantile,
            warm_rows,
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib: None,
            mad_anchor: false,
        };
        let cfg = learned(16, 1_000);
        // E73's full profile fits: 10,000 features, 50 targets, 16 bins.
        assert!(cfg.validate(10_000, 50).is_ok());
        // Sixty targets passed the three-value count, and do not fit.
        const {
            assert!(
                3 * 8 * 60 * 160_000 <= MEMORY_BUDGET,
                "the old count passed it"
            )
        };
        let err = cfg.validate(10_000, 60).unwrap_err();
        assert!(err.contains("the bin histogram"), "{err}");
        assert!(err.contains("each group's model"), "{err}");
        // A hold of many targets passed at eight bytes a value.
        const {
            assert!(
                1_000 * (30_000 + 3_500) * 8 <= MEMORY_BUDGET,
                "the old count passed it"
            )
        };
        let err = learned(2, 1_000).validate(30_000, 3_500).unwrap_err();
        assert!(err.contains("warm-up hold"), "{err}");
        assert!(learned(2, 1_000).validate(30_000, 10).is_ok());
    }

    /// `bin_budget` sets the limit: a larger one allows what the default
    /// refuses, a smaller one refuses what it allows, infinity is no bound,
    /// and a budget that is not a positive number is refused by name
    /// (docs/PLAN.md task 131).
    #[test]
    fn the_budget_can_be_set() {
        let with = |budget_mib| BinCfg {
            n_bins: 16,
            edges: None,
            rule: BinRule::Quantile,
            warm_rows: 1_000,
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib,
            mad_anchor: false,
        };
        // Sixty targets at 10,000 features pass 256 MiB; 512 holds them.
        assert!(with(None).validate(10_000, 60).is_err());
        assert!(with(Some(512.0)).validate(10_000, 60).is_ok());
        assert!(with(Some(f64::INFINITY)).validate(10_000, 60).is_ok());
        // E73's full profile fits the default and not 100 MiB.
        assert!(with(None).validate(10_000, 50).is_ok());
        let err = with(Some(100.0)).validate(10_000, 50).unwrap_err();
        assert!(
            err.contains("100 MiB") && err.contains("raise bin_budget"),
            "{err}"
        );
        for bad in [0.0, -1.0, f64::NAN] {
            let err = with(Some(bad)).validate(10, 1).unwrap_err();
            assert!(
                err.contains("bin_budget must be a positive number"),
                "{err}"
            );
        }
        // Given edges are held to it too.
        let given = BinCfg {
            n_bins: 0,
            edges: Some(vec![vec![0.0, 1.0]; 10]),
            rule: BinRule::Quantile,
            warm_rows: 0,
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib: Some(1e-6),
            mad_anchor: false,
        };
        let err = given.validate(10, 1).unwrap_err();
        assert!(err.contains("the bin histogram"), "{err}");
    }

    /// The size is given in the budget's unit, MiB, with the digits to read
    /// above it: a small budget read "would need 0.00 GiB (...), over the
    /// 0.000001 MiB" (review 2026-10-06, CD16).
    #[test]
    fn a_refusal_names_the_size_in_the_budgets_unit() {
        let cfg = BinCfg {
            n_bins: 4,
            edges: None,
            rule: BinRule::Quantile,
            warm_rows: 8,
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib: Some(1e-6),
            mad_anchor: false,
        };
        let err = cfg.validate(1, 1).unwrap_err();
        assert!(!err.contains("GiB"), "{err}");
        let shown: f64 = err
            .split("would need ")
            .nth(1)
            .and_then(|s| s.split(" MiB (").next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no size in MiB: {err}"));
        let mib = hold_bytes(8, 1, 1) as f64 / f64::from(1 << 20);
        assert!(shown > 1e-6 && (shown - mib).abs() <= 0.005 * mib, "{err}");
        assert!(err.contains("over the 0.000001 MiB"), "{err}");
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
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib: None,
            mad_anchor: false,
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
            b.update_row(&[-1.0], &[Some(4.0)], 1.0);
        }
        assert!(b.best_split(0, 0).is_none(), "one bin, constant target");
    }

    /// A pair's bins are its own target's, over two targets and two features
    /// of different bin counts: each bin's weight and mean are those of the
    /// rows whose feature fell in it (a value on an edge counted above it)
    /// and which carried that target.
    #[test]
    fn each_target_reads_its_own_bins() {
        let edges = vec![vec![-0.5, 0.0, 0.5], vec![0.0]];
        let mut b = MarginalBins::new(2, 2, edges.clone()).unwrap();
        let mut seed = 9u64;
        let mut rows = Vec::new();
        for i in 0..60 {
            let x = [lcg(&mut seed), lcg(&mut seed)];
            let y = [
                Some(3.0 * x[0] + 0.1 * lcg(&mut seed)),
                (i % 4 != 0).then(|| 10.0 - x[1].abs()),
            ];
            b.update_row(&x, &y, 1.0);
            rows.push((x, y));
        }
        for t in 0..2 {
            for (j, e) in edges.iter().enumerate() {
                let bins = b.bins(t, j);
                assert_eq!(bins.len(), e.len() + 1);
                for (k, bin) in bins.iter().enumerate() {
                    let ys: Vec<f64> = rows
                        .iter()
                        .filter(|(x, _)| e.iter().filter(|v| **v <= x[j]).count() == k)
                        .filter_map(|(_, y)| y[t])
                        .collect();
                    let n = ys.len() as f64;
                    let mean = ys.iter().sum::<f64>() / n;
                    assert_eq!(bin.n, n, "target {t}, feature {j}, bin {k}");
                    assert!(
                        (bin.mean_y - mean).abs() < 1e-12,
                        "target {t}, feature {j}, bin {k}: {} vs {mean}",
                        bin.mean_y
                    );
                }
            }
        }
    }

    /// `folds_at` is whether `decay` would fold: a scale that lands on
    /// `RENORM_AT` exactly is kept and one below it folds, a factor of zero
    /// folds, a factor that is not a number in `[0, ∞)` is refused by both,
    /// and an empty histogram is never aged, so never folds.
    #[test]
    fn folds_at_is_whether_decay_folds() {
        let mut b = MarginalBins::new(1, 1, vec![vec![0.0]]).unwrap();
        for lam in [1e-200, 0.0] {
            assert!(!b.folds_at(lam), "empty, {lam}");
        }
        b.update_row(&[1.0], &[Some(2.0)], 1.0);
        assert_eq!(b.scale, 1.0);
        for (lam, folds) in [
            (RENORM_AT, false),
            (RENORM_AT.next_down(), true),
            (0.0, true),
            (0.5, false),
            (f64::NAN, false),
            (-1.0, false),
            (f64::INFINITY, false),
        ] {
            assert_eq!(b.folds_at(lam), folds, "{lam}");
            if lam.is_finite() && lam >= 0.0 {
                // A fold puts the scale back at 1; otherwise it is `lam`.
                let mut c = b.clone();
                c.decay(lam);
                assert_eq!(c.scale == 1.0, folds, "decay({lam}) left {}", c.scale);
            }
        }
    }

    /// After a total gap the bins hold no weight but keep their old means.
    /// A split read from rows that then fill some bins and not others is
    /// the stump's on those rows alone, against the definition: the grand
    /// mean is not taken from an empty bin's leftover, an empty bin at the
    /// low end cuts nothing, and two cuts with an empty bin between them
    /// split the same rows, the lower edge reported.
    #[test]
    fn a_split_after_a_wipe_reads_only_the_bins_that_hold_weight() {
        let edges = vec![-1.0, 0.0, 1.0, 2.0];
        let mut b = MarginalBins::new(1, 1, vec![edges.clone()]).unwrap();
        for x in [-2.0, -0.5, 0.5, 1.5, 3.0] {
            b.update_row(&[x], &[Some(1e20)], 1.0);
        }
        b.decay(0.0);
        assert!(b.is_empty() && b.mean.iter().all(|m| *m == 1e20));
        // Rows in bins 1, 2 and 4: none below -1, none in [1, 2).
        let mut seed = 5u64;
        let mut rows = Vec::new();
        for i in 0..90 {
            let u = lcg(&mut seed);
            let (x, level) = match i % 3 {
                0 => (-0.5 + 0.4 * u, 1.0),
                1 => (0.5 + 0.4 * u, 1.5),
                _ => (3.0 + 0.5 * u, 6.0),
            };
            let y = level + 0.3 * lcg(&mut seed);
            b.update_row(&[x], &[Some(y)], 1.0);
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
            if l.is_empty() || r.is_empty() {
                continue;
            }
            let (nl, nr) = (l.len() as f64, r.len() as f64);
            let g = (total - (nl * var_of(&l) + nr * var_of(&r)) / (nl + nr)) / total;
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
        assert_eq!((got.at, best_at), (1.0, 1.0));
    }

    /// The budget holds a histogram whose buffers come to it exactly and
    /// refuses one a byte over, counting bins, one more than the edges, for
    /// every feature: three features of two edges are nine bins.
    #[test]
    fn the_budget_is_inclusive_and_counts_bins() {
        let (p, t) = (3, 2);
        let bytes = histogram_bytes(p, t, 9, CELL_VALUES);
        let at = |bytes: usize| BinCfg {
            n_bins: 0,
            edges: Some(vec![vec![0.0, 1.0]; 3]),
            rule: BinRule::Quantile,
            warm_rows: 0,
            stat: BinStat::Mean,
            refit_rows: 0,
            budget_mib: Some(bytes as f64 / (1u64 << 20) as f64),
            mad_anchor: false,
        };
        assert!(at(bytes).validate(p, t).is_ok());
        let err = at(bytes - 1).validate(p, t).unwrap_err();
        assert!(err.contains("9 bins over 3 features x 2 targets"), "{err}");
    }

    /// Each quantile edge closes a bin of the weight still unassigned over
    /// the bins still to come: ten equal rows into four bins close `10/4` ->
    /// `{1, 2}`, then `8/3` -> `{3, 4}`, then `6/2` -> `{5, 6, 7}`, leaving
    /// `{8, 9, 10}`.
    #[test]
    fn each_quantile_edge_shares_what_is_left() {
        let mut v = sample(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(
            edges_from(BinRule::Quantile, 4, &mut v),
            vec![3.0, 5.0, 8.0]
        );
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

    /// A row whose features bin nothing -- every one not finite -- is
    /// taken by the row update as the per-target update takes it: nothing
    /// written, no panic (review 2026-09-26, B1: the row update sliced
    /// `mean_lo` for every present target before it could skip).
    #[test]
    fn a_row_that_bins_nothing_writes_nothing() {
        let mut a = MarginalBins::new(2, 1, vec![vec![0.0], vec![1.0]]).unwrap();
        a.update_row(&[0.5, 2.0], &[Some(1.0)], 1.0);
        let mut b = a.clone();
        a.update_row(&[f64::NAN, f64::NAN], &[Some(1.0)], 1.0);
        b.update_target(0, &[f64::NAN, f64::NAN], 1.0, 1.0);
        assert_eq!(bits(&a), bits(&b), "nothing was written");
        a.update_row(&[0.5, 2.0], &[Some(2.0)], 1.0);
        b.update_target(0, &[0.5, 2.0], 2.0, 1.0);
        assert_eq!(bits(&a), bits(&b));
    }

    /// A restored histogram whose offsets disagree with its edges, or whose
    /// low parts are not at the cells' length, is refused as the wrong shape
    /// rather than written through (review 2026-09-26, B2); absent low parts
    /// too, which a state written before them carried and a repair sized at
    /// zero (docs/PLAN.md task 198).
    #[test]
    fn a_bin_state_whose_offsets_or_low_parts_disagree_is_refused() {
        let good = MarginalBins::new(3, 2, vec![vec![0.0], vec![], vec![1.0, 2.0]]).unwrap();
        assert!(good.has_shape(3, 2));
        let mut off = good.clone();
        off.off[1] += 1;
        assert!(!off.has_shape(3, 2), "an offset past its feature's bins");
        let mut lo = good.clone();
        lo.mean_lo.pop();
        assert!(!lo.has_shape(3, 2), "low parts at the wrong length");
        lo.mean_lo = Vec::new();
        assert!(!lo.has_shape(3, 2), "absent low parts");
    }

    /// The best split read after the scale folded is the split of the same
    /// weights unfolded: the stump's gain from the brute-force sums, which
    /// never fold (review 2026-09-26, B missing 3).
    #[test]
    fn the_split_gain_survives_a_fold() {
        let stump = |sums: &[[f64; 3]]| -> f64 {
            let w: f64 = sums.iter().map(|s| s[0]).sum();
            let sy: f64 = sums.iter().map(|s| s[1]).sum();
            let mean = sy / w;
            let total = sums.iter().map(|s| s[2]).sum::<f64>() - w * mean * mean;
            let (mut best, mut wl, mut sl) = (0.0f64, 0.0, 0.0);
            for s in &sums[..sums.len() - 1] {
                wl += s[0];
                sl += s[1];
                let (wr, sr) = (w - wl, sy - sl);
                if wl > 0.0 && wr > 0.0 {
                    let between = wl * (sl / wl - mean).powi(2) + wr * (sr / wr - mean).powi(2);
                    best = best.max(between / total);
                }
            }
            best
        };
        for (lam, rows, folded) in [(0.9, 300, false), (0.74, 2000, true)] {
            let (b, sums) = brute(lam, rows, &[-0.5, 0.0, 0.5]);
            // `decay` folds whenever the scale would fall below `RENORM_AT`,
            // so the stream folded exactly when its decay passes that point.
            let passes = lam.powi(rows as i32 - 1) < RENORM_AT;
            assert_eq!(passes, folded, "lam {lam}, scale {}", b.scale);
            assert!(b.scale >= RENORM_AT, "scale {}", b.scale);
            let got = b.best_split(0, 0).unwrap().gain;
            let want = stump(&sums);
            assert!(want > 0.5, "the cut explains most of a line: {want}");
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "lam {lam}: {got} vs {want}"
            );
        }
    }

    /// `Fixed` edges against a level the widths round away at: the sweep
    /// leaves the one distinct edge, strictly increasing and above the
    /// smallest value, which `MarginalBins::new` takes.
    #[test]
    fn fixed_edges_collapse_against_a_large_level() {
        let mut vals = vec![(1e16, 1.0), (1e16 + 2.0, 1.0)];
        let edges = edges_from(BinRule::Fixed, 8, &mut vals);
        assert_eq!(edges, vec![1e16 + 2.0]);
        assert!(MarginalBins::new(1, 1, vec![edges]).is_ok());
    }

    mod generated {
        use super::*;
        use proptest::prelude::*;

        /// A warm-up sample as a stream can give it: values mostly ordinary,
        /// with repeats, junk and the input bound; weights mostly ordinary,
        /// with zeros, subnormals, huge ones and junk.
        fn sample() -> impl Strategy<Value = Vec<(f64, f64)>> {
            let value = prop_oneof![
                5 => -3.0..3.0f64,
                2 => Just(0.0),
                1 => Just(f64::NAN),
                1 => Just(f64::INFINITY),
                1 => (-1.0..1.0f64).prop_map(|u| u * 1e100),
            ];
            let weight = prop_oneof![
                5 => 0.1..2.0f64,
                1 => Just(0.0),
                1 => Just(5e-324),
                1 => Just(1e300),
                1 => Just(f64::NAN),
            ];
            prop::collection::vec((value, weight), 0..60)
        }

        // proptest's own count, 256, or `PROPTEST_CASES`, which the
        // essentials gate sets low (docs/TESTING.md, "Two tiers").
        proptest! {
            #![proptest_config(ProptestConfig::default())]

            /// Whatever the sample, the edges are finite, strictly
            /// increasing, above the smallest usable value, no more than
            /// asked for, and a histogram takes them (review 2026-09-26, B
            /// missing 6).
            #[test]
            fn edges_from_gives_edges_a_histogram_accepts(
                mut s in sample(),
                n_bins in 2usize..12,
                fixed in any::<bool>(),
            ) {
                let rule = if fixed { BinRule::Fixed } else { BinRule::Quantile };
                let lo = s
                    .iter()
                    .filter(|(v, w)| v.is_finite() && w.is_finite() && *w > 0.0)
                    .map(|(v, _)| *v)
                    .fold(f64::INFINITY, f64::min);
                let edges = edges_from(rule, n_bins, &mut s);
                prop_assert!(edges.len() < n_bins);
                prop_assert!(edges.iter().all(|e| e.is_finite() && *e > lo), "{edges:?}");
                prop_assert!(edges.windows(2).all(|w| w[1] > w[0]), "{edges:?}");
                prop_assert!(MarginalBins::new(1, 1, vec![edges]).is_ok());
            }
        }
    }

    /// `BinStat::Huber`: each bin's location is Huber's M-location, one
    /// pass (the module doc; docs/PLAN.md task 272 (c)).
    mod huber {
        use super::*;

        const HUBER: BinStat = BinStat::Huber { delta: 1.0 };

        /// A stream of one feature against two targets, the second absent
        /// on a fifth of the rows, with weights (zeros among them), spikes
        /// of 40 on one row in twenty, and a decay of `lam` a row.
        fn rows(n: usize) -> Vec<(f64, [Option<f64>; 2], f64)> {
            let mut seed = 11u64;
            (0..n)
                .map(|i| {
                    let x = lcg(&mut seed);
                    let spike = if (lcg(&mut seed) + 1.0) * 10.0 < 1.0 {
                        40.0
                    } else {
                        0.0
                    };
                    let y0 = 2.0 * x * x + 0.3 * lcg(&mut seed) + spike;
                    let y1 = (lcg(&mut seed) > -0.6).then(|| -x + 0.2 * lcg(&mut seed));
                    let w = if i % 17 == 3 {
                        0.0
                    } else {
                        0.5 + 0.5 * (i % 3) as f64
                    };
                    (x, [Some(y0), y1], w)
                })
                .collect()
        }

        fn run(stat: BinStat, lam: f64, n: usize) -> MarginalBins {
            let mut b = MarginalBins::new_with(1, 2, vec![vec![-0.3, 0.3]], stat).unwrap();
            for (i, (x, y, w)) in rows(n).into_iter().enumerate() {
                if i > 0 {
                    b.decay(lam);
                }
                b.update_row(&[x], &y, w);
            }
            b
        }

        /// The stream of [`rows`] with each target times `c`, through a
        /// histogram whose MAD sketches are anchored or not.
        fn run_scaled(n: usize, c: f64, anchored: bool) -> MarginalBins {
            let edges = vec![vec![-0.3, 0.3]];
            let mut b = MarginalBins::new_with_anchor(1, 2, edges, HUBER, anchored).unwrap();
            for (i, (x, y, w)) in rows(n).into_iter().enumerate() {
                if i > 0 {
                    b.decay(0.97);
                }
                b.update_row(&[x], &y.map(|v| v.map(|v| v * c)), w);
            }
            b
        }

        /// Anchored sketches (task 278, the user's 2026-10-26): a decimal
        /// scale of the targets moves every cell's location by rounding,
        /// where the fixed buckets moved the scale, and every cut and
        /// location it judged, by up to a bucket. The unanchored move is
        /// printed and held above rounding as evidence that the cuts bound.
        #[test]
        fn anchored_sketches_scale_the_locations_by_rounding() {
            let n = 2000;
            let (base, fixed) = (run_scaled(n, 1.0, true), run_scaled(n, 1.0, false));
            assert!(base.mad_anchor() && !fixed.mad_anchor());
            let spread = base.mean.iter().fold(0.0f64, |a, m| a.max(m.abs()));
            let (mut worst_anchored, mut worst_fixed) = (0.0f64, 0.0f64);
            for c in [1e-6, 1e-3, 1e3, 1e6] {
                let (a, f) = (run_scaled(n, c, true), run_scaled(n, c, false));
                for i in 0..base.mean.len() {
                    let d = (a.mean[i] - c * base.mean[i]).abs() / (c * spread);
                    worst_anchored = worst_anchored.max(d);
                    let d = (f.mean[i] - c * fixed.mean[i]).abs() / (c * spread);
                    worst_fixed = worst_fixed.max(d);
                }
            }
            println!(
                "location move over the spread: anchored {worst_anchored:e}, fixed {worst_fixed:e}"
            );
            assert!(worst_anchored < 1e-13, "{worst_anchored:e}");
            assert!(
                worst_fixed > 1e-9,
                "the fixed buckets moved {worst_fixed:e}"
            );
        }

        /// `delta = inf` clips nothing, and a cell nothing was clipped in
        /// takes the mean's operations: the histogram is the mean's to the
        /// bit, through folds, spikes, zero weights and absent targets.
        #[test]
        fn an_infinite_delta_is_the_mean_to_the_bit() {
            let inf = BinStat::Huber {
                delta: f64::INFINITY,
            };
            for lam in [1.0, 0.97, 0.74] {
                let (mean, huber) = (run(BinStat::Mean, lam, 2000), run(inf, lam, 2000));
                assert_eq!(huber.stat(), inf);
                for (a, b) in [
                    (&mean.w, &huber.w),
                    (&mean.mean, &huber.mean),
                    (&mean.m2, &huber.m2),
                    (&mean.mean_lo, &huber.mean_lo),
                    // Every row counted at its full weight.
                    (&mean.w, &huber.om),
                ] {
                    let bits = |v: &Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(a), bits(b), "lam {lam}");
                }
                assert_eq!(mean.scale.to_bits(), huber.scale.to_bits());
            }
        }

        /// Until a target has a scale -- its first two rows -- nothing is
        /// clipped and the bins are the mean's to the bit; the third row's
        /// spike is the first a scale judges.
        #[test]
        fn nothing_is_clipped_before_the_target_has_a_scale() {
            let ys = [1.0, 1.5, 500.0];
            let xs = [-0.5, -0.4, -0.45];
            let mut m = MarginalBins::new(1, 1, vec![vec![0.0]]).unwrap();
            let mut h = MarginalBins::new_with(1, 1, vec![vec![0.0]], HUBER).unwrap();
            for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
                let judge = h.scales.as_ref().unwrap().judge(0);
                assert_eq!(judge.cut.is_finite(), i == 2, "row {i}: {judge:?}");
                m.update_row(&[*x], &[Some(y)], 1.0);
                h.update_row(&[*x], &[Some(y)], 1.0);
                let (a, b) = (m.bins(0, 0)[0], h.bins(0, 0)[0]);
                if i < 2 {
                    assert_eq!(a.mean_y.to_bits(), b.mean_y.to_bits(), "row {i}");
                    assert_eq!(a.var_y.to_bits(), b.var_y.to_bits(), "row {i}");
                } else {
                    // The scale is |1.5 - 1.0| / Φ⁻¹(3/4), within the
                    // sketch's 0.78%; the spike is clipped at it, and counts
                    // for `cut/|e|` of a row: `Ω' = 2 + cut/|e|`.
                    let cut = judge.cut;
                    assert!((cut - 0.5 / crate::robust::NORMAL_Q3).abs() < 0.008 * cut);
                    let om = 2.0 + cut / (500.0 - 1.25);
                    assert!((b.mean_y - (1.25 + cut / om)).abs() < 1e-12, "{b:?}");
                    assert!(a.mean_y > 160.0, "the mean takes the spike: {a:?}");
                }
                assert_eq!(
                    a.n.to_bits(),
                    b.n.to_bits(),
                    "the count is the rows' weight"
                );
            }
        }

        /// A thin bin and one spike: the mean moves by the spike's size
        /// times its share, the location by the cut times its share, and
        /// no more. The target's two thick bins at -5 and +5 give it a
        /// scale near 7.4, which the thin bin's noise never reaches, so
        /// every one of its rows counted at full weight before the spike.
        #[test]
        fn a_spike_in_a_thin_bin_moves_the_location_by_the_cut_times_its_share() {
            let edges = vec![vec![-1.0, 1.0]];
            let mut m = MarginalBins::new(1, 1, edges.clone()).unwrap();
            let mut h = MarginalBins::new_with(1, 1, edges, HUBER).unwrap();
            let mut seed = 5u64;
            for i in 0..420 {
                let (x, level) = match i % 21 {
                    0 => (0.0, 0.0),
                    k if k % 2 == 0 => (-2.0, -5.0),
                    _ => (2.0, 5.0),
                };
                let y = level + 0.01 * lcg(&mut seed);
                m.update_row(&[x], &[Some(y)], 1.0);
                h.update_row(&[x], &[Some(y)], 1.0);
            }
            let thin = |b: &MarginalBins| b.bins(0, 0)[1];
            assert_eq!(thin(&h).n, 20.0);
            assert_eq!(h.om[1].to_bits(), h.w[1].to_bits(), "nothing clipped yet");
            let cut = h.scales.as_ref().unwrap().judge(0).cut;
            assert!((5.0..12.0).contains(&cut), "the target's scale: {cut}");
            let (m0, h0) = (thin(&m).mean_y, thin(&h).mean_y);
            m.update_row(&[0.0], &[Some(1000.0)], 1.0);
            h.update_row(&[0.0], &[Some(1000.0)], 1.0);
            let (dm, dh) = (thin(&m).mean_y - m0, thin(&h).mean_y - h0);
            assert!((dm - (1000.0 - m0) / 21.0).abs() < 1e-9, "the mean: {dm}");
            // At most `c·w/W`, `W = 20` the bin's weight before the row,
            // every row of which counted in full: the step is `c·w/Ω'`, the
            // spike counting for `c/|e|` of a row in `Ω' = 20 + c/|e|`.
            assert!(dh > 0.0 && dh <= cut / 20.0, "{dh} against {cut}");
            let om = 20.0 + cut / (1000.0 - h0);
            assert!(
                (dh - cut / om).abs() < 1e-12 * cut,
                "{dh} against {}",
                cut / om
            );
            assert_eq!(thin(&h).n, 21.0, "the spike still counts as a row");
        }

        /// The step's bound is `c·w/Ω'`, the bin's weight after the row,
        /// `λΩ + ω`, and not `c·w/Ω`: under a decay of `λ = 0.5` the spike
        /// moves the thin bin by `c/(10 + c/|e|)`, nearly twice `c/20`
        /// (docs/PLAN.md task 272 (e), R6-1).
        #[test]
        fn under_decay_the_step_is_bounded_by_the_weight_after_the_row() {
            let edges = vec![vec![-1.0, 1.0]];
            let mut h = MarginalBins::new_with(1, 1, edges, HUBER).unwrap();
            let mut seed = 5u64;
            for i in 0..420 {
                let (x, level) = match i % 21 {
                    0 => (0.0, 0.0),
                    k if k % 2 == 0 => (-2.0, -5.0),
                    _ => (2.0, 5.0),
                };
                h.update_row(&[x], &[Some(level + 0.01 * lcg(&mut seed))], 1.0);
            }
            let thin = |b: &MarginalBins| b.bins(0, 0)[1];
            assert_eq!(thin(&h).n, 20.0, "Ω = W = 20: nothing clipped yet");
            assert_eq!(h.om[1].to_bits(), h.w[1].to_bits());
            h.decay(0.5);
            let cut = h.scales.as_ref().unwrap().judge(0).cut;
            let h0 = thin(&h).mean_y;
            h.update_row(&[0.0], &[Some(1000.0)], 1.0);
            let dh = thin(&h).mean_y - h0;
            let om = 0.5 * 20.0 + cut / (1000.0 - h0);
            assert!(
                (dh - cut / om).abs() < 1e-12 * cut,
                "{dh} against c/Ω' = {}",
                cut / om
            );
            assert!(
                dh > 1.9 * cut / 20.0,
                "{dh} is not above c·w/Ω = {}",
                cut / 20.0
            );
        }

        /// The scale trick by brute force: every bin's and every target's
        /// weights decayed every row, in the definition's own terms -- the
        /// row's weight `ω = w·min(1, c/|e|)` from the cut and the location
        /// before the row, the location stepped over `Ω'`, a bin's first row
        /// winsorized at the target's location, the second moment
        /// `w·ψ²σ²·λΩ/Ω'` -- against the histogram's undecayed weights and
        /// folds, through spikes, zero weights, an absent target and, at
        /// `lam = 0.74`, several folds. The sketches are the same component
        /// on both sides, aged and fed alike.
        #[test]
        fn the_decayed_recursion_by_brute_force() {
            let BinStat::Huber { delta } = HUBER else {
                unreachable!("HUBER is a Huber")
            };
            for lam in [1.0, 0.97, 0.74] {
                let real = run(HUBER, lam, 2000);
                // [t][bin]: W, Ω, m, M2.
                let mut cells = [[[0.0f64; 4]; 3]; 2];
                let mut targets = [[0.0f64; 2]; 2]; // Ω_t, m_t
                let mut mad: Vec<EwQuantile> = (0..2)
                    .map(|_| EwQuantile::bounded(&[0.5], crate::robust::MAD_BAND_OCTAVES).unwrap())
                    .collect();
                let (mut clipped, mut seeded_at_target, mut learned) = (0, 0, false);
                for (i, (x, y, w)) in rows(2000).into_iter().enumerate() {
                    if i > 0 && learned {
                        for c in cells.iter_mut().flatten() {
                            c[0] *= lam;
                            c[1] *= lam;
                            c[3] *= lam;
                        }
                        for t in targets.iter_mut() {
                            t[0] *= lam;
                        }
                        mad.iter_mut().for_each(|q| q.age(lam));
                    }
                    if w <= 0.0 {
                        continue;
                    }
                    learned = true;
                    let k = [-0.3, 0.3].partition_point(|e: &f64| *e <= x);
                    for t in 0..2 {
                        let Some(v) = y[t] else { continue };
                        let cut =
                            crate::robust::mad_scale(&mad[t]).map_or(f64::INFINITY, |s| delta * s);
                        let has = targets[t][0] > 0.0;
                        let c = &mut cells[t][k];
                        if c[0] <= 0.0 {
                            let e = v - targets[t][1];
                            if has && e.abs() > cut {
                                *c = [w, w * cut / e.abs(), targets[t][1] + cut.copysign(e), 0.0];
                                seeded_at_target += 1;
                            } else {
                                *c = [w, w, v, 0.0];
                            }
                        } else {
                            let e = v - c[2];
                            let om = w * (cut / e.abs()).min(1.0);
                            clipped += usize::from(om < w);
                            let ob = c[1] + om;
                            let step = (om / ob) * e;
                            let psi = e.clamp(-cut, cut);
                            c[3] += w * psi * psi * (c[1] / ob);
                            c[2] += step;
                            c[1] = ob;
                            c[0] += w;
                        }
                        if has {
                            let e = v - targets[t][1];
                            mad[t].add(e.abs(), w);
                            let om = w * (cut / e.abs()).min(1.0);
                            targets[t][0] += om;
                            targets[t][1] += (om / targets[t][0]) * e;
                        } else {
                            targets[t] = [w, v];
                        }
                    }
                }
                assert!(
                    clipped > 100,
                    "lam {lam}: the case needs clipped rows: {clipped}"
                );
                assert!(
                    seeded_at_target > 0 || lam == 1.0,
                    "lam {lam}: a seed at the target"
                );
                for (t, cells_t) in cells.iter().enumerate() {
                    for (k, got) in real.bins(t, 0).iter().enumerate() {
                        let [w, _, m, m2] = cells_t[k];
                        assert!(
                            (got.n - w).abs() <= 1e-9 * w,
                            "lam {lam} t {t} bin {k}: weight"
                        );
                        assert!(
                            (got.mean_y - m).abs() <= 1e-9 * (1.0 + m.abs()),
                            "lam {lam} t {t} bin {k}: {} {m}",
                            got.mean_y
                        );
                        assert!(
                            (got.var_y - m2 / w).abs() <= 1e-7 * (1.0 + m2 / w),
                            "lam {lam} t {t} bin {k}: variance"
                        );
                    }
                }
            }
        }

        /// A clock gap past `gap_cap` wipes the bins, and the targets'
        /// locations and scales with them: the next row starts a target
        /// over, judged against nothing.
        #[test]
        fn a_total_gap_wipes_the_scales_too() {
            let mut h = run(HUBER, 0.9, 300);
            assert!(h.scales.as_ref().unwrap().judge(0).cut.is_finite());
            h.decay(0.0);
            assert!(h.is_empty());
            let s = h.scales.as_ref().unwrap();
            assert_eq!(s.judge(0), Judge::NONE);
            assert_eq!(s.mad[0].get(0), None);
        }

        /// A restored histogram must carry the robust part its statistic
        /// asks for, of the right shape.
        #[test]
        fn has_shape_holds_the_robust_part() {
            let h = run(HUBER, 0.97, 50);
            assert!(h.has_shape(1, 2));
            let mut short = h.clone();
            short.om.pop();
            assert!(!short.has_shape(1, 2));
            let mut none = h.clone();
            none.scales = None;
            assert!(
                !none.has_shape(1, 2),
                "cells' robust weights with no scales"
            );
            let mut targets = h.clone();
            targets.scales.as_mut().unwrap().w.pop();
            assert!(!targets.has_shape(1, 2));
            for bad in [f64::NAN, 0.0, -1.0] {
                let mut delta = h.clone();
                delta.scales.as_mut().unwrap().delta = bad;
                assert!(!delta.has_shape(1, 2), "a cut of {bad}");
            }
            // A target's weight is a finite number, not below 0.
            for bad in [-1.0, f64::INFINITY, f64::NAN] {
                let mut weight = h.clone();
                weight.scales.as_mut().unwrap().w[1] = bad;
                assert!(!weight.has_shape(1, 2), "a target's weight of {bad}");
            }
            let mut empty = h;
            empty.scales.as_mut().unwrap().w[1] = 0.0;
            assert!(empty.has_shape(1, 2), "a target with no weight yet");
            assert!(run(BinStat::Mean, 0.97, 50).has_shape(1, 2));
        }

        /// The budget counts the fifth value a cell keeps, and the targets'
        /// scales at their most.
        #[test]
        fn the_budget_counts_the_robust_weights() {
            let (p, t) = (6, 3);
            let mut h = MarginalBins::new_with(p, t, vec![vec![0.0, 1.0]; p], HUBER).unwrap();
            // The row's offsets are allocated by the first row.
            h.update_row(&vec![0.5; p], &vec![Some(1.0); t], 1.0);
            let cells =
                histogram_bytes(p, t, 3 * p, HUBER_CELL_VALUES) - t * SCALES_BYTES_PER_TARGET;
            assert_eq!(h.heap_bytes(), cells);
            assert_eq!(
                cells - histogram_bytes(p, t, 3 * p, CELL_VALUES),
                3 * p * t * std::mem::size_of::<f64>()
            );
            let cfg = |budget_mib| BinCfg {
                n_bins: 3,
                edges: Some(vec![vec![0.0, 1.0]; p]),
                rule: BinRule::Quantile,
                warm_rows: 3,
                stat: HUBER,
                refit_rows: 0,
                budget_mib: Some(budget_mib),
                mad_anchor: false,
            };
            let bytes = histogram_bytes(p, t, 3 * p, HUBER_CELL_VALUES) as f64;
            assert!(cfg(bytes / (1u64 << 20) as f64).validate(p, t).is_ok());
            assert!(
                cfg(0.999 * bytes / (1u64 << 20) as f64)
                    .validate(p, t)
                    .is_err()
            );
        }

        /// A cut must be a positive number; `inf` is the mean.
        #[test]
        fn delta_is_above_zero() {
            for bad in [0.0, -1.0, f64::NAN] {
                let e = MarginalBins::new_with(1, 1, vec![vec![]], BinStat::Huber { delta: bad });
                assert!(
                    e.unwrap_err().contains("bin_huber_delta must be > 0"),
                    "{bad}"
                );
            }
            assert!(
                MarginalBins::new_with(
                    1,
                    1,
                    vec![vec![]],
                    BinStat::Huber {
                        delta: f64::INFINITY
                    }
                )
                .is_ok()
            );
            // The spec's check, before any histogram is made, says the same.
            let spec = |delta| BinCfg {
                n_bins: 3,
                edges: Some(vec![vec![0.0, 1.0]]),
                rule: BinRule::Quantile,
                warm_rows: 3,
                stat: BinStat::Huber { delta },
                refit_rows: 0,
                budget_mib: None,
                mad_anchor: false,
            };
            for bad in [0.0, -1.0, f64::NAN] {
                let e = spec(bad).validate(1, 1).unwrap_err();
                assert!(e.contains("bin_huber_delta must be > 0"), "{bad}: {e}");
            }
            for good in [1e-9, 1.0, f64::INFINITY] {
                assert!(spec(good).validate(1, 1).is_ok(), "{good}");
            }
            // The refit is of Huber locations at a finite cut (task 281):
            // asked for under the mean or an infinite delta, it is refused;
            // off, every statistic is taken.
            let refit = |stat, refit_rows| BinCfg {
                stat,
                refit_rows,
                ..spec(1.0)
            };
            for stat in [
                BinStat::Mean,
                BinStat::Huber {
                    delta: f64::INFINITY,
                },
            ] {
                let e = refit(stat, 1).validate(1, 1).unwrap_err();
                assert!(e.contains("bin_refit_rows applies only"), "{stat:?}: {e}");
                assert!(refit(stat, 0).validate(1, 1).is_ok(), "{stat:?}");
            }
            assert!(
                refit(BinStat::Huber { delta: 1.0 }, 1)
                    .validate(1, 1)
                    .is_ok()
            );
        }

        /// Deviations whose span widens a bucket at a time: up, and both
        /// ways at once.
        fn widening(mode: usize, i: usize) -> f64 {
            let k = i as f64 / 64.0;
            match mode {
                0 => (0.25 * k).exp2(),
                _ if i.is_multiple_of(2) => (0.1 * k).exp2(),
                _ => -(-0.1 * k).exp2(),
            }
        }

        /// The budget's bound on a target's location and scale is what they
        /// allocate at their most, to the byte: three doubles, and a sketch
        /// of one level whose buckets fill its band, reached here by
        /// deviations whose span widens a bucket at a time. A bound under
        /// it would let a spec through whose model takes more than its
        /// budget; one over it refuses specs that fit (docs/PLAN.md task
        /// 272 (m)).
        #[test]
        fn the_scales_bound_is_what_a_targets_scale_allocates_at_its_most() {
            let mut h = MarginalBins::new_with(1, 2, vec![vec![0.0]], HUBER).unwrap();
            let mut most = [0usize; 2];
            for i in 0..20_000 {
                h.update_row(&[-1.0], &[Some(widening(0, i)), Some(widening(1, i))], 1.0);
                let sc = h.scales.as_ref().unwrap();
                for (t, m) in most.iter_mut().enumerate() {
                    // Each of `w`, `m` and `m_lo` holds one double a target.
                    assert_eq!([&sc.w, &sc.m, &sc.m_lo].map(|v| v.capacity()), [2, 2, 2]);
                    *m = (*m).max(3 * std::mem::size_of::<f64>() + sc.mad[t].bytes());
                }
            }
            let sc = h.scales.as_ref().unwrap();
            for (t, most) in most.iter().enumerate() {
                let (len, _) = sc.mad[t].buckets_held();
                assert_eq!(
                    len,
                    2 * 64 * crate::robust::MAD_BAND_OCTAVES as usize + 1,
                    "target {t}: the band filled"
                );
                assert_eq!(*most, SCALES_BYTES_PER_TARGET, "target {t}");
            }
        }

        /// The bound holds while the median of a target's deviations is 0
        /// too: ten rows at exactly the target's location, then one a hair
        /// off it and one far off. With half the weight on deviations of
        /// exactly 0 the sketch kept no band, and those two spanned 127,563
        /// buckets, a megabyte where the budget counted 8 KB (docs/PLAN.md
        /// task 272 (m2)).
        #[test]
        fn the_scales_bound_holds_while_the_median_deviation_is_zero() {
            let mut h = MarginalBins::new_with(1, 1, vec![vec![0.0]], HUBER).unwrap();
            for y in std::iter::repeat_n(0.0, 10).chain([1e-300, 1e300]) {
                h.update_row(&[-1.0], &[Some(y)], 1.0);
                let sc = h.scales.as_ref().unwrap();
                let bytes = 3 * std::mem::size_of::<f64>() + sc.mad[0].bytes();
                assert!(bytes <= SCALES_BYTES_PER_TARGET, "{y}: {bytes} bytes");
            }
            let sc = h.scales.as_ref().unwrap();
            assert_eq!(sc.mad[0].get(0), Some(0.0), "the median deviation is 0");
            assert_eq!(sc.judge(0).cut, f64::INFINITY, "so there is no cut");
        }

        /// The cut is inclusive: a bin's first row exactly the cut from the
        /// target's location is inside it, so it seeds the bin at its own
        /// value, at its full weight, as the location's `ψ(e) = e` for
        /// `|e| <= c` has it. Seeded as a clipped row would be, from the
        /// location stepped by the cut, it is the same number in exact
        /// arithmetic and not to the bit: the case is searched for where the
        /// two part, so the test reads which one the code took.
        #[test]
        fn a_first_row_exactly_at_the_cut_seeds_the_bin_at_its_own_value() {
            let mut h = MarginalBins::new_with(1, 1, vec![vec![0.0]], HUBER).unwrap();
            let mut seed = 17u64;
            let mut found = None;
            for _ in 0..400 {
                let level = if lcg(&mut seed) > 0.0 { 7.0 } else { 1.0 };
                h.update_row(&[-1.0], &[Some(level + 0.3 * lcg(&mut seed))], 1.0);
                let j = h.scales.as_ref().unwrap().judge(0);
                if !j.cut.is_finite() {
                    continue;
                }
                let below = j.m - j.cut;
                found = (-64..=64).find_map(|k: i64| {
                    let y = f64::from_bits((below.to_bits() as i64 + k) as u64);
                    let e = crate::comp::dev(y, j.m, j.m_lo);
                    let (mut hi, mut lo) = (j.m, j.m_lo);
                    crate::comp::add(&mut hi, &mut lo, j.cut.copysign(e));
                    (e.abs() == j.cut && hi.to_bits() != y.to_bits()).then_some((y, j))
                });
                if found.is_some() {
                    break;
                }
            }
            let (y, j) = found.expect("a row exactly at the cut where the two forms part");
            assert!(j.has && j.m_lo != 0.0, "{j:?}");
            assert!(h.w[1] == 0.0, "the second bin is empty");
            h.update_row(&[1.0], &[Some(y)], 1.0);
            let bin = h.bins(0, 0)[1];
            assert_eq!(
                bin.mean_y.to_bits(),
                y.to_bits(),
                "{} against {y}",
                bin.mean_y
            );
            assert_eq!(h.om[1], 1.0, "at its full weight");
            assert_eq!(bin.var_y, 0.0);
        }

        /// A clipped row's weight `u·c/|e|` can round to 0: then it takes no
        /// step, and counts in the bin's row weight alone. Here a bin's two
        /// rows, at a weight of `1e-300` and `1e30` from a target whose cut
        /// is under 1, each weigh under the least subnormal: the first
        /// seeds the bin at the location stepped by the cut, with no weight,
        /// and the second, judged against a weight of 0, leaves it there,
        /// where `c·u/0` would have taken it to infinity (hard rule 9).
        #[test]
        fn a_clipped_row_whose_weight_rounds_to_nothing_takes_no_step() {
            let mut h = MarginalBins::new_with(1, 1, vec![vec![0.0]], HUBER).unwrap();
            let mut seed = 23u64;
            for i in 0..40 {
                let y = f64::from(i % 2) + 0.01 * lcg(&mut seed);
                h.update_row(&[-1.0], &[Some(y)], 1.0);
            }
            let j = h.scales.as_ref().unwrap().judge(0);
            assert!(j.cut > 0.1 && j.cut < 1.0, "{j:?}");
            let (u, y) = (1e-300, 1e30);
            assert_eq!(
                u * (j.cut / (y - j.m)),
                0.0,
                "the clipped weight rounds to 0"
            );
            h.update_row(&[1.0], &[Some(y)], u);
            assert_eq!((h.w[1], h.om[1]), (u, 0.0), "seeded, clipped to no weight");
            let seeded = h.bins(0, 0)[1];
            assert!((seeded.mean_y - (j.m + j.cut)).abs() < 1e-12, "{seeded:?}");
            h.update_row(&[1.0], &[Some(y)], u);
            let bin = h.bins(0, 0)[1];
            assert_eq!(bin.mean_y.to_bits(), seeded.mean_y.to_bits(), "{bin:?}");
            assert_eq!(bin.var_y, 0.0);
            assert_eq!(h.om[1], 0.0);
            assert_eq!(bin.n, 2.0 * u, "both rows count in the row weight");
        }
    }
}
