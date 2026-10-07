//! Exponentially weighted k-means (docs/CLUSTERING.md §6.2; docs/PLAN.md
//! §11a, task 23): hard assignment to the nearest of `k` centres under a
//! diagonal metric read from the EW feature moments, mean-form centre
//! updates through a batch summary merged every `update_every` learned rows,
//! and a split–merge check every `split_merge_every` learned rows that frees the
//! emptier of the two closest clusters — or a dead one — and re-places it on
//! the far rows of the cluster that has collected the most of them since the
//! last check (ISODATA's split, on the rows themselves).
//!
//! ```text
//! metric      mw_i = 1 / v_i  (EW variance of feature i; 1 where v_i = 0)
//! assign      j* = argmin_j ‖z − c_j‖²_mw                 (first minimum wins)
//! outputs     cluster = j*,  dist = ‖z − c_j*‖_mw,  dist2 = the runner-up's
//! far         d² > f · R̃,  f = 1 + FAR_SIGMAS sqrt(2/p)
//! batch       B_j* ← absorb_plain(z, w, d²)  unless far   (mean form, §6.1)
//!             F_j* ← absorb(z, w)  when far   (Welford, §6.1);  V += w either way
//! checkpoint  C_j ← merge_plain(B_j), B_j ← ∅         every update_every rows
//!             R̃ ← mean of the trusted positive r2_j leaving out the largest (none: nothing is far)
//! check       every split_merge_every rows: r2_j ← (n_j r2_j + n(F_j) cut) / (n_j + n(F_j)), R̃ again
//!             closest pair (i, j) with d_ij < split_merge (r_i + r_j)
//!             → l = argmax_l n(F_l) counting F_j as F_i's, only if n(F_l) ≥ FAR_SHARE · V
//!             and rows(F_l) ≥ FAR_ROWS: C_i ← merge_welford(C_j), F_i ← merge_welford(F_j)
//!             else j = argmin n_j if n_j < dead_frac · n_eff / k, l = argmax_l n(F_l) if any
//!             then n_l /= 2, C_j ← (c(F_l), n_l, R̃);  every F ← ∅, V ← 0
//! decay       n_j *= lam, W *= lam, V *= lam, buffered w *= lam   lam = 0.5^(d/half_life)
//! ```
//!
//! A row is *far* when its squared distance to its centre exceeds what a
//! blob-shaped cluster of the typical radius² `R̃` produces: `d² ~ R̃ χ²_p
//! / p` has mean `R̃` and standard deviation `R̃ sqrt(2/p)`, and the cut
//! sits [`FAR_SIGMAS`] of those above the mean. `R̃` leaves out the largest
//! radius² so that a cluster swollen by rows it should not own does not
//! hide them, and counts only radii from at least [`RADIUS_ROWS`] rows;
//! until one cluster has such a radius nothing is far, or every row would
//! be. A far row is not learned into the cluster at all — not its centre,
//! its radius or its weight — but into the cluster's far summary: a row
//! the cluster should not own must not drag it, widen the radius that is
//! the yardstick of the far cut and of the closest-pair ratio, or keep a
//! centre that has lost its rows alive. The one thing far rows do to the
//! cluster is at the check: they count in its radius as if they sat at the
//! cut, with their weight `F` against its own `n`, so a burst of them
//! widens it a little, and a cluster whose rows are all far widens by
//! `(n + f F) / (n + F)` per check. The cut is `f` times the *typical*
//! radius, so that reaches the rows once the typical radius widens with
//! it: with `k = 1`, or when every cluster sees far rows — a cut the
//! metric left behind, or every blob moving at once — so a cut that has
//! fallen below the data cannot keep every row out for good. One blob
//! that jumped among `k ≥ 2` stays far (its cluster's widening radius is
//! the largest, which `R̃` leaves out) until the dead rule re-places its
//! centre, on its own far rows if they are the heaviest,
//! `log2(1/dead_frac)` half-lives after the jump — 4.3 at the default,
//! 2 at `dead_frac = 0.25`. (A cluster wider than the typical learns the
//! rows within the cut and offers the rest: where a centre comes free,
//! that is where it goes.) Far rows are summarised per cluster, not kept.
//! The freed centre is placed at the heaviest summary's mean with the
//! typical radius and half the source's weight, so a component born far
//! from every centre is found in one move; a merge, which costs a cluster,
//! wants a component's worth of far rows first — at least [`FAR_ROWS`] of
//! them weighing [`FAR_SHARE`] of what was learned since the last check
//! (`V`) — while a dead centre, already lost, goes to whatever far rows
//! there are. What the rule cannot see is a cluster that owns two blobs:
//! its rows are all within its own radius, and their far mean, when any
//! are far, is its own centre. Seeding with `lloyd` (the best of
//! [`LLOYD_RESTARTS`] k-means++ starts by inertia) is what keeps that from
//! happening.
//!
//! Seeding applies the same far rule to the buffer as a whole: rows whose
//! squared distance to the EW mean exceeds `f` times the buffer's mean
//! squared distance do not choose the seeds (the whole buffer does when the
//! rest cannot give `k` distinct seeds), and are replayed as far rows.
//!
//! Every output is read *before* the row is learned, so `cluster` is an
//! out-of-sample assignment (CLAUDE.md rule 2), and `n_eff` is the EW weight
//! before the row and before its own decay (rule 8). Seeding waits for
//! `warm_rows` learned rows, at least `k`, held whole in a buffer of that
//! many rows, so the warm-up holds `O(warm_rows·p)` -- at most
//! [`KMeans::WARM_BUDGET_MIB`] -- and the stream after it `O(k·p)`; until
//! then every output is null. Under `first`, which waits for `k` distinct
//! rows, the buffer goes on growing until it has them or holds
//! `max(warm_rows, BUF_CAP)` rows ([`KMeans::BUF_CAP`]), where duplicates
//! are taken. Standardization scales the metric, never the
//! coordinates (docs/CLUSTERING.md §10), so the centres stay in the features'
//! own units and `coef` reads as `k` rows of `p` feature values.

use serde::{Deserialize, Serialize};

use super::summary::{ClusterSummary, FeatureMoments, LONG_HALFLIVES, SplitMix64, dist, dist2};
use crate::clock::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// How the first `k` centres are chosen from the warm-up buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedRule {
    /// The first `k` distinct rows.
    First,
    /// Gonzalez' farthest-first traversal from the first row.
    Farthest,
    /// k-means++ (Arthur & Vassilvitskii 2007), weighted by the buffered
    /// row weights, from a `splitmix64` stream keyed by `seed`.
    Kmeanspp,
    /// The best of [`LLOYD_RESTARTS`] runs of k-means++ followed by
    /// [`LLOYD_ITERS`] weighted Lloyd iterations on the buffer, by the
    /// weighted sum of squared distances over the buffer (first minimum
    /// wins). One run lands in a split-one-blob, merge-two local optimum
    /// a third of the time on five blobs in four dimensions; its cost is
    /// ~1.8x the right partition's, so the restarts tell them apart.
    #[default]
    Lloyd,
}

/// Configuration for [`KMeans`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KMeansCfg {
    pub n_features: usize,
    /// Number of clusters, `1 ..= MAX_K`.
    pub k: usize,
    pub decay: Decay,
    /// Outputs are null while `n_eff < min_weight`.
    pub min_weight: f64,
    /// Learned rows buffered before seeding, at least `k`; the buffer of
    /// that many rows is held to [`KMeans::WARM_BUDGET_MIB`].
    pub warm_rows: usize,
    pub seed_rule: SeedRule,
    /// Seed of the generator behind `kmeanspp` / `lloyd`.
    pub seed: u64,
    /// Learned rows between checkpoints (batch → centre merges), `>= 1`.
    pub update_every: u32,
    /// Merge the two closest clusters when their centre distance is below
    /// this many summed radii; `0` disables split–merge and the dead rule.
    pub split_merge: f64,
    /// Learned rows between split–merge checks, `>= 1`.
    pub split_merge_every: u32,
    /// A cluster lighter than `dead_frac · n_eff / k` at a check is dead
    /// and re-placed on the far rows; `0` disables the rule. A centre
    /// whose blob vanished gets there `log2(1/dead_frac)` half-lives later
    /// (4.3 at 0.05, 2 at 0.25); a blob lighter than `dead_frac / k` of
    /// the stream loses its centre whenever any row is far. The rule runs
    /// at a split–merge check, so above 0 it is refused beside
    /// `split_merge = 0`, which runs none.
    pub dead_frac: f64,
    /// Measure distances in units of each feature's EW standard deviation.
    pub standardize: bool,
    /// Floor the metric's variance at this fraction of the feature's
    /// long-run variance (`FeatureMoments`' reference), so that a feature
    /// quiet for `Q` half-lives comes to count `2^(Q/8) / scale_floor` times
    /// what its history says, where `1 / var` alone gave `2^Q`; `0` is the
    /// EW variance alone, and what a state written before the floor loads
    /// with (docs/PLAN.md task 102).
    #[serde(default)]
    pub scale_floor: f64,
}

impl KMeansCfg {
    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("kmeans: {e}"))?;
        if self.n_features == 0 {
            return Err("kmeans: n_features must be >= 1".into());
        }
        if self.k == 0 {
            return Err("kmeans: k must be >= 1".into());
        }
        // A count the spec sizes the centres and the seeding by before the
        // stream says anything (review 2026-10-06, CF2).
        if self.k > KMeans::MAX_K {
            return Err(format!(
                "kmeans: k must be at most {} (2^16), got {}: every row is scored against every \
                 centre, and seeding is O(k²) over the warm-up buffer",
                KMeans::MAX_K,
                self.k
            ));
        }
        // The buffer seeds `k` centres from `warm_rows` rows; fewer rows
        // were floored to `k` in silence, where `hmm` refuses them (review
        // 2026-10-06, PC8).
        if self.warm_rows < self.k {
            return Err(format!(
                "kmeans: warm_rows must be at least k ({}) to seed that many centres, got {}",
                self.k, self.warm_rows
            ));
        }
        let mib = KMeans::warm_mib(self.warm_rows, self.n_features);
        if mib > KMeans::WARM_BUDGET_MIB {
            return Err(format!(
                "kmeans: the warm-up buffer would hold {} MiB (warm_rows {} rows of {} features), \
                 over the {} MiB it is held to; lower warm_rows (at least k), or narrow the \
                 features",
                crate::budget::mib_over(mib, KMeans::WARM_BUDGET_MIB),
                self.warm_rows,
                self.n_features,
                KMeans::WARM_BUDGET_MIB
            ));
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err("kmeans: min_weight must be >= 0".into());
        }
        if self.update_every == 0 {
            return Err("kmeans: update_every must be >= 1".into());
        }
        if self.split_merge_every == 0 {
            return Err("kmeans: split_merge_every must be >= 1".into());
        }
        if !self.split_merge.is_finite() || self.split_merge < 0.0 {
            return Err("kmeans: split_merge must be finite and >= 0".into());
        }
        if !self.dead_frac.is_finite() || self.dead_frac < 0.0 {
            return Err("kmeans: dead_frac must be finite and >= 0".into());
        }
        // The dead rule runs inside the split–merge check (review
        // 2026-10-06, CF6).
        if self.split_merge == 0.0 && self.dead_frac > 0.0 {
            return Err(format!(
                "kmeans: dead_frac re-places a centre at a split-merge check, and split_merge = 0 \
                 runs none, so dead_frac = {} would do nothing; give split_merge > 0, or \
                 dead_frac = 0",
                self.dead_frac
            ));
        }
        if !self.scale_floor.is_finite() || self.scale_floor < 0.0 {
            return Err("kmeans: scale_floor must be finite and >= 0".into());
        }
        Ok(())
    }
}

/// How many standard deviations of a blob's `χ²_p` squared distance a row
/// must sit above the radius² to count as far; see the [module docs](self).
pub const FAR_SIGMAS: f64 = 4.0;

/// A cluster's far rows must weigh at least this share of the weight learned
/// since the last split–merge check for a merge to place its freed centre on
/// them; a lighter summary is strays, not a component. See the [module
/// docs](self).
pub const FAR_SHARE: f64 = 0.05;

/// Rows a cluster must have learned before its radius enters the typical
/// radius; see the [module docs](self).
pub const RADIUS_ROWS: u64 = 10;

/// Far rows a summary must hold, besides its [`FAR_SHARE`] of the window,
/// for a merge to place its freed centre on them; see the [module docs](self).
pub const FAR_ROWS: u64 = 3;

/// Exponentially weighted k-means; see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KMeans {
    cfg: KMeansCfg,
    /// EW mean/variance per feature; its weight is `n_eff`.
    moments: FeatureMoments,
    /// The metric for the next row, refreshed at the end of each step.
    mw: Vec<f64>,
    /// Empty until seeded, then exactly `k`.
    clusters: Vec<ClusterSummary>,
    /// Rows learned into each cluster (a re-placed one starts at
    /// [`RADIUS_ROWS`]): the count behind its radius.
    rows: Vec<u64>,
    /// Rows learned since the last checkpoint, per cluster.
    batch: Vec<ClusterSummary>,
    /// Warm-up rows and their decayed weights; cleared at seeding.
    buf: Vec<Vec<f64>>,
    buf_w: Vec<f64>,
    /// The far rows of each cluster since the last split–merge check, as
    /// Welford summaries about their own mean; empty until seeded, then
    /// exactly `k`, and only fed while `split_merge > 0`.
    far: Vec<ClusterSummary>,
    /// Rows in each far summary.
    far_rows: Vec<u64>,
    /// `1 + FAR_SIGMAS sqrt(2/p)`: the far cut in units of the radius².
    far_factor: f64,
    /// The typical radius² `R̃` and the far cut `far_factor · R̃`, refreshed
    /// at every checkpoint (0 and infinite until seeded).
    r2_typical: f64,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    far_cut: f64,
    /// Weight learned since the last split–merge check.
    window_w: f64,
    /// Learned rows since the last checkpoint / split–merge check.
    since: u32,
    since_sm: u32,
    n_merges: u64,
    n_dead: u64,
}

impl KMeans {
    /// Where the `first` seeding rule, which waits for `k` distinct rows,
    /// gives up on distinctness and seeds with what it has: once the buffer
    /// holds `max(warm_rows, BUF_CAP)` rows. Not a cap on the buffer, which
    /// holds `warm_rows` rows whatever this is.
    pub const BUF_CAP: usize = 1000;

    /// The most centres a model takes. Each row is scored against every
    /// centre, and seeding is `O(k²)` over a buffer of at least `k` rows
    /// (k-means++ and the Lloyd restarts), so a `k` past this is a spec's
    /// mistake rather than a workload: `k = 10^9` was accepted and held that
    /// many rows before seeding (review 2026-10-06, CF2).
    pub const MAX_K: usize = 1 << 16;

    /// The most MiB the warm-up buffer may hold ([`Self::warm_mib`]), as a
    /// marginal's bins, a window's snapshots and a held Gram block are held
    /// to 256 MiB: `warm_rows = 10^9` was accepted (review 2026-10-06, CF2).
    pub const WARM_BUDGET_MIB: f64 = 256.0;

    /// The MiB a warm-up buffer of `rows` rows of `p` features takes: each
    /// row's vector, its `p` values and its weight. A double, so no count
    /// overflows.
    pub fn warm_mib(rows: usize, p: usize) -> f64 {
        let row = std::mem::size_of::<Vec<f64>>() as f64 + (p as f64 + 1.0) * 8.0;
        rows as f64 * row / f64::from(1 << 20)
    }

    pub fn new(cfg: KMeansCfg) -> Result<Self, String> {
        cfg.validate()?;
        let p = cfg.n_features;
        Ok(Self {
            moments: FeatureMoments::new(p),
            mw: vec![1.0; p],
            clusters: Vec::new(),
            rows: Vec::new(),
            batch: Vec::new(),
            buf: Vec::new(),
            buf_w: Vec::new(),
            far: Vec::new(),
            far_rows: Vec::new(),
            far_factor: 1.0 + FAR_SIGMAS * (2.0 / p as f64).sqrt(),
            r2_typical: 0.0,
            far_cut: f64::INFINITY,
            window_w: 0.0,
            since: 0,
            since_sm: 0,
            n_merges: 0,
            n_dead: 0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &KMeansCfg {
        &self.cfg
    }

    /// EW weight of the learned rows: the model's `n_eff`.
    pub fn n_eff(&self) -> f64 {
        self.moments.w
    }

    /// True once the centres exist.
    pub fn seeded(&self) -> bool {
        !self.clusters.is_empty()
    }

    /// The live clusters (empty before seeding).
    pub fn clusters(&self) -> &[ClusterSummary] {
        &self.clusters
    }

    /// The metric in force for the next row.
    pub fn metric(&self) -> &[f64] {
        &self.mw
    }

    pub fn moments(&self) -> &FeatureMoments {
        &self.moments
    }

    /// Rows held for seeding (empty once seeded).
    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Split–merge events so far: `(merges, dead re-placements)`.
    pub fn events(&self) -> (u64, u64) {
        (self.n_merges, self.n_dead)
    }

    /// The centres as `k` rows of `p` feature values; `None` before seeding.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        self.seeded()
            .then(|| self.clusters.iter().map(|c| c.c.clone()).collect())
    }

    /// Nearest and runner-up: `(j*, d²_j*, d²_second, j_second)`. First
    /// minimum wins; the runner-up is NaN, and `None`, when `k = 1`. Among
    /// squares that overflowed the overflow-free norms decide, so a row at
    /// the input bound still has a nearest and a runner-up (review
    /// 2026-09-26, G1: an infinite square never won against the infinity
    /// the search starts from, and the runner-up was never set).
    fn nearest2(&self, z: &[f64]) -> (usize, f64, f64, Option<usize>) {
        let closer = |d: f64, j: usize, bd: f64, bj: usize| {
            if d.is_finite() || bd.is_finite() {
                d < bd
            } else {
                dist(&self.clusters[j].c, z, &self.mw) < dist(&self.clusters[bj].c, z, &self.mw)
            }
        };
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        let mut second = f64::INFINITY;
        let mut second_j = None;
        for (j, c) in self.clusters.iter().enumerate() {
            let d = dist2(&c.c, z, &self.mw);
            if j == 0 || closer(d, j, best_d, best) {
                second = best_d;
                second_j = (j > 0).then_some(best);
                best_d = d;
                best = j;
            } else if second_j.is_none_or(|sj| closer(d, j, second, sj)) {
                second = d;
                second_j = Some(j);
            }
        }
        if self.clusters.len() < 2 {
            second = f64::NAN;
            second_j = None;
        }
        (best, best_d, second, second_j)
    }

    /// The distance to centre `j` from its squared distance: the root where
    /// that is finite, the overflow-free norm where the square overflowed
    /// (review 2026-09-26, G1; `summary::dist`).
    fn reported(&self, d2: f64, j: usize, z: &[f64]) -> f64 {
        if d2.is_finite() || d2.is_nan() {
            d2.sqrt()
        } else {
            dist(&self.clusters[j].c, z, &self.mw)
        }
    }

    fn score(&self, x: &[f64], valid: bool, n_eff: f64) -> Vec<f64> {
        let mut pred = vec![f64::NAN; 3];
        if valid && self.seeded() && n_eff >= self.cfg.min_weight {
            let (j, d2, d2_second, second_j) = self.nearest2(x);
            pred[0] = j as f64;
            pred[1] = self.reported(d2, j, x);
            pred[2] = match second_j {
                Some(sj) => self.reported(d2_second, sj, x),
                None => f64::NAN,
            };
        }
        pred
    }

    fn learn_row(&mut self, x: &[f64], w: f64) {
        let (j, d2, _, _) = self.nearest2(x);
        let far = self.cfg.split_merge > 0.0 && self.is_far(d2);
        self.absorb(j, x, w, d2, far);
        self.since += 1;
        self.since_sm += 1;
        if self.since >= self.cfg.update_every {
            self.checkpoint();
        }
    }

    /// Whether a row at squared distance `d2` from its centre is far.
    fn is_far(&self, d2: f64) -> bool {
        d2 > self.far_cut
    }

    /// Learn one row into cluster `j`'s batch — or, when far, into its far
    /// summary instead. The window weight counts every row while the rule
    /// is on.
    fn absorb(&mut self, j: usize, x: &[f64], w: f64, d2: f64, far: bool) {
        if self.cfg.split_merge > 0.0 {
            self.window_w += w;
        }
        if far {
            self.far[j].absorb(x, w, &self.mw);
            self.far_rows[j] = self.far_rows[j].saturating_add(1);
        } else {
            self.batch[j].absorb_plain(x, w, d2);
            self.rows[j] = self.rows[j].saturating_add(1);
        }
    }

    /// Merge every batch into its cluster and refresh the far cut; then,
    /// when due, the split–merge check, which always consumes the far rows.
    fn checkpoint(&mut self) {
        for (c, b) in self.clusters.iter_mut().zip(&mut self.batch) {
            c.merge_plain(b);
            b.n = 0.0;
            b.r2 = 0.0;
            b.c.iter_mut().for_each(|v| *v = 0.0);
        }
        self.since = 0;
        if self.cfg.split_merge > 0.0 {
            self.refresh_far_cut();
            if self.since_sm >= self.cfg.split_merge_every {
                self.since_sm = 0;
                self.winsorize_radii();
                self.refresh_far_cut();
                self.split_merge_check();
                self.window_w = 0.0;
                self.far_rows.iter_mut().for_each(|r| *r = 0);
                for f in &mut self.far {
                    f.n = 0.0;
                    f.r2 = 0.0;
                    f.c.iter_mut().for_each(|v| *v = 0.0);
                }
            }
        }
    }

    /// Far rows count in the radius as if they sat at the cut, with their
    /// weight against the cluster's: a burst of them widens it a little; a
    /// cluster whose rows are all far — a blob that jumped, or a cut the
    /// metric left behind — widens it by `far_factor` per check until its
    /// rows are within reach again.
    fn winsorize_radii(&mut self) {
        // While no cluster has a trusted radius there is no cut to place a
        // far row at. `try_seed` marks rows far by the buffer's own cut, so
        // the pool can be non-empty here before any cluster has
        // `RADIUS_ROWS`; folding it at a cut of ∞ set the radius to ∞, and
        // the next refresh a typical radius of ∞ or NaN (review 2026-09-18,
        // B5). The pool is cleared by the caller either way.
        if !self.far_cut.is_finite() {
            return;
        }
        for (c, f) in self.clusters.iter_mut().zip(&self.far) {
            if f.n > 0.0 {
                c.r2 = (c.n * c.r2 + f.n * self.far_cut) / (c.n + f.n);
            }
        }
    }

    /// `R̃` = the mean over the trusted positive radii² leaving out the
    /// largest (that one itself when it is alone), and `cut = far_factor ·
    /// R̃`; while no cluster has such a radius nothing is far, or every row
    /// would be.
    fn refresh_far_cut(&mut self) {
        let (mut sum, mut max, mut count) = (0.0, 0.0f64, 0usize);
        for (c, &rows) in self.clusters.iter().zip(&self.rows) {
            if rows >= RADIUS_ROWS && c.r2 > 0.0 {
                sum += c.r2;
                max = max.max(c.r2);
                count += 1;
            }
        }
        (self.r2_typical, self.far_cut) = match count {
            0 => (0.0, f64::INFINITY),
            1 => (sum, sum * self.far_factor),
            n => {
                let typical = (sum - max) / (n - 1) as f64;
                (typical, typical * self.far_factor)
            }
        };
    }

    /// The cluster holding the most far weight (the first maximum wins);
    /// `None` when no cluster has any.
    fn far_source(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (l, f) in self.far.iter().enumerate() {
            if f.n > 0.0 && best.is_none_or(|b| f.n > self.far[b].n) {
                best = Some(l);
            }
        }
        best
    }

    /// [`far_source`](Self::far_source) for a merge of `j` into `i`, which
    /// pools `j`'s far rows with `i`'s and, as it costs a cluster, wants a
    /// component's worth of them: at least [`FAR_ROWS`] rows and
    /// [`FAR_SHARE`] of the window.
    fn merge_source(&self, i: usize, j: usize) -> Option<usize> {
        let pooled = self.far[i].n + self.far[j].n;
        let mut best = (i, pooled, self.far_rows[i].saturating_add(self.far_rows[j]));
        for (l, f) in self.far.iter().enumerate() {
            if l != i && l != j && f.n > best.1 {
                best = (l, f.n, self.far_rows[l]);
            }
        }
        let (l, n, rows) = best;
        (n > 0.0 && rows >= FAR_ROWS && n >= FAR_SHARE * self.window_w).then_some(l)
    }

    /// ISODATA's split on the far rows themselves: `source` keeps half its
    /// weight, `target` takes the other half at the far rows' mean with the
    /// typical radius (`target == source` re-places a cluster on its own far
    /// rows).
    fn split(&mut self, target: usize, source: usize) {
        self.clusters[source].n *= 0.5;
        let n = self.clusters[source].n;
        let c = self.far[source].c.clone();
        self.clusters[target] = ClusterSummary::at(c, n, self.r2_typical);
        self.rows[target] = RADIUS_ROWS;
    }

    fn split_merge_check(&mut self) {
        let k = self.clusters.len();
        if k >= 3 {
            // The closest pair, in summed radii; a pair with no radius at
            // all is never "close".
            let mut best = f64::INFINITY;
            let mut pair = (0, 0);
            for i in 0..k {
                for j in (i + 1)..k {
                    let ri = self.clusters[i].r2.max(0.0).sqrt();
                    let rj = self.clusters[j].r2.max(0.0).sqrt();
                    let den = ri + rj;
                    let d = dist2(&self.clusters[i].c, &self.clusters[j].c, &self.mw).sqrt();
                    // Coincident centres are one cluster whatever their
                    // radii; apart, a pair with no radius is never close.
                    let ratio = if d == 0.0 {
                        0.0
                    } else if den > 0.0 {
                        d / den
                    } else {
                        f64::INFINITY
                    };
                    if ratio < best {
                        best = ratio;
                        pair = (i, j);
                    }
                }
            }
            if best < self.cfg.split_merge {
                let (i, j) = pair;
                // Without far rows to put the freed centre on, the merge
                // alone would only lose a cluster. The far rows of `j` are
                // `i`'s once the two are one.
                let Some(source) = self.merge_source(i, j) else {
                    return;
                };
                let other = self.clusters[j].clone();
                self.clusters[i].merge_welford(&other, &self.mw);
                self.rows[i] = self.rows[i].saturating_add(self.rows[j]);
                let far_j =
                    std::mem::replace(&mut self.far[j], ClusterSummary::empty(other.c.len()));
                self.far[i].merge_welford(&far_j, &self.mw);
                self.far_rows[i] = self.far_rows[i].saturating_add(self.far_rows[j]);
                self.far_rows[j] = 0;
                self.split(j, source);
                self.n_merges += 1;
                return;
            }
        }
        if k >= 2 && self.cfg.dead_frac > 0.0 {
            let mut jd = 0;
            for j in 1..k {
                if self.clusters[j].n < self.clusters[jd].n {
                    jd = j;
                }
            }
            let floor = self.cfg.dead_frac * self.moments.w / k as f64;
            if self.clusters[jd].n < floor
                && let Some(source) = self.far_source()
            {
                self.split(jd, source);
                self.n_dead += 1;
            }
        }
    }

    /// The buffer's far cut: `far_factor` times its weighted mean squared
    /// distance to the EW mean, which is the buffer's own weighted mean.
    fn buffer_cut(&self) -> f64 {
        let (mut sum, mut wsum) = (0.0, 0.0);
        for (z, &w) in self.buf.iter().zip(&self.buf_w) {
            sum += w * dist2(&self.moments.mean, z, &self.mw);
            wsum += w;
        }
        let mean = if wsum > 0.0 { sum / wsum } else { 0.0 };
        mean * self.far_factor
    }

    /// Seed once the buffer is full enough — from the rows within the
    /// buffer's far cut, when at least `k` are — then replay the whole
    /// buffer through the frozen seeds so the centres start as the means of
    /// their rows, the rows beyond the cut as far rows.
    fn try_seed(&mut self) {
        let k = self.cfg.k;
        let target = self.cfg.warm_rows.max(k);
        if self.buf.len() < target {
            return;
        }
        let allow_dup = self.buf.len() >= target.max(Self::BUF_CAP);
        let cut = self.buffer_cut();
        let kept: Vec<usize> = (0..self.buf.len())
            .filter(|&i| dist2(&self.moments.mean, &self.buf[i], &self.mw) <= cut)
            .collect();
        let mut seeds = None;
        if kept.len() >= k && kept.len() < self.buf.len() {
            let buf: Vec<Vec<f64>> = kept.iter().map(|&i| self.buf[i].clone()).collect();
            let w: Vec<f64> = kept.iter().map(|&i| self.buf_w[i]).collect();
            seeds = seed_centres(
                &buf,
                &w,
                k,
                self.cfg.seed_rule,
                self.cfg.seed,
                allow_dup,
                &self.mw,
            )
            .filter(|c| distinct(c));
        }
        let seeds = seeds.or_else(|| {
            seed_centres(
                &self.buf,
                &self.buf_w,
                k,
                self.cfg.seed_rule,
                self.cfg.seed,
                allow_dup,
                &self.mw,
            )
        });
        let Some(seeds) = seeds else {
            return;
        };
        let p = self.cfg.n_features;
        self.clusters = seeds
            .into_iter()
            .map(|c| ClusterSummary::at(c, 0.0, 0.0))
            .collect();
        self.rows = vec![0; k];
        self.batch = vec![ClusterSummary::empty(p); k];
        self.far = vec![ClusterSummary::empty(p); k];
        self.far_rows = vec![0; k];
        let buf = std::mem::take(&mut self.buf);
        let buf_w = std::mem::take(&mut self.buf_w);
        for (z, &w) in buf.iter().zip(&buf_w) {
            let (j, d2, _, _) = self.nearest2(z);
            let far = self.cfg.split_merge > 0.0 && dist2(&self.moments.mean, z, &self.mw) > cut;
            self.absorb(j, z, w, d2, far);
        }
        for (c, b) in self.clusters.iter_mut().zip(&mut self.batch) {
            c.merge_plain(b);
            b.n = 0.0;
            b.r2 = 0.0;
            b.c.iter_mut().for_each(|v| *v = 0.0);
        }
        if self.cfg.split_merge > 0.0 {
            self.refresh_far_cut();
        }
    }
}

/// Whether no two of the centres coincide.
fn distinct(centres: &[Vec<f64>]) -> bool {
    centres
        .iter()
        .enumerate()
        .all(|(i, a)| centres[..i].iter().all(|b| a != b))
}

/// The seeding rules over a buffer of rows with weights `w`, under the
/// metric `mw`. `None` only for `first` when fewer than `k` distinct rows
/// exist and duplicates are not yet allowed.
pub(crate) fn seed_centres(
    buf: &[Vec<f64>],
    w: &[f64],
    k: usize,
    rule: SeedRule,
    seed: u64,
    allow_dup: bool,
    mw: &[f64],
) -> Option<Vec<Vec<f64>>> {
    if buf.len() < k {
        return None;
    }
    match rule {
        SeedRule::First => {
            let mut out: Vec<Vec<f64>> = Vec::with_capacity(k);
            for z in buf {
                if !out.iter().any(|c| c == z) {
                    out.push(z.clone());
                    if out.len() == k {
                        return Some(out);
                    }
                }
            }
            allow_dup.then(|| buf[..k].to_vec())
        }
        SeedRule::Farthest => Some(farthest(buf, k, mw)),
        SeedRule::Kmeanspp => {
            let mut rng = SplitMix64::new(seed);
            Some(kmeanspp(buf, w, k, mw, &mut rng))
        }
        SeedRule::Lloyd => {
            // One generator across the restarts, so the draws are one
            // stream keyed by `seed` and the reference can replay them.
            let mut rng = SplitMix64::new(seed);
            let mut best: Option<(f64, Vec<Vec<f64>>)> = None;
            for _ in 0..LLOYD_RESTARTS {
                let init = kmeanspp(buf, w, k, mw, &mut rng);
                let centres = lloyd(buf, w, init, mw, LLOYD_ITERS);
                let cost = inertia(buf, w, &centres, mw);
                if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                    best = Some((cost, centres));
                }
            }
            best.map(|(_, c)| c)
        }
    }
}

/// Lloyd iterations the `lloyd` rule runs on the buffer, per restart.
pub(crate) const LLOYD_ITERS: usize = 10;

/// k-means++ starts the `lloyd` rule tries, keeping the lowest
/// [`inertia`].
pub(crate) const LLOYD_RESTARTS: usize = 10;

/// The weighted sum over the buffer of each row's squared distance to its
/// nearest centre: the k-means objective the restarts are ranked by.
fn inertia(buf: &[Vec<f64>], w: &[f64], centres: &[Vec<f64>], mw: &[f64]) -> f64 {
    let mut total = 0.0;
    for (z, &wi) in buf.iter().zip(w) {
        let mut best = f64::INFINITY;
        for c in centres {
            let d = dist2(c, z, mw);
            if d < best {
                best = d;
            }
        }
        total += wi * best;
    }
    total
}

/// `dd_i ← min(dd_i, ‖z_i − c‖²)` for a newly chosen centre `c`.
fn shrink(dd: &mut [f64], buf: &[Vec<f64>], c: &[f64], mw: &[f64]) {
    for (d, z) in dd.iter_mut().zip(buf) {
        let q = dist2(c, z, mw);
        if q < *d {
            *d = q;
        }
    }
}

fn farthest(buf: &[Vec<f64>], k: usize, mw: &[f64]) -> Vec<Vec<f64>> {
    let mut out = vec![buf[0].clone()];
    let mut dd = vec![f64::INFINITY; buf.len()];
    shrink(&mut dd, buf, &buf[0], mw);
    while out.len() < k {
        let mut idx = 0;
        for i in 1..dd.len() {
            if dd[i] > dd[idx] {
                idx = i;
            }
        }
        out.push(buf[idx].clone());
        shrink(&mut dd, buf, &buf[idx], mw);
    }
    out
}

fn kmeanspp(
    buf: &[Vec<f64>],
    w: &[f64],
    k: usize,
    mw: &[f64],
    rng: &mut SplitMix64,
) -> Vec<Vec<f64>> {
    let idx0 = rng.choice(w);
    let mut out = vec![buf[idx0].clone()];
    let mut dd = vec![f64::INFINITY; buf.len()];
    shrink(&mut dd, buf, &buf[idx0], mw);
    let mut pr = vec![0.0; buf.len()];
    while out.len() < k {
        for i in 0..pr.len() {
            pr[i] = w[i] * dd[i];
        }
        let idx = rng.choice(&pr);
        out.push(buf[idx].clone());
        shrink(&mut dd, buf, &buf[idx], mw);
    }
    out
}

/// Weighted Lloyd iterations: assign (first minimum wins), then each centre
/// becomes the weighted mean of its rows, sums taken in row order; a centre
/// with no weight stays put.
fn lloyd(
    buf: &[Vec<f64>],
    w: &[f64],
    mut centres: Vec<Vec<f64>>,
    mw: &[f64],
    iters: usize,
) -> Vec<Vec<f64>> {
    let k = centres.len();
    let p = mw.len();
    let mut sw = vec![0.0; k];
    let mut sx = vec![vec![0.0; p]; k];
    for _ in 0..iters {
        sw.iter_mut().for_each(|v| *v = 0.0);
        sx.iter_mut()
            .for_each(|s| s.iter_mut().for_each(|v| *v = 0.0));
        for (z, &wi) in buf.iter().zip(w) {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (j, c) in centres.iter().enumerate() {
                let d = dist2(c, z, mw);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            sw[best] += wi;
            for (s, &zi) in sx[best].iter_mut().zip(z) {
                *s += wi * zi;
            }
        }
        for j in 0..k {
            if sw[j] > 0.0 {
                for (c, &s) in centres[j].iter_mut().zip(&sx[j]) {
                    *c = s / sw[j];
                }
            }
        }
    }
    centres
}

impl OnlineModel for KMeans {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature or a weight past the input bound was
        // learned, the feature as a point, which no centre leaves.
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        let lam = self.cfg.decay.factor(d_clock);
        let n_before = self.moments.w;
        let valid = x.iter().all(|v| v.is_finite());
        let learn = weight > 0.0 && weight.is_finite() && valid;

        // The clock passes for everything the model holds.
        self.moments
            .decay(lam, self.cfg.decay.factor(d_clock / LONG_HALFLIVES));
        for c in &mut self.clusters {
            c.decay(lam);
        }
        for b in &mut self.batch {
            b.decay(lam);
        }
        for f in &mut self.far {
            f.decay(lam);
        }
        self.window_w *= lam;
        for w in &mut self.buf_w {
            *w *= lam;
        }

        // Decay moves no centre and the metric is refreshed only below, so
        // this is exactly what `predict` reads.
        let pred = self.score(x, valid, n_before);

        if learn {
            self.moments.absorb(x, weight);
            if self.seeded() {
                self.learn_row(x, weight);
            } else {
                self.buf.push(x.to_vec());
                self.buf_w.push(weight);
                self.try_seed();
            }
        }
        self.moments
            .metric(self.cfg.standardize, self.cfg.scale_floor, &mut self.mw);

        Step {
            pred,
            n_eff: n_before,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let valid = x.iter().all(|v| v.is_finite());
        Step {
            pred: self.score(x, valid, self.moments.w),
            n_eff: self.moments.w,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::KMeans(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::KMeans(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("kmeans", m.cfg.validate())?;
                // Empty until seeded, then exactly `k` of each per-cluster
                // vector, every centre and buffered row `p` wide (review
                // 2026-09-18, B3).
                let (p, k) = (m.cfg.n_features, m.cfg.k);
                let count = if m.clusters.is_empty() { 0 } else { k };
                let sums =
                    |s: &[ClusterSummary]| s.len() == count && s.iter().all(|c| c.c.len() == p);
                if !m.moments.has_shape(p)
                    || m.mw.len() != p
                    || !sums(&m.clusters)
                    || !sums(&m.batch)
                    || !sums(&m.far)
                    || m.rows.len() != count
                    || m.far_rows.len() != count
                    || m.buf_w.len() != m.buf.len()
                    || m.buf.iter().any(|z| z.len() != p)
                {
                    return Err(StateError::Invalid(
                        "kmeans: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "kmeans",
                found: other.kind(),
            }),
        }
    }

    /// Zero: `kmeans` regresses nothing, as `ew_cov` does.
    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// `cluster`, `dist`, `dist2`.
    fn n_outputs(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = KMeans::new(cfg(3)).unwrap();
        let mut s = m.state();
        let ModelState::KMeans(inner) = &mut s.model else {
            unreachable!()
        };
        inner.mw.pop();
        match KMeans::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    /// A state's configuration is held to what `new` holds a fresh one to:
    /// `k = 0` loaded, `lloyd` seeded one centre from the buffer while the
    /// per-cluster counts had none, and `absorb` indexed `rows[0]` on an
    /// ordinary row (review 2026-10-06, CF4).
    #[test]
    fn a_restored_state_whose_cfg_new_refuses_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let c = KMeansCfg {
            seed_rule: SeedRule::Lloyd,
            warm_rows: 4,
            ..cfg(2)
        };
        let mut s = KMeans::new(c).unwrap().state();
        let ModelState::KMeans(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cfg.k = 0;
        assert!(inner.cfg.validate().is_err(), "`new` refuses k = 0");
        match KMeans::restore(&s) {
            Err(StateError::Invalid(e)) => {
                assert!(
                    e.contains("configuration") && e.contains("k must be >= 1"),
                    "{e}"
                );
            }
            Ok(mut back) => {
                for i in 0..=KMeans::BUF_CAP {
                    back.step(&[i as f64, (i % 3) as f64], &[], 1.0, 1.0);
                }
                panic!("k = 0 loaded and ran");
            }
            Err(e) => panic!("{e}"),
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The two distances reported are numbers where their squares overflow:
    /// a metric weight of `2.5e199` (a spread of `4e-200`, which a row of
    /// weight `1e-100` followed by one of `1e100` leaves) against a row at
    /// the input bound (review 2026-09-26, G1).
    #[test]
    fn an_overflowed_distance_is_reported_as_the_distance() {
        use crate::OnlineModel;
        let mut m = KMeans::new(cfg(2)).unwrap();
        let mut s = 11u64;
        for i in 0..12 {
            let x = [lcg(&mut s) + f64::from(i % 2) * 10.0, lcg(&mut s)];
            m.step(&x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.seeded());
        m.mw = vec![2.5e199, 2.5e199];
        let far = m.predict(&[0.0, 1e100], 1.0);
        assert!(
            far.pred[1].is_finite() && far.pred[2].is_finite(),
            "{:?}",
            far.pred
        );
        assert!(
            far.pred[1] > 1e150 && far.pred[1] <= far.pred[2],
            "{:?}",
            far.pred
        );
        let near = m.predict(&[0.2, 0.1], 1.0);
        assert!(
            near.pred[1] <= near.pred[2] && near.pred[2].is_finite(),
            "{:?}",
            near.pred
        );
    }

    fn cfg(k: usize) -> KMeansCfg {
        KMeansCfg {
            n_features: 2,
            k,
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            warm_rows: 6,
            seed_rule: SeedRule::First,
            seed: 0,
            update_every: 1,
            split_merge: 0.0,
            split_merge_every: 1,
            dead_frac: 0.0,
            standardize: false,
            scale_floor: 0.0,
        }
    }

    /// Three well-separated blobs, `n` rows each, interleaved.
    fn blobs(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let centres = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut s = seed;
        let mut out = Vec::new();
        for _ in 0..n {
            for c in &centres {
                out.push([c[0] + 0.5 * lcg(&mut s), c[1] + 0.5 * lcg(&mut s)]);
            }
        }
        out
    }

    /// A far row in the seed buffer is a far summary from the first row
    /// after seeding, while `far_cut` is still `∞` (no cluster has
    /// `RADIUS_ROWS` rows yet). The first split-merge check then folded that
    /// summary into the cluster's radius at the cut -- `∞` -- and the radius
    /// was `∞`, then NaN at the next merge, and the cut with it: every row
    /// far, or none (the review of 2026-09-18, B5). Every radius and the
    /// cut stay finite after every checkpoint, whatever the buffer held.
    #[test]
    fn a_far_row_in_the_seed_buffer_leaves_every_radius_finite() {
        let mut c = cfg(3);
        c.warm_rows = 8;
        c.split_merge = 0.5;
        c.split_merge_every = 2;
        let mut m = KMeans::new(c).unwrap();
        let mut rows = blobs(20, 5);
        rows[3] = [100.0, 100.0];
        for (i, row) in rows.iter().enumerate() {
            m.step(row, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if !m.seeded() {
                continue;
            }
            for (j, cl) in m.clusters.iter().enumerate() {
                assert!(
                    cl.r2.is_finite(),
                    "row {i}: cluster {j} has radius² {}",
                    cl.r2
                );
            }
            assert!(
                m.far_cut.is_finite() || m.far_cut == f64::INFINITY,
                "row {i}: far_cut {}",
                m.far_cut
            );
            assert!(
                m.r2_typical.is_finite(),
                "row {i}: r2_typical {}",
                m.r2_typical
            );
        }
        assert!(m.seeded());
    }

    /// The review (2026-09-12, V25) asked whether `kmeans` replays its warm-up
    /// as `hmm` did, at `lam = 1` with each row's raw weight. It does not: it
    /// ages the buffered weights on every row, so the centres start out
    /// weighing what `n_eff` does. Written as the review's test, and it
    /// passed on the build the review read.
    #[test]
    fn the_seeded_centres_weigh_what_n_eff_does() {
        let mut c = cfg(3);
        c.decay = Decay::Halflife(10.0);
        c.warm_rows = 12;
        let mut m = KMeans::new(c).unwrap();
        let mut s = 9u64;
        for (i, row) in blobs(6, 3).iter().enumerate() {
            let w = 0.5 + lcg(&mut s).abs();
            m.step(row, &[], if i == 0 { 0.0 } else { 1.0 }, w);
            if m.seeded() {
                break;
            }
        }
        assert!(m.seeded(), "twelve rows seed three centres");
        let total: f64 = m.clusters.iter().map(|c| c.n).sum();
        assert!(
            (total - m.n_eff()).abs() <= 1e-12 * m.n_eff(),
            "{total} in the centres, {} in n_eff",
            m.n_eff()
        );
    }

    /// The recursion written out longhand for `first` seeding, per-row
    /// checkpoints and no split–merge, on an irregular clock with weights,
    /// so the test cannot share a mistake with the implementation: seeds are
    /// the first `k` distinct rows, every buffered row is assigned to its
    /// nearest seed with its decayed weight and the centres start as those
    /// weighted means, then every later row moves its centre by `w / n'`.
    #[allow(clippy::type_complexity)]
    fn reference(
        c: &KMeansCfg,
        rows: &[[f64; 2]],
        ds: &[f64],
        ws: &[f64],
    ) -> (Vec<[f64; 3]>, Vec<(f64, Vec<f64>, f64)>, f64) {
        let p = 2;
        let (mut w_sum, mut mean, mut var) = (0.0, vec![0.0; p], vec![0.0; p]);
        let mut mean_lo = vec![0.0; p];
        let mut mw = vec![1.0; p];
        let mut seeds: Option<Vec<(f64, Vec<f64>, f64)>> = None;
        let mut buf: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut preds = Vec::new();
        let d2 = |a: &[f64], b: &[f64], mw: &[f64]| -> f64 {
            let mut acc = 0.0;
            for i in 0..a.len() {
                let t = b[i] - a[i];
                acc += mw[i] * t * t;
            }
            acc
        };
        for ((row, &d), &w) in rows.iter().zip(ds).zip(ws) {
            let lam = c.decay.factor(d);
            let n_before = w_sum;
            w_sum *= lam;
            if let Some(cl) = seeds.as_mut() {
                for s in cl.iter_mut() {
                    s.0 *= lam;
                }
            }
            for b in buf.iter_mut() {
                b.1 *= lam;
            }
            // Read before learning.
            let mut out = [f64::NAN; 3];
            if let Some(cl) = &seeds
                && n_before >= c.min_weight
            {
                let dd: Vec<f64> = cl.iter().map(|s| d2(&s.1, row, &mw)).collect();
                let mut j = 0;
                for i in 1..dd.len() {
                    if dd[i] < dd[j] {
                        j = i;
                    }
                }
                let mut second = f64::INFINITY;
                for (i, &v) in dd.iter().enumerate() {
                    if i != j && v < second {
                        second = v;
                    }
                }
                out = [j as f64, dd[j].sqrt(), second.sqrt()];
            }
            preds.push(out);
            if w > 0.0 {
                // Feature moments (Welford, diagonal).
                let w_new = w_sum + w;
                let (a, b) = (w_sum / w_new, w / w_new);
                for i in 0..p {
                    let dlt = crate::comp::dev(row[i], mean[i], mean_lo[i]);
                    var[i] = a * var[i] + a * b * dlt * dlt;
                    crate::comp::add(&mut mean[i], &mut mean_lo[i], b * dlt);
                }
                w_sum = w_new;
                match seeds.as_mut() {
                    None => {
                        buf.push((row.to_vec(), w));
                        if buf.len() >= c.warm_rows.max(c.k) {
                            let mut cs: Vec<Vec<f64>> = Vec::new();
                            for (z, _) in &buf {
                                if !cs.contains(z) {
                                    cs.push(z.clone());
                                }
                                if cs.len() == c.k {
                                    break;
                                }
                            }
                            if cs.len() == c.k {
                                // Replay: batch means per seed, then merged
                                // into the (weightless) seeds.
                                let mut batch: Vec<(f64, Vec<f64>, f64)> =
                                    cs.iter().map(|_| (0.0, vec![0.0; p], 0.0)).collect();
                                for (z, bw) in &buf {
                                    let dd: Vec<f64> = cs.iter().map(|s| d2(s, z, &mw)).collect();
                                    let mut j = 0;
                                    for i in 1..dd.len() {
                                        if dd[i] < dd[j] {
                                            j = i;
                                        }
                                    }
                                    let n_new = batch[j].0 + bw;
                                    let bb = bw / n_new;
                                    for (ci, &zi) in batch[j].1.iter_mut().zip(z) {
                                        *ci += bb * (zi - *ci);
                                    }
                                    batch[j].2 += bb * (dd[j] - batch[j].2);
                                    batch[j].0 = n_new;
                                }
                                let mut cl: Vec<(f64, Vec<f64>, f64)> =
                                    cs.iter().map(|s| (0.0, s.clone(), 0.0)).collect();
                                for (s, b) in cl.iter_mut().zip(&batch) {
                                    if b.0 > 0.0 {
                                        let n_new = s.0 + b.0;
                                        let bb = b.0 / n_new;
                                        for i in 0..p {
                                            s.1[i] += bb * (b.1[i] - s.1[i]);
                                        }
                                        s.2 += bb * (b.2 - s.2);
                                        s.0 = n_new;
                                    }
                                }
                                seeds = Some(cl);
                                buf.clear();
                            }
                        }
                    }
                    Some(cl) => {
                        let dd: Vec<f64> = cl.iter().map(|s| d2(&s.1, row, &mw)).collect();
                        let mut j = 0;
                        for i in 1..dd.len() {
                            if dd[i] < dd[j] {
                                j = i;
                            }
                        }
                        // One-row batch, then merge: identical to a direct
                        // absorb, which is the point of the mean form.
                        let n_new = cl[j].0 + w;
                        let bb = w / n_new;
                        for (ci, &zi) in cl[j].1.iter_mut().zip(row) {
                            *ci += bb * (zi - *ci);
                        }
                        cl[j].2 += bb * (dd[j] - cl[j].2);
                        cl[j].0 = n_new;
                    }
                }
                for i in 0..p {
                    let inv = 1.0 / var[i];
                    mw[i] = if c.standardize && var[i] > 0.0 && inv.is_finite() {
                        inv
                    } else {
                        1.0
                    };
                }
            }
        }
        (preds, seeds.unwrap_or_default(), w_sum)
    }

    fn same(a: f64, b: f64, what: &str) {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => {}
            (false, false) => assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "{what}: {a} vs {b}"
            ),
            _ => panic!("{what}: {a} vs {b}"),
        }
    }

    #[test]
    fn every_step_matches_the_recursion_written_out() {
        for standardize in [false, true] {
            let rows = blobs(40, 11);
            let mut s = 3u64;
            let ds: Vec<f64> = (0..rows.len())
                .map(|i| {
                    if i == 0 {
                        0.0
                    } else {
                        [1.0, 0.25, 7.0, 1.0, 0.5, 3.0][i % 6]
                    }
                })
                .collect();
            let ws: Vec<f64> = (0..rows.len())
                .map(|i| {
                    if i % 11 == 0 {
                        0.0
                    } else {
                        0.5 + lcg(&mut s).abs()
                    }
                })
                .collect();
            let c = KMeansCfg {
                n_features: 2,
                k: 3,
                decay: Decay::Halflife(30.0),
                min_weight: 4.0,
                warm_rows: 9,
                standardize,
                ..cfg(3)
            };
            let (want, want_clusters, want_w) = reference(&c, &rows, &ds, &ws);
            let mut m = KMeans::new(c).unwrap();
            for (i, ((row, &d), &w)) in rows.iter().zip(&ds).zip(&ws).enumerate() {
                let step = m.step(row, &[], d, w);
                for (s, &w) in want[i].iter().enumerate() {
                    same(
                        step.pred[s],
                        w,
                        &format!("row {i} slot {s} std={standardize}"),
                    );
                }
            }
            assert!((m.n_eff() - want_w).abs() < 1e-12);
            assert_eq!(m.clusters().len(), 3);
            for (got, want) in m.clusters().iter().zip(&want_clusters) {
                same(got.n, want.0, "n");
                same(got.r2, want.2, "r2");
                for i in 0..2 {
                    same(got.c[i], want.1[i], "centre");
                }
            }
        }
    }

    #[test]
    fn a_zero_weight_row_advances_the_clock_and_learns_nothing_even_first() {
        let c = KMeansCfg {
            decay: Decay::Halflife(5.0),
            ..cfg(2)
        };
        let mut m = KMeans::new(c.clone()).unwrap();
        let first = m.step(&[1.0, 2.0], &[], 0.0, 0.0);
        assert_eq!(first.n_eff, 0.0);
        assert!(first.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.buffered(), 0);
        assert!(m.n_eff() == 0.0 && m.moments().mean == vec![0.0, 0.0]);
        // Once seeded, a zero-weight row halves every weight over a
        // half-life and moves no centre.
        let rows = blobs(4, 1);
        for (i, r) in rows.iter().enumerate() {
            m.step(r, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.seeded());
        let before = m.clone();
        let step = m.step(&[100.0, 100.0], &[], 5.0, 0.0);
        assert_eq!(step.n_eff, before.n_eff());
        assert!((m.n_eff() - 0.5 * before.n_eff()).abs() < 1e-12);
        for (a, b) in m.clusters().iter().zip(before.clusters()) {
            assert_eq!(a.c, b.c);
            assert!((a.n - 0.5 * b.n).abs() < 1e-12);
            assert_eq!(a.r2, b.r2);
        }
        assert!(step.pred[0] >= 0.0, "a zero-weight row is still scored");
    }

    #[test]
    fn a_null_or_non_finite_feature_row_is_not_scored_and_not_learned() {
        let mut m = KMeans::new(cfg(2)).unwrap();
        for (i, r) in blobs(4, 2).iter().enumerate() {
            m.step(r, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.clone();
        for bad in [[f64::NAN, 0.0], [0.0, f64::INFINITY]] {
            let step = m.step(&bad, &[], 1.0, 1.0);
            assert!(step.pred.iter().all(|v| v.is_nan()));
            assert_eq!(m.clusters(), before.clusters());
            assert_eq!(m.moments().mean, before.moments().mean);
        }
    }

    #[test]
    fn outputs_are_null_until_seeded_and_until_min_periods() {
        let c = KMeansCfg {
            warm_rows: 6,
            min_weight: 8.0,
            ..cfg(3)
        };
        let mut m = KMeans::new(c).unwrap();
        let rows = blobs(5, 4);
        for (i, r) in rows.iter().enumerate() {
            let step = m.step(r, &[], 1.0, 1.0);
            let ready = i >= 8;
            assert_eq!(step.pred[0].is_nan(), !ready, "row {i}");
            assert_eq!(m.seeded(), i >= 5, "row {i}");
            assert_eq!(step.n_eff, i as f64);
        }
        assert_eq!(m.buffered(), 0);
    }

    #[test]
    fn k_equals_one_has_no_runner_up() {
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 1,
            ..cfg(1)
        })
        .unwrap();
        m.step(&[1.0, 1.0], &[], 0.0, 1.0);
        let step = m.step(&[2.0, 1.0], &[], 1.0, 1.0);
        assert_eq!(step.pred[0], 0.0);
        assert!((step.pred[1] - 1.0).abs() < 1e-12);
        assert!(step.pred[2].is_nan());
        assert_eq!(m.coefficients(), Some(vec![vec![1.5, 1.0]]));
    }

    #[test]
    fn first_seeding_waits_for_k_distinct_rows_until_the_cap() {
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 3,
            ..cfg(3)
        })
        .unwrap();
        for i in 0..50 {
            m.step(&[1.0, (i % 2) as f64], &[], 1.0, 1.0);
            assert!(!m.seeded(), "two distinct rows cannot seed k = 3");
        }
        assert_eq!(m.buffered(), 50);
        m.step(&[5.0, 5.0], &[], 1.0, 1.0);
        assert!(m.seeded());
        assert_eq!(m.buffered(), 0);
        let cs = m.coefficients().unwrap();
        assert_eq!(cs.len(), 3);
        // Seeds were the three distinct rows; after the replay each centre is
        // the mean of the rows nearest to it.
        assert_eq!(cs[2], vec![5.0, 5.0]);

        // At the cap the rule gives up on distinctness.
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 3,
            ..cfg(3)
        })
        .unwrap();
        for _ in 0..KMeans::BUF_CAP {
            m.step(&[1.0, 1.0], &[], 1.0, 1.0);
        }
        assert!(m.seeded());
        assert_eq!(m.buffered(), 0);
        assert_eq!(m.coefficients().unwrap(), vec![vec![1.0, 1.0]; 3]);
        let step = m.step(&[1.0, 1.0], &[], 1.0, 1.0);
        assert_eq!(step.pred[0], 0.0, "ties go to the first centre");
    }

    #[test]
    fn every_seeding_rule_finds_three_separated_blobs() {
        let rows = blobs(200, 7);
        let truth: Vec<usize> = (0..rows.len()).map(|i| i % 3).collect();
        for rule in [
            SeedRule::First,
            SeedRule::Farthest,
            SeedRule::Kmeanspp,
            SeedRule::Lloyd,
        ] {
            for standardize in [false, true] {
                let c = KMeansCfg {
                    warm_rows: 30,
                    seed_rule: rule,
                    standardize,
                    ..cfg(3)
                };
                let mut m = KMeans::new(c).unwrap();
                let mut labels = Vec::new();
                for r in &rows {
                    let step = m.step(r, &[], 1.0, 1.0);
                    labels.push(step.pred[0]);
                }
                // Every assigned row after warm-up sits in a pure cluster:
                // the map blob -> label is a bijection.
                let mut map = [usize::MAX; 3];
                for (i, &l) in labels.iter().enumerate().skip(30) {
                    assert!(!l.is_nan(), "{rule:?}: row {i} unassigned");
                    let l = l as usize;
                    if map[truth[i]] == usize::MAX {
                        map[truth[i]] = l;
                    }
                    assert_eq!(map[truth[i]], l, "{rule:?}: impure cluster at row {i}");
                }
                let mut seen = map.to_vec();
                seen.sort_unstable();
                assert_eq!(seen, vec![0, 1, 2], "{rule:?}: not a bijection");
                for c in m.clusters() {
                    assert!(c.r2 < 0.2, "{rule:?}: r2 = {}", c.r2);
                }
            }
        }
    }

    #[test]
    fn seeding_is_deterministic_in_the_seed_and_moves_with_it() {
        let rows: Vec<[f64; 2]> = {
            let mut s = 9u64;
            (0..400).map(|_| [lcg(&mut s), lcg(&mut s)]).collect()
        };
        let run = |seed: u64| {
            let c = KMeansCfg {
                warm_rows: 400,
                seed_rule: SeedRule::Kmeanspp,
                seed,
                ..cfg(5)
            };
            let mut m = KMeans::new(c).unwrap();
            for r in &rows {
                m.step(r, &[], 1.0, 1.0);
            }
            m.coefficients().unwrap()
        };
        assert_eq!(run(0), run(0));
        assert_ne!(run(0), run(1));
    }

    #[test]
    fn checkpoints_every_n_rows_hold_the_batch_back() {
        // With `update_every = 5`, the centres move only at rows 5, 10, ...
        // after seeding, and the checkpointed centres equal the per-row
        // model's whenever both have just checkpointed (mean form).
        let rows = blobs(30, 5);
        let make = |every| {
            KMeans::new(KMeansCfg {
                warm_rows: 6,
                update_every: every,
                ..cfg(3)
            })
            .unwrap()
        };
        let (mut a, mut b) = (make(1), make(5));
        let mut since = 0;
        for (i, r) in rows.iter().enumerate() {
            let sa = a.step(r, &[], 1.0, 1.0);
            let before = b.clusters().to_vec();
            let sb = b.step(r, &[], 1.0, 1.0);
            if a.seeded() && i >= 6 {
                since += 1;
                if since % 5 == 0 {
                    for (x, y) in a.clusters().iter().zip(b.clusters()) {
                        assert!((x.n - y.n).abs() < 1e-12);
                        for d in 0..2 {
                            assert!((x.c[d] - y.c[d]).abs() < 1e-9, "row {i}");
                        }
                    }
                } else {
                    assert_eq!(
                        b.clusters(),
                        &before[..],
                        "row {i}: moved between checkpoints"
                    );
                }
            }
            // The assignment read before the row agrees whenever the
            // centres do; on this data they agree everywhere.
            assert_eq!(sa.pred[0].is_nan(), sb.pred[0].is_nan());
        }
    }

    #[test]
    fn split_merge_recovers_a_cluster_lost_to_a_regime_change() {
        // Two blobs are seeded with k = 3 (`first` puts two seeds in blob A,
        // which is then split in halves); when a third blob appears every
        // one of its rows is far, the halves are the closest pair, and the
        // check merges them and re-places the freed centre in the new blob.
        // Without split–merge the third blob stays glued to a stale centre.
        let run = |split_merge: f64| {
            // The dead rule runs at a split-merge check, so without one it
            // is 0: above it the pair is refused (review 2026-10-06, CF6).
            let c = KMeansCfg {
                warm_rows: 20,
                split_merge,
                split_merge_every: 20,
                dead_frac: if split_merge > 0.0 { 0.05 } else { 0.0 },
                ..cfg(3)
            };
            let mut m = KMeans::new(c).unwrap();
            let mut s = 21u64;
            for _ in 0..200 {
                m.step(&[0.5 * lcg(&mut s), 0.5 * lcg(&mut s)], &[], 1.0, 1.0);
                m.step(
                    &[10.0 + 0.5 * lcg(&mut s), 0.5 * lcg(&mut s)],
                    &[],
                    1.0,
                    1.0,
                );
            }
            let mut labels = Vec::new();
            for _ in 0..300 {
                let r = [0.5 * lcg(&mut s), 10.0 + 0.5 * lcg(&mut s)];
                labels.push(m.step(&r, &[], 1.0, 1.0).pred[0]);
            }
            (m, labels)
        };
        let (m, labels) = run(1.5);
        let (merges, dead) = m.events();
        assert!(
            merges + dead >= 1,
            "no split–merge event: {:?}",
            m.clusters()
        );
        let last = &labels[labels.len() - 50..];
        assert!(
            last.iter().all(|&l| l == last[0]),
            "third blob not settled: {last:?}"
        );
        assert_eq!(m.clusters().len(), 3, "the count never changes");
        let found = m.clusters().iter().any(|c| (c.c[1] - 10.0).abs() < 1.0);
        assert!(found, "{:?}", m.clusters());

        let (m, _) = run(0.0);
        assert_eq!(m.events(), (0, 0));
        assert!(
            !m.clusters().iter().any(|c| (c.c[1] - 10.0).abs() < 1.0),
            "{:?}",
            m.clusters()
        );
    }

    #[test]
    fn split_merge_merges_two_centres_in_one_blob_only_onto_far_rows() {
        // k = 3 on two blobs where two seeds land in the same blob: the
        // pair is closer than `split_merge` summed radii, but with no far
        // rows to put the freed centre on the merge alone would only lose a
        // cluster, so nothing moves. Once a third blob sends far rows the
        // pair is merged and the freed centre goes to them.
        let c = KMeansCfg {
            warm_rows: 3,
            split_merge: 1.0,
            split_merge_every: 10,
            dead_frac: 0.0,
            ..cfg(3)
        };
        let mut m = KMeans::new(c).unwrap();
        let mut s = 33u64;
        // Seeds: two rows from blob A, one from blob B.
        for r in [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]] {
            m.step(&r, &[], 1.0, 1.0);
        }
        assert!(m.seeded());
        let blob = |m: &mut KMeans, s: &mut u64, x0: f64, y0: f64| {
            m.step(&[x0 + 0.5 * lcg(s), y0 + 0.5 * lcg(s)], &[], 1.0, 1.0);
        };
        for _ in 0..100 {
            blob(&mut m, &mut s, 0.0, 0.0);
            blob(&mut m, &mut s, 10.0, 0.0);
        }
        assert_eq!(m.events(), (0, 0), "{:?}", m.clusters());
        assert!(m.clusters().iter().all(|c| c.n > 0.0));

        for _ in 0..100 {
            blob(&mut m, &mut s, 0.0, 0.0);
            blob(&mut m, &mut s, 10.0, 0.0);
            blob(&mut m, &mut s, 0.0, 10.0);
        }
        let (merges, _) = m.events();
        assert!(merges >= 1, "{:?}", m.clusters());
        assert_eq!(m.clusters().len(), 3, "the count never changes");
        assert!(m.clusters().iter().all(|c| c.n > 0.0));
        let found = m.clusters().iter().any(|c| (c.c[1] - 10.0).abs() < 1.0);
        assert!(found, "{:?}", m.clusters());
    }

    #[test]
    fn a_cut_too_small_for_the_data_widens_until_the_rows_are_learned() {
        // Seeded on a blob a thousandth as wide as the one that follows at
        // the same place, every later row is far. Far rows are not learned,
        // but at each check they count in the radius as if they sat at the
        // cut, so the radius grows by (n + f F) / (n + F) per check (F far
        // weight against n learned, f the factor) until the rows are inside
        // the cut: nothing can trap the model behind its own cut. Without
        // the winsorized radius the state would freeze for good.
        let c = KMeansCfg {
            warm_rows: 50,
            split_merge: 0.5,
            split_merge_every: 50,
            ..cfg(1)
        };
        let mut m = KMeans::new(c).unwrap();
        let mut s = 44u64;
        for _ in 0..100 {
            m.step(&[1e-3 * lcg(&mut s), 1e-3 * lcg(&mut s)], &[], 1.0, 1.0);
        }
        let tight = m.clusters()[0].r2;
        assert!(tight > 0.0 && tight < 1e-5, "{tight}");
        let mut learned = Vec::new();
        for _ in 0..3000 {
            m.step(&[lcg(&mut s), lcg(&mut s)], &[], 1.0, 1.0);
            learned.push(m.rows[0]);
        }
        // Rows were far first (the count stood still for whole checks),
        // then learned again: the gap of 1e6 in r2 closes in ~16 checks
        // at (100 + 5 * 50) / 150 = 2.3 per check.
        assert_eq!(learned[0], learned[49], "{:?}", &learned[..50]);
        let resumed = learned.iter().position(|&r| r > learned[0]).unwrap();
        assert!((500..1500).contains(&resumed), "{resumed}");
        assert!(
            m.rows[0] > learned[0] + 1500,
            "{} vs {}",
            m.rows[0],
            learned[0]
        );
        let r2 = m.clusters()[0].r2;
        assert!(r2 > 0.1 && r2 < 2.0, "{r2}");
        assert!(m.far_cut.is_finite() && m.far_cut > r2);
    }

    #[test]
    fn split_merge_disabled_never_touches_the_centres() {
        // `dead_frac` 0: above it the pair is refused, the rule running only
        // at a split-merge check (review 2026-10-06, CF6).
        let c = KMeansCfg {
            warm_rows: 3,
            split_merge: 0.0,
            split_merge_every: 1,
            dead_frac: 0.0,
            ..cfg(3)
        };
        let mut m = KMeans::new(c).unwrap();
        for r in [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]] {
            m.step(&r, &[], 1.0, 1.0);
        }
        for i in 0..100 {
            m.step(&[10.0 + (i % 3) as f64 * 0.1, 0.0], &[], 1.0, 1.0);
        }
        assert_eq!(m.events(), (0, 0));
    }

    /// A feature that moves and then holds, at a level of 1e8 (docs/PLAN.md
    /// task 102): the labels and distances are those of the same stream at
    /// a level of 0. With a plain mean the variance settled on the mean's
    /// rounding gap and each centre's own gap made a term of its own; with
    /// the means as pairs and the metric floored, the level leaves nothing.
    #[test]
    fn a_stopped_feature_at_a_level_leaves_the_assignments_alone() {
        let run = |level: f64| -> Vec<Vec<f64>> {
            let mut m = KMeans::new(KMeansCfg {
                n_features: 3,
                decay: Decay::Halflife(20.0),
                warm_rows: 30,
                seed_rule: SeedRule::Lloyd,
                standardize: true,
                scale_floor: 0.1,
                ..cfg(3)
            })
            .unwrap();
            let mut g = SplitMix64::new(11);
            blobs(400, 5)
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let third = if i < 300 {
                        level + g.uniform()
                    } else {
                        level + 0.37
                    };
                    crate::OnlineModel::step(&mut m, &[b[0], b[1], third], &[], 1.0, 1.0).pred
                })
                .collect()
        };
        let (base, high) = (run(0.0), run(1e8));
        assert!(
            base.iter().skip(300).all(|p| !p[0].is_nan()),
            "seeded before the stop"
        );
        for (i, (b, h)) in base.iter().zip(&high).enumerate() {
            if b[0].is_nan() {
                assert!(h[0].is_nan(), "row {i}");
                continue;
            }
            assert_eq!(b[0], h[0], "row {i}: the label");
            for j in 1..3 {
                assert!(
                    (b[j] - h[j]).abs() <= 1e-6 * (1.0 + b[j]),
                    "row {i}, slot {j}: {} at 1e8 against {} at 0",
                    h[j],
                    b[j]
                );
            }
        }
    }

    /// `scale_floor` is refused by name in the core too, negative and NaN.
    #[test]
    fn a_bad_scale_floor_is_refused_by_name() {
        for bad in [-0.5, f64::NAN, f64::INFINITY] {
            let err = KMeans::new(KMeansCfg {
                scale_floor: bad,
                ..cfg(2)
            })
            .expect_err("refused");
            assert!(err.contains("scale_floor"), "{err}");
        }
    }

    /// The reference decays at `LONG_HALFLIVES` times the half-life: over a
    /// clock of 2 a row, the model's moments carry the weights and variances
    /// a `FeatureMoments` given `decay.factor(2 / 8)` carries, to the bit.
    #[test]
    fn the_reference_decays_at_eight_halflives() {
        let decay = Decay::Halflife(20.0);
        let mut m = KMeans::new(KMeansCfg {
            decay,
            warm_rows: 6,
            ..cfg(2)
        })
        .unwrap();
        let mut by_hand = FeatureMoments::new(2);
        let mut g = SplitMix64::new(5);
        for i in 0..80 {
            let x = [g.uniform(), 10.0 * g.uniform()];
            let d_clock = if i == 0 { 0.0 } else { 2.0 };
            crate::OnlineModel::step(&mut m, &x, &[], d_clock, 1.0);
            by_hand.decay(
                decay.factor(d_clock),
                decay.factor(d_clock / LONG_HALFLIVES),
            );
            by_hand.absorb(&x, 1.0);
        }
        assert_eq!(m.moments().w_long, by_hand.w_long);
        assert_eq!(m.moments().var_long, by_hand.var_long);
        assert!(m.moments().w_long[0] > 5.0);
    }

    /// A state written before `scale_floor` loads, with the floor at 0: the
    /// metric it had (review 2026-09-25, hard rule 5).
    #[test]
    fn a_state_without_scale_floor_loads_with_the_metric_it_had() {
        let mut m = KMeans::new(KMeansCfg {
            scale_floor: 0.1,
            ..cfg(2)
        })
        .unwrap();
        let mut g = SplitMix64::new(2);
        for _ in 0..20 {
            crate::OnlineModel::step(&mut m, &[g.uniform(), g.uniform()], &[], 1.0, 1.0);
        }
        let mut old = serde_json::to_value(&m).unwrap();
        assert!(
            old["cfg"]
                .as_object_mut()
                .unwrap()
                .remove("scale_floor")
                .is_some()
        );
        let back: KMeans = serde_json::from_value(old).unwrap();
        assert_eq!(back.cfg.scale_floor, 0.0);
        assert_eq!(back.moments(), m.moments());
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 4,
            decay: Decay::Halflife(20.0),
            standardize: true,
            ..cfg(3)
        })
        .unwrap();
        let mut s = 2u64;
        for (i, r) in blobs(50, 8).iter().enumerate() {
            let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
            let w = if i % 7 == 0 { 0.0 } else { 1.0 };
            let before = m.clone();
            let want = m.predict(r, d);
            assert_eq!(m, before, "predict mutated the model");
            let got = m.step(r, &[], d, w);
            for s in 0..3 {
                same(got.pred[s], want.pred[s], &format!("row {i} slot {s}"));
            }
            assert_eq!(got.n_eff, want.n_eff);
            assert_eq!(got.extra, want.extra);
        }
    }

    #[test]
    fn state_round_trips_and_refuses_another_model() {
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 4,
            split_merge: 0.5,
            split_merge_every: 5,
            dead_frac: 0.1,
            ..cfg(3)
        })
        .unwrap();
        for r in blobs(10, 3) {
            m.step(&r, &[], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let state: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut r = KMeans::restore(&state).unwrap();
        assert_eq!(r, m);
        let row = [3.0, 4.0];
        assert_eq!(r.step(&row, &[], 1.0, 1.0), m.step(&row, &[], 1.0, 1.0));
        let other = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(2))));
        assert!(matches!(
            KMeans::restore(&other),
            Err(StateError::WrongModel {
                expected: "kmeans",
                found: "ew_cov_accumulator"
            })
        ));
    }

    #[test]
    fn rows_at_the_input_bound_leave_every_number_finite() {
        let b = crate::INPUT_BOUND;
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 3,
            standardize: true,
            split_merge: 0.5,
            split_merge_every: 4,
            dead_frac: 0.1,
            ..cfg(3)
        })
        .unwrap();
        let script = [
            [0.0, 0.0],
            [1.0, 1.0],
            [b, -b],
            [-b, b],
            [b, b],
            [0.0, 0.0],
            [1e-300, 1e-300],
            [b, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ];
        for (i, r) in script.iter().cycle().take(200).enumerate() {
            let step = m.step(r, &[], 1.0, 1.0);
            assert!(i < 3 || !step.pred[0].is_nan(), "row {i}");
            assert!(step.n_eff.is_finite());
        }
        for c in m.clusters() {
            assert!(c.n.is_finite() && c.r2.is_finite() && c.r2 >= 0.0, "{c:?}");
            // A convex combination of bounded rows, up to rounding.
            assert!(
                c.c.iter()
                    .all(|v| v.is_finite() && v.abs() <= b * (1.0 + 1e-12)),
                "{c:?}"
            );
        }
        assert!(m.metric().iter().all(|v| v.is_finite() && *v > 0.0));
        // And then ordinary data is measured again.
        let step = m.step(&[1.0, 1.0], &[], 1.0, 1.0);
        assert!(step.pred[1].is_finite());
    }

    #[test]
    fn coefficients_are_the_centres_and_absent_before_seeding() {
        let mut m = KMeans::new(cfg(2)).unwrap();
        assert_eq!(m.coefficients(), None);
        for r in blobs(4, 6) {
            m.step(&r, &[], 1.0, 1.0);
        }
        let cs = m.coefficients().unwrap();
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0], m.clusters()[0].c);
    }

    #[test]
    fn config_is_validated() {
        let bad = [
            KMeansCfg { k: 0, ..cfg(1) },
            KMeansCfg {
                n_features: 0,
                ..cfg(1)
            },
            KMeansCfg {
                update_every: 0,
                ..cfg(1)
            },
            KMeansCfg {
                split_merge_every: 0,
                ..cfg(1)
            },
            KMeansCfg {
                split_merge: -1.0,
                ..cfg(1)
            },
            KMeansCfg {
                split_merge: f64::INFINITY,
                ..cfg(1)
            },
            KMeansCfg {
                dead_frac: f64::NAN,
                ..cfg(1)
            },
            KMeansCfg {
                min_weight: -1.0,
                ..cfg(1)
            },
        ];
        for c in bad {
            assert!(KMeans::new(c.clone()).is_err(), "{c:?}");
        }
        // `max(warm_rows, k)` rows were buffered, so 0 meant `k` without a
        // word; refused, as `hmm` refuses it (review 2026-10-06, PC8).
        assert!(
            KMeans::new(KMeansCfg {
                warm_rows: 0,
                ..cfg(1)
            })
            .is_err(),
            "warm_rows 0 is below k"
        );
    }

    /// The warm-up buffers `warm_rows` rows and seeds `k` centres from them,
    /// so fewer rows than centres is refused by name, as `hmm` refuses it;
    /// it was floored to `k` in silence (review 2026-10-06, PC8).
    #[test]
    fn warm_rows_below_k_is_refused_by_name() {
        let Err(e) = KMeans::new(KMeansCfg {
            warm_rows: 2,
            ..cfg(3)
        }) else {
            panic!("accepted")
        };
        assert!(
            e.contains("kmeans: warm_rows must be at least k (3)") && e.contains("got 2"),
            "{e}"
        );
        assert!(
            KMeans::new(KMeansCfg {
                warm_rows: 3,
                ..cfg(3)
            })
            .is_ok()
        );
    }

    /// The dead rule runs at a split-merge check, and `split_merge = 0` runs
    /// none, so a `dead_frac` above 0 beside it did nothing: refused by
    /// name, as a parameter its mode does not read is elsewhere (review
    /// 2026-10-06, CF6).
    #[test]
    fn dead_frac_without_split_merge_is_refused_by_name() {
        let Err(e) = KMeans::new(KMeansCfg {
            split_merge: 0.0,
            dead_frac: 0.5,
            ..cfg(3)
        }) else {
            panic!("accepted")
        };
        assert!(
            e.contains("kmeans: dead_frac") && e.contains("split_merge = 0") && e.contains("0.5"),
            "{e}"
        );
        for (split_merge, dead_frac) in [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0)] {
            assert!(
                KMeans::new(KMeansCfg {
                    split_merge,
                    dead_frac,
                    ..cfg(3)
                })
                .is_ok()
            );
        }
    }

    /// `k` and the warm-up buffer are sized by the spec before the stream
    /// can say anything: `k` has a ceiling, 2^16, and the buffer of
    /// `warm_rows` rows of `n_features` doubles a budget, 256 MiB, each
    /// refused by name with the value. `k = 10^9` and `warm_rows = 10^9`
    /// were accepted, and held that many rows before seeding (review
    /// 2026-10-06, CF2).
    #[test]
    fn k_and_the_warm_up_buffer_have_ceilings() {
        let k = (1usize << 16) + 1;
        let Err(e) = KMeans::new(KMeansCfg {
            k,
            warm_rows: k,
            ..cfg(3)
        }) else {
            panic!("accepted")
        };
        assert!(
            e.contains("kmeans: k must be at most 65536") && e.contains(&format!("got {k}")),
            "{e}"
        );
        // A buffered row of two features is its vector (24 bytes), its two
        // values and its weight: 48 bytes, so 5,592,405 rows fit 256 MiB and
        // one more is 32 bytes past it.
        let Err(e) = KMeans::new(KMeansCfg {
            warm_rows: 5_592_406,
            ..cfg(3)
        }) else {
            panic!("accepted")
        };
        assert!(
            e.contains("kmeans: the warm-up buffer would hold 256.00003 MiB")
                && e.contains("warm_rows 5592406 rows of 2 features")
                && e.contains("over the 256 MiB"),
            "{e}"
        );
        for c in [
            KMeansCfg {
                warm_rows: 5_592_405,
                ..cfg(3)
            },
            KMeansCfg {
                k: 1 << 16,
                warm_rows: 1 << 16,
                ..cfg(3)
            },
        ] {
            assert!(KMeans::new(c.clone()).is_ok(), "{c:?}");
        }
    }

    /// A model seeded by `first` at `centres` (in raw units, no decay, no
    /// split-merge unless the caller sets it), each centre exactly its row.
    fn seeded_at(centres: &[[f64; 2]], c: KMeansCfg) -> KMeans {
        let mut m = KMeans::new(KMeansCfg {
            k: centres.len(),
            warm_rows: centres.len(),
            ..c
        })
        .unwrap();
        for x in centres {
            m.step(x, &[], 1.0, 1.0);
        }
        assert!(m.seeded());
        let got: Vec<Vec<f64>> = centres.iter().map(|x| x.to_vec()).collect();
        assert_eq!(m.coefficients().unwrap(), got, "the fixture");
        m
    }

    fn seeded(k: usize) -> KMeans {
        let centres: Vec<[f64; 2]> = (0..k).map(|i| [10.0 * i as f64, 0.0]).collect();
        seeded_at(&centres, cfg(k))
    }

    /// [`seeded_at`] with the centres written into the state, not learned:
    /// for centres past the input bound, whose rows every model refuses
    /// (task 183). From usable rows, a square overflows under a
    /// standardised metric over a feature of tiny spread instead.
    fn placed_at(centres: &[[f64; 2]], c: KMeansCfg) -> KMeans {
        let near: Vec<[f64; 2]> = (0..centres.len()).map(|i| [i as f64, 0.0]).collect();
        let mut m = seeded_at(&near, c);
        for (cluster, centre) in m.clusters.iter_mut().zip(centres) {
            cluster.c = centre.to_vec();
        }
        m
    }

    /// The metric is all ones in raw units, and `1 / v_i` standardized,
    /// `v_i` each feature's variance over the rows (no decay, unit weights:
    /// the plain population variance).
    #[test]
    fn the_metric_is_one_over_each_features_variance_when_standardizing() {
        let mut plain = KMeans::new(cfg(2)).unwrap();
        let mut scaled = KMeans::new(KMeansCfg {
            standardize: true,
            ..cfg(2)
        })
        .unwrap();
        let rows = [[0.0, 0.0], [2.0, 10.0], [4.0, -10.0], [2.0, 0.0]];
        for x in rows {
            plain.step(&x, &[], 1.0, 1.0);
            scaled.step(&x, &[], 1.0, 1.0);
        }
        assert_eq!(plain.metric(), &[1.0, 1.0]);
        assert_eq!(scaled.metric().len(), 2);
        for j in 0..2 {
            let mean = rows.iter().map(|x| x[j]).sum::<f64>() / 4.0;
            let var = rows.iter().map(|x| (x[j] - mean).powi(2)).sum::<f64>() / 4.0;
            let got = scaled.metric()[j];
            assert!(
                (got - 1.0 / var).abs() <= 1e-12 / var,
                "feature {j}: {got} against 1/{var}"
            );
        }
    }

    /// Among squares that overflow, the overflow-free norms decide the
    /// nearest and the runner-up, first minimum winning; where one square
    /// is finite, the squares decide, even against one that overflowed by a
    /// rounding step and whose norm rounds to the finite one's. The centres
    /// are placed ([`placed_at`]): the rows that put them there are past the
    /// input bound, refused now (task 183).
    #[test]
    fn overflowed_squares_are_ranked_by_their_norms() {
        let origin = [0.0, 0.0];
        // Norms 1e200, 2e200, 3e200: nearest the first, runner-up the second.
        let m = placed_at(&[[1e200, 0.0], [0.0, 2e200], [-3e200, 0.0]], cfg(3));
        assert!(
            m.clusters
                .iter()
                .all(|c| dist2(&c.c, &origin, &m.mw).is_infinite())
        );
        assert_eq!(m.predict(&origin, 1.0).pred, vec![0.0, 1e200, 2e200]);
        // Two equally far: the first is the nearest, the second the runner-up.
        let m = placed_at(&[[1e200, 0.0], [-1e200, 0.0], [0.0, 3e200]], cfg(3));
        assert_eq!(m.predict(&origin, 1.0).pred, vec![0.0, 1e200, 1e200]);
        // The smallest double whose square overflows, before one whose
        // square is just finite and whose norm rounds to the same number
        // (found by search in IEEE doubles).
        let over = [1.3407807929942597e154, 0.0];
        let under = [1.3262349803990826e154, 1.9696170091710862e153];
        let one = [1.0, 1.0];
        assert!(
            dist2(&over, &origin, &one).is_infinite()
                && dist2(&under, &origin, &one).is_finite()
                && dist(&under, &origin, &one) == dist(&over, &origin, &one),
            "the fixture"
        );
        let m = placed_at(&[over, under], cfg(2));
        assert_eq!(m.predict(&origin, 1.0).pred[0], 1.0, "the finite square");
    }

    /// A finite squared distance is reported as its root, to the bit, not
    /// as the overflow-free norm, which differs from it in the last bit
    /// here: `√18` against `3 √2`.
    #[test]
    fn a_finite_distance_is_the_root_of_its_square() {
        let m = seeded_at(&[[0.0, 0.0], [10.0, 0.0]], cfg(2));
        let row = [3.0, 3.0];
        assert_ne!(
            dist(&[0.0, 0.0], &row, &m.mw),
            18f64.sqrt(),
            "the fixture: the norm is another number"
        );
        let out = m.predict(&row, 1.0).pred;
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 18f64.sqrt());
        assert_eq!(out[2], dist2(&[10.0, 0.0], &row, &m.mw).sqrt());
    }

    /// Until a cluster has a trusted radius nothing is far (the module
    /// docs), not even a row whose squared distance overflows: it is
    /// learned into its cluster, not into the far summary. The square of a
    /// usable row overflows under the metric of a feature of spread
    /// `1e-150`, `1e300` (a feature of `1e200`, which overflowed it in raw
    /// units, is past the input bound, refused now: task 183).
    #[test]
    fn before_a_trusted_radius_not_even_an_overflowed_square_is_far() {
        let mut m = seeded_at(
            &[[0.0, 0.0], [1.0, 0.0]],
            KMeansCfg {
                split_merge: 0.5,
                split_merge_every: 1000,
                ..cfg(2)
            },
        );
        assert_eq!(m.far_cut, f64::INFINITY, "no trusted radius yet");
        m.mw = vec![1e300, 1e300];
        let row = [1e5, 0.0];
        assert!(dist2(&row, &[1.0, 0.0], &m.mw).is_infinite(), "the fixture");
        m.step(&row, &[], 1.0, 1.0);
        assert_eq!(m.rows.iter().sum::<u64>(), 3, "learned: {:?}", m.rows);
        assert_eq!(m.far_rows, vec![0, 0]);
    }

    /// The window weight `V` is the weight learned since the last check,
    /// decayed with the clock as every weight is (`V ← λV + w`, the module
    /// docs), from the buffer's replay on; with the rule off it counts
    /// nothing.
    #[test]
    fn the_window_weight_decays_and_counts_what_is_learned() {
        let decay = Decay::Halflife(5.0);
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: 3,
            split_merge: 0.5,
            split_merge_every: 1000,
            decay,
            ..cfg(3)
        })
        .unwrap();
        for x in [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]] {
            m.step(&x, &[], 1.0, 1.0);
        }
        assert!(m.seeded());
        let lam = decay.factor(1.0);
        let mut v = lam * lam + lam + 1.0;
        assert!(
            (m.window_w - v).abs() <= 1e-12 * v,
            "{} against {v}",
            m.window_w
        );
        let mut s = 7u64;
        for (i, x) in blobs(10, 2).iter().enumerate() {
            let d = [0.5, 1.0, 2.0][i % 3];
            let w = 0.5 + lcg(&mut s).abs();
            m.step(x, &[], d, w);
            v = decay.factor(d) * v + w;
            assert!(
                (m.window_w - v).abs() <= 1e-12 * v,
                "row {i}: {} against {v}",
                m.window_w
            );
        }
        let mut off = KMeans::new(KMeansCfg {
            warm_rows: 3,
            decay,
            ..cfg(3)
        })
        .unwrap();
        for x in blobs(10, 2) {
            off.step(&x, &[], 1.0, 1.0);
        }
        assert!(off.seeded());
        assert_eq!(off.window_w, 0.0);
    }

    /// At a check, far rows count in a cluster's radius at the cut, against
    /// its weight; a cluster without far rows keeps its radius, to the bit,
    /// and one without weight too (no `0/0`).
    #[test]
    fn a_cluster_without_far_rows_keeps_its_radius_at_the_check() {
        let mut m = seeded(3);
        m.far_cut = 1.0;
        for (c, (n, r2)) in m
            .clusters
            .iter_mut()
            .zip([(3.0, 0.1), (7.0, 0.3), (0.0, 0.2)])
        {
            (c.n, c.r2) = (n, r2);
        }
        m.far[1].n = 2.0;
        assert_ne!(
            (3.0 * 0.1) / 3.0,
            0.1,
            "the fixture: the round trip moves 0.1"
        );
        m.winsorize_radii();
        assert_eq!(m.clusters[0].r2, 0.1);
        assert_eq!(m.clusters[2].r2, 0.2);
        assert_eq!(m.clusters[1].r2, (7.0 * 0.3 + 2.0 * 1.0) / 9.0);
    }

    /// The far source is the cluster with the most far weight, the first of
    /// two equally heavy, and none while no cluster has any.
    #[test]
    fn the_far_source_is_the_first_heaviest_far_summary() {
        let mut m = seeded(4);
        let mut with = |ns: [f64; 4]| {
            for (f, n) in m.far.iter_mut().zip(ns) {
                f.n = n;
            }
            m.far_source()
        };
        assert_eq!(with([0.0; 4]), None);
        assert_eq!(with([0.0, 2.0, 5.0, 5.0]), Some(2));
    }

    /// A merge of `j` into `i` pools their far rows, and places the freed
    /// centre on the heaviest far summary -- the pool on a tie -- only when
    /// it is a component's worth: `FAR_ROWS` rows, `FAR_SHARE` of the
    /// window's weight, and some weight at all.
    #[test]
    fn a_merge_wants_a_components_worth_of_far_rows() {
        let mut m = seeded(4);
        let mut source = |ns: [f64; 4], rows: [u64; 4], window: f64| {
            for ((f, r), (n, rw)) in m
                .far
                .iter_mut()
                .zip(&mut m.far_rows)
                .zip(ns.into_iter().zip(rows))
            {
                (f.n, *r) = (n, rw);
            }
            m.window_w = window;
            m.merge_source(0, 1)
        };
        // Cluster 2 outweighs the pool of 0 and 1 (4 against 3).
        assert_eq!(source([1.0, 2.0, 4.0, 3.0], [1, 1, 5, 5], 40.0), Some(2));
        // A tie keeps the pool, which has the rows (2 + 2).
        assert_eq!(source([1.0, 2.0, 3.0, 0.0], [2, 2, 4, 0], 40.0), Some(0));
        // Too few rows; too little of the window; no weight at all.
        assert_eq!(source([1.0, 2.0, 0.0, 0.0], [1, 1, 0, 0], 40.0), None);
        assert_eq!(source([1.0, 0.9, 0.0, 0.0], [3, 3, 0, 0], 40.0), None);
        assert_eq!(source([0.0; 4], [3, 3, 0, 0], 0.0), None);
    }

    /// A split leaves the source half its weight and places the target on
    /// the source's far rows with the other half and the typical radius,
    /// trusted at once.
    #[test]
    fn a_split_halves_the_source_and_places_the_target_on_its_far_rows() {
        let mut m = seeded(3);
        m.clusters[2].n = 8.0;
        m.far[2].c = vec![5.0, -1.0];
        m.r2_typical = 0.25;
        m.split(0, 2);
        assert_eq!(m.clusters[2].n, 4.0);
        assert_eq!(
            m.clusters[0],
            ClusterSummary::at(vec![5.0, -1.0], 4.0, 0.25)
        );
        assert_eq!(m.rows[0], RADIUS_ROWS);
    }

    /// A model with clusters placed by hand: centres, weights and radii,
    /// far rows on `far_on` (weight 5, five rows, mean `(50, 50)`), a window
    /// of 20 and `n_eff` of 100.
    fn placed(
        centres: &[[f64; 2]],
        n: &[f64],
        far_on: usize,
        split_merge: f64,
        dead_frac: f64,
    ) -> KMeans {
        let mut m = seeded_at(
            centres,
            KMeansCfg {
                split_merge,
                dead_frac,
                ..cfg(centres.len())
            },
        );
        for (c, &w) in m.clusters.iter_mut().zip(n) {
            (c.n, c.r2) = (w, 1.0);
        }
        m.far[far_on] = ClusterSummary::at(vec![50.0, 50.0], 5.0, 0.0);
        m.far_rows[far_on] = 5;
        m.window_w = 20.0;
        m.moments.w = 100.0;
        m
    }

    /// The closest pair is measured in summed radii, `d / (r_i + r_j)`, and
    /// merges below `split_merge`, not at it: centres 3 apart with radii of
    /// 1 are 1.5 summed radii apart.
    #[test]
    fn a_pair_merges_below_split_merge_not_at_it() {
        let centres = [[0.0, 0.0], [3.0, 0.0], [100.0, 0.0]];
        let mut at = placed(&centres, &[10.0; 3], 2, 1.5, 0.0);
        at.split_merge_check();
        assert_eq!(at.events(), (0, 0), "1.5 is not below 1.5");
        let mut below = placed(&centres, &[10.0; 3], 2, 1.6, 0.0);
        below.split_merge_check();
        assert_eq!(below.events(), (1, 0));
        assert_eq!(below.clusters[0].c, vec![1.5, 0.0], "the pair merged");
        assert_eq!(below.clusters[1].c, vec![50.0, 50.0], "the freed centre");
    }

    /// Of two pairs equally close, the first is merged: centres at 0, 2
    /// and 4 on a line, each pair of neighbours one summed radius apart.
    #[test]
    fn of_two_pairs_equally_close_the_first_merges() {
        let mut m = placed(
            &[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            &[10.0; 3],
            0,
            1.5,
            0.0,
        );
        m.split_merge_check();
        assert_eq!(m.events(), (1, 0));
        let cs: Vec<Vec<f64>> = m.clusters.iter().map(|c| c.c.clone()).collect();
        assert_eq!(cs, vec![vec![1.0, 0.0], vec![50.0, 50.0], vec![4.0, 0.0]]);
    }

    /// The dead rule: the lightest cluster -- the first of two equally
    /// light -- goes to the far rows when lighter than `dead_frac · n_eff /
    /// k`, not when at it, and the event is counted. With one cluster there
    /// is nothing to free: the rule needs two.
    #[test]
    fn the_dead_rule_frees_the_first_lightest_below_its_floor() {
        // Floor 0.25 · 100 / 2 = 12.5.
        let mut m = placed(&[[0.0, 0.0], [10.0, 0.0]], &[20.0, 1.0], 0, 0.5, 0.25);
        m.split_merge_check();
        assert_eq!(m.events(), (0, 1));
        assert_eq!(m.clusters[1].c, vec![50.0, 50.0]);
        let mut m = placed(&[[0.0, 0.0], [10.0, 0.0]], &[20.0, 12.5], 0, 0.5, 0.25);
        m.split_merge_check();
        assert_eq!(m.events(), (0, 0), "at the floor, alive");
        // Two equally light, the pairs far apart in summed radii.
        let mut m = placed(
            &[[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]],
            &[5.0, 0.1, 0.1],
            0,
            0.5,
            0.25,
        );
        m.split_merge_check();
        assert_eq!(m.events(), (0, 1));
        assert_eq!(m.clusters[1].c, vec![50.0, 50.0], "the first of the two");
        assert_eq!(m.clusters[2].c, vec![0.0, 100.0]);
        // One cluster, far lighter than the floor.
        let mut m = placed(&[[0.0, 0.0]], &[1.0], 0, 0.5, 0.5);
        m.split_merge_check();
        assert_eq!(m.events(), (0, 0));
        assert_eq!(m.clusters[0].c, vec![0.0, 0.0]);
    }

    /// The buffer's cut: `far_factor` times its rows' weighted mean squared
    /// distance to the EW mean.
    #[test]
    fn the_buffers_cut_is_its_weighted_mean_squared_distance_times_the_factor() {
        let mut m = KMeans::new(cfg(2)).unwrap();
        m.moments.mean = vec![1.0, 2.0];
        // Squared distances 0, 25 and 1.
        m.buf = vec![vec![1.0, 2.0], vec![4.0, 6.0], vec![0.0, 2.0]];
        m.buf_w = vec![0.5, 2.0, 1.5];
        let want = m.far_factor * (0.5 * 0.0 + 2.0 * 25.0 + 1.5 * 1.0) / (0.5 + 2.0 + 1.5);
        assert!(
            (m.buffer_cut() - want).abs() <= 1e-12 * want,
            "{} against {want}",
            m.buffer_cut()
        );
    }

    /// Nine rows, one far from the rest -- far enough to be past the
    /// buffer's cut, which the row itself raises -- for `first` seeding at
    /// `k = 2`.
    const ONE_FAR: [[f64; 2]; 9] = [
        [0.0, 0.0],
        [100.0, 100.0],
        [10.0, 0.0],
        [0.1, 0.0],
        [10.1, 0.0],
        [0.0, 0.1],
        [10.0, 0.1],
        [0.1, 0.1],
        [10.1, 0.1],
    ];

    fn seed_on(rows: &[[f64; 2]], c: KMeansCfg) -> KMeans {
        let mut m = KMeans::new(KMeansCfg {
            warm_rows: rows.len(),
            ..c
        })
        .unwrap();
        for x in rows {
            m.step(x, &[], 1.0, 1.0);
        }
        assert!(m.seeded());
        m
    }

    /// A row of the seed buffer past its cut chooses no seed; under the
    /// split-merge rule it is replayed as a far row, so the centres are the
    /// means of the other rows; with the rule off every row is learned.
    #[test]
    fn a_far_row_in_the_seed_buffer_chooses_no_seed() {
        let on = seed_on(
            &ONE_FAR,
            KMeansCfg {
                split_merge: 0.5,
                split_merge_every: 1000,
                ..cfg(2)
            },
        );
        let near = |got: &[f64], want: [f64; 2]| {
            assert!(
                (got[0] - want[0]).abs() < 1e-12 && (got[1] - want[1]).abs() < 1e-12,
                "{got:?} against {want:?}"
            );
        };
        near(&on.clusters[0].c, [0.05, 0.05]);
        near(&on.clusters[1].c, [10.05, 0.05]);
        assert_eq!(on.far_rows, vec![0, 1], "replayed as far");
        let off = seed_on(&ONE_FAR, cfg(2));
        near(&off.clusters[0].c, [0.05, 0.05]);
        near(&off.clusters[1].c, [140.2 / 5.0, 100.2 / 5.0]);
        let total: f64 = off.clusters.iter().map(|c| c.n).sum();
        assert_eq!(total, off.n_eff(), "every row learned");
    }

    /// When the rows within the cut cannot give `k` distinct seeds, the
    /// whole buffer chooses them: nine equal rows and one far.
    #[test]
    fn seeds_that_coincide_are_chosen_again_from_the_whole_buffer() {
        let mut rows = vec![[1.0, 1.0]; 9];
        rows.push([100.0, 100.0]);
        let m = seed_on(
            &rows,
            KMeansCfg {
                seed_rule: SeedRule::Farthest,
                split_merge: 0.5,
                split_merge_every: 1000,
                ..cfg(2)
            },
        );
        assert_eq!(
            m.coefficients().unwrap(),
            vec![vec![1.0, 1.0], vec![100.0, 100.0]]
        );
    }

    /// A buffer of one row repeated has no spread and a cut of 0, and its
    /// rows, at the mean, are not past it: all learned.
    #[test]
    fn a_buffer_without_spread_learns_every_row() {
        let m = seed_on(
            &[[2.0, 3.0]; 3],
            KMeansCfg {
                split_merge: 0.5,
                split_merge_every: 1000,
                ..cfg(1)
            },
        );
        assert_eq!(m.clusters[0].n, 3.0);
        assert_eq!(m.far_rows, vec![0]);
    }

    /// Seeding on rows enough for a trusted radius sets the far cut at
    /// once, so the first row after it is already judged: a row far from
    /// the blob goes to the far summary.
    #[test]
    fn the_far_cut_holds_from_the_first_row_after_seeding() {
        let mut s = 5u64;
        let rows: Vec<[f64; 2]> = (0..12).map(|_| [lcg(&mut s), lcg(&mut s)]).collect();
        let mut m = seed_on(
            &rows,
            KMeansCfg {
                split_merge: 0.5,
                split_merge_every: 1000,
                ..cfg(1)
            },
        );
        assert!(m.far_cut.is_finite(), "{}", m.far_cut);
        assert_eq!(m.rows, vec![12]);
        m.step(&[50.0, 50.0], &[], 1.0, 1.0);
        assert_eq!((m.rows[0], m.far_rows[0]), (12, 1));
    }

    /// The restarts are ranked by the weighted sum over the buffer of each
    /// row's squared distance to its nearest centre.
    #[test]
    fn inertia_is_the_weighted_squared_distance_to_the_nearest_centre() {
        let buf = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![4.0, 4.0],
            vec![5.0, 4.0],
        ];
        let w = [1.0, 2.0, 0.5, 3.0];
        let centres = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        // Nearest squared distances 0, 1, 2 and 1.
        let want = 1.0 * 0.0 + 2.0 * 1.0 + 0.5 * 2.0 + 3.0 * 1.0;
        assert_eq!(inertia(&buf, &w, &centres, &[1.0, 1.0]), want);
    }

    /// Farthest-first takes the first of two rows equally far, as every
    /// argmin and argmax here does (and `tests/reference_cluster.py`).
    #[test]
    fn farthest_first_takes_the_first_of_two_equally_far() {
        let buf = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![-1.0, 0.0]];
        assert_eq!(
            farthest(&buf, 2, &[1.0, 1.0]),
            vec![vec![0.0, 0.0], vec![1.0, 0.0]]
        );
    }

    /// Lloyd's assignment gives a row equally near two centres to the
    /// first, and a centre no row is nearest to stays where it is.
    #[test]
    fn lloyd_assigns_a_tie_to_the_first_centre_and_leaves_an_empty_one_put() {
        let buf = vec![vec![0.0, 0.0], vec![-1.0, 0.0], vec![1.0, 0.0]];
        let w = [1.0; 3];
        let got = lloyd(
            &buf,
            &w,
            vec![vec![-1.0, 0.0], vec![1.0, 0.0]],
            &[1.0, 1.0],
            1,
        );
        assert_eq!(got, vec![vec![-0.5, 0.0], vec![1.0, 0.0]]);
        let got = lloyd(
            &buf,
            &w,
            vec![vec![0.0, 0.0], vec![50.0, 50.0]],
            &[1.0, 1.0],
            3,
        );
        assert_eq!(got[1], vec![50.0, 50.0]);
    }

    /// Each part of a state's shape is checked on its own: a centre, a
    /// batch, a far summary, a row count, a far-row count, a buffered weight
    /// or a buffered row of the wrong size, each with everything else
    /// right.
    #[test]
    fn each_part_of_a_states_shape_is_checked_on_its_own() {
        let mut warm = KMeans::new(KMeansCfg {
            warm_rows: 10,
            ..cfg(2)
        })
        .unwrap();
        for x in blobs(1, 1) {
            warm.step(&x, &[], 1.0, 1.0);
        }
        assert!(!warm.seeded() && warm.buffered() == 3);
        type Edit = fn(&mut KMeans);
        let edits: [(&str, &KMeans, Edit); 7] = [
            ("centre", &seeded(3), |m| m.clusters[1].c.push(0.0)),
            ("batch", &seeded(3), |m| m.batch.truncate(2)),
            ("far", &seeded(3), |m| m.far[2].c.truncate(1)),
            ("rows", &seeded(3), |m| m.rows.push(0)),
            ("far_rows", &seeded(3), |m| m.far_rows.truncate(2)),
            ("buf_w", &warm, |m| m.buf_w.push(1.0)),
            ("buf", &warm, |m| m.buf[1].push(1.0)),
        ];
        for (what, model, edit) in edits {
            let mut s = model.state();
            let ModelState::KMeans(inner) = &mut s.model else {
                unreachable!()
            };
            edit(inner);
            match KMeans::restore(&s) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
                other => panic!("{what}: {other:?}"),
            }
            assert!(
                KMeans::restore(&model.state()).is_ok(),
                "{what}: the unedited state"
            );
        }
    }

    #[test]
    fn seed_rule_names_are_snake_case() {
        for (rule, name) in [
            (SeedRule::First, "\"first\""),
            (SeedRule::Farthest, "\"farthest\""),
            (SeedRule::Kmeanspp, "\"kmeanspp\""),
            (SeedRule::Lloyd, "\"lloyd\""),
        ] {
            assert_eq!(serde_json::to_string(&rule).unwrap(), name);
        }
    }

    /// A row with a value that is not usable takes no row of the warm-up
    /// buffer (`OnlineModel`, task 183), as a row of weight 0 takes none:
    /// the buffer only ages over it, and the seeding replays the rows either
    /// side of it. With no decay the model with the row is the model without
    /// it, before the seeding and after. A feature or a weight past the
    /// input bound was buffered, and seeded a centre with it.
    #[test]
    fn a_refused_row_takes_no_warm_up_row() {
        let bytes = |m: &KMeans| rmp_serde::to_vec(&m.state()).unwrap();
        let rows = blobs(4, 19);
        for (bad, w) in [
            ([f64::NAN, 0.5], 1.0),
            ([2.0 * crate::INPUT_BOUND, 0.5], 1.0),
            ([0.5, 0.25], 2.0 * crate::INPUT_BOUND),
        ] {
            let case = format!("{bad:?} at weight {w}");
            let mut with = KMeans::new(cfg(3)).unwrap();
            let mut without = KMeans::new(cfg(3)).unwrap();
            for (i, x) in rows.iter().enumerate() {
                if i == 3 {
                    let buffered = with.buf.clone();
                    let out = with.step(&bad, &[], 1.0, w);
                    assert!(out.pred.iter().all(|v| v.is_nan()), "{case}: {out:?}");
                    assert_eq!(with.buf, buffered, "{case}: the buffer took the row");
                    assert_eq!(bytes(&with), bytes(&without), "{case}");
                }
                with.step(x, &[], 1.0, 1.0);
                without.step(x, &[], 1.0, 1.0);
            }
            assert!(with.seeded(), "{case}: the fixture");
            assert_eq!(bytes(&with), bytes(&without), "{case}");
        }
    }
}
