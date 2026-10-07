//! DenStream-style micro-clusters with a linkage macro step
//! (docs/CLUSTERING.md §6.5; docs/PLAN.md §11a, task 24): a bounded set of
//! mean-form summaries, each absorbing a row only while its radius stays
//! within `eps`, a new one opened where no summary can take the row, and a
//! checkpoint every `prune_every` clock units, or every `max_rows_between_prunes`
//! learned rows, whichever comes first, that prunes the faded ones
//! and links the potential ones into clusters by single linkage.
//!
//! ```text
//! metric      mw_i = 1 / v_i  (EW variance of feature i; 1 where v_i = 0)
//! bound       E = eps² p                 (eps per standardized coordinate)
//! absorb      j_p = nearest potential;  take it if r2_after(j_p, z, w̄) ≤ E
//!             else j_o = nearest outlier;  take it if r2_after(j_o, z, w̄) ≤ E,
//!             promoting it once n_j_o ≥ beta_mu w̄ (it takes the label of the
//!             nearest potential summary within L, else its own id)
//!             else open a summary at z with weight w and the next id
//!             (evicting the lightest outlier summary — else the lightest
//!             potential one — when max_clusters are live)
//!             the summary taken absorbs z with weight w, then r2 ← min(r2, E)
//!             r2_after(j, z, w) = a r2_j + a b ‖z − c_j‖²_mw,  a = n_j/(n_j + w),  b = w/(n_j + w)
//! outputs     cluster = label(j_p),  dist = ‖z − c_j_p‖_mw,
//!             micro_id = the id the row goes to,  outlier = not taken by a
//!             potential summary,  n_clusters, n_micro
//! checkpoint  every prune_every clock units or max_rows_between_prunes
//!             learned rows, whichever first: drop a potential summary with
//!             n < beta_mu w̄, and an outlier one with n < ξ(age) w̄
//!             ξ(a) = (2^(−(a + Tp)/h) − 1) / (2^(−Tp/h) − 1),
//!             Tp = ⌈h log2(beta_mu / (beta_mu − 1))⌉      (DenStream eq. 4.1–4.2)
//!             then link potential summaries with ‖c_a − c_b‖_mw ≤ L,
//!             label = the smallest id in the component
//!             L = macro_link · eps √p, or, unset, derived from the spacing:
//!             max(LINK_FLOOR, LINK_FACTOR · p90 of the nearest-neighbour distance) · eps √p
//! decay       n_j *= lam, W *= lam, age_j += d_clock       lam = 0.5^(d/half_life)
//! ```
//!
//! `w̄` is the EW mean weight of the rows learned from (docs/PLAN.md task
//! 147): `beta_mu` and `ξ` count rows of it, and a summary admits a row of
//! it, so a constant multiple of every weight moves nothing; at any constant
//! weight `w̄ = 1`. `ξ(a) = Σ_{i ≤ a/Tp} 2^(−i Tp/h)` is the weight of a
//! summary that took one such row every `Tp` clock units since it opened.
//! `beta_mu` is DenStream's point density, DBSCAN's `MinPts`, set against
//! the arrival rate (docs/PLAN.md task 151): with half-life `h` and `v` rows
//! per clock unit the stream's steady-state weight is about `1.44·v·h`, so a
//! summary meant to hold a share `s` of it needs `beta_mu ≈ 1.44·s·v·h`.
//!
//! `eps` is the bound on a summary's RMS radius *per standardized
//! coordinate*: in `p` dimensions the bound on the radius in the metric is
//! `eps √p`, so `eps = 0.1` means the same thing at `p = 2` and `p = 50`.
//! (docs/CLUSTERING.md §7.8: a bound fixed in the metric falls off a cliff
//! with `p`, and the failure is silent — every row opens an outlier summary,
//! none is promoted, and the output is all null.) The linkage threshold is
//! likewise in units of `eps √p`.
//!
//! The threshold is what decides whether the macro step follows a shape or
//! ignores it (§7.8): the potential summaries along a shape sit about
//! `1.8 eps √p` apart at the median and `2.2` at the 90th percentile, so a
//! threshold at `2` severs the chain every other step and one much above
//! `3` bridges genuinely separate clusters. Unset, `macro_link` is derived
//! at each checkpoint from the spacing the step already measures: the
//! 90th percentile of every potential summary's nearest-neighbour distance
//! times [`LINK_FACTOR`], and never below [`LINK_FLOOR`] — DenStream's own
//! rule that two summaries within `2 eps` of each other overlap. A value
//! given is an override in the same units; `0` links nothing, so each
//! potential summary is its own cluster. The step is `O(m²)` time over `m`
//! potential summaries. Up to [`LINK_MATRIX_MAX`] of them it keeps their
//! squared distances in an `m × m` matrix, each computed once and read
//! twice; past it, it takes each pair's distance when it needs it, in
//! `O(m)` memory, where the matrix was `max_clusters²` doubles at every
//! checkpoint (review 2026-10-06, CF2).
//!
//! Three rules make a variable-count output honest (§6.5): ids are
//! monotone and never reused — an evicted or pruned id never comes back;
//! a row's `micro_id` is the id it *would* be absorbed by, read before the
//! update, so the first row of a new summary already carries the new id;
//! and the count of live clusters is an output, so churn is visible
//! without diffing labels. The decision is made for a row of the mean
//! weight `w̄` whatever the row's weight: a row of weight `w` stands for
//! `w/w̄` such rows, the first of which is decided, and this is what lets
//! [`predict`](OnlineModel::predict) — which is never told a weight — say
//! exactly what the step will do. The row is then absorbed with its full
//! weight, and since `w > w̄` can carry the radius past the bound, the
//! radius is capped at the bound after the absorb. Left above it, the
//! summary would admit nothing — not even a row at its centre — until
//! decay brought its weight under `E / (r2 − E)`, half-lives later; capped,
//! it is merely full, and the cap costs one row's worth of spread that the
//! next rows re-estimate. (DenStream has no row weights; this is the
//! extension.) So `r2 ≤ E` holds for every summary at all times, and the
//! radius [`coefficients`](Micro::coefficients) reports is at most
//! `eps √p`.
//!
//! Every output is read *before* the row is learned (CLAUDE.md rule 2),
//! `n_eff` is the EW weight before the row and before its own decay (rule
//! 8), and the pruning checkpoint is decided on the rows' stamps, the
//! decayed clock held exactly, and on the learned-row count (docs/PLAN.md
//! tasks 163 and 180), neither of which a chunking of the stream can move.
//! Standardization scales
//! the metric, never the coordinates (§10), so the centres stay in the
//! features' own units; a summary's radius is in the metric that was in
//! force when its rows were absorbed.

use serde::{Deserialize, Serialize};

use super::summary::{ClusterSummary, FeatureMoments, LONG_HALFLIVES, dist, dist2, merged_radius2};
use crate::clock::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::since::Since;

/// Configuration for [`Micro`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroCfg {
    pub n_features: usize,
    pub decay: Decay,
    /// Outputs are null while `n_eff < min_weight`.
    pub min_weight: f64,
    /// Bound on a summary's RMS radius per standardized coordinate, `> 0`.
    pub eps: f64,
    /// Weight at which an outlier summary becomes potential, `> 0`.
    pub beta_mu: f64,
    /// Live summaries at most, `>= 1`.
    pub max_clusters: usize,
    /// Clock units between checkpoints, as `solve_every` is the
    /// regressions' (docs/PLAN.md task 163): `0` checkpoints on every row,
    /// `inf` never by the clock. DenStream checks every `Tp` clock units
    /// (the module docs). A row of weight zero advances the clock, so a quiet
    /// spell still checkpoints, and a gap capped at `gap_cap` counts as the
    /// cap; it is measured from the last checkpoint on the rows' stamps, the
    /// decayed clock held exactly, where the caller hands them (task 180).
    /// Infinite under the row cap alone, the default, which a JSON export
    /// writes as a tag (`crate::humanfloat`).
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub prune_every: f64,
    /// At most this many learned rows between checkpoints, `u32::MAX` for
    /// none; whichever of the two comes first checkpoints.
    pub max_rows_between_prunes: u32,
    /// Linkage threshold in units of `eps √p`; `None` derives it from the
    /// observed spacing at each checkpoint, `0` links nothing.
    pub macro_link: Option<f64>,
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

impl MicroCfg {
    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("micro: {e}"))?;
        if self.n_features == 0 {
            return Err("micro: n_features must be >= 1".into());
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err("micro: min_weight must be >= 0".into());
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("micro: eps must be finite and > 0".into());
        }
        if !self.beta_mu.is_finite() || self.beta_mu <= 0.0 {
            return Err("micro: beta_mu must be finite and > 0".into());
        }
        if self.max_clusters == 0 {
            return Err("micro: max_clusters must be >= 1".into());
        }
        if self.prune_every.is_nan() || self.prune_every < 0.0 {
            return Err(format!(
                "micro: prune_every must be >= 0 clock units (0 checkpoints on every row), got {}",
                self.prune_every
            ));
        }
        if let Some(l) = self.macro_link
            && (!l.is_finite() || l < 0.0)
        {
            return Err("micro: macro_link must be finite and >= 0".into());
        }
        if !self.scale_floor.is_finite() || self.scale_floor < 0.0 {
            return Err("micro: scale_floor must be finite and >= 0".into());
        }
        Ok(())
    }
}

/// The derived linkage threshold is this many times the 90th percentile
/// of the nearest-neighbour distance between potential summaries; see the
/// [module docs](self).
pub const LINK_FACTOR: f64 = 1.5;

/// The derived linkage threshold is never below this many `eps √p`:
/// two summaries whose radii are within `eps √p` and whose centres are
/// within twice that overlap (DenStream's density-reachability).
pub const LINK_FLOOR: f64 = 2.0;

/// The quantile of the nearest-neighbour spacing the derived threshold
/// reads (nearest rank).
pub const LINK_QUANTILE: f64 = 0.9;

/// The most potential summaries the linkage keeps a matrix of squared
/// distances for: `4096²` doubles, 128 MiB. Up to it each pair's distance
/// is computed once and read for both the spacing and the links; past it
/// each is taken when wanted, twice, in `O(m)` memory, 14-18% slower a
/// checkpoint at `m = 200` (review 2026-10-06, CF2). The two give the same
/// threshold, links and labels to the bit.
const LINK_MATRIX_MAX: usize = 4096;

/// The derived link threshold, squared: [`LINK_FACTOR`] times the
/// [`LINK_QUANTILE`] (nearest rank) of the nearest-neighbour distances,
/// whose squares `nn` holds, and never below [`LINK_FLOOR`]` · eps √p`.
fn derived_link2(mut nn: Vec<f64>, eps2: f64) -> f64 {
    let m = nn.len();
    nn.sort_by(f64::total_cmp);
    let rank = ((LINK_QUANTILE * m as f64).ceil() as usize).clamp(1, m) - 1;
    let p90 = nn[rank].sqrt();
    let floor = LINK_FLOOR * eps2.sqrt();
    let l = (LINK_FACTOR * p90).max(floor);
    l * l
}

/// Each of `m` summaries' component under single linkage, as the index of
/// its root, the smallest index in the component: `a` and `b` are linked
/// when `d2(a, b) <= link2` (`a < b`), and nothing is linked at a threshold
/// of 0.
fn components(m: usize, link2: f64, d2: impl Fn(usize, usize) -> f64) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    if link2 > 0.0 {
        for a in 0..m {
            for b in (a + 1)..m {
                if d2(a, b) <= link2 {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        // Ids ascend with the index, so the smaller index is
                        // the smaller id.
                        parent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
    }
    (0..m).map(|a| find(&mut parent, a)).collect()
}

/// One micro-cluster: a summary with its id, age, kind and cluster label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroCluster {
    /// Monotone, never reused.
    pub id: u64,
    pub s: ClusterSummary,
    /// Clock units since creation, for the pruning rule.
    pub age: f64,
    /// Potential (weight reached `beta_mu`) or still an outlier summary.
    pub potential: bool,
    /// The smallest id in its linkage component at the last checkpoint;
    /// its own id until then. Meaningful for potential summaries only.
    pub label: u64,
}

/// Where a row would go: the index of the summary that takes it, or `None`
/// for a new one; whether the nearest potential summary exists and its
/// squared distance; whether the row is an outlier to the potential ones.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Decision {
    target: Option<usize>,
    /// The nearest potential summary and the row's distance to it.
    nearest_potential: Option<(usize, f64)>,
    outlier: bool,
}

/// Micro-clusters; see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Micro {
    cfg: MicroCfg,
    /// EW mean/variance per feature; its weight is `n_eff`.
    moments: FeatureMoments,
    /// The metric for the next row, refreshed at the end of each step.
    mw: Vec<f64>,
    /// `eps² p`: the bound on a summary's radius² in the metric.
    eps2: f64,
    /// Live summaries in creation order (ids ascending).
    mc: Vec<MicroCluster>,
    next_id: u64,
    /// Learned rows since the last checkpoint.
    rows_since_prune: u32,
    /// Where `prune_every`'s clock stands (docs/PLAN.md task 163): the stamp
    /// of the last checkpoint, the decayed clock held exactly (task 180).
    since_prune: Since,
    /// Distinct labels among the potential summaries.
    n_clusters: usize,
    /// The linkage threshold² in force, from the last checkpoint.
    link2: f64,
    n_evicted: u64,
    n_pruned: u64,
    /// The EW mean weight of the rows learned from, and their EW count: the
    /// weight of the "unit" row a summary admits and of the rows `beta_mu`
    /// and `ξ` count, so a weight's scale moves nothing (docs/PLAN.md task
    /// 147). A state written before it reads as a mean weight of 1.
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
}

impl Micro {
    pub fn new(cfg: MicroCfg) -> Result<Self, String> {
        cfg.validate()?;
        let p = cfg.n_features;
        let eps2 = cfg.eps * cfg.eps * p as f64;
        let link2 = match cfg.macro_link {
            Some(l) => l * l * eps2,
            None => LINK_FLOOR * LINK_FLOOR * eps2,
        };
        Ok(Self {
            moments: FeatureMoments::new(p),
            mw: vec![1.0; p],
            eps2,
            mc: Vec::new(),
            next_id: 0,
            rows_since_prune: 0,
            since_prune: Since::default(),
            n_clusters: 0,
            link2,
            n_evicted: 0,
            n_pruned: 0,
            w_mean: 0.0,
            w_rows: 0.0,
            cfg,
        })
    }

    /// A row of the stream's mean weight: what `beta_mu` and `ξ` count, and
    /// what a summary admits. `1` before any row, and for any constant
    /// weight.
    fn unit(&self) -> f64 {
        if self.w_rows > 0.0 { self.w_mean } else { 1.0 }
    }

    pub fn cfg(&self) -> &MicroCfg {
        &self.cfg
    }

    /// EW weight of the learned rows: the model's `n_eff`.
    pub fn n_eff(&self) -> f64 {
        self.moments.w
    }

    /// The live summaries, in creation order.
    pub fn micro_clusters(&self) -> &[MicroCluster] {
        &self.mc
    }

    /// The metric in force for the next row.
    pub fn metric(&self) -> &[f64] {
        &self.mw
    }

    pub fn moments(&self) -> &FeatureMoments {
        &self.moments
    }

    /// Distinct labels among the potential summaries.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// The id the next new summary gets.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Cap evictions and checkpoint prunings so far.
    pub fn events(&self) -> (u64, u64) {
        (self.n_evicted, self.n_pruned)
    }

    /// The linkage threshold in force, in the metric (from the last
    /// checkpoint; the floor before the first).
    pub fn link(&self) -> f64 {
        self.link2.sqrt()
    }

    /// The potential summaries as `[id, label, n, radius, c_1 .. c_p]`
    /// rows, in id order; `None` while there is none.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        let rows: Vec<Vec<f64>> = self
            .mc
            .iter()
            .filter(|m| m.potential)
            .map(|m| {
                let mut row = Vec::with_capacity(self.cfg.n_features + 4);
                row.push(m.id as f64);
                row.push(m.label as f64);
                row.push(m.s.n);
                row.push(m.s.r2.max(0.0).sqrt());
                row.extend_from_slice(&m.s.c);
                row
            })
            .collect();
        (!rows.is_empty()).then_some(rows)
    }

    /// The nearest summary of the given kind, its squared distance and its
    /// distance (first minimum wins). The distance is the root of the square
    /// where that is finite, and the overflow-free norm ([`dist`]) where the
    /// square overflowed -- a row at the input bound against a variance at
    /// the opposite scale -- so what is reported is a number; among squares
    /// that all overflowed the norms decide, so the nearest is still the
    /// nearest (review 2026-09-26, G1).
    fn nearest(&self, z: &[f64], potential: bool) -> Option<(usize, f64, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (j, m) in self.mc.iter().enumerate() {
            if m.potential != potential {
                continue;
            }
            let d = dist2(&m.s.c, z, &self.mw);
            let closer = match best {
                None => true,
                Some((_, bd)) if d.is_finite() || bd.is_finite() => d < bd,
                Some((bj, _)) => dist(&m.s.c, z, &self.mw) < dist(&self.mc[bj].s.c, z, &self.mw),
            };
            if closer {
                best = Some((j, d));
            }
        }
        best.map(|(j, d2)| {
            let d = if d2.is_finite() {
                d2.sqrt()
            } else {
                dist(&self.mc[j].s.c, z, &self.mw)
            };
            (j, d2, d)
        })
    }

    /// Whether summary `j`, its weight decayed by `lam`, admits a row of the
    /// stream's mean weight at squared distance `d2`.
    fn admits(&self, j: usize, d2: f64, lam: f64) -> bool {
        let s = &self.mc[j].s;
        merged_radius2(s.n * lam, s.r2, d2, self.unit()) <= self.eps2
    }

    /// Where a unit-weight row at `z` goes, and what it is scored against,
    /// once every summary's weight has decayed by `lam` — `1` in
    /// [`step`](OnlineModel::step), which has already applied the clock,
    /// and the row's own factor in [`predict`](OnlineModel::predict).
    fn decide(&self, z: &[f64], lam: f64) -> Decision {
        let potential = self.nearest(z, true);
        let nearest_potential = potential.map(|(jp, _, d)| (jp, d));
        if let Some((jp, d2, _)) = potential
            && self.admits(jp, d2, lam)
        {
            return Decision {
                target: Some(jp),
                nearest_potential,
                outlier: false,
            };
        }
        if let Some((jo, d2, _)) = self.nearest(z, false)
            && self.admits(jo, d2, lam)
        {
            return Decision {
                target: Some(jo),
                nearest_potential,
                outlier: true,
            };
        }
        Decision {
            target: None,
            nearest_potential,
            outlier: true,
        }
    }

    /// The six outputs: `cluster`, `dist`, `micro_id`, `outlier`,
    /// `n_clusters`, `n_micro`; all NaN when not ready.
    fn score(&self, dec: Option<&Decision>, n_eff: f64) -> Vec<f64> {
        let mut pred = vec![f64::NAN; 6];
        if let Some(dec) = dec.filter(|_| n_eff >= self.cfg.min_weight) {
            if let Some((jp, d)) = dec.nearest_potential {
                pred[0] = self.mc[jp].label as f64;
                pred[1] = d;
            }
            pred[2] = match dec.target {
                Some(j) => self.mc[j].id as f64,
                None => self.next_id as f64,
            };
            pred[3] = if dec.outlier { 1.0 } else { 0.0 };
            pred[4] = self.n_clusters as f64;
            pred[5] = self.mc.len() as f64;
        }
        pred
    }

    fn learn_row(&mut self, z: &[f64], w: f64, dec: &Decision) {
        let promote_at = self.cfg.beta_mu * self.unit();
        match dec.target {
            Some(j) => {
                let m = &mut self.mc[j];
                m.s.absorb(z, w, &self.mw);
                // A unit row was admitted; a heavier one may overshoot.
                m.s.r2 = m.s.r2.min(self.eps2);
                if !m.potential && m.s.n >= promote_at {
                    m.potential = true;
                    self.attach(j);
                }
            }
            None => self.create(z, w),
        }
        self.rows_since_prune += 1;
    }

    /// Open a summary at `z`; at the cap, evict the lightest outlier
    /// summary first, else the lightest potential one (first minimum wins).
    fn create(&mut self, z: &[f64], w: f64) {
        if self.mc.len() >= self.cfg.max_clusters {
            let pick = |want_potential: bool| {
                let mut best: Option<(usize, f64)> = None;
                for (j, m) in self.mc.iter().enumerate() {
                    if m.potential == want_potential && best.is_none_or(|(_, bn)| m.s.n < bn) {
                        best = Some((j, m.s.n));
                    }
                }
                best.map(|(j, _)| j)
            };
            let j = pick(false)
                .or_else(|| pick(true))
                .expect("max_clusters >= 1, so a live summary exists at the cap");
            self.drop_at(j);
            self.n_evicted += 1;
        }
        let potential = w >= self.cfg.beta_mu * self.unit();
        let id = self.next_id;
        self.next_id += 1;
        self.mc.push(MicroCluster {
            id,
            s: ClusterSummary::at(z.to_vec(), w, 0.0),
            age: 0.0,
            potential,
            label: id,
        });
        if potential {
            self.attach(self.mc.len() - 1);
        }
    }

    /// A summary just promoted takes the label of the nearest other
    /// potential summary within the linkage threshold, so that a growing
    /// shape's rows keep their label between checkpoints; with none in
    /// reach it starts a cluster of its own.
    fn attach(&mut self, j: usize) {
        let mut best: Option<(usize, f64)> = None;
        for (o, m) in self.mc.iter().enumerate() {
            if o == j || !m.potential {
                continue;
            }
            let d = dist2(&m.s.c, &self.mc[j].s.c, &self.mw);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((o, d));
            }
        }
        match best {
            Some((o, d)) if self.link2 > 0.0 && d <= self.link2 => {
                self.mc[j].label = self.mc[o].label;
            }
            _ => {
                self.mc[j].label = self.mc[j].id;
                self.n_clusters += 1;
            }
        }
    }

    /// Remove summary `j`, keeping the cluster count right: a potential
    /// summary that was the last of its label takes the label with it.
    fn drop_at(&mut self, j: usize) {
        let gone = self.mc.remove(j);
        if gone.potential && !self.mc.iter().any(|m| m.potential && m.label == gone.label) {
            self.n_clusters -= 1;
        }
    }

    /// The decay's half-life in clock units (infinite for none).
    fn half_life(&self) -> f64 {
        match self.cfg.decay {
            Decay::Halflife(h) => h,
            Decay::Lam(l) => {
                if l >= 1.0 {
                    f64::INFINITY
                } else {
                    -std::f64::consts::LN_2 / l.ln()
                }
            }
        }
    }

    /// DenStream's `(Tp, 2^(−Tp/h))`: `None` when weights do not fade or
    /// `beta_mu ≤ 1`, in which case outlier summaries are never pruned and
    /// only the cap bounds them.
    fn prune_horizon(&self) -> Option<(f64, f64)> {
        let h = self.half_life();
        if !h.is_finite() || self.cfg.beta_mu <= 1.0 {
            return None;
        }
        let tp = (h * (self.cfg.beta_mu / (self.cfg.beta_mu - 1.0)).log2()).ceil();
        let f_tp = self.cfg.decay.factor(tp);
        (f_tp < 1.0).then_some((tp, f_tp))
    }

    /// Prune, then link.
    fn checkpoint(&mut self) {
        let horizon = self.prune_horizon();
        let unit = self.unit();
        for j in (0..self.mc.len()).rev() {
            let m = &self.mc[j];
            let dead = if m.potential {
                m.s.n < self.cfg.beta_mu * unit
            } else if let Some((_, f_tp)) = horizon {
                let age_decay = self.cfg.decay.factor(m.age);
                let xi = (age_decay * f_tp - 1.0) / (f_tp - 1.0);
                m.s.n < xi * unit
            } else {
                false
            };
            if dead {
                self.drop_at(j);
                self.n_pruned += 1;
            }
        }
        self.link_potential();
    }

    /// Single linkage over the potential summaries; labels = the smallest
    /// id in each component. Up to [`LINK_MATRIX_MAX`] of them through a
    /// matrix of their squared distances, past it through each pair's
    /// distance taken when it is wanted (review 2026-10-06, CF2).
    fn link_potential(&mut self) {
        let idx: Vec<usize> = (0..self.mc.len())
            .filter(|&j| self.mc[j].potential)
            .collect();
        if idx.len() <= LINK_MATRIX_MAX {
            self.link_by_matrix(&idx);
        } else {
            self.link_by_pairs(&idx);
        }
    }

    /// The linkage over the potential summaries `idx` with each pair's
    /// squared distance computed once, into an `m × m` matrix, and read for
    /// the nearest-neighbour spacing and for the links: the step as it
    /// always ran, `m²` doubles.
    fn link_by_matrix(&mut self, idx: &[usize]) {
        let m = idx.len();
        // Pairwise squared distances, upper triangle by (a, b), a < b.
        let mut d2 = vec![0.0; m * m];
        for a in 0..m {
            for b in (a + 1)..m {
                let d = dist2(&self.mc[idx[a]].s.c, &self.mc[idx[b]].s.c, &self.mw);
                d2[a * m + b] = d;
                d2[b * m + a] = d;
            }
        }
        if self.cfg.macro_link.is_none() && m >= 2 {
            let nn: Vec<f64> = (0..m)
                .map(|a| {
                    (0..m)
                        .filter(|&b| b != a)
                        .map(|b| d2[a * m + b])
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            self.link2 = derived_link2(nn, self.eps2);
        }
        let roots = components(m, self.link2, |a, b| d2[a * m + b]);
        self.label(idx, &roots);
    }

    /// The same linkage in `O(m)` memory: each pair's squared distance is
    /// taken when it is wanted, once for the nearest-neighbour spacing and
    /// once for the links, where the matrix would be `max_clusters²`
    /// doubles -- 8 TB at a cap of 10^6 (review 2026-10-06, CF2). The
    /// distances are the same numbers, taken with the same arguments, so the
    /// threshold, the links and the labels are the matrix's to the bit; the
    /// minimum over a summary's neighbours does not depend on their order.
    fn link_by_pairs(&mut self, idx: &[usize]) {
        let m = idx.len();
        let (mc, mw) = (&self.mc, &self.mw);
        // The squared distance of the pair `(a, b)`, `a < b`.
        let d2 = |a: usize, b: usize| dist2(&mc[idx[a]].s.c, &mc[idx[b]].s.c, mw);
        if self.cfg.macro_link.is_none() && m >= 2 {
            let mut nn = vec![f64::INFINITY; m];
            for a in 0..m {
                for b in (a + 1)..m {
                    let d = d2(a, b);
                    nn[a] = nn[a].min(d);
                    nn[b] = nn[b].min(d);
                }
            }
            self.link2 = derived_link2(nn, self.eps2);
        }
        let roots = components(m, self.link2, d2);
        self.label(idx, &roots);
    }

    /// Each potential summary `idx[a]` labelled with the id of its
    /// component's root `idx[roots[a]]`, and the clusters counted.
    fn label(&mut self, idx: &[usize], roots: &[usize]) {
        let mut n_clusters = 0;
        for (a, &root) in roots.iter().enumerate() {
            if root == a {
                n_clusters += 1;
            }
            self.mc[idx[a]].label = self.mc[idx[root]].id;
        }
        self.n_clusters = n_clusters;
    }
}

impl OnlineModel for Micro {
    /// The checkpoint's cadence measures its clock by the row's stamp (task
    /// 180).
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        self.since_prune.stamp_next(stamp);
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature or a weight past the input bound was
        // learned, the feature as a summary of its own.
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
        for m in &mut self.mc {
            m.s.decay(lam);
            m.age += d_clock;
        }
        self.w_rows *= lam;

        // Decided once, as a row of the mean weight, and read before
        // anything moves.
        let dec = valid.then(|| self.decide(x, 1.0));
        let pred = self.score(dec.as_ref(), n_before);

        if let Some(dec) = dec.filter(|_| learn) {
            self.w_rows += 1.0;
            self.w_mean += (weight - self.w_mean) / self.w_rows;
            self.moments.absorb(x, weight);
            self.learn_row(x, weight, &dec);
        }
        // The checkpoint, on the clock or the learned rows, whichever comes
        // first (task 163). The clock passes on every row, so a quiet spell
        // prunes on time, where the learned rows alone waited for the next
        // learned row; it runs where it did, after the row and before the
        // metric for the next one. The clock is the row's stamp (task 180),
        // and `0` checkpoints on every row whatever the stamps say.
        self.since_prune.step(d_clock);
        if self.cfg.prune_every <= 0.0
            || self.since_prune.reached(self.cfg.prune_every)
            || self.rows_since_prune >= self.cfg.max_rows_between_prunes
        {
            self.rows_since_prune = 0;
            self.since_prune.restart();
            self.checkpoint();
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
        // The clock decides admission: a summary's weight sets how far it
        // lets a row move its radius.
        let valid = x.iter().all(|v| v.is_finite());
        let dec = valid.then(|| self.decide(x, self.cfg.decay.factor(d_clock)));
        Step {
            pred: self.score(dec.as_ref(), self.moments.w),
            n_eff: self.moments.w,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Micro(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Micro(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("micro", m.cfg.validate())?;
                // The feature moments, the metric weights and every centre
                // `p` wide (review 2026-09-18, B3).
                let p = m.cfg.n_features;
                if !m.moments.has_shape(p)
                    || m.mw.len() != p
                    || m.mc.iter().any(|c| c.s.c.len() != p)
                {
                    return Err(StateError::Invalid(
                        "micro: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "micro",
                found: other.kind(),
            }),
        }
    }

    /// Zero: `micro` regresses nothing, as `kmeans` does.
    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// `cluster`, `dist`, `micro_id`, `outlier`, `n_clusters`, `n_micro`.
    fn n_outputs(&self) -> usize {
        6
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
        let m = Micro::new(cfg()).unwrap();
        let mut s = m.state();
        let ModelState::Micro(inner) = &mut s.model else {
            unreachable!()
        };
        inner.mw.pop();
        match Micro::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    /// A state's configuration is held to what `new` holds a fresh one to:
    /// `max_clusters = 0` loaded, and the first learned row evicted at the
    /// cap with no summary to evict, and hit `create`'s `expect` (review
    /// 2026-10-06, CF4).
    #[test]
    fn a_restored_state_whose_cfg_new_refuses_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let mut s = Micro::new(cfg()).unwrap().state();
        let ModelState::Micro(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cfg.max_clusters = 0;
        assert!(
            inner.cfg.validate().is_err(),
            "`new` refuses max_clusters = 0"
        );
        match Micro::restore(&s) {
            Err(StateError::Invalid(e)) => {
                assert!(
                    e.contains("configuration") && e.contains("max_clusters"),
                    "{e}"
                );
            }
            Ok(mut back) => {
                back.step(&[0.0, 0.0], &[], 0.0, 1.0);
                panic!("max_clusters = 0 loaded and ran");
            }
            Err(e) => panic!("{e}"),
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Box–Muller from the lcg.
    fn gauss(state: &mut u64) -> f64 {
        let u1 = lcg(state).max(1e-300);
        let u2 = lcg(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn cfg() -> MicroCfg {
        MicroCfg {
            n_features: 2,
            decay: Decay::Halflife(500.0),
            min_weight: 0.0,
            eps: 0.3,
            beta_mu: 3.0,
            max_clusters: 50,
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 50,
            macro_link: None,
            standardize: false,
            scale_floor: 0.0,
        }
    }

    /// The checkpoint's schedule (task 163): every `prune_every` clock units
    /// or every `max_rows_between_prunes` learned rows, whichever comes
    /// first. Held to that rule written out over irregular clock steps with
    /// rows of weight zero among them, which advance the clock and not the
    /// row count. A checkpoint shows as both counters reset on a row that
    /// moved one of them.
    #[test]
    fn checkpoints_follow_the_clock_or_the_learned_rows_whichever_comes_first() {
        let mut g = crate::SplitMix64::new(4);
        let rows: Vec<([f64; 2], f64, f64)> = (0..120)
            .map(|i| {
                let d = if i == 0 {
                    0.0
                } else {
                    0.25 + g.uniform() * 2.0
                };
                let w = if i % 4 == 3 { 0.0 } else { 1.0 };
                ([g.uniform(), g.uniform()], d, w)
            })
            .collect();
        for (every, cap) in [(6.0, u32::MAX), (f64::INFINITY, 7), (6.0, 7)] {
            let mut m = Micro::new(MicroCfg {
                prune_every: every,
                max_rows_between_prunes: cap,
                ..cfg()
            })
            .unwrap();
            let mut got = Vec::new();
            for (i, (x, d, w)) in rows.iter().enumerate() {
                let before = (m.rows_since_prune, m.since_prune.summed());
                crate::OnlineModel::step(&mut m, x, &[], *d, *w);
                let moved = before.0 + u32::from(*w > 0.0) > 0 || before.1 + d > 0.0;
                if moved && (m.rows_since_prune, m.since_prune.summed()) == (0, 0.0) {
                    got.push(i);
                }
            }
            let (mut want, mut clock, mut learned) = (Vec::new(), 0.0, 0u32);
            for (i, (_, d, w)) in rows.iter().enumerate() {
                clock += d;
                learned += u32::from(*w > 0.0);
                if clock >= every || learned >= cap {
                    want.push(i);
                    (clock, learned) = (0.0, 0);
                }
            }
            assert_eq!(
                got, want,
                "prune_every {every}, max_rows_between_prunes {cap}"
            );
            assert!(want.len() >= 8, "{want:?}");
        }
    }

    /// The reason for the clock (task 163): a summary that has faded is
    /// pruned on time through a quiet spell of rows of weight zero, where a
    /// cadence of learned rows alone waited for the next learned row.
    #[test]
    fn a_quiet_spell_still_prunes_on_the_clock() {
        let run = |every: f64| {
            let mut m = Micro::new(MicroCfg {
                decay: Decay::Halflife(20.0),
                prune_every: every,
                max_rows_between_prunes: 100,
                ..cfg()
            })
            .unwrap();
            // Three summaries far apart, then 400 clock units of rows that
            // teach nothing.
            for (i, x) in [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]].iter().enumerate() {
                crate::OnlineModel::step(&mut m, x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let opened = m.mc.len();
            for _ in 0..400 {
                crate::OnlineModel::step(&mut m, &[5.0, 5.0], &[], 1.0, 0.0);
            }
            (opened, m.mc.len())
        };
        assert_eq!(
            run(f64::INFINITY),
            (3, 3),
            "learned rows alone never checkpoint here"
        );
        let (opened, left) = run(25.0);
        assert_eq!(opened, 3);
        assert_eq!(left, 0, "the clock prunes the faded summaries");
    }

    /// Task 180: `prune_every` reads the stamps its caller hands: every
    /// 2,000th 1 ms row under 2 s, the clock running from the first row,
    /// where the clock summed since the last checkpoint, read when none is
    /// handed, is a row late each time, as it always was. Every row is
    /// learned, so the learned rows since the last are none only there.
    #[test]
    fn prune_every_measures_the_stamps_it_is_handed() {
        for (stamped, want) in [(true, [2000, 4000]), (false, [2001, 4002])] {
            let mut m = Micro::new(MicroCfg {
                prune_every: 2.0,
                max_rows_between_prunes: u32::MAX,
                ..cfg()
            })
            .unwrap();
            let mut g = crate::SplitMix64::new(6);
            let got = crate::since::events_on_millisecond_rows(4_100, stamped, |_, stamp, d| {
                if let Some(s) = stamp {
                    crate::OnlineModel::stamp_next(&mut m, s);
                }
                crate::OnlineModel::step(&mut m, &[g.uniform(), g.uniform()], &[], d, 1.0);
                m.rows_since_prune == 0
            });
            assert_eq!(got, want, "stamped: {stamped}");
        }
    }

    /// A save between checkpoints keeps the clock and the learned rows since
    /// the last, so a resumed run checkpoints where the unbroken one does
    /// (task 163).
    #[test]
    fn the_clock_since_the_last_checkpoint_survives_a_save() {
        let mut g = crate::SplitMix64::new(9);
        let rows: Vec<([f64; 2], f64, f64)> = (0..90)
            .map(|i| {
                let d = if i == 0 { 0.0 } else { 0.5 + g.uniform() };
                let w = if i % 5 == 4 { 0.0 } else { 1.0 };
                ([g.uniform(), g.uniform()], d, w)
            })
            .collect();
        let c = MicroCfg {
            prune_every: 4.0,
            max_rows_between_prunes: 9,
            ..cfg()
        };
        let mut a = Micro::new(c.clone()).unwrap();
        let (mut want, mut mid) = (Vec::new(), Vec::new());
        for (x, d, w) in &rows {
            want.push(crate::OnlineModel::step(&mut a, x, &[], *d, *w).pred);
            mid.push(a.since_prune.summed() > 0.0);
        }
        let cuts: Vec<usize> = [10, 37, 61]
            .iter()
            .map(|&from| (from..rows.len()).find(|&i| mid[i - 1]).unwrap())
            .collect();
        for cut in cuts {
            let mut b = Micro::new(c.clone()).unwrap();
            for (i, (x, d, w)) in rows.iter().enumerate() {
                if i == cut {
                    let bytes = rmp_serde::to_vec(&b).unwrap();
                    b = rmp_serde::from_slice(&bytes).unwrap();
                }
                let out = crate::OnlineModel::step(&mut b, x, &[], *d, *w).pred;
                let same = out
                    .iter()
                    .zip(&want[i])
                    .all(|(p, q)| p.to_bits() == q.to_bits());
                assert!(same, "cut {cut}, row {i}");
            }
        }
    }

    /// A row of weight `1e-100`, then one of `1e100`, leave a standardized
    /// spread of `4e-200`; the next row at the input bound is `5e199` scaled
    /// units away, whose square overflows. The distance reported is that
    /// number, the row is an outlier, and the state stays finite (review
    /// 2026-09-26, G1: the contract's proptest at the bound found `dist`
    /// infinite and the row absorbed into the summary it was infinitely far
    /// from).
    #[test]
    fn a_row_at_the_bound_against_a_vanishing_spread_is_far_not_absorbed() {
        use crate::OnlineModel;
        let mut m = Micro::new(MicroCfg {
            standardize: true,
            scale_floor: 0.1,
            min_weight: 3.0,
            eps: 0.6,
            beta_mu: 2.0,
            decay: Decay::Halflife(20.0),
            ..cfg()
        })
        .unwrap();
        m.step(&[0.0, 0.0], &[], 0.0, 1e-100);
        // Three rows at the bound's weight, so the summary holds more than
        // `beta_mu` rows of the stream's mean weight, which they dominate
        // (task 147).
        for _ in 0..3 {
            m.step(&[0.0, -2.0709248242631726], &[], 1.0, 1e100);
        }
        let far = m.step(&[0.0, 1e100], &[], 1.0, 1.0);
        assert!(far.pred[1].is_finite(), "{:?}", far.pred);
        assert!(far.pred[1] > 1e150, "{:?}", far.pred);
        assert_eq!(far.pred[3], 1.0, "an outlier: {:?}", far.pred);
        assert!(far.n_eff.is_finite());
        let after = m.predict(&[0.0, -2.0], 1.0);
        assert!(after.pred.iter().all(|v| v.is_finite()), "{:?}", after.pred);
        assert!(
            after.pred[1] < 1e3,
            "the summary stayed where it was: {:?}",
            after.pred
        );
    }

    /// Two blobs of unit-ish spread around (0, 0) and (10, 10), interleaved.
    fn blobs(n: usize, seed: u64) -> Vec<([f64; 2], usize)> {
        let mut st = seed;
        (0..n)
            .map(|i| {
                let k = i % 2;
                let c = 10.0 * k as f64;
                ([c + 0.5 * gauss(&mut st), c + 0.5 * gauss(&mut st)], k)
            })
            .collect()
    }

    /// Equal as bit patterns, so NaN slots compare equal.
    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn run(m: &mut Micro, rows: &[([f64; 2], usize)]) -> Vec<Vec<f64>> {
        rows.iter()
            .map(|(x, _)| m.step(x, &[], 1.0, 1.0).pred)
            .collect()
    }

    #[test]
    fn ids_are_monotone_and_never_reused() {
        let mut m = Micro::new(MicroCfg {
            max_clusters: 4,
            eps: 0.05,
            ..cfg()
        })
        .unwrap();
        let mut st = 7u64;
        let mut seen_max = -1i64;
        let mut live_ids: Vec<u64> = Vec::new();
        for _ in 0..2000 {
            let x = [10.0 * lcg(&mut st), 10.0 * lcg(&mut st)];
            let pred = m.step(&x, &[], 1.0, 1.0).pred;
            let micro = pred[2] as i64;
            // The id a row is sent to is at most the next id: never one
            // that was evicted.
            assert!(micro <= m.next_id() as i64);
            let ids: Vec<u64> = m.micro_clusters().iter().map(|c| c.id).collect();
            assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids ascend: {ids:?}");
            for &id in &ids {
                if !live_ids.contains(&id) {
                    assert!(id as i64 > seen_max, "id {id} came back");
                    seen_max = id as i64;
                }
            }
            live_ids = ids;
            assert!(m.micro_clusters().len() <= 4);
        }
        assert!(m.events().0 > 100, "the cap evicted: {:?}", m.events());
    }

    #[test]
    fn a_row_is_labelled_with_the_id_it_opens() {
        let mut m = Micro::new(cfg()).unwrap();
        let first = m.step(&[0.0, 0.0], &[], 1.0, 1.0).pred;
        assert_eq!(first[2], 0.0, "the first row opens id 0 and says so");
        assert_eq!(
            first[3], 1.0,
            "and is an outlier to the (no) potential summaries"
        );
        assert!(first[0].is_nan() && first[1].is_nan());
        assert_eq!(first[4], 0.0);
        assert_eq!(first[5], 0.0);
        // A far row opens id 1.
        let far = m.step(&[10.0, 10.0], &[], 1.0, 1.0).pred;
        assert_eq!(far[2], 1.0);
        assert_eq!(far[5], 1.0, "one summary was live before it");
        // A row near the first summary goes to it.
        let near = m.step(&[0.01, 0.0], &[], 1.0, 1.0).pred;
        assert_eq!(near[2], 0.0);
        assert_eq!(m.micro_clusters().len(), 2);
    }

    #[test]
    fn promotion_happens_at_beta_mu_and_a_potential_summary_labels_rows() {
        let mut m = Micro::new(cfg()).unwrap();
        for i in 0..3 {
            let pred = m.step(&[0.0, 0.001 * i as f64], &[], 1.0, 1.0).pred;
            assert!(pred[0].is_nan(), "no potential summary yet at row {i}");
            assert_eq!(pred[3], 1.0);
        }
        // Weight 3 (three rows at half-life 500 decay a little: 2.997) — not
        // yet; the fourth row promotes.
        assert!(!m.micro_clusters()[0].potential);
        m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        assert!(m.micro_clusters()[0].potential);
        assert_eq!(m.n_clusters(), 1);
        let pred = m.step(&[0.0, 0.0], &[], 1.0, 1.0).pred;
        assert_eq!(pred[0], 0.0, "cluster = the label = the id");
        assert!(pred[1] < 0.01, "the centre sits near the rows: {}", pred[1]);
        assert_eq!(pred[3], 0.0, "not an outlier");
        assert_eq!(pred[4], 1.0);
    }

    #[test]
    fn a_heavy_row_promotes_on_creation() {
        // Heavy against the stream's mean weight (task 147): after unit rows
        // elsewhere, a row of five times it opens a potential summary; a
        // first row at weight 5 is a row of the mean weight, and does not.
        let mut m = Micro::new(cfg()).unwrap();
        m.step(&[0.0, 0.0], &[], 1.0, 5.0);
        assert!(!m.micro_clusters()[0].potential);
        let mut m = Micro::new(cfg()).unwrap();
        // The mean weight counts the heavy row too: twenty unit rows and it
        // make about 1.19, so promotion wants 3.57.
        for _ in 0..20 {
            m.step(&[50.0, 50.0], &[], 1.0, 1.0);
        }
        m.step(&[0.0, 0.0], &[], 1.0, 5.0);
        let opened = m.micro_clusters().last().unwrap();
        assert!(opened.potential, "{opened:?}");
        assert_eq!(m.n_clusters(), 2);
    }

    /// Task 147: a hundred times every weight opens, promotes, prunes and
    /// labels the same summaries.
    #[test]
    fn a_weights_scale_moves_nothing() {
        use crate::OnlineModel;
        let run = |scale: f64| {
            let mut m = Micro::new(MicroCfg {
                decay: Decay::Halflife(30.0),
                prune_every: f64::INFINITY,
                max_rows_between_prunes: 7,
                ..cfg()
            })
            .unwrap();
            let mut s = 101u64;
            let mut out = Vec::new();
            for i in 0..600 {
                let c = [0.0, 6.0, 12.0][(i / 50) % 3];
                let x = [c + lcg(&mut s), lcg(&mut s)];
                let w = scale * (0.5 + 0.5 * (lcg(&mut s) + 1.0));
                out.push(m.step(&x, &[], 1.0, w).pred);
            }
            out
        };
        let (one, many) = (run(1.0), run(100.0));
        for (i, (a, b)) in one.iter().zip(&many).enumerate() {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * (1.0 + x.abs()),
                    "row {i}: {a:?} against {b:?}"
                );
            }
        }
    }

    #[test]
    fn two_blobs_become_two_clusters_with_pure_labels() {
        let rows = blobs(4000, 3);
        let mut m = Micro::new(cfg()).unwrap();
        let out = run(&mut m, &rows);
        // After the first checkpoint, every row of a blob carries one label
        // and the two blobs carry different ones.
        let mut labels = [None, None];
        for (i, (pred, (_, k))) in out.iter().zip(&rows).enumerate().skip(400) {
            assert!(!pred[0].is_nan(), "row {i} unlabelled");
            match labels[*k] {
                None => labels[*k] = Some(pred[0]),
                Some(l) => assert_eq!(l, pred[0], "row {i} of blob {k} changed label"),
            }
        }
        assert_ne!(labels[0], labels[1]);
        assert_eq!(m.n_clusters(), 2);
        assert!(out[3999][4] == 2.0);
        // A potential summary's radius is within the bound.
        for c in m.micro_clusters().iter().filter(|c| c.potential) {
            assert!(c.s.r2 <= 0.3 * 0.3 * 2.0 + 1e-12, "r2 {}", c.s.r2);
        }
    }

    #[test]
    fn the_derived_threshold_clears_the_spacing_and_the_floor() {
        let rows = blobs(4000, 5);
        let mut m = Micro::new(cfg()).unwrap();
        run(&mut m, &rows);
        let floor = LINK_FLOOR * 0.3 * 2f64.sqrt();
        assert!(m.link() >= floor - 1e-12, "{} < floor {floor}", m.link());
        // An override is used as given.
        let mut o = Micro::new(MicroCfg {
            macro_link: Some(0.5),
            ..cfg()
        })
        .unwrap();
        run(&mut o, &rows);
        assert!((o.link() - 0.5 * 0.3 * 2f64.sqrt()).abs() < 1e-12);
        // Zero links nothing: every potential summary is its own cluster.
        let mut z = Micro::new(MicroCfg {
            macro_link: Some(0.0),
            ..cfg()
        })
        .unwrap();
        run(&mut z, &rows);
        let potential = z.micro_clusters().iter().filter(|c| c.potential).count();
        assert_eq!(z.n_clusters(), potential);
        assert!(potential > 2);
    }

    #[test]
    fn pruning_drops_a_faded_outlier_summary_by_the_xi_rule() {
        let mut m = Micro::new(MicroCfg {
            decay: Decay::Halflife(100.0),
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 10,
            ..cfg()
        })
        .unwrap();
        // One stray row, then a blob elsewhere for a long time.
        m.step(&[50.0, 50.0], &[], 1.0, 1.0);
        assert_eq!(m.micro_clusters().len(), 1);
        let mut st = 1u64;
        let mut gone_at = None;
        for i in 0..2000 {
            let x = [0.1 * gauss(&mut st), 0.1 * gauss(&mut st)];
            m.step(&x, &[], 1.0, 1.0);
            if gone_at.is_none() && !m.micro_clusters().iter().any(|c| c.id == 0) {
                gone_at = Some(i);
            }
        }
        // xi(age) rises from 1 to beta_mu = 3 over a few Tp = 59 clock
        // units; a weight-1 summary decays below it within the first
        // checkpoints.
        let gone = gone_at.expect("the stray summary was pruned");
        assert!(gone < 100, "pruned at {gone}");
        assert!(m.events().1 >= 1);
    }

    #[test]
    fn without_decay_outlier_summaries_are_never_pruned_only_capped() {
        let mut m = Micro::new(MicroCfg {
            decay: Decay::Halflife(f64::INFINITY),
            max_clusters: 8,
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 5,
            ..cfg()
        })
        .unwrap();
        let mut st = 9u64;
        for _ in 0..500 {
            let x = [100.0 * lcg(&mut st), 100.0 * lcg(&mut st)];
            m.step(&x, &[], 1.0, 1.0);
        }
        assert_eq!(m.events().1, 0, "nothing pruned");
        assert!(m.events().0 > 400, "the cap did the bounding");
        assert_eq!(m.micro_clusters().len(), 8);
    }

    #[test]
    fn a_potential_summary_lighter_than_beta_mu_is_pruned_at_the_checkpoint() {
        let mut m = Micro::new(MicroCfg {
            decay: Decay::Halflife(20.0),
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 10,
            ..cfg()
        })
        .unwrap();
        for _ in 0..10 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        assert!(m.micro_clusters()[0].potential);
        // Rows elsewhere; the first summary fades below beta_mu.
        let mut seen = m.micro_clusters().len();
        for _ in 0..200 {
            m.step(&[30.0, 30.0], &[], 1.0, 1.0);
            seen = seen.max(m.micro_clusters().len());
        }
        assert!(seen >= 2);
        assert!(!m.micro_clusters().iter().any(|c| c.id == 0), "id 0 pruned");
        assert_eq!(m.n_clusters(), 1);
    }

    #[test]
    fn the_cap_evicts_the_lightest_outlier_before_any_potential_summary() {
        let mut m = Micro::new(MicroCfg {
            max_clusters: 3,
            decay: Decay::Halflife(f64::INFINITY),
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 1000,
            ..cfg()
        })
        .unwrap();
        for _ in 0..5 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0); // id 0, potential
        }
        m.step(&[10.0, 0.0], &[], 1.0, 1.0); // id 1, outlier, weight 1
        m.step(&[20.0, 0.0], &[], 1.0, 1.0); // id 2, outlier, weight 1
        m.step(&[20.0, 0.0], &[], 1.0, 1.0); // id 2 gains: weight 2
        let pred = m.step(&[30.0, 0.0], &[], 1.0, 1.0).pred; // opens id 3, evicts id 1
        assert_eq!(pred[2], 3.0);
        let ids: Vec<u64> = m.micro_clusters().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![0, 2, 3]);
        assert_eq!(m.n_clusters(), 1);
        // Only potential summaries left: the lightest of them goes, and the
        // cluster count follows.
        let mut all_pot = Micro::new(MicroCfg {
            max_clusters: 2,
            decay: Decay::Halflife(f64::INFINITY),
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 1000,
            ..cfg()
        })
        .unwrap();
        // Potential by rows of the mean weight (task 147): three and four.
        for _ in 0..3 {
            all_pot.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        for _ in 0..4 {
            all_pot.step(&[10.0, 0.0], &[], 1.0, 1.0);
        }
        assert_eq!(all_pot.n_clusters(), 2);
        let pred = all_pot.step(&[20.0, 0.0], &[], 1.0, 1.0).pred;
        assert_eq!(pred[4], 2.0, "read before the eviction");
        assert_eq!(all_pot.n_clusters(), 1);
        let ids: Vec<u64> = all_pot.micro_clusters().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn a_zero_weight_row_advances_the_clock_and_learns_nothing_even_first() {
        let mut m = Micro::new(cfg()).unwrap();
        let s = m.step(&[1.0, 2.0], &[], 1.0, 0.0);
        assert_eq!(s.n_eff, 0.0);
        assert!(m.micro_clusters().is_empty());
        assert_eq!(m.n_eff(), 0.0);
        assert_eq!(s.pred[2], 0.0, "it would open id 0");
        for _ in 0..5 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        let before = m.clone();
        let s = m.step(&[0.0, 0.0], &[], 3.0, 0.0);
        let lam = 0.5f64.powf(3.0 / 500.0);
        assert!((m.n_eff() - before.n_eff() * lam).abs() < 1e-12);
        assert!((m.micro_clusters()[0].s.n - before.micro_clusters()[0].s.n * lam).abs() < 1e-12);
        assert_eq!(m.micro_clusters()[0].s.c, before.micro_clusters()[0].s.c);
        assert_eq!(
            m.micro_clusters()[0].age,
            before.micro_clusters()[0].age + 3.0
        );
        assert_eq!(s.pred[3], 0.0, "a unit-weight row at the centre is taken");
        // A zero weight is asked as a unit weight: the same answer as predict.
        assert!(same(&s.pred, &before.predict(&[0.0, 0.0], 3.0).pred));
    }

    #[test]
    fn a_non_finite_feature_row_is_not_scored_and_not_learned() {
        let mut m = Micro::new(cfg()).unwrap();
        for _ in 0..5 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        let before = m.clone();
        let s = m.step(&[f64::NAN, 0.0], &[], 1.0, 1.0);
        assert!(s.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.micro_clusters()[0].s.c, before.micro_clusters()[0].s.c);
        assert_eq!(m.micro_clusters().len(), 1);
        assert!((m.n_eff() - before.n_eff() * 0.5f64.powf(1.0 / 500.0)).abs() < 1e-12);
    }

    #[test]
    fn outputs_are_null_until_min_periods() {
        let mut m = Micro::new(MicroCfg {
            min_weight: 3.0,
            decay: Decay::Halflife(f64::INFINITY),
            ..cfg()
        })
        .unwrap();
        let mut ready = Vec::new();
        for _ in 0..6 {
            let s = m.step(&[0.0, 0.0], &[], 1.0, 1.0);
            ready.push(!s.pred[2].is_nan());
        }
        assert_eq!(ready, vec![false, false, false, true, true, true]);
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let rows = blobs(600, 11);
        let mut m = Micro::new(cfg()).unwrap();
        for (x, _) in &rows {
            let before = m.clone();
            let p = before.predict(x, 1.0);
            let s = m.step(x, &[], 1.0, 1.0);
            assert!(same(&p.pred, &s.pred), "{:?} vs {:?}", p.pred, s.pred);
            assert_eq!(p.n_eff, s.n_eff);
        }
    }

    #[test]
    fn a_heavy_row_is_decided_as_a_unit_row() {
        let mut m = Micro::new(MicroCfg {
            decay: Decay::Halflife(f64::INFINITY),
            ..cfg()
        })
        .unwrap();
        for _ in 0..5 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        // A tight summary of weight 5 at the origin. Absorbing a row at
        // distance² q with weight w leaves radius² a·b·q, a = 5/(5+w),
        // b = w/(5+w): 0.139 q for a unit row, 0.25 q for one of weight 5.
        // At q = 1.1025 the unit row stays within eps² p = 0.18 and a
        // weight-5 row would not — but the decision is the unit row's, so
        // the heavy row is absorbed too, predict says so beforehand, the
        // centre moves by the full weight, and the radius is capped.
        let eps2 = 0.3f64.powi(2) * 2.0;
        let before = m.clone();
        let said = before.predict(&[1.05, 0.0], 1.0).pred;
        let did = m.step(&[1.05, 0.0], &[], 1.0, 5.0).pred;
        assert!(same(&said, &did), "{said:?} vs {did:?}");
        assert_eq!(did[3], 0.0, "{did:?}");
        assert_eq!(did[2], 0.0, "absorbed by id 0");
        assert_eq!(m.micro_clusters().len(), 1);
        let s = &m.micro_clusters()[0].s;
        assert_eq!(s.n, 10.0);
        assert_eq!(s.c, vec![0.525, 0.0]);
        assert_eq!(s.r2, eps2, "0.25 · 1.1025 = 0.276 capped at the bound");
        // Full, not locked: a row at the centre is still admitted, and
        // shrinks the radius.
        let next = m.step(&[0.525, 0.0], &[], 1.0, 1.0).pred;
        assert_eq!(next[2], 0.0, "{next:?}");
        assert_eq!(m.micro_clusters().len(), 1);
        assert_eq!(m.micro_clusters()[0].s.r2, 10.0 / 11.0 * eps2);
    }

    #[test]
    fn the_radius_never_exceeds_the_bound() {
        // Weights from 0.5 to 8 on two blobs, every row's summaries checked.
        let rows = blobs(1500, 21);
        let mut m = Micro::new(cfg()).unwrap();
        let eps2 = 0.3f64.powi(2) * 2.0;
        let mut capped = 0;
        for (i, (x, _)) in rows.iter().enumerate() {
            let w = [0.5, 1.0, 2.0, 8.0][i % 4];
            m.step(x, &[], 1.0, w);
            for mc in m.micro_clusters() {
                assert!(mc.s.r2 <= eps2, "row {i}: r2 {} > {eps2}", mc.s.r2);
                capped += usize::from(mc.s.r2 == eps2);
            }
        }
        assert!(capped > 0, "the cap was exercised");
    }

    #[test]
    fn state_round_trips_and_refuses_another_model() {
        let rows = blobs(700, 2);
        let mut m = Micro::new(cfg()).unwrap();
        run(&mut m, &rows);
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let back: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut r = Micro::restore(&back).unwrap();
        assert_eq!(r, m);
        let more = blobs(100, 4);
        assert_eq!(run(&mut r, &more), run(&mut m, &more));
        let other = crate::Holt::new(crate::HoltCfg {
            n_targets: 1,
            level_half_life: 10.0,
            trend_half_life: 40.0,
            min_weight: 0.0,
            trend: true,
        })
        .unwrap()
        .state();
        match Micro::restore(&other) {
            Err(StateError::WrongModel { expected, found }) => {
                assert_eq!((expected, found), ("micro", "holt"));
            }
            other => panic!("{other:?}"),
        }
    }

    /// A feature that moves and then holds, at a level of 1e8 (docs/PLAN.md
    /// task 102): every output is what the same stream gives at a level of
    /// 0, the distances to 1e-6. `kmeans` has the same test.
    #[test]
    fn a_stopped_feature_at_a_level_leaves_the_summaries_alone() {
        let run = |level: f64| -> Vec<Vec<f64>> {
            let mut m = Micro::new(MicroCfg {
                n_features: 3,
                decay: Decay::Halflife(20.0),
                eps: 0.4,
                standardize: true,
                scale_floor: 0.1,
                ..cfg()
            })
            .unwrap();
            let mut g = crate::SplitMix64::new(11);
            (0..1200)
                .map(|i| {
                    let c = if i % 2 == 0 { 0.0 } else { 10.0 };
                    let (u, v) = (g.uniform() - 0.5, g.uniform() - 0.5);
                    let third = if i < 300 {
                        level + g.uniform()
                    } else {
                        level + 0.37
                    };
                    crate::OnlineModel::step(&mut m, &[c + u, c + v, third], &[], 1.0, 1.0).pred
                })
                .collect()
        };
        let (base, high) = (run(0.0), run(1e8));
        assert!(
            base.iter().skip(300).any(|p| !p[0].is_nan()),
            "a summary is established"
        );
        for (i, (b, h)) in base.iter().zip(&high).enumerate() {
            for j in [0, 2, 3, 4, 5] {
                assert!(
                    b[j] == h[j] || (b[j].is_nan() && h[j].is_nan()),
                    "row {i}, slot {j}: {} at 1e8 against {} at 0",
                    h[j],
                    b[j]
                );
            }
            if !b[1].is_nan() {
                assert!(
                    (b[1] - h[1]).abs() <= 1e-6 * (1.0 + b[1]),
                    "row {i}: distance {} at 1e8 against {} at 0",
                    h[1],
                    b[1]
                );
            }
        }
    }

    /// `scale_floor` is refused by name in the core too, negative and NaN.
    #[test]
    fn a_bad_scale_floor_is_refused_by_name() {
        for bad in [-0.5, f64::NAN, f64::INFINITY] {
            let err = Micro::new(MicroCfg {
                scale_floor: bad,
                ..cfg()
            })
            .expect_err("refused");
            assert!(err.contains("scale_floor"), "{err}");
        }
    }

    /// The reference decays at `LONG_HALFLIVES` times the half-life: over a
    /// clock of 2 a row, the model's moments carry what a `FeatureMoments`
    /// given `decay.factor(2 / 8)` carries, to the bit.
    #[test]
    fn the_reference_decays_at_eight_halflives() {
        let decay = Decay::Halflife(20.0);
        let mut m = Micro::new(MicroCfg { decay, ..cfg() }).unwrap();
        let mut by_hand = FeatureMoments::new(2);
        let mut g = crate::SplitMix64::new(5);
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
        let mut m = Micro::new(MicroCfg {
            scale_floor: 0.1,
            ..cfg()
        })
        .unwrap();
        let mut g = crate::SplitMix64::new(2);
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
        let back: Micro = serde_json::from_value(old).unwrap();
        assert_eq!(back.cfg.scale_floor, 0.0);
        assert_eq!(back.moments(), m.moments());
    }

    #[test]
    fn rows_at_the_input_bound_leave_every_number_finite() {
        let mut m = Micro::new(MicroCfg {
            standardize: true,
            ..cfg()
        })
        .unwrap();
        let b = crate::INPUT_BOUND;
        for i in 0..200 {
            let x = if i % 7 == 0 { [b, -b] } else { [0.0, 1.0] };
            let s = m.step(&x, &[], 1.0, 1.0);
            assert!(s.n_eff.is_finite());
            for c in m.micro_clusters() {
                assert!(c.s.n.is_finite() && c.s.r2.is_finite());
                assert!(c.s.c.iter().all(|v| v.is_finite()), "{:?}", c.s.c);
            }
            assert!(m.metric().iter().all(|v| v.is_finite()));
        }
        assert!(m.n_clusters() >= 1);
    }

    #[test]
    fn coefficients_are_the_potential_summaries_and_absent_before_any() {
        let mut m = Micro::new(cfg()).unwrap();
        assert!(m.coefficients().is_none());
        m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        assert!(
            m.coefficients().is_none(),
            "an outlier summary is not reported"
        );
        for _ in 0..5 {
            m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        }
        let c = m.coefficients().unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 2 + 4);
        assert_eq!(&c[0][..2], &[0.0, 0.0], "id, label");
        assert!(c[0][2] > 5.9 && c[0][2] <= 6.0, "weight");
        assert_eq!(c[0][3], 0.0, "radius");
        assert_eq!(&c[0][4..], &[0.0, 0.0]);
    }

    #[test]
    fn config_is_validated() {
        let bad = [
            MicroCfg {
                n_features: 0,
                ..cfg()
            },
            MicroCfg { eps: 0.0, ..cfg() },
            MicroCfg {
                eps: f64::INFINITY,
                ..cfg()
            },
            MicroCfg {
                beta_mu: 0.0,
                ..cfg()
            },
            MicroCfg {
                max_clusters: 0,
                ..cfg()
            },
            // Clock units since task 163: a negative or NaN cadence is none.
            MicroCfg {
                prune_every: -1.0,
                ..cfg()
            },
            MicroCfg {
                prune_every: f64::NAN,
                ..cfg()
            },
            MicroCfg {
                macro_link: Some(-1.0),
                ..cfg()
            },
            MicroCfg {
                min_weight: -1.0,
                ..cfg()
            },
        ];
        for c in bad {
            assert!(Micro::new(c.clone()).is_err(), "{c:?}");
        }
        assert!(Micro::new(cfg()).is_ok());
        // 0 is every row by the clock, and a row cap of 0 or 1 every row.
        for (every, cap) in [(0.0, u32::MAX), (f64::INFINITY, 0), (2.5, 1)] {
            let c = MicroCfg {
                prune_every: every,
                max_rows_between_prunes: cap,
                ..cfg()
            };
            assert!(Micro::new(c).is_ok(), "{every}, {cap}");
        }
    }

    #[test]
    fn lam_decay_gives_the_same_prune_horizon_as_its_halflife() {
        let h = Micro::new(MicroCfg {
            decay: Decay::Halflife(100.0),
            ..cfg()
        })
        .unwrap();
        let l = Micro::new(MicroCfg {
            decay: Decay::Lam(0.5f64.powf(1.0 / 100.0)),
            ..cfg()
        })
        .unwrap();
        let (tp_h, f_h) = h.prune_horizon().unwrap();
        let (tp_l, f_l) = l.prune_horizon().unwrap();
        assert_eq!(tp_h, 59.0);
        assert_eq!(tp_h, tp_l);
        assert!((f_h - f_l).abs() < 1e-12);
        assert!(
            Micro::new(MicroCfg {
                beta_mu: 1.0,
                ..cfg()
            })
            .unwrap()
            .prune_horizon()
            .is_none()
        );
    }

    /// No decay, no checkpoint, no linkage unless a test asks for it: the
    /// summaries are exactly what the rows make them.
    fn still() -> MicroCfg {
        MicroCfg {
            decay: Decay::Halflife(f64::INFINITY),
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 1000,
            macro_link: Some(0.0),
            ..cfg()
        }
    }

    /// `rows` rows of weight 1 at `x`.
    fn feed(m: &mut Micro, x: [f64; 2], rows: usize) {
        for _ in 0..rows {
            m.step(&x, &[], 1.0, 1.0);
        }
    }

    /// A summary placed by hand, of weight 5 and no radius.
    fn placed(id: u64, c: Vec<f64>, potential: bool) -> MicroCluster {
        MicroCluster {
            id,
            s: ClusterSummary::at(c, 5.0, 0.0),
            age: 0.0,
            potential,
            label: id,
        }
    }

    /// A state written before the stream's mean weight (task 147) reads as
    /// a mean weight of 1: its summaries admit a unit row, so a row at a
    /// summary's centre is taken and one far from it opens a new one.
    #[test]
    fn a_state_without_the_mean_weight_admits_a_unit_row() {
        let mut m = Micro::new(cfg()).unwrap();
        feed(&mut m, [0.0, 0.0], 5);
        let mut old = serde_json::to_value(&m).unwrap();
        let fields = old.as_object_mut().unwrap();
        assert!(fields.remove("w_mean").is_some() && fields.remove("w_rows").is_some());
        let back: Micro = serde_json::from_value(old).unwrap();
        assert_eq!((back.w_mean, back.w_rows), (0.0, 0.0), "the fixture");
        let far = back.predict(&[5.0, 0.0], 1.0).pred;
        assert_eq!(far[2], 1.0, "a far row opens id 1: {far:?}");
        assert_eq!(far[3], 1.0);
        let near = back.predict(&[0.0, 0.0], 1.0).pred;
        assert_eq!(near[2], 0.0, "{near:?}");
    }

    /// The metric is all ones in raw units, and `1 / v_i` standardized,
    /// `v_i` each feature's variance over the rows (no decay, unit weights:
    /// the plain population variance).
    #[test]
    fn the_metric_is_one_over_each_features_variance_when_standardizing() {
        let mut plain = Micro::new(still()).unwrap();
        let mut scaled = Micro::new(MicroCfg {
            standardize: true,
            ..still()
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

    /// The nearest summary is decided on the squares, first minimum
    /// winning: between summaries equally far the first is the nearest,
    /// and between two whose squares differ by a rounding step the smaller
    /// square is, though the distances round to one number.
    #[test]
    fn the_nearest_summary_is_decided_on_the_squares() {
        // Equally far: (1, 0) and (-1, 0) from the origin.
        let mut m = Micro::new(MicroCfg {
            eps: 0.25,
            ..still()
        })
        .unwrap();
        feed(&mut m, [1.0, 0.0], 3);
        feed(&mut m, [-1.0, 0.0], 3);
        assert!(m.micro_clusters().iter().all(|c| c.potential) && m.n_clusters() == 2);
        let tie = m.predict(&[0.0, 0.0], 1.0).pred;
        assert_eq!(tie[0], 0.0, "the first of two equally far: {tie:?}");
        // A rounding step apart: 1 + 2^-52 against 1.
        let step = 2f64.powi(-26);
        let mut m = Micro::new(MicroCfg {
            eps: 0.25,
            ..still()
        })
        .unwrap();
        feed(&mut m, [1.0, step], 3);
        feed(&mut m, [-1.0, 0.0], 3);
        let a = dist2(&m.micro_clusters()[0].s.c, &[0.0, 0.0], m.metric());
        let b = dist2(&m.micro_clusters()[1].s.c, &[0.0, 0.0], m.metric());
        assert!(a > b && a.sqrt() == b.sqrt(), "the fixture: {a} and {b}");
        let near = m.predict(&[0.0, 0.0], 1.0).pred;
        assert_eq!(near[0], 1.0, "the smaller square: {near:?}");
    }

    /// Among squares that overflow, the overflow-free norms decide, first
    /// minimum winning; where one square is finite, the squares do, even
    /// against a square that overflowed by a rounding step and whose norm
    /// rounds to the finite one's.
    #[test]
    fn overflowed_squares_are_decided_by_their_norms() {
        let mut m = Micro::new(cfg()).unwrap();
        m.mc = vec![
            placed(0, vec![0.0, 3e200], true),
            placed(1, vec![1e200, 0.0], true),
            placed(2, vec![-1e200, 0.0], true),
            placed(3, vec![0.0, -2e200], true),
        ];
        (m.next_id, m.n_clusters) = (4, 4);
        let out = m.predict(&[0.0, 0.0], 1.0).pred;
        assert_eq!(out[0], 1.0, "the first of the two nearest: {out:?}");
        assert_eq!(out[1], 1e200);

        // The smallest double whose square overflows, before one whose
        // square is just finite and whose norm rounds to the same number
        // (found by search in IEEE doubles).
        let over = 1.3407807929942597e154;
        let under = [1.3262349803990826e154, 1.9696170091710862e153];
        let origin = [0.0, 0.0];
        let (d_over, d_under) = (
            dist2(&[over, 0.0], &origin, &[1.0, 1.0]),
            dist2(&under, &origin, &[1.0, 1.0]),
        );
        assert!(
            d_over.is_infinite()
                && d_under.is_finite()
                && dist(&under, &origin, &[1.0, 1.0]) == dist(&[over, 0.0], &origin, &[1.0, 1.0]),
            "the fixture"
        );
        m.mc = vec![
            placed(0, vec![over, 0.0], true),
            placed(1, under.to_vec(), true),
        ];
        (m.next_id, m.n_clusters) = (2, 2);
        let out = m.predict(&origin, 1.0).pred;
        assert_eq!(out[0], 1.0, "the finite square: {out:?}");
    }

    /// At the cap the lightest outlier summary goes, the first of two
    /// equally light: outliers of weights 2, 1 and 1 lose the first 1.
    #[test]
    fn the_cap_evicts_the_first_of_the_lightest_outliers() {
        let mut m = Micro::new(MicroCfg {
            max_clusters: 4,
            ..still()
        })
        .unwrap();
        feed(&mut m, [0.0, 0.0], 3); // id 0, potential
        feed(&mut m, [10.0, 0.0], 2); // id 1, weight 2
        feed(&mut m, [20.0, 0.0], 1); // id 2, weight 1
        feed(&mut m, [30.0, 0.0], 1); // id 3, weight 1
        feed(&mut m, [40.0, 0.0], 1); // id 4, evicting id 2
        let ids: Vec<u64> = m.micro_clusters().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![0, 1, 3, 4]);
    }

    /// A summary promoted takes the label of the nearest other potential
    /// summary within the linkage threshold -- the nearer of two, wherever
    /// it sits in creation order, and the first of two equally near -- and
    /// with none in reach, its own.
    #[test]
    fn a_promoted_summary_takes_the_label_of_the_nearest_within_reach() {
        // eps 0.25 (bound 0.125) keeps the three summaries apart; a
        // threshold of 4 eps √p is a squared distance of 2, so the outer
        // two (2 or 1.9 apart) are separate clusters and the middle one
        // reaches both.
        let linked = MicroCfg {
            eps: 0.25,
            macro_link: Some(4.0),
            ..still()
        };
        for (b, want) in [(0.9, 1u64), (1.0, 0)] {
            let mut m = Micro::new(linked.clone()).unwrap();
            assert_eq!(m.link2, 2.0, "the fixture");
            feed(&mut m, [-1.0, 0.0], 3);
            feed(&mut m, [b, 0.0], 3);
            assert_eq!(m.n_clusters(), 2, "b {b}: the outer two are apart");
            feed(&mut m, [0.0, 0.0], 3);
            let mc = m.micro_clusters();
            assert_eq!(mc.len(), 3, "b {b}");
            assert!(mc[2].potential, "b {b}");
            assert_eq!(mc[2].label, want, "b {b}: {mc:?}");
            assert_eq!(m.n_clusters(), 2, "b {b}: it joined a cluster");
        }
        // Out of reach: its own label.
        let mut m = Micro::new(MicroCfg {
            eps: 0.25,
            macro_link: Some(1.0),
            ..still()
        })
        .unwrap();
        feed(&mut m, [-1.0, 0.0], 3);
        feed(&mut m, [0.0, 0.0], 3);
        assert_eq!(m.micro_clusters()[1].label, 1);
        assert_eq!(m.n_clusters(), 2);
    }

    /// `macro_link = 0` links nothing, not even two potential summaries at
    /// one centre: when the second is promoted, and at a checkpoint.
    #[test]
    fn a_threshold_of_zero_links_nothing_even_at_one_centre() {
        let mut m = Micro::new(still()).unwrap();
        assert_eq!(m.link2, 0.0);
        m.mc = vec![
            placed(0, vec![1.0, 2.0], true),
            placed(1, vec![1.0, 2.0], false),
        ];
        (m.next_id, m.n_clusters) = (2, 1);
        m.mc[1].potential = true;
        m.attach(1);
        assert_eq!(m.mc[1].label, 1);
        assert_eq!(m.n_clusters(), 2);
        m.link_potential();
        assert_eq!(m.n_clusters(), 2);
        assert_eq!((m.mc[0].label, m.mc[1].label), (0, 1));
    }

    /// The checkpoint keeps a potential summary of exactly `beta_mu` rows,
    /// and an outlier one of exactly `ξ(age)` rows: DenStream drops one
    /// lighter than the bound, not one at it. A summary that has just taken
    /// its first row is at `ξ(0) = 1`.
    #[test]
    fn the_checkpoint_keeps_a_summary_at_its_bound() {
        let mut m = Micro::new(MicroCfg {
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 3,
            ..still()
        })
        .unwrap();
        feed(&mut m, [0.0, 0.0], 3);
        let mc = m.micro_clusters();
        assert!(
            mc.len() == 1 && mc[0].potential && mc[0].s.n == 3.0,
            "{mc:?}"
        );
        assert_eq!(m.events().1, 0, "nothing pruned");

        let mut m = Micro::new(MicroCfg {
            prune_every: f64::INFINITY,
            max_rows_between_prunes: 1,
            decay: Decay::Halflife(100.0),
            ..still()
        })
        .unwrap();
        assert!(m.prune_horizon().is_some(), "outliers are pruned here");
        feed(&mut m, [0.0, 0.0], 1);
        assert_eq!(m.micro_clusters().len(), 1, "{:?}", m.micro_clusters());
        assert_eq!(m.events().1, 0);
    }

    /// The linkage is single linkage over the potential summaries, written
    /// here from its definition with a flood fill: two potential summaries
    /// are linked when their squared distance is within the threshold, a
    /// cluster is a chain of links, and its label is the smallest id in it;
    /// the derived threshold is `LINK_FACTOR` times the 90th percentile
    /// (nearest rank) of the nearest-neighbour distances, floored at
    /// `LINK_FLOOR · eps √p`. Outlier summaries keep their labels. Held over
    /// random placements, each threshold rule, so that a rewrite of the step
    /// -- its memory taken from `m²` to `m` (review 2026-10-06, CF2) -- gives
    /// the same labels, counts and threshold.
    #[test]
    fn the_linkage_is_single_linkage_over_the_potential_summaries() {
        let mut s = 20261006u64;
        for trial in 0..60 {
            let n = 1 + trial % 37;
            let macro_link = match trial % 3 {
                0 => None,
                1 => Some(1.5 + lcg(&mut s) * 4.0),
                _ => Some(0.0),
            };
            let mut m = Micro::new(MicroCfg {
                macro_link,
                ..still()
            })
            .unwrap();
            let spread = 2.0 + 10.0 * lcg(&mut s);
            m.mc = (0..n)
                .map(|i| {
                    let c = vec![spread * lcg(&mut s), spread * lcg(&mut s)];
                    placed(i as u64 * 3 + 7, c, lcg(&mut s) < 0.7)
                })
                .collect();
            m.next_id = n as u64 * 3 + 7;
            let before: Vec<u64> = m.mc.iter().map(|c| c.label).collect();
            m.link_potential();

            // The definition, longhand.
            let pot: Vec<usize> = (0..n).filter(|&j| m.mc[j].potential).collect();
            let d2 = |a: usize, b: usize| {
                let (ca, cb) = (&m.mc[pot[a]].s.c, &m.mc[pot[b]].s.c);
                let (dx, dy) = (cb[0] - ca[0], cb[1] - ca[1]);
                dx * dx + dy * dy
            };
            let link2 = match macro_link {
                // `l` in units of `eps √p`, squared.
                Some(l) => l * l * m.eps2,
                None if pot.len() >= 2 => {
                    let mut nn: Vec<f64> = (0..pot.len())
                        .map(|a| {
                            (0..pot.len())
                                .filter(|&b| b != a)
                                .map(|b| if a < b { d2(a, b) } else { d2(b, a) })
                                .fold(f64::INFINITY, f64::min)
                        })
                        .collect();
                    nn.sort_by(f64::total_cmp);
                    let rank = (9 * pot.len()).div_ceil(10);
                    let l = (LINK_FACTOR * nn[rank - 1].sqrt()).max(LINK_FLOOR * m.eps2.sqrt());
                    l * l
                }
                // Fewer than two: the threshold the model started with.
                None => m.link2,
            };
            assert_eq!(
                m.link2.to_bits(),
                link2.to_bits(),
                "trial {trial}: threshold"
            );
            let mut component = vec![usize::MAX; pot.len()];
            let mut count = 0;
            for start in 0..pot.len() {
                if component[start] != usize::MAX {
                    continue;
                }
                component[start] = count;
                let mut stack = vec![start];
                while let Some(a) = stack.pop() {
                    let near: Vec<usize> = (0..pot.len())
                        .filter(|&b| {
                            b != a
                                && link2 > 0.0
                                && (if a < b { d2(a, b) } else { d2(b, a) }) <= link2
                        })
                        .collect();
                    for b in near {
                        if component[b] == usize::MAX {
                            component[b] = count;
                            stack.push(b);
                        }
                    }
                }
                count += 1;
            }
            assert_eq!(m.n_clusters, count, "trial {trial}: clusters");
            for (a, &j) in pot.iter().enumerate() {
                let smallest = pot
                    .iter()
                    .enumerate()
                    .filter(|&(b, _)| component[b] == component[a])
                    .map(|(_, &k)| m.mc[k].id)
                    .min()
                    .unwrap();
                assert_eq!(m.mc[j].label, smallest, "trial {trial}: summary {j}");
            }
            for j in (0..n).filter(|&j| !m.mc[j].potential) {
                assert_eq!(m.mc[j].label, before[j], "trial {trial}: outlier {j}");
            }
        }
    }

    /// The two paths of the linkage give the same threshold, the same
    /// clusters and the same labels, to the bit, on the same summaries: the
    /// matrix kept up to `LINK_MATRIX_MAX` potential summaries, and the
    /// pairs taken when wanted past it (review 2026-10-06, CF2). Random
    /// placements, outliers among them, under each threshold rule; and the
    /// rule `link_potential` takes by the count.
    #[test]
    fn the_matrix_and_the_pairs_link_alike() {
        let mut s = 20261007u64;
        let mut linked = 0;
        for trial in 0..90 {
            let n = 2 + trial % 53;
            let macro_link = match trial % 3 {
                0 => None,
                1 => Some(1.5 + lcg(&mut s) * 4.0),
                _ => Some(0.0),
            };
            let mut m = Micro::new(MicroCfg {
                macro_link,
                ..still()
            })
            .unwrap();
            let spread = 2.0 + 10.0 * lcg(&mut s);
            m.mc = (0..n)
                .map(|i| {
                    let c = vec![spread * lcg(&mut s), spread * lcg(&mut s)];
                    placed(i as u64 * 3 + 7, c, lcg(&mut s) < 0.8)
                })
                .collect();
            m.next_id = n as u64 * 3 + 7;
            let idx: Vec<usize> = (0..n).filter(|&j| m.mc[j].potential).collect();
            let (mut matrix, mut pairs, mut either) = (m.clone(), m.clone(), m.clone());
            matrix.link_by_matrix(&idx);
            pairs.link_by_pairs(&idx);
            either.link_potential();
            let labels = |m: &Micro| m.mc.iter().map(|c| c.label).collect::<Vec<_>>();
            for (path, got) in [("pairs", &pairs), ("link_potential", &either)] {
                assert_eq!(
                    got.link2.to_bits(),
                    matrix.link2.to_bits(),
                    "trial {trial}, {path}: threshold"
                );
                assert_eq!(
                    got.n_clusters, matrix.n_clusters,
                    "trial {trial}, {path}: clusters"
                );
                assert_eq!(
                    labels(got),
                    labels(&matrix),
                    "trial {trial}, {path}: labels"
                );
            }
            linked += usize::from(matrix.n_clusters < idx.len());
        }
        assert!(linked >= 20, "too few trials linked anything: {linked}");
        assert_eq!(LINK_MATRIX_MAX * LINK_MATRIX_MAX * 8, 128 << 20, "128 MiB");
    }

    /// The derived threshold: `LINK_FACTOR` times the 90th percentile
    /// (nearest rank) of the potential summaries' nearest-neighbour
    /// distances, and never below `LINK_FLOOR · eps √p`.
    #[test]
    fn the_derived_threshold_is_the_spacings_90th_percentile_or_the_floor() {
        let mut m = Micro::new(MicroCfg {
            macro_link: None,
            ..still()
        })
        .unwrap();
        // Gaps of 1 to 9 along a line: nearest-neighbour distances 1, 1, 2,
        // ..., 9.
        let xs = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0];
        m.mc = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| placed(i as u64, vec![x, 0.0], true))
            .collect();
        m.link_potential();
        let mut nn: Vec<f64> = (0..xs.len())
            .map(|a| {
                (0..xs.len())
                    .filter(|&b| b != a)
                    .map(|b| (xs[a] - xs[b]).abs())
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        nn.sort_by(f64::total_cmp);
        let rank = (9 * xs.len()).div_ceil(10);
        let want = LINK_FACTOR * nn[rank - 1];
        assert_eq!(want, 12.0, "the fixture");
        assert!(
            (m.link() - want).abs() < 1e-12,
            "{} against {want}",
            m.link()
        );

        // Spacing far below the floor.
        m.mc = (0..3)
            .map(|i| placed(i, vec![0.1 * i as f64, 0.0], true))
            .collect();
        m.link_potential();
        let floor = LINK_FLOOR * 0.3 * 2f64.sqrt();
        assert!(
            (m.link() - floor).abs() < 1e-12,
            "{} against {floor}",
            m.link()
        );
    }
}
