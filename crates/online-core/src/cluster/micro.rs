//! DenStream-style micro-clusters with a linkage macro step
//! (docs/CLUSTERING.md §6.5; docs/PLAN.md §11a, task 24): a bounded set of
//! mean-form summaries, each absorbing a row only while its radius stays
//! within `eps`, a new one opened where no summary can take the row, and a
//! checkpoint every `prune_every` learned rows that prunes the faded ones
//! and links the potential ones into clusters by single linkage.
//!
//! ```text
//! metric      mw_i = 1 / v_i  (EW variance of feature i; 1 where v_i = 0)
//! bound       E = eps² p                 (eps per standardized coordinate)
//! absorb      j_p = nearest potential;  take it if r2_after(j_p, z, 1) ≤ E
//!             else j_o = nearest outlier;  take it if r2_after(j_o, z, 1) ≤ E,
//!             promoting it once n_j_o ≥ beta_mu (it takes the label of the
//!             nearest potential summary within L, else its own id)
//!             else open a summary at z with weight w and the next id
//!             (evicting the lightest outlier summary — else the lightest
//!             potential one — when max_clusters are live)
//!             the summary taken absorbs z with weight w, then r2 ← min(r2, E)
//!             r2_after(j, z, w) = a r2_j + a b ‖z − c_j‖²_mw,  a = n_j/(n_j + w),  b = w/(n_j + w)
//! outputs     cluster = label(j_p),  dist = ‖z − c_j_p‖_mw,
//!             micro = the id the row goes to,  outlier = not taken by a
//!             potential summary,  n_clusters, n_micro
//! checkpoint  every prune_every rows: drop a potential summary with
//!             n < beta_mu, and an outlier one with n < ξ(age)
//!             ξ(a) = (2^(−(a + Tp)/h) − 1) / (2^(−Tp/h) − 1),
//!             Tp = ⌈h log2(beta_mu / (beta_mu − 1))⌉      (DenStream eq. 4.1–4.2)
//!             then link potential summaries with ‖c_a − c_b‖_mw ≤ L,
//!             label = the smallest id in the component
//!             L = macro_link · eps √p, or, unset, derived from the spacing:
//!             max(LINK_FLOOR, LINK_FACTOR · p90 of the nearest-neighbour distance) · eps √p
//! decay       n_j *= lam, W *= lam, age_j += d_clock       lam = 0.5^(d/halflife)
//! ```
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
//! potential summary is its own cluster.
//!
//! Three rules make a variable-count output honest (§6.5): ids are
//! monotone and never reused — an evicted or pruned id never comes back;
//! a row's `micro` is the id it *would* be absorbed by, read before the
//! update, so the first row of a new summary already carries the new id;
//! and the count of live clusters is an output, so churn is visible
//! without diffing labels. The decision is made for a unit-weight row
//! whatever the row's weight: a row of weight `w` stands for `w` identical
//! rows, the first of which is a unit row, and this is what lets
//! [`predict`](OnlineModel::predict) — which is never told a weight — say
//! exactly what the step will do. The row is then absorbed with its full
//! weight, and since `w > 1` can carry the radius past the bound, the
//! radius is capped at the bound after the absorb. Left above it, the
//! summary would admit nothing — not even a row at its centre — until
//! decay brought its weight under `E / (r2 − E)`, halflives later; capped,
//! it is merely full, and the cap costs one row's worth of spread that the
//! next rows re-estimate. (DenStream has no row weights; this is the
//! extension.) So `r2 ≤ E` holds for every summary at all times, and the
//! radius [`coefficients`](Micro::coefficients) reports is at most
//! `eps √p`.
//!
//! Every output is read *before* the row is learned (CLAUDE.md rule 2),
//! `n_eff` is the EW weight before the row and before its own decay (rule
//! 8), and pruning runs on a learned-row schedule rather than a clock one
//! so that a chunking of the stream cannot move it. Standardization scales
//! the metric, never the coordinates (§10), so the centres stay in the
//! features' own units; a summary's radius is in the metric that was in
//! force when its rows were absorbed.

use serde::{Deserialize, Serialize};

use super::summary::{ClusterSummary, FeatureMoments, dist2, merged_radius2};
use crate::clock::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// Configuration for [`Micro`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroCfg {
    pub n_features: usize,
    pub decay: Decay,
    /// Outputs are null while `n_eff < min_periods`.
    pub min_periods: f64,
    /// Bound on a summary's RMS radius per standardized coordinate, `> 0`.
    pub eps: f64,
    /// Weight at which an outlier summary becomes potential, `> 0`.
    pub beta_mu: f64,
    /// Live summaries at most, `>= 1`.
    pub max_clusters: usize,
    /// Learned rows between checkpoints, `>= 1`.
    pub prune_every: u32,
    /// Linkage threshold in units of `eps √p`; `None` derives it from the
    /// observed spacing at each checkpoint, `0` links nothing.
    pub macro_link: Option<f64>,
    /// Measure distances in units of each feature's EW standard deviation.
    pub standardize: bool,
}

impl MicroCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("micro: n_features must be >= 1".into());
        }
        if self.min_periods.is_nan() || self.min_periods < 0.0 {
            return Err("micro: min_periods must be >= 0".into());
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
        if self.prune_every == 0 {
            return Err("micro: prune_every must be >= 1".into());
        }
        if let Some(l) = self.macro_link {
            if !l.is_finite() || l < 0.0 {
                return Err("micro: macro_link must be finite and >= 0".into());
            }
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
    since: u32,
    /// Distinct labels among the potential summaries.
    n_clusters: usize,
    /// The linkage threshold² in force, from the last checkpoint.
    link2: f64,
    n_evicted: u64,
    n_pruned: u64,
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
            since: 0,
            n_clusters: 0,
            link2,
            n_evicted: 0,
            n_pruned: 0,
            cfg,
        })
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

    /// The nearest summary of the given kind and its squared distance
    /// (first minimum wins).
    fn nearest(&self, z: &[f64], potential: bool) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (j, m) in self.mc.iter().enumerate() {
            if m.potential != potential {
                continue;
            }
            let d = dist2(&m.s.c, z, &self.mw);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        best
    }

    /// Whether summary `j`, its weight decayed by `lam`, admits a unit row
    /// at squared distance `d2`.
    fn admits(&self, j: usize, d2: f64, lam: f64) -> bool {
        let s = &self.mc[j].s;
        merged_radius2(s.n * lam, s.r2, d2, 1.0) <= self.eps2
    }

    /// Where a unit-weight row at `z` goes, and what it is scored against,
    /// once every summary's weight has decayed by `lam` — `1` in
    /// [`step`](OnlineModel::step), which has already applied the clock,
    /// and the row's own factor in [`predict`](OnlineModel::predict).
    fn decide(&self, z: &[f64], lam: f64) -> Decision {
        let nearest_potential = self.nearest(z, true);
        if let Some((jp, d2)) = nearest_potential {
            if self.admits(jp, d2, lam) {
                return Decision {
                    target: Some(jp),
                    nearest_potential,
                    outlier: false,
                };
            }
        }
        if let Some((jo, d2)) = self.nearest(z, false) {
            if self.admits(jo, d2, lam) {
                return Decision {
                    target: Some(jo),
                    nearest_potential,
                    outlier: true,
                };
            }
        }
        Decision {
            target: None,
            nearest_potential,
            outlier: true,
        }
    }

    /// The six outputs: `cluster`, `dist`, `micro`, `outlier`,
    /// `n_clusters`, `n_micro`; all NaN when not ready.
    fn score(&self, dec: Option<&Decision>, n_eff: f64) -> Vec<f64> {
        let mut pred = vec![f64::NAN; 6];
        if let Some(dec) = dec.filter(|_| n_eff >= self.cfg.min_periods) {
            if let Some((jp, d2)) = dec.nearest_potential {
                pred[0] = self.mc[jp].label as f64;
                pred[1] = d2.sqrt();
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
        match dec.target {
            Some(j) => {
                let m = &mut self.mc[j];
                m.s.absorb(z, w, &self.mw);
                // A unit row was admitted; a heavier one may overshoot.
                m.s.r2 = m.s.r2.min(self.eps2);
                if !m.potential && m.s.n >= self.cfg.beta_mu {
                    m.potential = true;
                    self.attach(j);
                }
            }
            None => self.create(z, w),
        }
        self.since += 1;
        if self.since >= self.cfg.prune_every {
            self.since = 0;
            self.checkpoint();
        }
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
        let potential = w >= self.cfg.beta_mu;
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

    /// The decay's halflife in clock units (infinite for none).
    fn halflife(&self) -> f64 {
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
        let h = self.halflife();
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
        for j in (0..self.mc.len()).rev() {
            let m = &self.mc[j];
            let dead = if m.potential {
                m.s.n < self.cfg.beta_mu
            } else if let Some((_, f_tp)) = horizon {
                let age_decay = self.cfg.decay.factor(m.age);
                let xi = (age_decay * f_tp - 1.0) / (f_tp - 1.0);
                m.s.n < xi
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
    /// id in each component.
    fn link_potential(&mut self) {
        let idx: Vec<usize> = (0..self.mc.len())
            .filter(|&j| self.mc[j].potential)
            .collect();
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
            let mut nn: Vec<f64> = (0..m)
                .map(|a| {
                    (0..m)
                        .filter(|&b| b != a)
                        .map(|b| d2[a * m + b])
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            nn.sort_by(f64::total_cmp);
            let rank = ((LINK_QUANTILE * m as f64).ceil() as usize).clamp(1, m) - 1;
            let p90 = nn[rank].sqrt();
            let floor = LINK_FLOOR * self.eps2.sqrt();
            let l = (LINK_FACTOR * p90).max(floor);
            self.link2 = l * l;
        }
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        if self.link2 > 0.0 {
            for a in 0..m {
                for b in (a + 1)..m {
                    if d2[a * m + b] <= self.link2 {
                        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                        if ra != rb {
                            // Ids ascend with the index, so the smaller
                            // index is the smaller id.
                            parent[ra.max(rb)] = ra.min(rb);
                        }
                    }
                }
            }
        }
        let mut n_clusters = 0;
        for a in 0..m {
            let root = find(&mut parent, a);
            if root == a {
                n_clusters += 1;
            }
            self.mc[idx[a]].label = self.mc[idx[root]].id;
        }
        self.n_clusters = n_clusters;
    }
}

impl OnlineModel for Micro {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let lam = self.cfg.decay.factor(d_clock);
        let n_before = self.moments.w;
        let valid = x.iter().all(|v| v.is_finite());
        let learn = weight > 0.0 && weight.is_finite() && valid;

        // The clock passes for everything the model holds.
        self.moments.decay(lam);
        for m in &mut self.mc {
            m.s.decay(lam);
            m.age += d_clock;
        }

        // Decided once, as a unit row, and read before anything moves.
        let dec = valid.then(|| self.decide(x, 1.0));
        let pred = self.score(dec.as_ref(), n_before);

        if let Some(dec) = dec.filter(|_| learn) {
            self.moments.absorb(x, weight);
            self.learn_row(x, weight, &dec);
        }
        self.moments.metric(self.cfg.standardize, &mut self.mw);

        Step {
            pred,
            n_eff: n_before,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
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
            ModelState::Micro(m) => Ok((**m).clone()),
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

    /// `cluster`, `dist`, `micro`, `outlier`, `n_clusters`, `n_micro`.
    fn n_outputs(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            min_periods: 0.0,
            eps: 0.3,
            beta_mu: 3.0,
            max_clusters: 50,
            prune_every: 50,
            macro_link: None,
            standardize: false,
        }
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
        // Weight 3 (three rows at halflife 500 decay a little: 2.997) — not
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
        let mut m = Micro::new(cfg()).unwrap();
        m.step(&[0.0, 0.0], &[], 1.0, 5.0);
        assert!(m.micro_clusters()[0].potential);
        assert_eq!(m.n_clusters(), 1);
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
            prune_every: 10,
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
            prune_every: 5,
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
            prune_every: 10,
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
            prune_every: 1000,
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
            prune_every: 1000,
            ..cfg()
        })
        .unwrap();
        all_pot.step(&[0.0, 0.0], &[], 1.0, 4.0);
        all_pot.step(&[10.0, 0.0], &[], 1.0, 5.0);
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
            min_periods: 3.0,
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
            level_halflife: 10.0,
            trend_halflife: 40.0,
            min_periods: 0.0,
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
            MicroCfg {
                prune_every: 0,
                ..cfg()
            },
            MicroCfg {
                macro_link: Some(-1.0),
                ..cfg()
            },
            MicroCfg {
                min_periods: -1.0,
                ..cfg()
            },
        ];
        for c in bad {
            assert!(Micro::new(c.clone()).is_err(), "{c:?}");
        }
        assert!(Micro::new(cfg()).is_ok());
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
}
