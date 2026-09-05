//! The accumulator every clustering model here is built on (docs/CLUSTERING.md
//! §6.1): a cluster is a weight, a centre and a radius², all in **mean form**,
//! so no stored number ever exceeds the largest input and a zero-weight update
//! is a guarded no-op (CLAUDE.md rule 9). The metric is diagonal and read from
//! the EW feature moments *before* each row (E24's rule, §10: standardize the
//! metric, never the coordinates), and the generator the seeding rules draw
//! from is written out here so that a Python reference can be bit-exact.

use serde::{Deserialize, Serialize};

/// `Σ mw_i (z_i − c_i)²`: squared distance under a diagonal metric.
///
/// Never NaN for finite inputs and finite weights (every term is `≥ 0`, so
/// the sum is at most `+∞`), which is what lets a row at the input bound
/// against a variance at the opposite scale compare as "far" instead of
/// poisoning an argmin.
#[inline]
pub fn dist2(c: &[f64], z: &[f64], mw: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..c.len() {
        let t = z[i] - c[i];
        acc += mw[i] * t * t;
    }
    acc
}

/// The radius² a summary of weight `n` and radius² `r2` would have after
/// absorbing weight `w` at squared distance `q` from its centre — the merged
/// radius DenStream's absorption test reads, and exactly what
/// [`ClusterSummary::absorb`] then stores. `r2` unchanged when the merged
/// weight is not positive or `q` is not finite, as `absorb` leaves it.
pub fn merged_radius2(n: f64, r2: f64, q: f64, w: f64) -> f64 {
    let n_new = n + w;
    if n_new <= 0.0 || !q.is_finite() {
        return r2;
    }
    let (a, b) = (n / n_new, w / n_new);
    a * r2 + a * b * q
}

/// One cluster: weight `n`, centre `c`, radius² `r2`.
///
/// `r2` is whatever the model defines it as — `kmeans` keeps the EW mean of
/// each assigned row's squared distance to the centre *at assignment*
/// ([`ClusterSummary::absorb_plain`] / [`ClusterSummary::merge_plain`]),
/// `micro` keeps Welford's centred radius² ([`ClusterSummary::absorb`] /
/// [`ClusterSummary::merge_welford`]). The two forms share the weight and
/// centre arithmetic, which is the mean-form recursion of `ewcov.rs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterSummary {
    pub n: f64,
    pub c: Vec<f64>,
    pub r2: f64,
}

impl ClusterSummary {
    /// Empty: no weight, the origin, no radius.
    pub fn empty(p: usize) -> Self {
        Self {
            n: 0.0,
            c: vec![0.0; p],
            r2: 0.0,
        }
    }

    /// A summary placed at `c` with the given weight and radius².
    pub fn at(c: Vec<f64>, n: f64, r2: f64) -> Self {
        Self { n, c, r2 }
    }

    /// `n *= lam`: the clock passes. A mean does not decay.
    #[inline]
    pub fn decay(&mut self, lam: f64) {
        self.n *= lam;
    }

    /// Fold in one row of weight `w` whose squared distance to the centre
    /// was `d2`, keeping `r2` as the weighted mean of those distances:
    ///
    /// ```text
    /// n' = n + w,  b = w / n'
    /// c' = c + b (z − c)
    /// r2' = r2 + b (d2 − r2)        (skipped when d2 is not finite)
    /// ```
    ///
    /// Guarded: nothing changes when `n' <= 0`. An infinite `d2` (a row the
    /// metric cannot measure, see [`dist2`]) moves the centre but is not
    /// learned into the radius, which would otherwise hold `∞` for good.
    pub fn absorb_plain(&mut self, z: &[f64], w: f64, d2: f64) {
        let n_new = self.n + w;
        if n_new <= 0.0 {
            return;
        }
        let b = w / n_new;
        for (ci, zi) in self.c.iter_mut().zip(z) {
            *ci += b * (zi - *ci);
        }
        if d2.is_finite() {
            self.r2 += b * (d2 - self.r2);
        }
        self.n = n_new;
    }

    /// Merge a batch summary built by [`absorb_plain`](Self::absorb_plain)
    /// into this one, both means:
    ///
    /// ```text
    /// n' = n + m,  b = m / n'
    /// c' = c + b (c_o − c),   r2' = r2 + b (r2_o − r2)
    /// ```
    ///
    /// Guarded as `absorb_plain`. With a one-row batch this *is*
    /// `absorb_plain`, bit for bit, which is what makes `update_every = 1`
    /// the per-row model without a second code path.
    pub fn merge_plain(&mut self, other: &ClusterSummary) {
        let n_new = self.n + other.n;
        if n_new <= 0.0 {
            return;
        }
        let b = other.n / n_new;
        for (ci, oi) in self.c.iter_mut().zip(&other.c) {
            *ci += b * (oi - *ci);
        }
        self.r2 += b * (other.r2 - self.r2);
        self.n = n_new;
    }

    /// Welford absorption of one row of weight `w` under the metric `mw`:
    ///
    /// ```text
    /// n' = n + w,  a = n / n',  b = w / n'
    /// c' = c + b δ,   r2' = a r2 + a b ‖δ‖²_mw        δ = z − c
    /// ```
    ///
    /// `r2` is then the EW mean squared deviation about the centre —
    /// DenStream's radius² with the fading function being the decay. Guarded
    /// when `n' <= 0`; an infinite `‖δ‖²` leaves `r2` alone, as
    /// [`absorb_plain`](Self::absorb_plain) does.
    pub fn absorb(&mut self, z: &[f64], w: f64, mw: &[f64]) {
        let n_new = self.n + w;
        if n_new <= 0.0 {
            return;
        }
        let (a, b) = (self.n / n_new, w / n_new);
        let q = dist2(&self.c, z, mw);
        for (ci, zi) in self.c.iter_mut().zip(z) {
            *ci += b * (zi - *ci);
        }
        if q.is_finite() {
            self.r2 = a * self.r2 + a * b * q;
        }
        self.n = n_new;
    }

    /// What [`absorb`](Self::absorb) would leave in `r2`, without absorbing:
    /// [`merged_radius2`] at this summary's weight and radius.
    pub fn radius2_after(&self, z: &[f64], w: f64, mw: &[f64]) -> f64 {
        merged_radius2(self.n, self.r2, dist2(&self.c, z, mw), w)
    }

    /// Welford merge of two centred summaries, with the cross term:
    ///
    /// ```text
    /// n' = n + m,  a = n / n',  b = m / n'
    /// c' = c + b δ,   r2' = a r2 + b r2_o + a b ‖δ‖²_mw        δ = c_o − c
    /// ```
    pub fn merge_welford(&mut self, other: &ClusterSummary, mw: &[f64]) {
        let n_new = self.n + other.n;
        if n_new <= 0.0 {
            return;
        }
        let (a, b) = (self.n / n_new, other.n / n_new);
        let q = dist2(&self.c, &other.c, mw);
        for (ci, oi) in self.c.iter_mut().zip(&other.c) {
            *ci += b * (oi - *ci);
        }
        if q.is_finite() {
            self.r2 = a * self.r2 + b * other.r2 + a * b * q;
        }
        self.n = n_new;
    }
}

/// Diagonal EW moments of the features, for the metric: the same Welford
/// recursion as `ewcov.rs` without the co-moments (O(p) a row).
///
/// ```text
/// W' = lam W + w,  a = lam W / W',  b = w / W'
/// m' = m + b (x − m),   v' = a v + a b (x − m)²
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureMoments {
    /// EW sum of learned weights: the model's `n_eff`.
    pub w: f64,
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
}

impl FeatureMoments {
    pub fn new(p: usize) -> Self {
        Self {
            w: 0.0,
            mean: vec![0.0; p],
            var: vec![0.0; p],
        }
    }

    /// The clock passes: `W *= lam`.
    #[inline]
    pub fn decay(&mut self, lam: f64) {
        self.w *= lam;
    }

    /// Learn one row of weight `w > 0` (after [`decay`](Self::decay)).
    pub fn absorb(&mut self, x: &[f64], w: f64) {
        let w_new = self.w + w;
        if w_new <= 0.0 {
            return;
        }
        let (a, b) = (self.w / w_new, w / w_new);
        for ((m, v), &xi) in self.mean.iter_mut().zip(&mut self.var).zip(x) {
            let d = xi - *m;
            *m += b * d;
            *v = a * *v + a * b * d * d;
        }
        self.w = w_new;
    }

    /// The metric weights: `1 / var_i` where the variance is positive and its
    /// reciprocal finite, else `1` (raw units); all ones when not
    /// standardizing. A feature that is constant so far — or whose variance
    /// has gone subnormal — is measured in its own units rather than
    /// magnified without bound.
    pub fn metric(&self, standardize: bool, out: &mut [f64]) {
        for (o, &v) in out.iter_mut().zip(&self.var) {
            let inv = 1.0 / v;
            *o = if standardize && v > 0.0 && inv.is_finite() {
                inv
            } else {
                1.0
            };
        }
    }
}

/// splitmix64 (Steele, Lea & Flood 2014): the generator behind the
/// `kmeanspp` and `lloyd` seeding rules, written out so that the Python
/// reference draws the same numbers. Sixty-four bits of state, no
/// dependency, and a distinct stream per `seed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`: the top 53 bits scaled by `2⁻⁵³`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Index `i` with probability `w_i / Σw`: the first index whose running
    /// sum exceeds `u · Σw`. When the weights sum to nothing (or the sum is
    /// not finite) every index is equally likely: `⌊u · n⌋`.
    pub fn choice(&mut self, weights: &[f64]) -> usize {
        let n = weights.len();
        debug_assert!(n > 0);
        let u = self.uniform();
        let mut total = 0.0;
        for w in weights {
            total += w;
        }
        if total.is_nan() || total <= 0.0 || total.is_infinite() {
            return ((u * n as f64) as usize).min(n - 1);
        }
        let target = u * total;
        let mut acc = 0.0;
        let mut last = 0;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                acc += w;
                last = i;
                if acc > target {
                    return i;
                }
            }
        }
        // Rounding left `acc <= target` at the end: the last positive weight.
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist2_is_a_weighted_sum_of_squares_and_never_nan() {
        assert_eq!(dist2(&[0.0, 0.0], &[3.0, 4.0], &[1.0, 1.0]), 25.0);
        assert_eq!(dist2(&[1.0, 1.0], &[3.0, 4.0], &[0.25, 1.0]), 1.0 + 9.0);
        let big = dist2(&[0.0], &[1e200], &[1e200]);
        assert!(big.is_infinite() && big > 0.0);
        assert!(!dist2(&[1e200], &[1e200], &[1e300]).is_nan());
    }

    #[test]
    fn a_one_row_batch_merged_is_the_row_absorbed() {
        let mut direct = ClusterSummary::at(vec![1.0, 2.0], 3.0, 0.5);
        let mut via_batch = direct.clone();
        let z = [4.0, -1.0];
        let d2 = dist2(&direct.c, &z, &[1.0, 1.0]);
        direct.absorb_plain(&z, 0.7, d2);
        let mut batch = ClusterSummary::empty(2);
        batch.absorb_plain(&z, 0.7, d2);
        via_batch.merge_plain(&batch);
        assert_eq!(direct, via_batch);
        // The same numbers, longhand.
        let b = 0.7 / 3.7;
        assert_eq!(direct.n, 3.7);
        assert_eq!(direct.c[0], 1.0 + b * (4.0 - 1.0));
        assert_eq!(direct.c[1], 2.0 + b * (-1.0 - 2.0));
        assert_eq!(direct.r2, 0.5 + b * (d2 - 0.5));
    }

    #[test]
    fn a_zero_weight_row_is_a_no_op_even_first() {
        let mut s = ClusterSummary::empty(2);
        s.absorb_plain(&[1.0, 1.0], 0.0, 2.0);
        assert_eq!(s, ClusterSummary::empty(2));
        s.absorb(&[1.0, 1.0], 0.0, &[1.0, 1.0]);
        assert_eq!(s, ClusterSummary::empty(2));
        assert_eq!(s.radius2_after(&[1.0, 1.0], 0.0, &[1.0, 1.0]), 0.0);
        let mut m = FeatureMoments::new(2);
        m.absorb(&[1.0, 1.0], 0.0);
        assert_eq!(m, FeatureMoments::new(2));
    }

    #[test]
    fn welford_absorption_matches_the_batch_variance() {
        // Ten rows with weights: r2 must equal the weighted mean squared
        // deviation about the weighted mean, computed the batch way.
        let rows: Vec<[f64; 2]> = (0..10).map(|i| [i as f64, (i * i) as f64 * 0.1]).collect();
        let w: Vec<f64> = (0..10).map(|i| 0.5 + (i % 3) as f64).collect();
        let mw = [1.0, 0.5];
        let mut s = ClusterSummary::empty(2);
        for (r, &wi) in rows.iter().zip(&w) {
            s.absorb(r, wi, &mw);
        }
        let tot: f64 = w.iter().sum();
        let mean: Vec<f64> = (0..2)
            .map(|j| rows.iter().zip(&w).map(|(r, wi)| wi * r[j]).sum::<f64>() / tot)
            .collect();
        let var: f64 = rows
            .iter()
            .zip(&w)
            .map(|(r, wi)| wi * dist2(&mean, r, &mw))
            .sum::<f64>()
            / tot;
        assert!((s.n - tot).abs() < 1e-12);
        assert!((s.c[0] - mean[0]).abs() < 1e-12 && (s.c[1] - mean[1]).abs() < 1e-12);
        assert!((s.r2 - var).abs() < 1e-12 * var, "{} vs {var}", s.r2);
        // radius2_after is absorb without the absorb.
        let probe = s.radius2_after(&[3.0, 3.0], 2.0, &mw);
        let mut t = s.clone();
        t.absorb(&[3.0, 3.0], 2.0, &mw);
        assert_eq!(probe, t.r2);
    }

    #[test]
    fn welford_merge_is_the_union_of_the_rows() {
        let mw = [1.0, 2.0];
        let a_rows = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let b_rows = [[5.0, 5.0], [6.0, 5.0], [5.0, 7.0], [6.0, 7.0]];
        let (mut a, mut b, mut all) = (
            ClusterSummary::empty(2),
            ClusterSummary::empty(2),
            ClusterSummary::empty(2),
        );
        for r in &a_rows {
            a.absorb(r, 1.0, &mw);
            all.absorb(r, 1.0, &mw);
        }
        for r in &b_rows {
            b.absorb(r, 1.0, &mw);
            all.absorb(r, 1.0, &mw);
        }
        a.merge_welford(&b, &mw);
        assert!((a.n - all.n).abs() < 1e-12);
        assert!((a.c[0] - all.c[0]).abs() < 1e-12 && (a.c[1] - all.c[1]).abs() < 1e-12);
        assert!((a.r2 - all.r2).abs() < 1e-12);
    }

    #[test]
    fn an_infinite_distance_moves_the_centre_but_not_the_radius() {
        let mut s = ClusterSummary::at(vec![0.0], 1.0, 4.0);
        s.absorb_plain(&[1e200], 1.0, f64::INFINITY);
        assert_eq!(s.r2, 4.0);
        assert_eq!(s.c[0], 0.5e200);
        let mut t = ClusterSummary::at(vec![0.0], 1.0, 4.0);
        t.absorb(&[1e200], 1.0, &[1e200]);
        assert_eq!(t.r2, 4.0);
        assert!(t.c[0].is_finite());
    }

    #[test]
    fn feature_moments_are_welford_and_the_metric_guards_its_reciprocal() {
        let xs = [[1.0, 10.0], [3.0, 10.0], [2.0, 10.0], [6.0, 10.0]];
        let mut m = FeatureMoments::new(2);
        for x in &xs {
            m.decay(0.9);
            m.absorb(x, 1.0);
        }
        // Longhand.
        let (mut w, mut mean, mut var) = (0.0, [0.0; 2], [0.0; 2]);
        for x in &xs {
            let w_new = 0.9 * w + 1.0;
            let (a, b) = (0.9 * w / w_new, 1.0 / w_new);
            for i in 0..2 {
                let d = x[i] - mean[i];
                mean[i] += b * d;
                var[i] = a * var[i] + a * b * d * d;
            }
            w = w_new;
        }
        assert_eq!(m.w, w);
        assert_eq!(m.mean, mean);
        assert_eq!(m.var, var);
        assert_eq!(
            m.var[1], 0.0,
            "a constant feature has exactly zero variance"
        );
        let mut mw = [0.0; 2];
        m.metric(true, &mut mw);
        assert_eq!(mw, [1.0 / var[0], 1.0]);
        m.metric(false, &mut mw);
        assert_eq!(mw, [1.0, 1.0]);
        let mut tiny = FeatureMoments::new(1);
        tiny.var[0] = 1e-320;
        tiny.metric(true, &mut mw[..1]);
        assert_eq!(mw[0], 1.0, "a subnormal variance is not standardized by");
    }

    #[test]
    fn splitmix64_reference_values() {
        // The first outputs for seed 0 and seed 1, as published for the
        // algorithm (and as tests/reference_cluster.py must reproduce).
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(g.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        let mut g = SplitMix64::new(1);
        assert_eq!(g.next_u64(), 0x910A_2DEC_8902_5CC1);
        let mut g = SplitMix64::new(7);
        let u = g.uniform();
        assert!((0.0..1.0).contains(&u));
    }

    #[test]
    fn choice_walks_the_cumulative_weights() {
        // Force u by construction: with weights [0, 3, 0, 1] the index is 1
        // when u*4 < 3 and 3 otherwise; check the frequencies over many draws.
        let mut g = SplitMix64::new(42);
        let (mut ones, mut threes) = (0, 0);
        for _ in 0..4000 {
            match g.choice(&[0.0, 3.0, 0.0, 1.0]) {
                1 => ones += 1,
                3 => threes += 1,
                i => panic!("a zero-weight index {i} was chosen"),
            }
        }
        assert!((2800..3200).contains(&ones), "{ones}");
        assert_eq!(ones + threes, 4000);
        // No weight at all: uniform over the indices.
        let mut seen = [0; 3];
        for _ in 0..3000 {
            seen[g.choice(&[0.0, 0.0, 0.0])] += 1;
        }
        assert!(seen.iter().all(|&c| c > 800), "{seen:?}");
        assert!(g.choice(&[1.0]) == 0);
    }
}
