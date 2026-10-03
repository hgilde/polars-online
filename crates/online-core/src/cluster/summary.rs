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

/// `‖z − c‖` under the same metric, computed so that it cannot overflow
/// where [`dist2`] does: a row at the input bound against a variance at the
/// opposite scale squares past the double's range, and its distance,
/// `5e199` say, does not (review 2026-09-26, G1). Every scaled deviation is
/// taken relative to the largest, so the sum of squares stays near 1. Not
/// the root of `dist2` to the bit, so it is read only where that root is
/// not finite.
pub fn dist(c: &[f64], z: &[f64], mw: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    for i in 0..c.len() {
        scale = scale.max(((z[i] - c[i]) * mw[i].sqrt()).abs());
    }
    if !scale.is_finite() || scale == 0.0 {
        return scale;
    }
    let mut acc = 0.0;
    for i in 0..c.len() {
        let t = (z[i] - c[i]) * mw[i].sqrt() / scale;
        acc += t * t;
    }
    scale * acc.sqrt()
}

/// The radius² a summary of weight `n` and radius² `r2` would have after
/// absorbing weight `w` at squared distance `q` from its centre — the merged
/// radius DenStream's absorption test reads, and exactly what
/// [`ClusterSummary::absorb`] then stores. `r2` unchanged when the merged
/// weight is not positive or `q` is NaN, as `absorb` leaves it; an infinite
/// `q` -- a square that overflowed, the row infinitely far by that measure
/// -- gives an infinite radius, so no admission test takes the row
/// (review 2026-09-26, G1: it read as `r2`, and every such row was admitted).
pub fn merged_radius2(n: f64, r2: f64, q: f64, w: f64) -> f64 {
    let n_new = n + w;
    if n_new <= 0.0 || q.is_nan() {
        return r2;
    }
    if q.is_infinite() {
        return f64::INFINITY;
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

/// The metric's floor references each feature's long-run scale, tracked at
/// this many model half-lives (docs/PLAN.md task 102).
pub const LONG_HALFLIVES: f64 = 8.0;
/// The reference clips a squared deviation at this multiple of its own
/// variance (ten standard deviations), and the mean's step at the root.
pub const CLIP: f64 = 100.0;
/// A feature's reference starts from the medians of this many of its rows.
pub const START_ROWS: usize = 5;

/// Diagonal EW moments of the features, for the metric: the same Welford
/// recursion as `ewcov.rs` without the co-moments (O(p) a row), with the
/// means as pairs (`crate::comp`, docs/PLAN.md task 101), and beside them,
/// a feature at a time, the reference the metric's floor is a fraction of
/// (task 102): the same moments at [`LONG_HALFLIVES`] times the half-life,
/// each row's weight clipped at the reference's own and its squared
/// deviation at [`CLIP`] times the reference's variance, started from the
/// medians of the feature's first [`START_ROWS`] learned rows and started
/// over by a first move from no spread. A row at the input bound moves the
/// reference by a factor of `1 + CLIP / 4` at most, which a few of its
/// half-lives undo, where the EW variance takes a thousand to forget such a
/// row; a reference that took rows as they came never forgot one, and the
/// metric was lost for good (`model_contract`'s recovery).
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
    /// What `mean` leaves out: the low parts of the pairs. Empty in a state
    /// written before them, and sized at the next row.
    #[serde(default)]
    pub mean_lo: Vec<f64>,
    /// The reference, a feature at a time: its weight, its means as pairs
    /// and its variances, and the rows each starts from -- [`START_ROWS`]
    /// values and weights a feature, `n_start` of them kept. Empty in a
    /// state written before it, and started from the next rows.
    #[serde(default)]
    pub w_long: Vec<f64>,
    #[serde(default)]
    pub mean_long: Vec<f64>,
    #[serde(default)]
    pub mean_long_lo: Vec<f64>,
    #[serde(default)]
    pub var_long: Vec<f64>,
    #[serde(default)]
    pub start: Vec<f64>,
    #[serde(default)]
    pub start_w: Vec<f64>,
    #[serde(default)]
    pub n_start: Vec<u32>,
}

impl FeatureMoments {
    pub fn new(p: usize) -> Self {
        Self {
            w: 0.0,
            mean: vec![0.0; p],
            var: vec![0.0; p],
            mean_lo: vec![0.0; p],
            w_long: vec![0.0; p],
            mean_long: vec![0.0; p],
            mean_long_lo: vec![0.0; p],
            var_long: vec![0.0; p],
            start: vec![0.0; p * START_ROWS],
            start_w: vec![0.0; p * START_ROWS],
            n_start: vec![0; p],
        }
    }

    /// Whether the moments are those of `p` features: what a restored state
    /// must hold to be updated (review 2026-09-18, B3).
    pub fn has_shape(&self, p: usize) -> bool {
        self.mean.len() == p && self.var.len() == p
    }

    /// The clock passes: `W *= lam`, and each reference's weight by
    /// `lam_long`, the same decay over `d_clock / LONG_HALFLIVES`.
    #[inline]
    pub fn decay(&mut self, lam: f64, lam_long: f64) {
        self.w *= lam;
        for wl in &mut self.w_long {
            *wl *= lam_long;
        }
    }

    /// Learn one row of weight `w > 0` (after [`decay`](Self::decay)): the
    /// EW moments, then each feature's reference. A row of weight 0 takes
    /// no step in either (`crate::comp::add` says why for the pairs).
    pub fn absorb(&mut self, x: &[f64], w: f64) {
        let w_new = self.w + w;
        if w_new <= 0.0 {
            return;
        }
        self.size();
        let (a, b) = (self.w / w_new, w / w_new);
        for (((m, l), v), &xi) in self
            .mean
            .iter_mut()
            .zip(self.mean_lo.iter_mut())
            .zip(self.var.iter_mut())
            .zip(x)
        {
            let d = crate::comp::dev(xi, *m, *l);
            *v = a * *v + a * b * d * d;
            if b > 0.0 {
                crate::comp::add(m, l, b * d);
            }
        }
        self.w = w_new;
        if w > 0.0 {
            for (i, &xi) in x.iter().enumerate() {
                self.step_long(i, xi, w);
            }
        }
    }

    /// Feature `i`'s reference takes one row of weight `w > 0`. Its first
    /// [`START_ROWS`] rows are kept, and the reference starts from their
    /// medians -- of the values, of the squared deviations from that
    /// median, and for its weight [`START_ROWS`] times the median of the
    /// rows' weights -- which up to two rows at the input bound among them
    /// do not move, and which scale with the weights as every other moment
    /// does. From there, the Welford step at a weight clipped at the
    /// reference's own (a row takes at most half of it) and a deviation
    /// clipped at [`CLIP`] times its variance. A feature without spread has
    /// no scale to clip against, so its first move starts its reference
    /// over, that row the first of the next five; a reference whose weight
    /// has decayed to nothing takes the row whole and, having no spread
    /// then, starts over on the next.
    fn step_long(&mut self, i: usize, xi: f64, w: f64) {
        let s = START_ROWS;
        let kept = self.n_start[i] as usize;
        if kept < s {
            self.start[i * s + kept] = xi;
            self.start_w[i * s + kept] = w;
            self.n_start[i] += 1;
            if kept + 1 < s {
                return;
            }
            let mut col = self.start[i * s..(i + 1) * s].to_vec();
            let m = median(&mut col);
            let mut sq: Vec<f64> = col.iter().map(|v| (v - m) * (v - m)).collect();
            let mut ws = self.start_w[i * s..(i + 1) * s].to_vec();
            self.mean_long[i] = m;
            self.mean_long_lo[i] = 0.0;
            self.var_long[i] = median(&mut sq);
            self.w_long[i] = s as f64 * median(&mut ws);
            return;
        }
        let d = crate::comp::dev(xi, self.mean_long[i], self.mean_long_lo[i]);
        if self.var_long[i] == 0.0 && d != 0.0 {
            self.n_start[i] = 0;
            self.w_long[i] = 0.0;
            self.step_long(i, xi, w);
            return;
        }
        let wl = self.w_long[i];
        let w = if wl > 0.0 { w.min(wl) } else { w };
        let wl_new = wl + w;
        let (a, b) = (wl / wl_new, w / wl_new);
        let cap = CLIP * self.var_long[i];
        let d = if cap > 0.0 && d * d > cap {
            cap.sqrt().copysign(d)
        } else {
            d
        };
        self.var_long[i] = a * self.var_long[i] + a * b * d * d;
        crate::comp::add(&mut self.mean_long[i], &mut self.mean_long_lo[i], b * d);
        self.w_long[i] = wl_new;
    }

    /// The low parts and the references are as wide as the means. A state
    /// written before them starts them here: the low parts at 0, the
    /// references from the next rows.
    fn size(&mut self) {
        let p = self.mean.len();
        if self.mean_lo.len() != p {
            self.mean_lo = vec![0.0; p];
        }
        let s = START_ROWS;
        if self.w_long.len() != p
            || self.mean_long.len() != p
            || self.mean_long_lo.len() != p
            || self.var_long.len() != p
            || self.start.len() != p * s
            || self.start_w.len() != p * s
            || self.n_start.len() != p
            || self.n_start.iter().any(|&n| n as usize > s)
        {
            self.w_long = vec![0.0; p];
            self.mean_long = vec![0.0; p];
            self.mean_long_lo = vec![0.0; p];
            self.var_long = vec![0.0; p];
            self.start = vec![0.0; p * s];
            self.start_w = vec![0.0; p * s];
            self.n_start = vec![0; p];
        }
    }

    /// The metric weights: `1 / v_i`, with `v_i` the EW variance floored at
    /// `scale_floor` times the reference's variance, where that is positive
    /// and its reciprocal finite, else `1` (raw units); all ones when not
    /// standardizing. The floor slows what a feature that has gone quiet can
    /// come to count for: `1 / var` alone grows as `2^Q` over `Q` half-lives
    /// of quiet, a million at twenty, and the row on which the feature moves
    /// again is then infinitely far from every centre, which the argmin
    /// cancels but a radius does not; floored, the weight grows as
    /// `2^(Q / LONG_HALFLIVES) / scale_floor`, about 57 at twenty (docs/PLAN.md
    /// task 102). A feature that is constant so far -- or whose variance has
    /// gone subnormal -- is measured in its own units rather than magnified
    /// without bound.
    pub fn metric(&self, standardize: bool, scale_floor: f64, out: &mut [f64]) {
        for (i, (o, &v)) in out.iter_mut().zip(&self.var).enumerate() {
            let floor = scale_floor * self.var_long.get(i).copied().unwrap_or(0.0);
            let v = if floor > v { floor } else { v };
            let inv = 1.0 / v;
            *o = if standardize && v > 0.0 && inv.is_finite() {
                inv
            } else {
                1.0
            };
        }
    }
}

/// The middle value of an odd number of values, all finite here.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
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

    /// Where the square overflows the distance is still a number, close to
    /// the root of the square wherever that exists; and an overflowed square
    /// merges into an infinite radius, which no admission test takes
    /// (review 2026-09-26, G1: a proptest at the input bound found `micro`
    /// reporting `inf` and absorbing the row).
    #[test]
    fn an_overflowed_distance_is_far_and_still_measured() {
        for (c, z, mw) in [
            (vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 1.0]),
            (vec![1.0, 1.0], vec![3.0, 4.0], vec![0.25, 1.0]),
            (vec![-2.07, 0.5], vec![1e6, -3.0], vec![7.0, 0.01]),
        ] {
            let want = dist2(&c, &z, &mw).sqrt();
            let got = dist(&c, &z, &mw);
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want,
                "{got} vs {want}"
            );
        }
        assert_eq!(dist(&[1.0], &[1.0], &[3.0]), 0.0);
        let far = dist(&[-2.07, 0.0], &[1e100, 0.0], &[2.5e199, 1.0]);
        assert!(far.is_finite(), "{far}");
        assert!(
            (far - 1e100 * 2.5e199f64.sqrt()).abs() <= 1e-12 * far,
            "{far}"
        );
        assert!(dist2(&[-2.07, 0.0], &[1e100, 0.0], &[2.5e199, 1.0]).is_infinite());
        assert_eq!(merged_radius2(1.0, 0.1, f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(merged_radius2(1.0, 0.1, f64::NAN, 1.0), 0.1);
        assert_eq!(
            merged_radius2(0.0, 0.1, f64::INFINITY, 0.0),
            0.1,
            "no weight, no merge"
        );
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
            m.decay(0.9, 0.9f64.powf(1.0 / LONG_HALFLIVES));
            m.absorb(x, 1.0);
        }
        // Longhand, the means as pairs.
        let (mut w, mut mean, mut lo, mut var) = (0.0, [0.0; 2], [0.0; 2], [0.0; 2]);
        for x in &xs {
            let w_new = 0.9 * w + 1.0;
            let (a, b) = (0.9 * w / w_new, 1.0 / w_new);
            for i in 0..2 {
                let d = crate::comp::dev(x[i], mean[i], lo[i]);
                var[i] = a * var[i] + a * b * d * d;
                crate::comp::add(&mut mean[i], &mut lo[i], b * d);
            }
            w = w_new;
        }
        assert_eq!(m.w, w);
        assert_eq!(m.mean, mean);
        assert_eq!(m.mean_lo, lo);
        assert_eq!(m.var, var);
        assert_eq!(
            m.var[1], 0.0,
            "a constant feature has exactly zero variance"
        );
        let mut mw = [0.0; 2];
        m.metric(true, 0.0, &mut mw);
        assert_eq!(mw, [1.0 / var[0], 1.0]);
        m.metric(false, 0.0, &mut mw);
        assert_eq!(mw, [1.0, 1.0]);
        let mut tiny = FeatureMoments::new(1);
        tiny.var[0] = 1e-320;
        tiny.metric(true, 0.0, &mut mw[..1]);
        assert_eq!(mw[0], 1.0, "a subnormal variance is not standardized by");
    }

    fn lam_long(lam: f64) -> f64 {
        lam.powf(1.0 / LONG_HALFLIVES)
    }

    /// The metric's floor (docs/PLAN.md task 102), on two features of which
    /// one goes quiet: its EW variance decays without bound and the floor
    /// holds its weight at `1 / (scale_floor · var_long)`, the reference
    /// decaying at an eighth of the rate; the feature that keeps moving
    /// reads its EW variance at every floor; and a floor of 0 is the
    /// reciprocal as it was.
    #[test]
    fn the_metric_is_floored_at_a_fraction_of_the_long_run_variance() {
        let lam = 0.5;
        let mut m = FeatureMoments::new(2);
        for i in 0..40 {
            m.decay(lam, lam_long(lam));
            m.absorb(&[(i % 5) as f64, (i % 3) as f64], 1.0);
        }
        let reference = m.var_long[0];
        assert!(reference > 0.5 && reference < 4.0, "{reference}");
        for i in 0..60 {
            m.decay(lam, lam_long(lam));
            m.absorb(&[2.0, (i % 3) as f64], 1.0);
        }
        assert!(m.var[0] >= 0.0 && m.var[0] < 1e-12, "{}", m.var[0]);
        // Sixty rows at half-life 1 are 7.5 of the reference's half-lives.
        let decayed = m.var_long[0] / reference;
        assert!(decayed > 0.002 && decayed < 0.02, "{decayed}");
        assert!(
            m.var[1] > 0.1 * m.var_long[1],
            "the moving feature is above its floor"
        );
        let mut mw = [0.0; 2];
        m.metric(true, 0.0, &mut mw);
        assert_eq!(mw, [1.0 / m.var[0], 1.0 / m.var[1]]);
        m.metric(true, 0.1, &mut mw);
        assert_eq!(mw, [1.0 / (0.1 * m.var_long[0]), 1.0 / m.var[1]]);
        m.metric(false, 0.1, &mut mw);
        assert_eq!(mw, [1.0, 1.0]);
    }

    /// The reference starts from the medians of the feature's first five
    /// rows, and not before: nothing of it is set while the rows are still
    /// being kept. A row at the input bound among them does not move the
    /// medians, nor a weight at the bound: the values' median, the squared
    /// deviations' median, and five times the weights' median. The EW
    /// moments take that row, and forget it in a thousand half-lives; the
    /// reference never has it, and the floor sits far below the EW variance
    /// until then.
    #[test]
    fn the_reference_starts_from_the_medians_of_the_first_five_rows() {
        let mut m = FeatureMoments::new(1);
        for (v, w) in [
            (crate::INPUT_BOUND, 1.0),
            (1.0, crate::INPUT_BOUND),
            (2.0, 1.0),
            (3.0, 1.0),
            (4.0, 1.0),
        ] {
            m.decay(0.9, 0.99);
            m.absorb(&[v], w);
            assert_eq!(m.var_long.len(), 1);
            if m.n_start[0] < START_ROWS as u32 {
                assert_eq!(
                    (m.mean_long[0], m.var_long[0], m.w_long[0]),
                    (0.0, 0.0, 0.0),
                    "not started before its fifth row"
                );
            }
        }
        assert_eq!(
            (m.mean_long[0], m.var_long[0], m.w_long[0]),
            (3.0, 1.0, 5.0)
        );
        assert_eq!(m.n_start[0], START_ROWS as u32);
        assert!(
            m.var[0] > 1e90,
            "the EW variance took the rows: {}",
            m.var[0]
        );
        let mut mw = [0.0; 1];
        m.metric(true, 0.1, &mut mw);
        assert_eq!(mw[0], 1.0 / m.var[0]);
    }

    /// A feature that starts its reference over reads its own rows' weights,
    /// not the weights of the rows another feature started on: feature 1 is
    /// constant while feature 0 moves at weight 2, then moves at weight 1,
    /// and its reference's weight is five times 1.
    #[test]
    fn a_restarted_reference_reads_its_own_rows_weights() {
        let mut m = FeatureMoments::new(2);
        let mut g = SplitMix64::new(6);
        for _ in 0..START_ROWS {
            m.absorb(&[g.uniform(), 7.0], 2.0);
        }
        assert_eq!(m.w_long, vec![10.0, 10.0]);
        for _ in 0..START_ROWS {
            m.absorb(&[g.uniform(), 7.0 + g.uniform()], 1.0);
        }
        assert_eq!(m.n_start, vec![5, 5]);
        assert_eq!(m.w_long[1], 5.0, "five rows at weight 1");
        assert!(m.w_long[0] > 10.0, "feature 0 went on: {}", m.w_long[0]);
    }

    /// A row at the input bound, at a weight at the bound, moves the
    /// reference by a bounded factor -- `1 + CLIP / 4` in the variance (a
    /// row takes at most half of it, so `(1 − b)(1 + CLIP·b)` peaks at
    /// `b = 1/2`), half the root of `CLIP` standard deviations in the
    /// mean, twice in the weight -- and its half-lives undo it: the
    /// reference is within a factor of two of its old value eleven of them
    /// later, with and without decay.
    #[test]
    fn a_row_at_the_bound_moves_the_reference_by_a_bounded_factor() {
        for lam in [0.9, 1.0] {
            let mut m = FeatureMoments::new(1);
            let mut g = SplitMix64::new(3);
            for _ in 0..200 {
                m.decay(lam, lam_long(lam));
                m.absorb(&[g.uniform()], 1.0);
            }
            let (mean, var, weight) = (m.mean_long[0], m.var_long[0], m.w_long[0]);
            m.decay(lam, lam_long(lam));
            m.absorb(&[crate::INPUT_BOUND], crate::INPUT_BOUND);
            assert!(
                m.var_long[0] <= (1.0 + 0.25 * CLIP) * var,
                "lam {lam}: {} from {var}",
                m.var_long[0]
            );
            assert!(
                (m.mean_long[0] - mean).abs() <= 0.5 * (CLIP * var).sqrt() + 1e-12,
                "lam {lam}: {} from {mean}",
                m.mean_long[0]
            );
            assert!(
                m.w_long[0] <= 2.0 * weight,
                "lam {lam}: {} from {weight}",
                m.w_long[0]
            );
            assert!(
                m.var[0] > 1e100,
                "the EW variance took the row: {}",
                m.var[0]
            );
            if lam < 1.0 {
                // Six hundred rows at half-life 6.6 are eleven of the
                // reference's half-lives; 2.8 times its old value after six.
                for _ in 0..600 {
                    m.decay(lam, lam_long(lam));
                    m.absorb(&[g.uniform()], 1.0);
                }
                let back = m.var_long[0] / var;
                assert!(back > 0.5 && back < 2.0, "{back}");
            }
        }
    }

    /// A feature without spread has no scale to clip against, so its first
    /// move starts its reference over: five rows of one value, then a row a
    /// million away, then rows of unit spread. Taken as it came, the row put
    /// 1e11 into the reference and the floor held the feature under-weighted
    /// a millionfold for two hundred half-lives; started over, the reference
    /// is the medians of that row and the next four, at the unit spread.
    #[test]
    fn a_feature_without_spread_starts_its_reference_over_on_its_first_move() {
        let lam = 0.9;
        let mut m = FeatureMoments::new(1);
        for _ in 0..START_ROWS {
            m.decay(lam, lam_long(lam));
            m.absorb(&[7.0], 1.0);
        }
        assert_eq!((m.var_long[0], m.w_long[0]), (0.0, 5.0));
        m.decay(lam, lam_long(lam));
        m.absorb(&[1e6], 1.0);
        assert_eq!(m.n_start[0], 1, "started over on the move");
        for v in [7.4, 6.5, 7.9, 6.2] {
            m.decay(lam, lam_long(lam));
            m.absorb(&[v], 1.0);
        }
        assert_eq!(m.n_start[0], START_ROWS as u32);
        assert!(
            m.var_long[0] > 0.01 && m.var_long[0] < 10.0,
            "{}",
            m.var_long[0]
        );
        assert!(
            m.mean_long[0] > 6.0 && m.mean_long[0] < 8.0,
            "{}",
            m.mean_long[0]
        );
    }

    /// The reference scales with the rows' weights as every other moment
    /// does, its start included: the same stream at weights 1 and 1e-3
    /// reads the same floored metric, through a quiet spell.
    #[test]
    fn the_reference_scales_with_the_weights() {
        let lam = 0.5;
        let run = |scale: f64| {
            let mut m = FeatureMoments::new(2);
            let mut g = SplitMix64::new(9);
            let mut mws = Vec::new();
            for i in 0..120 {
                m.decay(lam, lam_long(lam));
                let x0 = if i < 40 { g.uniform() } else { 0.5 };
                m.absorb(&[x0, g.uniform()], scale * (0.5 + g.uniform()));
                let mut mw = [0.0; 2];
                m.metric(true, 0.1, &mut mw);
                mws.push(mw);
            }
            mws
        };
        let (a, b) = (run(1.0), run(1e-3));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            for j in 0..2 {
                assert!(
                    (x[j] - y[j]).abs() <= 1e-9 * x[j],
                    "row {i}, feature {j}: {} at weights 1 against {} at 1e-3",
                    x[j],
                    y[j]
                );
            }
        }
    }

    /// A row of weight 0 takes no step in the moments' pairs nor in the
    /// reference: the rows 0.7 and 5.292162135665459 leave a low part of a
    /// whole step, where a step of 0 would round `hi` up
    /// (`crate::comp::add`).
    #[test]
    fn a_row_of_no_weight_takes_no_step_in_the_moments() {
        let mut m = FeatureMoments::new(1);
        m.absorb(&[0.7], 1.0);
        m.absorb(&[5.292162135665459], 1.0);
        let before = m.clone();
        assert_eq!(
            (m.mean[0], m.mean_lo[0]),
            (2.996081067832729, 4.440892098500626e-16)
        );
        m.absorb(&[9.0], 0.0);
        assert_eq!(m, before);
    }

    /// A reference whose weight has decayed to nothing -- a gap of
    /// thousands of its half-lives -- takes the next row whole and starts
    /// over on the one after, with no NaN on the way.
    #[test]
    fn a_gap_past_the_references_weight_starts_it_over() {
        let mut m = FeatureMoments::new(1);
        let mut g = SplitMix64::new(4);
        for _ in 0..20 {
            m.decay(0.9, lam_long(0.9));
            m.absorb(&[g.uniform()], 1.0);
        }
        m.decay(0.0, 0.0);
        assert_eq!(m.w_long[0], 0.0);
        m.absorb(&[0.3], 1.0);
        assert_eq!((m.var_long[0], m.w_long[0], m.n_start[0]), (0.0, 1.0, 5));
        for v in [0.6, 0.2, 0.8, 0.4, 0.5] {
            m.decay(0.9, lam_long(0.9));
            m.absorb(&[v], 1.0);
        }
        assert_eq!(m.n_start[0], START_ROWS as u32);
        assert!(
            m.var_long[0] > 0.0 && m.var_long[0].is_finite(),
            "{}",
            m.var_long[0]
        );
        let mut mw = [0.0; 1];
        m.metric(true, 0.1, &mut mw);
        assert!(mw[0].is_finite() && mw[0] > 0.0);
    }

    /// A state saved between the second and third rows of a feature's start
    /// resumes to the bit: two rows, a round trip, three rows are five rows
    /// straight through.
    #[test]
    fn a_state_saved_mid_start_resumes_to_the_bit() {
        let rows = [
            [1.0, 5.0],
            [2.0, 4.0],
            [1.5, 6.0],
            [0.5, 4.5],
            [2.5, 5.5],
            [1.2, 5.1],
        ];
        let mut straight = FeatureMoments::new(2);
        let mut cut = FeatureMoments::new(2);
        for (i, x) in rows.iter().enumerate() {
            straight.decay(0.9, lam_long(0.9));
            straight.absorb(x, 1.0);
            if i == 2 {
                let bytes = rmp_serde::to_vec_named(&cut).unwrap();
                cut = rmp_serde::from_slice(&bytes).unwrap();
                let bytes = rmp_serde::to_vec(&cut).unwrap();
                cut = rmp_serde::from_slice(&bytes).unwrap();
            }
            cut.decay(0.9, lam_long(0.9));
            cut.absorb(x, 1.0);
        }
        assert_eq!(cut, straight);
        assert_eq!(cut.n_start, vec![5, 5]);
    }

    /// A state that lost one field of the reference -- any one, or has a
    /// start of the wrong width -- starts the whole reference over at the
    /// next rows, as one that lost them all does: the parts are sized
    /// together or not at all.
    #[test]
    fn a_state_missing_one_reference_field_starts_it_over() {
        let mut m = FeatureMoments::new(2);
        for i in 0..8 {
            m.absorb(&[i as f64, 1.0 - i as f64], 1.0);
        }
        for key in [
            "w_long",
            "mean_long",
            "mean_long_lo",
            "var_long",
            "start",
            "start_w",
            "n_start",
        ] {
            let mut old = serde_json::to_value(&m).unwrap();
            if key == "start" {
                old["start"] = serde_json::json!([1.0, 2.0, 3.0]);
            } else {
                assert!(old.as_object_mut().unwrap().remove(key).is_some(), "{key}");
            }
            let mut back: FeatureMoments = serde_json::from_value(old).unwrap();
            for i in 0..START_ROWS {
                back.absorb(&[10.0 + i as f64, 2.0], 1.0);
            }
            assert_eq!(back.n_start, vec![5, 5], "without {key}");
            assert_eq!(
                back.mean_long,
                vec![12.0, 2.0],
                "without {key}: the medians of the next five rows"
            );
            assert_eq!(back.w_long, vec![5.0, 5.0], "without {key}");
        }
    }

    /// A state written before the low parts and the reference loads, reads
    /// its metric without a floor, and starts them at the next rows, as
    /// wide as the means.
    #[test]
    fn a_state_without_the_reference_starts_it() {
        let mut m = FeatureMoments::new(2);
        for i in 0..6 {
            m.absorb(&[i as f64, 2.0 * i as f64], 1.0);
        }
        let mut old = serde_json::to_value(&m).unwrap();
        for key in [
            "mean_lo",
            "w_long",
            "mean_long",
            "mean_long_lo",
            "var_long",
            "start",
            "start_w",
            "n_start",
        ] {
            assert!(old.as_object_mut().unwrap().remove(key).is_some(), "{key}");
        }
        let mut back: FeatureMoments = serde_json::from_value(old).unwrap();
        assert!(back.mean_lo.is_empty() && back.var_long.is_empty() && back.n_start.is_empty());
        assert!(back.has_shape(2));
        let mut mw = [0.0; 2];
        back.metric(true, 0.1, &mut mw);
        assert_eq!(
            mw,
            [1.0 / back.var[0], 1.0 / back.var[1]],
            "no reference: no floor"
        );
        for i in 0..START_ROWS {
            back.absorb(&[10.0 + i as f64, 1.0], 1.0);
        }
        assert_eq!(
            (back.mean_lo.len(), back.var_long.len(), back.n_start),
            (2, 2, vec![5, 5])
        );
        assert_eq!(
            back.mean_long,
            vec![12.0, 1.0],
            "the medians of the next five rows"
        );
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
