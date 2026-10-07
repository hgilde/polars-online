//! Streaming scalar statistics (docs/ENHANCEMENTS.md E23).
//!
//! Two diagnostics that complement `EwCov`'s moments and answer questions a
//! standard deviation cannot:
//!
//! - [`EwQuantile`] — an exponentially weighted quantile sketch, DDSketch's
//!   buckets decayed on the model's clock, within `tanh(1/128)` of the
//!   exponentially weighted quantile at every level at once. A residual
//!   distribution with fat tails has a 99th percentile far above `2.33·σ`,
//!   and only a quantile estimate will say so. It replaced P² (Jain &
//!   Chlamtac, 1985), which never forgot (docs/PLAN.md task 146).
//! - [`EwAutoCorr`] — exponentially weighted lag-`k` autocorrelation. Residual
//!   autocorrelation is the classic sign that a model is mis-specified: an
//!   out-of-sample residual stream should look like noise, and does not when a
//!   feature is missing or the decay is too slow.

use serde::{Deserialize, Serialize};

/// Buckets per octave of [`EwQuantile`]: a bucket spans a ratio of at most
/// `e^(1/64)`, so its representative is within [`EW_QUANTILE_ALPHA`] of every
/// value in it.
const BINS: f64 = 64.0;

/// The relative accuracy of [`EwQuantile`]: `tanh(1 / (2 · BINS))`, about
/// 0.78%.
pub const EW_QUANTILE_ALPHA: f64 = 0.007_812_341_058_161_014;

/// The decay a sketch takes before its buckets are rescaled to it.
const RENORM: f64 = 5.421_010_862_427_522e-20; // 2^-64

/// An end bucket at or below this share of the total is dropped when the
/// sketch is rescaled.
const PRUNE: f64 = 1e-12;

/// An exponentially weighted quantile of non-negative values: DDSketch
/// (Masson, Rim & Lee, 2019, "DDSketch: a fast and fully-mergeable quantile
/// sketch with relative-error guarantees", PVLDB 12(12), 2195-2205), its
/// bucket weights decayed on the model's clock (docs/PLAN.md task 146).
///
/// A value `v = m · 2^e`, `m` in `[1, 2)`, lands in bucket `⌊64 (e + m − 1)⌋`:
/// `e + m − 1` is `log2 v` interpolated linearly between powers of two, so
/// the index is exact arithmetic on the float's bits and no libm result
/// enters the state. Its slope against `log2 v` is at least `ln 2`, so a
/// bucket spans a ratio of at most `e^(1/64)`, and its representative
/// `2 lo hi / (lo + hi)` is within `α = tanh(1/128)` of every value in it. A
/// value below the smallest normal float counts as 0.
///
/// Each row multiplies every bucket by its decay `λ`, and a value adds its
/// weight `w` to its bucket. The quantile at `p` is the representative of
/// the first bucket, ascending, at which the cumulative weight reaches `p`
/// of the total -- the bucket holding the smallest `x` with `W(≤ x) ≥ p W`,
/// the exponentially weighted quantile -- so the estimate is within `α` of
/// it, relatively. One sketch answers every level.
///
/// The decay is kept as one factor beside the buckets and folded into them
/// every 64 halvings, when an end bucket at or under `1e-12` of the total is
/// dropped; each level keeps the bucket its quantile sits in and the weight
/// below it, so a row costs a few comparisons, not a walk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwQuantile {
    /// The levels reported, each in (0, 1).
    levels: Vec<f64>,
    /// Bucket weights from index `lo` up, each over `scale`.
    buckets: Vec<f64>,
    lo: i32,
    /// The weight of the values read as 0, over `scale`.
    zero: f64,
    /// `zero` plus every bucket.
    total: f64,
    /// The decay since the buckets were last rescaled: a stored weight times
    /// `scale` is the weight.
    scale: f64,
    /// Per level, the bucket its quantile is in (`None`: the zero bucket) and
    /// the stored weight below that bucket.
    at: Vec<Option<i32>>,
    below: Vec<f64>,
}

impl EwQuantile {
    pub fn new(levels: &[f64]) -> Result<Self, String> {
        if levels.iter().any(|p| !(*p > 0.0 && *p < 1.0)) {
            return Err("EwQuantile: every level must be in (0, 1)".into());
        }
        Ok(Self::empty(levels.to_vec()))
    }

    /// No weight at `levels`, which the caller has checked.
    fn empty(levels: Vec<f64>) -> Self {
        let n = levels.len();
        Self {
            levels,
            buckets: Vec::new(),
            lo: 0,
            zero: 0.0,
            total: 0.0,
            scale: 1.0,
            at: vec![None; n],
            below: vec![0.0; n],
        }
    }

    /// Whether a restored sketch can be read and added to: a pointer and a
    /// weight below it per level, each level in (0, 1), every pointer at the
    /// zero bucket or inside the buckets, the buckets inside the indices a
    /// value can have, every weight finite and none below 0, and a scale in
    /// (0, 1]. What `add`, `settle` and `get` index by and divide by; a
    /// pointer outside the buckets loaded and the next value indexed
    /// `buckets[k - lo]` far past them (review 2026-10-06, CB2). The weight
    /// below a level is a running sum its pointer's moves add to and take
    /// from, which rounding can leave a hair under 0, so it is held to
    /// finite alone.
    pub fn has_shape(&self) -> bool {
        let n = self.levels.len();
        let lo = i64::from(self.lo);
        let end = lo + self.buckets.len() as i64;
        let indices = i64::from(Self::index(f64::MIN_POSITIVE))..=i64::from(Self::index(f64::MAX));
        let weight = |w: &f64| w.is_finite() && *w >= 0.0;
        self.at.len() == n
            && self.below.len() == n
            && self.levels.iter().all(|p| *p > 0.0 && *p < 1.0)
            && (self.buckets.is_empty() || (indices.contains(&lo) && indices.contains(&(end - 1))))
            && self
                .at
                .iter()
                .all(|a| a.is_none_or(|k| (lo..end).contains(&i64::from(k))))
            && self.buckets.iter().all(weight)
            && weight(&self.zero)
            && weight(&self.total)
            && self.below.iter().all(|b| b.is_finite())
            && self.scale > 0.0
            && self.scale <= 1.0
    }

    /// The levels this sketch reports, in the order [`EwQuantile::get`]
    /// takes them.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// The bucket of a positive normal value.
    fn index(v: f64) -> i32 {
        let bits = v.to_bits();
        let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
        let m = (bits & ((1u64 << 52) - 1)) as f64 / (1u64 << 52) as f64;
        // `64 e + ⌊64 m⌋`, not `⌊64 (e + m)⌋`: the sum would round `m` off at
        // a large `e`, and carry a value just under a power of two into the
        // next octave -- past the top one at `f64::MAX`.
        64 * e + (m * BINS).floor() as i32
    }

    /// `2^(y)` interpolated as [`EwQuantile::index`] reads it, `y = e + m − 1`.
    fn value_at(y: f64) -> f64 {
        let e = y.floor();
        let m = 1.0 + (y - e);
        let biased = e as i64 + 1023;
        if biased >= 2047 {
            return f64::INFINITY;
        }
        m * f64::from_bits((biased as u64) << 52)
    }

    /// The value reported for bucket `k`, `2 lo hi / (lo + hi)` of its
    /// bounds; the top bucket's upper bound is `2^1024`, so the ratio is
    /// taken an octave down, where both are finite.
    fn representative(k: i32) -> f64 {
        let (y0, y1) = (f64::from(k) / BINS, f64::from(k + 1) / BINS);
        let lo = Self::value_at(y0);
        let hi = Self::value_at(y1);
        let r = if hi.is_finite() {
            lo / hi
        } else {
            Self::value_at(y0 - 1.0) / Self::value_at(y1 - 1.0)
        };
        lo * (2.0 / (1.0 + r))
    }

    fn weight(&self, at: Option<i32>) -> f64 {
        match at {
            None => self.zero,
            Some(k) => self.buckets[(k - self.lo) as usize],
        }
    }

    fn next(&self, at: Option<i32>) -> Option<Option<i32>> {
        let end = self.lo + self.buckets.len() as i32;
        match at {
            None if self.buckets.is_empty() => None,
            None => Some(Some(self.lo)),
            Some(k) if k + 1 < end => Some(Some(k + 1)),
            Some(_) => None,
        }
    }

    fn prev(&self, at: Option<i32>) -> Option<Option<i32>> {
        match at {
            None => None,
            Some(k) if k > self.lo => Some(Some(k - 1)),
            Some(_) => Some(None),
        }
    }

    /// Move level `l`'s pointer to the bucket its quantile is in.
    fn settle(&mut self, l: usize) {
        let target = self.levels[l] * self.total;
        loop {
            let at = self.at[l];
            if self.below[l] + self.weight(at) < target {
                if let Some(n) = self.next(at) {
                    self.below[l] += self.weight(at);
                    self.at[l] = n;
                    continue;
                }
            } else if self.below[l] >= target
                && let Some(p) = self.prev(at)
            {
                self.at[l] = p;
                self.below[l] -= self.weight(p);
                continue;
            }
            break;
        }
    }

    /// Every pointer and the total from the buckets, by a walk.
    fn rebuild(&mut self) {
        self.total = self.zero + self.buckets.iter().sum::<f64>();
        for l in 0..self.levels.len() {
            self.at[l] = None;
            self.below[l] = 0.0;
            self.settle(l);
        }
    }

    /// The clock moves on by a step whose decay is `lam`.
    pub fn age(&mut self, lam: f64) {
        self.scale *= lam;
        if self.scale < RENORM {
            self.rescale();
        }
    }

    /// Fold the decay into the buckets, drop the ends that no longer count,
    /// and rebuild the pointers from the rescaled weights.
    fn rescale(&mut self) {
        let s = self.scale;
        self.zero *= s;
        self.buckets.iter_mut().for_each(|b| *b *= s);
        self.scale = 1.0;
        let total = self.zero + self.buckets.iter().sum::<f64>();
        let floor = PRUNE * total;
        let keep_from = self.buckets.iter().position(|b| *b > floor);
        match keep_from {
            None => {
                self.buckets.clear();
                self.lo = 0;
            }
            Some(first) => {
                let last = self
                    .buckets
                    .iter()
                    .rposition(|b| *b > floor)
                    .unwrap_or(first);
                self.buckets.truncate(last + 1);
                self.buckets.drain(..first);
                self.lo += first as i32;
            }
        }
        if self.zero <= floor {
            self.zero = 0.0;
        }
        self.rebuild();
    }

    /// One value `v >= 0` at weight `w > 0`; anything else is not seen.
    pub fn add(&mut self, v: f64, w: f64) {
        if !(v >= 0.0 && v.is_finite() && w > 0.0 && w.is_finite()) {
            return;
        }
        let s = w / self.scale;
        let at = if v < f64::MIN_POSITIVE {
            self.zero += s;
            None
        } else {
            let k = Self::index(v);
            if self.buckets.is_empty() {
                self.lo = k;
                self.buckets.push(0.0);
            } else if k < self.lo {
                let grow = (self.lo - k) as usize;
                self.buckets.splice(0..0, std::iter::repeat_n(0.0, grow));
                self.lo = k;
            } else if k >= self.lo + self.buckets.len() as i32 {
                self.buckets.resize((k - self.lo + 1) as usize, 0.0);
            }
            self.buckets[(k - self.lo) as usize] += s;
            Some(k)
        };
        self.total += s;
        for l in 0..self.levels.len() {
            // `None` sorts before every bucket, as the zero bucket does.
            if at < self.at[l] {
                self.below[l] += s;
            }
            self.settle(l);
        }
    }

    /// The quantile at level `l`, or `None` before any weight.
    pub fn get(&self, l: usize) -> Option<f64> {
        if self.total <= 0.0 || self.total.is_nan() {
            return None;
        }
        Some(match self.at[l] {
            None => 0.0,
            Some(k) => Self::representative(k),
        })
    }

    /// Back to no weight, at the same levels: valid, since `new` checked
    /// them and a restore holds a sketch to [`Self::has_shape`].
    pub fn reset(&mut self) {
        *self = Self::empty(std::mem::take(&mut self.levels));
    }
}

/// Exponentially weighted lag-`k` autocorrelation of a stream.
///
/// The variance is the EW mean of every value's squared deviation, at weight
/// `W`; the cross term the EW mean of each pair's product, at a weight `W_c`
/// of its own, since a value with no partner `k` back -- the first `k` of a
/// stream, and of each stretch after [`EwAutoCorr::clear_lags`] -- forms no
/// pair. Both weights decay on the clock, by [`EwAutoCorr::age`] on a step
/// with no value. Where every value has its partner, `W_c = W` and the two
/// share one recursion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwAutoCorr {
    lag: usize,
    /// Recent values, most recent last; length `lag + 1` once warm.
    buf: Vec<f64>,
    w: f64,
    mean: f64,
    /// EW second moments of (x_t, x_{t-lag}) around their shared mean.
    var: f64,
    cross: f64,
    /// What `mean` leaves out: the mean is a pair no step is rounded off
    /// ([`crate::comp`]; docs/PLAN.md task 101). A plain mean given one value
    /// row after row stopped short of it, fed `var` and `cross` the gap's
    /// square, and took their ratio to 1.
    #[serde(default)]
    mean_lo: f64,
    /// The weight behind `cross`: the pairs' (docs/PLAN.md task 146).
    #[serde(default)]
    w_pairs: f64,
}

impl EwAutoCorr {
    pub fn new(lag: usize) -> Result<Self, String> {
        if lag == 0 {
            return Err("EwAutoCorr: lag must be >= 1".into());
        }
        // The buffer holds `lag + 1` values and is sized here, so the lag
        // has the models' lag ceiling (review 2026-10-06, CD10).
        crate::ewlagcov::check_lag_ceiling("EwAutoCorr: lag", lag)?;
        Ok(Self {
            lag,
            buf: Vec::with_capacity(lag + 1),
            w: 0.0,
            mean: 0.0,
            var: 0.0,
            cross: 0.0,
            mean_lo: 0.0,
            w_pairs: 0.0,
        })
    }

    /// Whether a restored tracker is at `other`'s lag with a buffer that lag
    /// can have filled: what a saved diagnostic must hold to continue as
    /// this spec's (review 2026-09-18, B3).
    pub fn same_shape(&self, other: &Self) -> bool {
        self.lag == other.lag && self.buf.len() <= self.lag + 1
    }

    /// `None` until a lagged pair has been seen. `var > 0` is reached at the
    /// second distinct observation, but for `lag >= 2` no pair exists yet
    /// then, so the pairs' weight is the gate rather than the variance
    /// (review 2026-09-18, D1).
    pub fn get(&self) -> Option<f64> {
        (self.w_pairs > 0.0 && self.var > 0.0).then(|| (self.cross / self.var).clamp(-1.0, 1.0))
    }

    /// The clock moves on by a step whose decay is `lam`, with no value.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
        self.w_pairs *= lam;
    }

    /// The values behind this one are no longer adjacent to it -- a gap
    /// capped by `gap_cap`, or a session change -- so none of them is a
    /// partner for the next: the buffer goes, and the moments stay
    /// (docs/PLAN.md task 146, as `OnlineModel::clear_lags` does for a
    /// model's rings).
    pub fn clear_lags(&mut self) {
        self.buf.clear();
    }

    /// One observation with decay factor `lam`.
    ///
    /// A single mean and variance are used for both legs of the pair, which is
    /// the standard simplification for a stationary series and keeps the result
    /// in [−1, 1] by construction.
    pub fn update(&mut self, x: f64, lam: f64) {
        if !x.is_finite() {
            self.age(lam);
            return;
        }
        self.buf.push(x);
        if self.buf.len() > self.lag + 1 {
            self.buf.remove(0);
        }

        let w_new = lam * self.w + 1.0;
        let (a, b) = (lam * self.w / w_new, 1.0 / w_new);
        let d = crate::comp::dev(x, self.mean, self.mean_lo);
        self.var = a * self.var + a * b * d * d;
        if self.buf.len() == self.lag + 1 {
            // `a d` is `x` about the mean it joins; the partner is about the
            // mean before it, as the variance's Welford step has it.
            let wp_new = lam * self.w_pairs + 1.0;
            let (ap, bp) = (lam * self.w_pairs / wp_new, 1.0 / wp_new);
            let lagged = crate::comp::dev(self.buf[0], self.mean, self.mean_lo);
            self.cross = ap * self.cross + a * bp * d * lagged;
            self.w_pairs = wp_new;
        } else {
            self.w_pairs *= lam;
        }
        crate::comp::add(&mut self.mean, &mut self.mean_lo, b * d);
        self.w = w_new;
    }
}

/// Exponentially weighted evaluation metrics for one prediction slot
/// (docs/ENHANCEMENTS.md E22).
///
/// `eval.py` computes the same quantities in Polars over collected output,
/// which is right for analysis but needs the whole frame. This is the O(state)
/// version: it lives beside the model, so a long-running stream and the CLI can
/// report how the fit is doing without keeping the rows.
///
/// All three are exponentially weighted on the model's own clock. `hit_rate`
/// has two readings, chosen by the caller's `binary` flag at
/// [`SlotMetrics::update`] (docs/PLAN.md task 76):
///
/// ```text
/// ic       = corr(pred, y)
/// r2       = 1 − EW[(y − pred)²] / EW[(y − ȳ)²]
/// hit_rate = EW mean of 1{sign(pred) = sign(y)}, over rows where y ≠ 0
///            and pred ≠ 0                                               (binary = false)
///          = EW mean of 1{(pred > 0.5) = (y > 0.5)}, every row            (binary = true)
/// ```
///
/// A regression fit's `pred` and `y` share a sign convention, so agreement
/// on the sign is informative, and a row with `y = 0` or `pred = 0`
/// (neither up nor down) is excluded rather than scored either way, as
/// `po.eval` excludes it (docs/PLAN.md task 195, S6). A Poisson fit has no
/// sign to hit, and its `hit_rate` is `None` ([`HitTest::Undefined`]). A
/// classifier's `pred` is a
/// probability in `(0, 1)` and its `y` a 0/1 label: both are positive by
/// construction, so the sign test always agrees and reports 1.0 whatever the
/// fit does (docs/PLAN.md task 76 measured it at exactly that on a fit
/// trained on pure noise). Accuracy at the natural threshold is the binary
/// reading, and every row scores -- `y = 0` is one of the two classes, not
/// an excluded case, the way it is for a signed target.
///
/// `r2` and `ic` keep their definitions on a 0/1 target and are still worth
/// reading, under different names: `r2` is the Brier skill score against the
/// EW base rate, and `ic` is the point-biserial correlation between the
/// probability and the label. Neither is renamed -- the formula does not
/// change, only what it is usually called.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotMetrics {
    /// Joint moments of (pred, y).
    joint: crate::EwCov,
    /// EW mean squared error and its weight.
    mse: f64,
    /// EW hit rate and its weight. Rows with `y` or `pred` at the centre
    /// are excluded from the sign-agreement reading; every scored row counts
    /// toward the accuracy-at-threshold one (see the struct docs).
    hits: f64,
    hit_w: f64,
}

impl Default for SlotMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotMetrics {
    pub fn new() -> Self {
        Self {
            // A diagnostic, never windowed: no runs (docs/PLAN.md task 128).
            joint: crate::EwCov::new(2).without_runs(),
            mse: 0.0,
            hits: 0.0,
            hit_w: 0.0,
        }
    }

    /// EW count of scored rows.
    pub fn n_eff(&self) -> f64 {
        self.joint.n_eff()
    }

    /// Information coefficient: the correlation between prediction and target.
    pub fn ic(&self) -> Option<f64> {
        let d = self.joint.var(0).sqrt() * self.joint.var(1).sqrt();
        (d > 0.0).then(|| (self.joint.cov(0, 1) / d).clamp(-1.0, 1.0))
    }

    /// Out-of-sample R², against the EW mean of the target.
    ///
    /// Negative values are normal and meaningful: they say the model is doing
    /// worse than predicting the running mean.
    pub fn r2(&self) -> Option<f64> {
        let var_y = self.joint.var(1);
        (var_y > 0.0).then(|| 1.0 - self.mse / var_y)
    }

    /// Sign agreement (`binary = false`) or accuracy at a 0.5 threshold
    /// (`binary = true`) -- see the struct docs for which to read on which
    /// kind of fit. `binary` is a caller-declared fact about the model, not
    /// something read off the data, so the field's meaning does not depend
    /// on which rows a chunk happened to carry (docs/PLAN.md task 76).
    pub fn hit_rate(&self) -> Option<f64> {
        (self.hit_w > 0.0).then_some(self.hits)
    }

    /// Score one row. `lam` is the model's decay factor for this row.
    /// `binary` says `pred` is a probability and `y` a 0/1 label (the
    /// model's declared loss is `logistic`, not something sniffed from this
    /// row's value): the hit test in that case is accuracy at a 0.5
    /// threshold and every row scores, where the regression reading tests
    /// sign agreement and excludes `y == 0`.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64, binary: bool) {
        self.update_about(pred, y, lam, w, binary, 0.0);
    }

    /// [`Self::update`] with the regression hit test taken about `centre`
    /// rather than zero: agreement on which side of it `pred` and `y` fall,
    /// a `y` or a `pred` exactly there not scored. For a target that is a
    /// ratio, whose natural centre is 1 -- about zero, two positive numbers
    /// always agree, and the rate read 1.0 whatever the fit (review
    /// 2026-09-26, D3). `binary` ignores the centre. Centre 0 is
    /// [`Self::update`] to the bit.
    pub fn update_about(&mut self, pred: f64, y: f64, lam: f64, w: f64, binary: bool, centre: f64) {
        let hit = if binary {
            HitTest::Threshold
        } else {
            HitTest::About(centre)
        };
        self.update_with(pred, y, lam, w, hit);
    }

    /// Score one row under the hit test `hit` ([`HitTest`]).
    pub fn update_with(&mut self, pred: f64, y: f64, lam: f64, w: f64, hit: HitTest) {
        if !pred.is_finite() || !y.is_finite() || w <= 0.0 {
            // Age the estimates but do not score: a row with no prediction is
            // not evidence of a bad one. The means themselves are unchanged;
            // only the effective counts shrink.
            self.joint.decay(lam);
            self.hit_w *= lam;
            return;
        }
        self.joint.update(&[pred, y], lam, w);

        // Reuse the joint accumulator's own weights for the MSE, so every
        // metric here is averaged identically. After the update above,
        // `n_eff = lam·W_old + w`, so `(n_eff − w)/n_eff` is exactly the weight
        // EwCov gave the history and `w/n_eff` the weight it gave this row.
        let denom = self.joint.n_eff();
        if denom > 0.0 {
            let e = y - pred;
            self.mse = ((denom - w) * self.mse + w * e * e) / denom;
        }
        let scored = match hit {
            HitTest::Threshold => Some(f64::from((pred > 0.5) == (y > 0.5))),
            // A `y` or a `pred` exactly at the centre says neither up nor
            // down. `signum(+0.0)` is 1, so a prediction of exactly 0 was
            // "up", a hit on every rising row, where `po.eval`, whose
            // `sign(0)` is 0, scored it a miss (review round 4, YB8;
            // docs/PLAN.md task 195, S6).
            HitTest::About(centre) if y != centre && pred != centre => {
                Some(f64::from((pred - centre).signum() == (y - centre).signum()))
            }
            HitTest::About(_) | HitTest::Undefined => None,
        };
        match scored {
            Some(hit) => {
                let hw = lam * self.hit_w + w;
                self.hits = (lam * self.hit_w * self.hits + w * hit) / hw;
                self.hit_w = hw;
            }
            // A row with no sign to hit is not scored -- but it still
            // happened, so it must age the hit weight exactly as a skipped
            // row does above, or the hit rate runs on a different clock from
            // `ic` and `r2` on any stream with exact zeros (review
            // 2026-09-18, S1). Under `Undefined` no row is scored, and
            // `hit_rate` is `None` throughout.
            None => self.hit_w *= lam,
        }
    }
}

/// Which hit test a slot's `hit_rate` reads ([`SlotMetrics::update_with`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HitTest {
    /// Sign agreement about a centre: 0 for a signed target, 1 for a ratio.
    /// A row whose target or prediction sits exactly at the centre is not
    /// scored.
    About(f64),
    /// Accuracy at a 0.5 threshold, every row scored: a probability against
    /// a 0/1 label (the logistic losses).
    Threshold,
    /// None: a fit with no sign to hit -- a Poisson rate, positive by
    /// construction against a count that is never negative, where the sign
    /// test about zero agreed on every row and read 1.0 whatever the fit
    /// (review round 4, CC5; docs/PLAN.md task 195, S5). `hit_rate` is
    /// `None`, as `po.eval` nulls a metric that is not defined.
    Undefined,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The exponentially weighted quantile by its definition, from every
    /// value and its weight as it now stands: the smallest value whose
    /// cumulative weight, ascending, reaches `p` of the total.
    fn ew_quantile(seen: &[(f64, f64)], p: f64) -> f64 {
        let mut v = seen.to_vec();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let total: f64 = v.iter().map(|x| x.1).sum();
        let mut cum = 0.0;
        for (x, w) in &v {
            cum += w;
            if cum >= p * total {
                return *x;
            }
        }
        v.last().unwrap().0
    }

    /// Task 146: every level within `α` of the definition, row by row, on
    /// irregular steps and weights, with zeros, through a tenfold fall in
    /// scale -- which a sketch that never forgot (P²) missed by ninefold.
    #[test]
    fn an_ew_quantile_is_within_alpha_of_the_definition() {
        let levels = [0.01, 0.1, 0.5, 0.9, 0.99];
        let mut q = EwQuantile::new(&levels).unwrap();
        let h = 15.0;
        let mut s = 41u64;
        let mut seen: Vec<(f64, f64)> = Vec::new();
        for i in 0..3000 {
            let lam = (-(2.0 * lcg(&mut s) / h)).exp2();
            q.age(lam);
            seen.iter_mut().for_each(|e| e.1 *= lam);
            let scale = if i < 1500 { 1.0 } else { 0.1 };
            let v = if i % 37 == 0 {
                0.0
            } else {
                -scale * lcg(&mut s).max(1e-300).ln()
            };
            let w = 0.5 + lcg(&mut s);
            q.add(v, w);
            seen.push((v, w));
            if i % 7 != 0 {
                continue;
            }
            for (l, &p) in levels.iter().enumerate() {
                let (got, want) = (q.get(l).unwrap(), ew_quantile(&seen, p));
                assert!(
                    (got - want).abs() <= EW_QUANTILE_ALPHA * want * (1.0 + 1e-9),
                    "row {i}, p = {p}: {got} against {want}"
                );
            }
        }
        assert_eq!(q.get(0), Some(0.0), "the zeros hold the lowest level");
    }

    /// A short half-life folds the decay in every 32 rows and drops the end
    /// buckets nothing reaches; the answer holds through it, and the
    /// buckets stay within the range the recent values span.
    #[test]
    fn rescaling_and_pruning_keep_the_answer_and_bound_the_memory() {
        let mut q = EwQuantile::new(&[0.5, 0.95]).unwrap();
        let lam = 0.25; // two halvings a row
        let mut s = 43u64;
        let mut seen: Vec<(f64, f64)> = Vec::new();
        for i in 0..20_000 {
            q.age(lam);
            seen.iter_mut().for_each(|e| e.1 *= lam);
            // A huge value early on, then values over 2^0 .. 2^20.
            let v = if i == 10 {
                1e200
            } else {
                (20.0 * lcg(&mut s)).exp2()
            };
            q.add(v, 1.0);
            seen.push((v, 1.0));
        }
        for (l, p) in [0.5, 0.95].into_iter().enumerate() {
            let (got, want) = (q.get(l).unwrap(), ew_quantile(&seen, p));
            assert!((got - want).abs() <= EW_QUANTILE_ALPHA * want * (1.0 + 1e-9));
        }
        assert!(q.buckets.len() <= 21 * 64, "{} buckets", q.buckets.len());
    }

    #[test]
    fn nothing_is_reported_without_weight_and_a_full_decay_forgets_all() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        assert_eq!(q.get(0), None);
        for (v, w) in [
            (1.0, 0.0),
            (f64::NAN, 1.0),
            (-1.0, 1.0),
            (f64::INFINITY, 1.0),
        ] {
            q.add(v, w);
            assert_eq!(q.get(0), None, "({v}, {w}) is not seen");
        }
        q.add(3.0, 1.0);
        assert!(q.get(0).is_some());
        q.age(0.0);
        assert_eq!(q.get(0), None, "a decay of 0 forgets everything");
        q.add(5.0, 2.0);
        let got = q.get(0).unwrap();
        assert!((got - 5.0).abs() <= EW_QUANTILE_ALPHA * 5.0);
    }

    /// The bucket is read off the float's bits: a power of two starts one,
    /// and every value's representative is within `α` of it, from the
    /// smallest normal float to the largest.
    #[test]
    fn the_bucket_is_exact_on_the_bits_and_within_alpha() {
        for k in [-1022, -1, 0, 1, 10, 1023] {
            assert_eq!(EwQuantile::index(2f64.powi(k)), 64 * k, "2^{k}");
        }
        assert_eq!(EwQuantile::index(1.5), 32);
        let mut s = 47u64;
        let check = |v: f64| {
            let rep = EwQuantile::representative(EwQuantile::index(v));
            assert!(
                (rep - v).abs() <= EW_QUANTILE_ALPHA * v * (1.0 + 1e-12),
                "{v}: {rep}"
            );
        };
        check(f64::MIN_POSITIVE);
        check(f64::MAX);
        for _ in 0..100_000 {
            check((2000.0 * lcg(&mut s) - 1000.0).exp2() * (1.0 + lcg(&mut s)));
        }
    }

    #[test]
    fn the_levels_share_one_sketch_and_stay_ordered() {
        let levels = [0.05, 0.25, 0.5, 0.75, 0.95];
        let mut q = EwQuantile::new(&levels).unwrap();
        let mut s = 53u64;
        for _ in 0..5000 {
            q.age(0.99);
            q.add(lcg(&mut s) * 10.0, 1.0);
            let got: Vec<f64> = (0..levels.len()).map(|l| q.get(l).unwrap()).collect();
            assert!(got.windows(2).all(|w| w[0] <= w[1]), "{got:?}");
        }
    }

    #[test]
    fn a_saved_sketch_goes_on_as_the_live_one() {
        let mut live = EwQuantile::new(&[0.5, 0.9]).unwrap();
        let mut s = 59u64;
        for _ in 0..500 {
            live.age(0.9);
            live.add(lcg(&mut s), 1.0);
        }
        let bytes = rmp_serde::to_vec(&live).unwrap();
        let mut back: EwQuantile = rmp_serde::from_slice(&bytes).unwrap();
        for _ in 0..500 {
            let v = lcg(&mut s);
            live.age(0.9);
            back.age(0.9);
            live.add(v, 1.0);
            back.add(v, 1.0);
        }
        assert_eq!(live, back);
    }

    #[test]
    fn levels_outside_the_open_unit_interval_are_refused() {
        for bad in [0.0, 1.0, -0.5, f64::NAN] {
            assert!(EwQuantile::new(&[0.5, bad]).is_err(), "{bad}");
        }
    }

    #[test]
    fn autocorr_is_near_zero_for_noise() {
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 11u64;
        for _ in 0..20000 {
            ac.update(lcg(&mut s) - 0.5, 0.999);
        }
        assert!(ac.get().unwrap().abs() < 0.1, "got {:?}", ac.get());
    }

    #[test]
    fn autocorr_detects_a_persistent_series() {
        // An AR(1) with phi = 0.8 must show a strong positive lag-1.
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 13u64;
        let mut prev = 0.0;
        for _ in 0..40000 {
            prev = 0.8 * prev + (lcg(&mut s) - 0.5);
            ac.update(prev, 0.9995);
        }
        let got = ac.get().unwrap();
        assert!(
            got > 0.6,
            "AR(1) phi=0.8 should show strong lag-1, got {got}"
        );
    }

    /// A series that stops moving: its mean is a pair ([`crate::comp`];
    /// docs/PLAN.md task 101), so `var` and `cross` decay with their history
    /// and their ratio stays where the hold's first half-lives left it -- the
    /// held value's offset from the mean reads as persistence for those, in
    /// exact arithmetic too. A plain mean stopped short fed both the gap's
    /// square, and the ratio went to 1. Read at 5 half-lives, before the
    /// stall at every level here, and at 150.
    #[test]
    fn a_held_series_keeps_its_autocorrelation() {
        for level in [0.5, 1e8, -1e8, 1e12] {
            let mut ac = EwAutoCorr::new(1).unwrap();
            let lam = 0.5f64.powf(1.0 / 20.0);
            let (mut s, mut u, mut early) = (17u64, 0.0, f64::NAN);
            for i in 0..3300 {
                u = 0.5 * u + (lcg(&mut s) - 0.5);
                ac.update(if i < 300 { level + u } else { level + 0.37 }, lam);
                if i == 300 + 5 * 20 {
                    early = ac.get().unwrap();
                }
            }
            let end = ac.get().unwrap();
            assert!(
                (end - early).abs() < 0.01,
                "level {level}: {early} after 5 half_lives, {end} after 150"
            );
        }
    }

    #[test]
    fn autocorr_detects_alternation() {
        let mut ac = EwAutoCorr::new(1).unwrap();
        for i in 0..20000 {
            ac.update(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.999);
        }
        assert!(ac.get().unwrap() < -0.9, "got {:?}", ac.get());
    }

    #[test]
    fn autocorr_rejects_lag_zero() {
        assert!(EwAutoCorr::new(0).is_err());
    }

    /// The tracker's buffer holds `lag + 1` residuals and is sized when it
    /// is built, so the lag has the ceiling the models' lag rings have,
    /// 2^20, refused by name with the value: `2^62` asked for a buffer past
    /// any capacity, and `usize::MAX` overflowed `lag + 1` (review
    /// 2026-10-06, CD10).
    #[test]
    fn autocorr_refuses_a_lag_past_the_ceiling() {
        for lag in [(1usize << 20) + 1, 1 << 62, usize::MAX] {
            let Err(e) = EwAutoCorr::new(lag) else {
                panic!("accepted")
            };
            assert!(
                e.contains("lag must be at most 1048576") && e.contains(&format!("got {lag}")),
                "{e}"
            );
        }
        assert!(EwAutoCorr::new(1 << 20).is_ok());
    }

    #[test]
    fn autocorr_is_none_until_a_pair_of_the_right_lag_exists() {
        // At lag 3 the first pair is between observation 1 and observation 4,
        // so `get` is None through the first three distinct values (where
        // `var > 0` alone would have returned Some(0.0)) and Some only from
        // the fourth (review 2026-09-18, D1).
        let mut ac = EwAutoCorr::new(3).unwrap();
        for (i, x) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            ac.update(x, 1.0);
            assert!(ac.get().is_none(), "some after {} obs", i + 1);
        }
        ac.update(4.0, 1.0);
        assert!(ac.get().is_some(), "none after the fourth observation");
    }

    /// Task 146: a break clears the buffer and keeps the moments, so the
    /// estimate stands, the next value pairs with nothing, and the one after
    /// pairs with it -- not with anything from before the break.
    #[test]
    fn clearing_the_lags_keeps_the_estimate_and_pairs_nothing_across_it() {
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 61u64;
        let mut prev = 0.0;
        for _ in 0..500 {
            prev = 0.8 * prev + (lcg(&mut s) - 0.5);
            ac.update(prev, 0.99);
        }
        let before = ac.get().unwrap();
        ac.clear_lags();
        assert_eq!(ac.get(), Some(before), "the estimate stands");
        let (cross, w_pairs) = (ac.cross, ac.w_pairs);
        ac.update(5.0, 0.99);
        assert_eq!(ac.cross, cross, "no pair across the break");
        assert_eq!(ac.w_pairs, 0.99 * w_pairs, "the pairs' weight only ages");
        ac.update(-5.0, 0.99);
        assert_eq!(
            ac.buf,
            vec![5.0, -5.0],
            "the pair is the two since the break"
        );
        assert_eq!(ac.w_pairs, 0.99 * (0.99 * w_pairs) + 1.0);
    }

    /// A step with no value ages both weights, so the clock reaches the
    /// tracker on every row: two steps' decay before a value is one step's
    /// decay of their product.
    #[test]
    fn a_step_with_no_value_ages_both_weights() {
        let mut s = 67u64;
        let xs: Vec<f64> = (0..300).map(|_| lcg(&mut s)).collect();
        let (mut aged, mut once) = (EwAutoCorr::new(2).unwrap(), EwAutoCorr::new(2).unwrap());
        for (i, &x) in xs.iter().enumerate() {
            if i % 3 == 0 {
                aged.age(0.9);
                aged.update(x, 0.8);
                once.update(x, 0.9 * 0.8);
            } else {
                aged.update(x, 0.95);
                once.update(x, 0.95);
            }
        }
        assert!((aged.get().unwrap() - once.get().unwrap()).abs() < 1e-12);
        assert!((aged.w - once.w).abs() < 1e-12 * once.w);
        assert!((aged.w_pairs - once.w_pairs).abs() < 1e-12 * once.w_pairs);
        // A non-finite value is a step with no value.
        let mut nan = once.clone();
        nan.update(f64::NAN, 0.5);
        once.age(0.5);
        assert_eq!(nan, once);
    }

    #[test]
    fn autocorr_of_a_constant_series_is_none() {
        // Every pair exists, but a constant has no variance to divide by.
        let mut ac = EwAutoCorr::new(1).unwrap();
        for _ in 0..10 {
            ac.update(5.0, 0.9);
        }
        assert_eq!(ac.get(), None);
    }

    #[test]
    fn same_shape_takes_a_full_buffer_and_refuses_an_overfull_one() {
        let mut full = EwAutoCorr::new(2).unwrap();
        for x in [1.0, 2.0, 3.0, 4.0] {
            full.update(x, 0.9);
        }
        assert_eq!(full.buf.len(), 3);
        let fresh = EwAutoCorr::new(2).unwrap();
        assert!(full.same_shape(&fresh) && fresh.same_shape(&full));
        let mut over = full.clone();
        over.buf.push(5.0);
        assert!(
            !over.same_shape(&fresh),
            "a buffer lag 2 cannot have filled"
        );
        assert!(
            !full.same_shape(&EwAutoCorr::new(3).unwrap()),
            "another lag"
        );
    }

    #[test]
    fn autocorr_forgets_an_old_regime() {
        // phi = +0.9, then -0.9: a half-life of ~69 rows must have forgotten
        // the first 3000 rows by the end of the next 3000. A stationary
        // series cannot tell decay from none; this can.
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 17u64;
        let mut prev = 0.0;
        for t in 0..6000 {
            let phi = if t < 3000 { 0.9 } else { -0.9 };
            prev = phi * prev + (lcg(&mut s) - 0.5);
            ac.update(prev, 0.99);
        }
        let got = ac.get().unwrap();
        assert!(got < -0.8, "the new regime's -0.9 should show, got {got}");
    }

    #[test]
    fn autocorr_has_no_cross_moment_before_its_first_pair() {
        // At lag 3 the first three observations have no partner: the
        // co-moment is a sum over pairs, and there are none yet.
        let mut ac = EwAutoCorr::new(3).unwrap();
        for x in [1.0, 5.0, 2.0] {
            ac.update(x, 0.9);
            assert_eq!(ac.cross, 0.0);
        }
        ac.update(7.0, 0.9);
        assert_ne!(ac.cross, 0.0, "the first pair, (7, 1), is a co-moment");
    }

    #[test]
    fn autocorr_is_unchanged_by_a_shift_or_a_scale_of_the_series() {
        // A correlation, so the series' level and units cannot reach it; at
        // lag 2, so the pairs that wait for a partner are in play too.
        let mut s = 23u64;
        let mut prev = 0.0;
        let xs: Vec<f64> = (0..3000)
            .map(|_| {
                prev = 0.7 * prev + (lcg(&mut s) - 0.5);
                prev
            })
            .collect();
        let at = |f: &dyn Fn(f64) -> f64| {
            let mut ac = EwAutoCorr::new(2).unwrap();
            for &x in &xs {
                ac.update(f(x), 0.995);
            }
            ac.get().unwrap()
        };
        let base = at(&|x| x);
        assert!(
            base > 0.3,
            "an AR(1) at 0.7 has a lag-2 near 0.49, got {base}"
        );
        for (name, got) in [("shift", at(&|x| x + 100.0)), ("scale", at(&|x| 3.0 * x))] {
            assert!((got - base).abs() < 1e-9, "{name}: {got} vs {base}");
        }
    }

    /// The levels come back as given, in order, and `reset` returns the
    /// sketch to a new one at the same levels (task 158).
    #[test]
    fn a_sketch_reports_its_levels_and_resets_to_new() {
        let levels = [0.25, 0.5, 0.9];
        let mut q = EwQuantile::new(&levels).unwrap();
        assert_eq!(q.levels(), &levels);
        for i in 0..50 {
            q.age(0.9);
            q.add(f64::from(i), 1.0);
        }
        assert_ne!(q, EwQuantile::new(&levels).unwrap());
        q.reset();
        assert_eq!(q, EwQuantile::new(&levels).unwrap());
        assert_eq!(q.get(0), None);
    }

    /// The shape a restored sketch must hold (review 2026-10-06, CB2), one
    /// part broken at a time: a sketch this module made always holds it,
    /// at every level, across the zero bucket, the decay's folds and the
    /// extreme indices, and each break is caught alone.
    #[test]
    fn a_sketch_holds_its_shape_and_each_break_is_caught() {
        let mut q = EwQuantile::new(&[0.1, 0.5, 0.99]).unwrap();
        assert!(q.has_shape(), "a new sketch");
        // At 0.8 a step the decay is folded in about every 200 rows.
        for i in 0..400 {
            q.age(0.8);
            let v = match i % 5 {
                0 => 0.0,
                1 => f64::MIN_POSITIVE,
                2 => f64::MAX,
                _ => f64::from(i),
            };
            q.add(v, 1.0 + f64::from(i % 3));
            assert!(q.has_shape(), "row {i}: {q:?}");
        }
        type Break<'a> = (&'a str, &'a dyn Fn(&mut EwQuantile));
        let breaks: [Break; 11] = [
            ("a pointer short", &|q| {
                q.at.pop();
            }),
            ("a weight below short", &|q| {
                q.below.pop();
            }),
            ("a level of 1", &|q| q.levels[0] = 1.0),
            ("a pointer below the buckets", &|q| q.at[0] = Some(q.lo - 1)),
            ("a pointer past the buckets", &|q| {
                q.at[0] = Some(q.lo + q.buckets.len() as i32);
            }),
            ("buckets below the indices", &|q| q.lo = i32::MIN),
            ("a negative bucket", &|q| q.buckets[0] = -1.0),
            ("an infinite zero bucket", &|q| q.zero = f64::INFINITY),
            ("a NaN total", &|q| q.total = f64::NAN),
            ("a NaN weight below", &|q| q.below[0] = f64::NAN),
            ("a scale of 0", &|q| q.scale = 0.0),
        ];
        for (what, f) in breaks {
            let mut broken = q.clone();
            f(&mut broken);
            assert!(!broken.has_shape(), "{what}");
        }
        // A scale above 1 is no decay a step can leave, nor is a NaN one.
        for scale in [1.5, f64::NAN] {
            let mut broken = q.clone();
            broken.scale = scale;
            assert!(!broken.has_shape(), "a scale of {scale}");
        }
        // And `reset` keeps the levels without checking them again.
        q.reset();
        assert!(q.has_shape() && q.get(2).is_none());
    }

    /// "Anything else is not seen": a weight of 0 leaves the sketch exactly
    /// as it was, buckets and all, not a bucket of weight 0 (task 158).
    #[test]
    fn a_value_at_weight_zero_leaves_the_sketch_untouched() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        q.add(3.0, 1.0);
        let before = q.clone();
        q.add(1e10, 0.0);
        q.add(1e-10, 0.0);
        assert_eq!(q, before);
    }

    /// The smallest normal float is a value, not a zero: it lands in its
    /// bucket and is reported within `α`; anything below it counts as 0
    /// (task 158).
    #[test]
    fn the_smallest_normal_float_is_a_value_and_below_it_a_zero() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        q.add(f64::MIN_POSITIVE, 1.0);
        let got = q.get(0).unwrap();
        assert!(
            (got - f64::MIN_POSITIVE).abs() <= EW_QUANTILE_ALPHA * f64::MIN_POSITIVE,
            "{got:e}"
        );
        let mut z = EwQuantile::new(&[0.5]).unwrap();
        z.add(f64::MIN_POSITIVE / 2.0, 1.0);
        assert_eq!(z.get(0), Some(0.0));
    }

    /// A decay of 0 leaves the sketch as a new one: no bucket kept, of
    /// weight 0 or otherwise (task 158).
    #[test]
    fn a_full_decay_leaves_a_new_sketch() {
        let levels = [0.1, 0.9];
        let mut q = EwQuantile::new(&levels).unwrap();
        for v in [0.0, 1.0, 5.0, 1e9] {
            q.add(v, 1.0);
        }
        q.age(0.0);
        assert_eq!(q, EwQuantile::new(&levels).unwrap());
    }

    /// The decay is folded into the buckets once it passes `2^-64`, not at
    /// it: after 64 halvings the buckets still hold the weights as added and
    /// the factor beside them is `2^-64`; the 65th folds it in (task 158).
    #[test]
    fn the_decay_is_folded_in_once_it_passes_two_to_the_minus_64() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        q.add(1.0, 1.0);
        for _ in 0..64 {
            q.age(0.5);
        }
        assert_eq!(q.scale, RENORM);
        assert_eq!(q.buckets, vec![1.0]);
        q.age(0.5);
        assert_eq!(q.scale, 1.0);
        assert_eq!(q.buckets, vec![0.5 * RENORM]);
    }

    /// An end bucket at exactly the prune share of the total is dropped, as
    /// `rescale`'s `b > floor` has it: the bucket of 4 weighs `2^-65` after
    /// the fold, which is `PRUNE · total` to the bit. The docs once said
    /// "lighter than" and "under"; task 158 settled them on the code's side,
    /// which moves no number.
    #[test]
    fn an_end_bucket_at_exactly_the_prune_share_is_dropped() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        q.add(1.0, 999_999_999_999.0);
        q.add(4.0, 1.0);
        assert!(q.buckets.len() > 1, "two values, many buckets apart");
        q.age(2f64.powi(-65));
        let total = q.zero + q.buckets.iter().sum::<f64>();
        assert_eq!(q.buckets.len(), 1, "the bucket of 4 is dropped");
        // The one left is the bucket of 1; the dropped one weighed the floor.
        assert_eq!(2f64.powi(-65), PRUNE * (total + 2f64.powi(-65)));
    }

    /// At a fold, the ends that weigh at or under `1e-12` of the total go and
    /// every bucket between the heaviest ends stays: two values from the
    /// first rows, far below and far above the rest, weigh `2^-64` of it at
    /// the fold and are dropped, while the buckets of 1 and 2 (indices 0 and
    /// 64) are kept whole. At weights of `1e8`, so the share is of the
    /// total's size, not of 1 (task 158).
    #[test]
    fn a_fold_drops_the_light_ends_and_keeps_the_rest() {
        let mut q = EwQuantile::new(&[0.5]).unwrap();
        let w = 1e8;
        q.add(1e-100, w);
        q.add(1e100, w);
        let mut folded = false;
        for i in 0..70 {
            q.age(0.5);
            q.add(if i % 2 == 0 { 1.0 } else { 2.0 }, w);
            if q.scale == 1.0 {
                folded = true;
                break;
            }
        }
        assert!(folded, "the decay was folded in");
        assert_eq!(q.lo, 0, "the bucket of 1 is the lowest kept");
        assert_eq!(q.buckets.len(), 65, "up to the bucket of 2");
        assert!(q.buckets[0] > 0.0 && q.buckets[64] > 0.0);
    }

    /// The top level at `p` just under 1: `p · total` is the total less an
    /// ulp, while the walk's own sum to the last bucket rounds below it. The
    /// pointer stays on the last bucket rather than stepping past the end,
    /// and reports it (task 158).
    #[test]
    fn a_level_next_to_one_stays_on_the_last_bucket() {
        let p = 1.0 - f64::EPSILON / 2.0;
        let mut q = EwQuantile::new(&[p]).unwrap();
        q.add(1.0, 1.0);
        // `1 + s` rounds up a whole ulp each time; their own sum does not.
        let s = 0.625 * f64::EPSILON;
        let mut past = 0;
        for _ in 0..4 {
            q.add(2.0, s);
            let last = q.lo + q.buckets.len() as i32 - 1;
            let walked = q.below[0] + q.weight(q.at[0]);
            past += usize::from(q.at[0] == Some(last) && walked < p * q.total);
        }
        assert!(
            past > 0,
            "the walk's sum fell under the target at the last bucket"
        );
        let got = q.get(0).unwrap();
        assert!((got - 2.0).abs() <= EW_QUANTILE_ALPHA * 2.0, "{got}");
    }
}

#[cfg(test)]
mod metric_tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn a_perfect_prediction_scores_perfectly() {
        let mut m = SlotMetrics::new();
        let mut s = 3u64;
        for _ in 0..5000 {
            let y = lcg(&mut s);
            m.update(y, y, 1.0, 1.0, false);
        }
        assert!((m.ic().unwrap() - 1.0).abs() < 1e-9);
        assert!((m.r2().unwrap() - 1.0).abs() < 1e-9);
        assert!((m.hit_rate().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_uninformative_prediction_scores_at_chance() {
        let mut m = SlotMetrics::new();
        let mut s = 5u64;
        for _ in 0..40000 {
            m.update(lcg(&mut s), lcg(&mut s), 1.0, 1.0, false);
        }
        assert!(m.ic().unwrap().abs() < 0.05, "ic {:?}", m.ic());
        assert!(m.r2().unwrap() < 0.05, "r2 {:?}", m.r2());
        assert!(
            (m.hit_rate().unwrap() - 0.5).abs() < 0.05,
            "hit {:?}",
            m.hit_rate()
        );
    }

    #[test]
    fn predicting_the_mean_scores_zero_r2() {
        let mut m = SlotMetrics::new();
        let mut s = 7u64;
        for _ in 0..40000 {
            m.update(0.0, lcg(&mut s), 1.0, 1.0, false);
        }
        assert!(m.r2().unwrap().abs() < 0.05, "r2 {:?}", m.r2());
    }

    #[test]
    fn a_worse_than_mean_prediction_scores_negative_r2() {
        let mut m = SlotMetrics::new();
        let mut s = 11u64;
        for _ in 0..20000 {
            let y = lcg(&mut s);
            m.update(-2.0 * y, y, 1.0, 1.0, false);
        }
        assert!(m.r2().unwrap() < -1.0, "r2 {:?}", m.r2());
    }

    #[test]
    fn a_sign_flip_shows_up_as_negative_ic_and_low_hit_rate() {
        let mut m = SlotMetrics::new();
        let mut s = 13u64;
        for _ in 0..20000 {
            let y = lcg(&mut s);
            m.update(-y, y, 1.0, 1.0, false);
        }
        assert!((m.ic().unwrap() + 1.0).abs() < 1e-9);
        assert!(m.hit_rate().unwrap() < 1e-9);
    }

    #[test]
    fn decay_lets_it_forget_an_old_regime() {
        let mut m = SlotMetrics::new();
        let mut s = 17u64;
        let lam = 0.99;
        for _ in 0..3000 {
            let y = lcg(&mut s);
            m.update(-y, y, lam, 1.0, false); // wrong sign
        }
        assert!(m.hit_rate().unwrap() < 0.1);
        for _ in 0..3000 {
            let y = lcg(&mut s);
            m.update(y, y, lam, 1.0, false); // now right
        }
        assert!(
            m.hit_rate().unwrap() > 0.9,
            "did not forget: {:?}",
            m.hit_rate()
        );
    }

    #[test]
    fn an_excluded_zero_target_still_ages_the_hit_weight() {
        // A finite `y == 0` under a non-binary loss is not a hit either way,
        // but it must age `hit_w` like any other row: a run of them leaves the
        // hit rate hostage to evidence from before them, so the next real row
        // barely moves it (review 2026-09-18, S1). Build a near-perfect hit
        // rate, run a long stretch of excluded zeros, then miss once; with the
        // ageing the miss dominates a nearly weightless history.
        let lam = 0.5;
        let mut m = SlotMetrics::new();
        for _ in 0..20 {
            m.update(1.0, 1.0, lam, 1.0, false); // a hit
        }
        assert!(m.hit_rate().unwrap() > 0.99);
        for _ in 0..40 {
            m.update(1.0, 0.0, lam, 1.0, false); // excluded: y == 0
        }
        m.update(-1.0, 1.0, lam, 1.0, false); // a miss
        assert!(
            m.hit_rate().unwrap() < 0.01,
            "the zeros did not age the hit weight: {:?}",
            m.hit_rate()
        );
    }

    #[test]
    fn unscored_rows_are_ignored_not_counted_as_zero() {
        let mut m = SlotMetrics::new();
        let mut s = 19u64;
        for _ in 0..2000 {
            let y = lcg(&mut s);
            m.update(y, y, 1.0, 1.0, false);
            m.update(f64::NAN, y, 1.0, 1.0, false); // no prediction yet
        }
        assert!((m.ic().unwrap() - 1.0).abs() < 1e-6, "ic {:?}", m.ic());
    }

    #[test]
    fn a_row_is_unscored_if_the_prediction_or_the_target_or_the_weight_is_missing() {
        // Three independent reasons to skip, each of which must skip on its
        // own: an OR here, not an AND.
        for (pred, y, w) in [
            (f64::NAN, 1.0, 1.0),
            (f64::INFINITY, 1.0, 1.0),
            (1.0, f64::NAN, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, -1.0),
        ] {
            let mut m = SlotMetrics::new();
            let mut s = 31u64;
            for _ in 0..500 {
                let v = lcg(&mut s);
                m.update(v, v, 1.0, 1.0, false);
            }
            let (ic, r2, hr) = (m.ic(), m.r2(), m.hit_rate());
            m.update(pred, y, 1.0, w, false);
            assert_eq!(m.ic(), ic, "({pred}, {y}, {w}) must not score");
            assert_eq!(m.r2(), r2);
            assert_eq!(m.hit_rate(), hr);
        }

        // A skipped row still ages the estimates: the effective weight shrinks
        // even though the means do not move.
        let mut m = SlotMetrics::new();
        let mut s = 37u64;
        for _ in 0..500 {
            let v = lcg(&mut s);
            m.update(v, v, 1.0, 1.0, false);
        }
        let before = m.hit_w;
        m.update(f64::NAN, 1.0, 0.5, 1.0, false);
        assert!(
            (m.hit_w - before * 0.5).abs() < 1e-12,
            "the weight must decay"
        );
    }

    #[test]
    fn nothing_is_reported_before_any_row() {
        let m = SlotMetrics::new();
        assert!(m.ic().is_none() && m.r2().is_none() && m.hit_rate().is_none());
    }

    #[test]
    fn n_eff_is_the_ew_count_of_scored_rows() {
        // By its definition: each scored row adds its weight, and every row,
        // scored or not, decays what came before.
        let mut m = SlotMetrics::new();
        let mut want = 0.0;
        let mut s = 31u64;
        for t in 0..200 {
            let (lam, w) = (0.95, 0.5 + lcg(&mut s).abs());
            let pred = if t % 7 == 3 { f64::NAN } else { lcg(&mut s) };
            let scored = pred.is_finite();
            m.update(pred, lcg(&mut s), lam, w, false);
            want = lam * want + if scored { w } else { 0.0 };
            assert!(
                (m.n_eff() - want).abs() < 1e-12 * want,
                "t={t}: {} vs {want}",
                m.n_eff()
            );
        }
    }

    #[test]
    fn the_binary_threshold_is_strict_at_one_half() {
        // `(pred > 0.5) == (y > 0.5)`, as the struct docs state it: exactly
        // 0.5 is the lower class, for the prediction and the label alike.
        let hit = |pred: f64, y: f64| {
            let mut m = SlotMetrics::new();
            m.update(pred, y, 1.0, 1.0, true);
            m.hit_rate().unwrap()
        };
        assert_eq!(hit(0.5, 0.0), 1.0, "0.5 predicts the lower class");
        assert_eq!(hit(0.5, 1.0), 0.0);
        assert_eq!(hit(0.9, 0.5), 0.0, "a label of 0.5 is the lower class");
        assert_eq!(hit(0.1, 0.5), 1.0);
    }

    // `binary = true`: the classifier reading (docs/PLAN.md task 76). Every
    // regression test above keeps its `binary = false` call unchanged, so the
    // sign-agreement path is unaffected to the bit.

    #[test]
    fn a_probability_that_knows_nothing_does_not_score_a_perfect_hit_rate() {
        // The defect task 76 found: `pred.signum() == y.signum()` on a
        // probability against a 0/1 label always agreed, because both are
        // positive by construction. `y == 0.0` rows are no longer excluded
        // either -- 0 is one of the two classes, not the sign this fit has no
        // opinion on.
        let mut m = SlotMetrics::new();
        let mut s = 41u64;
        for _ in 0..40000 {
            // A coin flip label and a constant, uninformative "probability".
            let y = f64::from(lcg(&mut s) > 0.0);
            m.update(0.5, y, 1.0, 1.0, true);
        }
        // Accuracy at threshold 0.5 with pred == 0.5 exactly: `(0.5 > 0.5)`
        // is false, so every row calls it class 0, right half the time.
        assert!(
            (m.hit_rate().unwrap() - 0.5).abs() < 0.05,
            "hit {:?}",
            m.hit_rate()
        );
    }

    /// About a centre, the hit is the side of it both fall on, a target on
    /// the centre ages the weight without scoring, and centre 0 is `update`
    /// to the bit (review 2026-09-26, D3).
    #[test]
    fn the_hit_test_is_about_the_centre() {
        let mut m = SlotMetrics::new();
        m.update_about(1.2, 1.1, 0.9, 1.0, false, 1.0);
        assert_eq!(m.hit_rate(), Some(1.0), "both above 1");
        m.update_about(0.9, 1.1, 0.9, 1.0, false, 1.0);
        assert!(
            (m.hit_rate().unwrap() - 0.9 / 1.9).abs() < 1e-15,
            "{:?}",
            m.hit_rate()
        );
        let before = m.clone();
        m.update_about(0.9, 1.0, 0.5, 1.0, false, 1.0);
        assert_eq!(m.hit_rate(), before.hit_rate(), "on the centre: not scored");
        assert_eq!(m.hit_w, 0.5 * before.hit_w, "but aged");
        // Both below the centre agree too (task 158).
        let mut below = SlotMetrics::new();
        below.update_about(0.8, 0.9, 0.9, 1.0, false, 1.0);
        assert_eq!(below.hit_rate(), Some(1.0), "both below 1");
        let (mut a, mut b) = (SlotMetrics::new(), SlotMetrics::new());
        let mut s = 5u64;
        for i in 0..40 {
            let (p, y) = (lcg(&mut s) - 0.5, lcg(&mut s) - 0.5);
            let w = if i % 7 == 3 { 0.0 } else { 1.0 };
            a.update(p, y, 0.97, w, false);
            b.update_about(p, y, 0.97, w, false, 0.0);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn binary_hit_rate_is_accuracy_at_one_half() {
        let mut m = SlotMetrics::new();
        let mut s = 43u64;
        let mut correct = 0.0;
        let n = 20000;
        for _ in 0..n {
            let y = f64::from(lcg(&mut s) > 0.0);
            // A probability correlated with the label but not equal to it,
            // so both classes see some wrong calls -- a replica of the
            // rounded accuracy is checked against `hit_rate` directly.
            let p = if y > 0.5 {
                0.5 + 0.3 * lcg(&mut s).abs()
            } else {
                0.5 - 0.3 * lcg(&mut s).abs()
            };
            correct += f64::from((p > 0.5) == (y > 0.5));
            m.update(p, y, 1.0, 1.0, true);
        }
        assert!(
            (m.hit_rate().unwrap() - correct / f64::from(n)).abs() < 1e-9,
            "hit_rate {:?} vs replica {}",
            m.hit_rate(),
            correct / f64::from(n)
        );
        assert!(m.hit_rate().unwrap() > 0.9, "hit {:?}", m.hit_rate());
    }

    #[test]
    fn a_zero_label_scores_under_binary_where_it_is_excluded_under_sign() {
        let mut sign_reading = SlotMetrics::new();
        let mut binary_reading = SlotMetrics::new();
        for _ in 0..100 {
            sign_reading.update(0.9, 0.0, 1.0, 1.0, false);
            binary_reading.update(0.9, 0.0, 1.0, 1.0, true);
        }
        assert!(
            sign_reading.hit_rate().is_none(),
            "y = 0 excludes every row"
        );
        assert_eq!(
            binary_reading.hit_rate(),
            Some(0.0),
            "class 0, called class 1"
        );
    }

    /// A slot's metrics are never windowed, so they keep no runs
    /// (docs/PLAN.md task 128).
    #[test]
    fn the_metrics_keep_no_runs() {
        assert!(!SlotMetrics::new().joint.keeps_runs());
    }

    /// A prediction exactly at the hit test's centre is left out, as a
    /// target there is: it says neither up nor down, and the row ages the
    /// weight without scoring. `f64::signum(+0.0)` is 1, so a prediction of
    /// exactly 0 was "up", a hit on every rising row and a miss on every
    /// falling one, where `po.eval`, whose `sign(0)` is 0, scored it a miss
    /// on both (review round 4, YB8; docs/PLAN.md task 195, S6). `-0.0` is
    /// at the centre too.
    #[test]
    fn a_prediction_at_the_centre_is_not_scored() {
        for (centre, at) in [(0.0, 0.0), (0.0, -0.0), (1.0, 1.0)] {
            let mut m = SlotMetrics::new();
            m.update_about(at, centre + 1.0, 0.9, 1.0, false, centre);
            assert_eq!(m.hit_rate(), None, "centre {centre}: a rising row");
            m.update_about(at, centre - 1.0, 0.9, 1.0, false, centre);
            assert_eq!(m.hit_rate(), None, "centre {centre}: a falling row");
            m.update_about(centre + 0.5, centre + 1.0, 0.9, 1.0, false, centre);
            assert_eq!(m.hit_rate(), Some(1.0), "centre {centre}");
            m.update_about(at, centre + 1.0, 0.5, 1.0, false, centre);
            assert_eq!(m.hit_rate(), Some(1.0), "centre {centre}: not scored");
            assert_eq!(m.hit_w, 0.5, "centre {centre}: but aged");
            // The binary reading scores every row, at its threshold.
            let mut b = SlotMetrics::new();
            b.update_about(0.0, 1.0, 0.9, 1.0, true, centre);
            assert_eq!(b.hit_rate(), Some(0.0), "centre {centre}: binary");
        }
    }
}
