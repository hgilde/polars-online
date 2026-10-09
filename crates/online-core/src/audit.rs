//! `audit`: what a stream's columns hold, read in one pass (docs/PLAN.md task
//! 223 (b)).
//!
//! It learns nothing and predicts nothing. Per column it counts what the
//! models cannot learn from, and measures what makes a column hard to learn
//! from; per stream it measures the clock. Every statistic is over the rows
//! the model is stepped with, in the order they come, and each is held in
//! memory that does not grow with the stream: a column costs a few hundred
//! numbers, whatever the number of rows.
//!
//! **Per column**, on each row:
//!
//! - *what is not a number a model can take*: a null (the plumbing writes
//!   [`NULL`] for one), any other NaN, `+inf`, `-inf`, and a finite value
//!   past [`crate::INPUT_BOUND`], each counted apart. Every other value is
//!   *usable*, and the rest is over the usable values alone;
//! - *moments*: the count `n`, the mean, and the central sums `M2`, `M3`,
//!   `M4`, by Terriberry's one-pass update (Pébay 2008, eq. 2.1-2.3), the
//!   mean a compensated pair ([`crate::comp`]):
//!
//!   ```text
//!   n' = n + 1   d = x − m   e = d / n'   t = d·e·n
//!   M4' = M4 + t·e²·(n'² − 3n' + 3) + 6e²·M2 − 4e·M3
//!   M3' = M3 + t·e·(n' − 2) − 3e·M2
//!   M2' = M2 + t                      m' = m + e
//!   ```
//!
//!   read back as `std = √(M2 / (n − 1))`, `skew = √n·M3 / M2^1.5` and the
//!   excess kurtosis `n·M4 / M2² − 3` (both biased, scipy's defaults), with
//!   the smallest and largest values;
//! - *repeats*: the counts of the distinct values, exact while there are at
//!   most `distinct_cap` of them, so the distinct count is exact up to the
//!   cap. Past it the counts are a Misra-Gries summary (Misra & Gries 1982)
//!   of `distinct_cap` counters: a value not held decrements every counter,
//!   and each count is then at most `D` below its true one, `D` the number
//!   of decrements, which is at most `n / (distinct_cap + 1)`. The most
//!   repeated value and the next one's count are read from the counters,
//!   with `D` beside them;
//! - *runs*: the longest run of one value on consecutive rows, and how many
//!   rows hold the value of the row before, out of the rows whose row before
//!   was usable too, beside `Σ p_v²` over the counted values' shares, the
//!   share independent rows would give. A row that is not usable ends a
//!   run;
//! - *persistence*: the moments of the pairs `(x_{t-1}, x_t)` of consecutive
//!   usable rows, a bivariate Welford update, read back as the lag-1
//!   autocorrelation `S_ab / √(S_aa·S_bb)` and the Dickey-Fuller statistic of
//!   the regression of `x_t` on `x_{t-1}` with a constant, `τ = (ρ̂ − 1) /
//!   se(ρ̂)`, `ρ̂ = S_ab / S_aa`, `se² = (S_bb − S_ab²/S_aa) / ((n − 2)·S_aa)`
//!   (statsmodels' `adfuller` with no lags and a constant);
//! - *location and spread, robustly*: the median and the median absolute
//!   deviation, exact from the counts while they are exact, and otherwise
//!   from a t-digest (Dunning & Ertl 2019) of uniform rank resolution:
//!   centroids of at most `2n / COMPRESSION` rows each, so a rank is known
//!   to within about `1 / COMPRESSION` of the stream. The largest robust z
//!   is `max(max − median, median − min) / (1.4826·MAD)`. [`crate::EwQuantile`]
//!   is a relative-error sketch: it resolves a value to `0.78%` of its own
//!   size, so a column at a level of 1e4 with a spread of 1 would read its
//!   median to within 78 of its spread, and the digest reads it to a rank.
//!
//! **Per pair of columns**, when `pairs` asks (`k(k − 1)/2` of them): over
//! the rows where both are usable, the bivariate moments and the
//! correlation, and the rows on which the two are equal.
//!
//! **The clock**, when the stream has one: each step between consecutive
//! rows (`d_clock`, which the stream has capped at `gap_cap`) is a
//! *duplicate stamp* at 0, a *gap* at or past `gap_cap`, and otherwise a
//! *regular* step, whose mean, spread, coefficient of variation and largest
//! value are kept.
//!
//! **A row's weight** goes into `n_eff`, the accumulated weight before the
//! row, undecayed (CLAUDE.md hard rule 8; `audit` has no decay). Nothing
//! else reads it: a value is a value whatever its row weighs, so a row of
//! weight 0 is counted like any other (hard rule 9 asks only that it divide
//! nothing by zero, and nothing here divides by a weight). Nor does `audit`
//! refuse a value that is not usable, as every other model does
//! ([`OnlineModel`]): it counts it, which is the point.
//!
//! **Merging** ([`Audit::merge`]) two audits of different streams -- two
//! groups, or two runs over different files -- gives what one audit of both
//! would give for the counts, the moments (Pébay's pairwise formulas, exact
//! but for rounding), the longest run (the larger), the rows equal to the
//! row before, the lag pairs, the pair moments and the clock. Two are not
//! exact: the counters past the cap merge as Misra-Gries summaries do
//! (Agarwal et al. 2012), with the error bounds added, and the t-digests by
//! recompressing their centroids together. A run or a lag pair that would
//! have spanned the two streams is not counted, since they were not one.

use serde::{Deserialize, Serialize};

use crate::{INPUT_BOUND, ModelState, OnlineModel, State, StateError, Step, check_schema, comp};

/// The NaN the plumbing writes for a null, so `audit` can count a null
/// apart from a NaN the data holds. A quiet NaN with a payload no arithmetic
/// produces (the bytes spell "NUL"); every other consumer reads it as the
/// NaN it is.
pub const NULL: f64 = f64::from_bits(0x7FF8_0000_004E_554C);

/// Is `v` the null [`NULL`] stands for?
#[inline]
pub fn is_null(v: f64) -> bool {
    v.to_bits() == NULL.to_bits()
}

/// The t-digest's compression: a centroid holds at most `2n / COMPRESSION`
/// rows, so a rank is resolved to about 1% of the stream.
pub const COMPRESSION: f64 = 200.0;

/// Values the t-digest holds before it compresses them into its centroids.
const DIGEST_BUFFER: usize = 256;

/// The default `distinct_cap`: the counters per column, which bound the
/// error of the most repeated value's count by `n / 257`.
pub const DISTINCT_CAP: usize = 256;

/// The largest `distinct_cap`: `2^16` counters, 1 MiB a column.
pub const MAX_DISTINCT_CAP: usize = 1 << 16;

/// `1 / Φ⁻¹(3/4)`: the MAD of a normal sample over its standard deviation.
const MAD_SCALE: f64 = 1.482_602_218_505_602;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditCfg {
    pub n_columns: usize,
    /// Keep each pair of columns' moments and equal rows, `O(k²)`.
    pub pairs: bool,
    /// Counters per column: the distinct count is exact up to this many
    /// values, and the most repeated value's count within `n / (cap + 1)`
    /// past it. At least 1 and at most [`MAX_DISTINCT_CAP`].
    pub distinct_cap: usize,
    /// The stream's `gap_cap`: a step at it is a gap. `None` counts none.
    pub gap_cap: Option<f64>,
    /// Whether the stream has a clock column; without one every step is 1
    /// and nothing is said about the clock.
    pub has_clock: bool,
}

impl AuditCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_columns == 0 {
            return Err("audit: columns must name at least one column".into());
        }
        if !(1..=MAX_DISTINCT_CAP).contains(&self.distinct_cap) {
            return Err(format!(
                "audit: distinct_cap must be in 1..={MAX_DISTINCT_CAP}, got {}",
                self.distinct_cap
            ));
        }
        if let Some(c) = self.gap_cap
            && !(c.is_finite() && c > 0.0)
        {
            return Err(format!("audit: gap_cap must be finite and > 0, got {c}"));
        }
        Ok(())
    }
}

/// One-pass moments to the fourth, the mean a compensated pair, kept over a
/// power of two, `scale`.
///
/// The fourth power of a deviation of `1e100`, a value the input bound
/// allows, is `1e400`, past a double. So the moments are of `x / scale`:
/// `scale` is 1 until a value past `2^65` arrives, and `2^(e − 64)` from
/// then on, `e` the largest exponent seen, which keeps every value under
/// `2^65` and its fourth power finite. Dividing by a power of two is exact,
/// so every number is what the unscaled arithmetic gives wherever that
/// arithmetic stays finite, and a rescale when `scale` grows is exact too;
/// only a value below `2^-1022` times `scale` loses digits to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub n: u64,
    pub mean: f64,
    pub mean_lo: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub scale: f64,
}

impl Default for Moments {
    fn default() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            mean_lo: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            scale: 1.0,
        }
    }
}

/// `2^k`, for `k` in a double's normal range.
fn pow2(k: i64) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The scale a value needs: 1 below `2^65`, else `2^(e − 64)`, `e` its
/// exponent.
fn scale_for(x: f64) -> f64 {
    let e = ((x.to_bits() >> 52) & 0x7ff) as i64 - 1023;
    if e > 64 { pow2(e - 64) } else { 1.0 }
}

impl Moments {
    /// Keep the moments over `to`, a power of two, where it is larger than
    /// `scale`.
    fn rescale(&mut self, to: f64) {
        if to <= self.scale {
            return;
        }
        let k = self.scale / to;
        self.mean *= k;
        self.mean_lo *= k;
        self.m2 *= k * k;
        self.m3 *= k * k * k;
        self.m4 *= k * k * k * k;
        self.scale = to;
    }

    pub fn add(&mut self, x: f64) {
        self.rescale(scale_for(x));
        let x = x / self.scale;
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let d = comp::dev(x, self.mean, self.mean_lo);
        let e = d / n;
        let e2 = e * e;
        let t = d * e * n1;
        comp::add(&mut self.mean, &mut self.mean_lo, e);
        self.m4 += t * e2 * (n * n - 3.0 * n + 3.0) + 6.0 * e2 * self.m2 - 4.0 * e * self.m3;
        self.m3 += t * e * (n - 2.0) - 3.0 * e * self.m2;
        self.m2 += t;
    }

    /// Pébay's (2008) pairwise combination, eq. 3.1.
    pub fn merge(&mut self, o: &Moments) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = o.clone();
            return;
        }
        let mut o = o.clone();
        let to = self.scale.max(o.scale);
        self.rescale(to);
        o.rescale(to);
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let d = (o.mean - self.mean) + (o.mean_lo - self.mean_lo);
        let (d2, d3, d4) = (d * d, d * d * d, d * d * d * d);
        let m4 = self.m4
            + o.m4
            + d4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * d2 * (na * na * o.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * d * (na * o.m3 - nb * self.m3) / n;
        let m3 = self.m3
            + o.m3
            + d3 * na * nb * (na - nb) / (n * n)
            + 3.0 * d * (na * o.m2 - nb * self.m2) / n;
        let m2 = self.m2 + o.m2 + d2 * na * nb / n;
        comp::add(&mut self.mean, &mut self.mean_lo, d * nb / n);
        self.n += o.n;
        (self.m2, self.m3, self.m4) = (m2, m3, m4);
    }

    fn mean(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            (self.mean + self.mean_lo) * self.scale
        }
    }

    /// The sample standard deviation, `ddof = 1`.
    fn std(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            (self.m2.max(0.0) / (self.n as f64 - 1.0)).sqrt() * self.scale
        }
    }

    /// The population standard deviation, `ddof = 0`.
    fn pop_std(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            (self.m2.max(0.0) / self.n as f64).sqrt() * self.scale
        }
    }

    fn skew(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            f64::NAN
        } else {
            (self.n as f64).sqrt() * self.m3 / (self.m2 * self.m2.sqrt())
        }
    }

    fn kurtosis(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            f64::NAN
        } else {
            self.n as f64 * self.m4 / (self.m2 * self.m2) - 3.0
        }
    }

    fn finite(&self) -> bool {
        let power_of_two = self.scale.to_bits() & ((1u64 << 52) - 1) == 0;
        [self.mean, self.mean_lo, self.m2, self.m3, self.m4]
            .iter()
            .all(|v| v.is_finite())
            && self.scale.is_finite()
            && self.scale >= 1.0
            && power_of_two
    }
}

/// One-pass co-moments of two series, the means compensated pairs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoMoments {
    pub n: u64,
    pub ma: f64,
    pub ma_lo: f64,
    pub mb: f64,
    pub mb_lo: f64,
    pub saa: f64,
    pub sbb: f64,
    pub sab: f64,
}

impl CoMoments {
    pub fn add(&mut self, a: f64, b: f64) {
        self.n += 1;
        let n = self.n as f64;
        let da = comp::dev(a, self.ma, self.ma_lo);
        let db = comp::dev(b, self.mb, self.mb_lo);
        comp::add(&mut self.ma, &mut self.ma_lo, da / n);
        comp::add(&mut self.mb, &mut self.mb_lo, db / n);
        let ea = comp::dev(a, self.ma, self.ma_lo);
        let eb = comp::dev(b, self.mb, self.mb_lo);
        self.saa += da * ea;
        self.sbb += db * eb;
        self.sab += da * eb;
    }

    /// Chan, Golub & LeVeque's pairwise combination.
    pub fn merge(&mut self, o: &CoMoments) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = o.clone();
            return;
        }
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let da = (o.ma - self.ma) + (o.ma_lo - self.ma_lo);
        let db = (o.mb - self.mb) + (o.mb_lo - self.mb_lo);
        self.saa += o.saa + da * da * na * nb / n;
        self.sbb += o.sbb + db * db * na * nb / n;
        self.sab += o.sab + da * db * na * nb / n;
        comp::add(&mut self.ma, &mut self.ma_lo, da * nb / n);
        comp::add(&mut self.mb, &mut self.mb_lo, db * nb / n);
        self.n += o.n;
    }

    pub fn corr(&self) -> f64 {
        if self.n < 2 || self.saa <= 0.0 || self.sbb <= 0.0 {
            return f64::NAN;
        }
        (self.sab / (self.saa.sqrt() * self.sbb.sqrt())).clamp(-1.0, 1.0)
    }

    /// The Dickey-Fuller `τ` of `b` on `a` with a constant: `(ρ̂ − 1) /
    /// se(ρ̂)`. Infinite where the fit is exact.
    fn unit_root_t(&self) -> f64 {
        if self.n < 3 || self.saa <= 0.0 {
            return f64::NAN;
        }
        let rho = self.sab / self.saa;
        let ss = (self.sbb - self.sab * rho).max(0.0);
        let se = (ss / ((self.n as f64 - 2.0) * self.saa)).sqrt();
        (rho - 1.0) / se
    }

    fn finite(&self) -> bool {
        [
            self.ma, self.ma_lo, self.mb, self.mb_lo, self.saa, self.sbb, self.sab,
        ]
        .iter()
        .all(|v| v.is_finite())
    }
}

/// Exact counts of up to `cap` distinct values, a Misra-Gries summary past
/// them. Keys are a value's bits, `-0` read as `0`, sorted for a binary
/// search and so the state's bytes do not depend on a hash.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub counts: Vec<(u64, u64)>,
    /// The decrements so far: no count is more than this below its true
    /// one, and a value not held has a true count of at most this.
    pub decrements: u64,
    /// More than `cap` distinct values were seen: the counts are a summary.
    pub over: bool,
}

#[inline]
fn key(v: f64) -> u64 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

impl Counts {
    fn add(&mut self, v: f64, cap: usize) {
        let k = key(v);
        match self.counts.binary_search_by_key(&k, |e| e.0) {
            Ok(i) => self.counts[i].1 += 1,
            Err(i) if self.counts.len() < cap => self.counts.insert(i, (k, 1)),
            Err(_) => {
                self.over = true;
                self.decrements += 1;
                self.counts.iter_mut().for_each(|e| e.1 -= 1);
                self.counts.retain(|e| e.1 > 0);
            }
        }
    }

    /// Agarwal et al.'s (2012) merge: add the counters, then take the
    /// `(cap + 1)`-th largest count from every counter.
    fn merge(&mut self, o: &Counts, cap: usize) {
        let mut all: Vec<(u64, u64)> = Vec::with_capacity(self.counts.len() + o.counts.len());
        let (mut i, mut j) = (0, 0);
        while i < self.counts.len() || j < o.counts.len() {
            match (self.counts.get(i), o.counts.get(j)) {
                (Some(a), Some(b)) if a.0 == b.0 => {
                    all.push((a.0, a.1 + b.1));
                    i += 1;
                    j += 1;
                }
                (Some(a), Some(b)) if a.0 < b.0 => {
                    all.push(*a);
                    i += 1;
                }
                (Some(a), None) => {
                    all.push(*a);
                    i += 1;
                }
                (_, Some(b)) => {
                    all.push(*b);
                    j += 1;
                }
                (None, None) => unreachable!(),
            }
        }
        self.decrements += o.decrements;
        self.over |= o.over;
        if all.len() > cap {
            let mut c: Vec<u64> = all.iter().map(|e| e.1).collect();
            c.sort_unstable_by(|a, b| b.cmp(a));
            let cut = c[cap];
            all.iter_mut().for_each(|e| e.1 -= cut);
            all.retain(|e| e.1 > 0);
            self.decrements += cut;
            self.over = true;
        }
        self.counts = all;
    }

    /// The most repeated value and its count, and the next count: ties go
    /// to the smaller value. `None` before a value.
    fn top(&self) -> Option<(f64, u64, u64)> {
        let mut best: Option<(f64, u64)> = None;
        let mut second = 0;
        for &(k, c) in &self.counts {
            let v = f64::from_bits(k);
            match best {
                Some((bv, bc)) if c > bc || (c == bc && v < bv) => {
                    second = bc;
                    best = Some((v, c));
                }
                Some(_) => second = second.max(c),
                None => best = Some((v, c)),
            }
        }
        best.map(|(v, c)| (v, c, second))
    }

    /// `Σ (c / n)²` over the counters: the share of rows equal to the row
    /// before that independent rows of these frequencies would give. Exact
    /// while the counts are; past the cap a lower bound, each count being
    /// at most the decrements short.
    fn equal_by_chance(&self, n: u64) -> f64 {
        if n == 0 {
            return f64::NAN;
        }
        let n = n as f64;
        self.counts
            .iter()
            .map(|e| {
                let p = e.1 as f64 / n;
                p * p
            })
            .sum()
    }

    /// The exact values and counts, ascending by value; `None` past the cap.
    fn exact(&self) -> Option<Vec<(f64, u64)>> {
        if self.over {
            return None;
        }
        let mut v: Vec<(f64, u64)> = self
            .counts
            .iter()
            .map(|&(k, c)| (f64::from_bits(k), c))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        Some(v)
    }

    fn has_shape(&self, cap: usize) -> bool {
        self.counts.len() <= cap
            && self.counts.windows(2).all(|w| w[0].0 < w[1].0)
            && self.counts.iter().all(|e| {
                let v = f64::from_bits(e.0);
                e.1 > 0 && v.abs() <= INPUT_BOUND && e.0 != (-0.0f64).to_bits()
            })
    }
}

/// A merging t-digest of uniform rank resolution (Dunning & Ertl 2019, the
/// `k_0` scale): centroids `(mean, weight)` ascending, and the values not
/// yet folded into them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    pub centroids: Vec<(f64, f64)>,
    pub buffer: Vec<f64>,
}

impl Digest {
    fn add(&mut self, v: f64) {
        self.buffer.push(v);
        if self.buffer.len() >= DIGEST_BUFFER {
            self.compress();
        }
    }

    /// Fold the buffer into the centroids: sort every centroid and value
    /// by value, then merge neighbours while a centroid stays within
    /// `2n / COMPRESSION` rows.
    fn compress(&mut self) {
        let mut all: Vec<(f64, f64)> = self.centroids.clone();
        all.extend(self.buffer.drain(..).map(|v| (v, 1.0)));
        self.centroids = Self::fold(all);
    }

    fn fold(mut all: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = all.iter().map(|c| c.1).sum();
        let cap = (2.0 * total / COMPRESSION).max(1.0);
        let mut out: Vec<(f64, f64)> = Vec::new();
        for (m, w) in all {
            match out.last_mut() {
                Some(cur) if cur.1 + w <= cap => {
                    let s = cur.1 + w;
                    cur.0 += (m - cur.0) * (w / s);
                    cur.1 = s;
                }
                _ => out.push((m, w)),
            }
        }
        out
    }

    fn merge(&mut self, o: &Digest) {
        let mut all = self.centroids.clone();
        all.extend(o.centroids.iter().copied());
        all.extend(self.buffer.iter().chain(&o.buffer).map(|&v| (v, 1.0)));
        self.buffer.clear();
        self.centroids = Self::fold(all);
    }

    /// The centroids with the buffer folded in, the state untouched.
    fn settled(&self) -> Vec<(f64, f64)> {
        let mut all = self.centroids.clone();
        all.extend(self.buffer.iter().map(|&v| (v, 1.0)));
        Self::fold(all)
    }

    fn has_shape(&self) -> bool {
        self.buffer.len() < DIGEST_BUFFER
            && self.centroids.windows(2).all(|w| w[0].0 <= w[1].0)
            && self
                .centroids
                .iter()
                .all(|c| c.0.abs() <= INPUT_BOUND && c.1 >= 1.0 && c.1.is_finite())
            && self.buffer.iter().all(|v| v.abs() <= INPUT_BOUND)
    }
}

/// The distribution a column's median and MAD are read from: the exact
/// values, or the digest's centroids, each a point `(value, rank)` at the
/// middle of its weight, with the extremes at ranks 0 and `n`.
struct Cdf {
    points: Vec<(f64, f64)>,
    n: f64,
}

impl Cdf {
    fn of(centroids: &[(f64, f64)], lo: f64, hi: f64) -> Self {
        let n: f64 = centroids.iter().map(|c| c.1).sum();
        let mut points = vec![(lo, 0.0)];
        let mut below = 0.0;
        for &(m, w) in centroids {
            points.push((m, below + w / 2.0));
            below += w;
        }
        points.push((hi, n));
        Self { points, n }
    }

    /// The rank at or below `x`, interpolated linearly between points.
    fn rank(&self, x: f64) -> f64 {
        let p = &self.points;
        if x < p[0].0 {
            return 0.0;
        }
        // The last point at or below `x`.
        let i = p.partition_point(|q| q.0 <= x) - 1;
        if i + 1 == p.len() {
            return self.n;
        }
        let (a, b) = (p[i], p[i + 1]);
        if b.0 > a.0 {
            a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0))
        } else {
            a.1
        }
    }

    /// The value at rank `q·n`, the inverse of [`Cdf::rank`].
    fn quantile(&self, q: f64) -> f64 {
        let r = q * self.n;
        let p = &self.points;
        let i = p.partition_point(|pt| pt.1 < r).clamp(1, p.len() - 1);
        let (a, b) = (p[i - 1], p[i]);
        if b.1 > a.1 {
            a.0 + (b.0 - a.0) * ((r - a.1) / (b.1 - a.1))
        } else {
            b.0
        }
    }

    /// The median absolute deviation about `m`: the smallest `d` with half
    /// the rows inside `[m − d, m + d]`, by bisection.
    fn mad(&self, m: f64) -> f64 {
        let (lo, hi) = (self.points[0].0, self.points[self.points.len() - 1].0);
        let mut b = (hi - m).max(m - lo);
        let mut a = 0.0;
        if self.rank(m) - self.rank_below(m) >= self.n / 2.0 {
            return 0.0;
        }
        for _ in 0..200 {
            let d = a + (b - a) / 2.0;
            if d <= a || d >= b {
                break;
            }
            if self.rank(m + d) - self.rank_below(m - d) >= self.n / 2.0 {
                b = d;
            } else {
                a = d;
            }
        }
        b
    }

    /// The rank strictly below `x`: [`Cdf::rank`] approached from the left.
    fn rank_below(&self, x: f64) -> f64 {
        let p = &self.points;
        let i = p.partition_point(|q| q.0 < x);
        if i == 0 {
            return 0.0;
        }
        if i == p.len() {
            return self.n;
        }
        let (a, b) = (p[i - 1], p[i]);
        if b.0 > a.0 {
            a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0))
        } else {
            a.1
        }
    }
}

/// The median of values with counts, ascending by value, numpy's: the mean
/// of the two middle values for an even count.
fn weighted_median(v: &[(f64, u64)]) -> f64 {
    let n: u64 = v.iter().map(|e| e.1).sum();
    if n == 0 {
        return f64::NAN;
    }
    // The values at 0-based ranks (n − 1) / 2 and n / 2.
    let at = |r: u64| {
        let mut below = 0;
        for &(x, c) in v {
            below += c;
            if r < below {
                return x;
            }
        }
        v[v.len() - 1].0
    };
    let (a, b) = (at((n - 1) / 2), at(n / 2));
    a + (b - a) / 2.0
}

/// One column's counts and measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ColumnAudit {
    pub null: u64,
    pub nan: u64,
    pub pos_inf: u64,
    pub neg_inf: u64,
    pub beyond: u64,
    pub moments: Moments,
    pub min: f64,
    pub max: f64,
    pub counts: Counts,
    pub digest: Digest,
    /// The previous row's value, when it was usable and the run goes on.
    pub prev: Option<f64>,
    pub run: u64,
    pub longest_run: u64,
    pub equal_prev: u64,
    pub adjacent: u64,
    pub lag: CoMoments,
}

impl ColumnAudit {
    fn add(&mut self, v: f64, cap: usize) {
        if !crate::usable(v) {
            if is_null(v) {
                self.null += 1;
            } else if v.is_nan() {
                self.nan += 1;
            } else if v == f64::INFINITY {
                self.pos_inf += 1;
            } else if v == f64::NEG_INFINITY {
                self.neg_inf += 1;
            } else {
                self.beyond += 1;
            }
            self.prev = None;
            self.run = 0;
            return;
        }
        if self.moments.n == 0 {
            (self.min, self.max) = (v, v);
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.moments.add(v);
        self.counts.add(v, cap);
        self.digest.add(v);
        match self.prev {
            Some(p) => {
                self.adjacent += 1;
                if p == v {
                    self.equal_prev += 1;
                    self.run += 1;
                } else {
                    self.run = 1;
                }
                self.lag.add(p, v);
            }
            None => self.run = 1,
        }
        self.longest_run = self.longest_run.max(self.run);
        self.prev = Some(v);
    }

    fn merge(&mut self, o: &ColumnAudit, cap: usize) {
        self.null += o.null;
        self.nan += o.nan;
        self.pos_inf += o.pos_inf;
        self.neg_inf += o.neg_inf;
        self.beyond += o.beyond;
        if o.moments.n > 0 {
            if self.moments.n == 0 {
                (self.min, self.max) = (o.min, o.max);
            } else {
                self.min = self.min.min(o.min);
                self.max = self.max.max(o.max);
            }
        }
        self.moments.merge(&o.moments);
        self.counts.merge(&o.counts, cap);
        self.digest.merge(&o.digest);
        self.longest_run = self.longest_run.max(o.longest_run);
        self.equal_prev += o.equal_prev;
        self.adjacent += o.adjacent;
        self.lag.merge(&o.lag);
        // The two were not one stream: no run goes on across them.
        self.prev = None;
        self.run = 0;
    }

    fn report(&self) -> ColumnReport {
        let m = &self.moments;
        let usable = m.n > 0;
        let (median, mad) = if !usable {
            (f64::NAN, f64::NAN)
        } else if let Some(exact) = self.counts.exact() {
            let med = weighted_median(&exact);
            let mut dev: Vec<(f64, u64)> =
                exact.iter().map(|&(x, c)| ((x - med).abs(), c)).collect();
            dev.sort_by(|a, b| a.0.total_cmp(&b.0));
            (med, weighted_median(&dev))
        } else {
            let cdf = Cdf::of(&self.digest.settled(), self.min, self.max);
            let med = cdf.quantile(0.5).clamp(self.min, self.max);
            (med, cdf.mad(med))
        };
        let robust_z = if usable && mad > 0.0 {
            (self.max - median).max(median - self.min) / (MAD_SCALE * mad)
        } else {
            f64::NAN
        };
        let top = self.counts.top();
        ColumnReport {
            rows: self.null + self.nan + self.pos_inf + self.neg_inf + self.beyond + m.n,
            null: self.null,
            nan: self.nan,
            pos_inf: self.pos_inf,
            neg_inf: self.neg_inf,
            beyond: self.beyond,
            count: m.n,
            mean: m.mean(),
            std: m.std(),
            skew: m.skew(),
            kurtosis: m.kurtosis(),
            min: if usable { self.min } else { f64::NAN },
            max: if usable { self.max } else { f64::NAN },
            median,
            mad,
            robust_z,
            distinct: (!self.counts.over).then_some(self.counts.counts.len() as u64),
            top_value: top.map_or(f64::NAN, |t| t.0),
            top_count: top.map_or(0, |t| t.1),
            second_count: top.map_or(0, |t| t.2),
            count_error: self.counts.decrements,
            longest_run: self.longest_run,
            equal_prev: self.equal_prev,
            adjacent: self.adjacent,
            equal_by_chance: self.counts.equal_by_chance(m.n),
            autocorr: self.lag.corr(),
            unit_root_t: self.lag.unit_root_t(),
        }
    }

    fn has_shape(&self, cap: usize) -> bool {
        let m = &self.moments;
        m.finite()
            && self.lag.finite()
            && (m.n == 0 || (self.min.abs() <= INPUT_BOUND && self.max.abs() <= INPUT_BOUND))
            && self.counts.has_shape(cap)
            && self.digest.has_shape()
            && self.prev.is_none_or(|p| p.abs() <= INPUT_BOUND)
            && self.run <= self.longest_run
            && self.equal_prev <= self.adjacent
    }
}

/// What a column held, as [`Audit::column`] reads it. NaN where a
/// statistic is undefined: no usable value, a single one, or no spread.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnReport {
    /// Rows read: every kind below, and the usable ones.
    pub rows: u64,
    pub null: u64,
    pub nan: u64,
    pub pos_inf: u64,
    pub neg_inf: u64,
    /// Finite, but past [`crate::INPUT_BOUND`].
    pub beyond: u64,
    /// Usable values.
    pub count: u64,
    pub mean: f64,
    pub std: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub mad: f64,
    pub robust_z: f64,
    /// The distinct usable values, `None` past `distinct_cap`.
    pub distinct: Option<u64>,
    pub top_value: f64,
    /// The most repeated value's count, at most `count_error` short of the
    /// truth, and the next value's.
    pub top_count: u64,
    pub second_count: u64,
    pub count_error: u64,
    pub longest_run: u64,
    pub equal_prev: u64,
    /// Rows whose row before was usable too: `equal_prev`'s denominator.
    pub adjacent: u64,
    /// `Σ p_v²` over the values' shares: the share of `adjacent` rows that
    /// would equal the row before by chance, were the rows independent.
    /// Exact while `distinct` is; past the cap a lower bound.
    pub equal_by_chance: f64,
    pub autocorr: f64,
    pub unit_root_t: f64,
}

/// A pair of columns, over the rows where both are usable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PairAudit {
    pub moments: CoMoments,
    pub equal: u64,
}

/// What a pair of columns held, as [`Audit::pair`] reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct PairReport {
    pub count: u64,
    pub corr: f64,
    pub equal: u64,
}

/// The steps between consecutive rows.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClockAudit {
    pub steps: u64,
    pub duplicates: u64,
    pub gaps: u64,
    /// The regular steps: neither a duplicate nor a gap.
    pub regular: Moments,
    pub max_step: f64,
}

/// The clock, as [`Audit::clock`] reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockReport {
    pub steps: u64,
    pub duplicates: u64,
    /// `None` without a `gap_cap`.
    pub gaps: Option<u64>,
    pub regular: u64,
    pub step_mean: f64,
    pub step_std: f64,
    /// `step_std / step_mean`, the population spread of the regular steps.
    pub step_cv: f64,
    pub max_step: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub cfg: AuditCfg,
    /// The weight before this row, undecayed.
    n_eff: f64,
    columns: Vec<ColumnAudit>,
    /// `k(k − 1)/2` pairs, `(0, 1), (0, 2), .., (k − 2, k − 1)`, when
    /// `pairs`; none otherwise.
    pairs: Vec<PairAudit>,
    clock: ClockAudit,
    /// The next row starts afresh: no step before it, no row before it.
    fresh: bool,
}

impl Audit {
    pub fn new(cfg: AuditCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.n_columns;
        Ok(Self {
            columns: vec![ColumnAudit::default(); k],
            pairs: if cfg.pairs {
                vec![PairAudit::default(); k * k.saturating_sub(1) / 2]
            } else {
                Vec::new()
            },
            cfg,
            n_eff: 0.0,
            clock: ClockAudit::default(),
            fresh: true,
        })
    }

    pub fn cfg(&self) -> &AuditCfg {
        &self.cfg
    }

    /// The accumulated weight before the next row: every row's, undecayed.
    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// Start a new run: the next row has no step and no row before it, so
    /// it is neither a duplicate stamp nor equal to the row before. The
    /// stream calls it where it would rebuild another model (a restart at a
    /// step back or a session gap): an audit keeps what it has read.
    pub fn restart(&mut self) {
        self.fresh = true;
        for c in &mut self.columns {
            c.prev = None;
            c.run = 0;
        }
    }

    /// Column `j`'s counts and measurements.
    pub fn column(&self, j: usize) -> ColumnReport {
        self.columns[j].report()
    }

    /// The pair `(i, j)`, `i < j`, when the audit keeps pairs.
    pub fn pair(&self, i: usize, j: usize) -> Option<PairReport> {
        let p = self.pairs.get(self.pair_index(i, j)?)?;
        Some(PairReport {
            count: p.moments.n,
            corr: p.moments.corr(),
            equal: p.equal,
        })
    }

    fn pair_index(&self, i: usize, j: usize) -> Option<usize> {
        let k = self.cfg.n_columns;
        (self.cfg.pairs && i < j && j < k).then(|| i * (2 * k - i - 1) / 2 + (j - i - 1))
    }

    /// The clock's steps; `None` for a stream without a clock column.
    pub fn clock(&self) -> Option<ClockReport> {
        if !self.cfg.has_clock {
            return None;
        }
        let r = &self.clock.regular;
        let mean = r.mean();
        let pop = r.pop_std();
        Some(ClockReport {
            steps: self.clock.steps,
            duplicates: self.clock.duplicates,
            gaps: self.cfg.gap_cap.map(|_| self.clock.gaps),
            regular: r.n,
            step_mean: mean,
            step_std: pop,
            step_cv: if mean > 0.0 { pop / mean } else { f64::NAN },
            max_step: if self.clock.steps > 0 {
                self.clock.max_step
            } else {
                f64::NAN
            },
        })
    }

    /// Fold another audit of the same columns into this one, as if one
    /// audit had read both streams (the module's doc says which parts are
    /// exact).
    pub fn merge(&mut self, o: &Audit) -> Result<(), String> {
        if (o.cfg.n_columns, o.cfg.pairs, o.cfg.distinct_cap)
            != (self.cfg.n_columns, self.cfg.pairs, self.cfg.distinct_cap)
        {
            return Err("audit: only audits of the same columns and settings merge".into());
        }
        let cap = self.cfg.distinct_cap;
        self.n_eff += o.n_eff;
        for (a, b) in self.columns.iter_mut().zip(&o.columns) {
            a.merge(b, cap);
        }
        for (a, b) in self.pairs.iter_mut().zip(&o.pairs) {
            a.moments.merge(&b.moments);
            a.equal += b.equal;
        }
        let (c, d) = (&mut self.clock, &o.clock);
        if d.steps > 0 {
            c.max_step = if c.steps > 0 {
                c.max_step.max(d.max_step)
            } else {
                d.max_step
            };
        }
        c.steps += d.steps;
        c.duplicates += d.duplicates;
        c.gaps += d.gaps;
        c.regular.merge(&d.regular);
        self.fresh = true;
        for col in &mut self.columns {
            col.prev = None;
            col.run = 0;
        }
        Ok(())
    }

    fn has_shape(&self) -> bool {
        let k = self.cfg.n_columns;
        let cap = self.cfg.distinct_cap;
        let pairs = if self.cfg.pairs {
            k * k.saturating_sub(1) / 2
        } else {
            0
        };
        self.columns.len() == k
            && self.pairs.len() == pairs
            && self.columns.iter().all(|c| c.has_shape(cap))
            && self
                .pairs
                .iter()
                .all(|p| p.moments.finite() && p.equal <= p.moments.n)
            && self.clock.regular.finite()
            && self.n_eff.is_finite()
            && self.n_eff >= 0.0
            && (self.clock.steps == 0 || self.clock.max_step.is_finite())
    }

    fn step_clock(&mut self, d_clock: f64) {
        if !self.cfg.has_clock {
            return;
        }
        let c = &mut self.clock;
        c.max_step = if c.steps == 0 {
            d_clock
        } else {
            c.max_step.max(d_clock)
        };
        c.steps += 1;
        if d_clock == 0.0 {
            c.duplicates += 1;
        } else if self.cfg.gap_cap.is_some_and(|cap| d_clock >= cap) {
            c.gaps += 1;
        } else {
            c.regular.add(d_clock);
        }
    }
}

impl OnlineModel for Audit {
    /// Count the row: every column's value, the pairs' and the clock's step.
    /// The weight goes into `n_eff` alone, and a value that is not usable is
    /// counted, not refused (the module's doc).
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let out = self.predict(x, d_clock);
        let cap = self.cfg.distinct_cap;
        if !self.fresh && d_clock.is_finite() && d_clock >= 0.0 {
            self.step_clock(d_clock);
        }
        for (c, &v) in self.columns.iter_mut().zip(x) {
            c.add(v, cap);
        }
        if self.cfg.pairs {
            let k = self.cfg.n_columns;
            let mut p = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let (a, b) = (x[i], x[j]);
                    if a.abs() <= INPUT_BOUND && b.abs() <= INPUT_BOUND {
                        let pair = &mut self.pairs[p];
                        pair.moments.add(a, b);
                        if a == b {
                            pair.equal += 1;
                        }
                    }
                    p += 1;
                }
            }
        }
        if weight.abs() <= INPUT_BOUND && weight > 0.0 {
            self.n_eff += weight;
        }
        self.fresh = false;
        out
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        Step {
            pred: Vec::new(),
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Audit(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Audit(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("audit", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(StateError::Invalid(
                        "audit: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "audit",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_columns
    }

    /// Nothing per row: its value is its state, read with [`Audit::column`],
    /// [`Audit::pair`] and [`Audit::clock`].
    fn n_outputs(&self) -> usize {
        0
    }
}

#[cfg(test)]
#[path = "audit/tests.rs"]
mod tests;
