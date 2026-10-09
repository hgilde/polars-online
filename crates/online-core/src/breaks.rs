//! Coefficient breaks (docs/PLAN.md task 221 (b)): the CUSUM and CUSUM of
//! squares of the predictive studentized residual, with a memory of their
//! own, and the Wald distance between a fast and a slow least-squares fit.
//!
//! **The studentized residual.** Each scored row's residual over its own
//! error inflation is its recursive residual, and the studentized residual
//! is that over the spread of the recursive residuals before the row:
//!
//! ```text
//! v = resid / error_inflation,     error_inflation = sqrt(1 + h(x))
//! z = v / s,                       s² = Σ ω v² / Σ ω   (the rows before)
//! ```
//!
//! `h(x)` is the estimation variance of the row's prediction over the noise.
//! Run once with no ridge, `v` is Brown, Durbin and Evans' (1975) recursive
//! residual, `N(0, σ²)` under a constant relationship, and `s` is their
//! spread so far, so `z` is `t`-distributed and near `N(0, 1)`; `z` waits
//! until `s` has [`SCALE_ROWS`] rows of Kish's size. A model with no per-row
//! estimation variance (every one but `ewridge`, `rls` and `kalman`) stands
//! in `error_inflation = 1`: `v` is the residual, and `z` a `zscore` at the
//! diagnostic's memory. The `sigma` field is not the scale: it is the spread
//! of `resid`, which carries the estimation error that `error_inflation`
//! divides out, so `resid / (sigma · error_inflation)` counts it twice. Run
//! once, where the first residuals carry a great deal of it, that scale
//! flagged 47% of 200 streams with no break at the CUSUM of squares' 5%
//! boundary; this one flagged 5.0% (task 221's measurements).
//!
//! **[`Breaks`]**, per slot: the scale's sums and four of `z`, `ω` each
//! row's weight times the decay since it,
//!
//! ```text
//! S1 = Σ ω    S2 = Σ ω²    Z1 = Σ ω z    Z2 = Σ ω z²
//! m4 = Σ ω_s z⁴ / Σ ω_s        (at four times the memory)
//! cusum    = Z1 / sqrt(r · S2)
//! cusum_sq = (Z2 − S1) / sqrt(r₂ · (m4 − 1) · S2)
//! ```
//!
//! Under a constant relationship both are about `N(0, 1)` on every row:
//! `Z1` is a weighted sum of near-independent `N(0, 1)` with variance `S2`,
//! and `Z2 − S1` a weighted sum of `z² − 1`, whose variance is `E[z⁴] − 1`
//! -- 2 for Gaussian errors, `2 + κ` under an excess kurtosis `κ` -- so the
//! measured fourth moment stands where a Gaussian's 2 stood (task 232 (4);
//! task 222's F2: over 2, Student's t with 3 to 5 degrees of freedom passed
//! 1.96 on 11-33% of rows at a memory of 200 and 28-53% of streams' last
//! rows run once; over `m4 − 1`, on 4.7-5.5% and 2.5-7.5%). It is read at
//! four times the memory so that a change of spread does not hide itself in
//! it at once. `r` and `r₂` are the windowed nulls' shares of the plain
//! variance ([`cusum_null`]), 1 run once. Run once (no decay, unit weights)
//! `S2` is the row count `r`, so `cusum · sqrt(r)` is the CUSUM path `W_r`
//! of Brown, Durbin and Evans, read against their boundary `±a (sqrt(T) + 2
//! r / sqrt(T))` over a run of `T` rows (`a` = 0.948 at 5%). `cusum_sq` is
//! not their CUSUM of squares (review round 6, A-7), the share `s_r` of a
//! whole run's squares in its first `r` rows: it is the sum of `z² − 1`
//! over its own standard deviation, the spread the residuals have drifted
//! to from the one their studentizing scale held. Its run-once mean sits a
//! little above 0 (+0.1 to +0.4 at the last of 6,000 rows): the first rows'
//! `z` are read against a young scale and spread wider than 1. With a
//! memory, `cusum` is a moving sum (Chu, Hornik and Kuan 1995) with an
//! exponential window in place of a rectangular one.
//!
//! **Under a horizon** (task 232 (3); review round 6, A-4) a target that
//! looks ahead `h` rows leaves residuals that share their shocks over `h − 1`
//! rows, and the sums' variance is the long-run one. Each denominator takes
//! Newey and West's (1987) Bartlett-weighted lag products of the terms it
//! sums, over `L = 2h` lags of the studentized residuals scored before the
//! row:
//!
//! ```text
//! P1_l = Σ ω_t ω_{t−l} z_t z_{t−l}             P2_l = the same of z² − 1
//! cusum    = Z1 / sqrt(r (S2 + 2 Σ_l (1 − l/(L+1)) P1_l))
//! cusum_sq = (Z2 − S1) / sqrt(r₂ ((m4 − 1) S2 + 2 Σ_l (1 − l/(L+1)) P2_l))
//! ```
//!
//! and the same long-run factor `1 + 2 Σ_l (1 − l/(L+1)) P1_l / S2` scales
//! `break_wald`'s noise variance ([`TwinFit::wald`]). With no horizon `L` is
//! 0 and they are the forms above.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// The Kish size of the recursive residuals' spread before a studentized
/// residual is read against it: measured, 3, 10 and 30 flagged the same
/// share of streams with no break at Brown, Durbin and Evans' boundaries
/// (task 221 (b)), and from 10 on `t`'s tails are those of a normal to a
/// few percent.
pub const SCALE_ROWS: f64 = 10.0;

/// The CUSUM and CUSUM of squares of one slot's studentized residuals, and
/// the spread they are studentized by. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Breaks {
    s1: f64,
    s2: f64,
    z1: f64,
    z2: f64,
    /// `Σ ω_s z⁴` and `Σ ω_s` at four times the memory (the slow factor,
    /// [`slow_factor`]): the fourth moment `cusum_sq`'s null reads (task 232
    /// (4)), slow so that a change of spread does not hide itself in it.
    #[serde(default)]
    z4: f64,
    #[serde(default)]
    w4: f64,
    /// The recursive residuals' `Σ ω`, `Σ ω²` and `Σ ω v²`.
    v1: f64,
    v2: f64,
    vq: f64,
    /// Under a horizon, Newey and West's lags `L`, and per lag the products
    /// `P1_l` and `P2_l` (module docs); 0 and empty without one.
    #[serde(default, skip_serializing_if = "is_zero")]
    lags: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    p1: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    p2: Vec<f64>,
    /// The last `L` scored rows' `ω z` and `ω (z² − 1)` at their present
    /// weights, newest first.
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    ring1: VecDeque<f64>,
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    ring2: VecDeque<f64>,
    /// On a quantile fit the CUSUM sums each row's standardized indicator
    /// `(1{y < pred} − q) / sqrt(q (1 − q))` (task 232 (6)), and `c2` is its
    /// `Σ ω²`; `false` and 0 otherwise.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    indicator: bool,
    #[serde(default)]
    c2: f64,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

impl Breaks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sums whose denominators read Newey and West's `lags` lags (module
    /// docs), the plain forms at 0.
    pub fn with_lags(lags: usize) -> Self {
        Self {
            lags,
            p1: vec![0.0; lags],
            p2: vec![0.0; lags],
            ..Self::default()
        }
    }

    /// The same, whose CUSUM sums a quantile fit's indicators where `on`
    /// (task 232 (6)).
    pub fn with_indicator(mut self, on: bool) -> Self {
        self.indicator = on;
        self
    }

    /// Whether a restored accumulator can be read and folded at `lags`:
    /// finite sums, the weights' and the squares' not negative, a product
    /// per lag and no more scores than lags.
    pub fn has_shape(&self, lags: usize) -> bool {
        self.lags == lags
            && self.p1.len() == lags
            && self.p2.len() == lags
            && self.ring1.len() <= lags
            && self.ring2.len() <= lags
            && self.c2.is_finite()
            && self.c2 >= 0.0
            && self
                .p1
                .iter()
                .chain(&self.p2)
                .chain(&self.ring1)
                .chain(&self.ring2)
                .all(|v| v.is_finite())
            && [
                self.s1, self.s2, self.z1, self.z2, self.z4, self.w4, self.v1, self.v2, self.vq,
            ]
            .iter()
            .all(|v| v.is_finite())
            && [
                self.s1, self.s2, self.z2, self.z4, self.w4, self.v1, self.v2, self.vq,
            ]
            .iter()
            .all(|v| *v >= 0.0)
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.s1 *= lam;
        self.s2 *= lam * lam;
        self.z1 *= lam;
        self.z2 *= lam;
        let slow = slow_factor(lam);
        self.z4 *= slow;
        self.w4 *= slow;
        self.v1 *= lam;
        self.v2 *= lam * lam;
        self.vq *= lam;
        self.c2 *= lam * lam;
        let lam2 = lam * lam;
        self.p1
            .iter_mut()
            .chain(&mut self.p2)
            .for_each(|v| *v *= lam2);
        self.ring1
            .iter_mut()
            .chain(&mut self.ring2)
            .for_each(|v| *v *= lam);
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring1.clear();
        self.ring2.clear();
    }

    /// The studentized residual of a recursive residual `v` against the
    /// spread before it: `None` for a `v` that is not finite, and until the
    /// spread has [`SCALE_ROWS`] rows of Kish's size and is above 0.
    pub fn studentized(&self, v: f64) -> Option<f64> {
        if !v.is_finite() || self.v2 <= 0.0 || self.v1 * self.v1 < SCALE_ROWS * self.v2 {
            return None;
        }
        let ms = self.vq / self.v1;
        (ms > 0.0).then(|| v / ms.sqrt())
    }

    /// One recursive residual `v` at weight `w`, after the step's decay
    /// `lam`: its studentized residual joins the sums, then `v` the spread.
    /// On a quantile fit (`with_indicator`) the CUSUM sums `c`, the row's
    /// standardized indicator `(1{y < pred} − q) / sqrt(q (1 − q))`, in
    /// place of the studentized residual. A `v` that is not finite, or a
    /// weight of 0, only ages (CLAUDE.md hard rule 9).
    pub fn update(&mut self, v: f64, lam: f64, w: f64, c: Option<f64>) {
        let z = self.studentized(v);
        self.age(lam);
        if !(v.is_finite() && w > 0.0) {
            return;
        }
        // The CUSUM's term: the indicator on a quantile fit, else `z`.
        let term = if self.indicator { c } else { z };
        if let Some(t) = term {
            self.z1 += w * t;
            if self.indicator {
                self.c2 += w * w;
            }
            if self.lags > 0 {
                let u1 = w * t;
                for (l, b1) in self.ring1.iter().enumerate() {
                    self.p1[l] += u1 * b1;
                }
                self.ring1.push_front(u1);
                self.ring1.truncate(self.lags);
            }
        }
        if let Some(z) = z {
            self.s1 += w;
            self.s2 += w * w;
            self.z2 += w * z * z;
            self.z4 += w * z * z * z * z;
            self.w4 += w;
            if self.lags > 0 {
                let u2 = w * (z * z - 1.0);
                for (l, b2) in self.ring2.iter().enumerate() {
                    self.p2[l] += u2 * b2;
                }
                self.ring2.push_front(u2);
                self.ring2.truncate(self.lags);
            }
        }
        self.v1 += w;
        self.v2 += w * w;
        self.vq += w * v * v;
    }

    /// The CUSUM's `Σ ω²`: of the indicators on a quantile fit, else of the
    /// studentized residuals.
    fn cusum_s2(&self) -> f64 {
        if self.indicator { self.c2 } else { self.s2 }
    }

    /// `Σ_l (1 − l/(L+1)) P_l` over the lags: Bartlett's weights.
    fn bartlett(&self, p: &[f64]) -> f64 {
        let l1 = self.lags as f64 + 1.0;
        p.iter()
            .enumerate()
            .map(|(l, v)| (1.0 - (l + 1) as f64 / l1) * v)
            .sum()
    }

    /// The long-run variance of the studentized residuals over their
    /// short-run one, `1 + 2 Σ_l (1 − l/(L+1)) P1_l / S2` (module docs): 1
    /// with no horizon, `None` before the first studentized residual and
    /// where the lag products leave it at or below 0.
    pub fn long_run(&self) -> Option<f64> {
        let s2 = self.cusum_s2();
        if s2 <= 0.0 {
            return None;
        }
        let f = 1.0 + 2.0 * self.bartlett(&self.p1) / s2;
        (f > 0.0).then_some(f)
    }

    /// `Z1 / sqrt(r · S2)`, over the long-run variance under a horizon,
    /// `r` the windowed null's share of the plain variance
    /// ([`cusum_null`]; 1 run once): `None` before the first studentized
    /// residual.
    pub fn cusum(&self, r: f64) -> Option<f64> {
        let f = self.long_run()?;
        (r > 0.0).then(|| self.z1 / (r * self.cusum_s2() * f).sqrt())
    }

    /// `(Z2 − S1) / sqrt(r · (m4 − 1) · S2)`, `m4` the studentized
    /// residuals' measured fourth moment at four times the memory, over the
    /// long-run variance under
    /// a horizon, `r` the windowed null's share (1/2 under a memory, where
    /// the scale shares the sums' memory; 1 run once): `None` before the
    /// first studentized residual, and while `m4` is at most 1.
    pub fn cusum_sq(&self, r: f64) -> Option<f64> {
        if self.s1 <= 0.0 || self.s2 <= 0.0 || r <= 0.0 {
            return None;
        }
        if self.w4 <= 0.0 {
            return None;
        }
        let excess = self.z4 / self.w4 - 1.0;
        let var = r * (excess * self.s2 + 2.0 * self.bartlett(&self.p2));
        (excess > 0.0 && var > 0.0).then(|| (self.z2 - self.s1) / var.sqrt())
    }
}

/// The share of a windowed CUSUM's plain variance its null has (task 232
/// (4); review round 6, A-5, G-3), from the half-lives of the fit whose
/// residuals it sums, `h_m`, and of its own sums, `h_c`, in the same clock
/// units:
///
/// ```text
/// r = h_m / (h_m + h_c)
/// ```
///
/// A fit that forgets absorbs a level at its own pace: its out-of-sample
/// residual is the shock less the EW mean of the shocks before it, `e_t =
/// ε_t − κ_m ∫ e^{−κ_m s} ε_{t−s} ds` in continuous time (`κ = ln 2 / h`).
/// Summed at the rate `κ_c`, shock `ε_{t−x}` enters with the weight `(κ_m
/// e^{−κ_m x} − κ_c e^{−κ_c x}) / (κ_m − κ_c)`, whose square integrates to
/// `1 / (2 (κ_m + κ_c))` against the plain sum's `1 / (2 κ_c)`: the ratio is
/// `κ_c / (κ_m + κ_c)`, whatever the rate of the rows. At equal memories it
/// is 1/2, a standard deviation of 0.71, as measured; a fit that forgets
/// nothing (`h_m = inf`) and the run-once sums (`h_c = inf`) read 1. The
/// CUSUM of squares is studentized by a scale at its own memory, which
/// absorbs the squares' level the same way, so its share is 1/2 under a
/// memory.
pub fn cusum_null(fit_half_life: f64, own_half_life: f64) -> f64 {
    if !own_half_life.is_finite() || !fit_half_life.is_finite() {
        return 1.0;
    }
    fit_half_life / (fit_half_life + own_half_life)
}

/// The fast and slow fits of one target, and the sums their difference's
/// variance reads: two exponentially weighted least-squares fits of the
/// target on the features, one at the diagnostic's memory (fast) and one at
/// four times it (slow), each as EW means and centred co-moments
/// ([`crate::EwCov`]), and the Wald distance between their coefficients
/// with the exact variance of their difference (task 232 (5); review round
/// 6, A-1).
///
/// Each fit is `β = G⁻¹ Σ ω z y` over `z = (1, x − c)`, `c` the first row's
/// features, `G = Σ ω z z'`, so under a constant relationship with noise
/// of variance `σ²` the difference is `Δ = Σ_t (A_f ω_{f,t} − A_s ω_{s,t})
/// z_t ε_t`, `A = G⁻¹`, and
///
/// ```text
/// Var(Δ) = σ² (A_f H_ff A_f + A_s H_ss A_s − A_f H_fs A_s − A_s H_fs A_f)
/// H_ff = Σ ω_f² z z'    H_ss = Σ ω_s² z z'    H_fs = Σ ω_f ω_s z z'
/// wald   = Δ' Var(Δ)⁻¹ Δ                       ~ χ²(k) under no break
/// ```
///
/// `σ²` is the slow fit's residual mean square at Kish's degrees of
/// freedom, times the residuals' long-run over short-run variance under a
/// horizon ([`Breaks::long_run`]). Through the origin the same sums are
/// read in `x = c + (x − c)`'s coordinates. The slow factor is the fast
/// one's fourth root, two square roots, which are exact in every libm: no
/// platform's last bit enters the state.
///
/// Until task 232 the variance was `σ² G_s⁻¹ (1/n_f + 1/n_s − 2Σω_fω_s /
/// (Σω_f Σω_s))`, the slow fit's Gram for both: a design whose spread
/// changes reads its fast Gram far from its slow one, and a feature gone
/// quiet with nothing broken passed `chi2(3)`'s 0.01% value on 85% of 100
/// streams (A-1).
///
/// Four, measured against two and eight beside `ewridge` at a half-life of
/// 200 (200 streams of 3,000 rows, a break at row 1,500, the fast fit at
/// the model's memory; first passage of `chi2(3)`'s 0.01% value, 21.1, as
/// a median delay in rows), under the old variance:
///
/// ```text
///              no break               intercept +0.5 sd       slope 1.0 -> 1.6
///  slow   rows>5%  rows>1%  >21.1   rows>1%  >21.1 (delay)   rows>1%  >21.1 (delay)
///   2x    4.60%    0.91%    0%      74%      100% (116)      82%      100% (88)
///   4x    4.63%    0.86%    0%      92%      100% (122)      95%      100% (94)
///   8x    4.70%    0.83%    0%      94%      100% (125)      96%      100% (96)
/// ```
///
/// Each is calibrated with no break. Twice flags a break first by a few
/// rows but holds it on fewer rows after it, and passed `chi2(3)`'s 0.1%
/// value somewhere in 13% of the no-break streams against 9% at four and
/// eight times; eight is four within the noise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinFit {
    /// EW moments of `(x_0, ..., x_{k-1}, y)` at the fast memory.
    fast: crate::EwCov,
    /// The same at the slow memory.
    slow: crate::EwCov,
    /// The origin `c`, the first folded row's features; empty before it.
    #[serde(default)]
    origin: Vec<f64>,
    /// `H_ff`, `H_ss` and `H_fs`, row-major over `z = (1, x − c)`.
    #[serde(default)]
    hff: Vec<f64>,
    #[serde(default)]
    hss: Vec<f64>,
    #[serde(default)]
    hfs: Vec<f64>,
}

/// The slow memory's factor from the fast one's: its fourth root, a half-life
/// four times as long, by two square roots, which every libm rounds exactly.
pub fn slow_factor(lam: f64) -> f64 {
    lam.sqrt().sqrt()
}

/// `a b`, both `p × p` row-major.
fn mat_mul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            for j in 0..p {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    out
}

impl TwinFit {
    /// Two empty fits over `k` features.
    pub fn new(k: usize) -> Self {
        let d = (k + 1) * (k + 1);
        Self {
            fast: crate::EwCov::new(k + 1).without_runs(),
            slow: crate::EwCov::new(k + 1).without_runs(),
            origin: Vec::new(),
            hff: vec![0.0; d],
            hss: vec![0.0; d],
            hfs: vec![0.0; d],
        }
    }

    /// Whether a restored pair is shaped for `k` features.
    pub fn has_shape(&self, k: usize) -> bool {
        let d = (k + 1) * (k + 1);
        self.fast.has_shape(k + 1)
            && self.slow.has_shape(k + 1)
            && self.fast.q_sum().is_some()
            && self.slow.q_sum().is_some()
            && (self.origin.is_empty() || self.origin.len() == k)
            && [&self.hff, &self.hss, &self.hfs]
                .iter()
                .all(|h| h.len() == d)
            && self
                .hff
                .iter()
                .chain(&self.hss)
                .chain(&self.hfs)
                .chain(&self.origin)
                .all(|v| v.is_finite())
    }

    /// The clock moves on by a step whose fast decay is `lam`, with nothing
    /// learned.
    pub fn age(&mut self, lam: f64) {
        let slow = slow_factor(lam);
        self.fast.decay(lam);
        self.slow.decay(slow);
        let (ff, ss, fs) = (lam * lam, slow * slow, lam * slow);
        self.hff.iter_mut().for_each(|v| *v *= ff);
        self.hss.iter_mut().for_each(|v| *v *= ss);
        self.hfs.iter_mut().for_each(|v| *v *= fs);
    }

    /// One row: its features, its target and its weight, after the step's
    /// fast decay `lam`. A target that is not finite, or a weight of 0, only
    /// ages.
    pub fn update(&mut self, x: &[f64], y: f64, lam: f64, w: f64) {
        if !(y.is_finite() && w > 0.0 && x.iter().all(|v| v.is_finite())) {
            return self.age(lam);
        }
        let slow = slow_factor(lam);
        let mut row = Vec::with_capacity(x.len() + 1);
        row.extend_from_slice(x);
        row.push(y);
        self.fast.update(&row, lam, w);
        self.slow.update(&row, slow, w);
        if self.origin.is_empty() {
            self.origin = x.to_vec();
        }
        let z: Vec<f64> = std::iter::once(1.0)
            .chain(x.iter().zip(&self.origin).map(|(v, c)| v - c))
            .collect();
        let d = z.len();
        let (ff, ss, fs, w2) = (lam * lam, slow * slow, lam * slow, w * w);
        for i in 0..d {
            for j in 0..d {
                let zz = w2 * z[i] * z[j];
                let at = i * d + j;
                self.hff[at] = ff * self.hff[at] + zz;
                self.hss[at] = ss * self.hss[at] + zz;
                self.hfs[at] = fs * self.hfs[at] + zz;
            }
        }
    }

    /// The Wald distance between the two fits' coefficients over the
    /// features `idx` (positions in `x`), with the intercept when
    /// `intercept`: `None` until both fits are determined -- each with more
    /// than `k` rows of Kish's size and a Gram that factorizes -- while the
    /// memories coincide (no decay: the run-once form has no slow fit, and
    /// the difference no variance) or the slow fit leaves no residual.
    /// `long_run` scales the noise's variance by the residuals' long-run
    /// over short-run variance under a horizon ([`Breaks::long_run`]), 1
    /// without one.
    pub fn wald(&self, idx: &[usize], intercept: bool, long_run: f64) -> Option<f64> {
        let kx = self.fast.k() - 1;
        let k = idx.len() + usize::from(intercept);
        if k == 0 || self.origin.len() != kx {
            return None;
        }
        let (nf, ns) = (self.fast.n_kish()?, self.slow.n_kish()?);
        if nf <= k as f64 || ns <= k as f64 {
            return None;
        }
        let (bf, bs) = (
            self.slopes(&self.fast, idx, intercept)?,
            self.slopes(&self.slow, idx, intercept)?,
        );
        let mut delta = Vec::with_capacity(k);
        if intercept {
            delta.push(self.level(&self.fast, idx, &bf) - self.level(&self.slow, idx, &bs));
        }
        delta.extend(bf.iter().zip(&bs).map(|(f, s)| f - s));
        let (af, a_s) = (
            inverse(&self.gram(&self.fast, idx, intercept), k)?,
            inverse(&self.gram(&self.slow, idx, intercept), k)?,
        );
        let hff = self.pick(&self.hff, idx, intercept);
        let hss = self.pick(&self.hss, idx, intercept);
        let hfs = self.pick(&self.hfs, idx, intercept);
        let ff = mat_mul(&mat_mul(&af, &hff, k), &af, k);
        let ss = mat_mul(&mat_mul(&a_s, &hss, k), &a_s, k);
        let fs = mat_mul(&mat_mul(&af, &hfs, k), &a_s, k);
        let var: Vec<f64> = (0..k * k)
            .map(|n| ff[n] + ss[n] - fs[n] - fs[(n % k) * k + n / k])
            .collect();
        let fitted: f64 = idx
            .iter()
            .zip(&bs)
            .map(|(&i, b)| b * moment(&self.slow, intercept, i, kx))
            .sum();
        let s2 = (moment(&self.slow, intercept, kx, kx) - fitted) * ns / (ns - k as f64);
        if s2.is_nan() || s2 <= 0.0 || long_run.is_nan() || long_run <= 0.0 {
            return None;
        }
        // A variance that does not factorize -- the memories coincide, or
        // nothing tells the fits apart -- has no distance to read.
        let (x, attempts) = crate::solve_spd(&var, &delta, k, 1)?;
        if attempts > 0 {
            return None;
        }
        let quad: f64 = x.iter().zip(&delta).map(|(a, b)| a * b).sum();
        quad.is_finite().then(|| quad.max(0.0) / (s2 * long_run))
    }

    /// A fit's slopes over `idx`, from its centred moments under an
    /// intercept and its raw ones through the origin.
    fn slopes(&self, m: &crate::EwCov, idx: &[usize], intercept: bool) -> Option<Vec<f64>> {
        let kx = m.k() - 1;
        let p = idx.len();
        if p == 0 {
            return Some(Vec::new());
        }
        let mut a = Vec::with_capacity(p * p);
        for &i in idx {
            for &j in idx {
                a.push(moment(m, intercept, i, j));
            }
        }
        let b: Vec<f64> = idx.iter().map(|&i| moment(m, intercept, i, kx)).collect();
        crate::solve_spd(&a, &b, p, 1)
            .filter(|(_, attempts)| *attempts == 0)
            .map(|(beta, _)| beta)
    }

    /// A fit's intercept at the origin `c`: `m_y − β'(m_x − c)`.
    fn level(&self, m: &crate::EwCov, idx: &[usize], b: &[f64]) -> f64 {
        let kx = m.k() - 1;
        let mut out = m.mean(kx);
        for (&i, bi) in idx.iter().zip(b) {
            out -= bi * (m.mean(i) - self.origin[i]);
        }
        out
    }

    /// A fit's Gram in the coefficients' coordinates, from its moments: `W`
    /// times the second moments of `(1, x − c)` under an intercept, of `x`
    /// through the origin.
    fn gram(&self, m: &crate::EwCov, idx: &[usize], intercept: bool) -> Vec<f64> {
        let w = m.n_eff();
        let k = idx.len() + usize::from(intercept);
        let mut g = vec![0.0; k * k];
        if intercept {
            let dev: Vec<f64> = idx.iter().map(|&i| m.mean(i) - self.origin[i]).collect();
            g[0] = w;
            for (a, &i) in idx.iter().enumerate() {
                g[a + 1] = w * dev[a];
                g[(a + 1) * k] = w * dev[a];
                for (b, &j) in idx.iter().enumerate() {
                    g[(a + 1) * k + b + 1] = w * (m.cov(i, j) + dev[a] * dev[b]);
                }
            }
        } else {
            for (a, &i) in idx.iter().enumerate() {
                for (b, &j) in idx.iter().enumerate() {
                    g[a * k + b] = w * m.raw(i, j);
                }
            }
        }
        g
    }

    /// A kept sum over `z = (1, x − c)` in the coefficients' coordinates:
    /// `z` itself under an intercept, `x = c + (x − c)` through the origin.
    fn pick(&self, h: &[f64], idx: &[usize], intercept: bool) -> Vec<f64> {
        let d = self.origin.len() + 1;
        let rows: Vec<Vec<(usize, f64)>> = if intercept {
            std::iter::once(vec![(0, 1.0)])
                .chain(idx.iter().map(|&i| vec![(1 + i, 1.0)]))
                .collect()
        } else {
            idx.iter()
                .map(|&i| vec![(0, self.origin[i]), (1 + i, 1.0)])
                .collect()
        };
        let k = rows.len();
        let mut out = vec![0.0; k * k];
        for (a, ta) in rows.iter().enumerate() {
            for (b, tb) in rows.iter().enumerate() {
                let mut v = 0.0;
                for &(i, u) in ta {
                    for &(j, t) in tb {
                        v += u * t * h[i * d + j];
                    }
                }
                out[a * k + b] = v;
            }
        }
        out
    }
}

/// The second moment a fit reads: centred with an intercept, raw through
/// the origin.
fn moment(m: &crate::EwCov, intercept: bool, i: usize, j: usize) -> f64 {
    if intercept { m.cov(i, j) } else { m.raw(i, j) }
}

/// `g⁻¹` for a `k × k` Gram, `None` where it does not factorize unjittered.
fn inverse(g: &[f64], k: usize) -> Option<Vec<f64>> {
    let identity: Vec<f64> = (0..k * k)
        .map(|n| if n / k == n % k { 1.0 } else { 0.0 })
        .collect();
    crate::solve_spd(g, &identity, k, k)
        .filter(|(_, attempts)| *attempts == 0)
        .map(|(v, _)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// The scale and the four sums by their definition, from every row and
    /// its weight as it stands, under uneven weights, zeros and a decay:
    /// each row's `z` is its `v` over the spread of the rows before it, read
    /// once they have `SCALE_ROWS` rows of Kish's size.
    #[test]
    fn breaks_are_their_weighted_sums() {
        let mut b = Breaks::new();
        let mut s = 5u64;
        // (v, z or NaN, weight now)
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        // Each row's weight at the slow memory, aligned with `rows`.
        let mut slow_w: Vec<f64> = Vec::new();
        let lam = 0.98;
        for i in 0..300 {
            let v = 3.0 * lcg(&mut s);
            let w = if i % 7 == 0 { 0.0 } else { 1.0 + lcg(&mut s) };
            let (v1, v2, vq) = rows.iter().fold((0.0, 0.0, 0.0), |a, r| {
                (a.0 + r.2, a.1 + r.2 * r.2, a.2 + r.2 * r.0 * r.0)
            });
            let z = if v2 > 0.0 && v1 * v1 / v2 >= SCALE_ROWS {
                v / (vq / v1).sqrt()
            } else {
                f64::NAN
            };
            let got = b.studentized(v);
            assert_eq!(got.is_some(), z.is_finite(), "row {i}");
            if let Some(g) = got {
                assert!((g - z).abs() < 1e-12 * z.abs().max(1.0), "row {i}");
            }
            b.update(v, lam, w, None);
            rows.iter_mut().for_each(|r| r.2 *= lam);
            rows.push((v, z, w));
            slow_w.iter_mut().for_each(|r| *r *= slow_factor(lam));
            slow_w.push(w);
            let (z4, w4) = rows
                .iter()
                .zip(&slow_w)
                .filter(|(r, _)| r.1.is_finite() && r.2 > 0.0)
                .fold((0.0, 0.0), |a, (r, ws)| (a.0 + ws * r.1.powi(4), a.1 + ws));
            let scored: Vec<&(f64, f64, f64)> = rows
                .iter()
                .filter(|r| r.1.is_finite() && r.2 > 0.0)
                .collect();
            let s1: f64 = scored.iter().map(|r| r.2).sum();
            let s2: f64 = scored.iter().map(|r| r.2 * r.2).sum();
            let z1: f64 = scored.iter().map(|r| r.2 * r.1).sum();
            let z2: f64 = scored.iter().map(|r| r.2 * r.1 * r.1).sum();
            if s2 == 0.0 {
                assert!(
                    b.cusum(1.0).is_none() && b.cusum_sq(1.0).is_none(),
                    "row {i}"
                );
                continue;
            }
            let tol = 1e-10;
            assert!(
                (b.cusum(1.0).unwrap() - z1 / s2.sqrt()).abs() < tol,
                "row {i}"
            );
            // A windowed null's share scales the variance.
            let got = b.cusum(0.5).unwrap();
            assert!((got - z1 / (0.5 * s2).sqrt()).abs() < tol, "row {i}");
            // The measured fourth moment in place of a Gaussian's 3.
            let m4 = z4 / w4;
            let want = (z2 - s1) / ((m4 - 1.0) * s2).sqrt();
            assert!((b.cusum_sq(1.0).unwrap() - want).abs() < tol, "row {i}");
            let got = b.cusum_sq(0.5).unwrap();
            assert!((got - want * 2f64.sqrt()).abs() < tol, "row {i}");
        }
        assert!(b.has_shape(0));
    }

    /// Under a horizon, the denominators by their definition: Bartlett's
    /// weights on the lag products of the scored rows' `ω z` and `ω (z² −
    /// 1)`, each at its present weight, a pair `l` scored rows apart --
    /// under a decay, uneven weights, zeros, and studentized residuals that
    /// carry a lag; a cleared run pairs nothing across it.
    #[test]
    fn under_a_horizon_the_variance_is_newey_and_wests() {
        let lags = 3usize;
        let mut b = Breaks::with_lags(lags);
        let mut s = 21u64;
        let lam = 0.97;
        // (z, weight now) of the scored rows, newest last, per run.
        let mut runs: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
        // The same rows at the slow memory's weights.
        let mut slow: Vec<(f64, f64)> = Vec::new();
        let mut prev = 0.0;
        for i in 0..400 {
            let v = 0.7 * prev + lcg(&mut s);
            prev = v;
            let w = if i % 9 == 4 { 0.0 } else { 1.0 + lcg(&mut s) };
            if i == 250 {
                b.clear_lags();
                runs.push(Vec::new());
            }
            let z = b.studentized(v);
            b.update(v, lam, w, None);
            runs.iter_mut().flatten().for_each(|r| r.1 *= lam);
            slow.iter_mut().for_each(|r| r.1 *= slow_factor(lam));
            if let (Some(z), true) = (z, w > 0.0) {
                runs.last_mut().unwrap().push((z, w));
                slow.push((z, w));
            }
            if i < 40 || i % 29 != 0 {
                continue;
            }
            let all: Vec<&(f64, f64)> = runs.iter().flatten().collect();
            let s1: f64 = all.iter().map(|r| r.1).sum();
            let s2: f64 = all.iter().map(|r| r.1 * r.1).sum();
            let z1: f64 = all.iter().map(|r| r.1 * r.0).sum();
            let z2: f64 = all.iter().map(|r| r.1 * r.0 * r.0).sum();
            let z4: f64 = slow.iter().map(|r| r.1 * r.0.powi(4)).sum();
            let w4: f64 = slow.iter().map(|r| r.1).sum();
            let (mut p1, mut p2) = (0.0, 0.0);
            for run in &runs {
                for l in 1..=lags {
                    let bw = 1.0 - l as f64 / (lags as f64 + 1.0);
                    for t in l..run.len() {
                        let (a, c) = (run[t], run[t - l]);
                        p1 += bw * a.1 * c.1 * a.0 * c.0;
                        p2 += bw * a.1 * c.1 * (a.0 * a.0 - 1.0) * (c.0 * c.0 - 1.0);
                    }
                }
            }
            let tol = 1e-9;
            let want = z1 / (s2 + 2.0 * p1).sqrt();
            assert!((b.cusum(1.0).unwrap() - want).abs() < tol, "row {i}");
            let want = (z2 - s1) / ((z4 / w4 - 1.0) * s2 + 2.0 * p2).sqrt();
            assert!((b.cusum_sq(1.0).unwrap() - want).abs() < tol, "row {i}");
            let want = 1.0 + 2.0 * p1 / s2;
            assert!((b.long_run().unwrap() - want).abs() < tol, "row {i}");
        }
        assert!(
            b.long_run().unwrap() > 1.5,
            "the lag carries a long-run variance"
        );
        assert!(b.has_shape(lags) && !b.has_shape(0));
    }

    /// A windowed CUSUM's null share by its definition: the residuals of an
    /// EW mean at the fit's half-life, summed at the CUSUM's, on a grid
    /// fine enough to be the continuous form, against the plain sum's
    /// weights -- `h_m / (h_m + h_c)`; 1/2 at equal memories, 1 where either
    /// forgets nothing.
    #[test]
    fn a_windowed_cusums_null_share_is_its_absorption() {
        let dt = 0.01;
        for (hm, hc) in [(200.0, 200.0), (200.0, 800.0), (200.0, 50.0), (37.0, 91.0)] {
            let (km, kc) = (std::f64::consts::LN_2 / hm, std::f64::consts::LN_2 / hc);
            // Shock `x` ago enters the sum with weight `c(x) = e^(−κ_c x) −
            // κ_m I(x)`: its own weight less what the fit absorbed of it,
            // `I(x) = ∫_0^x e^(−κ_c (x − y)) e^(−κ_m y) dy`, the fit's weight
            // on the shock `y` after it, summed into the CUSUM's weights.
            let n = (60.0 * hm.max(hc) / dt) as usize;
            let (mut integral, mut num, mut den) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let x = (i as f64 + 0.5) * dt;
                integral = integral * (-kc * dt).exp() + (-km * x).exp() * dt;
                let c = (-kc * x).exp() - km * integral;
                num += c * c * dt;
                den += (-2.0 * kc * x).exp() * dt;
            }
            let want = num / den;
            let got = cusum_null(hm, hc);
            assert!((got - want).abs() < 2e-3, "{hm} {hc}: {got} vs {want}");
        }
        assert_eq!(cusum_null(f64::INFINITY, 50.0), 1.0);
        assert_eq!(cusum_null(50.0, f64::INFINITY), 1.0);
    }

    /// A zero weight, the first included, and a `v` that is not a number
    /// only age the sums.
    #[test]
    fn zero_weight_and_missing_rows_only_age() {
        let mut b = Breaks::new();
        b.update(2.0, 0.9, 0.0, None);
        b.update(f64::NAN, 0.9, 1.0, None);
        assert_eq!(b, Breaks::new());
        for v in [
            1.0, -0.5, 0.3, 2.0, -1.0, 0.7, -0.2, 1.1, -0.9, 0.4, 0.8, -0.6,
        ] {
            b.update(v, 0.9, 1.0, None);
        }
        let before = b.clone();
        b.update(5.0, 0.5, 0.0, None);
        assert_eq!(b.z1, before.z1 * 0.5);
        assert_eq!(b.vq, before.vq * 0.5);
        assert_eq!(b.s2, before.s2 * 0.25);
        assert_eq!(b.cusum(1.0), before.cusum(1.0));
    }

    /// The fast and slow fits' slopes, intercepts, residual spread and the
    /// cross sum against their definitions: weighted least squares by
    /// faer's LU (`crate::oracle`) from every row at each memory's weight,
    /// and the statistic built from them by the module's formula.
    #[test]
    fn the_wald_distance_is_the_two_fits_by_definition() {
        // (x, y, weight, its fast weight now, its slow weight now)
        type Row = ([f64; 2], f64, f64, f64, f64);
        let mut t = TwinFit::new(2);
        let mut s = 9u64;
        let lam = 0.97_f64;
        let slow = slow_factor(lam);
        let mut rows: Vec<Row> = Vec::new();
        for i in 0..400 {
            // The second feature goes quiet at row 150, the first's slope
            // breaks at 250.
            let quiet = if i < 150 { 1.0 } else { 0.1 };
            let x = [lcg(&mut s) * 2.0 + 3.0, quiet * lcg(&mut s)];
            let shift = if i < 250 { 0.0 } else { 1.0 };
            let y = 1.0 + (0.5 + shift) * x[0] - x[1] + 0.3 * lcg(&mut s);
            let w = if i % 9 == 4 {
                0.0
            } else {
                0.5 + lcg(&mut s) + 0.5
            };
            t.update(&x, y, lam, w);
            rows.iter_mut().for_each(|r| {
                r.3 *= lam;
                r.4 *= slow;
            });
            rows.push((x, y, w, w, w));
            if i < 30 || i % 37 != 0 {
                continue;
            }
            for intercept in [true, false] {
                let zs = |row: &Row| -> Vec<f64> {
                    if intercept {
                        vec![1.0, row.0[0], row.0[1]]
                    } else {
                        vec![row.0[0], row.0[1]]
                    }
                };
                let p = if intercept { 3 } else { 2 };
                // Each fit by weighted least squares, faer's LU, in the
                // features' own units.
                let fit = |pick: fn(&Row) -> f64| {
                    let (mut g, mut r, mut sw, mut sw2) =
                        (vec![0.0; p * p], vec![0.0; p], 0.0, 0.0);
                    for row in &rows {
                        let om = pick(row);
                        let z = zs(row);
                        for a in 0..p {
                            r[a] += om * z[a] * row.1;
                            for b in 0..p {
                                g[a * p + b] += om * z[a] * z[b];
                            }
                        }
                        sw += om;
                        sw2 += om * om;
                    }
                    let beta = crate::oracle::solve(&g, &r);
                    let ssr: f64 = rows
                        .iter()
                        .map(|row| {
                            let z = zs(row);
                            let e = row.1 - (0..p).map(|a| beta[a] * z[a]).sum::<f64>();
                            pick(row) * e * e
                        })
                        .sum();
                    (beta, g, sw, sw2, ssr)
                };
                let (bf, gf, _, _, _) = fit(|r| r.3);
                let (bs, gs, ws, ws2, ssr_s) = fit(|r| r.4);
                // The difference is `Σ_t d_t ε_t`, `d_t = (A_f ω_f,t − A_s
                // ω_s,t) z_t`: its variance over the noise's is `Σ d_t d_t'`,
                // each row's `d_t` built from the inverses outright.
                let (af, a_s) = (crate::oracle::inverse(&gf), crate::oracle::inverse(&gs));
                let mut var = vec![0.0; p * p];
                for row in &rows {
                    let z = zs(row);
                    let d: Vec<f64> = (0..p)
                        .map(|a| {
                            (0..p)
                                .map(|b| (af[a * p + b] * row.3 - a_s[a * p + b] * row.4) * z[b])
                                .sum()
                        })
                        .collect();
                    for a in 0..p {
                        for b in 0..p {
                            var[a * p + b] += d[a] * d[b];
                        }
                    }
                }
                let ns = ws * ws / ws2;
                let s2 = ssr_s / ws * ns / (ns - p as f64);
                let delta: Vec<f64> = (0..p).map(|a| bf[a] - bs[a]).collect();
                let want = crate::oracle::quad_form(&var, &delta) / s2;
                let got = t.wald(&[0, 1], intercept, 1.0).unwrap();
                assert!(
                    (got - want).abs() < 1e-6 * want.max(1.0),
                    "row {i}, intercept {intercept}: {got} vs {want}"
                );
                // A long-run factor divides it.
                let half = t.wald(&[0, 1], intercept, 2.0).unwrap();
                assert!((half - got / 2.0).abs() < 1e-12 * got.max(1.0));
            }
        }
        // After the break the slow fit lags the fast one.
        assert!(t.wald(&[0, 1], true, 1.0).unwrap() > 30.0);
        assert!(t.has_shape(2));
    }

    /// The null by simulation, sharing nothing of the implementation's
    /// formula but its definition: a feature goes quiet with nothing
    /// broken, and over 300 seeds the statistic's mean is the `chi2(3)`'s
    /// 3, and it passes its 5% value on about 5% of them -- where the slow
    /// fit's Gram for both read a mean of 19.3 (review round 6, A-1).
    #[test]
    fn a_feature_gone_quiet_breaks_nothing() {
        let lam = 0.5f64.powf(1.0 / 50.0);
        let (mut sum, mut past) = (0.0, 0usize);
        let seeds = 300u64;
        for seed in 0..seeds {
            let mut t = TwinFit::new(2);
            let mut s = 1_000 + seed;
            for i in 0..600 {
                let quiet = if i < 400 { 1.0 } else { 0.1 };
                let x = [2.0 * lcg(&mut s), quiet * 2.0 * lcg(&mut s)];
                let e = (lcg(&mut s) + lcg(&mut s) + lcg(&mut s)) * 2.0;
                t.update(&x, 0.5 + x[0] - x[1] + e, lam, 1.0);
            }
            let w = t.wald(&[0, 1], true, 1.0).unwrap();
            sum += w;
            past += usize::from(w > 7.815);
        }
        let mean = sum / seeds as f64;
        assert!((mean - 3.0).abs() < 0.5, "mean {mean}");
        let rate = past as f64 / seeds as f64;
        assert!(rate < 0.09, "rate {rate}");
    }
    /// Run once (no decay) the two memories coincide and there is no
    /// distance to read; through the origin the raw moments are read.
    #[test]
    fn no_decay_has_no_slow_fit_and_no_intercept_reads_raw_moments() {
        let mut t = TwinFit::new(1);
        let mut s = 2u64;
        for _ in 0..100 {
            let x = lcg(&mut s) + 2.0;
            t.update(&[x], 3.0 * x + 0.1 * lcg(&mut s), 1.0, 1.0);
        }
        assert!(t.wald(&[0], true, 1.0).is_none());
        let mut u = TwinFit::new(1);
        for i in 0..300 {
            let x = lcg(&mut s) + 2.0;
            let slope = if i < 200 { 3.0 } else { 3.5 };
            u.update(&[x], slope * x + 0.1 * lcg(&mut s), 0.95, 1.0);
        }
        assert!(u.wald(&[0], false, 1.0).unwrap() > 10.0);
        assert!(u.wald(&[], false, 1.0).is_none(), "nothing to compare");
    }
}
