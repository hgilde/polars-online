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
//! 0 and they are the forms above. On a quantile fit the CUSUM's terms are
//! the indicators (task 232 (6)), so `P1_l` is theirs; the residuals'
//! `P1_l` are kept beside them for `break_wald`, whose twin fits' noise is
//! the residuals' ([`Breaks::long_run_residuals`]; review round 7, A-2).

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
    /// On a quantile fit under a horizon, the lag products `P1_l` of the
    /// studentized residuals' `ω z` and their ring, which `P1` and `ring1`
    /// then hold of the indicators: `break_wald`'s noise is the residuals'
    /// (review round 7, A-2; [`Self::long_run_residuals`]). Empty
    /// otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pz: Vec<f64>,
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    ringz: VecDeque<f64>,
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
    /// (task 232 (6)); under lags it then keeps the residuals' lag products
    /// beside the indicators' (review round 7, A-2).
    pub fn with_indicator(mut self, on: bool) -> Self {
        self.indicator = on;
        let kept = if on { self.lags } else { 0 };
        self.pz = vec![0.0; kept];
        self.ringz.clear();
        self
    }

    /// The lags the residuals' own products are kept over: `lags` on a
    /// quantile fit, 0 elsewhere, where `P1` is already the residuals'.
    fn residual_lags(&self) -> usize {
        if self.indicator { self.lags } else { 0 }
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
            && self.pz.len() == self.residual_lags()
            && self.ringz.len() <= self.residual_lags()
            && self.c2.is_finite()
            && self.c2 >= 0.0
            && self
                .p1
                .iter()
                .chain(&self.p2)
                .chain(&self.ring1)
                .chain(&self.ring2)
                .chain(&self.pz)
                .chain(&self.ringz)
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
            .chain(&mut self.pz)
            .for_each(|v| *v *= lam2);
        self.ring1
            .iter_mut()
            .chain(&mut self.ring2)
            .chain(&mut self.ringz)
            .for_each(|v| *v *= lam);
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring1.clear();
        self.ring2.clear();
        self.ringz.clear();
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
            // On a quantile fit `P1` holds the indicators' products: the
            // residuals' own, for `break_wald`, are kept beside them.
            if self.residual_lags() > 0 {
                let uz = w * z;
                for (l, bz) in self.ringz.iter().enumerate() {
                    self.pz[l] += uz * bz;
                }
                self.ringz.push_front(uz);
                self.ringz.truncate(self.lags);
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

    /// The long-run variance of the CUSUM's terms over their short-run one,
    /// `1 + 2 Σ_l (1 − l/(L+1)) P1_l / S2` (module docs) -- the studentized
    /// residuals', or on a quantile fit the indicators': 1 with no horizon,
    /// `None` before the first term and where the lag products leave it at
    /// or below 0.
    pub fn long_run(&self) -> Option<f64> {
        Self::long_run_of(self.cusum_s2(), self.bartlett(&self.p1))
    }

    /// The studentized residuals' long-run variance over their short-run
    /// one, which scales `break_wald`'s noise ([`TwinFit::wald`]): the
    /// CUSUM's own ([`Self::long_run`]) but on a quantile fit, whose CUSUM
    /// sums indicators. The twin least-squares fits' noise is the
    /// residuals', whose autocorrelation under an overlap is larger than
    /// the indicators' (the arcsine law, `(2/π) asin ρ` at the median), so
    /// the indicators' factor left the Wald distance too large: on quiet
    /// five-row look-ahead streams it passed `chi2(3)`'s 5% value on 10.8%
    /// of rows beside `quantile` against 5.9% beside `ewridge` (review
    /// round 7, A-2).
    pub fn long_run_residuals(&self) -> Option<f64> {
        if !self.indicator {
            return self.long_run();
        }
        Self::long_run_of(self.s2, self.bartlett(&self.pz))
    }

    fn long_run_of(s2: f64, bartlett: f64) -> Option<f64> {
        if s2 <= 0.0 {
            return None;
        }
        let f = 1.0 + 2.0 * bartlett / s2;
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

    /// How far the studentized residuals' sums have filled their memory:
    /// `x = κ T`, the decay since the first of them, `κ = ln 2 / h` at the
    /// sums' memory (task 278, D-B5). The sums' own weights tell it, free of
    /// the weights' scale and the rows' rate: `S1 = Σ ω` decays at `κ` and
    /// `Σ ω_s` (`w4`) at `κ / 4`, over the same rows, so in continuous time
    /// at a steady rate their ratio is
    ///
    /// ```text
    /// S1 / Σ ω_s = (1 − p⁴) / (4 (1 − p)) = (1 + p)(1 + p²) / 4,    p = e^{−κ T / 4}
    /// ```
    ///
    /// 1 at the first row and 1/4 once filled. `p` is the cubic's root on
    /// `[0, 1]`, found by Newton's method from 1 (the cubic is convex and
    /// rising there, so the steps fall monotonically onto it), and `x = −4
    /// ln p`. On discrete rows a filled ratio sits a little above 1/4:
    /// with a step's decay `λ` it is `(1 − λ^{1/4}) / (1 − λ)`, which
    /// reads `p ≈ 0.75 / K`, `K = S1² / S2` the sums' Kish size (`2 / (1 −
    /// λ)` filled). A `p` within twice that of 0 is below the reading's
    /// resolution, and reads as filled: `inf`, where the shares are their
    /// steady forms to the bit. Under uneven weights the ratio is a
    /// weighted one, which the last rows' weights move: with weights `1 ±
    /// 0.5` it read within 10% of the rows' age, and long filled a `q =
    /// e^{−x}` near `4e-8`, not `inf`. A long gap ages `S1` past the ratio's 1/4,
    /// which also reads as filled. 0 before the first row and with nothing
    /// decaying (run once, where `S1 = Σ ω_s`).
    pub fn elapsed(&self) -> f64 {
        if !(self.s1 > 0.0 && self.s2 > 0.0 && self.w4 > 0.0) {
            return 0.0;
        }
        let target = 4.0 * self.s1 / self.w4;
        if target >= 4.0 {
            return 0.0;
        }
        if target <= 1.0 {
            return f64::INFINITY;
        }
        let mut p = 1.0f64;
        for _ in 0..100 {
            let f = ((p + 1.0) * p + 1.0) * p + 1.0 - target;
            let next = p - f / ((3.0 * p + 2.0) * p + 1.0);
            if next >= p || next.is_nan() {
                break;
            }
            p = next;
        }
        if p <= 1.5 * self.s2 / (self.s1 * self.s1) {
            return f64::INFINITY;
        }
        -4.0 * p.ln()
    }

    /// The scale's lead on the sums at `elapsed` ([`Self::elapsed`]), `D =
    /// q (1 − c)`, which [`cusum_sq_null_at`] reads: `q = e^{−x}` the sums'
    /// unfilled share and `c = e^{−κ τ}` the scale's when the sums began, `τ`
    /// earlier (it waits for [`SCALE_ROWS`] rows of Kish's size). The
    /// scale's `Σ ω` (`v1`) has filled `N = 1 − c q` where the sums' has
    /// filled `1 − q`, so `N = (1 − q) v1 / S1` and `D = N − (1 − q)`,
    /// held to `[0, q]`. 0 where `elapsed` is not a positive finite time.
    pub fn scale_lead(&self, elapsed: f64) -> f64 {
        if !(elapsed > 0.0 && elapsed.is_finite() && self.s1 > 0.0) {
            return 0.0;
        }
        let (q, m) = ((-elapsed).exp(), -(-elapsed).exp_m1());
        (m * self.v1 / self.s1 - m).clamp(0.0, q)
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

/// The share of a windowed CUSUM's plain variance its null has beside a
/// fit that absorbs a level over a window of `window` clock units cut from
/// a decay at `fit_half_life` (`inf` for none), its own sums at
/// `own_half_life` (review round 7, A-3). [`cusum_null`] is the window's
/// limit as it grows; read as an exponential of the window's Kish size, a
/// window of 100 rows with no decay read 1/2 where this reads 0.703, and
/// the CUSUM spread 1.19 on clean streams.
///
/// The fit's prediction is a weighted mean of the shocks before the row,
/// `g(y)` the weight on the one `y` ago, so the residual is `e_t = ε_t −
/// ∫ g(y) ε_{t−y} dy`. Summed at the rate `κ_c` (`ln 2` over the half-life,
/// weights `w(x) = e^{−κ_c x}`), shock `ε_{t−x}` enters with `c = w − w ∗ g`,
/// and since `∫ w(x) w(x − y) dx = e^{−κ_c |y|} / (2 κ_c)`, its square
/// over the plain sum's `∫ w² = 1 / (2 κ_c)` is
///
/// ```text
/// r = 1 − 2 ∫ g(y) e^{−κ_c y} dy + ∫∫ g(y) g(y') e^{−κ_c |y − y'|} dy dy'
/// ```
///
/// With `u = κ_c W`, `v = κ_m W` (`κ_m` the decay's rate) and `F(a) = (1 −
/// e^{−a}) / a`, `F(0) = 1`, the window's weights `g(y) = κ_m e^{−κ_m y} /
/// (1 − e^{−v})` on `[0, W]` give, with `C = v / (1 − e^{−v})`,
///
/// ```text
/// r = 1 − 2 C F(u + v) + (C² / v) (F(u + v) − e^{−2v} F(u − v))
/// ```
///
/// and, with no decay (`v → 0`, `g = 1 / W`),
///
/// ```text
/// r = 1 + 2 F'(u) = 1 + 2 e^{−u} / u − 2 (1 − e^{−u}) / u²
/// ```
///
/// which is 0.703 at the window's Kish half-life (`u = 2`). The double
/// integral is `2 C² / W²` times `∫_0^W e^{(κ_m − κ_c) p} ∫_p^W e^{−2 κ_m
/// s} ds dp` over the lag `p = y − y'`. As the window grows past the decay
/// (`u, v → ∞`), `r → u / (u + v)`, [`cusum_null`]'s `κ_c / (κ_m + κ_c)`.
/// Near `v = 0` the decayed form cancels, so a `v` below `1e-8` reads the
/// undecayed one (they part by about `v`); near `u = 0` so does `F'`, which
/// a series reads below 0.1.
///
/// **The decayed form's rounding is about `2⁻⁵²/v` absolute.** Its bracket
/// `F(u + v) − e^{−2v} F(u − v)` is of order `u + v` and rounds to about
/// `2⁻⁵²`, and `C²/v` scales that by `1/v`. Where the share is of order 1
/// that is nothing, but where the window is far shorter than both
/// memories (`u` and `v` small) the share itself is about `2u/3`, so its
/// relative error is about `1.5 · 2⁻⁵² / (u v)`: `4.6e-5` at a window of 1
/// beside a decay half-life of `1e7` and a CUSUM memory of `1e4`
/// (`2.3e-5` on the CUSUM, which reads the square root). Measured against
/// the share's series in `u` and `v`, `2u/3 − uv/6 − u²/4 + …`, the error
/// stayed below `0.9 · 2⁻⁵²/v` with Apple's and glibc's `exp` and below
/// `1.8 · 2⁻⁵²/v` with Windows', for `u ≤ 1e-4` and `1e-8 ≤ v ≤ 1e-5`. At
/// the switch, `v = 1e-8`, the rounding and the undecayed form's
/// truncation are the same size, `2e-8`. `inf` for `own_half_life` reads 1, as
/// [`cusum_null`] does for the run-once sums; an infinite or non-positive
/// `window` is no window, and reads [`cusum_null`].
pub fn cusum_null_window(window: f64, fit_half_life: f64, own_half_life: f64) -> f64 {
    if !(window.is_finite() && window > 0.0) {
        return cusum_null(fit_half_life, own_half_life);
    }
    if !own_half_life.is_finite() {
        return 1.0;
    }
    let ln2 = std::f64::consts::LN_2;
    let u = ln2 / own_half_life * window;
    // 0 where nothing decays (`ln 2 / inf`).
    let v = ln2 / fit_half_life * window;
    if v < 1e-8 {
        return 1.0 + 2.0 * d_absorbed(u);
    }
    let c = v / -(-v).exp_m1();
    // `e^{−2v} F(u − v)`, kept from overflowing where `v > u`.
    let tail = if u >= v {
        (-2.0 * v).exp() * absorbed(u - v)
    } else {
        (-(u + v)).exp() * absorbed(v - u)
    };
    1.0 - 2.0 * c * absorbed(u + v) + c * c / v * (absorbed(u + v) - tail)
}

/// `F(a) = (1 − e^{−a}) / a`, 1 at 0: [`cusum_null_window`]'s weight on a
/// rate `a` over a unit window.
fn absorbed(a: f64) -> f64 {
    if a == 0.0 { 1.0 } else { -(-a).exp_m1() / a }
}

/// `F'(u) = e^{−u} / u − (1 − e^{−u}) / u²`, by its series
/// `Σ_{n≥1} (−1)^n n u^{n−1} / (n + 1)!` below 0.1, where the closed form
/// cancels.
fn d_absorbed(u: f64) -> f64 {
    if u < 0.1 {
        let (mut sum, mut pow, mut fact) = (0.0, 1.0, 1.0);
        for n in 1..14 {
            fact *= f64::from(n + 1);
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * f64::from(n) * pow / fact;
            pow *= u;
        }
        return sum;
    }
    let q = (-u).exp();
    q / u - (1.0 - q) / (u * u)
}

/// The share of a windowed CUSUM's plain variance its null has after its
/// sums have run for `elapsed`, `x = κ_c T` ([`Breaks::elapsed`]), beside a
/// fit that has (task 278, D-B5). [`cusum_null`] and [`cusum_null_window`]
/// are its limit as the sums fill; a sum of a few rows is all but its plain
/// variance, since the fit's mean absorbs a level over the fit's memory,
/// of which those rows hold little. Divided by the filled share from its
/// first row, the CUSUM passed 1.96 on 13% to 34% of the 150 rows after the
/// diagnostics' readiness on clean streams beside `ewridge` at a half-life
/// of 200, by memory from 200 to 800, and on 6.6% to 23% still 300 to 600
/// rows in.
///
/// The fit has run long before the sums (the diagnostics wait for it to
/// settle), so its residual is stationary, `e_t = ε_t − ∫ g(y) ε_{t−y} dy`,
/// with autocovariance `δ(d) + K(|d|)`, `K = −g + g ⋆ g` (`(g ⋆ g)(d) = ∫
/// g(y) g(y + d) dy`). The sums began `T` ago, with weights `ω(x) = e^{−κ_c
/// x}` on `[0, T]`, so
///
/// ```text
/// Var Z = ∫_0^T ω² + 2 ∫_0^T K(d) ∫_0^{T−d} ω(x) ω(x + d) dx dd
/// r(T)  = 1 + 2 (J(κ_c) − e^{−2x} J(−κ_c)) / (1 − e^{−2x}),    J(s) = ∫_0^{min(T, W)} K(d) e^{−s d} dd
/// ```
///
/// over the plain `∫_0^T ω² = (1 − e^{−2x}) / (2 κ_c)`; `r(0) = 1`, and
/// `r(∞) = 1 + 2 J(κ_c)` over the fit's whole support, the steady form.
/// Beside an exponential fit (`g = κ_m e^{−κ_m y}`, `K = −(κ_m/2) e^{−κ_m
/// d}`), with `β = κ_m / κ_c = h_c / h_m` and `F(a) = (1 − e^{−a}) / a`,
///
/// ```text
/// r(x) = h_m / (h_m + h_c) + (2 h_c / (h_m + h_c)) x e^{−(1 + min(1, β)) x} F(|1 − β| x) / (1 − e^{−2x})
/// ```
///
/// Beside a window of `W` (`u = κ_c W`, `v = κ_m W`, `ℓ = min(T, W) / W`),
/// with no decay `K(d) = −d / W²` and `J(±κ_c) = −ℓ² H(±u ℓ)`, `H(a) = ∫_0^1
/// t e^{−a t} dt`; cut from a decay, with `C = v / (1 − e^{−v})`,
///
/// ```text
/// J(±κ_c) = −C ℓ F((v ± u) ℓ) + (C² ℓ / (2v)) (F((v ± u) ℓ) − e^{−2v} F((±u − v) ℓ))
/// ```
///
/// which at `ℓ = 1`, `T → ∞`, is [`cusum_null_window`]'s closed form. The
/// steady share is that function's own, and the transient is added to it,
/// so a filled sum reads it to the bit. Every exponential is formed with its
/// exponent at most 0. `elapsed` of 0 (or below) reads 1, the plain sum;
/// `inf` the steady share.
pub fn cusum_null_at(window: f64, fit_half_life: f64, own_half_life: f64, elapsed: f64) -> f64 {
    let steady = cusum_null_window(window, fit_half_life, own_half_life);
    if !own_half_life.is_finite() || elapsed.is_nan() || elapsed == f64::INFINITY {
        return steady;
    }
    if elapsed <= 0.0 {
        return 1.0;
    }
    let x = elapsed;
    let p = -(-2.0 * x).exp_m1();
    if !(window.is_finite() && window > 0.0) {
        if !fit_half_life.is_finite() {
            return steady;
        }
        let beta = own_half_life / fit_half_life;
        let lead = 2.0 * own_half_life / (fit_half_life + own_half_life);
        let decay = (-(1.0 + beta.min(1.0)) * x).exp();
        return steady + lead * x * decay * absorbed((1.0 - beta).abs() * x) / p;
    }
    let ln2 = std::f64::consts::LN_2;
    let u = ln2 / own_half_life * window;
    let v = ln2 / fit_half_life * window;
    let l = (x / u).min(1.0);
    let e2 = (-2.0 * x).exp();
    let j_full = window_j(u, v, 1.0, 1.0, 0.0);
    let head = if l < 1.0 {
        window_j(u, v, l, 1.0, 0.0) - j_full
    } else {
        0.0
    };
    steady + 2.0 * (head + e2 * j_full - window_j(u, v, l, -1.0, 2.0 * x)) / p
}

/// `e^{−scale} J(σ κ_c)` beside a window, `J` over `ℓ` of it
/// ([`cusum_null_at`]), `σ = ±1`; `v` below `1e-8` reads the window with no
/// decay, as [`cusum_null_window`] does.
fn window_j(u: f64, v: f64, l: f64, sigma: f64, scale: f64) -> f64 {
    let y = u * l;
    if v < 1e-8 {
        let h = if sigma > 0.0 {
            -d_absorbed(y) * (-scale).exp()
        } else {
            scaled_h_neg(y, scale)
        };
        return -l * l * h;
    }
    let c = v / -(-v).exp_m1();
    let near = scaled_absorbed((v + sigma * u) * l, scale);
    let far = scaled_absorbed((sigma * u - v) * l, scale + 2.0 * v);
    -c * l * near + c * c * l / (2.0 * v) * (near - far)
}

/// `e^{−scale} F(a)`, `F(a) = (1 − e^{−a}) / a`, for an `a` of either sign
/// and a `scale` at least `−a`, so no exponential overflows.
fn scaled_absorbed(a: f64, scale: f64) -> f64 {
    if a >= 0.0 {
        return (-scale).exp() * absorbed(a);
    }
    let y = -a;
    if y <= 1.0 {
        (-scale).exp() * y.exp_m1() / y
    } else {
        ((y - scale).exp() - (-scale).exp()) / y
    }
}

/// `e^{−scale} H(−y)`, `H(−y) = ∫_0^1 t e^{y t} dt = (1 + e^y (y − 1)) /
/// y²`, by its series `Σ_{n≥0} y^n / (n! (n + 2))` below 0.1, where the
/// closed form cancels; `y ≤ scale`, so no exponential overflows.
fn scaled_h_neg(y: f64, scale: f64) -> f64 {
    if y < 0.1 {
        let (mut sum, mut pow, mut fact) = (0.0, 1.0, 1.0);
        for n in 0..14 {
            if n > 0 {
                fact *= f64::from(n);
            }
            sum += pow / (fact * f64::from(n + 2));
            pow *= y;
        }
        return (-scale).exp() * sum;
    }
    ((-scale).exp() + (y - scale).exp() * (y - 1.0)) / (y * y)
}

/// The share of the CUSUM of squares' plain variance its null has after
/// its sums have run for `elapsed` ([`Breaks::elapsed`]), the scale having
/// led them by `lead` ([`Breaks::scale_lead`]) (task 278, D-B5): 1/2 once
/// filled ([`cusum_null`] at its own memory), 1 run once. Divided by 1/2
/// from its first row, the CUSUM of squares passed 1.96 on 12% to 18% of
/// the 150 rows after the diagnostics' readiness on clean streams, by
/// memory from 50 to 800, beside `ewridge`, `rls` and `sgd` alike.
///
/// `z² − 1` is, to first order, `(v² − s²) / σ²`: the square's shock less
/// the scale's weighted mean of those before it, at the sums' rate `κ`.
/// The scale began `τ` before the sums, so its weights on the shocks
/// before row `t` are `κ e^{−κ y} / N(t)` on `[0, t + τ]`, `N(t) = 1 −
/// c e^{−κ t}`, `c = e^{−κ τ}`. Two rows' residuals then covary by `−(κ/2)
/// e^{−κ (t − t')} N(t') / N(t)`, and the sum over `[0, T]`, against its
/// plain `(1 − q²) / (2κ)`, `q = e^{−κ T}`, has the share
///
/// ```text
/// r₂ = 1/2 + (q − D) / (1 + q) + D² ln(N / D) / (1 − q²),    D = q (1 − c),  N = N(T) = D + 1 − q
/// ```
///
/// 1 at `T = 0` and 1/2 + `q² κ T / (1 − q²)` beside a scale long filled
/// (`c = 0`). A filled sum reads 1/2 to the bit where its transient is
/// below half its last place, and `inf` reads it outright.
pub fn cusum_sq_null_at(own_half_life: f64, elapsed: f64, lead: f64) -> f64 {
    let steady = cusum_null(own_half_life, own_half_life);
    if !own_half_life.is_finite() || elapsed.is_nan() || elapsed == f64::INFINITY {
        return steady;
    }
    if elapsed <= 0.0 {
        return 1.0;
    }
    let (q, m) = ((-elapsed).exp(), -(-elapsed).exp_m1());
    let d = lead.clamp(0.0, q);
    let tail = if d > 0.0 {
        d * d * (m / d).ln_1p() / (m * (1.0 + q))
    } else {
        0.0
    };
    steady + (q - d) / (1.0 + q) + tail
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
/// The variance is `σ² Σ_t d_t d_t'`, `d_t = (A_f ω_{f,t} − A_s ω_{s,t})
/// z_t`, and `Σ_t d_t z_t' = A_f G_f − A_s G_s = I − I = 0`: the `d_t` lie
/// in the null space of the `k × n` design, so over `n` rows the variance
/// has rank at most `n − k`, and is singular until `n = 2k` rows, whatever
/// their weights. The distance waits for them. Read sooner, whether it
/// factorized was the rounding's to decide, and the distance if it did:
/// multiplying every weight by 2, which moves no sum but by an exact
/// factor, turned it on or off on 1 to 3 of the release probe's 400 rows
/// (task 270 (g)).
///
/// `σ²` is the slow fit's residual mean square at Kish's degrees of
/// freedom, times the residuals' long-run over short-run variance under a
/// horizon ([`Breaks::long_run_residuals`]). Through the origin the same sums are
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
#[serde(from = "TwinFitWire")]
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
    /// Rows learned since the history was last wiped numerically
    /// ([`history_wiped`]): the count [`Self::wald`] waits on for `2k`.
    /// `fast.rows_learned()` counted every row since the first, so after a
    /// gap that left the old rows below the rounding of the new ones the
    /// distance was read on the `k` to `2k` rows after it, off a variance
    /// whose rank was the new rows' alone (task 270, R1-2). A state saved
    /// before the count existed loads with every row it learned
    /// ([`TwinFitWire`]).
    fresh_rows: u64,
}

/// [`TwinFit`] as a file holds it: a state saved before
/// [`TwinFit::fresh_rows`] existed has none, and loads with every row it
/// has learned, the count that state's gate read. A stream no gap has
/// wiped counts the same, so such a state saves the bytes a current build
/// writes from the same stream, and one saved inside its first `2k` rows
/// waits out the rest of them as it did. (`u64::MAX`, "past the wait",
/// would have skipped that wait and saved other bytes.)
#[derive(Deserialize)]
struct TwinFitWire {
    fast: crate::EwCov,
    slow: crate::EwCov,
    #[serde(default)]
    origin: Vec<f64>,
    #[serde(default)]
    hff: Vec<f64>,
    #[serde(default)]
    hss: Vec<f64>,
    #[serde(default)]
    hfs: Vec<f64>,
    #[serde(default)]
    fresh_rows: Option<u64>,
}

impl From<TwinFitWire> for TwinFit {
    fn from(w: TwinFitWire) -> Self {
        let fresh_rows = w.fresh_rows.unwrap_or_else(|| w.fast.rows_learned());
        Self {
            fast: w.fast,
            slow: w.slow,
            origin: w.origin,
            hff: w.hff,
            hss: w.hss,
            hfs: w.hfs,
            fresh_rows,
        }
    }
}

/// Whether a history carried into a row at weight `carried` (the fast fit's
/// weight times the step's decay) is wiped numerically by a row of weight
/// `w`: its share of the weight after the row, `s = carried / (carried +
/// w)`, squared, at most `2^-53`. The old rows' terms in the fast fit's
/// `H_ff = Σ ω² z z'` sum to at most `carried²` times their `z z'`, the new
/// row's is `w²` times its own, so below that share they are under the
/// rounding of the new row's term: the fast half of the difference's
/// variance has the new rows' rank alone, `n − k` over `n` of them, and the
/// variance as a whole holds the old rows only through the slow fit, at
/// `λ^½` in `H_ss`, where the cancellation `A_f H_ff A_f + A_s H_ss A_s −
/// A_f H_fs A_s − A_s H_fs A_f` resolves them to `2^-53 / λ^½` of themselves
/// and the distance amplifies that some thousands of times. Measured on the
/// distance across weight scales of 2, 100 and 0.01 (20 seeds, 40 rows at a
/// decay of 0.97, one row after a decay of `λ`, then 20 rows; the largest
/// relative move): `λ = 2^-26`, 1.9e-9; `2^-40`, 1.8e-7; `2^-45`, 2.3e-6;
/// `2^-53`, 2.7e-5; `2^-60`, 8.6e-4; `2^-80`, 2.6; from `2^-100` the
/// distance flips between a number and none, and 121 times at a decay of 0.
/// The share composes over a gap split by zero-weight rows, which age the
/// fast fit's weight, and counts a row whose own weight dwarfs the history
/// as the wipe it is; a share is free of the weights' scale.
fn history_wiped(carried: f64, w: f64) -> bool {
    let share = carried / (carried + w);
    share * share <= 2f64.powi(-53)
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
            fresh_rows: 0,
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
        self.fresh_rows = if history_wiped(lam * self.fast.n_eff(), w) {
            1
        } else {
            self.fresh_rows.saturating_add(1)
        };
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
    /// `intercept`: `None` until both fits are determined and their
    /// difference has a variance -- `2k` rows learned since the history
    /// was last wiped numerically ([`history_wiped`]), each fit with more
    /// than `k` rows of Kish's size and a Gram that factorizes -- while the
    /// memories coincide (no decay: the run-once form has no slow fit, and
    /// the difference no variance) or the slow fit leaves no residual.
    /// `long_run` scales the noise's variance by the residuals' long-run
    /// over short-run variance under a horizon
    /// ([`Breaks::long_run_residuals`]), 1 without one.
    pub fn wald(&self, idx: &[usize], intercept: bool, long_run: f64) -> Option<f64> {
        let kx = self.fast.k() - 1;
        let k = idx.len() + usize::from(intercept);
        if k == 0 || self.origin.len() != kx {
            return None;
        }
        // The difference's variance is singular below `2k` rows, whatever
        // their weights (the struct's docs): reading it there would leave
        // whether it factorizes, and the distance if it does, to the
        // rounding (task 270 (g)). Rows since the history was last wiped
        // numerically, which a gap can do long after the first row (R1-2).
        if self.fresh_rows < 2 * k as u64 {
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

    /// The share beside a window by its definition (review round 7, A-3):
    /// shock `x` ago enters the CUSUM with weight `c(x) = e^(−κ_c x) −
    /// ∫_0^min(x,W) g(y) e^(−κ_c (x − y)) dy`, `g` the window's weights on
    /// the shocks before a row -- uniform with no decay, `e^(−κ_m y)`
    /// normalized beside one -- summed on a fine grid against the plain
    /// sum's weights. The closed form agrees to the grid's error at short,
    /// equal and long CUSUM memories (within 1e-4 here), with no decay, a decay much longer
    /// than the window, one beside it and one much shorter; with no decay
    /// it reads 0.703 at the window's Kish half-life whatever the window,
    /// and it runs into [`cusum_null`] as the window grows past the decay.
    #[test]
    fn a_windows_cusum_null_share_is_its_absorption() {
        let ln2 = std::f64::consts::LN_2;
        let by_grid = |w: f64, hm: f64, hc: f64| {
            let dt = 0.002 * w.min(hc);
            let (km, kc) = (ln2 / hm, ln2 / hc);
            let cells = (w / dt).round() as usize;
            let total: f64 = (0..cells)
                .map(|j| (-km * (j as f64 + 0.5) * dt).exp() * dt)
                .sum();
            // `acc` is `∫_0^min(x,W) g(y) e^(−κ_c (x − y)) dy` at the cell
            // edge `x`, each cell's weight taken at its middle; `c²` and the
            // plain sum's weights are integrated by the trapezoid rule.
            let n = cells + (60.0 * hc / dt) as usize;
            let (mut acc, mut c_lo, mut num, mut den) = (0.0, 1.0, 0.0, 0.0);
            for i in 0..n {
                let x = (i + 1) as f64 * dt;
                acc *= (-kc * dt).exp();
                if i < cells {
                    let mid = (i as f64 + 0.5) * dt;
                    acc += (-km * mid).exp() / total * dt * (-kc * dt / 2.0).exp();
                }
                let c_hi = (-kc * x).exp() - acc;
                num += (c_lo * c_lo + c_hi * c_hi) / 2.0 * dt;
                den += ((-2.0 * kc * (x - dt)).exp() + (-2.0 * kc * x).exp()) / 2.0 * dt;
                c_lo = c_hi;
            }
            num / den
        };
        let inf = f64::INFINITY;
        for w in [12.0f64, 100.0, 400.0] {
            let kish = ln2 / ((w + 1.0) / (w - 1.0)).ln();
            for hm in [inf, 1e7, 5.0 * w, w, 0.1 * w] {
                for hc in [kish / 20.0, kish / 3.0, kish, 4.0 * kish, 50.0 * kish] {
                    let got = cusum_null_window(w, hm, hc);
                    let want = by_grid(w, hm, hc);
                    assert!(
                        (got - want).abs() < 5e-4 * want.max(0.05),
                        "{w} {hm} {hc}: {got} vs {want}"
                    );
                }
            }
            // At the window's continuous Kish half-life, `W ln 2 / 2`.
            let r = cusum_null_window(w, inf, w * ln2 / 2.0);
            assert!((r - 0.70300).abs() < 1e-5, "{w}: {r}");
        }
        // The series and the closed form of `F'` meet at 0.1, and the
        // decayed form meets the undecayed one at `v = 1e-8`.
        let (lo, hi) = (d_absorbed(0.1 - 1e-12), d_absorbed(0.1));
        assert!((lo - hi).abs() < 1e-12, "{lo} vs {hi}");
        // A CUSUM far longer than the window, where `F'`'s closed form
        // would lose every digit: `r = 2u/3 − u²/4 + u³/15 − …`.
        for u in [1e-6, 1e-3] {
            let r = cusum_null_window(1.0, inf, ln2 / u);
            let want = 2.0 * u / 3.0 - u * u / 4.0 + u * u * u / 15.0;
            assert!((r - want).abs() < 1e-9 * want, "{u}: {r} vs {want}");
        }
        // Where the window is far shorter than both memories the decayed
        // form rounds to about `2⁻⁵²/v` absolute, as its doc says (review
        // round 8, C-1): held to that against the series `r = u (2/3 −
        // v/6) − u²/4 + u² v/12 + u³/15 + …`, whose next terms are below
        // `1e-14` here, at the reviewer's two points (`(W, h_m, h_c) = (1,
        // 1e7, 1e4)` and `(12, 1e7, 1.2e5)`, where the grid above cannot
        // see it) and across `u ≤ 1e-4`, `1e-8 ≤ v ≤ 1e-5`.
        let mut points = vec![(1.0, 1e7, 1e4), (12.0, 1e7, 1.2e5)];
        for u0 in [1e-6, 1e-5, 1e-4] {
            for v0 in [1.0001e-8, 1e-7, 1e-6, 1e-5] {
                points.push((1.0, ln2 / v0, ln2 / u0));
            }
        }
        for (w, hm, hc) in points {
            let (u, v) = (ln2 / hc * w, ln2 / hm * w);
            let want =
                u * (2.0 / 3.0 - v / 6.0) - u * u / 4.0 + u * u * v / 12.0 + u * u * u / 15.0;
            let got = cusum_null_window(w, hm, hc);
            // `(1.5 k + 0.75)·2⁻⁵²/v` for a libm `k` ulps off (docs/PLAN.md
            // task 273): the bracket `F(u + v) − e^{−2v} F(u − v)` holds
            // values just under 1, each `F` an `exp_m1` (`k` ulps of
            // `2⁻⁵³`) and a division (½), the tail an `exp` (`k`) and a
            // product (½), `(3k + 1.5)·2⁻⁵³` in all, and `c²/v` scales it
            // by `1/v`; `c`'s own `k` ulps reach the share unscaled. Held at
            // 9, twice the worst measured with a libm 2 ulps off: 0.9 with
            // Apple's and glibc's `exp` and 1.7 with Windows' (CI
            // 2026-10-17, which took task 269's bound from 1 to 4), and 4.3
            // with Apple's moved by up to 2 ulps, where the derivation (`k
            // = 2.5`) gives 4.5; 4 failed there.
            assert!(
                (got - want).abs() <= 9.0 * f64::EPSILON / v,
                "({w}, {hm:e}, {hc:e}): {got:e} vs {want:e}, {:e} past 2^-52/v",
                (got - want).abs() * v / f64::EPSILON
            );
        }
        // Each switch's edge reads the closed form, which parts from the
        // other form in its last bits: `u = 0.1` reads `F'` in closed form,
        // and `v = 1e-8` the decayed share.
        let q = (-0.1f64).exp();
        assert_eq!(hi, q / 0.1 - (1.0 - q) / (0.1 * 0.1));
        let mut h = ln2 / 1e-8;
        for _ in 0..64 {
            if ln2 / h == 1e-8 {
                break;
            }
            h = if ln2 / h > 1e-8 {
                h.next_up()
            } else {
                h.next_down()
            };
        }
        assert_eq!(ln2 / h, 1e-8);
        let edge = cusum_null_window(1.0, h, 0.35);
        let undecayed = 1.0 + 2.0 * d_absorbed(ln2 / 0.35);
        assert!(
            edge != undecayed && (edge - undecayed).abs() < 1e-6,
            "{edge}"
        );
        let w = 100.0;
        let at = |v: f64| cusum_null_window(w, ln2 * w / v, 30.0);
        assert!((at(0.99e-8) - at(1.01e-8)).abs() < 1e-7);
        assert!((at(1e-5) - at(1e-9)).abs() < 1e-4);
        // Equal rates, where `F(u − v)` is `F(0) = 1`, lie on the same
        // curve as their neighbours.
        let (w, h) = (100.0, 30.0);
        let equal = cusum_null_window(w, h, h);
        let near = cusum_null_window(w, h, h * (1.0 + 1e-9));
        assert!((equal - near).abs() < 1e-8, "{equal} vs {near}");
        assert!((equal - by_grid(w, h, h)).abs() < 5e-4 * equal);
        // Rates far apart either way, where `e^(|u − v|)` alone would
        // overflow: a CUSUM far shorter than the window takes almost the
        // whole of a sum, and a window far past a short decay is the
        // exponential's share.
        let short = cusum_null_window(1_000.0, 1_000.0, 0.5);
        assert!(short > 0.999 && short <= 1.0, "{short}");
        let past = cusum_null_window(1e5, 1.0, 50.0);
        assert!((past - cusum_null(1.0, 50.0)).abs() < 1e-9, "{past}");
        // A window far past the decay is the exponential's share.
        let long = cusum_null_window(1e6, 50.0, 80.0);
        assert!((long - cusum_null(50.0, 80.0)).abs() < 1e-9, "{long}");
        assert_eq!(cusum_null_window(inf, 50.0, 80.0), cusum_null(50.0, 80.0));
        assert_eq!(cusum_null_window(0.0, 50.0, 80.0), cusum_null(50.0, 80.0));
        assert_eq!(cusum_null_window(100.0, 50.0, inf), 1.0);
        assert_eq!(cusum_null_window(100.0, inf, inf), 1.0);
    }

    /// Composite Simpson's rule over `[lo, hi]` in `n` (even) intervals.
    fn simpson(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
        if hi <= lo {
            return 0.0;
        }
        let h = (hi - lo) / n as f64;
        let inner: f64 = (1..n)
            .map(|i| f(lo + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
            .sum();
        (f(lo) + f(hi) + inner) * h / 3.0
    }

    /// Simpson's rule over `[lo, hi]` in segments of at most `step`, eight
    /// intervals each: the integrands below change on the scale of the
    /// shortest memory, which `step` is a fraction of.
    fn simpson_by(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, step: f64) -> f64 {
        if hi <= lo {
            return 0.0;
        }
        let n = ((hi - lo) / step).ceil().max(1.0) as usize;
        let h = (hi - lo) / n as f64;
        (0..n)
            .map(|i| simpson(f, lo + i as f64 * h, lo + (i + 1) as f64 * h, 8))
            .sum()
    }

    /// A young CUSUM's null share by its definition (task 278, D-B5): the
    /// sums began `T` ago with weights `ω(x) = e^{−κ x}`; the shock at `t`
    /// enters them with `c(t) = ω(T − t) 1{t ≥ 0} − ∫ ω(T − s) k(s, t) ds`
    /// over the rows `s` in `(max(t, 0), upper(t)]` whose residual absorbed
    /// it, `k(s, t)` the absorber's weight on it; the share is `∫ c²` over
    /// the plain `∫_0^T ω²`. Simpson's rule on each piece between the kinks
    /// `pieces` (from the earliest shock that matters to `T`), in segments
    /// of at most `step`.
    fn share_by_definition(
        kappa: f64,
        t_end: f64,
        kernel: &dyn Fn(f64, f64) -> f64,
        upper: &dyn Fn(f64) -> f64,
        pieces: &[f64],
        step: f64,
    ) -> f64 {
        let om = |x: f64| (-kappa * x).exp();
        // A piece holds the shocks the sums take themselves where it lies
        // at or past 0: `c` jumps there, so each piece reads its own side.
        let c = |t: f64, summed: bool| {
            let own = if summed { om(t_end - t) } else { 0.0 };
            let lo = t.max(0.0);
            own - simpson_by(&|s| om(t_end - s) * kernel(s, t), lo, upper(t), step)
        };
        let num: f64 = pieces
            .windows(2)
            .map(|p| simpson_by(&|t| c(t, p[0] >= 0.0).powi(2), p[0], p[1], step))
            .sum();
        num / (-(-2.0 * kappa * t_end).exp_m1() / (2.0 * kappa))
    }

    /// Beside an exponential fit, the share from the sums' first row on is
    /// its definition's ([`share_by_definition`], the fit's residual
    /// stationary): 1 at `T = 0`, `h_m / (h_m + h_c)` filled, at short,
    /// equal and long CUSUM memories; `inf` reads the steady share to the
    /// bit, and so does a sum long filled, its transient below half its last
    /// place.
    #[test]
    fn a_young_cusums_null_share_is_its_absorption() {
        let ln2 = std::f64::consts::LN_2;
        let inf = f64::INFINITY;
        for (hm, hc) in [(200.0, 200.0), (200.0, 800.0), (200.0, 50.0), (37.0, 91.0)] {
            let (km, kc) = (ln2 / hm, ln2 / hc);
            let kernel = |s: f64, t: f64| km * (-km * (s - t)).exp();
            for frac in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0] {
                let t_end = frac * hc;
                let want = share_by_definition(
                    kc,
                    t_end,
                    &kernel,
                    &|_| t_end,
                    &[-20.0 * hm, 0.0, t_end],
                    hm.min(hc) / 4.0,
                );
                let got = cusum_null_at(inf, hm, hc, kc * t_end);
                assert!(
                    (got - want).abs() < 1e-6,
                    "{hm} {hc} {t_end}: {got} vs {want}"
                );
            }
            let steady = cusum_null(hm, hc);
            assert_eq!(cusum_null_at(inf, hm, hc, inf), steady);
            assert_eq!(cusum_null_at(inf, hm, hc, 0.0), 1.0);
            assert_eq!(cusum_null_at(inf, hm, hc, 800.0), steady);
            assert!(cusum_null_at(inf, hm, hc, 1e-300) > 1.0 - 1e-15);
            assert_eq!(cusum_null_at(inf, hm, hc, f64::NAN), steady);
        }
        // Equal memories, where `F(|1 − β| x)` is `F(0)`, lie on the same
        // curve as their neighbours.
        let (a, b) = (
            cusum_null_at(inf, 100.0, 100.0, 1.3),
            cusum_null_at(inf, 100.0 * (1.0 + 1e-9), 100.0, 1.3),
        );
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        // Nothing forgets: the fit absorbs nothing, the run-once sums are
        // plain.
        assert_eq!(cusum_null_at(inf, inf, 50.0, 0.7), 1.0);
        assert_eq!(cusum_null_at(inf, 50.0, inf, 0.7), 1.0);
    }

    /// Beside a window, with no decay, a decay much longer than the window,
    /// one beside it and one much shorter, the young share is its
    /// definition's from inside the window's first span to long past it,
    /// and meets the steady form ([`cusum_null_window`]) to the bit once
    /// filled; it is continuous where the sums' age passes the window.
    #[test]
    fn a_young_windows_cusum_null_share_is_its_absorption() {
        let ln2 = std::f64::consts::LN_2;
        let inf = f64::INFINITY;
        for w in [12.0f64, 100.0] {
            for hm in [inf, 5.0 * w, w, 0.1 * w] {
                let km = ln2 / hm;
                let norm = if hm.is_finite() {
                    km / -(-km * w).exp_m1()
                } else {
                    1.0 / w
                };
                // The window's support, `(t, t + W]`, is `upper`'s.
                let kernel = |s: f64, t: f64| norm * (-km * (s - t)).exp();
                for hc in [w / 20.0, w / 3.0, 4.0 * w] {
                    let kc = ln2 / hc;
                    for frac in [0.05, 0.5, 1.0, 2.5] {
                        let t_end = frac * w;
                        let mut pieces = vec![-w, t_end - w, 0.0, t_end];
                        pieces.sort_by(f64::total_cmp);
                        let want = share_by_definition(
                            kc,
                            t_end,
                            &kernel,
                            &|t| (t + w).min(t_end),
                            &pieces,
                            hm.min(hc).min(w) / 4.0,
                        );
                        let got = cusum_null_at(w, hm, hc, kc * t_end);
                        assert!(
                            (got - want).abs() < 1e-6 * want.max(0.05),
                            "{w} {hm} {hc} {t_end}: {got} vs {want}"
                        );
                    }
                    let steady = cusum_null_window(w, hm, hc);
                    assert_eq!(cusum_null_at(w, hm, hc, inf), steady);
                    assert_eq!(cusum_null_at(w, hm, hc, 0.0), 1.0);
                    let x_w = kc * w;
                    let (lo, hi) = (
                        cusum_null_at(w, hm, hc, x_w * (1.0 - 1e-12)),
                        cusum_null_at(w, hm, hc, x_w * (1.0 + 1e-12)),
                    );
                    assert!((lo - hi).abs() < 1e-9, "{w} {hm} {hc}: {lo} vs {hi}");
                }
            }
        }
        // Long filled, the transient is below the steady share's last place.
        assert_eq!(
            cusum_null_at(400.0, 200.0, 200.0, 26.6),
            cusum_null_window(400.0, 200.0, 200.0)
        );
        // Rates far apart, where an exponential of `(u − v) ℓ` alone would
        // overflow, read finite shares in `[0, 1]`.
        for (w, hm, hc, x) in [
            (1_000.0, 1_000.0, 0.5, 5.0),
            (1_000.0, 1_000.0, 0.5, 2_000.0),
            (1e5, 1.0, 50.0, 3.0),
            (1e5, 1.0, 50.0, 900.0),
            (1.0, 1e7, 1e4, 1e-5),
        ] {
            let r = cusum_null_at(w, hm, hc, x);
            assert!(
                r.is_finite() && (0.0..=1.0).contains(&r),
                "{w} {hm} {hc} {x}: {r}"
            );
        }
    }

    /// The CUSUM of squares' young share by its definition: the scale, at
    /// the sums' own memory, began `τ` before them and absorbs each shock
    /// before a row with its weight then, `κ e^{−κ (s − t)} / N(s)`, `N(s) =
    /// 1 − e^{−κ (s + τ)}`. 1 at `T = 0` and 1/2 filled, for a scale a few
    /// rows ahead and one long filled (`c = 0`); `inf` reads 1/2 to the bit.
    #[test]
    fn a_young_cusum_of_squares_share_is_its_scales_absorption() {
        let ln2 = std::f64::consts::LN_2;
        for hc in [50.0, 200.0] {
            let kc = ln2 / hc;
            for tau in [10.0, 0.5 * hc, 1e4 * hc] {
                let kernel =
                    |s: f64, t: f64| kc * (-kc * (s - t)).exp() / -(-kc * (s + tau)).exp_m1();
                for frac in [0.02, 0.2, 1.0, 3.0] {
                    let t_end = frac * hc;
                    let lo = -tau.min(20.0 * hc);
                    let want = share_by_definition(
                        kc,
                        t_end,
                        &kernel,
                        &|_| t_end,
                        &[lo, 0.0, t_end],
                        hc.min(tau) / 4.0,
                    );
                    let x = kc * t_end;
                    let lead = (-x).exp() * -(-kc * tau).exp_m1();
                    let got = cusum_sq_null_at(hc, x, lead);
                    assert!(
                        (got - want).abs() < 1e-6,
                        "{hc} {tau} {t_end}: {got} vs {want}"
                    );
                }
            }
            assert_eq!(cusum_sq_null_at(hc, f64::INFINITY, 0.3), 0.5);
            assert_eq!(cusum_sq_null_at(hc, 0.0, 0.3), 1.0);
            assert_eq!(cusum_sq_null_at(hc, 800.0, 0.0), 0.5);
            // A lead past the sums' unfilled share is held to it.
            assert_eq!(
                cusum_sq_null_at(hc, 2.0, 5.0),
                cusum_sq_null_at(hc, 2.0, (-2.0f64).exp())
            );
        }
        assert_eq!(cusum_sq_null_at(f64::INFINITY, 1.0, 0.1), 1.0);
    }

    /// The sums' age is read off their own weights: rows at unit steps and
    /// weights at half-lives of 50 and 200 -- within 1% of `κ` times the
    /// rows scored less one, from the first rows to six half-lives in --
    /// and uneven weights, within 10% (4% from the 50th row); filled (`inf`)
    /// once the reading is below its resolution with unit weights (with
    /// uneven ones, within `1e-6` of it), and after a gap long enough to
    /// age the sums past it; 0 before any row and run once. The scale's lead is `q (1 − c)`, `c =
    /// e^{−κ τ}` for the `τ` rows it read before the first studentized
    /// residual, to within a row's decay.
    #[test]
    fn the_sums_age_is_read_off_their_weights() {
        let ln2 = std::f64::consts::LN_2;
        assert_eq!(Breaks::new().elapsed(), 0.0);
        for (h, uneven) in [(50.0, false), (200.0, false), (200.0, true)] {
            let lam = 0.5f64.powf(1.0 / h);
            let kc = ln2 / h;
            let mut b = Breaks::new();
            let mut s = 7u64;
            let (mut scored, mut ahead, mut max_err) = (0usize, 0usize, 0.0f64);
            let mut max_late = 0.0f64;
            for i in 0..(60.0 * h) as usize {
                let v = lcg(&mut s) * 3.0;
                let w = if uneven { 1.0 + lcg(&mut s) } else { 1.0 };
                b.update(v, lam, w, None);
                if b.s1 > 0.0 {
                    scored += 1;
                } else {
                    ahead += 1;
                }
                let x = b.elapsed();
                if scored >= 2 && (scored as f64) < 6.0 * h {
                    let want = kc * (scored - 1) as f64;
                    let err = (x - want).abs() / want.max(kc * 10.0);
                    max_err = max_err.max(err);
                    if scored >= 50 {
                        max_late = max_late.max(err);
                    }
                    if scored as f64 == 2.0 * h {
                        let q = (-x).exp();
                        let lead = q * -(-kc * ahead as f64).exp_m1();
                        let got = b.scale_lead(x);
                        // On discrete rows the sums' age is read as their
                        // rows less one, the scale's to within a row of it:
                        // a row's decay, `κ q`, on `N` and so on the lead.
                        assert!(
                            (got - lead).abs() < 2.0 * kc * q,
                            "{h} {i}: {got} vs {lead}"
                        );
                    }
                }
            }
            // Measured 2026-10-25: 0.69% at 50 and 0.17% at 200 with unit
            // weights; with weights `1 ± 0.5` the ratio is a weighted one,
            // 7.5% over the first rows and 2.9% from the 50th.
            let (all, late) = if uneven { (0.1, 0.04) } else { (0.01, 0.01) };
            assert!(
                max_err < all && max_late < late,
                "{h} {uneven}: {max_err} {max_late}"
            );
            // Filled: below the reading's resolution with unit weights. With
            // uneven ones the ratio is a weighted one, which the last rows'
            // weights move: 60 half-lives in it read `q = e^{−x}` of 4.2e-8
            // (2026-10-25), a share within that of its steady form.
            if uneven {
                assert!((-b.elapsed()).exp() < 1e-6, "{h}: {}", b.elapsed());
            } else {
                assert_eq!(b.elapsed(), f64::INFINITY, "{h}: filled");
            }
            assert_eq!(b.scale_lead(f64::INFINITY), 0.0);
            // Young again after a restart, then a gap of many half-lives.
            let mut young = Breaks::new();
            for _ in 0..(h as usize) {
                young.update(lcg(&mut s), lam, 1.0, None);
            }
            assert!(young.elapsed().is_finite());
            young.age(lam.powf(20.0 * h));
            assert_eq!(young.elapsed(), f64::INFINITY);
        }
        let (mut once, mut s) = (Breaks::new(), 3u64);
        for _ in 0..50 {
            once.update(lcg(&mut s), 1.0, 1.0, None);
        }
        assert!(once.s1 > 0.0);
        assert_eq!(once.elapsed(), 0.0);
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

    /// `|A|·|B|` and `A·B`, `p × p` row-major.
    fn abs_mul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
        crate::sandwich::rounding::abs_mul(a, b, p)
    }

    /// [`TwinFit`]'s run replayed with running bounds on its rounding
    /// (docs/PLAN.md task 274): its two accumulators of `(x, y)`
    /// ([`crate::ewcov::rounding::Rounding`], each decayed by the same
    /// factor the definition's weights are products of), and its three
    /// sums `H = Σ ω ω' z zᵀ` about its origin, each decayed by a product of
    /// two factors (one rounding of it, one of the partial sum) and added to
    /// a term of four roundings (`z`'s subtraction, `w²`, two products) and
    /// one of the sum.
    struct TwinTrack {
        fast: crate::ewcov::rounding::Rounding,
        slow: crate::ewcov::rounding::Rounding,
        origin: Option<[f64; 2]>,
        h: [Vec<f64>; 3],
        eh: [Vec<f64>; 3],
    }

    impl TwinTrack {
        fn new() -> Self {
            Self {
                fast: crate::ewcov::rounding::Rounding::new(3, 0.0),
                slow: crate::ewcov::rounding::Rounding::new(3, 0.0),
                origin: None,
                h: [vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]],
                eh: [vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]],
            }
        }

        fn update(&mut self, x: &[f64; 2], y: f64, lam: f64, w: f64) {
            use crate::oracle::U;
            let slow = slow_factor(lam);
            let factors = [lam * lam, slow * slow, lam * slow];
            for ((h, eh), f) in self.h.iter_mut().zip(self.eh.iter_mut()).zip(factors) {
                for (hv, ev) in h.iter_mut().zip(eh.iter_mut()) {
                    let aged = f * *hv;
                    *ev = f * *ev + 2.0 * U * aged.abs();
                    *hv = aged;
                }
            }
            if w <= 0.0 {
                self.fast.decay(lam);
                self.slow.decay(slow);
                return;
            }
            let row = [x[0], x[1], y];
            self.fast.update(&row, lam, w);
            self.slow.update(&row, slow, w);
            let o = *self.origin.get_or_insert(*x);
            let z = [1.0, x[0] - o[0], x[1] - o[1]];
            for q in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        let t = w * w * z[a] * z[b];
                        let v = a * 3 + b;
                        self.h[q][v] += t;
                        self.eh[q][v] += 4.0 * U * t.abs() + U * self.h[q][v].abs();
                    }
                }
            }
        }
    }

    /// One fit by weighted least squares from the rows at one memory's
    /// weights (`pick`), `z = zs(row)`, by faer's LU: its coefficients, its
    /// Gram and right-hand side, the sums' weight and squared weight, the
    /// residuals' weighted sum of squares, and bounds on the rounding of
    /// each sum -- each weight a product of `age` decays (`age·u`), each term
    /// two products more, each addition `u` of its partial sum.
    #[allow(clippy::type_complexity)]
    fn def_fit(
        rows: &[([f64; 2], f64, f64, f64, f64)],
        ages: &[f64],
        pick: fn(&([f64; 2], f64, f64, f64, f64)) -> f64,
        zs: &dyn Fn(&([f64; 2], f64, f64, f64, f64)) -> Vec<f64>,
    ) -> (Vec<f64>, [Vec<f64>; 4], [f64; 6]) {
        use crate::oracle::U;
        let p = zs(&rows[0]).len();
        let (mut g, mut r, mut eg, mut er) = (
            vec![0.0; p * p],
            vec![0.0; p],
            vec![0.0; p * p],
            vec![0.0; p],
        );
        let (mut sw, mut sw2, mut esw, mut esw2) = (0.0, 0.0, 0.0, 0.0);
        for (row, age) in rows.iter().zip(ages) {
            let om = pick(row);
            let z = zs(row);
            for a in 0..p {
                let t = om * z[a] * row.1;
                r[a] += t;
                er[a] += (age + 2.0) * U * t.abs() + U * r[a].abs();
                for b in 0..p {
                    let t = om * z[a] * z[b];
                    g[a * p + b] += t;
                    eg[a * p + b] += (age + 2.0) * U * t.abs() + U * g[a * p + b].abs();
                }
            }
            sw += om;
            esw += age * U * om + U * sw;
            sw2 += om * om;
            esw2 += (2.0 * age + 1.0) * U * om * om + U * sw2;
        }
        let beta = crate::oracle::solve(&g, &r);
        let (mut ssr, mut essr) = (0.0, 0.0);
        for (row, age) in rows.iter().zip(ages) {
            let z = zs(row);
            let fit: f64 = (0..p).map(|a| beta[a] * z[a]).sum();
            let e = row.1 - fit;
            let de = crate::oracle::gamma(p + 1)
                * (row.1.abs() + (0..p).map(|a| (beta[a] * z[a]).abs()).sum::<f64>());
            let t = pick(row) * e * e;
            ssr += t;
            essr += pick(row) * 2.0 * e.abs() * de + (age + 2.0) * U * t + U * ssr;
        }
        (beta, [g, r, eg, er], [sw, sw2, ssr, esw, esw2, essr])
    }

    /// The fast and slow fits' slopes, intercepts, residual spread and the
    /// cross sum against their definitions: weighted least squares by
    /// faer's LU (`crate::oracle`) from every row at each memory's weight,
    /// and the statistic built from them by the module's formula -- with an
    /// intercept, about the first feature's level 3 (the statistic is the
    /// same in any frame of `z`'s origin, the model's being its first row).
    ///
    /// Task 274: within both sides' rounding, where it was `1e-6` relative.
    /// The statistic `δ'V⁻¹δ/s²` moves by `(2|y|ᵀΔδ + |y|ᵀΔV|y|)/s²`
    /// (`y = V⁻¹δ`) and `s²`'s share, and by each side's own form of it.
    /// The model's ([`TwinTrack`]): each fit's slopes from its moments by a
    /// Cholesky solve and its level `m_y − β'(m_x − c)`; `A = G⁻¹` by a
    /// Cholesky solve per column (`|ΔA| ≤ |A|(ΔG + E)|A|`, `G` from the
    /// moments and the weight); `V = A_fH_ffA_f + A_sH_ssA_s − A_fH_fsA_s −
    /// (·)ᵀ` through each factor's error, each triple product's `γ_{2k}`, and
    /// the four terms' cancellation, three roundings of each; `s²` from the
    /// slow moments, `Δvar_y + 2|b|ᵀΔc + |b|ᵀΔC|b|` and Kish's factor; each
    /// mapped to the test's frame, whose shift from the model's origin is
    /// under 1. The definition's: its sums ([`def_fit`]) through each fit's
    /// normal equations, its inverses by their residuals and its sums, each
    /// `d_t` its products' and difference's rounding and its factors'
    /// errors, `V = Σ d_t d_tᵀ` its terms' and partial sums', its residuals'
    /// sum of squares; and the oracle's LU form of `δ'V⁻¹δ`.
    /// Measured on aarch64 (macOS): at most 4.1e-4 (with an intercept, where the model's
    /// variance is four terms that cancel to rounding and is bounded term by term) and 8.1e-3
    /// (through the origin) of the bound.
    #[test]
    fn the_wald_distance_is_the_two_fits_by_definition() {
        use crate::oracle::{self, U, abs_form, gamma};
        // (x, y, weight, its fast weight now, its slow weight now)
        type Row = ([f64; 2], f64, f64, f64, f64);
        let mut t = TwinFit::new(2);
        let mut track = TwinTrack::new();
        let mut s = 9u64;
        let lam = 0.97_f64;
        let slow = slow_factor(lam);
        let mut rows: Vec<Row> = Vec::new();
        let mut born: Vec<usize> = Vec::new();
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
            track.update(&x, y, lam, w);
            rows.iter_mut().for_each(|r| {
                r.3 *= lam;
                r.4 *= slow;
            });
            rows.push((x, y, w, w, w));
            born.push(i);
            if i < 30 || i % 37 != 0 {
                continue;
            }
            let ages: Vec<f64> = born.iter().map(|&b| (i - b) as f64).collect();
            let o = track.origin.unwrap();
            for intercept in [true, false] {
                let zs = |row: &Row| -> Vec<f64> {
                    if intercept {
                        vec![1.0, row.0[0] - 3.0, row.0[1]]
                    } else {
                        vec![row.0[0], row.0[1]]
                    }
                };
                let p = if intercept { 3 } else { 2 };
                // Each fit by weighted least squares, faer's LU.
                let (bf, [gf, rf, egf, erf], _) = def_fit(&rows, &ages, |r| r.3, &zs);
                let (bs, [gs, rs, egs, ers], [ws, ws2, ssr_s, ews, ews2, essr]) =
                    def_fit(&rows, &ages, |r| r.4, &zs);
                // The difference is `Σ_t d_t ε_t`, `d_t = (A_f ω_f,t − A_s
                // ω_s,t) z_t`: its variance over the noise's is `Σ d_t d_t'`,
                // each row's `d_t` built from the inverses outright.
                let (af, a_s) = (oracle::inverse(&gf), oracle::inverse(&gs));
                let daf: Vec<f64> = {
                    let r = oracle::inverse_error(&gf);
                    let m = abs_mul(
                        &abs_mul(&af.iter().map(|v| v.abs()).collect::<Vec<_>>(), &egf, p),
                        &af.iter().map(|v| v.abs()).collect::<Vec<_>>(),
                        p,
                    );
                    r.iter().zip(&m).map(|(a, b)| a + b).collect()
                };
                let das: Vec<f64> = {
                    let r = oracle::inverse_error(&gs);
                    let m = abs_mul(
                        &abs_mul(&a_s.iter().map(|v| v.abs()).collect::<Vec<_>>(), &egs, p),
                        &a_s.iter().map(|v| v.abs()).collect::<Vec<_>>(),
                        p,
                    );
                    r.iter().zip(&m).map(|(a, b)| a + b).collect()
                };
                let mut var = vec![0.0; p * p];
                let mut evar = vec![0.0; p * p];
                for (row, age) in rows.iter().zip(&ages) {
                    let z = zs(row);
                    let d: Vec<f64> = (0..p)
                        .map(|a| {
                            (0..p)
                                .map(|b| (af[a * p + b] * row.3 - a_s[a * p + b] * row.4) * z[b])
                                .sum()
                        })
                        .collect();
                    let dd: Vec<f64> = (0..p)
                        .map(|a| {
                            (0..p)
                                .map(|b| {
                                    let (tf, ts) = (
                                        (af[a * p + b] * row.3 * z[b]).abs(),
                                        (a_s[a * p + b] * row.4 * z[b]).abs(),
                                    );
                                    (gamma(p + 3) + age * U) * (tf + ts)
                                        + (daf[a * p + b] * row.3 + das[a * p + b] * row.4)
                                            * z[b].abs()
                                })
                                .sum()
                        })
                        .collect();
                    for a in 0..p {
                        for b in 0..p {
                            var[a * p + b] += d[a] * d[b];
                            evar[a * p + b] += dd[a] * d[b].abs()
                                + d[a].abs() * dd[b]
                                + U * (d[a] * d[b]).abs()
                                + U * var[a * p + b].abs();
                        }
                    }
                }
                let ns = ws * ws / ws2;
                let s2 = ssr_s / ws * ns / (ns - p as f64);
                let delta: Vec<f64> = (0..p).map(|a| bf[a] - bs[a]).collect();
                let want = oracle::quad_form(&var, &delta) / s2;
                let got = t.wald(&[0, 1], intercept, 1.0).unwrap();
                // The definition's errors in `δ` and `s²`.
                let fe_f = oracle::forward_error(&gf, &bf, &egf, &erf);
                let fe_s = oracle::forward_error(&gs, &bs, &egs, &ers);
                let own_f = oracle::solve_error(&gf, &rf, &bf);
                let own_s = oracle::solve_error(&gs, &rs, &bs);
                let dd_test: Vec<f64> = (0..p)
                    .map(|a| fe_f[a] + fe_s[a] + own_f[a] + own_s[a] + U * delta[a].abs())
                    .collect();
                // `s² = ssr/W·n/(n − p)`, `n = W²/W₂`: the sums' errors, and
                // `n`'s through `p/(n − p)` of its relative error.
                let dn = 2.0 * ews / ws + ews2 / ws2 + 2.0 * U;
                let ds2_test = s2 * (essr / ssr_s + ews / ws + 4.0 * U)
                    + s2 * (p as f64) / (ns - p as f64) * dn;
                // The model's, in its origin's frame, then the test's.
                let (fa, sa) = (&track.fast, &track.slow);
                let cols: Vec<usize> = vec![0, 1];
                let lib_fit = |m: &crate::ewcov::rounding::Rounding| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
                    // (coefficients, their bound, G, ΔG) in the model's frame.
                    let mom = |i: usize, j: usize| -> (f64, f64) {
                        if intercept {
                            (m.c[i * 3 + j], m.ec[i * 3 + j])
                        } else {
                            let r = m.c[i * 3 + j] + m.m[i] * m.m[j];
                            (r, m.ec[i * 3 + j] + m.m[i].abs() * m.em[j] + m.em[i] * m.m[j].abs() + U * ((m.m[i] * m.m[j]).abs() + r.abs()))
                        }
                    };
                    let k = cols.len();
                    let a: Vec<f64> = (0..k * k).map(|ij| mom(cols[ij / k], cols[ij % k]).0).collect();
                    let da: Vec<f64> = (0..k * k).map(|ij| mom(cols[ij / k], cols[ij % k]).1).collect();
                    let b: Vec<f64> = (0..k).map(|i| mom(cols[i], 2).0).collect();
                    let db: Vec<f64> = (0..k).map(|i| mom(cols[i], 2).1).collect();
                    let beta = oracle::solve(&a, &b);
                    let e = oracle::cholesky_backward(&a);
                    let da_e: Vec<f64> = da.iter().zip(&e).map(|(x, y)| x + y).collect();
                    let dbeta = oracle::forward_error(&a, &beta, &da_e, &db);
                    let w = m.w;
                    let pk = if intercept { 3 } else { 2 };
                    let mut g = vec![0.0; pk * pk];
                    let mut dg = vec![0.0; pk * pk];
                    if intercept {
                        let dev = [m.m[0] - o[0], m.m[1] - o[1]];
                        g[0] = w;
                        dg[0] = m.ew * w + U * w;
                        for a2 in 0..2 {
                            g[a2 + 1] = w * dev[a2];
                            g[(a2 + 1) * 3] = g[a2 + 1];
                            let e1 = (dev[a2] * w).abs() * (m.ew + 2.0 * U) + w * m.em[a2];
                            dg[a2 + 1] = e1;
                            dg[(a2 + 1) * 3] = e1;
                            for b2 in 0..2 {
                                let v = m.c[a2 * 3 + b2] + dev[a2] * dev[b2];
                                g[(a2 + 1) * 3 + b2 + 1] = w * v;
                                dg[(a2 + 1) * 3 + b2 + 1] = (w * v).abs() * (m.ew + 3.0 * U)
                                    + w * (m.ec[a2 * 3 + b2] + dev[a2].abs() * m.em[b2] + m.em[a2] * dev[b2].abs());
                            }
                        }
                        let level = m.m[2] - beta[0] * dev[0] - beta[1] * dev[1];
                        let dlevel = m.em[2]
                            + (0..2).map(|a2| beta[a2].abs() * m.em[a2] + dev[a2].abs() * dbeta[a2]).sum::<f64>()
                            + gamma(3) * (m.m[2].abs() + (beta[0] * dev[0]).abs() + (beta[1] * dev[1]).abs());
                        (vec![level, beta[0], beta[1]], vec![dlevel, dbeta[0], dbeta[1]], g, dg)
                    } else {
                        for ij in 0..4 {
                            g[ij] = w * a[ij];
                            dg[ij] = w * da[ij] + (w * a[ij]).abs() * (m.ew + U);
                        }
                        (beta, dbeta, g, dg)
                    }
                };
                let (cf, dcf, gfm, dgfm) = lib_fit(fa);
                let (cs, dcs, gsm, dgsm) = lib_fit(sa);
                let inv_err = |g: &[f64], dg: &[f64]| -> (Vec<f64>, Vec<f64>) {
                    let a = oracle::inverse(g);
                    let aa: Vec<f64> = a.iter().map(|v| v.abs()).collect();
                    let e = oracle::cholesky_backward(g);
                    let tot: Vec<f64> = dg.iter().zip(&e).map(|(x, y)| x + y).collect();
                    let d = abs_mul(&abs_mul(&aa, &tot, p), &aa, p);
                    (a, d)
                };
                let (afm, dafm) = inv_err(&gfm, &dgfm);
                let (asm, dasm) = inv_err(&gsm, &dgsm);
                // The sums `H` in the coefficients' coordinates: through the
                // origin, `x = c + (x − c)` maps them.
                let hmap = |q: usize| -> (Vec<f64>, Vec<f64>) {
                    if intercept {
                        (track.h[q].clone(), track.eh[q].clone())
                    } else {
                        let tm = [o[0], 1.0, 0.0, o[1], 0.0, 1.0];
                        let mut h = vec![0.0; 4];
                        let mut e = vec![0.0; 4];
                        for a2 in 0..2 {
                            for b2 in 0..2 {
                                for i3 in 0..3 {
                                    for j3 in 0..3 {
                                        let c = tm[a2 * 3 + i3] * tm[b2 * 3 + j3];
                                        h[a2 * 2 + b2] += c * track.h[q][i3 * 3 + j3];
                                        e[a2 * 2 + b2] += c.abs()
                                            * (track.eh[q][i3 * 3 + j3]
                                                + 2.0 * U * track.h[q][i3 * 3 + j3].abs());
                                    }
                                }
                            }
                        }
                        (h, e)
                    }
                };
                let (hff, ehff) = hmap(0);
                let (hss, ehss) = hmap(1);
                let (hfs, ehfs) = hmap(2);
                let abs = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.abs()).collect() };
                let term = |a: &[f64],
                            da: &[f64],
                            h: &[f64],
                            dh: &[f64],
                            b: &[f64],
                            db: &[f64]|
                 -> (Vec<f64>, Vec<f64>) {
                    let (aa, ab, ah) = (abs(a), abs(b), abs(h));
                    let mag = abs_mul(&abs_mul(&aa, &ah, p), &ab, p);
                    let e1 = abs_mul(&abs_mul(da, &ah, p), &ab, p);
                    let e2 = abs_mul(&abs_mul(&aa, dh, p), &ab, p);
                    let e3 = abs_mul(&abs_mul(&aa, &ah, p), db, p);
                    let err = (0..p * p)
                        .map(|ij| e1[ij] + e2[ij] + e3[ij] + gamma(2 * p) * mag[ij])
                        .collect();
                    (mag, err)
                };
                let (mff, eff) = term(&afm, &dafm, &hff, &ehff, &afm, &dafm);
                let (mss, ess) = term(&asm, &dasm, &hss, &ehss, &asm, &dasm);
                let (mfs, efs) = term(&afm, &dafm, &hfs, &ehfs, &asm, &dasm);
                let dvar_lib: Vec<f64> = (0..p * p)
                    .map(|ij| {
                        let ji = (ij % p) * p + ij / p;
                        eff[ij]
                            + ess[ij]
                            + efs[ij]
                            + efs[ji]
                            + 3.0 * U * (mff[ij] + mss[ij] + mfs[ij] + mfs[ji])
                    })
                    .collect();
                let ddel_lib: Vec<f64> = (0..p)
                    .map(|a| dcf[a] + dcs[a] + U * (cf[a] - cs[a]).abs())
                    .collect();
                // The slow fit's `s²` and Kish's factor.
                let bsl = &cs[if intercept { 1 } else { 0 }..];
                let mom_s = |i: usize, j: usize| -> (f64, f64) {
                    if intercept {
                        (sa.c[i * 3 + j], sa.ec[i * 3 + j])
                    } else {
                        let r = sa.c[i * 3 + j] + sa.m[i] * sa.m[j];
                        (
                            r,
                            sa.ec[i * 3 + j]
                                + sa.m[i].abs() * sa.em[j]
                                + sa.em[i] * sa.m[j].abs()
                                + U * ((sa.m[i] * sa.m[j]).abs() + r.abs()),
                        )
                    }
                };
                let resid = mom_s(2, 2).0 - (0..2).map(|a2| bsl[a2] * mom_s(a2, 2).0).sum::<f64>();
                let dresid = mom_s(2, 2).1
                    + 2.0 * (0..2).map(|a2| bsl[a2].abs() * mom_s(a2, 2).1).sum::<f64>()
                    + (0..4)
                        .map(|ij| bsl[ij / 2].abs() * mom_s(ij / 2, ij % 2).1 * bsl[ij % 2].abs())
                        .sum::<f64>()
                    + gamma(3)
                        * (mom_s(2, 2).0.abs()
                            + (0..2)
                                .map(|a2| (bsl[a2] * mom_s(a2, 2).0).abs())
                                .sum::<f64>());
                let nsm = sa.w * sa.w / sa.q;
                let dns = nsm * (2.0 * sa.ew + sa.eq + 2.0 * U);
                let ds2_lib = s2
                    * (dresid / resid.abs()
                        + (p as f64) * dns / (nsm * (nsm - p as f64))
                        + 4.0 * U);
                // The model's frame to the test's: with an intercept, the
                // level at the origin `c` against the level at `(3, 0)`.
                let shift = [3.0 - o[0], -o[1]];
                let (ddel_m, dvar_m) = if intercept {
                    let r_abs = [
                        1.0,
                        shift[0].abs(),
                        shift[1].abs(),
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ];
                    let rt: Vec<f64> = (0..9).map(|ij| r_abs[(ij % 3) * 3 + ij / 3]).collect();
                    let dd = oracle::abs_times(&r_abs, &ddel_lib);
                    let dv = abs_mul(&abs_mul(&r_abs, &dvar_lib, 3), &rt, 3);
                    (dd, dv)
                } else {
                    (ddel_lib, dvar_lib)
                };
                let y = oracle::solve(&var, &delta);
                let ddel: Vec<f64> = (0..p).map(|a| dd_test[a] + ddel_m[a]).collect();
                let dvar: Vec<f64> = (0..p * p).map(|ij| evar[ij] + dvar_m[ij]).collect();
                let dquad = abs_form(&y, &dvar, &y)
                    + 2.0 * y.iter().zip(&ddel).map(|(a, b)| a.abs() * b).sum::<f64>()
                    + oracle::quad_error(&var, &delta, &oracle::cholesky_backward(&var))
                    + oracle::quad_form_error(&var, &delta);
                let bound = dquad / s2 + want * (ds2_test + ds2_lib) / s2 + 2.0 * U * want;
                oracle::assert_within(
                    got,
                    want,
                    bound,
                    format_args!("intercept {intercept}, row {i}"),
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

    /// On a quantile fit the CUSUM by its definition: `Σ ω c / sqrt(Σ ω²)`
    /// over the rows with an indicator `c`, from every such row at its
    /// present weight, under a decay, uneven weights, zeros and rows with no
    /// indicator -- read from the first indicator, before the studentized
    /// residuals' scale is ready.
    #[test]
    fn a_quantile_fits_cusum_sums_its_indicators() {
        let mut b = Breaks::new().with_indicator(true);
        let mut s = 31u64;
        let lam = 0.97;
        // (c, weight now) of the summed rows.
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let q = 0.3f64;
        for i in 0..200 {
            let v = lcg(&mut s);
            let below = lcg(&mut s) < q - 0.5;
            let c =
                (i % 13 != 6).then(|| (f64::from(u8::from(below)) - q) / (q * (1.0 - q)).sqrt());
            let w = if i % 7 == 3 { 0.0 } else { 1.0 + lcg(&mut s) };
            b.update(v, lam, w, c);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            if let (Some(c), true) = (c, w > 0.0) {
                rows.push((c, w));
            }
            let z1: f64 = rows.iter().map(|r| r.1 * r.0).sum();
            let c2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
            if c2 == 0.0 {
                assert!(b.cusum(1.0).is_none(), "row {i}");
                continue;
            }
            let got = b.cusum(1.0).unwrap();
            assert!((got - z1 / c2.sqrt()).abs() < 1e-10, "row {i}");
        }
        // The indicators' long-run variance reads their own weights.
        assert_eq!(b.long_run(), Some(1.0));
        assert_eq!(b.long_run_residuals(), Some(1.0));
        assert!(b.has_shape(0));
    }

    /// On a quantile fit under a horizon, two long-run factors by their
    /// definition (review round 7, A-2): the CUSUM's over the indicators'
    /// lag products, and `break_wald`'s over the studentized residuals'
    /// -- Bartlett's weights on the products of the scored rows' `ω z`, a
    /// pair `l` scored rows apart, at their present weights, under a
    /// decay, zeros, rows with no indicator and a cleared run. Residuals
    /// that carry a lag and indicators that carry less of it read two
    /// factors that differ; without an indicator the two are one.
    #[test]
    fn a_quantile_fits_break_wald_reads_the_residuals_long_run_variance() {
        let lags = 3usize;
        let mut b = Breaks::with_lags(lags).with_indicator(true);
        let mut plain = Breaks::with_lags(lags);
        let mut s = 44u64;
        let lam = 0.98;
        let q = 0.5f64;
        // Per run: (z, weight now) of the scored rows, and (c, weight
        // now) of the rows with an indicator.
        let mut zs: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
        let mut cs: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
        let mut prev = 0.0;
        let mut differed = 0;
        for i in 0..500 {
            let v = 0.8 * prev + lcg(&mut s);
            prev = v;
            let c =
                (i % 11 != 5).then(|| (f64::from(u8::from(v < 0.0)) - q) / (q * (1.0 - q)).sqrt());
            let w = if i % 9 == 4 { 0.0 } else { 1.0 + lcg(&mut s) };
            if i == 300 {
                b.clear_lags();
                plain.clear_lags();
                zs.push(Vec::new());
                cs.push(Vec::new());
            }
            let z = b.studentized(v);
            b.update(v, lam, w, c);
            plain.update(v, lam, w, None);
            zs.iter_mut().flatten().for_each(|r| r.1 *= lam);
            cs.iter_mut().flatten().for_each(|r| r.1 *= lam);
            if w > 0.0 {
                if let Some(z) = z {
                    zs.last_mut().unwrap().push((z, w));
                }
                if let Some(c) = c {
                    cs.last_mut().unwrap().push((c, w));
                }
            }
            if i < 40 || i % 23 != 0 {
                continue;
            }
            let factor = |runs: &Vec<Vec<(f64, f64)>>| {
                let s2: f64 = runs.iter().flatten().map(|r| r.1 * r.1).sum();
                let mut p = 0.0;
                for run in runs {
                    for l in 1..=lags {
                        let bw = 1.0 - l as f64 / (lags as f64 + 1.0);
                        for t in l..run.len() {
                            p += bw * run[t].1 * run[t - l].1 * run[t].0 * run[t - l].0;
                        }
                    }
                }
                1.0 + 2.0 * p / s2
            };
            let (want_z, want_c) = (factor(&zs), factor(&cs));
            let tol = 1e-9;
            assert!(
                (b.long_run_residuals().unwrap() - want_z).abs() < tol,
                "row {i}"
            );
            assert!((b.long_run().unwrap() - want_c).abs() < tol, "row {i}");
            // The plain accumulator over the same residuals reads the same
            // factor, from one set of products.
            assert!((plain.long_run().unwrap() - want_z).abs() < tol, "row {i}");
            assert_eq!(plain.long_run(), plain.long_run_residuals());
            if (want_z - want_c).abs() > 0.1 {
                differed += 1;
            }
        }
        assert!(
            differed >= 10,
            "the residuals' factor is not the indicators'"
        );
        assert!(b.has_shape(lags) && !b.has_shape(0));
        assert!(plain.has_shape(lags));
    }

    /// The residuals' products are written on a quantile fit under lags
    /// alone, and read back; a restored one with a product too few, or
    /// more rows in its ring than lags, is refused.
    #[test]
    fn the_residuals_products_are_written_on_a_quantile_fit_under_lags() {
        let named = |b: &Breaks| rmp_serde::to_vec_named(b).unwrap();
        let has = |bytes: &[u8], key: &[u8]| bytes.windows(key.len()).any(|w| w == key);
        // Twice over: the scale studentizes from its tenth Kish row on.
        let rows = [
            1.0, -0.5, 0.7, -1.2, 0.3, 0.9, -0.1, 0.4, -0.8, 1.1, 0.2, -0.6, 0.5, -0.3,
        ];
        let mut fed: Vec<Breaks> = Vec::new();
        for (lags, on) in [(0, false), (3, false), (0, true), (3, true)] {
            let mut b = Breaks::with_lags(lags).with_indicator(on);
            for v in rows.iter().chain(&rows).copied() {
                b.update(v, 0.9, 1.0, Some(if v < 0.0 { 1.0 } else { -1.0 }));
            }
            let bytes = named(&b);
            assert_eq!(
                has(&bytes, b"pz"),
                lags > 0 && on,
                "lags {lags}, indicator {on}"
            );
            assert_eq!(has(&bytes, b"ringz"), lags > 0 && on);
            let back: Breaks = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(back, b);
            assert!(back.has_shape(lags));
            fed.push(b);
        }
        let quantile = fed.pop().unwrap();
        assert_eq!(quantile.pz.len(), 3);
        assert_eq!(quantile.ringz.len(), 3);
        let bad = |edit: fn(&mut Breaks)| {
            let mut c = quantile.clone();
            edit(&mut c);
            !c.has_shape(3)
        };
        assert!(bad(|c| c.pz.pop().map(drop).unwrap()));
        assert!(bad(|c| c.pz[0] = f64::NAN));
        assert!(bad(|c| c.ringz.push_back(0.0)));
        assert!(bad(|c| c.ringz[1] = f64::INFINITY));
    }

    /// The lag products of the indicators can cancel the variance outright
    /// (`c = 2` then `c = −1` one lag apart, Bartlett's weight 1/2): then
    /// there is no long-run variance and no CUSUM, where dividing would
    /// read infinity.
    #[test]
    fn a_long_run_variance_of_zero_reads_nothing() {
        let mut b = Breaks::with_lags(1).with_indicator(true);
        b.update(1.0, 1.0, 1.0, Some(2.0));
        b.update(1.0, 1.0, 1.0, Some(-1.0));
        assert!(b.long_run().is_none());
        assert!(b.cusum(1.0).is_none());
    }

    /// No studentized residual for a residual that is not a number, nor
    /// against a spread of 0, which is what a fit that has predicted every
    /// row exactly leaves; and no CUSUM over a null share that is not
    /// positive.
    #[test]
    fn no_studentized_residual_without_a_number_or_a_spread() {
        let mut b = Breaks::new();
        for _ in 0..20 {
            b.update(0.0, 0.9, 1.0, None);
        }
        assert!(b.studentized(1.0).is_none(), "a spread of 0");
        for v in [1.0, -0.5, 0.7, -1.2] {
            b.update(v, 0.9, 1.0, None);
        }
        assert!(b.studentized(1.0).is_some());
        assert!(b.studentized(f64::NAN).is_none());
        assert!(b.studentized(f64::INFINITY).is_none());
        assert!(b.cusum(1.0).is_some() && b.cusum(0.0).is_none());
    }

    /// The CUSUM of squares by its guards, on restored sums: nothing
    /// without a weight, nor over a null share that is not positive, nor
    /// while the fourth moment is at most 1, nor where the lag products
    /// leave a variance that is not positive -- each a case where the
    /// division would read a number that is not the statistic.
    #[test]
    fn the_cusum_of_squares_reads_only_a_positive_variance() {
        // m4 = 3, excess 2.
        let base = Breaks {
            s1: 1.0,
            s2: 1.0,
            z2: 2.0,
            z4: 3.0,
            w4: 1.0,
            ..Breaks::default()
        };
        assert!(base.has_shape(0));
        let got = base.cusum_sq(1.0).unwrap();
        assert!((got - 1.0 / 2f64.sqrt()).abs() < 1e-15);
        // No weight, though a sum of squared weights is restored beside it.
        let no_weight = Breaks {
            s1: 0.0,
            ..base.clone()
        };
        assert!(no_weight.cusum_sq(1.0).is_none());
        // A share below 0 against lag products below 0: their product is
        // positive, and no variance.
        let lagged = |p2: f64, z4: f64| Breaks {
            lags: 1,
            p1: vec![0.0],
            p2: vec![p2],
            z4,
            ..base.clone()
        };
        assert!(lagged(-10.0, 3.0).has_shape(1));
        assert!(lagged(-10.0, 3.0).cusum_sq(-1.0).is_none());
        // A fourth moment of 1, beside lag products that would leave a
        // positive variance.
        assert!(lagged(4.0, 1.0).cusum_sq(1.0).is_none());
        // Lag products that cancel the variance exactly: 2 + 2·(1/2)·(−2).
        assert!(lagged(-2.0, 3.0).cusum_sq(1.0).is_none());
        assert!(lagged(-1.0, 3.0).cusum_sq(1.0).is_some());
    }

    /// A restored accumulator at the right lags is still refused for a sum
    /// that is not finite, or a weight's that is negative.
    #[test]
    fn a_restored_accumulator_must_be_finite() {
        let mut b = Breaks::with_lags(2);
        for v in [
            1.0, -0.5, 0.7, -1.2, 0.3, 0.9, -0.1, 0.4, -0.8, 1.1, 0.2, -0.6,
        ] {
            b.update(v, 0.9, 1.0, None);
        }
        assert!(b.has_shape(2) && !b.has_shape(1));
        let bad = |edit: fn(&mut Breaks)| {
            let mut c = b.clone();
            edit(&mut c);
            !c.has_shape(2)
        };
        assert!(bad(|c| c.z1 = f64::NAN));
        assert!(bad(|c| c.s2 = -1.0));
        assert!(bad(|c| c.vq = f64::INFINITY));
        assert!(bad(|c| c.c2 = -1.0));
        assert!(bad(|c| c.p2[1] = f64::NAN));
    }

    /// A horizon-free accumulator writes none of the horizon's fields, so
    /// its state keeps the form it had before they existed; one under a
    /// horizon writes them and reads back the same.
    #[test]
    fn the_horizons_fields_are_written_only_under_one() {
        let named = |b: &Breaks| rmp_serde::to_vec_named(b).unwrap();
        let has = |bytes: &[u8], key: &[u8]| bytes.windows(key.len()).any(|w| w == key);
        let plain = named(&Breaks::new());
        assert!(!has(&plain, b"lags") && !has(&plain, b"ring1"));
        let mut h = Breaks::with_lags(3);
        for v in [
            1.0, -0.5, 0.7, -1.2, 0.3, 0.9, -0.1, 0.4, -0.8, 1.1, 0.2, -0.6,
        ] {
            h.update(v, 0.9, 1.0, None);
        }
        let bytes = named(&h);
        assert!(has(&bytes, b"lags"));
        let back: Breaks = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, h);
        assert!(back.has_shape(3));
    }

    /// The slow memory's factor is the fast one's fourth root: a half-life
    /// four times as long, exactly where the root is a double.
    #[test]
    fn the_slow_factor_is_the_fourth_root() {
        assert_eq!(slow_factor(0.0625), 0.5);
        assert_eq!(slow_factor(1.0), 1.0);
        let lam = 0.97f64;
        assert!((slow_factor(lam).powi(4) - lam).abs() < 1e-15);
    }

    /// `mat_mul` is the matrix product, against the definition's sums.
    #[test]
    fn mat_mul_is_the_product() {
        let a = [1.0, 2.0, -1.0, 0.5, 3.0, 1.5, -2.0, 0.0, 4.0];
        let b = [2.0, -1.0, 0.0, 1.0, 1.0, 3.0, 0.5, -0.5, 2.0];
        let got = mat_mul(&a, &b, 3);
        for i in 0..3 {
            for j in 0..3 {
                let want: f64 = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
                assert_eq!(got[i * 3 + j], want, "({i}, {j})");
            }
        }
    }

    /// A restored pair is refused unless it is shaped for the features and
    /// every kept number is finite.
    #[test]
    fn a_restored_pair_must_be_shaped_and_finite() {
        let mut t = TwinFit::new(2);
        let mut s = 4u64;
        for _ in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            t.update(&x, x[0] + lcg(&mut s), 0.9, 1.0);
        }
        assert!(t.has_shape(2) && !t.has_shape(1) && !t.has_shape(3));
        let bad = |edit: fn(&mut TwinFit)| {
            let mut c = t.clone();
            edit(&mut c);
            !c.has_shape(2)
        };
        assert!(bad(|c| c.hff[4] = f64::NAN));
        assert!(bad(|c| c.hfs[0] = f64::INFINITY));
        assert!(bad(|c| c.origin[1] = f64::NAN));
        assert!(bad(|c| c.origin = vec![0.0]));
        assert!(bad(|c| c.hss.pop().map_or((), |_| ())));
        assert!(bad(|c| c.fast = crate::EwCov::new(2).without_runs()));
        assert!(bad(|c| c.slow = crate::EwCov::new(4).without_runs()));
    }

    /// A row with a weight of 0 -- the first included -- or a target or a
    /// feature that is not a number only ages the pair: it sets no origin
    /// and moves no sum.
    #[test]
    fn a_row_the_fits_cannot_learn_only_ages_them() {
        let mut t = TwinFit::new(2);
        let mut aged = TwinFit::new(2);
        t.update(&[1.0, 2.0], 3.0, 0.9, 0.0);
        aged.age(0.9);
        assert_eq!(t, aged);
        let mut s = 8u64;
        for _ in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + lcg(&mut s);
            t.update(&x, y, 0.9, 1.0);
            aged.update(&x, y, 0.9, 1.0);
        }
        t.update(&[1.0, 2.0], f64::NAN, 0.8, 1.0);
        t.update(&[f64::NAN, 2.0], 3.0, 0.8, 1.0);
        t.update(&[1.0, 2.0], 3.0, 0.8, 0.0);
        for _ in 0..3 {
            aged.age(0.8);
        }
        assert_eq!(t, aged);
    }

    /// The distance waits for each fit to have more than `k` rows of Kish's
    /// size: at a fast factor of 1/2 the fast fit never has more than 3,
    /// while the slow one has about 11.
    #[test]
    fn the_distance_waits_for_the_fast_fit() {
        let mut t = TwinFit::new(2);
        let mut s = 6u64;
        for _ in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            t.update(&x, x[0] - x[1] + lcg(&mut s), 0.5, 1.0);
        }
        assert!(t.fast.n_kish().unwrap() <= 3.0 && t.slow.n_kish().unwrap() > 3.0);
        assert!(t.wald(&[0, 1], true, 1.0).is_none());
        assert!(t.wald(&[0], true, 1.0).is_some(), "two coefficients");
    }

    /// No distance where the slow fit leaves no residual -- a constant
    /// target -- nor for a long-run factor that is not positive.
    #[test]
    fn no_distance_without_a_residual_or_a_long_run_variance() {
        let mut flat = TwinFit::new(1);
        let mut t = TwinFit::new(1);
        let mut s = 12u64;
        for _ in 0..300 {
            let x = [lcg(&mut s)];
            flat.update(&x, 2.0, 0.95, 1.0);
            t.update(&x, 2.0 * x[0] + lcg(&mut s), 0.95, 1.0);
        }
        assert!(flat.wald(&[0], true, 1.0).is_none());
        assert!(t.wald(&[0], true, 1.0).is_some());
        assert!(t.wald(&[0], true, 0.0).is_none());
        assert!(t.wald(&[0], true, -1.0).is_none());
        assert!(t.wald(&[0], true, f64::NAN).is_none());
    }

    /// Multiplying every weight by one constant moves no distance (task 270
    /// (g)): the fits are weighted means, `G` scales with the weight and
    /// `H` with its square, so `A H A` does not move. Before the distance
    /// waited for `2k` rows ([`TwinFit::wald`]) it read a variance that is
    /// singular in exact arithmetic on the rows between `k` and `2k`, so
    /// whether it factorized was the rounding's to decide, and it did
    /// differently at another scale: 1 to 3 rows of 400 in the release
    /// probe's stream, at x100, x0.01 and even x2, and 343 times here. At
    /// exactly `2k` rows the variance is a cancellation of terms far larger
    /// than it, and another scale's rounding moved the distance by up to
    /// 1.5e-7 of itself; from there on by less than 1e-8.
    #[test]
    fn the_distance_is_free_of_the_weights_scale() {
        let mut flips = 0;
        for seed in 0..20u64 {
            for mixed in [false, true] {
                for c in [100.0, 0.01, 2.0] {
                    let (mut a, mut b) = (TwinFit::new(2), TwinFit::new(2));
                    let mut s = 40 + seed;
                    for i in 0..120 {
                        let x = [lcg(&mut s), lcg(&mut s) + 1.0];
                        let y = 0.5 + x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
                        let w = match (mixed, i % 11) {
                            (true, 3) => 0.0,
                            (true, _) => 1.0 + lcg(&mut s),
                            (false, _) => 1.0,
                        };
                        let lam = if i % 7 == 0 { 0.9 } else { 0.97 };
                        a.update(&x, y, lam, w);
                        b.update(&x, y, lam, c * w);
                        for intercept in [true, false] {
                            for idx in [&[0usize][..], &[1], &[0, 1]] {
                                match (a.wald(idx, intercept, 1.0), b.wald(idx, intercept, 1.0)) {
                                    (Some(u), Some(v)) => assert!(
                                        (u - v).abs() <= 1e-6 * u.abs().max(1.0),
                                        "seed {seed} x{c} row {i}: {u} vs {v}"
                                    ),
                                    (None, None) => {}
                                    (u, v) => {
                                        flips += 1;
                                        eprintln!(
                                            "seed {seed} x{c} mixed {mixed} row {i} {intercept} {idx:?}: {u:?} vs {v:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(flips, 0);
    }

    /// The difference's variance over the noise's is `Σ_t d_t d_t'`, `d_t =
    /// (A_f ω_f,t − A_s ω_s,t) z_t`, and `Σ_t d_t z_t' = A_f G_f − A_s G_s =
    /// I − I = 0`: the `d_t` lie in the null space of the `k × n` design, so
    /// over `n` rows the variance has rank at most `n − k`, and is singular
    /// until `n = 2k`. By its definition (faer's eigensolver, `crate::oracle`)
    /// its smallest eigenvalue is the rounding's below `2k` rows and not at
    /// `2k`; the distance waits for them.
    #[test]
    fn the_difference_has_no_variance_before_twice_its_coefficients() {
        let lam = 0.9_f64;
        let slow = slow_factor(lam);
        let mut early = 0;
        for seed in 0..30u64 {
            let mut t = TwinFit::new(2);
            let mut s = 70 + seed;
            // (z with the intercept, fast weight now, slow weight now)
            let mut rows: Vec<([f64; 3], f64, f64)> = Vec::new();
            for n in 1..=8usize {
                let x = [lcg(&mut s), lcg(&mut s)];
                let w = 1.0 + lcg(&mut s);
                t.update(&x, x[0] + 0.3 * lcg(&mut s), lam, w);
                rows.iter_mut().for_each(|r| {
                    r.1 *= lam;
                    r.2 *= slow;
                });
                rows.push(([1.0, x[0], x[1]], w, w));
                let got = t.wald(&[0, 1], true, 1.0);
                if n < 6 {
                    early += usize::from(got.is_some());
                }
                if n < 3 {
                    continue;
                }
                let gram = |pick: fn(&([f64; 3], f64, f64)) -> f64| {
                    let mut g = vec![0.0; 9];
                    for r in &rows {
                        for a in 0..3 {
                            for b in 0..3 {
                                g[a * 3 + b] += pick(r) * r.0[a] * r.0[b];
                            }
                        }
                    }
                    crate::oracle::inverse(&g)
                };
                let (af, a_s) = (gram(|r| r.1), gram(|r| r.2));
                // `var` by its definition, and the scale of its two halves,
                // `Σ |A_f ω_f z|² + Σ |A_s ω_s z|²`, which its rounding is
                // relative to.
                let (mut var, mut scale) = (vec![0.0; 9], 0.0);
                for r in &rows {
                    let half = |a_: &[f64], om: f64, a: usize| -> f64 {
                        (0..3).map(|b| a_[a * 3 + b] * om * r.0[b]).sum()
                    };
                    let d: Vec<f64> = (0..3)
                        .map(|a| half(&af, r.1, a) - half(&a_s, r.2, a))
                        .collect();
                    scale += (0..3)
                        .map(|a| half(&af, r.1, a).powi(2) + half(&a_s, r.2, a).powi(2))
                        .sum::<f64>();
                    for a in 0..3 {
                        for b in 0..3 {
                            var[a * 3 + b] += d[a] * d[b];
                        }
                    }
                }
                let (ev, _) = crate::oracle::sym_eigen(&var);
                let null = ev.iter().filter(|e| e.abs() < 1e-12 * scale).count();
                // Rank at most `n − 3`, and generically exactly that.
                assert_eq!(
                    null,
                    6usize.saturating_sub(n).min(3),
                    "seed {seed}, {n} rows: {ev:?} at {scale}"
                );
            }
        }
        assert_eq!(early, 0, "distances read off a singular variance");
    }

    /// One row after a decay of `lam_gap`, past 40 rows at 0.97, under
    /// weight scales 2, 100 and 0.01 beside unit weights: how often the
    /// distance is a number on one side and none on the other over the 20
    /// rows after the gap, and the largest relative move between the sides,
    /// on the row where the count since the last wipe is exactly `2k` and
    /// on the rows past it. Every subset of the two features, with the
    /// intercept and without, 20 seeds. `split` ages the gap in two steps
    /// of `√lam_gap` before the row, as zero-weight rows do.
    fn across_a_gap(lam_gap: f64, split: bool) -> (usize, f64, f64) {
        let (mut flips, mut at, mut past) = (0, 0.0f64, 0.0f64);
        for seed in 0..20u64 {
            for c in [2.0, 100.0, 0.01] {
                let (mut a, mut b) = (TwinFit::new(2), TwinFit::new(2));
                let mut s = 900 + seed;
                for i in 0..61 {
                    let x = [lcg(&mut s), lcg(&mut s) + 1.0];
                    let y = 0.5 + x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
                    let lam = match (i, split) {
                        (40, false) => lam_gap,
                        (40, true) => {
                            let half = lam_gap.sqrt();
                            for t in [&mut a, &mut b] {
                                t.age(half);
                            }
                            half
                        }
                        _ => 0.97,
                    };
                    a.update(&x, y, lam, 1.0);
                    b.update(&x, y, lam, c);
                    if i < 40 {
                        continue;
                    }
                    for intercept in [true, false] {
                        for idx in [&[0usize][..], &[1], &[0, 1]] {
                            let k = (idx.len() + usize::from(intercept)) as u64;
                            match (a.wald(idx, intercept, 1.0), b.wald(idx, intercept, 1.0)) {
                                (Some(u), Some(v)) => {
                                    let r = (u - v).abs() / u.abs().max(1.0);
                                    if a.fresh_rows == 2 * k {
                                        at = at.max(r);
                                    } else {
                                        past = past.max(r);
                                    }
                                }
                                (None, None) => {}
                                _ => flips += 1,
                            }
                        }
                    }
                }
            }
        }
        (flips, at, past)
    }

    /// A gap that leaves the old rows below the rounding of the new ones
    /// starts the `2k`-row wait over (task 270, R1-2): the count was of
    /// every row since the first, so on the `k` to `2k` rows after such a
    /// gap the distance was read off a variance whose rank was the new
    /// rows' alone. Before: at a gap's decay of `2^-60`, no flip but a
    /// relative move of 8.6e-4 between weight scales; at 0, 121 flips
    /// between a number and none. Now no flip at any gap from `2^-30` to
    /// 0, nor at a gap split in two, and within 1e-6 past the `2k`-th row
    /// after the wipe (4.1e-10 at most). On that row itself the variance
    /// is the cancellation the struct's docs describe, of terms far larger
    /// than it, which moves with another scale's rounding: 1.3e-5 at most
    /// here, where a fresh pair fed the same rows moves 2.5e-7 (its origin,
    /// the first row it saw, is not the one kept from before the gap).
    #[test]
    fn a_gap_that_wipes_the_history_starts_the_wait_over() {
        for e in [60, 0, 30, 40, 45, 53, 80, 120] {
            let lam_gap = if e == 0 { 0.0 } else { 2f64.powi(-e) };
            for split in [false, true] {
                let (flips, at, past) = across_a_gap(lam_gap, split);
                let what = format!("a gap of 2^-{e}, split {split}");
                assert_eq!(flips, 0, "{what}");
                assert!(past <= 1e-6, "{what}: {past:e} past the 2k-th row");
                assert!(at <= 5e-5, "{what}: {at:e} on the 2k-th row");
            }
        }
    }

    /// [`history_wiped`]'s share is of the weight after the row, `carried +
    /// w`, the old rows' and the new one's together: a history at `2^-26.5`
    /// of a row's weight has a share of `2^-26.5 / (1 + 2^-26.5)`, whose
    /// square is `2^-53 (1 - 2^-25.5)` to first order, so the row wipes it.
    /// Read against the weight the row adds less the history, `w - carried`,
    /// the square would be `2^-53 (1 + 2^-25.5)`, and it would not; the
    /// band between the two is 2^-25.5 wide, far from the rounding.
    #[test]
    fn the_wipe_reads_the_share_of_the_weight_after_the_row() {
        let edge = 2f64.powi(-53).sqrt();
        for w in [1.0, 4.0, 0.01] {
            assert!(history_wiped(edge * w, w), "w = {w}");
            assert!(
                !history_wiped(edge * w * (1.0 + 2f64.powi(-24)), w),
                "w = {w}"
            );
        }
    }

    /// [`history_wiped`]'s edge, a share whose square is `2^-53`, from both
    /// sides: a row of weight 1 meeting a history carried at `2^-27` wipes
    /// it, at `2^-26` it does not; so does a row of weight 4 meeting four
    /// times either, the share being free of the scale; a first row wipes
    /// the history of none. And through the fits: a gap's decay putting
    /// the carried weight just past the edge waits `2k` rows again, just
    /// short of it reads a distance on a row before `2k` after the gap.
    #[test]
    fn the_wipe_is_a_shares_square_below_two_to_the_minus_53() {
        assert!(history_wiped(2f64.powi(-27), 1.0));
        assert!(!history_wiped(2f64.powi(-26), 1.0));
        assert!(history_wiped(4.0 * 2f64.powi(-27), 4.0));
        assert!(!history_wiped(4.0 * 2f64.powi(-26), 4.0));
        assert!(history_wiped(0.0, 1.0));
        let edge = 2f64.powf(-26.5);
        let after = |carried: f64| -> Vec<bool> {
            let mut t = TwinFit::new(2);
            let mut s = 3u64;
            let mut read = Vec::new();
            for i in 0..47 {
                let x = [lcg(&mut s), lcg(&mut s) + 1.0];
                let y = 0.5 + x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
                // The gap's decay that carries `carried` of the fast fit's
                // weight into the row.
                let lam = if i == 40 {
                    carried / t.fast.n_eff()
                } else {
                    0.97
                };
                t.update(&x, y, lam, 1.0);
                if i >= 40 {
                    read.push(t.wald(&[0, 1], true, 1.0).is_some());
                }
            }
            read
        };
        // `k` = 3, the gap's own row the first after it. Either way the
        // fast fit's Kish size is above `k` from the fourth row; wiped, the
        // distance waits for the sixth, `2k`, kept, it is read from the
        // fourth.
        let wiped = after(edge * 0.99);
        assert_eq!(wiped, [false, false, false, false, false, true, true]);
        let kept = after(edge * 1.01);
        assert_eq!(kept, [false, false, false, true, true, true, true]);
    }

    /// A state saved before the wipe count existed loads with every row it
    /// learned, which is what a stream no gap wiped counts: the same state,
    /// so the same bytes saved (rule 5's exact conversion of a kept
    /// fixture), whether it was saved inside its first `2k` rows or past
    /// them.
    #[test]
    fn a_state_without_the_count_loads_with_every_row_it_learned() {
        for rows in [3, 40] {
            let mut t = TwinFit::new(2);
            let mut s = 8u64;
            for _ in 0..rows {
                let x = [lcg(&mut s), lcg(&mut s)];
                t.update(&x, x[0] + 0.3 * lcg(&mut s), 0.97, 1.0);
            }
            assert_eq!(t.fresh_rows, rows);
            let mut v = serde_json::to_value(&t).unwrap();
            assert!(v.as_object_mut().unwrap().remove("fresh_rows").is_some());
            let back: TwinFit = serde_json::from_value(v).unwrap();
            assert_eq!(back, t, "{rows} rows");
            let named = |t: &TwinFit| rmp_serde::to_vec_named(t).unwrap();
            assert_eq!(named(&back), named(&t));
        }
    }
}
