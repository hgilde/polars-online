//! Adaptive conformal intervals (ENHANCEMENTS E36): a tracked quantile of the
//! conformity score `|resid|`, with a long-run coverage guarantee that needs
//! no distributional assumption and survives a shift in the residuals.
//!
//! One scalar recursion per slot. With target coverage `1 − α`, score
//! `s_t = |resid_t|` and step `η_t`:
//!
//! ```text
//! err_t = 1{s_t > q_t}                      the interval missed
//! q_{t+1} = max(0, q_t + η_t · (w_t / w̄_t) · (err_t − α))
//! ```
//!
//! with `w̄_t` the stream's EW mean weight, this row's included: the mean
//! over every row the layer is given with a residual and a weight, which
//! takes in the row that warm-starts the radius and a delayed row shown no
//! interval, though neither is scored. So a step reads a row's weight
//! against the stream's, and multiplying every weight by one constant moves
//! nothing (docs/PLAN.md task 147: the step was `η·w`, and the same rows at
//! weight 100 swung the band, width sd 3.10 against 0.21).
//!
//! which is online gradient descent on the pinball loss at level `1 − α` —
//! the "P" term of Angelopoulos, Candès & Tibshirani (2023), *Conformal PID
//! control for time series prediction*, and the quantile-tracking form of
//! Gibbs & Candès (2021)'s adaptive conformal inference. The interval for the
//! next row is `pred ± q`, read *before* the row (its own score is folded in
//! afterwards), so it is out-of-sample the way every other output here is.
//!
//! **Guarantee.** For a constant step `η` and scores in `[0, B]`, telescoping
//! the recursion gives `|q_{T+1} − q_1| = η·|Σ_{t≤T}(err_t − α)|`, and
//! `q` never leaves `[−η, B + η]`, so
//!
//! ```text
//! |(1/T) Σ_{t≤T} err_t − α| ≤ (B + η) / (η T)
//! ```
//!
//! for *every* sequence of scores: the long-run miscoverage is `α` whatever
//! the residual distribution is, and however it moves. The clamp at zero
//! only ever raises `q`, so it can only add coverage. With weights the sum
//! is `Σ (w_t/w̄_t)(err_t − α)`, so what is held to `α` is the miscoverage
//! weighted by `w/w̄`: a heavy row's miss counts for more, and the share of
//! rows missed can sit on either side of `α` when the weights follow the
//! misses (0.954 of rows covered against 0.900 weighted, with weight 5 on
//! the wild rows; review round 4, CA5). The step here is `η_t = rate ·
//! σ_t`, with `σ_t` the slot's EW residual standard deviation before the
//! row, so that `rate` is unit-free and the radius moves at the scale of
//! the errors it brackets; the bound then holds in `σ`-weighted form, at
//! the same rate.
//!
//! `q` starts where a Gaussian would put it, `σ · Φ⁻¹(1 − α/2)`, on the first
//! row that has a residual and a finite `σ > 0`, and tracks from there. The
//! realized coverage is an EW mean of `1{s_t ≤ q_t}` on the same clock as the
//! model, also read before the row.
//!
//! **Under a label delay** (docs/records/CONFORMAL-STEP.md, docs/PLAN.md
//! task 342). A row whose label arrives `n` releases after it was shown its
//! interval feeds the recursion `n` steps late: linearized about the true
//! quantile, the radius error follows `e_{t+1} = e_t − k·e_{t−n}` with
//! `k = rate · g(α)` ([`gauss_gain`]), stable only for `k·n` below about
//! `π/2` and free of overshoot below about `1/e`. The delay-aware step
//! divides the rate by `1 + rate · g(α) · n̄ / c`, with `n̄` the EW mean of
//! the rows waiting at each release and `c` = [`HELD_GAIN_BOUND`]: the
//! fixed step when nothing waits, and `k·n̄` held below `c` when much does.
//! `n̄` is a slow mean, not the count at the instant: a step that followed
//! the instant count would take its smallest steps exactly when the misses
//! pile up, and the miss rate the guarantee holds is weighted by the steps.

use serde::{Deserialize, Serialize};

/// Tracked conformal radius and realized coverage for one output slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "Saved")]
pub struct Conformal {
    /// Miscoverage target `α`; the interval aims at coverage `1 − α`.
    alpha: f64,
    /// Step per unit of `σ`: the step itself, or under the delay-aware rule
    /// the step it takes when nothing waits.
    rate: f64,
    /// The radius. NaN until the warm start.
    q: f64,
    /// EW mean of `1{|resid| ≤ q}` and its weight.
    cov: f64,
    cov_w: f64,
    /// The stream's EW mean weight, and the EW count behind it: every row
    /// with a residual and a weight, the warm-start row and a delayed row
    /// shown no interval included, though neither is scored. What a row's
    /// step reads its weight against (docs/PLAN.md task 147).
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
    /// The EW mean of the rows still waiting for their labels at each
    /// release, the released row aside, on the same rows and decay as
    /// `w_mean`: `n̄` of the delay-aware step (docs/PLAN.md task 342). 0
    /// wherever the stream passes no count, so the step is `rate`'s; not
    /// written while 0, so no other state's bytes move (schema 67).
    #[serde(default, skip_serializing_if = "is_zero")]
    held_mean: f64,
    /// `g(α) = 2φ(Φ⁻¹(1 − α/2))`, the step's loop gain per unit of rate on
    /// Gaussian residuals, derived from `α` when the tracker is made or
    /// loaded and never written: a value of the configuration, not of the
    /// stream.
    #[serde(skip_serializing)]
    gain: f64,
}

/// A [`Conformal`] as its file holds it: every field but the derived gain.
#[derive(Deserialize)]
struct Saved {
    alpha: f64,
    rate: f64,
    q: f64,
    cov: f64,
    cov_w: f64,
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
    #[serde(default)]
    held_mean: f64,
}

impl From<Saved> for Conformal {
    fn from(s: Saved) -> Self {
        Self {
            alpha: s.alpha,
            rate: s.rate,
            q: s.q,
            cov: s.cov,
            cov_w: s.cov_w,
            w_mean: s.w_mean,
            w_rows: s.w_rows,
            held_mean: s.held_mean,
            gain: gauss_gain(s.alpha),
        }
    }
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// The delay-aware step's constant `c`: the step is cut so that its loop
/// gain times the rows waiting stays below `c`, between the edge at which
/// the radius stops overshooting (`1/e`) and the edge of stability (`π/2`).
pub const HELD_GAIN_BOUND: f64 = 0.5;

impl Conformal {
    /// `coverage` is the target `1 − α`, strictly inside `(0, 1)`; `rate` the
    /// step per unit of `σ`, `> 0`.
    pub fn new(coverage: f64, rate: f64) -> Result<Self, String> {
        let mut no = crate::Refusals::new();
        if !(coverage > 0.0 && coverage < 1.0) {
            no.refuse(format!(
                "conformal must be above 0 and below 1, but it is {coverage}. Set conformal to \
                 the share of rows whose target the interval should cover, such as 0.9."
            ));
        }
        if !(rate > 0.0 && rate.is_finite()) {
            no.refuse(format!(
                "conformal_rate must be finite and above 0, but it is {rate}. Set \
                 conformal_rate to how fast the interval's width adapts, such as the default \
                 0.05."
            ));
        }
        no.finish()?;
        Ok(Self {
            alpha: 1.0 - coverage,
            rate,
            q: f64::NAN,
            cov: 0.0,
            cov_w: 0.0,
            w_mean: 0.0,
            w_rows: 0.0,
            held_mean: 0.0,
            gain: gauss_gain(1.0 - coverage),
        })
    }

    /// The half-width of the current interval, once warm-started.
    pub fn radius(&self) -> Option<f64> {
        self.q.is_finite().then_some(self.q)
    }

    /// Realized coverage so far: the EW fraction of scored rows whose
    /// interval contained the target.
    pub fn coverage(&self) -> Option<f64> {
        (self.cov_w > 0.0).then_some(self.cov)
    }

    /// Fold one row in. `resid` is the row's out-of-sample residual (NaN when
    /// there was no prediction to score), `sigma` the slot's EW residual
    /// standard deviation *before* the row (NaN until one exists), `lam` the
    /// model's decay factor for the row and `w` the row's weight.
    ///
    /// A row with no residual or no weight only ages the coverage: it is
    /// not evidence either way. The first row with a residual and a finite
    /// `sigma > 0` sets the radius and is not scored — no interval was
    /// emitted for it. Every later row is scored against the radius it was
    /// given, then moves it.
    pub fn update(&mut self, resid: f64, sigma: f64, lam: f64, w: f64) {
        self.update_against(resid, sigma, lam, w, None, 0);
    }

    /// [`Self::update`] for a row whose label arrives after the row was
    /// shown its interval (`embargo`, `label_clock`): the row is scored
    /// against the radius it was *shown*, `shown`, not the radius the
    /// recursion has reached since, and the step moves the current radius by
    /// that error -- the delayed-feedback form of adaptive conformal
    /// inference. Scored against the radius at release, the coverage
    /// described intervals nobody was shown (review 2026-09-12, C21's other
    /// half; docs/PLAN.md task 112). `None` is the current radius, which is
    /// what `shown` is with no delay; `Some(NaN)`, a row shown no interval
    /// because none existed yet, is no evidence either way: it ages the
    /// coverage and takes no step.
    ///
    /// `held` is the count of rows still waiting for their labels as this
    /// one is released, itself aside, for the delay-aware step: its EW mean
    /// `n̄`, over the rows `w̄` is taken over, divides the step,
    ///
    /// ```text
    /// η = rate · σ / (1 + rate · g(α) · n̄ / c)
    /// ```
    ///
    /// so that the loop gain `η·g/σ` times the rows in flight stays below
    /// `c` ([`HELD_GAIN_BOUND`]); see [`gauss_gain`]. 0 on every row leaves
    /// `n̄` at 0 and the step at `rate · σ` to the bit, which is how the
    /// fixed rate is kept.
    pub fn update_against(
        &mut self,
        resid: f64,
        sigma: f64,
        lam: f64,
        w: f64,
        shown: Option<f64>,
        held: usize,
    ) {
        if !resid.is_finite() || w <= 0.0 {
            self.cov_w *= lam;
            self.w_rows *= lam;
            return;
        }
        // The stream's mean weight, this row's included, so the first row
        // reads its own weight as 1; and the mean of the rows waiting, on the
        // same count.
        self.w_rows = lam * self.w_rows + 1.0;
        self.w_mean += (w - self.w_mean) / self.w_rows;
        // (A tracker that has never been passed a count skips the division,
        // whose result would be 0: the fixed step costs what it did.)
        if held > 0 || self.held_mean != 0.0 {
            self.held_mean += (held as f64 - self.held_mean) / self.w_rows;
        }
        let usable_sigma = sigma.is_finite() && sigma > 0.0;
        if !self.q.is_finite() {
            if usable_sigma {
                self.q = sigma * norm_ppf(1.0 - 0.5 * self.alpha);
            }
            self.cov_w *= lam;
            return;
        }
        let against = shown.unwrap_or(self.q);
        if !against.is_finite() {
            self.cov_w *= lam;
            return;
        }
        let s = resid.abs();
        let miss = s > against;
        let cw = lam * self.cov_w + w;
        self.cov = (lam * self.cov_w * self.cov + w * f64::from(!miss)) / cw;
        self.cov_w = cw;
        // A slot whose residuals have all been exactly zero has sigma 0 and
        // takes no step: the radius waits until there is an error scale.
        let mut eta = if usable_sigma { self.rate * sigma } else { 0.0 };
        // Nothing has waited: the fixed step, with no division to round it.
        if self.held_mean > 0.0 {
            eta /= 1.0 + self.rate * self.gain * self.held_mean / HELD_GAIN_BOUND;
        }
        let rel = if self.w_mean > 0.0 {
            w / self.w_mean
        } else {
            1.0
        };
        self.q = (self.q + eta * rel * (f64::from(miss) - self.alpha)).max(0.0);
    }

    /// `n̄`: the EW mean of the rows waiting at each release, 0 where none
    /// has been counted.
    pub fn held_mean(&self) -> f64 {
        self.held_mean
    }
}

/// `g(α) = 2φ(Φ⁻¹(1 − α/2))`: the density of `|Z|`, `Z` standard normal,
/// at its `1 − α` quantile. On Gaussian residuals a step of `rate · σ` moves
/// the expected miss rate by `rate · g` per unit of radius error in `σ`, so
/// `rate · g` is the tracker's loop gain: 0.636 at a coverage of 50%, 0.206
/// at 90%, 0.029 at 99%.
///
/// `φ`'s exponential is [`exp_neg`], basic arithmetic alone, so the gain is
/// the same bits on every platform and a radius stepped by it continues to
/// the bit wherever its state is loaded. `Φ⁻¹` is [`norm_ppf`], whose tails
/// beyond `p = 0.02425` (coverage above 95.15%) read the platform's `ln`, as
/// the warm start does.
pub fn gauss_gain(alpha: f64) -> f64 {
    let z = norm_ppf(1.0 - 0.5 * alpha);
    // 1 / sqrt(2π)
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    2.0 * INV_SQRT_2PI * exp_neg(0.5 * z * z)
}

/// `e^{−x}` for `x ≥ 0` in basic arithmetic, the same bits on every
/// platform: `x = k·ln 2 + r` with `0 ≤ r < ln 2` (`k` whole), `e^{−r}` by
/// its Taylor series to the term below the last bit, scaled by `2^{−k}`
/// exactly. Within a few ulps of the true value; NaN for NaN, 0 past the
/// smallest normal.
pub fn exp_neg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x > 708.0 {
        return 0.0;
    }
    // ln 2 split in two (Cody and Waite): `k·LN2_HI` is exact for every `k`
    // here, so `r` keeps its digits however large `k` is.
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let k = (x / std::f64::consts::LN_2).floor();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    // e^{-r}, r in [0, ln 2): the terms fall below 2^-53 of the sum by the
    // 18th.
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=20 {
        term *= -r / f64::from(i);
        sum += term;
    }
    // 2^-k, k in 0..=1021, exactly: the exponent field of a normal double.
    sum * f64::from_bits(((1023 - k as i64) as u64) << 52)
}

/// Inverse of the standard normal CDF, `Φ⁻¹(p)` for `p` in `(0, 1)`.
///
/// Acklam's rational approximation (relative error below `1.2e-9` over the
/// whole range), evaluated in a fixed order so the same input gives the same
/// bits on every platform. Good to every purpose here — it seeds a radius
/// that is then tracked — and a test pins it against known quantiles.
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
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

    /// Box–Muller from two uniforms.
    fn gauss(state: &mut u64) -> f64 {
        let (u1, u2) = (lcg(state).max(1e-300), lcg(state));
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn norm_ppf_matches_known_quantiles() {
        // (p, Φ⁻¹(p)) to 15 digits; both branches of the approximation.
        // The lower tail at `2⁻⁷` rather than 0.01 (docs/PLAN.md task 273):
        // `1 − 2⁻⁷` is exact, so `norm_ppf(1 − p)` takes the log of the same
        // number `norm_ppf(p)` does and the two are negatives to the bit on
        // any libm. At 0.01 the upper tail read `ln(1 − 0.99)`, 8.7e-18
        // from 0.01, and agreed with Apple's `ln` alone: 4.5 ulps apart with
        // it moved by 2. Φ⁻¹(2⁻⁷) from MPFR at 200 bits.
        for (p, z) in [
            (0.5, 0.0),
            (0.75, 0.6744897501960817),
            (0.9, 1.2815515655446004),
            (0.95, 1.6448536269514722),
            (0.975, 1.959963984540054),
            (0.99, 2.3263478740408408),
            (0.995, 2.5758293035489004),
            (0.9995, 3.2905267314919255),
            (0.0078125, -2.417559016236505),
        ] {
            let got = norm_ppf(p);
            assert!(
                (got - z).abs() <= 2e-9 * z.abs().max(1.0),
                "ppf({p}) = {got}, want {z}"
            );
            assert_eq!(norm_ppf(1.0 - p), -got, "symmetry at {p}");
        }
        assert!(norm_ppf(0.0).is_nan());
        assert!(norm_ppf(1.0).is_nan());
        assert!(norm_ppf(f64::NAN).is_nan());
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Conformal::new(0.0, 0.05).is_err());
        assert!(Conformal::new(1.0, 0.05).is_err());
        assert!(Conformal::new(f64::NAN, 0.05).is_err());
        assert!(Conformal::new(0.9, 0.0).is_err());
        assert!(Conformal::new(0.9, f64::INFINITY).is_err());
        assert!(Conformal::new(0.9, f64::NAN).is_err());
        assert!(Conformal::new(0.9, 0.05).is_ok());
        // Every problem at once, each a line of its own.
        let e = Conformal::new(1.5, 0.0).err().unwrap_or_default();
        let lines: Vec<&str> = e.lines().collect();
        assert_eq!(lines.len(), 2, "{e}");
        assert!(lines[0].starts_with("conformal must be above 0 and below 1, but it is 1.5."));
        assert!(lines[1].starts_with("conformal_rate must be finite and above 0, but it is 0."));
    }

    #[test]
    fn starts_undefined_and_warm_starts_at_the_gaussian_radius() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        assert_eq!(c.radius(), None);
        assert_eq!(c.coverage(), None);
        // No sigma yet: nothing happens.
        c.update(0.3, f64::NAN, 1.0, 1.0);
        assert_eq!(c.radius(), None);
        // First usable sigma: the radius appears; the row is not scored.
        c.update(0.3, 2.0, 1.0, 1.0);
        let z = norm_ppf(0.95);
        assert_eq!(c.radius(), Some(2.0 * z));
        assert_eq!(c.coverage(), None);
        // From here on every row is scored and moves the radius.
        c.update(0.3, 2.0, 1.0, 1.0);
        assert_eq!(c.coverage(), Some(1.0));
        assert_eq!(c.radius(), Some(2.0 * z + 0.05 * 2.0 * (0.0 - 0.1)));
    }

    #[test]
    fn a_miss_widens_and_a_hit_narrows_by_the_stated_amounts() {
        let mut c = Conformal::new(0.8, 0.1).unwrap();
        c.update(1.0, 1.0, 1.0, 1.0); // warm start at Φ⁻¹(0.9)
        let q0 = c.radius().unwrap();
        c.update(10.0, 1.0, 1.0, 1.0); // miss: + η(1 − α) = 0.1·0.8
        assert!((c.radius().unwrap() - (q0 + 0.1 * 0.8)).abs() < 1e-15);
        let q1 = c.radius().unwrap();
        c.update(0.0, 1.0, 1.0, 1.0); // hit: − ηα = 0.1·0.2
        assert!((c.radius().unwrap() - (q1 - 0.1 * 0.2)).abs() < 1e-15);
        // The step reads the weight against the scored rows' mean, this row's
        // included: three at 1 and one at 3 make 1.5, so the step doubles. A
        // zero-weight row only ages the coverage.
        let q2 = c.radius().unwrap();
        c.update(10.0, 1.0, 1.0, 3.0);
        assert!((c.radius().unwrap() - (q2 + 2.0 * 0.1 * 0.8)).abs() < 1e-14);
        let q3 = c.radius().unwrap();
        let cov = c.coverage().unwrap();
        c.update(10.0, 1.0, 0.5, 0.0);
        assert_eq!(c.radius(), Some(q3));
        assert_eq!(c.coverage(), Some(cov));
    }

    /// Task 147: the same rows at a hundred times the weight give the same
    /// interval and the same coverage -- the step was `η·w`, and they swung
    /// the band.
    #[test]
    fn a_weights_scale_moves_nothing() {
        let (mut one, mut many) = (
            Conformal::new(0.9, 0.05).unwrap(),
            Conformal::new(0.9, 0.05).unwrap(),
        );
        let mut s = 71u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let (r, w) = (4.0 * next() - 2.0, 0.5 + next());
            one.update(r, 1.0, 0.99, w);
            many.update(r, 1.0, 0.99, 100.0 * w);
            let (a, b) = (one.radius().unwrap(), many.radius().unwrap());
            assert!((a - b).abs() <= 1e-12 * (1.0 + a), "{a} against {b}");
        }
        let (a, b) = (one.coverage().unwrap(), many.coverage().unwrap());
        assert!((a - b).abs() <= 1e-12, "{a} against {b}");
    }

    #[test]
    fn the_radius_never_goes_negative() {
        let mut c = Conformal::new(0.5, 1.0).unwrap();
        for _ in 0..100 {
            c.update(0.0, 1.0, 1.0, 1.0);
        }
        assert_eq!(c.radius(), Some(0.0));
        assert_eq!(c.coverage(), Some(1.0));
    }

    /// Against the radius it was shown: a row inside the interval it was
    /// shown counts as covered however the radius has moved since, the step
    /// moves the current radius, a row shown no interval is no evidence, and
    /// `None` is `update` to the bit (docs/PLAN.md task 112).
    #[test]
    fn a_delayed_row_is_scored_against_the_radius_it_was_shown() {
        let mut c = Conformal::new(0.9, 0.1).unwrap();
        c.update(0.5, 1.0, 1.0, 1.0); // the warm start: q = 1.645
        let q0 = c.radius().unwrap();
        // |resid| 1.2 against a shown radius of 1.3: covered, though 1.2 is
        // past a radius of 1.0 it might have moved to since.
        let mut shown = c.clone();
        shown.update_against(1.2, 1.0, 1.0, 1.0, Some(1.3), 0);
        assert_eq!(shown.coverage(), Some(1.0));
        assert!((shown.radius().unwrap() - (q0 - 0.1 * 0.1)).abs() < 1e-15);
        let mut missed = c.clone();
        missed.update_against(1.2, 1.0, 1.0, 1.0, Some(1.1), 0);
        assert_eq!(missed.coverage(), Some(0.0));
        assert!((missed.radius().unwrap() - (q0 + 0.1 * 0.9)).abs() < 1e-15);
        // Shown nothing: no evidence, no step.
        let mut none = c.clone();
        none.update_against(9.0, 1.0, 0.5, 1.0, Some(f64::NAN), 0);
        assert_eq!((none.coverage(), none.radius()), (None, Some(q0)));
        // `None` is the current radius, which is `update`.
        let (mut a, mut b) = (c.clone(), c);
        for (r, w) in [(0.3, 1.0), (2.5, 0.5), (-1.7, 2.0), (0.0, 1.0)] {
            a.update(r, 1.1, 0.97, w);
            b.update_against(r, 1.1, 0.97, w, None, 0);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn coverage_converges_to_the_target_on_any_distribution() {
        // Gaussian, fat-tailed (a Gaussian mixture with 5% wide outliers) and
        // a distribution whose scale doubles half-way: the long-run coverage
        // is the target within the guarantee's bound, and the radius on the
        // fat-tailed stream is not the Gaussian one.
        for (case, alpha) in [(0usize, 0.1), (1, 0.1), (2, 0.05)] {
            let mut c = Conformal::new(1.0 - alpha, 0.05).unwrap();
            let mut s = 11u64 + case as u64;
            let (mut sig2, mut sw) = (0.0f64, 0.0f64);
            let n = 200_000;
            let mut misses = 0.0;
            for t in 0..n {
                let scale = if case == 2 && t >= n / 2 { 2.0 } else { 1.0 };
                let r = match case {
                    1 if lcg(&mut s) < 0.05 => 8.0 * gauss(&mut s),
                    _ => scale * gauss(&mut s),
                };
                let sigma = if sw > 0.0 { sig2.sqrt() } else { f64::NAN };
                if let Some(q) = c.radius() {
                    misses += f64::from(r.abs() > q);
                }
                c.update(r, sigma, 0.999, 1.0);
                // A plain EW variance, the way the stream layer keeps sigma.
                let w_new = 0.999 * sw + 1.0;
                sig2 = (0.999 * sw * sig2 + r * r) / w_new;
                sw = w_new;
            }
            let rate = misses / (n as f64 - 2.0);
            assert!(
                (rate - alpha).abs() < 0.01,
                "case {case}: miss rate {rate} vs alpha {alpha}"
            );
            assert!((c.coverage().unwrap() - (1.0 - alpha)).abs() < 0.03);
            if case == 1 {
                // The Gaussian radius, sigma · Φ⁻¹(0.95), is inflated by the
                // outliers; the tracked one brackets the bulk.
                let gaussian = sig2.sqrt() * norm_ppf(0.95);
                let q = c.radius().unwrap();
                assert!(q < 0.7 * gaussian, "q {q} vs gaussian {gaussian}");
            }
        }
    }

    #[test]
    fn a_shift_in_scale_is_tracked_without_being_told() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        let mut s = 5u64;
        let mut q_before = 0.0;
        for t in 0..40_000 {
            let scale = if t < 20_000 { 1.0 } else { 4.0 };
            let r = scale * gauss(&mut s);
            if t == 19_999 {
                q_before = c.radius().unwrap();
            }
            // Sigma follows with a lag; here it is the true scale so the
            // test isolates the tracking itself.
            c.update(r, scale, 0.999, 1.0);
        }
        let q_after = c.radius().unwrap();
        assert!((q_before - 1.645).abs() < 0.25, "q before {q_before}");
        assert!((q_after - 4.0 * 1.645).abs() < 1.0, "q after {q_after}");
    }

    #[test]
    fn a_zero_sigma_freezes_the_radius() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        c.update(1.0, 1.0, 1.0, 1.0);
        let q = c.radius().unwrap();
        c.update(5.0, 0.0, 1.0, 1.0);
        assert_eq!(c.radius(), Some(q));
        assert_eq!(c.coverage(), Some(0.0));
    }

    #[test]
    fn round_trips_through_msgpack() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        let mut s = 9u64;
        for _ in 0..50 {
            c.update(gauss(&mut s), 1.0, 0.99, 1.0);
        }
        let bytes = rmp_serde::to_vec(&c).unwrap();
        let back: Conformal = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, c);
    }

    // The mutation survivors of the weekly pass (docs/PLAN.md task 158).

    /// The recursion against its definitions, row for row, with each kind of
    /// row that is no evidence mixed in -- no residual, no weight, and a
    /// delayed row shown no interval -- under an uneven decay. The realized
    /// coverage is the weighted EW mean of `1{s ≤ q}` over the scored rows,
    /// every row ageing it; the step reads the row's weight against `w̄`, the
    /// EW mean weight of the rows with a residual and a weight, every row
    /// ageing their count. That is not the scored rows' mean: the row that
    /// warm-starts the radius and a delayed row shown no interval are pushed
    /// into `w̄` though neither is scored. Both are written out as sums over
    /// the rows they hold, each row's age the product of the factors after
    /// it.
    #[test]
    fn the_recursion_is_its_definitions_among_rows_that_are_no_evidence() {
        let (alpha, rate) = (0.2, 0.3);
        let mut c = Conformal::new(1.0 - alpha, rate).unwrap();
        // `(w, hit, age)` per scored row, and `(w, age)` per row in `w̄`.
        let mut scored: Vec<(f64, f64, f64)> = Vec::new();
        let mut means: Vec<(f64, f64)> = Vec::new();
        let mut q = f64::NAN;
        let mut s = 3u64;
        let mut kinds = [0usize; 4];
        for t in 0..300 {
            let lam = 0.9 + 0.09 * lcg(&mut s);
            let resid = 2.0 * gauss(&mut s);
            let w = 0.5 + lcg(&mut s);
            let sigma = 1.0 + 0.5 * lcg(&mut s);
            scored.iter_mut().for_each(|e| e.2 *= lam);
            means.iter_mut().for_each(|e| e.1 *= lam);
            match t % 7 {
                1 => {
                    c.update(f64::NAN, sigma, lam, w);
                    kinds[0] += 1;
                }
                3 => {
                    c.update(resid, sigma, lam, 0.0);
                    kinds[1] += 1;
                }
                5 if q.is_finite() => {
                    c.update_against(resid, sigma, lam, w, Some(f64::NAN), 0);
                    means.push((w, 1.0));
                    kinds[2] += 1;
                }
                _ => {
                    c.update(resid, sigma, lam, w);
                    means.push((w, 1.0));
                    if q.is_finite() {
                        let miss = resid.abs() > q;
                        scored.push((w, f64::from(!miss), 1.0));
                        let wbar = means.iter().map(|(w, a)| w * a).sum::<f64>()
                            / means.iter().map(|(_, a)| a).sum::<f64>();
                        q = (q + rate * sigma * (w / wbar) * (f64::from(miss) - alpha)).max(0.0);
                        kinds[3] += 1;
                    } else {
                        q = sigma * norm_ppf(1.0 - 0.5 * alpha);
                    }
                }
            }
            let got = c.radius().unwrap_or(f64::NAN);
            assert!(
                (got - q).abs() <= 1e-12 * (1.0 + q) || (got.is_nan() && q.is_nan()),
                "row {t}: radius {got} vs {q}"
            );
            let want = (!scored.is_empty()).then(|| {
                scored.iter().map(|(w, hit, a)| w * hit * a).sum::<f64>()
                    / scored.iter().map(|(w, _, a)| w * a).sum::<f64>()
            });
            match (c.coverage(), want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-12, "row {t}: {g} vs {w}"),
                (g, w) => assert_eq!(g, w, "row {t}"),
            }
        }
        assert!(kinds.iter().all(|&k| k > 30), "{kinds:?}");
    }

    /// The radius starts on the first row with a residual and an error
    /// scale: a `σ` of 0, or one that is not finite, starts nothing, and the
    /// next row with a finite `σ > 0` starts it at `σ·Φ⁻¹(1 − α/2)`.
    #[test]
    fn a_sigma_of_zero_or_infinity_does_not_start_the_radius() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        c.update(0.3, 0.0, 1.0, 1.0);
        assert_eq!(c.radius(), None);
        c.update(0.3, f64::INFINITY, 1.0, 1.0);
        assert_eq!(c.radius(), None);
        c.update(0.3, 2.0, 1.0, 1.0);
        assert_eq!(c.radius(), Some(2.0 * norm_ppf(0.95)));
    }

    /// A radius that overflows -- a step of `rate·σ` past the largest double
    /// -- is no radius: the next row with an error scale starts it again at
    /// the Gaussian one and, like the first warm start, is not scored, so it
    /// only ages the coverage, which carries on.
    #[test]
    fn a_radius_that_overflows_starts_again_and_the_coverage_ages() {
        let mut c = Conformal::new(0.9, 1e300).unwrap();
        c.update(1.0, 1e10, 0.5, 1.0); // the warm start
        c.update(1.0, 1.0, 0.5, 1.0); // a hit: the radius steps down to 0
        assert_eq!(c.radius(), Some(0.0));
        c.update(1.0, 1e10, 0.5, 1.0); // a miss, with a step of 1e310
        assert_eq!(c.radius(), None);
        let cov = c.coverage().unwrap();
        c.update(1.0, 1.0, 0.5, 1.0); // starts again
        assert_eq!(c.radius(), Some(norm_ppf(0.95)));
        assert_eq!(c.coverage(), Some(cov));
        c.update(0.1, 1.0, 0.5, 1.0); // a hit
        // The hit, the miss and the hit, aged by the rows after each.
        let want = (0.125 * 1.0 + 0.25 * 0.0 + 1.0 * 1.0) / (0.125 + 0.25 + 1.0);
        assert!((c.coverage().unwrap() - want).abs() < 1e-15, "{c:?}");
    }

    /// `Φ⁻¹` is odd about `1/2`, `Φ⁻¹(1 − p) = −Φ⁻¹(p)`, at the knot where
    /// Acklam's approximation changes branch too: `p = 0.02425` and its
    /// mirror both take the central branch, which is odd by construction.
    /// The tail branch is as accurate there, but on the other side of the
    /// true quantile (`-1.9729610513118845`, from Python's
    /// `statistics.NormalDist`), so a knot taken by both would break the
    /// symmetry by `4.4e-9`.
    #[test]
    fn norm_ppf_is_odd_at_the_knot() {
        let p = 0.02425;
        let (lo, hi) = (norm_ppf(p), norm_ppf(1.0 - p));
        assert!((lo + hi).abs() < 1e-12, "{lo} and {hi}");
        assert!((lo - -1.972_961_051_311_884_5).abs() < 3e-9, "{lo}");
    }

    // The delay-aware step (docs/PLAN.md task 342b).

    /// `exp_neg` against the platform's `exp` over the range the gain reads
    /// and past it, within a few ulps, and its edges.
    #[test]
    fn exp_neg_is_the_exponential() {
        let mut x = 0.0f64;
        let mut worst = 0.0f64;
        while x < 700.0 {
            let (got, want) = (exp_neg(x), (-x).exp());
            worst = worst.max((got - want).abs() / want);
            x += 0.0137 + x * 0.003;
        }
        assert!(worst < 1e-15, "worst relative error {worst}");
        assert_eq!(exp_neg(0.0), 1.0);
        assert_eq!(exp_neg(-1.0), 1.0);
        assert_eq!(exp_neg(1e4), 0.0);
        // 708 itself is still a normal double, `2^-1021` scaled (task 358).
        let (got, want) = (exp_neg(708.0), (-708.0f64).exp());
        assert!(
            want.is_normal() && (got - want).abs() <= 1e-15 * want,
            "{got} {want}"
        );
        assert_eq!(exp_neg(708.0f64.next_up()), 0.0);
        assert!(exp_neg(f64::NAN).is_nan());
    }

    /// `g(α)` is `2φ(Φ⁻¹(1 − α/2))`: the record's values, and the density of
    /// `|Z|` written with the platform's `exp`.
    #[test]
    fn the_gain_is_the_density_of_the_absolute_normal_at_its_quantile() {
        for (cov, g) in [(0.5, 0.636), (0.9, 0.206), (0.95, 0.117), (0.99, 0.029)] {
            let alpha = 1.0 - cov;
            let got = gauss_gain(alpha);
            let z = norm_ppf(1.0 - 0.5 * alpha);
            let want = 2.0 * (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            assert!((got - want).abs() <= 1e-15 * want, "{cov}: {got} vs {want}");
            assert!((got - g).abs() < 5e-4, "{cov}: {got} vs {g}");
        }
    }

    /// Fed a count of 0 on every row, the delay-aware tracker is the fixed
    /// one to the bit, and writes the same bytes.
    #[test]
    fn a_count_of_zero_is_the_fixed_step_to_the_bit() {
        let (mut a, mut b) = (
            Conformal::new(0.9, 0.05).unwrap(),
            Conformal::new(0.9, 0.05).unwrap(),
        );
        let mut s = 17u64;
        for t in 0..3000 {
            let (r, w, lam) = (2.0 * gauss(&mut s), 0.5 + lcg(&mut s), 0.99);
            let shown = (t % 3 == 0).then(|| a.radius().unwrap_or(f64::NAN));
            a.update_against(r, 1.3, lam, w, shown, 0);
            b.update_against(r, 1.3, lam, w, shown, 0);
            c_eq_bits(&a, &b);
        }
        let mut fixed = Conformal::new(0.9, 0.05).unwrap();
        let mut s = 17u64;
        for _ in 0..3000 {
            let (r, w) = (2.0 * gauss(&mut s), 0.5 + lcg(&mut s));
            fixed.update(r, 1.3, 0.99, w);
        }
        assert_eq!(fixed.held_mean(), 0.0);
        let bytes = rmp_serde::to_vec_named(&fixed).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("held_mean"), "a count of 0 is not written");
        assert!(!text.contains("gain"), "the gain is never written");
    }

    fn c_eq_bits(a: &Conformal, b: &Conformal) {
        assert_eq!(
            a.radius().map(f64::to_bits),
            b.radius().map(f64::to_bits),
            "{a:?} vs {b:?}"
        );
    }

    /// `n̄` and the step against their definitions, row for row: `n̄` the
    /// EW mean of the counts over the rows `w̄` is taken over, each row's age
    /// the product of the factors after it; the step `rate·σ / (1 + rate·g·n̄
    /// / c)` times `w / w̄` -- with rows that are no evidence mixed in, which
    /// age the count and add nothing to it.
    #[test]
    fn the_held_mean_and_the_step_are_their_definitions() {
        let (alpha, rate) = (0.1, 0.05);
        let g = gauss_gain(alpha);
        let mut c = Conformal::new(1.0 - alpha, rate).unwrap();
        let mut rows: Vec<(f64, f64, f64)> = Vec::new(); // (w, held, age)
        let mut q = f64::NAN;
        let mut s = 29u64;
        let mut stepped = 0;
        for t in 0..400 {
            let lam = 0.95 + 0.04 * lcg(&mut s);
            let (resid, w, sigma) = (gauss(&mut s), 0.5 + lcg(&mut s), 1.0 + lcg(&mut s));
            let held = (lcg(&mut s) * 300.0) as usize;
            rows.iter_mut().for_each(|e| e.2 *= lam);
            if t % 5 == 2 {
                c.update_against(f64::NAN, sigma, lam, w, None, held);
            } else {
                c.update_against(resid, sigma, lam, w, None, held);
                rows.push((w, held as f64, 1.0));
                let ages: f64 = rows.iter().map(|e| e.2).sum();
                let nbar = rows.iter().map(|e| e.1 * e.2).sum::<f64>() / ages;
                let wbar = rows.iter().map(|e| e.0 * e.2).sum::<f64>() / ages;
                assert!(
                    (c.held_mean() - nbar).abs() <= 1e-12 * (1.0 + nbar),
                    "row {t}: {} vs {nbar}",
                    c.held_mean()
                );
                if q.is_finite() {
                    let miss = f64::from(resid.abs() > q);
                    let eta = rate * sigma / (1.0 + rate * g * nbar / HELD_GAIN_BOUND);
                    q = (q + eta * (w / wbar) * (miss - alpha)).max(0.0);
                    stepped += 1;
                } else {
                    q = sigma * norm_ppf(1.0 - 0.5 * alpha);
                }
            }
            let got = c.radius().unwrap();
            assert!(
                (got - q).abs() <= 1e-12 * (1.0 + q),
                "row {t}: {got} vs {q}"
            );
        }
        assert!(stepped > 300 && c.held_mean() > 50.0, "{stepped} {c:?}");
    }

    /// `held_mean` survives a save, a file without it loads at 0, and the
    /// gain is derived again on load rather than read.
    #[test]
    fn the_held_mean_round_trips_and_an_older_file_loads_at_zero() {
        let mut c = Conformal::new(0.9, 0.05).unwrap();
        let mut s = 5u64;
        for i in 0..200 {
            c.update_against(gauss(&mut s), 1.0, 0.99, 1.0, None, i % 40);
        }
        assert!(c.held_mean() > 0.0);
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let back: Conformal = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.gain.to_bits(), gauss_gain(0.1 - 0.0).to_bits());
        // The same state as a build before the field wrote it.
        let mut old = c.clone();
        old.held_mean = 0.0;
        let older = rmp_serde::to_vec_named(&old).unwrap();
        assert!(older.len() < bytes.len());
        let back: Conformal = rmp_serde::from_slice(&older).unwrap();
        assert_eq!(back.held_mean(), 0.0);
        assert_eq!(back, old);
    }

    /// The derivation's simulation (docs/records/CONFORMAL-STEP.md §2.4):
    /// standard normal residuals, `σ = 1` known, each row's label exactly
    /// `n` releases late. At the fixed 0.05 with 300 rows waiting `k·n =
    /// 0.05 · g · 300 = 3.1`, past the stability edge `π/2`, and the radius
    /// swings: its mean is far above `q*` and its spread is of the order of
    /// `q*` itself. The delay-aware step holds `k·n̄` to `c = 1/2`, where the
    /// radius stays within a few percent of `q*` on average and spreads
    /// about a twentieth of it. The coverage holds near 90% in both: the
    /// guarantee does not depend on the step.
    #[test]
    fn under_a_long_delay_the_auto_step_holds_the_loop_gain_and_the_fixed_one_swings() {
        let (n, rows, burn) = (300usize, 200_000usize, 50_000usize);
        let q_star = norm_ppf(0.95);
        let run = |held_count: bool| {
            let mut c = Conformal::new(0.9, 0.05).unwrap();
            let mut s = 41u64;
            let mut shown: std::collections::VecDeque<(f64, f64)> = Default::default();
            let (mut sum, mut sum2, mut misses, mut m) = (0.0, 0.0, 0.0, 0.0);
            for t in 0..rows {
                let r = gauss(&mut s);
                if shown.len() == n {
                    let (r_old, q_old) = shown.pop_front().unwrap();
                    let held = if held_count { shown.len() } else { 0 };
                    c.update_against(r_old, 1.0, 1.0 - 1.0 / 2000.0, 1.0, Some(q_old), held);
                }
                let q = c.radius().unwrap_or(f64::NAN);
                if t >= burn && q.is_finite() {
                    sum += q;
                    sum2 += q * q;
                    misses += f64::from(r.abs() > q);
                    m += 1.0;
                }
                shown.push_back((r, q));
            }
            let mean = sum / m;
            let sd = (sum2 / m - mean * mean).max(0.0).sqrt();
            (mean / q_star, sd / q_star, 1.0 - misses / m, c)
        };
        let (fixed_mean, fixed_sd, fixed_cov, _) = run(false);
        let (auto_mean, auto_sd, auto_cov, c) = run(true);
        // n̄ is the n - 1 others waiting at every release.
        assert!(
            (c.held_mean() - (n - 1) as f64).abs() < 1e-9,
            "{}",
            c.held_mean()
        );
        let k_n =
            0.05 / (1.0 + 0.05 * c.gain * c.held_mean() / HELD_GAIN_BOUND) * c.gain * n as f64;
        assert!(k_n < HELD_GAIN_BOUND, "k·n {k_n}");
        assert!(
            fixed_mean > 1.5 && fixed_sd > 0.8,
            "fixed: {fixed_mean} {fixed_sd}"
        );
        assert!(
            (auto_mean - 1.0).abs() < 0.03 && auto_sd < 0.1,
            "auto: {auto_mean} {auto_sd}"
        );
        assert!((fixed_cov - 0.9).abs() < 0.01, "fixed coverage {fixed_cov}");
        assert!((auto_cov - 0.9).abs() < 0.005, "auto coverage {auto_cov}");
    }
}
