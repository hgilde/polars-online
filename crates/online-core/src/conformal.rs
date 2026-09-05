//! Adaptive conformal intervals (ENHANCEMENTS E36): a tracked quantile of the
//! conformity score `|resid|`, with a long-run coverage guarantee that needs
//! no distributional assumption and survives a shift in the residuals.
//!
//! One scalar recursion per slot. With target coverage `1 − α`, score
//! `s_t = |resid_t|` and step `η_t`:
//!
//! ```text
//! err_t = 1{s_t > q_t}                      the interval missed
//! q_{t+1} = max(0, q_t + η_t · w_t · (err_t − α))
//! ```
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
//! only ever raises `q`, so it can only add coverage; with a weight `w_t` the
//! sum is `w`-weighted. The step here is `η_t = rate · σ_t`, with `σ_t` the
//! slot's EW residual standard deviation before the row, so that `rate` is
//! unit-free and the radius moves at the scale of the errors it brackets;
//! the bound then holds in `σ`-weighted form, at the same rate.
//!
//! `q` starts where a Gaussian would put it, `σ · Φ⁻¹(1 − α/2)`, on the first
//! row that has a residual and a finite `σ > 0`, and tracks from there. The
//! realized coverage is an EW mean of `1{s_t ≤ q_t}` on the same clock as the
//! model, also read before the row.

use serde::{Deserialize, Serialize};

/// Tracked conformal radius and realized coverage for one output slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conformal {
    /// Miscoverage target `α`; the interval aims at coverage `1 − α`.
    alpha: f64,
    /// Step per unit of `σ`.
    rate: f64,
    /// The radius. NaN until the warm start.
    q: f64,
    /// EW mean of `1{|resid| ≤ q}` and its weight.
    cov: f64,
    cov_w: f64,
}

impl Conformal {
    /// `coverage` is the target `1 − α`, strictly inside `(0, 1)`; `rate` the
    /// step per unit of `σ`, `> 0`.
    pub fn new(coverage: f64, rate: f64) -> Result<Self, String> {
        if !(coverage > 0.0 && coverage < 1.0) {
            return Err(format!(
                "conformal coverage must be strictly between 0 and 1, got {coverage}"
            ));
        }
        if !(rate > 0.0 && rate.is_finite()) {
            return Err(format!("conformal_rate must be finite and > 0, got {rate}"));
        }
        Ok(Self {
            alpha: 1.0 - coverage,
            rate,
            q: f64::NAN,
            cov: 0.0,
            cov_w: 0.0,
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
        if !resid.is_finite() || w <= 0.0 {
            self.cov_w *= lam;
            return;
        }
        let usable_sigma = sigma.is_finite() && sigma > 0.0;
        if !self.q.is_finite() {
            if usable_sigma {
                self.q = sigma * norm_ppf(1.0 - 0.5 * self.alpha);
            }
            self.cov_w *= lam;
            return;
        }
        let s = resid.abs();
        let miss = s > self.q;
        let cw = lam * self.cov_w + w;
        self.cov = (lam * self.cov_w * self.cov + w * f64::from(!miss)) / cw;
        self.cov_w = cw;
        // A slot whose residuals have all been exactly zero has sigma 0 and
        // takes no step: the radius waits until there is an error scale.
        let eta = if usable_sigma { self.rate * sigma } else { 0.0 };
        self.q = (self.q + eta * w * (f64::from(miss) - self.alpha)).max(0.0);
    }
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
        for (p, z) in [
            (0.5, 0.0),
            (0.75, 0.6744897501960817),
            (0.9, 1.2815515655446004),
            (0.95, 1.6448536269514722),
            (0.975, 1.959963984540054),
            (0.99, 2.3263478740408408),
            (0.995, 2.5758293035489004),
            (0.9995, 3.2905267314919255),
            (0.01, -2.3263478740408408),
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
        // Weight scales the step; a zero-weight row only ages the coverage.
        let q2 = c.radius().unwrap();
        c.update(10.0, 1.0, 1.0, 3.0);
        assert!((c.radius().unwrap() - (q2 + 3.0 * 0.1 * 0.8)).abs() < 1e-14);
        let q3 = c.radius().unwrap();
        let cov = c.coverage().unwrap();
        c.update(10.0, 1.0, 0.5, 0.0);
        assert_eq!(c.radius(), Some(q3));
        assert_eq!(c.coverage(), Some(cov));
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
}
