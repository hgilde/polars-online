//! `bocpd`: Bayesian online changepoint detection (Adams & MacKay 2007;
//! docs/ENHANCEMENTS.md E61).
//!
//! Every other detector here asks "has something changed?" and answers with
//! a statistic. This one keeps a **posterior over how long the current run
//! has lasted**, and a changepoint is that posterior collapsing to zero. The
//! answer is a probability rather than a flag, and it comes with the run
//! length attached: "we are 40 rows into a regime" is a different piece of
//! information from "something broke".
//!
//! # The recursion
//!
//! Their Algorithm 1, in log space, with `H = 1/hazard`:
//!
//! ```text
//! growth:      P(rₜ = r+1, x₁:ₜ) = P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·(1 − H)
//! changepoint: P(rₜ = 0,   x₁:ₜ) = Σᵣ P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·H
//! ```
//!
//! `πₜ^{(r)}` is the posterior predictive of run `r` -- the density of this
//! row under the rows that run has seen. Each run keeps its own conjugate
//! sufficient statistics, so the whole thing is `O(runs·d²)` a row.
//!
//! **Line 6 of their algorithm is where the bookkeeping is easy to get
//! wrong**, and it is worth writing out: `ν⁽ʳ⁺¹⁾_{t+1} = ν⁽ʳ⁾_t + u(xₜ)`
//! and `ν⁽⁰⁾_{t+1} = ν_prior`. Slot `j` holds exactly the `j` rows that the
//! hypothesis `rₜ = j` says come before the next row in its run -- so the
//! slot pushed at the front of the vector holds **nothing**, and its
//! predictive is the prior's. Letting it take the row too (the easy
//! mistake, and what this file did first) puts one row of the old regime
//! inside every "brand new run": the estimated start of every run moves
//! back by one, and a 20-σ row stops looking like a changepoint at all,
//! because the hypothesis that it started a run is evaluated with the row
//! before it already inside the run.
//!
//! The run vector would grow by one every row, so runs whose normalised
//! mass falls below `truncate` are dropped and the rest renormalised, and
//! `max_run` folds the tail into the last kept run. Both are approximations
//! with a knob, and a test measures what the knob costs.
//!
//! # The emissions
//!
//! `gaussian` is the normal-inverse-Wishart conjugate pair, whose posterior
//! predictive is a multivariate Student-t; `diag` is the per-feature
//! normal-inverse-gamma, a product of univariate ones. Both are exact.
//!
//! `robust` is a **β-power-weighted** variant: a row enters a run's
//! sufficient statistics at weight
//!
//! ```text
//! w(x) = (πₜ(x) / πₜ(mode))^β  ∈ (0, 1]
//! ```
//!
//! so a row in the body of the predictive counts fully and a 20-σ row counts
//! for essentially nothing. The **same** weight tempers the message the row
//! passes in the recursion, `πᵣ^{w(x)}`: a row that is atypical for every
//! run then multiplies every joint by about 1 and the posterior does not
//! move. Both halves are needed. Weighting only what a run learns leaves
//! the outlier declaring a changepoint (`p_change` 0.91, measured) while
//! keeping the statistics clean; weighting both makes the row a non-event.
//! That is the mechanism a generalised-Bayes changepoint detector uses
//! (Knoblauch, Jewson & Damoulas 2018 build their β-divergence BOCPD on
//! it).
//!
//! `robust_beta` is a trade and not a free lunch: a whole new regime is a
//! run of individually forgiven rows, so above about `0.2` nothing is ever
//! detected again. Measured on a four-sigma shift, `0.05` and `0.1` find it
//! within five rows and date it to within a row; `0.3` and up never find
//! it. The default the polars layer fills in is `0.1`.
//!
//! **It is not the diffusion-score-matching posterior of Altamirano, Briol
//! & Knoblauch (2023)**: that paper was not read here, and this does not
//! claim to be its equations. Swapping the exact ABK posterior in behind
//! this name is a follow-up, and the behavioural test -- a 20-σ row that
//! restarts the plain run and does not move this one -- is what either has
//! to pass.

use serde::{Deserialize, Serialize};

use crate::solve::SpdFactor;

/// **Why the reported changepoint probability is `P(r ≤ 1)` and not
/// `P(r = 0)`.**
///
/// Algorithm 1 puts the *same* predictive `πₜ^{(r)}` on both branches, so
/// under a constant hazard
///
/// ```text
/// P(rₜ = 0)      = H·Σᵣ Jᵣπᵣ
/// Σᵣ P(rₜ = r+1) = (1 − H)·Σᵣ Jᵣπᵣ
/// ```
///
/// and the normalised mass at `r = 0` is **exactly `H`, on every row,
/// whatever the data**. `docs/PLAN.md` §11a called that quantity `p_r0` and
/// expected it to spike; it cannot. What can move is `r = 1`: the run that
/// started on the changepoint row is evaluated against the *prior*
/// predictive there, and if the row is better explained that way than by
/// the run in progress, the mass goes to it. `p_change = P(rₜ ≤ 1)` is
/// therefore what is reported.
///
/// It is an alarm and not the answer. Measured: a tenfold variance step
/// takes it to 0.83 on the row of the break; a four-sigma mean shift under
/// a diffuse prior barely lifts it; a change in the correlation alone never
/// moves it at all. `run_mode` finds all three -- one to three rows later,
/// and dated to the right row, since it is the pre-row run length and so
/// `t − run_mode` is the row the current run began on.
///
/// Which conjugate pair the runs carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BocpdEmission {
    /// Normal-inverse-Wishart; the predictive is a multivariate Student-t.
    Gaussian,
    /// Normal-inverse-gamma per feature; the predictive is their product.
    Diag,
    /// `diag`'s pair with β-power-weighted rows; see the module docs.
    Robust,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BocpdCfg {
    pub n_features: usize,
    /// Expected rows between changepoints; `H = 1/hazard`.
    pub hazard: f64,
    /// Read the hazard from the row instead (it rides in `y[0]`, the way
    /// `ew_class`'s label does), falling back to `hazard` where the value
    /// is missing. Explicit, so a model without a hazard column can never
    /// mistake a target for one.
    #[serde(default)]
    pub hazard_from_row: bool,
    pub emission: BocpdEmission,
    /// `μ₀`, zeros when absent.
    pub prior_mean: Option<Vec<f64>>,
    /// `κ₀`, the prior's weight in rows.
    pub prior_kappa: f64,
    /// `ν₀`; `d + 2` when absent, the smallest value with a finite mean.
    pub prior_nu: Option<f64>,
    /// `Ψ₀`: a scalar for `sI`, or a `d×d` matrix. **Set it from the data's
    /// scale** -- it is the prior guess at the covariance, and 1.0 on data
    /// measured in 1e-4 makes every row look like a changepoint.
    pub prior_scale: Option<Vec<f64>>,
    /// `robust`'s β; 0 makes it `diag`.
    pub robust_beta: f64,
    /// Drop runs below this normalised mass.
    pub truncate: f64,
    /// Cap the run vector here, folding every longer run into the last
    /// kept one. That entry then holds their summed mass and its own
    /// statistics -- the youngest of the folded group, since the vector
    /// runs newest-first -- so `max_run` bounds the memory of a run as
    /// well as the length of the vector, and `run_mode` saturates one
    /// below it.
    pub max_run: usize,
    pub min_periods: f64,
}

impl BocpdCfg {
    pub fn nu0(&self) -> f64 {
        self.prior_nu.unwrap_or(self.n_features as f64 + 2.0)
    }

    /// `Ψ₀` as a `d×d` matrix.
    pub fn psi0(&self) -> Vec<f64> {
        let d = self.n_features;
        let mut m = vec![0.0; d * d];
        match &self.prior_scale {
            Some(v) if v.len() == d * d => m.copy_from_slice(v),
            Some(v) if v.len() == 1 => {
                for i in 0..d {
                    m[i * d + i] = v[0];
                }
            }
            _ => {
                for i in 0..d {
                    m[i * d + i] = 1.0;
                }
            }
        }
        m
    }

    pub fn mu0(&self) -> Vec<f64> {
        self.prior_mean
            .clone()
            .unwrap_or_else(|| vec![0.0; self.n_features])
    }

    pub fn validate(&self) -> Result<(), String> {
        let d = self.n_features;
        if d == 0 {
            return Err("bocpd: at least one column is required".into());
        }
        if !(self.hazard > 1.0 && self.hazard.is_finite()) {
            return Err(
                "bocpd: hazard is the expected rows between changepoints and must be finite and \
                 > 1"
                .into(),
            );
        }
        if !(self.prior_kappa > 0.0 && self.prior_kappa.is_finite()) {
            return Err("bocpd: prior_kappa must be finite and > 0".into());
        }
        let nu = self.nu0();
        let floor = if self.emission == BocpdEmission::Gaussian {
            d as f64 + 1.0
        } else {
            1.0
        };
        if !(nu > floor && nu.is_finite()) {
            return Err(format!(
                "bocpd: prior_nu must be finite and > {floor} for this emission (got {nu}); \
                 below it the predictive has no finite variance"
            ));
        }
        if let Some(v) = &self.prior_scale {
            if v.len() != 1 && v.len() != d * d {
                return Err(format!(
                    "bocpd: prior_scale is a scalar or a {d}x{d} matrix, got {} values",
                    v.len()
                ));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err("bocpd: prior_scale must be finite".into());
            }
            // A scale matrix that is not positive definite gives a
            // predictive with no density: every row would report nulls and
            // count as a failure, with nothing to say why
            // (docs/REVIEW-E54-E64.md B2).
            if v.len() == 1 && v[0] <= 0.0 {
                return Err(format!(
                    "bocpd: a scalar prior_scale is the prior scale of the variance and must be \
                     > 0 (got {})",
                    v[0]
                ));
            }
            if v.len() == d * d {
                for i in 0..d {
                    for j in (i + 1)..d {
                        let (a, b) = (v[i * d + j], v[j * d + i]);
                        if (a - b).abs() > 1e-12 * (1.0 + a.abs().max(b.abs())) {
                            return Err(format!(
                                "bocpd: a matrix prior_scale must be symmetric; [{i}][{j}] is \
                                 {a} and [{j}][{i}] is {b}"
                            ));
                        }
                    }
                }
                if !matches!(SpdFactor::of(v, d), Some(f) if f.attempts() == 0) {
                    return Err(
                        "bocpd: a matrix prior_scale must be positive definite -- it is the \
                         prior's scale matrix Ψ₀, and the predictive has no density without one"
                            .into(),
                    );
                }
            }
        }
        if let Some(m) = &self.prior_mean {
            if m.len() != d {
                return Err(format!(
                    "bocpd: prior_mean must be {d} values, got {}",
                    m.len()
                ));
            }
            if m.iter().any(|x| !x.is_finite()) {
                return Err("bocpd: prior_mean must be finite".into());
            }
        }
        if self.robust_beta < 0.0 || !self.robust_beta.is_finite() {
            return Err("bocpd: robust_beta must be finite and >= 0".into());
        }
        if self.robust_beta > 0.0 && self.emission != BocpdEmission::Robust {
            return Err(format!(
                "bocpd: robust_beta applies to emission = \"robust\", not {:?}",
                self.emission
            ));
        }
        if self.emission == BocpdEmission::Robust && self.robust_beta <= 0.0 {
            return Err(
                "bocpd: emission = \"robust\" needs robust_beta > 0; at 0 it is \"diag\"".into(),
            );
        }
        if !(0.0..1.0).contains(&self.truncate) || self.truncate.is_nan() {
            return Err("bocpd: truncate must be in [0, 1)".into());
        }
        if self.max_run < 2 {
            return Err("bocpd: max_run must be >= 2".into());
        }
        if self.min_periods.is_nan() || self.min_periods < 0.0 {
            return Err("bocpd: min_periods must be >= 0".into());
        }
        Ok(())
    }
}

/// One run's conjugate sufficient statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Run {
    /// Rows this run has absorbed, i.e. Adams & MacKay's `r`. Not `n`:
    /// under a fractional row weight, or the `robust` emission's, the two
    /// differ, and the reported run *length* is a count of rows.
    #[serde(default)]
    len: f64,
    /// Accumulated weight (rows, or β-weights under `robust`).
    n: f64,
    /// `Σ w·x`, length `d`.
    sx: Vec<f64>,
    /// `Σ w·x x'`: `d*d` for `gaussian`, the diagonal (`d`) otherwise.
    sxx: Vec<f64>,
}

impl Run {
    fn new(d: usize, full: bool) -> Self {
        Self {
            len: 0.0,
            n: 0.0,
            sx: vec![0.0; d],
            sxx: vec![0.0; if full { d * d } else { d }],
        }
    }

    fn add(&mut self, x: &[f64], w: f64, full: bool) {
        let d = x.len();
        self.len += 1.0;
        self.n += w;
        for (s, xi) in self.sx.iter_mut().zip(x) {
            *s += w * xi;
        }
        if full {
            for i in 0..d {
                for j in 0..d {
                    self.sxx[i * d + j] += w * x[i] * x[j];
                }
            }
        } else {
            for (s, xi) in self.sxx.iter_mut().zip(x) {
                *s += w * xi * xi;
            }
        }
    }
}

/// `ln Γ(x)`, by the Lanczos approximation: the Student-t density needs it
/// and `f64::ln_gamma` is unstable.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos g = 7, n = 9; accurate to ~1e-13 over the range used here.
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection, for completeness; the callers stay above 0.5.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `ln Σ exp(v)`, about the maximum.
fn log_sum_exp(v: &[f64]) -> f64 {
    let top = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return top;
    }
    top + v.iter().map(|x| (x - top).exp()).sum::<f64>().ln()
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// What [`Bocpd::read`] hands the update: the new log joint over run
/// lengths, and the weight each existing run takes the row at.
type Update = (Vec<f64>, Vec<f64>);

/// Bayesian online changepoint detection; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bocpd {
    cfg: BocpdCfg,
    /// One per kept run length, shortest first (`runs[0]` is `r = 0`).
    runs: Vec<Run>,
    /// `ln P(rₜ = r, x₁:ₜ)`, aligned with `runs`. Unnormalised; see
    /// [`Bocpd::prune`] for why it is left that way.
    logjoint: Vec<f64>,
    n_eff: f64,
    /// Rows whose predictive could not be evaluated -- a scale matrix that
    /// would not factorize, or a non-finite value reaching the emission.
    /// The row reports nulls and the posterior does not move; without a
    /// count that is silent (docs/REVIEW-E54-E64.md B1).
    ///
    /// Skipped when zero, so a state that never hit one writes the bytes it
    /// always did and no schema bump is owed (see [`crate::SCHEMA_VERSION`],
    /// which records the same rule for task 38's sums).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub solve_failures: u64,
}

impl Bocpd {
    pub fn new(cfg: BocpdCfg) -> Result<Self, String> {
        cfg.validate()?;
        let full = cfg.emission == BocpdEmission::Gaussian;
        let d = cfg.n_features;
        Ok(Self {
            runs: vec![Run::new(d, full)],
            logjoint: vec![0.0],
            n_eff: 0.0,
            solve_failures: 0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &BocpdCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// The normalised run-length posterior as it stands, shortest run first.
    pub fn run_posterior(&self) -> Vec<f64> {
        let z = log_sum_exp(&self.logjoint);
        if !z.is_finite() {
            let k = self.logjoint.len();
            return vec![1.0 / k as f64; k];
        }
        self.logjoint.iter().map(|l| (l - z).exp()).collect()
    }

    /// Output slot labels in emission order.
    pub fn labels(names: &[String]) -> Vec<String> {
        let mut out = vec!["p_change".into(), "run_mode".into(), "run_mean".into()];
        out.extend(names.iter().map(|n| format!("pred_{n}")));
        out.push("logscore".into());
        out
    }

    pub fn n_outputs_for(d: usize) -> usize {
        4 + d
    }

    fn full(&self) -> bool {
        self.cfg.emission == BocpdEmission::Gaussian
    }

    /// One run's posterior mean: `(κ₀μ₀ + Σx)/(κ₀ + n)`.
    fn run_mean_of(&self, run: &Run) -> Vec<f64> {
        let k0 = self.cfg.prior_kappa;
        let mu0 = self.cfg.mu0();
        let kn = k0 + run.n;
        mu0.iter()
            .zip(&run.sx)
            .map(|(m, s)| (k0 * m + s) / kn)
            .collect()
    }

    /// `(ln π(x), ln π(mode))` for one run: the log posterior predictive of
    /// the row, and of the predictive's own mode, which is what the
    /// `robust` weight is relative to.
    fn log_predictive(&self, run: &Run, x: &[f64]) -> Option<(f64, f64)> {
        let d = self.cfg.n_features;
        let (k0, nu0) = (self.cfg.prior_kappa, self.cfg.nu0());
        let mu0 = self.cfg.mu0();
        let kn = k0 + run.n;
        let nun = nu0 + run.n;
        let mun = self.run_mean_of(run);
        if self.full() {
            // `Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)'`.
            let mut psi = self.cfg.psi0();
            let xbar: Vec<f64> = if run.n > 0.0 {
                run.sx.iter().map(|s| s / run.n).collect()
            } else {
                vec![0.0; d]
            };
            for i in 0..d {
                for j in 0..d {
                    let s = run.sxx[i * d + j] - run.n * xbar[i] * xbar[j];
                    let g = k0 * run.n / kn * (xbar[i] - mu0[i]) * (xbar[j] - mu0[j]);
                    psi[i * d + j] += s + g;
                }
            }
            let dof = nun - d as f64 + 1.0;
            if dof <= 0.0 {
                return None;
            }
            let scale = (kn + 1.0) / (kn * dof);
            let mut sigma = psi;
            sigma.iter_mut().for_each(|v| *v *= scale);
            let f = SpdFactor::of(&sigma, d)?;
            let delta: Vec<f64> = x.iter().zip(&mun).map(|(a, b)| a - b).collect();
            let q = f.quad_forms(&delta, d, 1)[0];
            let df = d as f64;
            let base = ln_gamma((dof + df) / 2.0)
                - ln_gamma(dof / 2.0)
                - 0.5 * df * (dof * std::f64::consts::PI).ln()
                - 0.5 * f.log_det();
            Some((base - 0.5 * (dof + df) * (1.0 + q / dof).ln(), base))
        } else {
            // Per feature, a normal-inverse-gamma: `t_{νₙ}` with scale²
            // `ψₙ(κₙ+1)/(νₙκₙ)`.
            let psi0 = self.cfg.psi0();
            let mut lp = 0.0;
            let mut mode = 0.0;
            for i in 0..d {
                let xbar = if run.n > 0.0 { run.sx[i] / run.n } else { 0.0 };
                let s = run.sxx[i] - run.n * xbar * xbar;
                let g = k0 * run.n / kn * (xbar - mu0[i]) * (xbar - mu0[i]);
                let psin = psi0[i * d + i] + s + g;
                let var = psin * (kn + 1.0) / (nun * kn);
                if var.is_nan() || var <= 0.0 || nun <= 0.0 {
                    return None;
                }
                let z = (x[i] - mun[i]) / var.sqrt();
                let base = ln_gamma((nun + 1.0) / 2.0)
                    - ln_gamma(nun / 2.0)
                    - 0.5 * (nun * std::f64::consts::PI * var).ln();
                lp += base - 0.5 * (nun + 1.0) * (1.0 + z * z / nun).ln();
                mode += base;
            }
            Some((lp, mode))
        }
    }

    /// This row's hazard: the value in the targets slot under
    /// `hazard_from_row`, the configured one otherwise. A missing value
    /// falls back to the configured hazard; an unusable one (`<= 1`, or not
    /// finite) is refused by the plumbing before it reaches here.
    fn hazard_of(&self, y: &[Option<f64>]) -> f64 {
        if self.cfg.hazard_from_row {
            y.first().copied().flatten().unwrap_or(self.cfg.hazard)
        } else {
            self.cfg.hazard
        }
    }

    /// The row's outputs, and what the update needs: the new log joint and
    /// the per-run weights the row enters with.
    fn read(&self, x: &[f64], hazard: f64) -> (Vec<f64>, Option<Update>) {
        let d = self.cfg.n_features;
        let nan = vec![f64::NAN; Self::n_outputs_for(d)];
        let h = 1.0 / hazard;
        if !(h > 0.0 && h < 1.0) {
            return (nan, None);
        }
        let r = self.runs.len();
        let mut logpi = Vec::with_capacity(r);
        let mut weights = Vec::with_capacity(r);
        for run in &self.runs {
            let Some((lp, mode)) = self.log_predictive(run, x) else {
                return (nan, None);
            };
            logpi.push(lp);
            weights.push(if self.cfg.emission == BocpdEmission::Robust {
                (self.cfg.robust_beta * (lp - mode)).exp().clamp(0.0, 1.0)
            } else {
                1.0
            });
        }
        // The pre-row run posterior, which `run_mode`, `run_mean` and the
        // predictive mean are read from.
        let pre = self.run_posterior();
        let mut mode_at = 0;
        for i in 1..r {
            if pre[i] > pre[mode_at] {
                mode_at = i;
            }
        }
        // Every run carries its own length, because the slot index stops
        // being it as soon as `truncate` drops a run from the middle or
        // `max_run` folds the tail.
        let run_mean: f64 = pre.iter().zip(&self.runs).map(|(p, run)| p * run.len).sum();
        let mut pred = vec![0.0; d];
        for (i, run) in self.runs.iter().enumerate() {
            let m = self.run_mean_of(run);
            for (o, v) in pred.iter_mut().zip(&m) {
                *o += pre[i] * v;
            }
        }
        // `ln Σ p̃ᵣ πᵣ`, the row's log predictive density.
        let mix: Vec<f64> = (0..r)
            .map(|i| {
                if pre[i] > 0.0 {
                    pre[i].ln() + logpi[i]
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let logscore = log_sum_exp(&mix);
        // Algorithm 1: growth into `r+1`, changepoint into 0. Under
        // `robust` the message the row passes is the *tempered* likelihood
        // `πᵣ^{w(x)}`, with the same weight that tempers what the run
        // learns: a row that is atypical for every run then multiplies
        // every joint by about 1, and the posterior does not move. That is
        // the robustness -- one wild row must not be a changepoint -- and
        // it is the same generalised-Bayes weight in both places.
        let mut new = vec![f64::NEG_INFINITY; r + 1];
        let mut cp = Vec::with_capacity(r);
        for i in 0..r {
            let base = self.logjoint[i] + weights[i] * logpi[i];
            new[i + 1] = base + (1.0 - h).ln();
            cp.push(base + h.ln());
        }
        new[0] = log_sum_exp(&cp);
        let z = log_sum_exp(&new);
        if !z.is_finite() {
            return (nan, None);
        }
        let mut out = Vec::with_capacity(Self::n_outputs_for(d));
        // `P(rₜ ≤ 1)`: see the note above the emission enum. `r = 0` alone
        // is exactly the hazard.
        let short = if new.len() > 1 {
            log_sum_exp(&new[..2])
        } else {
            new[0]
        };
        out.push((short - z).exp());
        out.push(self.runs[mode_at].len);
        out.push(run_mean);
        out.extend_from_slice(&pred);
        out.push(logscore);
        // `min_periods` gates what is *reported*, never what is learned: a
        // gated row still moves the posterior, as it does in every other
        // model here.
        if self.n_eff < self.cfg.min_periods {
            return (nan, Some((new, weights)));
        }
        (out, Some((new, weights)))
    }

    /// Drop the runs below `truncate` and fold the tail at `max_run`.
    ///
    /// The joint is **not** renormalised, deliberately. Subtracting `z` here
    /// would keep it near zero, and every output is a difference against `z`
    /// so no reported number would move -- but `z` comes out of `ln`, and
    /// libm's last bit is not the same on every platform. Feeding it back
    /// into the state made a bank saved on one OS continue differently on
    /// another: `state_schema5.rs`'s frozen file stopped reproducing on
    /// Linux, having been written on macOS. The drift it would have fixed is
    /// about 1.4 nats a row, so it costs nothing until well past 1e9 rows
    /// (docs/REVIEW-E54-E64.md B4, reverted 2026-09-06).
    fn prune(&mut self) {
        let z = log_sum_exp(&self.logjoint);
        if !z.is_finite() {
            return;
        }
        if self.cfg.truncate > 0.0 {
            let keep: Vec<bool> = self
                .logjoint
                .iter()
                .map(|l| (l - z).exp() >= self.cfg.truncate)
                .collect();
            // Never drop everything, and never drop `r = 0`: it is the
            // branch a changepoint arrives on.
            if keep.iter().any(|k| *k) {
                let mut i = 0;
                self.runs.retain(|_| {
                    let k = keep[i] || i == 0;
                    i += 1;
                    k
                });
                let mut i = 0;
                self.logjoint.retain(|_| {
                    let k = keep[i] || i == 0;
                    i += 1;
                    k
                });
            }
        }
        if self.runs.len() > self.cfg.max_run {
            // Fold the tail into the last kept run: it takes their summed
            // mass and keeps its own statistics. The vector runs
            // newest-first, so that is the *youngest* of the folded group,
            // and the effect is that no run ever accumulates more than
            // `max_run` rows -- a bounded memory, not only a bounded
            // vector. `max_run_folds_the_tail` pins both halves.
            let cut = self.cfg.max_run;
            let tail = log_sum_exp(&self.logjoint[cut - 1..]);
            self.logjoint.truncate(cut);
            self.runs.truncate(cut);
            let last = self.logjoint.len() - 1;
            self.logjoint[last] = tail;
        }
    }
}

impl crate::OnlineModel for Bocpd {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        let (pred, extra) = self.read(x, self.hazard_of(y));
        let out = crate::Step {
            pred,
            n_eff: self.n_eff,
            extra: None,
        };
        let _ = d_clock;
        if weight <= 0.0 {
            // Advance the clock and learn nothing: the posterior does not
            // move and no run sees the row. Nothing here decays, so `n_eff`
            // -- the accumulated weight -- does not move either.
            return out;
        }
        self.n_eff += weight;
        let Some((new, weights)) = extra else {
            // The predictive could not be evaluated: the row reports nulls
            // and the posterior stands. Counted, so a run of them is
            // visible in `diagnostics` rather than silent (B1).
            self.solve_failures += 1;
            return out;
        };
        let full = self.full();
        // Algorithm 1's line 6, and the indexing is the whole of it:
        // `ν⁽ʳ⁺¹⁾_{t+1} = ν⁽ʳ⁾_t + u(xₜ)` and `ν⁽⁰⁾_{t+1} = ν_prior`. Slot
        // `j` holds exactly the `j` rows that the hypothesis `rₜ = j` says
        // precede the next row in its run -- so the new slot at the front
        // holds **nothing**, and its predictive is the prior's. (Letting it
        // take this row too, which is the easy mistake, puts one row of the
        // old regime inside every "brand new run" and moves the estimated
        // start of every run back by one.)
        let mut runs = Vec::with_capacity(self.runs.len() + 1);
        runs.push(Run::new(self.cfg.n_features, full));
        for (i, run) in self.runs.iter().enumerate() {
            let mut grown = run.clone();
            grown.add(x, weight * weights[i], full);
            runs.push(grown);
        }
        self.runs = runs;
        self.logjoint = new;
        self.prune();
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> crate::Step {
        crate::Step {
            pred: self.read(x, self.cfg.hazard).0,
            n_eff: self.n_eff,
            extra: None,
        }
    }

    /// The hazard rides in `y[0]` under `hazard_from_row`, so the answer
    /// depends on it exactly as the step's does (C1).
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], _d_clock: f64) -> crate::Step {
        crate::Step {
            pred: self.read(x, self.hazard_of(y)).0,
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::Bocpd(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Bocpd(m) => Ok((**m).clone()),
            other => Err(crate::StateError::WrongModel {
                expected: "bocpd",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        Self::n_outputs_for(self.cfg.n_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnlineModel;

    fn cfg(d: usize) -> BocpdCfg {
        BocpdCfg {
            n_features: d,
            hazard: 100.0,
            hazard_from_row: false,
            emission: BocpdEmission::Diag,
            prior_mean: None,
            prior_kappa: 1.0,
            prior_nu: Some(2.0),
            prior_scale: Some(vec![1.0]),
            robust_beta: 0.0,
            truncate: 0.0,
            max_run: 100_000,
            min_periods: 0.0,
        }
    }

    struct Normals(crate::SplitMix64);

    impl Normals {
        fn new(seed: u64) -> Self {
            Self(crate::SplitMix64::new(seed))
        }
        fn unit(&mut self) -> f64 {
            ((self.0.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.unit(), self.unit());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// `ln Γ` against values it must reproduce.
    #[test]
    fn the_log_gamma_is_accurate() {
        for (x, want) in [
            (1.0, 0.0),
            (2.0, 0.0),
            (0.5, (std::f64::consts::PI).sqrt().ln()),
            (5.0, 24.0f64.ln()),
            (10.0, 362880.0f64.ln()),
            // Stirling: (z − ½)ln z − z + ½ln 2π + 1/(12z) = 361.4355…
            (100.5, 361.4355404677775),
        ] {
            assert!(
                (ln_gamma(x) - want).abs() < 1e-9,
                "{x}: {} vs {want}",
                ln_gamma(x)
            );
        }
    }

    /// The whole run-length posterior against Algorithm 1 written out, at
    /// every row, with no truncation.
    #[test]
    fn the_posterior_is_the_longhand_algorithm_one() {
        let hazard = 50.0;
        let h = 1.0 / hazard;
        let (k0, nu0, psi0) = (1.0, 2.0, 1.0);
        let mut m = Bocpd::new(BocpdCfg { hazard, ..cfg(1) }).unwrap();
        let mut n = Normals::new(3);
        // Longhand: one (n, sum, sumsq) per run, in probabilities.
        let mut runs: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0)];
        let mut joint: Vec<f64> = vec![1.0];
        for t in 0..150 {
            let x = if t < 75 { n.normal() } else { 4.0 + n.normal() };
            let step = m.step(&[x], &[], 1.0, 1.0);
            // The predictive of each run: a Student-t.
            let pi: Vec<f64> = runs
                .iter()
                .map(|(cnt, s, ss)| {
                    let kn = k0 + cnt;
                    let nun = nu0 + cnt;
                    let mun = s / kn;
                    let xbar = if *cnt > 0.0 { s / cnt } else { 0.0 };
                    let sq = ss - cnt * xbar * xbar;
                    let g = k0 * cnt / kn * xbar * xbar;
                    let psin = psi0 + sq + g;
                    let var = psin * (kn + 1.0) / (nun * kn);
                    let z = (x - mun) / var.sqrt();
                    (ln_gamma((nun + 1.0) / 2.0)
                        - ln_gamma(nun / 2.0)
                        - 0.5 * (nun * std::f64::consts::PI * var).ln()
                        - 0.5 * (nun + 1.0) * (1.0 + z * z / nun).ln())
                    .exp()
                })
                .collect();
            let total: f64 = joint.iter().sum();
            let pre: Vec<f64> = joint.iter().map(|j| j / total).collect();
            let want_score: f64 = pre.iter().zip(&pi).map(|(p, f)| p * f).sum::<f64>().ln();
            let mut new = vec![0.0; runs.len() + 1];
            for i in 0..runs.len() {
                new[i + 1] = joint[i] * pi[i] * (1.0 - h);
                new[0] += joint[i] * pi[i] * h;
            }
            let z: f64 = new.iter().sum();
            // `P(r ≤ 1)`, and `P(r = 0)` is exactly the hazard -- which the
            // longhand shows too, and is why it is not what is reported.
            let want_change = (new[0] + new.get(1).copied().unwrap_or(0.0)) / z;
            assert!(
                (new[0] / z - h).abs() < 1e-12,
                "row {t}: P(r = 0) is the hazard, {} vs {h}",
                new[0] / z
            );
            assert!(
                (step.pred[0] - want_change).abs() < 1e-10,
                "row {t}: p_change {} vs {want_change}",
                step.pred[0]
            );
            assert!(
                (step.pred[4] - want_score).abs() < 1e-9,
                "row {t}: logscore {} vs {want_score}",
                step.pred[4]
            );
            // The whole posterior, not just its first entry.
            let got = m.run_posterior();
            assert_eq!(got.len(), new.len());
            for (i, g) in got.iter().enumerate() {
                assert!((g - new[i] / z).abs() < 1e-10, "row {t} run {i}");
            }
            // Line 6: the new slot is the *prior*, `ν⁽⁰⁾ = ν_prior`, and
            // every old slot takes the row.
            let mut next: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0)];
            next.extend(runs.iter().map(|(c, s, ss)| (c + 1.0, s + x, ss + x * x)));
            runs = next;
            joint = new;
        }
    }

    /// Adams & MacKay's own finance fixture shape: zero-mean rows whose
    /// *variance* steps, a gamma prior on the inverse variance (`a = 1`,
    /// `b = 1e-4`, their values, which are `prior_nu = 2a` and
    /// `prior_scale = 2b` here) and a hazard of 250.
    #[test]
    fn a_variance_step_is_found() {
        let mut m = Bocpd::new(BocpdCfg {
            hazard: 250.0,
            prior_nu: Some(2.0),
            prior_scale: Some(vec![2e-4]),
            truncate: 1e-6,
            max_run: 600,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(5);
        let mut peaks = Vec::new();
        let mut after_down = 0.0f64;
        for t in 0..900 {
            let sd = if (300..600).contains(&t) { 0.05 } else { 0.005 };
            let step = m.step(&[sd * n.normal()], &[], 1.0, 1.0);
            if step.pred[0] > 0.5 {
                peaks.push(t);
            }
            if (600..700).contains(&t) {
                after_down = after_down.max(step.pred[0]);
            }
        }
        let near = |edge: usize| peaks.iter().any(|p| *p >= edge && *p < edge + 40);
        assert!(near(300), "the step up was missed: {peaks:?}");
        // **The two directions are not symmetric**, and that is a property
        // of the model rather than of this implementation: a variance
        // *increase* makes the next row wildly unlikely under the old run,
        // so the evidence is immediate; a *decrease* makes it merely
        // unsurprising, and the old wide predictive still explains it. The
        // step down shows up as a rise in the changepoint mass, not a
        // spike, and takes many rows of small values to accumulate.
        assert!(
            after_down > 5.0 / 250.0,
            "the step down moved nothing: {after_down}"
        );
    }

    /// What `robust_beta` costs. Tempering is not free: a whole new regime
    /// is a run of individually forgiven rows, so above about `0.2` the
    /// model stops detecting anything at all. Measured on a four-sigma
    /// mean shift: `0.05` and `0.1` find it within five rows and within a
    /// row of the right one; `0.3`, `0.5` and `1.0` never find it.
    #[test]
    fn a_sustained_shift_survives_a_small_beta_and_not_a_large_one() {
        let found = |beta: f64| {
            let mut m = Bocpd::new(BocpdCfg {
                emission: if beta > 0.0 {
                    BocpdEmission::Robust
                } else {
                    BocpdEmission::Diag
                },
                robust_beta: beta,
                hazard: 250.0,
                truncate: 1e-6,
                ..cfg(1)
            })
            .unwrap();
            let mut n = Normals::new(31);
            let mut at = None;
            for t in 0..300 {
                let x = if t < 150 {
                    n.normal()
                } else {
                    4.0 + n.normal()
                };
                let step = m.step(&[x], &[], 1.0, 1.0);
                // `run_mode` is the pre-row run length, so `t - mode` is the
                // row the current run began on.
                if at.is_none() && t > 150 && step.pred[1] < 20.0 {
                    at = Some((t, t as f64 - step.pred[1]));
                }
            }
            at
        };
        for beta in [0.0, 0.05, 0.1] {
            let (t, start) = found(beta).unwrap_or_else(|| panic!("beta {beta} missed it"));
            assert!(t <= 155, "beta {beta} took until row {t}");
            // The start is a mode of a posterior over run lengths, so a row
            // either side of the break is the accuracy on offer.
            assert!(
                (start - 150.0).abs() <= 1.0,
                "beta {beta} put the start at {start}"
            );
        }
        for beta in [0.3, 0.5, 1.0] {
            assert!(
                found(beta).is_none(),
                "beta {beta} still detects; the note is stale"
            );
        }
    }

    /// Truncation is an approximation with a knob, and this is what the
    /// knob costs.
    #[test]
    fn truncation_changes_little() {
        let mut exact = Bocpd::new(cfg(1)).unwrap();
        let mut cut = Bocpd::new(BocpdCfg {
            truncate: 1e-4,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(7);
        let mut worst = 0.0f64;
        for t in 0..400 {
            let x = if t < 200 {
                n.normal()
            } else {
                3.0 + n.normal()
            };
            let a = exact.step(&[x], &[], 1.0, 1.0);
            let b = cut.step(&[x], &[], 1.0, 1.0);
            worst = worst.max((a.pred[0] - b.pred[0]).abs());
        }
        // `truncate = 1e-4` drops runs holding up to that much mass each, so
        // `P(r ≤ 1)` can move by a small multiple of it; measured at 3e-4.
        assert!(worst < 1e-3, "p_change moved by {worst}");
        assert!(
            cut.runs.len() < exact.runs.len(),
            "and the vector is shorter"
        );
    }

    #[test]
    fn max_run_folds_the_tail() {
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 20,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(11);
        for _ in 0..200 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
        }
        assert_eq!(m.runs.len(), 20);
        let p = m.run_posterior();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[19] > 0.5, "the tail holds the mass of every longer run");
        // And the memory is bounded with the vector: after 200 rows no run
        // has seen more than `max_run - 1` of them.
        let longest = m.runs.iter().map(|r| r.len).fold(0.0, f64::max);
        assert_eq!(longest, 19.0, "the fold keeps the youngest of the group");
    }

    /// The robust emission's whole purpose: a single 20-σ row restarts the
    /// Gaussian run and does not move this one.
    #[test]
    fn the_robust_emission_ignores_one_wild_row() {
        let build = |beta: f64| {
            Bocpd::new(BocpdCfg {
                emission: if beta > 0.0 {
                    BocpdEmission::Robust
                } else {
                    BocpdEmission::Diag
                },
                robust_beta: beta,
                truncate: 1e-6,
                ..cfg(1)
            })
            .unwrap()
        };
        let (mut plain, mut robust) = (build(0.0), build(0.1));
        let mut n = Normals::new(13);
        let (mut plain_p, mut robust_p) = (0.0, 0.0);
        for t in 0..300 {
            let x = if t == 200 { 20.0 } else { n.normal() };
            let a = plain.step(&[x], &[], 1.0, 1.0);
            let b = robust.step(&[x], &[], 1.0, 1.0);
            if t == 200 {
                plain_p = a.pred[0];
                robust_p = b.pred[0];
            }
        }
        // One 20-σ row *is* a changepoint to the plain model: the `r = 0`
        // hypothesis is the prior predictive, which explains it far better
        // than a run fitted to N(0, 1) does.
        assert!(plain_p > 0.5, "the plain model restarts on it: {plain_p}");
        // The robust one does not move: the row is atypical for every run,
        // so every tempered likelihood is about 1 and the posterior stays
        // where it was -- within a whisker of the hazard.
        assert!(robust_p < 0.02, "the robust one ignores it: {robust_p}");
        // And it did not learn it either. The plain run that the outlier
        // started carries it in its mean; the robust runs do not.
        let mean = robust.run_mean_of(robust.runs.last().expect("a long run"))[0];
        assert!(
            mean.abs() < 0.3,
            "the outlier did not move the run's mean: {mean}"
        );
        let plain_mean = plain.run_mean_of(plain.runs.last().expect("a long run"))[0];
        assert!(
            plain_mean.abs() > 0.05,
            "the plain one absorbed it: {plain_mean}"
        );
    }

    #[test]
    fn the_hazard_can_come_from_the_row() {
        let build = || {
            Bocpd::new(BocpdCfg {
                hazard_from_row: true,
                ..cfg(1)
            })
            .unwrap()
        };
        let mut sticky = build();
        let mut jumpy = build();
        let mut n = Normals::new(17);
        for _ in 0..100 {
            let x = [n.normal()];
            sticky.step(&x, &[Some(10_000.0)], 1.0, 1.0);
            jumpy.step(&x, &[Some(3.0)], 1.0, 1.0);
        }
        // A short hazard means a changepoint every few rows, so the run
        // posterior sits on short runs.
        let (a, b) = (sticky.run_posterior(), jumpy.run_posterior());
        let mean = |p: &[f64]| p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>();
        assert!(mean(&a) > 5.0 * mean(&b), "{} vs {}", mean(&a), mean(&b));
    }

    #[test]
    fn a_zero_weight_row_leaves_the_posterior_untouched() {
        let mut m = Bocpd::new(cfg(1)).unwrap();
        let mut n = Normals::new(19);
        for _ in 0..50 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
        }
        let before = m.run_posterior();
        let runs = m.runs.len();
        m.step(&[1e6], &[], 1.0, 0.0);
        assert_eq!(m.run_posterior(), before);
        assert_eq!(m.runs.len(), runs);
        assert_eq!(m.n_eff(), 50.0);
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut m = Bocpd::new(BocpdCfg {
            truncate: 1e-6,
            ..cfg(2)
        })
        .unwrap();
        let mut n = Normals::new(23);
        for _ in 0..200 {
            let x = vec![n.normal(), n.normal()];
            let want = m.predict(&x, 1.0);
            let got = m.step(&x, &[], 1.0, 1.0);
            assert_eq!(want.n_eff, got.n_eff);
            assert_eq!(want.pred, got.pred);
        }
    }

    #[test]
    fn the_gaussian_emission_runs_on_several_columns() {
        let mut m = Bocpd::new(BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_nu: Some(5.0),
            truncate: 1e-6,
            ..cfg(3)
        })
        .unwrap();
        let mut n = Normals::new(29);
        let mut last = Vec::new();
        for t in 0..400 {
            let shift = if t < 200 { 0.0 } else { 4.0 };
            let f = n.normal();
            let x: Vec<f64> = (0..3).map(|_| shift + 0.7 * f + 0.7 * n.normal()).collect();
            last = m.step(&x, &[], 1.0, 1.0).pred;
        }
        assert_eq!(last.len(), Bocpd::n_outputs_for(3));
        assert!(last.iter().all(|v| v.is_finite()), "{last:?}");
        // The predictive mean followed the shift.
        assert!((last[3] - 4.0).abs() < 1.0, "{}", last[3]);
    }

    #[test]
    fn the_labels_and_slot_count_follow_the_columns() {
        let names = ["a".to_string(), "b".to_string()];
        assert_eq!(
            Bocpd::labels(&names),
            [
                "p_change", "run_mode", "run_mean", "pred_a", "pred_b", "logscore"
            ]
        );
        assert_eq!(Bocpd::n_outputs_for(2), 6);
    }

    #[test]
    fn a_bad_configuration_is_refused_by_name() {
        let bad = |c: BocpdCfg, msg: &str| {
            let e = Bocpd::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(
            BocpdCfg {
                hazard: 1.0,
                ..cfg(1)
            },
            "must be finite and > 1",
        );
        bad(
            BocpdCfg {
                prior_kappa: 0.0,
                ..cfg(1)
            },
            "prior_kappa must be finite and > 0",
        );
        bad(
            BocpdCfg {
                prior_nu: Some(0.5),
                ..cfg(1)
            },
            "prior_nu must be finite and >",
        );
        bad(
            BocpdCfg {
                robust_beta: 1.0,
                ..cfg(1)
            },
            "robust_beta applies to",
        );
        bad(
            BocpdCfg {
                emission: BocpdEmission::Robust,
                ..cfg(1)
            },
            "needs robust_beta > 0",
        );
        bad(
            BocpdCfg {
                truncate: 1.0,
                ..cfg(1)
            },
            "truncate must be in",
        );
        bad(
            BocpdCfg {
                max_run: 1,
                ..cfg(1)
            },
            "max_run must be >= 2",
        );
        bad(
            BocpdCfg {
                prior_mean: Some(vec![0.0, 0.0]),
                ..cfg(1)
            },
            "prior_mean must be 1 values",
        );
        bad(
            BocpdCfg {
                prior_scale: Some(vec![1.0, 2.0]),
                ..cfg(1)
            },
            "prior_scale is a scalar or a 1x1 matrix",
        );
        bad(
            BocpdCfg {
                prior_scale: Some(vec![0.0]),
                ..cfg(1)
            },
            "scalar prior_scale",
        );
        bad(
            BocpdCfg {
                n_features: 2,
                prior_scale: Some(vec![1.0, 0.5, 0.4, 1.0]),
                ..cfg(2)
            },
            "must be symmetric",
        );
        bad(
            BocpdCfg {
                n_features: 2,
                prior_scale: Some(vec![1.0, 2.0, 2.0, 1.0]),
                ..cfg(2)
            },
            "positive definite",
        );
        bad(
            BocpdCfg {
                prior_mean: Some(vec![f64::NAN]),
                ..cfg(1)
            },
            "prior_mean must be finite",
        );
    }
}
