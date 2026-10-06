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
//! mass falls below `prune_below` are dropped and the rest renormalised, and
//! `max_run` folds the tail into the last kept run. Both are approximations
//! with a knob, and a test measures what the knob costs. In a stationary
//! stream the pruning bounds little: one regime makes every "it began `j`
//! rows ago" hypothesis about as likely as the next, and the posterior
//! spreads over thousands of run lengths -- 2,001 kept after 2,000 rows and
//! 7,036 at most by 20,000 under the defaults (`prune_below = 1e-6`,
//! measured 2026-09-27). `max_run` is then the bound on the vector and on
//! the `O(runs·d²)` work a row costs; a shift collapses it to a few dozen.
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
//! & Knoblauch (2023)**, and does not claim to be its equations. Read
//! 2026-09-28 (docs/PLAN.md task 114): theirs is a Gaussian posterior over
//! the model's natural parameters, `Σ⁻¹ += 2ωΛ(x)`, `μ = Σ(Σ⁻¹μ − 2ων(x))`
//! (their Prop. 3.1 and §3.4), made robust by a weight matrix `m(x)` built
//! from a reference `θ*` (Prop. 3.2), which they take as the maximum
//! likelihood estimate on the whole data set, with `ω` tuned by matching the
//! standard posterior on the first rows. Its predictive is closed-form for a
//! Gaussian with a changing mean; with mean and variance both unknown, as
//! here, they sample it (their App. C.1). A whole-data `θ*`, a tuned `ω`
//! and a sampled predictive do not fit a streaming, deterministic model, so
//! `robust` stays what it is, under its name, and ABK is not built (the
//! user's decision, 2026-09-28). The behavioural test -- a 20-σ row that
//! restarts the plain run and does not move this one -- is what it is held
//! to.

//!
//! **Row weights** (docs/PLAN.md task 147). A row of weight `w` enters at
//! `w / w̄`, `w̄` the mean weight of the rows learned from, this one
//! included, in its run's sufficient statistics and in the likelihood it
//! passes the recursion, `πᵣ^{w/w̄}` -- with `robust`'s β-power weight on
//! top of both. It entered the statistics at its raw weight and the
//! recursion not at all, so a heavy row made the predictive confident
//! without counting as evidence of a change. `κ₀` and `ν₀` are in rows of
//! the mean weight, and a constant multiple of every weight changes nothing.
//! What the row reports, `P(r ≤ 1)` among it, is read as a row of the mean
//! weight, which is what `predict`, never told a weight, can say.
//!
//! # The hazard on the clock (docs/PLAN.md task 179)
//!
//! By default `hazard` is the expected rows between changepoints. `H =
//! 1/hazard`, the same on every row, enters at the end of each row's step,
//! in the changepoint branch above. It is the chance of a break **after**
//! the row, which is why the mass at `r = 0` is exactly `H`.
//!
//! With `hazard_on_clock` it is `τ`, the expected time between changepoints
//! in clock units. Breaks then arrive in time at rate `1/τ`, so the step of
//! `d` clock units into a row carries the chance `h = 1 − exp(−d/τ)` that a
//! break fell inside it. That chance belongs to the boundary **before** the
//! row, and a stream cannot know the step after a row until the next row
//! arrives. So it is applied at the start of the row's step, before the row
//! is read:
//!
//! ```text
//! P(r = 0) ← P(r = 0) + h·Σ_{r≥1} P(r)     the empty run: the row begins a run
//! P(r)     ← (1 − h)·P(r),  r ≥ 1          every run goes on
//! ```
//!
//! Then the recursion above runs with `H = 0`, since no time has passed
//! since the row and no break can follow it yet. After the row the mass at
//! `r = 0` is 0, and `p_change = P(r ≤ 1)` is the chance that the row began
//! a run; it is not "exactly `H`" plus that, as under a per-row hazard.
//! On regular steps the per-row hazard `1/h` puts `h` on the boundary after
//! each row and this form puts it on the boundary before the next. Every
//! row is then read against the same posterior, the first included, whose
//! step meets only the empty run it begins.
//!
//! `d` is the step the model is stepped with: the decayed clock, so a gap
//! past `gap_cap` counts as the cap and a session change as its
//! `session_gap`. A step of 0 has no chance of a break: the first row of a
//! stream or a group, a row after a reset, a stamp shared with the row
//! before. Then `ln h = −∞`, and the empty run gains nothing. Nothing breaks
//! on it: `log_sum_exp` of an all-`−∞` vector is `−∞`, `prune` keeps slot 0
//! whatever its mass, and a run of no mass grows into one of no mass. Under
//! `prune_below = 0` each such step leaves one run of no mass in the vector
//! for good, until `max_run` folds it; under any positive `prune_below` the
//! next prune drops it.
//!
//! A row of weight 0 advances the clock and learns nothing (hard rule 9):
//! its step's chance applies, as decay does elsewhere, and no run grows. The
//! transition composes. Two steps `d_a` and `d_b` with nothing learned
//! between them leave the posterior one step of `d_a + d_b` would, to
//! rounding, since `(1 − h_a)(1 − h_b) = exp(−(d_a + d_b)/τ)`. A row whose
//! predictive cannot be evaluated is the same: the time passes, and nothing
//! is learned. A step that is no step (negative, NaN or infinite, which the
//! plumbing never hands over) refuses the row, as an unusable hazard does.
//!
//! `ln(1 − h) = −d/τ` is a division: no libm, and exact where `(1 −
//! h).ln()` would round away an `h` below `1e-16`. `ln h = ln(−expm1(−d/τ))`,
//! with `expm1` for accuracy at small `d/τ`. Both enter the persisted log
//! joint, as `ln H`, every run's log predictive and `log_sum_exp` already
//! do: the joint is a log-space recursion and holds libm results by nature.
//! [`Bocpd::prune`]'s rule, from the reverted B4, is against adding one that
//! no output needs, and every output is read under this chance.

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
/// Under a hazard on the clock the chance of a break is applied before the
/// row instead (the module docs), so after the row the mass at `r = 0` is 0
/// and `P(r ≤ 1)` is the chance that the row began a run.
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
    /// Expected rows between changepoints, `H = 1/hazard` the chance of a
    /// break after each row; under `hazard_on_clock`, the expected clock
    /// time between them, `τ`.
    pub hazard: f64,
    /// Read the hazard from the row instead (it rides in `y[0]`, the way
    /// `ew_class`'s label does), falling back to `hazard` where the value
    /// is missing. Explicit, so a model without a hazard column can never
    /// mistake a target for one. The row's value is a per-row hazard, the
    /// chance of a break after its row, and is refused beside
    /// `hazard_on_clock`.
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
    pub prune_below: f64,
    /// Cap the run vector here, folding every longer run into the last
    /// kept one. That entry then holds their summed mass and its own
    /// statistics -- the youngest of the folded group, since the vector
    /// runs newest-first -- so `max_run` bounds the memory of a run as
    /// well as the length of the vector, and `run_mode` saturates one
    /// below it.
    pub max_run: usize,
    pub min_weight: f64,
    /// `hazard` is `τ`, the expected clock time between changepoints: the
    /// step of `d` clock units into a row carries the chance `1 −
    /// exp(−d/τ)` of a break, applied before the row is read (the module
    /// docs). Last, and left out where it is false, so a state with a
    /// per-row hazard writes the bytes it always did; a state from before
    /// it reads as a per-row hazard, which it was.
    #[serde(default, skip_serializing_if = "is_false")]
    pub hazard_on_clock: bool,
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
        if self.hazard_on_clock {
            // Any positive time is a rate of breaks; only a step of 0 has no
            // chance of one, and that is the step's business, not `τ`'s.
            if !(self.hazard > 0.0 && self.hazard.is_finite()) {
                return Err(format!(
                    "bocpd: hazard as a duration is the expected time between changepoints and \
                     must be > 0 (got {} clock units)",
                    self.hazard
                ));
            }
            if self.hazard_from_row {
                return Err(
                    "bocpd: a hazard read from the row is a per-row hazard, the chance of a \
                     break after its row, and cannot stand beside a hazard on the clock, the \
                     chance of the step before each row"
                        .into(),
                );
            }
        } else if !(self.hazard > 1.0 && self.hazard.is_finite()) {
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
        // The normal-inverse-Wishart predictive is a `t` on `ν − d + 1`
        // degrees of freedom, with a variance from `ν > d + 1`; the
        // per-feature normal-inverse-gamma predictive is a `t` on `ν`, with
        // a mean from `ν > 1` and a variance only from `ν > 2`, so its floor
        // buys the mean (task 159, D2: the message said variance for both).
        let (floor, what) = if self.emission == BocpdEmission::Gaussian {
            (d as f64 + 1.0, "variance")
        } else {
            (1.0, "mean")
        };
        if !(nu > floor && nu.is_finite()) {
            return Err(format!(
                "bocpd: prior_nu must be finite and > {floor} for this emission (got {nu}); \
                 below it the predictive has no finite {what}"
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
        if !(0.0..1.0).contains(&self.prune_below) || self.prune_below.is_nan() {
            return Err(
                "bocpd: prune_below must be in [0, 1): the share of the mass below which a run \
                 is dropped"
                    .into(),
            );
        }
        if self.max_run < 2 {
            return Err("bocpd: max_run must be >= 2".into());
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err("bocpd: min_weight must be >= 0".into());
        }
        Ok(())
    }
}

/// One run's conjugate sufficient statistics, centred: the weighted mean of
/// its rows and their scatter about it, kept by the weighted Welford step
/// `EwCov` takes. They were the raw sums `Σw·x` and `Σw·x x'`, and the
/// scatter `Σw·x x' − n·x̄x̄'` formed from them on every row lost `n·L²·ε` of
/// itself at a level `L` -- all of a unit variance at `1e8`, where every row
/// then read as a changepoint (review 2026-09-12, C23).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Run {
    /// Rows this run has absorbed, i.e. Adams & MacKay's `r`. Not `n`:
    /// under a fractional row weight, or the `robust` emission's, the two
    /// differ, and the reported run *length* is a count of rows.
    len: f64,
    /// Accumulated weight (rows, or β-weights under `robust`).
    n: f64,
    /// The weighted mean `x̄`, length `d`.
    mean: Vec<f64>,
    /// The weighted scatter `Σ w·(x − x̄)(x − x̄)'`: `d*d` for `gaussian`, the
    /// diagonal (`d`) otherwise.
    m2: Vec<f64>,
    /// What each `mean` leaves out: the mean is a pair no step is rounded
    /// off ([`crate::comp`]; docs/PLAN.md task 101). A run's mean steps by
    /// `w/n`, with no decay, and a plain one given one value row after row
    /// stopped a gap short of it that grows with the run, feeding the scatter
    /// that gap's square on every row. Empty in a state written before it.
    #[serde(default)]
    mean_lo: Vec<f64>,
}

impl Run {
    fn new(d: usize, full: bool) -> Self {
        Self {
            len: 0.0,
            n: 0.0,
            mean: vec![0.0; d],
            m2: vec![0.0; if full { d * d } else { d }],
            mean_lo: vec![0.0; d],
        }
    }

    /// The weighted Welford step: the deviation from the old mean enters the
    /// scatter at weight `w·n/n'` -- `w·(x − x̄)(x − x̄')'`, since `x − x̄' =
    /// (n/n')·(x − x̄)` -- and the mean then moves by `w/n'` of it. A row at
    /// weight 0 (the `robust` emission forgiving it entirely) lengthens the
    /// run and moves nothing else.
    fn add(&mut self, x: &[f64], w: f64, full: bool) {
        let d = x.len();
        self.len += 1.0;
        let n_new = self.n + w;
        if !(w > 0.0 && n_new > 0.0) {
            return;
        }
        let (b, c) = (w / n_new, w * self.n / n_new);
        if self.mean_lo.len() != d {
            self.mean_lo = vec![0.0; d];
        }
        // Deviations from the means as the pairs they are.
        let (mean, lo) = (&self.mean, &self.mean_lo);
        let dev = |i: usize| crate::comp::dev(x[i], mean[i], lo[i]);
        if full {
            // The upper triangle, mirrored, so the scatter stays symmetric
            // to the bit.
            let m2 = &mut self.m2;
            for i in 0..d {
                let ci = c * dev(i);
                for j in i..d {
                    let v = ci * dev(j);
                    m2[i * d + j] += v;
                    if j != i {
                        m2[j * d + i] += v;
                    }
                }
            }
        } else {
            for (i, s) in self.m2.iter_mut().enumerate() {
                let di = dev(i);
                *s += c * di * di;
            }
        }
        for ((m, lo), &xi) in self.mean.iter_mut().zip(self.mean_lo.iter_mut()).zip(x) {
            let di = crate::comp::dev(xi, *m, *lo);
            crate::comp::add(m, lo, b * di);
        }
        self.n = n_new;
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

fn is_false(v: &bool) -> bool {
    !*v
}

/// The normalised run-length posterior of a log joint, shortest run first:
/// uniform where the joint has no finite total.
fn posterior_of(joint: &[f64]) -> Vec<f64> {
    let z = log_sum_exp(joint);
    if !z.is_finite() {
        let k = joint.len();
        return vec![1.0 / k as f64; k];
    }
    joint.iter().map(|l| (l - z).exp()).collect()
}

/// A chance of a break, as the recursion reads it: `ln H` into the empty
/// run and `ln(1 − H)` into every run that goes on.
#[derive(Debug, Clone, Copy)]
struct Branches {
    ln_break: f64,
    ln_grow: f64,
}

impl Branches {
    /// No chance of a break: what follows a row under a hazard on the
    /// clock, since no time has passed since it (the module docs).
    const NONE: Branches = Branches {
        ln_break: f64::NEG_INFINITY,
        ln_grow: 0.0,
    };

    /// A per-row hazard, `H = 1/hazard`, which must leave both branches a
    /// probability: strictly inside `(0, 1)`, or `None`.
    fn per_row(hazard: f64) -> Option<Self> {
        let h = 1.0 / hazard;
        (h > 0.0 && h < 1.0).then(|| Self {
            ln_break: h.ln(),
            ln_grow: (1.0 - h).ln(),
        })
    }

    /// The step of `d` clock units under `τ`: `H = 1 − exp(−d/τ)`, with
    /// `ln(1 − H) = −d/τ` exactly. A step of 0 has no chance, `ln H = −∞`.
    /// `None` for a step that is no step: negative, NaN or infinite.
    fn over(d: f64, tau: f64) -> Option<Self> {
        let x = d / tau;
        (x >= 0.0 && x.is_finite()).then(|| Self {
            ln_break: (-(-x).exp_m1()).ln(),
            ln_grow: -x,
        })
    }
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
    /// The mean weight of the rows learned from, and their count: a row
    /// teaches at its weight over this mean, so a weight's scale moves
    /// nothing (docs/PLAN.md task 147). Nothing here decays, so neither
    /// does this. Ahead of `solve_failures`, which a positional encoding
    /// skips when it is zero.
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
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
            w_mean: 0.0,
            w_rows: 0.0,
            cfg,
        })
    }

    /// A row of weight `w > 0` against the mean weight of the rows learned
    /// from, this one included -- `1` for every row of a constant weight --
    /// with the mean and the count it leaves.
    fn relative(&self, w: f64) -> (f64, f64, f64) {
        let rows = self.w_rows + 1.0;
        let mean = self.w_mean + (w - self.w_mean) / rows;
        (w / mean, mean, rows)
    }

    pub fn cfg(&self) -> &BocpdCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// The normalised run-length posterior as it stands, shortest run first.
    pub fn run_posterior(&self) -> Vec<f64> {
        posterior_of(&self.logjoint)
    }

    /// Output slot labels in emission order.
    pub fn labels(names: &[String]) -> Vec<String> {
        let mut out = vec!["p_change".into(), "run_mode".into(), "run_mean".into()];
        out.extend(names.iter().map(|n| format!("pred_{n}")));
        out.push("loglik".into());
        out
    }

    pub fn n_outputs_for(d: usize) -> usize {
        4 + d
    }

    fn full(&self) -> bool {
        self.cfg.emission == BocpdEmission::Gaussian
    }

    /// One run's posterior mean, `(κ₀μ₀ + n·x̄)/κₙ`, written from the prior's
    /// side, `μ₀ + (n/κₙ)(x̄ − μ₀)`, so a run with no rows is the prior's mean
    /// to the bit.
    fn run_mean_of(&self, run: &Run) -> Vec<f64> {
        let k0 = self.cfg.prior_kappa;
        let mu0 = self.cfg.mu0();
        let kn = k0 + run.n;
        mu0.iter()
            .zip(&run.mean)
            .map(|(m0, m)| m0 + run.n / kn * (m - m0))
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
            // `Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)'`, every term centred.
            let mut psi = self.cfg.psi0();
            for i in 0..d {
                for j in 0..d {
                    let g = k0 * run.n / kn * (run.mean[i] - mu0[i]) * (run.mean[j] - mu0[j]);
                    psi[i * d + j] += run.m2[i * d + j] + g;
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
                let dm = run.mean[i] - mu0[i];
                let psin = psi0[i * d + i] + run.m2[i] + k0 * run.n / kn * dm * dm;
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

    /// What a row is read against, and the chance of a break after it.
    /// Under a per-row hazard: the log joint as it stands, and `H` from the
    /// row's hazard. On the clock: the joint after the step into the row,
    /// its chance of a break applied, and no chance after it, since no time
    /// has passed since the row (the module docs). `(None, None)` for a step
    /// the clock cannot read, which refuses the row.
    fn before_row(&self, y: &[Option<f64>], d_clock: f64) -> (Option<Vec<f64>>, Option<Branches>) {
        if !self.cfg.hazard_on_clock {
            return (None, Branches::per_row(self.hazard_of(y)));
        }
        match Branches::over(d_clock, self.cfg.hazard) {
            Some(step) => (Some(self.transition(step)), Some(Branches::NONE)),
            None => (None, None),
        }
    }

    /// The log joint after a step whose chance of a break is `step`: every
    /// run goes on with `1 − H` of its mass, and `H` of it moves to the
    /// empty run at slot 0, which keeps its own -- a break before a row that
    /// begins a run anyway changes nothing. Mass is kept, and two steps
    /// compose into one of their sum, `(1 − H_a)(1 − H_b) = 1 − H_ab`.
    fn transition(&self, step: Branches) -> Vec<f64> {
        let goes_on = &self.logjoint[1..];
        let moved = log_sum_exp(goes_on) + step.ln_break;
        let mut out = Vec::with_capacity(self.logjoint.len());
        out.push(log_sum_exp(&[self.logjoint[0], moved]));
        out.extend(goes_on.iter().map(|l| l + step.ln_grow));
        out
    }

    /// The row's outputs, read against `joint` -- the log joint as it
    /// stands before the row -- and what the update needs: the new log
    /// joint, with `next`'s chance of a break after the row, and the per-run
    /// weights the row enters with. No `next` refuses the row.
    fn read(
        &self,
        x: &[f64],
        joint: &[f64],
        next: Option<Branches>,
        rel: f64,
    ) -> (Vec<f64>, Option<Update>) {
        let d = self.cfg.n_features;
        let nan = vec![f64::NAN; Self::n_outputs_for(d)];
        let Some(next) = next else {
            return (nan, None);
        };
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
        let pre = posterior_of(joint);
        let mut mode_at = 0;
        for i in 1..r {
            if pre[i] > pre[mode_at] {
                mode_at = i;
            }
        }
        // Every run carries its own length, because the slot index stops
        // being it as soon as `prune_below` drops a run from the middle or
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
        let grow = |weights: &[f64]| {
            let mut new = vec![f64::NEG_INFINITY; r + 1];
            let mut cp = Vec::with_capacity(r);
            for i in 0..r {
                let base = joint[i] + weights[i] * logpi[i];
                new[i + 1] = base + next.ln_grow;
                cp.push(base + next.ln_break);
            }
            new[0] = log_sum_exp(&cp);
            new
        };
        // What the row reports is read as a row of the stream's mean weight,
        // which is what `predict`, never told a weight, can say (the model
        // contract); what it teaches is at its own weight against that mean,
        // `rel` (task 147), in the runs and in the recursion alike.
        let new = grow(&weights);
        let z = log_sum_exp(&new);
        if !z.is_finite() {
            return (nan, None);
        }
        let mut out = Vec::with_capacity(Self::n_outputs_for(d));
        // `P(rₜ ≤ 1)`: see the note above the emission enum. Under a per-row
        // hazard `r = 0` alone is exactly the hazard; on the clock it holds
        // nothing, and this is the chance that the row began a run.
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
        // `min_weight` gates what is *reported*, never what is learned: a
        // gated row still moves the posterior, as it does in every other
        // model here.
        let update = if rel == 1.0 {
            (new, weights)
        } else {
            let scaled: Vec<f64> = weights.iter().map(|w| rel * w).collect();
            (grow(&scaled), scaled)
        };
        if self.n_eff < self.cfg.min_weight {
            return (nan, Some(update));
        }
        (out, Some(update))
    }

    /// Drop the runs below `prune_below` and fold the tail at `max_run`.
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
        if self.cfg.prune_below > 0.0 {
            let keep: Vec<bool> = self
                .logjoint
                .iter()
                .map(|l| (l - z).exp() >= self.cfg.prune_below)
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
        // A row of no weight is scored as a row of the mean weight would be.
        let (rel, w_mean, w_rows) = if weight > 0.0 {
            self.relative(weight)
        } else {
            (1.0, self.w_mean, self.w_rows)
        };
        let (moved, next) = self.before_row(y, d_clock);
        let joint = moved.as_deref().unwrap_or(self.logjoint.as_slice());
        let (pred, extra) = self.read(x, joint, next, rel);
        let out = crate::Step {
            pred,
            n_eff: self.n_eff,
            extra: None,
        };
        if weight <= 0.0 {
            // Advance the clock and learn nothing: no run sees the row, and
            // nothing here decays, so `n_eff` -- the accumulated weight --
            // does not move. Under a per-row hazard the posterior does not
            // move either; on the clock the clock's advance is the step's
            // chance of a break, which applies all the same (hard rule 9).
            if let Some(moved) = moved {
                self.logjoint = moved;
            }
            return out;
        }
        self.n_eff += weight;
        (self.w_mean, self.w_rows) = (w_mean, w_rows);
        let Some((new, weights)) = extra else {
            // The predictive could not be evaluated: the row reports nulls
            // and no run sees it. Counted, so a run of them is visible in
            // `diagnostics` rather than silent (B1). The time still passed:
            // on the clock its chance of a break applies, as on a row of no
            // weight.
            self.solve_failures += 1;
            if let Some(moved) = moved {
                self.logjoint = moved;
            }
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
            grown.add(x, weights[i], full);
            runs.push(grown);
        }
        self.runs = runs;
        self.logjoint = new;
        self.prune();
        out
    }

    /// Never told the row's hazard column, so a per-row hazard read from
    /// the row falls back to the configured one.
    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        self.predict_with(x, &[], d_clock)
    }

    /// The hazard rides in `y[0]` under `hazard_from_row`, so the answer
    /// depends on it exactly as the step's does (C1); on the clock it
    /// depends on `d_clock` as the step's does.
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> crate::Step {
        let (moved, next) = self.before_row(y, d_clock);
        let joint = moved.as_deref().unwrap_or(self.logjoint.as_slice());
        crate::Step {
            pred: self.read(x, joint, next, 1.0).0,
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
            crate::ModelState::Bocpd(m) => {
                let m = (**m).clone();
                // One log-joint per run, at least one run, and every run's
                // moments at the emission's width (review 2026-09-18, B3).
                let d = m.cfg.n_features;
                let m2 = if m.cfg.emission == BocpdEmission::Gaussian {
                    d * d
                } else {
                    d
                };
                if m.runs.is_empty()
                    || m.runs.len() != m.logjoint.len()
                    || m.runs.iter().any(|r| r.mean.len() != d || r.m2.len() != m2)
                {
                    return Err(crate::StateError::Invalid(
                        "bocpd: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
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

    /// A log-joint without its run is refused, where it loaded and panicked
    /// on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Bocpd::new(cfg(2)).unwrap();
        let mut s = m.state();
        let ModelState::Bocpd(inner) = &mut s.model else {
            unreachable!()
        };
        inner.logjoint.push(0.0);
        match Bocpd::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::OnlineModel;

    /// A run's mean steps by `w/n` with no decay, and a plain one given one
    /// value row after row stops where its step rounds to nothing: at a
    /// level of 1e12, hundreds of rounding steps short of the exact mean
    /// after 5,300 rows (docs/PLAN.md task 101). As a pair it stays within
    /// two. Dyadic values make the exact mean's numerator an integer.
    #[test]
    fn a_run_given_one_value_follows_exact_arithmetic() {
        let level = 1e12;
        let mut run = Run::new(1, false);
        let (mut plain, mut n, mut eighths) = (0.0, 0.0, 0u64);
        for i in 0..5_300u64 {
            let e = if i < 300 { (i * 7919) % 8 } else { 3 };
            let x = level + e as f64 / 8.0;
            eighths += e;
            run.add(&[x], 1.0, false);
            n += 1.0;
            plain += (x - plain) / n;
        }
        let want = level + eighths as f64 / (8.0 * n);
        let step = f64::from_bits(want.to_bits() + 1) - want;
        let steps = |m: f64| (m - want).abs() / step;
        assert!(
            steps(plain) > 100.0,
            "the plain mean is close; no case: {} steps",
            steps(plain)
        );
        assert!(
            steps(run.mean[0]) <= 2.0,
            "{} against {want}: {} steps",
            run.mean[0],
            steps(run.mean[0])
        );
    }

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
            prune_below: 0.0,
            max_run: 100_000,
            min_weight: 0.0,
            hazard_on_clock: false,
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
            prune_below: 1e-6,
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
                prune_below: 1e-6,
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
            prune_below: 1e-4,
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
        // `prune_below = 1e-4` drops runs holding up to that much mass each, so
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

    /// A stationary stream keeps its run vector at `max_run`, and a shift
    /// after it is still found at once (REVIEW-2026-09-18 §6; docs/PLAN.md
    /// task 112). Pruning at `1e-6` bounds little here: one regime makes
    /// every "it began `j` rows ago" hypothesis about as likely as the next,
    /// so the posterior spreads over thousands of run lengths (measured
    /// under the spec's defaults: 2,001 runs after 2,000 rows, 7,036 at most
    /// by 20,000), and `max_run` is the bound -- here 1,000, so the fold is
    /// reached within the stream. Every output stays finite, and a 4σ shift
    /// after 3,000 rows takes the most likely run length to at most 3
    /// within 3 rows. (`p_change`, `P(r ≤ 1)`, peaks at the shifted row --
    /// 0.47 after 3,000 rows, measured -- and falls as the new run ages, so
    /// it is not the reading to wait on.)
    #[test]
    fn a_long_stationary_stream_is_held_to_max_run_and_still_sees_a_shift() {
        let max_run = 1_000;
        let mut m = Bocpd::new(BocpdCfg {
            hazard: 250.0,
            prune_below: 1e-6,
            max_run,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(29);
        let steady = 3_000;
        let mut longest = 0;
        let mut modes = Vec::new();
        for i in 0..steady + 3 {
            let x = if i < steady {
                n.normal()
            } else {
                4.0 + n.normal()
            };
            let out = m.step(&[x], &[], 1.0, 1.0);
            longest = longest.max(m.runs.len());
            assert!(m.runs.len() <= max_run, "row {i}: {} runs", m.runs.len());
            if i > 1 {
                assert!(
                    out.pred.iter().all(|v| v.is_finite()),
                    "row {i}: {:?}",
                    out.pred
                );
            }
            if i >= steady {
                modes.push(out.pred[1]);
            }
        }
        assert_eq!(longest, max_run, "the fold was reached");
        // `run_mode` is read before the row, so the first shifted row still
        // reports the old regime; the next ones do not.
        assert!(modes[2] <= 3.0, "run_mode after the shift: {modes:?}");
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
                prune_below: 1e-6,
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
    fn a_weights_scale_moves_nothing_and_a_heavy_row_is_more_evidence() {
        // Task 147: every output equal at a hundred times the weights, a
        // level shift included, `n_eff` aside, which is the weight.
        let run = |scale: f64, heavy: f64| {
            let mut m = Bocpd::new(cfg(1)).unwrap();
            let mut n = Normals::new(23);
            let mut out = Vec::new();
            for i in 0..300 {
                let shift = if i >= 150 { 3.0 } else { 0.0 };
                let w = scale
                    * if i == 150 {
                        heavy
                    } else {
                        0.5 + 0.1 * (i % 7) as f64
                    };
                out.push(m.step(&[shift + n.normal()], &[], 1.0, w).pred);
            }
            (out, m.n_eff())
        };
        let ((one, w1), (many, w100)) = (run(1.0, 1.0), run(100.0, 1.0));
        assert!((w100 - 100.0 * w1).abs() <= 1e-9 * w100);
        for (i, (a, b)) in one.iter().zip(&many).enumerate() {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * (1.0 + x.abs()),
                    "row {i}: {a:?} against {b:?}"
                );
            }
        }
        // The first row of the shift at three times the mean weight reports
        // as a row of the mean weight -- `predict` could say no more -- and
        // teaches at its own: the runs it enters move further toward it, so
        // the predicted mean of the next row is nearer the new level.
        let (light, _) = run(1.0, 1.0);
        let (heavy, _) = run(1.0, 3.0);
        for k in 0..light[150].len() {
            assert_eq!(
                heavy[150][k].to_bits(),
                light[150][k].to_bits(),
                "output {k}"
            );
        }
        assert!(
            heavy[151][3] > light[151][3],
            "{} against {}",
            heavy[151][3],
            light[151][3]
        );
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
            prune_below: 1e-6,
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
            prune_below: 1e-6,
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

    /// `the_posterior_is_the_longhand_algorithm_one` with the rows at `1e8`
    /// and the prior centred there, and the longhand's scatter formed
    /// two-pass -- `Σ(x − x̄)²` over the run's own rows, each taken from the
    /// level first, which is exact for rows within a factor of two of it --
    /// so the oracle loses nothing (review 2026-09-12, C23). The runs' raw
    /// sums lost the scatter to `n·L²·ε`: at `1e8` a unit variance came back
    /// as 0 or 2, and every row read as a changepoint. The tolerance is the
    /// data's own resolution, `ulp(1e8) ≈ 1.5e-8` a row, accumulated along
    /// a run's log joint.
    #[test]
    fn the_posterior_is_the_longhand_at_a_level() {
        let (hazard, level) = (50.0, 1e8);
        let h = 1.0 / hazard;
        let (k0, nu0, psi0) = (1.0, 2.0, 1.0);
        let mut m = Bocpd::new(BocpdCfg {
            hazard,
            prior_mean: Some(vec![level]),
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(3);
        // Each run as the deviations from the level of the rows it holds.
        let mut runs: Vec<Vec<f64>> = vec![vec![]];
        let mut joint: Vec<f64> = vec![1.0];
        let mut worst = 0.0f64;
        for t in 0..150 {
            let x = level + if t < 75 { n.normal() } else { 4.0 + n.normal() };
            let dx = x - level;
            let step = m.step(&[x], &[], 1.0, 1.0);
            let pi: Vec<f64> = runs
                .iter()
                .map(|rows| {
                    let cnt = rows.len() as f64;
                    let (kn, nun) = (k0 + cnt, nu0 + cnt);
                    let bar = if cnt > 0.0 {
                        rows.iter().sum::<f64>() / cnt
                    } else {
                        0.0
                    };
                    let sq: f64 = rows.iter().map(|r| (r - bar) * (r - bar)).sum();
                    let psin = psi0 + sq + k0 * cnt / kn * bar * bar;
                    let var = psin * (kn + 1.0) / (nun * kn);
                    let z = (dx - cnt * bar / kn) / var.sqrt();
                    (ln_gamma((nun + 1.0) / 2.0)
                        - ln_gamma(nun / 2.0)
                        - 0.5 * (nun * std::f64::consts::PI * var).ln()
                        - 0.5 * (nun + 1.0) * (1.0 + z * z / nun).ln())
                    .exp()
                })
                .collect();
            let mut new = vec![0.0; runs.len() + 1];
            for i in 0..runs.len() {
                new[i + 1] = joint[i] * pi[i] * (1.0 - h);
                new[0] += joint[i] * pi[i] * h;
            }
            let z: f64 = new.iter().sum();
            let want = (new[0] + new.get(1).copied().unwrap_or(0.0)) / z;
            worst = worst.max((step.pred[0] - want).abs());
            // Line 6: the new slot is the prior's, and every old one takes
            // the row.
            let mut next = vec![vec![]];
            next.extend(runs.iter().map(|r| {
                let mut r = r.clone();
                r.push(dx);
                r
            }));
            runs = next;
            joint = new.iter().map(|v| v / z).collect();
        }
        assert!(worst < 1e-6, "p_change missed the longhand by {worst}");
    }

    /// `the_gaussian_emission_runs_on_several_columns` shifted to `1e8`, the
    /// prior with it. The raw scatter lost the factorization there: `Ψₙ`
    /// stopped being positive definite, the row reported nulls and counted a
    /// solve failure that was not one (C23). The model is shift-invariant, so
    /// it must see what it sees at the origin, to the data's resolution.
    #[test]
    fn the_gaussian_emission_is_the_same_at_a_level() {
        let run = |level: f64| {
            let mut m = Bocpd::new(BocpdCfg {
                emission: BocpdEmission::Gaussian,
                prior_nu: Some(5.0),
                prior_mean: Some(vec![level; 3]),
                prune_below: 1e-6,
                ..cfg(3)
            })
            .unwrap();
            let mut n = Normals::new(29);
            let mut out = Vec::new();
            for t in 0..400 {
                let shift = if t < 200 { 0.0 } else { 4.0 };
                let f = n.normal();
                let x: Vec<f64> = (0..3)
                    .map(|_| level + shift + 0.7 * f + 0.7 * n.normal())
                    .collect();
                out.push(m.step(&x, &[], 1.0, 1.0).pred);
            }
            (out, m.solve_failures)
        };
        let (origin, _) = run(0.0);
        let (high, failures) = run(1e8);
        assert_eq!(failures, 0, "a solve failed at the level");
        for (t, (a, b)) in origin.iter().zip(&high).enumerate() {
            assert!(b.iter().all(|v| v.is_finite()), "row {t}: {b:?}");
            // `p_change` and `run_mean`; the predictive mean and the log
            // score carry the level and the data's rounding at it.
            assert!(
                (a[0] - b[0]).abs() < 1e-6,
                "row {t}: p_change {} vs {}",
                a[0],
                b[0]
            );
            assert!(
                (a[2] - b[2]).abs() < 1e-6 * (1.0 + a[2]),
                "row {t}: run_mean {} vs {}",
                a[2],
                b[2]
            );
        }
    }

    #[test]
    fn the_labels_and_slot_count_follow_the_columns() {
        let names = ["a".to_string(), "b".to_string()];
        assert_eq!(
            Bocpd::labels(&names),
            [
                "p_change", "run_mode", "run_mean", "pred_a", "pred_b", "loglik"
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
        // Named by the parameter a caller has: `truncate` named none
        // (review 2026-10-05, CD2).
        for prune_below in [1.0, -1e-9, f64::NAN] {
            bad(
                BocpdCfg {
                    prune_below,
                    ..cfg(1)
                },
                "bocpd: prune_below must be in [0, 1)",
            );
        }
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

    // The mutation survivors of the weekly pass (docs/PLAN.md task 158).

    /// `ln Γ(k/2)` for a whole `k >= 1`, by the recurrence `Γ(x + 1) =
    /// x·Γ(x)` from `Γ(1/2) = √π` and `Γ(1) = 1`: a sum of logs, sharing
    /// nothing with the Lanczos series the model uses.
    fn ln_gamma_half(k: u64) -> f64 {
        let (mut x, mut acc) = if k.is_multiple_of(2) {
            (1.0, 0.0)
        } else {
            (0.5, 0.5 * std::f64::consts::PI.ln())
        };
        while x < k as f64 / 2.0 {
            acc += f64::ln(x);
            x += 1.0;
        }
        acc
    }

    /// `ln Σ exp(v)`, written here so the longhands below do not borrow the
    /// model's.
    fn lse(v: &[f64]) -> f64 {
        let top = v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        top + v.iter().map(|x| (x - top).exp()).sum::<f64>().ln()
    }

    /// Below `1/2` the log gamma reflects, `Γ(x)·Γ(1 − x) = π / sin(πx)`.
    /// No caller reaches it -- `ν₀ > 1` keeps every argument above `1/2` --
    /// but the function says it handles it, and these are the tabled values
    /// of `Γ(1/4)`, `Γ(1/10)` and `Γ(1/3)`.
    #[test]
    fn the_log_gamma_reflects_below_one_half() {
        for (x, gamma) in [
            (0.25, 3.625609908221908),
            (0.1, 9.513507698668732),
            (1.0 / 3.0, 2.6789385347077475),
        ] {
            let want = f64::ln(gamma);
            assert!(
                (ln_gamma(x) - want).abs() < 1e-12,
                "{x}: {} vs {want}",
                ln_gamma(x)
            );
        }
        // The recurrence the longhands below use, against the same series
        // where both apply.
        for k in 1..40 {
            let x = k as f64 / 2.0;
            assert!(
                (ln_gamma_half(k) - ln_gamma(x)).abs() < 1e-11 * (1.0 + ln_gamma(x).abs()),
                "Γ({x})"
            );
        }
    }

    /// The `gaussian` emission's predictive is the normal-inverse-Wishart
    /// posterior predictive, a multivariate Student-t (Murphy 2007,
    /// "Conjugate Bayesian analysis of the Gaussian distribution", its
    /// normal-inverse-Wishart section):
    /// `t_{νₙ−d+1}(μₙ, Ψₙ(κₙ+1)/(κₙ(νₙ−d+1)))`, with `κₙ = κ₀ + n`,
    /// `νₙ = ν₀ + n`, `μₙ = (κ₀μ₀ + n·x̄)/κₙ` and `Ψₙ = Ψ₀ + S +
    /// (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)'`. Written out here from each run's own
    /// rows, two-pass, with the test oracle's eigenvalues for the determinant
    /// and its LU for the quadratic form (`crate::oracle`), the gamma
    /// function by its recurrence, and Algorithm 1 in probabilities: nothing
    /// is shared with the model's Cholesky, its Lanczos series or its
    /// centred updates. Three correlated columns, a full `Ψ₀` that is not
    /// diagonal, `κ₀ = 1/2` and a prior mean away from zero, so every term
    /// of the predictive is live; and every output is checked, the run
    /// length's mode and mean among them.
    #[test]
    fn the_gaussian_emission_is_the_longhand_normal_inverse_wishart() {
        use crate::oracle;
        let d = 3usize;
        let (hazard, k0, nu0) = (40.0, 0.5, 5u64);
        let h = 1.0 / hazard;
        let mu0 = [0.3, -0.2, 0.1];
        let psi0 = [1.5, 0.4, 0.2, 0.4, 1.0, -0.3, 0.2, -0.3, 2.0];
        let mut m = Bocpd::new(BocpdCfg {
            hazard,
            emission: BocpdEmission::Gaussian,
            prior_mean: Some(mu0.to_vec()),
            prior_kappa: k0,
            prior_nu: Some(nu0 as f64),
            prior_scale: Some(psi0.to_vec()),
            ..cfg(d)
        })
        .unwrap();
        // `(ln π(x), μₙ)` of the run holding `rows`.
        let predictive = |rows: &[Vec<f64>], x: &[f64]| -> (f64, Vec<f64>) {
            let n = rows.len();
            let nf = n as f64;
            let bar: Vec<f64> = (0..d)
                .map(|i| {
                    if n > 0 {
                        rows.iter().map(|r| r[i]).sum::<f64>() / nf
                    } else {
                        0.0
                    }
                })
                .collect();
            let kn = k0 + nf;
            let mun: Vec<f64> = (0..d).map(|i| (k0 * mu0[i] + nf * bar[i]) / kn).collect();
            // `νₙ − d + 1`, a whole number here.
            let dof = nu0 + n as u64 + 1 - d as u64;
            let dof_f = dof as f64;
            let scale = (kn + 1.0) / (kn * dof_f);
            let sigma: Vec<f64> = (0..d * d)
                .map(|ij| {
                    let (i, j) = (ij / d, ij % d);
                    let s: f64 = rows.iter().map(|r| (r[i] - bar[i]) * (r[j] - bar[j])).sum();
                    let g = k0 * nf / kn * (bar[i] - mu0[i]) * (bar[j] - mu0[j]);
                    (psi0[i * d + j] + s + g) * scale
                })
                .collect();
            let delta: Vec<f64> = (0..d).map(|i| x[i] - mun[i]).collect();
            let q = oracle::quad_form(&sigma, &delta);
            let log_det = oracle::log_det(&sigma);
            let lp = ln_gamma_half(dof + d as u64)
                - ln_gamma_half(dof)
                - 0.5 * d as f64 * (dof_f * std::f64::consts::PI).ln()
                - 0.5 * log_det
                - 0.5 * (dof_f + d as f64) * (q / dof_f).ln_1p();
            (lp, mun)
        };
        let mut runs: Vec<Vec<Vec<f64>>> = vec![vec![]];
        let mut joint = vec![1.0];
        let mut n = Normals::new(41);
        for t in 0..100 {
            let shift = if t < 50 { 0.0 } else { 3.0 };
            let f = n.normal();
            let x: Vec<f64> = (0..d)
                .map(|i| shift * (i as f64 - 1.0) + 0.8 * f + 0.6 * n.normal())
                .collect();
            let got = m.step(&x, &[], 1.0, 1.0).pred;
            let parts: Vec<(f64, Vec<f64>)> = runs.iter().map(|r| predictive(r, &x)).collect();
            let total: f64 = joint.iter().sum();
            let pre: Vec<f64> = joint.iter().map(|j| j / total).collect();
            let mut new = vec![0.0; runs.len() + 1];
            for (i, (lp, _)) in parts.iter().enumerate() {
                new[i + 1] = joint[i] * lp.exp() * (1.0 - h);
                new[0] += joint[i] * lp.exp() * h;
            }
            let z: f64 = new.iter().sum();
            let mode = (0..pre.len()).fold(0, |b, i| if pre[i] > pre[b] { i } else { b });
            let run_mean: f64 = pre.iter().enumerate().map(|(i, p)| p * i as f64).sum();
            let loglik = pre
                .iter()
                .zip(&parts)
                .map(|(p, (lp, _))| p * lp.exp())
                .sum::<f64>()
                .ln();
            assert!(
                (got[0] - (new[0] + new[1]) / z).abs() < 1e-10,
                "row {t}: p_change {} vs {}",
                got[0],
                (new[0] + new[1]) / z
            );
            assert_eq!(got[1], mode as f64, "row {t}: run_mode");
            assert!(
                (got[2] - run_mean).abs() < 1e-9 * (1.0 + run_mean),
                "row {t}: run_mean {} vs {run_mean}",
                got[2]
            );
            for j in 0..d {
                let want: f64 = pre.iter().zip(&parts).map(|(p, (_, mu))| p * mu[j]).sum();
                assert!(
                    (got[3 + j] - want).abs() < 1e-10 * (1.0 + want.abs()),
                    "row {t}: pred_{j} {} vs {want}",
                    got[3 + j]
                );
            }
            assert!(
                (got[3 + d] - loglik).abs() < 1e-9 * (1.0 + loglik.abs()),
                "row {t}: loglik {} vs {loglik}",
                got[3 + d]
            );
            let mut next = vec![vec![]];
            next.extend(runs.iter().map(|r| {
                let mut r = r.clone();
                r.push(x.clone());
                r
            }));
            runs = next;
            joint = new.iter().map(|v| v / z).collect();
        }
        assert_eq!(m.solve_failures, 0);
    }

    /// `robust`, with row weights, against the module docs written out: a
    /// run takes a row at `(w/w̄)·(π(x)/π(mode))^β`, `w̄` the mean weight of
    /// the rows learned from, this one included; the recursion passes
    /// `π^{(w/w̄)·(π(x)/π(mode))^β}`; what the row reports is read at the
    /// mean weight. Per feature a normal-inverse-gamma, with the predictive
    /// `t_{νₙ}(μₙ, ψₙ(κₙ+1)/(νₙκₙ))` from each run's weighted rows, two-pass,
    /// and `π(mode)` the density at `x = μₙ`. Two columns, so the mode is a
    /// product and not one feature's; a wild row, a shift, a heavy row and
    /// uneven weights, so the tempering and the relative weight both move.
    #[test]
    fn the_robust_emission_with_row_weights_is_the_longhand() {
        let d = 2usize;
        let (hazard, k0, nu0, beta, psi0) = (30.0, 0.7, 3.0, 0.4, 1.3);
        let h = 1.0 / hazard;
        let mu0 = [0.2, -0.1];
        let mut m = Bocpd::new(BocpdCfg {
            hazard,
            emission: BocpdEmission::Robust,
            robust_beta: beta,
            prior_mean: Some(mu0.to_vec()),
            prior_kappa: k0,
            prior_nu: Some(nu0),
            prior_scale: Some(vec![psi0]),
            ..cfg(d)
        })
        .unwrap();
        // `(ln π(x), ln π(mode), μₙ)` of a run's weighted rows.
        let predictive = |rows: &[(Vec<f64>, f64)], x: &[f64]| -> (f64, f64, Vec<f64>) {
            let n: f64 = rows.iter().map(|r| r.1).sum();
            let (kn, nun) = (k0 + n, nu0 + n);
            let (mut lp, mut mode, mut mun) = (0.0, 0.0, vec![0.0; d]);
            for i in 0..d {
                let bar = if n > 0.0 {
                    rows.iter().map(|(r, w)| w * r[i]).sum::<f64>() / n
                } else {
                    0.0
                };
                let s: f64 = rows.iter().map(|(r, w)| w * (r[i] - bar).powi(2)).sum();
                mun[i] = (k0 * mu0[i] + n * bar) / kn;
                let psin = psi0 + s + k0 * n / kn * (bar - mu0[i]).powi(2);
                let var = psin * (kn + 1.0) / (nun * kn);
                let base = ln_gamma((nun + 1.0) / 2.0)
                    - ln_gamma(nun / 2.0)
                    - 0.5 * (nun * std::f64::consts::PI * var).ln();
                lp += base - 0.5 * (nun + 1.0) * ((x[i] - mun[i]).powi(2) / (var * nun)).ln_1p();
                mode += base;
            }
            (lp, mode, mun)
        };
        let mut runs: Vec<Vec<(Vec<f64>, f64)>> = vec![vec![]];
        let mut logj = vec![0.0];
        let mut seen = Vec::new();
        let mut n = Normals::new(47);
        let mut least_tempered = 1.0f64;
        for t in 0..120 {
            let x: Vec<f64> = if t == 30 {
                vec![15.0, -12.0]
            } else {
                let shift = if t < 60 { 0.0 } else { 3.0 };
                vec![shift + n.normal(), 0.5 * n.normal()]
            };
            let w = if t == 61 {
                4.0
            } else {
                0.5 + 0.25 * ((t * 5) % 7) as f64
            };
            let got = m.step(&x, &[], 1.0, w).pred;
            seen.push(w);
            let rel = w / (seen.iter().sum::<f64>() / seen.len() as f64);
            let parts: Vec<(f64, f64, Vec<f64>)> = runs.iter().map(|r| predictive(r, &x)).collect();
            let temper: Vec<f64> = parts
                .iter()
                .map(|(lp, mode, _)| (beta * (lp - mode)).exp())
                .collect();
            least_tempered = temper.iter().fold(least_tempered, |a, &b| a.min(b));
            let message = |scale: f64| {
                let mut new = vec![f64::NEG_INFINITY; runs.len() + 1];
                let mut cp = Vec::new();
                for (i, (lp, _, _)) in parts.iter().enumerate() {
                    let b = logj[i] + scale * temper[i] * lp;
                    new[i + 1] = b + (1.0 - h).ln();
                    cp.push(b + h.ln());
                }
                new[0] = lse(&cp);
                new
            };
            let shown = message(1.0);
            let z_pre = lse(&logj);
            let pre: Vec<f64> = logj.iter().map(|l| (l - z_pre).exp()).collect();
            let p_change = (lse(&shown[..2]) - lse(&shown)).exp();
            let mode = (0..pre.len()).fold(0, |b, i| if pre[i] > pre[b] { i } else { b });
            let run_mean: f64 = pre.iter().enumerate().map(|(i, p)| p * i as f64).sum();
            let mix: Vec<f64> = pre
                .iter()
                .zip(&parts)
                .map(|(p, (lp, _, _))| p.ln() + lp)
                .collect();
            assert!(
                (got[0] - p_change).abs() < 1e-10,
                "row {t}: p_change {} vs {p_change}",
                got[0]
            );
            assert_eq!(got[1], runs[mode].len() as f64, "row {t}: run_mode");
            assert!(
                (got[2] - run_mean).abs() < 1e-9 * (1.0 + run_mean),
                "row {t}: run_mean {} vs {run_mean}",
                got[2]
            );
            for j in 0..d {
                let want: f64 = pre
                    .iter()
                    .zip(&parts)
                    .map(|(p, (_, _, mu))| p * mu[j])
                    .sum();
                assert!(
                    (got[3 + j] - want).abs() < 1e-10 * (1.0 + want.abs()),
                    "row {t}: pred_{j} {} vs {want}",
                    got[3 + j]
                );
            }
            assert!(
                (got[3 + d] - lse(&mix)).abs() < 1e-9 * (1.0 + lse(&mix).abs()),
                "row {t}: loglik {} vs {}",
                got[3 + d],
                lse(&mix)
            );
            logj = message(rel);
            let mut next = vec![vec![]];
            next.extend(runs.iter().zip(&temper).map(|(r, tw)| {
                let mut r = r.clone();
                r.push((x.clone(), rel * tw));
                r
            }));
            runs = next;
        }
        // The wild row was forgiven, nearly entirely, by every run.
        assert!(least_tempered < 1e-3, "{least_tempered}");
    }

    /// A row's hazard must leave both branches a probability, `H = 1/hazard`
    /// strictly inside `(0, 1)`. A row hazard of 1 (`H = 1`: no run ever
    /// grows) or of infinity (`H = 0`: no run ever starts) is refused at the
    /// row: it reports nulls, is counted, and moves nothing. (The plumbing
    /// refuses both before they arrive; the model holds the line too.)
    #[test]
    fn a_row_hazard_of_one_or_infinity_is_refused() {
        for bad in [1.0, f64::INFINITY] {
            let mut m = Bocpd::new(BocpdCfg {
                hazard_from_row: true,
                ..cfg(1)
            })
            .unwrap();
            let mut n = Normals::new(5);
            for _ in 0..20 {
                m.step(&[n.normal()], &[None], 1.0, 1.0);
            }
            let before = m.clone();
            let shown = m.predict_with(&[0.3], &[Some(bad)], 1.0);
            assert!(shown.pred.iter().all(|v| v.is_nan()), "{bad}: {shown:?}");
            let out = m.step(&[0.3], &[Some(bad)], 1.0, 1.0);
            assert!(out.pred.iter().all(|v| v.is_nan()), "{bad}: {out:?}");
            assert_eq!(m.runs, before.runs, "{bad}");
            assert_eq!(m.logjoint, before.logjoint, "{bad}");
            assert_eq!(m.solve_failures, before.solve_failures + 1, "{bad}");
            // And a usable row hazard is used.
            let fine = m.step(&[0.3], &[Some(50.0)], 1.0, 1.0);
            assert!(fine.pred.iter().all(|v| v.is_finite()), "{fine:?}");
        }
    }

    /// `run_mode` is the first maximum of the run posterior. With `H = 1/2`
    /// the first row splits the posterior between "a run of none" and "a run
    /// of one" exactly, and the shorter run is the one reported.
    #[test]
    fn a_tie_in_the_run_posterior_reports_the_shorter_run() {
        let mut m = Bocpd::new(BocpdCfg {
            hazard: 2.0,
            ..cfg(1)
        })
        .unwrap();
        m.step(&[0.4], &[], 1.0, 1.0);
        assert_eq!(m.logjoint.len(), 2);
        assert_eq!(m.logjoint[0], m.logjoint[1], "no tie to break");
        let out = m.step(&[-0.2], &[], 1.0, 1.0);
        assert_eq!(out.pred[1], 0.0);
    }

    /// The fold at `max_run` keeps the mass: the last kept run takes the
    /// summed probability of itself and of every longer run, and each
    /// shorter run keeps its own.
    #[test]
    fn the_fold_at_max_run_keeps_every_runs_mass() {
        let mut m = Bocpd::new(cfg(1)).unwrap();
        let mut n = Normals::new(37);
        for _ in 0..30 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
        }
        let p = m.run_posterior();
        assert_eq!(p.len(), 31);
        let cut = 10;
        m.cfg.max_run = cut;
        m.prune();
        let q = m.run_posterior();
        assert_eq!(q.len(), cut);
        for i in 0..cut - 1 {
            assert!((q[i] - p[i]).abs() < 1e-14, "run {i}: {} vs {}", q[i], p[i]);
        }
        let tail: f64 = p[cut - 1..].iter().sum();
        assert!(
            p[cut - 1] > 1e-3 && tail > p[cut - 1] + 1e-3,
            "both halves of the fold carry mass: {} of {tail}",
            p[cut - 1]
        );
        assert!(
            (q[cut - 1] - tail).abs() < 1e-12,
            "{} vs {tail}",
            q[cut - 1]
        );
    }

    /// A state's shape is its emission's: a `gaussian` state of three
    /// columns, whose runs hold `d²` scatter entries each, restores to
    /// itself; one with a run whose scatter is short is refused, its means
    /// the right length or not.
    #[test]
    fn a_gaussian_state_restores_and_a_short_scatter_is_refused() {
        use crate::{ModelState, StateError};
        let mut m = Bocpd::new(BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_nu: Some(5.0),
            ..cfg(3)
        })
        .unwrap();
        let mut n = Normals::new(53);
        for _ in 0..10 {
            m.step(&[n.normal(), n.normal(), n.normal()], &[], 1.0, 1.0);
        }
        assert_eq!(Bocpd::restore(&m.state()).unwrap(), m);
        let mut s = m.state();
        let ModelState::Bocpd(inner) = &mut s.model else {
            unreachable!()
        };
        inner.runs[2].m2.pop();
        match Bocpd::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    /// `solve_failures` is left out of a state that never counted one, so
    /// such a state writes the bytes it always did; it is written where
    /// there were some, so a round trip keeps the count. A row whose feature
    /// is not a number is one: the predictive cannot be evaluated.
    #[test]
    fn solve_failures_is_written_only_when_there_were_some() {
        let mut m = Bocpd::new(cfg(1)).unwrap();
        m.step(&[0.1], &[], 1.0, 1.0);
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("solve_failures").is_none(), "{v}");
        let out = m.step(&[f64::NAN], &[], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.solve_failures, 1);
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["solve_failures"], 1, "{v}");
        let bytes = rmp_serde::to_vec(&m).unwrap();
        let back: Bocpd = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, m);
    }

    /// A row of weight 0 lengthens a run and moves nothing else, to the bit
    /// (`Run::add`'s contract; it is what the `robust` emission's full
    /// forgiveness reaches). Two rows leave the mean a pair that adding zero
    /// would round afresh (`crate::comp::add` says why), so even a step of
    /// zero would show.
    #[test]
    fn a_row_of_no_weight_moves_nothing_in_a_run() {
        for full in [false, true] {
            let mut run = Run::new(1, full);
            run.add(&[0.04], 1.0, full);
            run.add(&[1.99], 1.0, full);
            assert_ne!(
                run.mean[0] + run.mean_lo[0],
                run.mean[0],
                "the pair is one adding zero leaves alone"
            );
            let before = run.clone();
            run.add(&[-7.0], 0.0, full);
            assert_eq!(run.len, before.len + 1.0);
            assert_eq!(run.n.to_bits(), before.n.to_bits());
            assert_eq!(run.mean[0].to_bits(), before.mean[0].to_bits());
            assert_eq!(run.mean_lo[0].to_bits(), before.mean_lo[0].to_bits());
            assert_eq!(run.m2, before.m2);
        }
    }

    /// The configuration's edges, each where the docs put it: `ν₀ > d + 1`
    /// for `gaussian` and `> 1` otherwise; `robust_beta >= 0` whatever the
    /// emission; `max_run >= 2`; `min_weight >= 0`; and `ν₀` absent is
    /// `d + 2`.
    #[test]
    fn the_configuration_edges_are_where_the_docs_put_them() {
        let bad = |c: BocpdCfg, msg: &str| {
            let e = Bocpd::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        let gaussian = |nu: f64| BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_nu: Some(nu),
            ..cfg(2)
        };
        bad(gaussian(3.0), "prior_nu must be finite and > 3 ");
        // The floor buys a finite variance under the Wishart, a finite mean
        // under the gamma, and the message says which (task 159, D2).
        bad(gaussian(3.0), "no finite variance");
        Bocpd::new(gaussian(3.0 + 1e-9)).unwrap();
        bad(
            BocpdCfg {
                prior_nu: Some(1.0),
                ..cfg(1)
            },
            "prior_nu must be finite and > 1 ",
        );
        bad(
            BocpdCfg {
                prior_nu: Some(1.0),
                ..cfg(1)
            },
            "no finite mean",
        );
        Bocpd::new(BocpdCfg {
            prior_nu: Some(1.5),
            ..cfg(1)
        })
        .unwrap();
        for beta in [-1.0, f64::NAN] {
            bad(
                BocpdCfg {
                    robust_beta: beta,
                    ..cfg(1)
                },
                "robust_beta must be finite and >= 0",
            );
        }
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 2,
            ..cfg(1)
        })
        .unwrap();
        for i in 0..5 {
            m.step(&[i as f64], &[], 1.0, 1.0);
        }
        assert_eq!(m.runs.len(), 2);
        bad(
            BocpdCfg {
                min_weight: -1.0,
                ..cfg(1)
            },
            "min_weight must be >= 0",
        );
        Bocpd::new(BocpdCfg {
            min_weight: 5.0,
            ..cfg(1)
        })
        .unwrap();
        assert_eq!(
            BocpdCfg {
                prior_nu: None,
                ..cfg(3)
            }
            .nu0(),
            5.0
        );
    }

    /// `Ψ₀`'s short forms are the matrices they name, run for run: absent is
    /// the identity and a scalar `s` is `sI`; and `ν₀` absent is `d + 2`.
    #[test]
    fn the_prior_scale_and_nu_short_forms_run_as_what_they_name() {
        let run = |c: BocpdCfg| {
            let d = c.n_features;
            let mut m = Bocpd::new(c).unwrap();
            let mut n = Normals::new(43);
            (0..60)
                .map(|t| {
                    let x: Vec<f64> = (0..d)
                        .map(|i| (if t < 30 { 0.0 } else { 2.5 }) * i as f64 + n.normal())
                        .collect();
                    m.step(&x, &[], 1.0, 1.0).pred
                })
                .collect::<Vec<_>>()
        };
        let gaussian = |scale: Option<Vec<f64>>| BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_nu: Some(4.5),
            prior_scale: scale,
            ..cfg(2)
        };
        let identity = run(gaussian(Some(vec![1.0, 0.0, 0.0, 1.0])));
        let twice = run(gaussian(Some(vec![2.0, 0.0, 0.0, 2.0])));
        assert_ne!(identity, twice, "the scale moves the outputs");
        assert_eq!(run(gaussian(None)), identity);
        assert_eq!(run(gaussian(Some(vec![2.0]))), twice);
        let diag = |nu: Option<f64>| BocpdCfg {
            prior_nu: nu,
            ..cfg(3)
        };
        assert_ne!(run(diag(Some(5.0))), run(diag(Some(6.0))));
        assert_eq!(run(diag(None)), run(diag(Some(5.0))));
    }

    /// A matrix `prior_scale` is held to symmetry cell by cell, each upper
    /// entry against its lower one, to `1e-12` of the larger plus `1e-12`.
    /// Of three columns, an asymmetry in `[1][2]` alone is found and named;
    /// one at the tolerance exactly is rounding and is taken, as is one under
    /// it at a large scale or a tiny one.
    #[test]
    fn a_matrix_prior_scale_is_symmetric_to_a_tolerance() {
        let with = |v: Vec<f64>| {
            let d = if v.len() == 9 { 3 } else { 2 };
            BocpdCfg {
                emission: BocpdEmission::Gaussian,
                prior_nu: Some(6.0),
                prior_scale: Some(v),
                ..cfg(d)
            }
        };
        let e = Bocpd::new(with(vec![4.0, 1.0, 0.5, 1.0, 4.0, 0.9, 0.5, 0.5, 4.0])).unwrap_err();
        assert!(e.contains("[1][2] is 0.9 and [2][1] is 0.5"), "{e}");
        // The asymmetry `c` whose tolerance is `c` itself.
        let mut c = 1e-12f64;
        while 1e-12 * (1.0 + c) != c {
            c = 1e-12 * (1.0 + c);
        }
        assert_eq!((0.0 - c).abs(), 1e-12 * (1.0 + 0.0f64.max(c)));
        Bocpd::new(with(vec![1.0, 0.0, c, 1.0])).unwrap();
        Bocpd::new(with(vec![1e7, 1e6, 1e6 * (1.0 + 1e-13), 1e7])).unwrap();
        Bocpd::new(with(vec![1.0, 0.0, 5e-13, 1.0])).unwrap();
    }

    // The hazard on the clock (docs/PLAN.md task 179).

    /// [`cfg`] with `hazard` the expected clock time between changepoints.
    fn on_clock(d: usize, tau: f64) -> BocpdCfg {
        BocpdCfg {
            hazard: tau,
            hazard_on_clock: true,
            ..cfg(d)
        }
    }

    /// Two numbers agree to rounding: relatively, or absolutely near 0, or
    /// both NaN.
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    /// On regular steps `d` a hazard on the clock and the per-row hazard it
    /// stands for, `1/h` with `h = 1 − exp(−d/τ)`, read every row against
    /// the same posterior, from the first row on: the per-row form puts `h`
    /// on the boundary after each row and the clock form on the boundary
    /// before the next, and the first row's step -- 0 here, as a stream's
    /// first step is -- meets only the empty run that row begins. So
    /// `run_mode` agrees exactly and `run_mean`, the predictive means and
    /// the log score to rounding, under each emission and uneven weights.
    /// `p_change` differs by the per-row form's mass at `r = 0`, which is
    /// `H` exactly: `p_row = H + (1 − H)·p_clock`. (Rows of weight 0 are
    /// left out: on the clock one applies its step, and under a per-row
    /// hazard it is no row at all.)
    #[test]
    fn regular_steps_on_the_clock_are_the_per_row_hazard_from_row_one() {
        let (d, tau) = (3.0f64, 120.0f64);
        let h = -(-d / tau).exp_m1();
        let emissions = [
            ("diag", cfg(2)),
            (
                "gaussian",
                BocpdCfg {
                    emission: BocpdEmission::Gaussian,
                    prior_nu: Some(5.0),
                    ..cfg(3)
                },
            ),
            (
                "robust",
                BocpdCfg {
                    emission: BocpdEmission::Robust,
                    robust_beta: 0.1,
                    ..cfg(2)
                },
            ),
        ];
        for (name, base) in emissions {
            let k = base.n_features;
            let mut rows = Bocpd::new(BocpdCfg {
                hazard: 1.0 / h,
                ..base.clone()
            })
            .unwrap();
            // What the per-row form reads its chance as.
            let big_h = 1.0 / rows.cfg.hazard;
            let mut clock = Bocpd::new(BocpdCfg {
                hazard: tau,
                hazard_on_clock: true,
                ..base
            })
            .unwrap();
            let mut n = Normals::new(61);
            let mut moved = 0usize;
            for t in 0..300 {
                let shift = if t < 150 { 0.0 } else { 3.0 };
                let x: Vec<f64> = (0..k).map(|_| shift + n.normal()).collect();
                let w = 0.5 + 0.25 * (t % 4) as f64;
                let step = if t == 0 { 0.0 } else { d };
                let a = rows.step(&x, &[], step, w).pred;
                let b = clock.step(&x, &[], step, w).pred;
                assert_eq!(a[1], b[1], "{name} row {t}: run_mode");
                for slot in 2..a.len() {
                    assert!(
                        close(a[slot], b[slot], 1e-11),
                        "{name} row {t} slot {slot}: {} against {}",
                        a[slot],
                        b[slot]
                    );
                }
                let want = (a[0] - big_h) / (1.0 - big_h);
                assert!(
                    close(b[0], want, 1e-11),
                    "{name} row {t}: p_change {} against {want}",
                    b[0]
                );
                moved += usize::from(b[0] > 0.5);
            }
            assert!(moved > 0, "{name}: the shift was never a break");
            assert_eq!(clock.solve_failures, 0);
        }
    }

    /// Hard rule 9 on the clock: a row of weight 0 advances the clock and
    /// learns nothing, so its step's chance of a break applies and no run
    /// grows. The transition composes: one, two or three rows of weight 0
    /// and then a learned row leave the learned row's outputs, and the
    /// posterior after it, as one step over their summed time would, to
    /// rounding, and the runs to the bit. The rows of weight 0 move the
    /// joint -- the time passed -- and nothing else. A learned row whose
    /// predictive cannot be read lets the time pass the same way.
    #[test]
    fn two_steps_with_nothing_learned_between_compose_into_one() {
        let mut base = Bocpd::new(on_clock(1, 30.0)).unwrap();
        let mut n = Normals::new(71);
        for t in 0..40 {
            let x = if t < 25 { n.normal() } else { 2.5 + n.normal() };
            base.step(&[x], &[], if t == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        for steps in [
            vec![2.0, 1.0],
            vec![0.5, 4.0, 1.5],
            vec![3.0, 0.0, 7.5, 2.0],
        ] {
            let (mut split, mut whole) = (base.clone(), base.clone());
            let (last, quiet) = steps.split_last().unwrap();
            for &d in quiet {
                let (runs, n_eff) = (split.runs.clone(), split.n_eff);
                let before = split.logjoint.clone();
                split.step(&[99.0], &[], d, 0.0);
                assert_eq!(split.runs, runs, "{steps:?}: a row of weight 0 grew a run");
                assert_eq!(split.n_eff, n_eff);
                if d > 0.0 {
                    assert_ne!(split.logjoint, before, "{steps:?}: the time did not pass");
                }
            }
            let x = [2.4];
            let a = split.step(&x, &[], *last, 1.0).pred;
            let b = whole.step(&x, &[], steps.iter().sum(), 1.0).pred;
            for (slot, (u, v)) in a.iter().zip(&b).enumerate() {
                assert!(
                    close(*u, *v, 1e-12),
                    "{steps:?} slot {slot}: {u} against {v}"
                );
            }
            assert_eq!(split.runs, whole.runs, "{steps:?}");
            for (u, v) in split.run_posterior().iter().zip(whole.run_posterior()) {
                assert!((u - v).abs() <= 1e-13, "{steps:?}: {u} against {v}");
            }
        }
        // A learned row whose feature is not a number: a failure, no run
        // sees it, and its step applies as a row of weight 0's would.
        let (mut failed, mut quiet) = (base.clone(), base.clone());
        failed.step(&[f64::NAN], &[], 5.0, 1.0);
        quiet.step(&[f64::NAN], &[], 5.0, 0.0);
        assert_eq!(failed.solve_failures, base.solve_failures + 1);
        assert_eq!(failed.runs, base.runs);
        assert_eq!(failed.logjoint, quiet.logjoint);
        assert_ne!(failed.logjoint, base.logjoint);
    }

    /// A step of 0 has no chance of a break: no time passed, so the row
    /// cannot begin a run, and it reports `p_change` 0 exactly whatever it
    /// holds -- a 20-σ row included. Its `ln h` is `−∞`, which breaks
    /// nothing: no NaN, no failure. Under `prune_below = 0` each such step
    /// leaves one run of no mass in the vector for good; a positive
    /// `prune_below` drops it. The first row's step of 0 is the first run's
    /// beginning, and reports 1.
    #[test]
    fn a_step_of_zero_carries_no_chance_of_a_break() {
        let zero = |t: usize| t == 0 || (15..18).contains(&t) || t == 30;
        for prune_below in [0.0, 1e-6] {
            let mut m = Bocpd::new(BocpdCfg {
                prune_below,
                ..on_clock(1, 20.0)
            })
            .unwrap();
            let mut n = Normals::new(67);
            for t in 0..50 {
                let x = if t == 30 { 20.0 } else { n.normal() };
                let out = m.step(&[x], &[], if zero(t) { 0.0 } else { 1.0 }, 1.0);
                assert!(out.pred.iter().all(|v| v.is_finite()), "row {t}: {out:?}");
                match t {
                    0 => assert_eq!(out.pred[0], 1.0),
                    _ if zero(t) => assert_eq!(out.pred[0], 0.0, "row {t}"),
                    _ => assert!(out.pred[0] > 0.0, "row {t}: {}", out.pred[0]),
                }
            }
            assert_eq!(m.solve_failures, 0);
            let dead = m.logjoint[1..]
                .iter()
                .filter(|l| **l == f64::NEG_INFINITY)
                .count();
            assert_eq!(
                dead,
                if prune_below == 0.0 { 4 } else { 0 },
                "{prune_below}"
            );
        }
    }

    /// `τ` is a rate of breaks, so any positive finite time is one, below a
    /// row's worth included; 0, a negative or a non-finite one is refused by
    /// name. And a hazard read from the row is a per-row hazard, the chance
    /// of a break after its row: refused beside one on the clock.
    #[test]
    fn a_hazard_on_the_clock_is_refused_where_it_cannot_run() {
        for tau in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = Bocpd::new(on_clock(1, tau)).unwrap_err();
            assert!(
                e.contains("hazard as a duration is the expected time between changepoints"),
                "{tau}: {e}"
            );
        }
        Bocpd::new(on_clock(1, 0.5)).unwrap();
        let e = Bocpd::new(BocpdCfg {
            hazard_from_row: true,
            ..on_clock(1, 60.0)
        })
        .unwrap_err();
        assert!(
            e.contains("cannot stand beside a hazard on the clock"),
            "{e}"
        );
    }

    /// A step that is no step -- negative, NaN or infinite, which no clock
    /// hands over -- has no chance of a break to read: the row reports
    /// nulls, a learned one counts a failure, and nothing moves.
    #[test]
    fn a_step_that_is_no_step_refuses_the_row() {
        let mut m = Bocpd::new(on_clock(1, 30.0)).unwrap();
        let mut n = Normals::new(73);
        for t in 0..20 {
            m.step(&[n.normal()], &[], if t == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        for (i, d) in [-1.0, f64::NAN, f64::INFINITY].into_iter().enumerate() {
            let before = m.clone();
            assert!(m.predict(&[0.3], d).pred.iter().all(|v| v.is_nan()), "{d}");
            for w in [0.0, 1.0] {
                let out = m.step(&[0.3], &[], d, w);
                assert!(out.pred.iter().all(|v| v.is_nan()), "{d}: {out:?}");
                assert_eq!(m.runs, before.runs, "{d}");
                assert_eq!(m.logjoint, before.logjoint, "{d}");
            }
            assert_eq!(m.solve_failures, i as u64 + 1, "{d}");
        }
    }

    /// The flag is the configuration's last field and is left out where it
    /// is false, so a state with a per-row hazard writes the bytes it wrote
    /// before it existed; one on the clock writes it. Both round-trip, named
    /// and compact, mid-stream, and go on as the model that never stopped.
    #[test]
    fn the_clock_is_written_only_where_it_is_set() {
        for c in [cfg(1), on_clock(1, 15.0)] {
            let on = c.hazard_on_clock;
            let mut m = Bocpd::new(c).unwrap();
            let mut n = Normals::new(79);
            for t in 0..30 {
                m.step(&[n.normal()], &[], if t % 4 == 0 { 0.0 } else { 2.0 }, 1.0);
            }
            let v = serde_json::to_value(m.cfg()).unwrap();
            assert_eq!(v.get("hazard_on_clock").is_some(), on, "{v}");
            for bytes in [
                rmp_serde::to_vec(&m).unwrap(),
                rmp_serde::to_vec_named(&m).unwrap(),
            ] {
                let mut back: Bocpd = rmp_serde::from_slice(&bytes).unwrap();
                assert_eq!(back, m);
                let mut twin = m.clone();
                for t in 0..10 {
                    let x = [n.normal()];
                    let d = if t % 3 == 0 { 0.0 } else { 1.5 };
                    assert_eq!(back.step(&x, &[], d, 1.0), twin.step(&x, &[], d, 1.0));
                }
            }
        }
    }
}
