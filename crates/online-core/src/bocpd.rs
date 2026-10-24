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
//! sufficient statistics, so a row costs `O(runs·d)` under the diagonal
//! emission and `O(runs·d³)` under the full one, which factors each run's
//! `d × d` predictive scale every row (measured with 50 runs held: 187,
//! 251, 464, 1,040 and 3,095 ns a run at d = 2, 4, 8, 16 and 32, ×3.0 for
//! the last doubling; review 2026-10-23, D-3).
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
//! mass falls below `prune_below` are dropped, and `max_run` caps a run's
//! length. The joint is not renormalised after a drop (see `prune`): every
//! output is read against its log-sum, so no reported number needs it. The runs that reach `max_run − 1` rows are
//! folded into the one of them carrying the most mass, the oldest among
//! equals, which keeps its statistics, scaled back to `max_run − 1` rows'
//! weight as a decay would scale them. The capped slot is then an
//! exponentially weighted run: each row a Welford step and a scale by
//! `cap / (cap + 1)`, `cap = max_run − 1`, so its mean steps by `1 / max_run`
//! of the row's deviation and its half-life is `ln 2 / ln((cap + 1) / cap)`,
//! about `0.69 (max_run − 1)` rows: 13.5 at `max_run = 20`, about 6,900 at
//! the default 10,000. Both are approximations
//! with a knob, and a test measures what the knob costs. In a stationary
//! stream the pruning bounds little: one regime makes every "it began `j`
//! rows ago" hypothesis about as likely as the next, and the posterior
//! spreads over thousands of run lengths -- 2,001 kept after 2,000 rows and
//! 7,036 at most by 20,000 under the defaults (`prune_below = 1e-6`,
//! measured 2026-09-27). `max_run` is then the bound on the vector -- at
//! most `max_run` distinct lengths -- and on the `O(runs·d)` (diagonal) or
//! `O(runs·d³)` (full) work a row costs; a shift collapses it to a few dozen.
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
//! **`robust` is not free of the data's units; centre and scale its
//! inputs.** A change of units multiplies every run's predictive density by
//! one constant `c`. Under `gaussian` and `diag` that constant multiplies
//! every joint alike, and the posterior's normalisation takes it out. The
//! tempered message `πᵣ^{w(x)}` carries it as `c^{w(x)}`, and `w(x)` differs
//! from run to run, so the runs' odds move with the units. Measured
//! (docs/PLAN.md task 195): at a level of `1e4` with a spread of `1e-2`,
//! `p_change` differed by up to 0.86 from the same stream at level 0 and
//! spread 1, and `run_mode` on 427 of 596 rows; at a spread of `2e3` the
//! break at row 200 was missed.
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
//! # The prior from the first rows (docs/PLAN.md task 195, U4)
//!
//! The prior decides what counts as surprising, and in the data's units: a
//! scale of 1 makes every row of data in 1e-4 a break, and a mean of 0
//! leaves no row of data at 1e4 any density under a run that begins on it.
//! So where `prior_mean` or `prior_scale` is left out, the first
//! `warm_rows` learned rows (default `d + 2`) are held and report nothing;
//! their weighted mean and covariance, with the weights over their largest,
//!
//! ```text
//! x̄ = Σ w x / V₁        S = Σ w (x − x̄)(x − x̄)ᵀ / (V₁ − V₂/V₁)        V₁ = Σ w,  V₂ = Σ w²
//! ```
//!
//! (the diagonal under `diag` and `robust`) set what was left out, and the
//! held rows are then read in order as rows of a model that had that prior
//! from the start. At the default `prior_nu` the prior's mean variance is
//! `S`. A variance that is not above 0 -- a feature that did not move --
//! takes `2^-52·max(x̄², 1)`; under `gaussian` an `S` that is not positive
//! definite gives its diagonal. A row of weight 0 inside the warm-up learns
//! nothing, and its step joins the next held row's. `gaussian` and `diag`
//! are then free of the data's units; `robust`'s tempered message `π^w`
//! still carries them, as `c^-w` with `w` different from run to run.
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
//! is learned; a usable row that fails so is counted, in `n_eff` and as a
//! failure. A row with a feature or a weight that is not usable -- not a
//! number, infinite, or past the input bound -- is a row of weight 0 by the
//! rule every model keeps (`OnlineModel`): its step's chance applies, and it
//! is counted nowhere. A hazard in the targets slot that is not usable is
//! absent, the configured one (task 183). A step that is no step (negative,
//! NaN or infinite, which the plumbing never hands over) refuses the row, as
//! a hazard at or below 1 does.
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
/// It is an alarm and not the answer. Measured: a tenfold step in the
/// standard deviation, under a prior of the rows' scale, takes it to 0.83
/// on the row of the break; a four-sigma mean shift under
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
    /// `diag`'s pair with β-power-weighted rows; see the module docs. Not
    /// free of the data's units, as `diag` and `gaussian` are: centre and
    /// scale its inputs.
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
    /// `μ₀`; when absent, the first `warm_rows` learned rows' sample mean
    /// ([`Bocpd`]'s warm-up).
    pub prior_mean: Option<Vec<f64>>,
    /// `κ₀`, the prior's weight in rows.
    pub prior_kappa: f64,
    /// `ν₀`; when absent, the smallest integer that gives the prior's
    /// variance a mean: `d + 2` for `gaussian`'s inverse-Wishart (`ν > d +
    /// 1`), 3 for the per-feature normal-inverse-gamma of `diag` and
    /// `robust` (`ν > 2`), whatever `d` (docs/PLAN.md task 195, U5).
    pub prior_nu: Option<f64>,
    /// `Ψ₀`: a scalar for `sI`, or a `d×d` matrix -- the prior guess at
    /// the covariance, in the data's units. When absent, the first
    /// `warm_rows` learned rows' sample covariance, or its diagonal under
    /// `diag` and `robust` ([`Bocpd`]'s warm-up). The identity it was in
    /// their place made every row of data in 1e-4 look like a changepoint
    /// and no row of data at 1e4 ever begin a run (docs/PLAN.md task 195,
    /// U4).
    pub prior_scale: Option<Vec<f64>>,
    /// `robust`'s β; 0 makes it `diag`.
    pub robust_beta: f64,
    /// Drop runs below this normalised mass.
    pub prune_below: f64,
    /// Cap a run's length here. The runs that reach `max_run − 1` rows are
    /// folded into the one of them carrying the most mass, the oldest among
    /// equals: it holds their summed mass and keeps its own statistics,
    /// scaled back to `max_run − 1` rows, its weight and scatter times
    /// `(max_run − 1) / len`, its mean unchanged. So `max_run` bounds the
    /// memory of a run, and with it the vector, at most `max_run` lengths
    /// (`0 ..= max_run − 1`), and `run_mode` saturates one below it. The
    /// capped run is an exponentially weighted run, scaled each row by
    /// `(max_run − 1) / max_run`: its mean steps by `1 / max_run` of the
    /// row's deviation, and its half-life is `ln 2 / ln(max_run / (max_run −
    /// 1))`, about `0.69 (max_run − 1)` rows. The run that dominates a quiet
    /// stream keeps its memory at the cap, however far pruning has thinned
    /// the runs younger than it; after a break the post-break run takes the
    /// slot once it reaches the cap.
    pub max_run: usize,
    pub min_weight: f64,
    /// Learned rows buffered to set the prior from, where `prior_mean` or
    /// `prior_scale` is absent; `None` is `d + 2`, the name and the default
    /// `hmm` and `kmeans` give theirs (docs/PLAN.md task 195, U4). Refused
    /// beside both, which leave nothing to learn.
    #[serde(default)]
    pub warm_rows: Option<usize>,
    /// `hazard` is `τ`, the expected clock time between changepoints: the
    /// step of `d` clock units into a row carries the chance `1 −
    /// exp(−d/τ)` of a break, applied before the row is read (the module
    /// docs). Last, and left out where it is false, so a state with a
    /// per-row hazard writes the bytes it always did; a state from before
    /// it reads as a per-row hazard, which it was.
    #[serde(default, skip_serializing_if = "is_false")]
    pub hazard_on_clock: bool,
}

/// The most rows the warm-up buffers, MiB: `kmeans`' budget for its own.
pub const WARM_BUDGET_MIB: f64 = 256.0;

impl BocpdCfg {
    pub fn nu0(&self) -> f64 {
        self.prior_nu.unwrap_or(match self.emission {
            BocpdEmission::Gaussian => self.n_features as f64 + 2.0,
            BocpdEmission::Diag | BocpdEmission::Robust => 3.0,
        })
    }

    /// Whether the prior is set from the first rows: a `prior_mean` or a
    /// `prior_scale` left out.
    pub fn warms_up(&self) -> bool {
        self.prior_mean.is_none() || self.prior_scale.is_none()
    }

    /// The learned rows the warm-up buffers: `warm_rows`, or `d + 2`.
    pub fn warm_rows_or_default(&self) -> usize {
        self.warm_rows
            .unwrap_or_else(|| self.n_features.saturating_add(2))
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
            return Err("bocpd: mean_precision_prior must be finite and > 0".into());
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
                "bocpd: degrees_of_freedom_prior must be finite and > {floor} for this covariance_type (got {nu}); \
                 below it the predictive has no finite {what}"
            ));
        }
        if let Some(v) = &self.prior_scale {
            if v.len() != 1 && v.len() != d * d {
                return Err(format!(
                    "bocpd: covariance_prior is a scalar or a {d}x{d} matrix, got {} values",
                    v.len()
                ));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err("bocpd: covariance_prior must be finite".into());
            }
            // A scale matrix that is not positive definite gives a
            // predictive with no density: every row would report nulls and
            // count as a failure, with nothing to say why
            // (docs/REVIEW-E54-E64.md B2).
            if v.len() == 1 && v[0] <= 0.0 {
                return Err(format!(
                    "bocpd: a scalar covariance_prior is the prior scale of the variance and must be \
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
                                "bocpd: a matrix covariance_prior must be symmetric; [{i}][{j}] is \
                                 {a} and [{j}][{i}] is {b}"
                            ));
                        }
                    }
                }
                if !matches!(SpdFactor::of(v, d), Some(f) if f.attempts() == 0) {
                    return Err(
                        "bocpd: a matrix covariance_prior must be positive definite -- it is the \
                         prior's scale matrix Ψ₀, and the predictive has no density without one"
                            .into(),
                    );
                }
            }
        }
        if let Some(m) = &self.prior_mean {
            if m.len() != d {
                return Err(format!(
                    "bocpd: mean_prior must be {d} values, got {}",
                    m.len()
                ));
            }
            if m.iter().any(|x| !x.is_finite()) {
                return Err("bocpd: mean_prior must be finite".into());
            }
        }
        if self.robust_beta < 0.0 || !self.robust_beta.is_finite() {
            return Err("bocpd: robust_beta must be finite and >= 0".into());
        }
        if self.robust_beta > 0.0 && self.emission != BocpdEmission::Robust {
            return Err(format!(
                "bocpd: robust_beta tempers the per-feature predictive (covariance_type = \
                 \"diag\"), not {:?}",
                self.emission
            ));
        }
        if self.emission == BocpdEmission::Robust && self.robust_beta <= 0.0 {
            return Err(
                "bocpd: the tempered predictive needs robust_beta > 0; at 0 it is untempered"
                    .into(),
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
        if let Some(n) = self.warm_rows {
            if !self.warms_up() {
                return Err(
                    "bocpd: warm_rows sets the prior from the first rows, and prior_mean and \
                     prior_scale are both given; leave one out, or warm_rows"
                        .into(),
                );
            }
            if n < 2 {
                return Err(format!(
                    "bocpd: warm_rows must be at least 2, got {n}: the prior's scale is a \
                     variance, which takes two rows"
                ));
            }
        }
        // The buffer is `d` values a row, sized by a spec value, so it has a
        // ceiling, as `kmeans`' has (review 2026-10-06, CD10).
        let rows = self.warm_rows_or_default() as f64;
        let mib = rows * (self.n_features as f64 + 4.0) * 8.0 / f64::from(1 << 20);
        if self.warms_up() && mib > WARM_BUDGET_MIB {
            return Err(format!(
                "bocpd: the warm-up buffer would hold {} MiB (warm_rows {rows} rows of {d} \
                 features), over the {WARM_BUDGET_MIB} MiB it is held to; lower warm_rows",
                crate::budget::mib_over(mib, WARM_BUDGET_MIB),
            ));
        }
        Ok(())
    }
}

/// The prior the first rows set, where the cfg leaves `prior_mean` or
/// `prior_scale` out: their weighted sample mean and covariance, `Ψ₀` as a
/// `d×d` matrix (diagonal under `diag` and `robust`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DataPrior {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl DataPrior {
    /// The weighted mean `x̄ = Σw·x / V₁` and covariance `S = Σw·(x −
    /// x̄)(x − x̄)' / (V₁ − V₂/V₁)`, `V₁ = Σw` and `V₂ = Σw²` -- reliability
    /// weights, numpy's `cov(aweights=)`, the sample covariance at equal
    /// weights -- with the weights over their largest, so a weight's scale
    /// moves nothing. A variance that is not above 0, a feature that did
    /// not move over the rows, takes the floor `2^-52·max(x̄², 1)`: the
    /// spread of a value known to `2^-26` of its level, or of 1 below a
    /// level of 1. A predictive of no spread has no density, and every row
    /// after it failed. Under `gaussian` a covariance that is not positive
    /// definite with that floor -- collinear features -- gives its diagonal.
    fn of(rows: &[WarmRow], d: usize, full: bool) -> Self {
        let top = rows.iter().map(|r| r.weight).fold(0.0, f64::max);
        let w: Vec<f64> = rows.iter().map(|r| r.weight / top).collect();
        let v1: f64 = w.iter().sum();
        let v2: f64 = w.iter().map(|w| w * w).sum();
        let mut mean = vec![0.0; d];
        for (r, wi) in rows.iter().zip(&w) {
            for (m, x) in mean.iter_mut().zip(&r.x) {
                *m += wi * x;
            }
        }
        mean.iter_mut().for_each(|m| *m /= v1);
        let reliable = v1 - v2 / v1;
        let denom = if reliable > 0.0 { reliable } else { v1 };
        let mut scale = vec![0.0; d * d];
        for (r, wi) in rows.iter().zip(&w) {
            for i in 0..d {
                let di = r.x[i] - mean[i];
                let upto = if full { d } else { i + 1 };
                for j in i..upto {
                    scale[i * d + j] += wi * di * (r.x[j] - mean[j]);
                }
            }
        }
        // The upper triangle, mirrored: the matrix is symmetric to the bit.
        for i in 0..d {
            for j in i..d {
                let v = scale[i * d + j] / denom;
                scale[i * d + j] = v;
                scale[j * d + i] = v;
            }
        }
        for i in 0..d {
            let v = scale[i * d + i];
            if !(v > 0.0 && v.is_finite()) {
                scale[i * d + i] = 2f64.powi(-52) * (mean[i] * mean[i]).max(1.0);
            }
        }
        if full && !matches!(SpdFactor::of(&scale, d), Some(f) if f.attempts() == 0) {
            for i in 0..d {
                for j in 0..d {
                    if i != j {
                        scale[i * d + j] = 0.0;
                    }
                }
            }
        }
        Self { mean, scale }
    }
}

/// A learned row the warm-up holds until the prior is set, to be read then:
/// its features, its hazard where the row carries one, its step -- with the
/// steps of the rows of no weight before it, which learn nothing and whose
/// time the next row's step carries, as two steps compose into one -- and
/// its weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WarmRow {
    x: Vec<f64>,
    hazard: Option<f64>,
    d_clock: f64,
    weight: f64,
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
    /// that gap's square on every row. One per mean.
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

    /// Scale the run back to `len` rows, as a decay scales a run's
    /// statistics: the weight `n` and the scatter by `len / self.len`, the
    /// mean left where it is. The run's predictive keeps its centre and its
    /// spread and loses confidence in both, as a run of `len` rows like
    /// these would have.
    fn shrink_to(&mut self, len: f64) {
        let ratio = len / self.len;
        self.n *= ratio;
        for s in self.m2.iter_mut() {
            *s *= ratio;
        }
        self.len = len;
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
/// lengths, the weight each existing run takes the row at, and the new
/// joint's `log_sum_exp` where `read` took it, which [`Bocpd::prune`] would
/// take again (task 250).
type Update = (Vec<f64>, Vec<f64>, Option<f64>);

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
    /// The prior the first rows set, once they have, where the cfg leaves
    /// `prior_mean` or `prior_scale` out ([`DataPrior`]; docs/PLAN.md task
    /// 195, U4).
    data_prior: Option<DataPrior>,
    /// The learned rows held until there are `warm_rows` of them, and the
    /// step of the rows of no weight since the last one, which the next
    /// carries. Empty once the prior is set.
    warm: Vec<WarmRow>,
    warm_gap: f64,
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
    /// The predictive's gamma-function terms already taken, by the degrees
    /// of freedom they were taken at (task 250). Not state.
    #[serde(skip)]
    gammas: GammaCache,
}

/// The Student-t predictive's gamma terms, `ln Γ((ν+a)/2) − ln Γ(ν/2)`:
/// `a` is the dimension the density is over, `d` under `gaussian` and 1 per
/// feature otherwise. Grouped as the density's expression groups them, so
/// the value is the one the expression took inline.
fn gamma_diff(nu: f64, a: f64) -> f64 {
    ln_gamma((nu + a) / 2.0) - ln_gamma(nu / 2.0)
}

/// [`gamma_diff`] by the bits of `ν` and `a` (task 250): a run's `ν` is the
/// prior's plus its weight, so under rows of one weight the run of length
/// `r` asks this row for the value the run of length `r` asked the last
/// row for, and two `ln_gamma` a run a row -- four logarithms and a Lanczos
/// sum -- become a lookup. A slot keeps the value `gamma_diff` gave for
/// exactly those bits, so a hit is the value a miss computes: two `ν` that
/// differ in any bit, `0.0` and `-0.0` included, never share an entry, and
/// a different `a` empties every slot. Direct-mapped, [`Self::SLOTS`]
/// entries at most, filled as it is asked. Not state: a state file carries
/// none of it, a restored model starts empty, and two models are the same
/// model whatever each has cached, so [`PartialEq`] ignores it.
#[derive(Debug, Clone, Default)]
struct GammaCache {
    /// The bits of the `a` the slots were filled for.
    a: u64,
    keys: Vec<u64>,
    vals: Vec<f64>,
    /// Lookups answered from a slot, and computed: the tests read them to
    /// show the cache was used.
    #[cfg(test)]
    hits: u64,
    #[cfg(test)]
    misses: u64,
}

impl PartialEq for GammaCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl GammaCache {
    const SLOTS: usize = 256;

    /// The slot for `ν`'s bits: their low bits mixed with the exponent's
    /// and the mantissa's top, which are what change between the `ν` of
    /// one stream's runs. Where two keys share a slot the later evicts the
    /// earlier; the key is checked whole, so the slot only places it.
    #[inline]
    fn slot(bits: u64) -> usize {
        ((bits ^ (bits >> 29) ^ (bits >> 47)) as usize) & (Self::SLOTS - 1)
    }

    /// [`gamma_diff`]`(nu, a)`, from the slot for `nu`'s bits when it holds
    /// them.
    #[inline]
    fn get(&mut self, nu: f64, a: f64) -> f64 {
        if nu.is_nan() {
            return gamma_diff(nu, a);
        }
        if self.keys.is_empty() || self.a != a.to_bits() {
            // No `nu` asked for here has a NaN's bits, so they mark a slot
            // never filled.
            self.a = a.to_bits();
            self.keys = vec![f64::NAN.to_bits(); Self::SLOTS];
            self.vals = vec![0.0; Self::SLOTS];
        }
        let bits = nu.to_bits();
        let at = Self::slot(bits);
        if self.keys[at] == bits {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            return self.vals[at];
        }
        #[cfg(test)]
        {
            self.misses += 1;
        }
        let v = gamma_diff(nu, a);
        self.keys[at] = bits;
        self.vals[at] = v;
        v
    }
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
            gammas: GammaCache::default(),
            w_mean: 0.0,
            w_rows: 0.0,
            data_prior: None,
            warm: Vec::new(),
            warm_gap: 0.0,
            cfg,
        })
    }

    /// Whether rows are still being held for the prior (the warm-up), so
    /// that a row reports nothing.
    pub fn warming(&self) -> bool {
        self.cfg.warms_up() && self.data_prior.is_none()
    }

    /// `μ₀` as the model reads it: the cfg's, or the first rows' mean.
    fn mu0(&self) -> Vec<f64> {
        match (&self.cfg.prior_mean, &self.data_prior) {
            (None, Some(p)) => p.mean.clone(),
            _ => self.cfg.mu0(),
        }
    }

    /// `Ψ₀` as the model reads it, `d×d`: the cfg's, or the first rows'
    /// covariance.
    fn psi0(&self) -> Vec<f64> {
        match (&self.cfg.prior_scale, &self.data_prior) {
            (None, Some(p)) => p.scale.clone(),
            _ => self.cfg.psi0(),
        }
    }

    /// The prior's mean and scale as the model reads them (`Ψ₀` as a `d×d`
    /// matrix), `None` while the first rows are still held for them.
    pub fn prior(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        (!self.warming()).then(|| (self.mu0(), self.psi0()))
    }

    /// A row the warm-up holds: it reports nothing and counts in `n_eff`,
    /// as a learned row does. The `warm_rows`-th sets the prior from them
    /// all and they are read, in order, as rows of a model that had it from
    /// the start, as `hmm` replays its warm-up into its states. A row of no
    /// weight learns nothing: its step joins the next row's.
    fn hold(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        let out = crate::Step {
            pred: vec![f64::NAN; Self::n_outputs_for(self.cfg.n_features)],
            n_eff: self.n_eff,
            extra: None,
        };
        if weight <= 0.0 {
            self.warm_gap += d_clock;
            return out;
        }
        self.warm.push(WarmRow {
            x: x.to_vec(),
            hazard: if self.cfg.hazard_from_row {
                y.first().copied().flatten()
            } else {
                None
            },
            d_clock: d_clock + self.warm_gap,
            weight,
        });
        self.warm_gap = 0.0;
        self.n_eff += weight;
        if self.warm.len() >= self.cfg.warm_rows_or_default() {
            let full = self.full();
            self.data_prior = Some(DataPrior::of(&self.warm, self.cfg.n_features, full));
            let rows = std::mem::take(&mut self.warm);
            // What the rows learn is counted as they are read, failures
            // left out, as on any row.
            self.n_eff = 0.0;
            for r in &rows {
                let hazard = [r.hazard];
                let y: &[Option<f64>] = if self.cfg.hazard_from_row {
                    &hazard
                } else {
                    &[]
                };
                self.learn(&r.x, y, r.d_clock, r.weight);
            }
        }
        out
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
        let mu0 = self.mu0();
        (0..mu0.len())
            .map(|i| self.run_mean_at(run, &mu0, i))
            .collect()
    }

    /// Coordinate `i` of [`Self::run_mean_of`], against the prior's mean
    /// `mu0` the caller read once: the same expression, without a vector per
    /// run per row (task 250).
    #[inline]
    fn run_mean_at(&self, run: &Run, mu0: &[f64], i: usize) -> f64 {
        let kn = self.cfg.prior_kappa + run.n;
        mu0[i] + run.n / kn * (run.mean[i] - mu0[i])
    }

    /// `(ln π(x), ln π(mode))` for one run: the log posterior predictive of
    /// the row, and of the predictive's own mode, which is what the
    /// `robust` weight is relative to. `mu0` and `psi0` are the prior's
    /// ([`Self::mu0`], [`Self::psi0`]), read once a row by the caller rather
    /// than copied for every run (task 250).
    fn log_predictive(
        &self,
        run: &Run,
        x: &[f64],
        mu0: &[f64],
        psi0: &[f64],
        cache: Option<&mut GammaCache>,
    ) -> Option<(f64, f64)> {
        let d = self.cfg.n_features;
        let (k0, nu0) = (self.cfg.prior_kappa, self.cfg.nu0());
        let kn = k0 + run.n;
        let nun = nu0 + run.n;
        if self.full() {
            let mun: Vec<f64> = (0..d).map(|i| self.run_mean_at(run, mu0, i)).collect();
            // `Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)'`, every term centred.
            let mut psi = psi0.to_vec();
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
            // A row that is not a number has no density. `step` and
            // `predict_with` refuse one before it gets here (`OnlineModel`,
            // task 183); the check stays beside the form, whose clamp once
            // took a NaN to 0, so the row read as sitting on the mean and
            // was learned (task 179), and which keeps the NaN now (task 181),
            // rather than leave it to how `log_sum_exp`'s `f64::max` reads a
            // NaN density.
            if delta.iter().any(|v| v.is_nan()) {
                return None;
            }
            let q = f.quad_forms(&delta, d, 1)[0];
            let df = d as f64;
            // `(Γ − Γ) − ..` as the expression groups, so the cached
            // difference is the same value (task 250).
            let gammas = match cache {
                Some(c) => c.get(dof, df),
                None => gamma_diff(dof, df),
            };
            let base = gammas - 0.5 * df * (dof * std::f64::consts::PI).ln() - 0.5 * f.log_det();
            Some((base - 0.5 * (dof + df) * (1.0 + q / dof).ln(), base))
        } else {
            // Per feature, a normal-inverse-gamma: `t_{νₙ}` with scale²
            // `ψₙ(κₙ+1)/(νₙκₙ)`. The features share `νₙ`, so the gamma
            // functions' difference is taken once a run, not once a feature:
            // the same value, `(Γ − Γ) − ½ ln(..)` as the expression groups
            // (task 250).
            let gammas = match cache {
                Some(c) => c.get(nun, 1.0),
                None => gamma_diff(nun, 1.0),
            };
            let mut lp = 0.0;
            let mut mode = 0.0;
            for i in 0..d {
                let dm = run.mean[i] - mu0[i];
                let psin = psi0[i * d + i] + run.m2[i] + k0 * run.n / kn * dm * dm;
                let var = psin * (kn + 1.0) / (nun * kn);
                if var.is_nan() || var <= 0.0 || nun <= 0.0 {
                    return None;
                }
                let z = (x[i] - self.run_mean_at(run, mu0, i)) / var.sqrt();
                let base = gammas - 0.5 * (nun * std::f64::consts::PI * var).ln();
                lp += base - 0.5 * (nun + 1.0) * (1.0 + z * z / nun).ln();
                mode += base;
            }
            Some((lp, mode))
        }
    }

    /// This row's hazard: the value in the targets slot under
    /// `hazard_from_row`, the configured one otherwise. A missing value
    /// falls back to the configured hazard, and so does one that is not
    /// usable (not a number, infinite, or past the input bound), which is
    /// absent by the time it reaches here (`OnlineModel`, task 183); a
    /// usable one at or below 1 is no hazard, and refuses the row
    /// (`Branches::per_row`).
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
        mut cache: Option<&mut GammaCache>,
    ) -> (Vec<f64>, Option<Update>) {
        let d = self.cfg.n_features;
        let nan = vec![f64::NAN; Self::n_outputs_for(d)];
        let Some(next) = next else {
            return (nan, None);
        };
        let r = self.runs.len();
        let mut logpi = Vec::with_capacity(r);
        let mut weights = Vec::with_capacity(r);
        let (mu0, psi0) = (self.mu0(), self.psi0());
        for run in &self.runs {
            let Some((lp, mode)) = self.log_predictive(run, x, &mu0, &psi0, cache.as_deref_mut())
            else {
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
        // The predictive mean `Σ pᵣ mᵣ`, mixed as deviations from the most
        // probable run's mean, `m* + Σ pᵣ (mᵣ − m*)`: the posterior sums to
        // 1 only to its rounding, and that times each `mᵣ` was the level's
        // share of it, 2.3e-13 of a feature at 1e12 (1,040 of its rounding
        // steps); against `m*` it is a share of the runs' spread
        // (docs/PLAN.md task 215, D3).
        let anchor = self.run_mean_of(&self.runs[mode_at]);
        let mut mixed = vec![0.0; d];
        for (i, run) in self.runs.iter().enumerate() {
            for (f, (o, a)) in mixed.iter_mut().zip(&anchor).enumerate() {
                *o += pre[i] * (self.run_mean_at(run, &mu0, f) - a);
            }
        }
        let pred: Vec<f64> = anchor.iter().zip(&mixed).map(|(a, o)| a + o).collect();
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
            (new, weights, Some(z))
        } else {
            let scaled: Vec<f64> = weights.iter().map(|w| rel * w).collect();
            (grow(&scaled), scaled, None)
        };
        if self.n_eff < self.cfg.min_weight {
            return (nan, Some(update));
        }
        (out, Some(update))
    }

    /// Drop the runs below `prune_below` and fold the runs at the length cap.
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
    fn prune(&mut self, z: Option<f64>) {
        // `read`'s, where it took it of this joint: the same function of
        // the same values.
        let z = z.unwrap_or_else(|| log_sum_exp(&self.logjoint));
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
        // Fold every run that has reached `max_run - 1` rows into the one of
        // them carrying the most mass, scaled back to `max_run - 1` rows: it
        // takes their summed mass, keeps its own statistics and takes the
        // tail's place, so no run is ever longer than `max_run - 1` rows --
        // a bounded memory, and so a bounded vector, of at most `max_run`
        // lengths. The vector runs newest-first, its lengths rising, so the
        // runs at the cap are its tail, and the oldest is the last. Keeping
        // the heaviest's statistics is what keeps the dominant run's memory
        // whichever run that is. On quiet data it is the oldest: folding
        // into the youngest run at the cap, which pruning can leave far
        // younger than the run carrying the mass, reset that run's memory
        // and moved `run_mode` down by up to `max_run - 1`, each reset read
        // as a change (task 243). After a break it is the post-break run:
        // folding into the oldest put that run's mass on the row-0 run's
        // statistics, the old regime's mean, and the capped slot predicted
        // badly until its memory adapted, so the mass went to the shorter
        // runs and the reported start jumped again, a second alarm about
        // `max_run` rows after the break (task 243b). Among equals the
        // oldest, so a flat tail folds as it always did. The capped slot is
        // then an exponentially weighted run: a Welford step and a scale by
        // `cap / (cap + 1)` each row, a decay at that `lam`, whose mean
        // steps by `1 / max_run` and whose half-life is `ln 2 / ln((cap +
        // 1) / cap)`, about `0.69 (max_run - 1)` rows. The cap is on the
        // length, not the position, since a slot's position stops being its
        // length after a prune. `max_run_folds_the_tail`,
        // `max_run_caps_the_length_after_a_prune`,
        // `heavy_pruning_keeps_the_dominant_runs_memory_at_the_cap`,
        // `after_a_break_the_fold_keeps_the_run_carrying_the_mass` and
        // `the_capped_run_is_an_exponentially_weighted_run` pin each part.
        let cap = (self.cfg.max_run - 1) as f64;
        let kept = self.runs.iter().take_while(|r| r.len < cap).count();
        if kept < self.runs.len() {
            let tail = log_sum_exp(&self.logjoint[kept..]);
            // The heaviest, and the last of the tail among equals: scanned
            // from the end, so only a strictly heavier run moves it.
            let mut heaviest = self.runs.len() - 1;
            for i in (kept..heaviest).rev() {
                if self.logjoint[i] > self.logjoint[heaviest] {
                    heaviest = i;
                }
            }
            self.runs.swap(kept, heaviest);
            self.runs.truncate(kept + 1);
            self.runs[kept].shrink_to(cap);
            self.logjoint.truncate(kept + 1);
            self.logjoint[kept] = tail;
        }
    }
}

impl crate::OnlineModel for Bocpd {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): learned as a row of weight 0 -- on the clock its
        // step's chance applies, and nothing is counted, neither in `n_eff`
        // nor as a failure -- and a hazard that is not usable is absent
        // (the module docs; docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        if self.warming() {
            return self.hold(x, y, d_clock, weight);
        }
        self.learn(x, y, d_clock, weight)
    }

    /// Never told the row's hazard column, so a per-row hazard read from
    /// the row falls back to the configured one.
    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        self.predict_with(x, &[], d_clock)
    }

    /// The hazard rides in `y[0]` under `hazard_from_row`, so the answer
    /// depends on it exactly as the step's does (C1), absent where it is not
    /// usable; on the clock it depends on `d_clock` as the step's does.
    /// Nothing while the warm-up holds the first rows.
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict_with(self, x, y, d_clock) {
            return refused;
        }
        if self.warming() {
            return crate::Step {
                pred: vec![f64::NAN; Self::n_outputs_for(self.cfg.n_features)],
                n_eff: self.n_eff,
                extra: None,
            };
        }
        let (moved, next) = self.before_row(y, d_clock);
        let joint = moved.as_deref().unwrap_or(self.logjoint.as_slice());
        crate::Step {
            pred: self.read(x, joint, next, 1.0, None).0,
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
                crate::model::check_cfg("bocpd", m.cfg.validate())?;
                // One log-joint per run, at least one run, and every run's
                // moments at the emission's width (review 2026-09-18, B3);
                // the warm-up's rows and the prior they set at the cfg's
                // width, and rows held only while no prior is set.
                let d = m.cfg.n_features;
                let m2 = if m.cfg.emission == BocpdEmission::Gaussian {
                    d * d
                } else {
                    d
                };
                let prior_ok = m
                    .data_prior
                    .as_ref()
                    .is_none_or(|p| p.mean.len() == d && p.scale.len() == d * d);
                if m.runs.is_empty()
                    || m.runs.len() != m.logjoint.len()
                    || m.runs
                        .iter()
                        .any(|r| r.mean.len() != d || r.m2.len() != m2 || r.mean_lo.len() != d)
                    || !prior_ok
                    || m.warm.iter().any(|r| r.x.len() != d)
                    || (m.data_prior.is_some() && !m.warm.is_empty())
                    || m.warm.len() >= m.cfg.warm_rows_or_default()
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

impl Bocpd {
    /// A row read against the prior as it stands and learned (Algorithm 1;
    /// the module docs): every row once the prior is set, and the warm-up's
    /// rows when it is.
    fn learn(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        // A row of no weight is scored as a row of the mean weight would be.
        let (rel, w_mean, w_rows) = if weight > 0.0 {
            self.relative(weight)
        } else {
            (1.0, self.w_mean, self.w_rows)
        };
        let (moved, next) = self.before_row(y, d_clock);
        let joint = moved.as_deref().unwrap_or(self.logjoint.as_slice());
        // The gamma cache, out of `self` while `read` borrows it.
        let mut gammas = std::mem::take(&mut self.gammas);
        let (pred, extra) = self.read(x, joint, next, rel, Some(&mut gammas));
        self.gammas = gammas;
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
        let Some((new, weights, z)) = extra else {
            // The predictive could not be evaluated: the row reports nulls
            // and no run sees it. A usable row that fails -- a run's
            // covariance no factorization takes, a usable hazard at or below
            // 1 -- is counted as a failure, so a run of them is visible in
            // `diagnostics` rather than silent (B1). The time still passed:
            // on the clock its chance of a break applies, as on a row of no
            // weight -- which is all the row is: `n_eff` and the weight
            // mean, which are those of the rows learned from, do not move.
            // They moved before this refusal, so `n_eff` counted rows no
            // run saw and the next rows' `w/w̄` read a mean holding one;
            // `hmm` ages a failed row as a row of no weight too (review
            // round 4, CE10).
            self.solve_failures += 1;
            if let Some(moved) = moved {
                self.logjoint = moved;
            }
            return out;
        };
        self.n_eff += weight;
        (self.w_mean, self.w_rows) = (w_mean, w_rows);
        let full = self.full();
        // Algorithm 1's line 6, and the indexing is the whole of it:
        // `ν⁽ʳ⁺¹⁾_{t+1} = ν⁽ʳ⁾_t + u(xₜ)` and `ν⁽⁰⁾_{t+1} = ν_prior`. Slot
        // `j` holds exactly the `j` rows that the hypothesis `rₜ = j` says
        // precede the next row in its run -- so the new slot at the front
        // holds **nothing**, and its predictive is the prior's. (Letting it
        // take this row too, which is the easy mistake, puts one row of the
        // old regime inside every "brand new run" and moves the estimated
        // start of every run back by one.)
        // Each run takes the row where it stands, and the empty run goes in
        // front: the same steps on the same statistics as on a copy of each
        // run, without the copies (task 250).
        self.runs.insert(0, Run::new(self.cfg.n_features, full));
        for (run, &w) in self.runs[1..].iter_mut().zip(&weights) {
            run.add(x, w, full);
        }
        self.logjoint = new;
        self.prune(z);
        out
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
        // The pair this build computes, to the bit (task 252 (b)): the rows
        // are dyadic and the steps are IEEE sums, products and quotients,
        // with no libm, so every platform computes the same pair. The
        // tolerance below is against exact arithmetic, an oracle computed
        // another way, which the pair meets to two rounding steps.
        assert_eq!(
            (
                run.mean[0].to_bits(),
                run.mean_lo[0].to_bits(),
                run.m2[0].to_bits()
            ),
            (
                4_786_511_204_640_099_358,
                13_762_857_705_633_405_824,
                4_628_000_683_156_221_124
            ),
            "{} {} {}",
            run.mean[0],
            run.mean_lo[0],
            run.m2[0]
        );
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

    /// A prior given whole, so no warm-up: the longhand tests read Algorithm
    /// 1 from the first row.
    fn cfg(d: usize) -> BocpdCfg {
        BocpdCfg {
            n_features: d,
            hazard: 100.0,
            hazard_from_row: false,
            emission: BocpdEmission::Diag,
            prior_mean: Some(vec![0.0; d]),
            prior_kappa: 1.0,
            prior_nu: Some(2.0),
            prior_scale: Some(vec![1.0]),
            robust_beta: 0.0,
            prune_below: 0.0,
            max_run: 100_000,
            min_weight: 0.0,
            warm_rows: None,
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
            // Tolerance (task 252 (b)): an approximation (Lanczos, ~1e-13)
            // through libm (`ln`), against tabled values.
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
            // Tolerances here (task 252 (b)): an oracle computed another way
            // -- Algorithm 1 in probabilities from raw sums, where the model
            // works in logs from centred ones -- through libm (`ln`, `exp`).
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
        // (The approximation's cost, not a rounding tolerance.)
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
        // Tolerance (task 252 (b)): libm (`exp`) and the rounding of a
        // normalized sum, which is 1 only in exact arithmetic.
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[19] > 0.5, "the tail holds the mass of every longer run");
        // And the memory is bounded with the vector: after 200 rows no run
        // is longer than `max_run - 1` of them.
        let longest = m.runs.iter().map(|r| r.len).fold(0.0, f64::max);
        assert_eq!(longest, 19.0, "the fold caps the length");
        // The tail is the run that began on the first row, scaled back to
        // 19 rows' weight each time it reaches 20: a weighted Welford step,
        // then `n` and the scatter times `19 / 20`, the mean left alone.
        let mut n = Normals::new(11);
        let (mut len, mut w, mut mean, mut m2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for _ in 0..200 {
            let x = n.normal();
            let w_new = w + 1.0;
            m2 += w / w_new * (x - mean) * (x - mean);
            mean += (x - mean) / w_new;
            (w, len) = (w_new, len + 1.0);
            if len > 19.0 {
                let ratio = 19.0 / len;
                (w, m2, len) = (w * ratio, m2 * ratio, 19.0);
            }
        }
        let tail = m.runs.last().unwrap();
        assert_eq!(tail.len, len);
        // Exact (task 252 (b)): the weight does not depend on the rows --
        // unit weights, a sum and a scale by `19 / 20` -- so the longhand's
        // recursion is the model's, to the bit, with no libm in it.
        assert_eq!(tail.n.to_bits(), w.to_bits(), "n {} vs {w}", tail.n);
        // Tolerances: an oracle computed another way. The model's mean is
        // a compensated pair stepped by `b·d` and its scatter is fed the
        // pair's deviation; the longhand keeps a plain mean and divides.
        // The rows come through libm (`ln`, `cos`) as well.
        assert!(
            (tail.mean[0] - mean).abs() < 1e-12,
            "mean {} vs {mean}",
            tail.mean[0]
        );
        assert!(
            (tail.m2[0] - m2).abs() < 1e-9,
            "scatter {} vs {m2}",
            tail.m2[0]
        );
        assert_eq!(w, 19.0, "19 rows' weight: {w}");
    }

    /// Under heavy pruning the run that dominates a stationary stream keeps
    /// its memory at the cap. Pruning leaves a gap below `max_run`, so the
    /// youngest run at the cap can be a few rows long. Folding the capped
    /// run into that one, as the fold once did, reset the dominant run's
    /// memory on these 600 quiet rows: `run_mode` sat at the cap on 34 of
    /// them and fell by up to 19 at once, and each reset read as evidence of
    /// a change, `p_change` summing to 34.1. Scaled in place, the capped run
    /// is the mode on 581 rows, `run_mode` never falls, and `p_change` sums
    /// to 5.9, about `1 / hazard` a row.
    #[test]
    fn heavy_pruning_keeps_the_dominant_runs_memory_at_the_cap() {
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 20,
            prune_below: 1e-2,
            hazard: 200.0,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(5);
        let (mut p_sum, mut prev, mut worst_drop, mut capped) = (0.0, 0.0, 0.0f64, 0);
        for i in 0..600 {
            let x = n.normal();
            // On every row the capped slot is the oldest run of the tail,
            // the dominant run itself, scaled to the cap. At this pruning no
            // younger run survives to the cap on these rows, so the tail is
            // that run alone; the leg below is where the choice arises.
            let probe = before_fold(&m, &[x]);
            let tail: Vec<usize> = (0..probe.runs.len())
                .filter(|&j| probe.runs[j].len >= 19.0)
                .collect();
            let out = m.step(&[x], &[], 1.0, 1.0);
            if let (Some(&first), Some(&oldest)) = (tail.first(), tail.last()) {
                let mut want = probe.runs[oldest].clone();
                want.shrink_to(19.0);
                assert_eq!(m.runs[first], want, "row {i}");
            }
            if i > 0 {
                p_sum += out.pred[0];
                worst_drop = worst_drop.max(prev - out.pred[1]);
                capped += usize::from(out.pred[1] == 19.0);
                prev = out.pred[1];
            }
        }
        assert!(capped > 500, "the cap was the mode on {capped} rows");
        assert!(worst_drop <= 5.0, "run_mode fell by {worst_drop}");
        assert!(p_sum < 12.0, "p_change summed to {p_sum}");
        // The fold keeps the tail run carrying the most mass. At
        // `prune_below = 1e-3` a younger run reaches the cap beside the
        // dominant one on 203 of these rows, and on every one the dominant
        // run, the oldest, is the heaviest: what the fold keeps on quiet
        // data is what it kept when it took the oldest by rule.
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 20,
            prune_below: 1e-3,
            hazard: 200.0,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(5);
        let mut folds = 0;
        for i in 0..600 {
            let x = n.normal();
            let probe = before_fold(&m, &[x]);
            let tail: Vec<usize> = (0..probe.runs.len())
                .filter(|&j| probe.runs[j].len >= 19.0)
                .collect();
            m.step(&[x], &[], 1.0, 1.0);
            if tail.len() > 1 {
                folds += 1;
                let oldest = *tail.last().unwrap();
                let heaviest = tail.iter().copied().fold(tail[0], |a, j| {
                    if probe.logjoint[j] >= probe.logjoint[a] {
                        j
                    } else {
                        a
                    }
                });
                assert_eq!(heaviest, oldest, "row {i}: the oldest run carries the mass");
                let mut want = probe.runs[oldest].clone();
                want.shrink_to(19.0);
                assert_eq!(m.runs[tail[0]], want, "row {i}");
            }
        }
        assert!(folds > 150, "a younger run reached the cap on {folds} rows");
    }

    /// The state the fold reads: a clone that folds nothing (`max_run` past
    /// any length) stepped on the same row holds the grown and pruned vector
    /// as `prune` sees it before the fold, since the fold is the last thing
    /// a row does and the only reader of `max_run`.
    fn before_fold(m: &Bocpd, x: &[f64]) -> Bocpd {
        let mut probe = m.clone();
        probe.cfg.max_run = usize::MAX;
        probe.step(x, &[], 1.0, 1.0);
        probe
    }

    /// The rows where the row the run began on, `t − run_mode`, jumps by more
    /// than `jump`: the alarm REGIMES §5 reads `bocpd` by.
    fn start_alarms(starts: &[f64], jump: f64) -> Vec<usize> {
        (1..starts.len())
            .filter(|&t| starts[t] - starts[t - 1] > jump)
            .collect()
    }

    /// After a break the run carrying the mass is the post-break run, and the
    /// fold keeps it: at the row it reaches the cap the capped slot is that
    /// run to the bit (the uncapped model's run of the same length), under
    /// the tail's summed mass. Folding into the oldest run at the cap kept
    /// the row-0 run instead -- the old regime's mean at the cap's weight --
    /// under the post-break mass, so the capped slot predicted badly until
    /// its memory adapted, the mass went to the shorter runs, and the start
    /// `t − run_mode` reports jumped again: a second alarm about `max_run`
    /// rows after the break. On a 3-sigma mean shift at row 300 of 1,000 at
    /// `max_run = 200`, 38 of the first 40 seeds raised 3 to 14 alarms after
    /// row 310 where the uncapped posterior raised none or one; on these
    /// four it raised none, and the capped model 3, 2, 3 and 7 (task 243b,
    /// measured at `e70dd1b`).
    #[test]
    fn after_a_break_the_fold_keeps_the_run_carrying_the_mass() {
        let (max_run, brk, rows) = (200usize, 300usize, 1000usize);
        let cap = (max_run - 1) as f64;
        // The post-break run holds rows `brk..=t`, so it reaches the cap on
        // row `brk + cap - 1`.
        let fold_row = brk + max_run - 2;
        let base = BocpdCfg {
            max_run,
            hazard: 250.0,
            ..cfg(1)
        };
        for seed in [0u64, 3, 5, 9] {
            let mut capped = Bocpd::new(base.clone()).unwrap();
            let mut whole = Bocpd::new(BocpdCfg {
                max_run: 100_000,
                ..base.clone()
            })
            .unwrap();
            let mut n = Normals::new(seed);
            let (mut starts_c, mut starts_w) = (vec![], vec![]);
            for t in 0..rows {
                let x = n.normal() + if t >= brk { 3.0 } else { 0.0 };
                if t == fold_row {
                    let probe = before_fold(&capped, &[x]);
                    // With nothing pruned the tail is the run reaching the cap
                    // and the capped slot, grown past it.
                    let lens: Vec<f64> = probe.runs.iter().map(|r| r.len).collect();
                    assert_eq!(&lens[max_run - 1..], &[cap, cap + 1.0], "{seed}");
                    let (post, old) = (max_run - 1, max_run);
                    assert!(
                        probe.logjoint[post] > probe.logjoint[old],
                        "{seed}: the post-break run carries the mass: {} vs {}",
                        probe.logjoint[post],
                        probe.logjoint[old]
                    );
                    // (By now the capped slot is itself a run born a few rows
                    // before the break, each row's fold having kept the
                    // heaviest; the post-break run is heavier still.)
                    assert!(
                        probe.runs[post].mean[0] > 2.5,
                        "{seed}: the post-break regime's mean: {}",
                        probe.runs[post].mean[0]
                    );
                    let a = capped.step(&[x], &[], 1.0, 1.0);
                    let b = whole.step(&[x], &[], 1.0, 1.0);
                    starts_c.push(t as f64 - a.pred[1]);
                    starts_w.push(t as f64 - b.pred[1]);
                    assert_eq!(capped.runs.len(), max_run, "{seed}");
                    // The kept run is the post-break run to the bit: scaling
                    // a run of `cap` rows to `cap` multiplies by 1.
                    assert_eq!(capped.runs[post], probe.runs[post], "{seed}");
                    assert_eq!(capped.runs[post], whole.runs[post], "{seed}");
                    let tail = log_sum_exp(&probe.logjoint[post..]);
                    assert_eq!(capped.logjoint[post], tail, "{seed}");
                    assert_eq!(&capped.logjoint[..post], &probe.logjoint[..post], "{seed}");
                } else {
                    let a = capped.step(&[x], &[], 1.0, 1.0);
                    let b = whole.step(&[x], &[], 1.0, 1.0);
                    starts_c.push(t as f64 - a.pred[1]);
                    starts_w.push(t as f64 - b.pred[1]);
                }
            }
            let (ac, aw) = (start_alarms(&starts_c, 10.0), start_alarms(&starts_w, 10.0));
            let late = |a: &[usize]| a.iter().filter(|&&t| t > brk + 10).count();
            assert!(
                ac.iter().any(|&t| (brk..=brk + 10).contains(&t)),
                "{seed}: the break is found: {ac:?}"
            );
            assert_eq!(
                late(&aw),
                0,
                "{seed}: the uncapped posterior is quiet after it: {aw:?}"
            );
            assert_eq!(late(&ac), 0, "{seed}: and so is the capped one: {ac:?}");
        }
    }

    /// A run at the cap is an exponentially weighted run. Each row it takes
    /// the row at a Welford step, `cap` rows' weight becoming `cap + 1`, and
    /// is scaled back to `cap`: a decay at `lam = cap / (cap + 1)`, whose
    /// weight stays at `cap`, whose mean steps by `1 / max_run` of the row's
    /// deviation, and whose half-life is `ln 2 / ln((cap + 1) / cap)`, 13.5
    /// rows at `max_run = 20` and about `0.69 (max_run − 1)` in general.
    #[test]
    fn the_capped_run_is_an_exponentially_weighted_run() {
        let max_run = 20usize;
        let cap = (max_run - 1) as f64;
        let mut m = Bocpd::new(BocpdCfg { max_run, ..cfg(1) }).unwrap();
        let mut n = Normals::new(3);
        for _ in 0..40 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
        }
        for _ in 0..100 {
            let before = m.runs.last().unwrap().clone();
            assert_eq!(before.len, cap);
            // Exact (task 252 (b)): unit weights, `cap + 1` scaled by
            // `cap / (cap + 1)`, no libm and nothing from the rows.
            assert_eq!(before.n, cap, "the weight stays at the cap");
            let x = n.normal();
            m.step(&[x], &[], 1.0, 1.0);
            let after = m.runs.last().unwrap();
            assert_eq!(after.len, cap);
            let step = after.mean[0] - before.mean[0];
            let want = (x - before.mean[0]) / max_run as f64;
            // Tolerances (task 252 (b)): a closed form computed another way.
            // The model steps a compensated pair by `b·d`, `d` the deviation
            // from the pair, so its high part moves by that less what the
            // low part takes; the closed form reads the high part alone.
            assert!(
                (step - want).abs() < 1e-14,
                "the mean steps by 1/max_run: {step} vs {want}"
            );
            // The scatter takes the deviation at `cap / (cap + 1)` and is
            // scaled by the same.
            let lam = cap / (cap + 1.0);
            let dev = x - before.mean[0];
            let m2 = (before.m2[0] + lam * dev * dev) * lam;
            assert!((after.m2[0] - m2).abs() < 1e-10, "{} vs {m2}", after.m2[0]);
        }
        let half_life = 2f64.ln() / ((cap + 1.0) / cap).ln();
        // The docs' rounded figure, through libm (`ln`).
        assert!((half_life - 13.51).abs() < 0.01, "{half_life}");
    }

    /// `max_run` caps a run's length, not its position. Once pruning has
    /// dropped runs from the middle of the vector the position stops being
    /// the length, and a cap on the position let the oldest run grow with
    /// the stream: a `run_mode` of 599 after 600 stationary rows at
    /// `max_run = 20` and `prune_below = 1e-2`. Here no run passes 19 rows
    /// on any row, the cap is reached, pruning did drop middle runs (a run
    /// sits past its position), and the posterior still sums to 1. Without
    /// pruning a run's position is its length on every row, so the two caps
    /// fold the same runs.
    #[test]
    fn max_run_caps_the_length_after_a_prune() {
        for prune_below in [1e-3, 1e-2] {
            let mut m = Bocpd::new(BocpdCfg {
                max_run: 20,
                prune_below,
                hazard: 200.0,
                ..cfg(1)
            })
            .unwrap();
            let mut n = Normals::new(5);
            let (mut longest, mut moved) = (0.0f64, false);
            for i in 0..600 {
                let out = m.step(&[n.normal()], &[], 1.0, 1.0);
                if i > 0 {
                    assert!(out.pred[1] <= 19.0, "row {i}: run_mode {}", out.pred[1]);
                    assert!(out.pred[2] <= 19.0, "row {i}: run_mean {}", out.pred[2]);
                }
                assert!(m.runs.iter().all(|r| r.len <= 19.0), "row {i}");
                assert!(m.runs.len() <= 20, "row {i}");
                longest = longest.max(m.runs.last().unwrap().len);
                moved |= m.runs.iter().enumerate().any(|(j, r)| r.len > j as f64);
                let p = m.run_posterior();
                // Tolerance (task 252 (b)): libm (`exp`) and the rounding
                // of a normalized sum.
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "row {i}");
            }
            assert_eq!(longest, 19.0, "{prune_below}: the cap was reached");
            assert!(
                moved,
                "{prune_below}: pruning dropped a run from the middle"
            );
        }
        // Without pruning a run's position is its length, so the length cap
        // folds exactly where the position cap did.
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 20,
            ..cfg(1)
        })
        .unwrap();
        let mut n = Normals::new(5);
        for _ in 0..100 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
            let lens: Vec<f64> = m.runs.iter().map(|r| r.len).collect();
            let at: Vec<f64> = (0..m.runs.len()).map(|j| j as f64).collect();
            assert_eq!(lens, at);
        }
        assert_eq!(m.runs.len(), 20);
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
    #[ignore = "extended: a second or more: a long stream (2.8 s)"]
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
        // Tolerances (task 252 (b)): scale invariance holds in exact
        // arithmetic only. `0.5 + 0.1·k` times 100 is not exact, nor are
        // the ratios `w / n` built from it, and the outputs pass through
        // libm (`ln`, `exp`); the code promises no bit-level invariance.
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
            // Tolerance below (task 252 (b)): an oracle computed another way
            // (probabilities, two-pass scatter) through libm, and the
            // data's own resolution at the level, `ulp(1e8)` a row.
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
            // Tolerances (task 252 (b)): the rows at `1e8` are rounded to
            // `ulp(1e8)`, so the two streams are not the same numbers.
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
            "mean_precision_prior must be finite and > 0",
        );
        bad(
            BocpdCfg {
                prior_nu: Some(0.5),
                ..cfg(1)
            },
            "degrees_of_freedom_prior must be finite and >",
        );
        bad(
            BocpdCfg {
                robust_beta: 1.0,
                ..cfg(1)
            },
            "robust_beta tempers the per-feature predictive",
        );
        bad(
            BocpdCfg {
                emission: BocpdEmission::Robust,
                ..cfg(1)
            },
            "the tempered predictive needs robust_beta > 0",
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
            "mean_prior must be 1 values",
        );
        bad(
            BocpdCfg {
                prior_scale: Some(vec![1.0, 2.0]),
                ..cfg(1)
            },
            "covariance_prior is a scalar or a 1x1 matrix",
        );
        bad(
            BocpdCfg {
                prior_scale: Some(vec![0.0]),
                ..cfg(1)
            },
            "scalar covariance_prior",
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
            "mean_prior must be finite",
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
            // Tolerances (task 252 (b)): an approximation (Lanczos) and a
            // recurrence of logs, both through libm, against tabled values.
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
            // Tolerances (task 252 (b)): an oracle computed another way --
            // two-pass scatter, faer's eigenvalues and LU, the gamma
            // recurrence, probabilities -- through libm.
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
            // Tolerances (task 252 (b)): an oracle computed another way (the
            // tempered message written out from each run's rows) through
            // libm.
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
    /// grows) is a usable value and no hazard: refused at the row, it
    /// reports nulls, is counted, and moves nothing. (The plumbing refuses it
    /// before it arrives; the model holds the line too.) A hazard that is not
    /// usable -- infinite (`H = 0`: no run ever starts), not a number, or
    /// past the input bound -- is absent, as a target that is not usable is
    /// (`OnlineModel`, task 183): the row reads the configured hazard, as one
    /// with no hazard in the slot does, where infinity and NaN refused it and
    /// counted it, and `2e100` was taken as a hazard.
    #[test]
    fn a_row_hazard_of_one_is_refused_and_one_not_usable_is_absent() {
        let warm = || {
            let mut m = Bocpd::new(BocpdCfg {
                hazard_from_row: true,
                ..cfg(1)
            })
            .unwrap();
            let mut n = Normals::new(5);
            for _ in 0..20 {
                m.step(&[n.normal()], &[None], 1.0, 1.0);
            }
            m
        };
        let mut m = warm();
        let before = m.clone();
        let shown = m.predict_with(&[0.3], &[Some(1.0)], 1.0);
        assert!(shown.pred.iter().all(|v| v.is_nan()), "{shown:?}");
        let out = m.step(&[0.3], &[Some(1.0)], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| v.is_nan()), "{out:?}");
        assert_eq!(m.runs, before.runs);
        assert_eq!(m.logjoint, before.logjoint);
        assert_eq!(m.solve_failures, before.solve_failures + 1);
        // And a usable row hazard is used.
        let fine = m.step(&[0.3], &[Some(50.0)], 1.0, 1.0);
        assert!(fine.pred.iter().all(|v| v.is_finite()), "{fine:?}");

        for bad in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            2.0 * crate::INPUT_BOUND,
        ] {
            let (mut m, mut absent) = (warm(), warm());
            let shown = m.predict_with(&[0.3], &[Some(bad)], 1.0);
            assert_eq!(shown, absent.predict_with(&[0.3], &[None], 1.0), "{bad}");
            let out = m.step(&[0.3], &[Some(bad)], 1.0, 1.0);
            assert!(out.pred.iter().all(|v| v.is_finite()), "{bad}: {out:?}");
            assert_eq!(out, absent.step(&[0.3], &[None], 1.0, 1.0), "{bad}");
            assert_eq!(m, absent, "{bad}");
            assert_eq!(m.solve_failures, 0, "{bad}");
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

    /// The fold at `max_run` keeps the mass: the folded slot takes the
    /// summed probability of every run that reached the cap, and each
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
        let before = m.clone();
        m.cfg.max_run = cut;
        m.prune(None);
        let q = m.run_posterior();
        assert_eq!(q.len(), cut);
        // To the bit (task 252 (b)): the fold leaves every shorter run's
        // log joint where it was and gives the folded slot the log-sum-exp
        // of the tail's, the model's own function of the same doubles.
        assert_eq!(m.logjoint[..cut - 1], before.logjoint[..cut - 1]);
        assert_eq!(
            m.logjoint[cut - 1].to_bits(),
            log_sum_exp(&before.logjoint[cut - 1..]).to_bits()
        );
        // Tolerances on the probabilities: libm (`exp`, `ln`) and a
        // normalizer summed over another grouping -- the tail now one term
        // -- which nothing promises to round the same.
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
        // The folded slot keeps the oldest run's statistics, scaled to
        // `cut - 1` rows as a decay would scale them: `n` and the scatter
        // by `9 / 30`, the mean untouched.
        let ratio = (cut - 1) as f64 / 30.0;
        let (old, kept) = (&before.runs[30], &m.runs[cut - 1]);
        assert_eq!(kept.len, (cut - 1) as f64);
        assert_eq!(kept.n, old.n * ratio);
        assert_eq!(kept.m2, vec![old.m2[0] * ratio]);
        assert_eq!((&kept.mean, &kept.mean_lo), (&old.mean, &old.mean_lo));
        // Every shorter run is untouched.
        assert_eq!(m.runs[..cut - 1], before.runs[..cut - 1]);
    }

    /// Among tail runs of equal mass the fold keeps the oldest, the last of
    /// the tail, so a flat tail folds as it did when the oldest was taken by
    /// rule: the scan from the end moves only on a strictly heavier run. A
    /// strictly heavier younger run is kept wherever it sits.
    #[test]
    fn the_fold_keeps_the_oldest_of_equally_heavy_runs() {
        let mut m = Bocpd::new(cfg(1)).unwrap();
        let mut n = Normals::new(37);
        for _ in 0..30 {
            m.step(&[n.normal()], &[], 1.0, 1.0);
        }
        let cut = 10;
        let top = m.logjoint[cut - 1..]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        for l in &mut m.logjoint[cut - 1..] {
            *l = top;
        }
        let before = m.clone();
        m.cfg.max_run = cut;
        m.prune(None);
        let mut want = before.runs[30].clone();
        want.shrink_to((cut - 1) as f64);
        assert_eq!(m.runs[cut - 1], want, "the oldest of the equals");
        // The tail is runs `cut - 1 ..= 30`, each at `top`.
        let tail = (32 - cut) as f64;
        // Exact (task 252 (b)): `exp(0)` is 1 by IEEE 754 and C Annex F, so
        // the sum is the integer 22 and both sides take `ln` of the same
        // double on the same platform: the same bits wherever it runs.
        assert_eq!(
            m.logjoint[cut - 1].to_bits(),
            (top + tail.ln()).to_bits(),
            "{} against {}",
            m.logjoint[cut - 1],
            top + tail.ln()
        );
        // One run in the middle of the tail strictly heavier than the rest:
        // it is the one kept, scaled from its own length.
        let mut m = before.clone();
        m.logjoint[20] = top + 1.0;
        m.cfg.max_run = cut;
        m.prune(None);
        let mut want = before.runs[20].clone();
        want.shrink_to((cut - 1) as f64);
        assert_eq!(m.runs[cut - 1], want, "the strictly heavier run");
        assert_eq!(m.runs.len(), cut);
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
    /// there were some, so a round trip keeps the count. A row with a hazard
    /// of 1, a usable value and no hazard, is one. (A row whose feature is
    /// not a number was one; it is a row of weight 0 now, counted nowhere,
    /// task 183.)
    #[test]
    fn solve_failures_is_written_only_when_there_were_some() {
        let mut m = Bocpd::new(BocpdCfg {
            hazard_from_row: true,
            ..cfg(1)
        })
        .unwrap();
        m.step(&[0.1], &[None], 1.0, 1.0);
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("solve_failures").is_none(), "{v}");
        m.step(&[f64::NAN], &[None], 1.0, 1.0);
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("solve_failures").is_none(), "{v}");
        let out = m.step(&[0.2], &[Some(1.0)], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.solve_failures, 1);
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["solve_failures"], 1, "{v}");
        let bytes = rmp_serde::to_vec(&m).unwrap();
        let back: Bocpd = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, m);
    }

    /// A row whose feature is not usable is a row of weight 0 by the rule
    /// every model keeps (`OnlineModel`, task 183), under the full emission
    /// as under the diagonal one, on a per-row hazard and on the clock: it
    /// reports nulls, no run sees it, its step's chance of a break applies
    /// where there is one (on the clock), and it is counted nowhere, neither
    /// in `n_eff` nor as a failure, where task 179 counted it in both (B1).
    /// The whole state is the same row's at weight 0, its feature finite,
    /// and the rows after it read as before. The full emission's quadratic
    /// form was clamped at 0, and `f64::max` took a NaN to that 0, so the
    /// row read as sitting on every run's mean and was learned. Every run's
    /// statistics then held a NaN, and every learned row after it was a
    /// failure: 88 of the 99 that followed one, in a stream task 179 ran. The
    /// bank never hands a NaN feature over; the Rust API does.
    #[test]
    fn a_row_that_is_not_a_number_moves_nothing_under_either_emission() {
        for (emission, on_the_clock) in [
            (BocpdEmission::Gaussian, false),
            (BocpdEmission::Diag, false),
            (BocpdEmission::Gaussian, true),
            (BocpdEmission::Diag, true),
        ] {
            for bad in [f64::NAN, f64::INFINITY, -2.0 * crate::INPUT_BOUND] {
                let case = format!("{emission:?}, on the clock {on_the_clock}, {bad}");
                let mut m = Bocpd::new(BocpdCfg {
                    emission,
                    prior_nu: Some(5.0),
                    hazard: if on_the_clock { 30.0 } else { 50.0 },
                    hazard_on_clock: on_the_clock,
                    ..cfg(3)
                })
                .unwrap();
                let mut n = Normals::new(83);
                for t in 0..30 {
                    let d = if t == 0 { 0.0 } else { 1.0 };
                    m.step(&[n.normal(), n.normal(), n.normal()], &[], d, 1.0);
                }
                let before = m.clone();
                let mut no_weight = m.clone();
                no_weight.step(&[0.4, 0.1, 0.2], &[], 1.0, 0.0);
                let out = m.step(&[bad, 0.1, 0.2], &[], 1.0, 1.0);
                assert!(out.pred.iter().all(|v| v.is_nan()), "{case}: {out:?}");
                assert_eq!(out.n_eff, before.n_eff, "{case}");
                assert_eq!(m, no_weight, "{case}");
                assert_eq!(m.solve_failures, 0, "{case}");
                assert_eq!(m.n_eff, before.n_eff, "{case}");
                assert_eq!(m.runs, before.runs, "{case}");
                if on_the_clock {
                    assert_ne!(m.logjoint, before.logjoint, "{case}: the time did not pass");
                } else {
                    assert_eq!(m.logjoint, before.logjoint, "{case}");
                }
                for t in 0..20 {
                    let out = m.step(&[n.normal(), n.normal(), n.normal()], &[], 1.0, 1.0);
                    assert!(
                        out.pred.iter().all(|v| v.is_finite()),
                        "{case} row {t} after: {out:?}"
                    );
                }
                assert_eq!(m.solve_failures, 0, "{case}");
            }
        }
    }

    /// A usable row whose predictive cannot be evaluated is no row a run
    /// learned from, so it moves neither `n_eff` nor the weight mean a
    /// later row's `w/w̄` reads: it ages as a row of weight 0 does, and is
    /// counted, as `hmm` treats one. Both moved before the refusal, so
    /// `n_eff` counted rows no run saw (review round 4, CE10). Under a
    /// per-row hazard the row brings a hazard of 1, which is no hazard; on
    /// the clock its step is one the clock cannot read. A row at `1e300`
    /// was the case first; past the input bound, it is now a row of weight 0
    /// that counts nothing (task 183), and no usable row gives every run a
    /// density of 0.
    #[test]
    fn a_finite_row_whose_predictive_fails_moves_no_weight() {
        for emission in [BocpdEmission::Gaussian, BocpdEmission::Diag] {
            for hazard_on_clock in [false, true] {
                let case = format!("{emission:?}, hazard on the clock {hazard_on_clock}");
                let mut base = Bocpd::new(BocpdCfg {
                    emission,
                    prior_nu: Some(5.0),
                    hazard_on_clock,
                    hazard_from_row: !hazard_on_clock,
                    ..cfg(2)
                })
                .unwrap();
                // The rows around it read the configured hazard.
                let y: &[Option<f64>] = if hazard_on_clock { &[] } else { &[None] };
                let (bad_y, bad_d): (&[Option<f64>], f64) = if hazard_on_clock {
                    (&[], -1.0)
                } else {
                    (&[Some(1.0)], 1.0)
                };
                let mut n = Normals::new(89);
                for _ in 0..30 {
                    let w = 1.0 + n.unit();
                    base.step(&[n.normal(), n.normal()], y, 1.0, w);
                }
                let (mut failed, mut quiet) = (base.clone(), base.clone());
                let out = failed.step(&[0.5, -0.5], bad_y, bad_d, 2.5);
                assert!(out.pred.iter().all(|v| v.is_nan()), "{case}: {out:?}");
                assert_eq!(
                    failed.solve_failures,
                    base.solve_failures + 1,
                    "{case}: the case needs a failure"
                );
                quiet.step(&[0.5, -0.5], bad_y, bad_d, 0.0);
                assert_eq!(failed.n_eff.to_bits(), base.n_eff.to_bits(), "{case}");
                assert_eq!(
                    (failed.w_mean.to_bits(), failed.w_rows.to_bits()),
                    (base.w_mean.to_bits(), base.w_rows.to_bits()),
                    "{case}"
                );
                assert_eq!(failed.runs, quiet.runs, "{case}");
                assert_eq!(failed.logjoint, quiet.logjoint, "{case}");
                for t in 0..10 {
                    let (x, w) = ([n.normal(), n.normal()], 1.0 + n.unit());
                    let a = failed.step(&x, y, 1.0, w).pred;
                    let b = quiet.step(&x, y, 1.0, w).pred;
                    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&a), bits(&b), "{case}, row {t} after");
                }
            }
        }
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
        bad(
            gaussian(3.0),
            "degrees_of_freedom_prior must be finite and > 3 ",
        );
        // The floor buys a finite variance under the Wishart, a finite mean
        // under the gamma, and the message says which (task 159, D2).
        bad(gaussian(3.0), "no finite variance");
        Bocpd::new(gaussian(3.0 + 1e-9)).unwrap();
        bad(
            BocpdCfg {
                prior_nu: Some(1.0),
                ..cfg(1)
            },
            "degrees_of_freedom_prior must be finite and > 1 ",
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
        // `ν₀` absent: 3 for the per-feature emissions, `d + 2` for the
        // inverse-Wishart (task 195, U5).
        assert_eq!(
            BocpdCfg {
                prior_nu: None,
                ..cfg(3)
            }
            .nu0(),
            3.0
        );
        assert_eq!(
            BocpdCfg {
                prior_nu: None,
                emission: BocpdEmission::Gaussian,
                ..cfg(3)
            }
            .nu0(),
            5.0
        );
    }

    /// `Ψ₀`'s short forms are the matrices they name, run for run: a scalar
    /// `s` is `sI`; and `ν₀` absent is `d + 2` under `gaussian` and 3 under
    /// `diag`. Absent, `Ψ₀` is the first rows' covariance, which the warm-up
    /// tests hold (task 195, U4).
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
        assert_eq!(run(gaussian(Some(vec![1.0]))), identity);
        assert_eq!(run(gaussian(Some(vec![2.0]))), twice);
        let diag = |nu: Option<f64>| BocpdCfg {
            prior_nu: nu,
            ..cfg(3)
        };
        assert_ne!(run(diag(Some(3.0))), run(diag(Some(6.0))));
        assert_eq!(run(diag(None)), run(diag(Some(3.0))));
        let wishart = |nu: Option<f64>| BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_nu: nu,
            prior_mean: Some(vec![0.0; 3]),
            ..cfg(3)
        };
        assert_eq!(run(wishart(None)), run(wishart(Some(5.0))));
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
    #[ignore = "extended: a second or more (1.8 s)"]
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
                // Tolerances (task 252 (b)): two forms equal in exact
                // arithmetic -- `1/h` from `exp_m1` against the clock's own
                // `ln` and `exp` of the step -- through libm.
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
    /// joint -- the time passed -- and nothing else. A row with a feature
    /// that is not usable is one of them, whatever its weight (task 183).
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
            // Tolerances (task 252 (b)): libm. Composed steps multiply
            // `exp(-d/τ)` factors where the whole step takes one `exp` of
            // the sum; equal in exact arithmetic only.
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
        // A row with a weight and a feature that is not a number is a row of
        // weight 0: no run sees it, its step applies, and it is counted
        // nowhere, where task 179 counted it in `n_eff` and as a failure.
        let (mut refused, mut quiet) = (base.clone(), base.clone());
        refused.step(&[f64::NAN], &[], 5.0, 1.0);
        quiet.step(&[0.7], &[], 5.0, 0.0);
        assert_eq!(refused, quiet);
        assert_eq!(refused.solve_failures, base.solve_failures);
        assert_eq!(refused.n_eff, base.n_eff);
        assert_eq!(refused.runs, base.runs);
        assert_ne!(refused.logjoint, base.logjoint);
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

    /// The emissions with their default priors: no `prior_mean`, no
    /// `prior_scale`, no `prior_nu`, `robust`'s β at the polars layer's 0.1.
    fn from_the_data(d: usize, emission: BocpdEmission) -> BocpdCfg {
        BocpdCfg {
            emission,
            robust_beta: if emission == BocpdEmission::Robust {
                0.1
            } else {
                0.0
            },
            hazard: 250.0,
            prior_mean: None,
            prior_nu: None,
            prior_scale: None,
            prune_below: 1e-6,
            max_run: 1000,
            ..cfg(d)
        }
    }

    /// The breaks a stream shows: the rows the run-length posterior's mode
    /// restarts on. `run_mode` is the pre-row run length, so a row whose
    /// mode is under 20, after one at 20 or over, dates a break at `t -
    /// mode`, the row the run began on: `robust` forgives the first rows of
    /// a new regime one at a time, so its mode moves a few rows late
    /// (`a_sustained_shift_survives_a_small_beta_and_not_a_large_one`).
    fn breaks_of(m: &mut Bocpd, rows: impl Iterator<Item = Vec<f64>>) -> Vec<usize> {
        let (mut found, mut last) = (Vec::new(), 0.0);
        for (t, x) in rows.enumerate() {
            let mode = m.step(&x, &[], 1.0, 1.0).pred[1];
            if mode.is_finite() {
                if mode < 20.0 && last >= 20.0 {
                    found.push(t - mode as usize);
                }
                last = mode;
            }
        }
        found
    }

    /// Two features whose means step at rows 200 and 400, at `level` with a
    /// spread of `sd`: up 4 spreads and back, a shift `robust` dates (its
    /// trade is that a larger one is forgiven row by row).
    fn stepping(level: f64, sd: f64) -> impl Iterator<Item = Vec<f64>> {
        let mut n = Normals::new(17);
        (0..600).map(move |t| {
            let shift = if (200..400).contains(&t) { 4.0 } else { 0.0 };
            vec![
                level + sd * (shift + n.normal()),
                level + sd * (0.5 * shift + n.normal()),
            ]
        })
    }

    /// With no `prior_mean` and no `prior_scale` the prior is the first
    /// rows' own: their sample mean and covariance (docs/PLAN.md task 195,
    /// U4). So a stream at a level of 1e4 with a spread of 1e-2 shows the
    /// breaks the same stream at level 0 with a spread of 1 shows, on the
    /// same rows, under `gaussian` and `diag`, whose posterior is free of
    /// the data's units once the prior is (3e-11 apart in `p_change`,
    /// measured). The prior was a mean of 0 and the identity in the data's
    /// units: at 1e4 a run that begins on a row has the prior's predictive,
    /// under which no row of the data has any density, so no break was ever
    /// found (review round 4, CE5). `robust` is held at level 0 alone: its
    /// tempered message `π^w` carries the data's units as a factor `c^-w`,
    /// which differs from run to run, so its posterior moves with the scale
    /// whatever the prior -- at a spread of 2e3 it dates breaks that are not
    /// there and misses the one at 200 (a property of the emission, reported
    /// beside task 195 and not changed by it).
    #[test]
    #[ignore = "extended: a grid of levels and scales (1.3 s)"]
    fn the_prior_from_the_first_rows_finds_the_breaks_at_any_level_and_scale() {
        for emission in [
            BocpdEmission::Gaussian,
            BocpdEmission::Diag,
            BocpdEmission::Robust,
        ] {
            let found = |level: f64, sd: f64| {
                let mut m = Bocpd::new(from_the_data(2, emission)).unwrap();
                breaks_of(&mut m, stepping(level, sd))
            };
            let unit = found(0.0, 1.0);
            for edge in [200, 400] {
                assert!(
                    unit.iter().any(|b| b.abs_diff(edge) <= 1),
                    "{emission:?}: the break at {edge} was missed: {unit:?}"
                );
            }
            if emission == BocpdEmission::Robust {
                continue;
            }
            for (level, sd) in [(1e4, 1e-2), (-3e6, 2e3)] {
                assert_eq!(found(level, sd), unit, "{emission:?} at {level}");
            }
        }
    }

    /// A feature that does not move over the first rows has no variance to
    /// give the prior, and a predictive of no spread has no density. It
    /// takes a floor, and the model runs: it finds the break in the other
    /// feature, reports a number on every row after the first ones, and
    /// counts no failure, at level 0 and away from it, and after the
    /// constant starts to move (docs/PLAN.md task 195, U4).
    #[test]
    fn a_feature_constant_over_the_first_rows_does_not_break_the_prior() {
        for emission in [
            BocpdEmission::Gaussian,
            BocpdEmission::Diag,
            BocpdEmission::Robust,
        ] {
            for level in [0.0, 5.0, 1e4] {
                let mut m = Bocpd::new(from_the_data(2, emission)).unwrap();
                let mut n = Normals::new(23);
                let (mut found, mut last) = (Vec::new(), 0.0);
                for t in 0..500usize {
                    let moving = if t < 300 { 0.0 } else { n.normal() };
                    let shift = if t < 200 { 0.0 } else { 6.0 };
                    let x = [level + moving, shift + n.normal()];
                    let p = m.step(&x, &[], 1.0, 1.0).pred;
                    if t >= 10 {
                        assert!(
                            p.iter().all(|v| v.is_finite()),
                            "{emission:?} at {level}, row {t}: {p:?}"
                        );
                    }
                    let mode = p[1];
                    if mode.is_finite() {
                        if mode < 20.0 && last >= 20.0 {
                            found.push(t - mode as usize);
                        }
                        last = mode;
                    }
                }
                assert_eq!(m.solve_failures, 0, "{emission:?} at {level}");
                assert!(
                    found.iter().any(|b| b.abs_diff(200) <= 1),
                    "{emission:?} at {level}: {found:?}"
                );
            }
        }
    }

    /// The warm-up: the first `warm_rows` learned rows (default `d + 2`)
    /// report nothing; their sample mean and covariance -- the diagonal
    /// under `diag` -- are the prior, held to the definition; and they are
    /// then read as rows of a model given that prior from the start: every
    /// row after them reports what that model reports, to the bit. A state
    /// saved inside the warm-up goes on as the model that never stopped
    /// (docs/PLAN.md task 195, U4).
    #[test]
    fn the_first_rows_set_the_prior_and_are_read_as_rows_of_it() {
        for emission in [BocpdEmission::Gaussian, BocpdEmission::Diag] {
            let d = 3;
            let c = from_the_data(d, emission);
            assert_eq!(c.warm_rows_or_default(), 5);
            let mut n = Normals::new(41);
            let rows: Vec<Vec<f64>> = (0..60)
                .map(|t| {
                    let a = n.normal();
                    let lift = if t < 30 { 0.0 } else { 4.0 };
                    vec![
                        10.0 + a,
                        -3.0 + 0.5 * a + n.normal(),
                        lift + 0.1 * n.normal(),
                    ]
                })
                .collect();
            let mut m = Bocpd::new(c.clone()).unwrap();
            let mut outs = Vec::new();
            for (t, x) in rows.iter().enumerate() {
                if t == 3 {
                    let back = Bocpd::restore(&crate::OnlineModel::state(&m)).unwrap();
                    assert_eq!(back, m, "{emission:?}: inside the warm-up");
                }
                let p = m.step(x, &[], 1.0, 1.0).pred;
                assert_eq!(p.iter().all(|v| v.is_nan()), t < 5, "{emission:?}, row {t}");
                outs.push(p);
            }
            // The definition: the sample mean and the unbiased covariance of
            // the five rows.
            let (mean, scale) = m.prior().expect("set");
            let first = &rows[..5];
            for i in 0..d {
                let want = first.iter().map(|x| x[i]).sum::<f64>() / 5.0;
                // Tolerances (task 252 (b)): the definition computed another
                // way (two-pass sums, where the model steps a weighted
                // Welford update), on rows drawn through libm.
                assert!((mean[i] - want).abs() <= 1e-14 * want.abs().max(1.0));
                for j in 0..d {
                    let mj = first.iter().map(|x| x[j]).sum::<f64>() / 5.0;
                    let cov = first
                        .iter()
                        .map(|x| (x[i] - want) * (x[j] - mj))
                        .sum::<f64>()
                        / 4.0;
                    let want = if emission == BocpdEmission::Gaussian || i == j {
                        cov
                    } else {
                        0.0
                    };
                    let got = scale[i * d + j];
                    assert!(
                        (got - want).abs() <= 1e-13 * want.abs().max(1e-3),
                        "{emission:?} [{i}][{j}]: {got} against {want}"
                    );
                }
            }
            // A model given that prior from the start reads every row after
            // the warm-up to the bit.
            let mut given = Bocpd::new(BocpdCfg {
                prior_mean: Some(mean),
                prior_scale: Some(scale),
                ..c
            })
            .unwrap();
            for (t, x) in rows.iter().enumerate() {
                let p = given.step(x, &[], 1.0, 1.0).pred;
                if t >= 5 {
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&p), bits(&outs[t]), "{emission:?}, row {t}");
                }
            }
        }
    }

    /// The prior's weights are the rows' own against their largest, so a
    /// weight's scale moves nothing, and a row of no weight is not one of
    /// the first rows: it learns nothing and is not counted; its step joins
    /// the next row's. A variance of 0 takes the floor `2^-52·max(x̄², 1)`,
    /// and under `gaussian` a covariance that is not positive definite --
    /// two features that are one -- gives its diagonal.
    #[test]
    fn the_warm_up_reads_weights_floors_a_constant_and_drops_collinearity() {
        let c = BocpdCfg {
            warm_rows: Some(3),
            ..from_the_data(2, BocpdEmission::Gaussian)
        };
        let rows = [[1.0, 2.0], [3.0, 6.0], [2.0, 4.0], [5.0, 10.0]];
        let weights = [1.0, 0.0, 3.0, 2.0];
        let prior_at = |scale: f64| {
            let mut m = Bocpd::new(c.clone()).unwrap();
            for (x, w) in rows.iter().zip(weights) {
                m.step(x, &[], 1.0, scale * w);
            }
            m.prior().expect("three learned rows")
        };
        let (mean, scale) = prior_at(1.0);
        // Rows 0, 2 and 3 at weights 1, 3 and 2: V1 = 6, V2 = 14.
        let want = (1.0 * 1.0 + 3.0 * 2.0 + 2.0 * 5.0) / 6.0;
        // Tolerances (task 252 (b)): the closed form computed another way
        // (one division of the weighted sums; the model steps a Welford
        // update by inexact ratios such as `3/4`).
        assert!((mean[0] - want).abs() <= 1e-15 * want);
        let ss =
            1.0 * (1.0 - want).powi(2) + 3.0 * (2.0 - want).powi(2) + 2.0 * (5.0 - want).powi(2);
        let var = ss / (6.0 - 14.0 / 6.0);
        assert!((scale[0] - var).abs() <= 1e-14 * var, "{scale:?}");
        // x1 = 2·x0: the covariance is singular, so its diagonal.
        assert_eq!((scale[1], scale[2]), (0.0, 0.0), "{scale:?}");
        assert!((scale[3] - 4.0 * var).abs() <= 1e-14 * var, "{scale:?}");
        assert_eq!(prior_at(2f64.powi(-300)), prior_at(1.0), "a weight's scale");
        for level in [0.0, 0.5, 7.0] {
            let mut m = Bocpd::new(BocpdCfg {
                warm_rows: Some(3),
                ..from_the_data(1, BocpdEmission::Diag)
            })
            .unwrap();
            for _ in 0..3 {
                m.step(&[level], &[], 1.0, 1.0);
            }
            let floor = 2f64.powi(-52) * (level * level).max(1.0);
            assert_eq!(m.prior().unwrap().1, vec![floor], "level {level}");
        }
    }

    /// `warm_rows` is refused where it reads no row -- beside both priors --
    /// and below 2, where there is no variance to take, and past the
    /// buffer's budget.
    #[test]
    fn warm_rows_is_refused_where_it_cannot_set_a_prior() {
        let both = BocpdCfg {
            warm_rows: Some(5),
            ..cfg(2)
        };
        assert!(both.validate().unwrap_err().contains("warm_rows"));
        let one = BocpdCfg {
            warm_rows: Some(1),
            ..from_the_data(2, BocpdEmission::Diag)
        };
        assert!(one.validate().unwrap_err().contains("at least 2"));
        let huge = BocpdCfg {
            warm_rows: Some(1 << 40),
            ..from_the_data(2, BocpdEmission::Diag)
        };
        assert!(huge.validate().unwrap_err().contains("MiB"));
    }

    /// `prior_nu`'s default is the smallest integer that gives the prior's
    /// variance a mean: `d + 2` for the inverse-Wishart of `gaussian`
    /// (`ν > d + 1`), and 3 for the per-feature normal-inverse-gamma of
    /// `diag` and `robust` (`ν > 2`) whatever `d`. It was `d + 2` under
    /// every emission, so a per-feature prior grew stronger with the
    /// feature count for no reason (review round 4, CE2; docs/PLAN.md task
    /// 195, U5).
    #[test]
    fn the_default_prior_nu_is_the_smallest_giving_each_emission_a_mean() {
        for d in [1usize, 3, 6] {
            let with = |emission| BocpdCfg {
                emission,
                prior_nu: None,
                ..cfg(d)
            };
            assert_eq!(with(BocpdEmission::Gaussian).nu0(), d as f64 + 2.0);
            assert_eq!(with(BocpdEmission::Diag).nu0(), 3.0, "d = {d}");
            assert_eq!(with(BocpdEmission::Robust).nu0(), 3.0, "d = {d}");
        }
    }

    /// `learn` as it stood before task 250's fast path (`c65ef07`): no
    /// gamma cache, every run copied to take the row, and `prune` taking its
    /// own `log_sum_exp` of the joint. A refused row goes back through it,
    /// as `step` goes back through itself. The fast path is held to it to
    /// the bit below.
    fn learn_reference(
        m: &mut Bocpd,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
    ) -> crate::Step {
        if let Some(refused) = crate::model::refused_by(m, x, y, d_clock, weight, learn_reference) {
            return refused;
        }
        assert!(!m.warming(), "the reference starts after the warm-up");
        let (rel, w_mean, w_rows) = if weight > 0.0 {
            m.relative(weight)
        } else {
            (1.0, m.w_mean, m.w_rows)
        };
        let (moved, next) = m.before_row(y, d_clock);
        let joint = moved.as_deref().unwrap_or(m.logjoint.as_slice());
        let (pred, extra) = m.read(x, joint, next, rel, None);
        let out = crate::Step {
            pred,
            n_eff: m.n_eff,
            extra: None,
        };
        if weight <= 0.0 {
            if let Some(moved) = moved {
                m.logjoint = moved;
            }
            return out;
        }
        let Some((new, weights, _)) = extra else {
            m.solve_failures += 1;
            if let Some(moved) = moved {
                m.logjoint = moved;
            }
            return out;
        };
        m.n_eff += weight;
        (m.w_mean, m.w_rows) = (w_mean, w_rows);
        let full = m.full();
        let mut runs = Vec::with_capacity(m.runs.len() + 1);
        runs.push(Run::new(m.cfg.n_features, full));
        for (i, run) in m.runs.iter().enumerate() {
            let mut grown = run.clone();
            grown.add(x, weights[i], full);
            runs.push(grown);
        }
        m.runs = runs;
        m.logjoint = new;
        m.prune(None);
        out
    }

    fn step_bits(s: &crate::Step) -> (Vec<u64>, u64) {
        (
            s.pred.iter().map(|v| v.to_bits()).collect(),
            s.n_eff.to_bits(),
        )
    }

    /// The fast path -- the gamma cache, the runs growing in place, `prune`
    /// handed `read`'s `log_sum_exp` -- is the reference above to the bit,
    /// row by row and in the state's bytes, on streams that reach every
    /// branch it touches: each emission (the cache keyed on `νₙ` and on the
    /// inverse-Wishart's `ν − d + 1`), the data's prior and the cfg's, a
    /// per-row hazard and one on the clock, regime breaks, `prune_below`
    /// dropping runs and `max_run` folding a tail of several, rows of no
    /// weight, rows not a number, and a stretch of uneven weights (where the
    /// update is re-grown at `rel ≠ 1`, `prune` takes its own sum, and the
    /// cache misses). Each leg counts that its branches ran (task 252).
    #[test]
    fn the_fast_path_is_the_reference_to_the_bit() {
        let gaussian = BocpdCfg {
            emission: BocpdEmission::Gaussian,
            prior_scale: Some(vec![1.0, 0.2, 0.0, 0.2, 1.0, 0.1, 0.0, 0.1, 1.0]),
            prior_nu: Some(6.0),
            ..cfg(3)
        };
        let robust = BocpdCfg {
            emission: BocpdEmission::Robust,
            robust_beta: 0.1,
            ..cfg(2)
        };
        let legs: Vec<(&str, BocpdCfg)> = vec![
            ("diag", cfg(2)),
            ("gaussian", gaussian),
            ("robust", robust),
            ("diag, data prior", from_the_data(2, BocpdEmission::Diag)),
            (
                "gaussian, data prior",
                from_the_data(3, BocpdEmission::Gaussian),
            ),
            ("diag, on the clock", on_clock(2, 40.0)),
        ];
        for (name, c) in legs {
            let c = BocpdCfg {
                max_run: 25,
                prune_below: 1e-6,
                hazard: if c.hazard_on_clock { 40.0 } else { 60.0 },
                ..c
            };
            let d = c.n_features;
            let cap = (c.max_run - 1) as f64;
            let mut fast = Bocpd::new(c).unwrap();
            let mut n = Normals::new(1234);
            let row = |n: &mut Normals, t: usize| -> Vec<f64> {
                let level = [0.0, 4.0, -3.0, 1.0][(t / 150) % 4];
                (0..d).map(|_| level + n.normal()).collect()
            };
            // The warm-up, where the prior comes from the data, then the
            // reference takes a copy and both read the same rows.
            let mut t = 0;
            while fast.warming() {
                fast.step(&row(&mut n, t), &[], 1.0, 1.0);
                t += 1;
            }
            let mut slow = fast.clone();
            slow.gammas = GammaCache::default();
            let (mut pruned, mut folded, mut folded_many) = (0, 0, 0);
            let (mut zero, mut nan, mut uneven) = (0, 0, 0);
            for t in t..1_200 {
                let mut x = row(&mut n, t);
                let mut w = 1.0;
                if t % 97 == 13 {
                    w = 0.0;
                    zero += 1;
                } else if t % 131 == 7 {
                    x[0] = f64::NAN;
                    nan += 1;
                } else if (700..760).contains(&t) {
                    w = 0.5 + (t % 5) as f64 * 0.4;
                    uneven += usize::from(w != 1.0);
                }
                let dc = if t % 11 == 0 { 3.5 } else { 1.0 };
                let before = fast.runs.len();
                let at_cap = fast.runs.iter().filter(|r| r.len >= cap - 1.0).count();
                let a = fast.step(&x, &[], dc, w);
                let b = learn_reference(&mut slow, &x, &[], dc, w);
                assert_eq!(step_bits(&a), step_bits(&b), "{name}, row {t}");
                if w > 0.0 && x[0].is_finite() {
                    let grew = before + 1;
                    let capped = fast.runs.last().unwrap().len == cap;
                    folded += usize::from(capped);
                    folded_many += usize::from(capped && at_cap > 1);
                    pruned += usize::from(!capped && fast.runs.len() < grew);
                }
                if t % 100 == 0 {
                    assert_eq!(
                        rmp_serde::to_vec(&fast).unwrap(),
                        rmp_serde::to_vec(&slow).unwrap(),
                        "{name}, row {t}"
                    );
                }
            }
            assert_eq!(
                rmp_serde::to_vec_named(&fast).unwrap(),
                rmp_serde::to_vec_named(&slow).unwrap(),
                "{name}"
            );
            assert_eq!(fast, slow, "{name}");
            let g = &fast.gammas;
            eprintln!(
                "{name}: {} hits, {} misses; pruned {pruned}, folded {folded} ({folded_many} of several)",
                g.hits, g.misses
            );
            assert!(
                g.hits > 1_000 && g.misses > 100,
                "{name}: the cache hit {} times and missed {}",
                g.hits,
                g.misses
            );
            assert_eq!(slow.gammas.hits, 0, "{name}: the reference took no cache");
            assert!(
                pruned > 5 && folded > 1_000 && folded_many > 5,
                "{name}: pruned on {pruned} rows, folded on {folded}, {folded_many} of several"
            );
            assert!(zero > 5 && nan > 5 && uneven > 30, "{name}");
        }
    }

    /// The cache on a long stream of one weight hits nearly every lookup,
    /// and every value it holds is `gamma_diff` taken again at its key, to
    /// the bit.
    #[test]
    fn the_gamma_cache_hits_on_a_long_stream_and_holds_what_it_computes() {
        for (c, a) in [
            (cfg(2), 1.0),
            (
                BocpdCfg {
                    emission: BocpdEmission::Gaussian,
                    prior_scale: Some(vec![1.0, 0.0, 0.0, 1.0]),
                    prior_nu: Some(5.0),
                    ..cfg(2)
                },
                2.0,
            ),
        ] {
            let mut m = Bocpd::new(BocpdCfg {
                max_run: 60,
                prune_below: 1e-6,
                ..c
            })
            .unwrap();
            let mut n = Normals::new(8);
            for _ in 0..5_000 {
                m.step(&[n.normal(), n.normal()], &[], 1.0, 1.0);
            }
            let g = &m.gammas;
            assert!(
                g.hits > 50 * g.misses,
                "{} hits, {} misses",
                g.hits,
                g.misses
            );
            assert_eq!(g.a, f64::to_bits(a));
            assert_eq!(
                (g.keys.len(), g.vals.len()),
                (GammaCache::SLOTS, GammaCache::SLOTS)
            );
            let empty = f64::NAN.to_bits();
            let filled: Vec<usize> = (0..g.keys.len()).filter(|&i| g.keys[i] != empty).collect();
            assert!(filled.len() >= 30, "{} slots filled", filled.len());
            for i in filled {
                let nu = f64::from_bits(g.keys[i]);
                assert_eq!(g.vals[i].to_bits(), gamma_diff(nu, a).to_bits(), "nu {nu}");
            }
        }
    }

    /// A slot is keyed on every bit of `ν`: two values in the same slot
    /// evict each other and each reads its own value, `0.0` and `-0.0` are
    /// two keys, another `a` empties the cache, a NaN is computed and never
    /// stored, and no number of distinct keys grows it past its slots.
    #[test]
    fn the_gamma_cache_never_shares_an_entry_between_two_keys() {
        let slot = |nu: f64| GammaCache::slot(nu.to_bits());
        let mut c = GammaCache::default();
        assert_eq!(
            c.get(f64::NAN, 1.0).to_bits(),
            gamma_diff(f64::NAN, 1.0).to_bits()
        );
        assert!(
            c.keys.is_empty() && c.hits + c.misses == 0,
            "a NaN is not stored"
        );
        let p: f64 = 3.5;
        let q = (1..)
            .map(|k| f64::from_bits(p.to_bits() + k))
            .find(|q| slot(*q) == slot(p))
            .unwrap();
        for (nu, hits, misses) in [(p, 0, 1), (p, 1, 1), (q, 1, 2), (p, 1, 3), (q, 1, 4)] {
            assert_eq!(
                c.get(nu, 1.0).to_bits(),
                gamma_diff(nu, 1.0).to_bits(),
                "{nu}"
            );
            assert_eq!((c.hits, c.misses), (hits, misses), "{nu}");
        }
        c.get(0.0, 1.0);
        let misses = c.misses;
        c.get(-0.0, 1.0);
        assert_eq!(c.misses, misses + 1, "-0.0 is not 0.0");
        // Another `a` asks for another function: nothing from the first is
        // reused.
        let misses = c.misses;
        assert_eq!(c.get(q, 3.0).to_bits(), gamma_diff(q, 3.0).to_bits());
        assert_eq!(c.misses, misses + 1, "a new a starts empty");
        assert_eq!(c.get(q, 3.0).to_bits(), gamma_diff(q, 3.0).to_bits());
        assert_eq!(c.misses, misses + 1, "and then fills");
        for k in 0..10_000u64 {
            c.get(1.0 + k as f64 * 0.37, 3.0);
        }
        assert_eq!(
            (c.keys.len(), c.vals.len()),
            (GammaCache::SLOTS, GammaCache::SLOTS)
        );
    }

    /// The cache is not state: a state written with a warm cache is the
    /// bytes of one written by a model that never had one, a restored model
    /// starts empty and equal to the one it came from, and it goes on as
    /// that model, to the bit, rebuilding its cache as it goes.
    #[test]
    fn a_restored_model_starts_with_no_cache_and_goes_on_to_the_bit() {
        let mut m = Bocpd::new(BocpdCfg {
            max_run: 40,
            prune_below: 1e-5,
            ..from_the_data(2, BocpdEmission::Diag)
        })
        .unwrap();
        let mut n = Normals::new(31);
        for _ in 0..600 {
            m.step(&[n.normal(), n.normal()], &[], 1.0, 1.0);
        }
        assert!(m.gammas.hits > 0 && !m.gammas.keys.is_empty());
        let mut cold = m.clone();
        cold.gammas = GammaCache::default();
        assert_eq!(
            rmp_serde::to_vec(&m).unwrap(),
            rmp_serde::to_vec(&cold).unwrap()
        );
        assert_eq!(
            rmp_serde::to_vec_named(&m).unwrap(),
            rmp_serde::to_vec_named(&cold).unwrap()
        );
        // A state read back from its file, as the bank loads one; and the
        // model's own bytes.
        let saved = rmp_serde::to_vec(&m.state()).unwrap();
        let state: crate::State = rmp_serde::from_slice(&saved).unwrap();
        let mut back = Bocpd::restore(&state).unwrap();
        let bytes = rmp_serde::to_vec(&m).unwrap();
        let mut loaded: Bocpd = rmp_serde::from_slice(&bytes).unwrap();
        for r in [&back, &loaded] {
            assert!(r.gammas.keys.is_empty() && r.gammas.hits + r.gammas.misses == 0);
            assert_eq!(*r, m, "equality ignores the cache");
        }
        for t in 0..400 {
            let x = [n.normal() + if t > 200 { 3.0 } else { 0.0 }, n.normal()];
            let want = step_bits(&m.step(&x, &[], 1.0, 1.0));
            assert_eq!(step_bits(&back.step(&x, &[], 1.0, 1.0)), want, "row {t}");
            assert_eq!(step_bits(&loaded.step(&x, &[], 1.0, 1.0)), want, "row {t}");
        }
        assert_eq!(
            rmp_serde::to_vec(&back).unwrap(),
            rmp_serde::to_vec(&m).unwrap()
        );
        assert!(back.gammas.misses > 0 && back.gammas.hits > back.gammas.misses);
    }

    /// Task 252: no columns, and a scale prior that is not finite, are
    /// refused by name; with no scale prior the configuration's `Ψ₀` reads
    /// as the identity.
    #[test]
    fn no_columns_or_a_scale_that_is_not_finite_is_refused_and_none_is_the_identity() {
        let bad = |c: BocpdCfg, msg: &str| {
            let e = Bocpd::new(c).unwrap_err();
            assert_eq!(e, msg);
        };
        bad(
            BocpdCfg {
                n_features: 0,
                prior_mean: Some(vec![]),
                ..cfg(1)
            },
            "bocpd: at least one column is required",
        );
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            bad(
                BocpdCfg {
                    prior_scale: Some(vec![v]),
                    ..cfg(1)
                },
                "bocpd: covariance_prior must be finite",
            );
            bad(
                BocpdCfg {
                    prior_scale: Some(vec![1.0, 0.0, 0.0, v]),
                    ..cfg(2)
                },
                "bocpd: covariance_prior must be finite",
            );
        }
        // And a pruning share no run reaches drops none: with `H = 1/2`
        // the first row splits the mass evenly, below 0.9 on each side, and
        // never-pruning `r = 0` beside a dropped rest is not done either --
        // the runs and the joint are the unpruned model's to the bit.
        let mut pruned = Bocpd::new(BocpdCfg {
            hazard: 2.0,
            prune_below: 0.9,
            ..cfg(1)
        })
        .unwrap();
        let mut plain = Bocpd::new(BocpdCfg {
            hazard: 2.0,
            ..cfg(1)
        })
        .unwrap();
        pruned.step(&[0.3], &[], 1.0, 1.0);
        plain.step(&[0.3], &[], 1.0, 1.0);
        assert_eq!(pruned.runs.len(), 2, "both runs kept");
        assert_eq!(pruned.runs, plain.runs);
        assert_eq!(
            pruned
                .logjoint
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            plain
                .logjoint
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let none = BocpdCfg {
            prior_scale: None,
            ..cfg(3)
        };
        assert_eq!(none.psi0(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
}
