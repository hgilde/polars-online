//! `corrchange`: has the correlation structure changed?
//! (docs/ENHANCEMENTS.md E59)
//!
//! Two tests, because there are two questions.
//!
//! # `monitor`: the constancy test of Wied, Krämer & Dehling (2012)
//!
//! A **closed-sample** fluctuation test, run span by span. Over a span of
//! `T` rows, per pair:
//!
//! ```text
//! Q = max_{2≤j≤T} (j/√T)·|ρ̂ⱼ − ρ̂_T| / D̂
//! ```
//!
//! `ρ̂ⱼ` is the sample correlation of the span's first `j` rows and `D̂` is
//! the delta-method long-run standard deviation of `ρ̂`: the five raw
//! moments `Uₜ = (x², y², x, y, xy)` centred at their span means, their
//! Bartlett long-run covariance `Σ̂` at bandwidth `γ_T = ⌊ln T⌋`, mapped to
//! `(σ_x², σ_y², σ_xy)` by `D₂` and to `ρ` by `D₃`, so `D̂² = D₃D₂Σ̂D₂'D₃'`.
//! It is computed on the span centred at its means and scaled by its
//! standard deviations, which leaves `ρ̂` and `D̂` where they were (the
//! shifted moments are an affine image of the raw ones, and the Jacobians
//! cancel) and puts the arithmetic at the spread's scale rather than the
//! level's: there `m_x = m_y = 0` and both variances are 1, so `D₃D₂ =
//! (−ρ̂/2, −ρ̂/2, 0, 0, 1)` and `D̂²` is the Bartlett long-run variance of
//! the one series `ξₜ = x̃ₜỹₜ − (ρ̂/2)(x̃ₜ² + ỹₜ²)`. Formed from the raw
//! moments, `σ_x² = E[x²] − E[x]²` lost the level's digits: 1.2e-5 of `D̂`
//! at a level of 1e5, and NaN at 1e8 (docs/PLAN.md task 103).
//!
//! The kernel is WKD's: `D̂₁ = ΣₜΣᵤ k((t−u)/γ_T)VₜVᵤ'` with `k(x) = 1 − |x|`
//! (their Appendix A.1), so lag `l` is weighted `1 − l/γ` and lag `γ` not at
//! all. Until docs/PLAN.md task 114 (the user's decision, 2026-09-28) it
//! was Newey–West's `1 − l/(γ+1)` over lags `0..=γ`; measured at WKD's Table
//! 1 and 2 settings, the change moved the size by at most 0.0008 and the
//! power by 0.003. At `bandwidth = 1` only lag 0 is left.
//!
//! Under the null `Q →_d sup|B|`, a Brownian bridge, whose quantiles are
//! the Kolmogorov distribution -- computed from the series here rather than
//! pinned, so a test can check it reproduces 1.3581 at 5 %.
//!
//! The paper's own form is *sequential*, with a boundary function, in Wied
//! & Galeano (2013), read from their SFB 823 preprint (Discussion Paper
//! 12/2012) on 2026-09-28, docs/PLAN.md task 114: the detector
//! `V_k = D̂·(k/√m)·(ρ̂^{m+k}_{m+1} − ρ̂^m_1)` against `c·w(k/m)`, `w(b) =
//! (1 + b)(b/(1 + b))^γ`, with `D̂` from the `m` historical rows and `c`
//! simulated. What ships is the closed test run over consecutive spans of
//! `horizon` rows. The cost is a delay of at most `horizon` rows and the
//! benefit is a null with published tables -- which is the whole point of
//! the exercise.
//!
//! # `window`: two adjacent windows
//!
//! `‖vech(R̂_pre − R̂_post)‖`, the size of the change in the correlation
//! matrix between two adjacent windows, against a fixed threshold or a
//! **permutation** critical value. Not a sign-flip null: negating a whole
//! row leaves every `xₜxₜ'` and so every correlation matrix exactly where
//! it was, so a sign-flip null has no spread at all. The exchangeable null
//! for "the two windows share a distribution" is a permutation of the
//! pooled rows between them, in blocks of `perm_block` so that serially
//! dependent rows do not make it too liberal.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::{Decay, EwDiag, SplitMix64};

/// Which test; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrChangeKind {
    Monitor,
    Window,
    /// Wied & Galeano's (2013) detector: a history of `span_rows` rows, then
    /// every row of a monitoring period tested against it.
    Sequential,
}

/// The norm the `window` kind measures the change in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeNorm {
    L1,
    LInf,
}

/// `P(sup|B| ≤ x) = 1 − 2·Σ_{k≥1} (−1)^{k−1}·exp(−2k²x²)`, the Kolmogorov
/// distribution: the limiting law of the `monitor` statistic.
pub fn kolmogorov_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 1..200 {
        let term = (-2.0 * (k * k) as f64 * x * x).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (1.0 - 2.0 * sum).clamp(0.0, 1.0)
}

/// The `1 − alpha` quantile of `sup|B|`, by bisection on
/// [`kolmogorov_cdf`]. `1.3581` at 5 %, `1.6276` at 1 %, `1.2239` at 10 %.
pub fn kolmogorov_quantile(alpha: f64) -> f64 {
    if !(alpha > 0.0 && alpha < 1.0) {
        return f64::NAN;
    }
    let target = 1.0 - alpha;
    let (mut lo, mut hi) = (0.0f64, 10.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if kolmogorov_cdf(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The Bartlett kernel `k(x) = 1 − |x|` on `[−1, 1]`.
#[inline]
fn bartlett(x: f64) -> f64 {
    let a = x.abs();
    if a < 1.0 { 1.0 - a } else { 0.0 }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrChangeCfg {
    pub n_features: usize,
    pub kind: CorrChangeKind,
    /// Rows per comparison block: the span `T` that `"monitor"` tests for
    /// constancy, or the length of each of the two adjacent blocks
    /// `"window"` compares. One number, because it is one concept -- the two
    /// kinds only differ in what they do with the block.
    pub span_rows: usize,
    /// Nominal level; under `monitor` and `sequential` the critical value is
    /// the quantile at `1 − alpha/npairs` under Bonferroni. Under
    /// `sequential`, a share below `2^-52` with no `crit` is refused
    /// ([`Self::validate`]).
    pub alpha: f64,
    /// `"bonferroni"` (default) or `"none"`: whether `monitor` and
    /// `sequential` split `alpha` over the pairs. Under `window` it changes
    /// nothing, since there is one statistic over all the pairs: the
    /// permutation critical value is taken at `alpha`.
    pub alpha_adjust: String,
    /// `monitor`: the Bartlett bandwidth, `⌊ln T⌋` when `None`.
    pub bandwidth: Option<usize>,
    /// `monitor`: run the CUSUM on the equicorrelation of the standardised
    /// row instead of every pair.
    pub scalar: bool,
    /// `scalar`: the standardiser's decay.
    pub decay: Decay,
    /// `window`: a fixed critical value, or `None` for the permutation one.
    pub crit: Option<f64>,
    pub n_perm: usize,
    pub permute_every: usize,
    pub perm_block: usize,
    pub norm: ChangeNorm,
    pub seed: u64,
    /// Empty both rings and start over at a flag.
    pub reset: bool,
    /// `sequential`: the rows monitored after each history, W&G's `⌊mT⌋`,
    /// so their `T` is `monitor_rows / span_rows`.
    #[serde(default)]
    pub monitor_rows: usize,
    /// `sequential`: the boundary's exponent `γ` in `w(b) = (1 + b)(b/(1 +
    /// b))^γ`, `0 ≤ γ ≤ 0.49`; 0 is the straight boundary `1 + b`. W&G
    /// allow up to 1/2, but the critical value is solved for, and the
    /// solve's work grows as `1/(1/2 − γ)`.
    #[serde(default)]
    pub boundary_gamma: f64,
}

/// The largest `boundary_gamma` a configuration may give. W&G's boundary
/// takes `γ < 1/2`, but the critical value is solved for, and the solve's
/// work grows as `1/(1/2 − γ)`: seconds here, and hours just under 1/2
/// (review 2026-10-05, CD1).
const MAX_BOUNDARY_GAMMA: f64 = 0.49;

impl CorrChangeCfg {
    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("corrchange: {e}"))?;
        if let Some(c) = self.crit {
            // A NaN never flags and a non-positive one flags every row, both
            // in silence (docs/REVIEW-E54-E64.md CC1).
            if !(c > 0.0 && c.is_finite()) {
                return Err(format!(
                    "corrchange: crit is the critical value the statistic is compared with and \
                     must be finite and > 0 (got {c}); leave it out for the tabulated one"
                ));
            }
        }
        if self.n_features < 2 {
            return Err(
                "corrchange: at least two columns are needed; a correlation is between two \
                 things"
                    .into(),
            );
        }
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err("corrchange: alpha must be strictly between 0 and 1".into());
        }
        if !["bonferroni", "none"].contains(&self.alpha_adjust.as_str()) {
            return Err(format!(
                "corrchange: unknown alpha_adjust {:?}; expected \"bonferroni\" or \"none\"",
                self.alpha_adjust
            ));
        }
        match self.kind {
            CorrChangeKind::Monitor => {
                if self.span_rows < 8 {
                    return Err(
                        "corrchange: kind = \"monitor\" needs span_rows of at least 8; the \
                         statistic is a maximum over the span"
                            .into(),
                    );
                }
                if self.bandwidth == Some(0) {
                    return Err("corrchange: bandwidth must be >= 1".into());
                }
            }
            CorrChangeKind::Sequential => {
                if self.span_rows < 8 {
                    return Err(
                        "corrchange: kind = \"sequential\" needs span_rows of at least 8; the \
                         history's long-run variance is read from them"
                            .into(),
                    );
                }
                if self.monitor_rows < 2 {
                    return Err(
                        "corrchange: kind = \"sequential\" needs monitor_rows of at least 2; \
                         a correlation over the monitored rows needs two"
                            .into(),
                    );
                }
                // Not up to 1/2: the critical value is solved for
                // (`crate::boundary`), and the solve's work grows as 1/(1/2
                // − γ). Measured under load on a release build, 0.9 s at
                // 0.45, 2.8 s at 0.49 and 32 s at 0.499, and about an hour
                // at 0.49999 by its step count, in every process and on
                // every load (review 2026-10-05, CD1).
                if !(0.0..=MAX_BOUNDARY_GAMMA).contains(&self.boundary_gamma) {
                    return Err(format!(
                        "corrchange: boundary_gamma must be in [0, {MAX_BOUNDARY_GAMMA}] (got {}); \
                         the critical value's solve takes work that grows as 1/(1/2 − γ), \
                         seconds at {MAX_BOUNDARY_GAMMA} and hours just under 1/2; at 1/2 the \
                         boundary is crossed with probability 1 whatever the data",
                        self.boundary_gamma
                    ));
                }
                if self.bandwidth == Some(0) {
                    return Err("corrchange: bandwidth must be >= 1".into());
                }
                // The critical value is a quantile at `1 − alpha/npairs`
                // (`alpha` itself without Bonferroni), and below `2^-52`
                // that is 1, or a double a step or two from it: the quantile
                // of 1 is NaN, and the detector ran, reported its statistic
                // and could never flag, saying nothing (review round 4,
                // CD12). The tail cannot be read in its own right either:
                // the series and the solve both give the mass inside the
                // boundary, `1 −` the tail, and the solve's error is far
                // above a tail that small. A configured `crit` reads no
                // `alpha`.
                let floor = 2f64.powi(-52);
                if self.crit.is_none() && self.pair_alpha() < floor {
                    let share = if self.alpha_adjust == "bonferroni" {
                        format!(
                            "alpha / {} pairs = {:e} (Bonferroni's share)",
                            self.npairs(),
                            self.pair_alpha()
                        )
                    } else {
                        format!("alpha = {:e}", self.alpha)
                    };
                    return Err(format!(
                        "corrchange: kind = \"sequential\" tests each pair at {share}, below \
                         2^-52: as a double, 1 minus that is 1 or within two steps of it, where \
                         no critical value can be read; raise alpha, or give crit"
                    ));
                }
            }
            CorrChangeKind::Window => {
                if self.span_rows < 3 {
                    return Err(
                        "corrchange: kind = \"window\" needs span_rows of at least 3".into(),
                    );
                }
                if self.crit.is_none() && self.n_perm < 20 {
                    return Err(
                        "corrchange: a permutation critical value needs n_perm >= 20 draws".into(),
                    );
                }
                if self.perm_block == 0 || self.perm_block > self.span_rows {
                    return Err(format!(
                        "corrchange: perm_block must be 1..={} (span_rows)",
                        self.span_rows
                    ));
                }
                if self.permute_every == 0 {
                    return Err("corrchange: permute_every must be >= 1".into());
                }
                if self.scalar {
                    return Err(
                        "corrchange: scalar applies to kind = \"monitor\"; the window statistic \
                         is already one number"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Pairs the statistic is a maximum over.
    fn npairs(&self) -> usize {
        if self.scalar {
            1
        } else {
            self.n_features * (self.n_features - 1) / 2
        }
    }

    /// The level each pair is tested at: `alpha`, over the pairs under
    /// Bonferroni.
    fn pair_alpha(&self) -> f64 {
        if self.alpha_adjust == "bonferroni" {
            self.alpha / self.npairs() as f64
        } else {
            self.alpha
        }
    }

    /// The critical value: the Kolmogorov quantile under `monitor`, the
    /// configured one under `window`, and under `sequential` the configured
    /// one or Wied & Galeano's (`crate::boundary::sequential_crit`).
    fn fixed_crit(&self) -> Option<f64> {
        match self.kind {
            CorrChangeKind::Monitor => Some(kolmogorov_quantile(self.pair_alpha())),
            CorrChangeKind::Window => self.crit,
            CorrChangeKind::Sequential => self.crit.or_else(|| {
                Some(crate::boundary::sequential_crit(
                    self.pair_alpha(),
                    self.boundary_gamma,
                    self.monitor_rows as f64 / self.span_rows as f64,
                ))
            }),
        }
    }

    /// Output slots per statistic: one per pair, or one for `scalar`.
    fn width(&self) -> usize {
        if self.scalar { 1 } else { self.n_features }
    }
}

/// `sequential`'s monitoring period: what the history left, and the
/// monitored rows so far (docs/PLAN.md task 114). Before it, while the
/// history fills, the history is the model's `ring`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Monitoring {
    /// Per pair (`a < b` in order), or the one `scalar` series: the
    /// history's correlation, or the mean of `u`.
    level: Vec<f64>,
    /// And its long-run standard deviation, the inverse of W&G's `D̂`;
    /// NaN where the history gives none (a constant column, a collinear
    /// pair), and that pair has no verdict until the next history.
    scale: Vec<f64>,
    /// The monitored rows so far, `k − 1` on the `k`-th monitored row.
    rows: Vec<Vec<f64>>,
    /// Their running means, second moments and co-moments (per pair, in
    /// the order of `level`), the mean-form (Welford) sums.
    mean: Vec<f64>,
    m2: Vec<f64>,
    cross: Vec<f64>,
}

/// The constancy monitor; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrChange {
    cfg: CorrChangeCfg,
    /// The span's rows (`monitor`) or the two adjacent windows (`window`,
    /// oldest first: the first `window` are `pre`).
    ring: VecDeque<Vec<f64>>,
    /// `scalar`: the standardiser the row is divided by.
    diag: EwDiag,
    n_eff: f64,
    /// Learned rows since the last flag; `None` before the first.
    since_flag: Option<u64>,
    /// The permutation critical value in force, and the rows since it was
    /// drawn.
    perm_crit: Option<f64>,
    since_perm: usize,
    rng: SplitMix64,
    /// `sequential`: the monitoring period in progress, `None` while the
    /// history fills `ring` (and under the other kinds).
    #[serde(default)]
    monitoring: Option<Monitoring>,
    /// `sequential`: the critical value, a function of the cfg alone. Not
    /// state: it calls `exp`, whose last bits are the platform's, so it is
    /// computed on construction and on load.
    #[serde(skip)]
    seq_crit: f64,
}

impl CorrChange {
    pub fn new(cfg: CorrChangeCfg) -> Result<Self, String> {
        cfg.validate()?;
        let d = cfg.n_features;
        let seq_crit = Self::sequential_crit_of(&cfg);
        Ok(Self {
            ring: VecDeque::new(),
            diag: EwDiag::new(d),
            n_eff: 0.0,
            since_flag: None,
            perm_crit: None,
            since_perm: 0,
            rng: SplitMix64::new(cfg.seed),
            monitoring: None,
            seq_crit,
            cfg,
        })
    }

    /// `sequential`'s critical value, or NaN under the other kinds.
    fn sequential_crit_of(cfg: &CorrChangeCfg) -> f64 {
        match cfg.kind {
            CorrChangeKind::Sequential => cfg.fixed_crit().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    pub fn cfg(&self) -> &CorrChangeCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// Rows in the ring, which is the span so far or the two windows.
    pub fn depth(&self) -> usize {
        self.ring.len()
    }

    /// Output slot labels in emission order.
    pub fn labels() -> Vec<String> {
        vec![
            "stat".into(),
            "crit".into(),
            "flag".into(),
            "since_flag".into(),
            "since_change".into(),
        ]
    }

    pub fn n_outputs_for() -> usize {
        5
    }

    /// The sample correlation of `rows[..n]` for the pair `(a, b)`.
    fn corr_of(rows: &[Vec<f64>], n: usize, a: usize, b: usize) -> f64 {
        if n < 2 {
            return f64::NAN;
        }
        let nf = n as f64;
        let (mut sa, mut sb) = (0.0, 0.0);
        for r in &rows[..n] {
            sa += r[a];
            sb += r[b];
        }
        let (ma, mb) = (sa / nf, sb / nf);
        let (mut vaa, mut vbb, mut vab) = (0.0, 0.0, 0.0);
        for r in &rows[..n] {
            let (da, db) = (r[a] - ma, r[b] - mb);
            vaa += da * da;
            vbb += db * db;
            vab += da * db;
        }
        let den = (vaa * vbb).sqrt();
        if den > 0.0 { vab / den } else { f64::NAN }
    }

    /// `D̂`, the delta-method long-run standard deviation of `ρ̂` over the
    /// span, for the pair `(a, b)`: the Bartlett long-run standard deviation
    /// of `ξ_t = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)`, the rows centred at the
    /// span's means and scaled by its standard deviations (the module docs
    /// derive it from the five raw moments).
    fn long_run_sd(rows: &[Vec<f64>], a: usize, b: usize, gamma: usize) -> f64 {
        let t = rows.len();
        if t < 4 {
            return f64::NAN;
        }
        let tf = t as f64;
        // The span's means as pairs (`crate::comp`): a second pass over the
        // exact differences from the first pass's mean picks up what its
        // division rounded away, a shift of the whole span that a first-order
        // term in `ξ` would otherwise carry (1e-10 of `D̂` at a level of 1e8).
        let (mut mx, mut my) = (0.0, 0.0);
        for r in rows {
            mx += r[a];
            my += r[b];
        }
        let (mx, my) = (mx / tf, my / tf);
        let (mut mx_lo, mut my_lo) = (0.0, 0.0);
        for r in rows {
            mx_lo += r[a] - mx;
            my_lo += r[b] - my;
        }
        let (mx_lo, my_lo) = (mx_lo / tf, my_lo / tf);
        let dev = |r: &Vec<f64>| {
            (
                crate::comp::dev(r[a], mx, mx_lo),
                crate::comp::dev(r[b], my, my_lo),
            )
        };
        // Its moments about them.
        let (mut sx2, mut sy2, mut sxy) = (0.0, 0.0, 0.0);
        for r in rows {
            let (dx, dy) = dev(r);
            sx2 += dx * dx;
            sy2 += dy * dy;
            sxy += dx * dy;
        }
        let (sx2, sy2, sxy) = (sx2 / tf, sy2 / tf, sxy / tf);
        if !(sx2 > 0.0 && sy2 > 0.0) {
            return f64::NAN;
        }
        let (sx, sy) = (sx2.sqrt(), sy2.sqrt());
        let rho = sxy / (sx * sy);
        // A pair within rounding of `|ρ̂| = 1` has no long-run standard
        // deviation to divide by: the delta method's variance is
        // `(1 − ρ²)²·(…)`, 0 on a collinear pair, and this `D̂` reaches that
        // 0 where the raw-moment form's own noise stood in for it, so the
        // numerator's rounding divided by it flagged every span of `y = 2x +
        // 3` (review 2026-09-25). NaN is no verdict, as for a constant
        // column.
        if 1.0 - rho * rho <= 64.0 * f64::EPSILON {
            return f64::NAN;
        }
        // `ξ`, whose span mean is 0 to rounding by construction: `E[x̃ỹ] =
        // ρ̂` and `E[x̃²] = E[ỹ²] = 1` over the same rows, so there is
        // nothing to centre.
        let xi: Vec<f64> = rows
            .iter()
            .map(|r| {
                let (dx, dy) = dev(r);
                let (u, v) = (dx / sx, dy / sy);
                u * v - 0.5 * rho * (u * u + v * v)
            })
            .collect();
        // Its Bartlett long-run variance: the lag-0 term once, each other
        // lag's autocovariance twice, at WKD's `k(l/γ)` (their Appendix A.1),
        // which puts lag `γ` at 0; Newey–West's `1 − l/(γ+1)` stood here
        // until docs/PLAN.md task 114.
        let mut v = 0.0;
        for lag in 0..t.min(gamma) {
            let w = bartlett(lag as f64 / gamma as f64);
            let mut acf = 0.0;
            for s in lag..t {
                acf += xi[s] * xi[s - lag];
            }
            v += w * acf / tf * if lag > 0 { 2.0 } else { 1.0 };
        }
        if v > 0.0 { v.sqrt() } else { f64::NAN }
    }

    /// `Q` for the **mean** of a one-column span: `max_j (j/√T)·|ūⱼ − ū_T|
    /// / D̂ᵤ`, the same CUSUM with the Bartlett long-run standard deviation
    /// of `u` in place of the delta-method one. That is `scalar = true`'s
    /// statistic: the equicorrelation is already one number, so there is a
    /// mean to test and no pair.
    #[cfg(test)]
    fn scalar_stat(&self, rows: &[Vec<f64>]) -> f64 {
        self.scalar_stat_at(rows).0
    }

    /// [`Self::scalar_stat`], and the `j` its maximum is attained at.
    fn scalar_stat_at(&self, rows: &[Vec<f64>]) -> (f64, usize) {
        let t = rows.len();
        if t < 4 {
            return (f64::NAN, 0);
        }
        let tf = t as f64;
        let u: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let mean = u.iter().sum::<f64>() / tf;
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((tf).ln().floor() as usize).max(1));
        let sd = Self::scalar_long_run_sd(&u, gamma);
        if sd.is_nan() {
            return (f64::NAN, 0);
        }
        let mut run = 0.0;
        let (mut best, mut at) = (0.0f64, 0);
        for (j, uj) in u.iter().enumerate() {
            run += uj;
            let jj = j as f64 + 1.0;
            let q = (jj / tf.sqrt()) * (run / jj - mean).abs() / sd;
            if q > best {
                (best, at) = (q, j + 1);
            }
        }
        (best, at)
    }

    /// The Bartlett long-run standard deviation of one series about its
    /// mean, at WKD's `k(l/γ)`; NaN where it is not positive.
    fn scalar_long_run_sd(u: &[f64], gamma: usize) -> f64 {
        let t = u.len();
        let tf = t as f64;
        let mean = u.iter().sum::<f64>() / tf;
        let v: Vec<f64> = u.iter().map(|x| x - mean).collect();
        let mut var = 0.0;
        for lag in 0..t.min(gamma) {
            let w = bartlett(lag as f64 / gamma as f64);
            for s in lag..t {
                let term = w * v[s] * v[s - lag] / tf;
                var += term;
                if lag > 0 {
                    var += term;
                }
            }
        }
        if var.is_nan() || var <= 0.0 {
            f64::NAN
        } else {
            var.sqrt()
        }
    }

    /// `Q` for one pair over the span in the ring.
    #[cfg(test)]
    fn monitor_stat(&self, rows: &[Vec<f64>], a: usize, b: usize) -> f64 {
        self.monitor_stat_at(rows, a, b).0
    }

    /// [`Self::monitor_stat`], and the `j` its maximum is attained at: the
    /// last row before the change, as Wied & Galeano (2013, Eq. 8) date a
    /// change from the same CUSUM.
    fn monitor_stat_at(&self, rows: &[Vec<f64>], a: usize, b: usize) -> (f64, usize) {
        let t = rows.len();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((t as f64).ln().floor() as usize).max(1));
        let sd = Self::long_run_sd(rows, a, b, gamma);
        // NaN or non-positive: no scale to divide by.
        if sd.is_nan() || sd <= 0.0 {
            return (f64::NAN, 0);
        }
        let rho_t = Self::corr_of(rows, t, a, b);
        if !rho_t.is_finite() {
            return (f64::NAN, 0);
        }
        let tf = t as f64;
        let (mut best, mut at) = (0.0f64, 0);
        for j in 2..=t {
            let rj = Self::corr_of(rows, j, a, b);
            if !rj.is_finite() {
                continue;
            }
            let v = (j as f64 / tf.sqrt()) * (rj - rho_t).abs() / sd;
            if v > best {
                (best, at) = (v, j);
            }
        }
        (best, at)
    }

    /// The pair `(a, b)`, `a < b`, at position `idx` in the order the pairs
    /// are walked (`a` outer, `b` inner).
    fn pair_at(d: usize, mut idx: usize) -> (usize, usize) {
        for a in 0..d {
            let row = d - a - 1;
            if idx < row {
                return (a, a + 1 + idx);
            }
            idx -= row;
        }
        unreachable!("pair index past the pairs")
    }

    /// `sequential`: what the history leaves the monitoring period -- per
    /// pair the history's correlation and the long-run standard deviation
    /// of it (the inverse of W&G's `D̂`, WKD's Appendix A.1 estimator on the
    /// `m` history rows, bandwidth `⌊ln m⌋` unless given), or for `scalar`
    /// the mean of `u` and its long-run standard deviation.
    fn begin_monitoring(&self, history: &[Vec<f64>]) -> Monitoring {
        let m = history.len();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((m as f64).ln().floor() as usize).max(1));
        let (mut level, mut scale) = (Vec::new(), Vec::new());
        if self.cfg.scalar {
            let u: Vec<f64> = history.iter().map(|r| r[0]).collect();
            level.push(u.iter().sum::<f64>() / m as f64);
            scale.push(Self::scalar_long_run_sd(&u, gamma));
        } else {
            let d = self.cfg.n_features;
            for a in 0..d {
                for b in (a + 1)..d {
                    level.push(Self::corr_of(history, m, a, b));
                    let sd = Self::long_run_sd(history, a, b, gamma);
                    scale.push(if sd > 0.0 { sd } else { f64::NAN });
                }
            }
        }
        let width = self.cfg.width();
        let pairs = if self.cfg.scalar { 0 } else { level.len() };
        Monitoring {
            level,
            scale,
            rows: Vec::new(),
            mean: vec![0.0; width],
            m2: vec![0.0; width],
            cross: vec![0.0; pairs],
        }
    }

    /// `sequential`'s detector on the row that would be the `k`-th
    /// monitored one: `max |V_k|/w(k/m)` over the pairs, `V_k = (k/√m)·
    /// (ρ̂^{m+k}_{m+1} − ρ̂^m_1)/σ̂` (W&G's Eq. 1, their `D̂` being `1/σ̂`),
    /// `w(b) = (1 + b)(b/(1 + b))^γ` (their Eq. 5), and the pair it is
    /// attained at. `ρ̂^{m+k}_{m+1}` is the correlation of the monitored
    /// rows with this one, from the mean-form sums a step on; NaN below two
    /// rows, where a correlation is not defined.
    fn sequential_stat(&self, mon: &Monitoring, row: &[f64]) -> (f64, usize) {
        let k = mon.rows.len() + 1;
        if k < 2 {
            return (f64::NAN, 0);
        }
        let kf = k as f64;
        let m = self.cfg.span_rows as f64;
        let b = kf / m;
        let w = (1.0 + b) * (b / (1.0 + b)).powf(self.cfg.boundary_gamma);
        let factor = kf / m.sqrt();
        let width = mon.mean.len();
        let dev: Vec<f64> = (0..width).map(|i| row[i] - mon.mean[i]).collect();
        let mean: Vec<f64> = (0..width).map(|i| mon.mean[i] + dev[i] / kf).collect();
        if self.cfg.scalar {
            let v = factor * (mean[0] - mon.level[0]) / mon.scale[0];
            return (v.abs() / w, 0);
        }
        let d = self.cfg.n_features;
        let (mut best, mut at) = (f64::NAN, 0);
        let mut idx = 0;
        for a in 0..d {
            let m2a = mon.m2[a] + dev[a] * (row[a] - mean[a]);
            for bb in (a + 1)..d {
                let m2b = mon.m2[bb] + dev[bb] * (row[bb] - mean[bb]);
                let c = mon.cross[idx] + dev[a] * (row[bb] - mean[bb]);
                let den = (m2a * m2b).sqrt();
                let rho = if den > 0.0 { c / den } else { f64::NAN };
                let ratio = (factor * (rho - mon.level[idx]) / mon.scale[idx]).abs() / w;
                if ratio.is_finite() && (best.is_nan() || ratio > best) {
                    (best, at) = (ratio, idx);
                }
                idx += 1;
            }
        }
        (best, at)
    }

    /// W&G's Eq. 8: on a flag at the `τ`-th monitored row, the change is
    /// dated to `k̂ = argmax_{j ≤ τ−1} j·|ρ̂_j − ρ̂_{τ−1}|`, `ρ̂_j` the
    /// correlation of the first `j` monitored rows (for `scalar`, the mean
    /// of `u`), over the rows before the flag -- the history does not enter,
    /// which they found distorted the estimate -- and the positive factor
    /// `D̂/√τ` left out of an argmax. The output is `τ − k̂`, the rows from
    /// the first changed one through the flag's; NaN below two rows.
    fn sequential_since_change(&self, mon: &Monitoring, at: usize) -> f64 {
        let n = mon.rows.len();
        let prefix: Vec<f64> = if self.cfg.scalar {
            let mut run = 0.0;
            mon.rows
                .iter()
                .enumerate()
                .map(|(j, r)| {
                    run += r[0];
                    run / (j + 1) as f64
                })
                .collect()
        } else {
            let (a, b) = Self::pair_at(self.cfg.n_features, at);
            let (mut ma, mut mb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            mon.rows
                .iter()
                .enumerate()
                .map(|(j, r)| {
                    let jf = (j + 1) as f64;
                    let (da, db) = (r[a] - ma, r[b] - mb);
                    ma += da / jf;
                    mb += db / jf;
                    saa += da * (r[a] - ma);
                    sbb += db * (r[b] - mb);
                    sab += da * (r[b] - mb);
                    let den = (saa * sbb).sqrt();
                    if den > 0.0 { sab / den } else { f64::NAN }
                })
                .collect()
        };
        let Some(&last) = prefix.last() else {
            return f64::NAN;
        };
        let (mut best, mut k_hat) = (f64::NEG_INFINITY, None);
        for (j, p) in prefix.iter().enumerate() {
            let score = (j + 1) as f64 * (p - last).abs();
            if score.is_finite() && score > best {
                (best, k_hat) = (score, Some(j + 1));
            }
        }
        match (last.is_finite(), k_hat) {
            (true, Some(k)) => (n + 1 - k) as f64,
            _ => f64::NAN,
        }
    }

    /// `sequential`: a learned row enters the history, or the monitoring
    /// period. A full history starts the monitoring; a flag or the
    /// period's last row ends it, and the next learned row starts a history.
    fn advance_sequential(&mut self, row: Vec<f64>, flagged: bool) {
        let (limit, scalar) = (self.cfg.monitor_rows, self.cfg.scalar);
        let Some(mon) = self.monitoring.as_mut() else {
            self.ring.push_back(row);
            if self.ring.len() >= self.cfg.span_rows {
                let history: Vec<Vec<f64>> = self.ring.drain(..).collect();
                self.monitoring = Some(self.begin_monitoring(&history));
            }
            return;
        };
        // The same arithmetic as `sequential_stat`'s step, now kept.
        let kf = (mon.rows.len() + 1) as f64;
        let width = mon.mean.len();
        let dev: Vec<f64> = (0..width).map(|i| row[i] - mon.mean[i]).collect();
        for i in 0..width {
            mon.mean[i] += dev[i] / kf;
            mon.m2[i] += dev[i] * (row[i] - mon.mean[i]);
        }
        if !scalar {
            // The pairs in `level`'s order: `a` outer, `b > a` inner.
            let Monitoring { cross, mean, .. } = &mut *mon;
            let mut cross = cross.iter_mut();
            for (a, &da) in dev.iter().enumerate() {
                for (xb, mb) in row[a + 1..].iter().zip(&mean[a + 1..]) {
                    *cross.next().expect("one co-moment a pair") += da * (xb - mb);
                }
            }
        }
        mon.rows.push(row);
        if flagged || mon.rows.len() >= limit {
            self.monitoring = None;
        }
    }

    /// The row standardised by the diag, for `scalar`.
    fn scalarise(&self, x: &[f64]) -> f64 {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for (i, xi) in x.iter().enumerate() {
            let v = self.diag.var(i);
            if v.is_nan() || v <= 0.0 {
                return f64::NAN;
            }
            let r = self.diag.deviation(i, *xi) / v.sqrt();
            s1 += r;
            s2 += r * r;
        }
        let n = x.len() as f64;
        if s2 > 0.0 {
            (s1 * s1 - s2) / ((n - 1.0) * s2)
        } else {
            f64::NAN
        }
    }

    /// `‖vech(R̂_pre − R̂_post)‖` over the two windows in `rows`.
    fn window_stat(&self, rows: &[Vec<f64>]) -> f64 {
        let w = self.cfg.span_rows;
        if rows.len() < 2 * w {
            return f64::NAN;
        }
        let (pre, post) = rows.split_at(w);
        let d = self.cfg.n_features;
        let mut acc = 0.0f64;
        for a in 0..d {
            for b in (a + 1)..d {
                let x = Self::corr_of(pre, w, a, b);
                let y = Self::corr_of(post, w, a, b);
                if !(x.is_finite() && y.is_finite()) {
                    return f64::NAN;
                }
                let diff = (x - y).abs();
                match self.cfg.norm {
                    ChangeNorm::L1 => acc += diff,
                    ChangeNorm::LInf => acc = acc.max(diff),
                }
            }
        }
        acc
    }

    /// The `(1 − alpha)` quantile of the statistic under a permutation of
    /// the pooled rows between the two windows, in blocks.
    fn permutation_crit(&mut self) -> Option<f64> {
        let w = self.cfg.span_rows;
        if self.ring.len() < 2 * w {
            return None;
        }
        let rows: Vec<Vec<f64>> = self.ring.iter().cloned().collect();
        let block = self.cfg.perm_block;
        let nblocks = (2 * w).div_ceil(block);
        let mut stats = Vec::with_capacity(self.cfg.n_perm);
        let mut order: Vec<usize> = (0..nblocks).collect();
        for _ in 0..self.cfg.n_perm {
            // Fisher-Yates over the blocks, from the state's own generator
            // so the draws are a function of the data alone.
            for i in (1..nblocks).rev() {
                let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let mut shuffled: Vec<Vec<f64>> = Vec::with_capacity(2 * w);
            for &bi in &order {
                let start = bi * block;
                for r in rows.iter().take((start + block).min(2 * w)).skip(start) {
                    shuffled.push(r.clone());
                }
            }
            let s = self.window_stat(&shuffled);
            if s.is_finite() {
                stats.push(s);
            }
        }
        if stats.is_empty() {
            return None;
        }
        stats.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let q = ((1.0 - self.cfg.alpha) * stats.len() as f64).ceil() as usize;
        Some(stats[q.min(stats.len()) - 1])
    }

    /// The row as the ring would hold it, or `None` when the standardiser
    /// cannot yet make one (`scalar` before the first variances).
    fn ring_row(&self, x: &[f64]) -> Option<Vec<f64>> {
        if !self.cfg.scalar {
            return Some(x.to_vec());
        }
        let u = self.scalarise(x);
        // A one-column ring: the CUSUM is over `u` itself.
        u.is_finite().then(|| vec![u])
    }

    /// What this row produces, from the state as it stands: nothing except
    /// where a statistic is due.
    ///
    /// A pure function of the state and the row, so `step` and `predict`
    /// share it -- which is what makes `predict` the step's answer without
    /// the step even on the row that closes a span.
    /// A row's report: the statistic for the span **ending at this row**,
    /// the critical value in force, the flag and the rows since the last
    /// one. The row itself is always part of the span reported here -- it
    /// is reported before the update, which is what makes the flag out of
    /// sample -- so a row that will not be learned (`weight = 0`) is
    /// reported as if it would be, and then does not enter the ring, does
    /// not advance `since` for the rows after it and does not reset it if
    /// it flags. That is the reading `predict` has to give too, since
    /// `predict` does not know the weight (docs/REVIEW-E54-E64.md CC2).
    fn read(&self, x: &[f64]) -> Vec<f64> {
        let mut pred = vec![f64::NAN; Self::n_outputs_for()];
        let Some(row) = self.ring_row(x) else {
            return pred;
        };
        let since = self.since_flag.unwrap_or(0) + 1;
        match self.cfg.kind {
            CorrChangeKind::Monitor => {
                if self.ring.len() + 1 < self.cfg.span_rows {
                    return pred;
                }
                let mut rows: Vec<Vec<f64>> = self.ring.iter().cloned().collect();
                rows.push(row);
                let (mut stat, mut at) = (f64::NAN, 0);
                if self.cfg.scalar {
                    (stat, at) = self.scalar_stat_at(&rows);
                } else {
                    for a in 0..self.cfg.n_features {
                        for b in (a + 1)..self.cfg.n_features {
                            let (q, j) = self.monitor_stat_at(&rows, a, b);
                            if q.is_finite() && (stat.is_nan() || q > stat) {
                                (stat, at) = (q, j);
                            }
                        }
                    }
                }
                let crit = self.cfg.fixed_crit().unwrap_or(f64::NAN);
                let flag = stat.is_finite() && crit.is_finite() && stat > crit;
                pred[0] = stat;
                pred[1] = crit;
                pred[2] = f64::from(flag);
                pred[3] = since as f64;
                if flag && at > 0 {
                    // The rows after the CUSUM's argmax, through this one.
                    pred[4] = (rows.len() - at) as f64;
                }
            }
            CorrChangeKind::Window => {
                let w = self.cfg.span_rows;
                let mut rows: Vec<Vec<f64>> = self.ring.iter().cloned().collect();
                rows.push(row);
                while rows.len() > 2 * w {
                    rows.remove(0);
                }
                if rows.len() < 2 * w {
                    return pred;
                }
                let stat = self.window_stat(&rows);
                // The value **in force**: a redraw happens after the row is
                // reported, so `predict` and `step` see the same one.
                let crit = self.cfg.crit.or(self.perm_crit);
                let flag = matches!((stat.is_finite(), crit), (true, Some(c)) if stat > c);
                pred[0] = stat;
                pred[1] = crit.unwrap_or(f64::NAN);
                pred[2] = f64::from(flag);
                pred[3] = since as f64;
                if flag {
                    // The second window, by construction.
                    pred[4] = w as f64;
                }
            }
            CorrChangeKind::Sequential => {
                let Some(mon) = &self.monitoring else {
                    return pred;
                };
                if mon.rows.is_empty() {
                    return pred;
                }
                let (stat, at) = self.sequential_stat(mon, &row);
                let crit = self.seq_crit;
                let flag = stat.is_finite() && crit.is_finite() && stat > crit;
                pred[0] = stat;
                pred[1] = crit;
                pred[2] = f64::from(flag);
                pred[3] = since as f64;
                if flag {
                    pred[4] = self.sequential_since_change(mon, at);
                }
            }
        }
        pred
    }
}

impl crate::OnlineModel for CorrChange {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a row with a weight was counted in `n_eff`, and
        // fed to the scalar statistic's moments, whatever its features
        // held (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        if weight <= 0.0 {
            // Hard rule 8: `n_eff` is the accumulated weight before the
            // row's update and before its own decay, and a zero-weight row
            // still decays it (docs/REVIEW-E54-E64.md H3).
            self.n_eff *= lam;
            if self.cfg.scalar {
                self.diag.update(x, lam, 0.0);
            }
            return out;
        }
        let row = self.ring_row(x);
        if self.cfg.scalar {
            self.diag.update(x, lam, weight);
        }
        self.n_eff = lam * self.n_eff + weight;
        let Some(row) = row else { return out };
        self.since_flag = Some(self.since_flag.unwrap_or(0) + 1);
        let flagged = out.pred[2] == 1.0;
        if flagged {
            self.since_flag = Some(0);
        }
        match self.cfg.kind {
            CorrChangeKind::Sequential => self.advance_sequential(row, flagged),
            CorrChangeKind::Monitor => {
                self.ring.push_back(row);
                if self.ring.len() >= self.cfg.span_rows {
                    // Spans are disjoint by construction: the next row
                    // starts a new one, so `reset` has nothing to add here.
                    self.ring.clear();
                }
            }
            CorrChangeKind::Window => {
                self.ring.push_back(row);
                let w = self.cfg.span_rows;
                while self.ring.len() > 2 * w {
                    self.ring.pop_front();
                }
                if flagged && self.cfg.reset {
                    self.ring.clear();
                    self.perm_crit = None;
                    self.since_perm = 0;
                } else if self.cfg.crit.is_none() && self.ring.len() >= 2 * w {
                    // The critical value is refreshed **after** the row is
                    // reported, so `predict` and `step` see the one in
                    // force and cannot disagree.
                    if self.perm_crit.is_none() || self.since_perm >= self.cfg.permute_every {
                        self.perm_crit = self.permutation_crit();
                        self.since_perm = 0;
                    } else {
                        self.since_perm += 1;
                    }
                }
            }
        }
        out
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        crate::Step {
            pred: self.read(x),
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn clear_lags(&mut self) {
        // The rings pair rows the clock says are no longer adjacent; a
        // `monitor` span that spans a break is not a span.
        self.ring.clear();
        self.perm_crit = None;
        self.since_perm = 0;
        // And a `sequential` cycle across a break is not one cycle.
        self.monitoring = None;
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::CorrChange(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::CorrChange(m) => {
                let mut m = (**m).clone();
                // First, so nothing below is read from a cfg `new` refuses:
                // the sequential critical value is solved from it.
                crate::model::check_cfg("corrchange", m.cfg.validate())?;
                // The diagnostics and every ring row at the cfg's width
                // (review 2026-09-18, B3); a `scalar` ring holds `u`.
                let d = m.cfg.n_features;
                let width = m.cfg.width();
                let wrong_mon = m.monitoring.as_ref().is_some_and(|mon| {
                    let values = if m.cfg.scalar { 1 } else { m.cfg.npairs() };
                    let pairs = if m.cfg.scalar { 0 } else { values };
                    mon.level.len() != values
                        || mon.scale.len() != values
                        || mon.mean.len() != width
                        || mon.m2.len() != width
                        || mon.cross.len() != pairs
                        || mon.rows.iter().any(|r| r.len() != width)
                });
                if m.diag.k() != d || m.ring.iter().any(|r| r.len() != width) || wrong_mon {
                    return Err(crate::StateError::Invalid(
                        "corrchange: the state has the wrong shape".into(),
                    ));
                }
                // Configuration, not state: recomputed on load.
                m.seq_crit = Self::sequential_crit_of(&m.cfg);
                Ok(m)
            }
            other => Err(crate::StateError::WrongModel {
                expected: "corrchange",
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
        Self::n_outputs_for()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ring row narrower than the cfg is refused, where it loaded and
    /// panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = CorrChange::new(cfg(2, CorrChangeKind::Monitor)).unwrap();
        let mut s = m.state();
        let ModelState::CorrChange(inner) = &mut s.model else {
            unreachable!()
        };
        inner.ring.push_back(vec![0.0]);
        match CorrChange::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    /// A state's configuration is held to what `new` holds a fresh one to:
    /// a `window` state whose `perm_block` is 0 passed the shape checks,
    /// which read the ring and the monitoring period alone, and the first
    /// row that filled both windows divided by it in `permutation_crit`
    /// (review 2026-10-06, CD14). The cfg is checked before the sequential
    /// critical value is solved from it, which a damaged `boundary_gamma`
    /// would have kept busy for hours.
    #[test]
    fn a_restored_corrchange_whose_cfg_new_refuses_is_refused() {
        use crate::{ModelState, StateError};
        let m = CorrChange::new(CorrChangeCfg {
            span_rows: 5,
            n_perm: 20,
            ..cfg(2, CorrChangeKind::Window)
        })
        .unwrap();
        let mut v = serde_json::to_value(&m).unwrap();
        v["cfg"]["perm_block"] = serde_json::json!(0);
        let edited: CorrChange = serde_json::from_value(v).unwrap();
        assert!(
            edited.cfg.validate().is_err(),
            "`new` refuses perm_block = 0"
        );
        let s = crate::State::new(ModelState::CorrChange(Box::new(edited)));
        match CorrChange::restore(&s) {
            Err(StateError::Invalid(e)) => {
                assert!(
                    e.contains("configuration") && e.contains("perm_block"),
                    "{e}"
                );
            }
            Ok(mut back) => {
                let mut n = Normals::new(7);
                for _ in 0..10 {
                    back.step(&n.pair(0.3), &[], 1.0, 1.0);
                }
                panic!("perm_block = 0 loaded and ran");
            }
            Err(e) => panic!("{e}"),
        }
    }
    use crate::OnlineModel;

    fn cfg(d: usize, kind: CorrChangeKind) -> CorrChangeCfg {
        CorrChangeCfg {
            n_features: d,
            kind,
            span_rows: 200,
            alpha: 0.05,
            alpha_adjust: "bonferroni".into(),
            bandwidth: None,
            scalar: false,
            decay: Decay::Halflife(f64::INFINITY),
            crit: None,
            n_perm: 50,
            permute_every: 25,
            perm_block: 1,
            norm: ChangeNorm::L1,
            seed: 3,
            reset: false,
            monitor_rows: 0,
            boundary_gamma: 0.0,
        }
    }

    /// A `sequential` cfg: `m` rows of history, `k` monitored.
    fn seq_cfg(d: usize, m: usize, k: usize, gamma: f64) -> CorrChangeCfg {
        CorrChangeCfg {
            span_rows: m,
            monitor_rows: k,
            boundary_gamma: gamma,
            ..cfg(d, CorrChangeKind::Sequential)
        }
    }

    /// `d` columns with correlation `rho` between every two.
    fn equicorrelated(n: &mut Normals, d: usize, rho: f64) -> Vec<f64> {
        let f = n.normal();
        (0..d)
            .map(|_| rho.sqrt() * f + (1.0 - rho).sqrt() * n.normal())
            .collect()
    }

    fn same(a: f64, b: f64) -> bool {
        a == b || (a.is_nan() && b.is_nan())
    }

    /// A correlated Gaussian pair, by Box-Muller from a `SplitMix64`.
    struct Normals(SplitMix64);

    impl Normals {
        fn new(seed: u64) -> Self {
            Self(SplitMix64::new(seed))
        }

        fn unit(&mut self) -> f64 {
            // `(0, 1)`, never exactly 0, so `ln` is finite.
            ((self.0.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u, v) = (self.unit(), self.unit());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }

        fn pair(&mut self, rho: f64) -> Vec<f64> {
            let a = self.normal();
            let b = rho * a + (1.0 - rho * rho).sqrt() * self.normal();
            vec![a, b]
        }
    }

    /// Under `"sequential"` an `alpha` whose share per pair is below
    /// `2^-52` is refused, by name: `1 − alpha/npairs` rounds to 1 there,
    /// the boundary's quantile of 1 is NaN, and the detector ran, reported
    /// its statistic and could never flag, saying nothing (review round 4,
    /// CD12). A share at the limit runs, and its critical value is a
    /// number. A configured `crit` reads no `alpha`, and `"monitor"`'s
    /// quantile saturates, so neither refuses one.
    #[test]
    fn an_alpha_too_small_for_its_quantile_is_refused_under_sequential() {
        let limit = 2f64.powi(-52);
        // Three columns: three pairs under Bonferroni.
        let with = |alpha: f64, adjust: &str| CorrChangeCfg {
            alpha,
            alpha_adjust: adjust.into(),
            ..seq_cfg(3, 50, 50, 0.0)
        };
        for (alpha, adjust) in [
            (1e-17, "bonferroni"),
            (2.9 * limit, "bonferroni"),
            (1e-17, "none"),
            (0.9 * limit, "none"),
        ] {
            let err = with(alpha, adjust).validate().unwrap_err();
            assert!(
                err.contains("alpha") && err.contains("2^-52"),
                "{alpha:e} {adjust}: {err}"
            );
        }
        for (alpha, adjust) in [
            (3.0 * limit, "bonferroni"),
            (limit, "none"),
            (0.05, "bonferroni"),
        ] {
            let c = with(alpha, adjust);
            c.validate().unwrap();
            let crit = c.fixed_crit().expect("sequential has one");
            assert!(crit.is_finite(), "{alpha:e} {adjust}: {crit}");
        }
        CorrChangeCfg {
            crit: Some(2.0),
            ..with(1e-300, "none")
        }
        .validate()
        .unwrap();
        CorrChangeCfg {
            alpha: 1e-300,
            ..cfg(3, CorrChangeKind::Monitor)
        }
        .validate()
        .unwrap();
    }

    /// The Kolmogorov quantiles the paper's tables are read at.
    #[test]
    fn the_kolmogorov_quantiles_are_the_published_ones() {
        assert!((kolmogorov_quantile(0.05) - 1.3581).abs() < 1e-3);
        assert!((kolmogorov_quantile(0.01) - 1.6276).abs() < 1e-3);
        assert!((kolmogorov_quantile(0.10) - 1.2239).abs() < 1e-3);
        // And the series is a distribution function.
        assert!(kolmogorov_cdf(0.0) == 0.0);
        assert!(kolmogorov_cdf(10.0) == 1.0);
        for x in [0.5, 1.0, 1.5, 2.0] {
            assert!(kolmogorov_cdf(x) < kolmogorov_cdf(x + 0.1));
        }
        assert!(kolmogorov_quantile(0.0).is_nan() && kolmogorov_quantile(1.0).is_nan());
    }

    /// A correlation is free of its columns' units, so `D̂` -- the delta
    /// method's long-run standard deviation of `ρ̂` -- must be too: the same
    /// rows with one column multiplied by 100, or the other by 0.01, give the
    /// same number. The gradient carried the wrong powers of `σ_x` and `σ_y`,
    /// which cancelled only at unit variance, so on any other scale `D̂` was
    /// off by up to `σ²` and the monitor flagged nothing or everything
    /// (review 2026-09-18, S3).
    #[test]
    fn the_long_run_sd_is_free_of_the_columns_units() {
        let t = 80usize;
        let mut n = Normals::new(17);
        let rows: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.4)).collect();
        let gamma = ((t as f64).ln().floor() as usize).max(1);
        let base = CorrChange::long_run_sd(&rows, 0, 1, gamma);
        assert!(base.is_finite() && base > 0.0, "{base}");
        for (cx, cy) in [(1.0, 100.0), (0.01, 1.0), (3.0, 0.2)] {
            let scaled: Vec<Vec<f64>> = rows.iter().map(|r| vec![cx * r[0], cy * r[1]]).collect();
            let got = CorrChange::long_run_sd(&scaled, 0, 1, gamma);
            assert!(
                (got - base).abs() <= 1e-9 * base,
                "x × {cx}, y × {cy}: {got} vs {base}"
            );
        }
        // And against a central difference of `ρ` in each moment, which is
        // what the gradient claims to be.
        let rho = |sx2: f64, sy2: f64, sxy: f64| sxy / (sx2 * sy2).sqrt();
        let (sx2, sy2, sxy): (f64, f64, f64) = (2.0, 0.5, 0.4);
        let (sx, sy) = (sx2.sqrt(), sy2.sqrt());
        let d3 = [
            -0.5 * sxy / (sx * sx * sx * sy),
            -0.5 * sxy / (sx * sy * sy * sy),
            1.0 / (sx * sy),
        ];
        let h = 1e-6;
        let fd = [
            (rho(sx2 + h, sy2, sxy) - rho(sx2 - h, sy2, sxy)) / (2.0 * h),
            (rho(sx2, sy2 + h, sxy) - rho(sx2, sy2 - h, sxy)) / (2.0 * h),
            (rho(sx2, sy2, sxy + h) - rho(sx2, sy2, sxy - h)) / (2.0 * h),
        ];
        for i in 0..3 {
            assert!(
                (d3[i] - fd[i]).abs() < 1e-7,
                "entry {i}: {} vs {}",
                d3[i],
                fd[i]
            );
        }
    }

    /// `D̂` does not depend on the level the columns sit at: `ρ̂` is
    /// shift-invariant and so is the delta-method variance of it (the
    /// shifted moments are an affine image of the raw ones, and the
    /// Jacobians cancel). The deviations are put on a grid so that adding a
    /// level is exact and every level sees the same rows: 2⁻²⁰ is exact
    /// below 2³³ (8.6e9), 2⁻¹² below 2⁴¹. Formed from `E[x²] − E[x]²` this
    /// failed at every level (docs/PLAN.md task 103); computed on the span
    /// centred at a compensated mean it is the same double at every level,
    /// held here to 1e-14. The span is 79 rows, a prime, so that neither
    /// mean's division lands on the level's grid by chance (one in five did
    /// at 80, and the x side's second pass went untested), and the pair is
    /// taken both ways round, so each column is `x` once.
    #[test]
    fn the_long_run_sd_is_free_of_the_columns_level() {
        let t = 79usize;
        let mut n = Normals::new(17);
        let raw: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.4)).collect();
        let gamma = ((t as f64).ln().floor() as usize).max(1);
        let mut off = Vec::new();
        for (level, bits) in [(1e3, 20), (1e5, 20), (1e8, 20), (4e9, 20), (1e12, 12)] {
            let grid = (1u64 << bits) as f64;
            let rows: Vec<Vec<f64>> = raw
                .iter()
                .map(|r| r.iter().map(|v| (v * grid).round() / grid).collect())
                .collect();
            let base = CorrChange::long_run_sd(&rows, 0, 1, gamma);
            assert!(base.is_finite() && base > 0.0, "{base}");
            let shifted: Vec<Vec<f64>> = rows
                .iter()
                .map(|r| vec![level + r[0], level + r[1]])
                .collect();
            for (s, r) in shifted.iter().zip(&rows) {
                assert_eq!(s[0] - level, r[0], "the shift is exact at {level:e}");
                assert_eq!(s[1] - level, r[1], "the shift is exact at {level:e}");
            }
            for (a, b) in [(0, 1), (1, 0)] {
                let got = CorrChange::long_run_sd(&shifted, a, b, gamma);
                let off_by = (got - base).abs();
                if off_by.is_nan() || off_by > 1e-14 * base {
                    off.push(format!(
                        "at {level:e}, pair ({a}, {b}): {got} against {base} ({:e} of it)",
                        off_by / base
                    ));
                }
            }
        }
        assert!(off.is_empty(), "{}", off.join("; "));
    }

    /// The statistic itself at a level: `Q` reads `corr_of` too, whose
    /// one-pass mean is second order in the level's rounding.
    #[test]
    fn the_statistic_is_free_of_the_columns_level() {
        let t = 60usize;
        let mut n = Normals::new(11);
        let grid = (1u64 << 20) as f64;
        let rows: Vec<Vec<f64>> = (0..t)
            .map(|_| {
                n.pair(0.4)
                    .iter()
                    .map(|v| (v * grid).round() / grid)
                    .collect()
            })
            .collect();
        let m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let base = m.monitor_stat(&rows, 0, 1);
        assert!(base.is_finite() && base > 0.0, "{base}");
        for level in [1e5, 1e8] {
            let shifted: Vec<Vec<f64>> = rows
                .iter()
                .map(|r| vec![level + r[0], level + r[1]])
                .collect();
            let got = m.monitor_stat(&shifted, 0, 1);
            assert!(
                (got - base).abs() <= 1e-10 * base,
                "at {level:e}: {got} against {base}"
            );
        }
    }

    /// Too few rows for a long-run variance: three give none, four give one.
    #[test]
    fn a_span_of_four_rows_is_the_first_with_a_long_run_sd() {
        let mut n = Normals::new(5);
        let rows: Vec<Vec<f64>> = (0..4).map(|_| n.pair(0.3)).collect();
        assert!(CorrChange::long_run_sd(&rows[..3], 0, 1, 1).is_nan());
        let four = CorrChange::long_run_sd(&rows, 0, 1, 1);
        assert!(four.is_finite() && four > 0.0, "{four}");
    }

    /// `bandwidth` overrides `⌊ln T⌋`: the statistic is the one `long_run_sd`
    /// gives at that bandwidth, and two bandwidths give two statistics.
    ///
    /// Over thirty streams, because the one that stood here alone passed on
    /// macOS and failed on Linux and Windows (found by CI on the 0.11.0 tag):
    /// at `usize::MAX` every Bartlett weight is about 1, so the long-run
    /// variance is about `(Σξ)²/T`, which is 0 but for rounding -- `ξ` has
    /// span mean 0 by construction -- and its sign is that rounding's. The
    /// streams come from Box–Muller, whose `ln` and `cos` differ in their last
    /// bits between Apple's libm and the others, so a variance that rounded
    /// positive here rounded to 0 or below there, where the model says NaN,
    /// no verdict. The longhand folded its NaNs away with `f64::max` and said
    /// 0.0. It takes the model's rule now (a long-run sd that is NaN or not
    /// positive is no statistic), and NaN equals NaN.
    #[test]
    fn the_bandwidth_override_reaches_the_kernel() {
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        let t = 60usize;
        for seed in 11..=40 {
            let mut n = Normals::new(seed);
            let rows: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.4)).collect();
            let stat_at = |bandwidth: usize| {
                let m = CorrChange::new(CorrChangeCfg {
                    span_rows: t,
                    bandwidth: Some(bandwidth),
                    ..cfg(2, CorrChangeKind::Monitor)
                })
                .unwrap();
                m.monitor_stat(&rows, 0, 1)
            };
            let longhand = |gamma: usize| {
                let sd = CorrChange::long_run_sd(&rows, 0, 1, gamma);
                if sd.is_nan() || sd <= 0.0 {
                    return f64::NAN;
                }
                let rho_t = CorrChange::corr_of(&rows, t, 0, 1);
                (2..=t)
                    .map(|j| {
                        (j as f64 / (t as f64).sqrt())
                            * (CorrChange::corr_of(&rows, j, 0, 1) - rho_t).abs()
                            / sd
                    })
                    .fold(0.0f64, f64::max)
            };
            for gamma in [1, 20, usize::MAX] {
                let (got, want) = (stat_at(gamma), longhand(gamma));
                assert!(
                    same(got, want),
                    "seed {seed}, bandwidth {gamma}: {got} vs {want}"
                );
            }
            // At the two bandwidths a span of 60 can support, a statistic,
            // and not the same one.
            assert!(
                stat_at(1).is_finite() && stat_at(20).is_finite(),
                "seed {seed}"
            );
            assert_ne!(stat_at(1), stat_at(20), "seed {seed}");
            // A bandwidth past the span is the kernel's answer, not an
            // overflow (`gamma.saturating_add(1)`): a number or no verdict.
            assert!(!stat_at(usize::MAX).is_infinite(), "seed {seed}");
        }
    }

    /// A pair within rounding of `|ρ̂| = 1` has no verdict: the delta
    /// method's variance is 0 on a collinear pair, and this `D̂` reaches it,
    /// where dividing the numerator's rounding by it flagged every span of a
    /// derived column (review 2026-09-25). A pair off collinear by 1e-6 of
    /// the spread still has its verdict.
    #[test]
    fn a_collinear_pair_has_no_verdict() {
        let t = 79usize;
        let mut n = Normals::new(23);
        let x: Vec<f64> = (0..t).map(|_| n.pair(0.0)[0]).collect();
        let m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let gamma = ((t as f64).ln().floor() as usize).max(1);
        for (slope, offset) in [
            (2.0, 3.0),
            (1.8, 32.0),
            (-2.5, 1e3),
            (1.0, 0.0),
            (-1.0, 0.0),
        ] {
            let rows: Vec<Vec<f64>> = x.iter().map(|&v| vec![v, slope * v + offset]).collect();
            for (a, b) in [(0, 1), (1, 0)] {
                assert!(
                    CorrChange::long_run_sd(&rows, a, b, gamma).is_nan(),
                    "y = {slope}x + {offset}, pair ({a}, {b})"
                );
            }
            assert!(
                m.monitor_stat(&rows, 0, 1).is_nan(),
                "y = {slope}x + {offset}"
            );
        }
        let mut z = Normals::new(29);
        let rows: Vec<Vec<f64>> = x
            .iter()
            .map(|&v| vec![v, v + 1e-6 * z.pair(0.0)[0]])
            .collect();
        let q = m.monitor_stat(&rows, 0, 1);
        assert!(
            q.is_finite() && q < 3.0,
            "a verdict off collinear by 1e-6: {q}"
        );
    }

    /// A column without spread has no correlation to test: `D̂` is NaN, and
    /// so is the statistic, which the bank reads as "no verdict". The
    /// constant is held at values whose mean the first pass rounds (1/3, a
    /// level of 1e8), where the raw-moment form gave a finite `D̂` from the
    /// rounding alone; centred at the compensated mean every deviation is
    /// exactly 0.
    #[test]
    fn a_constant_column_has_no_long_run_sd() {
        let t = 40usize;
        let mut n = Normals::new(5);
        let rows: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.3)).collect();
        let gamma = ((t as f64).ln().floor() as usize).max(1);
        assert!(CorrChange::long_run_sd(&rows, 0, 1, gamma).is_finite());
        for c in [2.5, 1.0 / 3.0, 1e8 + 0.1] {
            let mut held = rows.clone();
            for r in &mut held {
                r[0] = c;
            }
            assert!(CorrChange::long_run_sd(&held, 0, 1, gamma).is_nan(), "{c}");
            assert!(CorrChange::long_run_sd(&held, 1, 0, gamma).is_nan(), "{c}");
            let m = CorrChange::new(CorrChangeCfg {
                span_rows: t,
                ..cfg(2, CorrChangeKind::Monitor)
            })
            .unwrap();
            assert!(m.monitor_stat(&held, 0, 1).is_nan(), "{c}");
        }
    }

    /// `scalar = true`'s statistic against its definition written out: the
    /// Bartlett long-run variance of the one column, the lag-0 term once and
    /// every other lag twice at weight `k(l/γ) = 1 − l/γ`, WKD's Appendix
    /// A.1 (lag `γ` at 0; docs/PLAN.md task 114), then the CUSUM of its
    /// running mean; at a bandwidth given and at the default `⌊ln T⌋`.
    #[test]
    fn the_scalar_statistic_is_its_definition() {
        let t = 60usize;
        let mut n = Normals::new(13);
        let rows: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.0)).collect();
        for bandwidth in [Some(3), None] {
            let m = CorrChange::new(CorrChangeCfg {
                span_rows: t,
                scalar: true,
                bandwidth,
                ..cfg(2, CorrChangeKind::Monitor)
            })
            .unwrap();
            let gamma = bandwidth.unwrap_or(((t as f64).ln().floor() as usize).max(1));
            let tf = t as f64;
            let u: Vec<f64> = rows.iter().map(|r| r[0]).collect();
            let mean = u.iter().sum::<f64>() / tf;
            let v: Vec<f64> = u.iter().map(|x| x - mean).collect();
            let mut var = 0.0;
            for lag in 0..=gamma.min(t - 1) {
                let w = 1.0 - lag as f64 / gamma as f64;
                let mut acf = 0.0;
                for s in lag..t {
                    acf += w * v[s] * v[s - lag] / tf;
                }
                var += if lag == 0 { acf } else { 2.0 * acf };
            }
            let sd = var.sqrt();
            let mut want = 0.0f64;
            for j in 1..=t {
                let mean_j = u[..j].iter().sum::<f64>() / j as f64;
                want = want.max((j as f64 / tf.sqrt()) * (mean_j - mean).abs() / sd);
            }
            let got = m.scalar_stat(&rows);
            assert!(
                (got - want).abs() <= 1e-12 * (1.0 + want),
                "bandwidth {bandwidth:?}: {got} against {want}"
            );
        }
    }

    /// `scalarise` against its definition: each feature standardized by the
    /// diag's pair and standard deviation, then the equicorrelation
    /// `(s1² − s2) / ((n − 1) s2)` of the standardized row.
    #[test]
    fn the_scalar_row_is_its_definition() {
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: 20,
            scalar: true,
            ..cfg(3, CorrChangeKind::Monitor)
        })
        .unwrap();
        let mut n = Normals::new(7);
        for _ in 0..30 {
            let (a, b) = (n.pair(0.5)[0], n.pair(0.5)[1]);
            crate::OnlineModel::step(&mut m, &[a, b, a - b], &[], 1.0, 1.0);
        }
        let x = [0.4, -1.1, 0.7];
        let (mut s1, mut s2) = (0.0, 0.0);
        for (i, &xi) in x.iter().enumerate() {
            let r = m.diag.deviation(i, xi) / m.diag.var(i).sqrt();
            s1 += r;
            s2 += r * r;
        }
        let want = (s1 * s1 - s2) / (2.0 * s2);
        assert_eq!(m.scalarise(&x), want);
    }

    /// `D̂` and `Q` against the same arithmetic written out, `D̂₁` as WKD's
    /// Appendix A.1 writes it: `ΣₜΣᵤ k((t−u)/γ_T)VₜVᵤ'`, `k(x) = 1 − |x|`.
    #[test]
    fn the_statistic_is_its_definition() {
        let t = 60usize;
        let mut n = Normals::new(11);
        let rows: Vec<Vec<f64>> = (0..t).map(|_| n.pair(0.4)).collect();
        let gamma = ((t as f64).ln().floor() as usize).max(1);

        // Longhand `D̂`.
        let u: Vec<[f64; 5]> = rows
            .iter()
            .map(|r| [r[0] * r[0], r[1] * r[1], r[0], r[1], r[0] * r[1]])
            .collect();
        let mut mean = [0.0; 5];
        for row in &u {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += row[j] / t as f64;
            }
        }
        let v: Vec<[f64; 5]> = u
            .iter()
            .map(|row| std::array::from_fn(|j| row[j] - mean[j]))
            .collect();
        let mut sigma = [[0.0f64; 5]; 5];
        for s in 0..t {
            for q in 0..t {
                let w = bartlett((s as f64 - q as f64) / gamma as f64);
                if w == 0.0 {
                    continue;
                }
                for i in 0..5 {
                    for j in 0..5 {
                        sigma[i][j] += w * v[s][i] * v[q][j] / t as f64;
                    }
                }
            }
        }
        let (mx, my) = (mean[2], mean[3]);
        let (sx2, sy2) = (mean[0] - mx * mx, mean[1] - my * my);
        let sxy = mean[4] - mx * my;
        let (sx, sy) = (sx2.sqrt(), sy2.sqrt());
        let d3 = [
            -0.5 * sxy / (sx.powi(3) * sy),
            -0.5 * sxy / (sx * sy.powi(3)),
            1.0 / (sx * sy),
        ];
        let d2 = [
            [1.0, 0.0, -2.0 * mx, 0.0, 0.0],
            [0.0, 1.0, 0.0, -2.0 * my, 0.0],
            [0.0, 0.0, -my, -mx, 1.0],
        ];
        let f: [f64; 5] = std::array::from_fn(|j| (0..3).map(|i| d3[i] * d2[i][j]).sum());
        let mut var = 0.0;
        for i in 0..5 {
            for j in 0..5 {
                var += f[i] * sigma[i][j] * f[j];
            }
        }
        let want_sd = var.sqrt();
        let got_sd = CorrChange::long_run_sd(&rows, 0, 1, gamma);
        assert!(
            (got_sd - want_sd).abs() <= 1e-12 * (1.0 + want_sd),
            "{got_sd} vs {want_sd}"
        );

        // Longhand `Q`.
        let corr = |n: usize| {
            let nf = n as f64;
            let (mut sa, mut sb) = (0.0, 0.0);
            for r in &rows[..n] {
                sa += r[0];
                sb += r[1];
            }
            let (ma, mb) = (sa / nf, sb / nf);
            let (mut vaa, mut vbb, mut vab) = (0.0, 0.0, 0.0);
            for r in &rows[..n] {
                vaa += (r[0] - ma) * (r[0] - ma);
                vbb += (r[1] - mb) * (r[1] - mb);
                vab += (r[0] - ma) * (r[1] - mb);
            }
            vab / (vaa * vbb).sqrt()
        };
        let rho_t = corr(t);
        let want_q = (2..=t)
            .map(|j| (j as f64 / (t as f64).sqrt()) * (corr(j) - rho_t).abs() / want_sd)
            .fold(0.0f64, f64::max);
        let m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let got_q = m.monitor_stat(&rows, 0, 1);
        assert!(
            (got_q - want_q).abs() <= 1e-12 * (1.0 + want_q),
            "{got_q} vs {want_q}"
        );
    }

    /// The empirical size against Wied, Krämer & Dehling's Table 1. Theirs
    /// is 5000 replications of i.i.d. bivariate innovations at the 5 %
    /// level and reads `.040 / .035 / .041` at `T = 500` for `ρ = −0.5 / 0
    /// / 0.5`. This runs fewer draws and a Gaussian pair, so the band is
    /// the binomial one around those numbers; **`|ρ| ≤ 0.5` only**, because
    /// the test over-rejects badly at `|ρ| = 0.9` for `T ≤ 500` (`.142` in
    /// their own table) and that is the paper's finding, not a bug here.
    #[test]
    fn the_size_is_the_papers() {
        let t = 500usize;
        let reps = 300usize;
        for (rho, want) in [(-0.5, 0.040), (0.0, 0.035), (0.5, 0.041)] {
            let mut rejects = 0;
            for rep in 0..reps {
                let mut n = Normals::new(1000 + rep as u64);
                let mut m = CorrChange::new(CorrChangeCfg {
                    span_rows: t,
                    alpha_adjust: "none".into(),
                    ..cfg(2, CorrChangeKind::Monitor)
                })
                .unwrap();
                let mut last = f64::NAN;
                for _ in 0..t {
                    last = m.step(&n.pair(rho), &[], 1.0, 1.0).pred[2];
                }
                rejects += usize::from(last == 1.0);
            }
            let size = rejects as f64 / reps as f64;
            // A binomial band at `reps` draws around the paper's number,
            // widened for the smaller sample and the different innovation.
            let se = (want * (1.0 - want) / reps as f64).sqrt();
            assert!(
                (size - want).abs() < 4.0 * se + 0.02,
                "rho {rho}: size {size} against the paper's {want} (se {se:.4})"
            );
        }
    }

    /// Power against their Table 2: a `0.5 → 0.7` break at `T/2` rejects
    /// `.587` of the time at `T = 500` and `.830` at `T = 1000`. The break
    /// is what the test exists to find, so the assertion is one-sided --
    /// power at least the paper's, less the sampling band.
    #[test]
    fn the_power_is_at_least_the_papers() {
        for (t, want) in [(500usize, 0.587), (1000usize, 0.830)] {
            let reps = 200usize;
            let mut rejects = 0;
            for rep in 0..reps {
                let mut n = Normals::new(7000 + rep as u64);
                let mut m = CorrChange::new(CorrChangeCfg {
                    span_rows: t,
                    alpha_adjust: "none".into(),
                    ..cfg(2, CorrChangeKind::Monitor)
                })
                .unwrap();
                let mut last = f64::NAN;
                for i in 0..t {
                    let rho = if i < t / 2 { 0.5 } else { 0.7 };
                    last = m.step(&n.pair(rho), &[], 1.0, 1.0).pred[2];
                }
                rejects += usize::from(last == 1.0);
            }
            let power = rejects as f64 / reps as f64;
            let se = (want * (1.0 - want) / reps as f64).sqrt();
            assert!(
                power > want - 4.0 * se - 0.05,
                "T {t}: power {power} against the paper's {want} (se {se:.4})"
            );
        }
    }

    #[test]
    fn nothing_is_reported_except_on_a_spans_last_row() {
        let mut n = Normals::new(5);
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: 40,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        for i in 1..=120 {
            let step = m.step(&n.pair(0.3), &[], 1.0, 1.0);
            let due = i % 40 == 0;
            assert_eq!(step.pred[0].is_finite(), due, "row {i}");
            assert_eq!(step.pred[1].is_finite(), due, "row {i}");
            if due {
                assert_eq!(m.depth(), 0, "the span is closed at its last row");
            }
        }
    }

    #[test]
    fn the_window_statistic_is_the_norm_of_the_difference() {
        let mut n = Normals::new(9);
        let w = 40usize;
        let mut m = CorrChange::new(CorrChangeCfg {
            kind: CorrChangeKind::Window,
            span_rows: w,
            crit: Some(0.5),
            ..cfg(3, CorrChangeKind::Window)
        })
        .unwrap();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut last = f64::NAN;
        for i in 0..2 * w {
            let rho = if i < w { 0.1 } else { 0.9 };
            let mut r = n.pair(rho);
            r.push(n.normal());
            rows.push(r.clone());
            last = m.step(&r, &[], 1.0, 1.0).pred[0];
        }
        // Longhand: the L1 norm over the strict upper triangle.
        let corr = |rs: &[Vec<f64>], a: usize, b: usize| CorrChange::corr_of(rs, rs.len(), a, b);
        let (pre, post) = rows.split_at(w);
        let mut want = 0.0;
        for a in 0..3 {
            for b in (a + 1)..3 {
                want += (corr(pre, a, b) - corr(post, a, b)).abs();
            }
        }
        assert!((last - want).abs() < 1e-12, "{last} vs {want}");
        assert!(last > 0.5, "a 0.1 -> 0.9 break is a big change");
    }

    #[test]
    fn a_sign_flip_null_would_have_no_spread() {
        // The reason the permutation null is a permutation: negating a
        // whole row leaves every correlation exactly where it was.
        let mut n = Normals::new(13);
        let rows: Vec<Vec<f64>> = (0..40).map(|_| n.pair(0.6)).collect();
        let flipped: Vec<Vec<f64>> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i % 3 == 0 {
                    vec![-r[0], -r[1]]
                } else {
                    r.clone()
                }
            })
            .collect();
        let a = CorrChange::corr_of(&rows, rows.len(), 0, 1);
        let b = CorrChange::corr_of(&flipped, flipped.len(), 0, 1);
        // Not exactly equal -- the means move -- but the *statistic* a
        // sign-flip null would produce is the same to three digits, which
        // is no null at all.
        assert!((a - b).abs() < 5e-2, "{a} vs {b}");
    }

    #[test]
    fn the_permutation_critical_value_flags_a_real_break_and_not_noise() {
        let w = 40usize;
        let build = || {
            CorrChange::new(CorrChangeCfg {
                kind: CorrChangeKind::Window,
                span_rows: w,
                crit: None,
                n_perm: 100,
                permute_every: 1000,
                ..cfg(2, CorrChangeKind::Window)
            })
            .unwrap()
        };
        // Stationary: the statistic sits under the permutation quantile.
        let mut n = Normals::new(17);
        let mut m = build();
        let mut flags = 0;
        for _ in 0..400 {
            flags += usize::from(m.step(&n.pair(0.4), &[], 1.0, 1.0).pred[2] == 1.0);
        }
        assert!(flags < 40, "{flags} flags on a stationary stream");
        // A break: it is found.
        let mut n = Normals::new(19);
        let mut m = build();
        let mut found = false;
        for i in 0..400 {
            let rho = if i < 200 { 0.0 } else { 0.9 };
            found |= m.step(&n.pair(rho), &[], 1.0, 1.0).pred[2] == 1.0;
        }
        assert!(found, "a 0 -> 0.9 break went unflagged");
    }

    /// The scalar CUSUM is a **mean** test, not a correlation one: the
    /// equicorrelation is already one number. A break in its level is what
    /// it finds.
    #[test]
    fn the_scalar_form_finds_a_break_in_the_equicorrelation() {
        let run = |rho_a: f64, rho_b: f64, seed: u64| {
            let mut n = Normals::new(seed);
            let mut m = CorrChange::new(CorrChangeCfg {
                // Fewer than the rows fed: the first rows have no
                // standardiser yet and never enter the span.
                span_rows: 300,
                scalar: true,
                alpha_adjust: "none".into(),
                decay: Decay::Halflife(100.0),
                ..cfg(5, CorrChangeKind::Monitor)
            })
            .unwrap();
            let mut out = (f64::NAN, false);
            for i in 0..500 {
                let rho: f64 = if i < 250 { rho_a } else { rho_b };
                let a = n.normal();
                let x: Vec<f64> = (0..5)
                    .map(|_| rho.sqrt() * a + (1.0 - rho).sqrt() * n.normal())
                    .collect();
                let step = m.step(&x, &[], 1.0, 1.0);
                if step.pred[0].is_finite() {
                    out = (step.pred[0], step.pred[2] == 1.0);
                }
            }
            out
        };
        let (steady, flagged) = run(0.4, 0.4, 41);
        assert!(steady.is_finite() && !flagged, "steady stat {steady}");
        let (broken, flagged) = run(0.1, 0.9, 43);
        assert!(
            flagged,
            "a 0.1 -> 0.9 break in the equicorrelation: {broken}"
        );
        assert!(broken > steady);
    }

    #[test]
    fn the_scalar_form_runs_on_the_equicorrelation() {
        let mut n = Normals::new(23);
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: 100,
            scalar: true,
            decay: Decay::Halflife(200.0),
            ..cfg(4, CorrChangeKind::Monitor)
        })
        .unwrap();
        let mut last = f64::NAN;
        for i in 0..400 {
            let rho: f64 = if i < 200 { 0.1 } else { 0.9 };
            let a = n.normal();
            let x: Vec<f64> = (0..4)
                .map(|_| rho.sqrt() * a + (1.0 - rho).sqrt() * n.normal())
                .collect();
            let step = m.step(&x, &[], 1.0, 1.0);
            if step.pred[0].is_finite() {
                last = step.pred[0];
            }
        }
        assert!(last.is_finite() && last >= 0.0, "{last}");
    }

    #[test]
    fn a_zero_weight_row_is_not_a_row_of_the_span() {
        let mut n = Normals::new(29);
        let mut a = CorrChange::new(cfg(2, CorrChangeKind::Monitor)).unwrap();
        let mut b = a.clone();
        for i in 0..300 {
            let r = n.pair(0.3);
            a.step(&r, &[], 1.0, 1.0);
            if i == 150 {
                a.step(&[1e6, -1e6], &[], 1.0, 0.0);
            }
            b.step(&r, &[], 1.0, 1.0);
        }
        assert_eq!(a.depth(), b.depth());
        assert_eq!(a.n_eff(), b.n_eff());
    }

    /// What a zero-weight row *reports*: the span it would close if it were
    /// learned -- the row is always part of its own report, which is what
    /// makes the flag out of sample -- and then it leaves nothing behind.
    /// `since` does not advance across it and a flag on it does not reset
    /// the count (docs/REVIEW-E54-E64.md CC2).
    #[test]
    fn a_zero_weight_row_reports_the_span_it_would_have_closed() {
        let mut n = Normals::new(41);
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: 20,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        // Nineteen rows: the twentieth closes the span, so the row under
        // test is one that actually reports.
        for _ in 0..19 {
            m.step(&n.pair(0.3), &[], 1.0, 1.0);
        }
        let before = (m.depth(), m.since_flag);
        // The same row twice, once weightless and once not: the report is
        // the same both times, and only the second one moves the model.
        let row = n.pair(0.3);
        let zero = m.step(&row, &[], 1.0, 0.0);
        assert_eq!(
            (m.depth(), m.since_flag),
            before,
            "a zero-weight row entered the span"
        );
        assert!(zero.pred[0].is_finite(), "the span still closes here");
        let learned = m.step(&row, &[], 1.0, 1.0);
        for (slot, (a, b)) in zero.pred.iter().zip(&learned.pred).enumerate() {
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "slot {slot}: weightless said {a}, learned said {b}"
            );
        }
        // The learned row closed the span, so the ring starts over; the
        // weightless one before it left no trace at all.
        assert_eq!(m.depth(), 0, "the span did not close on the learned row");
    }

    #[test]
    fn clear_lags_abandons_the_span() {
        let mut n = Normals::new(31);
        let mut m = CorrChange::new(cfg(2, CorrChangeKind::Monitor)).unwrap();
        for _ in 0..50 {
            m.step(&n.pair(0.3), &[], 1.0, 1.0);
        }
        assert_eq!(m.depth(), 50);
        let w = m.n_eff();
        m.clear_lags();
        assert_eq!(m.depth(), 0);
        assert_eq!(m.n_eff(), w, "clear_lags is not a reset");
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut n = Normals::new(37);
        let mut m = CorrChange::new(cfg(2, CorrChangeKind::Monitor)).unwrap();
        for _ in 0..250 {
            let x = n.pair(0.3);
            let want = m.predict(&x, 1.0);
            let before = m.clone();
            let got = m.step(&x, &[], 1.0, 1.0);
            assert_eq!(want.n_eff, before.n_eff());
            // The step's answer without the step, on the span-closing row
            // as much as on any other.
            assert!(
                want.pred
                    .iter()
                    .zip(&got.pred)
                    .all(|(a, b)| a == b || (a.is_nan() && b.is_nan())),
                "{:?} vs {:?}",
                want.pred,
                got.pred
            );
        }
    }

    #[test]
    fn a_bad_configuration_is_refused_by_name() {
        let bad = |c: CorrChangeCfg, msg: &str| {
            let e = CorrChange::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(cfg(1, CorrChangeKind::Monitor), "at least two columns");
        bad(
            CorrChangeCfg {
                span_rows: 4,
                ..cfg(2, CorrChangeKind::Monitor)
            },
            "span_rows of at least 8",
        );
        bad(
            CorrChangeCfg {
                alpha: 1.5,
                ..cfg(2, CorrChangeKind::Monitor)
            },
            "alpha must be strictly between",
        );
        bad(
            CorrChangeCfg {
                alpha_adjust: "nope".into(),
                ..cfg(2, CorrChangeKind::Monitor)
            },
            "unknown alpha_adjust",
        );
        bad(
            CorrChangeCfg {
                kind: CorrChangeKind::Window,
                span_rows: 2,
                ..cfg(2, CorrChangeKind::Window)
            },
            "span_rows of at least 3",
        );
        bad(
            CorrChangeCfg {
                kind: CorrChangeKind::Window,
                n_perm: 2,
                ..cfg(2, CorrChangeKind::Window)
            },
            "n_perm >= 20",
        );
        bad(
            CorrChangeCfg {
                kind: CorrChangeKind::Window,
                scalar: true,
                ..cfg(2, CorrChangeKind::Window)
            },
            "scalar applies to",
        );
        // docs/REVIEW-E54-E64.md CC1 and CC3: the cases `validate` refuses
        // and nothing exercised.
        for c in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
            bad(
                CorrChangeCfg {
                    crit: Some(c),
                    ..cfg(2, CorrChangeKind::Monitor)
                },
                "crit is the critical value",
            );
        }
        for b in [0usize, 51] {
            bad(
                CorrChangeCfg {
                    kind: CorrChangeKind::Window,
                    span_rows: 50,
                    perm_block: b,
                    ..cfg(2, CorrChangeKind::Window)
                },
                "perm_block must be 1..=",
            );
        }
        bad(
            CorrChangeCfg {
                kind: CorrChangeKind::Window,
                permute_every: 0,
                ..cfg(2, CorrChangeKind::Window)
            },
            "permute_every must be >= 1",
        );
        bad(
            CorrChangeCfg {
                bandwidth: Some(0),
                ..cfg(2, CorrChangeKind::Monitor)
            },
            "bandwidth must be >= 1",
        );
    }

    // --- sequential (Wied & Galeano 2013; docs/PLAN.md task 114) ---------

    /// The detector against its definition, written out on the rows
    /// themselves: the history's correlation and long-run sd per pair
    /// (`corr_of`, `long_run_sd` at `⌊ln m⌋`), the monitored rows'
    /// correlation from scratch, `V_k = (k/√m)(ρ̂_k − ρ̂_h)/σ̂`, the ratio to
    /// `w(k/m) = (1 + k/m)((k/m)/(1 + k/m))^γ`, the maximum over the three
    /// pairs, W&G's Eq. 8 on a flag, and the cycle: a history of `m`
    /// learned rows, then at most `monitor_rows`, a new history after a flag
    /// or the last of them. A break in the middle of the stream makes flags.
    #[test]
    fn the_sequential_detector_is_its_definition() {
        let (d, m, limit, gamma) = (3usize, 40usize, 60usize, 0.25);
        let mut model = CorrChange::new(seq_cfg(d, m, limit, gamma)).unwrap();
        let crit = crate::boundary::sequential_crit(0.05 / 3.0, gamma, 1.5);
        let bw = ((m as f64).ln().floor() as usize).max(1);
        let mut n = Normals::new(41);
        let (mut hist, mut mon): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (vec![], vec![]);
        let mut monitoring = false;
        let (mut flags, mut stats) = (0, 0);
        for i in 0..400 {
            let x = equicorrelated(&mut n, d, if i < 200 { 0.2 } else { 0.85 });
            let got = model.step(&x, &[], 1.0, 1.0).pred;
            let mut want = [f64::NAN; 5];
            if monitoring && !mon.is_empty() {
                let mut rows = mon.clone();
                rows.push(x.clone());
                let k = rows.len() as f64;
                let b = k / m as f64;
                let w = (1.0 + b) * (b / (1.0 + b)).powf(gamma);
                let (mut best, mut at) = (f64::NAN, (0, 0));
                for a in 0..d {
                    for bb in (a + 1)..d {
                        let level = CorrChange::corr_of(&hist, m, a, bb);
                        let sd = CorrChange::long_run_sd(&hist, a, bb, bw);
                        let rho = CorrChange::corr_of(&rows, rows.len(), a, bb);
                        let r = ((k / (m as f64).sqrt()) * (rho - level) / sd).abs() / w;
                        if r.is_finite() && (best.is_nan() || r > best) {
                            (best, at) = (r, (a, bb));
                        }
                    }
                }
                want[0] = best;
                want[1] = crit;
                want[2] = f64::from(best > crit);
                if best > crit {
                    let last = CorrChange::corr_of(&mon, mon.len(), at.0, at.1);
                    let (mut top, mut k_hat) = (f64::NEG_INFINITY, 0);
                    for j in 1..=mon.len() {
                        let s = j as f64 * (CorrChange::corr_of(&mon, j, at.0, at.1) - last).abs();
                        if s.is_finite() && s > top {
                            (top, k_hat) = (s, j);
                        }
                    }
                    // No finite score below two rows: no date.
                    want[4] = if k_hat == 0 {
                        f64::NAN
                    } else {
                        (mon.len() + 1 - k_hat) as f64
                    };
                }
            }
            for s in [0, 2, 4] {
                assert!(
                    (got[s] - want[s]).abs() <= 1e-9 * (1.0 + want[s].abs())
                        || same(got[s], want[s]),
                    "row {i}, slot {s}: {} against {}",
                    got[s],
                    want[s]
                );
            }
            assert!(
                same(got[1], want[1]) || (got[1] - want[1]).abs() < 1e-12,
                "row {i}"
            );
            if got[0].is_finite() {
                stats += 1;
            }
            let flagged = got[2] == 1.0;
            flags += usize::from(flagged);
            if !monitoring {
                hist.push(x);
                monitoring = hist.len() == m;
            } else {
                mon.push(x);
                if flagged || mon.len() == limit {
                    (monitoring, hist, mon) = (false, vec![], vec![]);
                }
            }
        }
        assert!(
            flags >= 1 && stats > 100,
            "{flags} flags, {stats} statistics"
        );
    }

    /// A break in the correlation is flagged in the monitoring period, and
    /// dated near where it happened: Eq. 8's estimate over the monitored
    /// rows, `τ − k̂` rows from the first changed one through the flag.
    #[test]
    fn a_break_is_flagged_and_dated() {
        let (m, limit, change) = (200usize, 400usize, 150usize);
        let mut model = CorrChange::new(seq_cfg(2, m, limit, 0.0)).unwrap();
        let mut n = Normals::new(5);
        let mut found = None;
        for i in 0..(m + limit) {
            let rho = if i < m + change { 0.1 } else { 0.8 };
            let out = model.step(&n.pair(rho), &[], 1.0, 1.0).pred;
            if out[2] == 1.0 {
                found = Some((i - m + 1, out[4]));
                break;
            }
        }
        let (tau, since) = found.expect("the break is flagged");
        assert!(
            tau > change,
            "flagged at monitored row {tau}, before the break"
        );
        let dated = tau as f64 - since;
        assert!(
            (dated - change as f64).abs() <= 40.0,
            "dated after monitored row {dated}, the change after {change}"
        );
    }

    /// The cycle: `m` rows of history report nothing, the first monitored
    /// row reports nothing (a correlation needs two rows), rows 2..=limit
    /// report, and the next learned row starts a history. With a critical
    /// value nothing crosses, cycles run back to back; with one everything
    /// crosses, each flag ends its cycle at once.
    #[test]
    fn the_cycle_restarts_after_its_period_and_after_a_flag() {
        let (m, limit) = (10usize, 5usize);
        let pattern = |crit: f64| {
            let mut model = CorrChange::new(CorrChangeCfg {
                crit: Some(crit),
                ..seq_cfg(2, m, limit, 0.0)
            })
            .unwrap();
            let mut n = Normals::new(9);
            (0..40)
                .map(|_| model.step(&n.pair(0.3), &[], 1.0, 1.0).pred[0].is_finite())
                .collect::<Vec<bool>>()
        };
        // 10 history, 1 silent, 4 reporting: a period of 15.
        let never = pattern(1e9);
        for (i, &live) in never.iter().enumerate() {
            let at = i % 15;
            assert_eq!(live, (11..15).contains(&at), "row {i}");
        }
        // Every statistic flags, so each cycle is 10 + 2 rows.
        let always = pattern(1e-12);
        for (i, &live) in always.iter().enumerate() {
            assert_eq!(live, i % 12 == 11, "row {i}");
        }
    }

    /// `scalar = true`: the same detector on the mean of `u`, the
    /// equicorrelation of the standardized row, with the history's mean and
    /// Bartlett long-run sd of `u` written out.
    #[test]
    fn the_scalar_sequential_detector_is_its_definition() {
        let (d, m, limit) = (4usize, 30usize, 45usize);
        let mut model = CorrChange::new(CorrChangeCfg {
            scalar: true,
            ..seq_cfg(d, m, limit, 0.1)
        })
        .unwrap();
        let crit = crate::boundary::sequential_crit(0.05, 0.1, 1.5);
        let bw = ((m as f64).ln().floor() as usize).max(1);
        let mut n = Normals::new(3);
        let (mut hist, mut mon): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        let (mut monitoring, mut stats) = (false, 0);
        for i in 0..300 {
            let x = equicorrelated(&mut n, d, if i < 150 { 0.2 } else { 0.7 });
            let u = model.scalarise(&x);
            let got = model.step(&x, &[], 1.0, 1.0).pred;
            if !u.is_finite() {
                continue;
            }
            if monitoring && !mon.is_empty() {
                let k = (mon.len() + 1) as f64;
                let mean = (mon.iter().sum::<f64>() + u) / k;
                let level = hist.iter().sum::<f64>() / m as f64;
                let tf = m as f64;
                let v: Vec<f64> = hist.iter().map(|h| h - level).collect();
                let mut var = 0.0;
                for lag in 0..bw.min(m) {
                    let w = 1.0 - lag as f64 / bw as f64;
                    let acf: f64 = (lag..m).map(|s| v[s] * v[s - lag]).sum::<f64>() / tf;
                    var += w * acf * if lag == 0 { 1.0 } else { 2.0 };
                }
                let b = k / tf;
                let w = (1.0 + b) * (b / (1.0 + b)).powf(0.1);
                let want = ((k / tf.sqrt()) * (mean - level) / var.sqrt()).abs() / w;
                assert!(
                    (got[0] - want).abs() <= 1e-9 * (1.0 + want),
                    "row {i}: {} against {want}",
                    got[0]
                );
                assert!((got[1] - crit).abs() < 1e-12);
                stats += 1;
            } else {
                assert!(got[0].is_nan(), "row {i}");
            }
            if !monitoring {
                hist.push(u);
                monitoring = hist.len() == m;
            } else {
                mon.push(u);
                if got[2] == 1.0 || mon.len() == limit {
                    (monitoring, hist, mon) = (false, vec![], vec![]);
                }
            }
        }
        assert!(stats > 50, "{stats}");
    }

    /// A zero-weight row is reported as the next monitored row would be and
    /// then not learned: the stream around it is the stream without it.
    #[test]
    fn a_zero_weight_row_is_not_a_monitored_row() {
        let mut n = Normals::new(21);
        let rows: Vec<Vec<f64>> = (0..120).map(|_| n.pair(0.4)).collect();
        let mut with = CorrChange::new(seq_cfg(2, 40, 60, 0.2)).unwrap();
        let mut without = with.clone();
        for (i, x) in rows.iter().enumerate() {
            if i == 70 {
                let ghost = with.step(&[3.0, -3.0], &[], 1.0, 0.0).pred;
                let as_if = without.predict(&[3.0, -3.0], 1.0).pred;
                assert!(ghost.iter().zip(&as_if).all(|(a, b)| same(*a, *b)));
            }
            let (a, b) = (
                with.step(x, &[], 1.0, 1.0).pred,
                without.step(x, &[], 1.0, 1.0).pred,
            );
            assert!(a.iter().zip(&b).all(|(p, q)| same(*p, *q)), "row {i}");
        }
    }

    /// `predict` is the step's answer without the step, on every row of a
    /// cycle; a break (`clear_lags`) starts a new history.
    #[test]
    fn sequential_predict_is_the_step_and_a_break_restarts_the_cycle() {
        let mut n = Normals::new(8);
        let mut m = CorrChange::new(seq_cfg(2, 20, 30, 0.3)).unwrap();
        for _ in 0..120 {
            let x = n.pair(0.5);
            let want = m.predict(&x, 1.0).pred;
            let got = m.step(&x, &[], 1.0, 1.0).pred;
            assert!(want.iter().zip(&got).all(|(a, b)| same(*a, *b)));
        }
        m.clear_lags();
        assert!(m.monitoring.is_none() && m.ring.is_empty());
        for _ in 0..20 {
            assert!(m.step(&n.pair(0.5), &[], 1.0, 1.0).pred[0].is_nan());
        }
        assert!(m.monitoring.is_some(), "twenty rows are a history");
    }

    /// A state saved in the history, in the monitoring period, and -- for
    /// `scalar`, whose ring holds `u` -- in a `monitor` span, loads and goes
    /// on to the bit. A `scalar` monitor saved mid-span was refused on load:
    /// the shape check held its one-value rows to the feature count.
    #[test]
    fn a_state_saved_mid_cycle_goes_on_to_the_bit() {
        let cases = [
            seq_cfg(3, 30, 50, 0.2),
            CorrChangeCfg {
                scalar: true,
                ..seq_cfg(3, 30, 50, 0.0)
            },
            CorrChangeCfg {
                scalar: true,
                span_rows: 30,
                ..cfg(3, CorrChangeKind::Monitor)
            },
        ];
        for c in cases {
            for cut in [10usize, 45, 70] {
                let mut n = Normals::new(cut as u64);
                let rows: Vec<Vec<f64>> =
                    (0..130).map(|_| equicorrelated(&mut n, 3, 0.3)).collect();
                let mut whole = CorrChange::new(c.clone()).unwrap();
                let mut first = CorrChange::new(c.clone()).unwrap();
                for x in &rows[..cut] {
                    whole.step(x, &[], 1.0, 1.0);
                    first.step(x, &[], 1.0, 1.0);
                }
                let bytes = rmp_serde::to_vec_named(&first.state()).unwrap();
                let mut back = CorrChange::restore(&rmp_serde::from_slice(&bytes).unwrap())
                    .unwrap_or_else(|e| panic!("{:?} at {cut}: {e}", c.kind));
                for (i, x) in rows[cut..].iter().enumerate() {
                    let (a, b) = (
                        whole.step(x, &[], 1.0, 1.0).pred,
                        back.step(x, &[], 1.0, 1.0).pred,
                    );
                    assert!(
                        a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                        "{:?} cut {cut}, row {i}: {a:?} vs {b:?}",
                        c.kind
                    );
                }
            }
        }
    }

    /// `since_change` on the older kinds: a `monitor` flag dates the change
    /// after the CUSUM's argmax, `T − j*` rows through the span's last; a
    /// `window` flag says the second window. Null where nothing flags.
    #[test]
    fn the_older_kinds_date_their_flags_too() {
        let t = 200usize;
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let mut n = Normals::new(2);
        let rows: Vec<Vec<f64>> = (0..t)
            .map(|i| n.pair(if i < 120 { 0.0 } else { 0.9 }))
            .collect();
        let mut last = vec![];
        for x in &rows {
            last = m.step(x, &[], 1.0, 1.0).pred;
        }
        assert_eq!(last[2], 1.0, "a break of 0 to 0.9 is flagged");
        let fresh = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let (_, j) = fresh.monitor_stat_at(&rows, 0, 1);
        assert_eq!(last[4], (t - j) as f64);
        assert!((last[4] - 80.0).abs() <= 25.0, "{}", last[4]);

        let mut w = CorrChange::new(CorrChangeCfg {
            span_rows: 50,
            crit: Some(1e-9),
            ..cfg(2, CorrChangeKind::Window)
        })
        .unwrap();
        let outs: Vec<Vec<f64>> = (0..120)
            .map(|_| w.step(&n.pair(0.3), &[], 1.0, 1.0).pred)
            .collect();
        for o in &outs {
            if o[2] == 1.0 {
                assert_eq!(o[4], 50.0);
            } else {
                assert!(o[4].is_nan());
            }
        }
    }

    /// The configurations `sequential` refuses, by name.
    #[test]
    fn a_bad_sequential_configuration_is_refused_by_name() {
        let bad = |c: CorrChangeCfg, msg: &str| {
            let e = CorrChange::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(seq_cfg(2, 7, 10, 0.0), "span_rows of at least 8");
        bad(seq_cfg(2, 20, 1, 0.0), "monitor_rows of at least 2");
        for g in [0.5, -0.1, f64::NAN, f64::INFINITY] {
            bad(seq_cfg(2, 20, 10, g), "boundary_gamma must be in [0, 0.49]");
        }
        // Past 0.49 the critical value's solve, whose work grows as 1/(1/2 −
        // γ), runs for minutes to hours in every process and on every load
        // (review 2026-10-05, CD1): refused by `validate`, before it runs,
        // and 0.49 itself kept. Read through `validate` alone, since the
        // model a configuration builds solves for its critical value.
        for g in [0.4999, 0.491, 0.49f64.next_up()] {
            let e = seq_cfg(2, 20, 10, g).validate().unwrap_err();
            assert!(
                e.contains("boundary_gamma must be in [0, 0.49]"),
                "{g}: {e}"
            );
            assert!(e.contains("1/(1/2 − γ)"), "{g}: the message says why: {e}");
        }
        seq_cfg(2, 20, 10, 0.49).validate().unwrap();
        bad(
            CorrChangeCfg {
                bandwidth: Some(0),
                ..seq_cfg(2, 20, 10, 0.0)
            },
            "bandwidth must be >= 1",
        );
        // And a given critical value replaces the computed one.
        let m = CorrChange::new(CorrChangeCfg {
            crit: Some(2.5),
            ..seq_cfg(2, 20, 10, 0.2)
        })
        .unwrap();
        assert_eq!(m.seq_crit, 2.5);
    }

    // --- task 158: the mutation survivors --------------------------------

    /// Each bound `validate` draws sits where its message puts it: `alpha`
    /// strictly inside (0, 1), and every "at least" inclusive -- `span_rows`
    /// 8 for `monitor` and `sequential` and 3 for `window`, `monitor_rows`
    /// 2, `n_perm` 20, and a `perm_block` of the whole span.
    #[test]
    fn each_configuration_bound_is_where_its_message_puts_it() {
        for alpha in [0.0, 1.0] {
            let e = CorrChange::new(CorrChangeCfg {
                alpha,
                ..cfg(2, CorrChangeKind::Monitor)
            })
            .unwrap_err();
            assert!(e.contains("alpha must be strictly between"), "{alpha}: {e}");
        }
        for ok in [
            CorrChangeCfg {
                span_rows: 8,
                ..cfg(2, CorrChangeKind::Monitor)
            },
            seq_cfg(2, 8, 10, 0.0),
            seq_cfg(2, 20, 2, 0.0),
            CorrChangeCfg {
                span_rows: 3,
                ..cfg(2, CorrChangeKind::Window)
            },
            CorrChangeCfg {
                n_perm: 20,
                ..cfg(2, CorrChangeKind::Window)
            },
            CorrChangeCfg {
                span_rows: 50,
                perm_block: 50,
                ..cfg(2, CorrChangeKind::Window)
            },
        ] {
            let shown = format!("{ok:?}");
            assert!(CorrChange::new(ok).is_ok(), "{shown}");
        }
    }

    /// The output slots, named in the order they are emitted.
    #[test]
    fn the_labels_name_the_five_outputs_in_order() {
        assert_eq!(
            CorrChange::labels(),
            ["stat", "crit", "flag", "since_flag", "since_change"]
        );
        assert_eq!(CorrChange::labels().len(), CorrChange::n_outputs_for());
    }

    /// The Kolmogorov distribution has a second series, the theta-function
    /// form `√(2π)/x · Σ_{k≥1} exp(−(2k−1)²π²/(8x²))`, fast where the first is
    /// slow (Feller 1948, eq. 1.4; Marsaglia, Tsang & Wang 2003). The two
    /// agree at every `x`: below the quantiles, where the first series needs
    /// many terms, as much as at them.
    #[test]
    fn the_kolmogorov_series_is_the_theta_series() {
        use std::f64::consts::PI;
        for x in [0.3, 0.5, 0.7, 1.0, 1.3581, 2.0] {
            let theta = (2.0 * PI).sqrt() / x
                * (1..60)
                    .map(|k| {
                        let n = f64::from(2 * k - 1);
                        (-(n * n) * PI * PI / (8.0 * x * x)).exp()
                    })
                    .sum::<f64>();
            let got = kolmogorov_cdf(x);
            assert!(
                (got - theta).abs() < 1e-12,
                "x = {x}: {got} against {theta}"
            );
        }
    }

    /// `since_flag` is the learned rows since the last flag, this row
    /// included: the row's ordinal before any flag, counted again from the
    /// row after one. A span that does not flag dates nothing.
    #[test]
    fn since_flag_counts_the_rows_since_the_last_flag() {
        let t = 40usize;
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let mut n = Normals::new(2);
        let (mut last_flag, mut flags) = (0usize, Vec::new());
        for i in 1..=4 * t {
            // A break of 0 to 0.95 halfway through the second span.
            let rho = if (t + t / 2..=2 * t).contains(&i) {
                0.95
            } else {
                0.0
            };
            let out = m.step(&n.pair(rho), &[], 1.0, 1.0).pred;
            if i % t != 0 {
                assert!(out[3].is_nan(), "row {i}: nothing off a span's last row");
                continue;
            }
            assert_eq!(out[3], (i - last_flag) as f64, "row {i}");
            if out[2] == 1.0 {
                (last_flag, flags) = (i, [flags, vec![i]].concat());
            } else {
                assert!(out[4].is_nan(), "row {i}: no flag, no date");
            }
        }
        assert_eq!(flags, [2 * t], "the break's span flags, and only it");
    }

    /// The window kind reports nothing until both windows are full, not even
    /// its critical value. With `reset` a flag empties both, and the next
    /// report waits for both again; without it a flag empties nothing.
    #[test]
    fn the_window_kind_waits_for_both_windows() {
        let w = 10usize;
        let reports = |crit: f64, reset: bool| {
            let mut m = CorrChange::new(CorrChangeCfg {
                span_rows: w,
                crit: Some(crit),
                reset,
                ..cfg(2, CorrChangeKind::Window)
            })
            .unwrap();
            let mut n = Normals::new(4);
            (1..=6 * w)
                .filter(|_| m.step(&n.pair(0.3), &[], 1.0, 1.0).pred[1].is_finite())
                .collect::<Vec<usize>>()
        };
        let every: Vec<usize> = (2 * w..=6 * w).collect();
        assert_eq!(reports(1e9, true), every, "nothing flags");
        assert_eq!(reports(1e-9, false), every, "every report flags");
        assert_eq!(reports(1e-9, true), [2 * w, 4 * w, 6 * w], "and resets");
    }

    /// With a fixed critical value no permutation is drawn: the generator
    /// is where the seed put it, and no permutation value is held.
    #[test]
    fn a_fixed_critical_value_draws_no_permutations() {
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: 10,
            crit: Some(0.5),
            ..cfg(2, CorrChangeKind::Window)
        })
        .unwrap();
        let mut n = Normals::new(6);
        for _ in 0..60 {
            m.step(&n.pair(0.3), &[], 1.0, 1.0);
        }
        assert!(m.perm_crit.is_none());
        assert_eq!(m.rng, SplitMix64::new(m.cfg.seed));
    }

    /// The permutation critical value is redrawn on its cadence: drawn after
    /// the first row that fills both windows reports, then kept for
    /// `permute_every` rows and redrawn after the next, so each value is in
    /// force for `permute_every + 1` reports.
    #[test]
    fn the_permutation_critical_value_is_redrawn_on_its_cadence() {
        let (w, every) = (10usize, 4usize);
        let mut m = CorrChange::new(CorrChangeCfg {
            span_rows: w,
            crit: None,
            permute_every: every,
            ..cfg(2, CorrChangeKind::Window)
        })
        .unwrap();
        let mut n = Normals::new(8);
        let crits: Vec<f64> = (0..2 * w + 5 * (every + 1))
            .map(|_| m.step(&n.pair(0.3), &[], 1.0, 1.0).pred[1])
            .collect();
        assert!(crits[..2 * w].iter().all(|c| c.is_nan()), "{crits:?}");
        let blocks: Vec<&[f64]> = crits[2 * w..].chunks(every + 1).collect();
        for (b, block) in blocks.iter().enumerate() {
            assert!(block.iter().all(|c| *c == block[0]), "block {b}: {block:?}");
            if b > 0 {
                assert_ne!(block[0], blocks[b - 1][0], "block {b} was not redrawn");
            }
        }
    }

    /// A `sequential` state saved in its monitoring period is held to the
    /// cfg's shape on load, each of the period's vectors on its own.
    #[test]
    fn each_monitoring_vector_is_held_to_its_shape_on_load() {
        let mut m = CorrChange::new(seq_cfg(3, 20, 40, 0.0)).unwrap();
        let mut n = Normals::new(12);
        for _ in 0..25 {
            m.step(&equicorrelated(&mut n, 3, 0.3), &[], 1.0, 1.0);
        }
        assert!(m.monitoring.as_ref().is_some_and(|mon| mon.rows.len() == 5));
        type Spoil = (&'static str, fn(&mut Monitoring));
        let cases: [Spoil; 6] = [
            ("level", |mon| mon.level.push(0.0)),
            ("scale", |mon| mon.scale.push(1.0)),
            ("mean", |mon| mon.mean.push(0.0)),
            ("m2", |mon| mon.m2.push(0.0)),
            ("cross", |mon| mon.cross.push(0.0)),
            ("a row", |mon| mon.rows[0].push(0.0)),
        ];
        for (what, spoil) in cases {
            let mut bad = m.clone();
            spoil(bad.monitoring.as_mut().unwrap());
            match CorrChange::restore(&bad.state()) {
                Err(crate::StateError::Invalid(e)) => {
                    assert!(e.contains("wrong shape"), "{what}: {e}");
                }
                other => panic!("{what}: {other:?}"),
            }
        }
        assert!(CorrChange::restore(&m.state()).is_ok());
    }

    /// `pair_at` walks the pairs `a < b`, `a` outer and `b` inner: the order
    /// `level` holds them in.
    #[test]
    fn pair_at_walks_the_upper_triangle_in_order() {
        for d in 2..=6usize {
            let want: Vec<(usize, usize)> = (0..d)
                .flat_map(|a| ((a + 1)..d).map(move |b| (a, b)))
                .collect();
            let got: Vec<(usize, usize)> =
                (0..want.len()).map(|i| CorrChange::pair_at(d, i)).collect();
            assert_eq!(got, want, "d = {d}");
        }
    }

    fn scalar_monitor() -> CorrChange {
        CorrChange::new(CorrChangeCfg {
            span_rows: 8,
            scalar: true,
            ..cfg(3, CorrChangeKind::Monitor)
        })
        .unwrap()
    }

    fn column(us: &[f64]) -> Vec<Vec<f64>> {
        us.iter().map(|&u| vec![u]).collect()
    }

    /// The scalar statistic needs four rows, as the pair one does
    /// (`a_span_of_four_rows_is_the_first_with_a_long_run_sd`): three give
    /// none, four give one.
    #[test]
    fn a_scalar_span_of_four_rows_is_the_first_with_a_statistic() {
        let m = scalar_monitor();
        let u = column(&[0.3, -0.2, 0.5, 0.1]);
        assert!(m.scalar_stat(&u[..3]).is_nan());
        let four = m.scalar_stat(&u);
        assert!(four.is_finite() && four > 0.0, "{four}");
    }

    /// A series without spread has no long-run standard deviation and so no
    /// statistic: eight values of 0.25, whose mean is exact, so every
    /// deviation is exactly 0.
    #[test]
    fn a_constant_scalar_series_has_no_statistic() {
        assert!(CorrChange::scalar_long_run_sd(&[0.25; 8], 2).is_nan());
        assert!(scalar_monitor().scalar_stat(&column(&[0.25; 8])).is_nan());
    }

    /// The scalar CUSUM's argmax is the row, counted from 1, at which its
    /// maximum is first reached. Here `(j/√T)·|ūⱼ − ū|` is the same double at
    /// rows 1 and 2 (the steps between them are powers of two) and smaller
    /// after, so the change is dated after row 1.
    #[test]
    fn the_scalar_cusum_dates_at_the_first_maximum() {
        let (stat, at) =
            scalar_monitor().scalar_stat_at(&column(&[1.0, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0, 0.0]));
        assert!(stat.is_finite() && stat > 0.0, "{stat}");
        assert_eq!(at, 1);
    }

    /// The pair CUSUM's argmax is the first `j` at which its maximum is
    /// reached. These ten integer rows (found by a search) have `ρ̂_T = 0`
    /// exactly and `(j/√T)·|ρ̂ⱼ − ρ̂_T|` the same double at `j = 2` and `j = 4`
    /// (`ρ̂₂ = −1`, `ρ̂₄ = −1/2`, and the steps between are powers of two),
    /// the largest over the span, so the change is dated after row 2.
    #[test]
    fn the_pair_cusum_dates_at_the_first_maximum() {
        let rows: Vec<Vec<f64>> = [
            (3, -3),
            (1, -2),
            (0, -3),
            (0, 0),
            (2, 2),
            (1, 2),
            (3, -3),
            (-1, -2),
            (-1, -3),
            (2, -3),
        ]
        .iter()
        .map(|&(x, y)| vec![f64::from(x), f64::from(y)])
        .collect();
        let t = rows.len();
        assert_eq!(CorrChange::corr_of(&rows, t, 0, 1), 0.0);
        assert_eq!(CorrChange::corr_of(&rows, 2, 0, 1), -1.0);
        assert_eq!(CorrChange::corr_of(&rows, 4, 0, 1), -0.5);
        let m = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(2, CorrChangeKind::Monitor)
        })
        .unwrap();
        let (stat, at) = m.monitor_stat_at(&rows, 0, 1);
        assert!(stat.is_finite() && stat > 0.0, "{stat}");
        assert_eq!(at, 2);
    }

    /// `sequential`'s detector reports the first pair, in the walk order,
    /// at which its largest ratio is reached: that pair dates the change. A
    /// monitoring period built by hand, and a row at its means, so that
    /// each pair's correlation is its co-moment over the root of its
    /// second moments exactly: `±2/√(4·4) = ±1/2`, against levels 0 and
    /// −1, ties the first two pairs at `|ρ − level| = 1/2`.
    #[test]
    fn the_sequential_detector_reports_the_first_pair_on_a_tie() {
        let m = CorrChange::new(seq_cfg(3, 20, 40, 0.0)).unwrap();
        let mon = Monitoring {
            level: vec![0.0, -1.0, 0.0],
            scale: vec![1.0; 3],
            rows: vec![vec![0.0; 3]; 4],
            mean: vec![0.0; 3],
            m2: vec![4.0; 3],
            cross: vec![2.0, -2.0, 0.0],
        };
        let (ratio, at) = m.sequential_stat(&mon, &[0.0, 0.0, 0.0]);
        assert!(ratio.is_finite() && ratio > 0.0, "{ratio}");
        assert_eq!(at, 0);
    }

    /// A `scalar` monitoring period of the given values of `u`.
    fn scalar_period(us: &[f64]) -> Monitoring {
        Monitoring {
            level: vec![0.0],
            scale: vec![1.0],
            rows: column(us),
            mean: vec![0.0],
            m2: vec![0.0],
            cross: vec![],
        }
    }

    /// W&G's Eq. 8 for `scalar`, written out on the monitored values: `k̂`
    /// maximises `j·|ūⱼ − ūₙ|` over the prefix means, the earliest `j` on a
    /// tie, and the output is `n + 1 − k̂`. The first period's scores are 1,
    /// 1, 0, 0; the others are shifted Gaussian series of several lengths.
    #[test]
    fn the_scalar_change_is_dated_by_eq_8() {
        let m = CorrChange::new(CorrChangeCfg {
            scalar: true,
            ..seq_cfg(3, 20, 40, 0.0)
        })
        .unwrap();
        let tie = scalar_period(&[1.0, 0.0, -1.0, 0.0]);
        assert_eq!(m.sequential_since_change(&tie, 0), 4.0);
        let mut n = Normals::new(31);
        for trial in 0..20usize {
            let (len, shift) = (10 + trial, 3 + trial % 6);
            let us: Vec<f64> = (0..len)
                .map(|i| n.normal() + if i >= shift { 2.0 } else { 0.0 })
                .collect();
            let mean = |j: usize| us[..j].iter().sum::<f64>() / j as f64;
            let last = mean(len);
            let (mut best, mut k_hat) = (f64::NEG_INFINITY, 0);
            for j in 1..=len {
                let score = j as f64 * (mean(j) - last).abs();
                if score > best {
                    (best, k_hat) = (score, j);
                }
            }
            assert_eq!(
                m.sequential_since_change(&scalar_period(&us), 0),
                (len + 1 - k_hat) as f64,
                "trial {trial}"
            );
        }
    }

    /// A window in which one pair has no correlation (a column constant in
    /// the first window) gives no statistic under either norm: the L∞
    /// norm's running maximum would otherwise pass over the NaN (`f64::max`
    /// ignores one) and report the other pairs' largest.
    #[test]
    fn a_pair_without_a_correlation_leaves_no_window_statistic() {
        let w = 20usize;
        let mut n = Normals::new(14);
        let rows: Vec<Vec<f64>> = (0..2 * w)
            .map(|i| {
                let mut r = n.pair(0.4);
                r.push(if i < w { 1.5 } else { n.normal() });
                r
            })
            .collect();
        for norm in [ChangeNorm::L1, ChangeNorm::LInf] {
            let m = |d: usize| {
                CorrChange::new(CorrChangeCfg {
                    span_rows: w,
                    crit: Some(0.5),
                    norm,
                    ..cfg(d, CorrChangeKind::Window)
                })
                .unwrap()
            };
            assert!(m(3).window_stat(&rows).is_nan(), "{norm:?}");
            assert!(
                m(2).window_stat(&rows).is_finite(),
                "{norm:?}: the first pair"
            );
        }
    }

    /// The window statistic is free of the columns' units: one column scaled
    /// by a power of two gives the same statistic, to the bit. At `2^-270`
    /// the column's sums of squares (`~2^-540`) are normal and their squares
    /// are not, so no correlation of the column with itself could be formed
    /// there; the statistic is over the pairs `a < b` alone.
    #[test]
    fn the_window_statistic_is_free_of_a_columns_units() {
        let w = 20usize;
        let mut n = Normals::new(15);
        let rows: Vec<Vec<f64>> = (0..2 * w)
            .map(|i| equicorrelated(&mut n, 3, if i < w { 0.2 } else { 0.6 }))
            .collect();
        for norm in [ChangeNorm::L1, ChangeNorm::LInf] {
            let m = CorrChange::new(CorrChangeCfg {
                span_rows: w,
                crit: Some(0.5),
                norm,
                ..cfg(3, CorrChangeKind::Window)
            })
            .unwrap();
            let base = m.window_stat(&rows);
            assert!(base.is_finite() && base > 0.0, "{base}");
            for e in [-270, 300] {
                let s = 2f64.powi(e);
                let scaled: Vec<Vec<f64>> =
                    rows.iter().map(|r| vec![r[0], s * r[1], r[2]]).collect();
                assert_eq!(m.window_stat(&scaled), base, "{norm:?} at 2^{e}");
            }
        }
    }

    /// The permutation null in blocks: three blocks have six orders, and two
    /// hundred Fisher–Yates draws visit every one, so at a level that reads
    /// the largest draw the critical value is the largest statistic over the
    /// six orders, written out here, and at 50 % it is one of them below
    /// that. The largest is at an odd order only, which a shuffle that only
    /// ever composes 3-cycles would never reach.
    #[test]
    fn the_permutation_null_visits_every_block_order() {
        let (w, block) = (3usize, 2usize);
        let mut n = Normals::new(21);
        let rows: Vec<Vec<f64>> = (0..2 * w).map(|_| n.pair(0.3)).collect();
        let base = CorrChange::new(CorrChangeCfg {
            span_rows: w,
            crit: None,
            n_perm: 200,
            perm_block: block,
            ..cfg(2, CorrChangeKind::Window)
        })
        .unwrap();
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let odd = [false, true, true, false, false, true];
        let all: Vec<f64> = orders
            .iter()
            .map(|order| {
                let shuffled: Vec<Vec<f64>> = order
                    .iter()
                    .flat_map(|&b| rows[b * block..(b + 1) * block].to_vec())
                    .collect();
                base.window_stat(&shuffled)
            })
            .collect();
        let top = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            all.iter().zip(odd).all(|(&s, o)| s < top || o),
            "the largest must be at an odd order only: {all:?}"
        );
        for (alpha, at_top) in [(0.001, true), (0.5, false)] {
            let mut m = CorrChange::new(CorrChangeCfg {
                alpha,
                ..base.cfg.clone()
            })
            .unwrap();
            m.ring = rows.iter().cloned().collect();
            let crit = m.permutation_crit().expect("both windows are full");
            assert!(all.contains(&crit), "alpha {alpha}: {crit} not in {all:?}");
            if at_top {
                assert_eq!(crit, top, "alpha {alpha}");
            } else {
                assert!(crit < top, "alpha {alpha}: {crit} against {top}");
            }
        }
    }

    /// The monitor's statistic is the largest of its pairs': with three
    /// columns and a break in the correlation of the first and the third
    /// alone, it is that pair's -- not the first pair's, not the last's, not
    /// the smallest.
    #[test]
    fn the_monitor_statistic_is_the_largest_pair() {
        let t = 80usize;
        let fresh = CorrChange::new(CorrChangeCfg {
            span_rows: t,
            ..cfg(3, CorrChangeKind::Monitor)
        })
        .unwrap();
        let mut m = fresh.clone();
        let mut n = Normals::new(16);
        let rows: Vec<Vec<f64>> = (0..t)
            .map(|i| {
                let p = n.pair(if i < t / 2 { 0.0 } else { 0.9 });
                vec![p[0], n.normal(), p[1]]
            })
            .collect();
        let mut last = vec![];
        for x in &rows {
            last = m.step(x, &[], 1.0, 1.0).pred;
        }
        let q: Vec<f64> = [(0, 1), (0, 2), (1, 2)]
            .iter()
            .map(|&(a, b)| fresh.monitor_stat(&rows, a, b))
            .collect();
        assert!(q[1] > q[0] && q[1] > q[2], "{q:?}");
        assert_eq!(last[0], q[1]);
    }

    /// The detector flags where it exceeds the critical value, not where it
    /// meets it (W&G stop at `|V_k| > c·w(k/m)`): set to the largest
    /// statistic of a cycle, the critical value lets that row pass, and
    /// every row before it.
    #[test]
    fn a_statistic_at_the_critical_value_does_not_flag() {
        let make = |crit: f64| {
            CorrChange::new(CorrChangeCfg {
                crit: Some(crit),
                ..seq_cfg(2, 20, 30, 0.0)
            })
            .unwrap()
        };
        let mut n = Normals::new(18);
        let rows: Vec<Vec<f64>> = (0..50).map(|_| n.pair(0.3)).collect();
        let mut free = make(1e9);
        let (mut at, mut top) = (0, f64::NEG_INFINITY);
        for (i, x) in rows.iter().enumerate() {
            let s = free.step(x, &[], 1.0, 1.0).pred[0];
            if s > top {
                (at, top) = (i, s);
            }
        }
        assert!(top.is_finite() && at > 20, "{at}: {top}");
        let mut held = make(top);
        for (i, x) in rows.iter().enumerate().take(at + 1) {
            let out = held.step(x, &[], 1.0, 1.0).pred;
            assert_ne!(out[2], 1.0, "row {i}: {} against {top}", out[0]);
            if i == at {
                assert_eq!(out[0], top);
            }
        }
    }

    /// A row with a value that is not usable takes no place in a span
    /// (`OnlineModel`, task 183), as a row of weight 0 takes none: with no
    /// decay the test is exactly as it was, so the stream with the row is
    /// the stream without it, and every span closes on the same learned row.
    /// A row with a weight was counted in `n_eff` whatever its features
    /// held, and fed to the scalar statistic's moments, where a NaN never
    /// leaves. Under both readings of the row; for a feature that is not a
    /// number, one past the input bound and a weight that is not usable.
    #[test]
    fn a_refused_row_takes_no_place_in_a_span() {
        // Bytes, not `==`: a state holds a NaN where a statistic is not set.
        let bytes = |m: &CorrChange| rmp_serde::to_vec(&m.state()).unwrap();
        let mut s = 17u64;
        let mut u = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let rows: Vec<[f64; 3]> = (0..50).map(|_| [u(), u(), u()]).collect();
        for scalar in [false, true] {
            for (bad, w) in [
                ([f64::NAN, 0.5, 0.1], 1.0),
                ([0.5, -2.0 * crate::INPUT_BOUND, 0.1], 1.0),
                ([0.5, 0.25, 0.1], f64::INFINITY),
            ] {
                let case = format!("scalar {scalar}, {bad:?} at weight {w}");
                let c = CorrChangeCfg {
                    span_rows: 20,
                    scalar,
                    ..cfg(3, CorrChangeKind::Monitor)
                };
                let mut with = CorrChange::new(c.clone()).unwrap();
                let mut without = CorrChange::new(c).unwrap();
                let mut closed = 0;
                for (i, x) in rows.iter().enumerate() {
                    if i == 30 {
                        let ring = with.ring.clone();
                        let out = with.step(&bad, &[], 1.0, w);
                        assert_eq!(with.ring, ring, "{case}: the span took the row");
                        assert!(
                            !w.is_finite() || out.pred.iter().all(|v| v.is_nan()),
                            "{case}: {out:?}"
                        );
                        assert_eq!(bytes(&with), bytes(&without), "{case}");
                    }
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let a = with.step(x, &[], d, 1.0);
                    let b = without.step(x, &[], d, 1.0);
                    assert!(
                        a.pred.len() == b.pred.len()
                            && a.pred.iter().zip(&b.pred).all(|(p, q)| {
                                p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan())
                            }),
                        "{case}, row {i}: {a:?} against {b:?}"
                    );
                    closed += usize::from(a.pred[0].is_finite());
                }
                assert_eq!(bytes(&with), bytes(&without), "{case}");
                assert!(closed >= 2, "{case}: spans closed {closed}");
            }
        }
    }
}
