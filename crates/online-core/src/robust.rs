//! Robust regression: Huber and quantile (docs/PLAN.md §4.5).
//!
//! IRLS-style reweighting on the EW-ridge update: each row's weight is scaled by
//! the robust weight of its *prior* residual, so the reweighting is still
//! out-of-sample (the residual comes from the prediction made before the update).
//!
//! Huber, with `d = huber_delta` in units of the target's residual scale `s_j`:
//!
//! ```text
//! w_robust = 1                 if |r| <= d * s
//!          = d * s / |r|       otherwise
//! ```
//!
//! Quantile (check loss at level tau) does not reweight: it takes one Newton
//! step on the check loss smoothed by a uniform kernel of half-width
//! `h = quantile_eps * s`, linearised at the fit the row was scored with. The
//! smoothed loss has curvature `1/(2h)` inside the band and none outside, so
//! with `psi(r) = tau - 1{r < 0}` the row is:
//!
//! ```text
//! |r| <  h:  a least-squares row, target y + 2h(tau - 1/2)
//! |r| >= h:  no weight in the Gram; 2h * psi(r) * z into the cross-moment
//! ```
//!
//! Both arms come from one identity: the Newton system's row is
//! `z z' beta = z z' beta_prior + 2h * psi(r) * z`, and inside the band
//! `2h * psi(r) = 2h(tau - 1/2) + r`, which folds the prior fit back out. The
//! IRLS weight this replaced, `psi(r)/r`, is that step's *secant* where this is
//! its tangent, and it is unbounded as `r -> 0`: a row whose prior residual
//! happened to be near zero kept a weight of up to `1/quantile_eps` for ever,
//! and the fit settled a fraction `1/(1 + ln(1/eps))` of the way from the mean
//! of its own past fits to the quantile regression -- 0.164 short of
//! `statsmodels`' `QuantReg` at the median of a skewed noise after 20 000 rows,
//! and 0.477 at the 0.9 quantile (review 2026-09-12, N9).
//!
//! A nudge enters the cross-moment, a mean, as a step over the band's
//! weight, and the step is bounded so that the next solve moves the row's
//! prediction by at most its residual: the linearisation holds inside the
//! band, and a step past the row's own target is more than one term of the
//! score can justify.
//!
//! ```text
//! |step| <= |r| / (1 + u' A^-1 u)    (u' A^-1 u through the origin)
//! ```
//!
//! `A` is the band system the solve factorizes: the centred Gram over the
//! kept features, scaled under `standardize`, plus the ridge, or the raw
//! Gram through the origin. `u` is the row's deviation from the band's
//! means over the same columns, scaled alike, or its raw values through the
//! origin. The bound dates from review 2026-09-26 (G2), which read the
//! row's leverage off the Gram's diagonal; the full leverage `u' A^-1 u`
//! is review 2026-10-05's (TC1b).
//!
//! Under three rows per coefficient of the rows the target was present on,
//! the quantile fit warms up as ordinary least squares: a Newton step needs a
//! Hessian, and a band around a fit built from a handful of rows is not one.
//! And the band is never narrower than `(k/n)^(2/5)` of `s` for the target's
//! effective sample `n`, its present rows counted one each and decayed, so a
//! weight's scale reaches neither (docs/PLAN.md task 147) -- the smoothed-quantile bandwidth rate, which a long
//! stream leaves behind and which keeps the Hessian fed under a short
//! half-life, where the band's share of the sample is a few rows. The warm-up
//! read the band's weight until the second review of 2026-09-15 (F3), which a
//! half-life caps at that share, so a tail quantile at `half_life = 30` kept
//! falling back into warm-up and covered 0.825 where 0.9 was asked. What
//! that warm-up also did was rebuild a fit a row at the input bound had
//! left behind. Such a row sets the Gram and the cross-moment at its own
//! scale, and a mean-form accumulator forgets only through rows entering
//! it; every later row is outside the band (its residual is at the bound's
//! scale, where the band is a fraction of it), so only nudges arrive, each
//! `2h * psi * z` over the band's weight -- nothing beside the bound's
//! moments until that weight has decayed to nothing, and a step that
//! outgrows the band once it has. The fit oscillates, the band's weight
//! underflows after 1400 half-lives, and the prediction is withheld (the
//! bounded-extremes contract). So a band holding under one row per
//! coefficient takes least-squares rows until it holds rows again: from
//! one row up an outside row's step, `2h * |psi| / wj`, lands inside the
//! band, and a band the floor keeps at a share of the sample is never near
//! one row in steady state, so the warm-up's bias does not return.
//!
//! `s` is the residual scale of that target as the row arrives (under the
//! RMS, the quantile band's, the EW residual std; under the MAD, below, the
//! EW median of `|e|` over `Φ⁻¹(3/4)`). The target has one once it is
//! finite and above 0 (under the RMS, once `s²` is); before that -- no residual yet,
//! or every one so far exactly zero -- there is nothing to judge an outlier
//! against, and Huber down-weights no row: its weight is 1. Nor is there a
//! band to draw, and a quantile row is a least-squares row, as in its
//! warm-up. `s` was taken as 1 there, a cut and a band in the target's own
//! units, so a target in millions had its first predicted rows
//! down-weighted as outliers and one in millionths none, and a quantile
//! band drawn there, past the warm-up, put a target in millionths up to 2e5
//! of its own units off on the scaling test's stream (task 177). `s²`'s
//! weight is not consulted: `s²` is written only by a row of positive
//! weight, and a mean-form estimate keeps its value across a gap while its
//! weight ages, as the fit itself does (CLAUDE.md hard rule 8).
//!
//! **The Huber cut's scale is a configuration, `huber_scale`**
//! ([`HuberScale`]; docs/PLAN.md task 271, docs/HUBER.md):
//!
//! - **`Mad`**, a spec's default: `s = m / Φ⁻¹(3/4)`, `m` the EW median of
//!   `|e|` read from one [`EwQuantile`] at 0.5 per target (within
//!   `tanh(1/128)` of the exact EW median; bounded to [`MAD_BAND_OCTAVES`]
//!   either side of the median, task 271 (b)), fed the residuals and raw
//!   weights `σ²` is. An outlier moves a median by its weight whatever its
//!   size, so the outliers cannot widen the cut drawn to catch them, and a
//!   fit's first residuals -- the level's size, while the intercept is near
//!   0 -- leave it once the fit has the level. At 5% one-sided outliers on a
//!   target at 1,000 with no decay the slope is 2.041 where statsmodels'
//!   converged `RLM` with the MAD gives 2.039.
//! - **`Rms`**: `s = √σ²`, the plain EW mean of squared residuals, in which
//!   the rows Huber down-weights count at full weight, so an outlier
//!   inflates the scale the next rows' cuts are drawn in, and without decay
//!   the first residuals stay in it for good (review 2026-09-12, D4; C-1 of
//!   review round 9): 2.676 on the same stream, least squares 4.05. The
//!   only scale before schema 63, and what a state written before it loads
//!   as ([`HuberScale::of_a_state_before_63`]).
//!
//! Either ages on `huber_scale_decay`, the model's decay unless given: a
//! robust scale holds the fit back after a level shift until the median
//! itself has moved (every post-shift row is an outlier until then), and a
//! shorter memory follows the shift sooner (docs/HUBER.md §4). A quantile
//! fit's band is drawn in `√σ²` on the model's clock, whatever these say.
//!
//! Because the weights are per target, the `S` accumulator is per target here
//! (one [`EwCov`] each) — unlike [`crate::EwRidge`], which shares one.
//!
//! **Each coefficient's data share** (`support_coef`; docs/PLAN.md task 116,
//! docs/WARMUP-AND-CONVERGENCE.md §2.2). The ridge is mean-form: each solve
//! inverts `A = G + λI`, `G` the band Gram -- the rows as the loss weighs
//! them, centred and scaled as the solve scales them, or raw through the
//! origin -- and `λ` the ridge plus any jitter the factor needed. So
//! `S = G A⁻¹ = I − λ A⁻¹` is the shrinkage matrix, and
//!
//! ```text
//! support_coef_j = S_jj = 1 − λ (A⁻¹)_jj    in [0, 1]
//! ```
//!
//! is the share of coefficient `j` the data, as the robust loss weighs it,
//! determined rather than the ridge: `ewridge`'s statistic on the system
//! this model solves. Taken at each solve, `O(k³)`, and kept per target;
//! NaN in the intercept's slot, 0 for a column the solve dropped.

use serde::{Deserialize, Serialize};

use std::sync::OnceLock;

use crate::model::{Fit, ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::since::Since;
use crate::solve::{QuadWork, dot_aug};
use crate::{Decay, EwCov, EwQuantile, SpdFactor};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustLoss {
    /// Huber with `delta` in units of the residual scale
    /// ([`RobustCfg::huber_scale`]).
    Huber { delta: f64 },
    /// Quantile regression at level `tau`.
    Quantile { tau: f64 },
}

/// `Φ⁻¹(3/4)`, the third quartile of the standard normal: the median of
/// `|Z|` for `Z` standard normal, so `median|e| / Φ⁻¹(3/4)` is the normal's
/// standard deviation for normal errors, the MAD's consistency constant.
/// A literal, not a libm result (CLAUDE.md, no libm in the state); held to
/// scipy's `norm.ppf(0.75)` by `tests/test_huber_scale.py`.
pub const NORMAL_Q3: f64 = 0.674_489_750_196_081_7;

/// The scale a Huber cut is drawn in, per target (the module docs;
/// docs/PLAN.md task 271).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HuberScale {
    /// The exponentially weighted median of `|e|` over [`NORMAL_Q3`]: one
    /// [`EwQuantile`] at 0.5 per target, within its `tanh(1/128)` of the
    /// exact EW median. Breakdown 50%: an outlier moves the median by one
    /// row's weight whatever its size.
    Mad,
    /// The exponentially weighted root mean square of `e`, `√σ²`. Breakdown
    /// 0: one wild row sets it.
    Rms,
}

impl HuberScale {
    /// What a configuration written before schema 63, which had no such
    /// field, was fitted with: the root mean square, the only scale there
    /// was. A spec's default is [`HuberScale::Mad`]; a state's missing
    /// field is not a spec left at its default but a fit made under the
    /// RMS, and it must go on as one, to the bit (CLAUDE.md hard rule 5).
    pub fn of_a_state_before_63() -> Self {
        Self::Rms
    }

    /// The word a spec spells it with.
    pub fn word(self) -> &'static str {
        match self {
            Self::Mad => "mad",
            Self::Rms => "rms",
        }
    }
}

/// A target's MAD scale from its sketch of `|e|`: the median over
/// [`NORMAL_Q3`], once it is finite and above 0. `None` before any
/// residual, or while at least half the weight is on residuals of exactly
/// 0: no scale to judge an outlier against (the module docs).
pub(crate) fn mad_scale(q: &EwQuantile) -> Option<f64> {
    let s = q.get(0)? / NORMAL_Q3;
    (s > 0.0 && s.is_finite()).then_some(s)
}

/// The weighted median of `a` at weights `v`: the smallest value at which
/// the cumulative weight reaches half the total (the warm-up refit's scale,
/// [`RobustCfg::warm_rows`]). NaN when there is no weight.
fn weighted_median(a: &[f64], v: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..a.len()).filter(|&i| v[i] > 0.0).collect();
    idx.sort_by(|&p, &q| a[p].total_cmp(&a[q]));
    let total: f64 = idx.iter().map(|&i| v[i]).sum();
    let mut cum = 0.0;
    for &i in &idx {
        cum += v[i];
        if cum >= 0.5 * total {
            return a[i];
        }
    }
    f64::NAN
}

/// The octaves a MAD sketch keeps either side of its median's bucket
/// ([`EwQuantile::bounded`]): 16 octaves in all, so a median that falls
/// from the start-up residuals to the noise's by up to a factor of 2^16
/// folds nothing it later reads (docs/PLAN.md task 271 (b)).
pub const MAD_BAND_OCTAVES: u32 = 8;

/// One sketch of `|e|` per target under the MAD, none under the RMS.
pub(crate) fn mad_sketches(scale: HuberScale, n: usize) -> Vec<EwQuantile> {
    match scale {
        HuberScale::Mad => vec![
            EwQuantile::bounded(&[0.5], MAD_BAND_OCTAVES)
                .expect("0.5 is in (0, 1) and the band in range");
            n
        ],
        HuberScale::Rms => Vec::new(),
    }
}

/// Whether a restored state's sketches are the ones its scale keeps: `n`
/// sketches at the level 0.5 under the MAD, each one a sketch that can be
/// read and added to, and none under the RMS.
pub(crate) fn mad_sketches_fit(scale: HuberScale, n: usize, q: &[EwQuantile]) -> bool {
    match scale {
        HuberScale::Mad => q.len() == n && q.iter().all(|s| s.levels() == [0.5] && s.has_shape()),
        HuberScale::Rms => q.is_empty(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    pub loss: RobustLoss,
    pub ridge: f64,
    pub standardize: bool,
    pub min_weight: f64,
    /// Solve cadence in clock units; <= 0 solves every row. Measured from
    /// the last solve on the rows' stamps, the decayed clock held exactly,
    /// where the caller hands them (`since::Since`, docs/PLAN.md task 180).
    /// A row of weight 0 never solves: a solve due on one waits for the next
    /// row with weight (hard rule 9, docs/PLAN.md task 214).
    pub solve_every: f64,
    /// Row cap between solves, counted on the rows with weight.
    pub max_rows_between_solves: u32,
    /// The default cadence (docs/PLAN.md task 115 (b)): solve once the weight
    /// learned since the last solve reaches this share of the weight the fit
    /// holds, in place of `solve_every`'s clock. In steady state that is the
    /// clock's own `half_life / 50` at a share of `ln 2 / 50`; where they part
    /// -- warm-up, after a gap, a half-life far longer than the stream -- it
    /// keeps the fit that close to its data, where the clock solved once and
    /// never again. `None` keeps the clock.
    #[serde(default)]
    pub solve_share: Option<f64>,
    /// Half-width of the band a quantile fit takes its Newton step in, in
    /// units of the EW residual std: rows inside carry the curvature, rows
    /// outside only the score (the module docs). It was the floor under `|r|`
    /// in an IRLS weight, bounding a weight rather than naming a band (review
    /// 2026-09-12, N9).
    pub quantile_eps: f64,
    /// The scale the Huber cut is drawn in (the module docs). A state
    /// written before schema 63 has no such field and was fitted under the
    /// RMS, so it loads as [`HuberScale::Rms`]
    /// ([`HuberScale::of_a_state_before_63`]), where a spec's default is
    /// the MAD. A quantile fit keeps [`HuberScale::Rms`]: its band reads
    /// the EW root mean square, which this does not change.
    #[serde(default = "HuberScale::of_a_state_before_63")]
    pub huber_scale: HuberScale,
    /// The scale's own memory: `None` ages it on the model's `decay`, as
    /// before schema 63. Huber only.
    #[serde(default)]
    pub huber_scale_decay: Option<Decay>,
    /// Named subsets of the features (0-based, the intercept excluded),
    /// each a fit of its own from the same per-target sums, as
    /// [`crate::EwRidgeCfg::feature_sets`] are; output slots are
    /// target-major, then set. Empty: one fit over every feature. Huber
    /// only. A row's Huber weight is one number per target, since the sets
    /// share the target's Gram, so it is read from the residual of the
    /// largest set ([`Self::weighing_set`]): weighting each set by its own
    /// residual would need a Gram per set (docs/PLAN.md task 272 (a)).
    /// State since schema 63, empty in a state written before it.
    #[serde(default)]
    pub feature_sets: Vec<(String, Vec<usize>)>,
    /// RLS's decaying prior on the sum scale in place of the ridge on the
    /// means, as [`crate::EwRidgeCfg::ridge_scale`]: each target solves
    /// `(W_j S_j + p_j · ridge · I) β = W_j r_j` over its raw moments, `W_j`
    /// the weight its sums hold (Huber's reweighted rows) and `p_j` the
    /// product of the decay factors they have aged by, so the prior fades as
    /// the rows arrive and the intercept is penalized with the slopes. At
    /// `huber_delta = inf` that is `rls`'s fit, and `ewridge`'s under the
    /// same setting. Huber only, and refused beside `standardize` and two or
    /// more feature sets, as `ewridge`'s is (docs/PLAN.md task 272 (a)).
    /// State since schema 63, false in a state written before it.
    #[serde(default)]
    pub ridge_scale: bool,
    /// Report each target's residual scale `s` with every prediction, as
    /// [`crate::Extra::Huber`]: the scale the row's Huber cut is drawn in,
    /// read before the row's own residual enters it, so a caller can use it
    /// as a weight or a scoring scale that is causal by construction. NaN
    /// where the target has no scale yet (the module docs). Huber only
    /// (docs/PLAN.md task 272 (a)). State since schema 63, false in a state
    /// written before it.
    #[serde(default)]
    pub emit_scale: bool,
    /// Scratch (prototype, branch `huber-warmup`; docs/HUBER.md §3): hold
    /// each target's first `warm_rows` learned rows (weight above 0), and on
    /// the row the count is reached refit them by IRLS to convergence --
    /// every pass's Huber weights from the residuals against the previous
    /// pass's fit, in a scale re-estimated from those residuals -- and
    /// rebuild the target's sums from the converged weights, then go on
    /// online. 0 is off. Huber only, not beside `ridge_scale`. Skipped when
    /// 0, so a state written without it has the bytes it had.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub warm_rows: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// The most IRLS passes a warm-up refit takes ([`RobustCfg::warm_rows`]).
const WARM_PASSES: usize = 50;
/// A warm-up refit has converged when no coefficient moved by more than
/// this, relative to `1 + max|β|`.
const WARM_TOL: f64 = 1e-10;

/// One target's warm-up buffer ([`RobustCfg::warm_rows`]): from the first
/// row it learned, every row as its sums took it, flat, each `x` (the
/// features), `y`, the raw weight, the row's decay factor and the scale's
/// (`huber_scale_decay`) -- a row that
/// only aged the sums (the target null, or no weight) as zeros, weight 0
/// and its factor, so that the replay ages them factor by factor, as the
/// stream did, and rebuilds them to the bit; the rows learned; and whether
/// the refit has run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WarmBuf {
    rows: Vec<f64>,
    learned: u32,
    done: bool,
}

impl WarmBuf {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            learned: 0,
            done: false,
        }
    }

    /// A row that only aged the target's sums, once a row has been learned.
    fn age(&mut self, nf: usize, lam: f64, lam_scale: f64) {
        if !self.done && self.learned > 0 {
            self.rows.extend(std::iter::repeat_n(0.0, nf));
            self.rows.extend_from_slice(&[0.0, 0.0, lam, lam_scale]);
        }
    }
}

/// A target's band system as a solve factorizes it -- `A`, the band Gram
/// over the kept columns, scaled, with the ridge on its diagonal -- the
/// columns kept and their scales: what a nudge reads its row's full leverage
/// from (review 2026-10-05, TC1b). `factor` is `None` where no column was
/// kept: a step then moves the intercept alone, and through the origin
/// nothing. A scale of a column not kept can be infinite (a variance that
/// overflowed), so the scales are written with [`crate::humanfloat`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BandSystem {
    factor: Option<SpdFactor>,
    keep: Vec<usize>,
    #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
    s: Vec<f64>,
}

/// Each target's [`BandSystem`], kept for the nudges between solves: decay
/// moves neither the band Gram's moments nor `A`, and a row inside the band
/// moves the Gram, and the system with it where the move is exact
/// ([`Robust::move_band_system`]) or drops it, for the next nudge to build
/// again (task 170). Only a quantile fit keeps one: its nudges are the one
/// reader. State since schema 33: a system kept from a solve or built at a
/// nudge holds the Gram's system to the bit, but one a band row moved holds
/// it to rounding ([`SpdFactor::MAX_MOVES`]), and a model restored without
/// it built a fresh one where the saved model read the moved one, so a
/// resumed fit at `ridge = 0` parted from the uninterrupted one by a
/// rounding (`1e-13`). Saved, a restored model reads the factor the saved
/// one held, its moves and its cap with it, and two models with different
/// systems are not equal. A model state written before 33 carries none,
/// and its first nudge builds one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct BandSystems(Vec<Option<BandSystem>>);

impl BandSystems {
    /// Target `j`'s slot among `n`, the slots made on first use (a model
    /// state written before schema 33 carries none).
    fn slot(&mut self, n: usize, j: usize) -> &mut Option<BandSystem> {
        if self.0.len() != n {
            self.0 = vec![None; n];
        }
        &mut self.0[j]
    }
}

/// A nudge's working space, kept between rows so that reading a row's
/// leverage allocates nothing ([`Robust::nudge_movement`]; task 170): the
/// row's scaled deviations, and the column their quadratic form is solved
/// in. Not state, and equal whatever it holds.
#[derive(Debug, Clone, Default)]
struct NudgeScratch {
    u: Vec<f64>,
    work: QuadWork,
}

impl PartialEq for NudgeScratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// What one row does to a target's accumulators ([`Robust::row_update`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RowUpdate {
    /// A weighted least-squares row: `w` into the Gram, `target` into the
    /// cross-moment.
    Fit { w: f64, target: f64 },
    /// A row outside the quantile band: the Gram takes nothing, the
    /// cross-moment takes `nudge * z` (review 2026-09-12, N9).
    Nudge { nudge: f64 },
}

/// Rows per coefficient a quantile fit accumulates as ordinary least squares
/// before it starts taking Newton steps ([`Robust::row_update`]). Measured
/// (review 2026-09-12, N9): at one row per coefficient the Gram is near
/// singular and the ridge turns it into coefficients in the hundreds; three
/// was stable at every bandwidth and feature count tried, and the fit it
/// leaves is `QuantReg`'s to within one of its standard errors.
const WARM_ROWS: f64 = 3.0;

impl RobustCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    /// The fits per target: one per feature set, or one over every feature.
    pub fn n_sets(&self) -> usize {
        self.feature_sets.len().max(1)
    }

    /// The set whose prediction a row's Huber weight and the residual scale
    /// read: the one with the most features, the first of them where two
    /// tie, and 0 without sets (the field's docs).
    pub fn weighing_set(&self) -> usize {
        let mut best = 0;
        for (i, (_, idx)) in self.feature_sets.iter().enumerate() {
            if idx.len() > self.feature_sets[best].1.len() {
                best = i;
            }
        }
        best
    }

    /// Set `c`'s features, `None` without sets: every feature.
    fn set_of(&self, c: usize) -> Option<&[usize]> {
        self.feature_sets.get(c).map(|(_, idx)| idx.as_slice())
    }

    /// The model a spec names this loss by, which every message leads with:
    /// `huber` or `quantile`, where it said `robust`, a name no spec has
    /// (review 2026-10-06, CF5; docs/PLAN.md task 196, N1).
    pub fn kind(&self) -> &'static str {
        match self.loss {
            RobustLoss::Huber { .. } => "huber",
            RobustLoss::Quantile { .. } => "quantile",
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let kind = self.kind();
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("{kind}: {e}"))?;
        if self
            .solve_share
            .is_some_and(|f| !(f.is_finite() && f > 0.0))
        {
            return Err("solve_share must be finite and > 0".into());
        }
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.warm_rows > 0
            && (!matches!(self.loss, RobustLoss::Huber { .. }) || self.ridge_scale)
        {
            return Err(format!(
                "{kind}: huber_warm_rows is for huber, and not beside ridge_scale \"sum\""
            ));
        }
        match self.loss {
            RobustLoss::Huber { delta } => {
                if delta <= 0.0 || delta.is_nan() {
                    return Err(format!("{kind}: huber_delta must be > 0, got {delta}"));
                }
                if let Some(d) = self.huber_scale_decay {
                    d.check()
                        .map_err(|e| format!("{kind}: huber_scale_half_life: {e}"))?;
                }
            }
            RobustLoss::Quantile { tau } => {
                if !(0.0..=1.0).contains(&tau) || tau == 0.0 || tau == 1.0 {
                    return Err(format!("{kind}: quantile must be in (0, 1), got {tau}"));
                }
                // The band is drawn in the residuals' EW root mean square,
                // on the model's clock: a scale of Huber's is not this
                // loss's (docs/PLAN.md task 271).
                if self.huber_scale != HuberScale::Rms || self.huber_scale_decay.is_some() {
                    return Err(format!(
                        "{kind}: huber_scale and huber_scale_half_life are for huber"
                    ));
                }
            }
        }
        if !self.feature_sets.is_empty() && !matches!(self.loss, RobustLoss::Huber { .. }) {
            return Err(format!(
                "{kind}: feature_sets are for huber: a quantile fit's nudges read one band \
                 system per target"
            ));
        }
        if self.emit_scale && !matches!(self.loss, RobustLoss::Huber { .. }) {
            return Err(format!(
                "{kind}: emit_huber_scale is for huber: a quantile fit's band has no cut"
            ));
        }
        if self.ridge_scale {
            if !matches!(self.loss, RobustLoss::Huber { .. }) {
                return Err(format!(
                    "{kind}: ridge_scale is for huber: a quantile fit's nudges step the mean-form \
                     system"
                ));
            }
            if self.standardize || self.n_sets() > 1 {
                return Err(format!(
                    "{kind}: ridge_scale is incompatible with standardize and with two or more \
                     feature sets"
                ));
            }
        }
        for (name, idx) in &self.feature_sets {
            if idx.is_empty() {
                return Err(format!("{kind}: feature set {name:?} is empty"));
            }
            if idx.iter().any(|&i| i >= self.n_features) {
                return Err(format!(
                    "{kind}: feature set {name:?} has out-of-range indices"
                ));
            }
            let mut seen = idx.clone();
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) {
                return Err(format!(
                    "{kind}: feature set {name:?} lists a feature more than once"
                ));
            }
        }
        // What the spec layer refuses, refused here too, so the Rust API and
        // a state file are held to it (review 2026-10-06, CC7): a NaN ridge
        // passed `< 0` and failed every solve; a NaN `quantile_eps` passed
        // `<= 0` and read as the band's floor alone, and at `inf` every row
        // was a band row aimed at `y + inf·(tau − 1/2)`.
        if !(self.ridge.is_finite() && self.ridge >= 0.0) {
            return Err(format!(
                "{kind}: ridge must be finite and >= 0, got {}",
                self.ridge
            ));
        }
        if !(self.quantile_eps.is_finite() && self.quantile_eps > 0.0) {
            return Err(format!(
                "{kind}: quantile_eps must be finite and > 0, got {}",
                self.quantile_eps
            ));
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err(format!(
                "{kind}: min_weight must be >= 0, got {}",
                self.min_weight
            ));
        }
        // A NaN cadence never comes due on the clock, and left the row cap
        // alone to schedule the solves, with no word (review round 5, C6).
        if self.solve_every.is_nan() {
            return Err(format!(
                "{kind}: solve_every must not be NaN (<= 0 solves every row)"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Robust {
    cfg: RobustCfg,
    /// One accumulator per target (the robust weights are per target).
    cov: Vec<EwCov>,
    /// Per target, the weight its accumulators hold: Huber's reweighted rows,
    /// or the quantile band's. The mean-form cross-moment `cross` is over it,
    /// and so is `cov`, whose weight it equals.
    wj: Vec<f64>,
    /// Per target, the rows it was present on at their raw weights, decayed:
    /// what the per-target `min_weight` gate reads (hard rule 8, S2) and the
    /// quantile fit's warm-up counts. `wj` stood in for it, and for the
    /// quantile that is the band's weight, which a half-life caps at the
    /// band's share of the sample (the second review of 2026-09-15, F1).
    wobs: Vec<f64>,
    /// Per target, the same rows counted one each, decayed alike: `wobs`
    /// over the rows' mean weight, which the quantile fit's warm-up and band
    /// floor read so that a weight's scale reaches neither (docs/PLAN.md
    /// task 147: rows at weight 100 left the warm-up on their first row,
    /// and the fit reached 1e51).
    nobs: Vec<f64>,
    /// Per target, the centred cross-moment `c_j = E[(z − m_j)(y − ȳ_j)]`
    /// over `wj`, with `m_j` its accumulator's mean, and `ȳ_j` beside it
    /// (`ybar`). With an intercept slot 0 is exactly 0: `z_0 − m_0 = 0` once
    /// a row has entered. They were kept raw, `E[z·y]`, and the solves read
    /// the raw normal equations or centred them by subtraction, which loses
    /// `level²·ε` -- the whole fit at `1e8` -- where `ewridge` and `lasso`
    /// were moved to centred moments in the 2026-09-12 round and this model
    /// was not (the review of 2026-09-18, S2). A quantile nudge, a sum's worth
    /// of `2h·psi·z` over `wj`, enters as `ȳ += nudge/wj` and `c += nudge·(z
    /// − m)/wj`: the raw step `E[z·y] += nudge·z/wj` transformed exactly.
    cross: Vec<Vec<f64>>,
    ybar: Vec<f64>,
    /// EW residual variance per target (drives the robust scale).
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    /// EW count of *observations* using the raw row weights, i.e. ignoring
    /// what the loss does with them. This is what `n_eff` and `min_weight`
    /// mean everywhere else, so the robust models report it too: Huber scales
    /// the accumulators by its weights and a quantile fit weighs only the rows
    /// inside its band, and the observation count must follow neither. (The
    /// IRLS weights a quantile fit once used reached `2 / quantile_eps`, so
    /// counting them inflated `n_eff` by ~1000x -- T-A5.)
    w_raw: f64,
    /// The last solve's coefficients per target: NaN for a target no solve
    /// gave a fit (review round 4, CC1).
    beta: Option<Fit<Vec<Vec<f64>>>>,
    /// Where `solve_every`'s clock stands: the stamp of the last solve, the
    /// decayed clock held exactly (docs/PLAN.md task 180).
    since_solve: Since,
    rows_since_solve: u32,
    /// Weight learned since the last solve, for `solve_share`.
    #[serde(default)]
    weight_since_solve: f64,
    pub solve_failures: u64,
    /// What each `ybar` leaves out: the mean is `ybar[j] + ybar_lo[j]`
    /// ([`crate::comp`]; docs/PLAN.md task 101). One per target.
    ybar_lo: Vec<f64>,
    /// Each target's band system, kept from its last solve or nudge for the
    /// nudges that follow ([`BandSystems`]): state since schema 33 (task
    /// 170), empty in a state written before it. Before the two fields that
    /// are skipped, which must stay last.
    #[serde(default)]
    systems: BandSystems,
    /// Per target, the threshold its own weight is checked against, as a
    /// bank checks it on the output (hard rule 8): the weight at which a
    /// target no solve has fit has its own first solve (docs/PLAN.md task
    /// 195, S9b; [`Self::set_target_min_weight`]). Empty: `min_weight` for
    /// every target.
    #[serde(default)]
    target_min_weight: Vec<f64>,
    /// Per target, each coefficient's data share from the last solve that
    /// fit it, `k_total` long (docs/WARMUP-AND-CONVERGENCE.md §2.2; the
    /// module doc): NaN in the intercept slot and for a target no solve has
    /// fit, 0 for a column the system dropped. Empty before the first solve.
    /// State since schema 45 (docs/PLAN.md task 116).
    #[serde(default)]
    support: Shares,
    /// Per target, the sketch of `|e|` whose median over [`NORMAL_Q3`] is
    /// the Huber cut's scale under [`HuberScale::Mad`]: fed the residuals
    /// and raw weights `sig2` is, and aged on the scale's clock as `wsig`
    /// is. Empty under the RMS, and in a state written before schema 63
    /// (docs/PLAN.md task 271).
    #[serde(default)]
    mad: Vec<EwQuantile>,
    /// Per target, the warm-up buffer ([`RobustCfg::warm_rows`]): empty
    /// when it is off, so a state without it has the bytes it had.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    warm: Vec<WarmBuf>,
    /// The row's augmented values ([`RowBuf`]). Not state.
    #[serde(skip)]
    zbuf: RowBuf,
    /// A nudge's working space ([`NudgeScratch`]). Not state.
    #[serde(skip)]
    nudge: NudgeScratch,
}

/// The row's augmented values, `[1, x]` with an intercept and `x` through
/// the origin, written in full at the start of every row before anything
/// reads them. Not state, and two models are equal whatever it holds -- a
/// restored model's is zeros where the saved one's holds its last row -- as
/// `EwCov`'s scratch is; it compared, and no restored model equalled the one
/// saved (task 170).
#[derive(Debug, Clone, Default)]
struct RowBuf(Vec<f64>);

impl PartialEq for RowBuf {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::ops::Deref for RowBuf {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}

impl std::ops::DerefMut for RowBuf {
    fn deref_mut(&mut self) -> &mut Vec<f64> {
        &mut self.0
    }
}

/// One solve's data shares for one target, taken when something reads them
/// (docs/PLAN.md task 116): `A⁻¹`'s diagonal is `O(k³)` and a solve runs
/// every few rows, so it waits for a `coef` row, the summary, a save or the
/// end of the stream's chunk, as `ewridge`'s does (tasks 140 and 265). The
/// arithmetic is the same whenever it runs.
#[derive(Debug, Clone)]
struct PendingShares {
    factor: SpdFactor,
    /// The ridge plus the jitter the factor needed.
    lam: f64,
    /// Per kept column, in the factor's order, the coefficient slot its
    /// share goes to.
    to: Vec<usize>,
    /// The coefficients a target has, and whether the first is an
    /// intercept.
    k: usize,
    intercept: bool,
    /// The fit's feature set, `None` for every feature: kept only under
    /// sets, so a plain solve allocates nothing for it. A base of `k`
    /// shares allocated at every solve, read or not, cost a 20-feature
    /// `huber` solve 3% (docs/PLAN.md task 272 (e), R5-4).
    set: Option<Vec<usize>>,
    shares: OnceLock<Vec<f64>>,
}

impl PendingShares {
    /// Write into `out` the shares before any column's: NaN in the
    /// intercept's slot, which is not a share, and in a slot outside the
    /// fit's feature set (`set`, `None` for every feature), which has no
    /// coefficient to share, as `ewridge`'s sets have it; and 0 elsewhere, a
    /// column the solve dropped having none of the data's.
    fn fill_base(out: &mut [f64], intercept: bool, set: Option<&[usize]>) {
        let off = usize::from(intercept);
        out.fill(0.0);
        if let Some(set) = set {
            for (i, v) in out.iter_mut().enumerate().skip(off) {
                if !set.contains(&(i - off)) {
                    *v = f64::NAN;
                }
            }
        }
        if intercept && let Some(first) = out.first_mut() {
            *first = f64::NAN;
        }
    }

    /// `1 − λ (A⁻¹)_jj` for each kept column, in its slot (the module doc).
    fn shares(&self) -> &[f64] {
        self.shares.get_or_init(|| {
            let mut out = vec![0.0; self.k];
            Self::fill_base(&mut out, self.intercept, self.set.as_deref());
            let inv = self.factor.inverse_diagonal(self.to.len());
            for (&slot, inv) in self.to.iter().zip(inv) {
                // NaN in the base is no share: the intercept, which the
                // sum-scale system solves with the slopes.
                if !out[slot].is_nan() {
                    out[slot] = (1.0 - self.lam * inv).clamp(0.0, 1.0);
                }
            }
            out
        })
    }
}

/// Each target's data shares (`support_coef`): the ones stored, and per
/// target the last solve's while nothing has read them. A save writes the
/// shares a read would give, as [`Fit`] writes a fit, and two are equal on
/// those values, by their bits (the intercept's is NaN).
#[derive(Debug, Clone, Default)]
struct Shares {
    stored: Vec<Vec<f64>>,
    pending: Vec<Option<PendingShares>>,
}

impl Shares {
    /// `n` targets of `k` coefficients: NaN, no fit, for a target the
    /// table had no room for -- all of them before the first solve.
    fn shape(&mut self, n: usize, k: usize) {
        if self.stored.len() != n || self.stored.iter().any(|v| v.len() != k) {
            self.stored = vec![vec![f64::NAN; k]; n];
        }
        if self.pending.len() != n {
            self.pending = vec![None; n];
        }
    }

    /// Target `j` has no fit, so no shares: NaN.
    fn unfit(&mut self, j: usize) {
        self.stored[j].fill(f64::NAN);
        self.pending[j] = None;
    }

    /// Every target's shares, a pending solve's taken.
    fn settled(&self) -> Vec<Vec<f64>> {
        self.stored
            .iter()
            .enumerate()
            .map(|(j, v)| match self.pending.get(j) {
                Some(Some(p)) => p.shares().to_vec(),
                _ => v.clone(),
            })
            .collect()
    }

    /// Store every pending solve's shares and drop its factor.
    fn settle(&mut self) {
        if self.pending.iter().any(Option::is_some) {
            self.stored = self.settled();
            self.pending.iter_mut().for_each(|p| *p = None);
        }
    }
}

impl Serialize for Shares {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Fit(self.settled()).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Shares {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let stored = Fit::<Vec<Vec<f64>>>::deserialize(d)?.0;
        let pending = vec![None; stored.len()];
        Ok(Self { stored, pending })
    }
}

impl PartialEq for Shares {
    fn eq(&self, other: &Self) -> bool {
        Fit(self.settled()) == Fit(other.settled())
    }
}

impl Robust {
    pub fn new(cfg: RobustCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        Ok(Self {
            // No window here, so no runs (docs/PLAN.md task 128).
            cov: vec![EwCov::new(k).without_runs(); m],
            wj: vec![0.0; m],
            wobs: vec![0.0; m],
            nobs: vec![0.0; m],
            cross: vec![vec![0.0; k]; m],
            ybar: vec![0.0; m],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            w_raw: 0.0,
            beta: None,
            since_solve: Since::default(),
            rows_since_solve: 0,
            weight_since_solve: 0.0,
            solve_failures: 0,
            ybar_lo: vec![0.0; m],
            zbuf: RowBuf(vec![0.0; k]),
            systems: BandSystems(vec![None; m]),
            target_min_weight: Vec::new(),
            support: Shares::default(),
            mad: if matches!(cfg.loss, RobustLoss::Huber { .. }) {
                mad_sketches(cfg.huber_scale, m)
            } else {
                Vec::new()
            },
            nudge: NudgeScratch::default(),
            warm: if cfg.warm_rows > 0 {
                vec![WarmBuf::new(); m]
            } else {
                Vec::new()
            },
            cfg,
        })
    }

    pub fn cfg(&self) -> &RobustCfg {
        &self.cfg
    }

    /// Each target's own `min_weight`, as a bank holds a list of them: one
    /// value `>= 0` per target, or none for `min_weight` everywhere. A
    /// target no solve has fit is solved on the row its own weight first
    /// reaches its own threshold, as a fresh model's first solve fires on
    /// the row its weight reaches `min_weight` (docs/PLAN.md task 195, S9b).
    pub fn set_target_min_weight(&mut self, own: Vec<f64>) -> Result<(), String> {
        crate::model::check_target_min_weight("robust", &own, self.cfg.n_targets)?;
        self.target_min_weight = own;
        Ok(())
    }

    /// Whether target `j`'s own weight, before this row and after it,
    /// crosses into its own threshold while no solve has fit it: the row
    /// its own first solve falls on (S9b). A target's weight grows only on
    /// a row that carries it, `lam·W + w` there as in the update below.
    fn own_first_solve(&self, j: usize, present: bool, lam: f64, weight: f64) -> bool {
        if !present || !(weight > 0.0 && weight.is_finite()) {
            return false;
        }
        let t = crate::model::min_weight_of(&self.target_min_weight, self.cfg.min_weight, j);
        let reached = |w: f64| w >= t && w > 0.0;
        let before = self.wobs[j];
        !reached(before) && reached(lam * before + weight) && !self.fitted(j)
    }

    /// Whether a solve has fit target `j`: its weighing set's slot holds no
    /// NaN ([`RobustCfg::weighing_set`]), the slot its Huber weights read.
    fn fitted(&self, j: usize) -> bool {
        let slot = j * self.cfg.n_sets() + self.cfg.weighing_set();
        self.beta
            .as_ref()
            .is_some_and(|b| b[slot].iter().all(|v| !v.is_nan()))
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Target `j`'s residual scale, once it has one: under
    /// [`HuberScale::Mad`] the EW median of `|e|` over [`NORMAL_Q3`]
    /// ([`mad_scale`]), and otherwise -- the RMS, and every quantile fit --
    /// the EW std `s = √σ²` of its residuals, `σ²` finite and above 0.
    /// `None` before -- no residual yet, or every one so far exactly zero
    /// (under the MAD, half the weight on such residuals) -- where the
    /// Huber cut has nothing to be drawn in (the module docs; task 177).
    fn residual_scale(&self, j: usize) -> Option<f64> {
        if let Some(q) = self.mad.get(j) {
            return mad_scale(q);
        }
        let s2 = self.sig2[j];
        (s2 > 0.0 && s2.is_finite()).then(|| s2.sqrt())
    }

    /// Per target, the sketch of `|e|` the MAD scale is read from: empty
    /// under the RMS.
    pub fn mad_sketch(&self) -> &[EwQuantile] {
        &self.mad
    }

    /// EW count of observations under the raw row weights (`w_raw`).
    pub fn n_eff(&self) -> f64 {
        self.w_raw
    }

    pub fn coefficients(&self) -> Option<&[Vec<f64>]> {
        self.beta.as_deref().map(Vec::as_slice)
    }

    /// What a row does to one target's accumulators (the module docs).
    ///
    /// Huber reweights it: an ordinary least-squares row at `min(1, delta*s/|r|)`,
    /// a weight bounded by 1, and at weight 1 while the target has no scale
    /// to draw the cut in (task 177). The quantile loss linearises it
    /// instead, inside the band or outside it, and the two arms are
    /// [`RowUpdate`]'s; while the target has no scale to draw the band in,
    /// it takes a least-squares row, as in its warm-up.
    ///
    /// `pred` is the prediction the row was scored with, so both stay
    /// out-of-sample, and `scale` is the residual scale, `None` until one
    /// exists ([`Self::residual_scale`]).
    /// `present` is the count of the rows this target was present
    /// on, decayed to the row (`nobs`): under `WARM_ROWS` of them per
    /// coefficient the quantile fit takes ordinary least-squares rows, since a
    /// Newton step needs a Hessian to lean on and a band around a fit built
    /// from a handful of rows is not one. That is the warm-up, and it is what
    /// rebuilds the fit after a gap or a reset has aged the weight away. Past
    /// it the band is at least `(k/present)^(2/5)` of `scale` wide, and
    /// `aged`, the band's own weight decayed to the row in rows of the
    /// target's mean weight, under one row per coefficient is a fit the data
    /// has left behind, which takes least-squares rows until the band holds
    /// rows again (the module docs). Both are counts, so a weight's scale
    /// reaches neither (docs/PLAN.md task 147).
    fn row_update(
        &self,
        yj: f64,
        pred: f64,
        scale: Option<f64>,
        weight: f64,
        present: f64,
        aged: f64,
    ) -> RowUpdate {
        match self.cfg.loss {
            RobustLoss::Huber { delta } => {
                // No prediction is no residual, and no scale is nothing to
                // judge one against: either way the row is down-weighted by
                // nothing.
                let w_rob = match scale {
                    Some(scale) if pred.is_finite() => {
                        let cut = delta * scale;
                        let a = (yj - pred).abs();
                        if a <= cut || a == 0.0 { 1.0 } else { cut / a }
                    }
                    _ => 1.0,
                };
                RowUpdate::Fit {
                    w: weight * w_rob,
                    target: yj,
                }
            }
            RobustLoss::Quantile { tau } => {
                let k = self.cfg.k_total() as f64;
                let least_squares = RowUpdate::Fit {
                    w: weight,
                    target: yj,
                };
                if !pred.is_finite() || present < WARM_ROWS * k || aged < k {
                    return least_squares;
                }
                // A band is a width in units of the scale: with none yet
                // there is no band, and the row is least squares.
                let Some(scale) = scale else {
                    return least_squares;
                };
                let floor = (k / present).powf(0.4);
                let h = scale * self.cfg.quantile_eps.max(floor);
                let r = yj - pred;
                if r.abs() < h {
                    RowUpdate::Fit {
                        w: weight,
                        target: yj + 2.0 * h * (tau - 0.5),
                    }
                } else {
                    let psi = if r > 0.0 { tau } else { tau - 1.0 };
                    RowUpdate::Nudge {
                        nudge: weight * 2.0 * h * psi,
                    }
                }
            }
        }
    }

    /// The warm-up refit of target `j` ([`RobustCfg::warm_rows`]): IRLS on
    /// its buffered rows to convergence, then its sums rebuilt from the
    /// converged weights. Each pass reads the residuals `r_i` of the rows
    /// against the weighing set's fit (the online fit as it stands on the
    /// first pass, or unit weights where none exists), the scale `s` from
    /// them at the rows' decayed raw weights `v_i` (the weighted median of
    /// `|r|` over [`NORMAL_Q3`], or the weighted root mean square), the
    /// weights `w_i · min(1, δs/|r_i|)`, and replays the rows into fresh sums
    /// in their order, each aged by the decay since the row before it, as
    /// the stream entered them; then solves. `n_eff` and the per-target
    /// counts read raw weights and do not move (hard rule 8); nor does the
    /// residual scale's sketch, which keeps the out-of-sample residuals.
    #[inline(never)]
    fn warm_refit(&mut self, j: usize) {
        let RobustLoss::Huber { delta } = self.cfg.loss else {
            return;
        };
        let nf = self.cfg.n_features;
        let stride = nf + 4;
        let buf = std::mem::take(&mut self.warm[j].rows);
        self.warm[j].done = true;
        let n = buf.len() / stride;
        let row = |i: usize| &buf[i * stride..(i + 1) * stride];
        // Each row's raw weight decayed to this row, the last row this one:
        // `v_i = w_i · Π_{l > i} λ_l` (0 on a row that only aged the sums).
        let mut v = vec![0.0; n];
        let mut age = 1.0;
        for i in (0..n).rev() {
            v[i] = row(i)[nf + 1] * age;
            age *= row(i)[nf + 2];
        }
        let k = self.cfg.k_total();
        let c = self.cfg.weighing_set();
        let slot = j * self.cfg.n_sets() + c;
        let mut beta: Option<Vec<f64>> = if self.fitted(j) {
            self.beta.as_ref().map(|b| b[slot].clone())
        } else {
            None
        };
        let mut z = vec![0.0; k];
        let fill = |z: &mut [f64], x: &[f64], intercept: bool| {
            if intercept {
                z[0] = 1.0;
                z[1..].copy_from_slice(x);
            } else {
                z.copy_from_slice(x);
            }
        };
        let intercept = self.cfg.fit_intercept;
        let mut absr = vec![0.0; n];
        let mut u = vec![1.0; n];
        for _ in 0..WARM_PASSES {
            u.fill(1.0);
            if let Some(b) = &beta {
                for (i, a) in absr.iter_mut().enumerate() {
                    let r = row(i);
                    *a = (r[nf] - dot_aug(b, &r[..nf], intercept)).abs();
                }
                let s = match self.cfg.huber_scale {
                    HuberScale::Mad => weighted_median(&absr, &v) / NORMAL_Q3,
                    HuberScale::Rms => {
                        let (num, den) = absr
                            .iter()
                            .zip(&v)
                            .fold((0.0, 0.0), |(p, q), (&a, &w)| (p + w * a * a, q + w));
                        (num / den).sqrt()
                    }
                };
                if s > 0.0 && s.is_finite() {
                    let cut = delta * s;
                    for (ui, &a) in u.iter_mut().zip(&absr) {
                        if a > cut && a != 0.0 {
                            *ui = cut / a;
                        }
                    }
                }
            }
            // Fresh sums, the rows replayed in order (`step`'s fit arm).
            self.cov[j] = EwCov::new(k).without_runs();
            self.cross[j].fill(0.0);
            self.ybar[j] = 0.0;
            self.ybar_lo[j] = 0.0;
            self.wj[j] = 0.0;
            for (i, &ui) in u.iter().enumerate() {
                let r = row(i);
                let (y, lam) = (r[nf], r[nf + 2]);
                let w = r[nf + 1] * ui;
                let aged = lam * self.wj[j];
                // A row that only aged the sums has weight 0 (rule 9).
                if w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    continue;
                }
                fill(&mut z, &r[..nf], intercept);
                let wj_new = aged + w;
                let a = aged / wj_new;
                let bb = w / wj_new;
                let dy = crate::comp::dev(y, self.ybar[j], self.ybar_lo[j]);
                let ab_dy = a * bb * dy;
                let cov = &self.cov[j];
                for (i2, (ci, &zi)) in self.cross[j].iter_mut().zip(z.iter()).enumerate() {
                    *ci = a * *ci + ab_dy * cov.deviation(i2, zi);
                }
                crate::comp::add(&mut self.ybar[j], &mut self.ybar_lo[j], bb * dy);
                self.cov[j].update(&z, lam, w);
                self.wj[j] = wj_new;
            }
            let solved = if intercept {
                self.solve_centred(k, j, c)
            } else {
                self.solve_through_origin(k, j, c)
            };
            let Some((next, _)) = solved else {
                break;
            };
            let moved = beta.as_ref().map_or(f64::INFINITY, |b| {
                b.iter()
                    .zip(&next)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f64::max)
            });
            let size = next.iter().map(|q| q.abs()).fold(0.0, f64::max);
            beta = Some(next);
            if moved <= WARM_TOL * (1.0 + size) {
                break;
            }
        }
        // The residual scale rebuilt from the converged fit's residuals, at
        // the rows' raw weights, aged on the scale's clock as the stream
        // aged it: the out-of-sample residuals it held are those of the
        // early fits, as far from the converged one as the fits were, and a
        // cut drawn in them lets the next rows pull the fit back.
        let Some(b) = beta else {
            return;
        };
        let mut q = mad_sketches(self.cfg.huber_scale, 1).pop();
        let (mut s2, mut ws) = (0.0, 0.0);
        for i in 0..n {
            let r = row(i);
            let (w, lam_s) = (r[nf + 1], r[nf + 3]);
            ws *= lam_s;
            if let Some(q) = q.as_mut() {
                q.age(lam_s);
            }
            if w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let e = r[nf] - dot_aug(&b, &r[..nf], intercept);
            let ws_new = ws + w;
            let next = (ws * s2 + w * e * e) / ws_new;
            if next.is_finite() {
                s2 = next;
                ws = ws_new;
            }
            if let Some(q) = q.as_mut() {
                q.add(e.abs(), w);
            }
        }
        self.sig2[j] = s2;
        self.wsig[j] = ws;
        if let Some(q) = q {
            self.mad[j] = q;
        }
    }

    // Not inlined into `step`: a solve is `O(k³)`, a call costs it nothing,
    // and the row's own path keeps its size (task 272 (e), R5-4).
    #[inline(never)]
    fn solve(&mut self) {
        let k = self.cfg.k_total();
        // A target nothing solved has no fit, NaN, which predicts nothing
        // and which the bank writes as null: one with no weight in its
        // accumulators that no solve has fit yet, and one whose first solve
        // fails. Zeros, what `beta` started at, were predicted as exactly
        // 0.0 until the next solve once the target's rows arrived -- 56 rows
        // of 80 under `solve_every = 1000` -- since this solve spends the
        // first-solve trigger (review round 4, CC1). Such a target waits for
        // the next solve on the cadence. One a solve has fit is solved with
        // no weight too: its mean-form moments hold the history it was fit
        // from, which a gap ages and does not move (hard rule 8), and a
        // weight that underflows to 0 forgets as one just short of it does
        // (docs/PLAN.md task 115 (c)), where it took zeros too.
        // One slot per target and feature set, target-major
        // ([`RobustCfg::feature_sets`]), each solved from its target's sums.
        let nc = self.cfg.n_sets();
        let mut beta = vec![vec![f64::NAN; k]; self.cfg.n_targets * nc];
        // The data shares move in place, slot by slot, as the fit does: a
        // slot this solve skips has none, one it fits the solve's, one whose
        // solve fails keeps its own.
        self.support.shape(self.cfg.n_targets * nc, k);
        for j in 0..self.cfg.n_targets {
            for c in 0..nc {
                let slot = j * nc + c;
                let fit_before = self
                    .beta
                    .as_ref()
                    .is_some_and(|b| !b[slot].iter().any(|v| v.is_nan()));
                if self.wj[j] <= 0.0 && !fit_before {
                    self.support.unfit(slot);
                    continue;
                }
                // `None` is a solve that failed at every jitter: counted, and
                // the previous fit kept. It returned through `?` before the
                // count (review 2026-09-12, S13).
                let solved = if self.cfg.ridge_scale {
                    self.solve_sum(k, j, c)
                } else if self.cfg.fit_intercept {
                    self.solve_centred(k, j, c)
                } else {
                    self.solve_through_origin(k, j, c)
                };
                match solved {
                    Some((sol, shares)) => {
                        beta[slot] = sol;
                        match shares {
                            Some(p) => self.support.pending[slot] = Some(p),
                            None => {
                                // No column kept: nothing to invert, every
                                // slope 0.
                                self.support.pending[slot] = None;
                                PendingShares::fill_base(
                                    &mut self.support.stored[slot],
                                    self.cfg.fit_intercept,
                                    self.cfg.set_of(c),
                                );
                            }
                        }
                    }
                    None => {
                        self.solve_failures += 1;
                        if let Some(prev) = &self.beta {
                            beta[slot] = prev[slot].clone();
                        }
                        // The previous fit is kept, and its shares with it.
                    }
                }
            }
        }
        self.beta = Some(Fit(beta));
        self.since_solve.restart();
        self.rows_since_solve = 0;
        self.weight_since_solve = 0.0;
    }

    /// The solve with an intercept, plain or standardized, on the centred
    /// system, as [`crate::EwRidge`]'s `solve_centred`: the slopes from the
    /// accumulator's centred co-moments over the features and the target's
    /// centred cross-moment, then `beta_0 = ȳ − m·beta`. Plain, nothing is
    /// scaled and nothing dropped, the estimator the raw normal equations
    /// with an unpenalized intercept define; standardized, the co-moments
    /// are scaled to correlation form and a ~zero-variance feature is
    /// dropped (coefficient 0). Nothing level-sized is subtracted from
    /// anything on the way (S2). `None` when every jitter failed.
    fn solve_centred(
        &mut self,
        k: usize,
        j: usize,
        c: usize,
    ) -> Option<(Vec<f64>, Option<PendingShares>)> {
        let kf = k - 1;
        let (asub, keep, s) = self.band_system(j, c);
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut shares = None;
        let mut jitter = 0u32;
        let mut factor = None;
        if kk > 0 {
            let bsub: Vec<f64> = keep.iter().map(|&i| self.cross[j][i + 1] / s[i]).collect();
            // The factor's own solve is `solve_spd`'s, to the bit; it is kept
            // for the nudges that follow (`Self::nudge_movement`).
            let Some(f) = SpdFactor::of(&asub, kk) else {
                *self.systems.slot(self.cfg.n_targets, j) = None;
                return None;
            };
            let sol = f.solve(&bsub, kk, 1);
            jitter = f.attempts();
            for (i2, &i) in keep.iter().enumerate() {
                out[i + 1] = sol[i2] / s[i];
            }
            let (pending, kept) =
                self.pending_shares(f, keep.iter().map(|&i| i + 1).collect(), k, c);
            shares = Some(pending);
            factor = kept;
        }
        let cov = &self.cov[j];
        let mut b0 = self.ybar[j];
        for i in 0..kf {
            b0 -= cov.mean(i + 1) * out[i + 1];
        }
        out[0] = b0;
        self.solve_failures += u64::from(jitter);
        self.keep_band_system(j, BandSystem { factor, keep, s });
        Some((out, shares))
    }

    /// The data shares of the system `f` factorizes, to be taken when
    /// something reads them ([`PendingShares`]): `λ` the ridge plus the
    /// jitter the factor needed, since that is what it carries, and the
    /// coefficient slot each kept column's share goes to. The factor moves
    /// into them; a quantile fit, whose nudges read it as well
    /// ([`BandSystem`]), gets a copy back, and a Huber fit, which keeps no
    /// band system, none -- a copy a solve it would only have dropped.
    fn pending_shares(
        &self,
        f: SpdFactor,
        to: Vec<usize>,
        k: usize,
        c: usize,
    ) -> (PendingShares, Option<SpdFactor>) {
        let kept = matches!(self.cfg.loss, RobustLoss::Quantile { .. }).then(|| f.clone());
        let lam = self.cfg.ridge + f.jitter();
        let pending = PendingShares {
            factor: f,
            lam,
            to,
            k,
            intercept: self.cfg.fit_intercept,
            set: self.cfg.set_of(c).map(<[usize]>::to_vec),
            shares: OnceLock::new(),
        };
        (pending, kept)
    }

    /// Take the data shares of every solve nothing has read yet, and drop
    /// the factors they wait on: what the stream calls at the end of each
    /// run, as it does `ewridge`'s (docs/PLAN.md task 140). The values are
    /// the ones a read would have given.
    pub fn settle_readiness(&mut self) {
        self.support.settle();
    }

    /// Targets whose data shares still hold the factor of the solve they
    /// come from: 0 once [`Self::settle_readiness`] has run.
    pub fn pending_readiness(&self) -> usize {
        self.support.pending.iter().flatten().count()
    }

    /// Keep target `j`'s band system from its solve for the nudges that
    /// follow. Only a quantile fit nudges, so only a quantile fit keeps
    /// one; a Huber fit's would be state that nothing reads (task 170).
    fn keep_band_system(&mut self, j: usize, sys: BandSystem) {
        let kept = matches!(self.cfg.loss, RobustLoss::Quantile { .. }).then_some(sys);
        *self.systems.slot(self.cfg.n_targets, j) = kept;
    }

    /// Whether the band systems a state holds fit the model (task 170):
    /// none, as a state written before schema 33 holds, or one slot per
    /// target. A system is built, kept or moved only from the Gram as it
    /// stands, and only a band row moves the Gram, moving or dropping the
    /// system with it, so a system's columns and scales are the ones
    /// [`Self::band_scales`] gives from the Gram it is saved beside, to the
    /// bit; and it holds a factor exactly when a column is kept, of their
    /// number. Nothing is factorized: the factor's own entries are checked
    /// where they are read ([`SpdFactor`]'s state form).
    fn band_systems_fit(&self) -> Result<(), String> {
        let n = self.cfg.n_targets;
        if !self.systems.0.is_empty() && self.systems.0.len() != n {
            return Err(format!(
                "{} band systems for {n} targets",
                self.systems.0.len()
            ));
        }
        for (j, sys) in self.systems.0.iter().enumerate() {
            let Some(sys) = sys else {
                continue;
            };
            let (s, keep) = self.band_scales(j);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            if keep != sys.keep || bits(&s) != bits(&sys.s) {
                return Err(format!(
                    "target {j}'s band system is not of its Gram's columns and scales"
                ));
            }
            let fits = match &sys.factor {
                Some(f) => !keep.is_empty() && f.order() == keep.len(),
                None => keep.is_empty(),
            };
            if !fits {
                return Err(format!(
                    "target {j}'s band factor is not of its {} kept columns",
                    keep.len()
                ));
            }
        }
        Ok(())
    }

    /// Target `j`'s band system as its solve factorizes it, from the band
    /// Gram as it stands: `A` row-major over the kept columns, the columns
    /// kept and their scales. With an intercept, the centred Gram over the
    /// features (column `i` is slot `i + 1`), scaled by the centred
    /// deviations under `standardize` and keeping the columns whose variance
    /// is usable; through the origin, the raw Gram over every slot, scaled
    /// by the raw second moments under `standardize` and keeping the columns
    /// with one. The ridge is on `A`'s diagonal either way. Under feature
    /// sets, set `c`'s columns alone (a quantile fit has none, and its
    /// nudges pass 0).
    fn band_system(&self, j: usize, c: usize) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let cov = &self.cov[j];
        let (s, mut keep) = self.band_scales(j);
        if let Some(set) = self.cfg.set_of(c) {
            keep.retain(|i| set.contains(i));
        }
        // Read straight from the Gram, where the centred block was copied
        // out first, one allocation and `k²` copies a system (task 170).
        let kk = keep.len();
        let mut asub = vec![0.0; kk * kk];
        for (i2, &i) in keep.iter().enumerate() {
            for (j2, &jj) in keep.iter().enumerate() {
                let g = if self.cfg.fit_intercept {
                    cov.cov(i + 1, jj + 1)
                } else {
                    cov.raw(i, jj)
                };
                asub[i2 * kk + j2] = g / (s[i] * s[jj]);
            }
            asub[i2 * kk + i2] += self.cfg.ridge;
        }
        (asub, keep, s)
    }

    /// Target `j`'s band system's scales and kept columns, from the band
    /// Gram as it stands ([`Self::band_system`]): with an intercept, the
    /// centred deviations of the features under `standardize`, keeping the
    /// columns whose variance is usable; through the origin, the raw second
    /// moments' roots under `standardize`, keeping the columns with one;
    /// unstandardized, scales of 1 and every column. One rule for the
    /// system a nudge builds and the one a band row moves.
    fn band_scales(&self, j: usize) -> (Vec<f64>, Vec<usize>) {
        let k = self.cfg.k_total();
        let cov = &self.cov[j];
        if self.cfg.fit_intercept {
            let kf = k - 1;
            if self.cfg.standardize {
                let s = (0..kf)
                    .map(|i| cov.cov(i + 1, i + 1).max(0.0).sqrt())
                    .collect();
                let keep = (0..kf)
                    .filter(|&i| {
                        let var = cov.cov(i + 1, i + 1);
                        crate::variance_is_usable(var, cov.raw(i + 1, i + 1))
                    })
                    .collect();
                (s, keep)
            } else {
                (vec![1.0; kf], (0..kf).collect())
            }
        } else {
            let s: Vec<f64> = if self.cfg.standardize {
                (0..k).map(|i| cov.raw(i, i).max(0.0).sqrt()).collect()
            } else {
                vec![1.0; k]
            };
            // No centering here, so no cancellation: any strictly positive raw
            // moment is usable.
            let keep = (0..k).filter(|&i| s[i] > 0.0).collect();
            (s, keep)
        }
    }

    /// How far a unit step moves the nudged row's prediction at the next
    /// solve, the row in `zbuf`: `1 + uᵀA⁻¹u` with an intercept, `u` the
    /// row's centred deviations over the kept columns, scaled, and `uᵀA⁻¹u`
    /// through the origin, `u` its scaled values -- the row's full leverage
    /// against the solve's own system (`Self::band_system`), ridge included,
    /// so a step bounded by it moves the row by no more than the bound. The
    /// diagonal's `Σ dᵢ²/vᵢ` read a row at (+1, −1) against features
    /// correlated at 0.999 at about 2, where its full leverage is about
    /// 2,000, and the solve threw such a row 6 to 13 times its residual
    /// past its target, 30 to 127 times at 0.9999 (review 2026-10-05,
    /// TC1b). The factor is kept, from the last solve or the last nudge
    /// that made one, and a nudge costs `O(k²)`. A row inside the band moves
    /// the Gram, and the factor with it in `O(k²)` where that is exact
    /// ([`Self::move_band_system`]); elsewhere it drops the factor, and the
    /// first nudge after it makes the `O(k³)` factorization again. A kept
    /// feature with no spread in the band, on which the row deviates, has no
    /// curvature to lean on: the movement is unbounded and the step 0 (G2),
    /// as for a system no jitter factorizes, and for a form that is not a
    /// number (task 181).
    fn nudge_movement(&mut self, j: usize) -> f64 {
        let n = self.cfg.n_targets;
        if self.systems.slot(n, j).is_none() {
            let (a, keep, s) = self.band_system(j, 0);
            let kk = keep.len();
            let factor = if kk > 0 {
                let Some(f) = SpdFactor::of(&a, kk) else {
                    return f64::INFINITY;
                };
                Some(f)
            } else {
                None
            };
            *self.systems.slot(n, j) = Some(BandSystem { factor, keep, s });
        }
        let sys = self.systems.0[j]
            .as_ref()
            .expect("the system is built above");
        let cov = &self.cov[j];
        let intercept = self.cfg.fit_intercept;
        // The deviations and their form in the kept working space: a nudge
        // allocated both, and the form's matrix and vector, every row (task
        // 170).
        let u = &mut self.nudge.u;
        u.clear();
        for &i in &sys.keep {
            let (d, spread) = if intercept {
                (
                    cov.deviation(i + 1, self.zbuf[i + 1]),
                    cov.cov(i + 1, i + 1),
                )
            } else {
                (self.zbuf[i], cov.raw(i, i))
            };
            if (spread.is_nan() || spread <= 0.0) && d != 0.0 {
                return f64::INFINITY;
            }
            u.push(d / sys.s[i]);
        }
        let q = match &sys.factor {
            Some(f) => f.quad_form(u, &mut self.nudge.work),
            None => 0.0,
        };
        // A form that is not a number -- terms past the double's range at
        // `+∞` and `−∞` -- reads no leverage either: unbounded, the step 0.
        // `f64::max` took it to 0, a leverage of 1 with an intercept and 0
        // through the origin, and the step was bounded by the whole residual
        // or by nothing; NaN would bound nothing, `|r| / NaN` failing every
        // comparison (task 181).
        if q.is_nan() {
            return f64::INFINITY;
        }
        if intercept { 1.0 + q } else { q }
    }

    /// The step a row inside the band makes to target `j`'s kept band
    /// system, read before the Gram moves: `v` with `A' = a·A + v vᵀ` before
    /// any rescaling, or `None` where no move is exact or wanted (task 170).
    ///
    /// The row moves the mean-form Gram as `EwCov::update` does, with the
    /// row's `a` and `b` and its deviations `d` from the means before it:
    /// the centred `C' = a·C + a·b·d dᵀ`, and through the origin, where the
    /// system is the raw Gram `R = C + m mᵀ` and `m' = m + b·d`, `R' = a·R +
    /// b·z zᵀ`, since `a + b = 1`. The scaled system is `A = D⁻¹ C D⁻¹ + r·I`
    /// (`R` through the origin), `D` the scales, so with `ũ = D⁻¹d` (`D⁻¹z`)
    /// `A' = E (a·(A − r·I) + v vᵀ) E + r·I` with `v = √(a·b)·ũ` (`√b·ũ`)
    /// and `E = D'⁻¹D`, the scales' change, the identity unstandardized. A
    /// ridge `r > 0` leaves a diagonal shift `(1 − a)·r·I` over the move,
    /// which no `O(k²)` move makes, so the move is exact only at `r = 0`.
    /// It is wanted only by the quantile loss, whose nudges read the system;
    /// a Huber fit drops it, as every model did before.
    fn band_step(&self, j: usize, a: f64, b: f64) -> Option<Vec<f64>> {
        if self.cfg.ridge != 0.0 || !matches!(self.cfg.loss, RobustLoss::Quantile { .. }) {
            return None;
        }
        let sys = self.systems.0.get(j)?.as_ref()?;
        if !sys.factor.as_ref()?.can_move() {
            return None;
        }
        let cov = &self.cov[j];
        Some(if self.cfg.fit_intercept {
            let root = (a * b).sqrt();
            sys.keep
                .iter()
                .map(|&i| root * (cov.deviation(i + 1, self.zbuf[i + 1]) / sys.s[i]))
                .collect()
        } else {
            let root = b.sqrt();
            sys.keep
                .iter()
                .map(|&i| root * (self.zbuf[i] / sys.s[i]))
                .collect()
        })
    }

    /// Target `j`'s kept band system after a row inside the band moved its
    /// Gram, the row's step read before by [`Self::band_step`]: moved in
    /// `O(k²)` -- the factor updated by `a` and the step, then rescaled by
    /// the scales' change under `standardize` -- where the move is exact,
    /// else dropped for the next nudge to build again. Unstandardized the
    /// scales and columns cannot change; standardized, a column the Gram
    /// now keeps or no longer keeps changes the system's shape, which no
    /// move makes. A factor that refuses its move -- jittered, at its cap of
    /// moves, or left with a pivot that is not a positive number -- is
    /// dropped too (`SpdFactor::updated`).
    fn move_band_system(&mut self, j: usize, a: f64, step: Option<Vec<f64>>) {
        let n = self.cfg.n_targets;
        let moved = step.and_then(|mut v| {
            let mut sys = self.systems.slot(n, j).take()?;
            let mut factor = sys.factor.take()?.updated(a, &mut v)?;
            if self.cfg.standardize {
                let (s, keep) = self.band_scales(j);
                if keep != sys.keep {
                    return None;
                }
                let e: Vec<f64> = keep.iter().map(|&i| sys.s[i] / s[i]).collect();
                factor = factor.congruent(&e)?;
                sys.s = s;
            }
            sys.factor = Some(factor);
            Some(sys)
        });
        *self.systems.slot(n, j) = moved;
    }

    /// The solve under `ridge_scale` (`RobustCfg::ridge_scale`): RLS's
    /// system on the sum scale over the raw moments of the slots set `c`
    /// keeps, the intercept's among them, `(W·E[z zᵀ] + p·ridge·I) β =
    /// W·E[z·y]`, with `E[z·y] = c + m·ȳ` one step from what is kept, as
    /// [`Self::solve_through_origin`] forms it. `ewridge`'s `ridge_scale`
    /// solve, on this target's own sums. The data shares read `λ = p·ridge`
    /// plus the factor's jitter, as `ewridge`'s do; the intercept's is NaN.
    /// `None` when every jitter failed.
    fn solve_sum(
        &mut self,
        k: usize,
        j: usize,
        c: usize,
    ) -> Option<(Vec<f64>, Option<PendingShares>)> {
        let cov = &self.cov[j];
        let w = cov.n_eff();
        let prior = cov.prior_scale() * self.cfg.ridge;
        let off = usize::from(self.cfg.fit_intercept);
        let keep: Vec<usize> = (0..k)
            .filter(|&i| {
                i < off
                    || self
                        .cfg
                        .set_of(c)
                        .is_none_or(|set| set.contains(&(i - off)))
            })
            .collect();
        let kk = keep.len();
        let mut a = vec![0.0; kk * kk];
        for (p, &i) in keep.iter().enumerate() {
            for (q, &jj) in keep.iter().enumerate() {
                a[p * kk + q] = w * cov.raw(i, jj);
            }
            a[p * kk + p] += prior;
        }
        let ybar = self.ybar[j];
        let b: Vec<f64> = keep
            .iter()
            .map(|&i| w * (self.cross[j][i] + cov.mean(i) * ybar))
            .collect();
        let f = SpdFactor::of(&a, kk)?;
        let sol = f.solve(&b, kk, 1);
        self.solve_failures += u64::from(f.attempts());
        let mut out = vec![0.0; k];
        for (p, &i) in keep.iter().enumerate() {
            out[i] = sol[p];
        }
        let lam = prior + f.jitter();
        let pending = PendingShares {
            factor: f,
            lam,
            to: keep,
            k,
            intercept: self.cfg.fit_intercept,
            set: self.cfg.set_of(c).map(<[usize]>::to_vec),
            shares: OnceLock::new(),
        };
        Some((out, Some(pending)))
    }

    /// The solve through the origin, on the raw system: every slot is a
    /// slope and every slot is penalized, and the right-hand side is the
    /// uncentred `E[z·y] = c + m·ȳ`, one step from what is kept. Nothing is
    /// centred through the origin -- a level cannot be absorbed there -- so
    /// the raw form is the fit asked for, as `EwRidge`'s no-intercept branch
    /// has it. Standardized, the system is scaled by the raw second-moment
    /// diagonals (a slot with none is dropped): this centred the Gram and
    /// kept the raw right-hand side once, the hybrid system C8 found in
    /// `lasso`, least squares only when every feature has mean zero (review
    /// 2026-09-12, C11). `None` when every jitter failed.
    fn solve_through_origin(
        &mut self,
        k: usize,
        j: usize,
        c: usize,
    ) -> Option<(Vec<f64>, Option<PendingShares>)> {
        let cov = &self.cov[j];
        let ybar = self.ybar[j];
        let b: Vec<f64> = (0..k)
            .map(|i| self.cross[j][i] + cov.mean(i) * ybar)
            .collect();
        let (asub, keep, s) = self.band_system(j, c);
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut shares = None;
        let mut jitter = 0u32;
        let mut factor = None;
        if kk > 0 {
            let bsub: Vec<f64> = keep.iter().map(|&i| b[i] / s[i]).collect();
            // Kept for the nudges that follow, as in `Self::solve_centred`.
            let Some(f) = SpdFactor::of(&asub, kk) else {
                *self.systems.slot(self.cfg.n_targets, j) = None;
                return None;
            };
            let sol = f.solve(&bsub, kk, 1);
            jitter = f.attempts();
            for (i2, &i) in keep.iter().enumerate() {
                out[i] = sol[i2] / s[i];
            }
            let (pending, kept) = self.pending_shares(f, keep.clone(), k, c);
            shares = Some(pending);
            factor = kept;
        }
        self.solve_failures += u64::from(jitter);
        self.keep_band_system(j, BandSystem { factor, keep, s });
        Some((out, shares))
    }
}

impl OnlineModel for Robust {
    fn set_solve_share(&mut self, share: Option<f64>) {
        self.cfg.solve_share = share;
    }

    fn solve_share(&self) -> Option<f64> {
        self.cfg.solve_share
    }

    /// The solve cadence measures its clock by the row's stamp (task 180).
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        self.since_solve.stamp_next(stamp);
    }

    /// Each coefficient's data share from the last solve, per target
    /// (docs/WARMUP-AND-CONVERGENCE.md §2.2): `None` before the first solve.
    fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        self.beta.as_ref()?;
        Some(self.support.settled())
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.wobs);
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): learned as a row of weight 0, so counted nowhere,
        // `n_eff` and the target's present weight included. A feature that
        // is not a number arrived at the loss as a least-squares row, with
        // no finite prediction, and was learned there (task 182, which then
        // refused it in that arm but counted it).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = RowBuf(vec![0.0; k]);
        }
        let lam = self.cfg.decay.factor(d_clock);
        // The scale's own clock (`huber_scale_decay`), the model's unless
        // given: the same factor, to the bit, where it is not.
        let lam_scale = self
            .cfg
            .huber_scale_decay
            .map_or(lam, |d| d.factor(d_clock));
        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict (state before the update) ----
        let mut out = self.predict(x, d_clock);
        let pred = &out.pred;
        // Each target's scale as this row's cut reads it, after its decay
        // and before its residual (`RobustCfg::emit_scale`).
        let emit = self.cfg.emit_scale;
        let mut seen = if emit { vec![f64::NAN; m] } else { Vec::new() };
        // The slot whose prediction weighs the row, per target: under
        // feature sets the largest set's, since the sets share the target's
        // sums and a row enters them at one weight (`RobustCfg::feature_sets`).
        let (nc, weighing) = (self.cfg.n_sets(), self.cfg.weighing_set());

        // ---- the solve schedule, decided before the targets learn the row:
        // it reads nothing they move, and a band row whose system the solve
        // at the end of the row rebuilds need not move it first (task 170) ----
        self.w_raw = lam * self.w_raw + weight;
        // The clock since the last solve, on the row's stamp (task 180).
        self.since_solve.step(d_clock);
        // A zero-weight row is clock alone (hard rule 9): no row of the row
        // cap, and never a solve. One the clock or the weight's share brings
        // due on it is due on the next row with weight too, so it waits for
        // that row, and the solves fall where they fall without it
        // (`ewridge`'s rule, docs/PLAN.md task 214: a clock-due solve fired
        // on the zero-weight row, and the fit read until the next solve
        // moved 8.4e-8 under `huber`, 1.0e-7 under `quantile`). Such a row
        // takes no band step either: it moves no target's Gram.
        // A weight here is usable, finite and `>= 0` (`OnlineModel::step`;
        // `refused_step` steps a row whose weight is not as a row of 0).
        let teaches = weight > 0.0;
        if teaches {
            self.rows_since_solve += 1;
            self.weight_since_solve += weight;
        }
        let by_cadence = teaches
            && match self.cfg.solve_share {
                Some(share) => self.weight_since_solve >= share * self.w_raw,
                None => {
                    self.cfg.solve_every <= 0.0 || self.since_solve.reached(self.cfg.solve_every)
                }
            };
        // A fresh model's first solve fires on the row its weight reaches
        // `min_weight`; so does a target's own, on the row its own weight
        // first reaches its own, where no solve has fit it -- a target that
        // joins after the first solve. It waited for the cadence's next
        // solve, null until then: under `solve_every = 1000` all 56 of its
        // rows of 80 (review round 4, CC1; docs/PLAN.md task 195, S9b).
        let due = teaches
            && (by_cadence
                || self.rows_since_solve >= self.cfg.max_rows_between_solves
                || (self.beta.is_none() && self.w_raw >= self.cfg.min_weight)
                || (0..m).any(|j| self.own_first_solve(j, y[j].is_some(), lam, weight)));

        // ---- update: Huber reweights the row, the quantile linearises it ----
        // Targets whose warm-up buffer this row fills (`RobustCfg::warm_rows`).
        let mut refits: Vec<usize> = Vec::new();
        for j in 0..m {
            // `σ²`'s weight ages on every row, as `wj` does, and a row adds to
            // it only with a target, a weight and a prediction to measure the
            // residual from. A zero or NaN weight skipped the ageing, and so
            // did a row with no prediction (`min_weight` unmet after a clock
            // gap), so `σ²` -- the scale of every cut -- forgot less across
            // either than across a null (review 2026-09-12, S13; N6).
            self.wsig[j] *= lam_scale;
            if let Some(q) = self.mad.get_mut(j) {
                q.age(lam_scale);
            }
            if emit {
                seen[j] = self.residual_scale(j).unwrap_or(f64::NAN);
            }
            let present = lam * self.wobs[j];
            let rows = lam * self.nobs[j];
            let Some(yj) = y[j] else {
                if let Some(b) = self.warm.get_mut(j) {
                    b.age(self.cfg.n_features, lam, lam_scale);
                }
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                self.wobs[j] = present;
                self.nobs[j] = rows;
                continue;
            };
            // A row the target is present on counts at its raw weight, whatever
            // the loss then does with it (hard rule 8), and as one row.
            let counts = weight > 0.0 && weight.is_finite();
            self.wobs[j] = present + if counts { weight } else { 0.0 };
            self.nobs[j] = rows + if counts { 1.0 } else { 0.0 };
            let scale = self.residual_scale(j);
            let aged = lam * self.wj[j];
            // The band's weight in rows of the target's mean weight.
            let aged_rows = if present > 0.0 {
                aged * (rows / present)
            } else {
                0.0
            };
            let pj = pred[j * nc + weighing];
            match self.row_update(yj, pj, scale, weight, rows, aged_rows) {
                // NaN is `inf / inf` from an overflowed residual against an
                // overflowed scale; such a row cannot be learned from either.
                // It is a usable row that fails, and counted as present above;
                // a row holding a value that is not usable never gets here.
                RowUpdate::Fit { w, .. } if w.is_nan() || w <= 0.0 => {
                    if let Some(b) = self.warm.get_mut(j) {
                        b.age(self.cfg.n_features, lam, lam_scale);
                    }
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    continue;
                }
                RowUpdate::Fit { w, target } => {
                    // The same `a`/`b` as `EwCov::update` forms from the same
                    // weights, and the deviations from the means *before* the
                    // row, as it takes them -- so the cross-moment is read
                    // before the accumulator moves.
                    let wj_new = aged + w;
                    let a = aged / wj_new;
                    let bb = w / wj_new;
                    let dy = crate::comp::dev(target, self.ybar[j], self.ybar_lo[j]);
                    let ab_dy = a * bb * dy;
                    let cov = &self.cov[j];
                    for (i, (ci, &zi)) in self.cross[j].iter_mut().zip(self.zbuf.iter()).enumerate()
                    {
                        *ci = a * *ci + ab_dy * cov.deviation(i, zi);
                    }
                    crate::comp::add(&mut self.ybar[j], &mut self.ybar_lo[j], bb * dy);
                    // The Gram moves: the band system kept from the last
                    // solve or nudge moves with it where that is exact, from
                    // the row's step read before the means move, and is
                    // dropped elsewhere, or where the solve at the end of
                    // the row builds it afresh (task 170).
                    let step = if due { None } else { self.band_step(j, a, bb) };
                    self.cov[j].update(&self.zbuf, lam, w);
                    self.wj[j] = wj_new;
                    self.move_band_system(j, a, step);
                    if let Some(b) = self.warm.get_mut(j).filter(|b| !b.done) {
                        b.rows.extend_from_slice(x);
                        b.rows.extend_from_slice(&[yj, weight, lam, lam_scale]);
                        b.learned += 1;
                        if b.learned >= self.cfg.warm_rows {
                            refits.push(j);
                        }
                    }
                }
                RowUpdate::Nudge { nudge } => {
                    // Outside the band a row is one term of the score and none
                    // of the Hessian: no weight in the Gram, and `2h*psi(r)*z`
                    // into the cross-moment, which is a mean over `wj` -- so a
                    // sum's worth of nudge enters divided by it (N9). Centred,
                    // that is `ȳ += nudge/wj` and `c += nudge·(z − m)/wj`.
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    // A row of weight 0 learns nothing, the residual variance
                    // included, as the fit arm leaves it above: its nudge is
                    // 0, and `(wsig·σ² + 0)/(wsig + 0)` is not `σ²` to the bit
                    // (hard rule 9; review 2026-10-05, CC5).
                    if weight.is_nan() || weight <= 0.0 {
                        continue;
                    }
                    if nudge.is_finite() && aged > 0.0 {
                        // The step moves the fit at this row by the step
                        // times the row's full leverage against the band
                        // system (`Self::nudge_movement`), unbounded for a
                        // row far outside the data's spread, where the Gram
                        // holds no curvature for it. The linearisation holds
                        // inside the band, so a step past the row's own
                        // residual overshoots what one term of the score can
                        // justify: bounded to it, the row is brought at most
                        // to its target, not thrown past it. A target of
                        // `1e100` and features of `1e100` at weights from
                        // `1e-100` moved a slope to `1e248` this way, and the
                        // prediction at `1e100` read `-inf` (review
                        // 2026-09-26, G2). The leverage was the Gram's
                        // diagonal reading, which correlated features take
                        // far below the solve's (review 2026-10-05, TC1b).
                        let movement = self.nudge_movement(j);
                        let cov = &self.cov[j];
                        let most = (yj - pj).abs() / movement;
                        let raw = nudge / aged;
                        let step = if raw.abs() > most {
                            most.copysign(raw)
                        } else {
                            raw
                        };
                        for (i, (ci, &zi)) in
                            self.cross[j].iter_mut().zip(self.zbuf.iter()).enumerate()
                        {
                            *ci += step * cov.deviation(i, zi);
                        }
                        // A step of nothing is not taken (`crate::comp::add`
                        // says why): a nudge from a row of weight 0.
                        if step != 0.0 {
                            crate::comp::add(&mut self.ybar[j], &mut self.ybar_lo[j], step);
                        }
                    }
                }
            }
            if pj.is_finite() {
                let resid = yj - pj;
                let ws_new = self.wsig[j] + weight;
                let s2 = (self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                // Skipped when it would not be finite: an `inf` scale makes
                // the Huber cut and the quantile band infinite, and every row
                // after it a plain least-squares one for good
                // (docs/IMPROVEMENTS.md C2).
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
                // The MAD's sketch takes the same residual at the same raw
                // weight; a weight of 0 or an `|e|` that is not finite adds
                // nothing (`EwQuantile::add`), so a zero-weight row only
                // ages it (hard rule 9).
                if let Some(q) = self.mad.get_mut(j) {
                    q.add(resid.abs(), weight);
                }
            }
        }

        // The warm-up refits, after the row's own update and scale: the row
        // was predicted before either, so it stays out of sample (hard rule
        // 2). A refit rebuilds the sums, so the row solves.
        for &j in &refits {
            self.warm_refit(j);
        }
        if due || !refits.is_empty() {
            self.solve();
        }
        if emit {
            out.extra = Some(crate::Extra::Huber { scale: seen });
        }
        out
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let n_eff = self.w_raw;
        let nc = self.cfg.n_sets();
        let mut pred = vec![f64::NAN; self.cfg.n_targets * nc];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_weight, &self.beta) {
            for (slot, p) in pred.iter_mut().enumerate() {
                if self.wj[slot / nc] > 0.0 {
                    *p = dot_aug(&beta[slot], x, self.cfg.fit_intercept);
                }
            }
        }
        // Each target's scale as it stands: a row scored and not learned
        // (held under an embargo) reads it so (`RobustCfg::emit_scale`).
        let extra = self.cfg.emit_scale.then(|| crate::Extra::Huber {
            scale: (0..self.cfg.n_targets)
                .map(|j| self.residual_scale(j).unwrap_or(f64::NAN))
                .collect(),
        });
        Step { pred, n_eff, extra }
    }

    fn state(&self) -> State {
        State::new(ModelState::Robust(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Robust(m) => {
                let mut m = (**m).clone();
                let kind = m.cfg.kind();
                crate::model::check_cfg(kind, m.cfg.validate())?;
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                // One fit per target and feature set (`RobustCfg::feature_sets`).
                let slots = n * m.cfg.n_sets();
                // One accumulator, one cross-moment row and one of each
                // scalar per target, all at the cfg's width; a short one
                // loaded and panicked on the first `step` (review
                // 2026-09-18, B3).
                // Every per-target vector, the row counts and the means' low
                // parts among them: a state written before either was
                // repaired, the counts from the weights and the low parts at
                // zero, and so was a damaged one (docs/PLAN.md task 198).
                let per_target = [
                    &m.wj, &m.wobs, &m.nobs, &m.ybar, &m.ybar_lo, &m.sig2, &m.wsig,
                ];
                if m.cov.len() != n
                    || m.cov.iter().any(|c| !c.has_shape(k))
                    || m.cross.len() != n
                    || m.cross.iter().any(|c| c.len() != k)
                    || per_target.iter().any(|v| v.len() != n)
                    || m.beta
                        .as_ref()
                        .is_some_and(|b| b.len() != slots || b.iter().any(|v| v.len() != k))
                {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the accumulators have the wrong shape"
                    )));
                }
                // The band systems are state (task 170), held as they were
                // saved and never factorized again, so one that does not fit
                // its Gram is refused rather than read.
                if let Err(e) = m.band_systems_fit() {
                    return Err(StateError::Invalid(format!("{kind}: {e}")));
                }
                if let Err(e) =
                    crate::model::check_target_min_weight("robust", &m.target_min_weight, n)
                {
                    return Err(StateError::Invalid(e));
                }
                // The data shares are a solve's: none before the first, and
                // a target's `k_total` each after it (docs/PLAN.md task 116).
                let shares_fit = match &m.beta {
                    None => m.support.stored.is_empty(),
                    Some(_) => {
                        m.support.stored.len() == slots
                            && m.support.stored.iter().all(|v| v.len() == k)
                    }
                };
                if !shares_fit {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the data shares have the wrong shape"
                    )));
                }
                // The MAD's sketches, one per target at 0.5 exactly under
                // it and none otherwise: the next residual indexes a
                // sketch's buckets by its pointers (docs/PLAN.md task 271).
                let scale = match m.cfg.loss {
                    RobustLoss::Huber { .. } => m.cfg.huber_scale,
                    RobustLoss::Quantile { .. } => HuberScale::Rms,
                };
                if !mad_sketches_fit(scale, n, &m.mad) {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the residual sketches do not match huber_scale"
                    )));
                }
                // The warm-up buffers (`RobustCfg::warm_rows`): one per
                // target when it is on, each whole rows short of the count
                // (none once refit), and none when it is off.
                let stride = m.cfg.n_features + 4;
                let warm_fits = if m.cfg.warm_rows == 0 {
                    m.warm.is_empty()
                } else {
                    m.warm.len() == n
                        && m.warm.iter().all(|b| {
                            if b.done {
                                b.rows.is_empty()
                            } else {
                                b.rows.len() % stride == 0
                                    && b.learned < m.cfg.warm_rows
                                    && (b.learned > 0 || b.rows.is_empty())
                            }
                        })
                };
                if !warm_fits {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the warm-up buffers do not match huber_warm_rows"
                    )));
                }
                // No window here: a state written before the runs' flag keeps
                // none from here (review 2026-09-26, C4).
                m.cov.iter_mut().for_each(crate::EwCov::set_runs_off);
                m.zbuf = RowBuf(vec![0.0; k]);
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "robust",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// One slot per target and feature set (`RobustCfg::feature_sets`).
    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.n_sets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = m.state();
        let ModelState::Robust(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cross[0].pop();
        match Robust::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::{EwRidge, EwRidgeCfg};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The stream the model contract's proptest failed on once its values
    /// reached the input bound (review 2026-09-26, G2): a target of `1e100`
    /// once, then features of `1e100` at weights from `1e-100` up. The fit
    /// extrapolated to `-1.5e199` at `1e100`, that residual set the scale to
    /// `5.8e148`, and the next row outside the band nudged the slope to
    /// `1e248` -- one term of the score against a Gram holding no curvature
    /// for a row that far out -- so the prediction at `1e100` read `-inf`.
    /// Bounded by the row's leverage, the nudge brings the row to its band's
    /// edge and every prediction is a number.
    #[test]
    fn a_quantile_fit_stays_finite_through_the_bound() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2, RobustLoss::Quantile { tau: 0.5 });
        c.decay = Decay::Halflife(20.0);
        c.ridge = 1e-6;
        c.min_weight = 3.0;
        let mut m = Robust::new(c).unwrap();
        let rows: Vec<([f64; 2], Option<f64>, f64)> = vec![
            ([0.0, 0.0], None, 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.1785268524789025),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(1e100), 2.0699847121199655),
            ([0.0, 0.0], Some(0.0), 1.4511913482607992),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([1.4589775263789706, 0.0], Some(0.0), 1.0),
            ([1e100, 0.0], Some(0.0), 1e-100),
            ([0.0, 0.0], Some(0.0), 3.8874731502776667),
            ([1e100, 0.0], Some(0.0), 1.0),
            ([1e100, 0.0], None, 1.0),
        ];
        for (i, (x, y0, w)) in rows.iter().enumerate() {
            let out = m.step(x, &[*y0, None], if i == 0 { 0.0 } else { 1.0 }, *w);
            assert!(
                out.pred[0].is_nan() || out.pred[0].is_finite(),
                "row {i}: {:?}",
                out.pred
            );
            // Row 0 carries no target: no fit yet (review round 4, CC1).
            let beta = &m.beta.as_ref().unwrap()[0];
            let none_yet = i == 0 && beta.iter().all(|b| b.is_nan());
            assert!(
                none_yet || beta.iter().all(|b| b.is_finite()),
                "row {i}: {beta:?}"
            );
        }
        // The last nudge brought the row toward its band rather than past
        // it: the fit at `x = 1e100` is within the residual it started from.
        let at = m.predict(&[1e100, 0.0], 1.0).pred[0];
        assert!(at.is_finite() && at.abs() < 1e199, "{at:e}");
    }

    fn cfg(k: usize, m: usize, loss: RobustLoss) -> RobustCfg {
        RobustCfg {
            n_features: k,
            n_targets: m,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss,
            ridge: 1e-8,
            standardize: false,
            min_weight: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            quantile_eps: 1e-3,
            huber_scale: HuberScale::Rms,
            huber_scale_decay: None,
            feature_sets: Vec::new(),
            ridge_scale: false,
            emit_scale: false,
            warm_rows: 0,
        }
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let huber = RobustLoss::Huber { delta: 1.5 };
        let bad = |loss: RobustLoss, f: &dyn Fn(&mut RobustCfg), want: &str| {
            let mut c = cfg(2, 1, loss);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        bad(huber, &|c| c.n_features = 0, "must be >= 1");
        bad(huber, &|c| c.n_targets = 0, "must be >= 1");

        // delta is the crossover from squared to absolute loss.
        for d in [0.0, -1.0, f64::NAN] {
            bad(
                huber,
                &|c| c.loss = RobustLoss::Huber { delta: d },
                "huber_delta",
            );
        }
        cfg(2, 1, RobustLoss::Huber { delta: 1e-9 })
            .validate()
            .unwrap();

        // The quantile is an open interval: 0 and 1 are not quantiles a
        // weighted least-squares reformulation can represent.
        for t in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
            bad(
                huber,
                &|c| c.loss = RobustLoss::Quantile { tau: t },
                "quantile must be in",
            );
        }
        for t in [1e-6, 0.5, 1.0 - 1e-6] {
            cfg(2, 1, RobustLoss::Quantile { tau: t })
                .validate()
                .unwrap();
        }

        // ridge may be zero; quantile_eps may not (it divides). Each named
        // with its value (review 2026-10-06, CC7).
        bad(
            huber,
            &|c| c.ridge = -1e-9,
            "ridge must be finite and >= 0, got -0.000000001",
        );
        let mut ok = cfg(2, 1, huber);
        ok.ridge = 0.0;
        ok.validate().unwrap();
        bad(
            huber,
            &|c| c.quantile_eps = 0.0,
            "quantile_eps must be finite and > 0, got 0",
        );
        bad(
            huber,
            &|c| c.quantile_eps = -1.0,
            "quantile_eps must be finite and > 0, got -1",
        );
    }

    #[test]
    fn n_eff_counts_observations_not_irls_weights() {
        // The defect T-A5 found: the IRLS weights a quantile fit then used
        // reached `2 / quantile_eps`, so counting them made `n_eff` -- and
        // therefore `min_weight` -- meaningless. It must be the plain
        // weighted observation count, identical to every other model's, and it
        // still is now that the fit weighs the rows in its band (N9).
        for loss in [
            RobustLoss::Huber { delta: 1.5 },
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Quantile { tau: 0.9 },
        ] {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(20.0);
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut want = 0.0;
            for i in 0..30 {
                let d = if i == 0 { 0.0 } else { 1.0 };
                let step = m.step(&[i as f64], &[Some(3.0 * i as f64)], d, 1.0);
                assert!(
                    (step.n_eff - want).abs() < 1e-12,
                    "{loss:?} row {i}: {} vs {want}",
                    step.n_eff
                );
                want = want * 0.5f64.powf(d / 20.0) + 1.0;
            }
            assert!(
                want < 31.0,
                "an observation count cannot exceed the row count"
            );
        }
    }

    #[test]
    fn a_solve_failure_is_counted_and_the_previous_fit_is_kept() {
        // Two perfectly collinear features with no ridge: the normal equations
        // are singular. The model must keep its last good coefficients rather
        // than emit NaN, and say so in `solve_failures`.
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.fit_intercept = true;
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            // x1 == x0, and the intercept is constant: rank deficient by two.
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        assert!(
            beta.iter().all(|v| v.is_finite()),
            "never NaN, even singular: {beta:?}"
        );
        // Either the jitter rescued it (counted) or the solve failed (counted);
        // silently succeeding on a singular system is the outcome to rule out.
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` on the
    /// standardized path, as the review wrote it (2026-09-12, S13). The two
    /// collinear features give a singular correlation matrix, which the
    /// jitter ladder rescues and counts; a solve that fails outright is the
    /// next test's.
    #[test]
    fn a_standardized_solve_failure_is_counted_and_the_previous_fit_is_kept() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.standardize = true;
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        assert!(
            beta.iter().all(|v| v.is_finite()),
            "never NaN, even singular: {beta:?}"
        );
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// A standardized solve that fails at every jitter keeps the previous
    /// fit, as the plain one does, and is counted as the plain one is: it
    /// returned through `?` before the count (review 2026-09-12, S13). No
    /// stream of rows gives a correlation matrix that every jitter fails on,
    /// so the accumulator is handed one: a correlation of 2.
    #[test]
    fn a_standardized_solve_that_fails_outright_is_counted() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.standardize = true;
        let mut m = Robust::new(c).unwrap();
        let mut s = 103u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        let before = m.solve_failures;
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            w,
            q,
        );
        m.solve();
        assert_eq!(
            m.solve_failures,
            before + 1,
            "every jitter failed: one failure"
        );
        assert_eq!(
            m.coefficients().unwrap()[0],
            beta,
            "and the previous fit is kept"
        );
    }

    /// `null_targets_decay_the_residual_variance_weight_without_adding_to_it`
    /// (ewridge.rs) with the target present at weight zero: a row that
    /// teaches nothing leaves `σ²` where it was and ages its weight, as a
    /// null row does. The zero weight skipped the ageing, so `σ²` -- the
    /// scale of every Huber cut -- forgot less across such a row than across
    /// a null (review 2026-09-12, S13; S9 was the same in `kalman`).
    #[test]
    fn a_zero_weight_row_ages_the_residual_variance_as_a_null_does() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(10.0);
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 53u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let (sig, w) = (m.sig2[0], m.wsig[0]);
        assert!(sig > 0.0 && w > 0.0);
        let lam = 0.5f64.powf(3.0 / 10.0);
        m.step(&[0.5], &[Some(-500.0)], 3.0, 0.0);
        assert_eq!(m.sig2[0], sig, "weight 0 must not move sigma2");
        assert!(
            (m.wsig[0] - w * lam).abs() < 1e-12,
            "but its weight ages: {} vs {}",
            m.wsig[0],
            w * lam
        );
    }

    /// A row of weight 0 learns nothing, the residual variance included:
    /// across every such row `σ²` keeps its bits, not `(a σ²) / a` --
    /// `kalman`'s twin (task 158) for both losses. A quantile row outside
    /// the band is a nudge, which skipped no zero weight before the `σ²`
    /// update, and moved it by an ulp on some rows (review 2026-10-05,
    /// CC5); the Huber row of weight 0 is a fit of weight 0, which did.
    #[test]
    fn a_zero_weight_row_keeps_the_residual_variance_to_the_bit() {
        for loss in [
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Quantile { tau: 0.9 },
            RobustLoss::Huber { delta: 1.345 },
        ] {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(9.0);
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 107u64;
            let mut checked = 0;
            for i in 0..3000 {
                let x = [lcg(&mut s)];
                let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
                // Far outside the band, where a quantile row is a nudge.
                let zero = i % 3 == 2;
                let (y, w) = if zero { (y + 50.0, 0.0) } else { (y, 1.0) };
                let before = m.sigma2()[0];
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 0.7 }, w);
                if zero && i > 60 {
                    assert!(before > 0.0, "{loss:?}, row {i}");
                    assert_eq!(
                        m.sigma2()[0].to_bits(),
                        before.to_bits(),
                        "{loss:?}, row {i}"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 900, "{loss:?}: {checked}");
        }
    }

    /// `σ²` is the EW mean of the squared out-of-sample errors: every row
    /// ages its weight, and a row with a target, a weight and a prediction
    /// adds `w·r²` (the row weight, not the robust one). Held on a stream
    /// with zero-weight rows (S13) and a clock gap that takes `n_eff` under
    /// `min_weight`, whose next rows have a target and no prediction and
    /// aged nothing either (N6).
    #[test]
    fn the_residual_variance_ages_on_every_row() {
        let hl = 10.0;
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 3.0;
        let mut m = Robust::new(c).unwrap();
        let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
        let mut s = 61u64;
        for i in 0..160 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.2 * lcg(&mut s);
            let d = match i {
                0 => 0.0,
                80 => 200.0,
                _ => 1.0,
            };
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(y), 0.0),
                _ => (Some(y), 1.0),
            };
            let p = m.step(&x, &[y], d, w).pred[0];
            wsig *= 0.5f64.powf(d / hl);
            match y {
                Some(y) if w > 0.0 && p.is_finite() => {
                    let r = y - p;
                    let ws_new = wsig + w;
                    want = (wsig * want + w * r * r) / ws_new;
                    wsig = ws_new;
                }
                Some(_) if w > 0.0 && wsig > 0.0 => unpredicted += 1,
                _ => {}
            }
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
            );
        }
        assert!(
            unpredicted >= 2,
            "the gap must leave rows with no prediction"
        );
    }

    #[test]
    fn huber_resists_outliers_that_break_least_squares() {
        let mut hub = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut ols = EwRidge::new(EwRidgeCfg {
            n_features: 1,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-8],
            feature_sets: vec![],
            standardize: false,
            ridge_scale: false,
            session_shrink: None,
            long_half_life: None,
            coef_prior: None,
            min_weight: 2.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: crate::TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut s = 77u64;
        for i in 0..600 {
            let x = [lcg(&mut s)];
            // clean relationship y = 2x, with 3% enormous outliers
            let outlier = i % 33 == 7;
            let y = if outlier {
                500.0 * lcg(&mut s)
            } else {
                2.0 * x[0]
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            hub.step(&x, &[Some(y)], d, 1.0);
            ols.step(&x, &[Some(y)], d, 1.0);
        }
        let h = hub.coefficients().unwrap()[0][1];
        let o = ols.coefficients().unwrap()[0][1];
        assert!(
            (h - 2.0).abs() < (o - 2.0).abs(),
            "huber {h} should beat ols {o} (truth 2.0)"
        );
        assert!((h - 2.0).abs() < 0.5, "huber slope {h}");
    }

    #[test]
    fn huber_matches_least_squares_without_outliers() {
        // With a huge delta nothing is downweighted, so it must reduce to OLS.
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1e9 })).unwrap();
        let mut s = 78u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.5 * x[1] + 0.25;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0];
        assert!((b[0] - 0.25).abs() < 1e-6);
        assert!((b[1] - 1.5).abs() < 1e-6);
        assert!((b[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn quantile_tracks_the_requested_quantile() {
        // y = 1 + noise with an asymmetric spread; tau = 0.9 must sit clearly
        // above tau = 0.5, which must sit above tau = 0.1.
        let mut lo = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.1 })).unwrap();
        let mut mid = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.5 })).unwrap();
        let mut hi = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.9 })).unwrap();
        let mut s = 79u64;
        for i in 0..4000 {
            let x = [lcg(&mut s)];
            let y = 1.0 + 2.0 * lcg(&mut s); // uniform(-1,3) around the level
            let d = if i == 0 { 0.0 } else { 1.0 };
            lo.step(&x, &[Some(y)], d, 1.0);
            mid.step(&x, &[Some(y)], d, 1.0);
            hi.step(&x, &[Some(y)], d, 1.0);
        }
        let (a, b, c) = (
            lo.coefficients().unwrap()[0][0],
            mid.coefficients().unwrap()[0][0],
            hi.coefficients().unwrap()[0][0],
        );
        assert!(a < b && b < c, "quantile levels out of order: {a} {b} {c}");
    }

    /// The weight the per-target gate reads is the rows the target was
    /// present on, decayed -- hard rule 8's -- whatever the loss does with
    /// them. From N9 to the second review's F1 the quantile fit reported its
    /// band's weight, which a half-life caps at the band's share of the
    /// effective sample, so a `min_weight` above that share closed the gate
    /// for good.
    #[test]
    fn the_target_weight_is_the_rows_present_whatever_the_band_holds() {
        let hl = 30.0;
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let (mut want, mut s, mut out) = (0.0f64, 93u64, Vec::new());
        for i in 0..400 {
            let x = [lcg(&mut s)];
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(x[0] + lcg(&mut s)), 0.0),
                _ => (Some(x[0] + lcg(&mut s)), 1.0),
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.target_n_eff_into(&mut out);
            assert!(
                (out[0] - want).abs() <= 1e-12 * want.max(1.0),
                "row {i}: {} reported, {want} present",
                out[0]
            );
            m.step(&x, &[y], d, w);
            want *= 0.5f64.powf(d / hl);
            if y.is_some() && w > 0.0 {
                want += w;
            }
        }
        assert!(want > 30.0, "the stream must have settled: {want}");
    }

    /// The bounded-extremes contract's case (batch 7): a row at the input
    /// bound sets the Gram and the cross-moment at its own scale, every later
    /// row is outside the band, and the nudges, `2h * psi * z` over the
    /// band's weight, cannot move the fit until that weight has decayed to
    /// nothing -- where each is a step that outgrows the band. The warm-up
    /// rebuilt such a fit while it read the band's weight; since F3 moved it
    /// to the rows present, a band holding under one row per coefficient
    /// takes least-squares rows until it holds rows again.
    #[test]
    fn a_band_under_a_row_per_coefficient_takes_least_squares_rows() {
        let hl = 20.0;
        let lam = 0.5f64.powf(1.0 / hl);
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..300 {
            let x = [lcg(&mut s)];
            let y = Some(x[0] + 0.1 * lcg(&mut s));
            m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let mut out = Vec::new();
        m.target_n_eff_into(&mut out);
        assert!(
            out[0] > 6.0 && m.wj[0] > 2.0,
            "settled: {} present, {} in the band",
            out[0],
            m.wj[0]
        );

        // Past the warm-up, with the band holding rows: a row far outside
        // it is a nudge, and the band's weight only decays.
        let x = [0.3];
        let wj = m.wj[0];
        m.step(&x, &[Some(1e6)], 1.0, 1.0);
        assert!(
            (m.wj[0] - lam * wj).abs() <= 1e-12 * wj,
            "a nudge weighs nothing: {} from {wj}",
            m.wj[0]
        );

        // The band starved to under a row per coefficient (`k = 2`): the same
        // row is a least-squares row, weighed at its weight and aimed at `y`.
        // `wj` is set by hand, so the accumulator's own weight is left where
        // it was; the target mean's share is the one `wj` gives.
        m.wj[0] = 1.5;
        let y0 = m.ybar[0];
        m.step(&x, &[Some(1e6)], 1.0, 1.0);
        let aged = lam * 1.5;
        assert!(
            (m.wj[0] - (aged + 1.0)).abs() <= 1e-12,
            "the row's weight enters: {} for {}",
            m.wj[0],
            aged + 1.0
        );
        let want = y0 + 1.0 / (aged + 1.0) * (1e6 - y0);
        assert!(
            (m.ybar[0] - want).abs() <= 1e-9 * want.abs(),
            "the target mean took the row at its target: {} for {want}",
            m.ybar[0]
        );
    }

    /// A level costs the fit nothing (the review of 2026-09-18, S2), as
    /// `ewridge`'s test of the same name says of it: Huber at a `delta` that
    /// makes it least squares, on the same stream at the origin and shifted
    /// by `1e8` -- features and target alike, a price regressed on prices --
    /// gives the same slopes and predictions that differ by the shift, plain
    /// and standardized. The cross-moments were raw, and both solves lost
    /// `L²·ε`, which at `1e8` was the whole fit. The tolerances are the
    /// data's own resolution at `1e8`, `ulp ≈ 1.5e-8`.
    #[test]
    fn a_level_costs_the_fit_nothing() {
        for standardize in [false, true] {
            let run = |level: f64| {
                let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
                c.standardize = standardize;
                c.ridge = 1e-6;
                c.decay = Decay::Halflife(200.0);
                c.min_weight = 10.0;
                let mut m = Robust::new(c).unwrap();
                let mut s = 29u64;
                let mut preds = Vec::new();
                for i in 0..600 {
                    let u = [lcg(&mut s), lcg(&mut s)];
                    let x = [level + u[0], level + u[1]];
                    let y = level + 2.0 * u[0] - u[1] + 0.1 * lcg(&mut s);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0] - level);
                }
                (preds, m.coefficients().unwrap()[0].clone())
            };
            let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
            for i in 1..3 {
                assert!(
                    (b0[i] - b8[i]).abs() < 1e-6,
                    "standardize {standardize}: slope {i}: {} vs {}",
                    b0[i],
                    b8[i]
                );
            }
            let mut worst = 0.0f64;
            for (t, (a, b)) in p0.iter().zip(&p8).enumerate() {
                assert_eq!(a.is_finite(), b.is_finite(), "row {t}: {a} vs {b}");
                if a.is_finite() {
                    worst = worst.max((a - b).abs());
                }
            }
            assert!(
                worst < 1e-5,
                "standardize {standardize}: the predictions part by {worst}"
            );
        }
    }

    /// The same for the quantile loss, whose nudge enters the same centred
    /// cross-moment: `ȳ += nudge/wj`, `c += nudge·(z − m)/wj` (S2).
    #[test]
    fn a_level_costs_the_quantile_fit_nothing() {
        let run = |level: f64| {
            let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.75 });
            c.decay = Decay::Halflife(500.0);
            c.min_weight = 20.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 37u64;
            let mut preds = Vec::new();
            for i in 0..1500 {
                let u = lcg(&mut s);
                let y = level + 2.0 * u + lcg(&mut s);
                let d = if i == 0 { 0.0 } else { 1.0 };
                preds.push(m.step(&[level + u], &[Some(y)], d, 1.0).pred[0] - level);
            }
            (preds, m.coefficients().unwrap()[0].clone())
        };
        let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
        assert!(
            (b0[1] - 2.0).abs() < 0.2,
            "the fixture is not what it claims: {b0:?}"
        );
        assert!(
            (b0[1] - b8[1]).abs() < 1e-6,
            "slope: {} vs {}",
            b0[1],
            b8[1]
        );
        let mut worst = 0.0f64;
        for (a, b) in p0.iter().zip(&p8) {
            assert_eq!(a.is_finite(), b.is_finite());
            if a.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "the predictions part by {worst}");
    }

    /// With an intercept the cross-moment's slot 0 is exactly 0: `z_0 = 1`
    /// and `m_0 = 1` once a row has entered, so nothing level-sized ever sits
    /// in the right-hand side (S2).
    #[test]
    fn the_intercept_slot_of_the_cross_moment_is_exactly_zero() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 41u64;
        for i in 0..50 {
            let x = [1e8 + lcg(&mut s), 1e8 + lcg(&mut s)];
            let y = 1e8 + x[0] - x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            assert_eq!(m.cross[0][0], 0.0, "row {i}");
        }
    }

    /// N9: after the warm-up a row outside the band moves the fit by its
    /// nudge alone -- the Gram's weight does not see it at all.
    #[test]
    fn a_quantile_row_outside_the_band_nudges_and_weighs_nothing() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 90u64;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let gram_before = m.cov[0].n_eff();
        let mut present = Vec::new();
        m.target_n_eff_into(&mut present);
        let (p_before, b_before) = (present[0], m.coefficients().unwrap()[0].clone());
        // Far above the fit, where the band is `quantile_eps * sigma` wide.
        m.step(&[0.5], &[Some(500.0)], 1.0, 1.0);
        m.target_n_eff_into(&mut present);
        let b_after = m.coefficients().unwrap()[0].clone();
        assert!(
            (m.cov[0].n_eff() - gram_before).abs() < 1e-12,
            "a row outside the band weighs nothing in the Gram: {gram_before} -> {}",
            m.cov[0].n_eff()
        );
        assert!(
            (present[0] - p_before - 1.0).abs() < 1e-12,
            "and is a row the target was present on: {p_before} -> {}",
            present[0]
        );
        assert!(
            b_after[0] > b_before[0],
            "tau = 0.9 follows a row above it: {b_before:?} -> {b_after:?}"
        );
        assert!(
            b_after[0] - b_before[0] < 1.0,
            "by a nudge, not by the row itself: {b_before:?} -> {b_after:?}"
        );
    }

    /// Task 147: the same rows at a hundred times the weight make the same
    /// quantile fit. The warm-up and the band's floor read the target's
    /// weight against row counts, so rows at weight 100 left the warm-up on
    /// their first row, and the fit, leaning on a one-row Gram, reached 1e51.
    #[test]
    fn a_quantile_fit_is_the_same_at_a_hundred_times_the_weight() {
        let fit = |scale: f64| {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.8 });
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 97u64;
            let mut preds = Vec::new();
            for i in 0..600 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
                let w = scale * (0.5 + 0.5 * (lcg(&mut s) + 1.0));
                preds.push(
                    m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w)
                        .pred[0],
                );
            }
            preds
        };
        let (one, many) = (fit(1.0), fit(100.0));
        for (i, (a, b)) in one.iter().zip(&many).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + a.abs()),
                "row {i}: {a} against {b}"
            );
        }
    }

    /// N9: under the warm-up a quantile fit is ordinary least squares, bit for
    /// bit -- a Newton step needs a Hessian, and a band around a fit built from
    /// a handful of rows is not one.
    #[test]
    fn a_quantile_fit_warms_up_as_least_squares() {
        let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        qc.min_weight = 0.0;
        let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        lc.min_weight = 0.0;
        let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
        let mut s = 91u64;
        // `WARM_ROWS` per coefficient is nine rows here, so the tenth is the
        // first the quantile fit takes a step on.
        for i in 0..12 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let qs = q.step(&x, &[Some(y)], d, 1.0);
            let ls = l.step(&x, &[Some(y)], d, 1.0);
            if (1..9).contains(&i) {
                assert_eq!(qs.pred, ls.pred, "row {i} is inside the warm-up");
            }
        }
        assert_ne!(
            q.coefficients().unwrap()[0],
            l.coefficients().unwrap()[0],
            "and it parts from least squares once the band is in force"
        );
    }

    #[test]
    fn reweighting_uses_the_prior_residual_only() {
        // An enormous single observation must not be able to fully absorb
        // itself: the weight comes from the prediction made BEFORE the update.
        let mut m = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = 80u64;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.coefficients().unwrap()[0].clone();
        m.step(&[1.0], &[Some(1e6)], 1.0, 1.0);
        let after = m.coefficients().unwrap()[0].clone();
        // it moves, but nowhere near 1e6
        assert!(
            (after[1] - before[1]).abs() < 100.0,
            "{:?} -> {:?}",
            before,
            after
        );
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 81u64;
        let rows: Vec<([f64; 2], f64)> = (0..120)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..60].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    /// The band systems are state (task 170; [`BandSystems`]): a model and
    /// its copy without them are not equal, and a model restored from its
    /// state -- msgpack named and compact, and JSON -- equals the one saved,
    /// its band systems included. Until schema 33 they were not state, a
    /// restore built them again at the first nudge, and any two models that
    /// differed only in them compared equal (task 166).
    #[test]
    fn the_band_systems_are_state() {
        use crate::OnlineModel;
        let mut m = Robust::new(cfg(3, 1, RobustLoss::Quantile { tau: 0.5 })).unwrap();
        let mut s = 7u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = x[0] - x[2] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(
            m.systems.0.iter().any(Option::is_some),
            "the case needs a system"
        );
        let mut bare = m.clone();
        bare.systems = BandSystems::default();
        assert_ne!(bare, m, "the band systems are compared");
        let state = m.state();
        let named: crate::State =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&state).unwrap()).unwrap();
        let compact: crate::State =
            rmp_serde::from_slice(&rmp_serde::to_vec(&state).unwrap()).unwrap();
        let json: crate::State =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        for (how, back) in [("named", named), ("compact", compact), ("json", json)] {
            assert_eq!(Robust::restore(&back).unwrap(), m, "{how}");
        }
    }

    #[test]
    fn new_surfaces_the_validation_error() {
        let e = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.0 })).unwrap_err();
        assert!(e.contains("quantile must be in"), "{e}");
    }

    /// A robust fit has no window, so it keeps no runs (docs/PLAN.md task
    /// 128).
    #[test]
    fn a_robust_fit_keeps_no_runs() {
        let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        assert!(m.cov.iter().all(|c| !c.keeps_runs()));
    }

    /// A state from before the runs' flag keeps no runs once restored: this
    /// model has no window to read them (review 2026-09-26, C4).
    #[test]
    fn restore_keeps_no_runs() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.35 })).unwrap();
        let mut s = 3u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &x, &[Some(x[0])], d, 1.0);
        }
        let mut v = serde_json::to_value(crate::OnlineModel::state(&m)).unwrap();
        crate::window::json_edit(&mut v, "runs", &mut |x| {
            *x = serde_json::json!({"x": [], "start": []});
        });
        let m =
            <Robust as crate::OnlineModel>::restore(&serde_json::from_value(v).unwrap()).unwrap();
        assert!(m.cov.iter().all(|c| !c.keeps_runs()));
    }

    // --- task 158: the mutation survivors --------------------------------

    /// `solve_share` is refused unless it is finite and positive, and
    /// accepted at any such value.
    #[test]
    fn a_bad_solve_share_is_refused() {
        let huber = RobustLoss::Huber { delta: 1.5 };
        for f in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut c = cfg(2, 1, huber);
            c.solve_share = Some(f);
            match c.validate() {
                Err(e) => assert!(e.contains("solve_share must be finite and > 0"), "{f}: {e}"),
                Ok(()) => panic!("solve_share {f} accepted"),
            }
        }
        for f in [1e-9, 0.02, 1.0, 5.0] {
            let mut c = cfg(2, 1, huber);
            c.solve_share = Some(f);
            c.validate().unwrap();
        }
    }

    /// The warm-up is `WARM_ROWS` rows per coefficient, and no more: with
    /// three coefficients the row with nine before it is the first Newton
    /// step, so the tenth row's prediction is the first to part from least
    /// squares.
    #[test]
    fn the_warm_up_ends_at_three_rows_per_coefficient() {
        let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        qc.min_weight = 0.0;
        let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        lc.min_weight = 0.0;
        let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
        let mut s = 91u64;
        for i in 0..=10 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let (qs, ls) = (
                q.step(&x, &[Some(y)], d, 1.0),
                l.step(&x, &[Some(y)], d, 1.0),
            );
            if i == 10 {
                assert_ne!(qs.pred, ls.pred, "the ninth row was a Newton step");
            } else if i > 0 {
                assert_eq!(qs.pred, ls.pred, "row {i} follows least-squares rows only");
            }
        }
    }

    /// The band's weight is read in rows of the target's mean weight
    /// (task 147): at a weight of 4 a row, a band holding 6 holds 1.5 rows,
    /// under one row per coefficient (`k = 2`), and the next row is a
    /// least-squares row; a band holding 8 holds exactly 2, one row per
    /// coefficient, and a row far outside it is a nudge. With no decay and
    /// equal weights the count is exactly a quarter of the weight.
    #[test]
    fn the_band_is_counted_in_rows_of_the_mean_weight() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 13u64;
        for i in 0..300 {
            let x = [lcg(&mut s)];
            let y = x[0] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 4.0);
        }
        assert_eq!((m.nobs[0], m.wobs[0]), (300.0, 1200.0));
        for (band, least_squares) in [(6.0, true), (8.0, false)] {
            let mut t = m.clone();
            t.wj[0] = band;
            t.step(&[0.3], &[Some(1e6)], 1.0, 4.0);
            let want = if least_squares { band + 4.0 } else { band };
            assert_eq!(
                t.wj[0],
                want,
                "a band of {band} ({} rows): least squares {least_squares}",
                band / 4.0
            );
        }
    }

    /// The band is open, `|r| < h`: a residual of exactly `h` is outside it.
    /// Rows of zeros keep the fit at 0 and every residual exactly 0, so the
    /// target has no scale, and a row is least squares whatever its residual
    /// (task 177). With an `s²` of 1 put in, `h` is `quantile_eps` itself
    /// once the floor `(k/n)^(2/5)` is under it; a row at `h` then nudges
    /// (the band's weight stays) and one just under it is a fit row (the
    /// band's weight takes it). The test read the band at the literal scale
    /// of 1 the model took before a residual, which is gone.
    #[test]
    fn the_band_is_open_at_its_edge() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        c.quantile_eps = 0.5;
        let mut m = Robust::new(c).unwrap();
        for i in 0..20 {
            m.step(&[0.0], &[Some(0.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.sig2[0], 0.0, "every residual so far is exactly 0");
        assert_eq!(m.predict(&[0.0], 1.0).pred[0], 0.0);
        assert!(
            (2.0f64 / 20.0).powf(0.4) < 0.5,
            "the floor is under quantile_eps"
        );
        let wj = m.wj[0];
        let mut unscaled = m.clone();
        unscaled.step(&[0.0], &[Some(0.5)], 1.0, 1.0);
        assert_eq!(unscaled.wj[0], wj + 1.0, "with no scale, least squares");
        m.sig2[0] = 1.0;
        let mut edge = m.clone();
        edge.step(&[0.0], &[Some(0.5)], 1.0, 1.0);
        assert_eq!(edge.wj[0], wj, "a residual of h is outside the band");
        let mut inside = m.clone();
        inside.step(&[0.0], &[Some(0.5 - 1e-12)], 1.0, 1.0);
        assert_eq!(inside.wj[0], wj + 1.0, "one just under h is inside");
    }

    /// Through the origin and standardized, a feature that has only ever been
    /// 0 has no scale and is dropped with a coefficient of 0: the fit of the
    /// other is the fit without it, and no solve fails.
    #[test]
    fn a_feature_without_scale_is_dropped_through_the_origin() {
        let run = |dead: bool| {
            let mut c = cfg(
                if dead { 2 } else { 1 },
                1,
                RobustLoss::Huber { delta: 1e9 },
            );
            c.fit_intercept = false;
            c.standardize = true;
            let mut m = Robust::new(c).unwrap();
            let mut s = 17u64;
            for i in 0..50 {
                let a = lcg(&mut s);
                let x = if dead { vec![a, 0.0] } else { vec![a] };
                let y = 1.5 * a + 0.1 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            (m.coefficients().unwrap()[0].clone(), m.solve_failures)
        };
        let ((with, failed), (without, _)) = (run(true), run(false));
        assert_eq!(failed, 0, "no solve failed");
        assert_eq!(with[1], 0.0, "the dead feature's coefficient");
        assert!(
            (with[0] - without[0]).abs() <= 1e-12 * without[0].abs(),
            "{} against {}",
            with[0],
            without[0]
        );
    }

    /// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` through the
    /// origin: two equal features and no ridge make the raw Gram singular,
    /// the jitter ladder rescues it, and the rescue is counted.
    #[test]
    fn a_solve_failure_through_the_origin_is_counted() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.fit_intercept = false;
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = &m.coefficients().unwrap()[0];
        assert!(beta.iter().all(|v| v.is_finite()), "{beta:?}");
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// The rows a model solves on: where its coefficients appear or move.
    /// Every row is fresh, so a solve always moves them.
    fn solve_rows(m: &mut Robust, n: usize) -> Vec<usize> {
        let mut s = 7u64;
        let mut out = Vec::new();
        for i in 0..n {
            let before = m.coefficients().map(<[Vec<f64>]>::to_vec);
            let x = [lcg(&mut s)];
            let y = x[0] + lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if m.coefficients().map(<[Vec<f64>]>::to_vec) != before {
                out.push(i);
            }
        }
        out
    }

    /// The clock cadence: the first solve on the first row with `min_weight`
    /// (two rows here), then one each `solve_every` of clock (5), with
    /// `max_rows_between_solves` too far off to decide.
    #[test]
    fn the_clock_cadence_solves_every_solve_every() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = 5.0;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(solve_rows(&mut m, 30), [1, 6, 11, 16, 21, 26]);
    }

    /// The row cadence: no clock (`solve_every` infinite), a solve every
    /// `max_rows_between_solves` rows (4) after the first.
    #[test]
    fn the_row_cadence_solves_every_max_rows_between_solves() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 4;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(solve_rows(&mut m, 20), [1, 5, 9, 13, 17]);
    }

    /// Task 180: `solve_every` reads the stamps its caller hands, for both
    /// losses: every 2,000th 1 ms row under 2 s after the first solve, where
    /// the clock summed since the last solve, read when none is handed, is
    /// a row late each time, as it always was.
    #[test]
    fn solve_every_measures_the_stamps_it_is_handed() {
        let losses = [
            RobustLoss::Huber { delta: 1.0 },
            RobustLoss::Quantile { tau: 0.5 },
        ];
        for loss in losses {
            for (stamped, want) in [(true, [0, 2000, 4000]), (false, [0, 2001, 4002])] {
                let mut c = cfg(1, 1, loss);
                c.min_weight = 0.0;
                c.solve_every = 2.0;
                c.max_rows_between_solves = u32::MAX;
                let mut m = Robust::new(c).unwrap();
                let got =
                    crate::since::events_on_millisecond_rows(4_100, stamped, |i, stamp, d| {
                        if let Some(s) = stamp {
                            m.stamp_next(s);
                        }
                        let x = (i % 7) as f64;
                        m.step(&[x], &[Some(1.0 + 2.0 * x)], d, 1.0);
                        m.rows_since_solve == 0
                    });
                assert_eq!(got, want, "{loss:?}, stamped: {stamped}");
            }
        }
    }

    /// The share cadence, set as the spec sets it (`set_solve_share`, after
    /// building): a solve once the weight learned since the last one reaches
    /// `solve_share` of the weight the fit holds (docs/PLAN.md task 115
    /// (b)), the clock left out. The rows are the rule's, written out.
    #[test]
    fn the_share_cadence_solves_at_its_share_of_the_weight() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(m.solve_share(), None);
        m.set_solve_share(Some(0.25));
        assert_eq!(m.solve_share(), Some(0.25));
        let (mut held, mut since, mut fit, mut want) = (0.0, 0.0, false, Vec::new());
        for i in 0..60 {
            held += 1.0;
            since += 1.0;
            if since >= 0.25 * held || (!fit && held >= 2.0) {
                want.push(i);
                (since, fit) = (0.0, true);
            }
        }
        assert!(want.len() > 8 && want[want.len() - 1] - want[want.len() - 2] > 5);
        assert_eq!(solve_rows(&mut m, 60), want);
        m.set_solve_share(None);
        assert_eq!(m.solve_share(), None);
    }

    /// A target that has never been present has no fit, so no prediction,
    /// beside one that has.
    #[test]
    fn a_target_never_present_has_no_prediction() {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 3u64;
        for i in 0..20 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(x[0]), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let p = m.predict(&[0.5], 1.0).pred;
        assert!(p[0].is_finite() && p[1].is_nan(), "{p:?}");
    }

    /// A target with no weight at a solve has no fit: its row of
    /// coefficients is NaN, so once its rows arrive it predicts nothing
    /// until a solve has seen them. The solve left it the zeros `beta`
    /// starts at and spent the first-solve trigger, so the target was
    /// predicted as exactly 0.0 until the next scheduled solve, 56 rows of
    /// 80 under `solve_every = 1000` (review round 4, CC1). And a late
    /// target has a first solve of its own, as a fresh model has: on the row
    /// its own weight first reaches its `min_weight`, here the second of its
    /// rows, row 11, so it is predicted from row 12 on. It waited for the
    /// next solve on the cadence, the row cap 25 rows after the first solve
    /// at row 1, and was null until row 27 (docs/PLAN.md task 195, S9b).
    #[test]
    fn a_target_that_joins_after_the_first_solve_is_not_predicted_from_zeros() {
        for loss in [
            RobustLoss::Huber { delta: 1.5 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            let mut c = cfg(1, 2, loss);
            c.min_weight = 2.0;
            c.solve_every = f64::INFINITY;
            c.max_rows_between_solves = 25;
            let mut m = Robust::new(c).unwrap();
            let mut s = 11u64;
            for i in 0..40 {
                let x = [lcg(&mut s)];
                let b = (i >= 10).then(|| 1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s));
                let p = m.step(
                    &x,
                    &[Some(0.5 + x[0]), b],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
                match i {
                    // No weight before its first row is learned, and one
                    // row's worth before its second.
                    0..=11 => assert!(p.pred[1].is_nan(), "{loss:?}, row {i}: {:?}", p.pred),
                    // Its own first solve, at the end of row 11, saw them.
                    _ => assert!(
                        (p.pred[1] - (1.0 - 2.0 * x[0])).abs() < 0.1,
                        "{loss:?}, row {i}: {:?}",
                        p.pred
                    ),
                }
                // After the row: the target's own first solve at the end of
                // row 11 has seen it, and none before.
                if (2..=11).contains(&i) {
                    let beta = m.coefficients().expect("solved at row 1");
                    assert_eq!(
                        beta[1].iter().all(|v| v.is_nan()),
                        i < 11,
                        "{loss:?}, row {i}: {beta:?}"
                    );
                    assert!(beta[0].iter().all(|v| v.is_finite()), "{loss:?}, row {i}");
                }
            }
        }
    }

    /// The threshold a late target's own first solve waits for is its own,
    /// as a bank holds one per target: at 5 the target joining at row 10 is
    /// first solved at the end of its fifth row, row 14, where at the
    /// model's 2 it was row 11; one solve, and none on the rows between. A
    /// threshold list of the wrong length, or one below 0, is refused, and
    /// so is a state holding one (docs/PLAN.md task 195, S9b).
    #[test]
    fn a_late_targets_first_solve_waits_for_its_own_min_weight() {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.345 });
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        assert!(m.set_target_min_weight(vec![2.0]).is_err());
        assert!(m.set_target_min_weight(vec![2.0, -1.0]).is_err());
        m.set_target_min_weight(vec![2.0, 5.0]).unwrap();
        let mut s = 7u64;
        for i in 0..20 {
            let x = [lcg(&mut s)];
            let b = (i >= 10).then(|| 1.0 - 2.0 * x[0]);
            m.step(&x, &[Some(x[0]), b], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let fit = m
                .coefficients()
                .is_some_and(|b| b[1].iter().all(|v| v.is_finite()));
            assert_eq!(fit, i >= 14, "row {i}");
        }
        let mut st = m.state();
        let crate::ModelState::Robust(inner) = &mut st.model else {
            unreachable!()
        };
        inner.target_min_weight = vec![1.0];
        assert!(matches!(Robust::restore(&st), Err(StateError::Invalid(_))));
    }

    /// A target whose weight a gap takes to exactly 0 keeps the fit its
    /// moments hold, as at one half-life short of the underflow: a fit does
    /// not move on a gap (hard rule 8), and past the underflow the decay
    /// forgets as just short of it (docs/PLAN.md task 115 (c)). The solve
    /// on the row after the gap gave it zeros, predicted as 0.0 once rows
    /// came (review round 4, CC1).
    #[test]
    fn a_target_whose_weight_underflows_keeps_its_fit() {
        for loss in [
            RobustLoss::Huber { delta: 1.5 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            let fit = |gap: f64| {
                let mut c = cfg(1, 1, loss);
                c.decay = Decay::Halflife(10.0);
                c.min_weight = 0.0;
                let mut m = Robust::new(c).unwrap();
                let mut s = 29u64;
                for i in 0..60 {
                    let x = [lcg(&mut s)];
                    let y = 1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s);
                    m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                }
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(1.0 - 2.0 * x[0])], gap, 0.0);
                (m.wj[0], m.coefficients().unwrap()[0].clone())
            };
            let (w_forgot, forgot) = fit(10_750.0);
            let (w_aged, aged) = fit(10_740.0);
            assert!(w_forgot == 0.0 && w_aged > 0.0, "{loss:?}: the case");
            for (a, b) in forgot.iter().zip(&aged) {
                assert!(
                    (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                    "{loss:?}: {forgot:?} against {aged:?}"
                );
            }
            assert!((aged[1] + 2.0).abs() < 0.05, "{loss:?}: {aged:?}");
        }
    }

    /// A first solve that fails at every jitter leaves no fit: NaN, not the
    /// zeros `beta` starts at, which predicted 0.0 as though learned (review
    /// round 4, CC1; `ewridge`'s CA5). No stream of rows gives a
    /// correlation matrix every jitter fails on, so the accumulator is
    /// handed one, as `a_standardized_solve_that_fails_outright_is_counted`
    /// hands it.
    #[test]
    fn a_first_solve_that_fails_leaves_no_fit() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.standardize = true;
        c.min_weight = 1e9;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = u32::MAX;
        let mut m = Robust::new(c).unwrap();
        let mut s = 103u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert!(
            m.coefficients().is_none(),
            "min_weight held the first solve"
        );
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            w,
            q,
        );
        m.solve();
        assert_eq!(m.solve_failures, 1, "every jitter failed");
        let beta = m.coefficients().expect("a solve ran");
        assert!(beta[0].iter().all(|v| v.is_nan()), "{beta:?}");
    }

    /// A model deserialized on its own, outside `restore`, has no row buffer
    /// (it is not state); its first step makes one, and it goes on as the
    /// model it was saved from.
    #[test]
    fn a_model_deserialized_without_restore_steps_on() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 5u64;
        let mut row = || {
            let x = [lcg(&mut s), lcg(&mut s)];
            (x, x[0] - x[1])
        };
        for i in 0..10 {
            let (x, y) = row();
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&m).unwrap();
        let mut back: Robust = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.zbuf.is_empty(), "the buffer is not saved");
        for _ in 0..10 {
            let (x, y) = row();
            assert_eq!(
                m.step(&x, &[Some(y)], 1.0, 1.0).pred,
                back.step(&x, &[Some(y)], 1.0, 1.0).pred
            );
        }
    }

    /// Each of the shape checks on load refuses on its own: an accumulator
    /// of the wrong width, a cross-moment row too many, a per-target vector
    /// short, a coefficient row short, a coefficient row too many.
    #[test]
    fn each_shape_check_refuses_on_its_own() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = 9u64;
        for i in 0..10 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.beta.is_some());
        type Spoil = (&'static str, fn(&mut Robust));
        let cases: [Spoil; 5] = [
            ("an accumulator two wide", |m| {
                m.cov[0] = EwCov::new(2).without_runs();
            }),
            ("a cross-moment row too many", |m| {
                m.cross.push(vec![0.0; 3])
            }),
            ("no residual variance", |m| m.sig2.clear()),
            ("a short coefficient row", |m| {
                m.beta.as_mut().unwrap()[0].truncate(2);
            }),
            ("a coefficient row too many", |m| {
                m.beta.as_mut().unwrap().push(vec![0.0; 3]);
            }),
        ];
        for (what, spoil) in cases {
            let mut bad = m.clone();
            spoil(&mut bad);
            match Robust::restore(&bad.state()) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
                other => panic!("{what}: {other:?}"),
            }
        }
        assert!(Robust::restore(&m.state()).is_ok());
    }

    /// A zero-weight row is not one of the rows a quantile fit counts toward
    /// its warm-up and band floor (hard rule 9): with no decay, a stream with
    /// zero-weight rows among its first rows is the stream without them.
    #[test]
    fn zero_weight_rows_do_not_count_toward_the_warm_up() {
        let run = |ghosts: bool| {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 91u64;
            let mut preds = Vec::new();
            for i in 0..60 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
                if ghosts && i < 8 {
                    m.step(&[-x[1], x[0]], &[Some(-y)], 0.0, 0.0);
                }
                let d = if i == 0 { 0.0 } else { 1.0 };
                preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0]);
            }
            (preds, m.nobs[0])
        };
        let ((with, n_with), (without, n_without)) = (run(true), run(false));
        assert_eq!(n_with, n_without, "the zero-weight rows were counted");
        for (i, (a, b)) in with.iter().zip(&without).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "row {i}: {a} against {b}"
            );
        }
    }

    /// What `row_update` makes of target `j`'s next row, `y` scored at
    /// `pred` at weight `w` after a clock step of `d`, under `scale`: its
    /// other inputs read from the model's state before the row, as `step`
    /// forms them.
    fn update_of(
        m: &Robust,
        j: usize,
        (pred, y, d, w): (f64, f64, f64, f64),
        scale: Option<f64>,
    ) -> RowUpdate {
        let lam = m.cfg.decay.factor(d);
        let (present, rows, aged) = (lam * m.wobs[j], lam * m.nobs[j], lam * m.wj[j]);
        let aged_rows = if present > 0.0 {
            aged * (rows / present)
        } else {
            0.0
        };
        m.row_update(y, pred, scale, w, rows, aged_rows)
    }

    /// Whether target 0's next row, `y` scored at `pred` at weight `w` after
    /// a clock step of `d`, is a nudge: `row_update`'s own reading of the
    /// model's state before the row, as `step` forms its inputs.
    fn is_nudge(m: &Robust, pred: f64, y: f64, d: f64, w: f64) -> bool {
        matches!(
            update_of(m, 0, (pred, y, d, w), m.residual_scale(0)),
            RowUpdate::Nudge { .. }
        )
    }

    /// A nudge brings its row at most to its target, never past it (the
    /// module docs): the solve after a nudged row moves that row's
    /// prediction by no more than its residual. The step was bounded by the
    /// band Gram's diagonal leverage, `Σ d²/v`, where the solve moves the
    /// row by its full leverage, `1 + uᵀA⁻¹u` over the solve's own system:
    /// against features correlated at 0.999 a row at (+1, −1) reads about 2
    /// on the diagonal and about 2,000 in full, and was thrown 6 to 13 times
    /// its residual, 30 to 127 times at 0.9999 (review 2026-10-05, TC1b;
    /// this test on the old code: 5.9 to 7.5, and 30 to 36). Every
    /// nudged row is held to it -- with an intercept, standardized or not,
    /// and through the origin -- and on uncorrelated features, where the
    /// two leverages agree. Every hundredth row past the warm-up runs
    /// against the correlation, at (+1, −1), 0.4 above the true line.
    #[test]
    fn a_nudge_never_moves_its_row_past_its_residual() {
        for rho in [0.0, 0.999, 0.9999] {
            for (fit_intercept, standardize) in [(true, false), (true, true), (false, false)] {
                let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
                c.decay = Decay::Halflife(500.0);
                c.min_weight = 0.0;
                c.ridge = 1e-6;
                c.quantile_eps = 0.2;
                c.fit_intercept = fit_intercept;
                c.standardize = standardize;
                let mut m = Robust::new(c).unwrap();
                let mut s = 5u64;
                let (mut checked, mut glitches, mut worst) = (0, 0, (0.0f64, 0usize));
                for i in 0..3000usize {
                    let a = 3f64.sqrt() * lcg(&mut s);
                    let b = rho * a + (1.0 - rho * rho).sqrt() * 3f64.sqrt() * lcg(&mut s);
                    let glitch = i >= 1000 && i % 100 == 0;
                    let x = if glitch { [1.0, -1.0] } else { [a, b] };
                    let noise = if glitch { 0.4 } else { 0.5 * lcg(&mut s) };
                    // Through the origin the line through the origin.
                    let level = if fit_intercept { 1.0 } else { 0.0 };
                    let y = level + 0.8 * x[0] - 0.4 * x[1] + noise;
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let before = m.predict(&x, d).pred[0];
                    let nudge = before.is_finite() && is_nudge(&m, before, y, d, 1.0);
                    m.step(&x, &[Some(y)], d, 1.0);
                    if nudge {
                        let after = m.predict(&x, 1.0).pred[0];
                        let moved = (after - before).abs() / (y - before).abs();
                        // NaN, a movement not a number, counts as the worst.
                        if moved.is_nan() || moved > worst.0 {
                            worst = (moved, i);
                        }
                        checked += 1;
                        glitches += usize::from(glitch);
                    }
                }
                let case =
                    format!("rho {rho}, intercept {fit_intercept}, standardize {standardize}");
                assert!(checked > 300, "{case}: {checked} nudges");
                assert!(glitches > 0, "{case}: no glitch row was nudged");
                assert!(
                    worst.0 <= 1.0 + 1e-9,
                    "{case}: row {} moved {:.3} times its residual",
                    worst.1,
                    worst.0
                );
            }
        }
    }

    /// On uncorrelated features the full leverage is the diagonal one: a
    /// band Gram set to a diagonal matrix reads a row's leverage as `Σ
    /// d²/(v + ridge)` unstandardized and `Σ (d²/v)/(1 + ridge)` standardized,
    /// to rounding, and so as the old reading, `Σ d²/v`, to the ridge's
    /// share of the smallest variance.
    #[test]
    fn the_full_leverage_is_the_diagonal_one_on_uncorrelated_features() {
        for standardize in [false, true] {
            let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
            c.ridge = 1e-6;
            c.standardize = standardize;
            let mut m = Robust::new(c).unwrap();
            let mut s = 7u64;
            for i in 0..50 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                m.step(
                    &x,
                    &[Some(x[0] - x[2])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            let (mean, var) = ([1.0, 0.2, -0.3, 0.5], [0.0, 0.7, 1.9, 0.04]);
            let mut cen = vec![0.0; 16];
            for i in 0..4 {
                cen[i * 4 + i] = var[i];
            }
            let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
            m.cov[0].set_moments(&mean, &cen, w, q);
            m.systems.0[0] = None;
            m.zbuf = RowBuf(vec![1.0, 1.1, -2.0, 0.9]);
            let ridge = 1e-6;
            let term = |i: usize| (m.zbuf[i] - mean[i]).powi(2);
            let with_ridge: f64 = 1.0
                + (1..4)
                    .map(|i| {
                        if standardize {
                            term(i) / var[i] / (1.0 + ridge)
                        } else {
                            term(i) / (var[i] + ridge)
                        }
                    })
                    .sum::<f64>();
            let old: f64 = 1.0 + (1..4).map(|i| term(i) / var[i]).sum::<f64>();
            let full = m.nudge_movement(0);
            assert!(
                (full - with_ridge).abs() <= 1e-12 * with_ridge,
                "standardize {standardize}: {full} against {with_ridge}"
            );
            assert!(
                (full - old).abs() <= ridge / 0.04 * old,
                "standardize {standardize}: {full} against the old {old}"
            );
        }
    }

    /// A nudge is bounded by the row's own residual over its leverage (G2).
    /// At the value a feature has always taken the leverage is 0, so a heavy
    /// row outside the band moves the fit exactly to its target and no
    /// further; off that value the leverage is unbounded (the Gram holds no
    /// curvature for it) and the fit does not move. At a weight of `1e4` the
    /// step before the bound is far past the residual, so the bound decides.
    #[test]
    fn a_nudge_on_a_feature_without_spread_is_bounded_by_the_residual() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 23u64;
        for i in 0..40 {
            let y = 0.1 * lcg(&mut s);
            m.step(&[1.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.cov[0].cov(1, 1), 0.0, "the feature has no spread");
        let p = m.predict(&[1.0], 1.0).pred[0];
        let mut on = m.clone();
        on.step(&[1.0], &[Some(p + 0.5)], 1.0, 1e4);
        assert_eq!(on.wj[0], m.wj[0], "the row was a nudge");
        let q = on.predict(&[1.0], 1.0).pred[0];
        assert!(
            (q - (p + 0.5)).abs() <= 1e-12,
            "brought to its target: {p} -> {q}, target {}",
            p + 0.5
        );
        let mut off = m.clone();
        let p2 = off.predict(&[2.0], 1.0).pred[0];
        off.step(&[2.0], &[Some(p2 + 0.5)], 1.0, 1e4);
        assert_eq!(off.wj[0], m.wj[0], "the row was a nudge");
        assert_eq!(off.coefficients(), m.coefficients(), "and moved nothing");
    }

    /// A leverage whose quadratic form is not a number is unbounded, and
    /// the nudge's step 0, as for a feature with no spread. Features
    /// correlated at 0.99 and a row at `(1e160, 5e159)`: `A⁻¹u` has entries
    /// of opposite signs, so the form's two terms overflow to `+∞` and `−∞`,
    /// and their sum is NaN. `f64::max(NaN, 0.0)` read it as 0: with an
    /// intercept a leverage of 1, the least there is, bounding the step by
    /// the whole residual, and through the origin a leverage of 0, bounding
    /// it by nothing -- the overshoot the bound is there to stop (review
    /// 2026-09-26, G2). Read as NaN, `|r| / NaN` bounds nothing either
    /// (task 181). With an intercept and through the origin.
    #[test]
    fn a_leverage_that_is_not_a_number_is_unbounded() {
        for fit_intercept in [true, false] {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
            c.fit_intercept = fit_intercept;
            let mut m = Robust::new(c).unwrap();
            let mut s = 29u64;
            for i in 0..50 {
                let x = [lcg(&mut s), lcg(&mut s)];
                m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            // The band Gram set to unit variances correlated at 0.99, about
            // means of 0: centred with an intercept, raw through the origin.
            let off = usize::from(fit_intercept);
            let k = 2 + off;
            let mut mean = vec![0.0; k];
            let mut moments = vec![0.0; k * k];
            if fit_intercept {
                mean[0] = 1.0;
            }
            for i in 0..2 {
                for j in 0..2 {
                    moments[(i + off) * k + j + off] = if i == j { 1.0 } else { 0.99 };
                }
            }
            let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
            m.cov[0].set_moments(&mean, &moments, w, q);
            m.systems.0[0] = None;
            let row = [1e160, 0.5e160];
            m.zbuf = RowBuf(if fit_intercept {
                vec![1.0, row[0], row[1]]
            } else {
                row.to_vec()
            });
            let q = m.nudge_movement(0);
            assert_eq!(q, f64::INFINITY, "intercept {fit_intercept}: {q}");
            // The same system reads a row at the data's scale.
            m.zbuf[off] = 1.0;
            m.zbuf[off + 1] = 0.5;
            let q = m.nudge_movement(0);
            assert!(q.is_finite() && q > 1.0, "intercept {fit_intercept}: {q}");
        }
    }

    /// A row with a feature that is not a finite number has no finite
    /// prediction, so it reached the loss as a least-squares row and was
    /// learned: its NaN went into the Gram and the cross-moment, every solve
    /// after it failed -- the 40 of 40 rows that followed one -- and the fit
    /// froze where it stood. Task 182 refused it in that arm, and counted it
    /// in `n_eff` and the target's present weight. By the rule every model
    /// keeps (`OnlineModel`, task 183) it is a row of weight 0 and counted
    /// nowhere: it reports nothing, the Gram ages, the cross-moment and the
    /// target's mean stay, and the state is the same row's at weight 0, its
    /// features finite, byte for byte. Huber and the quantile loss alike; a
    /// NaN, an infinity and a feature past the input bound alike.
    #[test]
    fn a_row_that_is_not_a_number_is_not_learned() {
        let bytes = |m: &Robust| rmp_serde::to_vec(&m.state()).unwrap();
        for loss in [
            RobustLoss::Huber { delta: 1.0 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            for bad in [
                [f64::NAN, 0.5],
                [0.5, f64::NEG_INFINITY],
                [2.0 * crate::INPUT_BOUND, 0.5],
            ] {
                let case = format!("{loss:?}, {bad:?}");
                let mut c = cfg(2, 1, loss);
                c.decay = Decay::Halflife(50.0);
                c.ridge = 1e-6;
                c.min_weight = 3.0;
                let lam = c.decay.factor(1.0);
                let mut m = Robust::new(c).unwrap();
                let mut s = 9u64;
                let mut at_the_row = None;
                for i in 0..80 {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = 1.0 + x[0] + 2.0 * x[1] + 0.1 * lcg(&mut s);
                    if i == 40 {
                        let before = m.clone();
                        let mut no_weight = m.clone();
                        no_weight.step(&x, &[Some(1.0)], 1.0, 0.0);
                        let out = m.step(&bad, &[Some(1.0)], 1.0, 1.0);
                        assert!(out.pred[0].is_nan(), "{case}: {out:?}");
                        assert_eq!(m.cross, before.cross, "{case}");
                        assert_eq!(m.ybar, before.ybar, "{case}");
                        assert_eq!(m.cov[0].means(), before.cov[0].means(), "{case}");
                        assert_eq!(m.cov[0].comoments(), before.cov[0].comoments(), "{case}");
                        assert_eq!(m.wj[0], lam * before.wj[0], "{case}");
                        assert_eq!(m.w_raw, lam * before.w_raw, "{case}");
                        assert_eq!(m.wobs[0], lam * before.wobs[0], "{case}");
                        assert_eq!(m.nobs[0], lam * before.nobs[0], "{case}");
                        assert_eq!(m.sig2, before.sig2, "{case}");
                        assert_eq!(bytes(&m), bytes(&no_weight), "{case}");
                        at_the_row = m.coefficients().map(<[_]>::to_vec);
                        continue;
                    }
                    let out = m.step(&x, &[Some(y)], 1.0, 1.0);
                    if i > 40 {
                        assert!(out.pred[0].is_finite(), "{case}, row {i}: {out:?}");
                    }
                }
                assert_eq!(m.solve_failures, 0, "{case}");
                let beta = m.coefficients().unwrap()[0].clone();
                assert_ne!(
                    Some(vec![beta.clone()]),
                    at_the_row,
                    "{case}: the fit moved on"
                );
                for (b, want) in beta.iter().zip([1.0, 1.0, 2.0]) {
                    assert!((b - want).abs() < 0.1, "{case}: {beta:?}");
                }
            }
        }
    }

    // --- task 170: the band factor moved where the move is exact ---------

    /// The leverage a nudge would read for `probe` against target 0's band
    /// system, from the system `m` keeps and from one built afresh from its
    /// Gram: `(kept, fresh)`.
    fn movement_kept_and_fresh(m: &Robust, probe: &[f64]) -> (f64, f64) {
        let z: Vec<f64> = if m.cfg.fit_intercept {
            std::iter::once(1.0).chain(probe.iter().copied()).collect()
        } else {
            probe.to_vec()
        };
        let (mut kept, mut fresh) = (m.clone(), m.clone());
        fresh.systems = BandSystems::default();
        kept.zbuf.copy_from_slice(&z);
        fresh.zbuf.copy_from_slice(&z);
        (kept.nudge_movement(0), fresh.nudge_movement(0))
    }

    /// Under `ridge = 0` a row inside the band moves the band system's
    /// matrix by a scale and a rank-one step -- `A' = a·A + a·b·ũ ũᵀ` from the
    /// centred Gram, `a·A + b·x̃ x̃ᵀ` from the raw one through the origin --
    /// and under `standardize` by a rescaling after it, so the kept factor is
    /// moved with it in `O(k²)` instead of dropped and rebuilt in `O(k³)` at
    /// the next nudge (task 170, TC1b's cost). Held, after every such row a
    /// solve did not follow, to a system built afresh from the Gram: the same
    /// columns and scales to the bit, and the leverage a nudge reads -- on the
    /// row itself, on a row against the features' correlation and on a third
    /// -- within `1e-12` of the fresh system's (measured: 9.5e-15). With an
    /// intercept and through the origin, standardized and not, under a
    /// half-life and without one.
    #[test]
    #[ignore = "extended: a second or more (2.7 s)"]
    fn without_a_ridge_a_band_row_moves_the_kept_factor() {
        for (fit_intercept, standardize) in
            [(true, false), (true, true), (false, false), (false, true)]
        {
            for hl in [200.0, f64::INFINITY] {
                let case =
                    format!("intercept {fit_intercept}, standardize {standardize}, half-life {hl}");
                let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
                c.decay = Decay::Halflife(hl);
                c.ridge = 0.0;
                c.min_weight = 0.0;
                c.quantile_eps = 0.2;
                c.fit_intercept = fit_intercept;
                c.standardize = standardize;
                c.solve_every = f64::INFINITY;
                c.max_rows_between_solves = 50;
                let mut m = Robust::new(c).unwrap();
                let mut s = 17u64;
                let (mut moved, mut worst) = (0usize, 0.0f64);
                for i in 0..3000usize {
                    let a = lcg(&mut s);
                    let x = [a, 0.9 * a + 0.4 * lcg(&mut s), lcg(&mut s)];
                    let level = if fit_intercept { 1.0 } else { 0.0 };
                    let y = level + x[0] - 0.5 * x[1] + 0.25 * x[2] + 0.5 * lcg(&mut s);
                    m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                    let Some(sys) = m.systems.0.first().and_then(Option::as_ref) else {
                        continue;
                    };
                    // A factor no band row moved is fresh from a solve or a
                    // nudge, the to-the-bit case of `solve.rs`.
                    if sys.factor.as_ref().is_none_or(|f| f.moves() == 0) {
                        continue;
                    }
                    moved += 1;
                    let (_, keep, scales) = m.band_system(0, 0);
                    assert_eq!(keep, sys.keep, "{case}, row {i}");
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&scales), bits(&sys.s), "{case}, row {i}");
                    for probe in [x, [1.0, -1.0, 0.5], [0.3, 0.2, -0.8]] {
                        let (kept, fresh) = movement_kept_and_fresh(&m, &probe);
                        assert!(fresh.is_finite() && fresh > 0.0, "{case}, row {i}: {fresh}");
                        worst = worst.max((kept - fresh).abs() / fresh);
                    }
                }
                assert!(moved > 300, "{case}: {moved} rows read a moved factor");
                assert!(worst <= 1e-12, "{case}: the leverage parts by {worst:e}");
            }
        }
    }

    /// With a ridge on the band system, a band row moves `A` by a diagonal
    /// shift as well, `(1 − a)·ridge·I`, which no `O(k²)` move makes, so the
    /// kept factor is dropped and the next nudge factorizes afresh, as before
    /// task 170: the default configuration (`ridge = 1e-6`) keeps TC1b's
    /// numbers to the bit. A Huber fit drops it too: no nudge reads it.
    #[test]
    fn with_a_ridge_or_under_huber_a_band_row_drops_the_kept_factor() {
        for (loss, ridge) in [
            (RobustLoss::Quantile { tau: 0.5 }, 1e-6),
            (RobustLoss::Huber { delta: 1.345 }, 0.0),
        ] {
            let mut c = cfg(3, 1, loss);
            c.decay = Decay::Halflife(200.0);
            c.ridge = ridge;
            c.min_weight = 0.0;
            c.quantile_eps = 0.2;
            c.solve_every = f64::INFINITY;
            c.max_rows_between_solves = 50;
            let mut m = Robust::new(c).unwrap();
            let mut s = 17u64;
            let mut fits = 0;
            for i in 0..3000usize {
                let a = lcg(&mut s);
                let x = [a, 0.9 * a + 0.4 * lcg(&mut s), lcg(&mut s)];
                let y = 1.0 + x[0] - 0.5 * x[1] + 0.25 * x[2] + 0.5 * lcg(&mut s);
                let wj = m.wj[0];
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                let fit = m.wj[0] > wj * 0.5f64.powf(1.0 / 200.0) * (1.0 + 1e-12);
                if fit && m.rows_since_solve > 0 {
                    fits += 1;
                    assert!(m.systems.0[0].is_none(), "{loss:?}, row {i}");
                }
            }
            assert!(fits > 300, "{loss:?}: {fits} fit rows");
        }
    }

    /// A band row that gives a standardized fit a column it did not keep --
    /// a feature that had held one value takes another -- changes the
    /// system's shape, which no move makes: the kept factor is dropped, and
    /// the next nudge builds the system over the new columns (task 170).
    #[test]
    fn a_band_row_that_changes_the_kept_columns_drops_the_factor() {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
        c.ridge = 0.0;
        c.standardize = true;
        c.min_weight = 0.0;
        c.quantile_eps = 0.2;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        let mut s = 31u64;
        for i in 0..300 {
            let a = lcg(&mut s);
            let y = 0.5 + a + 0.3 * lcg(&mut s);
            m.step(&[a, 0.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // A nudge, so that a system is kept.
        let p = m.predict(&[0.1, 0.0], 1.0).pred[0];
        m.step(&[0.1, 0.0], &[Some(p + 50.0)], 1.0, 1.0);
        let sys = m.systems.0[0].as_ref().expect("the nudge kept a system");
        assert_eq!(sys.keep, [0], "the feature without spread is not kept");
        // A row at its prediction is inside the band, and moves feature 1.
        let p = m.predict(&[0.1, 0.5], 1.0).pred[0];
        let wj = m.wj[0];
        m.step(&[0.1, 0.5], &[Some(p)], 1.0, 1.0);
        assert!(m.wj[0] > wj, "the row was inside the band");
        assert!(m.systems.0[0].is_none(), "the system's columns changed");
        let p = m.predict(&[0.1, 0.0], 1.0).pred[0];
        m.step(&[0.1, 0.0], &[Some(p + 50.0)], 1.0, 1.0);
        assert_eq!(m.systems.0[0].as_ref().unwrap().keep, [0, 1]);
    }

    // --- task 170: the band systems are state ---------------------------

    /// TC1b's stream: two features correlated at 0.999 and, from row 1000,
    /// a row against the correlation every hundred rows, at (+1, −1), on
    /// the true line, whose leverage is near 2,000 and whose nudge's bound
    /// binds (`a_nudge_never_moves_its_row_past_its_residual`).
    fn tc1b_rows(n: usize) -> Vec<([f64; 2], f64)> {
        let mut s = 5u64;
        (0..n)
            .map(|i| {
                let a = 3f64.sqrt() * lcg(&mut s);
                let b = 0.999 * a + (1.0 - 0.999f64 * 0.999).sqrt() * 3f64.sqrt() * lcg(&mut s);
                let glitch = i >= 1000 && i % 100 == 0;
                let x = if glitch { [1.0, -1.0] } else { [a, b] };
                let noise = if glitch { 0.4 } else { 0.5 * lcg(&mut s) };
                (x, 1.0 + 0.8 * x[0] - 0.4 * x[1] + noise)
            })
            .collect()
    }

    /// A quantile fit on `tc1b_rows`' configuration, at `ridge`.
    fn tc1b_cfg(ridge: f64, standardize: bool, share: bool) -> RobustCfg {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
        c.decay = Decay::Halflife(500.0);
        c.min_weight = 0.0;
        c.ridge = ridge;
        c.quantile_eps = 0.2;
        c.standardize = standardize;
        c.solve_every = 10.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_share = share.then_some(crate::DEFAULT_SOLVE_SHARE);
        c
    }

    /// Target 0's kept factor's moves, if a factor is kept.
    fn band_moves(m: &Robust) -> Option<u32> {
        m.systems
            .0
            .first()?
            .as_ref()?
            .factor
            .as_ref()
            .map(SpdFactor::moves)
    }

    /// A quantile fit resumes to the bit at every save point (task 170):
    /// saved -- named msgpack, as a bank writes it -- and restored at the 49
    /// rows of the probe and at the three rows before each row against the
    /// correlation, it goes on as the fit that never stopped, its band factor
    /// read back as it was saved, the moves its band rows made since the
    /// last solve included. Before the band systems were state, a restore
    /// built a fresh factor where the saved fit read a moved one, and at
    /// `ridge = 0` the probe's save point before row 1100 parted from the
    /// uninterrupted fit by up to 7.5e-14 (1 of 49). Under both cadences,
    /// standardized and not, and at the default ridge, whose kept factor
    /// was already a fresh one. Each comparison runs the 300 rows after its
    /// save point: a parting persists, and a solve comes within ten.
    #[test]
    #[ignore = "extended: a save at every row (1.9 s)"]
    fn a_quantile_fit_resumes_to_the_bit_at_every_save_point() {
        use crate::OnlineModel;
        let rows = tc1b_rows(3000);
        let mut cuts: Vec<usize> = (1100..2900).step_by(37).collect();
        assert_eq!(cuts.len(), 49);
        cuts.extend((1100..2900).step_by(100).flat_map(|g| [g - 2, g - 1, g]));
        let step = |m: &mut Robust, i: usize| {
            let (x, y) = rows[i];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
                .pred[0]
                .to_bits()
        };
        for ridge in [0.0, 1e-6] {
            for standardize in [false, true] {
                for share in [false, true] {
                    let case = format!("ridge {ridge}, standardize {standardize}, share {share}");
                    let c = tc1b_cfg(ridge, standardize, share);
                    let mut straight = Robust::new(c.clone()).unwrap();
                    let mut saved = Vec::new();
                    let mut preds = Vec::new();
                    for i in 0..rows.len() {
                        if cuts.contains(&i) {
                            saved.push((i, rmp_serde::to_vec_named(&straight.state()).unwrap()));
                        }
                        preds.push(step(&mut straight, i));
                    }
                    let (mut off, mut moved) = (Vec::new(), 0);
                    for (cut, bytes) in saved {
                        let mut back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap())
                            .unwrap_or_else(|e| panic!("{case}, cut {cut}: {e}"));
                        moved += usize::from(band_moves(&back).is_some_and(|k| k > 0));
                        if (cut..(cut + 300).min(rows.len()))
                            .any(|i| step(&mut back, i) != preds[i])
                        {
                            off.push(cut);
                        }
                    }
                    assert!(
                        off.is_empty(),
                        "{case}: {} save points off the bit: {off:?}",
                        off.len()
                    );
                    if ridge == 0.0 {
                        assert!(
                            moved > 10,
                            "{case}: {moved} save points held a moved factor"
                        );
                    }
                }
            }
        }
    }

    /// A state holding a moved band factor goes through JSON and back as
    /// itself (task 170): the restored fit equals the saved one, its factor
    /// with the moves it had made, and goes on to the bit.
    #[test]
    fn a_moved_band_factor_survives_a_json_round_trip() {
        use crate::OnlineModel;
        let rows = tc1b_rows(1400);
        for standardize in [false, true] {
            let mut m = Robust::new(tc1b_cfg(0.0, standardize, true)).unwrap();
            let mut i = 0;
            while i < 1000 || band_moves(&m).is_none_or(|k| k == 0) {
                let (x, y) = rows[i];
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                i += 1;
            }
            let json = serde_json::to_string(&m.state()).unwrap();
            let mut back = Robust::restore(&serde_json::from_str(&json).unwrap()).unwrap();
            assert_eq!(back, m, "standardize {standardize}");
            assert_eq!(
                band_moves(&back),
                band_moves(&m),
                "standardize {standardize}"
            );
            for (x, y) in &rows[i..] {
                assert_eq!(
                    back.step(x, &[Some(*y)], 1.0, 1.0).pred[0].to_bits(),
                    m.step(x, &[Some(*y)], 1.0, 1.0).pred[0].to_bits(),
                    "standardize {standardize}"
                );
            }
        }
    }

    /// Each way a band system can be damaged in a state is refused on its
    /// own (task 170), where the model would read a leverage from a system
    /// that is not its Gram's: a slot too many, a column the Gram does not
    /// keep, a scale an ulp off, no factor where columns are kept, one of
    /// the wrong order, and one where none is kept. A factor that is not one
    /// -- a pivot not positive, an entry not finite or missing, a jitter that
    /// does not fit its rung, more moves than the cap, any on a jittered
    /// factor -- is refused where the state is read; beside them a jittered
    /// factor that made no move is read.
    #[test]
    fn each_damaged_band_system_is_refused_on_its_own() {
        use crate::OnlineModel;
        use serde_json::json;
        let fitted = |constant: bool| {
            let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
            c.standardize = true;
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 7u64;
            for i in 0..60 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                let x = if constant { [1.0, 2.0, 3.0] } else { x };
                let y = x[0] - x[2] + 0.3 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            assert!(m.systems.0[0].is_some(), "the case needs a system");
            m.state()
        };
        let base = fitted(false);
        let none_kept = fitted(true);
        type Spoil = (&'static str, bool, fn(&mut BandSystems), &'static str);
        let cases: [Spoil; 6] = [
            (
                "a slot too many",
                false,
                |b| b.0.push(None),
                "band systems for",
            ),
            (
                "a column the Gram does not keep",
                false,
                |b| {
                    b.0[0].as_mut().unwrap().keep.pop();
                },
                "not of its Gram's",
            ),
            (
                "a scale an ulp off",
                false,
                |b| {
                    let s = &mut b.0[0].as_mut().unwrap().s;
                    s[1] = f64::from_bits(s[1].to_bits() + 1);
                },
                "not of its Gram's",
            ),
            (
                "no factor where columns are kept",
                false,
                |b| {
                    b.0[0].as_mut().unwrap().factor = None;
                },
                "band factor is not of its 3",
            ),
            (
                "a factor of the wrong order",
                false,
                |b| {
                    b.0[0].as_mut().unwrap().factor = SpdFactor::of(&[1.0], 1);
                },
                "band factor is not of its 3",
            ),
            (
                "a factor where no column is kept",
                true,
                |b| {
                    b.0[0].as_mut().unwrap().factor = SpdFactor::of(&[1.0], 1);
                },
                "band factor is not of its 0",
            ),
        ];
        for (what, constant, spoil, want) in cases {
            let mut s = if constant {
                none_kept.clone()
            } else {
                base.clone()
            };
            let ModelState::Robust(inner) = &mut s.model else {
                unreachable!()
            };
            spoil(&mut inner.systems);
            match Robust::restore(&s) {
                Err(StateError::Invalid(e)) => assert!(e.contains(want), "{what}: {e}"),
                other => panic!("{what}: {other:?}"),
            }
        }
        assert!(Robust::restore(&base).is_ok() && Robust::restore(&none_kept).is_ok());

        let v = serde_json::to_value(&base).unwrap();
        type JsonEdit = (&'static str, fn(&mut serde_json::Value));
        type Edit = (&'static str, &'static [JsonEdit]);
        let read = |edits: &[JsonEdit]| {
            let mut v = v.clone();
            for (key, edit) in edits {
                crate::window::json_edit(&mut v, key, &mut |x| edit(x));
            }
            serde_json::from_value::<crate::State>(v)
        };
        let refused: [Edit; 7] = [
            (
                "a pivot that is not positive",
                &[("lower", |x| x[0] = json!(-1.0))],
            ),
            (
                "an entry that is not finite",
                &[("lower", |x| x[1] = json!("nan"))],
            ),
            (
                "an entry missing",
                &[("lower", |x| {
                    x.as_array_mut().unwrap().pop();
                })],
            ),
            (
                "a shift on the first rung",
                &[("jitter", |x| *x = json!(0.5))],
            ),
            (
                "a shift that is not finite",
                &[
                    ("attempts", |x| *x = json!(1)),
                    ("jitter", |x| *x = json!("inf")),
                ],
            ),
            ("more moves than the cap", &[("moves", |x| *x = json!(65))]),
            (
                "a move on a jittered factor",
                &[
                    ("attempts", |x| *x = json!(1)),
                    ("jitter", |x| *x = json!(1e-12)),
                    ("moves", |x| *x = json!(1)),
                ],
            ),
        ];
        for (what, edits) in refused {
            assert!(read(edits).is_err(), "{what}");
        }
        assert!(
            read(&[
                ("attempts", |x| *x = json!(5)),
                ("jitter", |x| *x = json!(1e-3))
            ])
            .is_err()
        );
        let jittered = read(&[
            ("attempts", |x| *x = json!(1)),
            ("jitter", |x| *x = json!(1e-12)),
        ]);
        assert!(
            Robust::restore(&jittered.unwrap()).is_ok(),
            "a jittered factor is read"
        );
    }

    /// Until a target has a residual scale, Huber down-weights no row: it is
    /// least squares to the bit, through the row that gives the scale its
    /// first residual (task 177). The target is held at exactly zero for ten
    /// rows, so every residual is exactly zero and no scale exists, and then
    /// jumps by 1 000, which the old cut of `delta` times the literal 1
    /// weighed at 0.0015. From that row the scale exists, and the next jump
    /// it judges is cut: the fits part.
    #[test]
    fn before_a_scale_exists_huber_is_least_squares() {
        let run = |delta: f64| {
            let mut c = cfg(1, 1, RobustLoss::Huber { delta });
            c.min_weight = 2.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 131u64;
            let mut out = Vec::new();
            for i in 0..30 {
                let x = [lcg(&mut s)];
                let y = if i < 10 { 0.0 } else { 1000.0 + x[0] };
                let scaled = m.residual_scale(0).is_some();
                let p = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                out.push((p.pred[0], scaled));
            }
            out
        };
        let (hub, ls) = (run(1.5), run(f64::INFINITY));
        for (i, (_, scaled)) in hub.iter().enumerate() {
            assert_eq!(*scaled, i > 10, "row {i}: a scale from row 11 on");
        }
        for (i, ((h, _), (l, _))) in hub.iter().zip(&ls).enumerate().take(12) {
            assert!(
                h.to_bits() == l.to_bits() || (h.is_nan() && l.is_nan()),
                "row {i}: huber {h} against least squares {l}"
            );
        }
        assert!(hub[10].0 == 0.0 && hub[11].0 > 50.0, "{:?}", &hub[9..12]);
        assert!(
            (hub[12].0 - ls[12].0).abs() > 1.0,
            "row 12: huber {} against least squares {}: the scale cut row 11",
            hub[12].0,
            ls[12].0
        );
    }

    /// Until a target has a residual scale, a quantile row past the warm-up
    /// is least squares, to the bit, through the row that gives the scale
    /// its first residual (task 177): a band is a width in units of the
    /// scale, and there is none to draw. The target is held at exactly zero
    /// for twenty rows, well past the warm-up, so every residual is exactly
    /// zero; the band the literal scale of 1 drew took them as band rows
    /// aimed `2h(tau - 1/2)` above the target, and the fit left zero. Then
    /// the target jumps by 1 000. From that row the scale exists, and the
    /// band shapes the rows after it: the fits part.
    #[test]
    fn before_a_scale_exists_a_quantile_row_is_least_squares() {
        let run = |loss: RobustLoss| {
            let mut c = cfg(1, 1, loss);
            c.min_weight = 2.0;
            c.quantile_eps = 0.2;
            let mut m = Robust::new(c).unwrap();
            let mut s = 137u64;
            let mut out = Vec::new();
            for i in 0..40 {
                let x = [lcg(&mut s)];
                let y = if i < 20 {
                    0.0
                } else {
                    1000.0 + x[0] + lcg(&mut s)
                };
                let scaled = m.residual_scale(0).is_some();
                let p = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                out.push((p.pred[0], scaled));
            }
            out
        };
        let q = run(RobustLoss::Quantile { tau: 0.9 });
        let ls = run(RobustLoss::Huber {
            delta: f64::INFINITY,
        });
        for (i, ((a, _), (b, _))) in q.iter().zip(&ls).enumerate().take(22) {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "row {i}: quantile {a} against least squares {b}"
            );
        }
        for (i, (_, scaled)) in q.iter().enumerate() {
            assert_eq!(*scaled, i > 20, "row {i}: a scale from row 21 on");
        }
        assert!(q[20].0 == 0.0 && q[21].0 > 20.0, "{:?}", &q[19..22]);
        assert!(
            (q[22].0 - ls[22].0).abs() > 1.0,
            "row 22: quantile {} against least squares {}: the band shaped row 21",
            q[22].0,
            ls[22].0
        );
    }

    /// A row of [`units_rows`]: features, targets, clock step, weight.
    type UnitsRow = ([f64; 2], [Option<f64>; 3], f64, f64);

    /// The stream [`scaling_the_targets_scales_every_prediction_to_the_bit`]
    /// runs: three targets on two features. The first is held at exactly
    /// zero for twelve rows, so its residuals are exactly zero until it
    /// moves; the second joins on row 3 and misses one row in five; the
    /// third carries an outlier one row in seven, for the cut to act on once
    /// a scale exists. A row of weight 0 and a gap of five clock units fall
    /// before any residual, a gap of 300 takes the weight under
    /// `min_weight`, and one of 40 000 (1 333 half-lives) ages every weight
    /// to exactly 0.
    fn units_rows() -> Vec<UnitsRow> {
        let mut s = 113u64;
        (0..260)
            .map(|i| {
                let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
                let y0 = 0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s);
                let y1 = -x[0] + 0.7 * x[1] + 0.3 * lcg(&mut s);
                let kick = if i % 7 == 3 { 25.0 * lcg(&mut s) } else { 0.0 };
                let y2 = 1.0 + 2.0 * x[0] + 0.1 * lcg(&mut s) + kick;
                let ys = [
                    match i {
                        _ if i < 12 => Some(0.0),
                        _ if i % 11 == 7 => None,
                        _ => Some(y0),
                    },
                    (i >= 3 && i % 5 != 2).then_some(y1),
                    Some(y2),
                ];
                let d = match i {
                    0 => 0.0,
                    4 => 5.0,
                    150 => 300.0,
                    200 => 40_000.0,
                    _ => 1.0,
                };
                let w = if i == 2 { 0.0 } else { 1.5 + lcg(&mut s) };
                (x, ys, d, w)
            })
            .collect()
    }

    /// Scaling every target by `c` scales every prediction by `c`, `σ²` by
    /// `c²` and every coefficient by `c`: the fit has no unit of its own.
    /// Until a target had a residual scale -- no residual yet, or every one
    /// so far exactly zero -- its Huber cut was `delta` times the literal 1,
    /// in the target's units, so a target in millions had its first
    /// predicted rows down-weighted as outliers and one in millionths none
    /// (task 177; `kalman`'s twin is task 172's). Now such a row is
    /// down-weighted by nothing. Powers of two keep every operation exact,
    /// so the comparison is to the bit, over the warm-up and after
    /// ([`units_rows`] says what the stream holds), with an intercept and
    /// without, standardized and not.
    #[test]
    fn scaling_the_targets_scales_every_prediction_to_the_bit() {
        let rows = units_rows();
        // Predictions, the model after the stream, the rows judged before
        // their target had a scale (a prediction, a weight and a residual
        // other than 0, and `σ²` still 0), and the rows the scale cut.
        let run = |base: &RobustCfg, c: f64| {
            let mut m = Robust::new(base.clone()).unwrap();
            let (mut preds, mut unscaled, mut cut) = (Vec::new(), 0, 0);
            for (x, ys, d, w) in &rows {
                let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
                let p = m.predict(x, *d).pred;
                for (j, y) in ys.iter().enumerate() {
                    let Some(y) = *y else { continue };
                    let r = y - p[j];
                    if !r.is_finite() || *w <= 0.0 || r == 0.0 {
                        continue;
                    }
                    let RobustLoss::Huber { delta } = m.cfg.loss else {
                        unreachable!()
                    };
                    if m.sig2[j] == 0.0 {
                        unscaled += 1;
                    } else if r.abs() > delta * m.sig2[j].sqrt() {
                        cut += 1;
                    }
                }
                preds.push(m.step(x, &ys, *d, *w).pred);
            }
            (preds, m, unscaled, cut)
        };
        let mut cases = Vec::new();
        for intercept in [true, false] {
            for standardize in [false, true] {
                let mut c = cfg(2, 3, RobustLoss::Huber { delta: 1.5 });
                c.decay = Decay::Halflife(30.0);
                c.min_weight = 3.0;
                c.fit_intercept = intercept;
                c.standardize = standardize;
                cases.push((
                    format!("intercept {intercept}, standardize {standardize}"),
                    c,
                ));
            }
        }
        // The first predictions later in the stream, at another cut.
        let mut c = cfg(2, 3, RobustLoss::Huber { delta: 1.345 });
        c.decay = Decay::Halflife(30.0);
        c.min_weight = 20.0;
        cases.push(("min_weight 20, delta 1.345".into(), c));

        for (name, base) in &cases {
            let (want, m1, unscaled, cut) = run(base, 1.0);
            // Every target is judged once before it has a scale: the first
            // when it leaves zero, the others on their first prediction.
            assert!(unscaled >= 3, "{name}: {unscaled} rows before a scale");
            assert!(cut >= 20, "{name}: the scale cut {cut} rows");
            let predicted = want
                .iter()
                .filter(|p| p.iter().all(|v| v.is_finite()))
                .count();
            assert!(predicted > 200, "{name}: {predicted} rows predicted");
            for c in [2f64.powi(20), 2f64.powi(-20)] {
                let (got, mc, _, _) = run(base, c);
                assert_scaled(name, c, (&want, &m1), (&got, &mc));
            }
        }
    }

    /// The run on targets scaled by `c` (`got` and its model) is the run on
    /// the targets as they are (`want`) scaled by `c`, to the bit: every
    /// prediction by `c`, a NaN where `want` has one, and after the stream
    /// `σ²` by `c²` and every coefficient by `c`.
    fn assert_scaled(
        name: &str,
        c: f64,
        (want, m1): (&[Vec<f64>], &Robust),
        (got, mc): (&[Vec<f64>], &Robust),
    ) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            for (j, (g, w)) in g.iter().zip(w).enumerate() {
                let scaled = w * c;
                assert!(
                    g.to_bits() == scaled.to_bits() || (g.is_nan() && w.is_nan()),
                    "{name}, c = {c:e}, row {i}, target {j}: {g} against {scaled} \
                     ({:.2e} apart, relative)",
                    ((g - scaled) / scaled).abs()
                );
            }
        }
        let (bc, b1) = (mc.coefficients().unwrap(), m1.coefficients().unwrap());
        for j in 0..m1.cfg.n_targets {
            assert_eq!(
                mc.sig2[j].to_bits(),
                (m1.sig2[j] * c * c).to_bits(),
                "{name}, c = {c:e}, target {j}"
            );
            for (a, b) in bc[j].iter().zip(&b1[j]) {
                assert_eq!(a.to_bits(), (b * c).to_bits(), "{name}, c = {c:e}");
            }
        }
    }

    /// [`scaling_the_targets_scales_every_prediction_to_the_bit`] for the
    /// quantile loss. Its band is `quantile_eps`, or the floor, times `s`,
    /// and `s` was the literal 1 until the target had a residual scale, so a
    /// band drawn before one was in the target's own units. Such a band is
    /// drawn past the warm-up only: on a target's first prediction when
    /// `min_weight` holds it back past three rows per coefficient, and while
    /// every residual so far is exactly zero (the first target, held at zero
    /// here past its warm-up). Now such a row is a least-squares row, as in
    /// the warm-up (task 177). At `tau = 0.5` an in-band row is a
    /// least-squares row anyway, so only a nudge parted the units there.
    #[test]
    fn scaling_the_targets_scales_every_quantile_prediction_to_the_bit() {
        let mut rows = units_rows();
        for row in rows.iter_mut().take(24) {
            row.1[0] = Some(0.0);
        }
        // Predictions, the model after the stream, the rows before a scale
        // that the literal 1 would have decided otherwise (a band row, or a
        // nudge, in place of a least-squares row), and the rows after one
        // that the band shaped (a band row or a nudge).
        let run = |base: &RobustCfg, c: f64| {
            let mut m = Robust::new(base.clone()).unwrap();
            let (mut preds, mut decided, mut banded) = (Vec::new(), 0, 0);
            for (x, ys, d, w) in &rows {
                let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
                let p = m.predict(x, *d).pred;
                for (j, y) in ys.iter().enumerate() {
                    let Some(y) = *y else { continue };
                    if !p[j].is_finite() || *w <= 0.0 {
                        continue;
                    }
                    let row = (p[j], y, *d, *w);
                    let scale = m.residual_scale(j);
                    let update = update_of(&m, j, row, scale);
                    if scale.is_none() {
                        decided += usize::from(update_of(&m, j, row, Some(1.0)) != update);
                    } else {
                        banded += usize::from(update != RowUpdate::Fit { w: *w, target: y });
                    }
                }
                preds.push(m.step(x, &ys, *d, *w).pred);
            }
            (preds, m, decided, banded)
        };
        let quantile = |tau: f64, min_weight: f64, intercept: bool, standardize: bool| {
            let mut c = cfg(2, 3, RobustLoss::Quantile { tau });
            c.decay = Decay::Halflife(30.0);
            c.quantile_eps = 0.2;
            c.min_weight = min_weight;
            c.fit_intercept = intercept;
            c.standardize = standardize;
            c
        };
        let mut ridge_free = quantile(0.25, 20.0, false, false);
        ridge_free.ridge = 0.0;
        let cases = [
            ("tau 0.9, min_weight 3", quantile(0.9, 3.0, true, false)),
            (
                "tau 0.5, min_weight 20, standardized",
                quantile(0.5, 20.0, true, true),
            ),
            ("tau 0.25, through the origin, ridge 0", ridge_free),
            (
                "tau 0.75, through the origin, standardized",
                quantile(0.75, 3.0, false, true),
            ),
        ];
        for (name, base) in &cases {
            let (want, m1, _, banded) = run(base, 1.0);
            assert!(banded >= 100, "{name}: the band shaped {banded} rows");
            let predicted = want
                .iter()
                .filter(|p| p.iter().all(|v| v.is_finite()))
                .count();
            assert!(predicted > 200, "{name}: {predicted} rows predicted");
            let mut decided = 0;
            for c in [2f64.powi(20), 2f64.powi(-20)] {
                let (got, mc, d, _) = run(base, c);
                assert_scaled(name, c, (&want, &m1), (&got, &mc));
                decided += d;
            }
            // At `2^20` a residual before a scale is far outside the band
            // the literal drew, so a row past the warm-up was a nudge.
            assert!(decided >= 1, "{name}: the literal decided {decided} rows");
        }
    }

    // ---- data shares (docs/PLAN.md task 116, B) ----

    /// With `huber_delta` past every residual no row is down-weighted, so
    /// the band Gram is the EW Gram of the rows: `support_coef_j` is
    /// `1 − λ ((Σ + λI)⁻¹)_jj`, `Σ` the rows' centred covariance (in
    /// correlation form under `standardize`), here summed from the rows
    /// with no decay and inverted by `crate::oracle`'s LU.
    #[test]
    fn the_data_shares_are_the_shrinkage_diagonal() {
        for standardize in [false, true] {
            let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
            c.ridge = 0.4;
            c.standardize = standardize;
            let mut m = Robust::new(c).unwrap();
            let mut s = 7u64;
            let mut rows = Vec::new();
            for i in 0..200 {
                let a = 1.5 * lcg(&mut s);
                let x = [a, 0.6 * lcg(&mut s) + 0.5 * a];
                let y = 1.0 + x[0] - x[1] + 0.3 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                rows.push(x);
            }
            let n = rows.len() as f64;
            let mean = [0, 1].map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n);
            let mut a = [0.0; 4];
            for r in &rows {
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    a[i * 2 + j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                }
            }
            if standardize {
                let sd = [a[0].sqrt(), a[3].sqrt()];
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    a[i * 2 + j] /= sd[i] * sd[j];
                }
            }
            a[0] += 0.4;
            a[3] += 0.4;
            let inv = crate::oracle::inverse(&a);
            let got = m.support_coef().unwrap();
            assert!(got[0][0].is_nan(), "the intercept is not a share");
            // Task 274: a share moves by `0.4·Δ(A⁻¹)ⱼⱼ`, and `(A⁻¹)ⱼⱼ` by
            // `cⱼᵀ·ΔA·cⱼ` (`cⱼ` its column) for every error `ΔA` of the system
            // on either side: the model's co-moments, carried along its
            // weighted Welford run ([`crate::ewcov::rounding::Rounding`]); the
            // test's, two-pass, each term three roundings and each addition
            // `u` of its partial sum, the means' errors second order; under
            // `standardize` each side's correlation `C_ij/√(C_ii C_jj)`, its
            // inputs' errors through `ΔC_ij/(sᵢsⱼ) + ½|R_ij|(ΔC_ii/C_ii +
            // ΔC_jj/C_jj)` and three roundings; the ridge's addition; the
            // model's `diag(A⁻¹)`, a Cholesky solve per column (its backward
            // error); and the oracle's inverse, by its residual. It was
            // `1e-9`.
            // Measured on aarch64 (macOS): at most 0.0071 (plain) and 0.0020 (standardized) of
            // the bound: two hundred rows with no decay, the moments' worst-case running bound
            // linear in their count.
            use crate::oracle::{self, U, abs_form};
            let mut tr = crate::ewcov::rounding::Rounding::new(2, 0.0);
            for r in &rows {
                tr.update(r, 1.0, 1.0);
            }
            let mut et = [0.0f64; 4];
            let mut acc = [0.0f64; 4];
            for r in &rows {
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let t = (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                    acc[i * 2 + j] += t;
                    et[i * 2 + j] += 3.0 * U * t.abs() + U * acc[i * 2 + j].abs();
                }
            }
            let cmat = acc;
            let to_system = |e: &[f64]| -> Vec<f64> {
                if !standardize {
                    return e.to_vec();
                }
                let sd = [cmat[0].sqrt(), cmat[3].sqrt()];
                (0..4)
                    .map(|ij| {
                        let (i, j) = (ij / 2, ij % 2);
                        let r = cmat[ij] / (sd[i] * sd[j]);
                        e[ij] / (sd[i] * sd[j])
                            + 0.5 * r.abs() * (e[i * 3] / cmat[i * 3] + e[j * 3] / cmat[j * 3])
                            + 3.0 * U * r.abs()
                    })
                    .collect()
            };
            let (dl, dt) = (to_system(&tr.ec), to_system(&et));
            let e = oracle::cholesky_backward(&a);
            let own = oracle::inverse_error(&a);
            for j in 0..2 {
                let want = 1.0 - 0.4 * inv[j * 2 + j];
                let cj = [inv[j], inv[2 + j]];
                let da: Vec<f64> = (0..4)
                    .map(|ij| {
                        dl[ij]
                            + dt[ij]
                            + e[ij]
                            + if ij % 3 == 0 {
                                2.0 * U * a[ij].abs()
                            } else {
                                0.0
                            }
                    })
                    .collect();
                let dinv = abs_form(&cj, &da, &cj) + own[j * 2 + j];
                let bound = 0.4 * dinv + 2.0 * U * (want.abs() + 0.4 * inv[j * 2 + j].abs());
                oracle::assert_within(
                    got[0][j + 1],
                    want,
                    bound,
                    format_args!("standardize {standardize}, share {j}"),
                );
            }
        }
    }

    /// A duplicated pair splits its coefficient evenly in the band Gram
    /// either loss keeps, through the origin as well; a target no solve
    /// has fit has no shares; and the shares survive a state round trip.
    #[test]
    fn a_duplicated_pair_reads_half_under_either_loss() {
        for loss in [
            RobustLoss::Huber { delta: 1.345 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            for fit_intercept in [true, false] {
                let mut c = cfg(3, 2, loss);
                c.fit_intercept = fit_intercept;
                let mut m = Robust::new(c).unwrap();
                assert!(m.support_coef().is_none(), "before the first solve");
                let mut s = 11u64;
                for i in 0..400 {
                    let a = lcg(&mut s);
                    let x = [a, lcg(&mut s), a];
                    let y = 1.0 + 2.0 * a - x[1] + 0.3 * lcg(&mut s);
                    m.step(&x, &[Some(y), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                }
                let got = m.support_coef().unwrap();
                let off = usize::from(fit_intercept);
                let shares = &got[0][off..];
                assert!(
                    (shares[0] - 0.5).abs() < 1e-3
                        && (shares[1] - 1.0).abs() < 1e-3
                        && (shares[2] - 0.5).abs() < 1e-3,
                    "{loss:?}, intercept {fit_intercept}: {:?}",
                    got[0]
                );
                assert!(got[1].iter().all(|v| v.is_nan()), "no row, no fit");
                let back = Robust::restore(&m.state()).unwrap();
                assert_eq!(back, m);
                let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                let again: crate::State = rmp_serde::from_slice(&bytes).unwrap();
                // Compared by their bits: the intercept's share is NaN.
                assert_eq!(Robust::restore(&again).unwrap(), m);
            }
        }
    }

    /// The shares wait for a read (`PendingShares`): read now, read after
    /// the stream settles them, or saved, they are the same bits, and a
    /// settled model holds no factor for them.
    #[test]
    fn the_shares_are_the_same_whenever_they_are_taken() {
        for loss in [
            RobustLoss::Huber { delta: 1.345 },
            RobustLoss::Quantile { tau: 0.3 },
        ] {
            let mut m = Robust::new(cfg(3, 1, loss)).unwrap();
            let mut s = 23u64;
            for i in 0..150 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                let y = 1.0 + x[0] - x[2] + 0.3 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            assert_eq!(m.pending_readiness(), 1, "{loss:?}: the last solve waits");
            let unread = m.clone();
            let read = m.support_coef().unwrap();
            m.settle_readiness();
            assert_eq!(m.pending_readiness(), 0);
            let settled = m.support_coef().unwrap();
            let bits = |v: &[Vec<f64>]| v.concat().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&read), bits(&settled), "{loss:?}");
            assert_eq!(unread, m, "equal on the values a read gives");
            let saved = rmp_serde::to_vec(&unread.state()).unwrap();
            assert_eq!(saved, rmp_serde::to_vec(&m.state()).unwrap());
            // The shares are a ridge's: below 1, above 0.
            assert!(
                settled[0][1..].iter().all(|v| *v > 0.9 && *v < 1.0),
                "{settled:?}"
            );
        }
    }

    /// A stream of `rows` rows whose first and third features are one
    /// column twice, the second a column of its own; target 1 null
    /// throughout when `second` is false.
    fn duplicated(m: &mut Robust, rows: usize, second: bool) {
        let mut s = 11u64;
        for i in 0..rows {
            let a = lcg(&mut s);
            let x = [a, lcg(&mut s), a];
            let y = 1.0 + 2.0 * a - x[1] + 0.3 * lcg(&mut s);
            let y1 = second.then(|| 0.5 - a + 0.3 * lcg(&mut s));
            let ys: Vec<Option<f64>> = [Some(y), y1][..m.cfg.n_targets].to_vec();
            m.step(&x, &ys, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
    }

    /// With no ridge, a feature the rows never move leaves a row and a
    /// column of 0 in the band Gram, whose factor then takes the jitter
    /// ladder's shift `δ`: the shares are the shifted system's,
    /// `1 − (ridge + δ) (A⁻¹)_jj` with `A` carrying `δ`
    /// (`Robust::pending_shares`), so the still feature reads 0, the data
    /// carrying nothing of it, and the others all but 1, under either loss
    /// (docs/PLAN.md task 220). Read with the shift taken off, the still
    /// feature's share was 2, clamped to 1.
    #[test]
    fn a_feature_the_rows_never_move_reads_no_share_through_the_jitter() {
        for loss in [
            RobustLoss::Huber { delta: 1.345 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            let mut c = cfg(3, 1, loss);
            c.ridge = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 17u64;
            for i in 0..200 {
                let x = [lcg(&mut s), 0.25, lcg(&mut s)];
                let y = 1.0 + 2.0 * x[0] - x[2] + 0.3 * lcg(&mut s);
                let before = m.solve_failures;
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                if m.solve_failures == before || i < 10 {
                    continue;
                }
                let got = m.support_coef().unwrap();
                let shares = &got[0][1..];
                assert!(
                    shares[1].abs() < 1e-9
                        && [shares[0], shares[2]]
                            .iter()
                            .all(|v| *v > 0.99 && *v <= 1.0),
                    "{loss:?}, row {i}: {:?}",
                    got[0]
                );
            }
            assert!(m.solve_failures > 100, "{loss:?}: {}", m.solve_failures);
        }
    }

    /// A target whose solve fails keeps its fit and the shares of that fit
    /// (`Robust::solve`): stored, as the stream leaves them at the end of a
    /// run, and then a solve that fails at every jitter, the accumulator
    /// handed a correlation of 2 as `a_standardized_solve_that_fails_outright_is_counted`
    /// hands it. The shares keep their bits (docs/PLAN.md task 220).
    #[test]
    fn a_solve_that_fails_keeps_the_shares_of_the_fit_it_keeps() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.standardize = true;
        let mut m = Robust::new(c).unwrap();
        let mut s = 103u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        m.settle_readiness();
        let (beta, shares) = (
            m.coefficients().unwrap().to_vec(),
            m.support_coef().unwrap(),
        );
        assert!(
            shares[0][1..].iter().all(|v| *v > 0.0 && *v < 1.0),
            "{shares:?}"
        );
        let before = m.solve_failures;
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            w,
            q,
        );
        m.solve();
        assert_eq!(m.solve_failures, before + 1, "every jitter failed");
        assert_eq!(m.coefficients().unwrap(), beta, "the fit is kept");
        let bits = |v: &[Vec<f64>]| v.concat().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&m.support_coef().unwrap()), bits(&shares));
    }

    /// A target no solve has fit has no shares, NaN, whatever a file held
    /// for it: one edited to hold numbers there loads, its shape being
    /// right, and the next solve, which skips the target, clears them
    /// (`Shares::unfit`; docs/PLAN.md task 220). The fitted target's are
    /// the solve's.
    #[test]
    fn a_target_no_solve_has_fit_holds_no_shares_whatever_a_file_held() {
        let mut m = Robust::new(cfg(3, 2, RobustLoss::Huber { delta: 1.345 })).unwrap();
        duplicated(&mut m, 60, false);
        m.settle_readiness();
        let mut s = m.state();
        let ModelState::Robust(inner) = &mut s.model else {
            unreachable!()
        };
        assert!(inner.support.stored[1].iter().all(|v| v.is_nan()));
        inner.support.stored[1] = vec![0.5; 4];
        let mut back = Robust::restore(&s).unwrap();
        back.step(&[0.1, 0.2, 0.1], &[Some(1.0), None], 1.0, 1.0);
        let got = back.support_coef().unwrap();
        assert!(got[1].iter().all(|v| v.is_nan()), "{got:?}");
        assert!(got[0][1..].iter().all(|v| v.is_finite()), "{got:?}");
    }

    /// Two models are equal only where their shares are, by their bits:
    /// one share a rounding step apart makes them two (`Shares`'s
    /// `PartialEq`; docs/PLAN.md task 220).
    #[test]
    fn models_whose_shares_differ_are_not_equal() {
        let mut m = Robust::new(cfg(3, 1, RobustLoss::Huber { delta: 1.345 })).unwrap();
        duplicated(&mut m, 60, false);
        m.settle_readiness();
        assert_eq!(m, m.clone());
        let mut other = m.clone();
        other.support.stored[0][2] = other.support.stored[0][2].next_up();
        assert_ne!(m, other);
    }

    /// The data shares a state carries are a solve's shape, each break
    /// refused alone (docs/PLAN.md task 220): after a solve, a target's
    /// shares a value short, and one target too many with each row of the
    /// right width; before the first solve, any.
    #[test]
    fn a_state_whose_shares_have_the_wrong_shape_is_refused() {
        let fresh = Robust::new(cfg(3, 2, RobustLoss::Huber { delta: 1.345 })).unwrap();
        let mut m = fresh.clone();
        duplicated(&mut m, 60, true);
        m.settle_readiness();
        type Spoil = (&'static str, fn(&mut Shares));
        let cases: [Spoil; 2] = [
            ("a target's shares a value short", |s| {
                s.stored[1].pop();
            }),
            ("a target too many", |s| s.stored.push(vec![0.5; 4])),
        ];
        for (what, spoil) in cases {
            for base in [&fresh, &m] {
                let mut s = base.state();
                let ModelState::Robust(inner) = &mut s.model else {
                    unreachable!()
                };
                if inner.support.stored.is_empty() {
                    inner.support.stored = vec![vec![0.5; 4]; 2];
                }
                spoil(&mut inner.support);
                match Robust::restore(&s) {
                    Err(StateError::Invalid(e)) => {
                        assert!(e.contains("data shares"), "{what}: {e}");
                    }
                    other => panic!("{what}: {other:?}"),
                }
            }
        }
        assert!(Robust::restore(&fresh.state()).is_ok() && Robust::restore(&m.state()).is_ok());
    }

    /// Task 252: through the origin and standardized, features that are
    /// always 0 have no raw second moment, so the band system keeps no
    /// column: no factorization, every coefficient 0, nothing counted as a
    /// failure -- in the solve and in the nudges of the rows outside the
    /// band alike, whose leverage is then 0.
    #[test]
    fn features_with_no_second_moment_keep_no_column_through_the_origin() {
        for loss in [
            RobustLoss::Huber { delta: 1.0 },
            RobustLoss::Quantile { tau: 0.5 },
        ] {
            let mut c = cfg(2, 1, loss);
            c.fit_intercept = false;
            c.standardize = true;
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 21u64;
            for i in 0..40 {
                let y = 1.0 + 3.0 * lcg(&mut s);
                let out = m.step(&[0.0, 0.0], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
                if i > 0 {
                    assert_eq!(out.pred[0], 0.0, "{loss:?}, row {i}");
                }
            }
            assert_eq!(m.coefficients().unwrap()[0], vec![0.0, 0.0], "{loss:?}");
            assert_eq!(m.solve_failures, 0, "{loss:?}");
        }
    }

    /// Task 252: with an intercept and standardized, a feature that never
    /// moves has no usable variance and is dropped, so a quantile nudge
    /// from a row outside the band reads a band system of no column: its
    /// leverage is the intercept's alone, and it is taken -- the target's
    /// mean moves.
    #[test]
    fn a_nudge_against_a_band_system_of_no_column_reads_the_intercepts_leverage() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.standardize = true;
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 22u64;
        for i in 0..30 {
            let y = 2.0 + 0.1 * lcg(&mut s);
            m.step(&[1.0], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
        }
        m.systems.0[0] = None;
        let ybar = m.ybar[0];
        m.step(&[1.0], &[Some(50.0)], 1.0, 1.0);
        assert!(
            m.ybar[0] > ybar,
            "the nudge was taken: {} against {ybar}",
            m.ybar[0]
        );
        if let Some(sys) = m.systems.0[0].as_ref() {
            assert!(sys.keep.is_empty() && sys.factor.is_none());
        }
    }

    /// Task 252: a quantile nudge from a row that deviates on a feature
    /// with no spread among the rows in the band has no curvature to lean
    /// on: its movement is unbounded and its step 0, so neither the
    /// target's mean nor its cross-moments move.
    #[test]
    fn a_nudge_along_a_feature_with_no_spread_takes_no_step() {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..40 {
            let x1 = lcg(&mut s);
            let y = 2.0 * x1 + 0.01 * lcg(&mut s);
            m.step(&[1.0, x1], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
        }
        let (ybar, lo, cross) = (m.ybar[0], m.ybar_lo[0], m.cross[0].clone());
        let out = m.step(&[2.0, 0.5], &[Some(100.0)], 1.0, 1.0);
        assert!(out.pred[0].is_finite(), "the row was scored");
        assert_eq!(
            (m.ybar[0].to_bits(), m.ybar_lo[0].to_bits()),
            (ybar.to_bits(), lo.to_bits())
        );
        assert_eq!(m.cross[0], cross);
    }

    /// A late target's own first solve reads whether a solve has fit *it*:
    /// under feature sets each target has one slot per set, target-major,
    /// and the one read is its own weighing set's. With two targets and two
    /// sets the late target's slot is the fourth; the first target's
    /// weighing slot, the second, was fit at row 1, and read in its place
    /// the late target would wait for the row cap, 25 rows after it, null
    /// meanwhile (docs/PLAN.md task 272 (m)).
    #[test]
    fn a_late_targets_first_solve_reads_its_own_slot_under_feature_sets() {
        let mut c = cfg(2, 2, RobustLoss::Huber { delta: 1.5 });
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 25;
        c.feature_sets = vec![("a".into(), vec![0]), ("all".into(), vec![0, 1])];
        let mut m = Robust::new(c).unwrap();
        assert_eq!((m.cfg.n_sets(), m.cfg.weighing_set()), (2, 1));
        let mut s = 4u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let late = (i >= 10).then(|| 1.0 - 2.0 * x[0] + x[1] + 0.01 * lcg(&mut s));
            let y0 = 0.5 + x[0] - x[1] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y0), late], f64::from(u8::from(i > 0)), 1.0);
            if i >= 1 {
                let beta = m.coefficients().expect("solved at row 1");
                assert!(beta[..2].iter().flatten().all(|v| v.is_finite()), "row {i}");
                // Its own first solve at the end of its second row, row 11.
                for slot in [2, 3] {
                    assert_eq!(
                        beta[slot].iter().all(|v| v.is_finite()),
                        i >= 11,
                        "row {i}, slot {slot}: {beta:?}"
                    );
                }
            }
        }
    }

    /// A target that the first solve fit -- present on the rows before it,
    /// below its own threshold -- has no first solve of its own when its
    /// weight later reaches that threshold: a solve has fit it, and the
    /// next falls on the cadence, here the row cap 25 rows on, so the
    /// coefficients hold until then (docs/PLAN.md task 195, S9b, and 272
    /// (m)).
    #[test]
    fn a_target_the_first_solve_fit_waits_for_the_cadence_at_its_own_threshold() {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 25;
        let mut m = Robust::new(c).unwrap();
        m.set_target_min_weight(vec![2.0, 5.0]).unwrap();
        let mut s = 6u64;
        let mut first = None;
        for i in 0..24 {
            let x = [lcg(&mut s)];
            let y = [
                Some(0.5 + x[0] + 0.01 * lcg(&mut s)),
                Some(1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s)),
            ];
            m.step(&x, &y, f64::from(u8::from(i > 0)), 1.0);
            if i == 1 {
                let beta = m.coefficients().expect("the first solve, at row 1");
                assert!(beta.iter().flatten().all(|v| v.is_finite()), "both fit");
                assert!(
                    m.wobs[1] < 5.0,
                    "the second target is below its own threshold"
                );
                first = Some(beta.to_vec());
            }
            if i > 1 {
                assert_eq!(m.coefficients(), first.as_deref(), "row {i}");
            }
        }
        assert!(
            m.wobs[1] >= 5.0,
            "the second target's weight passed its threshold"
        );
    }

    /// A target whose weight its decay has taken to exactly 0 has forgotten
    /// its rows, as every model forgets what a decay that underflows takes
    /// (`model_contract.rs`): it predicts nothing, while a target with
    /// weight goes on, though the fit the last solve made for it is kept
    /// for when its rows return (docs/PLAN.md task 115 (c), and 272 (m)).
    #[test]
    fn a_target_whose_weight_has_decayed_to_nothing_predicts_nothing() {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(1.0);
        c.min_weight = 1.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..30 {
            let x = [lcg(&mut s)];
            let y = [
                Some(0.5 + x[0]),
                Some(1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s)),
            ];
            m.step(&x, &y, f64::from(u8::from(i > 0)), 1.0);
        }
        // A gap of 2,000 halflives, on a row the second target is absent
        // from: its weight decays to 0, the first's starts over.
        m.step(&[0.3], &[Some(0.8), None], 2000.0, 1.0);
        assert_eq!(m.wj[1], 0.0);
        assert!(m.wj[0] > 0.0 && m.w_raw >= 1.0);
        let beta = m.coefficients().unwrap();
        assert!(beta[1].iter().all(|v| v.is_finite()), "the kept fit");
        let p = m.predict(&[0.3], 0.0);
        assert!(p.pred[0].is_finite(), "{:?}", p.pred);
        assert!(p.pred[1].is_nan(), "{:?}", p.pred);
    }
}

#[cfg(test)]
#[path = "robust/scale_tests.rs"]
mod scale_tests;

#[cfg(test)]
#[path = "robust/options_tests.rs"]
mod options_tests;

#[cfg(test)]
#[path = "robust/warm_tests.rs"]
mod warm_tests;
