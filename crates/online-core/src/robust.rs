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
//! RMS, the EW residual std; under the MAD, below, the EW median of `|e|`
//! over `Φ⁻¹(3/4)`, which a quantile fit's band reads too). The target has one once it is
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
//! **`min_weight` withholds the output alone** (docs/PLAN.md task 278,
//! D10). The fit learns from the prediction it makes on a row the gate
//! withholds: the scale takes its residual, Huber weighs it, the band is
//! drawn around it. So the fit is the same at any `min_weight`. What a
//! target's first prediction waits for, reported or learned from, is the
//! model's own floor ([`RobustCfg::floor`]): its first row per coefficient
//! with weight, counted once and again after its history is forgotten, the
//! row it reaches them solving it. Before schema 64 the gate held learning
//! back instead -- a withheld row taught the scale nothing and Huber took it
//! at weight 1, and the first solve waited for `min_weight` -- and a state
//! written then goes on that way ([`RobustCfg::min_weight_holds_learning`]).
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
//!   converged `RLM` with the MAD gives 2.039. Under
//!   [`RobustCfg::mad_anchor`], which a spec sets, each sketch keeps `|e|`
//!   in units of its first ([`EwQuantile::anchored`], task 278), so a target
//!   rescaled by any factor rescales `s`, and the fit, to rounding; on the
//!   fixed buckets a factor other than a power of two moved `m` by up to a
//!   bucket.
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
//! shorter memory follows the shift sooner (docs/HUBER.md §4).
//!
//! **A quantile fit's band is drawn in the MAD too**, on the model's clock
//! (docs/PLAN.md task 278, D-Q). In the RMS, the only scale before schema
//! 64 and what a quantile state written before it keeps, a wild row
//! inflated the band drawn to bound its pull: at 5% one-sided wild targets
//! (a hundred noise deviations) the RMS stood near 11 against a noise of
//! 0.5, the band of 0.2 of it held every clean row as a least-squares row
//! and gave every wild row a pull of the band's width, and all 50 seeds
//! ended more than 0.05 from statsmodels' `QuantReg` on the same rows (the
//! worst 0.098 at 5%, 0.98 at 20%). The median of `|e|` moves by a wild
//! row's weight, not its size, so the band stays the clean rows' width.
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

use crate::huber_refit::{
    Event, REFIT_PASSES, RefitBuf, Rows, converged, fitted, huber_weights, rebuild_scale, scale_of,
};
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

/// The octaves a MAD sketch keeps either side of its median's bucket
/// ([`EwQuantile::bounded`]): 16 octaves in all, so a median that falls
/// from the start-up residuals to the noise's by up to a factor of 2^16
/// folds nothing it later reads (docs/PLAN.md task 271 (b)).
pub const MAD_BAND_OCTAVES: u32 = 8;

/// One sketch of `|e|` per target under the MAD, none under the RMS;
/// anchored ([`EwQuantile::anchored`]) where the cfg says so
/// ([`RobustCfg::mad_anchor`]).
pub(crate) fn mad_sketches(scale: HuberScale, anchored: bool, n: usize) -> Vec<EwQuantile> {
    match scale {
        HuberScale::Mad => vec![
            if anchored {
                EwQuantile::anchored(&[0.5], MAD_BAND_OCTAVES)
            } else {
                EwQuantile::bounded(&[0.5], MAD_BAND_OCTAVES)
            }
            .expect("0.5 is in (0, 1) and the band in range");
            n
        ],
        HuberScale::Rms => Vec::new(),
    }
}

/// Whether a restored state's sketches are the ones its scale keeps: `n`
/// sketches at the level 0.5 under the MAD, each one a sketch that can be
/// read and added to and anchored exactly when the cfg says so, and none
/// under the RMS.
pub(crate) fn mad_sketches_fit(
    scale: HuberScale,
    anchored: bool,
    n: usize,
    q: &[EwQuantile],
) -> bool {
    match scale {
        HuberScale::Mad => {
            q.len() == n
                && q.iter().all(|s| {
                    s.levels() == [0.5] && s.has_shape() && s.anchor().is_some() == anchored
                })
        }
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
    /// units of the residual scale `s`: rows inside carry the curvature, rows
    /// outside only the score (the module docs). It was the floor under `|r|`
    /// in an IRLS weight, bounding a weight rather than naming a band (review
    /// 2026-09-12, N9).
    pub quantile_eps: f64,
    /// The scale the Huber cut is drawn in (the module docs). A state
    /// written before schema 63 has no such field and was fitted under the
    /// RMS, so it loads as [`HuberScale::Rms`]
    /// ([`HuberScale::of_a_state_before_63`]), where a spec's default is
    /// the MAD. A quantile fit's band is drawn in it too: a spec builds
    /// [`HuberScale::Mad`], and a quantile state written before schema 64
    /// says [`HuberScale::Rms`], the band's only scale then, and goes on in
    /// it (docs/PLAN.md task 278, D-Q).
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
    /// Each target's first `refit_rows` learned rows refit to convergence,
    /// once, and its residual scale rebuilt from the converged residuals
    /// ([`crate::huber_refit`]): a spec's `huber_refit_rows`, 200 by
    /// default; 0 is off. Huber only, and not beside `ridge_scale`. A state
    /// written before schema 64 has no such field and was fitted without a
    /// refit, so it loads as 0 and goes on as it was, never refitting
    /// rows its stream has long passed.
    #[serde(default)]
    pub refit_rows: u32,
    /// Whether each MAD sketch buckets `|e|` in units of the first `|e|` it
    /// takes ([`EwQuantile::anchored`]), so the scale, and the fit it
    /// weighs, moves with a rescaled target by the factor to rounding,
    /// where the fixed buckets moved it by up to a bucket at any factor
    /// but a power of two. A spec sets it. A state written before schema
    /// 64 has no such field and its sketches bucket `|e|` as it is, so it
    /// loads as `false` and goes on as it was, to the bit.
    #[serde(default)]
    pub mad_anchor: bool,
    /// Whether `min_weight` holds learning back as well as the output, as
    /// it did before schema 64: a row it withheld had no prediction, so
    /// its residual taught the scale nothing and Huber weighed it at 1,
    /// and the first solve waited for the row whose weight reached it (and
    /// a late target's for its own threshold). A spec builds `false`:
    /// `min_weight` withholds the output alone, the fit learns from the
    /// prediction it would have made, its first solve falls on the first
    /// row with weight, and a target's first rows each solve it until it
    /// holds a row per coefficient ([`Robust`]'s `own_solve`), so a fit is
    /// the same with any `min_weight`. A state written before
    /// schema 64 has no such field and was fitted the old way, so it loads
    /// as `true` and goes on to the bit
    /// ([`RobustCfg::gate_held_learning_before_64`]).
    #[serde(default = "RobustCfg::gate_held_learning_before_64")]
    pub min_weight_holds_learning: bool,
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

    /// What a configuration written before schema 64, which had no
    /// [`Self::min_weight_holds_learning`], was fitted with: a gate that held
    /// learning back too. It must go on as one, to the bit (CLAUDE.md hard
    /// rule 5).
    pub fn gate_held_learning_before_64() -> bool {
        true
    }

    /// The weight at which a fresh model's first solve fires, where the
    /// gate holds learning back: `min_weight` ([`Self::min_weight_holds_learning`]).
    /// Otherwise no weight fires it: each target's own floor does
    /// ([`Robust`]'s `own_solve`).
    fn first_solve_at(&self) -> f64 {
        if self.min_weight_holds_learning {
            self.min_weight
        } else {
            f64::INFINITY
        }
    }

    /// The model's own floor: a target predicts once it has been present,
    /// with weight, on a row per coefficient, so no prediction, reported or
    /// learned from, is read off a fit of fewer rows than unknowns, which
    /// only the ridge determines. A count of its first rows, not decayed:
    /// a fit needs them once, and a half-life too short to hold them at a
    /// time (one row with one feature) is the caller's memory to choose,
    /// not a fit to withhold for good; a count, so a weight's scale does
    /// not reach it (task 147). Not `min_weight`, a gate on the output
    /// alone (task 278, D10). A state written before schema 64 has none:
    /// its gate stood in this floor's place. Under feature sets the
    /// coefficients counted are the weighing set's, whose prediction the
    /// row is learned from ([`Self::weighing_set`]), so that set is the fit
    /// of a `huber` on its features alone, to the bit.
    fn floor(&self) -> u32 {
        if self.min_weight_holds_learning {
            return 0;
        }
        let k = self
            .set_of(self.weighing_set())
            .map_or(self.n_features, <[usize]>::len)
            + usize::from(self.fit_intercept);
        u32::try_from(k).unwrap_or(u32::MAX)
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
        if self.refit_rows > 0
            && (!matches!(self.loss, RobustLoss::Huber { delta } if delta.is_finite())
                || self.ridge_scale)
        {
            return Err(format!(
                "{kind}: huber_refit_rows is for huber with a finite huber_delta, and not \
                 beside ridge_scale \"sum\": an infinite cut is least squares, with no \
                 outliers to refit for, and the sum-scale prior ages with every row before \
                 the first, which a refit of the buffered rows does not hold"
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
                // The band is drawn in the residuals' MAD (a spec's), or
                // their EW root mean square (a state written before schema
                // 64), on the model's clock: a memory of its own is
                // Huber's (docs/PLAN.md tasks 271 and 278, D-Q).
                if self.huber_scale_decay.is_some() {
                    return Err(format!("{kind}: huber_scale_half_life is for huber"));
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
    /// Per target, the refit's buffer ([`RobustCfg::refit_rows`]): none
    /// when it is off, and in a state written before schema 64.
    #[serde(default)]
    refit: Vec<RefitBuf>,
    /// Per target, the rows with weight it has been present on, counted to
    /// the model's floor and no further ([`RobustCfg::floor`]): what its
    /// first prediction and first solve wait for. None in a state written
    /// before schema 64, and under the gate that held learning back, which
    /// has no floor.
    #[serde(default)]
    seen: Vec<u32>,
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
            // A quantile fit's band reads the same scale as Huber's cut
            // (docs/PLAN.md task 278, D-Q).
            mad: mad_sketches(cfg.huber_scale, cfg.mad_anchor, m),
            nudge: NudgeScratch::default(),
            refit: RefitBuf::per_target(cfg.refit_rows, m),
            seen: if cfg.min_weight_holds_learning {
                Vec::new()
            } else {
                vec![0; m]
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

    /// Whether this row solves the fit for target `j`'s sake, outside the
    /// cadence: the row on which it reaches the model's floor, a row per
    /// coefficient ([`RobustCfg::floor`]), so the first row it predicts
    /// reads a fit of its own rows, as a fresh model's and a target that
    /// joins after the first solve alike, not the cadence's next solve. It
    /// reads no `min_weight`, which withholds the output alone (task 278,
    /// D10).
    ///
    /// Under the gate that held learning back (a state written before
    /// schema 64; [`RobustCfg::min_weight_holds_learning`]) it is the row
    /// whose own weight, before it and after it, crosses into its own
    /// `min_weight` while no solve has fit it: the row its own first solve
    /// fell on (S9b). A target's weight grows only on a row that carries
    /// it, `lam·W + w` there as in the update below.
    fn own_solve(&self, j: usize, present: bool, lam: f64, weight: f64) -> bool {
        if !present || !(weight > 0.0 && weight.is_finite()) {
            return false;
        }
        if !self.cfg.min_weight_holds_learning {
            return self.seen.get(j).is_some_and(|&n| n + 1 == self.cfg.floor());
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

    /// Whether `min_weight` withholds this row's output: the weight before
    /// the row, `n_eff`, short of it.
    fn withheld(&self) -> bool {
        let past = self.w_raw >= self.cfg.min_weight;
        !past
    }

    /// Each slot's prediction from the last solve, whatever `min_weight`
    /// says: NaN for a slot no solve has fit, for a target short of the
    /// model's floor, a row per coefficient ([`RobustCfg::floor`]), and for
    /// a target that has forgotten its rows -- its weight 0, or, since the
    /// fit learns from this prediction (task 278, D10), below the smallest
    /// normal double, where a decay one half-life longer would have taken
    /// it to 0 (task 215's line): the fit then learns the same from a
    /// history aged to `2^-1074` of its weight as from one aged to nothing.
    fn slot_predictions(&self, x: &[f64]) -> Vec<f64> {
        let nc = self.cfg.n_sets();
        let (least, floor) = if self.cfg.min_weight_holds_learning {
            (0.0, 0)
        } else {
            (f64::MIN_POSITIVE, self.cfg.floor())
        };
        let mut pred = vec![f64::NAN; self.cfg.n_targets * nc];
        if let Some(beta) = &self.beta {
            for (slot, p) in pred.iter_mut().enumerate() {
                let j = slot / nc;
                let w = self.wj[j];
                let reached = self.seen.get(j).is_none_or(|&n| n >= floor);
                if w > 0.0 && w >= least && reached {
                    *p = dot_aug(&beta[slot], x, self.cfg.fit_intercept);
                }
            }
        }
        pred
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Target `j`'s residual scale, once it has one: under
    /// [`HuberScale::Mad`] the EW median of `|e|` over [`NORMAL_Q3`]
    /// ([`mad_scale`]), for either loss, and otherwise -- the RMS --
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

    /// Per target, how its refit stands ([`RobustCfg::refit_rows`]): empty
    /// when the refit is off.
    pub fn refit_status(&self) -> Vec<crate::RefitStatus> {
        self.refit.iter().map(RefitBuf::status).collect()
    }

    /// Give every target's refit up before it begins, past the spec's
    /// `refit_budget` ([`RobustCfg::refit_rows`]): a bank does this to a
    /// group whose reservation the budget cannot take, and to the models
    /// a reset rebuilds for it.
    pub fn refit_give_up(&mut self) {
        self.refit
            .iter_mut()
            .for_each(RefitBuf::give_up_over_budget);
    }

    /// The bytes this model's refit buffers can hold at most
    /// ([`crate::refit_reservation`] a target), 0 with the refit off.
    pub fn refit_reservation(&self) -> usize {
        if self.refit.is_empty() {
            return 0;
        }
        self.cfg.n_targets * crate::refit_reservation(self.cfg.n_features, self.cfg.refit_rows)
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

    /// The refit of target `j` ([`RobustCfg::refit_rows`],
    /// [`crate::huber_refit`]): IRLS on its buffered rows to convergence,
    /// each pass's sums rebuilt by replaying the rows in their order, each
    /// aged by its own factor as the stream entered it, at `w_i · u_i`, then
    /// solved by the model's own solve on the weighing set; the first
    /// pass reads the online fit as it stands, or unit weights where there
    /// is none. Then the residual scale is rebuilt from the converged
    /// residuals, and the target's own slots are solved from the rebuilt
    /// sums if a solve had fit it -- no other target's, and the cadence is
    /// left as it stands, so no other fit moves. One that no solve has fit
    /// keeps waiting for its first, which then reads the rebuilt sums.
    #[inline(never)]
    fn refit_target(&mut self, j: usize) {
        let RobustLoss::Huber { delta } = self.cfg.loss else {
            return;
        };
        let nf = self.cfg.n_features;
        let (buf, ages) = self.refit[j].take();
        let rows = Rows::new(&buf, &ages, nf);
        let n = rows.len();
        // The scale reads the residuals at the weights its own clock has
        // left them (`huber_scale_decay`), as the stream's scale holds them.
        let vs = rows.scale_weights();
        let k = self.cfg.k_total();
        let c = self.cfg.weighing_set();
        let slot = j * self.cfg.n_sets() + c;
        let fitted_before = self.fitted(j);
        let mut beta: Option<Vec<f64>> = if fitted_before {
            self.beta.as_ref().map(|b| b[slot].clone())
        } else {
            None
        };
        let intercept = self.cfg.fit_intercept;
        let mut z = vec![0.0; k];
        let mut absr = vec![0.0; n];
        let mut u = vec![1.0; n];
        // The passes' factor jitters are the refit's, not failures of the
        // fit: the count is left as it stood, and one refit that no pass
        // solves counts once.
        let failures = self.solve_failures;
        let (mut passes, mut solved) = (0, false);
        for _ in 0..REFIT_PASSES {
            passes += 1;
            match &beta {
                Some(b) => {
                    for (i, a) in absr.iter_mut().enumerate() {
                        *a = (rows.y(i) - fitted(b, rows.x(i), intercept)).abs();
                    }
                    huber_weights(
                        &absr,
                        scale_of(self.cfg.huber_scale, &absr, &vs),
                        delta,
                        &mut u,
                    );
                }
                None => u.fill(1.0),
            }
            self.replay(j, &rows, &u, &mut z);
            let next = if intercept {
                self.solve_centred(k, j, c)
            } else {
                self.solve_through_origin(k, j, c)
            };
            let Some((next, _)) = next else {
                break;
            };
            solved = true;
            let done = converged(beta.as_deref(), &next);
            beta = Some(next);
            if done {
                break;
            }
        }
        self.solve_failures = failures;
        self.refit[j].ran(passes);
        let Some(b) = beta.filter(|_| solved) else {
            self.refit[j].fail();
            self.solve_failures += 1;
            return;
        };
        // A refit that took every pass is counted, as a seeding's Lloyd run
        // stopped at its cap is (task 278): its fit stands, the last pass's.
        if self.refit[j].status().ran_to_cap() {
            self.solve_failures += 1;
        }
        let scale = rebuild_scale(&rows, self.cfg.huber_scale, self.cfg.mad_anchor, |i| {
            rows.y(i) - fitted(&b, rows.x(i), intercept)
        });
        self.sig2[j] = scale.sig2;
        self.wsig[j] = scale.wsig;
        if let Some(q) = scale.mad {
            self.mad[j] = q;
        }
        if fitted_before && let Some(prev) = self.beta.take() {
            let mut out: Vec<Vec<f64>> = prev.to_vec();
            self.solve_target(j, Some(prev.as_slice()), &mut out);
            self.beta = Some(Fit(out));
        }
    }

    /// Target `j`'s sums rebuilt from the buffered rows at weights
    /// `w_i · u_i`, in the order the stream took them: `step`'s fit arm, row
    /// for row, and an ageing run's rows one by one ([`Event::age_by`]; one
    /// decay by `λⁿ` past 65,536 rows). So at `u = 1` -- every Huber weight 1
    /// -- the replay is the sums the stream built, to the bit, unless a run
    /// of ageings was that long. It is that only when every row the stream
    /// learned had a Huber weight of 1; otherwise it is the sums at the
    /// refit's weights, which is what it is for.
    fn replay(&mut self, j: usize, rows: &Rows<'_>, u: &[f64], z: &mut [f64]) {
        let k = self.cfg.k_total();
        let intercept = self.cfg.fit_intercept;
        self.cov[j] = EwCov::new(k).without_runs();
        self.cross[j].fill(0.0);
        self.ybar[j] = 0.0;
        self.ybar_lo[j] = 0.0;
        self.wj[j] = 0.0;
        for e in rows.events() {
            let i = match e {
                Event::Age { lam, count, .. } => {
                    let (cov, wj) = (&mut self.cov[j], &mut self.wj[j]);
                    Event::age_by(lam, count, |l| {
                        cov.decay(l);
                        *wj *= l;
                    });
                    continue;
                }
                Event::Row(i) => i,
            };
            let lam = rows.lam(i);
            let w = rows.w(i) * u[i];
            let aged = lam * self.wj[j];
            // A Huber weight of 0, from a residual of `inf`: the row ages the
            // sums alone (hard rule 9).
            if w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                self.cov[j].decay(lam);
                self.wj[j] = aged;
                continue;
            }
            if intercept {
                z[0] = 1.0;
                z[1..].copy_from_slice(rows.x(i));
            } else {
                z.copy_from_slice(rows.x(i));
            }
            let wj_new = aged + w;
            let a = aged / wj_new;
            let bb = w / wj_new;
            let dy = crate::comp::dev(rows.y(i), self.ybar[j], self.ybar_lo[j]);
            let ab_dy = a * bb * dy;
            let cov = &self.cov[j];
            for (i2, (ci, &zi)) in self.cross[j].iter_mut().zip(z.iter()).enumerate() {
                *ci = a * *ci + ab_dy * cov.deviation(i2, zi);
            }
            crate::comp::add(&mut self.ybar[j], &mut self.ybar_lo[j], bb * dy);
            self.cov[j].update(z, lam, w);
            self.wj[j] = wj_new;
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
        let prev = self.beta.take();
        for j in 0..self.cfg.n_targets {
            self.solve_target(j, prev.as_deref().map(Vec::as_slice), &mut beta);
        }
        self.beta = Some(Fit(beta));
        self.since_solve.restart();
        self.rows_since_solve = 0;
        self.weight_since_solve = 0.0;
    }

    /// Target `j`'s slots solved into `beta`, one per feature set, each from
    /// the target's sums ([`Self::solve`]'s loop body): `prev` is the fit
    /// before, which a slot whose solve fails keeps, and which says whether
    /// a slot with no weight has a fit to keep solving. The data shares move
    /// in place, slot by slot, as the fit does.
    fn solve_target(&mut self, j: usize, prev: Option<&[Vec<f64>]>, beta: &mut [Vec<f64>]) {
        let k = self.cfg.k_total();
        let nc = self.cfg.n_sets();
        self.support.shape(self.cfg.n_targets * nc, k);
        for c in 0..nc {
            let slot = j * nc + c;
            let fit_before = prev.is_some_and(|b| !b[slot].iter().any(|v| v.is_nan()));
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
                    if let Some(prev) = prev {
                        beta[slot] = prev[slot].clone();
                    }
                    // The previous fit is kept, and its shares with it.
                }
            }
        }
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

    /// On the row clock the solve cadence counts from a row before the
    /// first, as `max_rows_between_solves` does (task 278, D13).
    fn stamp_next_on_rows(&mut self, stamp: crate::Stamp) {
        self.since_solve.stamp_next_on_rows(stamp);
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
        // The fit learns from the prediction it makes, whether or not
        // `min_weight` withholds it from the output; only a state written
        // before schema 64 learns from the output's
        // (`RobustCfg::min_weight_holds_learning`). Built only on a
        // withheld row, so a row past the gate allocates nothing more.
        let own = (!self.cfg.min_weight_holds_learning && self.withheld())
            .then(|| self.slot_predictions(x));
        let pred = own.as_deref().unwrap_or(&out.pred);
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
        // A fresh model's first solve fires on its first row with weight,
        // and a target's first rows each solve it until it holds a row per
        // coefficient (`Self::own_solve`) -- a target that joins after the
        // first solve among them. It waited for the cadence's next solve,
        // null until then: under `solve_every = 1000` all 56 of its rows of
        // 80 (review round 4, CC1; docs/PLAN.md task 195, S9b). A state
        // written before schema 64 fires the first on the row the weight
        // reaches `min_weight`, and a late target's on the row its own
        // reaches its own, as it did then (`RobustCfg::first_solve_at`): the
        // gate held the fit back, and the fit moved with it (task 278, D10).
        let due = teaches
            && (by_cadence
                || self.rows_since_solve >= self.cfg.max_rows_between_solves
                || (self.beta.is_none() && self.w_raw >= self.cfg.first_solve_at())
                || (0..m).any(|j| self.own_solve(j, y[j].is_some(), lam, weight)));

        // ---- update: Huber reweights the row, the quantile linearises it ----
        // Targets whose refit buffer this row fills (`RobustCfg::refit_rows`).
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
                if let Some(b) = self.refit.get_mut(j) {
                    b.age(lam, lam_scale, self.cfg.refit_rows);
                }
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                self.wobs[j] = present;
                self.nobs[j] = rows;
                continue;
            };
            // A history aged below the smallest normal double of the row's
            // weight is forgotten, as a decay that underflows to 0 forgets it
            // (`gaps::decay_into`, task 215): kept, it gave a standardized
            // solve of the row a spread of its own, and the solve read the
            // old fit back, which the fit now learns from (task 278, D10;
            // `Self::slot_predictions` predicts nothing from such a target).
            // A state written before schema 64 keeps it, as it did. The
            // refit's buffer keeps the model's factor, a factor of 0 being
            // no row it holds (`RefitBuf::fits`): a history that far below
            // a row's weight is no history its replay reads apart from 0.
            let lam_t = if self.cfg.min_weight_holds_learning {
                lam
            } else {
                crate::gaps::decay_into(lam, self.wobs[j], weight)
            };
            let (present, rows) = (lam_t * self.wobs[j], lam_t * self.nobs[j]);
            // A row the target is present on counts at its raw weight, whatever
            // the loss then does with it (hard rule 8), and as one row.
            let counts = weight > 0.0 && weight.is_finite();
            self.wobs[j] = present + if counts { weight } else { 0.0 };
            self.nobs[j] = rows + if counts { 1.0 } else { 0.0 };
            // The floor's count of the target's rows, which starts over
            // where its history is gone -- forgotten above, or aged to
            // nothing while it was absent -- as a fresh target's does.
            let floor = self.cfg.floor();
            if let Some(n) = self.seen.get_mut(j)
                && counts
            {
                if present == 0.0 {
                    *n = 0;
                }
                if *n < floor {
                    *n += 1;
                }
            }
            let scale = self.residual_scale(j);
            let aged = lam_t * self.wj[j];
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
                    if let Some(b) = self.refit.get_mut(j) {
                        b.age(lam, lam_scale, self.cfg.refit_rows);
                    }
                    self.cov[j].decay(lam_t);
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
                    self.cov[j].update(&self.zbuf, lam_t, w);
                    self.wj[j] = wj_new;
                    self.move_band_system(j, a, step);
                    if let Some(b) = self.refit.get_mut(j)
                        && b.learn(x, yj, weight, lam, lam_scale, self.cfg.refit_rows)
                    {
                        refits.push(j);
                    }
                }
                RowUpdate::Nudge { nudge } => {
                    // Outside the band a row is one term of the score and none
                    // of the Hessian: no weight in the Gram, and `2h*psi(r)*z`
                    // into the cross-moment, which is a mean over `wj` -- so a
                    // sum's worth of nudge enters divided by it (N9). Centred,
                    // that is `ȳ += nudge/wj` and `c += nudge·(z − m)/wj`.
                    self.cov[j].decay(lam_t);
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

        if due {
            self.solve();
        }
        // The refits, after the row's own update, scale and solve: the row
        // was predicted before any of them, so it stays out of sample (hard
        // rule 2), and each solves its own target alone (`refit_target`).
        for &j in &refits {
            self.refit_target(j);
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
        let pred = if self.withheld() {
            vec![f64::NAN; self.cfg.n_targets * self.cfg.n_sets()]
        } else {
            self.slot_predictions(x)
        };
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
                if !mad_sketches_fit(m.cfg.huber_scale, m.cfg.mad_anchor, n, &m.mad) {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the residual sketches do not match huber_scale"
                    )));
                }
                // The refit's buffers (`RobustCfg::refit_rows`): one per
                // target when it is on, each one this configuration writes,
                // and none when it is off.
                let refit_fits = if m.cfg.refit_rows == 0 {
                    m.refit.is_empty()
                } else {
                    m.refit.len() == n
                        && m.refit
                            .iter()
                            .all(|b| b.fits(m.cfg.n_features, m.cfg.refit_rows))
                };
                if !refit_fits {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the refit's buffers do not match huber_refit_rows"
                    )));
                }
                // The floor's counts: one a target, none above the floor,
                // where the gate leaves learning alone; none where it held
                // learning back (`RobustCfg::min_weight_holds_learning`).
                let seen_fits = if m.cfg.min_weight_holds_learning {
                    m.seen.is_empty()
                } else {
                    m.seen.len() == n && m.seen.iter().all(|&c| c <= m.cfg.floor())
                };
                if !seen_fits {
                    return Err(StateError::Invalid(format!(
                        "{kind}: the floor's row counts do not match the targets"
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
#[path = "robust/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "robust/scale_tests.rs"]
mod scale_tests;

#[cfg(test)]
#[path = "robust/options_tests.rs"]
mod options_tests;

#[cfg(test)]
#[path = "robust/refit_tests.rs"]
mod refit_tests;

#[cfg(test)]
#[path = "robust/gate_tests.rs"]
mod gate_tests;
