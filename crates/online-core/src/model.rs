//! The core contract every model implements (docs/PLAN.md §2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{MIN_SCHEMA_VERSION, SCHEMA_VERSION};

/// Output of one [`OnlineModel::step`].
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// Per output slot (targets, or targets x grid combos); NaN when not ready.
    pub pred: Vec<f64>,
    /// EW count of observations (with the model's decay).
    pub n_eff: f64,
    /// Model-specific extras.
    pub extra: Option<Extra>,
}

/// Model-specific step extras (docs/PLAN.md §4).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Extra {
    /// Lasso path: selected lambda per target (docs/PLAN.md §4.3).
    Lasso { lam_selected: Vec<f64> },
}

/// Versioned, serializable model state. The bank wraps this with its own header
/// (spec, package version) before writing msgpack (docs/PLAN.md §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub schema_version: u32,
    pub model: ModelState,
}

impl State {
    pub fn new(model: ModelState) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            model,
        }
    }
}

/// One variant per model; grows as models land (tasks 4-14).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelState {
    EwCov(Box<crate::EwCov>),
    EwRidge(Box<crate::EwRidge>),
    Rls(Box<crate::Rls>),
    Lasso(Box<crate::Lasso>),
    Kalman(Box<crate::Kalman>),
    Robust(Box<crate::Robust>),
    Ftrl(Box<crate::Ftrl>),
    EwCovModel(Box<crate::EwCovModel>),
    Sgd(Box<crate::Sgd>),
    Pa(Box<crate::Pa>),
    Holt(Box<crate::Holt>),
    KMeans(Box<crate::KMeans>),
    Micro(Box<crate::Micro>),
    EwClass(Box<crate::EwClass>),
    SeqTest(Box<crate::SeqTest>),
    Marginal(Box<crate::Marginal>),
    Deco(Box<crate::Deco>),
    Rcov(Box<crate::Rcov>),
    Hmm(Box<crate::Hmm>),
    CorrChange(Box<crate::CorrChange>),
    Bocpd(Box<crate::Bocpd>),
}

#[derive(Debug, Error)]
pub enum StateError {
    /// A version [`check_schema`] does not accept: older than
    /// [`MIN_SCHEMA_VERSION`], or newer than [`SCHEMA_VERSION`], a file a
    /// later build wrote. The message names the range and which side the
    /// version is on (review 2026-10-06, CF11).
    #[error("{}", schema_refusal(*.found, *.current))]
    SchemaVersion { found: u32, current: u32 },
    #[error("state is for a different model: expected {expected}, found {found}")]
    WrongModel {
        expected: &'static str,
        found: &'static str,
    },
    #[error("invalid state: {0}")]
    Invalid(String),
}

/// [`StateError::SchemaVersion`]'s message: a state from a newer build is
/// told to upgrade, where it was told the version was not supported in
/// the words an old one is.
fn schema_refusal(found: u32, current: u32) -> String {
    if found > current {
        format!(
            "state schema version {found} was written by a newer version (this build reads \
             {MIN_SCHEMA_VERSION}..={current}): upgrade to read it"
        )
    } else {
        format!(
            "state schema version {found} not supported (this build reads \
             {MIN_SCHEMA_VERSION}..={current}; {MIN_SCHEMA_VERSION} is the oldest it accepts)"
        )
    }
}

/// A restored state's configuration, held to what the model's `new` holds
/// a fresh one to: every `restore` runs its cfg's `validate` (which runs
/// [`crate::Decay::check`]) before it reads anything the cfg sizes or
/// divides by. No `restore` did, so a state whose cfg its own `new` refuses
/// -- `k = 0`, a half-life of 0, a feature index past the features, a lag
/// of 0 -- loaded and failed at a later row, as a panic or as a NaN for
/// good (review 2026-10-06, CF4, CA6, CD14).
pub(crate) fn check_cfg(kind: &str, checked: Result<(), String>) -> Result<(), StateError> {
    checked.map_err(|e| {
        StateError::Invalid(format!("{kind}: the state's configuration is refused: {e}"))
    })
}

impl ModelState {
    pub fn kind(&self) -> &'static str {
        match self {
            // The bare accumulator, which the bank does not run; "ew_cov" is
            // the model, and a `WrongModel` must say which it found (review
            // 2026-09-12, S4).
            ModelState::EwCov(_) => "ew_cov_accumulator",
            ModelState::EwRidge(_) => "ewridge",
            ModelState::Rls(_) => "rls",
            ModelState::Lasso(_) => "lasso",
            ModelState::Kalman(_) => "kalman",
            ModelState::Robust(_) => "robust",
            ModelState::Ftrl(_) => "ftrl",
            ModelState::EwCovModel(_) => "ew_cov",
            ModelState::Sgd(_) => "sgd",
            ModelState::Pa(_) => "pa",
            ModelState::Holt(_) => "holt",
            ModelState::KMeans(_) => "kmeans",
            ModelState::Micro(_) => "micro",
            ModelState::EwClass(_) => "ew_class",
            ModelState::SeqTest(_) => "seqtest",
            ModelState::Marginal(_) => "marginal",
            ModelState::Deco(_) => "deco",
            ModelState::Rcov(_) => "rcov",
            ModelState::Hmm(_) => "hmm",
            ModelState::CorrChange(_) => "corrchange",
            ModelState::Bocpd(_) => "bocpd",
        }
    }
}

/// Age each target's own weight by this row's decay and add the row's weight
/// to the targets it carried: the weight a target's `min_weight` is checked
/// against (hard rule 8, docs/PLAN.md task 115 (d)), where the emitted
/// `n_eff` is every row's. For the update-form models, `pa`, `sgd`, `ftrl`
/// and `rls`, whose coefficients a row with a null target does not move; a
/// zero weight only ages it (hard rule 9).
pub(crate) fn age_target_weights(
    w: &mut [f64],
    carried: impl Fn(usize) -> bool,
    lam: f64,
    weight: f64,
) {
    for (j, wt) in w.iter_mut().enumerate() {
        *wt = lam * *wt + if carried(j) { weight } else { 0.0 };
    }
}

/// A state from before task 115 (d) keeps no per-target weight, and loads
/// with each target at the shared weight -- the one its gate read, so the
/// restored model withholds and predicts where it did. `false` when the
/// state holds one of the wrong length.
pub(crate) fn restore_target_weights(w: &mut Vec<f64>, w_sum: f64, n_targets: usize) -> bool {
    if w.is_empty() {
        *w = vec![w_sum; n_targets];
    }
    w.len() == n_targets
}

/// Each target's own `min_weight`, as a bank holds a list of them: none,
/// for the model's one threshold everywhere, or one value `>= 0` per target
/// (`lasso`'s `target_min_weight`, and the solving models' own first solve,
/// docs/PLAN.md task 195, S9b).
pub(crate) fn check_target_min_weight(
    who: &str,
    own: &[f64],
    n_targets: usize,
) -> Result<(), String> {
    if !(own.is_empty() || own.len() == n_targets) || own.iter().any(|t| t.is_nan() || *t < 0.0) {
        return Err(format!(
            "{who}: target_min_weight must be empty or one value >= 0 per target, got {own:?}"
        ));
    }
    Ok(())
}

/// Target `j`'s own threshold: its entry of `own`, or `min_weight`.
pub(crate) fn min_weight_of(own: &[f64], min_weight: f64, j: usize) -> f64 {
    own.get(j).copied().unwrap_or(min_weight)
}

/// A solving model's coefficients, `ewridge`'s, `huber`'s and `quantile`'s
/// and `lasso`'s, written as the vectors themselves and compared by their
/// bits. A slot nothing solved is NaN by definition -- no fit, which
/// predicts nothing (review round 4, CC1) -- so it is tagged in a
/// human-readable export ([`crate::humanfloat`]; msgpack's bytes are the
/// vectors' own), and `NaN != NaN` would make a model holding one unequal
/// to its own clone, as the readiness shares' NaN would (`ewridge`'s
/// `same_bits`).
#[derive(Debug, Clone)]
pub(crate) struct Fit<T>(pub(crate) T);

impl Serialize for Fit<Vec<Vec<f64>>> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        crate::humanfloat::vec_vec_f64_or_tag::serialize(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for Fit<Vec<Vec<f64>>> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        crate::humanfloat::vec_vec_f64_or_tag::deserialize(d).map(Fit)
    }
}

impl Serialize for Fit<Vec<Vec<Vec<f64>>>> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        crate::humanfloat::vec_vec_vec_f64_or_tag::serialize(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for Fit<Vec<Vec<Vec<f64>>>> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        crate::humanfloat::vec_vec_vec_f64_or_tag::deserialize(d).map(Fit)
    }
}

/// Floats compared by their bits, through any nesting of vectors.
pub(crate) trait SameBits {
    fn same_bits(&self, other: &Self) -> bool;
}

impl SameBits for f64 {
    fn same_bits(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl<T: SameBits> SameBits for Vec<T> {
    fn same_bits(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.same_bits(b))
    }
}

impl<T: SameBits> PartialEq for Fit<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_bits(&other.0)
    }
}

impl<T> std::ops::Deref for Fit<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Fit<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// Check a state's schema version before dispatching to a model's `restore`.
///
/// Layout migrations do not live here: a model whose layout changed accepts
/// every version it can convert in its own `Deserialize` (see `rls`), so by
/// the time a `State` exists the migration has already happened. This gate
/// only rejects versions no model can convert.
pub fn check_schema(state: &State) -> Result<(), StateError> {
    if !(MIN_SCHEMA_VERSION..=SCHEMA_VERSION).contains(&state.schema_version) {
        return Err(StateError::SchemaVersion {
            found: state.schema_version,
            current: SCHEMA_VERSION,
        });
    }
    Ok(())
}

/// The largest magnitude of a feature, target or weight a model has to cope
/// with. Any value beyond it is missing, like a null or a NaN ([`usable`]):
/// to the plumbing (`online-polars`), which never hands a model one, and to
/// every model itself, which refuses one by the rule [`OnlineModel`] states
/// (docs/PLAN.md task 183).
///
/// Every model must keep a finite state, and go on learning, through any row
/// within the bound -- including a weight of `1e100` and a feature of `1e100`
/// on the same row -- and its predictions must return to a clean copy's once
/// such a row has decayed (`tests/model_contract.rs`, docs/IMPROVEMENTS.md
/// C2). The bound is where that is provable with `f64`: squares of `1e100`
/// still fit, products of a weight and a square (`1e300`) still fit.
pub const INPUT_BOUND: f64 = 1e100;

/// Is a feature, target or weight value one a model learns from? Null
/// (extracted as NaN), NaN, infinities and magnitudes beyond [`INPUT_BOUND`]
/// are all "missing": a feature or weight that is not usable makes the row
/// one of weight 0 that keeps nothing of its own, and a target that is not
/// usable is absent, predict-only ([`OnlineModel`]'s rule; docs/PLAN.md §3,
/// docs/IMPROVEMENTS.md C2). The plumbing (`online-polars`) reads values by
/// this one definition too: it skips such a row, and makes such a target
/// predict-only.
#[inline]
pub fn usable(v: f64) -> bool {
    // One comparison: it is false for a NaN and for either infinity, so it
    // is `v.is_finite() && v.abs() <= INPUT_BOUND` without a second test.
    v.abs() <= INPUT_BOUND
}

/// [`usable`] over every value of a row. Not `Iterator::all`: its early
/// exit is worth nothing on rows that are nearly always usable, and a plain
/// fold over the compare vectorises where the exit does not
/// (docs/PERFORMANCE.md §20).
#[inline]
pub fn all_usable(row: &[f64]) -> bool {
    row.iter().fold(true, |ok, &v| ok & usable(v))
}

/// Every present target [`usable`]: a fold, as [`all_usable`] is.
#[inline]
fn targets_usable(y: &[Option<f64>]) -> bool {
    y.iter().fold(true, |ok, v| ok & v.is_none_or(usable))
}

/// The targets as a model reads them: one that is not [`usable`] is absent.
fn usable_targets(y: &[Option<f64>]) -> Vec<Option<f64>> {
    y.iter().map(|v| v.filter(|t| usable(*t))).collect()
}

/// The [`Step`] of a row with a feature that is not usable: `s` with NaN in
/// every slot and in every number of its `extra`, and its `n_eff`.
fn reports_nothing(mut s: Step) -> Step {
    s.pred.fill(f64::NAN);
    if let Some(Extra::Lasso { lam_selected }) = s.extra.as_mut() {
        lam_selected.fill(f64::NAN);
    }
    s
}

/// What every model's [`OnlineModel::step`] does first: the rule the trait
/// states for a value that is not [`usable`]. `None` for a row whose every
/// value is usable, which the model then learns as it stands. Otherwise the
/// row's [`Step`], the row learned by the rule:
///
/// - a target that is not usable is absent;
/// - a row with a feature or a weight that is not usable is the same row at
///   weight 0, with every feature 0. What a row of weight 0 leaves does not
///   depend on its features, which a zero weight reads into no moment, ring,
///   warm-up buffer or likelihood (hard rule 9, held by
///   `tests/model_contract.rs` against the row itself at weight 0), so this
///   is that row's state with nothing of the row's own in it. And no `0 · d`
///   meets a feature that is not a number: a zero-weight update forms one
///   for each feature, and `0 · NaN` is NaN (docs/PLAN.md tasks 181-183);
/// - such a row reports nothing where a feature is not usable, and what
///   [`OnlineModel::predict_with`] reports, which never reads a weight,
///   where only the weight is not.
///
/// The row goes back through the model's own `step`, every value usable,
/// so the zero-weight path it takes is the one every row of weight 0 takes.
/// A model that reads no features hands in an empty `x`.
///
/// The check is inlined into every `step` and the refusal is not: the
/// refusal steps the row again, so a `step` that called the whole of it
/// called into a cycle the compiler would not inline, and a step of two
/// features paid 12% for it (docs/PLAN.md task 183).
#[inline(always)]
pub(crate) fn refused_step<M: OnlineModel>(
    m: &mut M,
    x: &[f64],
    y: &[Option<f64>],
    d_clock: f64,
    weight: f64,
) -> Option<Step> {
    if all_usable(x) & usable(weight) & targets_usable(y) {
        return None;
    }
    Some(refuse(m, x, y, d_clock, weight, M::step))
}

/// [`refused_step`] for a model with a second way to step a row
/// (`Marginal::step_sharded`), which goes back through `step`, as `step`
/// goes back through itself.
#[inline(always)]
pub(crate) fn refused_by<M: OnlineModel>(
    m: &mut M,
    x: &[f64],
    y: &[Option<f64>],
    d_clock: f64,
    weight: f64,
    step: impl FnOnce(&mut M, &[f64], &[Option<f64>], f64, f64) -> Step,
) -> Option<Step> {
    if all_usable(x) & usable(weight) & targets_usable(y) {
        return None;
    }
    Some(refuse(m, x, y, d_clock, weight, step))
}

/// The refusal [`refused_step`] makes, for a row it found a value in that
/// is not usable.
#[cold]
#[inline(never)]
fn refuse<M: OnlineModel>(
    m: &mut M,
    x: &[f64],
    y: &[Option<f64>],
    d_clock: f64,
    weight: f64,
    step: impl FnOnce(&mut M, &[f64], &[Option<f64>], f64, f64) -> Step,
) -> Step {
    let (readable, weighed) = (all_usable(x), usable(weight));
    let y = usable_targets(y);
    if readable && weighed {
        return step(m, x, &y, d_clock, weight);
    }
    let out = readable.then(|| m.predict_with(x, &y, d_clock));
    let learned = step(m, &vec![0.0; x.len()], &y, d_clock, 0.0);
    out.unwrap_or_else(|| reports_nothing(learned))
}

/// What every model's [`OnlineModel::predict`] that reads its features does
/// first: `None` for a row whose features are all [`usable`], and for one
/// that is not, its [`Step`] -- nothing reported, `n_eff` as on any row,
/// read off the model with every feature 0, as [`refused_step`] learns it.
#[inline(always)]
pub(crate) fn refused_predict<M: OnlineModel>(m: &M, x: &[f64], d_clock: f64) -> Option<Step> {
    if all_usable(x) {
        return None;
    }
    Some(refuse_reading(m, x, d_clock))
}

/// The answer [`refused_predict`] gives for a row it refuses.
#[cold]
#[inline(never)]
fn refuse_reading<M: OnlineModel>(m: &M, x: &[f64], d_clock: f64) -> Step {
    reports_nothing(m.predict(&vec![0.0; x.len()], d_clock))
}

/// [`refused_predict`] for a model that reads a number out of the targets
/// slot ([`OnlineModel::predict_with`]), which is absent where it is not
/// [`usable`], as a target is.
#[inline(always)]
pub(crate) fn refused_predict_with<M: OnlineModel>(
    m: &M,
    x: &[f64],
    y: &[Option<f64>],
    d_clock: f64,
) -> Option<Step> {
    if all_usable(x) & targets_usable(y) {
        return None;
    }
    Some(refuse_reading_with(m, x, y, d_clock))
}

/// The answer [`refused_predict_with`] gives for a row it refuses.
#[cold]
#[inline(never)]
fn refuse_reading_with<M: OnlineModel>(m: &M, x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
    let y = usable_targets(y);
    if all_usable(x) {
        m.predict_with(x, &y, d_clock)
    } else {
        reports_nothing(m.predict_with(&vec![0.0; x.len()], &y, d_clock))
    }
}

/// One row in, one [`Step`] out (docs/PLAN.md §2).
///
/// Invariants:
/// - `pred` uses state *before* the update with this row (out-of-sample by
///   construction);
/// - deterministic given input order;
/// - no allocation in the hot path after warmup (buffers preallocated);
/// - the state stays finite, and the model keeps learning, after any row.
///
/// `x` excludes the intercept (the model adds it if configured); `y[j] = None`
/// means predict-only for target j; `d_clock` is already capped/gap-adjusted
/// (see [`crate::ClockState`]); `weight >= 0` scales the row, and `0` means
/// "advance the clock, learn nothing".
///
/// **A value that is not [`usable`]** -- not a number, infinite, or beyond
/// [`INPUT_BOUND`] -- is refused, by every model the same way
/// (docs/PLAN.md task 183):
///
/// 1. A target that is not usable is absent: predict-only for that target.
///    So is a number read out of the targets slot (`bocpd`'s hazard,
///    `hmm`'s exogenous value), in [`Self::predict_with`] too.
/// 2. A row with a feature or a weight that is not usable is learned as a
///    row of weight 0 that keeps nothing of its own. Its `d_clock` ages the
///    state as a zero weight does, and nothing of its values enters the
///    state: no moment, ring slot, warm-up row, count, cluster or
///    likelihood. `n_eff` and every decayed weight are `λ^d` times what they
///    were: the row is not counted, as the plumbing, which skips it, never
///    counts it.
/// 3. A row with a feature that is not usable reports nothing, NaN in every
///    slot and in `extra`, from `step` and `predict` alike, and `n_eff` as
///    any row does. A row whose weight alone is not usable reports what
///    `predict` does, which never reads a weight.
///
/// `tests/model_contract.rs` holds every model to the three, against the
/// same row at weight 0 with the feature finite, whose state the refused
/// row's must equal byte for byte.
///
/// ```
/// use online_core::{Holt, HoltCfg, OnlineModel};
///
/// // Holt's linear trend has no features, so `x` is empty.
/// let mut model = Holt::new(HoltCfg {
///     n_targets: 1,
///     level_half_life: 2.0,
///     trend_half_life: 4.0,
///     min_weight: 2.0,
///     trend: true,
/// })?;
/// for t in 0..60 {
///     model.step(&[], &[Some(t as f64)], 1.0, 1.0);
/// }
///
/// // A missing target is predict-only: the forecast extrapolates one clock
/// // unit ahead and nothing is learned from the row.
/// let step = model.step(&[], &[None], 1.0, 1.0);
/// assert!((step.pred[0] - 60.0).abs() < 1e-3);
///
/// // A zero weight advances the clock and learns nothing, however wild the
/// // target: the coefficients (here `[level, trend]`) do not move.
/// let before = model.coefficients();
/// model.step(&[], &[Some(1e9)], 1.0, 0.0);
/// assert_eq!(model.coefficients(), before);
///
/// // `predict` is the step's answer without the step: the forecast three
/// // clock units past the last row. Neither row above was learned from, so
/// // `holt` extrapolates from its last observation (level 59, trend 1) over
/// // their two clock units as well, `59 + 5·1`. Nothing -- not even the
/// // clock -- has moved.
/// let ahead = model.predict(&[], 3.0);
/// assert!((ahead.pred[0] - 64.0).abs() < 1e-3);
/// assert_eq!(model.predict(&[], 3.0), ahead);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait OnlineModel: Sized {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step;
    /// The [`Step`] that [`Self::step`] would return for this row, without
    /// the update: the same `pred`, the same `n_eff`, the same `extra`, and
    /// the state untouched. `d_clock` is the clock elapsed since the last
    /// learned row -- a trend model (`holt`) extrapolates over it, on top of
    /// the clock since its target was last observed, and a proximal model
    /// (`ftrl`) sees its accumulators decayed by it; the
    /// coefficient models ignore it, since decay alone never moves a mean.
    ///
    /// `tests/model_contract.rs` holds every model to the equality
    /// `predict(x, d) == step(x, y, d, w)` on `pred`, `n_eff` and `extra`,
    /// row by row (docs/ENHANCEMENTS.md E31).
    fn predict(&self, x: &[f64], d_clock: f64) -> Step;

    /// [`Self::predict`] for a model that reads a number out of the
    /// **targets** slot rather than regressing it: `bocpd`'s hazard column
    /// and `hmm`'s exogenous column ride there, the way a label does, and
    /// `predict` alone cannot see them -- so it answered from the
    /// configured default and disagreed with the step on every row where
    /// the column differed from it (docs/REVIEW-E54-E64.md C1).
    ///
    /// The default ignores `y`, which is right for every model that only
    /// regresses its targets: `predict` is out-of-sample precisely because
    /// it does not read the row's answer.
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        let _ = y;
        self.predict(x, d_clock)
    }

    /// Drop whatever this model keeps that is indexed by *rows back* -- a
    /// ring of past feature vectors, a partially filled window -- because
    /// the rows behind it are no longer adjacent to the next one
    /// (docs/PLAN.md task 47).
    ///
    /// The stream calls it on a **session change** and on a **capped clock
    /// gap** ([`crate::ClockAdvance::capped`]), and on nothing else: a reset
    /// already rebuilds the model, and an ordinary gap is what decay is for.
    /// A model with no row-lagged state does nothing, which is every model
    /// but the ones that keep one -- hence the default.
    ///
    /// It is **not** a reset: means, co-moments, coefficients and `n_eff`
    /// are untouched. Only the lags go.
    fn clear_lags(&mut self) {}

    /// The next learned row's place on the decayed clock, held exactly
    /// ([`crate::Stamp`], docs/PLAN.md task 175). A caller that keeps one
    /// hands it before each [`Self::step`] -- the stream does, from
    /// [`crate::ClockState::advance_stamped`] -- and a model with a window
    /// keys that row's snapshot by it, and decides its edge and its snapshot
    /// spacing from it. A step handed none takes the model's own clock,
    /// summed from the `d_clock`s it is stepped with, as every window did
    /// before. Decay reads `d_clock` either way. A model without a window has
    /// no edge to decide -- hence the default, which ignores it.
    fn stamp_next(&mut self, _stamp: crate::Stamp) {}

    /// Bound this model's window, if it has one ([`crate::WindowBudget`]).
    /// Configuration, not state: a caller sets it after building or
    /// restoring the model, and a model without a window ignores it --
    /// hence the default.
    fn set_window_budget(&mut self, _budget: Option<crate::WindowBudget>) {}

    /// Which edge of this model's window holds a row exactly one window old
    /// ([`crate::WindowClosed`]; docs/PLAN.md task 196): `Right`, the
    /// default a window is built with, or `Both`. Configuration that travels
    /// with the state, since what the ring has trimmed follows it: a caller
    /// sets it once, after building the model and before its first row, and a
    /// restored model keeps the one it was saved with. A model without a
    /// window ignores it -- hence the default.
    fn set_window_closed(&mut self, _closed: crate::WindowClosed) {}

    /// Set the weight-share cadence of a model that solves on a schedule
    /// (`ewridge`, `lasso`, `huber`, `quantile`; docs/PLAN.md task 115 (b)).
    /// Configuration from the spec: a caller sets it after building or
    /// restoring, so a state saved before the rule existed takes it on load;
    /// a model with no schedule ignores it -- hence the default.
    fn set_solve_share(&mut self, _share: Option<f64>) {}

    /// The weight-share cadence this model runs by, if any.
    fn solve_share(&self) -> Option<f64> {
        None
    }

    /// What a refusing budget saw the window reach: its snapshots' bytes,
    /// and its cadence (`window_every` and `max_rows_between_snapshots`,
    /// doubled by any thinning). `None` while under budget, and for a model
    /// without a window.
    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        None
    }

    /// The window's ring as a [`crate::WindowShadow`], for a caller that
    /// must know before it runs rows whether they would take the ring past
    /// a refusing budget (docs/PLAN.md task 115 (d)). `None` for a model
    /// without a window.
    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        None
    }

    /// Each target's own accumulated weight, with `n_eff`'s meaning --
    /// before this row's update and before its own decay, and inside the
    /// window under one -- for the per-target `min_weight` gate: the stream
    /// checks each target's threshold against its own weight, where the
    /// shared `n_eff` is the feature side's, the same for every target
    /// (review 2026-09-12, S2). Clears and fills `out`, an entry a target,
    /// and says whether it did; a model that keeps no weight per target
    /// leaves every target on the shared `n_eff` -- hence the default.
    fn target_n_eff_into(&self, _out: &mut Vec<f64>) -> bool {
        false
    }

    /// Per output slot, how much estimation error is expected to inflate a
    /// prediction's error over the noise floor, read from the state before
    /// the row: `sqrt(1 + edf / n_kish)`, the effective degrees of freedom
    /// the last solve used over Kish's effective sample size behind the
    /// slot's Gram (docs/WARMUP-AND-CONVERGENCE.md §2.1). Infinite before
    /// the first solve, and where the Gram has no weight. Clears and fills
    /// `out`, an entry a slot, and says whether it did. Only `ewridge`
    /// reports it. Every other model keeps the default, `false`, the linear
    /// fits `rls`, `lasso`, `huber`, `quantile` and `kalman` among them, and
    /// the spec refuses `max_error_inflation` and `emit_error_inflation` for
    /// every model but `ewridge`.
    fn error_inflation_into(&self, _out: &mut Vec<f64>) -> bool {
        false
    }

    /// [`Self::error_inflation_into`] for the noise gate at `limit`, which
    /// reads only which side of `limit` each slot is on: exact wherever the
    /// ratio could reach `limit`, while below it a bound may stand in, so a
    /// model can skip work the gate would not look at (docs/PLAN.md task
    /// 140). The largest value is then exact whenever any slot is at or
    /// above `limit`. The default is the exact statistic.
    fn error_inflation_gate_into(&self, out: &mut Vec<f64>, limit: f64) -> bool {
        let _ = limit;
        self.error_inflation_into(out)
    }

    /// The same for *this* row's features: `sqrt(1 + h(x))` per slot with
    /// `h(x) = x' Σ̂⁻¹ x / n_kish`, the leverage of the row against the
    /// factor the fit came from, so a row whose `x` leans on a direction the
    /// data never showed reads large where the stream average cannot see it.
    /// One triangular solve per slot, `O(k²)`, which is why it rides on an
    /// opt-in field (`emit_error_inflation`) and the model keeps the factor
    /// only when asked to. `false` where [`Self::error_inflation_into`] is.
    fn row_error_inflation_into(&self, _x: &[f64], _out: &mut Vec<f64>) -> bool {
        false
    }

    /// Per output slot, each coefficient's data share
    /// (docs/WARMUP-AND-CONVERGENCE.md §2.2): `1 − λ (Σ̂⁻¹)_jj`, the fraction
    /// of the coefficient the data determined rather than the ridge, in
    /// `[0, 1]`, laid out like the coefficients (`k_total` per slot). NaN in
    /// the intercept slot, which is not a share, and for a coefficient
    /// outside the slot's feature set; 0 for a column the standardiser
    /// dropped. `None` before the first solve, and for a model with no
    /// ridge system to read it from.
    fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        None
    }

    fn state(&self) -> State;
    fn restore(s: &State) -> Result<Self, StateError>;
    fn n_targets(&self) -> usize;
    fn n_features(&self) -> usize;
    /// Number of prediction slots (`n_targets * grid combos`; usually `n_targets`).
    fn n_outputs(&self) -> usize {
        self.n_targets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each state names its own kind, so a `WrongModel` says which one it
    /// found: the bare accumulator answered "ew_cov", the name of the bank
    /// model it is not (review 2026-09-12, S4).
    #[test]
    fn the_bare_accumulator_is_not_named_as_the_ew_cov_model() {
        let bare = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))));
        assert_ne!(bare.model.kind(), "ew_cov");
        match crate::EwCovModel::restore(&bare) {
            Err(StateError::WrongModel { expected, found }) => {
                assert_eq!(expected, "ew_cov");
                assert_ne!(found, expected, "the error must say what it found");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A model that keeps none of the optional statistics answers as the
    /// trait's defaults say: no schedule, no window, no weight per target,
    /// no noise inflation, no data shares -- and says so, leaving each
    /// buffer it was handed as it found it. `rls` overrides none of them
    /// (task 158: the defaults were changed to answers no test noticed).
    #[test]
    fn a_model_without_the_optional_statistics_reports_none_of_them() {
        let mut rls = crate::Rls::new(crate::RlsCfg {
            n_features: 1,
            n_targets: 2,
            fit_intercept: true,
            decay: crate::Decay::Halflife(10.0),
            delta: 1.0,
            coef_prior: None,
            min_weight: 0.0,
        })
        .unwrap();
        rls.step(&[1.0], &[Some(2.0), Some(-1.0)], 0.0, 1.0);
        rls.step(&[3.0], &[Some(5.0), None], 1.0, 1.0);
        assert_eq!(rls.solve_share(), None);
        assert_eq!(rls.window_over_budget(), None);
        assert!(rls.window_shadow().is_none());
        assert_eq!(rls.support_coef(), None);
        let mut out = vec![7.0];
        assert!(!rls.error_inflation_into(&mut out));
        assert!(!rls.error_inflation_gate_into(&mut out, 2.0));
        assert!(!rls.row_error_inflation_into(&[1.0], &mut out));
        assert_eq!(out, [7.0], "a model with none fills nothing");
        // `rls` keeps a weight per target, the rows it learned from: here
        // the first row only, since a row with a null target teaches nothing,
        // decayed by the one clock unit since, at a half-life of 10.
        assert!(rls.target_n_eff_into(&mut out));
        let first = 0.5f64.powf(0.1);
        assert!(
            out.len() == 2 && out.iter().all(|w| (w - first).abs() < 1e-15),
            "{out:?}"
        );
        // A model that keeps only the shared weight fills nothing.
        let mut seq = crate::SeqTest::new(crate::SeqTestCfg {
            n_targets: 2,
            min_weight: 0.0,
        })
        .unwrap();
        seq.step(&[], &[Some(1.0), Some(-1.0)], 0.0, 1.0);
        let mut out = vec![7.0];
        assert!(!seq.target_n_eff_into(&mut out));
        assert_eq!(out, [7.0]);
    }

    /// The noise gate's default is the exact statistic: a model that reports
    /// [`OnlineModel::error_inflation_into`] and keeps no bound of its own
    /// hands the gate that statistic, whatever the limit.
    #[test]
    fn the_noise_gates_default_is_the_exact_statistic() {
        struct Inflated;
        impl OnlineModel for Inflated {
            fn step(&mut self, x: &[f64], _: &[Option<f64>], d: f64, _: f64) -> Step {
                self.predict(x, d)
            }
            fn predict(&self, _: &[f64], _: f64) -> Step {
                Step {
                    pred: vec![f64::NAN; 2],
                    n_eff: 0.0,
                    extra: None,
                }
            }
            fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
                out.clear();
                out.extend([1.25, f64::INFINITY]);
                true
            }
            fn state(&self) -> State {
                State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))))
            }
            fn restore(_: &State) -> Result<Self, StateError> {
                Err(StateError::Invalid("a test model keeps no state".into()))
            }
            fn n_targets(&self) -> usize {
                2
            }
            fn n_features(&self) -> usize {
                0
            }
        }
        for limit in [0.5, 1.25, 2.0, f64::INFINITY] {
            let mut out = Vec::new();
            assert!(Inflated.error_inflation_gate_into(&mut out, limit));
            assert_eq!(out, [1.25, f64::INFINITY], "at the limit {limit}");
        }
    }

    #[test]
    fn schema_check() {
        let s = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))));
        assert!(check_schema(&s).is_ok());
        let old = State {
            schema_version: MIN_SCHEMA_VERSION,
            ..s.clone()
        };
        assert!(check_schema(&old).is_ok());
        for v in [MIN_SCHEMA_VERSION - 1, SCHEMA_VERSION + 1] {
            let bad = State {
                schema_version: v,
                ..s.clone()
            };
            assert!(matches!(
                check_schema(&bad),
                Err(StateError::SchemaVersion { .. })
            ));
        }
    }

    /// The refusal names the oldest version the gate accepts, and a state
    /// from a newer build is told so: it was told "not supported (current:
    /// N)", which names neither the floor nor the cause (review 2026-10-06,
    /// CF11).
    #[test]
    fn a_schema_refusal_names_the_range_and_a_newer_build() {
        let s = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))));
        let refused = |v: u32| {
            check_schema(&State {
                schema_version: v,
                ..s.clone()
            })
            .unwrap_err()
            .to_string()
        };
        let range = format!("{MIN_SCHEMA_VERSION}..={SCHEMA_VERSION}");
        let older = refused(MIN_SCHEMA_VERSION - 1);
        assert!(
            older.contains(&range)
                && older.contains(&format!("{MIN_SCHEMA_VERSION} is the oldest it accepts"))
                && !older.contains("newer"),
            "{older}"
        );
        let newer = refused(SCHEMA_VERSION + 1);
        assert!(
            newer.contains("written by a newer version")
                && newer.contains(&range)
                && newer.contains("upgrade"),
            "{newer}"
        );
    }

    /// A fit with a slot nothing solved, NaN (review round 4, CC1), writes
    /// the vectors' own msgpack bytes, tags the NaN in JSON and reads both
    /// back to the same bits; and two such fits are equal, where `NaN !=
    /// NaN` made a model holding one unequal to its own clone.
    #[test]
    fn a_fit_of_nan_round_trips_and_equals_itself() {
        let raw = vec![vec![1.5, -0.25], vec![f64::NAN, f64::NAN]];
        let fit = Fit(raw.clone());
        assert_eq!(fit, fit.clone());
        assert_ne!(fit, Fit(vec![vec![1.5, -0.25], vec![0.0, 0.0]]));
        let bytes = rmp_serde::to_vec(&fit).unwrap();
        assert_eq!(
            bytes,
            rmp_serde::to_vec(&raw).unwrap(),
            "msgpack is the vectors'"
        );
        assert_eq!(
            rmp_serde::from_slice::<Fit<Vec<Vec<f64>>>>(&bytes).unwrap(),
            fit
        );
        let json = serde_json::to_string(&fit).unwrap();
        assert_eq!(json, r#"[[1.5,-0.25],["nan","nan"]]"#);
        assert_eq!(
            serde_json::from_str::<Fit<Vec<Vec<f64>>>>(&json).unwrap(),
            fit
        );
        let path = Fit(vec![raw.clone(), raw]);
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(
            serde_json::from_str::<Fit<Vec<Vec<Vec<f64>>>>>(&json).unwrap(),
            path
        );
    }
}
