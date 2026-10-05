//! `marginal`: exponentially weighted moments of every feature against every
//! target, one pair at a time (docs/ENHANCEMENTS.md E44).
//!
//! An `ew_cov` over `[x_1..x_p, y_1..y_T]` keeps the whole `(p+T)²` Gram.
//! This model keeps the diagonal and the cross column -- per (target `t`,
//! feature `j`) the two means, the two centred variances and the covariance
//! -- which is `O(p·T)` state and `O(p·T)` work per row where the Gram is
//! `O(p²)`. It emits nothing per row but `n_eff`; its value is its state,
//! read back per pair with [`Marginal::pair`].
//!
//! Per target `t`, on a row where `y_t` is present, with weight `w` and decay
//! factor `lam`:
//!
//! ```text
//! W'_t       = lam·W_t + w            a = lam·W_t / W'_t        b = w / W'_t
//! Q'_t       = lam²·Q_t + w²                                    (Σw², for Kish)
//! dy         = y_t − m_y[t]           dx_j = x_j − m_x[t,j]
//! S'_yy[t]   = a·S_yy[t]   + a·b·dy·dy
//! S'_xx[t,j] = a·S_xx[t,j] + a·b·dx_j·dx_j
//! S'_xy[t,j] = a·S_xy[t,j] + a·b·dx_j·dy
//! m_y[t]    += b·dy                   m_x[t,j] += b·dx_j
//! ```
//!
//! This is the weighted Welford form [`crate::EwCov`] uses, operation for
//! operation, so the pair `(x_j, y_t)` here and the pair in an `ew_cov` over
//! `[x_j, y_t]` fed the same rows give the same correlation to the bit
//! (`tests/test_marginal.py` holds them to it). A missing `y_t` ages the
//! target's accumulators (`W_t·lam`, `Q_t·lam²`) and moves nothing else, as
//! a missing target does in `ew_ridge`. The feature moments are kept per
//! *pair*, not per feature: they are over the rows where the target was
//! present, so both sides of a correlation are over the same rows whatever
//! the target's missingness.
//!
//! Read back per pair ([`Pair`]): `n_eff = W_t`; `n_kish = W_t² / Q_t`,
//! Kish's effective sample size -- `(1+lam^d)/(1−lam^d)` in the limit for
//! unit weights `d` clock units apart, about twice `n_eff`, and the `n` a standard error wants; the
//! moments; and from them `corr = S_xy / √(S_xx·S_yy)`, `beta = S_xy /
//! S_xx` (the slope of `y` on `x`) and `t = corr·√((n_kish − 2) / (1 −
//! corr²))`, the t-statistic of the correlation at Kish's `n`. `corr`, `beta`
//! and `t` are NaN while `W_t < min_weight`; the moments are always
//! reported. The t is descriptive: the rows of a stream are rarely
//! independent, and nothing here pretends otherwise.
//!
//! # Shared feature moments
//!
//! Under [`FeatureMomentLayout::Shared`] (docs/PLAN.md task 125, E72) the
//! feature's mean and variance are kept once per feature, over **every**
//! learned row, with the row's own mix over the model's weight `W`, and each
//! pair keeps only its covariance, stepped with the target's mix:
//!
//! ```text
//! a = lam·W / W'   b = w / W'         (W' = lam·W + w, every learned row)
//! dx_j       = x_j − m_x[j]           (the shared mean before the row)
//! S'_xy[t,j] = a_t·S_xy[t,j] + a_t·b_t·dx_j·dy      (on target t's rows)
//! S'_xx[j]   = a·S_xx[j] + a·b·dx_j·dx_j            m_x[j] += b·dx_j
//! ```
//!
//! That is `p` means and variances where the default keeps `p·T`, and each
//! pair's step one multiply-add. Where every target is on every learned row,
//! `W_t = W` and each target's feature means are the shared ones, so the
//! pairs are the default's to the bit. Where a target is absent on some
//! rows it is a different estimator: `mean_x` and `var_x` are the feature's
//! over every learned row, and `cov` is the target's rows centred on that
//! mean, so `corr`, `beta` and `t` move with them. That is sound where the
//! absence says nothing about the feature. Under `lags` the feature's
//! autocovariance at each lag is kept per feature too, stepped with the
//! row's mix, and the cross terms per pair with the target's
//! ([`crate::MarginalLags::update_shared`]). A window is refused: its
//! subtraction reads each covariance as centred on the pair's own mean.

use serde::{Deserialize, Serialize};

use crate::{Decay, OnlineModel, State, StateError, Step};

/// A windowed weight `wn` (current minus the aged boundary), zeroed when it
/// is a rounding crumb below `EMPTY_FRACTION` of the untruncated weight `w` --
/// the same emptiness test `Marginal::cut` applies to each pair, so the
/// model's `n_eff` and its pairs agree after a gap (review 2026-09-18, S4).
fn empty_or(wn: f64, w: f64) -> f64 {
    if wn <= crate::window::EMPTY_FRACTION * w || !wn.is_finite() {
        0.0
    } else {
        wn
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub decay: Decay,
    /// Weight each target must have accumulated before its pairs' `corr`,
    /// `beta` and `t` are reported, one entry per target; the moments never
    /// wait.
    pub min_weight: Vec<f64>,
    /// Lags to accumulate pair moments at (docs/ENHANCEMENTS.md E66),
    /// strictly increasing and `>= 1`, counted in **learned rows within the
    /// group** — not in rows where a particular target was present, since the
    /// ring is shared. Empty for none, which is what every state written
    /// before E66 has.
    #[serde(default)]
    pub lags: Vec<usize>,
    /// How `n_serial` is formed from the lags that were kept: `"truncated"`
    /// sums them as they are (a lower bound on the correction, so an upper
    /// bound on `n_serial`), `"bartlett"` weights lag `l` by `1 − l/(L + 1)`,
    /// `"geometric"` fits one decay per series and extrapolates the tail in
    /// closed form. `None` keeps the lag moments and derives nothing.
    #[serde(default)]
    pub serial_rule: Option<SerialRule>,
    /// Binned target moments per pair (docs/ENHANCEMENTS.md E67): the
    /// feature's response curve and its best single split, which is what a
    /// nonlinear relation shows up in when `corr` cannot see it. `default`
    /// but **not** skipped, for the reason `Marginal::lag` gives.
    #[serde(default)]
    pub bins: Option<Box<crate::BinCfg>>,
    /// The lags to keep the cross moments at, `lagcorr_xy` and `lagcorr_yx`
    /// (E70, docs/PLAN.md task 123): strictly increasing, each one of
    /// `lags`. `None` keeps them at every lag, as before the option existed;
    /// empty keeps none. `n_serial` reads the autocorrelations alone, which
    /// every lag keeps whatever this says. `default` but not skipped, for
    /// the reason `window` gives.
    #[serde(default)]
    pub cross_lags: Option<Vec<usize>>,
    /// Where the feature moments are kept ([`FeatureMomentLayout`]; docs/PLAN.md
    /// task 125). `default`, so a state written before it keeps them per
    /// target, and not skipped, for the reason `window` gives.
    #[serde(default)]
    pub feature_moments: FeatureMomentLayout,
    /// Clock units of history the pairs are computed from, with a **hard**
    /// cutoff: a row older than this contributes nothing (docs/PLAN.md §13).
    /// Inside the window the weights are still exponential.
    ///
    /// **Last, with `window_every`, and they must stay last**: the compact
    /// msgpack encoding writes a struct as an array, so a
    /// `skip_serializing_if` field anywhere else shifts what follows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<f64>,
    /// Rows between the window's snapshots, counted on every row the model
    /// is stepped with, rows of weight zero included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_every: Option<usize>,
}

/// How the serial-dependence correction behind `n_serial` is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerialRule {
    /// Sum the kept lags as they are. Truncating the sum can only make the
    /// correction smaller, so this is an upper bound on `n_serial` — safe in
    /// the direction that does not overstate significance only when the
    /// omitted lags are negative, which is why it is not the default. A sum
    /// that takes the factor `1 + 2·Σ ρ_x(l)·ρ_y(l)` to zero or below -- two
    /// series whose autocorrelations have opposite signs -- is outside the
    /// parameter space, and `n_serial` and `t_serial` are NaN.
    Truncated,
    /// Fit `rho(l) = phi^l` per series by least squares on `log rho` over the
    /// kept lags where `rho > 0`, then sum the tail in closed form:
    /// `1 + 2·phi_x·phi_y/(1 − phi_x·phi_y)`. The right choice when both
    /// series are exponentially weighted, and the reason the lags need not be
    /// dense. Needs at least two kept lags with `rho > 0` on each side.
    Geometric,
    /// Newey and West's Bartlett weights on the kept lags: `1 + 2·Σ_l (1 −
    /// l/(L + 1))·ρ_x(l)·ρ_y(l)`, `L` the longest kept lag (docs/PLAN.md
    /// task 135). The weights shrink the long lags, where the estimates are
    /// noisiest, and they are what keeps the factor positive for a
    /// positive-definite sequence of products over every lag `1..=L` -- the
    /// factor is then `1ᵀC1/(L + 1)` for the Toeplitz `C` of the products.
    /// With lags missing, or on these EW estimates, that is not guaranteed,
    /// and a factor at or below zero is NaN, as under `Truncated`.
    Bartlett,
}

/// Where a pair's feature moments are kept (docs/PLAN.md task 125,
/// docs/ENHANCEMENTS.md E72).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureMomentLayout {
    /// Each pair keeps the feature's mean and variance over the rows its
    /// target was present: `p·T` of each, beside the `p·T` covariances.
    #[default]
    PerTarget,
    /// One mean and one variance per feature, over every learned row, and
    /// each pair its covariance: `p` of each beside the `p·T` covariances.
    /// Where every target is on every learned row the two are the same
    /// numbers, to the bit. Where a target is absent on some, `mean_x` and
    /// `var_x` are the feature's over every learned row, and `cov` is the
    /// target's rows centred on that mean: a different estimator, sound
    /// where the absence says nothing about the feature (the module docs).
    Shared,
}

impl MarginalCfg {
    /// Whether the feature moments are one per feature.
    pub fn shared(&self) -> bool {
        self.feature_moments == FeatureMomentLayout::Shared
    }

    /// The length of each feature-moment vector: `p`, or `p·T` per target.
    fn feature_len(&self) -> usize {
        if self.shared() {
            self.n_features
        } else {
            self.n_features * self.n_targets
        }
    }

    /// Where pair `(t, j)`'s feature moments sit.
    fn feature_at(&self, t: usize, j: usize) -> usize {
        if self.shared() {
            j
        } else {
            t * self.n_features + j
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("marginal: at least one feature is required".into());
        }
        if self.n_targets == 0 {
            return Err("marginal: at least one target is required".into());
        }
        if self.min_weight.len() != self.n_targets {
            return Err(format!(
                "marginal: min_weight has {} entries for {} targets",
                self.min_weight.len(),
                self.n_targets
            ));
        }
        // `inf` is a gate that never opens, as for every other model and as
        // the builders document; this model alone refused it (review
        // 2026-09-12, S27).
        if let Some(bad) = self.min_weight.iter().find(|v| v.is_nan() || **v < 0.0) {
            return Err(format!("marginal: min_weight must be >= 0, got {bad}"));
        }
        // `MarginalLags::new` checks the lags themselves; this is the pair of
        // rules that involve `serial_rule`, which it cannot see.
        if self.lags.is_empty() && self.serial_rule.is_some() {
            return Err(
                "marginal: serial_rule needs `lags`; the correction is built from the \
                 autocorrelations at those lags"
                    .into(),
            );
        }
        if self.lags.is_empty() && self.cross_lags.is_some() {
            return Err(
                "marginal: cross_lags needs `lags`; it names which of them keep the cross terms"
                    .into(),
            );
        }
        if self.window.is_none() && self.window_every.is_some() {
            return Err("marginal: window_every needs `window_size`".into());
        }
        if self.shared() && self.window.is_some() {
            return Err(
                "marginal: feature_moments = \"shared\" takes no window. The window subtracts \
                 each pair's moments at its boundary by the pooling identity, which reads a \
                 covariance as centred on the pair's own mean; the shared mean also moves on \
                 rows the pair's target missed, so the subtraction would not be the window's \
                 covariance. Use feature_moments = \"per_target\" with a window."
                    .into(),
            );
        }
        // A window with lags snapshots the lag moments too (docs/PLAN.md
        // task 137), at a price the spec layer asks the caller to accept by
        // name (`window_lags`); the core takes the pair as it is.
        if let Some(b) = self.bins.as_ref() {
            b.validate(self.n_features, self.n_targets)?;
            if self.window.is_some() {
                return Err(
                    "marginal: bins and window cannot be combined. A window subtracts an old \
                     snapshot of the accumulators, and a snapshot of the histogram is bins times \
                     the size of one -- too large to keep per snapshot. Use one or the other."
                        .into(),
                );
            }
        }
        Ok(())
    }
}

/// One (feature, target) pair as the state stands (see the module doc).
/// Not `Copy`: with `lags` a pair carries one list per lag family.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    /// Accumulated weight behind the pair: `W_t`, the rows where the target
    /// was present.
    pub n_eff: f64,
    /// Kish's effective sample size `W_t² / Q_t`; NaN before the first row.
    pub n_kish: f64,
    pub mean_x: f64,
    /// Centred variance of the feature over the rows the target was present.
    pub var_x: f64,
    pub mean_y: f64,
    pub var_y: f64,
    /// Centred covariance.
    pub cov: f64,
    /// Per configured lag: `rho_x(l)`, the feature's own autocorrelation,
    /// and `rho_y(l)`, the target's. Per cross lag -- every lag unless
    /// `cross_lags` names fewer -- the two cross-correlations: `lagcorr_xy`
    /// is the feature *now* against the target `l` rows ago, `lagcorr_yx`
    /// the target now against the feature `l` rows ago. Empty without
    /// `lags`, and the cross pair empty under `cross_lags = []`.
    ///
    /// Each is the lagged covariance over the two contemporaneous standard
    /// deviations, as `ew_cov`'s `lagcorr` is, and like it **not clamped**:
    /// a lagged correlation is not bounded by one in finite samples, and
    /// clamping would hide that. The serial correction guards itself.
    pub lagcorr_xx: Vec<f64>,
    pub lagcorr_yy: Vec<f64>,
    pub lagcorr_xy: Vec<f64>,
    pub lagcorr_yx: Vec<f64>,
    /// `n_kish` divided by Bartlett's serial-dependence factor
    /// `1 + 2·Σ_l rho_x(l)·rho_y(l)`, per `serial_rule`; NaN without one.
    pub n_serial: f64,
    /// `corr·sqrt((n_serial − 2)/(1 − corr²))`: the same statistic as `t`
    /// against a count that has been told about serial dependence, and
    /// `±inf` where `t` is.
    pub t_serial: f64,
    /// The fitted per-row decays under `SerialRule::Geometric`; NaN
    /// otherwise, and NaN when fewer than two kept lags had `rho > 0`.
    pub phi_x: f64,
    pub phi_y: f64,
    /// `cov / √(var_x·var_y)`, clamped to `[-1, 1]`; NaN when either side is
    /// constant, or below `min_weight`.
    pub corr: f64,
    /// The slope of `y` on `x`, `cov / var_x`; NaN when the feature is
    /// constant, or below `min_weight`.
    pub beta: f64,
    /// `corr·√((n_kish − 2) / (1 − corr²))`; NaN when `n_kish <= 2`, or
    /// below `min_weight`. Enormous or `±inf` for a perfect correlation,
    /// which is the honest value.
    pub t: f64,
    /// The feature's bin edges, and the target's weight, mean and variance
    /// inside each bin: the response curve. One more bin than edges, the
    /// outer two open. Empty without `bins`, and until the edges are fixed.
    pub bin_edges: Vec<f64>,
    pub bin_n: Vec<f64>,
    pub bin_mean_y: Vec<f64>,
    pub bin_var_y: Vec<f64>,
    /// Fraction of the target's variance removed by the best single cut of
    /// this feature -- a regression stump's gain, in `[0, 1]`. This is the
    /// number that sees a threshold, a V or a saturation, all of which can
    /// sit at `corr = 0`. NaN without `bins`, below `min_weight`, or when
    /// the target does not vary.
    pub split_gain: f64,
    /// The edge that achieves it, in the feature's own units.
    pub split_at: f64,
    /// `√((n − 2)·g/(1 − g))`, the `t` its `corr` would need to match that
    /// gain, against `n_serial` when there is one and `n_kish` otherwise;
    /// `+inf` at a gain of one, as `t` is at `corr = ±1`.
    ///
    /// **Optimistic by construction**: the cut was chosen by maximizing over
    /// the `bins − 1` candidates, and this statistic does not know that. Read
    /// it as a ranking of features, not as a p-value; a rough correction is
    /// to require it to clear the threshold for `bins − 1` comparisons.
    pub split_gain_t: f64,
}

/// See the module doc. Vectors indexed `[t * n_features + j]` are per pair;
/// the ones of length `n_targets` are per target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marginal {
    cfg: MarginalCfg,
    /// Accumulated weight of every learned row, targets present or not: the
    /// model's `n_eff`.
    w_sum: f64,
    /// `W_t`: accumulated weight of the rows where target `t` was present.
    wt: Vec<f64>,
    /// `Q_t`: accumulated squared weight of the same rows.
    qt: Vec<f64>,
    my: Vec<f64>,
    syy: Vec<f64>,
    mx: Vec<f64>,
    sxx: Vec<f64>,
    sxy: Vec<f64>,
    /// Lagged pair moments, when the spec asks for lags (E66). **Boxed**, as
    /// the bins are and for the same reason: inline they add hundreds of
    /// bytes to every `Marginal`, which pushes the vectors the per-row loop
    /// reads further apart and costs a stream that uses neither feature
    /// (docs/PERFORMANCE.md §17). `default` so
    /// a state written before E66 loads with none, but **not** skipped: only
    /// the last field may skip, and that is `win`. Two skipping fields in a
    /// row are ambiguous in the compact encoding, which writes a struct as an
    /// array -- with this one skipped and `win` present, `win`'s value lands
    /// in this slot. `tests/state_encoding.rs` holds the rule down.
    #[serde(default)]
    lag: Option<Box<crate::MarginalLags>>,
    /// Binned target moments (E67), when the spec asks for bins. Boxed for
    /// the reason `lag` gives; `default` and not skipped for the other
    /// reason it gives.
    #[serde(default)]
    bins: Option<Box<Binned>>,
    /// What each mean in `mx` and `my` leaves out: each is a pair no step
    /// is rounded off, as `ew_cov`'s are, so the pairs still agree with it to
    /// the bit ([`crate::comp`]; docs/PLAN.md task 101). Empty in a state
    /// written before them. Ahead of `win`, which skips and must stay last.
    #[serde(default)]
    mx_lo: Vec<f64>,
    #[serde(default)]
    my_lo: Vec<f64>,
    /// Per (target, feature) pair and per target, the value it has held on
    /// the target's rows since it last changed, and the weight of those rows
    /// ([`crate::Runs`]): what lets a window say that a slot held one value
    /// over it, which its subtraction cannot (docs/PLAN.md task 94). Empty in
    /// a state written before them. Ahead of `win`.
    #[serde(default)]
    x_runs: crate::Runs,
    #[serde(default)]
    y_runs: crate::Runs,
    /// Per target, its learned rows so far: what its runs' start rows count
    /// in, and what the window's snapshot records. Ahead of `win`.
    #[serde(default)]
    rows_t: Vec<u64>,
    /// The hard-cutoff window, when the spec asks for one. Last, for the
    /// reason `MarginalCfg::window` gives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
    /// Rows whose pair work waits for a sharded flush
    /// ([`Self::step_sharded`]). Not state: serde skips it, and every reader
    /// of the pairs flushes first. Boxed, as `lag` and `bins` are and for
    /// the same reason: its seven vectors would otherwise sit between the
    /// ones the unsplit row reads.
    #[serde(skip)]
    defer: Box<Deferred>,
}
/// The window's clock and the snapshots it subtracts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Windowed {
    clock: f64,
    snaps: crate::Snapshots<MarginalMoments>,
}

/// Every accumulator a pair is read from, before a row and decayed to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MarginalMoments {
    w_sum: f64,
    wt: Vec<f64>,
    qt: Vec<f64>,
    my: Vec<f64>,
    syy: Vec<f64>,
    mx: Vec<f64>,
    sxx: Vec<f64>,
    sxy: Vec<f64>,
    /// Each target's count of learned rows when the snapshot was taken
    /// (`crate::window::Moments::rows`); `None` in a snapshot written before
    /// it, which then holds nothing.
    #[serde(default)]
    rows: Option<Vec<u64>>,
    /// The lag moments, under a `window` with `lags` (docs/PLAN.md task 137):
    /// `L·T + (L + 2C)·p·T` doubles beside the `(3p + 5)·T` above, `C` the
    /// cross lags. Last, and not skipped, so both encodings read a snapshot
    /// without it.
    #[serde(default)]
    lag: Option<crate::LagMoments>,
}

impl crate::Footprint for MarginalMoments {
    /// Every vector the snapshot holds, the per-target row counts included
    /// (docs/PLAN.md task 130).
    fn footprint(&self) -> usize {
        std::mem::size_of::<f64>()
            + [
                &self.wt, &self.qt, &self.my, &self.syy, &self.mx, &self.sxx, &self.sxy,
            ]
            .iter()
            .map(|v| crate::window::floats(v))
            .sum::<usize>()
            + self
                .rows
                .as_ref()
                .map_or(0, |r| std::mem::size_of_val(r.as_slice()))
            + self.lag.as_ref().map_or(0, crate::Footprint::footprint)
    }
}

/// Where a held row's lag reads the row `lag` learned rows back.
#[derive(Debug, Clone, Copy)]
enum Src {
    /// The `i`-th row of the ring as it stood when the batch began: the
    /// ring is not touched while rows are held, since a held row may still
    /// read a row a push would drop.
    Ring(usize),
    /// The `r`-th held row.
    Row(usize),
}

/// Rows whose pair work waits for a flush ([`Marginal::step_sharded`],
/// docs/PLAN.md task 126). Everything a pair needs from a row is fixed when
/// the row arrives -- the row's features and targets, each target's mix,
/// where each lag's row back is, the bins' weight -- and none of it depends
/// on another pair, so the pairs can be stepped later, row by row in
/// order, split by feature.
#[derive(Debug, Clone, Default)]
struct Deferred {
    /// Rows held.
    n: usize,
    /// Their features, `p` a row.
    xs: Vec<f64>,
    /// Their targets, `T` a row: what a later row's lag reads back.
    ys: Vec<Option<f64>>,
    /// Per row and target, the mix its pairs take; `None` where they do not
    /// move.
    mixes: Vec<Option<TargetMix>>,
    /// Per row, the mix the shared feature moments take under
    /// [`FeatureMomentLayout::Shared`] ([`shared_mix`]); empty otherwise.
    shared: Vec<Option<(f64, f64)>>,
    /// The row's feature deviations, a buffer [`pair_kernel_shared`] fills:
    /// kept here so the unsplit row allocates none.
    dx: Vec<f64>,
    /// Per row, the weight its cells take in the bins, `None` where it
    /// writes none.
    u: Vec<Option<f64>>,
    /// Per row and lag, where the row that far back is; `None` where the
    /// ring does not reach it yet, and the lag waits.
    backs: Vec<Option<Src>>,
    /// Per row, target and lag, the target's value that far back against
    /// its mean before this row; `None` where it was absent then.
    dy_lag: Vec<Option<f64>>,
    /// The rows a lag can reach, oldest first, as the ring would hold them
    /// had each held row been pushed.
    ring: std::collections::VecDeque<Src>,
}

/// Held rows are never equal to anything: two models compare equal only
/// once both have flushed, which is when their pairs say what they are.
impl PartialEq for Deferred {
    fn eq(&self, other: &Self) -> bool {
        self.n == 0 && other.n == 0
    }
}

impl Deferred {
    /// Hold nothing, keeping the buffers for the next batch when `keep`.
    fn clear(&mut self, keep: bool) {
        if !keep {
            *self = Self::default();
            return;
        }
        self.n = 0;
        self.xs.clear();
        self.ys.clear();
        self.mixes.clear();
        self.shared.clear();
        self.u.clear();
        self.backs.clear();
        self.dy_lag.clear();
        self.ring.clear();
    }
}

/// Bartlett's serial-dependence factor `1 + 2·Σ_l rho_x(l)·rho_y(l)`, and
/// the two fitted decays when the rule is geometric.
///
/// `"truncated"` sums the kept lags as they are, and `"bartlett"` weights lag
/// `l` by `1 − l/(L + 1)`, `L` the longest kept. `"geometric"` fits
/// `rho(l) = phi^l` per series by least squares on `log rho` over the kept
/// lags with `rho > 0` — a straight line through the origin in `l`, so
/// `log phi = Σ l·log rho / Σ l²` — and sums the tail in closed form,
/// `2·p/(1 − p)` with `p = phi_x·phi_y`. Fewer than two usable lags on
/// either side, or a product that does not converge, gives NaN rather than a
/// number the caller would have to distrust.
fn serial_factor(
    rule: SerialRule,
    lags: &[usize],
    rho_x: &[f64],
    rho_y: &[f64],
) -> (f64, f64, f64) {
    match rule {
        SerialRule::Truncated | SerialRule::Bartlett => {
            let longest = lags.iter().copied().max().unwrap_or(0) as f64;
            let mut sum = 0.0;
            for ((&l, rx), ry) in lags.iter().zip(rho_x).zip(rho_y) {
                if rx.is_finite() && ry.is_finite() {
                    let w = match rule {
                        SerialRule::Bartlett => 1.0 - l as f64 / (longest + 1.0),
                        _ => 1.0,
                    };
                    sum += w * rx * ry;
                }
            }
            // A factor at or below zero is an estimate outside the parameter
            // space -- two series whose autocorrelations have opposite signs
            // -- and says nothing, as `Geometric` does, where a floor at
            // `f64::MIN_POSITIVE` reported an infinite count (review
            // 2026-09-12, S18). Uniform weights on the lags do not keep the
            // sum positive; Bartlett's `1 − l/(L + 1)` do on a dense,
            // positive-definite sequence, and fall back to this rule where
            // the sequence is neither.
            let factor = 1.0 + 2.0 * sum;
            let factor = if factor > 0.0 { factor } else { f64::NAN };
            (factor, f64::NAN, f64::NAN)
        }
        SerialRule::Geometric => {
            let fit = |rho: &[f64]| -> f64 {
                let (mut num, mut den, mut used) = (0.0, 0.0, 0);
                for (&l, &r) in lags.iter().zip(rho) {
                    if r.is_finite() && r > 0.0 {
                        let l = l as f64;
                        num += l * r.ln();
                        den += l * l;
                        used += 1;
                    }
                }
                if used < 2 || den <= 0.0 {
                    return f64::NAN;
                }
                (num / den).exp()
            };
            let (px, py) = (fit(rho_x), fit(rho_y));
            if !px.is_finite() || !py.is_finite() {
                return (f64::NAN, px, py);
            }
            let prod = px * py;
            if !(0.0..1.0).contains(&prod) {
                // A product at or above one is a series the geometric tail
                // does not sum for: say nothing rather than a negative count.
                return (f64::NAN, px, py);
            }
            (1.0 + 2.0 * prod / (1.0 - prod), px, py)
        }
    }
}

/// The histogram and, until its edges exist, the rows waiting for them.
///
/// Learned edges cost nothing in accuracy: the warm-up rows are **held**, not
/// spent, and replayed with their own decays the moment the edges are fixed,
/// so the histogram is exactly what it would have been had the edges been
/// known before the first row. Exactly, to the bit, unless rows of weight
/// zero fall inside the warm-up: each of those is held as its decay alone,
/// folded into the next held row so that a run of them cannot grow the hold,
/// and a product of decays rounds differently from the same decays applied
/// one at a time -- by about 1e-15 of the data's scale
/// (`learned_edges_lose_no_row_across_targets`). The price is memory, which
/// `BinCfg::validate` bounds up front.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Binned {
    /// `None` until the edges are fixed: either at construction, when they
    /// were given, or at the `warm_rows`-th learned row.
    hist: Option<crate::MarginalBins>,
    /// Warm-up rows in arrival order. Empty once the edges exist.
    held: Vec<crate::margbins::HeldRow>,
    /// Decay from zero-weight rows since the last held row, which teach
    /// nothing but still age the histogram. Folded into the next held row so
    /// a run of them cannot grow `held`.
    pending_lam: f64,
}

impl Binned {
    /// Whether the histogram, once the edges exist, and every held row are
    /// those of `p` features and `t` targets (review 2026-09-18, B3).
    fn has_shape(&self, p: usize, t: usize) -> bool {
        self.hist.as_ref().is_none_or(|h| h.has_shape(p, t))
            && self.held.iter().all(|r| r.x.len() == p && r.y.len() == t)
    }
}

impl Binned {
    /// Fix the edges from the held rows and replay everything held, in
    /// order, with its decay. Only ever reached for learned edges: given
    /// ones build the histogram in [`Marginal::new`] and nothing is held.
    fn freeze(&mut self, cfg: &crate::BinCfg, p: usize, n_targets: usize) {
        debug_assert!(cfg.edges.is_none(), "given edges never hold rows");
        let edges: Vec<Vec<f64>> = (0..p)
            .map(|j| {
                let mut vals: Vec<(f64, f64)> = self
                    .held
                    .iter()
                    .map(|r| (r.x.get(j).copied().unwrap_or(f64::NAN), r.w))
                    .collect();
                crate::edges_from(cfg.rule, cfg.n_bins, &mut vals)
            })
            .collect();
        let mut hist = crate::MarginalBins::new(p, n_targets, edges)
            .expect("edges_from gives one finite, strictly increasing list per feature");
        for row in &self.held {
            hist.decay(row.lam);
            hist.update_row(&row.x, &row.y, row.w);
        }
        // `feed_bins` folded the pending decay into the row it just held
        // before calling, so there is none left to apply.
        debug_assert_eq!(self.pending_lam, 1.0);
        self.held = Vec::new();
        self.hist = Some(hist);
    }
}

impl Marginal {
    pub fn new(cfg: MarginalCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (p, t) = (cfg.n_features, cfg.n_targets);
        let lag = if cfg.lags.is_empty() {
            None
        } else {
            Some(Box::new(crate::MarginalLags::new(
                cfg.n_features,
                cfg.n_targets,
                cfg.lags.clone(),
                cfg.cross_lags.clone(),
                cfg.shared(),
            )?))
        };
        let windowed = cfg.window.is_some();
        let win = match cfg.window {
            Some(w) => Some(Windowed {
                clock: 0.0,
                snaps: crate::Snapshots::new(w, cfg.window_every.unwrap_or(1))?,
            }),
            None => None,
        };
        let bins = match cfg.bins.as_ref() {
            None => None,
            Some(bc) => {
                let hist = match &bc.edges {
                    Some(e) => Some(crate::MarginalBins::new(p, t, e.clone())?),
                    None => None,
                };
                Some(Box::new(Binned {
                    hist,
                    held: Vec::new(),
                    pending_lam: 1.0,
                }))
            }
        };
        // One mean and variance per feature under `"shared"`, one per pair
        // otherwise; the covariances are always per pair.
        let fx = cfg.feature_len();
        Ok(Self {
            cfg,
            w_sum: 0.0,
            wt: vec![0.0; t],
            qt: vec![0.0; t],
            my: vec![0.0; t],
            syy: vec![0.0; t],
            mx: vec![0.0; fx],
            sxx: vec![0.0; fx],
            sxy: vec![0.0; p * t],
            lag,
            bins,
            mx_lo: vec![0.0; fx],
            my_lo: vec![0.0; t],
            // Runs are a window's (docs/PLAN.md tasks 124 and 128): without
            // one they are not even allocated, where at `p·T` slots they
            // were 16 bytes a pair in memory and in every state file.
            x_runs: if windowed {
                crate::Runs::new(p * t)
            } else {
                crate::Runs::off()
            },
            y_runs: if windowed {
                crate::Runs::new(t)
            } else {
                crate::Runs::off()
            },
            rows_t: vec![0; t],
            win,
            defer: Box::default(),
        })
    }

    /// One row into the histogram, decayed first so that the row's own
    /// weight is undecayed -- the same recursion the pair moments use.
    /// Before the edges exist the row is held instead; the `warm_rows`-th
    /// learned row fixes the edges and replays every held row.
    ///
    /// Counting *learned* rows, and holding raw values, is what makes this
    /// chunk-invariant: the edges are fixed at the same row of the stream
    /// however the stream arrives.
    fn feed_bins(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, weight: f64) {
        let Some(cfg) = self.cfg.bins.as_ref() else {
            return;
        };
        let Some(b) = self.bins.as_mut() else {
            return;
        };
        if let Some(hist) = b.hist.as_mut() {
            hist.decay(lam);
            hist.update_row(x, y, weight);
            return;
        }
        if weight > 0.0 && weight.is_finite() {
            // Grown by doubling up to `bin_warm_rows`, never reserved whole
            // up front: a short group holds only its rows, and the budget's
            // count (`margbins::hold_bytes`) stays the most the hold can
            // reach (review 2026-09-26, A2). A hold restored from a state has
            // no spare room and grows from here.
            if b.held.len() == b.held.capacity() {
                let want = (2 * b.held.len()).clamp(4, cfg.warm_rows.max(4));
                b.held.reserve_exact(want.saturating_sub(b.held.len()));
            }
            b.held.push(crate::margbins::HeldRow {
                x: x.to_vec(),
                y: y.to_vec(),
                lam: lam * b.pending_lam,
                w: weight,
            });
            b.pending_lam = 1.0;
            if b.held.len() >= cfg.warm_rows {
                b.freeze(cfg, self.cfg.n_features, self.cfg.n_targets);
            }
        } else {
            // Learns nothing, but time still passed.
            b.pending_lam *= lam;
        }
    }

    pub fn cfg(&self) -> &MarginalCfg {
        &self.cfg
    }

    /// The boundary snapshot and the factor that decays it forward, when a
    /// `window` is set and something has aged out of it.
    fn boundary(&self) -> Option<(&MarginalMoments, f64)> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        if old.w_sum == 0.0 {
            return None;
        }
        Some((old, self.cfg.decay.factor(win.clock - u)))
    }

    /// One truncated `(weight, mean, centred second moment)` triple:
    /// `A(t) - f·A(u)` by the centred pooling identity `window::truncated`
    /// uses, one pair at a time so a readout stays O(1). Every term is a
    /// centred moment or a difference of two means, so a pair against a
    /// price-level column keeps its precision; going back through the raw
    /// moment `s + ma·mb` did not (review 2026-09-12, C17).
    fn cut(
        w: f64,
        w_old: f64,
        f: f64,
        (ma, mb): (f64, f64),
        (ma_old, mb_old): (f64, f64),
        s: f64,
        s_old: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let wo = f * w_old;
        let wn = w - wo;
        // A remainder at rounding size is an empty window, not a tiny one
        // (`window::EMPTY_FRACTION`, review C2).
        if wn <= crate::window::EMPTY_FRACTION * w || !wn.is_finite() {
            return None;
        }
        let (ratio, g) = (wo / wn, w / wn);
        let (da, db) = (ma_old - ma, mb_old - mb);
        let a = ma - ratio * da;
        let b = mb - ratio * db;
        Some((wn, a, b, g * s - ratio * s_old - ratio * g * da * db))
    }

    /// The accumulated weight the pairs are read from: under a `window`, the
    /// weight inside it. The remainder is zeroed below the same
    /// `EMPTY_FRACTION` the pairs use, so after a gap that empties the window
    /// this reports 0 exactly rather than the rounding crumb the subtraction
    /// leaves, matching every `Pair::n_eff` (review 2026-09-18, S4).
    pub fn n_eff(&self) -> f64 {
        match self.boundary() {
            Some((old, f)) => empty_or(self.w_sum - f * old.w_sum, self.w_sum),
            None => self.w_sum,
        }
    }

    /// `W_t`, the weight behind target `t`'s pairs.
    pub fn target_weight(&self, t: usize) -> f64 {
        match self.boundary() {
            Some((old, f)) => empty_or(self.wt[t] - f * old.wt[t], self.wt[t]),
            None => self.wt[t],
        }
    }

    /// The statistics of feature `j` against target `t`. The moments are
    /// reported at any weight; `corr`, `beta` and `t` wait for
    /// `min_weight`.
    pub fn pair(&self, t: usize, j: usize) -> Pair {
        // Loud in every build: a pair read over held rows is wrong for good
        // (review 2026-09-26, A3).
        assert_eq!(
            self.defer.n, 0,
            "marginal: rows held for a sharded flush are flushed before a pair is read"
        );
        let i = t * self.cfg.n_features + j;
        // The feature's moments: the pair's own, or the feature's under
        // `"shared"`, which takes no window.
        let fi = self.cfg.feature_at(t, j);
        // With a `window`, every moment this pair is built from is truncated
        // to it first: the weight, the two means, and the three centred
        // second moments. Each is the same subtraction `EwCov` makes.
        let (n_eff, n_kish, mean_x, mean_y, var_x, var_y, cov) = match self.boundary() {
            None => (
                self.wt[t],
                self.wt[t] * self.wt[t] / self.qt[t],
                self.mx[fi],
                self.my[t],
                self.sxx[fi].max(0.0),
                self.syy[t].max(0.0),
                self.sxy[i],
            ),
            Some((old, f)) => {
                let cut = |ms, ms_old, s, s_old| {
                    Self::cut(self.wt[t], old.wt[t], f, ms, ms_old, s, s_old)
                };
                match (
                    cut(
                        (self.mx[i], self.mx[i]),
                        (old.mx[i], old.mx[i]),
                        self.sxx[i],
                        old.sxx[i],
                    ),
                    cut(
                        (self.my[t], self.my[t]),
                        (old.my[t], old.my[t]),
                        self.syy[t],
                        old.syy[t],
                    ),
                    cut(
                        (self.mx[i], self.my[t]),
                        (old.mx[i], old.my[t]),
                        self.sxy[i],
                        old.sxy[i],
                    ),
                ) {
                    (Some((w, mx, _, sxx)), Some((_, my, _, syy)), Some((_, _, _, sxy))) => {
                        let q = (self.qt[t] - f * f * old.qt[t]).max(0.0);
                        let (mut mx, mut my, mut sxx, mut syy, mut sxy) =
                            (mx, my, sxx.max(0.0), syy.max(0.0), sxy);
                        // A slot that held one value over every row inside
                        // the window has no spread there, and the subtraction
                        // cannot say so; its run can, when it started at or
                        // before the first of the target's learned rows
                        // inside the window (`crate::truncated`, docs/PLAN.md
                        // task 94).
                        let (p, n) = (self.cfg.n_features, self.cfg.n_targets);
                        let first = old.rows.as_ref().and_then(|r| r.get(t)).map(|r| r + 1);
                        if let Some(first) = first {
                            if let Some(value) = self.x_runs.started_by(n * p, i, first) {
                                (mx, sxx, sxy) = (value, 0.0, 0.0);
                            }
                            if let Some(value) = self.y_runs.started_by(n, t, first) {
                                (my, syy, sxy) = (value, 0.0, 0.0);
                            }
                        }
                        (w, w * w / q, mx, my, sxx, syy, sxy)
                    }
                    // Nothing inside the window: report nothing, not stale
                    // moments (hard rule 9).
                    _ => (
                        0.0,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                    ),
                }
            }
        };
        let (corr, beta, t_stat) = if n_eff >= self.cfg.min_weight[t] {
            // The same product of the same two roots `ew_cov` takes, so the
            // correlations agree to the bit.
            let d = var_x.sqrt() * var_y.sqrt();
            let corr = if d > 0.0 {
                (cov / d).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            };
            let beta = if var_x > 0.0 { cov / var_x } else { f64::NAN };
            let t_stat = if n_kish > 2.0 {
                corr * ((n_kish - 2.0) / (1.0 - corr * corr)).sqrt()
            } else {
                f64::NAN
            };
            (corr, beta, t_stat)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        // The lag statistics, when the spec asked for lags. Correlations
        // rather than covariances, because the correction is a product of
        // two autocorrelations and the caller reads them as such.
        let (mut lagcorr_xx, mut lagcorr_yy) = (Vec::new(), Vec::new());
        let (mut lagcorr_xy, mut lagcorr_yx) = (Vec::new(), Vec::new());
        let (mut n_serial, mut t_serial) = (f64::NAN, f64::NAN);
        let (mut phi_x, mut phi_y) = (f64::NAN, f64::NAN);
        if let Some(lag) = self.lag.as_ref() {
            // `C_l / (sd_a · sd_b)` in every orientation, the auto terms
            // included -- `ew_cov`'s expression, so the two surfaces agree
            // to the bit (`tests/test_marginal_lags.py` holds them to it).
            let sd_x = var_x.sqrt();
            let sd_y = var_y.sqrt();
            let norm = |v: f64, d: f64| if d > 0.0 { v / d } else { f64::NAN };
            // Under a `window` each lag moment is truncated as the pair's
            // are, in sum form with the target's weight: `(W_t·C −
            // f·W_u·C_u) / W_R`, the increments made inside the window
            // (docs/PLAN.md task 137). A snapshot without them -- none is
            // taken unless the spec asks for lags under the window -- has no
            // lag statistics to give.
            let cut = self
                .boundary()
                .map(|(old, f)| old.lag.as_ref().map(|m| (f * old.wt[t], m)));
            let at = |live: f64, then: &dyn Fn(&crate::LagMoments) -> f64| match cut {
                None => live,
                Some(None) => f64::NAN,
                Some(Some((wo, m))) => {
                    let wn = self.wt[t] - wo;
                    if wn > crate::window::EMPTY_FRACTION * self.wt[t] && wn.is_finite() {
                        (self.wt[t] * live - wo * then(m)) / wn
                    } else {
                        f64::NAN
                    }
                }
            };
            for li in 0..lag.lags().len() {
                let cxx = at(lag.cxx(li, t, j), &|m| m.cxx(li, t, j));
                let cyy = at(lag.cyy(li, t), &|m| m.cyy(li, t));
                lagcorr_xx.push(norm(cxx, sd_x * sd_x));
                lagcorr_yy.push(norm(cyy, sd_y * sd_y));
            }
            // Over the cross lags, which are every lag unless `cross_lags`
            // names fewer (E70).
            for ci in 0..lag.cross_lags().len() {
                let cxy = at(lag.cxy(ci, t, j), &|m| m.cxy(ci, t, j));
                let cyx = at(lag.cyx(ci, t, j), &|m| m.cyx(ci, t, j));
                lagcorr_xy.push(norm(cxy, sd_x * sd_y));
                lagcorr_yx.push(norm(cyx, sd_x * sd_y));
            }
            if let Some(rule) = self.cfg.serial_rule {
                let (factor, px, py) = serial_factor(rule, lag.lags(), &lagcorr_xx, &lagcorr_yy);
                phi_x = px;
                phi_y = py;
                if factor.is_finite() && factor > 0.0 {
                    n_serial = n_kish / factor;
                    // `±inf` at `corr = ±1`, as `t` is: the honest value.
                    if n_serial > 2.0 && corr.is_finite() {
                        t_serial = corr * ((n_serial - 2.0) / (1.0 - corr * corr)).sqrt();
                    }
                }
            }
        }
        // The binned view of the same pair. Gated by `min_weight` like the
        // linear statistics, since it answers the same question.
        let (mut bin_edges, mut bin_n) = (Vec::new(), Vec::new());
        let (mut bin_mean_y, mut bin_var_y) = (Vec::new(), Vec::new());
        let (mut split_gain, mut split_at, mut split_gain_t) = (f64::NAN, f64::NAN, f64::NAN);
        if let Some(hist) = self.bins.as_ref().and_then(|b| b.hist.as_ref()) {
            bin_edges = hist.edges(j).to_vec();
            for b in hist.bins(t, j) {
                bin_n.push(b.n);
                bin_mean_y.push(b.mean_y);
                bin_var_y.push(b.var_y);
            }
            if n_eff >= self.cfg.min_weight[t] {
                if let Some(split) = hist.best_split(t, j) {
                    split_gain = split.gain;
                    split_at = split.at;
                    // Serial dependence, when it was measured, deflates this
                    // count for the same reason it deflates `t`.
                    let n = if n_serial.is_finite() {
                        n_serial
                    } else {
                        n_kish
                    };
                    // `+inf` at a gain of one, as `t` is at `corr = ±1`.
                    if n > 2.0 {
                        split_gain_t = ((n - 2.0) * split.gain / (1.0 - split.gain)).sqrt();
                    }
                }
            }
        }
        Pair {
            n_eff,
            n_kish,
            bin_edges,
            bin_n,
            bin_mean_y,
            bin_var_y,
            split_gain,
            split_at,
            split_gain_t,
            lagcorr_xx,
            lagcorr_yy,
            lagcorr_xy,
            lagcorr_yx,
            n_serial,
            t_serial,
            phi_x,
            phi_y,
            mean_x,
            var_x,
            mean_y,
            var_y,
            cov,
            corr,
            beta,
            t: t_stat,
        }
    }

    /// The lagged moments, against the same old means and the same `a`/`b`
    /// the pair update is about to apply -- which is what makes a lag-0
    /// moment identical to the contemporaneous one, and what `EwLagCov` does
    /// beside `EwCov`. Run *before* [`Marginal::learn`], since it reads the
    /// means that call is about to move.
    ///
    /// This is a separate pass rather than a branch inside `learn`'s loop on
    /// purpose. A call in that loop -- even one that is never taken -- makes
    /// the compiler assume the callee could reallocate `mx`, `sxx` and
    /// `sxy`, so it reloads their base pointers every iteration and stops
    /// vectorizing. Measured: 54M rows/s with the branch inside, 74M with it
    /// out, on a stream with no lags at all (docs/PERFORMANCE.md §17). The
    /// arithmetic is unchanged -- `a` and `b` are recomputed from the same
    /// values with the same operations, so the moments stay bit-identical to
    /// `ew_cov(lags=)`, which `lagged_pair_moments_are_ew_covs_to_the_bit`
    /// checks.
    fn learn_lags(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, w: f64) {
        if w < 0.0 {
            return;
        }
        self.size_lo();
        let p = self.cfg.n_features;
        let Some(lag) = self.lag.as_mut() else {
            return;
        };
        if self.cfg.shared() {
            // The feature's autocovariance once, with the row's mix over the
            // model's weight before the row; each present target's terms
            // with its own (docs/PLAN.md task 125).
            let targets: Vec<Option<crate::TargetLag>> = y
                .iter()
                .enumerate()
                .map(|(t, yt)| {
                    let yt = (*yt)?;
                    let w_new = lam * self.wt[t] + w;
                    (w_new > 0.0).then(|| crate::TargetLag {
                        a: lam * self.wt[t] / w_new,
                        b: w / w_new,
                        yt,
                        my: self.my[t],
                        my_lo: self.my_lo[t],
                    })
                })
                .collect();
            let row = shared_mix(lam, self.w_sum, w);
            lag.update_shared(x, row, &targets, &self.mx, &self.mx_lo);
            return;
        }
        for (t, yt) in y.iter().enumerate() {
            // A target that is absent, or a row with no weight behind it or
            // on it, moves nothing: these are normalized moments and their
            // decay rides on `W_t` (the `marglag` module doc says why they
            // must not be aged on their own).
            let Some(yt) = *yt else {
                continue;
            };
            let w_new = lam * self.wt[t] + w;
            if w_new <= 0.0 {
                continue;
            }
            let row = t * p;
            lag.update_target(
                t,
                x,
                yt,
                crate::PairMix {
                    mx: &self.mx[row..row + p],
                    mx_lo: &self.mx_lo[row..row + p],
                    my: self.my[t],
                    my_lo: self.my_lo[t],
                    a: lam * self.wt[t] / w_new,
                    b: w / w_new,
                },
            );
        }
    }

    /// The means' low parts at the means' own lengths, which a state written
    /// before them does not carry: they start at zero. Here, not per element
    /// inside the pair loop, whose vectorizing a possible reallocation would
    /// stop (see `learn_lags`).
    fn size_lo(&mut self) {
        if self.mx_lo.len() != self.mx.len() {
            self.mx_lo = vec![0.0; self.mx.len()];
        }
        if self.my_lo.len() != self.my.len() {
            self.my_lo = vec![0.0; self.my.len()];
        }
        if self.rows_t.len() != self.cfg.n_targets {
            self.rows_t = vec![0; self.cfg.n_targets];
        }
    }

    fn learn(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        debug_assert!(w >= 0.0, "marginal requires a non-negative weight, got {w}");
        if w < 0.0 {
            return;
        }
        let (p, n_targets) = (self.cfg.n_features, self.cfg.n_targets);
        self.size_lo();
        // The model-level weight: every row, present targets or not. A
        // zero-weight first row leaves it at zero, which is legal (rule 9).
        let w_before = self.w_sum;
        self.w_sum = lam * self.w_sum + w;
        if self.cfg.shared() {
            // Every target's own side first -- its mix reads its state before
            // the row, which the pairs below do not touch -- then the pairs,
            // centred on the shared means before the row.
            let mixes: Vec<Option<TargetMix>> = y
                .iter()
                .enumerate()
                .map(|(t, &yt)| self.advance_target(t, yt, lam, w))
                .collect();
            let mut sxy: Vec<&mut [f64]> = self.sxy.chunks_exact_mut(p).collect();
            pair_kernel_shared(
                shared_mix(lam, w_before, w),
                &mixes,
                x,
                &mut self.mx,
                &mut self.mx_lo,
                &mut self.sxx,
                &mut sxy,
                &mut self.defer.dx,
            );
            return;
        }
        for (t, &yt) in y.iter().enumerate() {
            let Some(mix) = self.advance_target(t, yt, lam, w) else {
                continue;
            };
            let r = t * p..(t + 1) * p;
            if let Some(row) = mix.runs_row {
                if let Some((values, start)) = self.x_runs.slots_mut(n_targets * p) {
                    crate::runs::track_slots(&mut values[r.clone()], &mut start[r.clone()], x, row);
                }
            }
            pair_kernel(
                mix,
                x,
                &mut self.mx[r.clone()],
                &mut self.mx_lo[r.clone()],
                &mut self.sxx[r.clone()],
                &mut self.sxy[r],
            );
        }
    }

    /// The window's snapshot, when the row is one it keeps: every
    /// accumulator as it stands *before* this row, decayed to this row's
    /// clock, so subtracting it later retains this row and everything
    /// after. Keyed by a clock the model accumulates itself, so the boundary
    /// cannot depend on the chunking.
    fn offer_snapshot(&mut self, d_clock: f64, lam: f64) {
        if let Some(win) = self.win.as_mut() {
            let t = win.clock + d_clock;
            // Built inside the closure so the snapshot is only formed on the
            // rows `offer` keeps, not on every row (review 2026-09-18, P1).
            win.snaps.offer(t, || MarginalMoments {
                w_sum: self.w_sum * lam,
                wt: self.wt.iter().map(|w| w * lam).collect(),
                qt: self.qt.iter().map(|q| q * lam * lam).collect(),
                my: self.my.clone(),
                syy: self.syy.clone(),
                mx: self.mx.clone(),
                sxx: self.sxx.clone(),
                sxy: self.sxy.clone(),
                rows: Some(self.rows_t.clone()),
                lag: self.lag.as_ref().map(|l| l.moments()),
            });
            win.clock = t;
            win.snaps.trim(t);
        }
    }

    /// [`OnlineModel::step`], with the row's pair work held back and done
    /// by `shards` a batch of rows at a time (docs/PLAN.md task 126).
    ///
    /// Every target's own statistics, the model's weight and what the row
    /// reports are advanced now, as `step` advances them; none of them reads
    /// a pair. What each pair takes from the row -- the features, each
    /// target's mix, where each lag's row back is, the bins' weight -- is
    /// fixed now and held. A flush then steps each pair through the held
    /// rows in order, the pairs split into `shards.count` ranges of
    /// features and run by `shards.run`. A pair touches only its own cells,
    /// so every number is the one `step` gives, to the bit, whatever the
    /// count and whichever threads run the ranges.
    ///
    /// Held rows are flushed when the batch is full, before a window
    /// snapshot (which copies the pairs) and before the bins fold their
    /// scale into every cell. Everything that reads a pair needs the
    /// caller's [`Self::flush`] first: `pair`, `state`, a comparison. A
    /// plain [`OnlineModel::step`] flushes on its own, on this thread.
    ///
    /// Why a batch and not a row: a fork-join costs about ten
    /// microseconds, idle threads being woken for it, and at `p = 10,000` a
    /// row of the moments alone is about six. At one and nine targets, one
    /// fork-join per row ran at best 0.6 and 1.7 times as fast as no split;
    /// one per 64 rows, 3.9 and 6.5 times (docs/PERFORMANCE.md §25).
    pub fn step_sharded(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        shards: &Shards<'_>,
    ) -> Step {
        if shards.count <= 1 {
            self.flush(shards);
            return OnlineModel::step(self, x, y, d_clock, weight);
        }
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        // The snapshot copies the pairs as they stand before this row.
        if self
            .win
            .as_ref()
            .is_some_and(|w| w.snaps.takes(w.clock + d_clock))
        {
            self.flush_rows(shards, true);
        }
        self.offer_snapshot(d_clock, lam);
        self.defer_row(x, y, lam, weight, shards);
        out
    }

    /// Learn every row [`Self::step_sharded`] holds, split and run as
    /// `shards` says, and let go of the buffers that held them. A no-op
    /// when nothing is held.
    pub fn flush(&mut self, shards: &Shards<'_>) {
        self.flush_rows(shards, false);
    }

    /// Rows held for a flush.
    pub fn held_rows(&self) -> usize {
        self.defer.n
    }

    /// One row into the batch: `learn_lags`, `feed_bins` and `learn` with
    /// every pair's part held back.
    fn defer_row(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, w: f64, shards: &Shards<'_>) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        debug_assert!(w >= 0.0, "marginal requires a non-negative weight, got {w}");
        let (p, n_targets) = (self.cfg.n_features, self.cfg.n_targets);
        // The bins before the pairs, as `step` feeds them.
        let u = self.defer_bins(x, y, lam, w, shards);
        if w < 0.0 {
            return;
        }
        self.size_lo();
        let n_lags = self.lag.as_ref().map_or(0, |l| l.lags().len());
        if self.defer.n == 0 {
            // A batch begins with the ring as it stands.
            let depth = self.lag.as_ref().map_or(0, |l| l.depth());
            self.defer.ring.extend((0..depth).map(Src::Ring));
        }
        let r = self.defer.n;
        if self.cfg.shared() {
            self.defer.shared.push(shared_mix(lam, self.w_sum, w));
        }
        self.w_sum = lam * self.w_sum + w;
        if let Some(lag) = self.lag.as_ref() {
            let depth = self.defer.ring.len();
            for &l in lag.lags() {
                let back = (l <= depth).then(|| self.defer.ring[depth - l]);
                self.defer.backs.push(back);
            }
        }
        for (t, &yt) in y.iter().enumerate() {
            let mix = self.advance_target(t, yt, lam, w);
            self.defer.mixes.push(mix);
            let Some(lag) = self.lag.as_deref_mut() else {
                continue;
            };
            let d = &mut self.defer;
            for li in 0..n_lags {
                // The target that far back, against its mean before this
                // row, as `MarginalLags::update_target` takes it; its own
                // lagged moment moves now, with the target's scalars.
                let dy_lag = match (mix, d.backs[r * n_lags + li]) {
                    (Some(m), Some(src)) => {
                        let back = match src {
                            Src::Ring(i) => lag.ring_row(i).1[t],
                            Src::Row(q) => d.ys[q * n_targets + t],
                        };
                        back.map(|v| crate::comp::dev(v, m.my, m.my_lo))
                    }
                    _ => None,
                };
                if let Some(m) = mix {
                    let c = lag.cyy_mut(li, t);
                    *c = crate::marglag::step_cyy(m.a, m.b, m.dy, dy_lag, *c);
                }
                d.dy_lag.push(dy_lag);
            }
        }
        self.defer.xs.extend_from_slice(x);
        self.defer.ys.extend_from_slice(y);
        self.defer.u.push(u);
        self.defer.n += 1;
        // The ring holds rows that taught something, as `step` pushes them.
        if let Some(lag) = self.lag.as_ref() {
            if w > 0.0 && w.is_finite() && x.iter().all(|v| v.is_finite()) {
                self.defer.ring.push_back(Src::Row(r));
                if self.defer.ring.len() > lag.max_lag() {
                    self.defer.ring.pop_front();
                }
            }
        }
        if self.defer.n >= batch_rows(p) {
            self.flush_rows(shards, true);
        }
    }

    /// The bins' side of a held row: [`Self::feed_bins`] with the cell
    /// writes held back. The histogram ages by the row -- the held rows
    /// learned first where the ageing folds the scale into every cell, since
    /// they were weighed against the scale before it -- or, before the edges
    /// exist, the row is held for them as `feed_bins` holds it. The weight
    /// the row's cells take, `None` where it writes none.
    fn defer_bins(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        lam: f64,
        w: f64,
        shards: &Shards<'_>,
    ) -> Option<f64> {
        let hist = self.bins.as_ref().and_then(|b| b.hist.as_ref());
        let Some(folds) = hist.map(|h| h.folds_at(lam)) else {
            // No bins, or the warm-up: no cell is written until the edges
            // exist, and the replay that writes the held rows then is the
            // model's own.
            if self.bins.is_some() {
                self.feed_bins(x, y, lam, w);
            }
            return None;
        };
        if folds {
            self.flush_rows(shards, true);
        }
        let hist = self
            .bins
            .as_mut()
            .and_then(|b| b.hist.as_mut())
            .expect("the histogram exists: checked above");
        hist.decay(lam);
        let u = hist.row_weight(y, w)?;
        // A row that writes a cell makes the histogram not empty now, as the
        // write itself would, so the next row's ageing is not skipped.
        if hist.is_empty() && x.iter().any(|v| v.is_finite()) {
            hist.mark_written();
        }
        Some(u)
    }

    /// Learn the held rows' pairs: every pair vector cut into
    /// `shards.count` ranges of features, one [`MarginalShard`] each, run by
    /// `shards.run`; then the ring is the one those rows leave. `keep` keeps
    /// the buffers for another batch.
    fn flush_rows(&mut self, shards: &Shards<'_>, keep: bool) {
        if self.defer.n == 0 {
            self.defer.clear(keep);
            return;
        }
        let d = std::mem::take(&mut self.defer);
        let (p, n_targets) = (self.cfg.n_features, self.cfg.n_targets);
        let count = shards.count.clamp(1, p.max(1));
        let cuts: Vec<usize> = (0..=count).map(|k| k * p / count).collect();
        let cross_of = self.lag.as_ref().map(|l| l.cross_of()).unwrap_or_default();
        let tracks_runs = d
            .mixes
            .iter()
            .any(|m| m.is_some_and(|m| m.runs_row.is_some()));
        let writes_cells = d.u.iter().any(Option::is_some);
        let shared = self.cfg.shared();
        if p > 0 {
            let Marginal {
                mx,
                mx_lo,
                sxx,
                sxy,
                lag,
                bins,
                x_runs,
                ..
            } = self;
            let (moments, ring_x) = match lag.as_deref_mut() {
                Some(l) => {
                    let crate::marglag::LagParts {
                        cxx,
                        cxy,
                        cyx,
                        ring_x,
                    } = l.parts();
                    (Some([cxx, cxy, cyx]), Some(ring_x))
                }
                None => (None, None),
            };
            let cells = bins
                .as_deref_mut()
                .and_then(|b| b.hist.as_mut())
                .filter(|_| writes_cells)
                .map(|h| h.cell_parts());
            let job = Job {
                p,
                n_targets,
                shared,
                d: &d,
                ring_x,
                cross_of: &cross_of,
                bins: cells.as_ref().map(|c| (c.edges, c.off)),
            };
            let mut parts: Vec<MarginalShard<'_>> = cuts
                .windows(2)
                .map(|c| MarginalShard::new(c[0]..c[1], &job))
                .collect();
            deal(mx, p, &cuts, |k, v| parts[k].mx.push(v));
            deal(mx_lo, p, &cuts, |k, v| parts[k].mx_lo.push(v));
            deal(sxx, p, &cuts, |k, v| parts[k].sxx.push(v));
            deal(sxy, p, &cuts, |k, v| parts[k].sxy.push(v));
            if tracks_runs {
                if let Some((values, start)) = x_runs.slots_mut(n_targets * p) {
                    deal(values, p, &cuts, |k, v| parts[k].runs.push(v));
                    deal(start, p, &cuts, |k, v| parts[k].runs_start.push(v));
                }
            }
            if let Some([cxx, cxy, cyx]) = moments {
                for v in cxx.iter_mut() {
                    deal(v, p, &cuts, |k, v| parts[k].cxx.push(v));
                }
                for v in cxy.iter_mut() {
                    deal(v, p, &cuts, |k, v| parts[k].cxy.push(v));
                }
                for v in cyx.iter_mut() {
                    deal(v, p, &cuts, |k, v| parts[k].cyx.push(v));
                }
            }
            if let Some(c) = cells {
                let at: Vec<usize> = cuts.iter().map(|&j| c.off[j]).collect();
                let width = c.off[p];
                deal(c.w, width, &at, |k, v| parts[k].w.push(v));
                deal(c.mean, width, &at, |k, v| parts[k].mean.push(v));
                deal(c.m2, width, &at, |k, v| parts[k].m2.push(v));
                deal(c.mean_lo, width, &at, |k, v| parts[k].mean_lo.push(v));
            }
            (shards.run)(&mut parts);
            debug_assert!(
                parts.iter().all(|s| !s.wrote)
                    || bins
                        .as_ref()
                        .and_then(|b| b.hist.as_ref())
                        .is_none_or(|h| !h.is_empty()),
                "a written cell was marked when its row was held"
            );
        }
        // The ring those rows leave: the rows a lag can reach, oldest first.
        // The batch's ring keeps the newest rows of the ring as it stood, in
        // order, and the held rows after them; the former stay where they
        // are, the latter are pushed as `step` would have pushed them
        // (review 2026-09-26, A8: every ring row was copied at every flush).
        if let Some(lag) = self.lag.as_deref_mut() {
            let kept = d.ring.iter().filter(|s| matches!(s, Src::Ring(_))).count();
            lag.keep_last(kept);
            for src in &d.ring {
                if let Src::Row(r) = *src {
                    lag.push(
                        &d.xs[r * p..(r + 1) * p],
                        &d.ys[r * n_targets..(r + 1) * n_targets],
                    );
                }
            }
        }
        self.defer = d;
        self.defer.clear(keep);
    }

    /// Target `t`'s own side of a row: its weight, squared weight, mean and
    /// centred second moment stepped, and under a window its learned rows
    /// and run; and the mix its pairs take, from the target's state before
    /// the row. `None` where its pairs do not move: the target is absent,
    /// or there is no weight behind the row or on it.
    ///
    /// Nothing here reads a pair, and a pair reads nothing here but the mix,
    /// which is what lets [`Self::step_sharded`] advance every target now
    /// and do the pairs later, split by feature.
    fn advance_target(&mut self, t: usize, yt: Option<f64>, lam: f64, w: f64) -> Option<TargetMix> {
        let Some(yt) = yt else {
            // Time passes for a target that is not there: its weight ages,
            // its moments hold, as `ew_ridge` treats a missing target.
            self.wt[t] *= lam;
            self.qt[t] *= lam * lam;
            return None;
        };
        let w_new = lam * self.wt[t] + w;
        if w_new <= 0.0 {
            // No weight in the history and none on this row: nothing to
            // average, and `a`/`b` would be 0/0 (CLAUDE.md rule 9). The
            // weights still take the row's decay -- `lam = 0` with a
            // zero-weight row is a clock gap past `gap_cap` on a row that
            // teaches nothing, and the target's `n_eff` must not outlive the
            // gap while the model's does not.
            self.wt[t] = 0.0;
            self.qt[t] = 0.0;
            return None;
        }
        let a = lam * self.wt[t] / w_new;
        let b = w / w_new;
        let mut runs_row = None;
        // A row counts for the runs when its weight is something next to the
        // target's (`EwCov::update`, `crate::Runs`).
        if w > 0.0 && w > crate::window::EMPTY_FRACTION * (lam * self.wt[t]) {
            self.rows_t[t] += 1;
            // The runs are read by a window's truncated pair alone, and a
            // window is fixed when the model is built: without one they were
            // kept for nothing, and at `p·T` slots that was half the model's
            // time at width (docs/PERFORMANCE.md §24).
            if self.win.is_some() {
                self.y_runs
                    .track_one(self.cfg.n_targets, t, yt, self.rows_t[t]);
                runs_row = Some(self.rows_t[t]);
            }
        }
        use crate::comp::{add, dev};
        let (my, my_lo) = (self.my[t], self.my_lo[t]);
        let dy = dev(yt, my, my_lo);
        // Co-moments from the deviations against the OLD means, then the
        // means advance -- `EwCov::update`'s order, operation for operation,
        // so the pair agrees with `ew_cov` to the bit.
        self.syy[t] = a * self.syy[t] + a * b * dy * dy;
        // A row of weight 0 takes no step (`crate::comp::add` says why).
        if b > 0.0 {
            add(&mut self.my[t], &mut self.my_lo[t], b * dy);
        }
        self.wt[t] = w_new;
        self.qt[t] = lam * lam * self.qt[t] + w * w;
        Some(TargetMix {
            a,
            b,
            dy,
            my,
            my_lo,
            runs_row,
        })
    }
}

/// What a target's pairs take from a row ([`Marginal::advance_target`]):
/// the mix `a = lam·W/W'` and `b = w/W'`, the target's deviation from its
/// mean before the row, that mean itself (the lags centre on it), and the
/// learned row the target's runs start from when the row counts for them.
#[derive(Debug, Clone, Copy)]
struct TargetMix {
    a: f64,
    b: f64,
    dy: f64,
    my: f64,
    my_lo: f64,
    runs_row: Option<u64>,
}

/// One target's pair moments over a range of features, the module doc's
/// recursion on slices of one length: `x` the row's features, and the
/// target's means, their low parts and the two co-moments for the same
/// features. The only copy of the pair update: an unsplit row runs it over
/// every feature, a shard over its own ([`MarginalShard`]).
#[inline]
fn pair_kernel(
    mix: TargetMix,
    x: &[f64],
    mx: &mut [f64],
    mx_lo: &mut [f64],
    sxx: &mut [f64],
    sxy: &mut [f64],
) {
    use crate::comp::{add, dev};
    let (a, b, dy) = (mix.a, mix.b, mix.dy);
    // A row of weight 0 takes no step (`crate::comp::add` says why).
    let step = b > 0.0;
    let means = mx.iter_mut().zip(mx_lo.iter_mut());
    let moments = sxx.iter_mut().zip(sxy.iter_mut());
    for ((&xj, (m, lo)), (s, c)) in x.iter().zip(means).zip(moments) {
        let dx = dev(xj, *m, *lo);
        *s = a * *s + a * b * dx * dx;
        *c = a * *c + a * b * dx * dy;
        if step {
            add(m, lo, b * dx);
        }
    }
}

/// The mix of every learned row, which the shared feature moments take
/// ([`FeatureMomentLayout::Shared`]): `a = lam·W/W'` and `b = w/W'` over the
/// model's own weight, in the expressions [`Marginal::advance_target`] uses
/// for a target's. `None` where there is no weight behind the row or on it:
/// nothing to average, and `a` and `b` would be 0/0 (CLAUDE.md rule 9).
fn shared_mix(lam: f64, w_before: f64, w: f64) -> Option<(f64, f64)> {
    let w_new = lam * w_before + w;
    (w_new > 0.0).then(|| (lam * w_before / w_new, w / w_new))
}

/// The pair moments under [`FeatureMomentLayout::Shared`] over a range of
/// features: each feature's deviation from its shared mean before the row,
/// every present target's covariance stepped with that target's mix, then the
/// feature's variance and mean stepped with the row's own mix (`shared`,
/// from [`shared_mix`]). `sxy` holds each target's covariances over the
/// range. Element for element these are the expressions [`pair_kernel`]
/// evaluates, so where every target is on every learned row -- each
/// target's mix then being the row's, and each target's means the shared
/// ones -- the numbers are its numbers, to the bit. The deviations go
/// through `dx`, a buffer the caller keeps, so each target's covariances
/// are one contiguous pass: stepping every target inside the feature loop
/// strode across them and ran slower than the per-target layout it replaces
/// (docs/PLAN.md task 125).
#[allow(clippy::too_many_arguments)]
#[inline]
fn pair_kernel_shared(
    shared: Option<(f64, f64)>,
    mixes: &[Option<TargetMix>],
    x: &[f64],
    mx: &mut [f64],
    mx_lo: &mut [f64],
    sxx: &mut [f64],
    sxy: &mut [&mut [f64]],
    dx: &mut Vec<f64>,
) {
    use crate::comp::{add, dev};
    // One target: one pass, the per-target kernel's shape, in a function of
    // its own that takes the mixes by value as `pair_kernel` does; inside
    // this one the pass ran at 1.7 times `pair_kernel`'s time (docs/PLAN.md
    // task 125). A target with a mix has weight behind it, so the row has
    // too.
    if let ([mix], [c]) = (mixes, &mut *sxy) {
        if let Some((a, b)) = shared {
            let target = mix.map(|t| (t.a, t.b, t.dy));
            pair_kernel_one(target, (a, b), x, mx, mx_lo, sxx, c);
        }
        return;
    }
    dx.clear();
    dx.extend(
        x.iter()
            .zip(mx.iter().zip(mx_lo.iter()))
            .map(|(&xj, (&m, &lo))| dev(xj, m, lo)),
    );
    for (mix, c) in mixes.iter().zip(sxy.iter_mut()) {
        let Some(m) = mix else {
            continue;
        };
        let (ta, tb, dy) = (m.a, m.b, m.dy);
        for (c, &d) in c.iter_mut().zip(dx.iter()) {
            *c = ta * *c + ta * tb * d * dy;
        }
    }
    if let Some((a, b)) = shared {
        // A row of weight 0 takes no step (`crate::comp::add` says why).
        let step = b > 0.0;
        let means = mx.iter_mut().zip(mx_lo.iter_mut());
        for ((s, &d), (m, lo)) in sxx.iter_mut().zip(dx.iter()).zip(means) {
            *s = a * *s + a * b * d * d;
            if step {
                add(m, lo, b * d);
            }
        }
    }
}

/// [`pair_kernel_shared`] for one target: the target's mix `(a, b, dy)`
/// where it is present steps the covariance, and the row's `(a, b)` the
/// feature's variance and mean, in one pass with no test inside but the one
/// [`pair_kernel`] has.
#[inline]
fn pair_kernel_one(
    target: Option<(f64, f64, f64)>,
    (a, b): (f64, f64),
    x: &[f64],
    mx: &mut [f64],
    mx_lo: &mut [f64],
    sxx: &mut [f64],
    sxy: &mut [f64],
) {
    use crate::comp::{add, dev};
    // A row of weight 0 takes no step (`crate::comp::add` says why).
    let step = b > 0.0;
    let means = mx.iter_mut().zip(mx_lo.iter_mut());
    match target {
        Some((ta, tb, dy)) => {
            let moments = sxx.iter_mut().zip(sxy.iter_mut());
            for ((&xj, (m, lo)), (s, c)) in x.iter().zip(means).zip(moments) {
                let d = dev(xj, *m, *lo);
                *c = ta * *c + ta * tb * d * dy;
                *s = a * *s + a * b * d * d;
                if step {
                    add(m, lo, b * d);
                }
            }
        }
        None => {
            for ((&xj, (m, lo)), s) in x.iter().zip(means).zip(sxx.iter_mut()) {
                let d = dev(xj, *m, *lo);
                *s = a * *s + a * b * d * d;
                if step {
                    add(m, lo, b * d);
                }
            }
        }
    }
}

/// The most rows a batch holds, and the bytes of features it may hold:
/// enough rows that a fork-join is a small part of a flush (one per 64 rows
/// ran within 6% of one per 256 at nine targets, and 29% short of it at one,
/// docs/PERFORMANCE.md §25), few enough that a group's batch stays small
/// beside its model.
const BATCH_ROWS: usize = 256;
const BATCH_BYTES: usize = 8 << 20;

/// Rows a batch of `p` features holds before it is flushed.
fn batch_rows(p: usize) -> usize {
    (BATCH_BYTES / (8 * p.max(1))).clamp(1, BATCH_ROWS)
}

/// A pair's moments on one row, in nanoseconds on one thread, and what
/// each part a spec adds costs beside them in units of that: a lagged
/// moment (a lag's autocovariance, or one of a cross lag's two terms) and
/// the bins' cell. Fitted to `marg_shard_bench` on an M4 Pro, within a
/// fifth on the eleven shapes of docs/PERFORMANCE.md §25; only their order
/// of magnitude matters to what they decide.
const PAIR_NS: f64 = 0.6;
const LAG_UNITS: f64 = 0.4;
const BIN_UNITS: f64 = 9.0;

/// The pair work a shard of a flush must have for the fork-join to be a
/// small part of it, in nanoseconds (docs/PERFORMANCE.md §25).
const SHARD_NS: f64 = 100_000.0;

impl MarginalCfg {
    /// `"auto"`'s shard count on a pool of `threads` (docs/PLAN.md task
    /// 126): as many as a flush's pair work keeps busy at a tenth of a
    /// millisecond each, and up to twice the threads, since more ranges
    /// than threads lets the fast cores take the slow ones' share. A row
    /// with too little work for two shards is not split. The flush's work
    /// is its rows -- the batch's, or under a window the snapshot cadence's
    /// where that is shorter -- times the row's pairs times each pair's
    /// cost, from the moments, the lagged moments and the bins.
    pub fn auto_shards(&self, threads: usize) -> usize {
        let (p, t) = (self.n_features, self.n_targets);
        let lags = self.lags.len();
        let cross = self.cross_lags.as_ref().map_or(lags, Vec::len);
        let bins = if self.bins.is_some() { BIN_UNITS } else { 0.0 };
        let per_pair = PAIR_NS * (1.0 + LAG_UNITS * (lags + 2 * cross) as f64 + bins);
        // A windowed row flushes before every snapshot, so a flush holds at
        // most the snapshot cadence's rows: every row at the default, which
        // no width can keep busy (review 2026-09-26, A1).
        let rows = match self.window {
            Some(_) => batch_rows(p).min(self.window_every.unwrap_or(1)),
            None => batch_rows(p),
        };
        let flush = rows as f64 * (p * t) as f64 * per_pair;
        let fits = (flush / SHARD_NS).floor() as usize;
        let most = if threads <= 1 { 1 } else { 2 * threads };
        if fits < 2 { 1 } else { fits.min(most) }
    }
}

/// What runs a flush's shards: every one of them, in any order and on any
/// threads ([`Shards::run`]). A closure may borrow what it counts with for
/// `'r`; a bare `dyn` alias would have asked for `'static`.
pub type ShardRunner<'r> = dyn Fn(&mut [MarginalShard<'_>]) + Sync + 'r;

/// How [`Marginal::step_sharded`] splits the pair work, and what runs the
/// parts.
#[derive(Clone, Copy)]
pub struct Shards<'r> {
    /// Ranges of features to split the pairs into, each a [`MarginalShard`];
    /// one or fewer steps the pairs row by row, unsplit.
    pub count: usize,
    /// Runs every shard of a flush, in any order and on any threads: each
    /// writes its own features' cells and nothing else, so the order cannot
    /// change a number.
    pub run: &'r ShardRunner<'r>,
}

impl Shards<'static> {
    /// One shard, run on this thread: what [`OnlineModel::step`] uses to
    /// learn rows a sharded step left held.
    pub fn inline() -> Self {
        Self {
            count: 1,
            run: &run_in_order,
        }
    }
}

/// Run each shard in turn, on this thread.
pub fn run_in_order(shards: &mut [MarginalShard<'_>]) {
    shards.iter_mut().for_each(MarginalShard::run);
}

/// What every shard of a flush reads: the held rows, the ring as it stood
/// when they began, and how the lags and the bins are laid out.
struct Job<'a> {
    p: usize,
    n_targets: usize,
    /// [`FeatureMomentLayout::Shared`]: each shard holds one block of feature
    /// moments, where it holds one per target otherwise.
    shared: bool,
    d: &'a Deferred,
    ring_x: Option<&'a std::collections::VecDeque<Vec<f64>>>,
    /// Per lag, its place among the cross lags, `None` where it keeps none.
    cross_of: &'a [Option<usize>],
    /// The bins' edges and feature offsets, where a held row writes a cell.
    bins: Option<(&'a [Vec<f64>], &'a [usize])>,
}

impl Job<'_> {
    /// The features of the row `src` names.
    fn row_x(&self, src: Src) -> &[f64] {
        match src {
            Src::Ring(i) => &self
                .ring_x
                .expect("a ring row is read only where there are lags")[i],
            Src::Row(r) => &self.d.xs[r * self.p..(r + 1) * self.p],
        }
    }
}

/// One part of a flush ([`Marginal::step_sharded`]): a range of features,
/// with every target's cells for them in every per-pair vector, stepped
/// through every held row in order by [`Self::run`].
pub struct MarginalShard<'a> {
    j: std::ops::Range<usize>,
    job: &'a Job<'a>,
    /// Per target, this range of its means, their low parts and its two
    /// co-moments.
    mx: Vec<&'a mut [f64]>,
    mx_lo: Vec<&'a mut [f64]>,
    sxx: Vec<&'a mut [f64]>,
    sxy: Vec<&'a mut [f64]>,
    /// Per target, this range of its runs, where a window keeps them.
    runs: Vec<&'a mut [f64]>,
    runs_start: Vec<&'a mut [u64]>,
    /// `[lag · T + t]` and `[cross lag · T + t]`: this range of each lagged
    /// moment.
    cxx: Vec<&'a mut [f64]>,
    cxy: Vec<&'a mut [f64]>,
    cyx: Vec<&'a mut [f64]>,
    /// Per target, this range's cells in the bins.
    w: Vec<&'a mut [f64]>,
    mean: Vec<&'a mut [f64]>,
    m2: Vec<&'a mut [f64]>,
    mean_lo: Vec<&'a mut [f64]>,
    /// Whether a cell was written.
    wrote: bool,
}

impl<'a> MarginalShard<'a> {
    fn new(j: std::ops::Range<usize>, job: &'a Job<'a>) -> Self {
        Self {
            j,
            job,
            mx: Vec::new(),
            mx_lo: Vec::new(),
            sxx: Vec::new(),
            sxy: Vec::new(),
            runs: Vec::new(),
            runs_start: Vec::new(),
            cxx: Vec::new(),
            cxy: Vec::new(),
            cyx: Vec::new(),
            w: Vec::new(),
            mean: Vec::new(),
            m2: Vec::new(),
            mean_lo: Vec::new(),
            wrote: false,
        }
    }

    /// The features this shard steps.
    pub fn features(&self) -> std::ops::Range<usize> {
        self.j.clone()
    }

    /// Step this range's pairs through every held row, in order: per row,
    /// the lags against the means before it, then the bins, then the runs
    /// and the pair moments -- `step`'s order, so the same numbers.
    pub fn run(&mut self) {
        use crate::marglag::{Lagged, step_all, step_cross, step_xx, wait};
        let job = self.job;
        let d = job.d;
        let (p, n_targets, n_lags) = (job.p, job.n_targets, job.cross_of.len());
        let (j0, j1) = (self.j.start, self.j.end);
        let mut at = Vec::new();
        let mut dx = Vec::with_capacity(if job.shared { j1 - j0 } else { 0 });
        for r in 0..d.n {
            let x = &d.xs[r * p + j0..r * p + j1];
            let mixes = &d.mixes[r * n_targets..(r + 1) * n_targets];
            if n_lags > 0 && job.shared {
                // One block of feature moments: the feature's autocovariance
                // at each lag once, with the row's mix, and each present
                // target's cross terms with its own, as
                // `MarginalLags::update_shared` steps them.
                let backs = &d.backs[r * n_lags..(r + 1) * n_lags];
                let row = d.shared[r];
                for (li, (&back, &ci)) in backs.iter().zip(job.cross_of).enumerate() {
                    let Some(src) = back else {
                        // Nothing that far back yet: the moments age and wait.
                        if let Some((a, _)) = row {
                            wait(a, self.cxx[li]);
                        }
                        for (t, mix) in mixes.iter().enumerate() {
                            if let (Some(m), Some(ci)) = (*mix, ci) {
                                wait(m.a, self.cxy[ci * n_targets + t]);
                                wait(m.a, self.cyx[ci * n_targets + t]);
                            }
                        }
                        continue;
                    };
                    let x_lag = &job.row_x(src)[j0..j1];
                    if let Some((a, b)) = row {
                        let lagged = Lagged {
                            a,
                            b,
                            dy_now: 0.0,
                            dy_lag: None,
                            x,
                            x_lag,
                            mx: self.mx[0],
                            mx_lo: self.mx_lo[0],
                        };
                        step_xx(lagged, self.cxx[li]);
                    }
                    let Some(ci) = ci else {
                        continue;
                    };
                    for (t, mix) in mixes.iter().enumerate() {
                        let Some(m) = *mix else {
                            continue;
                        };
                        let lagged = Lagged {
                            a: m.a,
                            b: m.b,
                            dy_now: m.dy,
                            dy_lag: d.dy_lag[(r * n_targets + t) * n_lags + li],
                            x,
                            x_lag,
                            mx: self.mx[0],
                            mx_lo: self.mx_lo[0],
                        };
                        step_cross(
                            lagged,
                            self.cxy[ci * n_targets + t],
                            self.cyx[ci * n_targets + t],
                        );
                    }
                }
            } else if n_lags > 0 {
                let backs = &d.backs[r * n_lags..(r + 1) * n_lags];
                for (t, mix) in mixes.iter().enumerate() {
                    let Some(m) = *mix else {
                        continue;
                    };
                    for (li, (&back, &ci)) in backs.iter().zip(job.cross_of).enumerate() {
                        let xx = li * n_targets + t;
                        let Some(src) = back else {
                            // Nothing that far back yet: the moments age
                            // and wait.
                            wait(m.a, self.cxx[xx]);
                            if let Some(ci) = ci {
                                wait(m.a, self.cxy[ci * n_targets + t]);
                                wait(m.a, self.cyx[ci * n_targets + t]);
                            }
                            continue;
                        };
                        let lagged = Lagged {
                            a: m.a,
                            b: m.b,
                            dy_now: m.dy,
                            dy_lag: d.dy_lag[(r * n_targets + t) * n_lags + li],
                            x,
                            x_lag: &job.row_x(src)[j0..j1],
                            mx: self.mx[t],
                            mx_lo: self.mx_lo[t],
                        };
                        match ci {
                            Some(ci) => step_all(
                                lagged,
                                self.cxx[xx],
                                self.cxy[ci * n_targets + t],
                                self.cyx[ci * n_targets + t],
                            ),
                            None => step_xx(lagged, self.cxx[xx]),
                        }
                    }
                }
            }
            if let (Some(u), Some((edges, off))) = (d.u[r], job.bins) {
                // Each feature's bin once a row, then every target's cells.
                crate::margbins::bin_offsets_in(edges, off, j0, x, &mut at);
                for t in 0..n_targets {
                    let Some(v) = d.ys[r * n_targets + t].filter(|v| v.is_finite()) else {
                        continue;
                    };
                    self.wrote |= crate::margbins::update_cells(
                        u,
                        v,
                        &at,
                        self.w[t],
                        self.mean[t],
                        self.m2[t],
                        self.mean_lo[t],
                    );
                }
            }
            if job.shared {
                // One block of feature moments, and no runs: `"shared"`
                // takes no window.
                pair_kernel_shared(
                    d.shared[r],
                    mixes,
                    x,
                    self.mx[0],
                    self.mx_lo[0],
                    self.sxx[0],
                    &mut self.sxy,
                    &mut dx,
                );
                continue;
            }
            for (t, mix) in mixes.iter().enumerate() {
                let Some(m) = *mix else {
                    continue;
                };
                if let (Some(row), Some(values)) = (m.runs_row, self.runs.get_mut(t)) {
                    crate::runs::track_slots(values, self.runs_start[t], x, row);
                }
                pair_kernel(m, x, self.mx[t], self.mx_lo[t], self.sxx[t], self.sxy[t]);
            }
        }
    }
}

/// Deal `v`, laid out `[t · width + i]`, into shards: shard `k` takes
/// `v[t·width + at[k] .. t·width + at[k+1]]` of every block `t`, in block
/// order. `at` runs from 0 to `width`.
fn deal<'a, E>(
    v: &'a mut [E],
    width: usize,
    at: &[usize],
    mut take: impl FnMut(usize, &'a mut [E]),
) {
    debug_assert!(at.first() == Some(&0) && at.last() == Some(&width));
    if width == 0 {
        return;
    }
    for block in v.chunks_exact_mut(width) {
        let mut rest = block;
        for (k, c) in at.windows(2).enumerate() {
            let (piece, tail) = std::mem::take(&mut rest).split_at_mut(c[1] - c[0]);
            take(k, piece);
            rest = tail;
        }
    }
}

impl OnlineModel for Marginal {
    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, usize)> {
        self.win.as_ref().and_then(|win| win.snaps.over_budget())
    }

    /// The ring's shadow, its snapshots formed as the step forms them
    /// (docs/PLAN.md task 115 (d)).
    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        let win = self.win.as_ref()?;
        Some(crate::WindowShadow::new(win.clock, &win.snaps, || {
            MarginalMoments {
                w_sum: self.w_sum,
                wt: self.wt.clone(),
                qt: self.qt.clone(),
                my: self.my.clone(),
                syy: self.syy.clone(),
                mx: self.mx.clone(),
                sxx: self.sxx.clone(),
                sxy: self.sxy.clone(),
                rows: Some(self.rows_t.clone()),
                lag: self.lag.as_ref().map(|l| l.moments()),
            }
        }))
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // Rows held for a sharded flush were learned first: this row
        // follows them (`Marginal::step_sharded`).
        if self.defer.n > 0 {
            self.flush(&Shards::inline());
        }
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        self.offer_snapshot(d_clock, lam);
        if self.lag.is_some() {
            self.learn_lags(x, y, lam, weight);
        }
        if self.bins.is_some() {
            self.feed_bins(x, y, lam, weight);
        }
        self.learn(x, y, lam, weight);
        // The ring holds rows that *taught* something, so a later row can be
        // `l` of them back. A zero-weight row aged the moments above and is
        // deliberately not pushed.
        //
        // The cheap test is on the outside: `x.iter().all(...)` walks every
        // feature, and a stream with no lags must not pay for a ring it does
        // not have.
        if let Some(lag) = self.lag.as_mut() {
            if weight > 0.0 && weight.is_finite() && x.iter().all(|v| v.is_finite()) {
                lag.push(x, y);
            }
        }
        out
    }

    /// No prediction slots: the step reports `n_eff` and nothing else --
    /// under a `window`, the weight inside it, which is what the pairs are
    /// gated on. This reported `w_sum`, the whole history, so a windowed
    /// `marginal` emitted a count that rose for the life of the stream beside
    /// pairs read from the window (review 2026-09-12, S17).
    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        Step {
            pred: Vec::new(),
            n_eff: self.n_eff(),
            extra: None,
        }
    }

    /// A session change or a clock gap beyond `gap_cap` means the next
    /// row is not one learned row after the last one, so the ring cannot say
    /// what `l` rows ago was. The moments stay; only the ring empties.
    ///
    /// The bin warm-up hold stays too, and so does the histogram. Nothing in
    /// either depends on rows being adjacent: a gap does not make the values
    /// a feature took any less representative of the values it takes.
    fn clear_lags(&mut self) {
        // Rows held for a sharded flush may still read the ring as it
        // stands: only the ring those rows will leave empties.
        if self.defer.n > 0 {
            self.defer.ring.clear();
            return;
        }
        if let Some(lag) = self.lag.as_mut() {
            lag.clear();
        }
    }

    fn state(&self) -> State {
        // Loud in every build: `defer` is not in the state, so a state read
        // over held rows could never learn them (review 2026-09-26, A3).
        assert_eq!(
            self.defer.n, 0,
            "marginal: rows held for a sharded flush are flushed before the state is read"
        );
        State::new(crate::ModelState::Marginal(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Marginal(m) => {
                let m = (**m).clone();
                let (p, t) = (m.cfg.n_features, m.cfg.n_targets);
                // Every per-pair vector at `p·t`, every per-target one at
                // `t`, and the boxed parts exactly as the cfg asks (review
                // 2026-09-18, B3).
                let lag_ok = match &m.lag {
                    Some(l) => {
                        l.has_shape(p, t, &m.cfg.lags, m.cfg.cross_lags.as_deref())
                            && l.is_shared() == m.cfg.shared()
                    }
                    None => m.cfg.lags.is_empty(),
                };
                let bins_ok = match (&m.bins, &m.cfg.bins) {
                    (Some(b), Some(_)) => b.has_shape(p, t),
                    (None, None) => true,
                    _ => false,
                };
                // Every snapshot at the same widths, its lag moments those of
                // the lags the model keeps: a narrower one indexed out of
                // bounds when a pair was read (B3).
                let fx = m.cfg.feature_len();
                let snaps_ok = m.win.as_ref().is_none_or(|w| {
                    w.snaps.iter().all(|s| {
                        [&s.wt, &s.qt, &s.my, &s.syy].iter().all(|v| v.len() == t)
                            && [&s.mx, &s.sxx].iter().all(|v| v.len() == fx)
                            && s.sxy.len() == p * t
                            && match (&s.lag, &m.lag) {
                                (Some(sl), Some(l)) => sl.matches(l),
                                (Some(_), None) => false,
                                (None, _) => true,
                            }
                    })
                });
                if [&m.wt, &m.qt, &m.my, &m.syy].iter().any(|v| v.len() != t)
                    || [&m.mx, &m.sxx].iter().any(|v| v.len() != fx)
                    || m.sxy.len() != p * t
                    || m.cfg.min_weight.len() != t
                    || !lag_ok
                    || !bins_ok
                    || !snaps_ok
                    || m.win.is_some() != m.cfg.window.is_some()
                {
                    return Err(StateError::Invalid(
                        "marginal: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "marginal",
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

    /// Zero: the model predicts nothing per row. Its outputs are read from
    /// the state with [`Marginal::pair`].
    fn n_outputs(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests;
