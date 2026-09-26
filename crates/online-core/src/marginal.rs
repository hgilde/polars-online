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
//! Kish's effective sample size -- `(1+lam)/(1−lam)` in the limit for unit
//! weights, about twice `n_eff`, and the `n` a standard error wants; the
//! moments; and from them `corr = S_xy / √(S_xx·S_yy)`, `beta = S_xy /
//! S_xx` (the slope of `y` on `x`) and `t = corr·√((n_kish − 2) / (1 −
//! corr²))`, the t-statistic of the correlation at Kish's `n`. `corr`, `beta`
//! and `t` are NaN while `W_t < min_periods`; the moments are always
//! reported. The t is descriptive: the rows of a stream are rarely
//! independent, and nothing here pretends otherwise.

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
    pub min_periods: Vec<f64>,
    /// Lags to accumulate pair moments at (docs/ENHANCEMENTS.md E66),
    /// strictly increasing and `>= 1`, counted in **learned rows within the
    /// group** — not in rows where a particular target was present, since the
    /// ring is shared. Empty for none, which is what every state written
    /// before E66 has.
    #[serde(default)]
    pub lags: Vec<usize>,
    /// How `n_serial` is formed from the lags that were kept: `"truncated"`
    /// sums them as they are (a lower bound on the correction, so an upper
    /// bound on `n_serial`), `"geometric"` fits one decay per series and
    /// extrapolates the tail in closed form. `None` keeps the lag moments and
    /// derives nothing.
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
    /// Clock units of history the pairs are computed from, with a **hard**
    /// cutoff: a row older than this contributes nothing (docs/PLAN.md §13).
    /// Inside the window the weights are still exponential.
    ///
    /// **Last, with `window_every`, and they must stay last**: the compact
    /// msgpack encoding writes a struct as an array, so a
    /// `skip_serializing_if` field anywhere else shifts what follows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<f64>,
    /// Learned rows between the window's snapshots.
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
}

impl MarginalCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("marginal: at least one feature is required".into());
        }
        if self.n_targets == 0 {
            return Err("marginal: at least one target is required".into());
        }
        if self.min_periods.len() != self.n_targets {
            return Err(format!(
                "marginal: min_periods has {} entries for {} targets",
                self.min_periods.len(),
                self.n_targets
            ));
        }
        // `inf` is a gate that never opens, as for every other model and as
        // the builders document; this model alone refused it (review
        // 2026-09-12, S27).
        if let Some(bad) = self.min_periods.iter().find(|v| v.is_nan() || **v < 0.0) {
            return Err(format!("marginal: min_periods must be >= 0, got {bad}"));
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
            return Err("marginal: window_every needs `window`".into());
        }
        // The lag ring keeps no snapshot, so under a window every lagged
        // co-moment was the whole history's, divided by the window's
        // variances: a hybrid of two histories in `lagcorr`, `n_serial` and
        // what is built from them. `ew_cov` refuses the pair for the same
        // reason (review 2026-09-12, C18).
        if self.window.is_some() && !self.lags.is_empty() {
            return Err(
                "marginal: window and lags cannot be combined. The lag ring keeps no snapshot, \
                 so a windowed lagcorr would divide the whole history's lagged co-moment by the \
                 window's variances. Use one or the other."
                    .into(),
            );
        }
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
    /// constant, or below `min_periods`.
    pub corr: f64,
    /// The slope of `y` on `x`, `cov / var_x`; NaN when the feature is
    /// constant, or below `min_periods`.
    pub beta: f64,
    /// `corr·√((n_kish − 2) / (1 − corr²))`; NaN when `n_kish <= 2`, or
    /// below `min_periods`. Enormous or `±inf` for a perfect correlation,
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
    /// sit at `corr = 0`. NaN without `bins`, below `min_periods`, or when
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
    /// of the pairs flushes first.
    #[serde(skip)]
    defer: Deferred,
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
        self.u.clear();
        self.backs.clear();
        self.dy_lag.clear();
        self.ring.clear();
    }
}

/// Bartlett's serial-dependence factor `1 + 2·Σ_l rho_x(l)·rho_y(l)`, and
/// the two fitted decays when the rule is geometric.
///
/// `"truncated"` sums the kept lags as they are. `"geometric"` fits
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
        SerialRule::Truncated => {
            let mut sum = 0.0;
            for (rx, ry) in rho_x.iter().zip(rho_y) {
                if rx.is_finite() && ry.is_finite() {
                    sum += rx * ry;
                }
            }
            // A factor at or below zero is an estimate outside the parameter
            // space -- two series whose autocorrelations have opposite signs
            // -- and says nothing, as `Geometric` does, where a floor at
            // `f64::MIN_POSITIVE` reported an infinite count (review
            // 2026-09-12, S18). Uniform weights on the lags do not keep the
            // sum positive; Bartlett's `1 − l/(L + 1)` would.
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
        hist.decay(self.pending_lam);
        self.pending_lam = 1.0;
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
        Ok(Self {
            cfg,
            w_sum: 0.0,
            wt: vec![0.0; t],
            qt: vec![0.0; t],
            my: vec![0.0; t],
            syy: vec![0.0; t],
            mx: vec![0.0; p * t],
            sxx: vec![0.0; p * t],
            sxy: vec![0.0; p * t],
            lag,
            bins,
            mx_lo: vec![0.0; p * t],
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
            defer: Deferred::default(),
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
            // Reserved at exactly the rows it will hold, which is what the
            // budget counts (`margbins::hold_bytes`); a hold restored from a
            // state is reserved again here.
            if b.held.capacity() < cfg.warm_rows {
                b.held.reserve_exact(cfg.warm_rows - b.held.len());
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
    /// `min_periods`.
    pub fn pair(&self, t: usize, j: usize) -> Pair {
        debug_assert_eq!(
            self.defer.n, 0,
            "marginal: rows held for a sharded flush are flushed before a pair is read"
        );
        let i = t * self.cfg.n_features + j;
        // With a `window`, every moment this pair is built from is truncated
        // to it first: the weight, the two means, and the three centred
        // second moments. Each is the same subtraction `EwCov` makes.
        let (n_eff, n_kish, mean_x, mean_y, var_x, var_y, cov) = match self.boundary() {
            None => (
                self.wt[t],
                self.wt[t] * self.wt[t] / self.qt[t],
                self.mx[i],
                self.my[t],
                self.sxx[i].max(0.0),
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
        let (corr, beta, t_stat) = if n_eff >= self.cfg.min_periods[t] {
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
            for li in 0..lag.lags().len() {
                lagcorr_xx.push(norm(lag.cxx(li, t, j), sd_x * sd_x));
                lagcorr_yy.push(norm(lag.cyy(li, t), sd_y * sd_y));
            }
            // Over the cross lags, which are every lag unless `cross_lags`
            // names fewer (E70).
            for ci in 0..lag.cross_lags().len() {
                lagcorr_xy.push(norm(lag.cxy(ci, t, j), sd_x * sd_y));
                lagcorr_yx.push(norm(lag.cyx(ci, t, j), sd_x * sd_y));
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
        // The binned view of the same pair. Gated by `min_periods` like the
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
            if n_eff >= self.cfg.min_periods[t] {
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
        self.w_sum = lam * self.w_sum + w;
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
        if let Some(lag) = self.lag.as_deref_mut() {
            let rows = d
                .ring
                .iter()
                .map(|src| match *src {
                    Src::Ring(i) => {
                        let (x, y) = lag.ring_row(i);
                        (x.to_vec(), y.to_vec())
                    }
                    Src::Row(r) => (
                        d.xs[r * p..(r + 1) * p].to_vec(),
                        d.ys[r * n_targets..(r + 1) * n_targets].to_vec(),
                    ),
                })
                .collect();
            lag.set_ring(rows);
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
            // zero-weight row is a clock gap past `max_dclock` on a row that
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

/// The most rows a batch holds, and the bytes of features it may hold:
/// enough rows that a fork-join is a small part of a flush (one per 64 rows
/// ran within 6% of one per 256, docs/PERFORMANCE.md §25), few enough that
/// a group's batch stays small beside its model.
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
    /// is its rows times the row's pairs times each pair's cost, from the
    /// moments, the lagged moments and the bins.
    pub fn auto_shards(&self, threads: usize) -> usize {
        let (p, t) = (self.n_features, self.n_targets);
        let lags = self.lags.len();
        let cross = self.cross_lags.as_ref().map_or(lags, Vec::len);
        let bins = if self.bins.is_some() { BIN_UNITS } else { 0.0 };
        let per_pair = PAIR_NS * (1.0 + LAG_UNITS * (lags + 2 * cross) as f64 + bins);
        let flush = batch_rows(p) as f64 * (p * t) as f64 * per_pair;
        let fits = (flush / SHARD_NS).floor() as usize;
        let most = if threads <= 1 { 1 } else { 2 * threads };
        if fits < 2 { 1 } else { fits.min(most) }
    }
}

/// What runs a flush's shards: every one of them, in any order and on any
/// threads ([`Shards::run`]).
pub type ShardRunner = dyn Fn(&mut [MarginalShard<'_>]) + Sync;

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
    pub run: &'r ShardRunner,
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
        use crate::marglag::{Lagged, step_all, step_xx, wait};
        let job = self.job;
        let d = job.d;
        let (p, n_targets, n_lags) = (job.p, job.n_targets, job.cross_of.len());
        let (j0, j1) = (self.j.start, self.j.end);
        let mut at = Vec::new();
        for r in 0..d.n {
            let x = &d.xs[r * p + j0..r * p + j1];
            let mixes = &d.mixes[r * n_targets..(r + 1) * n_targets];
            if n_lags > 0 {
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

    /// A session change or a clock gap beyond `max_dclock` means the next
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
        debug_assert_eq!(
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
                    Some(l) => l.has_shape(p, t, &m.cfg.lags, m.cfg.cross_lags.as_deref()),
                    None => m.cfg.lags.is_empty(),
                };
                let bins_ok = match (&m.bins, &m.cfg.bins) {
                    (Some(b), Some(_)) => b.has_shape(p, t),
                    (None, None) => true,
                    _ => false,
                };
                if [&m.wt, &m.qt, &m.my, &m.syy].iter().any(|v| v.len() != t)
                    || [&m.mx, &m.sxx, &m.sxy].iter().any(|v| v.len() != p * t)
                    || m.cfg.min_periods.len() != t
                    || !lag_ok
                    || !bins_ok
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
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Marginal::new(cfg(2, 1)).unwrap();
        let mut s = m.state();
        let ModelState::Marginal(inner) = &mut s.model else {
            unreachable!()
        };
        inner.mx.pop();
        match Marginal::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::{EwCov, EwCovCfg, EwCovModel, EwCovStat};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(p: usize, t: usize) -> MarginalCfg {
        MarginalCfg {
            n_features: p,
            n_targets: t,
            decay: Decay::Halflife(20.0),
            min_periods: vec![0.0; t],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins: None,
            window: None,
            window_every: None,
        }
    }

    /// A windowed marginal over one stream: feature 0 moves; feature 1 and
    /// the target are what `x1` and `y` give row `i`; `weight` and `present`
    /// say how each row is learned.
    fn windowed_pairs(
        window: f64,
        h: f64,
        n: usize,
        x1: impl Fn(usize, f64) -> f64,
        y: impl Fn(usize, f64, f64) -> Option<f64>,
        weight: impl Fn(usize) -> f64,
    ) -> Marginal {
        let mut c = cfg(2, 1);
        c.decay = Decay::Halflife(h);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..n {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.step(&[b, x1(i, a)], &[y(i, a, b)], d, weight(i));
        }
        m
    }

    /// PLAN task 94, for the pairs. A feature, or a target, that holds one
    /// value over every row inside a window has no spread there, and the
    /// windowed pair says so exactly: zero variance and covariance, the value
    /// as its mean, no correlation, and a slope of zero on a target that
    /// holds. The subtraction left a remainder that grows with the level and
    /// the rows since the boundary, and `beta` divided it by itself. Found by
    /// the sweep of every running mean for task 101. The slot that moves
    /// keeps its spread.
    #[test]
    fn a_slot_held_over_the_window_has_no_spread_there() {
        for level in [0.0, 1e3, 1e6, -1e8] {
            for (window, h) in [(0.5, 10.0), (9.0, 10.0), (30.0, 8.0), (199.0, 70.0)] {
                for before in [0usize, 3] {
                    for held_feature in [true, false] {
                        let n = 400 + window as usize;
                        let from = n - 1 - window as usize - before;
                        let m = windowed_pairs(
                            window,
                            h,
                            n,
                            |i, a| {
                                if held_feature && i >= from {
                                    level + 0.37
                                } else {
                                    level + 1e-3 * a
                                }
                            },
                            |i, a, b| {
                                Some(if !held_feature && i >= from {
                                    level + 0.25
                                } else {
                                    level + 1.0 + 0.5 * a + b
                                })
                            },
                            |_| 1.0,
                        );
                        let case = format!(
                            "level {level}, window {window}, h {h}, from {before} before, \
                             held feature {held_feature}"
                        );
                        let pair = m.pair(0, 1);
                        assert_eq!(pair.cov, 0.0, "{case}: covariance");
                        assert!(pair.corr.is_nan(), "{case}: correlation {}", pair.corr);
                        if held_feature {
                            assert_eq!(pair.var_x, 0.0, "{case}: variance");
                            assert_eq!(pair.mean_x, level + 0.37, "{case}: mean");
                            assert!(pair.beta.is_nan(), "{case}: slope {}", pair.beta);
                        } else {
                            assert_eq!(pair.var_y, 0.0, "{case}: variance");
                            assert_eq!(pair.mean_y, level + 0.25, "{case}: mean");
                            if window >= 1.0 {
                                assert_eq!(pair.beta, 0.0, "{case}: slope");
                            }
                        }
                        if window >= 1.0 {
                            let moving = m.pair(0, 0);
                            assert!(moving.var_x > 0.0, "{case}: the moving feature's spread");
                        }
                    }
                }
            }
        }
    }

    /// The window's own weight counts `EMPTY_FRACTION` of the live weight as
    /// nothing, and so may a run: a boundary row at a weight of 1e-12,
    /// carrying another value before the run begins, leaves the held
    /// feature with no spread inside the window.
    #[test]
    fn a_boundary_row_of_no_account_does_not_end_a_run() {
        let (window, n) = (9.0, 300);
        let boundary = n - 1 - window as usize;
        let m = windowed_pairs(
            window,
            10.0,
            n,
            |i, a| match i {
                _ if i == boundary => 1e3 - 5.0,
                _ if i > boundary => 1e3 + 0.37,
                _ => 1e3 + 1e-3 * a,
            },
            |_, a, b| Some(1.0 + 0.5 * a + b),
            |i| if i == boundary { 1e-12 } else { 1.0 },
        );
        let pair = m.pair(0, 1);
        assert_eq!(pair.var_x, 0.0, "a boundary row of no account");
        assert_eq!(pair.mean_x, 1e3 + 0.37);
    }

    /// A row without the target ages its runs as it ages its weight, so a run
    /// that began inside the window is not credited with weight it no longer
    /// carries: a feature that moved inside the window keeps its spread there
    /// when the target is present on every other row.
    #[test]
    fn a_row_without_the_target_ages_its_runs() {
        let (window, n) = (30.0, 400);
        let m = windowed_pairs(
            window,
            3.0,
            n,
            |i, a| if i >= n - 20 { 1e3 + 0.37 } else { 1e3 + a },
            |i, a, b| (i % 2 == 0).then_some(1.0 + 0.5 * a + b),
            |_| 1.0,
        );
        let pair = m.pair(0, 1);
        assert!(
            pair.var_x > 0.0,
            "the feature moved inside the window: {}",
            pair.var_x
        );
        assert!(pair.cov != 0.0);
    }

    /// A window that holds all but 2^-60 of the weight reads what no window
    /// reads, field for field, to the rounding of the subtraction: the
    /// weight, Kish's size, the means and the three second moments.
    #[test]
    fn a_window_holding_the_weight_reads_what_no_window_does() {
        let n = 400;
        let windowed = windowed_pairs(60.0, 1.0, n, |_, a| 3.0 + a, |_, a, b| Some(a - b), |_| 1.0);
        let mut c = cfg(2, 1);
        c.decay = Decay::Halflife(1.0);
        let mut plain = Marginal::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..n {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            plain.step(
                &[b, 3.0 + a],
                &[Some(a - b)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        for j in 0..2 {
            let (w, p) = (windowed.pair(0, j), plain.pair(0, j));
            for (what, got, want) in [
                ("n_eff", w.n_eff, p.n_eff),
                ("n_kish", w.n_kish, p.n_kish),
                ("mean_x", w.mean_x, p.mean_x),
                ("mean_y", w.mean_y, p.mean_y),
                ("var_x", w.var_x, p.var_x),
                ("var_y", w.var_y, p.var_y),
                ("cov", w.cov, p.cov),
            ] {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1e-300),
                    "feature {j}: {what} {got} windowed, {want} without"
                );
            }
        }
    }

    /// A row of weight 0 takes no step in a pair's means, to the bit: the
    /// rows 0.7 and 5.292162135665459 at unit weight and no decay leave each
    /// mean with a low part of a whole rounding step, where a zero step would
    /// round the double up (`crate::comp::add`).
    #[test]
    fn a_row_of_no_weight_leaves_the_means_as_they_were() {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY);
        let mut m = Marginal::new(c).unwrap();
        for v in [0.7, 5.292162135665459] {
            m.step(&[v], &[Some(v)], 1.0, 1.0);
        }
        assert_eq!(
            (m.mx[0], m.my[0]),
            (2.996081067832729, 2.996081067832729),
            "the fixture"
        );
        assert_eq!(
            m.mx_lo[0], 4.440892098500626e-16,
            "a whole step below the double"
        );
        m.step(&[1.0], &[Some(1.0)], 1.0, 0.0);
        assert_eq!((m.mx[0], m.my[0]), (2.996081067832729, 2.996081067832729));
        assert_eq!(
            (m.mx_lo[0], m.my_lo[0]),
            (4.440892098500626e-16, 4.440892098500626e-16)
        );
    }

    /// A row without a target ages that target's runs and no other's, over
    /// its own slots and no other's: two features held over a window, and
    /// two targets each present on half the rows, so that every pair reads
    /// its feature as held. A run aged on the wrong rows, or the wrong
    /// slots, falls short of the window's weight.
    #[test]
    fn a_row_without_a_target_ages_that_targets_runs_alone() {
        let (window, n) = (30.0, 400);
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(3.0);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 13u64;
        let from = n - 1 - window as usize - 3;
        for i in 0..n {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            let x = if i >= from {
                [1e3 + 0.37, 1e3 + 0.25]
            } else {
                [1e3 + a, 1e3 + b]
            };
            let y = [(i % 2 == 1).then_some(a - b), (i % 2 == 0).then_some(a + b)];
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        for t in 0..2 {
            for (j, held) in [1e3 + 0.37, 1e3 + 0.25].into_iter().enumerate() {
                let pair = m.pair(t, j);
                assert_eq!(pair.var_x, 0.0, "target {t}, feature {j}: variance");
                assert_eq!(pair.mean_x, held, "target {t}, feature {j}: mean");
            }
        }
    }

    /// A state written before the means' low parts, and before the runs,
    /// loads and steps: each starts at the size of what it belongs to.
    #[test]
    fn a_state_without_the_low_parts_loads_and_steps() {
        let mut c = cfg(2, 1);
        c.window = Some(9.0);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 3u64;
        for i in 0..40 {
            m.step(
                &[lcg(&mut s), lcg(&mut s)],
                &[Some(lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let mut old = serde_json::to_value(&m).unwrap();
        for key in ["mx_lo", "my_lo", "x_runs", "y_runs"] {
            assert!(old.as_object_mut().unwrap().remove(key).is_some(), "{key}");
        }
        let mut back: Marginal = serde_json::from_value(old).unwrap();
        assert!(back.mx_lo.is_empty() && back.my_lo.is_empty());
        for i in 0..40 {
            back.step(&[lcg(&mut s), lcg(&mut s)], &[Some(lcg(&mut s))], 1.0, 1.0);
            let p = back.pair(0, i % 2);
            assert!(p.mean_x.is_finite() && p.var_x.is_finite(), "row {i}");
        }
        assert_eq!((back.mx_lo.len(), back.my_lo.len()), (2, 1));
    }

    /// Under a window, Kish's size is `ew_cov`'s over the same rows: the
    /// same weights, the same `Sum w²`, and the same boundary, so the same
    /// subtraction.
    #[test]
    fn the_windowed_kish_size_is_ew_covs() {
        let (window, h) = (30.0, 10.0);
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(h);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();
        let mut ec = EwCovModel::new(EwCovCfg {
            n_features: 2,
            decay: Decay::Halflife(h),
            stats: vec![EwCovStat::Mean],
            min_periods: 0.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0,
            lags: Vec::new(),
            window: Some(window),
            window_every: None,
        })
        .unwrap();
        let mut s = 5u64;
        for i in 0..200 {
            let (x, y) = (lcg(&mut s), lcg(&mut s));
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
            crate::OnlineModel::step(&mut ec, &[x, y], &[], d, 1.0);
            if i >= 100 {
                let want = ec.windowed_cov().n_kish().unwrap();
                let got = m.pair(0, 0).n_kish;
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "row {i}: {got} against ew_cov's {want}"
                );
            }
        }
    }

    /// A stream with three features and three targets that reaches every
    /// lag event: serially dependent series, targets absent on a fifth of
    /// the rows, unequal weights with zeros, and the ring cleared once, as a
    /// session change clears it. The lags `[1, 2, 5, 10]`, with the cross
    /// terms kept where `cross_lags` says.
    fn cross_lag_run(cross_lags: Option<Vec<usize>>) -> Marginal {
        let mut c = cfg(3, 3);
        c.decay = Decay::Halflife(40.0);
        c.lags = vec![1, 2, 5, 10];
        c.serial_rule = Some(SerialRule::Geometric);
        c.cross_lags = cross_lags;
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 77u64;
        let (mut x, mut y) = ([0.0_f64; 3], [0.0_f64; 3]);
        for i in 0..600 {
            for v in x.iter_mut() {
                *v = 0.8 * *v + lcg(&mut seed);
            }
            let ys: Vec<Option<f64>> = (0..3)
                .map(|t| {
                    y[t] = 0.7 * y[t] + 0.5 * x[t] + lcg(&mut seed);
                    (lcg(&mut seed) > -0.6).then_some(y[t])
                })
                .collect();
            let w = if i % 11 == 4 {
                0.0
            } else {
                0.5 + (lcg(&mut seed) + 1.0) / 2.0
            };
            if i == 300 {
                OnlineModel::clear_lags(&mut m);
            }
            OnlineModel::step(&mut m, &x, &ys, step_clock(i), w);
        }
        m
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Each lagged correlation is its lagged covariance over the two
    /// contemporaneous standard deviations, the autocorrelations over one
    /// of them squared -- `ew_cov`'s expression -- to the bit, at every lag
    /// and every cross lag.
    #[test]
    fn a_lagged_correlation_is_its_covariance_over_the_two_deviations() {
        for cross in [None, Some(vec![2, 10])] {
            let m = cross_lag_run(cross.clone());
            let lag = m.lag.as_ref().unwrap();
            for t in 0..3 {
                for j in 0..3 {
                    let q = m.pair(t, j);
                    let (sx, sy) = (q.var_x.sqrt(), q.var_y.sqrt());
                    assert!(sx > 0.0 && sy > 0.0);
                    for li in 0..4 {
                        assert_eq!(
                            q.lagcorr_xx[li].to_bits(),
                            (lag.cxx(li, t, j) / (sx * sx)).to_bits()
                        );
                        assert_eq!(
                            q.lagcorr_yy[li].to_bits(),
                            (lag.cyy(li, t) / (sy * sy)).to_bits()
                        );
                    }
                    for ci in 0..lag.cross_lags().len() {
                        assert_eq!(
                            q.lagcorr_xy[ci].to_bits(),
                            (lag.cxy(ci, t, j) / (sx * sy)).to_bits()
                        );
                        assert_eq!(
                            q.lagcorr_yx[ci].to_bits(),
                            (lag.cyx(ci, t, j) / (sx * sy)).to_bits()
                        );
                    }
                }
            }
        }
    }

    /// A feature constant on every row its target was present on, but not on
    /// a row between them where the target was absent: the ring is shared,
    /// so the target now against the feature a row back sees that row, and
    /// the lagged covariance is not zero where the feature's spread is. Its
    /// correlation is undefined, NaN, not the infinity the division gives.
    #[test]
    fn a_lagged_correlation_over_no_spread_is_nan() {
        let mut c = cfg(1, 1);
        c.lags = vec![1];
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 3u64;
        for i in 0..40 {
            // Odd rows carry the target with the feature at 2; even rows
            // leave the target out and move the feature.
            let (x, y) = if i % 2 == 1 {
                (2.0, Some(lcg(&mut seed)))
            } else {
                (lcg(&mut seed), None)
            };
            OnlineModel::step(&mut m, &[x], &[y], step_clock(i), 1.0);
        }
        let q = m.pair(0, 0);
        let lag = m.lag.as_ref().unwrap();
        assert_eq!(q.var_x, 0.0, "the feature is constant on the target's rows");
        assert!(q.var_y > 0.0);
        assert!(lag.cyx(0, 0, 0) != 0.0, "the ring saw the rows between");
        assert!(q.lagcorr_yx[0].is_nan(), "{}", q.lagcorr_yx[0]);
        assert!(q.lagcorr_xx[0].is_nan() && q.lagcorr_xy[0].is_nan());
    }

    /// E70 (docs/PLAN.md task 123): `cross_lags` chooses which cross terms
    /// are kept and touches nothing else. The autocorrelations, `n_serial`,
    /// `t_serial` and the fitted decays are the same to the bit whether the
    /// cross terms are kept at every lag, at some, or at none.
    #[test]
    fn cross_lags_leave_the_serial_correction_to_the_bit() {
        let all = cross_lag_run(None);
        let mut corrected = 0;
        for cross in [vec![1], vec![], vec![2, 10]] {
            let some = cross_lag_run(Some(cross.clone()));
            for t in 0..3 {
                for j in 0..3 {
                    let (a, b) = (all.pair(t, j), some.pair(t, j));
                    let why = format!("cross_lags {cross:?}, target {t}, feature {j}");
                    assert_eq!(bits(&a.lagcorr_xx), bits(&b.lagcorr_xx), "{why}");
                    assert_eq!(bits(&a.lagcorr_yy), bits(&b.lagcorr_yy), "{why}");
                    let serial = |q: &Pair| [q.n_serial, q.t_serial, q.phi_x, q.phi_y, q.corr, q.t];
                    assert_eq!(bits(&serial(&a)), bits(&serial(&b)), "{why}");
                    assert_eq!(a.lagcorr_xy.len(), 4, "{why}: every lag by default");
                    assert_eq!(b.lagcorr_xy.len(), cross.len(), "{why}");
                    assert_eq!(b.lagcorr_yx.len(), cross.len(), "{why}");
                    corrected += usize::from(b.n_serial.is_finite());
                }
            }
        }
        // The correction was computed, not NaN on both sides alike.
        assert_eq!(corrected, 27, "every pair's n_serial is a number");
    }

    /// ... and the cross terms it keeps are the default's at those lags, to
    /// the bit: the accumulators, not only the correlations read from them.
    /// Every lag named outright takes the general path and the default its
    /// own loop (`MarginalLags::update_target`), so this is also what holds
    /// the two loops to the same bits.
    #[test]
    fn cross_lags_keep_the_default_cross_terms_at_their_lags() {
        let all = cross_lag_run(None);
        let la = all.lag.as_ref().unwrap();
        for cross in [vec![1, 5], vec![1, 2, 5, 10]] {
            let some = cross_lag_run(Some(cross.clone()));
            let ls = some.lag.as_ref().unwrap();
            assert_eq!(ls.cross_lags(), cross.as_slice());
            let at: Vec<usize> = cross
                .iter()
                .map(|l| [1, 2, 5, 10].iter().position(|m| m == l).unwrap())
                .collect();
            for t in 0..3 {
                for j in 0..3 {
                    for (ci, &li) in at.iter().enumerate() {
                        assert_eq!(ls.cxy(ci, t, j).to_bits(), la.cxy(li, t, j).to_bits());
                        assert_eq!(ls.cyx(ci, t, j).to_bits(), la.cyx(li, t, j).to_bits());
                    }
                    for li in 0..4 {
                        assert_eq!(ls.cxx(li, t, j).to_bits(), la.cxx(li, t, j).to_bits());
                        assert_eq!(ls.cyy(li, t).to_bits(), la.cyy(li, t).to_bits());
                    }
                    let (a, b) = (all.pair(t, j), some.pair(t, j));
                    let pick = |v: &[f64]| at.iter().map(|&li| v[li]).collect::<Vec<_>>();
                    assert_eq!(bits(&b.lagcorr_xy), bits(&pick(&a.lagcorr_xy)));
                    assert_eq!(bits(&b.lagcorr_yx), bits(&pick(&a.lagcorr_yx)));
                }
            }
        }
    }

    #[test]
    fn a_bad_cross_lags_is_refused_by_name() {
        for (lags, cross, msg) in [
            (
                vec![1, 2, 5],
                vec![3],
                "cross_lags must each be one of lags, and 3 is not",
            ),
            (
                vec![1, 2, 5],
                vec![1, 0],
                "cross_lags must be strictly increasing",
            ),
            (
                vec![1, 2, 5],
                vec![5, 1],
                "cross_lags must be strictly increasing",
            ),
            (
                vec![1, 2, 5],
                vec![2, 2],
                "cross_lags must be strictly increasing",
            ),
            (
                vec![1],
                vec![0],
                "cross_lags must each be one of lags, and 0 is not",
            ),
            (vec![], vec![1], "cross_lags needs `lags`"),
            (vec![], vec![], "cross_lags needs `lags`"),
        ] {
            let mut c = cfg(1, 1);
            c.lags = lags.clone();
            c.cross_lags = Some(cross.clone());
            let err = Marginal::new(c).unwrap_err();
            assert!(
                err.contains(msg),
                "lags {lags:?}, cross_lags {cross:?}: {err}"
            );
        }
    }

    /// A state written before E70 has no `cross_lags`, in the config or in
    /// the lag moments. It loads as cross terms at every lag, which is what
    /// it holds, and learns on exactly as the state that wrote it.
    #[test]
    fn a_state_without_cross_lags_loads_with_every_lag() {
        let mut m = cross_lag_run(None);
        let mut old = serde_json::to_value(&m).unwrap();
        assert!(
            old["cfg"]
                .as_object_mut()
                .unwrap()
                .remove("cross_lags")
                .is_some()
        );
        assert!(
            old["lag"]
                .as_object_mut()
                .unwrap()
                .remove("cross_lags")
                .is_some()
        );
        let mut back: Marginal = serde_json::from_value(old).unwrap();
        assert_eq!(back, m);
        let mut seed = 5u64;
        for i in 0..50 {
            let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
            let y = [Some(lcg(&mut seed)), None, Some(lcg(&mut seed))];
            OnlineModel::step(&mut m, &x, &y, step_clock(i + 1), 1.0);
            OnlineModel::step(&mut back, &x, &y, step_clock(i + 1), 1.0);
        }
        assert_eq!(back, m);
        assert_eq!(back.pair(2, 1).lagcorr_xy.len(), 4);
    }

    /// E66 test 1: the lagged moments are `ew_cov(lags=)`'s, to the bit. Both
    /// centre each leg at the pre-row mean and mix with the same `a`/`b`, so
    /// there is no room for them to differ -- and if the two ever drift, one
    /// of the two recursions has been changed without the other.
    #[test]
    fn lagged_pair_moments_are_ew_covs_to_the_bit() {
        let lags = vec![1usize, 3, 7];
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(30.0);
        c.lags = lags.clone();
        let mut m = Marginal::new(c).unwrap();

        // `ew_cov` over the same two columns, in the order [x, y]: its lagged
        // co-moment for (0, 1) is `E[dx_t·dy_{t-l}]`, which is `cxy`.
        let mut ec = EwCovModel::new(EwCovCfg {
            n_features: 2,
            decay: Decay::Halflife(30.0),
            stats: vec![EwCovStat::Mean],
            min_periods: 0.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0,
            lags: lags.clone(),
            window: None,
            window_every: None,
        })
        .unwrap();

        let mut seed = 21u64;
        for i in 0..80 {
            let x = lcg(&mut seed) * 2.0;
            let y = 0.6 * x + lcg(&mut seed);
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
            crate::OnlineModel::step(&mut ec, &[x, y], &[], d, 1.0);
        }
        let pair = m.pair(0, 0);
        let want = ec.lag().expect("ew_cov has lags");
        for (li, _) in lags.iter().enumerate() {
            // Covariances, not the correlations the pair reports: compare the
            // accumulators themselves so the normalization cannot hide a
            // difference.
            let lag = m.lag.as_ref().unwrap();
            assert_eq!(
                lag.cxx(li, 0, 0).to_bits(),
                want.get(li, 0, 0).to_bits(),
                "lag {li}: the feature's own autocovariance"
            );
            assert_eq!(
                lag.cyy(li, 0).to_bits(),
                want.get(li, 1, 1).to_bits(),
                "lag {li}: the target's"
            );
            assert_eq!(
                lag.cxy(li, 0, 0).to_bits(),
                want.get(li, 0, 1).to_bits(),
                "lag {li}: x now against y back"
            );
            assert_eq!(
                lag.cyx(li, 0, 0).to_bits(),
                want.get(li, 1, 0).to_bits(),
                "lag {li}: y now against x back"
            );
        }
        assert_eq!(pair.lagcorr_xx.len(), lags.len());
    }

    /// The identity above for every feature of every target: two of each,
    /// so a slot read as `t·p + j` is one of four, and `ew_cov` over the
    /// four columns `[x0, x1, y0, y1]` has each lagged co-moment by index.
    #[test]
    fn lagged_pair_moments_are_ew_covs_for_every_feature_and_target() {
        let lags = vec![1usize, 3];
        let (p, nt) = (2usize, 2usize);
        let mut c = cfg(p, nt);
        c.decay = Decay::Halflife(30.0);
        c.lags = lags.clone();
        let mut m = Marginal::new(c).unwrap();
        let mut ec = EwCovModel::new(EwCovCfg {
            n_features: p + nt,
            decay: Decay::Halflife(30.0),
            stats: vec![EwCovStat::Mean],
            min_periods: 0.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0,
            lags: lags.clone(),
            window: None,
            window_every: None,
        })
        .unwrap();
        let mut seed = 23u64;
        for i in 0..80 {
            let x0 = lcg(&mut seed) * 2.0;
            let x1 = lcg(&mut seed) + 0.3 * x0;
            let y0 = 0.6 * x0 + lcg(&mut seed);
            let y1 = -0.4 * x1 + lcg(&mut seed);
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &[x0, x1], &[Some(y0), Some(y1)], d, 1.0);
            crate::OnlineModel::step(&mut ec, &[x0, x1, y0, y1], &[], d, 1.0);
        }
        let lag = m.lag.as_ref().unwrap();
        let want = ec.lag().expect("ew_cov has lags");
        for li in 0..lags.len() {
            for t in 0..nt {
                assert_eq!(
                    lag.cyy(li, t).to_bits(),
                    want.get(li, p + t, p + t).to_bits(),
                    "lag {li}, target {t}: the target's own"
                );
                for j in 0..p {
                    assert_eq!(
                        lag.cxx(li, t, j).to_bits(),
                        want.get(li, j, j).to_bits(),
                        "lag {li}, target {t}, feature {j}: the feature's own"
                    );
                    assert_eq!(
                        lag.cxy(li, t, j).to_bits(),
                        want.get(li, j, p + t).to_bits(),
                        "lag {li}, target {t}, feature {j}: x now against y back"
                    );
                    assert_eq!(
                        lag.cyx(li, t, j).to_bits(),
                        want.get(li, p + t, j).to_bits(),
                        "lag {li}, target {t}, feature {j}: y now against x back"
                    );
                }
            }
        }
    }

    /// The runs are a window's, and a model without one keeps none: every
    /// slot's run stays unstarted, where under a window each slot's run
    /// starts at its first learned row (docs/PERFORMANCE.md §24).
    #[test]
    fn only_a_window_keeps_the_runs() {
        for window in [None, Some(50.0)] {
            let mut c = cfg(2, 2);
            c.window = window;
            let mut m = Marginal::new(c).unwrap();
            let mut s = 9u64;
            for i in 0..10 {
                let x = [lcg(&mut s), 0.25];
                m.step(&x, &[Some(lcg(&mut s)), Some(1.0)], step_clock(i), 1.0);
            }
            assert_eq!(m.rows_t, vec![10, 10], "the counts are kept either way");
            // Not even allocated without one (docs/PLAN.md task 128).
            assert_eq!(m.x_runs.is_off(), window.is_none());
            assert_eq!(m.y_runs.is_off(), window.is_none());
            let started = |r: &crate::Runs, k: usize| {
                (0..k)
                    .filter(|&i| r.started_by(k, i, u64::MAX).is_some())
                    .count()
            };
            let (xs, ys) = (started(&m.x_runs, 4), started(&m.y_runs, 2));
            if window.is_some() {
                assert_eq!((xs, ys), (4, 2), "every slot has a run under a window");
                assert_eq!(m.x_runs.started_by(4, 1, 1), Some(0.25), "the held feature");
            } else {
                assert_eq!((xs, ys), (0, 0), "no run without one");
            }
        }
    }

    /// Each target counts its own learned rows, and a row of weight 0 or of
    /// no account counts for none (`EwCov::update`).
    #[test]
    fn each_target_counts_its_learned_rows() {
        let mut m = Marginal::new(cfg(1, 2)).unwrap();
        for i in 0..20 {
            let y0 = (i % 2 == 0).then_some(1.0);
            let w = if i % 5 == 4 { 0.0 } else { 1.0 };
            m.step(&[0.5], &[y0, Some(2.0)], if i == 0 { 0.0 } else { 1.0 }, w);
        }
        assert_eq!(m.rows_t, vec![8, 16]);
        // A row at exactly the fraction of target 0's decayed weight is
        // nothing to it, and below the fraction of target 1's, heavier.
        let lam = m.cfg.decay.factor(1.0);
        let nothing = crate::window::EMPTY_FRACTION * (lam * m.wt[0]);
        m.step(&[0.5], &[Some(1.0), Some(2.0)], 1.0, nothing);
        assert_eq!(m.rows_t, vec![8, 16], "a row at the fraction is nothing");
        // Just above it, the row counts for target 0 alone.
        m.step(&[0.5], &[Some(1.0), Some(2.0)], 1.0, nothing * (1.0 + 1e-6));
        assert_eq!(
            m.rows_t,
            vec![9, 16],
            "a row just above the fraction counts"
        );
        m.step(&[0.5], &[Some(1.0), Some(2.0)], 1.0, 1.0);
        assert_eq!(m.rows_t, vec![10, 17]);
    }

    /// A standard normal from the module's LCG, by Box-Muller.
    fn normal(state: &mut u64) -> f64 {
        let u1 = (lcg(state) + 1.0) / 2.0;
        let u2 = (lcg(state) + 1.0) / 2.0;
        let u1 = u1.max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A truncated factor at or below zero -- two series whose
    /// autocorrelations have opposite signs, `1 + 2·0.8·(−0.8) = −0.28` -- is
    /// an estimate outside the parameter space, and says so with NaN, the
    /// answer `Geometric` gives a factor it cannot form. It was floored at
    /// `f64::MIN_POSITIVE`, so `n_serial` came out `+inf` and `t_serial`
    /// `±inf`: infinite evidence from a correction that had failed (review
    /// 2026-09-12, S18).
    #[test]
    fn a_truncated_factor_that_is_not_positive_is_nan() {
        for (rx, ry) in [(0.8, -0.8), (0.5, -1.0)] {
            let (f, px, py) = serial_factor(SerialRule::Truncated, &[1], &[rx], &[ry]);
            assert!(f.is_nan(), "({rx}, {ry}): {f}");
            assert!(px.is_nan() && py.is_nan());
        }
        let (f, _, _) = serial_factor(SerialRule::Truncated, &[1], &[0.8], &[0.8]);
        assert!((f - 2.28).abs() < 1e-12, "{f}");
    }

    /// E66 tests 3 and 4, the ones the feature exists for. Two *independent*
    /// AR(1) series, so the true correlation is zero and `t` should be
    /// N(0,1) — but it is not: with both series smooth, the sample
    /// correlation's variance is `[1 + 2·p/(1−p)]/n` with `p = phi_x·phi_y`,
    /// so `t` is over-dispersed by `sqrt` of that bracket. `t_serial` is the
    /// same statistic against a count that has been told, and it is the one
    /// that comes out standard.
    #[test]
    fn t_serial_is_standard_where_t_is_over_dispersed() {
        let (phi_x, phi_y) = (0.9_f64, 0.8_f64);
        let reps = 200;
        let n = 1500;
        let prod = phi_x * phi_y;
        let inflation = (1.0 + 2.0 * prod / (1.0 - prod)).sqrt();

        let (mut t_vals, mut ts_vals) = (Vec::new(), Vec::new());
        let (mut px_hat, mut py_hat) = (0.0, 0.0);
        for rep in 0..reps {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(f64::INFINITY); // lam = 1: the classical result
            c.lags = vec![1, 2, 3, 5, 8, 13];
            c.serial_rule = Some(SerialRule::Geometric);
            let mut m = Marginal::new(c).unwrap();
            let mut seed = 1000 + rep as u64;
            let (mut x, mut y) = (0.0, 0.0);
            for i in 0..n {
                x = phi_x * x + (1.0 - phi_x * phi_x).sqrt() * normal(&mut seed);
                y = phi_y * y + (1.0 - phi_y * phi_y).sqrt() * normal(&mut seed);
                crate::OnlineModel::step(
                    &mut m,
                    &[x],
                    &[Some(y)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            let pair = m.pair(0, 0);
            t_vals.push(pair.t);
            ts_vals.push(pair.t_serial);
            px_hat += pair.phi_x / reps as f64;
            py_hat += pair.phi_y / reps as f64;
        }
        let sd = |v: &[f64]| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let (sd_t, sd_ts) = (sd(&t_vals), sd(&ts_vals));

        // The fitted decays land on the truth (test 4).
        assert!((px_hat - phi_x).abs() < 0.05, "phi_x {px_hat} vs {phi_x}");
        assert!((py_hat - phi_y).abs() < 0.05, "phi_y {py_hat} vs {phi_y}");
        // `t` is over-dispersed by roughly the Bartlett inflation...
        assert!(
            sd_t > 0.6 * inflation && sd_t < 1.6 * inflation,
            "sd(t) {sd_t} should be near the inflation {inflation}"
        );
        // ... and `t_serial` is standard, which is the whole point.
        assert!(
            (0.75..1.35).contains(&sd_ts),
            "sd(t_serial) {sd_ts} should be near 1 (sd(t) was {sd_t}, inflation {inflation})"
        );
    }

    /// PLAN §13.4 for `marginal`: one pair, computed directly over the rows
    /// inside the window and nothing else. The boundary is inclusive, as
    /// `window.rs` states it: a row exactly `window` old is inside.
    #[test]
    fn a_windowed_pair_is_the_pair_of_the_rows_inside_the_window() {
        let (halflife, window) = (25.0, 70.0);
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(halflife);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();

        let mut seed = 99u64;
        let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
        let mut on_the_boundary = 0;
        for i in 0..140 {
            // `lcg` is [-1, 1): a clock increment must be its magnitude, or
            // the clock runs backwards and the window means nothing.
            // Quarter units, exact in a double, so some rows land exactly
            // one window old (the boundary is inclusive).
            let d = if i == 0 {
                0.0
            } else {
                0.25 * (2.0 + (lcg(&mut seed).abs() * 8.0).floor())
            };
            clock += d;
            let x = lcg(&mut seed) * 3.0;
            let y = 2.0 * x + lcg(&mut seed) * 0.2;
            crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
            xs.push(x);
            ys.push(y);
            t.push(clock);

            if i >= 5 {
                let now = clock;
                let keep: Vec<usize> = (0..=i).filter(|&j| now - t[j] <= window).collect();
                on_the_boundary += keep.iter().filter(|&&j| now - t[j] == window).count();
                let w: Vec<f64> = keep
                    .iter()
                    .map(|&j| 0.5_f64.powf((now - t[j]) / halflife))
                    .collect();
                let wsum: f64 = w.iter().sum();
                let mx: f64 = keep.iter().zip(&w).map(|(&j, wi)| wi * xs[j]).sum::<f64>() / wsum;
                let my: f64 = keep.iter().zip(&w).map(|(&j, wi)| wi * ys[j]).sum::<f64>() / wsum;
                let vx: f64 = keep
                    .iter()
                    .zip(&w)
                    .map(|(&j, wi)| wi * (xs[j] - mx) * (xs[j] - mx))
                    .sum::<f64>()
                    / wsum;
                let cov: f64 = keep
                    .iter()
                    .zip(&w)
                    .map(|(&j, wi)| wi * (xs[j] - mx) * (ys[j] - my))
                    .sum::<f64>()
                    / wsum;
                let got = m.pair(0, 0);
                let tol = |a: f64, b: f64| (a - b).abs() < 1e-8 * b.abs().max(1.0);
                assert!(
                    tol(got.n_eff, wsum),
                    "row {i} n_eff: {} vs {wsum}",
                    got.n_eff
                );
                assert!(
                    tol(got.mean_x, mx),
                    "row {i} mean_x: {} vs {mx}",
                    got.mean_x
                );
                assert!(
                    tol(got.mean_y, my),
                    "row {i} mean_y: {} vs {my}",
                    got.mean_y
                );
                assert!(tol(got.var_x, vx), "row {i} var_x: {} vs {vx}", got.var_x);
                assert!(tol(got.cov, cov), "row {i} cov: {} vs {cov}", got.cov);
            }
        }
        // The boundary was exercised, not only the rows either side of it.
        assert!(
            on_the_boundary >= 5,
            "{on_the_boundary} reads had a row exactly one window old"
        );
    }

    /// Review 2026-09-12, C17: the same pair against a feature and a target
    /// that sit at `1e8`. `cut` used to go back through the raw moment `s +
    /// ma·mb`, which at this level loses the variance and the covariance
    /// entirely; the oracle is two-pass, so it is right at any offset. The
    /// boundary is inclusive, as `window.rs` states it.
    #[test]
    fn a_windowed_pair_keeps_its_precision_at_a_large_offset() {
        let (halflife, window) = (25.0, 70.0);
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(halflife);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();

        let mut seed = 4242u64;
        let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
        let mut on_the_boundary = 0;
        for i in 0..160 {
            // Quarter units, exact in a double, so some rows land exactly
            // one window old (the boundary is inclusive).
            let d = if i == 0 {
                0.0
            } else {
                0.25 * (2.0 + (lcg(&mut seed).abs() * 8.0).floor())
            };
            clock += d;
            let u = lcg(&mut seed) * 3.0;
            let x = 1e8 + u;
            let y = 1e8 + 2.0 * u + lcg(&mut seed) * 0.2;
            crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
            xs.push(x);
            ys.push(y);
            t.push(clock);

            if i >= 5 {
                let now = clock;
                let keep: Vec<usize> = (0..=i).filter(|&j| now - t[j] <= window).collect();
                on_the_boundary += keep.iter().filter(|&&j| now - t[j] == window).count();
                let w: Vec<f64> = keep
                    .iter()
                    .map(|&j| 0.5_f64.powf((now - t[j]) / halflife))
                    .collect();
                let wsum: f64 = w.iter().sum();
                let mean =
                    |v: &[f64]| keep.iter().zip(&w).map(|(&j, wj)| wj * v[j]).sum::<f64>() / wsum;
                let (mx, my) = (mean(&xs), mean(&ys));
                let co = |a: &[f64], ma: f64, b: &[f64], mb: f64| {
                    keep.iter()
                        .zip(&w)
                        .map(|(&j, wj)| wj * (a[j] - ma) * (b[j] - mb))
                        .sum::<f64>()
                        / wsum
                };
                let (vx, vy, cov) = (
                    co(&xs, mx, &xs, mx),
                    co(&ys, my, &ys, my),
                    co(&xs, mx, &ys, my),
                );
                let got = m.pair(0, 0);
                let close = |got: f64, want: f64| (got - want).abs() < 1e-6 * want.abs().max(1.0);
                assert!(
                    (got.n_eff - wsum).abs() < 1e-9 * wsum,
                    "row {i} n_eff: {} vs {wsum}",
                    got.n_eff
                );
                assert!(
                    (got.mean_x - mx).abs() < 1e-12 * mx,
                    "row {i} mean_x: {} vs {mx}",
                    got.mean_x
                );
                assert!(
                    (got.mean_y - my).abs() < 1e-12 * my,
                    "row {i} mean_y: {} vs {my}",
                    got.mean_y
                );
                assert!(close(got.var_x, vx), "row {i} var_x: {} vs {vx}", got.var_x);
                assert!(close(got.var_y, vy), "row {i} var_y: {} vs {vy}", got.var_y);
                assert!(close(got.cov, cov), "row {i} cov: {} vs {cov}", got.cov);
            }
        }
        assert!(
            on_the_boundary >= 5,
            "{on_the_boundary} reads had a row exactly one window old"
        );
    }

    /// A window no stream reaches leaves every pair exactly as it was.
    #[test]
    fn a_marginal_window_no_stream_reaches_changes_nothing() {
        let mk = |window: Option<f64>| {
            let mut c = cfg(2, 1);
            c.window = window;
            Marginal::new(c).unwrap()
        };
        let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
        let mut seed = 4u64;
        for i in 0..60 {
            let x = [lcg(&mut seed), lcg(&mut seed)];
            let y = x[0] - x[1];
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut plain, &x, &[Some(y)], d, 1.0);
            crate::OnlineModel::step(&mut windowed, &x, &[Some(y)], d, 1.0);
        }
        for j in 0..2 {
            let (a, b) = (plain.pair(0, j), windowed.pair(0, j));
            assert_eq!(a.mean_x.to_bits(), b.mean_x.to_bits(), "feature {j} mean");
            assert_eq!(a.cov.to_bits(), b.cov.to_bits(), "feature {j} cov");
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "feature {j} n_eff");
        }
    }

    /// After a gap that empties the window, the model's `n_eff` is 0 exactly,
    /// as every pair's is -- not the rounding crumb the truncating
    /// subtraction leaves (review 2026-09-18, S4).
    #[test]
    fn a_gap_that_empties_the_window_leaves_n_eff_exactly_zero() {
        let (halflife, window) = (25.0, 70.0);
        let mut c = cfg(2, 1);
        c.decay = Decay::Halflife(halflife);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 7u64;
        // Fill the window with real rows.
        for i in 0..40 {
            let x = [lcg(&mut seed), lcg(&mut seed)];
            let y = x[0] - x[1];
            let d = if i == 0 { 0.0 } else { 2.0 };
            crate::OnlineModel::step(&mut m, &x, &[Some(y)], d, 1.0);
        }
        // Advance the clock well past the window with zero-weight rows: the
        // clock moves, nothing is learned, so the window empties.
        for _ in 0..60 {
            crate::OnlineModel::step(&mut m, &[0.0, 0.0], &[Some(0.0)], 2.0, 0.0);
        }
        assert_eq!(m.n_eff(), 0.0, "n_eff is a crumb, not 0");
        assert_eq!(m.target_weight(0), 0.0, "target_weight is a crumb, not 0");
        for j in 0..2 {
            assert_eq!(m.pair(0, j).n_eff, 0.0, "pair {j} n_eff is a crumb, not 0");
        }
    }

    /// Weighted, decayed moments of one pair written out longhand as sums
    /// over the rows the target was present: the oracle the recursion is
    /// held to.
    struct Longhand {
        xs: Vec<f64>,
        ys: Vec<f64>,
        ws: Vec<f64>,
        lams: Vec<f64>,
    }

    impl Longhand {
        /// The decay each row has suffered by the end: the product of the
        /// factors of every later row (present or not -- time passes for
        /// them all).
        fn weights(&self) -> Vec<f64> {
            let n = self.ws.len();
            (0..n)
                .map(|i| self.ws[i] * self.lams[i + 1..].iter().product::<f64>())
                .collect()
        }

        fn moments(&self) -> (f64, f64, f64, f64, f64, f64, f64) {
            let w = self.weights();
            let sw: f64 = w.iter().sum();
            let sq: f64 = w.iter().map(|v| v * v).sum();
            let mx = w.iter().zip(&self.xs).map(|(w, x)| w * x).sum::<f64>() / sw;
            let my = w.iter().zip(&self.ys).map(|(w, y)| w * y).sum::<f64>() / sw;
            let sxx = w
                .iter()
                .zip(&self.xs)
                .map(|(w, x)| w * (x - mx) * (x - mx))
                .sum::<f64>()
                / sw;
            let syy = w
                .iter()
                .zip(&self.ys)
                .map(|(w, y)| w * (y - my) * (y - my))
                .sum::<f64>()
                / sw;
            let sxy = w
                .iter()
                .zip(&self.xs)
                .zip(&self.ys)
                .map(|((w, x), y)| w * (x - mx) * (y - my))
                .sum::<f64>()
                / sw;
            (sw, sq, mx, my, sxx, syy, sxy)
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn validation_rejects_each_bad_field() {
        let err = |c: MarginalCfg, what: &str| {
            let e = Marginal::new(c).unwrap_err();
            assert!(e.contains(what), "{e}");
        };
        err(cfg(0, 1), "at least one feature");
        err(cfg(1, 0), "at least one target");
        let mut c = cfg(1, 1);
        c.min_periods = vec![-1.0];
        err(c.clone(), "min_periods");
        // A gate that never opens, as for every other model (S27).
        c.min_periods = vec![f64::INFINITY];
        assert!(Marginal::new(c.clone()).is_ok());
        c.min_periods = vec![f64::NAN];
        err(c.clone(), "min_periods");
        c.min_periods = vec![1.0, 2.0];
        err(c.clone(), "2 entries for 1 targets");
        c.min_periods = vec![];
        err(c, "0 entries for 1 targets");
        // The lag ring has no snapshot, so a window beside it reported the
        // whole history's lagged co-moment over the window's variance;
        // `ew_cov` refuses the pair, and so does this (review 2026-09-12,
        // C18).
        let mut c = cfg(1, 1);
        c.window = Some(50.0);
        c.lags = vec![1];
        err(c, "window and lags");
        Marginal::new(cfg(3, 2)).unwrap();
    }

    #[test]
    fn matches_the_longhand_moments_with_weights_gaps_and_a_missing_target() {
        // Two targets, the second missing on every third row, uneven weights
        // (some zero), and clock gaps: each target's pair moments must equal
        // the decayed weighted sums over the rows where that target was
        // present, with the decay of every row -- present or not -- applied.
        let mut m = Marginal::new(cfg(3, 2)).unwrap();
        let mut s = 7u64;
        let mut long: Vec<Vec<Longhand>> = (0..2)
            .map(|_| {
                (0..3)
                    .map(|_| Longhand {
                        xs: vec![],
                        ys: vec![],
                        ws: vec![],
                        lams: vec![],
                    })
                    .collect()
            })
            .collect();
        for i in 0..300 {
            let x: Vec<f64> = (0..3).map(|_| 2.0 * lcg(&mut s) + 100.0).collect();
            let y0 = x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
            let y1 = -x[2] + 0.1 * lcg(&mut s);
            let y = [Some(y0), (i % 3 != 2).then_some(y1)];
            let w = match i % 7 {
                0 => 0.0,
                1 => 2.5,
                _ => 1.0,
            };
            let d = if i == 0 {
                0.0
            } else if i % 50 == 0 {
                15.0
            } else {
                1.0
            };
            let lam = m.cfg().decay.factor(d);
            for (t, yt) in y.iter().enumerate() {
                for j in 0..3 {
                    let l = &mut long[t][j];
                    // Every row ages the target's history; only a present
                    // target adds a row.
                    // A missing target is a zero-weight row in the longhand:
                    // it adds nothing to the sums and `weights()` still
                    // applies its decay factor to every earlier row.
                    let (xv, yv, wv) = match yt {
                        Some(yt) => (x[j], *yt, w),
                        None => (0.0, 0.0, 0.0),
                    };
                    l.xs.push(xv);
                    l.ys.push(yv);
                    l.ws.push(wv);
                    l.lams.push(lam);
                }
            }
            let step = m.step(&x, &y, d, w);
            assert!(step.pred.is_empty());
            assert!(step.n_eff.is_finite());
        }
        for (t, row) in long.iter().enumerate() {
            for (j, l) in row.iter().enumerate() {
                let (sw, sq, mx, my, sxx, syy, sxy) = l.moments();
                let p = m.pair(t, j);
                assert!(close(p.n_eff, sw, 1e-12), "W_{t}: {} vs {sw}", p.n_eff);
                assert!(
                    close(p.n_kish, sw * sw / sq, 1e-12),
                    "kish_{t}: {} vs {}",
                    p.n_kish,
                    sw * sw / sq
                );
                assert!(
                    close(p.mean_x, mx, 1e-12),
                    "mx[{t},{j}] {} vs {mx}",
                    p.mean_x
                );
                assert!(close(p.mean_y, my, 1e-12), "my[{t}] {} vs {my}", p.mean_y);
                // Centred second moments around an offset of 100: the
                // Welford form keeps them to ~1e-12 relative; a raw
                // `E[x²] − m²` would have lost them.
                assert!(
                    close(p.var_x, sxx, 1e-9),
                    "sxx[{t},{j}] {} vs {sxx}",
                    p.var_x
                );
                assert!(close(p.var_y, syy, 1e-9), "syy[{t}] {} vs {syy}", p.var_y);
                assert!(close(p.cov, sxy, 1e-9), "sxy[{t},{j}] {} vs {sxy}", p.cov);
                let corr = sxy / (sxx * syy).sqrt();
                assert!(
                    close(p.corr, corr, 1e-9),
                    "corr[{t},{j}] {} vs {corr}",
                    p.corr
                );
                assert!(close(p.beta, sxy / sxx, 1e-9), "beta[{t},{j}]");
                let n = sw * sw / sq;
                let tt = corr * ((n - 2.0) / (1.0 - corr * corr)).sqrt();
                assert!(close(p.t, tt, 1e-9), "t[{t},{j}] {} vs {tt}", p.t);
            }
        }
        // The model-level weight counts every row, including the ones where
        // a target was missing: it is `W_0` here, since target 0 was always
        // present.
        assert_eq!(m.n_eff().to_bits(), m.target_weight(0).to_bits());
        assert!(m.target_weight(1) < m.target_weight(0));
    }

    #[test]
    fn a_pair_is_the_ew_cov_of_the_two_columns_to_the_bit() {
        // `ew_cov` over `[x_j, y]` and the pair `(j, 0)` here, fed the same
        // rows: the same recursion in the same order gives the same bits --
        // moments, and the correlation.
        let mut m = Marginal::new(cfg(2, 1)).unwrap();
        let mut covs = [EwCov::new(2), EwCov::new(2)];
        let mut full = EwCovModel::new(EwCovCfg {
            n_features: 2,
            decay: Decay::Halflife(20.0),
            stats: vec![EwCovStat::Corr],
            min_periods: 0.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 1,
            lags: Vec::new(),
            window: None,
            window_every: None,
        })
        .unwrap();
        let mut s = 99u64;
        for i in 0..200 {
            let x = [lcg(&mut s) * 3.0, lcg(&mut s) + 5.0];
            let y = x[0] + 0.7 * x[1] + 0.2 * lcg(&mut s);
            let w = if i % 5 == 0 {
                0.0
            } else {
                1.0 + 0.5 * (i % 4) as f64
            };
            let d = if i == 0 { 0.0 } else { 0.5 + (i % 3) as f64 };
            let lam = m.cfg().decay.factor(d);
            // The ew_cov reference is read *before* the row too.
            let corr_ref = full.step(&[x[0], y], &[], d, w).pred[0];
            let before = m.pair(0, 0);
            if i > 0 {
                assert_eq!(before.corr.to_bits(), corr_ref.to_bits(), "row {i}");
            }
            m.step(&x, &[Some(y)], d, w);
            for (j, c) in covs.iter_mut().enumerate() {
                c.update(&[x[j], y], lam, w);
            }
        }
        for (j, c) in covs.iter().enumerate() {
            let p = m.pair(0, j);
            assert_eq!(p.n_eff.to_bits(), c.n_eff().to_bits());
            assert_eq!(p.mean_x.to_bits(), c.mean(0).to_bits());
            assert_eq!(p.mean_y.to_bits(), c.mean(1).to_bits());
            assert_eq!(p.var_x.to_bits(), c.var(0).to_bits());
            assert_eq!(p.var_y.to_bits(), c.var(1).to_bits());
            assert_eq!(p.cov.to_bits(), c.cov(0, 1).to_bits());
        }
    }

    #[test]
    fn n_eff_is_the_weight_before_the_row_and_min_periods_gates_the_derived_values() {
        let mut c = cfg(1, 1);
        c.min_periods = vec![3.0];
        let mut m = Marginal::new(c).unwrap();
        let lam = 0.5f64.powf(1.0 / 20.0);
        let mut expect = 0.0;
        for i in 0..6 {
            let x = [i as f64];
            let step = m.step(
                &x,
                &[Some(2.0 * i as f64 + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            assert_eq!(step.n_eff, expect, "row {i}");
            expect = if i == 0 { 1.0 } else { lam * expect + 1.0 };
            let p = m.pair(0, 0);
            // The moments are there from the first row; the derived values
            // wait for the weight.
            assert!(p.mean_x.is_finite());
            if p.n_eff < 3.0 {
                assert!(
                    p.corr.is_nan() && p.beta.is_nan() && p.t.is_nan(),
                    "row {i}"
                );
            } else {
                assert!(
                    (p.corr - 1.0).abs() < 1e-12,
                    "y = 2x + 1: corr 1, got {}",
                    p.corr
                );
                assert!((p.beta - 2.0).abs() < 1e-9, "beta 2, got {}", p.beta);
                // `corr` is 1 to rounding, so `1 − corr²` is tiny or zero
                // and t is enormous or +inf: either is the honest value.
                assert!(p.t > 1e3, "perfect fit: t huge, got {}", p.t);
            }
        }
    }

    #[test]
    fn a_zero_weight_first_row_and_a_constant_column_stay_finite() {
        let mut m = Marginal::new(cfg(2, 1)).unwrap();
        // Weight 0 on the very first row: nothing to average, no 0/0.
        let s0 = m.step(&[1.0, 2.0], &[Some(3.0)], 0.0, 0.0);
        assert_eq!(s0.n_eff, 0.0);
        let p = m.pair(0, 0);
        assert_eq!(p.n_eff, 0.0);
        assert!(p.n_kish.is_nan(), "0/0 before any weight");
        assert_eq!(p.mean_x, 0.0);
        assert!(p.corr.is_nan());
        // Then a constant feature (column 1): var_x = 0 exactly, corr and
        // beta NaN rather than infinite, everything else finite.
        for i in 0..20 {
            let xi = (i as f64).sin();
            m.step(&[xi, 7.0], &[Some(2.0 * xi)], 1.0, 1.0);
        }
        let p = m.pair(0, 1);
        assert_eq!(p.var_x, 0.0);
        assert_eq!(p.cov, 0.0);
        assert!(p.corr.is_nan() && p.beta.is_nan());
        assert!(p.mean_y.is_finite() && p.var_y > 0.0);
        let p = m.pair(0, 0);
        assert!((p.corr - 1.0).abs() < 1e-12);
        assert!((p.beta - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kish_size_of_unit_weights_tends_to_one_plus_lam_over_one_minus_lam() {
        let mut m = Marginal::new(cfg(1, 1)).unwrap();
        let mut s = 3u64;
        for i in 0..5000 {
            let x = lcg(&mut s);
            m.step(
                &[x],
                &[Some(x + lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let lam = 0.5f64.powf(1.0 / 20.0);
        let p = m.pair(0, 0);
        assert!(close(p.n_eff, 1.0 / (1.0 - lam), 1e-9));
        assert!(
            close(p.n_kish, (1.0 + lam) / (1.0 - lam), 1e-9),
            "{}",
            p.n_kish
        );
        // Unequal weights lower it: a single heavy row dominates.
        let mut m2 = Marginal::new(cfg(1, 1)).unwrap();
        for i in 0..50 {
            let w = if i == 49 { 1000.0 } else { 1.0 };
            m2.step(
                &[i as f64],
                &[Some(i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                w,
            );
        }
        let p2 = m2.pair(0, 0);
        assert!(p2.n_kish < 1.1, "one row carries the weight: {}", p2.n_kish);
        assert!(p2.n_eff > 1000.0);
    }

    #[test]
    fn state_round_trips_and_continues_identically() {
        let mut m = Marginal::new(cfg(3, 2)).unwrap();
        let mut s = 11u64;
        let row = |s: &mut u64| {
            let x: Vec<f64> = (0..3).map(|_| lcg(s)).collect();
            let y = [Some(x[0] + lcg(s)), (lcg(s) > 0.0).then(|| x[1] - lcg(s))];
            (x, y)
        };
        for i in 0..50 {
            let (x, y) = row(&mut s);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let mut r = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(r, m);
        for _ in 0..50 {
            let (x, y) = row(&mut s);
            let a = m.step(&x, &y, 1.0, 1.5);
            let b = r.step(&x, &y, 1.0, 1.5);
            assert_eq!(a, b);
        }
        assert_eq!(r, m);
        let wrong = Marginal::restore(
            &EwCovModel::new(EwCovCfg {
                n_features: 1,
                decay: Decay::Halflife(1.0),
                stats: vec![],
                min_periods: 0.0,
                precision_prior: None,
                mahal_quantiles: vec![],
                pca: 0,
                pca_every: 1,
                lags: Vec::new(),
                window: None,
                window_every: None,
            })
            .unwrap()
            .state(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            wrong.contains("marginal") && wrong.contains("ew_cov"),
            "{wrong}"
        );
    }

    #[test]
    fn shape_accessors() {
        let m = Marginal::new(cfg(4, 3)).unwrap();
        assert_eq!(m.n_features(), 4);
        assert_eq!(m.n_targets(), 3);
        assert_eq!(m.n_outputs(), 0);
        assert_eq!(m.n_eff(), 0.0);
        assert_eq!(m.state().model.kind(), "marginal");
        let p = m.predict(&[0.0; 4], 1.0);
        assert!(p.pred.is_empty() && p.n_eff == 0.0 && p.extra.is_none());
    }
    /// `d_clock` is the gap since the previous row, so the first row has none.
    fn step_clock(i: usize) -> f64 {
        if i == 0 { 0.0 } else { 1.0 }
    }

    fn bins_cfg(n_bins: usize, warm_rows: usize) -> Box<crate::BinCfg> {
        Box::new(crate::BinCfg {
            n_bins,
            edges: None,
            rule: crate::BinRule::Quantile,
            warm_rows,
            budget_mib: None,
        })
    }

    fn marginal_with(bins: Option<Box<crate::BinCfg>>) -> Marginal {
        Marginal::new(MarginalCfg {
            n_features: 1,
            n_targets: 1,
            decay: Decay::Halflife(200.0),
            min_periods: vec![0.0],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins,
            window: None,
            window_every: None,
        })
        .unwrap()
    }

    /// The warm-up rows are held, not spent. Learning the edges from the
    /// first 50 rows and then replaying them must land on exactly the state
    /// a model given those same edges up front reaches -- to the bit, since
    /// the replay performs the same operations in the same order.
    #[test]
    fn learned_edges_lose_no_row() {
        let rows: Vec<(f64, f64)> = {
            let mut seed = 99u64;
            (0..400)
                .map(|_| {
                    let x = lcg(&mut seed);
                    (x, x.abs() + 0.2 * lcg(&mut seed))
                })
                .collect()
        };
        let mut learned = marginal_with(Some(bins_cfg(4, 50)));
        for (i, (x, y)) in rows.iter().enumerate() {
            OnlineModel::step(&mut learned, &[*x], &[Some(*y)], step_clock(i), 1.0);
        }
        let got = learned.pair(0, 0);
        assert_eq!(got.bin_edges.len(), 3, "four bins means three edges");

        let mut given = marginal_with(Some(Box::new(crate::BinCfg {
            n_bins: 4,
            edges: Some(vec![got.bin_edges.clone()]),
            rule: crate::BinRule::Quantile,
            warm_rows: 50,
            budget_mib: None,
        })));
        for (i, (x, y)) in rows.iter().enumerate() {
            OnlineModel::step(&mut given, &[*x], &[Some(*y)], step_clock(i), 1.0);
        }
        let want = given.pair(0, 0);
        assert_eq!(got.bin_n, want.bin_n, "bin weights");
        assert_eq!(got.bin_mean_y, want.bin_mean_y, "bin means");
        assert_eq!(got.bin_var_y, want.bin_var_y, "bin variances");
        assert_eq!(got.split_gain, want.split_gain);
        assert_eq!(got.split_at, want.split_at);
    }

    /// The same across targets, through the row update that bins each
    /// feature once (E71, docs/PLAN.md task 122): five targets, each absent
    /// on a random third of the rows, and three features, one of them
    /// constant so that it keeps no edges. The replay lands on the state
    /// given edges reach, pair by pair, every bin and split number to the
    /// bit.
    ///
    /// Except by a rounding where rows of weight zero fall inside the
    /// warm-up: each is held as its decay alone, folded into the next held
    /// row so that a run of them cannot grow the hold, and the replay ages
    /// the histogram by the product, `scale·(a·b)` where given edges age it
    /// `(scale·a)·b` (docs/MARGINAL-LAGS-AND-BINS.md, "Invariance"). Measured
    /// here, the two differ by 1.5e-15 of the data's scale; the bound below
    /// is ten times that, and the test fails if they stop differing, since
    /// the docs would then understate the replay.
    #[test]
    fn learned_edges_lose_no_row_across_targets() {
        let (p, n_targets) = (3, 5);
        let mk = |edges: Option<Vec<Vec<f64>>>| {
            Marginal::new(MarginalCfg {
                n_features: p,
                n_targets,
                decay: Decay::Halflife(200.0),
                min_periods: vec![0.0; n_targets],
                lags: Vec::new(),
                serial_rule: None,
                cross_lags: None,
                bins: Some(Box::new(crate::BinCfg {
                    n_bins: 6,
                    edges,
                    rule: crate::BinRule::Quantile,
                    warm_rows: 60,
                    budget_mib: None,
                })),
                window: None,
                window_every: None,
            })
            .unwrap()
        };
        // The distance between two numbers on the data's scale, which is
        // about 1: equal bits (a NaN for a bin or a split that has none) are
        // 0 apart, and a NaN against a number is infinitely far.
        let apart = |a: f64, b: f64| -> f64 {
            if a.to_bits() == b.to_bits() {
                0.0
            } else if a.is_finite() && b.is_finite() {
                (a - b).abs() / a.abs().max(b.abs()).max(1.0)
            } else {
                f64::INFINITY
            }
        };
        let mut worst_with_zero_weights = 0.0_f64;
        for zero_weights in [false, true] {
            let mut seed = 31u64;
            let rows: Vec<(Vec<f64>, Vec<Option<f64>>, f64)> = (0..500)
                .map(|i| {
                    let x = vec![lcg(&mut seed), 0.25, lcg(&mut seed).powi(3)];
                    let y = (0..n_targets)
                        .map(|t| {
                            let v = x[0] * t as f64 + x[2].abs() + 0.1 * lcg(&mut seed);
                            (lcg(&mut seed) >= -1.0 / 3.0).then_some(v)
                        })
                        .collect();
                    let w = if zero_weights && i % 17 == 5 {
                        0.0
                    } else {
                        1.0
                    };
                    (x, y, w)
                })
                .collect();
            let run = |m: &mut Marginal| {
                for (i, (x, y, w)) in rows.iter().enumerate() {
                    OnlineModel::step(m, x, y, step_clock(i), *w);
                }
            };
            let mut learned = mk(None);
            run(&mut learned);
            let edges: Vec<Vec<f64>> = (0..p).map(|j| learned.pair(0, j).bin_edges).collect();
            assert_eq!(edges[0].len(), 5, "six bins means five edges");
            assert!(edges[1].is_empty(), "a constant feature keeps no edges");
            let mut given = mk(Some(edges));
            run(&mut given);
            for t in 0..n_targets {
                for j in 0..p {
                    let (a, b) = (learned.pair(t, j), given.pair(t, j));
                    assert!(a.bin_n.iter().any(|n| *n > 0.0), "target {t} feature {j}");
                    let numbers = |q: &Pair| {
                        let mut v =
                            [q.bin_n.clone(), q.bin_mean_y.clone(), q.bin_var_y.clone()].concat();
                        v.extend([q.split_gain, q.split_at, q.split_gain_t]);
                        v
                    };
                    let (na, nb) = (numbers(&a), numbers(&b));
                    assert_eq!(na.len(), nb.len());
                    let worst = na
                        .iter()
                        .zip(&nb)
                        .map(|(x, y)| apart(*x, *y))
                        .fold(0.0, f64::max);
                    if zero_weights {
                        worst_with_zero_weights = worst_with_zero_weights.max(worst);
                    } else {
                        assert_eq!(worst, 0.0, "target {t} feature {j}: {na:?} vs {nb:?}");
                    }
                }
            }
        }
        assert!(
            worst_with_zero_weights > 0.0 && worst_with_zero_weights < 1.5e-14,
            "{worst_with_zero_weights:e} apart with rows of weight zero in the warm-up"
        );
    }

    /// A zero-weight row inside the warm-up teaches nothing: its feature
    /// values cannot reach the histogram, so putting absurd ones there must
    /// change nothing at all. (That it also advances the clock is the shared
    /// rule, checked for every model in `model_contract.rs`.)
    #[test]
    fn a_zero_weight_row_in_the_warm_up_teaches_nothing() {
        let run = |junk: f64| {
            let mut m = marginal_with(Some(bins_cfg(4, 20)));
            let mut seed = 5u64;
            for i in 0..200 {
                let x = lcg(&mut seed);
                if i == 5 {
                    OnlineModel::step(&mut m, &[junk], &[Some(junk)], 1.0, 0.0);
                }
                OnlineModel::step(&mut m, &[x], &[Some(x.abs())], step_clock(i), 1.0);
            }
            m.pair(0, 0)
        };
        let tame = run(0.5);
        let absurd = run(1e9);
        assert_eq!(
            tame.bin_edges, absurd.bin_edges,
            "a zero-weight row set an edge"
        );
        assert_eq!(
            tame.bin_n, absurd.bin_n,
            "a zero-weight row landed in a bin"
        );
        assert_eq!(tame.bin_mean_y, absurd.bin_mean_y);
        assert_eq!(tame.split_gain, absurd.split_gain);
    }

    /// What bins are for: a V-shaped relation has no linear signal at all,
    /// and a split finds it. Checked against the gain computed the long way
    /// over the same edges.
    #[test]
    fn a_v_shape_is_invisible_to_corr_and_obvious_to_a_split() {
        let mut m = marginal_with(Some(bins_cfg(8, 200)));
        let mut seed = 21u64;
        let mut rows = Vec::new();
        for i in 0..4000 {
            let x = lcg(&mut seed);
            let y = x.abs();
            OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
            rows.push((x, y));
        }
        let p = m.pair(0, 0);
        // The half-life is 200, so the effective sample is about 290 rows and
        // `corr` has a standard error near 0.06 whatever the truth is. The
        // claim is not that it is zero, but that it is noise: the split
        // explains an order of magnitude more of the target's variance than
        // the linear fit's own R-squared does.
        assert!(
            p.corr.abs() < 0.2,
            "a symmetric V should have no linear signal, got corr = {}",
            p.corr
        );
        assert!(
            p.split_gain > 10.0 * p.corr * p.corr,
            "gain {} against a linear R-squared of {}",
            p.split_gain,
            p.corr * p.corr
        );
        assert!(
            p.split_gain > 0.2,
            "a split should see it, got gain = {}",
            p.split_gain
        );
        // Overwhelming for an effective sample of ~290, even allowing for
        // the cut having been chosen by maximizing over seven candidates.
        assert!(
            p.split_gain_t > 10.0,
            "and say so loudly, got {}",
            p.split_gain_t
        );

        // The same gain, computed from the raw rows over the same edges.
        // Undecayed, so only the recent-weighted answer differs, and the
        // half-life is long enough here for that to be small.
        let edges = &p.bin_edges;
        let var_of = |rs: &[(f64, f64)]| {
            let n = rs.len() as f64;
            let m = rs.iter().map(|r| r.1).sum::<f64>() / n;
            rs.iter().map(|r| (r.1 - m) * (r.1 - m)).sum::<f64>() / n
        };
        let total = var_of(&rows);
        let best = edges
            .iter()
            .map(|c| {
                let (l, r): (Vec<_>, Vec<_>) = rows.iter().partition(|(x, _)| x < c);
                let (nl, nr) = (l.len() as f64, r.len() as f64);
                (total - (nl * var_of(&l) + nr * var_of(&r)) / (nl + nr)) / total
            })
            .fold(0.0_f64, f64::max);
        assert!(
            (p.split_gain - best).abs() < 0.05,
            "gain {} vs the long way {best}",
            p.split_gain
        );
    }

    /// Nothing is reported before the edges exist, and `bins` is refused
    /// where it cannot be honoured.
    #[test]
    fn bins_before_and_beyond_what_it_can_do() {
        let mut m = marginal_with(Some(bins_cfg(4, 100)));
        for i in 0..10 {
            OnlineModel::step(&mut m, &[i as f64], &[Some(1.0)], step_clock(i), 1.0);
        }
        let p = m.pair(0, 0);
        assert!(
            p.bin_edges.is_empty(),
            "no curve before the edges are fixed"
        );
        assert!(p.split_gain.is_nan());

        let with_window = |bins| {
            Marginal::new(MarginalCfg {
                n_features: 1,
                n_targets: 1,
                decay: Decay::Halflife(50.0),
                min_periods: vec![0.0],
                lags: Vec::new(),
                serial_rule: None,
                cross_lags: None,
                bins,
                window: Some(100.0),
                window_every: None,
            })
        };
        let err = with_window(Some(bins_cfg(4, 100))).unwrap_err();
        assert!(err.contains("bins and window"), "{err}");
        assert!(with_window(None).is_ok());

        let too_few = Marginal::new(MarginalCfg {
            n_features: 1,
            n_targets: 1,
            decay: Decay::Halflife(50.0),
            min_periods: vec![0.0],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins: Some(bins_cfg(8, 4)),
            window: None,
            window_every: None,
        })
        .unwrap_err();
        assert!(too_few.contains("bin_warm_rows"), "{too_few}");
    }

    /// A feature with two values gets one edge, and a constant one gets
    /// none, rather than an error: real inputs contain both.
    #[test]
    fn degenerate_features_keep_the_bins_they_can_support() {
        let mut m = Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: Decay::Halflife(100.0),
            min_periods: vec![0.0],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins: Some(bins_cfg(4, 20)),
            window: None,
            window_every: None,
        })
        .unwrap();
        let mut seed = 3u64;
        for i in 0..100 {
            let x = lcg(&mut seed);
            let binary = if x > 0.0 { 1.0 } else { 0.0 };
            OnlineModel::step(
                &mut m,
                &[x, binary, 7.0],
                &[Some(binary)],
                step_clock(i),
                1.0,
            );
        }
        assert_eq!(
            m.pair(0, 0).bin_edges.len(),
            3,
            "a spread feature: four bins"
        );
        assert_eq!(
            m.pair(0, 1).bin_edges.len(),
            1,
            "a binary feature: two bins"
        );
        let constant = m.pair(0, 2);
        assert!(constant.bin_edges.is_empty(), "a constant feature: one bin");
        assert!(constant.split_gain.is_nan(), "and no split to report");
    }

    /// Two edge lists are the same thing in two layouts: bins learned from
    /// a warm-up, or given up front. This is the given kind.
    fn given_edges(edges: Vec<f64>) -> Box<crate::BinCfg> {
        Box::new(crate::BinCfg {
            n_bins: 2,
            edges: Some(vec![edges]),
            rule: crate::BinRule::Quantile,
            warm_rows: 2,
            budget_mib: None,
        })
    }

    /// The lagged moments are normalized (`E_w`), and their decay rides on
    /// the target's weight through `a = lam·W/W'` on the rows that learn. A
    /// row where the target is absent must therefore leave them exactly
    /// where they are, as it leaves the pair moments: ageing them on their
    /// own biased every lagged autocorrelation toward zero by the missing
    /// fraction (`marglag` module doc).
    #[test]
    fn lag_moments_hold_where_the_target_is_missing() {
        let mut c = cfg(1, 1);
        c.lags = vec![1, 2];
        c.serial_rule = Some(SerialRule::Truncated);
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 5u64;
        let mut x = 0.0;
        for i in 0..300 {
            x = 0.9 * x + lcg(&mut seed);
            let y = x + 0.1 * lcg(&mut seed);
            OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
        }
        let before = m.pair(0, 0);
        assert!(before.lagcorr_xx[0] > 0.5, "{:?}", before.lagcorr_xx);
        for _ in 0..100 {
            x = 0.9 * x + lcg(&mut seed);
            OnlineModel::step(&mut m, &[x], &[None], 1.0, 1.0);
        }
        let after = m.pair(0, 0);
        assert_eq!(before.var_x, after.var_x, "the pair moments hold");
        assert_eq!(before.corr, after.corr);
        assert_eq!(
            before.lagcorr_xx, after.lagcorr_xx,
            "so the lag moments hold"
        );
        assert_eq!(before.lagcorr_yy, after.lagcorr_yy);
        assert_eq!(before.lagcorr_xy, after.lagcorr_xy);
        assert_eq!(before.lagcorr_yx, after.lagcorr_yx);
        // `n_kish = W²/Q` is scale-free, so the correction is unchanged too
        // (to rounding: `W` and `Q` aged by `lam` and `lam²` a hundred times).
        assert!(close(before.n_serial, after.n_serial, 1e-12));
        assert!(after.n_eff < before.n_eff / 20.0, "only the weight aged");
    }

    /// A clock gap long enough that `lam` underflows to exactly zero, on a
    /// row that carries no weight: the model's `n_eff` is zero after it, and
    /// so must every target's be. The recursion `W' = lam·W + w` gives that
    /// by itself; the `0/0` guard on `a` and `b` must not skip it.
    #[test]
    fn a_zero_weight_row_across_a_total_gap_ages_the_target_weight() {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(1.0);
        let mut m = Marginal::new(c).unwrap();
        for i in 0..50 {
            OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], step_clock(i), 1.0);
        }
        assert!(m.pair(0, 0).n_eff > 1.9);
        assert_eq!(Decay::Halflife(1.0).factor(2000.0), 0.0, "a total gap");
        OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 2000.0, 0.0);
        assert_eq!(m.n_eff(), 0.0);
        let p = m.pair(0, 0);
        assert_eq!(p.n_eff, 0.0, "the target's weight decayed with the model's");
        assert!(p.n_kish.is_nan(), "0/0, reported as nothing");
        // And the stream resumes from nothing, as after a first row.
        OnlineModel::step(&mut m, &[2.0], &[Some(3.0)], 1.0, 1.0);
        let p = m.pair(0, 0);
        assert_eq!((p.n_eff, p.mean_x, p.mean_y), (1.0, 2.0, 3.0));
    }

    /// The histogram is the pair moments' companion, so a gap that takes the
    /// pair's weight to nothing (`lam = 0`) must empty it too -- and a chain
    /// of gaps whose product underflows the scale must not leave it dead.
    /// The invariant: the bin weights sum to the target's `n_eff`.
    #[test]
    fn a_total_gap_empties_the_histogram_with_the_moments() {
        let sum = |p: &Pair| p.bin_n.iter().sum::<f64>();
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(1.0);
        c.bins = Some(given_edges(vec![0.0]));
        let mut m = Marginal::new(c).unwrap();
        for i in 0..100 {
            let x = if i % 2 == 0 { -1.0 } else { 1.0 };
            OnlineModel::step(&mut m, &[x], &[Some(10.0)], step_clock(i), 1.0);
        }
        let p = m.pair(0, 0);
        assert!(p.bin_n.iter().all(|n| *n > 0.5), "{:?}", p.bin_n);
        assert!((sum(&p) - p.n_eff).abs() < 1e-9);

        // lam = 0: the old rows weigh nothing, and the histogram says so.
        OnlineModel::step(&mut m, &[-1.0], &[Some(0.0)], 2000.0, 1.0);
        let p = m.pair(0, 0);
        assert_eq!(p.bin_n, vec![1.0, 0.0], "one row, in the left bin");
        assert_eq!(p.bin_mean_y[0], 0.0, "and nothing of the 10s survives");
        assert!(p.bin_mean_y[1].is_nan());
        assert!(p.split_gain.is_nan(), "one bin, no variance, no split");
        for _ in 0..9 {
            OnlineModel::step(&mut m, &[-1.0], &[Some(0.0)], 1.0, 1.0);
        }
        let p = m.pair(0, 0);
        assert!(
            (sum(&p) - p.n_eff).abs() < 1e-9,
            "{} vs {}",
            sum(&p),
            p.n_eff
        );
        assert_eq!(p.bin_n[1], 0.0);

        // Two gaps whose factors multiply below the smallest scale the
        // histogram keeps: it folds, and keeps counting.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(1.0);
        c.bins = Some(given_edges(vec![0.0]));
        let mut m = Marginal::new(c).unwrap();
        OnlineModel::step(&mut m, &[-1.0], &[Some(1.0)], 0.0, 1.0);
        OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 465.0, 1.0);
        let p = m.pair(0, 0);
        assert!(
            (sum(&p) - p.n_eff).abs() < 1e-9,
            "{} vs {}",
            sum(&p),
            p.n_eff
        );
        OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 700.0, 1.0);
        for _ in 0..50 {
            OnlineModel::step(&mut m, &[1.0], &[Some(2.0)], 1.0, 1.0);
        }
        let p = m.pair(0, 0);
        assert!(p.n_eff > 1.99, "{}", p.n_eff);
        assert!(
            (sum(&p) - p.n_eff).abs() < 1e-9,
            "{} vs {}",
            sum(&p),
            p.n_eff
        );
        assert_eq!(p.bin_n[0], 0.0, "the first row is below f64 range");
        assert!((p.bin_mean_y[1] - 2.0).abs() < 1e-9, "{}", p.bin_mean_y[1]);
        assert!(p.bin_var_y[1] < 1e-9, "{}", p.bin_var_y[1]);
    }

    /// A cut that explains everything: `split_gain` is exactly one, and its
    /// statistic is `+inf`, as `t` is at `corr = ±1`.
    #[test]
    fn a_perfect_split_has_an_infinite_statistic() {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY);
        c.bins = Some(given_edges(vec![0.0]));
        let mut m = Marginal::new(c).unwrap();
        for i in 0..100 {
            let x = if i % 2 == 0 { -1.0 } else { 1.0 };
            let y = if x > 0.0 { 1.0 } else { 0.0 };
            OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
        }
        let p = m.pair(0, 0);
        assert_eq!(p.split_gain, 1.0);
        assert_eq!(p.split_at, 0.0);
        assert_eq!(p.split_gain_t, f64::INFINITY);
        // The same target is a perfect line in `x`, and `t` agrees.
        assert_eq!(p.corr, 1.0);
        assert_eq!(p.t, f64::INFINITY);
    }

    // -----------------------------------------------------------------
    // Sharded steps (docs/PLAN.md task 126).
    // -----------------------------------------------------------------

    /// One row of a stream that exercises everything a held row carries:
    /// three targets, two of them often absent, weights of zero, clock gaps
    /// long enough to fold the bins' scale and one that empties everything,
    /// and the lags' ring cleared now and then.
    struct ShardRow {
        x: Vec<f64>,
        y: Vec<Option<f64>>,
        d: f64,
        w: f64,
        clear: bool,
    }

    fn shard_stream(n: usize, p: usize) -> Vec<ShardRow> {
        let mut seed = 2026u64;
        let mut level = vec![0.0_f64; p];
        (0..n)
            .map(|i| {
                for (j, v) in level.iter_mut().enumerate() {
                    *v = 0.9 * *v + lcg(&mut seed) + if j == 2 { 1e6 } else { 0.0 } * 1e-6;
                }
                // Feature 2 sits at a level, feature 4 holds one value for
                // long runs: the cases the compensated means and the runs are
                // for.
                let mut x = level.clone();
                x[2] += 1e6;
                if p > 4 {
                    x[4] = if (i / 40) % 2 == 0 { 3.0 } else { x[4] };
                }
                let y0 = x[0] - 0.5 * x[1] + 0.3 * lcg(&mut seed);
                let y = vec![
                    Some(y0),
                    (lcg(&mut seed) > -0.4).then(|| 2.0 * x[1] + lcg(&mut seed)),
                    (lcg(&mut seed) > 0.0).then(|| (x[0] * x[3]).abs() + 0.1 * lcg(&mut seed)),
                ];
                let d = match i {
                    0 => 0.0,
                    _ if i % 211 == 7 => 1e6,
                    _ if i % 53 == 11 => 300.0,
                    _ => 1.0,
                };
                let w = if i % 13 == 5 {
                    0.0
                } else {
                    0.5 + (lcg(&mut seed) + 1.0) / 2.0
                };
                ShardRow {
                    x,
                    y,
                    d,
                    w,
                    clear: i % 97 == 50,
                }
            })
            .collect()
    }

    /// Every configuration a held row must carry exactly.
    fn shard_cfgs(p: usize) -> Vec<(&'static str, MarginalCfg)> {
        let base = || {
            let mut c = cfg(p, 3);
            c.decay = Decay::Halflife(15.0);
            c.min_periods = vec![3.0; 3];
            c
        };
        let with = |f: &dyn Fn(&mut MarginalCfg)| {
            let mut c = base();
            f(&mut c);
            c
        };
        let learned = |c: &mut MarginalCfg| {
            c.bins = Some(Box::new(crate::BinCfg {
                n_bins: 4,
                edges: None,
                rule: crate::BinRule::Quantile,
                warm_rows: 40,
                budget_mib: None,
            }))
        };
        vec![
            ("moments", base()),
            (
                "lags",
                with(&|c| {
                    c.lags = vec![1, 3, 8];
                    c.serial_rule = Some(SerialRule::Geometric);
                }),
            ),
            (
                "cross lags",
                with(&|c| {
                    c.lags = vec![1, 3, 8];
                    c.cross_lags = Some(vec![3]);
                }),
            ),
            (
                "no cross lags",
                with(&|c| {
                    c.lags = vec![1, 2];
                    c.cross_lags = Some(vec![]);
                }),
            ),
            ("learned bins", with(&learned)),
            (
                "given bins",
                with(&|c| {
                    c.bins = Some(Box::new(crate::BinCfg {
                        n_bins: 3,
                        edges: Some((0..p).map(|j| vec![-0.5 + j as f64 * 0.1, 0.5]).collect()),
                        rule: crate::BinRule::Quantile,
                        warm_rows: 2,
                        budget_mib: None,
                    }))
                }),
            ),
            // A window takes neither lags nor bins (`MarginalCfg::validate`).
            ("window", with(&|c| c.window = Some(40.0))),
            (
                "window every 3",
                with(&|c| {
                    c.window = Some(25.0);
                    c.window_every = Some(3);
                }),
            ),
        ]
    }

    fn state_bytes(m: &Marginal) -> Vec<u8> {
        rmp_serde::to_vec(&m.state()).unwrap()
    }

    /// Each shard on a thread of its own.
    fn threaded(shards: &mut [MarginalShard<'_>]) {
        std::thread::scope(|s| {
            for sh in shards.iter_mut() {
                s.spawn(move || sh.run());
            }
        });
    }

    /// The last shard first: the order a pool runs them in is its own.
    fn reversed(shards: &mut [MarginalShard<'_>]) {
        shards.iter_mut().rev().for_each(MarginalShard::run);
    }

    /// The whole point of the split (docs/PLAN.md task 126): whatever the
    /// shard count, whichever threads run the shards and in what order, and
    /// wherever the held rows are flushed, a sharded model is the unsplit
    /// one to the bit -- every row's `n_eff`, and every byte of the state,
    /// which holds every pair, lag, bin, run and window snapshot. Past a
    /// batch (256 rows), across window snapshots, bin folds, a gap that
    /// empties everything and a cleared ring.
    #[test]
    fn a_sharded_step_is_the_unsplit_step_to_the_bit() {
        let p = 7;
        let rows = shard_stream(700, p);
        let runners: [(&str, &ShardRunner); 3] = [
            ("in order", &run_in_order),
            ("threads", &threaded),
            ("reversed", &reversed),
        ];
        for (name, c) in shard_cfgs(p) {
            let mut plain = Marginal::new(c.clone()).unwrap();
            let steps: Vec<Step> = rows
                .iter()
                .map(|r| {
                    if r.clear {
                        OnlineModel::clear_lags(&mut plain);
                    }
                    OnlineModel::step(&mut plain, &r.x, &r.y, r.d, r.w)
                })
                .collect();
            let want = state_bytes(&plain);
            for count in [2, 3, p, 50] {
                for (how, run) in runners {
                    let shards = Shards { count, run };
                    let mut m = Marginal::new(c.clone()).unwrap();
                    let mut held = 0;
                    for (i, r) in rows.iter().enumerate() {
                        if r.clear {
                            OnlineModel::clear_lags(&mut m);
                        }
                        let got = m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
                        assert_eq!(
                            got.n_eff.to_bits(),
                            steps[i].n_eff.to_bits(),
                            "{name}, {count} shards {how}: n_eff at row {i}"
                        );
                        held = held.max(m.held_rows());
                        // A flush wherever the caller likes, and the state
                        // there is the unsplit one's there.
                        if i == 333 {
                            m.flush(&shards);
                            let mut part = Marginal::new(c.clone()).unwrap();
                            for r in &rows[..=i] {
                                if r.clear {
                                    OnlineModel::clear_lags(&mut part);
                                }
                                OnlineModel::step(&mut part, &r.x, &r.y, r.d, r.w);
                            }
                            assert!(
                                state_bytes(&m) == state_bytes(&part),
                                "{name}, {count} shards {how}: the state at row {i} differs"
                            );
                        }
                    }
                    // A window snapshot every row flushes every row.
                    let most = if name == "window" { 1 } else { 2 };
                    assert!(held >= most, "{name}: rows were held ({held})");
                    m.flush(&shards);
                    assert_eq!(m.held_rows(), 0);
                    assert!(
                        state_bytes(&m) == want,
                        "{name}, {count} shards {how}: the state differs"
                    );
                }
            }
        }
    }

    /// The stream above reaches what it is meant to: rows held past a
    /// batch, window snapshots taken while rows were held, bins that fold
    /// and empty, a ring cleared with rows held -- each counted, since a
    /// comparison of two models that never met a case says nothing about
    /// it.
    #[test]
    fn the_sharded_stream_reaches_every_case() {
        let p = 7;
        let rows = shard_stream(700, p);
        let shards = Shards {
            count: 3,
            run: &run_in_order,
        };
        let cfgs = shard_cfgs(p);
        let find = |name: &str| cfgs.iter().find(|(n, _)| *n == name).unwrap().1.clone();
        for name in ["window", "window every 3"] {
            let mut m = Marginal::new(find(name)).unwrap();
            let mut snapshots_while_held = 0;
            for r in &rows {
                let win = m.win.as_ref().unwrap();
                if m.held_rows() > 0 && win.snaps.takes(win.clock + r.d) {
                    snapshots_while_held += 1;
                }
                m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
            }
            assert!(
                snapshots_while_held > 100,
                "{name}: snapshots with rows held: {snapshots_while_held}"
            );
        }
        let mut m = Marginal::new(find("learned bins")).unwrap();
        let (mut folds, mut empties, mut folds_while_held) = (0, 0, 0);
        for r in &rows {
            let lam = m.cfg.decay.factor(r.d);
            if let Some(h) = m.bins.as_ref().and_then(|b| b.hist.as_ref()) {
                if h.folds_at(lam) {
                    folds += 1;
                    empties += usize::from(lam == 0.0);
                    folds_while_held += usize::from(m.held_rows() > 0);
                }
            }
            m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
        }
        assert!(
            m.bins.as_ref().unwrap().hist.is_some(),
            "the edges were learned"
        );
        assert!(
            folds > 2 && empties > 1 && folds_while_held == folds,
            "folds {folds}, of them empties {empties}, with rows held {folds_while_held}"
        );
        // Without a window, the batch fills, and a clear meets held rows.
        let mut lagged = Marginal::new(find("lags")).unwrap();
        let (mut most_held, mut cleared_while_held) = (0, 0);
        for r in &rows {
            if r.clear && lagged.held_rows() > 0 {
                cleared_while_held += 1;
            }
            if r.clear {
                OnlineModel::clear_lags(&mut lagged);
            }
            lagged.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
            most_held = most_held.max(lagged.held_rows());
        }
        assert_eq!(most_held, batch_rows(p) - 1, "a full batch is flushed");
        assert!(
            cleared_while_held > 2,
            "clears with rows held: {cleared_while_held}"
        );
    }

    /// A plain step after sharded ones learns the held rows first, and a
    /// state saved after a flush under one count continues under another,
    /// or none: the count is not in the state.
    #[test]
    fn the_shard_count_is_not_in_the_state() {
        let p = 7;
        let rows = shard_stream(500, p);
        let cfgs = shard_cfgs(p);
        let c = &cfgs.iter().find(|(n, _)| *n == "lags").unwrap().1;
        let mut plain = Marginal::new(c.clone()).unwrap();
        for r in &rows {
            if r.clear {
                OnlineModel::clear_lags(&mut plain);
            }
            OnlineModel::step(&mut plain, &r.x, &r.y, r.d, r.w);
        }
        let five = Shards {
            count: 5,
            run: &run_in_order,
        };
        let two = Shards {
            count: 2,
            run: &threaded,
        };
        let mut m = Marginal::new(c.clone()).unwrap();
        for (i, r) in rows.iter().enumerate() {
            if r.clear {
                OnlineModel::clear_lags(&mut m);
            }
            match i {
                // Rows held, then a plain step: it learns them first.
                0..150 => {
                    m.step_sharded(&r.x, &r.y, r.d, r.w, &five);
                }
                150..160 => {
                    OnlineModel::step(&mut m, &r.x, &r.y, r.d, r.w);
                }
                // Saved and restored with rows flushed, then another count.
                160 => {
                    m.flush(&five);
                    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                    m = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                    m.step_sharded(&r.x, &r.y, r.d, r.w, &two);
                }
                _ => {
                    m.step_sharded(&r.x, &r.y, r.d, r.w, &two);
                }
            }
        }
        m.flush(&two);
        assert!(state_bytes(&m) == state_bytes(&plain));
    }

    /// One shard or fewer is the unsplit step, row by row: nothing is held.
    #[test]
    fn one_shard_holds_nothing() {
        let rows = shard_stream(50, 7);
        let mut m = Marginal::new(cfg(7, 3)).unwrap();
        for r in &rows {
            for count in [0, 1] {
                let shards = Shards {
                    count,
                    run: &run_in_order,
                };
                m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
                assert_eq!(m.held_rows(), 0);
            }
        }
    }

    /// Reading a pair with rows held is a caller's bug, and says so in a
    /// debug build rather than reporting pairs that miss those rows.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "flushed before a pair is read")]
    fn a_pair_read_with_rows_held_is_refused() {
        let mut m = Marginal::new(cfg(4, 1)).unwrap();
        let shards = Shards {
            count: 2,
            run: &run_in_order,
        };
        m.step_sharded(&[1.0, 2.0, 3.0, 4.0], &[Some(1.0)], 0.0, 1.0, &shards);
        let _ = m.pair(0, 0);
    }

    /// `"auto"` splits a row only when a flush has work for two shards,
    /// never into more than twice the threads, and lags, cross terms and
    /// bins add to the work (docs/PERFORMANCE.md §25).
    #[test]
    fn auto_splits_only_what_fills_two_shards() {
        let shape =
            |p: usize, t: usize, lags: Vec<usize>, cross: Option<Vec<usize>>, bins: bool| {
                let mut c = cfg(p, t);
                c.lags = lags;
                c.cross_lags = cross;
                c.bins = bins.then(|| bins_cfg(16, 1_000));
                c
            };
        let moments = |p, t| shape(p, t, vec![], None, false);
        // Measured not to pay: split, 1,000 features ran 1.15 times as
        // fast at best and slower past two shards.
        assert_eq!(moments(1_000, 1).auto_shards(14), 1);
        assert_eq!(moments(4_000, 1).auto_shards(14), 6);
        assert_eq!(moments(10_000, 9).auto_shards(14), 28, "twice the threads");
        assert_eq!(moments(10_000, 9).auto_shards(1), 1, "one thread, no split");
        assert_eq!(moments(10_000, 9).auto_shards(0), 1);
        let lags = |cross| shape(1_000, 1, vec![1, 2, 5, 10, 20, 50], cross, false);
        assert!(lags(Some(vec![1])).auto_shards(14) > 1);
        assert!(lags(None).auto_shards(14) > lags(Some(vec![1])).auto_shards(14));
        assert!(
            shape(300, 1, vec![], None, true).auto_shards(14) > 1,
            "bins are work"
        );
        // The batch holds fewer rows of a wider row: the work per flush is
        // what counts.
        assert_eq!(batch_rows(10_000), 104);
        assert_eq!(batch_rows(1_000), BATCH_ROWS);
    }

    /// The warm-up hold's budget is what the held rows take, to the byte:
    /// reserved at exactly the rows it will hold, again after a restore
    /// partway through, with rows of weight zero taking nothing. And the
    /// learned histogram that follows stays inside its budget
    /// (docs/PLAN.md task 129).
    #[test]
    fn the_hold_budget_is_what_the_held_rows_take() {
        use crate::margbins::{HeldRow, histogram_bytes, hold_bytes};
        // Not a power of two, nor twice the restore point: a hold grown by
        // doubling cannot land on it by chance.
        let (p, t, warm, n_bins) = (9, 3, 37, 4);
        let held_bytes = |m: &Marginal| -> usize {
            let b = m.bins.as_ref().unwrap();
            b.held.capacity() * std::mem::size_of::<HeldRow>()
                + b.held
                    .iter()
                    .map(|r| {
                        r.x.capacity() * std::mem::size_of::<f64>()
                            + r.y.capacity() * std::mem::size_of::<Option<f64>>()
                    })
                    .sum::<usize>()
        };
        let mut c = cfg(p, t);
        c.bins = Some(bins_cfg(n_bins, warm));
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 5u64;
        let (mut held, mut i, mut restored) = (0, 0, false);
        let row = |seed: &mut u64| -> (Vec<f64>, Vec<Option<f64>>) {
            let x = (0..p).map(|_| lcg(seed)).collect();
            let y = (0..t).map(|_| Some(lcg(seed))).collect();
            (x, y)
        };
        while held < warm - 1 {
            let (x, y) = row(&mut seed);
            let w = if i % 5 == 2 { 0.0 } else { 1.0 };
            held += usize::from(w > 0.0);
            OnlineModel::step(&mut m, &x, &y, step_clock(i), w);
            i += 1;
            if held == 20 && !restored {
                // A restored hold has no spare room; the next push reserves.
                let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                m = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                restored = true;
            }
        }
        let b = m.bins.as_ref().unwrap();
        assert!(b.hist.is_none(), "the edges are not fixed yet");
        assert_eq!(b.held.len(), warm - 1);
        assert_eq!(
            b.held.capacity(),
            warm,
            "reserved at exactly the rows it holds"
        );
        // One row short of full: the hold's budget less that row's vectors.
        let last = p * std::mem::size_of::<f64>() + t * std::mem::size_of::<Option<f64>>();
        assert_eq!(held_bytes(&m) + last, hold_bytes(warm, p, t));
        // The next row fixes the edges and lets the hold go.
        let (x, y) = row(&mut seed);
        OnlineModel::step(&mut m, &x, &y, step_clock(i), 1.0);
        let b = m.bins.as_ref().unwrap();
        assert!(b.held.is_empty() && b.held.capacity() == 0);
        let hist = b.hist.as_ref().unwrap();
        assert!(hist.heap_bytes() <= histogram_bytes(p, t, p * n_bins));
    }

    /// A shard is sent to a pool's threads.
    #[test]
    fn a_shard_can_be_sent() {
        fn send<T: Send>() {}
        send::<MarginalShard<'_>>();
    }

    /// The window's snapshot counts every vector it holds in its footprint,
    /// the per-target row counts included (docs/PLAN.md task 130).
    #[test]
    fn the_window_footprint_counts_every_vector() {
        let snap = |p: usize, t: usize| {
            let mut c = cfg(p, t);
            c.window = Some(10.0);
            let mut m = Marginal::new(c).unwrap();
            let mut s = 9u64;
            for i in 0..30 {
                let x: Vec<f64> = (0..p).map(|_| lcg(&mut s)).collect();
                let y: Vec<Option<f64>> = (0..t)
                    .map(|k| (k == 0 || i % 3 != 1).then(|| lcg(&mut s)))
                    .collect();
                m.step(&x, &y, step_clock(i), 1.0);
            }
            m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
        };
        crate::window::assert_footprint_counts_every_vector(&snap(2, 1), &snap(5, 3), "marginal");
    }
}
