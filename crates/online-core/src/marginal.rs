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
    /// omitted lags are negative, which is why it is not the default.
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
        if let Some(bad) = self
            .min_periods
            .iter()
            .find(|v| v.is_nan() || v.is_infinite() || **v < 0.0)
        {
            return Err(format!(
                "marginal: min_periods must be finite and >= 0, got {bad}"
            ));
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
        if self.window.is_none() && self.window_every.is_some() {
            return Err("marginal: window_every needs `window`".into());
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
    /// `rho_y(l)`, the target's, and the two cross-correlations —
    /// `lagcorr_xy` is the feature *now* against the target `l` rows ago,
    /// `lagcorr_yx` the target now against the feature `l` rows ago. Empty
    /// without `lags`.
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
    /// The hard-cutoff window, when the spec asks for one. Last, for the
    /// reason `MarginalCfg::window` gives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
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
            ((1.0 + 2.0 * sum).max(f64::MIN_POSITIVE), f64::NAN, f64::NAN)
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
/// known before the first row. The price is memory, which
/// `BinCfg::validate` bounds up front.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Binned {
    /// `None` until the edges are fixed: either at construction, when they
    /// were given, or at the `warm_rows`-th learned row.
    hist: Option<crate::MarginalBins>,
    /// Warm-up rows in arrival order. Empty once the edges exist.
    held: Vec<HeldRow>,
    /// Decay from zero-weight rows since the last held row, which teach
    /// nothing but still age the histogram. Folded into the next held row so
    /// a run of them cannot grow `held`.
    pending_lam: f64,
}

/// One warm-up row, kept whole so the replay can be exact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct HeldRow {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    lam: f64,
    w: f64,
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
            for (t, yt) in row.y.iter().enumerate().take(n_targets) {
                if let Some(v) = yt {
                    hist.update_target(t, &row.x, *v, row.w);
                }
            }
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
            )?))
        };
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
            win,
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
            if weight > 0.0 && weight.is_finite() {
                for (t, yt) in y.iter().enumerate().take(self.cfg.n_targets) {
                    if let Some(v) = yt {
                        hist.update_target(t, x, *v, weight);
                    }
                }
            }
            return;
        }
        if weight > 0.0 && weight.is_finite() {
            b.held.push(HeldRow {
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
    /// `A(t) - f·A(u)` back through raw moments and re-centred, which is the
    /// same identity `EwCov` uses, one pair at a time so a readout stays O(1).
    fn cut(
        w: f64,
        w_old: f64,
        f: f64,
        (ma, mb): (f64, f64),
        (ma_old, mb_old): (f64, f64),
        s: f64,
        s_old: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let wn = w - f * w_old;
        if wn <= 0.0 || !wn.is_finite() {
            return None;
        }
        let a = (w * ma - f * w_old * ma_old) / wn;
        let b = (w * mb - f * w_old * mb_old) / wn;
        let raw = w * (s + ma * mb) - f * w_old * (s_old + ma_old * mb_old);
        Some((wn, a, b, raw / wn - a * b))
    }

    /// The accumulated weight the pairs are read from: under a `window`, the
    /// weight inside it.
    pub fn n_eff(&self) -> f64 {
        match self.boundary() {
            Some((old, f)) => (self.w_sum - f * old.w_sum).max(0.0),
            None => self.w_sum,
        }
    }

    /// `W_t`, the weight behind target `t`'s pairs.
    pub fn target_weight(&self, t: usize) -> f64 {
        match self.boundary() {
            Some((old, f)) => (self.wt[t] - f * old.wt[t]).max(0.0),
            None => self.wt[t],
        }
    }

    /// The statistics of feature `j` against target `t`. The moments are
    /// reported at any weight; `corr`, `beta` and `t` wait for
    /// `min_periods`.
    pub fn pair(&self, t: usize, j: usize) -> Pair {
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
                        (w, w * w / q, mx, my, sxx.max(0.0), syy.max(0.0), sxy)
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
            for li in 0..lag.lags().len() {
                let norm = |v: f64, d: f64| if d > 0.0 { v / d } else { f64::NAN };
                lagcorr_xx.push(norm(lag.cxx(li, t, j), sd_x * sd_x));
                lagcorr_yy.push(norm(lag.cyy(li, t), sd_y * sd_y));
                lagcorr_xy.push(norm(lag.cxy(li, t, j), sd_x * sd_y));
                lagcorr_yx.push(norm(lag.cyx(li, t, j), sd_x * sd_y));
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
                    my: self.my[t],
                    a: lam * self.wt[t] / w_new,
                    b: w / w_new,
                },
            );
        }
    }

    fn learn(&mut self, x: &[f64], y: &[Option<f64>], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        debug_assert!(w >= 0.0, "marginal requires a non-negative weight, got {w}");
        if w < 0.0 {
            return;
        }
        let p = self.cfg.n_features;
        // The model-level weight: every row, present targets or not. A
        // zero-weight first row leaves it at zero, which is legal (rule 9).
        self.w_sum = lam * self.w_sum + w;
        for (t, yt) in y.iter().enumerate() {
            let Some(yt) = *yt else {
                // Time passes for a target that is not there: its weight
                // ages, its moments hold, as `ew_ridge` treats a missing
                // target.
                self.wt[t] *= lam;
                self.qt[t] *= lam * lam;
                continue;
            };
            let w_new = lam * self.wt[t] + w;
            if w_new <= 0.0 {
                // No weight in the history and none on this row: nothing to
                // average, and `a`/`b` would be 0/0 (CLAUDE.md rule 9). The
                // weights still take the row's decay -- `lam = 0` with a
                // zero-weight row is a clock gap past `max_dclock` on a row
                // that teaches nothing, and the target's `n_eff` must not
                // outlive the gap while the model's does not.
                self.wt[t] = 0.0;
                self.qt[t] = 0.0;
                continue;
            }
            let a = lam * self.wt[t] / w_new;
            let b = w / w_new;
            let dy = yt - self.my[t];
            // Co-moments from the deviations against the OLD means, then the
            // means advance -- `EwCov::update`'s order, operation for
            // operation, so the pair agrees with `ew_cov` to the bit.
            self.syy[t] = a * self.syy[t] + a * b * dy * dy;
            let row = t * p;
            for (i, xj) in (row..row + p).zip(x) {
                let dx = xj - self.mx[i];
                self.sxx[i] = a * self.sxx[i] + a * b * dx * dx;
                self.sxy[i] = a * self.sxy[i] + a * b * dx * dy;
            }
            for (mi, xj) in self.mx[row..row + p].iter_mut().zip(x) {
                *mi += b * (xj - *mi);
            }
            self.my[t] += b * dy;
            self.wt[t] = w_new;
            self.qt[t] = lam * lam * self.qt[t] + w * w;
        }
    }
}

impl OnlineModel for Marginal {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        // The snapshot is every accumulator as it stands *before* this row,
        // decayed to this row's clock, so subtracting it later retains this
        // row and everything after. Keyed by a clock the model accumulates
        // itself, so the boundary cannot depend on the chunking.
        if let Some(win) = self.win.as_mut() {
            let t = win.clock + d_clock;
            let snap = MarginalMoments {
                w_sum: self.w_sum * lam,
                wt: self.wt.iter().map(|w| w * lam).collect(),
                qt: self.qt.iter().map(|q| q * lam * lam).collect(),
                my: self.my.clone(),
                syy: self.syy.clone(),
                mx: self.mx.clone(),
                sxx: self.sxx.clone(),
                sxy: self.sxy.clone(),
            };
            win.snaps.offer(t, || snap);
            win.clock = t;
            win.snaps.trim(t);
        }
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

    /// No prediction slots: the step reports `n_eff` and nothing else.
    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        Step {
            pred: Vec::new(),
            n_eff: self.w_sum,
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
        if let Some(lag) = self.lag.as_mut() {
            lag.clear();
        }
    }

    fn state(&self) -> State {
        State::new(crate::ModelState::Marginal(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Marginal(m) => Ok((**m).clone()),
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
            bins: None,
            window: None,
            window_every: None,
        }
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

    /// A standard normal from the module's LCG, by Box-Muller.
    fn normal(state: &mut u64) -> f64 {
        let u1 = (lcg(state) + 1.0) / 2.0;
        let u2 = (lcg(state) + 1.0) / 2.0;
        let u1 = u1.max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
    /// inside the window and nothing else.
    #[test]
    fn a_windowed_pair_is_the_pair_of_the_rows_inside_the_window() {
        let (halflife, window) = (25.0, 70.0);
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(halflife);
        c.window = Some(window);
        let mut m = Marginal::new(c).unwrap();

        let mut seed = 99u64;
        let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
        for i in 0..140 {
            // `lcg` is [-1, 1): a clock increment must be its magnitude, or
            // the clock runs backwards and the window means nothing.
            let d = if i == 0 {
                0.0
            } else {
                0.5 + 2.0 * lcg(&mut seed).abs()
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
                let keep: Vec<usize> = (0..=i).filter(|&j| now - t[j] < window).collect();
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
        c.min_periods = vec![f64::INFINITY];
        err(c.clone(), "min_periods");
        c.min_periods = vec![f64::NAN];
        err(c.clone(), "min_periods");
        c.min_periods = vec![1.0, 2.0];
        err(c.clone(), "2 entries for 1 targets");
        c.min_periods = vec![];
        err(c, "0 entries for 1 targets");
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
}
