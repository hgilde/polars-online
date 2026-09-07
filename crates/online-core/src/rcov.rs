//! `rcov`: a block's realised covariance, robust to microstructure noise
//! (docs/ENHANCEMENTS.md E57).
//!
//! A plain realised covariance over ticks is biased by noise (each price is
//! the efficient one plus an error, and the error's variance accumulates with
//! every tick) and attenuated by asynchrony (the Epps effect). Both are
//! estimated away by published estimators that are **sums over lags**, which
//! is exactly what a stream can accumulate.
//!
//! Rows are **returns**: the caller differences upstream, which for a
//! refresh-time grid ([`crate`]'s sibling `po.prep.refresh_time`) is
//! `.diff().over(by)`. There is no decay and no per-row output -- the model's
//! value is its state at the group's close, and the block is what a
//! `group_close` row carries.
//!
//! # The three kinds
//!
//! **`plain`** is `Σⱼ xⱼxⱼ'`, accumulated as it arrives. At close it equals
//! `n ×` an `ew_cov(lam = 1)`'s uncentred second moment, to the bit, which is
//! both the cross-check and the reference the other two are measured against:
//! the difference is the noise and the attenuation.
//!
//! **`kernel`** is the multivariate realised kernel of Barndorff-Nielsen,
//! Hansen, Lunde & Shephard (2011):
//!
//! ```text
//! K = Σ_{h=−H}^{H} k(h/(H+1))·Γ̂_h        Γ̂_h = Σ_{j=|h|+1}^{n} xⱼ x'_{j−h}
//! Γ̂_{−h} = Γ̂_h'                          k = Parzen
//! ```
//!
//! Parzen because it is a positive-definite function (so `K` is PSD up to
//! rounding) and because BNHLS measure its efficiency at 0.97 against the
//! quadratic spectral's 0.93; the Bartlett kernel is *not* consistent for
//! this estimator, so it is not offered. The end points are jittered by
//! averaging the first and last `jitter` observations, which is what makes
//! the estimator's end effects vanish -- their `m = 1` is mean-square optimal
//! and `m = 1..4` moves the estimate by under 0.5 %, so the default of 2 is
//! immaterial and the docstring says so.
//!
//! **`preavg`** is the modulated realised covariance of Christensen,
//! Kinnebrock & Podolskij (2010): the returns are pre-averaged over a window
//! of `kₙ = ⌊θ√n⌋` with the weight function `g(x) = min(x, 1−x)`, which
//! averages the noise away, and the residual bias is subtracted.
//!
//! # Nothing reads a future row
//!
//! The jittered **end** point is formed at close from observations already in
//! state, and a product enters `Γ̂_h` only once both its legs are final -- a
//! return `m` rows back can no longer turn out to be part of the end jitter.
//! That is what makes the state a ring of `max_bandwidth + m` vectors, the close
//! `O(max_bandwidth·k²)`, and the whole thing chunk-invariant.
//!
//! # A break inside a block
//!
//! Nothing here decays, so `halflife` is refused -- but a clock still
//! matters: a gap over `max_dclock`, or a session change, says the returns
//! on either side of it are not adjacent, and a covariance of adjacent
//! returns is the whole statistic. Such a break splits the group into
//! **stretches** ([`crate::OnlineModel::clear_lags`]). Each stretch is
//! closed as the group's last one is -- leading jitter, interior, trailing
//! jitter -- and `Γ̂_h` is the sum over stretches, so no product ever pairs
//! two returns across the break. A stretch that ends before its tail is
//! full contributes only what it had already emitted.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Which estimator; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RcovKind {
    Plain,
    Kernel,
    Preavg,
}

impl RcovKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RcovKind::Plain => "plain",
            RcovKind::Kernel => "kernel",
            RcovKind::Preavg => "preavg",
        }
    }
}

/// The Parzen kernel, `k(0) = 1` and `k(x) = 0` beyond 1.
#[inline]
pub fn parzen(x: f64) -> f64 {
    let x = x.abs();
    if x <= 0.5 {
        1.0 - 6.0 * x * x + 6.0 * x * x * x
    } else if x <= 1.0 {
        let u = 1.0 - x;
        2.0 * u * u * u
    } else {
        0.0
    }
}

/// The pre-averaging weight `g(x) = min(x, 1−x)` on `[0, 1]`.
#[inline]
pub fn preavg_g(x: f64) -> f64 {
    x.min(1.0 - x)
}

/// CKP's finite-sample `ψ₁^k = k·Σ_{i=1}^{k}(g(i/k) − g((i−1)/k))²`; its
/// limit is 1.
pub fn psi1(k: usize) -> f64 {
    let kf = k as f64;
    kf * (1..=k)
        .map(|i| {
            let d = preavg_g(i as f64 / kf) - preavg_g((i as f64 - 1.0) / kf);
            d * d
        })
        .sum::<f64>()
}

/// CKP's finite-sample `ψ₂^k = (1/k)·Σ_{i=1}^{k−1} g(i/k)²`; its limit is
/// `1/12`.
pub fn psi2(k: usize) -> f64 {
    let kf = k as f64;
    (1..k).map(|i| preavg_g(i as f64 / kf).powi(2)).sum::<f64>() / kf
}

/// `φ₁^k(j) = Σ_{i=j+1}^{k−1}(g((i−1)/k) − g(i/k))·(g((i−j−1)/k) − g((i−j)/k))`.
pub fn phi1(k: usize, j: usize) -> f64 {
    let kf = k as f64;
    let g = |i: usize| preavg_g(i as f64 / kf);
    (j + 1..k)
        .map(|i| (g(i - 1) - g(i)) * (g(i - j - 1) - g(i - j)))
        .sum()
}

/// `φ₂^k(j) = Σ_{i=j+1}^{k−1} g(i/k)·g((i−j)/k)`.
pub fn phi2(k: usize, j: usize) -> f64 {
    let kf = k as f64;
    let g = |i: usize| preavg_g(i as f64 / kf);
    (j + 1..k).map(|i| g(i) * g(i - j)).sum()
}

/// `Φ₁₁^k = k·(Σⱼ φ₁(j)² − ½φ₁(0)²)`; its limit is `1/6`.
pub fn phi_11(k: usize) -> f64 {
    let s: f64 = (0..k).map(|j| phi1(k, j).powi(2)).sum();
    k as f64 * (s - 0.5 * phi1(k, 0).powi(2))
}

/// `Φ₁₂^k = (1/k)·(Σⱼ φ₁(j)φ₂(j) − ½φ₁(0)φ₂(0))`; its limit is `1/96`.
pub fn phi_12(k: usize) -> f64 {
    let s: f64 = (0..k).map(|j| phi1(k, j) * phi2(k, j)).sum();
    (s - 0.5 * phi1(k, 0) * phi2(k, 0)) / k as f64
}

/// `Φ₂₂^k = (1/k³)·(Σⱼ φ₂(j)² − ½φ₂(0)²)`; its limit is `151/80640`.
pub fn phi_22(k: usize) -> f64 {
    let s: f64 = (0..k).map(|j| phi2(k, j).powi(2)).sum();
    (s - 0.5 * phi2(k, 0).powi(2)) / (k as f64).powi(3)
}

/// BNHLS's Parzen constant `c* = ((12)²/0.269)^{1/5}`.
pub fn parzen_c_star() -> f64 {
    (144.0f64 / 0.269).powf(0.2)
}

/// Realised variance and quarticity on a subsampled grid, averaged over the
/// stride's offsets: `stride = 1` is every return.
///
/// Offset `o` accumulates returns `o, o + 1, ...` into `stride`-step sums;
/// each completed sum contributes its square to the variance and its fourth
/// power to the quarticity. Averaging over offsets is BNHLS's own recipe
/// (their §4.1) and is what makes a sparse estimate use all the data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StridedRv {
    stride: usize,
    k: usize,
    /// `stride * k` partial sums.
    partial: Vec<f64>,
    /// `stride * k` sums of squares, and of fourth powers.
    rv: Vec<f64>,
    rq: Vec<f64>,
    /// Completed steps per offset.
    counts: Vec<u64>,
    /// Returns fed so far.
    t: u64,
}

impl StridedRv {
    fn new(k: usize, stride: usize) -> Self {
        Self {
            stride,
            k,
            partial: vec![0.0; stride * k],
            rv: vec![0.0; stride * k],
            rq: vec![0.0; stride * k],
            counts: vec![0; stride],
            t: 0,
        }
    }

    fn push(&mut self, x: &[f64]) {
        let (s, k) = (self.stride, self.k);
        for o in 0..s {
            if self.t < o as u64 {
                continue;
            }
            for (i, xi) in x.iter().enumerate().take(k) {
                self.partial[o * k + i] += xi;
            }
            if (self.t - o as u64 + 1) % s as u64 == 0 {
                for i in 0..k {
                    let v = self.partial[o * k + i];
                    self.rv[o * k + i] += v * v;
                    self.rq[o * k + i] += v * v * v * v;
                    self.partial[o * k + i] = 0.0;
                }
                self.counts[o] += 1;
            }
        }
        self.t += 1;
    }

    /// Mean over offsets of that offset's realised variance, per feature.
    fn mean_rv(&self) -> Vec<f64> {
        self.mean(&self.rv, 1.0)
    }

    /// Mean over offsets of `RV / (2n)`, per feature: BNHLS's noise-variance
    /// estimate, deliberately biased upward (their §4.1).
    fn noise(&self) -> Vec<f64> {
        let k = self.k;
        let mut out = vec![0.0; k];
        let mut live = 0.0;
        for o in 0..self.stride {
            if self.counts[o] == 0 {
                continue;
            }
            live += 1.0;
            let d = 2.0 * self.counts[o] as f64;
            for (i, v) in out.iter_mut().enumerate() {
                *v += self.rv[o * k + i] / d;
            }
        }
        if live == 0.0 {
            return vec![f64::NAN; k];
        }
        out.iter().map(|v| v / live).collect()
    }

    /// Mean over offsets of `(n/3)·Σ x⁴`, per feature: an integrated
    /// quarticity proxy, and labelled one.
    fn quarticity(&self) -> Vec<f64> {
        let k = self.k;
        let mut out = vec![0.0; k];
        let mut live = 0.0;
        for o in 0..self.stride {
            if self.counts[o] == 0 {
                continue;
            }
            live += 1.0;
            let n = self.counts[o] as f64;
            for (i, v) in out.iter_mut().enumerate() {
                *v += n / 3.0 * self.rq[o * k + i];
            }
        }
        if live == 0.0 {
            return vec![f64::NAN; k];
        }
        out.iter().map(|v| v / live).collect()
    }

    fn mean(&self, from: &[f64], scale: f64) -> Vec<f64> {
        let k = self.k;
        let mut out = vec![0.0; k];
        let mut live = 0.0;
        for o in 0..self.stride {
            if self.counts[o] == 0 {
                continue;
            }
            live += 1.0;
            for (i, v) in out.iter_mut().enumerate() {
                *v += from[o * k + i] * scale;
            }
        }
        if live == 0.0 {
            return vec![f64::NAN; k];
        }
        out.iter().map(|v| v / live).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RcovCfg {
    pub n_features: usize,
    pub kind: RcovKind,
    /// Only `"parzen"`; another name is refused.
    pub kernel: String,
    /// A fixed `H`, or `None` for BNHLS's `"auto"` rule (which needs
    /// `block_rows`).
    pub bandwidth: Option<usize>,
    /// Observations averaged at each end, `m`. Default 2; `1` is no jitter.
    pub jitter: usize,
    /// Pre-averaging window scale: `kₙ = ⌊θ√block_rows⌋`.
    pub theta: f64,
    /// Clip negative eigenvalues at close.
    pub psd: bool,
    /// The block's expected length, which sizes the ring: `"auto"` and
    /// pre-averaging both need a window fixed before the first row.
    pub block_rows: Option<usize>,
    /// Ring depth; defaults to `⌈c*·block_rows^{3/5}⌉` under `"auto"`.
    pub max_bandwidth: Option<usize>,
    /// A fixed `kₙ`, instead of `⌊θ√block_rows⌋`.
    pub preavg_rows: Option<usize>,
    /// Subsampling stride for the noise estimate `ω̂²` (default 1, the dense
    /// grid).
    pub noise_stride: usize,
    /// Subsampling stride for the sparse integrated variance `IV̂` and the
    /// quarticity (default 20).
    pub iv_stride: usize,
}

impl RcovCfg {
    /// The pre-averaging window this configuration fixes before the block
    /// starts.
    pub fn window_for(&self) -> Option<usize> {
        if let Some(w) = self.preavg_rows {
            return Some(w);
        }
        let n = self.block_rows? as f64;
        Some(
            if self.psd {
                // CKP §3's longer, PSD configuration. **The exponent is
                // Hautsch-Podolskij's reading (1/2 + delta, delta = 0.1) and is
                // the one thing in this model taken second-hand**; CKP §3 is the
                // source, and if it disagrees this line and the dropped bias
                // term move together (docs/PLAN.md §11a, task 50).
                (self.theta * n.powf(0.6)).ceil() as usize
            } else {
                // CKP's own floor.
                (self.theta * n.sqrt()).floor() as usize
            }
            .max(2),
        )
    }

    /// The ring depth: `max_bandwidth`, or `⌈c*·block_rows^{3/5}⌉` when the bandwidth is
    /// automatic -- the depth at which `ξ̂ = 1`, a noise variance equal to
    /// the block's integrated variance, beyond which the estimator is not
    /// worth having.
    pub fn ring_for(&self) -> Option<usize> {
        if let Some(h) = self.max_bandwidth {
            return Some(h);
        }
        match (self.kind, self.bandwidth) {
            (RcovKind::Kernel, Some(h)) => Some(h),
            (RcovKind::Kernel, None) => {
                let n = self.block_rows? as f64;
                Some(((parzen_c_star() * n.powf(0.6)).ceil() as usize).max(1))
            }
            _ => Some(0),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("rcov: at least one column is required".into());
        }
        if self.kernel != "parzen" {
            return Err(format!(
                "rcov: unknown kernel {:?}; only \"parzen\" is offered -- the Bartlett kernel is \
                 not consistent for this estimator, and Parzen beats the quadratic spectral on \
                 efficiency (BNHLS 2011)",
                self.kernel
            ));
        }
        if self.jitter == 0 {
            return Err("rcov: jitter must be >= 1 (1 is no jitter at all)".into());
        }
        if !(self.theta > 0.0 && self.theta.is_finite()) {
            return Err("rcov: theta must be finite and > 0".into());
        }
        if self.noise_stride == 0 || self.iv_stride == 0 {
            return Err("rcov: noise_stride and iv_stride must be >= 1".into());
        }
        if self.bandwidth.is_some() && self.kind != RcovKind::Kernel {
            return Err(format!(
                "rcov: bandwidth applies to kind = \"kernel\", not {:?}",
                self.kind.as_str()
            ));
        }
        if self.preavg_rows.is_some() && self.kind != RcovKind::Preavg {
            return Err(format!(
                "rcov: window applies to kind = \"preavg\", not {:?}",
                self.kind.as_str()
            ));
        }
        if self.kind == RcovKind::Kernel && self.bandwidth.is_none() && self.block_rows.is_none() {
            return Err(
                "rcov: an automatic bandwidth needs `block_rows` (the block's expected length): the \
                 ring has to be sized before the first row, and `n` is known only at the close"
                    .into(),
            );
        }
        if self.kind == RcovKind::Preavg && self.window_for().is_none() {
            return Err(
                "rcov: pre-averaging needs `block_rows` or an explicit `window`: the window has to be \
                 fixed before the first row"
                    .into(),
            );
        }
        if let Some(w) = self.preavg_rows {
            if w < 2 {
                return Err(format!(
                    "rcov: window must be >= 2 (got {w}); the pre-averaged return is a weighted \
                     sum over `window − 1` returns, so below 2 there is nothing to average and \
                     no block ever accumulates"
                ));
            }
        }
        if self.block_rows == Some(0) {
            return Err(
                "rcov: block_rows is the block's expected length in returns and must be >= 1; the \
                 ring and the window are sized from it"
                    .into(),
            );
        }
        if let (Some(h), Some(b)) = (self.max_bandwidth, self.bandwidth) {
            if h < b {
                return Err(format!(
                    "rcov: max_bandwidth = {h} caps the ring below bandwidth = {b}, so the lags the \
                     bandwidth asks for are not there and it is silently reduced to {h}; raise \
                     max_bandwidth or lower the bandwidth"
                ));
            }
        }
        Ok(())
    }
}

/// A closed block's estimate, as the group-close row reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct RcovEstimate {
    /// `k*k` row-major; `None` when the block was too short.
    pub rcov: Option<Vec<f64>>,
    pub rcorr: Option<Vec<f64>>,
    /// Effective returns behind it.
    pub n: i64,
    pub kind: &'static str,
    /// The bandwidth actually used (`kernel` only).
    pub bandwidth_used: Option<i64>,
    /// The noise variance and sparse integrated variance behind an automatic
    /// bandwidth, per feature, and the quarticity proxy.
    pub omega2: Option<Vec<f64>>,
    pub iv_sparse: Option<Vec<f64>>,
    pub iq: Option<Vec<f64>>,
    /// A negative eigenvalue had to be clipped.
    pub psd_repaired: bool,
}

/// The block estimator; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rcov {
    cfg: RcovCfg,
    /// Returns accepted (weight 1).
    n: u64,
    /// Accumulated weight, which is `n`: the `n_eff` contract, undecayed.
    w_sum: f64,
    /// `Σ xⱼxⱼ'`, always accumulated: the `plain` estimate, and the bias
    /// term pre-averaging subtracts.
    raw: Vec<f64>,
    /// The first `jitter` returns, for the leading jittered return.
    head: Vec<Vec<f64>>,
    /// The last `jitter` raw returns of the current stretch, which are the
    /// ones still eligible for the trailing jitter: everything older has
    /// been emitted.
    tail: VecDeque<Vec<f64>>,
    /// The last `max_bandwidth` *final* returns, for the lagged products.
    fin: VecDeque<Vec<f64>>,
    /// `(max_bandwidth + 1) * k * k`: `Γ̂_h` for `h = 0 ..= max_bandwidth`.
    gamma: Vec<f64>,
    /// Effective returns emitted so far.
    emitted: u64,
    /// Pre-averaging: the ring of `kₙ − 1` returns and `Σ ȲȲ'`.
    pre_ring: VecDeque<Vec<f64>>,
    pre_sum: Vec<f64>,
    pre_count: u64,
    /// The noise and sparse-variance accumulators.
    dense: StridedRv,
    sparse: StridedRv,
}

impl Rcov {
    pub fn new(cfg: RcovCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.n_features;
        let max_bandwidth = cfg.ring_for().unwrap_or(0);
        let kn = cfg.window_for().unwrap_or(0);
        Ok(Self {
            n: 0,
            w_sum: 0.0,
            raw: vec![0.0; k * k],
            head: Vec::with_capacity(cfg.jitter),
            tail: VecDeque::with_capacity(cfg.jitter.max(1)),
            fin: VecDeque::with_capacity(max_bandwidth.max(1)),
            gamma: vec![0.0; (max_bandwidth + 1) * k * k],
            emitted: 0,
            pre_ring: VecDeque::with_capacity(kn.max(1) + 1),
            pre_sum: vec![0.0; k * k],
            pre_count: 0,
            dense: StridedRv::new(k, cfg.noise_stride),
            sparse: StridedRv::new(k, cfg.iv_stride),
            cfg,
        })
    }

    pub fn cfg(&self) -> &RcovCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Returns accepted so far.
    pub fn n(&self) -> u64 {
        self.n
    }

    fn k(&self) -> usize {
        self.cfg.n_features
    }

    /// The leading jittered return `x̃₁ = Σ_{i=1}^{m−1}(i/m)·xᵢ + x_m`, which
    /// is `X_m − mean(X₀..X_{m−1})` written in returns.
    fn lead(&self) -> Vec<f64> {
        let m = self.cfg.jitter;
        let mut out = vec![0.0; self.k()];
        for (idx, x) in self.head.iter().enumerate() {
            // `head[idx]` is `x_{idx+1}`.
            let i = idx + 1;
            let w = if i == m { 1.0 } else { i as f64 / m as f64 };
            for (o, v) in out.iter_mut().zip(x) {
                *o += w * v;
            }
        }
        out
    }

    /// The trailing jittered return `x̃_end = (1/m)·Σ_{i=n−m+1}^{n}(n−i+1)·xᵢ`,
    /// which is `mean(X_{n−m+1}..X_n) − X_{n−m}` written in returns.
    fn trail(&self) -> Vec<f64> {
        let m = self.cfg.jitter;
        let mut out = vec![0.0; self.k()];
        // The ring holds the `m` returns that have not been emitted, which
        // are exactly the end jitter -- fewer while the stretch is still
        // short of them.
        let take = self.tail.len().min(m);
        let skip = self.tail.len() - take;
        for (idx, x) in self.tail.iter().skip(skip).enumerate() {
            // The last entry is `x_n`, weight 1/m; the first of the `m` is
            // `x_{n−m+1}`, weight m/m.
            let w = (take - idx) as f64 / m as f64;
            for (o, v) in out.iter_mut().zip(x) {
                *o += w * v;
            }
        }
        out
    }

    /// One final effective return into the lagged products.
    fn emit(&mut self, y: Vec<f64>) {
        let k = self.k();
        let max_bandwidth = self.cfg.ring_for().unwrap_or(0);
        for h in 0..=max_bandwidth {
            let past = if h == 0 {
                Some(&y)
            } else {
                self.fin.len().checked_sub(h).map(|i| &self.fin[i])
            };
            let Some(past) = past else { continue };
            let block = h * k * k;
            for (i, &yi) in y.iter().enumerate() {
                let row = &mut self.gamma[block + i * k..block + (i + 1) * k];
                for (g, &pj) in row.iter_mut().zip(past) {
                    *g += yi * pj;
                }
            }
        }
        if max_bandwidth > 0 {
            if self.fin.len() == max_bandwidth {
                self.fin.pop_front();
            }
            self.fin.push_back(y);
        }
        self.emitted += 1;
    }

    /// One accepted return.
    fn push(&mut self, x: &[f64]) {
        let k = self.k();
        for (i, &xi) in x.iter().enumerate() {
            let row = &mut self.raw[i * k..(i + 1) * k];
            for (r, &xj) in row.iter_mut().zip(x) {
                *r += xi * xj;
            }
        }
        self.dense.push(x);
        self.sparse.push(x);
        self.n += 1;
        let m = self.cfg.jitter;
        if self.cfg.kind == RcovKind::Kernel {
            // Which phase of the stretch this return is in is carried by
            // `head`, not by a count against `n`: a break in the clock
            // (`clear_lags`) starts a new stretch part-way through the
            // group, and the phases have to restart with it
            // (docs/REVIEW-E54-E64.md R2).
            if self.head.len() < m {
                // The leading jitter: the first `m` returns of the stretch
                // make one effective return between them and nothing else.
                self.head.push(x.to_vec());
                if self.head.len() == m {
                    let y = self.lead();
                    self.emit(y);
                }
            } else {
                // The interior: a return leaves the tail -- and so can no
                // longer become part of the end jitter -- when the `m` after
                // it have arrived.
                if self.tail.len() == m {
                    let y = self.tail.pop_front().expect("m kept");
                    self.emit(y);
                }
                self.tail.push_back(x.to_vec());
            }
        }
        if self.cfg.kind == RcovKind::Preavg {
            let kn = self.cfg.window_for().unwrap_or(0);
            if kn >= 2 {
                self.pre_ring.push_back(x.to_vec());
                if self.pre_ring.len() == kn {
                    // `Ȳ_i = Σ_{j=1}^{kn−1} g(j/kn)·x_{i+j}`: the window is
                    // the `kn − 1` returns *after* the oldest in the ring,
                    // which is what makes the first one start at `i = 0`.
                    let mut y = vec![0.0; k];
                    for j in 1..kn {
                        let w = preavg_g(j as f64 / kn as f64);
                        for (o, v) in y.iter_mut().zip(&self.pre_ring[j]) {
                            *o += w * v;
                        }
                    }
                    for (i, &yi) in y.iter().enumerate() {
                        let row = &mut self.pre_sum[i * k..(i + 1) * k];
                        for (p, &yj) in row.iter_mut().zip(&y) {
                            *p += yi * yj;
                        }
                    }
                    self.pre_count += 1;
                    self.pre_ring.pop_front();
                }
            }
        }
    }

    /// The block, from the state as it stands. Called at the group's close;
    /// `&self`, so it can be read without disturbing anything.
    pub fn estimate(&self) -> RcovEstimate {
        let k = self.k();
        let kind = self.cfg.kind.as_str();
        let omega2 = Some(self.dense.noise());
        let iv_sparse = Some(self.sparse.mean_rv());
        let iq = Some(self.sparse.quarticity());
        let short = |n: i64| RcovEstimate {
            rcov: None,
            rcorr: None,
            n,
            kind,
            bandwidth_used: None,
            omega2: omega2.clone(),
            iv_sparse: iv_sparse.clone(),
            iq: iq.clone(),
            psd_repaired: false,
        };
        let (mut cov, n_eff, bandwidth_used) = match self.cfg.kind {
            RcovKind::Plain => {
                if self.n < 1 {
                    return short(self.n as i64);
                }
                (self.raw.clone(), self.n as i64, None)
            }
            RcovKind::Kernel => {
                let m = self.cfg.jitter as u64;
                // The last stretch contributes a trailing jitter only if it
                // ran far enough to fill the tail; one cut short by a break
                // contributes what it emitted before the break and no more.
                // With no break the two agree: `n >= 2m` is exactly a full
                // tail, so an unbroken group is unchanged to the bit.
                let closes = self.tail.len() == self.cfg.jitter;
                if self.n < 2 * m || (self.emitted == 0 && !closes) {
                    return short(0);
                }
                let max_bandwidth = self.cfg.ring_for().unwrap_or(0);
                // The trailing jittered return, formed here from state.
                let mut gamma = self.gamma.clone();
                if closes {
                    let y = self.trail();
                    let fin = &self.fin;
                    for h in 0..=max_bandwidth {
                        let past = if h == 0 {
                            Some(&y)
                        } else {
                            fin.len().checked_sub(h).map(|i| &fin[i])
                        };
                        let Some(past) = past else { continue };
                        let block = h * k * k;
                        for (i, &yi) in y.iter().enumerate() {
                            let row = &mut gamma[block + i * k..block + (i + 1) * k];
                            for (g, &pj) in row.iter_mut().zip(past) {
                                *g += yi * pj;
                            }
                        }
                    }
                }
                let n_eff = self.emitted as i64 + i64::from(closes);
                let h = self.bandwidth(n_eff as f64).min(max_bandwidth);
                let mut out = vec![0.0; k * k];
                for lag in 0..=h {
                    let w = parzen(lag as f64 / (h as f64 + 1.0));
                    let block = lag * k * k;
                    for i in 0..k {
                        for j in 0..k {
                            let g = gamma[block + i * k + j];
                            if lag == 0 {
                                out[i * k + j] += w * g;
                            } else {
                                // `Γ̂_{−h} = Γ̂_h'`, so the pair enters as
                                // the matrix and its transpose.
                                out[i * k + j] += w * g;
                                out[j * k + i] += w * g;
                            }
                        }
                    }
                }
                (out, n_eff, Some(h as i64))
            }
            RcovKind::Preavg => {
                let kn = self.cfg.window_for().unwrap_or(0);
                if self.n < kn as u64 || kn < 2 {
                    return short(self.pre_count as i64);
                }
                let n = self.n as f64;
                let knf = kn as f64;
                let (p1, p2) = (psi1(kn), psi2(kn));
                let scale = n / (n - knf + 2.0) / (p2 * knf);
                let bias = if self.cfg.psd {
                    // CKP §3's PSD configuration drops the bias term with the
                    // longer window (see `window_for`).
                    0.0
                } else {
                    p1 / (self.cfg.theta * self.cfg.theta * p2) / (2.0 * n)
                };
                let out = self
                    .pre_sum
                    .iter()
                    .zip(&self.raw)
                    .map(|(p, r)| scale * p - bias * r)
                    .collect();
                (out, self.pre_count as i64, None)
            }
        };
        // Symmetrize against rounding: every path sums outer products, so
        // the two triangles differ only in the last bits.
        for i in 0..k {
            for j in (i + 1)..k {
                let v = 0.5 * (cov[i * k + j] + cov[j * k + i]);
                cov[i * k + j] = v;
                cov[j * k + i] = v;
            }
        }
        let mut psd_repaired = false;
        if self.cfg.psd {
            if let Some(fixed) = clip_psd(&cov, k) {
                psd_repaired = fixed.1;
                cov = fixed.0;
            }
        }
        let rcorr = correlation(&cov, k);
        RcovEstimate {
            rcov: Some(cov),
            rcorr: Some(rcorr),
            n: n_eff,
            kind,
            bandwidth_used,
            omega2,
            iv_sparse,
            iq,
            psd_repaired,
        }
    }

    /// The bandwidth for a block of `n` effective returns: the configured
    /// one, or BNHLS's `H = ⌈c*·ξ̂^{4/5}·n^{3/5}⌉` averaged over features.
    fn bandwidth(&self, n: f64) -> usize {
        if let Some(h) = self.cfg.bandwidth {
            return h;
        }
        let omega2 = self.dense.noise();
        let iv = self.sparse.mean_rv();
        let c = parzen_c_star();
        let per: Vec<f64> = omega2
            .iter()
            .zip(&iv)
            .map(|(w, v)| {
                if *v > 0.0 && w.is_finite() && *w >= 0.0 {
                    let xi2 = w / v;
                    c * xi2.sqrt().powf(0.8) * n.powf(0.6)
                } else {
                    f64::NAN
                }
            })
            .filter(|v| v.is_finite())
            .collect();
        if per.is_empty() {
            return 1;
        }
        let mean = per.iter().sum::<f64>() / per.len() as f64;
        (mean.ceil().max(1.0)) as usize
    }
}

/// The correlation matrix of a covariance, `NaN` where a variance is not
/// positive.
fn correlation(cov: &[f64], k: usize) -> Vec<f64> {
    let sd: Vec<f64> = (0..k).map(|i| cov[i * k + i].max(0.0).sqrt()).collect();
    let mut out = vec![f64::NAN; k * k];
    for i in 0..k {
        for j in 0..k {
            let d = sd[i] * sd[j];
            if d > 0.0 {
                out[i * k + j] = (cov[i * k + j] / d).clamp(-1.0, 1.0);
            }
        }
    }
    out
}

/// `cov` with its negative eigenvalues clipped at zero, and whether any had
/// to be. `None` when the decomposition fails.
fn clip_psd(cov: &[f64], k: usize) -> Option<(Vec<f64>, bool)> {
    use faer::Side;
    use faer::prelude::*;
    if cov.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mat = Mat::from_fn(k, k, |i, j| cov[i * k + j]);
    let evd = mat.self_adjoint_eigen(Side::Lower).ok()?;
    let (s, u) = (evd.S(), evd.U());
    let neg = (0..k).any(|i| s[i] < 0.0);
    if !neg {
        return Some((cov.to_vec(), false));
    }
    let mut out = vec![0.0; k * k];
    for c in 0..k {
        let lam = s[c].max(0.0);
        if lam == 0.0 {
            continue;
        }
        for i in 0..k {
            let ui = u[(i, c)] * lam;
            for j in 0..k {
                out[i * k + j] += ui * u[(j, c)];
            }
        }
    }
    Some((out, true))
}

impl crate::OnlineModel for Rcov {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], _d_clock: f64, weight: f64) -> crate::Step {
        let out = self.predict(x, _d_clock);
        // A zero-weight row advances the clock and is not a return of the
        // block: it enters no accumulator and no ring.
        if weight > 0.0 {
            self.push(x);
            self.w_sum += 1.0;
        }
        out
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> crate::Step {
        crate::Step {
            pred: Vec::new(),
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn clear_lags(&mut self) {
        // A break in the clock is a break in the block: the rings pair rows
        // that are no longer adjacent. The stretch that ends here is closed
        // the way `estimate` closes the last one -- its trailing jitter
        // emitted against the returns it *is* adjacent to -- and the next
        // return starts a new stretch, leading jitter and all. Dropping the
        // tail instead lost the `m` returns in it and then re-emitted the
        // first return after the break `m + 1` times
        // (docs/REVIEW-E54-E64.md R2).
        if self.cfg.kind == RcovKind::Kernel && self.tail.len() == self.cfg.jitter {
            let y = self.trail();
            self.emit(y);
        }
        // A stretch that never got past its leading jitter contributes what
        // it already emitted and no more: a partial end jitter is not the
        // paper's statistic, and there is no way to un-emit the lead.
        self.head.clear();
        self.tail.clear();
        self.fin.clear();
        self.pre_ring.clear();
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::Rcov(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Rcov(m) => {
                let mut m = (**m).clone();
                // A state written before the stretch rewrite kept `m + 1`
                // returns in the tail, its front already emitted. The tail
                // now holds exactly the unemitted `m`, and a longer one
                // would never reach the emission test again -- so drop the
                // extra from the front, which is the entry that had already
                // gone (docs/REVIEW-E54-E64.md R2).
                while m.tail.len() > m.cfg.jitter {
                    m.tail.pop_front();
                }
                Ok(m)
            }
            other => Err(crate::StateError::WrongModel {
                expected: "rcov",
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

    /// Nothing per row: the block is the value, read at the group's close.
    fn n_outputs(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EwCov, OnlineModel};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(k: usize, kind: RcovKind) -> RcovCfg {
        RcovCfg {
            n_features: k,
            kind,
            kernel: "parzen".into(),
            bandwidth: (kind == RcovKind::Kernel).then_some(4),
            jitter: 2,
            theta: 1.0,
            psd: false,
            block_rows: Some(400),
            max_bandwidth: None,
            preavg_rows: None,
            noise_stride: 1,
            iv_stride: 20,
        }
    }

    /// An efficient price plus i.i.d. noise, differenced: the returns a
    /// noisy tick stream produces.
    fn returns(n: usize, k: usize, seed: u64, noise: f64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut prev = vec![0.0; k];
        let mut level = vec![0.0; k];
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let f = lcg(&mut s);
            for (i, l) in level.iter_mut().enumerate() {
                *l += 0.3 * f + 0.7 * lcg(&mut s) * (1.0 + i as f64 * 0.1);
            }
            let obs: Vec<f64> = level.iter().map(|l| l + noise * lcg(&mut s)).collect();
            out.push(obs.iter().zip(&prev).map(|(o, p)| o - p).collect());
            prev = obs;
        }
        out
    }

    fn feed(m: &mut Rcov, rows: &[Vec<f64>]) {
        for x in rows {
            m.step(x, &[], 1.0, 1.0);
        }
    }

    /// The acceptance: `plain` at close is `n ×` an `ew_cov(lam = 1)`'s
    /// uncentred second moment, to the bit -- the same products in the same
    /// order.
    #[test]
    fn plain_is_the_raw_second_moment_times_n() {
        let (k, n) = (3, 200);
        let rows = returns(n, k, 7, 0.0);
        let mut m = Rcov::new(cfg(k, RcovKind::Plain)).unwrap();
        feed(&mut m, &rows);
        let mut cov = EwCov::new(k);
        for x in &rows {
            cov.update(x, 1.0, 1.0);
        }
        let est = m.estimate();
        let got = est.rcov.unwrap();
        for i in 0..k {
            for j in 0..k {
                // `EwCov` keeps means, so `raw` is the second moment; times
                // `n` it is the sum of products.
                let want = cov.raw(i, j) * n as f64;
                assert!(
                    (got[i * k + j] - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "[{i}][{j}]: {} vs {want}",
                    got[i * k + j]
                );
            }
        }
        assert_eq!(est.n, n as i64);
    }

    /// The kernel against the definition, written out over the effective
    /// return series (jittered ends and all).
    #[test]
    fn the_kernel_is_its_definition_on_the_effective_returns() {
        let (k, n, m_j, h) = (2usize, 60usize, 2usize, 4usize);
        let rows = returns(n, k, 11, 0.4);
        let mut model = Rcov::new(RcovCfg {
            bandwidth: Some(h),
            jitter: m_j,
            ..cfg(k, RcovKind::Kernel)
        })
        .unwrap();
        feed(&mut model, &rows);
        let got = model.estimate().rcov.unwrap();

        // The effective series, longhand.
        let mut y: Vec<Vec<f64>> = Vec::new();
        let lead: Vec<f64> = (0..k)
            .map(|c| {
                (1..=m_j)
                    .map(|i| {
                        let w = if i == m_j { 1.0 } else { i as f64 / m_j as f64 };
                        w * rows[i - 1][c]
                    })
                    .sum()
            })
            .collect();
        y.push(lead);
        for row in rows.iter().take(n - m_j).skip(m_j) {
            y.push(row.clone());
        }
        let trail: Vec<f64> = (0..k)
            .map(|c| {
                (n - m_j..n)
                    .map(|i| (n - i) as f64 / m_j as f64 * rows[i][c])
                    .sum()
            })
            .collect();
        y.push(trail);
        assert_eq!(y.len(), n - 2 * m_j + 2);

        let mut want = vec![0.0; k * k];
        for lag in 0..=h {
            let w = parzen(lag as f64 / (h as f64 + 1.0));
            let mut g = vec![0.0; k * k];
            for t in lag..y.len() {
                for i in 0..k {
                    for j in 0..k {
                        g[i * k + j] += y[t][i] * y[t - lag][j];
                    }
                }
            }
            for i in 0..k {
                for j in 0..k {
                    want[i * k + j] += w * g[i * k + j];
                    if lag > 0 {
                        want[j * k + i] += w * g[i * k + j];
                    }
                }
            }
        }
        // The estimate is symmetrized; compare the symmetric part.
        for i in 0..k {
            for j in 0..k {
                let w = 0.5 * (want[i * k + j] + want[j * k + i]);
                assert!(
                    (got[i * k + j] - w).abs() <= 1e-9 * (1.0 + w.abs()),
                    "[{i}][{j}]: {} vs {w}",
                    got[i * k + j]
                );
            }
        }
    }

    /// The MRC against its definition, pre-averaged windows and all.
    #[test]
    fn the_preaveraged_estimate_is_its_definition() {
        let (k, n) = (2usize, 300usize);
        let rows = returns(n, k, 13, 0.5);
        let mut model = Rcov::new(RcovCfg {
            bandwidth: None,
            preavg_rows: Some(10),
            ..cfg(k, RcovKind::Preavg)
        })
        .unwrap();
        feed(&mut model, &rows);
        let got = model.estimate().rcov.unwrap();

        let kn = 10usize;
        let (p1, p2) = (psi1(kn), psi2(kn));
        let mut sum = vec![0.0; k * k];
        let mut count = 0u64;
        for start in 0..=(n - kn) {
            let ybar: Vec<f64> = (0..k)
                .map(|c| {
                    (1..kn)
                        .map(|j| preavg_g(j as f64 / kn as f64) * rows[start + j][c])
                        .sum()
                })
                .collect();
            for i in 0..k {
                for j in 0..k {
                    sum[i * k + j] += ybar[i] * ybar[j];
                }
            }
            count += 1;
        }
        let mut raw = vec![0.0; k * k];
        for x in &rows {
            for i in 0..k {
                for j in 0..k {
                    raw[i * k + j] += x[i] * x[j];
                }
            }
        }
        let nf = n as f64;
        let scale = nf / (nf - kn as f64 + 2.0) / (p2 * kn as f64);
        let bias = p1 / p2 / (2.0 * nf);
        assert_eq!(count as i64, model.estimate().n);
        for i in 0..k {
            for j in 0..k {
                let w = 0.5
                    * ((scale * sum[i * k + j] - bias * raw[i * k + j])
                        + (scale * sum[j * k + i] - bias * raw[j * k + i]));
                assert!(
                    (got[i * k + j] - w).abs() <= 1e-9 * (1.0 + w.abs()),
                    "[{i}][{j}]: {} vs {w}",
                    got[i * k + j]
                );
            }
        }
    }

    /// The `psi` and `Phi` constants converge to their published closed
    /// forms for `g = min(x, 1−x)`.
    #[test]
    fn the_preaveraging_constants_reach_their_closed_forms() {
        let k = 4000;
        assert!((psi1(k) - 1.0).abs() < 1e-6, "psi1 = {}", psi1(k));
        assert!((psi2(k) - 1.0 / 12.0).abs() < 1e-6, "psi2 = {}", psi2(k));
        // The Phi constants converge slowly, so a smaller `k` and a looser
        // band; the point is that the finite-sample sums are the ones the
        // paper defines.
        let k = 400;
        assert!(
            (phi_11(k) - 1.0 / 6.0).abs() < 5e-3,
            "Phi11 = {}",
            phi_11(k)
        );
        assert!(
            (phi_12(k) - 1.0 / 96.0).abs() < 5e-4,
            "Phi12 = {}",
            phi_12(k)
        );
        assert!(
            (phi_22(k) - 151.0 / 80640.0).abs() < 5e-5,
            "Phi22 = {}",
            phi_22(k)
        );
    }

    /// Parzen at the points the paper pins.
    #[test]
    fn the_parzen_kernel_is_its_definition() {
        assert_eq!(parzen(0.0), 1.0);
        assert!((parzen(0.5) - 0.25).abs() < 1e-15);
        assert_eq!(parzen(1.0), 0.0);
        assert_eq!(parzen(1.5), 0.0);
        assert_eq!(parzen(-0.5), parzen(0.5));
        assert!(
            (parzen_c_star() - 3.5134).abs() < 1e-3,
            "{}",
            parzen_c_star()
        );
    }

    /// Nothing reads a future row: the close from a stream fed one row at a
    /// time equals the offline computation over the whole block.
    #[test]
    fn the_close_is_the_same_however_the_rows_arrive() {
        for kind in [RcovKind::Plain, RcovKind::Kernel, RcovKind::Preavg] {
            let (k, n) = (2usize, 120usize);
            let rows = returns(n, k, 17, 0.3);
            let mut a = Rcov::new(RcovCfg {
                preavg_rows: (kind == RcovKind::Preavg).then_some(8),
                bandwidth: (kind == RcovKind::Kernel).then_some(3),
                ..cfg(k, kind)
            })
            .unwrap();
            let mut b = a.clone();
            feed(&mut a, &rows);
            // The same rows, with a save/load in the middle.
            feed(&mut b, &rows[..50]);
            let bytes = rmp_serde::to_vec(&b.state()).unwrap();
            let mut b = Rcov::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            feed(&mut b, &rows[50..]);
            assert_eq!(a.estimate(), b.estimate(), "{kind:?}");
        }
    }

    #[test]
    fn a_zero_weight_row_advances_the_clock_and_enters_nothing() {
        let (k, n) = (2usize, 40usize);
        let rows = returns(n, k, 19, 0.2);
        let mut with = Rcov::new(cfg(k, RcovKind::Kernel)).unwrap();
        let mut without = Rcov::new(cfg(k, RcovKind::Kernel)).unwrap();
        for (i, x) in rows.iter().enumerate() {
            with.step(x, &[], 1.0, 1.0);
            if i == 20 {
                // A wild row at zero weight, in the middle.
                with.step(&[1e6, -1e6], &[], 1.0, 0.0);
            }
            without.step(x, &[], 1.0, 1.0);
        }
        assert_eq!(with.estimate(), without.estimate());
        assert_eq!(with.n_eff(), without.n_eff());
    }

    #[test]
    fn a_short_block_gives_nulls_rather_than_a_panic() {
        // Fewer than `2 * jitter` returns: the kernel has no effective
        // series at all.
        let mut m = Rcov::new(cfg(2, RcovKind::Kernel)).unwrap();
        m.step(&[1.0, 2.0], &[], 1.0, 1.0);
        let e = m.estimate();
        assert!(e.rcov.is_none() && e.rcorr.is_none() && e.n == 0);
        // And a block with no rows at all.
        let m = Rcov::new(cfg(2, RcovKind::Plain)).unwrap();
        assert!(m.estimate().rcov.is_none());
        // Pre-averaging with fewer rows than its window.
        let mut m = Rcov::new(RcovCfg {
            preavg_rows: Some(20),
            ..cfg(2, RcovKind::Preavg)
        })
        .unwrap();
        for _ in 0..5 {
            m.step(&[1.0, 2.0], &[], 1.0, 1.0);
        }
        assert!(m.estimate().rcov.is_none());
    }

    #[test]
    fn the_psd_clip_repairs_a_negative_eigenvalue_and_says_so() {
        // A one-spike stream makes the kernel estimate indefinite at a
        // bandwidth this large relative to the block.
        let k = 2;
        let mut m = Rcov::new(RcovCfg {
            psd: true,
            bandwidth: Some(6),
            ..cfg(k, RcovKind::Kernel)
        })
        .unwrap();
        let mut s = 23u64;
        for i in 0..40 {
            let x = if i == 20 {
                vec![50.0, -50.0]
            } else {
                vec![lcg(&mut s), lcg(&mut s)]
            };
            m.step(&x, &[], 1.0, 1.0);
        }
        let e = m.estimate();
        let cov = e.rcov.unwrap();
        // Whatever happened, the result is PSD.
        let ev = {
            use faer::Side;
            use faer::prelude::*;
            let mat = Mat::from_fn(k, k, |i, j| cov[i * k + j]);
            let evd = mat.self_adjoint_eigen(Side::Lower).unwrap();
            (0..k).map(|i| evd.S()[i]).collect::<Vec<_>>()
        };
        assert!(ev.iter().all(|v| *v >= -1e-9), "{ev:?}");
    }

    #[test]
    fn adversarial_streams_do_not_panic() {
        for kind in [RcovKind::Plain, RcovKind::Kernel, RcovKind::Preavg] {
            for rows in [
                vec![vec![0.0]; 50], // a constant series
                vec![vec![1e50]; 3], // enormous
                (0..50)
                    .map(|i| vec![if i == 25 { 1.0 } else { 0.0 }])
                    .collect(),
            ] {
                let mut m = Rcov::new(RcovCfg {
                    psd: true,
                    preavg_rows: (kind == RcovKind::Preavg).then_some(4),
                    bandwidth: (kind == RcovKind::Kernel).then_some(2),
                    ..cfg(1, kind)
                })
                .unwrap();
                feed(&mut m, &rows);
                let e = m.estimate();
                if let Some(c) = e.rcov {
                    assert!(c.iter().all(|v| !v.is_nan()), "{kind:?}");
                }
            }
        }
    }

    #[test]
    fn the_auto_bandwidth_is_its_formula() {
        let (k, n) = (1usize, 400usize);
        let rows = returns(n, k, 29, 0.5);
        let mut m = Rcov::new(RcovCfg {
            bandwidth: None,
            block_rows: Some(n),
            ..cfg(k, RcovKind::Kernel)
        })
        .unwrap();
        feed(&mut m, &rows);
        let e = m.estimate();
        let (w, v) = (e.omega2.unwrap()[0], e.iv_sparse.unwrap()[0]);
        let xi2 = w / v;
        let want = (parzen_c_star() * xi2.sqrt().powf(0.8) * (e.n as f64).powf(0.6)).ceil() as i64;
        let max_bandwidth = m.cfg().ring_for().unwrap() as i64;
        assert_eq!(e.bandwidth_used, Some(want.min(max_bandwidth)));
        assert!(w > 0.0 && v > 0.0);
    }

    #[test]
    fn clear_lags_drops_the_rings_and_keeps_the_sums() {
        let mut m = Rcov::new(cfg(2, RcovKind::Kernel)).unwrap();
        feed(&mut m, &returns(30, 2, 31, 0.1));
        let raw = m.raw.clone();
        let n = m.n;
        m.clear_lags();
        assert!(m.fin.is_empty() && m.tail.is_empty() && m.pre_ring.is_empty());
        assert_eq!(m.raw, raw);
        assert_eq!(m.n, n);
    }

    /// A break splits the group into stretches: the estimate is what the two
    /// stretches give as separate groups, added. At `H = 0` that is exactly
    /// `Σ ỹỹ'` over each stretch's own effective returns, so the two sides
    /// are comparable term by term -- and with `jitter = 1` the effective
    /// returns are the raw ones, so both sides are `plain` as well.
    ///
    /// The tail used to be dropped rather than closed: the `m` returns in it
    /// were lost and the first return after the break was re-emitted until
    /// the ring refilled, which at `m = 1` moved the estimate by
    /// `−x_10 x_10' + x_11 x_11'` (docs/REVIEW-E54-E64.md R2).
    #[test]
    fn a_break_splits_the_group_into_stretches() {
        let rows = returns(20, 2, 77, 0.1);
        for m in 1..=3usize {
            let kern = || {
                Rcov::new(RcovCfg {
                    bandwidth: Some(0),
                    max_bandwidth: Some(0),
                    jitter: m,
                    ..cfg(2, RcovKind::Kernel)
                })
                .unwrap()
            };
            let piece = |rows: &[Vec<f64>]| {
                let mut p = kern();
                feed(&mut p, rows);
                p.estimate()
            };
            let (a, b) = (piece(&rows[..10]), piece(&rows[10..]));

            let mut broken = kern();
            feed(&mut broken, &rows[..10]);
            broken.clear_lags();
            feed(&mut broken, &rows[10..]);
            let got = broken.estimate();
            for ((g, x), y) in got
                .rcov
                .as_ref()
                .unwrap()
                .iter()
                .zip(a.rcov.as_ref().unwrap())
                .zip(b.rcov.as_ref().unwrap())
            {
                assert!(
                    (g - (x + y)).abs() < 1e-9,
                    "jitter {m}: broken {g} != {x} + {y}"
                );
            }
            assert_eq!(got.n, a.n + b.n, "jitter {m}: effective returns");
        }
        // `jitter = 1` is no jitter at all, so every effective return is a
        // raw one and both runs are the plain sum of squares.
        let mut plain = Rcov::new(cfg(2, RcovKind::Plain)).unwrap();
        feed(&mut plain, &rows);
        let want = plain.estimate().rcov.unwrap();
        for cut in [rows.len(), 10] {
            let mut m = Rcov::new(RcovCfg {
                bandwidth: Some(0),
                max_bandwidth: Some(0),
                jitter: 1,
                ..cfg(2, RcovKind::Kernel)
            })
            .unwrap();
            feed(&mut m, &rows[..cut]);
            if cut < rows.len() {
                m.clear_lags();
                feed(&mut m, &rows[cut..]);
            }
            let got = m.estimate();
            for (g, w) in got.rcov.unwrap().iter().zip(&want) {
                assert!((g - w).abs() < 1e-9, "cut {cut}: {g} != plain {w}");
            }
            assert_eq!(got.n, rows.len() as i64, "cut {cut}: every return emitted");
        }
    }

    /// A stretch too short to fill the tail contributes what it emitted and
    /// no more, and a group made only of such stretches reports nothing
    /// rather than a covariance built from no returns.
    #[test]
    fn a_group_of_stretches_shorter_than_the_jitter_reports_nothing() {
        let rows = returns(9, 2, 5, 0.1);
        let mut m = Rcov::new(RcovCfg {
            bandwidth: Some(0),
            max_bandwidth: Some(0),
            jitter: 4,
            ..cfg(2, RcovKind::Kernel)
        })
        .unwrap();
        for chunk in rows.chunks(3) {
            feed(&mut m, chunk);
            m.clear_lags();
        }
        let e = m.estimate();
        assert!(e.rcov.is_none() && e.n == 0, "{e:?}");
    }

    #[test]
    fn a_bad_configuration_is_refused_by_name() {
        let bad = |c: RcovCfg, msg: &str| {
            let e = Rcov::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(
            RcovCfg {
                kernel: "bartlett".into(),
                ..cfg(2, RcovKind::Kernel)
            },
            "not consistent",
        );
        bad(
            RcovCfg {
                jitter: 0,
                ..cfg(2, RcovKind::Kernel)
            },
            "jitter must be >= 1",
        );
        for w in [0, 1] {
            bad(
                RcovCfg {
                    preavg_rows: Some(w),
                    ..cfg(2, RcovKind::Preavg)
                },
                "window must be >= 2",
            );
        }
        bad(
            RcovCfg {
                block_rows: Some(0),
                bandwidth: None,
                ..cfg(2, RcovKind::Kernel)
            },
            "block_rows is the block's expected length",
        );
        bad(
            RcovCfg {
                max_bandwidth: Some(0),
                bandwidth: Some(4),
                ..cfg(2, RcovKind::Kernel)
            },
            "caps the ring below bandwidth",
        );
        bad(
            RcovCfg {
                bandwidth: None,
                block_rows: None,
                ..cfg(2, RcovKind::Kernel)
            },
            "needs `block_rows`",
        );
        bad(
            RcovCfg {
                bandwidth: None,
                block_rows: None,
                preavg_rows: None,
                ..cfg(2, RcovKind::Preavg)
            },
            "needs `block_rows` or an explicit `window`",
        );
        bad(
            RcovCfg {
                bandwidth: Some(2),
                ..cfg(2, RcovKind::Plain)
            },
            "bandwidth applies to",
        );
        bad(
            RcovCfg {
                preavg_rows: Some(2),
                ..cfg(2, RcovKind::Kernel)
            },
            "window applies to",
        );
        bad(
            RcovCfg {
                theta: 0.0,
                ..cfg(2, RcovKind::Preavg)
            },
            "theta must be",
        );
    }
}
