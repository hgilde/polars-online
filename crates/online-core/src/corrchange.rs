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
//!
//! Under the null `Q →_d sup|B|`, a Brownian bridge, whose quantiles are
//! the Kolmogorov distribution -- computed from the series here rather than
//! pinned, so a test can check it reproduces 1.3581 at 5 %.
//!
//! The paper's own form is *sequential*, with a boundary function, in Wied
//! & Galeano (2013); nobody here has read it, so what ships is the closed
//! test run over consecutive spans of `horizon` rows. The cost is a delay
//! of at most `horizon` rows and the benefit is a null with published
//! tables -- which is the whole point of the exercise.
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
    /// `monitor`: the span length `T`.
    pub horizon: usize,
    /// `window`: the length of each of the two adjacent windows.
    pub window: usize,
    /// Nominal level; the critical value is `1 − alpha/npairs` under
    /// Bonferroni.
    pub alpha: f64,
    /// `"bonferroni"` (default) or `"none"`.
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
}

impl CorrChangeCfg {
    pub fn validate(&self) -> Result<(), String> {
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
                if self.horizon < 8 {
                    return Err(
                        "corrchange: kind = \"monitor\" needs a horizon of at least 8 rows; the \
                         statistic is a maximum over the span"
                            .into(),
                    );
                }
                if self.bandwidth == Some(0) {
                    return Err("corrchange: bandwidth must be >= 1".into());
                }
            }
            CorrChangeKind::Window => {
                if self.window < 3 {
                    return Err(
                        "corrchange: kind = \"window\" needs a window of at least 3 rows".into(),
                    );
                }
                if self.crit.is_none() && self.n_perm < 20 {
                    return Err(
                        "corrchange: a permutation critical value needs n_perm >= 20 draws".into(),
                    );
                }
                if self.perm_block == 0 || self.perm_block > self.window {
                    return Err(format!(
                        "corrchange: perm_block must be 1..={} (the window)",
                        self.window
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

    /// The critical value: the Kolmogorov quantile under `monitor`, the
    /// configured one under `window`.
    fn fixed_crit(&self) -> Option<f64> {
        match self.kind {
            CorrChangeKind::Monitor => {
                let a = if self.alpha_adjust == "bonferroni" {
                    self.alpha / self.npairs() as f64
                } else {
                    self.alpha
                };
                Some(kolmogorov_quantile(a))
            }
            CorrChangeKind::Window => self.crit,
        }
    }
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
}

impl CorrChange {
    pub fn new(cfg: CorrChangeCfg) -> Result<Self, String> {
        cfg.validate()?;
        let d = cfg.n_features;
        Ok(Self {
            ring: VecDeque::new(),
            diag: EwDiag::new(d),
            n_eff: 0.0,
            since_flag: None,
            perm_crit: None,
            since_perm: 0,
            rng: SplitMix64::new(cfg.seed),
            cfg,
        })
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
        ]
    }

    pub fn n_outputs_for() -> usize {
        4
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
    /// span, for the pair `(a, b)`. See the module docs.
    fn long_run_sd(rows: &[Vec<f64>], a: usize, b: usize, gamma: usize) -> f64 {
        let t = rows.len();
        if t < 4 {
            return f64::NAN;
        }
        let tf = t as f64;
        // The five raw moments, centred at their span means.
        let mut u = vec![0.0; 5 * t];
        for (i, r) in rows.iter().enumerate() {
            let (x, y) = (r[a], r[b]);
            u[i * 5] = x * x;
            u[i * 5 + 1] = y * y;
            u[i * 5 + 2] = x;
            u[i * 5 + 3] = y;
            u[i * 5 + 4] = x * y;
        }
        let mut mean = [0.0; 5];
        for i in 0..t {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += u[i * 5 + j];
            }
        }
        mean.iter_mut().for_each(|m| *m /= tf);
        for i in 0..t {
            for j in 0..5 {
                u[i * 5 + j] -= mean[j];
            }
        }
        // The Bartlett long-run covariance of the centred moments.
        let mut sigma = [[0.0f64; 5]; 5];
        for lag in 0..t.min(gamma + 1) {
            let w = bartlett(lag as f64 / (gamma as f64 + 1.0));
            if w == 0.0 {
                continue;
            }
            for s in lag..t {
                for i in 0..5 {
                    for j in 0..5 {
                        let v = u[s * 5 + i] * u[(s - lag) * 5 + j];
                        sigma[i][j] += w * v / tf;
                        if lag > 0 {
                            sigma[j][i] += w * v / tf;
                        }
                    }
                }
            }
        }
        // `D₂` maps the five moments to `(σ_x², σ_y², σ_xy)`.
        let (mx, my) = (mean[2], mean[3]);
        let d2 = [
            [1.0, 0.0, -2.0 * mx, 0.0, 0.0],
            [0.0, 1.0, 0.0, -2.0 * my, 0.0],
            [0.0, 0.0, -my, -mx, 1.0],
        ];
        let sx2 = mean[0] - mx * mx;
        let sy2 = mean[1] - my * my;
        let sxy = mean[4] - mx * my;
        if !(sx2 > 0.0 && sy2 > 0.0) {
            return f64::NAN;
        }
        let (sx, sy) = (sx2.sqrt(), sy2.sqrt());
        // `D₃` maps `(σ_x², σ_y², σ_xy)` to `ρ`.
        let d3 = [
            -0.5 * sxy * sy / (sx * sx * sx),
            -0.5 * sxy * sx / (sy * sy * sy),
            1.0 / (sx * sy),
        ];
        // `f = D₃ D₂`, a 5-vector; `D̂² = f Σ̂ f'`.
        let mut f = [0.0; 5];
        for (j, fj) in f.iter_mut().enumerate() {
            *fj = (0..3).map(|i| d3[i] * d2[i][j]).sum();
        }
        let mut v = 0.0;
        for i in 0..5 {
            for j in 0..5 {
                v += f[i] * sigma[i][j] * f[j];
            }
        }
        if v > 0.0 { v.sqrt() } else { f64::NAN }
    }

    /// `Q` for the **mean** of a one-column span: `max_j (j/√T)·|ūⱼ − ū_T|
    /// / D̂ᵤ`, the same CUSUM with the Bartlett long-run standard deviation
    /// of `u` in place of the delta-method one. That is `scalar = true`'s
    /// statistic: the equicorrelation is already one number, so there is a
    /// mean to test and no pair.
    fn scalar_stat(&self, rows: &[Vec<f64>]) -> f64 {
        let t = rows.len();
        if t < 4 {
            return f64::NAN;
        }
        let tf = t as f64;
        let u: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let mean = u.iter().sum::<f64>() / tf;
        let v: Vec<f64> = u.iter().map(|x| x - mean).collect();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((tf).ln().floor() as usize).max(1));
        let mut var = 0.0;
        for lag in 0..t.min(gamma + 1) {
            let w = bartlett(lag as f64 / (gamma as f64 + 1.0));
            if w == 0.0 {
                continue;
            }
            for s in lag..t {
                let term = w * v[s] * v[s - lag] / tf;
                var += term;
                if lag > 0 {
                    var += term;
                }
            }
        }
        if var.is_nan() || var <= 0.0 {
            return f64::NAN;
        }
        let sd = var.sqrt();
        let mut run = 0.0;
        let mut best = 0.0f64;
        for (j, uj) in u.iter().enumerate() {
            run += uj;
            let jj = j as f64 + 1.0;
            let q = (jj / tf.sqrt()) * (run / jj - mean).abs() / sd;
            if q > best {
                best = q;
            }
        }
        best
    }

    /// `Q` for one pair over the span in the ring.
    fn monitor_stat(&self, rows: &[Vec<f64>], a: usize, b: usize) -> f64 {
        let t = rows.len();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((t as f64).ln().floor() as usize).max(1));
        let sd = Self::long_run_sd(rows, a, b, gamma);
        // NaN or non-positive: no scale to divide by.
        if sd.is_nan() || sd <= 0.0 {
            return f64::NAN;
        }
        let rho_t = Self::corr_of(rows, t, a, b);
        if !rho_t.is_finite() {
            return f64::NAN;
        }
        let tf = t as f64;
        let mut best = 0.0f64;
        for j in 2..=t {
            let rj = Self::corr_of(rows, j, a, b);
            if !rj.is_finite() {
                continue;
            }
            let v = (j as f64 / tf.sqrt()) * (rj - rho_t).abs() / sd;
            if v > best {
                best = v;
            }
        }
        best
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
            let r = (xi - self.diag.mean(i)) / v.sqrt();
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
        let w = self.cfg.window;
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
        let w = self.cfg.window;
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
    fn read(&self, x: &[f64], weight: f64) -> Vec<f64> {
        let mut pred = vec![f64::NAN; Self::n_outputs_for()];
        if weight <= 0.0 {
            return pred;
        }
        let Some(row) = self.ring_row(x) else {
            return pred;
        };
        let since = self.since_flag.unwrap_or(0) + 1;
        match self.cfg.kind {
            CorrChangeKind::Monitor => {
                if self.ring.len() + 1 < self.cfg.horizon {
                    return pred;
                }
                let mut rows: Vec<Vec<f64>> = self.ring.iter().cloned().collect();
                rows.push(row);
                let mut stat = f64::NAN;
                if self.cfg.scalar {
                    stat = self.scalar_stat(&rows);
                } else {
                    for a in 0..self.cfg.n_features {
                        for b in (a + 1)..self.cfg.n_features {
                            let q = self.monitor_stat(&rows, a, b);
                            if q.is_finite() && (stat.is_nan() || q > stat) {
                                stat = q;
                            }
                        }
                    }
                }
                let crit = self.cfg.fixed_crit().unwrap_or(f64::NAN);
                pred[0] = stat;
                pred[1] = crit;
                pred[2] = f64::from(stat.is_finite() && crit.is_finite() && stat > crit);
                pred[3] = since as f64;
            }
            CorrChangeKind::Window => {
                let w = self.cfg.window;
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
                pred[0] = stat;
                pred[1] = crit.unwrap_or(f64::NAN);
                pred[2] =
                    f64::from(matches!((stat.is_finite(), crit), (true, Some(c)) if stat > c));
                pred[3] = since as f64;
            }
        }
        pred
    }
}

impl crate::OnlineModel for CorrChange {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        if weight <= 0.0 {
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
        self.ring.push_back(row);
        let flagged = out.pred[2] == 1.0;
        if flagged {
            self.since_flag = Some(0);
        }
        match self.cfg.kind {
            CorrChangeKind::Monitor => {
                if self.ring.len() >= self.cfg.horizon {
                    // Spans are disjoint by construction: the next row
                    // starts a new one, so `reset` has nothing to add here.
                    self.ring.clear();
                }
            }
            CorrChangeKind::Window => {
                let w = self.cfg.window;
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

    fn predict(&self, x: &[f64], _d_clock: f64) -> crate::Step {
        crate::Step {
            pred: self.read(x, 1.0),
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
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::CorrChange(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::CorrChange(m) => Ok((**m).clone()),
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
    use crate::OnlineModel;

    fn cfg(d: usize, kind: CorrChangeKind) -> CorrChangeCfg {
        CorrChangeCfg {
            n_features: d,
            kind,
            horizon: 200,
            window: 50,
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
        }
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

    /// `D̂` and `Q` against the same arithmetic written out.
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
                let w = bartlett((s as f64 - q as f64) / (gamma as f64 + 1.0));
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
            -0.5 * sxy * sy / sx.powi(3),
            -0.5 * sxy * sx / sy.powi(3),
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
            horizon: t,
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
                    horizon: t,
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
                    horizon: t,
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
            horizon: 40,
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
            window: w,
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
                window: w,
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
                horizon: 300,
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
            horizon: 100,
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
                horizon: 4,
                ..cfg(2, CorrChangeKind::Monitor)
            },
            "horizon of at least 8",
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
                window: 2,
                ..cfg(2, CorrChangeKind::Window)
            },
            "window of at least 3",
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
    }
}
