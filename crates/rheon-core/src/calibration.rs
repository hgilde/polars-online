//! `Calibration`: Mincer and Zarnowitz's (1969) regression of the outcome on
//! its prediction, kept as exponentially weighted moments with a memory of
//! its own (docs/PLAN.md task 221 (a)).
//!
//! A prediction is calibrated when `E[y | pred] = pred`: the regression
//! `y = a + b·pred + e` has `a = 0` and `b = 1`. The slope is the multiplier
//! to put on a prediction, and the intercept what to add after it. Each
//! row's `(pred, y)` joins EW means and centred co-moments, the weighted
//! Welford recursion of [`crate::EwCov`] with weights `ω`, the row's weight
//! times the decay since it:
//!
//! ```text
//! W'  = lam·W + w            Q' = lam²·Q + w²        n_kish = W² / Q
//! m_p, m_y                   the EW means
//! C_pp, C_py, C_yy           the EW centred co-moments (mean form, over W)
//! b   = C_py / C_pp          a = m_y − b·m_p
//! s²  = C_yy − C_py² / C_pp  the residual's EW mean square
//! ```
//!
//! The joint test of `a = 0, b = 1` is Wald's, read at Kish's effective
//! sample size in place of `n`:
//!
//! ```text
//! wald = (n_kish − 2) · ((m_y − m_p)² + (b − 1)²·C_pp) / s²
//! ```
//!
//! — `d'·M·d / (s²/(n_kish − 2))` with `d = (a, b − 1)` and
//! `M = [[1, m_p], [m_p, C_pp + m_p²]]` the EW second moments of `(1, pred)`;
//! `a + (b − 1)·m_p` is `m_y − m_p`, the mean residual, which makes the
//! form one subtraction short of cancellation. With no memory (`lam = 1`)
//! and unit weights `n_kish = n`, and `wald / 2` is the F statistic of the
//! ordinary least squares test (statsmodels' `f_test`), `F(2, n − 2)` under
//! Gaussian errors and `wald` itself `χ²(2)` as `n` grows. With a memory the
//! weights are uneven and the variance of a weighted mean is `1/n_kish` of
//! one row's, so the same form is approximately `χ²(2)` on each row; the
//! rows' values are then correlated over about a half-life.
//!
//! **Under a horizon** (a target that looks ahead `h` rows; task 232 (3))
//! the residuals overlap and the form above reads them as independent, so
//! the test is Wald's with Newey and West's variance in place of `s²/n`:
//! `wald = d' V⁻¹ d` with `V` the sandwich [`crate::Sandwich`] of the
//! regression on `(1, pred)` at `L = 2h` lags, its meat built from the
//! residual under the null, `y − pred` (`a = 0, b = 1` leave nothing else),
//! so it needs no fit of its own.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Mincer–Zarnowitz calibration of one prediction slot. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// The EW moments of `(pred, y)`, never windowed: no runs.
    joint: crate::EwCov,
    /// Under a horizon, the sandwich of the regression on `(1, pred)` at
    /// Newey and West's lags, its meat from `y − pred` (module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hac: Option<crate::Sandwich>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibration {
    pub fn new() -> Self {
        Self::with_lags(0)
    }

    /// A calibration whose Wald statistic reads Newey and West's variance
    /// at `lags` lags (module docs), the plain form at 0.
    pub fn with_lags(lags: usize) -> Self {
        Self {
            joint: crate::EwCov::new(2).without_runs(),
            hac: (lags > 0).then(|| crate::Sandwich::new(1, true, lags)),
        }
    }

    /// Whether a restored accumulator is shaped as a fresh one at `lags`:
    /// its moments two wide, with the sum of squared weights Kish's size
    /// reads, and the sandwich at those lags.
    pub fn has_shape(&self, lags: usize) -> bool {
        self.joint.has_shape(2)
            && self.joint.q_sum().is_some()
            && match &self.hac {
                None => lags == 0,
                Some(h) => lags > 0 && h.has_shape(1, true, lags),
            }
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        if let Some(h) = self.hac.as_mut() {
            h.clear_lags();
        }
    }

    /// The EW weight of the scored rows.
    pub fn n_eff(&self) -> f64 {
        self.joint.n_eff()
    }

    /// Kish's effective sample size `W²/Q`, `None` before the first scored
    /// row of positive weight.
    pub fn n_kish(&self) -> Option<f64> {
        self.joint.n_kish()
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored: the means stay, the weights shrink.
    pub fn age(&mut self, lam: f64) {
        self.joint.decay(lam);
        if let Some(h) = self.hac.as_mut() {
            h.age(lam);
        }
    }

    /// One scored row: its prediction, its outcome, the step's decay and its
    /// weight. A row with either value not finite, or of weight 0, only ages
    /// the moments (CLAUDE.md hard rule 9): it is not evidence about the
    /// calibration, and a zero-weight first row would be `0/0`.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        if !(pred.is_finite() && y.is_finite() && w > 0.0) {
            return self.age(lam);
        }
        self.joint.update(&[pred, y], lam, w);
        if let Some(h) = self.hac.as_mut() {
            h.update(&[pred], y - pred, lam, w);
        }
    }

    /// `b`, the slope of `y` on `pred`: `None` until the predictions have a
    /// spread.
    pub fn slope(&self) -> Option<f64> {
        let var_p = self.joint.var(0);
        (var_p > 0.0 && self.n_eff() > 0.0).then(|| self.joint.cov(0, 1) / var_p)
    }

    /// `a`, the intercept: `m_y − b·m_p`, where the slope is.
    pub fn intercept(&self) -> Option<f64> {
        self.slope()
            .map(|b| self.joint.mean(1) - b * self.joint.mean(0))
    }

    /// The Wald statistic of `a = 0, b = 1` at Kish's size (module docs).
    /// `None` until there is a slope, more than two rows' worth of Kish's
    /// size, and a residual with a spread: a fit through every point has no
    /// error to scale the distance by.
    pub fn wald(&self) -> Option<f64> {
        let b = self.slope()?;
        let n = self.n_kish()?;
        if n <= 2.0 {
            return None;
        }
        if let Some(h) = &self.hac {
            let a = self.joint.mean(1) - b * self.joint.mean(0);
            return h.wald(&[0], true, &[0, 1], &[a, b - 1.0]);
        }
        let (c_pp, c_py, c_yy) = (self.joint.var(0), self.joint.cov(0, 1), self.joint.var(1));
        let s2 = c_yy - c_py * c_py / c_pp;
        // A NaN spread is no spread either.
        if s2.is_nan() || s2 <= 0.0 {
            return None;
        }
        let mean_resid = self.joint.mean(1) - self.joint.mean(0);
        let d = mean_resid * mean_resid + (b - 1.0) * (b - 1.0) * c_pp;
        Some((n - 2.0) * d / s2)
    }
}

/// The calibration of a quantile fit (task 232 (6); review round 6, G-1,
/// F-2): a prediction of the `q`-th quantile is calibrated when the outcome
/// falls below it on a share `q` of rows, which Mincer and Zarnowitz's
/// regression of the outcome on the prediction -- a mean's calibration --
/// does not test. Each scored row's indicator `d = 1{y < pred} − q` joins EW
/// sums at weight `ω`:
///
/// ```text
/// coverage = q + Σ ω d / Σ ω                     (the share below the prediction)
/// wald     = (Σ ω d)² / (q (1 − q) Σ ω²)          ~ χ²(1)
/// ```
///
/// the binomial test of the share at Kish's size: `d` has mean 0 and
/// variance `q (1 − q)` at a calibrated quantile, and `(Σ ω d)²/Σ ω²` is
/// `n_kish (coverage − q)²`. Under a horizon the indicators overlap as the
/// residuals do, and the variance takes Newey and West's Bartlett-weighted
/// lag products of `ω d` over `L` lags, as [`crate::Breaks`]' does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coverage {
    /// The quantile level, in `(0, 1)`.
    q: f64,
    /// `Σ ω`, `Σ ω²` and `Σ ω d` over the scored rows.
    w: f64,
    w2: f64,
    d: f64,
    /// Newey and West's lags, the products per lag and the last `ω d`
    /// newest first; 0 and empty without a horizon.
    #[serde(default)]
    lags: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    p: Vec<f64>,
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    ring: VecDeque<f64>,
}

impl Coverage {
    /// An empty coverage of the `q`-th quantile at `lags` Newey-West lags.
    pub fn new(q: f64, lags: usize) -> Self {
        Self {
            q,
            w: 0.0,
            w2: 0.0,
            d: 0.0,
            lags,
            p: vec![0.0; lags],
            ring: VecDeque::with_capacity(lags),
        }
    }

    /// Whether a restored accumulator is shaped for `q` and `lags`.
    pub fn has_shape(&self, q: f64, lags: usize) -> bool {
        self.q == q
            && self.lags == lags
            && self.p.len() == lags
            && self.ring.len() <= lags
            && [self.w, self.w2, self.d]
                .iter()
                .chain(&self.p)
                .chain(&self.ring)
                .all(|v| v.is_finite())
            && self.w >= 0.0
            && self.w2 >= 0.0
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
        self.w2 *= lam * lam;
        self.d *= lam;
        self.p.iter_mut().for_each(|v| *v *= lam * lam);
        self.ring.iter_mut().for_each(|v| *v *= lam);
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// One scored row: its prediction, its outcome, the step's decay and its
    /// weight. A value that is not finite, or a weight of 0, only ages
    /// (CLAUDE.md hard rule 9).
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(pred.is_finite() && y.is_finite() && w > 0.0) {
            return;
        }
        let d = f64::from(u8::from(y < pred)) - self.q;
        self.w += w;
        self.w2 += w * w;
        self.d += w * d;
        if self.lags > 0 {
            let u = w * d;
            for (l, back) in self.ring.iter().enumerate() {
                self.p[l] += u * back;
            }
            self.ring.push_front(u);
            self.ring.truncate(self.lags);
        }
    }

    /// The EW share of rows whose outcome fell below the prediction: `None`
    /// before the first scored row.
    pub fn coverage(&self) -> Option<f64> {
        (self.w > 0.0).then(|| self.q + self.d / self.w)
    }

    /// The binomial Wald statistic of `coverage = q` at Kish's size, over
    /// the long-run variance under a horizon: `None` before the first scored
    /// row, and where the lag products leave no variance.
    pub fn wald(&self) -> Option<f64> {
        if self.w2 <= 0.0 {
            return None;
        }
        let l1 = self.lags as f64 + 1.0;
        let lagged: f64 = self
            .p
            .iter()
            .enumerate()
            .map(|(l, v)| (1.0 - (l + 1) as f64 / l1) * v)
            .sum();
        let var = self.q * (1.0 - self.q) * self.w2 + 2.0 * lagged;
        (var > 0.0).then(|| self.d * self.d / var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// The weighted least squares fit of `y` on `(1, pred)` and its Wald
    /// statistic at Kish's size, from every row and its weight as it now
    /// stands: the normal equations solved by faer's LU (`crate::oracle`),
    /// the residual summed row by row. No moment of the accumulator's.
    fn by_definition(rows: &[(f64, f64, f64)]) -> (f64, f64, f64) {
        let (mut m, mut r) = (vec![0.0; 4], vec![0.0; 2]);
        let (mut sw, mut sw2) = (0.0, 0.0);
        for &(p, y, w) in rows {
            m[0] += w;
            m[1] += w * p;
            m[2] += w * p;
            m[3] += w * p * p;
            r[0] += w * y;
            r[1] += w * p * y;
            sw += w;
            sw2 += w * w;
        }
        let beta = crate::oracle::solve(&m, &r);
        let ssr: f64 = rows
            .iter()
            .map(|&(p, y, w)| w * (y - beta[0] - beta[1] * p).powi(2))
            .sum();
        let n_kish = sw * sw / sw2;
        let s2 = ssr / sw * n_kish / (n_kish - 2.0);
        let d = [beta[0], beta[1] - 1.0];
        // `d'·(X'ΩX / W)·d`, the EW second moments of `(1, pred)` summed
        // directly.
        let quad: f64 = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| d[i] * m[i * 2 + j] / sw * d[j])
            .sum();
        (beta[0], beta[1], n_kish * quad / s2)
    }

    /// How far [`by_definition`]'s own rounding may have moved its slope,
    /// intercept and statistic (docs/PLAN.md task 274), to first order:
    ///
    /// - its sums, terms of one to three products each, every addition `u`
    ///   of the partial sum it makes, read off the run (Higham §3.3), through
    ///   the normal equations' forward error ([`crate::oracle::forward_error`])
    ///   in the frame of the predictions' level, and the LU's own (its
    ///   residual);
    /// - its weights, each decayed `age` times by repeated products and so
    ///   within `age·u` of itself, through `∂β/∂(ln wᵣ) = wᵣM⁻¹zᵣeᵣ`;
    /// - the statistic `(n − 2)·d'Md/ssr`: through `β`, `2|Md|ᵀΔβ/d'Md` of
    ///   it (the residuals' sum of squares does not move with `β` to first
    ///   order at the least-squares fit), through the sums `d'ΔMd`, the
    ///   weights through `∂ ln W/∂ ln wᵣ = wᵣ[2(1/Σw − wᵣ/Σw²)n/(n − 2) +
    ///   ((d'zᵣ)² + 2(d'zᵣ)eᵣ)/d'Md − eᵣ²/ssr]`, the residuals' rounding
    ///   into `ssr`, and twelve roundings of its own formula.
    fn definition_error(rows: &[(f64, f64, f64)], now: usize) -> [f64; 3] {
        use crate::oracle::{self, U, gamma};
        let (mut m, mut r) = (vec![0.0; 4], vec![0.0; 2]);
        let (mut em, mut er) = (vec![0.0; 4], vec![0.0; 2]);
        let (mut sw, mut sw2, mut esw, mut esw2) = (0.0, 0.0, 0.0, 0.0);
        let add = |s: &mut f64, e: &mut f64, t: f64, k: f64| {
            *s += t;
            *e += k * U * t.abs() + U * s.abs();
        };
        for &(p, y, w) in rows {
            add(&mut m[0], &mut em[0], w, 0.0);
            add(&mut m[1], &mut em[1], w * p, 1.0);
            add(&mut m[2], &mut em[2], w * p, 1.0);
            add(&mut m[3], &mut em[3], w * p * p, 2.0);
            add(&mut r[0], &mut er[0], w * y, 1.0);
            add(&mut r[1], &mut er[1], w * p * y, 2.0);
            add(&mut sw, &mut esw, w, 0.0);
            add(&mut sw2, &mut esw2, w * w, 1.0);
        }
        let beta = oracle::solve(&m, &r);
        // The errors are carried in the frame of `(1, p − 5)`, the
        // predictions' level, where the normal equations are well
        // conditioned: `β = Sβ_c`, `S = [[1, −5], [0, 1]]`, `M_c = SᵀMS`,
        // and the sums' errors `|S|ᵀΔM|S|` there. In the raw frame an
        // error in the intercept is −5 times one in the slope, and bounds
        // taken entry by entry there part the two and lose that.
        let lv = 5.0;
        let s_abs = [1.0, lv, 0.0, 1.0];
        let st_abs = [1.0, 0.0, lv, 1.0];
        let mc = {
            let ms = [m[0], m[1] - lv * m[0], m[2], m[3] - lv * m[2]];
            [ms[0], ms[1], ms[2] - lv * ms[0], ms[3] - lv * ms[1]]
        };
        let bc = [beta[0] + lv * beta[1], beta[1]];
        let emc = {
            let a = crate::sandwich::rounding::abs_mul(&st_abs, &em, 2);
            crate::sandwich::rounding::abs_mul(&a, &s_abs, 2)
        };
        let erc = [er[0], er[1] + lv * er[0]];
        let mci = oracle::inverse(&mc);
        let res = oracle::solve_error(&m, &r, &beta);
        let mut dbc: Vec<f64> = oracle::forward_error(&mc, &bc, &emc, &erc);
        // The LU's own, `β − β̂ = M⁻¹·res`, is `M_c⁻¹Sᵀ·res` in this frame;
        // `res` read back from its bound in the raw frame, `|M|·Δβ̂`.
        let raw_res = oracle::abs_times(&m, &res);
        let own = oracle::abs_times(&mci, &[raw_res[0], raw_res[1] + lv * raw_res[0]]);
        dbc[0] += own[0];
        dbc[1] += own[1];
        // The same in the raw frame, which for the coefficients themselves
        // (not the statistic) can be the tighter of the two; both are
        // bounds, and each coefficient takes the smaller.
        let mi = oracle::inverse(&m);
        let mut dbr: Vec<f64> = oracle::forward_error(&m, &beta, &em, &er)
            .iter()
            .zip(&res)
            .map(|(a, b)| a + b)
            .collect();
        let resid = |p: f64, y: f64| y - beta[0] - beta[1] * p;
        let n = sw * sw / sw2;
        let d = [beta[0], beta[1] - 1.0];
        let dc = [d[0] + lv * d[1], d[1]];
        let g = [mc[0] * dc[0] + mc[1] * dc[1], mc[2] * dc[0] + mc[3] * dc[1]];
        let dmd = dc[0] * g[0] + dc[1] * g[1];
        let ssr: f64 = rows.iter().map(|&(p, y, w)| w * resid(p, y).powi(2)).sum();
        let total = rows.len();
        let mut dln_w = 0.0;
        let mut dssr = 0.0;
        for (t, &(p, y, w)) in rows.iter().enumerate() {
            let e = resid(p, y);
            let age = (now - t) as f64 * U;
            for (a, dba) in dbc.iter_mut().enumerate() {
                *dba += (w * (mci[a * 2] + mci[a * 2 + 1] * (p - lv)) * e).abs() * age;
            }
            for (a, dba) in dbr.iter_mut().enumerate() {
                *dba += (w * (mi[a * 2] + mi[a * 2 + 1] * p) * e).abs() * age;
            }
            let dz = d[0] + d[1] * p;
            let gw = w
                * (2.0 * (1.0 / sw - w / sw2) * n / (n - 2.0) + (dz * dz + 2.0 * dz * e) / dmd
                    - e * e / ssr);
            dln_w += gw.abs() * age;
            // `e` rounds by `u` of each of its three terms, its square and
            // its weight by two more; the sum by `γ_N`.
            let de = U * (y.abs() + beta[0].abs() + 2.0 * (beta[1] * p).abs());
            dssr += w * (2.0 * e.abs() * de + 3.0 * U * e * e);
        }
        dssr += gamma(total) * ssr;
        let wald = (n - 2.0) * dmd / ssr;
        let dn = 2.0 * esw / sw + esw2 / sw2;
        let dwald = wald
            * (2.0 * (g[0].abs() * dbc[0] + g[1].abs() * dbc[1]) / dmd
                + oracle::abs_form(&dc, &emc, &dc) / dmd
                + dln_w
                + dssr / ssr
                + dn * n / (n - 2.0)
                + 12.0 * U);
        let db = [(dbc[0] + lv * dbc[1]).min(dbr[0]), dbc[1].min(dbr[1])];
        [db[0], db[1], dwald]
    }

    /// How far [`Calibration`]'s own rounding may have moved its slope,
    /// intercept and statistic, from its moments' errors as
    /// [`crate::ewcov::rounding::Rounding`] carries them along the run:
    /// `b = C_py/C_pp`, `a = m_y − b·m_p` and `(n − 2)·d/s²`, `n = W²/Q`,
    /// `s² = C_yy − C_py²/C_pp`, `d = (m_y − m_p)² + (b − 1)²C_pp`, their
    /// inputs' errors carried to first order ([`crate::oracle::propagate`])
    /// and each formula's own roundings counted.
    fn model_error(tr: &crate::ewcov::rounding::Rounding) -> [f64; 3] {
        use crate::oracle::{U, propagate};
        let (c, m) = (&tr.c, &tr.m);
        let x = [tr.w, tr.q, m[0], m[1], c[0], c[1], c[3]];
        let dx = [
            tr.ew * tr.w,
            tr.eq * tr.q,
            tr.em[0],
            tr.em[1],
            tr.ec[0],
            tr.ec[1],
            tr.ec[3],
        ];
        let slope = |v: &[f64]| v[5] / v[4];
        let icept = |v: &[f64]| v[3] - v[5] / v[4] * v[2];
        let wald = |v: &[f64]| {
            let n = v[0] * v[0] / v[1];
            let b = v[5] / v[4];
            let s2 = v[6] - v[5] * v[5] / v[4];
            let d = (v[3] - v[2]).powi(2) + (b - 1.0).powi(2) * v[4];
            (n - 2.0) * d / s2
        };
        let (b, a, w) = (slope(&x), icept(&x), wald(&x));
        let s2 = c[3] - c[1] * c[1] / c[0];
        let mr = m[1] - m[0];
        let dd = (mr * mr + (b - 1.0).powi(2) * c[0]).abs();
        let rw = 5.0 * U
            + U * (2.0 * c[1] * c[1] / c[0] + s2.abs()) / s2.abs()
            + U * (3.0 * mr * mr + 4.0 * (b - 1.0).powi(2) * c[0] + dd) / dd;
        [
            propagate(slope, &x, &dx) + U * b.abs(),
            propagate(icept, &x, &dx)
                + 2.0 * U * ((b * m[0]).abs() + a.abs())
                + U * b.abs() * m[0].abs(),
            propagate(wald, &x, &dx) + rw * w.abs(),
        ]
    }

    /// Slope, intercept and Wald against the definition, row by row, under a
    /// decay, uneven weights with zeros among them, and a miscalibrated
    /// prediction, `y = 0.3 + 0.7·pred + e`.
    #[test]
    fn calibration_matches_weighted_least_squares_at_kish_size() {
        let mut c = Calibration::new();
        let mut tr = crate::ewcov::rounding::Rounding::new(2, 0.0);
        let mut s = 7u64;
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let lam = 0.97;
        for i in 0..400 {
            let p = 2.0 * lcg(&mut s) + 5.0;
            let y = 0.3 + 0.7 * p + 0.4 * lcg(&mut s);
            let w = if i % 11 == 3 {
                0.0
            } else {
                0.5 + lcg(&mut s) + 0.5
            };
            c.update(p, y, lam, w);
            if w > 0.0 {
                tr.update(&[p, y], lam, w);
            } else {
                tr.decay(lam);
            }
            rows.iter_mut().for_each(|r| r.2 *= lam);
            rows.push((p, y, w));
            if i < 5 {
                continue;
            }
            let (a, b, wald) = by_definition(&rows);
            // Task 274: each side's rounding, the model's carried along its
            // run ([`model_error`]) and the definition's ([`definition_error`]),
            // where each was `1e-9` (relative to `max(1, W)` for the statistic).
            // Measured on aarch64 (macOS): at most 0.090 (slope), 0.091 (intercept) and 0.0084
            // (statistic) of the bound.
            let (dm, dd) = (model_error(&tr), definition_error(&rows, i));
            crate::oracle::assert_within(
                c.slope().unwrap(),
                b,
                dm[0] + dd[1],
                format_args!("slope at row {i}"),
            );
            crate::oracle::assert_within(
                c.intercept().unwrap(),
                a,
                dm[1] + dd[0],
                format_args!("intercept at row {i}"),
            );
            crate::oracle::assert_within(
                c.wald().unwrap(),
                wald,
                dm[2] + dd[2],
                format_args!("wald at row {i}"),
            );
        }
        let (a, b) = (c.intercept().unwrap(), c.slope().unwrap());
        assert!((b - 0.7).abs() < 0.05 && (a - 0.3).abs() < 0.3, "{a} {b}");
        assert!(c.wald().unwrap() > 100.0, "a slope of 0.7 is far from 1");
    }

    /// No memory and unit weights: Kish's size is the row count, and the
    /// statistic is twice the ordinary least squares F (the Python suite
    /// holds it to statsmodels' `f_test`).
    #[test]
    fn without_memory_kish_size_is_the_row_count() {
        let mut c = Calibration::new();
        let mut s = 3u64;
        for i in 0..50 {
            let p = lcg(&mut s);
            c.update(p, p + 0.1 * lcg(&mut s), 1.0, 1.0);
            assert!((c.n_kish().unwrap() - (i + 1) as f64).abs() < 1e-9);
        }
    }

    /// Under a horizon, Wald's statistic of `a = 0, b = 1` with Newey and
    /// West's variance by its definition: `B = Σ ω z z'`, `z = (1, pred)`,
    /// the meat's double sum of `u_t = ω_t (y_t − pred_t) z_t` over pairs at
    /// most `L` rows apart at Bartlett's weights, `B⁻¹ S B⁻¹` and its
    /// inverse by faer's LU (`crate::oracle`), under a decay, uneven weights
    /// and zeros, with residuals that carry a lag. The variance is summed
    /// about the predictions' level, 5, and mapped to `(1, pred)`
    /// ([`crate::sandwich::rounding::covariance_bound`]).
    ///
    /// Task 274: within the two sides' rounding, where it was `1e-7`
    /// relative: the variance's, entry by entry, through `−x'ΔVx` (`x =
    /// V⁻¹d`, `d` the model's own distance on both sides) in the level's
    /// frame, and the two quadratic forms', the model's Cholesky and the
    /// oracle's LU.
    /// Measured on aarch64 (macOS): at most 0.019 of the bound.
    #[test]
    fn under_a_horizon_the_wald_is_newey_and_wests() {
        use crate::sandwich::rounding::{DefRow, Track, covariance_bound};
        let lags = 4usize;
        let lam = 0.995;
        let mut c = Calibration::with_lags(lags);
        let mut track = Track::new(1, lags);
        let mut s = 31u64;
        // Folded rows only.
        let mut rows: Vec<DefRow> = Vec::new();
        let mut prev = 0.0;
        for i in 0..300 {
            let p = 2.0 * lcg(&mut s) + 5.0;
            let shock = 0.6 * prev + lcg(&mut s);
            prev = shock;
            let y = 0.2 + 0.9 * p + shock;
            let w = if i % 13 == 6 { 0.0 } else { 1.0 + lcg(&mut s) };
            c.update(p, y, lam, w);
            // The calibration ages its sandwich, or updates it, as the
            // sandwich updates itself on a row of weight 0.
            track.update(&[p], y - p, lam, w);
            rows.iter_mut().for_each(|r| r.w *= lam);
            if w > 0.0 {
                rows.push(DefRow {
                    x: vec![p],
                    e: y - p,
                    w,
                    born: i,
                });
            }
            if i < 30 || i % 41 != 0 {
                continue;
            }
            let b = covariance_bound(&track, &rows, i, &[0.0, 5.0], (lags, lags));
            let d = [c.intercept().unwrap(), c.slope().unwrap() - 1.0];
            let want = crate::oracle::quad_form(&b.v, &d);
            let got = c.wald().unwrap();
            // The variance's errors move the form by `x_l'ΔV_l x_l` in the
            // level's frame (`d_l = (d₀ + 5d₁, d₁)`), where they are bounded
            // entry by entry without the level's cancellation.
            let dl = [d[0] + 5.0 * d[1], d[1]];
            let xl = crate::oracle::solve(&b.v_level, &dl);
            let bound = crate::oracle::abs_form(&xl, &b.dv_level, &xl)
                + crate::oracle::quad_error(&b.v, &d, &crate::oracle::cholesky_backward(&b.v))
                + crate::oracle::quad_form_error(&b.v, &d);
            crate::oracle::assert_within(got, want, bound, format_args!("wald at row {i}"));
        }
        assert!(c.has_shape(lags) && !c.has_shape(0));
    }

    /// A quantile fit's coverage and its Wald statistic by their definition,
    /// from every scored row at its present weight -- the share below the
    /// prediction, and `(Σωd)² / (q(1−q)Σω² + 2 Σ_l b_l Σ ω_t ω_{t−l} d_t
    /// d_{t−l})` -- under a decay, uneven weights, zeros, a row with no
    /// outcome and indicators that carry a lag.
    #[test]
    fn coverage_is_its_definition() {
        let (q, lags, lam) = (0.8, 3usize, 0.98);
        let mut c = Coverage::new(q, lags);
        let mut s = 41u64;
        // (d, weight now), scored rows only.
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let mut prev = 0.0;
        for i in 0..300 {
            let shock = 0.7 * prev + lcg(&mut s);
            prev = shock;
            let (pred, y) = (0.3, shock + 0.6);
            let w = if i % 11 == 5 { 0.0 } else { 1.0 + lcg(&mut s) };
            let y = if i % 17 == 9 { f64::NAN } else { y };
            c.update(pred, y, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            if w > 0.0 && y.is_finite() {
                rows.push((f64::from(u8::from(y < pred)) - q, w));
            }
            if i < 20 || i % 23 != 0 {
                continue;
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let sw2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
            let sd: f64 = rows.iter().map(|r| r.1 * r.0).sum();
            let mut lagged = 0.0;
            for l in 1..=lags {
                let b = 1.0 - l as f64 / (lags as f64 + 1.0);
                for t in l..rows.len() {
                    lagged += b * rows[t].1 * rows[t - l].1 * rows[t].0 * rows[t - l].0;
                }
            }
            let tol = 1e-10;
            assert!(
                (c.coverage().unwrap() - (q + sd / sw)).abs() < tol,
                "row {i}"
            );
            let want = sd * sd / (q * (1.0 - q) * sw2 + 2.0 * lagged);
            assert!(
                (c.wald().unwrap() - want).abs() < tol * want.max(1.0),
                "row {i}"
            );
        }
        assert!(c.has_shape(q, lags) && !c.has_shape(0.5, lags));
        let mut z = Coverage::new(q, 0);
        z.update(1.0, 0.0, 0.9, 0.0);
        assert!(
            z.coverage().is_none() && z.wald().is_none(),
            "a zero weight only ages"
        );
    }

    /// A zero-weight row, the first included, and a row with a value not a
    /// number only age the moments; until two rows with a spread there is
    /// no slope, and with no residual spread no statistic.
    #[test]
    fn zero_weight_and_missing_rows_only_age() {
        let mut c = Calibration::new();
        c.update(1.0, 2.0, 0.9, 0.0);
        assert_eq!(c.n_eff(), 0.0);
        assert!(c.slope().is_none() && c.wald().is_none() && c.n_kish().is_none());
        c.update(f64::NAN, 2.0, 0.9, 1.0);
        c.update(1.0, f64::INFINITY, 0.9, 1.0);
        assert_eq!(c.n_eff(), 0.0);
        c.update(1.0, 2.0, 0.9, 1.0);
        assert!(c.slope().is_none(), "one row has no spread");
        let before = c.clone();
        c.update(3.0, 1.0, 0.5, 0.0);
        assert_eq!(c.n_eff(), before.n_eff() * 0.5);
        assert_eq!(c.joint.mean(0), before.joint.mean(0));
        // Points on a line: a slope, and no residual to scale the test by.
        let mut line = Calibration::new();
        for p in [1.0, 2.0, 3.0, 4.0] {
            line.update(p, 2.0 * p, 1.0, 1.0);
        }
        assert!((line.slope().unwrap() - 2.0).abs() < 1e-12);
        assert!(line.wald().is_none());
        assert!(line.has_shape(0));
        // A memory decayed to nothing -- a gap past 1,075 half-lives, whose
        // factor is 0 -- keeps the means and has no slope to read.
        let mut gone = line.clone();
        gone.age(0.0);
        assert_eq!(gone.n_eff(), 0.0);
        assert!(gone.slope().is_none() && gone.intercept().is_none());
    }

    /// A restored calibration is refused unless its sandwich is at the lags
    /// asked for: none without a horizon, the same count with one.
    #[test]
    fn a_restored_calibration_must_be_at_its_lags() {
        assert!(Calibration::with_lags(0).has_shape(0));
        assert!(Calibration::with_lags(3).has_shape(3));
        assert!(!Calibration::with_lags(3).has_shape(2));
        assert!(!Calibration::with_lags(3).has_shape(0));
        assert!(!Calibration::with_lags(0).has_shape(3));
        // A sandwich with no lags where none is asked for is not the shape
        // a fresh one has.
        let odd = Calibration {
            joint: crate::EwCov::new(2).without_runs(),
            hac: Some(crate::Sandwich::new(1, true, 0)),
        };
        assert!(!odd.has_shape(0));
    }

    /// The rows behind a cleared run are not lags of the rows after it:
    /// the Wald statistic under a horizon is the definition's with the lag
    /// products taken within each run, against a calibration fed the same
    /// rows without the cut.
    #[test]
    fn a_cleared_run_pairs_nothing_across_it() {
        let lags = 3usize;
        let (mut cut, mut whole) = (Calibration::with_lags(lags), Calibration::with_lags(lags));
        let mut s = 17u64;
        let mut prev = 0.0;
        for i in 0..200 {
            if i == 120 {
                cut.clear_lags();
            }
            let p = 2.0 * lcg(&mut s) + 5.0;
            prev = 0.8 * prev + lcg(&mut s);
            let y = 0.1 + p + prev;
            cut.update(p, y, 0.99, 1.0);
            whole.update(p, y, 0.99, 1.0);
        }
        let (a, b) = (cut.wald().unwrap(), whole.wald().unwrap());
        assert!((a - b).abs() > 1e-6 * b, "{a} vs {b}");
        assert_eq!(cut.slope(), whole.slope(), "the moments do not see it");
    }

    /// A coverage's cleared run pairs nothing across it: the products of
    /// the rows on either side are left out of the variance, by the
    /// definition's double sum within each run.
    #[test]
    fn a_coverages_cleared_run_pairs_nothing_across_it() {
        let (q, lags) = (0.3, 2usize);
        let mut c = Coverage::new(q, lags);
        let mut s = 23u64;
        let mut runs: Vec<Vec<f64>> = vec![Vec::new()];
        for i in 0..60 {
            if i == 31 {
                c.clear_lags();
                runs.push(Vec::new());
            }
            let y = lcg(&mut s);
            c.update(0.0, y, 1.0, 1.0);
            runs.last_mut()
                .unwrap()
                .push(f64::from(u8::from(y < 0.0)) - q);
        }
        let all: Vec<f64> = runs.iter().flatten().copied().collect();
        let sd: f64 = all.iter().sum();
        let mut lagged = 0.0;
        for run in &runs {
            for l in 1..=lags {
                let b = 1.0 - l as f64 / (lags as f64 + 1.0);
                for t in l..run.len() {
                    lagged += b * run[t] * run[t - l];
                }
            }
        }
        let want = sd * sd / (q * (1.0 - q) * all.len() as f64 + 2.0 * lagged);
        let got = c.wald().unwrap();
        assert!(
            (got - want).abs() < 1e-12 * want.max(1.0),
            "{got} vs {want}"
        );
    }

    /// An outcome at the prediction is not below it: the indicator is
    /// `1{y < pred}`, as the coverage's definition reads.
    #[test]
    fn an_outcome_at_the_prediction_is_not_below_it() {
        let mut c = Coverage::new(0.25, 0);
        c.update(1.0, 1.0, 1.0, 1.0);
        assert_eq!(c.coverage(), Some(0.0));
        c.update(1.0, 0.5, 1.0, 1.0);
        assert_eq!(c.coverage(), Some(0.5));
    }

    /// A variance that underflows to 0 -- one row whose weight's square is
    /// the least subnormal, which `q (1 − q)` takes to 0 -- reads no
    /// statistic, where the division would read `0/0`.
    #[test]
    fn a_coverage_with_no_variance_reads_no_statistic() {
        let mut c = Coverage::new(0.5, 0);
        let w = 2.3e-162;
        assert!(w * w > 0.0 && 0.25 * (w * w) == 0.0);
        c.update(1.0, 0.0, 1.0, w);
        assert!(c.coverage().is_some());
        assert!(c.wald().is_none());
    }

    /// Task 252: the default calibration is a fresh one -- no lags, no rows,
    /// nothing to read -- and learns as one.
    #[test]
    fn the_default_calibration_is_a_fresh_one() {
        let mut d = Calibration::default();
        assert_eq!(d, Calibration::new());
        assert!(d.has_shape(0) && !d.has_shape(2));
        assert_eq!(d.n_eff(), 0.0);
        assert!(d.slope().is_none() && d.wald().is_none());
        let mut fresh = Calibration::new();
        for (p, y) in [(0.5, 1.0), (1.5, 1.25), (-0.5, -1.0)] {
            d.update(p, y, 0.9, 1.0);
            fresh.update(p, y, 0.9, 1.0);
        }
        assert_eq!(d, fresh);
        assert!(d.n_eff() > 0.0);
    }
}
