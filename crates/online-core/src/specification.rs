//! `Specification`: three tests of what a fit is missing, from its
//! out-of-sample residuals, with a memory of their own (docs/PLAN.md task
//! 221 (d)). Each is an exponentially weighted sum, `ω` each scored row's
//! weight times the decay since it, and each statistic reads Kish's size
//! `n = (Σω)² / Σω²` in place of the row count.
//!
//! **Ljung and Box (1978)**, over the lags `s+1 ..= s+L`: `s` is the
//! horizon a look-ahead target correlates its residuals over by
//! construction (0 for a target that does not look ahead). With `ē` the EW
//! mean residual,
//!
//! ```text
//! ρ_l = Σ ω_t (e_t − ē)(e_{t−l} − ē) / Σ ω_t (e_t − ē)²
//! Q   = n (n + 2) Σ_l ρ_l² / (n − l)                      ~ χ²(L)
//! ```
//!
//! `e_{t−l}` is the `l`-th scored residual before `e_t`; a pair is weighed
//! by the later row's `ω`. Run once with unit weights it is
//! `statsmodels`' `acorr_ljungbox` over those lags.
//!
//! **Under a finite memory it is Box and Pierce's (1970) `Q = n Σ_l ρ_l²`**
//! at Kish's `n` (review round 6, B-3). Ljung and Box's `(n + 2)/(n − l)`
//! corrects the variance of `ρ_l` in a sample of `n` rows, of which `n − l`
//! have a partner; under exponential weights every row but the oldest has
//! one, and `Var(ρ_l)` is `Σω²/(Σω)² = 1/n` itself. At Kish's size the
//! factor inflated `Q`: on iid residuals over 2,000 streams 20 half-lives
//! long, with 10 lags, Ljung and Box's form passed its 5% value on 15.8%,
//! 9.7%, 6.6%, 6.7% and 4.8% of streams at half-lives of 10, 20, 50, 100
//! and 200 rows, and Box and Pierce's on 4.8%, 5.7%, 5.3%, 6.0% and 4.6%.
//! Run once, Ljung and Box's held (5.3-7.7% at 30 to 1,000 rows) where Box
//! and Pierce's fell to 2.7% at 30, so the run-once form keeps it.
//!
//! Past a horizon the autocorrelations are read against Bartlett's (1946)
//! covariance for them beyond a moving average's order: a target summing
//! the next `s + 1` rows' shocks leaves an MA(`s`) residual, whose
//! autocorrelations past `s` are zero but spread wider than `1/n` and move
//! together. With `r_l = ρ_l sqrt((n + 2)/(n − l))` over the tested lags
//! run once, and `r_l = ρ_l` under a finite memory,
//!
//! ```text
//! γ(v)  = Σ_{|j| ≤ s} ρ_j ρ_{j+v}        (ρ_0 = 1, ρ_{−j} = ρ_j, 0 past s)
//! Q     = n r' C⁻¹ r,   C_{kl} = γ(|k − l|)                    ~ χ²(L)
//! ```
//!
//! which is the `Q` above when `s = 0`. On a five-row look-ahead
//! target with nothing missing, the plain `Q` over lags 5-14 passed its 5%
//! value on 61% of the rows of 200 streams, `Q` over Bartlett's variance
//! alone on 13-14%, and this one on the figures in `po.spec`'s table
//! (task 221 (d)).
//!
//! **Breusch and Pagan (1979)**, in Koenker's (1981) studentized form: the
//! squared residual regressed on the features, `LM = n R²`, `χ²(k)` with
//! `k` features, the `R²` from EW centred moments of `(x, e²)`.
//!
//! **Ramsey's (1969) RESET**, the residual regressed on `p`, `p²` and `p³`
//! (`p` the prediction, about the first one scored): the Lagrange
//! multiplier for the two powers,
//!
//! ```text
//! LM = n (R²_full − R²_p) / (1 − R²_p)                    ~ χ²(2)
//! ```
//!
//! `R²_p` the residual's on `p` alone. Run once it is `statsmodels`'
//! `compare_lm_test` of the two regressions.
//!
//! **Under a horizon** (task 232 (3); review round 6, B-1) the residuals
//! of a target that looks ahead `h` rows overlap, and against a persistent
//! feature `n R²` reads their shared shocks as a spread or a curvature:
//! beside an AR(1) feature at 0.95 and a five-row horizon, Breusch and
//! Pagan's form passed its 5% value on 26% of the rows of streams missing
//! nothing and RESET's on 41% (28% and 54% run once). So each becomes
//! Wald's test of the same coefficients with Newey and West's variance at
//! `L = 2h` lags, as `se_coef_hac` reads them ([`crate::Sandwich`]):
//!
//! ```text
//! Breusch-Pagan   γ' V_γ⁻¹ γ            γ the slopes of e² on x       ~ χ²(k)
//! RESET           β' V_β⁻¹ β            β those of p², p³ in e on
//!                                         (p, p², p³)                ~ χ²(2)
//! ```
//!
//! each `V` the sandwich of its regression, its meat built from the
//! residual under the null: `e² − ē²`, the squared residual less its EW mean
//! before the row, and `e − ē − b (p − p̄)`, the residual less its EW fit
//! on `p` alone before the row.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// The three tests of one slot. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Specification {
    /// The lags skipped and the lags tested.
    skip: usize,
    lags: usize,
    /// `Σω`, `Σω²`, `Σωe`, `Σωe²` over the scored rows.
    w: f64,
    w2: f64,
    e1: f64,
    e2: f64,
    /// Per lag, the skipped ones included: `Σ ω e e_lag`, `Σ ω e`, `Σ ω e_lag`, `Σ ω` over the
    /// rows with a partner that far back.
    pairs: Vec<[f64; 4]>,
    /// The last `skip + lags` scored residuals, newest first.
    ring: VecDeque<f64>,
    /// EW moments of `(x, e²)`, for Breusch and Pagan.
    het: crate::EwCov,
    /// EW moments of `(p, p², p³, e)`, `p` about `origin`, for RESET.
    reset: crate::EwCov,
    origin: Option<f64>,
    /// Under a horizon, the sandwiches of Breusch and Pagan's and RESET's
    /// regressions at Newey and West's lags (module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    het_hac: Option<crate::Sandwich>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reset_hac: Option<crate::Sandwich>,
}

impl Specification {
    /// Empty tests over `k` features, skipping `skip` lags and testing
    /// `lags`, Breusch and Pagan's and RESET's at `nw` Newey-West lags
    /// (their plain forms at 0).
    pub fn new(k: usize, skip: usize, lags: usize, nw: usize) -> Self {
        Self {
            skip,
            lags,
            w: 0.0,
            w2: 0.0,
            e1: 0.0,
            e2: 0.0,
            pairs: vec![[0.0; 4]; skip + lags],
            ring: VecDeque::with_capacity(skip + lags),
            het: crate::EwCov::new(k + 1).without_runs(),
            reset: crate::EwCov::new(4).without_runs(),
            origin: None,
            het_hac: (nw > 0).then(|| crate::Sandwich::new(k, true, nw)),
            reset_hac: (nw > 0).then(|| crate::Sandwich::new(3, true, nw)),
        }
    }

    /// Whether a restored set is shaped for `k` features and these lags.
    pub fn has_shape(&self, k: usize, skip: usize, lags: usize, nw: usize) -> bool {
        let hac = |h: &Option<crate::Sandwich>, d: usize| match h {
            None => nw == 0,
            Some(h) => nw > 0 && h.has_shape(d, true, nw),
        };
        hac(&self.het_hac, k)
            && hac(&self.reset_hac, 3)
            && self.skip == skip
            && self.lags == lags
            && self.pairs.len() == skip + lags
            && self.ring.len() <= skip + lags
            && self.het.has_shape(k + 1)
            && self.reset.has_shape(4)
            && self.het.q_sum().is_some()
            && self.reset.q_sum().is_some()
            && [self.w, self.w2, self.e1, self.e2]
                .iter()
                .chain(self.pairs.iter().flatten())
                .chain(&self.ring)
                .all(|v| v.is_finite())
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
        self.w2 *= lam * lam;
        self.e1 *= lam;
        self.e2 *= lam;
        self.pairs.iter_mut().flatten().for_each(|v| *v *= lam);
        self.het.decay(lam);
        self.reset.decay(lam);
        for h in [&mut self.het_hac, &mut self.reset_hac]
            .into_iter()
            .flatten()
        {
            h.age(lam);
        }
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring.clear();
        for h in [&mut self.het_hac, &mut self.reset_hac]
            .into_iter()
            .flatten()
        {
            h.clear_lags();
        }
    }

    /// One scored row: its features, prediction and out-of-sample residual,
    /// after the step's decay `lam`, at weight `w`. A residual, prediction
    /// or feature that is not finite, or a weight of 0, only ages
    /// (CLAUDE.md hard rule 9).
    pub fn update(&mut self, x: &[f64], p: f64, e: f64, lam: f64, w: f64) {
        if !(e.is_finite() && p.is_finite() && w > 0.0 && x.iter().all(|v| v.is_finite())) {
            return self.age(lam);
        }
        self.w = lam * self.w + w;
        self.w2 = lam * lam * self.w2 + w * w;
        self.e1 = lam * self.e1 + w * e;
        self.e2 = lam * self.e2 + w * e * e;
        for (j, pair) in self.pairs.iter_mut().enumerate() {
            pair.iter_mut().for_each(|v| *v *= lam);
            if let Some(&back) = self.ring.get(j) {
                pair[0] += w * e * back;
                pair[1] += w * e;
                pair[2] += w * back;
                pair[3] += w;
            }
        }
        if self.skip + self.lags > 0 {
            self.ring.push_front(e);
            self.ring.truncate(self.skip + self.lags);
        }
        let c = *self.origin.get_or_insert(p);
        let q = p - c;
        // The residuals under each null, from the sums before the row.
        if let Some(h) = self.het_hac.as_mut() {
            let k = self.het.k() - 1;
            let u = if self.het.n_eff() > 0.0 {
                e * e - self.het.mean(k)
            } else {
                0.0
            };
            h.update(x, u, lam, w);
        }
        if let Some(h) = self.reset_hac.as_mut() {
            let r = if self.reset.n_eff() > 0.0 {
                let var_q = self.reset.var(0);
                let b = if var_q > 0.0 {
                    self.reset.cov(0, 3) / var_q
                } else {
                    0.0
                };
                e - self.reset.mean(3) - b * (q - self.reset.mean(0))
            } else {
                0.0
            };
            h.update(&[q, q * q, q * q * q], r, lam, w);
        }
        let mut row = Vec::with_capacity(x.len() + 1);
        row.extend_from_slice(x);
        row.push(e * e);
        self.het.update(&row, lam, w);
        self.reset.update(&[q, q * q, q * q * q, e], lam, w);
    }

    fn n_kish(&self) -> Option<f64> {
        (self.w2 > 0.0).then(|| self.w * self.w / self.w2)
    }

    /// Ljung and Box's `Q` over the tested lags, or Box and Pierce's where
    /// the sums `forget` (a finite memory; the module docs): `None` until
    /// each lag has a pair, the residuals a spread, and Kish's size passes
    /// the largest lag.
    pub fn ljung_box(&self, forget: bool) -> Option<f64> {
        let n = self.n_kish()?;
        if self.lags == 0 || n <= (self.skip + self.lags) as f64 {
            return None;
        }
        let mean = self.e1 / self.w;
        let den = self.e2 - self.w * mean * mean;
        if den.is_nan() || den <= 0.0 {
            return None;
        }
        let mut rho = Vec::with_capacity(self.pairs.len());
        for [s, a, b, c] in &self.pairs {
            if *c <= 0.0 {
                return None;
            }
            rho.push((s - mean * (a + b) + c * mean * mean) / den);
        }
        let (skip, lags) = (self.skip, self.lags);
        let r: Vec<f64> = (0..lags)
            .map(|j| {
                let l = skip + j + 1;
                if forget {
                    rho[l - 1]
                } else {
                    rho[l - 1] * ((n + 2.0) / (n - l as f64)).sqrt()
                }
            })
            .collect();
        if skip == 0 {
            return Some(n * r.iter().map(|v| v * v).sum::<f64>());
        }
        // Bartlett's covariance of the autocorrelations past an MA(skip).
        let at = |j: isize| -> f64 {
            match j.unsigned_abs() {
                0 => 1.0,
                a if a <= skip => rho[a - 1],
                _ => 0.0,
            }
        };
        let s = skip as isize;
        let gamma = |v: usize| -> f64 { (-s..=s).map(|j| at(j) * at(j + v as isize)).sum() };
        let c: Vec<f64> = (0..lags)
            .flat_map(|k| (0..lags).map(move |l| (k, l)))
            .map(|(k, l)| gamma(k.abs_diff(l)))
            .collect();
        let (x, attempts) = crate::solve_spd(&c, &r, lags, 1)?;
        (attempts == 0).then(|| n * x.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>())
    }

    /// The slopes of column `y` of `m` regressed on the columns `xs`, from
    /// the centred moments; `None` where they do not factorize.
    fn slopes(m: &crate::EwCov, xs: &[usize], y: usize) -> Option<Vec<f64>> {
        let p = xs.len();
        let a: Vec<f64> = xs
            .iter()
            .flat_map(|&i| xs.iter().map(move |&j| (i, j)))
            .map(|(i, j)| m.cov(i, j))
            .collect();
        let b: Vec<f64> = xs.iter().map(|&i| m.cov(i, y)).collect();
        let (beta, attempts) = crate::solve_spd(&a, &b, p, 1)?;
        (attempts == 0).then_some(beta)
    }

    /// The `R²` of column `y` of `m` regressed on the columns `xs`, from the
    /// centred moments; `None` where the regressors' moments do not
    /// factorize or `y` has no spread.
    fn r2(m: &crate::EwCov, xs: &[usize], y: usize) -> Option<f64> {
        let var_y = m.var(y);
        if var_y.is_nan() || var_y <= 0.0 {
            return None;
        }
        if xs.is_empty() {
            return Some(0.0);
        }
        let p = xs.len();
        let a: Vec<f64> = xs
            .iter()
            .flat_map(|&i| xs.iter().map(move |&j| (i, j)))
            .map(|(i, j)| m.cov(i, j))
            .collect();
        let b: Vec<f64> = xs.iter().map(|&i| m.cov(i, y)).collect();
        let (beta, attempts) = crate::solve_spd(&a, &b, p, 1)?;
        if attempts > 0 {
            return None;
        }
        let fit: f64 = beta.iter().zip(&b).map(|(x, y)| x * y).sum();
        Some((fit / var_y).clamp(0.0, 1.0))
    }

    /// Breusch and Pagan's `n R²` over the features `idx`: `None` with no
    /// feature, until Kish's size passes their count, and while their
    /// moments do not factorize.
    pub fn breusch_pagan(&self, idx: &[usize]) -> Option<f64> {
        let n = self.het.n_kish()?;
        let k = self.het.k() - 1;
        if idx.is_empty() || n <= (idx.len() + 1) as f64 {
            return None;
        }
        if let Some(h) = &self.het_hac {
            // Wald's test of the slopes with Newey and West's variance
            // (module docs): the slopes sit after the intercept.
            let gamma = Self::slopes(&self.het, idx, k)?;
            let pick: Vec<usize> = (1..=idx.len()).collect();
            return h.wald(idx, true, &pick, &gamma);
        }
        Some(n * Self::r2(&self.het, idx, k)?)
    }

    /// RESET's Lagrange multiplier for `p²` and `p³` beside `p`: `None`
    /// until Kish's size passes 4, and while the powers' moments do not
    /// factorize -- a prediction with no spread.
    pub fn reset(&self) -> Option<f64> {
        let n = self.reset.n_kish()?;
        if n <= 4.0 {
            return None;
        }
        if let Some(h) = &self.reset_hac {
            // Wald's test of `p²`'s and `p³`'s coefficients with Newey and
            // West's variance (module docs), after the intercept and `p`'s.
            let beta = Self::slopes(&self.reset, &[0, 1, 2], 3)?;
            return h.wald(&[0, 1, 2], true, &[2, 3], &beta[1..]);
        }
        let full = Self::r2(&self.reset, &[0, 1, 2], 3)?;
        let base = Self::r2(&self.reset, &[0], 3)?;
        (base < 1.0).then(|| n * (full - base).max(0.0) / (1.0 - base))
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

    /// Ljung and Box's statistic by its definition, from every scored row
    /// at its present weight, with skipped lags, a decay, uneven weights
    /// and zeros, and residuals that carry a lag.
    #[test]
    fn ljung_box_is_its_definition() {
        let (skip, lags, lam) = (2usize, 3usize, 0.99);
        let mut s = Specification::new(1, skip, lags, 0);
        let mut st = 7u64;
        let mut rows: Vec<(f64, f64)> = Vec::new(); // (e, present weight), scored only
        let mut prev = 0.0;
        for i in 0..400 {
            let e = 0.8 * prev + lcg(&mut st) + 0.2;
            prev = e;
            let w = if i % 13 == 2 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            s.update(&[lcg(&mut st)], 1.0 + lcg(&mut st), e, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            if w > 0.0 {
                rows.push((e, w));
            }
            if i < 30 || i % 37 != 0 {
                continue;
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let sw2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
            let n = sw * sw / sw2;
            let mean = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
            let den: f64 = rows.iter().map(|r| r.1 * (r.0 - mean).powi(2)).sum();
            let rho = |l: usize| {
                (l..rows.len())
                    .map(|t| rows[t].1 * (rows[t].0 - mean) * (rows[t - l].0 - mean))
                    .sum::<f64>()
                    / den
            };
            // Bartlett's covariance past an MA(skip), inverted by faer's
            // LU.
            let at = |j: isize| -> f64 {
                let a = j.unsigned_abs();
                if a == 0 {
                    1.0
                } else if a <= skip {
                    rho(a)
                } else {
                    0.0
                }
            };
            let sk = skip as isize;
            let gamma = |v: usize| -> f64 { (-sk..=sk).map(|j| at(j) * at(j + v as isize)).sum() };
            let c: Vec<f64> = (0..lags)
                .flat_map(|k| (0..lags).map(move |l| gamma(k.abs_diff(l))))
                .collect();
            // Ljung and Box's factor run once; Box and Pierce's plain
            // autocorrelations under a memory that forgets.
            for forget in [false, true] {
                let r: Vec<f64> = (skip + 1..=skip + lags)
                    .map(|l| {
                        let f = if forget {
                            1.0
                        } else {
                            (n + 2.0) / (n - l as f64)
                        };
                        rho(l) * f.sqrt()
                    })
                    .collect();
                let x = crate::oracle::solve(&c, &r);
                let want = n * x.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>();
                let got = s.ljung_box(forget).unwrap();
                assert!(
                    (got - want).abs() < 1e-8 * want.max(1.0),
                    "row {i}, forget {forget}: {got} vs {want}"
                );
            }
        }
        assert!(s.ljung_box(false).unwrap() > s.ljung_box(true).unwrap());
        assert!(s.has_shape(1, skip, lags, 0));
    }

    /// With no horizon and a memory that forgets, the statistic is Box and
    /// Pierce's `n Σ ρ_l²` at Kish's size, by its definition (review round
    /// 6, B-3).
    #[test]
    fn box_pierce_under_a_memory() {
        let (lags, lam) = (4usize, 0.97);
        let mut s = Specification::new(1, 0, lags, 0);
        let mut st = 5u64;
        let mut rows: Vec<(f64, f64)> = Vec::new();
        for _ in 0..300 {
            let e = lcg(&mut st) + 0.1;
            let w = 0.5 + lcg(&mut st) + 0.5;
            s.update(&[lcg(&mut st)], 1.0, e, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            rows.push((e, w));
        }
        let sw: f64 = rows.iter().map(|r| r.1).sum();
        let n = sw * sw / rows.iter().map(|r| r.1 * r.1).sum::<f64>();
        let mean = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
        let den: f64 = rows.iter().map(|r| r.1 * (r.0 - mean).powi(2)).sum();
        let want = n
            * (1..=lags)
                .map(|l| {
                    let rho = (l..rows.len())
                        .map(|t| rows[t].1 * (rows[t].0 - mean) * (rows[t - l].0 - mean))
                        .sum::<f64>()
                        / den;
                    rho * rho
                })
                .sum::<f64>();
        let got = s.ljung_box(true).unwrap();
        assert!((got - want).abs() < 1e-10 * want, "{got} vs {want}");
    }

    /// Breusch and Pagan's and RESET's statistics by their definitions:
    /// least squares by faer's LU (`crate::oracle`) on every row, centred
    /// by hand, at Kish's size.
    #[test]
    fn breusch_pagan_and_reset_are_their_definitions() {
        let mut s = Specification::new(2, 0, 1, 0);
        let mut st = 11u64;
        let mut rows: Vec<([f64; 2], f64, f64)> = Vec::new(); // (x, p, e)
        for _ in 0..300 {
            let x = [lcg(&mut st) * 2.0, lcg(&mut st)];
            let p = 3.0 + x[0] - x[1];
            let e = (0.5 + x[0].abs()) * lcg(&mut st) + 0.3 * (p - 3.0).powi(2);
            s.update(&x, p, e, 1.0, 1.0);
            rows.push((x, p, e));
        }
        let n = rows.len() as f64;
        // R² of y on centred regressors.
        type Row = ([f64; 2], f64, f64);
        let r2 = |cols: &dyn Fn(&Row) -> Vec<f64>, y: &dyn Fn(&Row) -> f64| {
            let k = cols(&rows[0]).len();
            let mx: Vec<f64> = (0..k)
                .map(|j| rows.iter().map(|r| cols(r)[j]).sum::<f64>() / n)
                .collect();
            let my = rows.iter().map(y).sum::<f64>() / n;
            let mut a = vec![0.0; k * k];
            let mut b = vec![0.0; k];
            let mut vy = 0.0;
            for r in &rows {
                let c = cols(r);
                let dy = y(r) - my;
                vy += dy * dy;
                for i in 0..k {
                    b[i] += (c[i] - mx[i]) * dy;
                    for j in 0..k {
                        a[i * k + j] += (c[i] - mx[i]) * (c[j] - mx[j]);
                    }
                }
            }
            let beta = crate::oracle::solve(&a, &b);
            beta.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() / vy
        };
        let bp = n * r2(&|r| r.0.to_vec(), &|r| r.2 * r.2);
        let got = s.breusch_pagan(&[0, 1]).unwrap();
        assert!((got - bp).abs() < 1e-8 * bp, "{got} vs {bp}");
        let q = |r: &([f64; 2], f64, f64)| r.1 - rows[0].1;
        let full = r2(&|r| vec![q(r), q(r).powi(2), q(r).powi(3)], &|r| r.2);
        let base = r2(&|r| vec![q(r)], &|r| r.2);
        let want = n * (full - base) / (1.0 - base);
        let got = s.reset().unwrap();
        assert!((got - want).abs() < 1e-7 * want, "{got} vs {want}");
        assert!(got > 20.0, "a curvature the fit missed");
    }

    /// A zero weight, the first included, and a residual that is not a
    /// number only age; a cleared ring pairs nothing across the break.
    #[test]
    fn zero_weights_missing_values_and_breaks() {
        let mut s = Specification::new(1, 0, 2, 0);
        s.update(&[1.0], 1.0, 0.5, 0.9, 0.0);
        s.update(&[1.0], 1.0, f64::NAN, 0.9, 1.0);
        assert_eq!((s.w, s.w2, s.e1, s.e2), (0.0, 0.0, 0.0, 0.0));
        assert!(s.ring.is_empty() && s.origin.is_none() && s.het.n_eff() == 0.0);
        s.update(&[1.0], 1.0, 0.5, 0.9, 1.0);
        s.clear_lags();
        s.update(&[2.0], 1.5, -0.5, 0.9, 1.0);
        assert_eq!(s.pairs[0][3], 0.0, "no pair across a break");
        assert!(s.ljung_box(true).is_none() && s.breusch_pagan(&[0]).is_none());
    }
}
