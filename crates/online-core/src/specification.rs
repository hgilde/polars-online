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
    /// The row Breusch and Pagan's moments take, `(x, e²)`, kept so a row
    /// allocates nothing (task 270, F-3): not state.
    #[serde(skip)]
    row: crate::ewcov::Scratch,
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
            row: Default::default(),
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
        let row = &mut self.row.0;
        row.clear();
        row.extend_from_slice(x);
        row.push(e * e);
        self.het.update(row, lam, w);
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

    /// Ljung and Box's statistic from its autocorrelations `ρ_l` and Kish's
    /// size, as both sides read it once they have them: each tested lag
    /// scaled by `√((n + 2)/(n − l))` unless the memory `forget`s, the
    /// lags past `skip` solved against Bartlett's covariance by the
    /// oracle's LU, `n·x'r`.
    fn lb_of(rho: &[f64], n: f64, skip: usize, lags: usize, forget: bool) -> f64 {
        let r: Vec<f64> = (skip + 1..=skip + lags)
            .map(|l| {
                let f = if forget {
                    1.0
                } else {
                    (n + 2.0) / (n - l as f64)
                };
                rho[l - 1] * f.sqrt()
            })
            .collect();
        let at = |j: isize| -> f64 {
            let a = j.unsigned_abs();
            if a == 0 {
                1.0
            } else if a <= skip {
                rho[a - 1]
            } else {
                0.0
            }
        };
        let sk = skip as isize;
        let gamma = |v: usize| -> f64 { (-sk..=sk).map(|j| at(j) * at(j + v as isize)).sum() };
        let c: Vec<f64> = (0..lags)
            .flat_map(|k| (0..lags).map(move |l| gamma(k.abs_diff(l))))
            .collect();
        let x = crate::oracle::solve(&c, &r);
        n * x.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>()
    }

    /// The most the model's Ljung–Box statistic may be off the
    /// definition's (docs/PLAN.md task 274), to first order. Both read the
    /// statistic off each lag's autocorrelation and Kish's size, through
    /// [`lb_of`], whose sensitivity carries their errors
    /// ([`crate::oracle::propagate`]); its own rounding on each side --
    /// Bartlett's covariance, its terms' products and sums `γ_{2s+1}`, the
    /// scaling's three roundings, the model's Cholesky form of `x'r` and
    /// the oracle's LU form, the product by `n` -- is counted beside.
    ///
    /// - The model's: its sums `Σω`, `Σω²`, `Σωe`, `Σωe²` and each lag's four,
    ///   decayed by `λ` and added to on each row, carried along its run by
    ///   `sums` (each decay one rounding of the partial sum, two for `Σω²`,
    ///   each term its products and the addition one of the sum it makes);
    ///   `ρ_l = (Σωee' − m(Σωe + Σωe') + m²Σω)/(Σωe² − Wm²)`, `m = Σωe/W`,
    ///   their errors through its derivatives and its own subtractions'
    ///   rounding.
    /// - The definition's: each row's weight a product of `age` decays,
    ///   `(age + 4)u` of each term with its products, each addition `u` of
    ///   its partial sum, the mean's error first order in a lag's sum over
    ///   the rows that have that partner (`Σ ω(e − m)` over them is not 0).
    fn lb_error(
        rows: &[(f64, f64, usize)],
        now: usize,
        sums: &LbTrack,
        (skip, lags, forget): (usize, usize, bool),
        want: f64,
    ) -> f64 {
        use crate::oracle::{self, U, gamma};
        let nl = skip + lags;
        // The definition's pieces and their rounding.
        let (mut sw, mut esw, mut sw2, mut esw2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut se, mut ese) = (0.0f64, 0.0f64);
        for &(e, w, born) in rows {
            let age = (now - born) as f64;
            sw += w;
            esw += age * U * w + U * sw;
            sw2 += w * w;
            esw2 += (2.0 * age + 1.0) * U * w * w + U * sw2;
            se += w * e;
            ese += (age + 1.0) * U * (w * e).abs() + U * se.abs();
        }
        let mean = se / sw;
        let dmean = ese / sw + mean.abs() * (esw / sw + U);
        let (mut den, mut eden) = (0.0f64, 0.0f64);
        for &(e, w, born) in rows {
            let t = w * (e - mean).powi(2);
            den += t;
            eden += ((now - born) as f64 + 4.0) * U * t + U * den;
        }
        let n = sw * sw / sw2;
        let dn = n * (2.0 * esw / sw + esw2 / sw2 + 2.0 * U);
        let mut rho = vec![0.0; nl];
        let mut drho_t = vec![0.0; nl];
        for l in 1..=nl {
            let (mut num, mut enu) = (0.0f64, 0.0f64);
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for t in l..rows.len() {
                let (et, wt, born) = rows[t];
                let el = rows[t - l].0;
                let v = wt * (et - mean) * (el - mean);
                num += v;
                enu += ((now - born) as f64 + 4.0) * U * v.abs() + U * num.abs();
                s1 += wt * (et - mean);
                s2 += wt * (el - mean);
            }
            rho[l - 1] = num / den;
            drho_t[l - 1] =
                (enu + dmean * (s1.abs() + s2.abs())) / den + rho[l - 1].abs() * (eden / den + U);
        }
        // The model's pieces from its tracked sums.
        let t = sums;
        let mm = t.e1.0 / t.w.0;
        let dmm = (t.e1.1 + mm.abs() * t.w.1) / t.w.0 + U * mm.abs();
        let dden_m = t.e2.1
            + t.w.1 * mm * mm
            + 2.0 * (t.w.0 * mm).abs() * dmm
            + gamma(3) * (t.e2.0.abs() + (t.w.0 * mm * mm).abs());
        let den_m = t.e2.0 - t.w.0 * mm * mm;
        let mut drho_m = vec![0.0; nl];
        for (j, (p, ep)) in t.pairs.iter().enumerate() {
            let [sv, a, b, c] = *p;
            let num = sv - mm * (a + b) + c * mm * mm;
            let r = num / den_m;
            let dnum = ep[0]
                + mm.abs() * (ep[1] + ep[2])
                + (a + b).abs() * dmm
                + ep[3] * mm * mm
                + 2.0 * (c * mm).abs() * dmm
                + gamma(5) * (sv.abs() + (mm * (a + b)).abs() + (c * mm * mm).abs());
            drho_m[j] = dnum / den_m.abs() + r.abs() * (dden_m / den_m.abs() + U);
        }
        let drho: Vec<f64> = (0..nl).map(|j| drho_t[j] + drho_m[j]).collect();
        let dn_m = n * (2.0 * t.w.1 / t.w.0 + t.w2.1 / t.w2.0 + 2.0 * U);
        let mut inputs = rho.clone();
        inputs.push(n);
        let mut dinputs = drho;
        dinputs.push(dn + dn_m);
        let moved = oracle::propagate(
            |v| lb_of(&v[..nl], v[nl], skip, lags, forget),
            &inputs,
            &dinputs,
        );
        // Each side's own evaluation from its `ρ` and `n`.
        let r: Vec<f64> = (skip + 1..=skip + lags)
            .map(|l| {
                let f = if forget {
                    1.0
                } else {
                    (n + 2.0) / (n - l as f64)
                };
                rho[l - 1] * f.sqrt()
            })
            .collect();
        let at = |j: isize| -> f64 {
            let a = j.unsigned_abs();
            if a == 0 {
                1.0
            } else if a <= skip {
                rho[a - 1]
            } else {
                0.0
            }
        };
        let sk = skip as isize;
        let c: Vec<f64> = (0..lags)
            .flat_map(|k| (0..lags).map(move |l| (k, l)))
            .map(|(k, l)| {
                (-sk..=sk)
                    .map(|j| at(j) * at(j + k.abs_diff(l) as isize))
                    .sum()
            })
            .collect();
        let cabs: Vec<f64> = (0..lags)
            .flat_map(|k| (0..lags).map(move |l| (k, l)))
            .map(|(k, l)| {
                gamma(2 * skip + 2)
                    * (-sk..=sk)
                        .map(|j| (at(j) * at(j + k.abs_diff(l) as isize)).abs())
                        .sum::<f64>()
            })
            .collect();
        let x = oracle::solve(&c, &r);
        let dr: Vec<f64> = r.iter().map(|v| 4.0 * U * v.abs()).collect();
        let own = n
            * 2.0
            * (oracle::abs_form(&x, &cabs, &x)
                + 2.0 * x.iter().zip(&dr).map(|(a, b)| a.abs() * b).sum::<f64>())
            + n * (oracle::quad_error(&c, &r, &oracle::cholesky_backward(&c))
                + oracle::quad_form_error(&c, &r))
            + 2.0 * U * want.abs();
        moved + own
    }

    /// The model's Ljung–Box sums replayed with a running bound on each
    /// one's rounding: `(value, error)` for `Σω`, `Σω²`, `Σωe`, `Σωe²`, and
    /// each lag's four sums.
    #[derive(Default)]
    struct LbTrack {
        w: (f64, f64),
        w2: (f64, f64),
        e1: (f64, f64),
        e2: (f64, f64),
        pairs: Vec<([f64; 4], [f64; 4])>,
        ring: std::collections::VecDeque<f64>,
    }

    impl LbTrack {
        fn new(nl: usize) -> Self {
            Self {
                pairs: vec![([0.0; 4], [0.0; 4]); nl],
                ..Default::default()
            }
        }

        /// [`Specification::update`]'s sums: `S ← λS + term`, one rounding
        /// of `λS` (two under `λ²`), the term's products, and the sum's.
        fn update(&mut self, e: f64, lam: f64, w: f64) {
            use crate::oracle::U;
            let step = |s: &mut (f64, f64), l: f64, lr: f64, t: f64, tr: f64| {
                let (v, e0) = *s;
                let nv = l * v + t;
                *s = (
                    nv,
                    l * e0 + lr * U * (l * v).abs() + tr * U * t.abs() + U * nv.abs(),
                );
            };
            if w > 0.0 {
                step(&mut self.w, lam, 1.0, w, 0.0);
                step(&mut self.w2, lam * lam, 2.0, w * w, 1.0);
                step(&mut self.e1, lam, 1.0, w * e, 1.0);
                step(&mut self.e2, lam, 1.0, w * e * e, 2.0);
                let nl = self.pairs.len();
                for j in 0..nl {
                    let back = self.ring.get(j).copied();
                    let (p, ep) = &mut self.pairs[j];
                    let terms = match back {
                        Some(b) => [(w * e * b, 2.0), (w * e, 1.0), (w * b, 1.0), (w, 0.0)],
                        None => [(0.0, 0.0); 4],
                    };
                    for (q, (t, tr)) in terms.iter().enumerate() {
                        let v = lam * p[q];
                        let nv = v + t;
                        ep[q] = lam * ep[q] + U * v.abs() + tr * U * t.abs() + U * nv.abs();
                        p[q] = nv;
                    }
                }
                if nl > 0 {
                    self.ring.push_front(e);
                    self.ring.truncate(nl);
                }
            } else {
                for s in [&mut self.w, &mut self.e1, &mut self.e2] {
                    s.0 *= lam;
                    s.1 = lam * s.1 + U * s.0.abs();
                }
                self.w2.0 *= lam * lam;
                self.w2.1 = lam * lam * self.w2.1 + 2.0 * U * self.w2.0;
                for (p, ep) in self.pairs.iter_mut() {
                    for q in 0..4 {
                        p[q] *= lam;
                        ep[q] = lam * ep[q] + U * p[q].abs();
                    }
                }
            }
        }
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
        let mut born: Vec<usize> = Vec::new();
        let mut track = LbTrack::new(skip + lags);
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
            track.update(e, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            if w > 0.0 {
                rows.push((e, w));
                born.push(i);
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
                // Task 274: [`lb_error`], where it was `1e-8` relative to
                // `max(1, Q)`.
                // Measured on aarch64 (macOS): at most 0.0034 of the bound: the sums'
                // worst-case running bound over some 100 rows of memory against their realized
                // rounding.
                let tagged: Vec<(f64, f64, usize)> = rows
                    .iter()
                    .zip(&born)
                    .map(|(r, &b)| (r.0, r.1, b))
                    .collect();
                let bound = lb_error(&tagged, i, &track, (skip, lags, forget), want);
                crate::oracle::assert_within(
                    got,
                    want,
                    bound,
                    format_args!("forget {forget}, row {i}"),
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

    /// The most the model's `R²` of column `y` on columns `xs` may be off
    /// the definition's (docs/PLAN.md task 274), the definition two-pass
    /// over `cols` (each row's regressors) and `ys` at unit weights, the
    /// model's from its moments' run `tr` (column `y` of it the target), to
    /// first order. `R² = b'A⁻¹b/v` on both sides, in either's scale (the
    /// ratio has none), moves by `(2|x|ᵀΔb + |x|ᵀΔA|x| + R²Δv)/v` for each
    /// side's errors in `A`, `b` and `v` (`x = A⁻¹b`): the model's from its
    /// weighted Welford run ([`crate::ewcov::rounding::Rounding`]), the
    /// definition's from its sums (each term three roundings with its
    /// deviations, each addition `u` of its partial sum; a deviation from a
    /// computed mean is off it only to second order); and by each side's own
    /// form `b'A⁻¹b`, the model's Cholesky ([`crate::oracle::quad_error`])
    /// and the oracle's LU ([`crate::oracle::quad_form_error`]), and its
    /// quotient. Returns `(R², its bound)`.
    fn r2_error(
        cols: &[Vec<f64>],
        ys: &[f64],
        tr: &crate::ewcov::rounding::Rounding,
        xs: &[usize],
        y: usize,
    ) -> (f64, f64) {
        use crate::oracle::{self, U, abs_form};
        let n = cols.len() as f64;
        let k = cols[0].len();
        let mx: Vec<f64> = (0..k)
            .map(|j| cols.iter().map(|c| c[j]).sum::<f64>() / n)
            .collect();
        let my = ys.iter().sum::<f64>() / n;
        let run = |f: &dyn Fn(usize) -> f64, rounds: f64| -> (f64, f64) {
            let (mut sm, mut e) = (0.0f64, 0.0f64);
            for i in 0..cols.len() {
                let t = f(i);
                sm += t;
                e += rounds * U * t.abs() + U * sm.abs();
            }
            (sm, e)
        };
        let mut a = vec![0.0; k * k];
        let mut da = vec![0.0; k * k];
        let mut b = vec![0.0; k];
        let mut db = vec![0.0; k];
        for i in 0..k {
            for j in 0..k {
                let (v, e) = run(&|r| (cols[r][i] - mx[i]) * (cols[r][j] - mx[j]), 3.0);
                a[i * k + j] = v;
                da[i * k + j] = e;
            }
            let (v, e) = run(&|r| (cols[r][i] - mx[i]) * (ys[r] - my), 3.0);
            b[i] = v;
            db[i] = e;
        }
        let (vy, dvy) = run(&|r| (ys[r] - my) * (ys[r] - my), 3.0);
        // The model's errors in the definition's scale: its moments are
        // means, the definition's sums over the `n` rows.
        let kt = tr.m.len();
        let ec = |i: usize, j: usize| n * tr.ec[i * kt + j];
        for (p, &i) in xs.iter().enumerate() {
            for (q, &j) in xs.iter().enumerate() {
                da[p * k + q] += ec(i, j);
            }
            db[p] += ec(i, y);
        }
        let dvy = dvy + ec(y, y);
        let x = oracle::solve(&a, &b);
        let fit: f64 = x.iter().zip(&b).map(|(p, q)| p * q).sum();
        let r2 = fit / vy;
        let e = oracle::cholesky_backward(&a);
        let dfit = 2.0 * x.iter().zip(&db).map(|(p, q)| p.abs() * q).sum::<f64>()
            + abs_form(&x, &da, &x)
            + oracle::quad_error(&a, &b, &e)
            + oracle::quad_form_error(&a, &b);
        (r2, (dfit + r2 * dvy) / vy + 4.0 * U * r2)
    }

    /// Breusch and Pagan's and RESET's statistics by their definitions:
    /// least squares by faer's LU (`crate::oracle`) on every row, centred
    /// by hand, at Kish's size.
    #[test]
    fn breusch_pagan_and_reset_are_their_definitions() {
        let mut s = Specification::new(2, 0, 1, 0);
        let mut st = 11u64;
        let mut rows: Vec<([f64; 2], f64, f64)> = Vec::new(); // (x, p, e)
        // The model's two accumulators' rounding along its run (task 274):
        // `(x, e²)` for Breusch and Pagan, `(q, q², q³, e)` for RESET.
        let mut het = crate::ewcov::rounding::Rounding::new(3, 0.0);
        let mut res = crate::ewcov::rounding::Rounding::new(4, 0.0);
        let mut origin: Option<f64> = None;
        for _ in 0..300 {
            let x = [lcg(&mut st) * 2.0, lcg(&mut st)];
            let p = 3.0 + x[0] - x[1];
            let e = (0.5 + x[0].abs()) * lcg(&mut st) + 0.3 * (p - 3.0).powi(2);
            s.update(&x, p, e, 1.0, 1.0);
            het.update(&[x[0], x[1], e * e], 1.0, 1.0);
            let q = p - *origin.get_or_insert(p);
            res.update(&[q, q * q, q * q * q, e], 1.0, 1.0);
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
        // Task 274: each `R²` within [`r2_error`] (`n` exact on both sides,
        // unit weights), where they were `1e-8` and `1e-7` relative; RESET's
        // `n(full − base)/(1 − base)` carries both through its derivatives
        // and rounds four times.
        // Measured on aarch64 (macOS): at most 1.6e-4 (Breusch and Pagan, its `R²` 0.04, a
        // one-ulp error) and 5.1e-3 (RESET) of the bound: three hundred rows with no decay,
        // whose worst-case running bound grows with their count and whose realized rounding
        // does not.
        let xs: Vec<Vec<f64>> = rows.iter().map(|r| r.0.to_vec()).collect();
        let e2: Vec<f64> = rows.iter().map(|r| r.2 * r.2).collect();
        let (_, dbp) = r2_error(&xs, &e2, &het, &[0, 1], 2);
        crate::oracle::assert_within(
            got,
            bp,
            n * dbp + 2.0 * crate::oracle::U * bp,
            "breusch pagan",
        );
        let q = |r: &([f64; 2], f64, f64)| r.1 - rows[0].1;
        let full = r2(&|r| vec![q(r), q(r).powi(2), q(r).powi(3)], &|r| r.2);
        let base = r2(&|r| vec![q(r)], &|r| r.2);
        let want = n * (full - base) / (1.0 - base);
        let got = s.reset().unwrap();
        let es: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let qs3: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| vec![q(r), q(r).powi(2), q(r).powi(3)])
            .collect();
        let qs1: Vec<Vec<f64>> = rows.iter().map(|r| vec![q(r)]).collect();
        let (_, dfull) = r2_error(&qs3, &es, &res, &[0, 1, 2], 3);
        let (_, dbase) = r2_error(&qs1, &es, &res, &[0], 3);
        let dreset = n * (dfull + dbase) / (1.0 - base)
            + n * (full - base).abs() * dbase / (1.0 - base).powi(2)
            + 4.0 * crate::oracle::U * want.abs();
        crate::oracle::assert_within(got, want, dreset, "reset");
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

    /// The weighted least-squares slopes of `y` on `cols` over `rows` at
    /// their present weights `w`, centred by hand and solved by faer's LU.
    fn wls_slopes(cols: &[Vec<f64>], y: &[f64], w: &[f64]) -> Vec<f64> {
        let k = cols[0].len();
        let sw: f64 = w.iter().sum();
        let mx: Vec<f64> = (0..k)
            .map(|j| cols.iter().zip(w).map(|(c, w)| w * c[j]).sum::<f64>() / sw)
            .collect();
        let my = y.iter().zip(w).map(|(y, w)| w * y).sum::<f64>() / sw;
        let mut a = vec![0.0; k * k];
        let mut b = vec![0.0; k];
        for ((c, y), w) in cols.iter().zip(y).zip(w) {
            for i in 0..k {
                b[i] += w * (c[i] - mx[i]) * (y - my);
                for j in 0..k {
                    a[i * k + j] += w * (c[i] - mx[i]) * (c[j] - mx[j]);
                }
            }
        }
        crate::oracle::solve(&a, &b)
    }

    /// Weighted least squares' slopes of `y` on `cols` at weights `w`, as
    /// [`wls_slopes`] forms them, and a bound on how far they and a model's
    /// slopes from its moments' run `tr` (columns `xs`, target `y_col`) may
    /// part (docs/PLAN.md task 274), to first order: the definition's sums,
    /// two-pass, each weight within `age·u` of itself and each term three
    /// roundings more, each addition `u` of its partial sum; the model's
    /// weighted Welford run ([`crate::ewcov::rounding::Rounding`]); both
    /// through `|A⁻¹|(ΔA|β| + Δb)` in the means' scale, beside the model's
    /// Cholesky and the oracle's LU.
    fn slopes_error(
        cols: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        ages: &[f64],
        tr: &crate::ewcov::rounding::Rounding,
        (xs, y_col): (&[usize], usize),
    ) -> (Vec<f64>, Vec<f64>) {
        use crate::oracle::{self, U};
        let k = cols[0].len();
        let (mut sw, mut esw) = (0.0f64, 0.0f64);
        for (wi, ai) in w.iter().zip(ages) {
            sw += wi;
            esw += ai * U * wi + U * sw;
        }
        let mx: Vec<f64> = (0..k)
            .map(|j| cols.iter().zip(w).map(|(c, w)| w * c[j]).sum::<f64>() / sw)
            .collect();
        let my = y.iter().zip(w).map(|(y, w)| w * y).sum::<f64>() / sw;
        let run = |f: &dyn Fn(usize) -> f64| -> (f64, f64) {
            let (mut sm, mut e) = (0.0f64, 0.0f64);
            for (i, age) in ages.iter().enumerate() {
                let t = f(i);
                sm += t;
                e += (age + 3.0) * U * t.abs() + U * sm.abs();
            }
            (sm / sw, e / sw + (sm / sw).abs() * (esw / sw + U))
        };
        let mut a = vec![0.0; k * k];
        let mut da = vec![0.0; k * k];
        let mut b = vec![0.0; k];
        let mut db = vec![0.0; k];
        let kt = tr.m.len();
        for i in 0..k {
            for j in 0..k {
                let (v, e) = run(&|r| w[r] * (cols[r][i] - mx[i]) * (cols[r][j] - mx[j]));
                a[i * k + j] = v;
                da[i * k + j] = e + tr.ec[xs[i] * kt + xs[j]];
            }
            let (v, e) = run(&|r| w[r] * (cols[r][i] - mx[i]) * (y[r] - my));
            b[i] = v;
            db[i] = e + tr.ec[xs[i] * kt + y_col];
        }
        let beta = oracle::solve(&a, &b);
        let e = oracle::cholesky_backward(&a);
        let both: Vec<f64> = da.iter().zip(&e).map(|(x, y)| x + y).collect();
        let fe = oracle::forward_error(&a, &beta, &both, &db);
        let own = oracle::solve_error(&a, &b, &beta);
        let bound = (0..k)
            .map(|p| fe[p] + own[p] + 2.0 * U * beta[p].abs())
            .collect();
        (beta, bound)
    }

    /// The sandwich's meat moves with its residuals `u`: `|Δu_t|` on row `t`
    /// moves `Σ_t Σ_l k_l ω_tω_s u_t u_s z_t z_sᵀ` by `k_l ω_tω_s(|Δu_t||u_s| +
    /// |u_t||Δu_s|)|z_t||z_s|ᵀ` (and its transpose past lag 0), and `V =
    /// B⁻¹MB⁻¹` by `|B⁻¹|·ΔM·|B⁻¹|`: `z` about `level`, the slopes' block
    /// the same in any such frame.
    fn residual_error(
        rows: &[crate::sandwich::rounding::DefRow],
        du: &[f64],
        level: &[f64],
        lags: usize,
    ) -> Vec<f64> {
        let d = level.len();
        let z: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| {
                std::iter::once(1.0)
                    .chain(r.x.iter().zip(&level[1..]).map(|(x, c)| x - c))
                    .collect()
            })
            .collect();
        let mut dm = vec![0.0; d * d];
        let mut b = vec![0.0; d * d];
        for t in 0..rows.len() {
            for a in 0..d {
                for c in 0..d {
                    b[a * d + c] += rows[t].w * z[t][a] * z[t][c];
                }
            }
            for l in 0..=lags.min(t) {
                let s = t - l;
                let k = if l == 0 {
                    1.0
                } else {
                    1.0 - l as f64 / (lags as f64 + 1.0)
                };
                let mag =
                    k * rows[t].w * rows[s].w * (du[t] * rows[s].e.abs() + rows[t].e.abs() * du[s]);
                for a in 0..d {
                    for c in 0..d {
                        let v = mag * (z[t][a] * z[s][c]).abs();
                        dm[a * d + c] += v;
                        if l > 0 {
                            dm[c * d + a] += v;
                        }
                    }
                }
            }
        }
        let bi: Vec<f64> = crate::oracle::inverse(&b).iter().map(|v| v.abs()).collect();
        let r = crate::sandwich::rounding::abs_mul(&bi, &dm, d);
        crate::sandwich::rounding::abs_mul(&r, &bi, d)
    }

    /// Under a horizon, Breusch and Pagan's and RESET's statistics are
    /// Wald's tests with Newey and West's variance (module docs), by their
    /// definitions on every folded row: the slopes of `e²` on `x` and of `e`
    /// on `(q, q², q³)`, `q` the prediction about the first one scored, each
    /// sandwich's meat built from the residual under the null -- `e²` less
    /// its EW mean before the row, `e − ē − b (q − q̄)` from the EW fit on
    /// `q` before the row, and 0 on the first row, which has none before it
    /// -- under a decay, uneven weights, zeros, a row that is not a number,
    /// and residuals that carry a lag. The sandwich is summed about the
    /// regressors' weighted means and its slopes' block read there, where it
    /// is the same as in `z = (1, x)`'s own frame
    /// ([`crate::sandwich::rounding::covariance_bound`]).
    #[test]
    fn breusch_pagan_and_reset_under_a_horizon_are_their_definitions() {
        let (nw, lam) = (2usize, 0.99);
        let mut s = Specification::new(2, 0, 1, nw);
        let mut st = 19u64;
        // (x, q, e, present weight, BP's null residual, RESET's), folded
        // rows only, and the row each was folded on.
        type Row = ([f64; 2], f64, f64, f64, f64, f64);
        let mut rows: Vec<Row> = Vec::new();
        let mut born: Vec<usize> = Vec::new();
        // Task 274: the model's runs' rounding -- its two accumulators and
        // its two sandwiches -- and each null residual's own error on both
        // sides: the model's mean before the row off by its run's bound, the
        // definition's by its sums', and each residual's own subtractions.
        use crate::oracle::{self, U};
        use crate::sandwich::rounding::{DefRow, Track, covariance_bound};
        let mut het = crate::ewcov::rounding::Rounding::new(3, 0.0);
        let mut res = crate::ewcov::rounding::Rounding::new(4, 0.0);
        let (mut het_hac, mut res_hac) = (Track::new(2, nw), Track::new(3, nw));
        let (mut du, mut dr): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut first: Option<f64> = None;
        let mut prev = 0.0;
        for i in 0..260 {
            let x = [lcg(&mut st) * 2.0 + 5.0, lcg(&mut st)];
            let p = 40.0 + x[0] - 3.0 * x[1] + lcg(&mut st);
            let shock = (0.5 + (x[0] - 5.0).abs()) * lcg(&mut st);
            let e = 0.5 * prev + shock + (p - 40.0).powi(2);
            prev = e;
            let w = if i % 17 == 4 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            let e_in = if i % 29 == 11 { f64::NAN } else { e };
            s.update(&x, p, e_in, lam, w);
            rows.iter_mut().for_each(|r| r.3 *= lam);
            let folds = w > 0.0 && e_in.is_finite();
            if folds {
                let q = p - *first.get_or_insert(p);
                let (u, r) = if rows.is_empty() {
                    (0.0, 0.0)
                } else {
                    let sw: f64 = rows.iter().map(|r| r.3).sum();
                    let mean =
                        |f: &dyn Fn(&Row) -> f64| rows.iter().map(|r| r.3 * f(r)).sum::<f64>() / sw;
                    let (e2, mq, me) = (mean(&|r| r.2 * r.2), mean(&|r| r.1), mean(&|r| r.2));
                    let var_q = mean(&|r| (r.1 - mq).powi(2));
                    let cov = mean(&|r| (r.1 - mq) * (r.2 - me));
                    let b = if var_q > 0.0 { cov / var_q } else { 0.0 };
                    (e * e - e2, e - me - b * (q - mq))
                };
                // Each residual's errors: the definition's means' sums (each
                // weight `age·u` off, each term and partial sum `u`), the
                // model's means' runs, and the subtractions.
                let (dui, dri) = if rows.is_empty() {
                    (0.0, 0.0)
                } else {
                    let ages: Vec<f64> = born.iter().map(|&b| (i - b) as f64).collect();
                    let sw: f64 = rows.iter().map(|r| r.3).sum();
                    let rel = |f: &dyn Fn(&Row) -> f64| -> f64 {
                        let (mut sm, mut e) = (0.0f64, 0.0f64);
                        for (r, a) in rows.iter().zip(&ages) {
                            let t = r.3 * f(r);
                            sm += t;
                            e += (a + 2.0) * U * t.abs() + U * sm.abs();
                        }
                        e / sw + 2.0 * U * (sm / sw).abs()
                    };
                    let me = rows.iter().map(|r| r.3 * r.2).sum::<f64>() / sw;
                    let mq = rows.iter().map(|r| r.3 * r.1).sum::<f64>() / sw;
                    let var_q = res.c[0];
                    let b = if var_q > 0.0 { res.c[3] / var_q } else { 0.0 };
                    let db = (res.ec[3] + b.abs() * res.ec[0]) / var_q.max(f64::MIN_POSITIVE);
                    let d_e2 = rel(&|r| r.2 * r.2) + het.em[2];
                    let d_me = rel(&|r| r.2) + res.em[3];
                    let d_mq = rel(&|r| r.1) + res.em[0];
                    (
                        d_e2 + 2.0 * U * ((e * e).abs() + u.abs()),
                        d_me + b.abs() * d_mq
                            + 2.0 * (q - mq).abs() * db
                            + 4.0 * U * (e.abs() + me.abs() + (b * (q - mq)).abs()),
                    )
                };
                het_hac.update(&x, u, lam, w);
                res_hac.update(&[q, q * q, q * q * q], r, lam, w);
                het.update(&[x[0], x[1], e * e], lam, w);
                res.update(&[q, q * q, q * q * q, e], lam, w);
                rows.push((x, q, e, w, u, r));
                born.push(i);
                du.push(dui);
                dr.push(dri);
            } else {
                // The sandwiches only age.
                het_hac.update(&[0.0; 2], f64::NAN, lam, 0.0);
                res_hac.update(&[0.0; 3], f64::NAN, lam, 0.0);
                het.decay(lam);
                res.decay(lam);
            }
            if i < 30 || i % 37 != 0 {
                continue;
            }
            let w: Vec<f64> = rows.iter().map(|r| r.3).collect();
            let ages: Vec<f64> = born.iter().map(|&b| (i - b) as f64).collect();
            let xs: Vec<Vec<f64>> = rows.iter().map(|r| r.0.to_vec()).collect();
            let e2: Vec<f64> = rows.iter().map(|r| r.2 * r.2).collect();
            let u: Vec<f64> = rows.iter().map(|r| r.4).collect();
            let gamma = wls_slopes(&xs, &e2, &w);
            let got = s.breusch_pagan(&[0, 1]).unwrap();
            // Task 274: each Wald statistic within both sides' rounding,
            // where each was `1e-7` relative: the slopes' ([`slopes_error`])
            // through `2|y|ᵀΔd`, the slopes' block of the variance -- the same
            // in every frame of `z`'s origin, so bounded in the level's
            // ([`covariance_bound`]) with the residuals' own errors
            // ([`residual_error`]) -- through `|y|ᵀΔV|y|` (`y = V⁻¹d`), and the
            // model's Cholesky form and the oracle's LU form of `d'V⁻¹d`.
            // Measured on aarch64 (macOS): at most 3.7e-3 (Breusch and Pagan) and 8.6e-5
            // (RESET, whose regressors are a prediction's first three powers, nearly collinear)
            // of the bound.
            let wald_bound = |track: &Track,
                              defs: &[DefRow],
                              level: &[f64],
                              pick: &[usize],
                              d: &[f64],
                              dd: &[f64],
                              dres: &[f64]|
             -> (f64, f64) {
                let p = level.len();
                let b = covariance_bound(track, defs, i, level, (nw, nw));
                let extra = residual_error(defs, dres, level, nw);
                let sub = |m: &[f64]| -> Vec<f64> {
                    pick.iter()
                        .flat_map(|&a| pick.iter().map(move |&c| m[a * p + c]))
                        .collect()
                };
                let (vs, dvs, xs) = (sub(&b.v_level), sub(&b.dv_level), sub(&extra));
                let y = oracle::solve(&vs, d);
                let dv: Vec<f64> = dvs.iter().zip(&xs).map(|(a, c)| a + c).collect();
                let bound = oracle::abs_form(&y, &dv, &y)
                    + 2.0 * y.iter().zip(dd).map(|(a, c)| a.abs() * c).sum::<f64>()
                    + oracle::quad_error(&vs, d, &oracle::cholesky_backward(&vs))
                    + oracle::quad_form_error(&vs, d);
                (oracle::quad_form(&vs, d), bound)
            };
            let defs = |cols: &[Vec<f64>], resid: &[f64]| -> Vec<DefRow> {
                (0..rows.len())
                    .map(|t| DefRow {
                        x: cols[t].clone(),
                        e: resid[t],
                        w: w[t],
                        born: born[t],
                    })
                    .collect()
            };
            let (_, dg) = slopes_error(&xs, &e2, &w, &ages, &het, (&[0, 1], 2));
            let mean_of = |c: usize, cols: &[Vec<f64>]| -> f64 {
                cols.iter().zip(&w).map(|(v, w)| w * v[c]).sum::<f64>() / w.iter().sum::<f64>()
            };
            let level_x = [0.0, mean_of(0, &xs), mean_of(1, &xs)];
            let (bp, bound) = wald_bound(
                &het_hac,
                &defs(&xs, &u),
                &level_x,
                &[1, 2],
                &gamma,
                &dg,
                &du,
            );
            oracle::assert_within(got, bp, bound, format_args!("breusch pagan at row {i}"));
            let qs: Vec<Vec<f64>> = rows
                .iter()
                .map(|r| vec![r.1, r.1 * r.1, r.1.powi(3)])
                .collect();
            let es: Vec<f64> = rows.iter().map(|r| r.2).collect();
            let rs: Vec<f64> = rows.iter().map(|r| r.5).collect();
            let beta = wls_slopes(&qs, &es, &w);
            let got = s.reset().unwrap();
            let (_, db) = slopes_error(&qs, &es, &w, &ages, &res, (&[0, 1, 2], 3));
            let level_q = [0.0, mean_of(0, &qs), mean_of(1, &qs), mean_of(2, &qs)];
            let (want, bound) = wald_bound(
                &res_hac,
                &defs(&qs, &rs),
                &level_q,
                &[2, 3],
                &beta[1..],
                &db[1..],
                &dr,
            );
            oracle::assert_within(got, want, bound, format_args!("reset at row {i}"));
        }
        // The meat is built under the null, so the curvature it carries
        // widens `V`: RESET reads 2.65 here, Breusch and Pagan 32.
        assert!(
            s.breusch_pagan(&[0, 1]).unwrap() > 20.0,
            "a spread the fit missed"
        );
        assert!(s.has_shape(2, 0, 1, nw));
    }

    /// RESET reads the prediction about the first one scored (module docs),
    /// so its powers keep their spread at any level: predictions 10,000
    /// above the same ones give the same statistic, plain and under a
    /// horizon.
    #[test]
    fn reset_does_not_see_the_predictions_level() {
        for nw in [0, 2] {
            let (mut low, mut high) = (
                Specification::new(1, 0, 1, nw),
                Specification::new(1, 0, 1, nw),
            );
            let mut st = 31u64;
            for _ in 0..200 {
                let x = [lcg(&mut st)];
                let p = 1.0 + 2.0 * x[0];
                let e = 0.4 * (p - 1.0).powi(2) + 0.3 * lcg(&mut st);
                low.update(&x, p, e, 0.995, 1.0);
                high.update(&x, p + 1e4, e, 0.995, 1.0);
            }
            let (a, b) = (low.reset().unwrap(), high.reset().unwrap());
            assert!(a > 10.0 && (a - b).abs() < 1e-6 * a, "nw {nw}: {a} vs {b}");
        }
    }

    /// The shape a restored set must hold, one part broken at a time: the
    /// configuration it was made for, the Newey-West sandwiches there or
    /// not as `nw` says (none at `nw = 0`, not even one at no lags), each
    /// sum's size, the ring's depth, the moments'
    /// `Σω²`, and every number finite. A set with no lag skipped holds it
    /// with its ring full.
    #[test]
    fn each_part_of_the_shape_is_checked_alone() {
        let mut s = Specification::new(2, 1, 2, 3);
        let mut plain = Specification::new(2, 0, 2, 0);
        let mut st = 37u64;
        for _ in 0..20 {
            let x = [lcg(&mut st), lcg(&mut st)];
            let (p, e) = (lcg(&mut st), lcg(&mut st));
            s.update(&x, p, e, 0.95, 1.0);
            plain.update(&x, p, e, 0.95, 1.0);
        }
        assert_eq!((s.ring.len(), plain.ring.len()), (3, 2));
        assert!(s.has_shape(2, 1, 2, 3) && plain.has_shape(2, 0, 2, 0));
        assert!(!s.has_shape(2, 1, 2, 4), "a sandwich at other lags");
        assert!(!s.has_shape(2, 1, 2, 0) && !plain.has_shape(2, 0, 2, 3));
        let mut stray = plain.clone();
        stray.het_hac = Some(crate::Sandwich::new(2, true, 0));
        assert!(!stray.has_shape(2, 0, 2, 0), "a sandwich at no lags");
        type Edit<'a> = (&'a str, &'a dyn Fn(&mut Specification));
        let broken: [Edit; 14] = [
            ("het sandwich", &|s| s.het_hac = None),
            ("reset sandwich", &|s| s.reset_hac = None),
            ("het sandwich's shape", &|s| {
                s.het_hac = Some(crate::Sandwich::new(1, true, 3));
            }),
            ("skip", &|s| s.skip = 2),
            ("lags", &|s| s.lags = 1),
            ("pairs", &|s| {
                s.pairs.pop();
            }),
            ("ring", &|s| s.ring.push_back(0.0)),
            ("het", &|s| s.het = crate::EwCov::new(2)),
            ("reset", &|s| s.reset = crate::EwCov::new(3)),
            ("het's Σω²", &|s| {
                let (m, c, w) = (
                    s.het.means().to_vec(),
                    s.het.comoments().to_vec(),
                    s.het.n_eff(),
                );
                s.het.set_moments(&m, &c, w, None);
            }),
            ("reset's Σω²", &|s| {
                let (m, c, w) = (
                    s.reset.means().to_vec(),
                    s.reset.comoments().to_vec(),
                    s.reset.n_eff(),
                );
                s.reset.set_moments(&m, &c, w, None);
            }),
            ("w2 NaN", &|s| s.w2 = f64::NAN),
            ("pair inf", &|s| s.pairs[2][1] = f64::INFINITY),
            ("ring NaN", &|s| s.ring[1] = f64::NAN),
        ];
        for (what, edit) in broken {
            let mut t = s.clone();
            edit(&mut t);
            assert!(!t.has_shape(2, 1, 2, 3), "{what}");
        }
    }

    /// Each statistic is `None`, never a number read from nothing: Ljung
    /// and Box's with no lag tested, while Kish's size has not passed the
    /// largest lag though the rows have, with residuals of no spread, and
    /// once the memory has faded past what a double holds of `Σω²`;
    /// Breusch and Pagan's with no feature, at Kish's size of one more than
    /// the features, and with squared residuals of no spread; RESET's with
    /// a prediction on two values, whose powers have one direction between
    /// them, and with a residual that is the prediction's line, which
    /// leaves the powers nothing to explain.
    #[test]
    fn each_statistic_refuses_what_it_cannot_read() {
        let mut st = 41u64;
        let mut feed = |s: &mut Specification, rows: &[f64], e: &dyn Fn(f64, usize) -> f64| {
            for (i, &w) in rows.iter().enumerate() {
                let x = [lcg(&mut st), lcg(&mut st)];
                let p = lcg(&mut st);
                s.update(&x, p, e(p, i), 1.0, w);
            }
        };
        let noise: Vec<f64> = {
            let mut st = 43u64;
            (0..64).map(|_| lcg(&mut st)).collect()
        };
        let e = |_: f64, i: usize| noise[i];
        let mut none = Specification::new(2, 0, 0, 0);
        feed(&mut none, &[1.0; 30], &e);
        assert!(
            none.ljung_box(false).is_none() && none.ljung_box(true).is_none(),
            "no lag"
        );
        // Five rows past two lags, one carrying nearly all the weight.
        let mut heavy = Specification::new(2, 0, 2, 0);
        feed(&mut heavy, &[1.0, 1.0, 1.0, 1.0, 100.0], &e);
        assert!(heavy.n_kish().unwrap() < 2.0 && heavy.pairs[1][3] > 0.0);
        assert!(
            heavy.ljung_box(false).is_none() && heavy.ljung_box(true).is_none(),
            "Kish"
        );
        let mut flat = Specification::new(2, 0, 2, 0);
        feed(&mut flat, &[1.0; 30], &|_, _| 0.5);
        assert!(flat.ljung_box(true).is_none(), "no spread");
        assert!(flat.breusch_pagan(&[0, 1]).is_none(), "e² of no spread");
        let mut faded = Specification::new(2, 0, 2, 0);
        feed(&mut faded, &[1.0; 30], &e);
        assert!(faded.ljung_box(true).is_some());
        faded.age(1e-170);
        assert!(faded.w > 0.0 && faded.w2 == 0.0, "Σω² underflows first");
        assert!(
            faded.ljung_box(true).is_none() && faded.ljung_box(false).is_none(),
            "faded"
        );
        let mut bp = Specification::new(2, 0, 1, 0);
        feed(&mut bp, &[1.0; 3], &e);
        assert!(bp.breusch_pagan(&[]).is_none(), "no feature");
        assert!(
            bp.breusch_pagan(&[0, 1]).is_none(),
            "Kish's size 3, two features"
        );
        feed(&mut bp, &[1.0], &e);
        assert!(bp.breusch_pagan(&[0, 1]).is_some(), "Kish's size 4");
        let mut two = Specification::new(1, 0, 1, 0);
        for (i, &e) in noise.iter().enumerate().take(40) {
            let p = if i % 2 == 0 { 1.0 } else { 2.0 };
            two.update(&[lcg(&mut st)], p, e, 1.0, 1.0);
        }
        assert!(two.reset().is_none(), "a prediction on two values");
        let mut line = Specification::new(1, 0, 1, 0);
        for i in 0..40 {
            let p = (i % 7) as f64;
            line.update(&[lcg(&mut st)], p, 2.0 * p, 1.0, 1.0);
        }
        assert_eq!(Specification::r2(&line.reset, &[0], 3), Some(1.0));
        assert!(
            line.reset().is_none(),
            "a residual on the prediction's line"
        );
    }

    /// Task 252: under Newey and West's lags a break clears the rows the
    /// two sandwiches pair across it as well as the residuals' own ring,
    /// and nothing else: the state is the one whose three rings were
    /// cleared by hand, and not the one that saw no break.
    #[test]
    fn a_break_clears_the_sandwiches_lags_too() {
        let mut s = Specification::new(1, 0, 2, 3);
        for i in 0..12 {
            let x = f64::from(i % 4) - 1.5;
            s.update(&[x], 0.5 * x, 0.1 * x * x - 0.05, 0.95, 1.0);
        }
        assert!(s.het_hac.is_some() && s.reset_hac.is_some());
        let (mut by_hand, unbroken) = (s.clone(), s.clone());
        s.clear_lags();
        by_hand.ring.clear();
        by_hand.het_hac.as_mut().unwrap().clear_lags();
        by_hand.reset_hac.as_mut().unwrap().clear_lags();
        assert_eq!(s, by_hand);
        assert_ne!(s, unbroken, "the sandwiches held lags to clear");
    }
}
