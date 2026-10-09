//! `Sandwich`: heteroskedasticity- and autocorrelation-robust covariances of
//! an exponentially weighted least-squares fit, from its out-of-sample
//! residuals (docs/PLAN.md task 221 (c)).
//!
//! The fit is `β = B⁻¹ Σ ω z y`, with `z = (1, x − c)` -- the features about
//! an origin `c`, the first row's, which keeps the sums well conditioned
//! far from zero -- and `ω` each row's weight times the decay since it.
//! Its error is `B⁻¹ Σ ω z ε`, so its covariance is the sandwich
//!
//! ```text
//! B   = Σ ω z z'                                       the bread
//! M   = Σ_t Σ_s ω_t ω_s E[e_t e_s] z_t z_s'            the meat
//! Cov = B⁻¹ M B⁻¹
//! ```
//!
//! White's (1980) HC0 keeps the meat's diagonal, `Σ ω² e² z z'`; Newey and
//! West (1987) add each lag `l` up to `L` at Bartlett's weight `1 − l/(L+1)`:
//!
//! ```text
//! Γ_l = Σ_t ω_t ω_{t−l} e_t e_{t−l} z_t z_{t−l}'
//! M   = Γ_0 + Σ_{l=1..L} (1 − l/(L+1)) (Γ_l + Γ_l')
//! ```
//!
//! `t − l` is the `l`-th folded row before `t`. Each sum is kept as it
//! stands at the current row: on each step `B` decays by `λ`, `Γ_l` by
//! `λ²`, and each of the last `L` scores `u = ω e z` by `λ`, so the lag
//! products are those of the two rows' present weights. `e` is the row's
//! out-of-sample residual, the one the model had not learned, where the
//! classical sandwich has the in-sample residual of the final fit: larger
//! by its estimation error, `sqrt(1 + h)` on a row of leverage `h`, which
//! errs large while the fit is young and fades as it settles.
//!
//! The covariance is mapped to the features' own units: the slopes are the
//! fit's, and the intercept `β_0 − c'β_x`, whose variance is
//! `v_00 − 2 c'v_0x + c'V_xx c`. With no decay and unit weights it is
//! `statsmodels`' `OLS.fit(cov_type="HC0")` and `cov_type="HAC"` with
//! `maxlags = L` on the same rows and residuals.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// The sandwich of one slot's fit. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sandwich {
    /// Features, and whether `z` leads with the intercept's 1.
    k: usize,
    intercept: bool,
    /// Newey and West's lags, `0` for HC0 alone.
    lags: usize,
    /// The origin the features are taken about: the first folded row's,
    /// empty before it (and always without an intercept, which cannot
    /// absorb a shift).
    origin: Vec<f64>,
    /// `B`, `Γ_0` and each `Γ_l`, row-major over `z`.
    bread: Vec<f64>,
    meat: Vec<f64>,
    gammas: Vec<Vec<f64>>,
    /// The last `lags` scores `u = ω e z`, newest first, at their present
    /// weights.
    ring: VecDeque<Vec<f64>>,
}

impl Sandwich {
    /// An empty sandwich over `k` features, with or without an intercept,
    /// keeping `lags` lags.
    pub fn new(k: usize, intercept: bool, lags: usize) -> Self {
        let d = k + usize::from(intercept);
        Self {
            k,
            intercept,
            lags,
            origin: Vec::new(),
            bread: vec![0.0; d * d],
            meat: vec![0.0; d * d],
            gammas: vec![vec![0.0; d * d]; lags],
            ring: VecDeque::with_capacity(lags),
        }
    }

    fn dim(&self) -> usize {
        self.k + usize::from(self.intercept)
    }

    /// Whether a restored sandwich is shaped as this configuration's: its
    /// sums sized for `z`, its lags and ring within them, every number
    /// finite.
    pub fn has_shape(&self, k: usize, intercept: bool, lags: usize) -> bool {
        let d = k + usize::from(intercept);
        self.k == k
            && self.intercept == intercept
            && self.lags == lags
            && (self.origin.is_empty() || self.origin.len() == k)
            && self.bread.len() == d * d
            && self.meat.len() == d * d
            && self.gammas.len() == lags
            && self.gammas.iter().all(|g| g.len() == d * d)
            && self.ring.len() <= lags
            && self.ring.iter().all(|u| u.len() == d)
            && self
                .bread
                .iter()
                .chain(&self.meat)
                .chain(self.gammas.iter().flatten())
                .chain(self.ring.iter().flatten())
                .chain(&self.origin)
                .all(|v| v.is_finite())
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// folded: the sums keep their rows at their present weights.
    pub fn age(&mut self, lam: f64) {
        let lam2 = lam * lam;
        self.bread.iter_mut().for_each(|v| *v *= lam);
        self.meat.iter_mut().for_each(|v| *v *= lam2);
        self.gammas.iter_mut().flatten().for_each(|v| *v *= lam2);
        self.ring.iter_mut().flatten().for_each(|v| *v *= lam);
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none of them is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// One row: its features, its out-of-sample residual, the step's decay
    /// and its weight. A residual or a feature that is not finite, or a
    /// weight of 0, only ages (CLAUDE.md hard rule 9).
    pub fn update(&mut self, x: &[f64], e: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(e.is_finite() && w > 0.0 && x.len() == self.k && x.iter().all(|v| v.is_finite())) {
            return;
        }
        if self.intercept && self.origin.is_empty() {
            self.origin = x.to_vec();
        }
        let d = self.dim();
        let mut z = Vec::with_capacity(d);
        if self.intercept {
            z.push(1.0);
            z.extend(x.iter().zip(&self.origin).map(|(v, c)| v - c));
        } else {
            z.extend_from_slice(x);
        }
        let u: Vec<f64> = z.iter().map(|v| w * e * v).collect();
        for i in 0..d {
            for j in 0..d {
                self.bread[i * d + j] += w * z[i] * z[j];
                self.meat[i * d + j] += u[i] * u[j];
            }
        }
        for (g, back) in self.gammas.iter_mut().zip(&self.ring) {
            for i in 0..d {
                for j in 0..d {
                    g[i * d + j] += u[i] * back[j];
                }
            }
        }
        if self.lags > 0 {
            self.ring.push_front(u);
            self.ring.truncate(self.lags);
        }
    }

    /// The standard errors of the coefficients over the features `idx`
    /// (positions in `x`), in the layout `coef` reads -- the intercept
    /// first where there is one, then each of the `k` features, NaN for one
    /// outside `idx` -- with Newey and West's lags (`hac`) or HC0's
    /// diagonal alone. `None` until a row is folded, and while the bread
    /// over `idx` does not factorize: fewer rows than coefficients, or a
    /// feature with no spread.
    pub fn standard_errors(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
        let d = self.dim();
        let off = usize::from(self.intercept);
        let pos: Vec<usize> = (0..off).chain(idx.iter().map(|&i| i + off)).collect();
        let p = pos.len();
        if p == 0 {
            return None;
        }
        let pick = |m: &[f64]| -> Vec<f64> {
            pos.iter()
                .flat_map(|&i| pos.iter().map(move |&j| m[i * d + j]))
                .collect()
        };
        let bread = pick(&self.bread);
        let mut meat = pick(&self.meat);
        if hac {
            let lags = self.lags as f64;
            for (l, g) in self.gammas.iter().enumerate() {
                let b = 1.0 - (l + 1) as f64 / (lags + 1.0);
                let g = pick(g);
                for i in 0..p {
                    for j in 0..p {
                        meat[i * p + j] += b * (g[i * p + j] + g[j * p + i]);
                    }
                }
            }
        }
        let factor = crate::SpdFactor::of(&bread, p).filter(|f| f.attempts() == 0)?;
        // `B⁻¹ M B⁻¹`: the meat's columns solved, transposed (`M B⁻¹`, `M`
        // symmetric), and solved again. Column-major in and out.
        let col = |m: &[f64]| -> Vec<f64> { (0..p * p).map(|n| m[(n % p) * p + n / p]).collect() };
        let left = factor.solve(&col(&meat), p, p);
        let left_t: Vec<f64> = (0..p * p).map(|n| left[(n % p) * p + n / p]).collect();
        let v = factor.solve(&left_t, p, p);
        // Column-major `v`: entry (i, j) at `j·p + i`; symmetric.
        let at = |i: usize, j: usize| 0.5 * (v[j * p + i] + v[i * p + j]);
        let mut out = vec![f64::NAN; self.k + off];
        for (a, &i) in idx.iter().enumerate() {
            out[off + i] = at(a + off, a + off).max(0.0).sqrt();
        }
        if self.intercept {
            let c: Vec<f64> = idx
                .iter()
                .map(|&i| self.origin.get(i).copied().unwrap_or(0.0))
                .collect();
            let mut var = at(0, 0);
            for (a, ca) in c.iter().enumerate() {
                var -= 2.0 * ca * at(0, a + 1);
                for (b, cb) in c.iter().enumerate() {
                    var += ca * cb * at(a + 1, b + 1);
                }
            }
            out[0] = var.max(0.0).sqrt();
        }
        out.iter().any(|v| v.is_finite()).then_some(out)
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

    /// The sandwich by its definition, from every folded row at its present
    /// weight: `X'ΩX`, the meat's double sum over pairs at most `L` folded
    /// rows apart at Bartlett's weights, inverted by faer's LU
    /// (`crate::oracle`) in the features' own units -- `z = (1, x)`, no
    /// origin -- under a decay, uneven weights, zeros and residuals that
    /// carry a lag.
    #[test]
    fn the_sandwich_is_its_definition() {
        let lags = 3usize;
        let lam = 0.99;
        let mut s = Sandwich::new(2, true, lags);
        let mut st = 13u64;
        // (z, e, present weight); folded rows only, in order.
        let mut rows: Vec<([f64; 3], f64, f64)> = Vec::new();
        let mut prev = 0.0;
        for i in 0..300 {
            let x = [lcg(&mut st) + 50.0, 2.0 * lcg(&mut st) - 7.0];
            let shock = lcg(&mut st) * (1.0 + (x[0] - 50.0).abs());
            let e = 0.6 * prev + shock;
            prev = e;
            let w = if i % 11 == 5 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            s.update(&x, e, lam, w);
            rows.iter_mut().for_each(|r| r.2 *= lam);
            if w > 0.0 {
                rows.push(([1.0, x[0], x[1]], e, w));
            }
            if i < 20 || i % 41 != 0 {
                continue;
            }
            for hac in [false, true] {
                let n = rows.len();
                let (mut b, mut m) = (vec![0.0; 9], vec![0.0; 9]);
                for r in &rows {
                    for a in 0..3 {
                        for c in 0..3 {
                            b[a * 3 + c] += r.2 * r.0[a] * r.0[c];
                        }
                    }
                }
                let reach = if hac { lags } else { 0 };
                for t in 0..n {
                    for l in 0..=reach.min(t) {
                        let (rt, rs) = (&rows[t], &rows[t - l]);
                        let k = if l == 0 {
                            1.0
                        } else {
                            1.0 - l as f64 / (lags as f64 + 1.0)
                        };
                        for a in 0..3 {
                            for c in 0..3 {
                                let p = rt.2 * rt.1 * rt.0[a] * rs.2 * rs.1 * rs.0[c];
                                m[a * 3 + c] += k * p;
                                if l > 0 {
                                    m[c * 3 + a] += k * p;
                                }
                            }
                        }
                    }
                }
                let bi = crate::oracle::inverse(&b);
                let mut v = [0.0; 9];
                for a in 0..3 {
                    for c in 0..3 {
                        for p in 0..3 {
                            for q in 0..3 {
                                v[a * 3 + c] += bi[a * 3 + p] * m[p * 3 + q] * bi[q * 3 + c];
                            }
                        }
                    }
                }
                let got = s.standard_errors(&[0, 1], hac).unwrap();
                for a in 0..3 {
                    let want = v[a * 3 + a].sqrt();
                    assert!(
                        (got[a] - want).abs() < 1e-8 * want,
                        "row {i}, hac {hac}, coefficient {a}: {} vs {want}",
                        got[a]
                    );
                }
            }
        }
        assert!(s.has_shape(2, true, lags));
    }

    /// A feature set reads its own sub-block, NaN outside it; no intercept
    /// reads the raw features; a zero weight, the first included, only
    /// ages; a cleared ring adds no lag product across the break.
    #[test]
    fn sets_origins_zero_weights_and_breaks() {
        let mut s = Sandwich::new(2, false, 1);
        s.update(&[1.0, 2.0], 0.5, 0.9, 0.0);
        assert!(s.standard_errors(&[0, 1], true).is_none());
        assert_eq!(s, Sandwich::new(2, false, 1));
        let mut st = 3u64;
        for _ in 0..50 {
            let x = [lcg(&mut st), lcg(&mut st)];
            s.update(&x, lcg(&mut st), 1.0, 1.0);
        }
        let one = s.standard_errors(&[1], false).unwrap();
        assert!(one[0].is_nan() && one[1].is_finite());
        let before = s.gammas[0].clone();
        s.clear_lags();
        s.update(&[0.3, -0.2], 0.4, 1.0, 1.0);
        assert_eq!(s.gammas[0], before, "no lag product across a break");
        assert!(s.has_shape(2, false, 1));
    }
}
