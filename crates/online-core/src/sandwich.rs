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
    /// weight of 0, only ages (CLAUDE.md hard rule 9), and so does a row
    /// whose score `u = w e z` or bread term `w z z'` would put a sum past
    /// `f64`: kept, its `inf` would stay in the sums for good.
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
        // The bread and the meat are sums of positive semidefinite terms, so
        // each entry is bounded by its diagonal ones (Cauchy-Schwarz), and
        // each lag sum by the meat's diagonal at the two rows' present
        // weights: a finite diagonal after the row keeps every sum finite
        // (docs/PLAN.md task 270, review round 9 B-2).
        let fits = (0..d).all(|i| {
            (self.bread[i * d + i] + w * z[i] * z[i]).is_finite()
                && (self.meat[i * d + i] + u[i] * u[i]).is_finite()
        });
        if !fits {
            return;
        }
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

    /// The covariance of the coefficients over the features `idx`
    /// (positions in `x`), in the features' own units -- the intercept
    /// first where there is one, then each of `idx` -- with Newey and
    /// West's lags (`hac`) or HC0's diagonal alone, `p × p` row-major.
    /// `None` until a row is folded, and while the bread over `idx` does not
    /// factorize: fewer rows than coefficients, or a feature with no spread.
    pub fn covariance(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
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
        let mut out: Vec<f64> = (0..p * p)
            .map(|n| {
                let (i, j) = (n / p, n % p);
                0.5 * (v[j * p + i] + v[i * p + j])
            })
            .collect();
        if self.intercept {
            // The intercept in the features' own units is `β₀ − c'β_x`:
            // `T V T'` with `T = [[1, −c'], [0, I]]`.
            let c: Vec<f64> = std::iter::once(0.0)
                .chain(
                    idx.iter()
                        .map(|&i| self.origin.get(i).copied().unwrap_or(0.0)),
                )
                .collect();
            // Row 0 of `T V`: `v_0j − Σ_a c_a v_aj`.
            let row0: Vec<f64> = (0..p)
                .map(|j| out[j] - (1..p).map(|a| c[a] * out[a * p + j]).sum::<f64>())
                .collect();
            let v00 = row0[0] - (1..p).map(|b| c[b] * row0[b]).sum::<f64>();
            for j in 1..p {
                out[j] = row0[j];
                out[j * p] = row0[j];
            }
            out[0] = v00;
        }
        Some(out)
    }

    /// The standard errors of the coefficients over the features `idx`
    /// (positions in `x`), in the layout `coef` reads -- the intercept
    /// first where there is one, then each of the `k` features, NaN for one
    /// outside `idx` -- with Newey and West's lags (`hac`) or HC0's
    /// diagonal alone. `None` as [`Self::covariance`] is.
    pub fn standard_errors(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
        let v = self.covariance(idx, hac)?;
        let off = usize::from(self.intercept);
        let p = idx.len() + off;
        let mut out = vec![f64::NAN; self.k + off];
        // A rounding below 0 is 0, but a NaN variance stays NaN: `f64::max`
        // read it as 0.0, a standard error that says the coefficient is
        // exact (review round 9, B-2).
        for (a, &i) in idx.iter().enumerate() {
            out[off + i] = crate::solve::clamp_rounding(v[(a + off) * p + a + off]).sqrt();
        }
        if self.intercept {
            out[0] = crate::solve::clamp_rounding(v[0]).sqrt();
        }
        out.iter().any(|v| v.is_finite()).then_some(out)
    }

    /// Wald's statistic `d' V⁻¹ d` for the coefficients at `pick` --
    /// positions in [`Self::covariance`]'s layout over `idx` -- against the
    /// distance `d` of their estimates from the null, `V` the covariance
    /// with Newey and West's lags (`hac`). `None` where the covariance is
    /// not there, or its part over `pick` does not factorize.
    pub fn wald(&self, idx: &[usize], hac: bool, pick: &[usize], d: &[f64]) -> Option<f64> {
        let v = self.covariance(idx, hac)?;
        let p = idx.len() + usize::from(self.intercept);
        let q = pick.len();
        if q == 0 || d.len() != q || pick.iter().any(|&i| i >= p) {
            return None;
        }
        let v = &v;
        let sub: Vec<f64> = pick
            .iter()
            .flat_map(|&i| pick.iter().map(move |&j| v[i * p + j]))
            .collect();
        let (x, attempts) = crate::solve_spd(&sub, d, q, 1)?;
        let w: f64 = x.iter().zip(d).map(|(a, b)| a * b).sum();
        (attempts == 0 && w.is_finite()).then_some(w.max(0.0))
    }
}

#[cfg(test)]
pub(crate) mod rounding;

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
    /// (`crate::oracle`), about the level `(50, −7)` and mapped to the
    /// features' own units -- not the model's origin, its first row -- under
    /// a decay, uneven weights, zeros and residuals that carry a lag.
    ///
    /// Task 274: each standard error within a bound on both sides'
    /// rounding ([`rounding::covariance_bound`]), the root halving its
    /// variance's relative error, where it was `1e-8` relative. Summed about
    /// 0, the definition's bread held the level's square, and its own
    /// rounding, `u` of entries 2,500 times the spread, was all it could
    /// see: 3e-11 of the model's numbers, against the model's 1e-15.
    /// Measured on aarch64 (macOS): at most 0.012 of the bound.
    #[test]
    fn the_sandwich_is_its_definition() {
        use rounding::{DefRow, Track, covariance_bound};
        let lags = 3usize;
        let lam = 0.99;
        let mut s = Sandwich::new(2, true, lags);
        let mut track = Track::new(2, lags);
        let mut st = 13u64;
        // Folded rows only, in order.
        let mut rows: Vec<DefRow> = Vec::new();
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
            track.update(&x, e, lam, w);
            rows.iter_mut().for_each(|r| r.w *= lam);
            if w > 0.0 {
                rows.push(DefRow {
                    x: x.to_vec(),
                    e,
                    w,
                    born: i,
                });
            }
            if i < 20 || i % 41 != 0 {
                continue;
            }
            for hac in [false, true] {
                let reach = if hac { lags } else { 0 };
                let b = covariance_bound(&track, &rows, i, &[0.0, 50.0, -7.0], (reach, lags));
                let (v, dv) = (b.v, b.dv);
                let got = s.standard_errors(&[0, 1], hac).unwrap();
                for a in 0..3 {
                    let want = v[a * 3 + a].sqrt();
                    let bound =
                        want * (0.5 * dv[a * 3 + a] / v[a * 3 + a] + 2.0 * crate::oracle::U);
                    crate::oracle::assert_within(
                        got[a],
                        want,
                        bound,
                        format_args!("row {i}, hac {hac}, coefficient {a}"),
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

    /// A row whose score `u = w e z` would put a sum past `f64` -- one
    /// wild row whose square overflows, or a row the meat's running sum
    /// cannot take -- only ages, as a residual that is not finite does: the
    /// sandwich after it is the one that aged past it, and its errors stay
    /// numbers. Folded, its `inf` stayed in the meat for good and every
    /// later standard error read `NaN.max(0.0) = 0.0`, an exact coefficient
    /// (review round 9, B-2; docs/PLAN.md task 270).
    #[test]
    fn a_score_past_f64_is_refused() {
        let mut st = 29u64;
        let rows: Vec<(f64, f64)> = (0..80).map(|_| (lcg(&mut st), lcg(&mut st))).collect();
        let (mut s, mut aged) = (Sandwich::new(1, true, 2), Sandwich::new(1, true, 2));
        for (i, &(x, e)) in rows.iter().enumerate() {
            if i == 40 {
                s.update(&[1e60], 1e100, 1.0, 1.0);
                aged.update(&[1e60], f64::NAN, 1.0, 1.0);
            }
            s.update(&[x], e, 1.0, 1.0);
            aged.update(&[x], e, 1.0, 1.0);
        }
        assert_eq!(s, aged, "the wild row only aged");
        for hac in [false, true] {
            let se = s.standard_errors(&[0], hac).unwrap();
            assert!(se.iter().all(|v| v.is_finite() && *v > 0.0), "{se:?}");
        }
        // Each square finite, their sum not: the second is refused.
        let mut s = Sandwich::new(1, true, 0);
        s.update(&[0.0], 1e154, 1.0, 1.0);
        let one = s.clone();
        s.update(&[0.0], 1e154, 1.0, 1.0);
        assert_eq!(s, one);
        assert!(s.has_shape(1, true, 0));
    }

    /// The guard reads each sum's own diagonal after the row, `B_ii + w
    /// z_i²` and `M_ii + u_i²`, and refuses the row only when one would be
    /// past `f64`. With no intercept, `z = x` on two features, the second
    /// at `1e154`: its square `1e308` fits, a second row's does not, and
    /// the row only ages, while the off-diagonal `Σ w z_1 z_2`, at `1e154`,
    /// would take it; the same of the meat, with `u_2 = 1e154` from a
    /// feature of `1e100` and a residual of `1e54`. Two diagonals of
    /// `1e200` sum to `2e200`, a number, and the second row is folded,
    /// though their product is not one. A weight of `1e-10` on a feature of
    /// `1e155` gives a term `(w z) z = 1e300`, folded, though `z²` alone is
    /// past `f64`.
    #[test]
    fn a_row_is_refused_by_its_own_diagonals_after_it() {
        let refused = |x: [f64; 2], e: f64, w: f64| {
            let mut s = Sandwich::new(2, false, 0);
            s.update(&x, e, 1.0, w);
            let one = s.clone();
            assert!(s.has_shape(2, false, 0), "the first row is folded");
            s.update(&x, e, 1.0, w);
            assert!(s.has_shape(2, false, 0), "{x:?}, e {e}: every sum a number");
            s == one
        };
        assert!(refused([1.0, 1e154], 1e-200, 1.0), "the bread's diagonal");
        assert!(refused([1.0, 1e100], 1e54, 1.0), "the meat's diagonal");
        assert!(!refused([1.0, 1e100], 1e-200, 1.0), "two terms of 1e200");
        let mut s = Sandwich::new(2, false, 0);
        s.update(&[1.0, 1e100], 1e-200, 1.0, 1.0);
        s.update(&[1.0, 1e100], 1e-200, 1.0, 1.0);
        assert_eq!(s.bread[3], 2e200);
        let mut s = Sandwich::new(2, false, 0);
        s.update(&[1.0, 1e155], 1e-200, 1.0, 1e-10);
        assert_eq!(s.bread[3], 1e-10 * 1e155 * 1e155);
        assert!(s.bread[3] > 1e299 && s.bread[3].is_finite());
    }

    /// A variance that is not a number -- here `B⁻¹ M B⁻¹` past `f64`, its
    /// intercept mapped by `inf − inf` -- is no standard error, not one of
    /// 0.0, which reads as an exact coefficient (review round 9, B-2).
    #[test]
    fn a_variance_that_is_not_a_number_is_no_standard_error() {
        let mut s = Sandwich::new(1, true, 0);
        let mut st = 31u64;
        for _ in 0..20 {
            s.update(&[5.0 + 1e-9 * lcg(&mut st)], 1e150 * lcg(&mut st), 1.0, 1.0);
        }
        let v = s.covariance(&[0], false).unwrap();
        assert!(
            v.iter().any(|v| v.is_nan()),
            "the case: a NaN variance, {v:?}"
        );
        let se = s.standard_errors(&[0], false);
        let se = se.as_deref().unwrap_or(&[]);
        assert!(se.iter().all(|v| *v != 0.0), "{se:?}");
    }

    /// A sandwich over `k` rows of `(1, x)` at `lags` lags, its ring full.
    fn folded(k: usize, lags: usize, rows: usize, seed: u64) -> Sandwich {
        let mut s = Sandwich::new(k, true, lags);
        let mut st = seed;
        for _ in 0..rows {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut st) + 3.0).collect();
            s.update(&x, lcg(&mut st), 0.95, 1.0);
        }
        s
    }

    /// The shape a restored sandwich must hold, one part broken at a time:
    /// the configuration it was made for, each sum's size, the ring's depth
    /// and width, and every number finite. A fresh one and one with its
    /// ring full both hold it.
    #[test]
    fn each_part_of_the_shape_is_checked_alone() {
        assert!(Sandwich::new(2, true, 2).has_shape(2, true, 2));
        let s = folded(2, 2, 6, 17);
        assert_eq!((s.origin.len(), s.ring.len()), (2, 2));
        assert!(s.has_shape(2, true, 2));
        type Edit<'a> = (&'a str, &'a dyn Fn(&mut Sandwich));
        let broken: [Edit; 15] = [
            ("k", &|s| s.k = 3),
            ("intercept", &|s| s.intercept = false),
            ("lags", &|s| s.lags = 3),
            ("origin", &|s| {
                s.origin.pop();
            }),
            ("bread", &|s| {
                s.bread.pop();
            }),
            ("meat", &|s| {
                s.meat.pop();
            }),
            ("gammas", &|s| {
                s.gammas.pop();
            }),
            ("a gamma", &|s| {
                s.gammas[1].pop();
            }),
            ("ring depth", &|s| s.ring.push_back(vec![0.0; 3])),
            ("ring width", &|s| s.ring[1].push(0.0)),
            ("bread NaN", &|s| s.bread[4] = f64::NAN),
            ("meat inf", &|s| s.meat[8] = f64::INFINITY),
            ("gamma NaN", &|s| s.gammas[1][3] = f64::NAN),
            ("ring NaN", &|s| s.ring[1][2] = f64::NAN),
            ("origin NaN", &|s| s.origin[1] = f64::NAN),
        ];
        for (what, edit) in broken {
            let mut t = s.clone();
            edit(&mut t);
            assert!(!t.has_shape(2, true, 2), "{what}");
        }
        assert!(!s.has_shape(2, true, 3) && !s.has_shape(2, false, 2));
    }

    /// Wald's statistic is `d' V⁻¹ d` over the picked block of the
    /// covariance, solved by faer's LU (`crate::oracle`); it refuses an
    /// empty pick, a distance of another length, a position outside the
    /// covariance, and a block that does not factorize -- a fit whose
    /// residuals were all 0, so the meat is.
    #[test]
    fn wald_is_its_definition_and_refuses_what_it_cannot_test() {
        let s = folded(2, 2, 40, 23);
        let idx = [0, 1];
        let v = s.covariance(&idx, true).unwrap();
        let pick = [1, 2];
        let d = [0.3, -0.2];
        let v = &v;
        let sub: Vec<f64> = pick
            .iter()
            .flat_map(|&i| pick.iter().map(move |&j| v[i * 3 + j]))
            .collect();
        let x = crate::oracle::solve(&sub, &d);
        let want: f64 = x.iter().zip(&d).map(|(a, b)| a * b).sum();
        let got = s.wald(&idx, true, &pick, &d).unwrap();
        // Task 274: the model's Cholesky form of the same block against the
        // oracle's LU form, each side's rounding bounded (`crate::oracle`);
        // it was `1e-10` relative.
        // Measured on aarch64 (macOS): equal to the bit; the result moved by 100u of itself
        // fails it (6.5 times the bound).
        let bound = crate::oracle::quad_error(&sub, &d, &crate::oracle::cholesky_backward(&sub))
            + crate::oracle::quad_form_error(&sub, &d);
        crate::oracle::assert_within(got, want, bound, "wald");
        assert!(s.wald(&idx, true, &[], &[]).is_none(), "nothing picked");
        assert!(s.wald(&idx, true, &pick, &d[..1]).is_none(), "d too short");
        assert!(s.wald(&idx, true, &[0, 3], &d).is_none(), "outside");
        let mut quiet = Sandwich::new(2, true, 2);
        let mut st = 29u64;
        for _ in 0..40 {
            let x = [lcg(&mut st), lcg(&mut st)];
            quiet.update(&x, 0.0, 0.95, 1.0);
        }
        let v = quiet.covariance(&idx, true).unwrap();
        assert!(v.iter().all(|&a| a == 0.0), "no residual, no meat");
        assert!(quiet.wald(&idx, true, &pick, &d).is_none());
    }

    /// Task 252: with no intercept and no feature asked for there is no
    /// coefficient, so no covariance, no standard error and no Wald test,
    /// however many rows were folded; the same sandwich over its feature
    /// has all three.
    #[test]
    fn no_coefficient_has_no_covariance() {
        let mut s = Sandwich::new(1, false, 2);
        for i in 0..20 {
            let x = f64::from(i % 5) - 2.0;
            s.update(&[x], 0.1 * x + 0.05 * f64::from(i % 3), 1.0, 1.0);
        }
        assert!(s.covariance(&[], false).is_none());
        assert!(s.covariance(&[], true).is_none());
        assert!(s.standard_errors(&[], true).is_none());
        assert!(s.wald(&[], true, &[0], &[1.0]).is_none());
        assert!(s.covariance(&[0], true).is_some(), "the control");
    }
}
