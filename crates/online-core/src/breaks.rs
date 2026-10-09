//! Coefficient breaks (docs/PLAN.md task 221 (b)): the CUSUM and CUSUM of
//! squares of the predictive studentized residual, with a memory of their
//! own, and the Wald distance between a fast and a slow least-squares fit.
//!
//! **The studentized residual.** Each scored row's residual over its own
//! error inflation is its recursive residual, and the studentized residual
//! is that over the spread of the recursive residuals before the row:
//!
//! ```text
//! v = resid / error_inflation,     error_inflation = sqrt(1 + h(x))
//! z = v / s,                       s² = Σ ω v² / Σ ω   (the rows before)
//! ```
//!
//! `h(x)` is the estimation variance of the row's prediction over the noise.
//! Run once with no ridge, `v` is Brown, Durbin and Evans' (1975) recursive
//! residual, `N(0, σ²)` under a constant relationship, and `s` is their
//! spread so far, so `z` is `t`-distributed and near `N(0, 1)`; `z` waits
//! until `s` has [`SCALE_ROWS`] rows of Kish's size. A model with no per-row
//! estimation variance (every one but `ewridge`, `rls` and `kalman`) stands
//! in `error_inflation = 1`: `v` is the residual, and `z` a `zscore` at the
//! diagnostic's memory. The `sigma` field is not the scale: it is the spread
//! of `resid`, which carries the estimation error that `error_inflation`
//! divides out, so `resid / (sigma · error_inflation)` counts it twice. Run
//! once, where the first residuals carry a great deal of it, that scale
//! flagged 47% of 200 streams with no break at the CUSUM of squares' 5%
//! boundary; this one flagged 5.0% (task 221's measurements).
//!
//! **[`Breaks`]**, per slot: the scale's sums and four of `z`, `ω` each
//! row's weight times the decay since it,
//!
//! ```text
//! S1 = Σ ω    S2 = Σ ω²    Z1 = Σ ω z    Z2 = Σ ω z²
//! cusum    = Z1 / sqrt(S2)
//! cusum_sq = (Z2 / S1 − 1) · S1 / sqrt(2 S2)
//! ```
//!
//! Under a constant relationship and Gaussian errors both are about `N(0, 1)`
//! on every row: `Z1` is a weighted sum of near-independent `N(0, 1)` with
//! variance `S2`, and `Z2 / S1` a weighted mean of `χ²(1)` with variance
//! `2 S2 / S1²`. Run once (no decay, unit weights) `S2` is the row count
//! `r`, so `cusum · sqrt(r)` is the CUSUM path `W_r` of Brown, Durbin and
//! Evans, read against their boundary `±a (sqrt(T) + 2 r / sqrt(T))` over a
//! run of `T` rows (`a` = 0.948 at 5%), and `Z2` the numerator of their
//! CUSUM of squares. With a memory, `cusum` is a moving sum (Chu, Hornik and
//! Kuan 1995) with an exponential window in place of a rectangular one.

use serde::{Deserialize, Serialize};

/// The Kish size of the recursive residuals' spread before a studentized
/// residual is read against it: measured, 3, 10 and 30 flagged the same
/// share of streams with no break at Brown, Durbin and Evans' boundaries
/// (task 221 (b)), and from 10 on `t`'s tails are those of a normal to a
/// few percent.
pub const SCALE_ROWS: f64 = 10.0;

/// The CUSUM and CUSUM of squares of one slot's studentized residuals, and
/// the spread they are studentized by. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Breaks {
    s1: f64,
    s2: f64,
    z1: f64,
    z2: f64,
    /// The recursive residuals' `Σ ω`, `Σ ω²` and `Σ ω v²`.
    v1: f64,
    v2: f64,
    vq: f64,
}

impl Breaks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a restored accumulator can be read and folded: finite sums,
    /// the weights' and the squares' not negative.
    pub fn has_shape(&self) -> bool {
        [
            self.s1, self.s2, self.z1, self.z2, self.v1, self.v2, self.vq,
        ]
        .iter()
        .all(|v| v.is_finite())
            && [self.s1, self.s2, self.z2, self.v1, self.v2, self.vq]
                .iter()
                .all(|v| *v >= 0.0)
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.s1 *= lam;
        self.s2 *= lam * lam;
        self.z1 *= lam;
        self.z2 *= lam;
        self.v1 *= lam;
        self.v2 *= lam * lam;
        self.vq *= lam;
    }

    /// The studentized residual of a recursive residual `v` against the
    /// spread before it: `None` for a `v` that is not finite, and until the
    /// spread has [`SCALE_ROWS`] rows of Kish's size and is above 0.
    pub fn studentized(&self, v: f64) -> Option<f64> {
        if !v.is_finite() || self.v2 <= 0.0 || self.v1 * self.v1 < SCALE_ROWS * self.v2 {
            return None;
        }
        let ms = self.vq / self.v1;
        (ms > 0.0).then(|| v / ms.sqrt())
    }

    /// One recursive residual `v` at weight `w`, after the step's decay
    /// `lam`: its studentized residual joins the sums, then `v` the spread.
    /// A `v` that is not finite, or a weight of 0, only ages (CLAUDE.md hard
    /// rule 9).
    pub fn update(&mut self, v: f64, lam: f64, w: f64) {
        let z = self.studentized(v);
        self.age(lam);
        if !(v.is_finite() && w > 0.0) {
            return;
        }
        if let Some(z) = z {
            self.s1 += w;
            self.s2 += w * w;
            self.z1 += w * z;
            self.z2 += w * z * z;
        }
        self.v1 += w;
        self.v2 += w * w;
        self.vq += w * v * v;
    }

    /// `Z1 / sqrt(S2)`: `None` before the first studentized residual.
    pub fn cusum(&self) -> Option<f64> {
        (self.s2 > 0.0).then(|| self.z1 / self.s2.sqrt())
    }

    /// `(Z2/S1 − 1) · S1 / sqrt(2 S2)`: `None` before the first studentized
    /// residual.
    pub fn cusum_sq(&self) -> Option<f64> {
        (self.s1 > 0.0 && self.s2 > 0.0)
            .then(|| (self.z2 / self.s1 - 1.0) * self.s1 / (2.0 * self.s2).sqrt())
    }
}

/// The fast and slow fits of one target, and the weights' cross sum:
/// two exponentially weighted least-squares fits of the target on the
/// features, one at the diagnostic's memory (fast) and one at four times it
/// (slow), each as EW means and centred co-moments ([`crate::EwCov`]), and
/// the Wald distance between their coefficients:
///
/// ```text
/// Δ        = β_fast − β_slow           (the intercept and the slopes)
/// Var(Δ)  ≈ σ² G⁻¹ c,   c = 1/n_fast + 1/n_slow − 2 Σω_f ω_s / (Σω_f · Σω_s)
/// wald     = Δ' G Δ / (σ² c)
/// ```
///
/// `G` is the slow fit's EW second moments of `(1, x)`, `σ²` its residual
/// mean square at Kish's degrees of freedom, `n_*` each fit's Kish size and
/// `Σω_f ω_s` the weights' cross sum, decayed by both factors. Under a
/// constant relationship, with `G` and `σ²` steady, `wald` is approximately
/// `χ²(k)` on each row, `k` the coefficients compared. The slow factor is
/// the fast one's fourth root, two square roots, which are exact in every
/// libm: no platform's last bit enters the state.
///
/// Four, measured against two and eight beside `ewridge` at a half-life of
/// 200 (200 streams of 3,000 rows, a break at row 1,500, the fast fit at
/// the model's memory; first passage of `chi2(3)`'s 0.01% value, 21.1, as
/// a median delay in rows):
///
/// ```text
///              no break               intercept +0.5 sd       slope 1.0 -> 1.6
///  slow   rows>5%  rows>1%  >21.1   rows>1%  >21.1 (delay)   rows>1%  >21.1 (delay)
///   2x    4.60%    0.91%    0%      74%      100% (116)      82%      100% (88)
///   4x    4.63%    0.86%    0%      92%      100% (122)      95%      100% (94)
///   8x    4.70%    0.83%    0%      94%      100% (125)      96%      100% (96)
/// ```
///
/// Each is calibrated with no break. Twice flags a break first by a few
/// rows but holds it on fewer rows after it, and passed `chi2(3)`'s 0.1%
/// value somewhere in 13% of the no-break streams against 9% at four and
/// eight times; eight is four within the noise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinFit {
    /// EW moments of `(x_0, ..., x_{k-1}, y)` at the fast memory.
    fast: crate::EwCov,
    /// The same at the slow memory.
    slow: crate::EwCov,
    /// `Σ ω_f ω_s`: each row's weight squared, decayed by both factors.
    cross: f64,
}

/// The slow memory's factor from the fast one's: its fourth root, a half-life
/// four times as long, by two square roots, which every libm rounds exactly.
pub fn slow_factor(lam: f64) -> f64 {
    lam.sqrt().sqrt()
}

impl TwinFit {
    /// Two empty fits over `k` features.
    pub fn new(k: usize) -> Self {
        Self {
            fast: crate::EwCov::new(k + 1).without_runs(),
            slow: crate::EwCov::new(k + 1).without_runs(),
            cross: 0.0,
        }
    }

    /// Whether a restored pair is shaped for `k` features.
    pub fn has_shape(&self, k: usize) -> bool {
        self.fast.has_shape(k + 1)
            && self.slow.has_shape(k + 1)
            && self.fast.q_sum().is_some()
            && self.slow.q_sum().is_some()
            && self.cross.is_finite()
    }

    /// The clock moves on by a step whose fast decay is `lam`, with nothing
    /// learned.
    pub fn age(&mut self, lam: f64) {
        let slow = slow_factor(lam);
        self.fast.decay(lam);
        self.slow.decay(slow);
        self.cross *= lam * slow;
    }

    /// One row: its features, its target and its weight, after the step's
    /// fast decay `lam`. A target that is not finite, or a weight of 0, only
    /// ages.
    pub fn update(&mut self, x: &[f64], y: f64, lam: f64, w: f64) {
        if !(y.is_finite() && w > 0.0 && x.iter().all(|v| v.is_finite())) {
            return self.age(lam);
        }
        let slow = slow_factor(lam);
        let mut row = Vec::with_capacity(x.len() + 1);
        row.extend_from_slice(x);
        row.push(y);
        self.fast.update(&row, lam, w);
        self.slow.update(&row, slow, w);
        self.cross = lam * slow * self.cross + w * w;
    }

    /// The Wald distance between the two fits' coefficients over the
    /// features `idx` (positions in `x`), with the intercept when
    /// `intercept`: `None` until both fits are determined -- each with more
    /// than `k` rows of Kish's size and a Gram that factorizes -- and while
    /// the memories coincide (no decay: the run-once form has no slow fit)
    /// or the slow fit leaves no residual.
    pub fn wald(&self, idx: &[usize], intercept: bool) -> Option<f64> {
        let kx = self.fast.k() - 1;
        let k = idx.len() + usize::from(intercept);
        if k == 0 {
            return None;
        }
        let (nf, ns) = (self.fast.n_kish()?, self.slow.n_kish()?);
        if nf <= k as f64 || ns <= k as f64 {
            return None;
        }
        let (wf, ws) = (self.fast.n_eff(), self.slow.n_eff());
        let c = 1.0 / nf + 1.0 / ns - 2.0 * self.cross / (wf * ws);
        if c.is_nan() || c <= 1e-12 * (1.0 / nf + 1.0 / ns) {
            return None;
        }
        // The second moments a fit reads: centred with an intercept, raw
        // through the origin.
        let moment = |m: &crate::EwCov, i: usize, j: usize| {
            if intercept { m.cov(i, j) } else { m.raw(i, j) }
        };
        let solve = |m: &crate::EwCov| -> Option<Vec<f64>> {
            let p = idx.len();
            if p == 0 {
                return Some(Vec::new());
            }
            let a: Vec<f64> = idx
                .iter()
                .flat_map(|&i| idx.iter().map(move |&j| (i, j)))
                .map(|(i, j)| moment(m, i, j))
                .collect();
            let b: Vec<f64> = idx.iter().map(|&i| moment(m, i, kx)).collect();
            crate::solve_spd(&a, &b, p, 1)
                .filter(|(_, attempts)| *attempts == 0)
                .map(|(beta, _)| beta)
        };
        let (bf, bs) = (solve(&self.fast)?, solve(&self.slow)?);
        let delta: Vec<f64> = bf.iter().zip(&bs).map(|(f, s)| f - s).collect();
        // `Δβ' C_s Δβ`, the slopes' part, in the slow fit's moments.
        let mut quad = 0.0;
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                quad += delta[a] * moment(&self.slow, i, j) * delta[b];
            }
        }
        if intercept {
            // `Δα + m_s'Δβ = (m_yf − m_ys) − β_f'(m_xf − m_xs)`.
            let mut level = self.fast.mean(kx) - self.slow.mean(kx);
            for (a, &i) in idx.iter().enumerate() {
                level -= bf[a] * (self.fast.mean(i) - self.slow.mean(i));
            }
            quad += level * level;
        }
        let fitted: f64 = idx
            .iter()
            .zip(&bs)
            .map(|(&i, b)| b * moment(&self.slow, i, kx))
            .sum();
        let s2 = (moment(&self.slow, kx, kx) - fitted) * ns / (ns - k as f64);
        if s2.is_nan() || s2 <= 0.0 {
            return None;
        }
        Some(quad / (s2 * c))
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

    /// The scale and the four sums by their definition, from every row and
    /// its weight as it stands, under uneven weights, zeros and a decay:
    /// each row's `z` is its `v` over the spread of the rows before it, read
    /// once they have `SCALE_ROWS` rows of Kish's size.
    #[test]
    fn breaks_are_their_weighted_sums() {
        let mut b = Breaks::new();
        let mut s = 5u64;
        // (v, z or NaN, weight now)
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let lam = 0.98;
        for i in 0..300 {
            let v = 3.0 * lcg(&mut s);
            let w = if i % 7 == 0 { 0.0 } else { 1.0 + lcg(&mut s) };
            let (v1, v2, vq) = rows.iter().fold((0.0, 0.0, 0.0), |a, r| {
                (a.0 + r.2, a.1 + r.2 * r.2, a.2 + r.2 * r.0 * r.0)
            });
            let z = if v2 > 0.0 && v1 * v1 / v2 >= SCALE_ROWS {
                v / (vq / v1).sqrt()
            } else {
                f64::NAN
            };
            let got = b.studentized(v);
            assert_eq!(got.is_some(), z.is_finite(), "row {i}");
            if let Some(g) = got {
                assert!((g - z).abs() < 1e-12 * z.abs().max(1.0), "row {i}");
            }
            b.update(v, lam, w);
            rows.iter_mut().for_each(|r| r.2 *= lam);
            rows.push((v, z, w));
            let scored: Vec<&(f64, f64, f64)> = rows
                .iter()
                .filter(|r| r.1.is_finite() && r.2 > 0.0)
                .collect();
            let s1: f64 = scored.iter().map(|r| r.2).sum();
            let s2: f64 = scored.iter().map(|r| r.2 * r.2).sum();
            let z1: f64 = scored.iter().map(|r| r.2 * r.1).sum();
            let z2: f64 = scored.iter().map(|r| r.2 * r.1 * r.1).sum();
            if s2 == 0.0 {
                assert!(b.cusum().is_none() && b.cusum_sq().is_none(), "row {i}");
                continue;
            }
            let tol = 1e-10;
            assert!((b.cusum().unwrap() - z1 / s2.sqrt()).abs() < tol, "row {i}");
            let want = (z2 / s1 - 1.0) * s1 / (2.0 * s2).sqrt();
            assert!((b.cusum_sq().unwrap() - want).abs() < tol, "row {i}");
        }
        assert!(b.has_shape());
    }

    /// A zero weight, the first included, and a `v` that is not a number
    /// only age the sums.
    #[test]
    fn zero_weight_and_missing_rows_only_age() {
        let mut b = Breaks::new();
        b.update(2.0, 0.9, 0.0);
        b.update(f64::NAN, 0.9, 1.0);
        assert_eq!(b, Breaks::new());
        for v in [
            1.0, -0.5, 0.3, 2.0, -1.0, 0.7, -0.2, 1.1, -0.9, 0.4, 0.8, -0.6,
        ] {
            b.update(v, 0.9, 1.0);
        }
        let before = b.clone();
        b.update(5.0, 0.5, 0.0);
        assert_eq!(b.z1, before.z1 * 0.5);
        assert_eq!(b.vq, before.vq * 0.5);
        assert_eq!(b.s2, before.s2 * 0.25);
        assert_eq!(b.cusum(), before.cusum());
    }

    /// The fast and slow fits' slopes, intercepts, residual spread and the
    /// cross sum against their definitions: weighted least squares by
    /// faer's LU (`crate::oracle`) from every row at each memory's weight,
    /// and the statistic built from them by the module's formula.
    #[test]
    fn the_wald_distance_is_the_two_fits_by_definition() {
        // (x, y, weight, its fast weight now, its slow weight now)
        type Row = ([f64; 2], f64, f64, f64, f64);
        let mut t = TwinFit::new(2);
        let mut s = 9u64;
        let lam = 0.97_f64;
        let slow = slow_factor(lam);
        let mut rows: Vec<Row> = Vec::new();
        for i in 0..400 {
            let x = [lcg(&mut s) * 2.0 + 3.0, lcg(&mut s)];
            let shift = if i < 250 { 0.0 } else { 1.0 };
            let y = 1.0 + (0.5 + shift) * x[0] - x[1] + 0.3 * lcg(&mut s);
            let w = if i % 9 == 4 {
                0.0
            } else {
                0.5 + lcg(&mut s) + 0.5
            };
            t.update(&x, y, lam, w);
            rows.iter_mut().for_each(|r| {
                r.3 *= lam;
                r.4 *= slow;
            });
            rows.push((x, y, w, w, w));
            if i < 30 || i % 37 != 0 {
                continue;
            }
            let fit = |pick: fn(&Row) -> f64| {
                let (mut g, mut r, mut sw, mut sw2) = (vec![0.0; 9], vec![0.0; 3], 0.0, 0.0);
                for row in &rows {
                    let om = pick(row);
                    let z = [1.0, row.0[0], row.0[1]];
                    for a in 0..3 {
                        r[a] += om * z[a] * row.1;
                        for b in 0..3 {
                            g[a * 3 + b] += om * z[a] * z[b];
                        }
                    }
                    sw += om;
                    sw2 += om * om;
                }
                let beta = crate::oracle::solve(&g, &r);
                let ssr: f64 = rows
                    .iter()
                    .map(|row| {
                        let e = row.1 - beta[0] - beta[1] * row.0[0] - beta[2] * row.0[1];
                        pick(row) * e * e
                    })
                    .sum();
                (beta, g, sw, sw2, ssr)
            };
            let (bf, _, wf, wf2, _) = fit(|r| r.3);
            let (bs, gs, ws, ws2, ssr_s) = fit(|r| r.4);
            let cross: f64 = rows.iter().map(|r| r.3 * r.4).sum();
            let (nf, ns) = (wf * wf / wf2, ws * ws / ws2);
            let c = 1.0 / nf + 1.0 / ns - 2.0 * cross / (wf * ws);
            let s2 = ssr_s / ws * ns / (ns - 3.0);
            let d: Vec<f64> = (0..3).map(|a| bf[a] - bs[a]).collect();
            let quad: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| d[a] * gs[a * 3 + b] / ws * d[b])
                .sum();
            let want = quad / (s2 * c);
            let got = t.wald(&[0, 1], true).unwrap();
            assert!(
                (got - want).abs() < 1e-7 * want.max(1.0),
                "row {i}: {got} vs {want}"
            );
        }
        // After the break the slow fit lags the fast one.
        assert!(t.wald(&[0, 1], true).unwrap() > 30.0);
        assert!(t.has_shape(2));
    }

    /// Run once (no decay) the two memories coincide and there is no
    /// distance to read; through the origin the raw moments are read.
    #[test]
    fn no_decay_has_no_slow_fit_and_no_intercept_reads_raw_moments() {
        let mut t = TwinFit::new(1);
        let mut s = 2u64;
        for _ in 0..100 {
            let x = lcg(&mut s) + 2.0;
            t.update(&[x], 3.0 * x + 0.1 * lcg(&mut s), 1.0, 1.0);
        }
        assert!(t.wald(&[0], true).is_none());
        let mut u = TwinFit::new(1);
        for i in 0..300 {
            let x = lcg(&mut s) + 2.0;
            let slope = if i < 200 { 3.0 } else { 3.5 };
            u.update(&[x], slope * x + 0.1 * lcg(&mut s), 0.95, 1.0);
        }
        assert!(u.wald(&[0], false).unwrap() > 10.0);
        assert!(u.wald(&[], false).is_none(), "nothing to compare");
    }
}
