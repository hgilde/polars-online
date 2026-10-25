//! The Huber refit: each target's first rows refit to convergence, once,
//! for `huber` ([`crate::Robust`]) and `sgd(loss = "huber")` ([`crate::Sgd`])
//! (docs/HUBER.md §3; docs/PLAN.md task 277).
//!
//! **Why.** A stream weights each row once, by its residual against the fit
//! before it. Over the first rows that fit is close to least squares, so a
//! share of outliers pulls it, and the residual scale `s` learns the pulled
//! fit's residuals: a scale several times the noise's. The cut `δs` is then
//! wide, the outliers keep entering at nearly full weight, and the fit stays
//! where they pulled it -- *captured*. Without decay it stays for good: the
//! early rows keep their share of the sums, and the scale its memory of
//! them. Measured on `y = 1000 + 2x + 0.5ε` with +50 on a share of the rows
//! with `x > 0`, 20,000 rows, no decay, 40 seeds: 6, 13 and 25 of 40 streams
//! ended more than 0.05 in slope from statsmodels' converged `RLM` at 10%,
//! 12.5% and 20% outliers, the worst by 0.19, 0.43 and 4.3, while the
//! median stream was within 0.002.
//!
//! **What.** Each target keeps its first `refit_rows` learned rows (a row
//! of positive weight with its target), and every row in between that only
//! aged its sums, in a [`RefitBuf`]. On the row the count is reached, after
//! the row's own prediction and update, the buffered rows are refit by
//! iteratively reweighted least squares to convergence:
//!
//! ```text
//! v_i = w_i · Π_{l > i} λ_l                 the row's raw weight, decayed to now
//! r_i = y_i − z_i'β                         against the last pass's fit
//! s   = wmedian(|r|; v) / Φ⁻¹(3/4)          (or √(Σ v r² / Σ v) under "rms")
//! u_i = min(1, δs / |r_i|)
//! β   = argmin Σ v_i u_i (y_i − z_i'β)²      the model's own solve
//! ```
//!
//! until no coefficient moves by more than [`REFIT_TOL`] of `max|β|`,
//! or [`REFIT_PASSES`] passes. That is statsmodels' `RLM` loop with the
//! MAD's scale (`scale_est = "mad"`, centred at 0), weighted by the decay.
//! Then the residual scale is rebuilt from the converged fit's residuals
//! `y_i − z_i'β`, at the rows' raw weights and aged on the scale's own
//! clock as the stream aged it: a refit that keeps the scale the pulled
//! fit taught is pulled back within a few hundred rows. The buffer is then
//! freed and the stream goes on online, as before.
//!
//! The weight a model reports and gates on (`n_eff`, each target's own) is
//! the raw weights' and does not move (CLAUDE.md hard rule 8); the refit
//! fires on a count of learned rows, never at a chunk's end (rule 3); a row
//! of weight 0 is buffered as an ageing alone and does not count (rule 9).

use serde::{Deserialize, Serialize};

use crate::robust::{HuberScale, NORMAL_Q3, mad_sketches};
use crate::{EwQuantile, SpdFactor};

/// The most IRLS passes a refit takes.
pub const REFIT_PASSES: usize = 50;

/// A refit has converged when no coefficient moved by more than this,
/// relative to `max|β|`.
pub const REFIT_TOL: f64 = 1e-10;

/// A spec's default `huber_refit_rows`: measured, 200 rows took the
/// captured streams from 13 and 25 of 40 to 0 and 4 at 12.5% and 20%
/// outliers, where 50 rows still left 14 of 40 at 20% (the module docs).
pub const DEFAULT_REFIT_ROWS: u32 = 200;

/// One target's refit buffer: from the first row it learned, every row as
/// its sums took it, flat -- the features, the target, the raw weight, the
/// row's decay factor and the scale's -- a row that only aged them (the
/// target null, or no weight) as zeros, weight 0 and its two factors, so a
/// replay ages factor by factor, as the stream did; the rows learned; and
/// whether the refit has run (its rows then freed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RefitBuf {
    rows: Vec<f64>,
    learned: u32,
    done: bool,
}

impl RefitBuf {
    pub(crate) fn new() -> Self {
        Self {
            rows: Vec::new(),
            learned: 0,
            done: false,
        }
    }

    /// One per target when the refit is on (`refit_rows > 0`), none off.
    pub(crate) fn per_target(refit_rows: u32, n: usize) -> Vec<Self> {
        if refit_rows > 0 {
            vec![Self::new(); n]
        } else {
            Vec::new()
        }
    }

    /// Whether the buffer still takes rows.
    pub(crate) fn open(&self) -> bool {
        !self.done
    }

    /// The numbers it holds: `(features + 4)` a row.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.rows.len()
    }

    /// A learned row: `true` on the row that fills the buffer to `cap`.
    pub(crate) fn learn(
        &mut self,
        x: &[f64],
        y: f64,
        w: f64,
        lam: f64,
        lam_s: f64,
        cap: u32,
    ) -> bool {
        if self.done {
            return false;
        }
        self.rows.extend_from_slice(x);
        self.rows.extend_from_slice(&[y, w, lam, lam_s]);
        self.learned += 1;
        self.learned >= cap
    }

    /// A row that only aged the target's sums, once a row has been learned
    /// (before that the sums are empty, and ageing them is nothing).
    pub(crate) fn age(&mut self, nf: usize, lam: f64, lam_s: f64) {
        if !self.done && self.learned > 0 {
            self.rows.extend(std::iter::repeat_n(0.0, nf));
            self.rows.extend_from_slice(&[0.0, 0.0, lam, lam_s]);
        }
    }

    /// The rows, taken: the buffer is done and holds none from here.
    pub(crate) fn take(&mut self) -> Vec<f64> {
        self.done = true;
        std::mem::take(&mut self.rows)
    }

    /// Whether a restored buffer is one this configuration writes: whole
    /// rows of `nf` features, fewer learned than `cap`, rows only once one
    /// was learned; none once done.
    pub(crate) fn fits(&self, nf: usize, cap: u32) -> bool {
        if self.done {
            return self.rows.is_empty();
        }
        self.rows.len().is_multiple_of(nf + 4)
            && self.learned < cap
            && (self.learned > 0 || self.rows.is_empty())
    }
}

/// A refit buffer's rows, read: row `i`'s features, target, raw weight and
/// factors ([`RefitBuf`]).
pub(crate) struct Rows<'a> {
    buf: &'a [f64],
    nf: usize,
}

impl<'a> Rows<'a> {
    pub(crate) fn new(buf: &'a [f64], nf: usize) -> Self {
        Self { buf, nf }
    }

    pub(crate) fn len(&self) -> usize {
        self.buf.len() / (self.nf + 4)
    }

    fn row(&self, i: usize) -> &'a [f64] {
        let s = self.nf + 4;
        &self.buf[i * s..(i + 1) * s]
    }

    pub(crate) fn x(&self, i: usize) -> &'a [f64] {
        &self.row(i)[..self.nf]
    }

    pub(crate) fn y(&self, i: usize) -> f64 {
        self.row(i)[self.nf]
    }

    pub(crate) fn w(&self, i: usize) -> f64 {
        self.row(i)[self.nf + 1]
    }

    pub(crate) fn lam(&self, i: usize) -> f64 {
        self.row(i)[self.nf + 2]
    }

    pub(crate) fn lam_s(&self, i: usize) -> f64 {
        self.row(i)[self.nf + 3]
    }

    /// Whether row `i` was learned: one that only aged the sums has weight 0.
    pub(crate) fn learned(&self, i: usize) -> bool {
        self.w(i) > 0.0
    }

    /// Each row's raw weight decayed to the buffer's last row, which is the
    /// row the refit runs on: `v_i = w_i · Π_{l > i} λ_l`.
    pub(crate) fn decayed_weights(&self) -> Vec<f64> {
        let n = self.len();
        let mut v = vec![0.0; n];
        let mut age = 1.0;
        for i in (0..n).rev() {
            v[i] = self.w(i) * age;
            age *= self.lam(i);
        }
        v
    }
}

/// `β'z` for the augmented row `z = [1, x]` with an intercept, `x` without.
pub(crate) fn fitted(beta: &[f64], x: &[f64], intercept: bool) -> f64 {
    crate::solve::dot_aug(beta, x, intercept)
}

/// The weighted median of `a` at weights `v`: the smallest value at which
/// the cumulative weight reaches half the total, over the entries of
/// positive weight. NaN when there is none.
pub(crate) fn weighted_median(a: &[f64], v: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..a.len()).filter(|&i| v[i] > 0.0).collect();
    idx.sort_by(|&p, &q| a[p].total_cmp(&a[q]));
    let total: f64 = idx.iter().map(|&i| v[i]).sum();
    let mut cum = 0.0;
    for &i in &idx {
        cum += v[i];
        if cum >= 0.5 * total {
            return a[i];
        }
    }
    f64::NAN
}

/// The scale a pass draws its cut in, from the residuals' sizes at the
/// rows' decayed weights: the weighted median over [`NORMAL_Q3`] under the
/// MAD, the weighted root mean square under the RMS; `None` where it is
/// not finite and above 0 (no cut: every weight 1, the models' rule).
pub(crate) fn scale_of(kind: HuberScale, absr: &[f64], v: &[f64]) -> Option<f64> {
    let s = match kind {
        HuberScale::Mad => weighted_median(absr, v) / NORMAL_Q3,
        HuberScale::Rms => {
            let (num, den) = absr
                .iter()
                .zip(v)
                .filter(|(_, w)| **w > 0.0)
                .fold((0.0, 0.0), |(p, q), (&a, &w)| (p + w * a * a, q + w));
            (num / den).sqrt()
        }
    };
    (s > 0.0 && s.is_finite()).then_some(s)
}

/// Each row's Huber weight `min(1, δs/|r|)` into `u`, 1 where there is no
/// scale.
pub(crate) fn huber_weights(absr: &[f64], s: Option<f64>, delta: f64, u: &mut [f64]) {
    u.fill(1.0);
    let Some(s) = s else {
        return;
    };
    let cut = delta * s;
    // `cut / 0` is `inf`, so a residual of 0 keeps the weight 1.
    for (ui, &a) in u.iter_mut().zip(absr) {
        *ui = (cut / a).min(1.0);
    }
}

/// Whether a pass has converged: no coefficient moved by more than
/// [`REFIT_TOL`] of `max|β|`, or none moved at all. Relative alone, so a
/// target scaled by a power of two takes the same passes and its fit is
/// the same fit scaled, to the bit.
pub(crate) fn converged(before: Option<&[f64]>, after: &[f64]) -> bool {
    let Some(before) = before else {
        return false;
    };
    let moved = before
        .iter()
        .zip(after)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    let size = after.iter().map(|q| q.abs()).fold(0.0, f64::max);
    moved <= REFIT_TOL * size
}

/// A target's residual scale rebuilt from the converged fit's residuals.
pub(crate) struct Rebuilt {
    pub sig2: f64,
    pub wsig: f64,
    /// The MAD's sketch, `None` under the RMS.
    pub mad: Option<EwQuantile>,
}

/// The residual scale rebuilt over the buffered rows from the residuals
/// `resid(i)` of the learned ones, at their raw weights, aged on the
/// scale's clock (`λ_s`) row by row as the stream aged it: the EW mean of
/// `e²` (`σ²` and its weight) and, under the MAD, a fresh sketch of `|e|`.
pub(crate) fn rebuild_scale(
    rows: &Rows<'_>,
    kind: HuberScale,
    resid: impl Fn(usize) -> f64,
) -> Rebuilt {
    let mut mad = mad_sketches(kind, 1).pop();
    let (mut sig2, mut wsig) = (0.0, 0.0);
    for i in 0..rows.len() {
        let lam_s = rows.lam_s(i);
        wsig *= lam_s;
        if let Some(q) = mad.as_mut() {
            q.age(lam_s);
        }
        if !rows.learned(i) {
            continue;
        }
        let (e, w) = (resid(i), rows.w(i));
        let ws_new = wsig + w;
        let next = (wsig * sig2 + w * e * e) / ws_new;
        if next.is_finite() {
            sig2 = next;
            wsig = ws_new;
        }
        if let Some(q) = mad.as_mut() {
            q.add(e.abs(), w);
        }
    }
    Rebuilt { sig2, wsig, mad }
}

/// Weighted least squares over the buffered rows at weights `wt`, in mean
/// form: with an intercept the centred Gram over the features, plus `ridge`
/// on its diagonal, and the intercept `ȳ − m'β`; without one the raw Gram
/// plus the ridge. `None` without weight, or when the factor fails at every
/// jitter.
fn wls(rows: &Rows<'_>, wt: &[f64], intercept: bool, ridge: f64) -> Option<Vec<f64>> {
    let nf = rows.nf;
    // Without weight, or with a weight that overflows, the means and
    // moments are `0/0` or `inf/inf`, and the fit's check below refuses it.
    let total: f64 = wt.iter().sum();
    let (mut mx, mut my) = (vec![0.0; nf], 0.0);
    if intercept {
        for (i, &w) in wt.iter().enumerate() {
            // A weight here is `>= 0`: a row of 0 adds nothing.
            if w == 0.0 {
                continue;
            }
            for (m, &x) in mx.iter_mut().zip(rows.x(i)) {
                *m += w * x;
            }
            my += w * rows.y(i);
        }
        mx.iter_mut().for_each(|m| *m /= total);
        my /= total;
    }
    let mut a = vec![0.0; nf * nf];
    let mut b = vec![0.0; nf];
    for (i, &w) in wt.iter().enumerate() {
        if w == 0.0 {
            continue;
        }
        let x = rows.x(i);
        let dy = rows.y(i) - my;
        for p in 0..nf {
            let dp = x[p] - mx[p];
            b[p] += w * dp * dy;
            for q in 0..nf {
                a[p * nf + q] += w * dp * (x[q] - mx[q]);
            }
        }
    }
    for p in 0..nf {
        b[p] /= total;
        for q in 0..nf {
            a[p * nf + q] /= total;
        }
        a[p * nf + p] += ridge;
    }
    let slopes = SpdFactor::of(&a, nf)?.solve(&b, nf, 1);
    let fit: Vec<f64> = if intercept {
        let b0 = my - mx.iter().zip(&slopes).map(|(m, s)| m * s).sum::<f64>();
        std::iter::once(b0).chain(slopes).collect()
    } else {
        slopes
    };
    fit.iter().all(|v| v.is_finite()).then_some(fit)
}

/// The refit of a model without sums of its own (`sgd`): IRLS over the
/// buffered rows from the weighted least-squares fit, statsmodels' start,
/// each pass a weighted least-squares solve ([`wls`]) at `v_i · u_i`, to
/// convergence or [`REFIT_PASSES`]. `None` where no pass solves.
pub(crate) fn irls(
    rows: &Rows<'_>,
    intercept: bool,
    delta: f64,
    kind: HuberScale,
    ridge: f64,
) -> Option<Vec<f64>> {
    let n = rows.len();
    let v = rows.decayed_weights();
    let mut beta = wls(rows, &v, intercept, ridge)?;
    let mut absr = vec![0.0; n];
    let mut u = vec![1.0; n];
    let mut wt = vec![0.0; n];
    for _ in 0..REFIT_PASSES {
        for (i, a) in absr.iter_mut().enumerate() {
            *a = (rows.y(i) - fitted(&beta, rows.x(i), intercept)).abs();
        }
        huber_weights(&absr, scale_of(kind, &absr, &v), delta, &mut u);
        for ((t, &vi), &ui) in wt.iter_mut().zip(&v).zip(&u) {
            *t = vi * ui;
        }
        let Some(next) = wls(rows, &wt, intercept, ridge) else {
            break;
        };
        let done = converged(Some(&beta), &next);
        beta = next;
        if done {
            break;
        }
    }
    Some(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weighted_median_is_the_halfway_point_of_the_weight() {
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0]), 2.0);
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[1.0, 0.0, 5.0]), 2.0);
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[4.0, 1.0, 1.0]), 3.0);
        assert!(weighted_median(&[1.0], &[0.0]).is_nan());
    }

    #[test]
    fn a_buffer_counts_learned_rows_and_ages_only_after_the_first() {
        let mut b = RefitBuf::new();
        b.age(1, 0.5, 0.5);
        assert!(b.rows.is_empty(), "nothing to age before a learned row");
        assert!(!b.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 2));
        b.age(1, 0.5, 0.25);
        assert!(
            b.learn(&[3.0], 4.0, 2.0, 0.9, 0.8, 2),
            "the second fills it"
        );
        let rows = b.rows.clone();
        let r = Rows::new(&rows, 1);
        assert_eq!(r.len(), 3);
        assert!(r.learned(0) && !r.learned(1) && r.learned(2));
        // v = [1 · 0.5 · 0.9, 0, 2]
        assert_eq!(r.decayed_weights(), vec![1.0 * 0.5 * 0.9, 0.0, 2.0]);
        assert!(b.fits(1, 3) && !b.fits(1, 2) && !b.fits(2, 3));
        // Rows held before any was learned are not a buffer this writes.
        let orphan = RefitBuf {
            rows: vec![0.0; 5],
            learned: 0,
            done: false,
        };
        assert!(!orphan.fits(1, 3));
        assert!(RefitBuf::new().fits(1, 3));
        assert_eq!(b.take(), rows);
        assert!(b.fits(1, 3) && !b.open() && b.rows.is_empty());
        assert!(
            !b.learn(&[1.0], 1.0, 1.0, 1.0, 1.0, 2),
            "a done buffer takes nothing"
        );
    }

    /// The rebuilt scale is the stream's own recursion over the rows, at
    /// their raw weights and aged on the scale's clock row by row: `σ²` the
    /// EW mean of `e²` and its weight, and under the MAD a sketch fed the
    /// same residuals. Three rows -- learned, an ageing, learned -- worked
    /// by hand: `W` = 1 · 0.5 · 0.5 + 2 = 2.25, and `σ²` = (0.25 · 4 + 2 · 1)
    /// / 2.25.
    #[test]
    fn the_rebuilt_scale_is_the_streams_recursion() {
        let mut b = RefitBuf::new();
        b.learn(&[1.0], 10.0, 1.0, 0.9, 0.5, 9);
        b.age(1, 0.9, 0.5);
        b.learn(&[2.0], 20.0, 2.0, 0.9, 0.5, 9);
        let rows = b.rows.clone();
        let r = Rows::new(&rows, 1);
        let resid = |i: usize| if i == 0 { 2.0 } else { -1.0 };
        let rms = rebuild_scale(&r, HuberScale::Rms, resid);
        assert_eq!(rms.wsig, 2.25);
        assert_eq!(rms.sig2, (0.25 * 4.0 + 2.0 * 1.0) / 2.25);
        assert!(rms.mad.is_none());
        let mad = rebuild_scale(&r, HuberScale::Mad, resid);
        assert_eq!((mad.wsig, mad.sig2), (rms.wsig, rms.sig2));
        let q = mad.mad.expect("a sketch under the MAD");
        // |e| of 2 at weight 0.25 and 1 at weight 2: the median is 1.
        assert!((q.get(0).unwrap() - 1.0).abs() < 0.01, "{:?}", q.get(0));
        // A residual that is not finite leaves the scale as it was.
        let inf = rebuild_scale(&r, HuberScale::Rms, |i| {
            if i == 0 { 2.0 } else { f64::INFINITY }
        });
        assert_eq!((inf.wsig, inf.sig2), (0.25, 4.0));
    }

    /// Weighted least squares on two rows through `y = 1 + 2x`: the line,
    /// with and without an intercept's slot; and no fit without weight,
    /// whose means would be `0/0`.
    #[test]
    fn weighted_least_squares_needs_weight() {
        let mut b = RefitBuf::new();
        b.learn(&[0.0], 1.0, 1.0, 1.0, 1.0, 9);
        b.age(1, 1.0, 1.0);
        b.learn(&[1.0], 3.0, 1.0, 1.0, 1.0, 9);
        let rows = b.rows.clone();
        let r = Rows::new(&rows, 1);
        let fit = wls(&r, &[1.0, 0.0, 3.0], true, 0.0).unwrap();
        assert!(
            (fit[0] - 1.0).abs() < 1e-12 && (fit[1] - 2.0).abs() < 1e-12,
            "{fit:?}"
        );
        let origin = wls(&r, &[1.0, 0.0, 1.0], false, 0.0).unwrap();
        assert!((origin[0] - 3.0).abs() < 1e-12, "{origin:?}");
        assert_eq!(wls(&r, &[0.0, 0.0, 0.0], true, 0.0), None);
        assert_eq!(wls(&r, &[0.0, 0.0, 0.0], false, 1e-6), None);
        assert_eq!(wls(&r, &[f64::INFINITY, 0.0, 1.0], true, 0.0), None);
    }

    #[test]
    fn the_scale_is_none_without_spread() {
        assert_eq!(scale_of(HuberScale::Mad, &[0.0, 0.0], &[1.0, 1.0]), None);
        assert_eq!(scale_of(HuberScale::Rms, &[0.0], &[1.0]), None);
        let s = scale_of(HuberScale::Rms, &[3.0, 4.0], &[1.0, 1.0]).unwrap();
        assert!((s - 12.5f64.sqrt()).abs() < 1e-15);
        // A row without weight is no part of the scale, whatever its residual.
        assert_eq!(
            scale_of(HuberScale::Rms, &[f64::INFINITY, 3.0], &[0.0, 1.0]),
            Some(3.0)
        );
        assert_eq!(
            scale_of(HuberScale::Mad, &[f64::INFINITY, 3.0], &[0.0, 1.0]),
            Some(3.0 / NORMAL_Q3)
        );
        let mut u = [0.0; 3];
        huber_weights(&[1.0, 4.0, 0.0], Some(1.0), 2.0, &mut u);
        assert_eq!(u, [1.0, 0.5, 1.0]);
        huber_weights(&[1.0, 4.0, 0.0], None, 2.0, &mut u);
        assert_eq!(u, [1.0, 1.0, 1.0]);
    }
}
