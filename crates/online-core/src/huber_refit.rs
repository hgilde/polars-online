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
//! of positive weight with its target), and between them the rows that only
//! aged its sums, as runs of their factors, in a [`RefitBuf`]: at most
//! [`REFIT_RUNS_PER_ROW`] runs a learned row of the count, past which the
//! target gives its refit up and keeps its one-pass fit, so a buffer is
//! bounded however sparse its target. On the row the count is reached, after
//! the row's own prediction and update, the buffered rows are refit by
//! iteratively reweighted least squares to convergence:
//!
//! ```text
//! v_i = w_i · Π_{l > i} λ_l                 the row's raw weight, decayed to now
//! v_i'                                      the same by the scale's factors
//! r_i = y_i − z_i'β                         against the last pass's fit
//! s   = wmedian(|r|; v') / Φ⁻¹(3/4)         (or √(Σ v' r² / Σ v') under "rms")
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

/// A buffer holds at most this many ageing runs per learned row its refit
/// waits for (`refit_rows`): past it the refit of that target is given up
/// ([`RefitBuf::age`]). Under a regular clock repeated ageings are one run,
/// so a target present on a few rows in many still fits; under an irregular
/// clock each ageing row is a run of its own, and a target present on fewer
/// than about one row in five gives up.
pub const REFIT_RUNS_PER_ROW: u32 = 4;

/// A run of more ageing rows than this is replayed as one decay by the
/// factor's power ([`pow_n`]), within rounding of the stream's own decays;
/// a shorter one decay by decay, to the bit.
const EXACT_RUN: u64 = 1 << 16;

/// How one target's refit stands ([`crate::Robust::refit_status`],
/// [`crate::Sgd::refit_status`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefitStatus {
    /// The learned rows it holds while it waits; its count once it ran.
    pub learned: u32,
    /// Whether the refit has run, or was given up.
    pub done: bool,
    /// The IRLS passes it took: [`REFIT_PASSES`] when it stopped at the cap
    /// without converging, 0 before it ran.
    pub passes: u32,
    /// Given up past the cap on ageing runs ([`REFIT_RUNS_PER_ROW`]).
    pub abandoned: bool,
    /// Ran, and no pass solved: the one-pass fit stands.
    pub failed: bool,
}

/// One target's refit buffer, from the first row it learned: the learned
/// rows, `(features + 4)` numbers each -- the features, the target, the raw
/// weight, the row's decay factor and the scale's -- and between them the
/// rows that only aged the target's sums (the target null, or no weight)
/// as runs of `(λ, λ_s, count, learned rows before it)`, a run lengthened
/// while its two factors repeat to the bit; the rows learned; whether the
/// refit has run (its rows then freed); and how it went: the IRLS passes it
/// took (the cap's number when it did not converge), or why it did not fit:
/// given up past the cap on runs (`abandoned`), or no pass solved
/// (`failed`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RefitBuf {
    rows: Vec<f64>,
    ages: Vec<f64>,
    learned: u32,
    done: bool,
    passes: u32,
    abandoned: bool,
    failed: bool,
}

impl RefitBuf {
    pub(crate) fn new() -> Self {
        Self {
            rows: Vec::new(),
            ages: Vec::new(),
            learned: 0,
            done: false,
            passes: 0,
            abandoned: false,
            failed: false,
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

    /// How the refit stands.
    pub(crate) fn status(&self) -> RefitStatus {
        RefitStatus {
            learned: self.learned,
            done: self.done,
            passes: self.passes,
            abandoned: self.abandoned,
            failed: self.failed,
        }
    }

    /// Mark the refit as one that no pass solved.
    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }

    /// Record the IRLS passes the refit took.
    pub(crate) fn ran(&mut self, passes: u32) {
        self.passes = passes;
    }

    /// The numbers it holds.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.rows.len() + self.ages.len()
    }

    /// The ageing runs it holds.
    #[cfg(test)]
    pub(crate) fn runs(&self) -> usize {
        self.ages.len() / 4
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
    /// (before that the sums are empty, and ageing them is nothing): the
    /// last run lengthened when it follows the same learned row with the
    /// same two factors, to the bit, or a new run. Past
    /// [`REFIT_RUNS_PER_ROW`] runs a row of `cap` the target's refit is given
    /// up: the buffer freed, and marked [`Self::abandoned`].
    pub(crate) fn age(&mut self, lam: f64, lam_s: f64, cap: u32) {
        if self.done || self.learned == 0 {
            return;
        }
        let before = f64::from(self.learned);
        let n = self.ages.len();
        if n >= 4 {
            let last = &mut self.ages[n - 4..];
            if last[3] == before
                && last[0].to_bits() == lam.to_bits()
                && last[1].to_bits() == lam_s.to_bits()
            {
                last[2] += 1.0;
                return;
            }
        }
        self.ages.extend_from_slice(&[lam, lam_s, 1.0, before]);
        if self.ages.len() / 4 > (REFIT_RUNS_PER_ROW * cap) as usize {
            self.done = true;
            self.abandoned = true;
            self.rows = Vec::new();
            self.ages = Vec::new();
        }
    }

    /// The rows and runs, taken: the buffer is done and holds none from here.
    pub(crate) fn take(&mut self) -> (Vec<f64>, Vec<f64>) {
        self.done = true;
        (
            std::mem::take(&mut self.rows),
            std::mem::take(&mut self.ages),
        )
    }

    /// Whether a restored buffer is one this configuration writes: exactly
    /// `learned` rows of `nf` features, fewer than `cap`, each weight above 0
    /// and each factor in (0, 1]; runs of whole counts, factors in (0, 1],
    /// following learned rows in order, at most the cap's number; none once
    /// done, and neither mark while open.
    pub(crate) fn fits(&self, nf: usize, cap: u32) -> bool {
        let factor = |f: f64| f > 0.0 && f <= 1.0;
        if self.done {
            return self.rows.is_empty() && self.ages.is_empty();
        }
        let stride = nf + 4;
        let rows_fit = self.rows.len() == self.learned as usize * stride
            && self.learned < cap
            && self.rows.chunks_exact(stride).all(|r| {
                r[nf + 1] > 0.0 && r[nf + 1].is_finite() && factor(r[nf + 2]) && factor(r[nf + 3])
            });
        let mut last = 1.0;
        let ages_fit = self.ages.len().is_multiple_of(4)
            && self.ages.len() / 4 <= (REFIT_RUNS_PER_ROW * cap) as usize
            && self.ages.as_chunks::<4>().0.iter().all(|a| {
                let ok = factor(a[0])
                    && factor(a[1])
                    && a[2] >= 1.0
                    && a[2].fract() == 0.0
                    && a[3] >= last
                    && a[3] <= f64::from(self.learned);
                last = a[3];
                ok
            });
        rows_fit && ages_fit && !self.abandoned && !self.failed && self.passes == 0
    }
}

/// `x` to the power `n`, by squaring: no libm, so the same bits on every
/// platform (CLAUDE.md, no libm in the state); `x` itself at `n = 1`.
pub(crate) fn pow_n(x: f64, n: u64) -> f64 {
    let (mut base, mut n, mut out) = (x, n, 1.0);
    if n == 1 {
        return x;
    }
    while n > 0 {
        if n & 1 == 1 {
            out *= base;
        }
        base *= base;
        n >>= 1;
    }
    out
}

/// One step of a buffer's history, in order ([`Rows::events`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Event {
    /// A run of `count` rows that only aged the sums, each by `lam` (and
    /// the scale by `lam_s`).
    Age { lam: f64, lam_s: f64, count: u64 },
    /// Learned row `i`.
    Row(usize),
}

impl Event {
    /// Apply an ageing run's `count` decays by `lam` through `decay`: one by
    /// one up to [`EXACT_RUN`], as the stream applied them, and past it as
    /// one decay by `lam^count`.
    pub(crate) fn age_by(lam: f64, count: u64, mut decay: impl FnMut(f64)) {
        if count <= EXACT_RUN {
            for _ in 0..count {
                decay(lam);
            }
        } else {
            decay(pow_n(lam, count));
        }
    }
}

/// A refit buffer's rows and runs, read: learned row `i`'s features, target,
/// raw weight and factors, and the history in order ([`RefitBuf`]).
pub(crate) struct Rows<'a> {
    buf: &'a [f64],
    ages: &'a [f64],
    nf: usize,
}

impl<'a> Rows<'a> {
    pub(crate) fn new(buf: &'a [f64], ages: &'a [f64], nf: usize) -> Self {
        Self { buf, ages, nf }
    }

    /// The learned rows.
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

    /// The history in the order the stream took it: each learned row after
    /// the ageing runs that came before it.
    pub(crate) fn events(&self) -> Vec<Event> {
        let mut out = Vec::with_capacity(self.len() + self.ages.len() / 4);
        let mut runs = self.ages.as_chunks::<4>().0.iter().peekable();
        for i in 0..self.len() {
            // A run's last number is the learned rows before it.
            while let Some(a) = runs.next_if(|a| a[3] as usize <= i) {
                out.push(Event::Age {
                    lam: a[0],
                    lam_s: a[1],
                    count: a[2] as u64,
                });
            }
            out.push(Event::Row(i));
        }
        for a in runs {
            out.push(Event::Age {
                lam: a[0],
                lam_s: a[1],
                count: a[2] as u64,
            });
        }
        out
    }

    /// Each learned row's raw weight decayed to the buffer's last row, the
    /// row the refit runs on, by the model's factors (`scale = false`) or
    /// the scale's (`true`): `v_i = w_i · Π_{after i} λ`.
    fn decayed(&self, scale: bool) -> Vec<f64> {
        let mut v = vec![0.0; self.len()];
        let mut age = 1.0;
        for e in self.events().into_iter().rev() {
            match e {
                Event::Age { lam, lam_s, count } => {
                    age *= pow_n(if scale { lam_s } else { lam }, count);
                }
                Event::Row(i) => {
                    v[i] = self.w(i) * age;
                    age *= if scale { self.lam_s(i) } else { self.lam(i) };
                }
            }
        }
        v
    }

    /// The learned rows' weights decayed by the model's clock: the fit's.
    pub(crate) fn decayed_weights(&self) -> Vec<f64> {
        self.decayed(false)
    }

    /// The learned rows' weights decayed by the scale's clock
    /// (`huber_scale_decay`): the scale's, as the stream's scale holds them.
    pub(crate) fn scale_weights(&self) -> Vec<f64> {
        self.decayed(true)
    }
}

/// `β'z` for the augmented row `z = [1, x]` with an intercept, `x` without.
pub(crate) fn fitted(beta: &[f64], x: &[f64], intercept: bool) -> f64 {
    crate::solve::dot_aug(beta, x, intercept)
}

/// The weighted median of `a` at weights `v`: the smallest value at which
/// the cumulative weight reaches half the total, over the entries of
/// positive weight. NaN when there is none. At an even count of equal
/// weights that is the lower of the two middles, where numpy and
/// statsmodels take their mean: the two differ by half the gap between the
/// middle residuals, a few in 10,000 of the scale at 200 rows of normal
/// noise.
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

/// The residual scale rebuilt over the buffer from the residuals
/// `resid(i)` of the learned rows, at their raw weights, aged on the
/// scale's clock (`λ_s`) as the stream aged it, an ageing run's rows one by
/// one ([`Event::age_by`]): the EW mean of `e²` (`σ²` and its weight) and,
/// under the MAD, a fresh sketch of `|e|`.
pub(crate) fn rebuild_scale(
    rows: &Rows<'_>,
    kind: HuberScale,
    resid: impl Fn(usize) -> f64,
) -> Rebuilt {
    let mut mad = mad_sketches(kind, 1).pop();
    let (mut sig2, mut wsig) = (0.0, 0.0);
    for e in rows.events() {
        match e {
            Event::Age { lam_s, count, .. } => Event::age_by(lam_s, count, |l| {
                wsig *= l;
                if let Some(q) = mad.as_mut() {
                    q.age(l);
                }
            }),
            Event::Row(i) => {
                let lam_s = rows.lam_s(i);
                wsig *= lam_s;
                if let Some(q) = mad.as_mut() {
                    q.age(lam_s);
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
        }
    }
    Rebuilt { sig2, wsig, mad }
}

/// Weighted least squares over the buffered rows at weights `wt`, in mean
/// form: with an intercept the centred Gram over the features, plus on its
/// diagonal `ridge` and each slope's `penalty` over the total weight, and the
/// intercept `ȳ − m'β`; without one the raw Gram plus the same. `penalty`
/// is a penalty's whole mass, `Σ_i c_i · l2_j`, which the mean form divides
/// by `Σ_i wt_i`. `None` without weight, or when the factor fails at every
/// jitter.
fn wls(
    rows: &Rows<'_>,
    wt: &[f64],
    intercept: bool,
    ridge: f64,
    penalty: &[f64],
) -> Option<Vec<f64>> {
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
        a[p * nf + p] += ridge + penalty[p] / total;
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
/// convergence or [`REFIT_PASSES`]. `l2` is each slope's penalty per row,
/// as the steps apply it, unweighted: so its mass is `l2_j · Σ_i c_i`, `c_i`
/// each learned row counted once and decayed as its weight is, and the
/// minimum is that of `Σ v_i ρ(r_i/s) + ½ Σ_j l2_j β_j² Σ_i c_i`. The fit,
/// and the passes taken (the last one at the cap means it did not
/// converge); `None` where no pass solves.
pub(crate) fn irls(
    rows: &Rows<'_>,
    intercept: bool,
    delta: f64,
    kind: HuberScale,
    ridge: f64,
    l2: &[f64],
) -> Option<(Vec<f64>, u32)> {
    let n = rows.len();
    let v = rows.decayed_weights();
    let vs = rows.scale_weights();
    let count: f64 = (0..n).map(|i| v[i] / rows.w(i)).sum();
    let penalty: Vec<f64> = l2.iter().map(|l| l * count).collect();
    let mut beta = wls(rows, &v, intercept, ridge, &penalty)?;
    let mut absr = vec![0.0; n];
    let mut u = vec![1.0; n];
    let mut wt = vec![0.0; n];
    let mut passes = 0;
    for _ in 0..REFIT_PASSES {
        passes += 1;
        for (i, a) in absr.iter_mut().enumerate() {
            *a = (rows.y(i) - fitted(&beta, rows.x(i), intercept)).abs();
        }
        huber_weights(&absr, scale_of(kind, &absr, &vs), delta, &mut u);
        for ((t, &vi), &ui) in wt.iter_mut().zip(&v).zip(&u) {
            *t = vi * ui;
        }
        let Some(next) = wls(rows, &wt, intercept, ridge, &penalty) else {
            break;
        };
        let done = converged(Some(&beta), &next);
        beta = next;
        if done {
            break;
        }
    }
    Some((beta, passes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weighted_median_is_the_halfway_point_of_the_weight() {
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0]), 2.0);
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[1.0, 0.0, 5.0]), 2.0);
        assert_eq!(weighted_median(&[3.0, 1.0, 2.0], &[4.0, 1.0, 1.0]), 3.0);
        // At an even count the lower middle, where numpy averages the two.
        assert_eq!(weighted_median(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4]), 2.0);
        assert!(weighted_median(&[1.0], &[0.0]).is_nan());
    }

    /// The buffer counts learned rows, ages only after the first, keeps
    /// repeated ageings as one run, and reads its history back in order:
    /// the weights decayed by the model's factors and by the scale's.
    #[test]
    fn a_buffer_keeps_learned_rows_and_runs_of_ageings() {
        let mut b = RefitBuf::new();
        b.age(0.5, 0.5, 2);
        assert_eq!(b.held(), 0, "nothing to age before a learned row");
        assert!(!b.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 2));
        for _ in 0..3 {
            b.age(0.5, 0.25, 2);
        }
        assert_eq!(b.runs(), 1, "three ageings at the same factors are one run");
        b.age(0.5, 0.5, 2);
        assert_eq!(b.runs(), 2, "another factor is another run");
        assert!(
            b.learn(&[3.0], 4.0, 2.0, 0.9, 0.8, 2),
            "the second fills it"
        );
        let (rows, ages) = (b.rows.clone(), b.ages.clone());
        let r = Rows::new(&rows, &ages, 1);
        assert_eq!(r.len(), 2);
        assert_eq!(
            r.events(),
            vec![
                Event::Row(0),
                Event::Age {
                    lam: 0.5,
                    lam_s: 0.25,
                    count: 3
                },
                Event::Age {
                    lam: 0.5,
                    lam_s: 0.5,
                    count: 1
                },
                Event::Row(1),
            ]
        );
        // v_0 = 1 · 0.5³ · 0.5 · 0.9, and by the scale's clock 1 · 0.25³ · 0.5 · 0.8.
        assert_eq!(r.decayed_weights(), vec![0.5f64.powi(4) * 0.9, 2.0]);
        assert_eq!(r.scale_weights(), vec![0.25f64.powi(3) * 0.5 * 0.8, 2.0]);
        assert_eq!(b.take(), (rows, ages));
        assert!(!b.open() && b.held() == 0);
        assert!(
            !b.learn(&[1.0], 1.0, 1.0, 1.0, 1.0, 2),
            "a done buffer takes nothing"
        );
    }

    /// A restored buffer must be one the configuration writes.
    #[test]
    fn a_buffer_that_was_not_written_so_does_not_fit() {
        let mut b = RefitBuf::new();
        b.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 5);
        b.age(0.5, 0.5, 5);
        assert!(b.fits(1, 5) && RefitBuf::new().fits(1, 5));
        assert!(!b.fits(1, 1), "as many learned as the count");
        assert!(!b.fits(2, 5), "rows of another width");
        let bad = |f: &dyn Fn(&mut RefitBuf)| {
            let mut c = b.clone();
            f(&mut c);
            !c.fits(1, 5)
        };
        assert!(bad(&|c| c.learned = 0), "rows held before any was learned");
        assert!(bad(&|c| c.rows[2] = 0.0), "a learned row of no weight");
        assert!(bad(&|c| c.rows[3] = 1.5), "a factor above 1");
        assert!(bad(&|c| c.ages[1] = 0.0), "a run's scale factor of 0");
        assert!(bad(&|c| c.ages[2] = 1.5), "a run of no whole count");
        assert!(bad(&|c| c.ages[2] = 0.0), "a run of no rows");
        assert!(bad(&|c| c.ages[3] = 2.0), "a run after a row not learned");
        assert!(bad(&|c| c.passes = 3), "passes before the refit");
        assert!(bad(&|c| c.failed = true), "a failure before the refit");
        assert!(bad(&|c| c.abandoned = true), "given up and still open");
        let mut done = b.clone();
        done.take();
        assert!(done.fits(1, 5));
        assert!(bad(&|c| c.done = true), "done and still holding rows");
    }

    /// Past [`REFIT_RUNS_PER_ROW`] runs a row of its count, a target's refit
    /// is given up: the buffer freed, done, and marked.
    #[test]
    fn a_buffer_past_its_cap_of_runs_gives_up() {
        let mut b = RefitBuf::new();
        b.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 2);
        for i in 0..8 {
            b.age(0.5 + 0.01 * f64::from(i), 0.5, 2);
        }
        assert!(b.open() && b.runs() == 8, "the cap, 4 runs a row of 2");
        b.age(0.4, 0.5, 2);
        assert!(!b.open() && b.status().abandoned && b.held() == 0);
        assert!(!b.learn(&[1.0], 1.0, 1.0, 1.0, 1.0, 2));
        // A regular clock's ageings are one run, however many.
        let mut r = RefitBuf::new();
        r.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 2);
        for _ in 0..10_000 {
            r.age(0.5, 0.5, 2);
        }
        assert!(r.open() && r.runs() == 1);
    }

    #[test]
    fn a_power_by_squaring_is_the_product() {
        assert_eq!(pow_n(0.5, 10), 0.5f64.powi(10));
        assert_eq!(pow_n(0.3, 1), 0.3);
        assert_eq!(pow_n(0.3, 0), 1.0);
        assert_eq!(pow_n(0.9, 3), 0.9 * 0.9 * 0.9);
        let mut seen = Vec::new();
        Event::age_by(0.5, 3, |l| seen.push(l));
        assert_eq!(seen, vec![0.5; 3], "a short run decay by decay");
        seen.clear();
        Event::age_by(0.999, EXACT_RUN + 1, |l| seen.push(l));
        assert_eq!(
            seen,
            vec![pow_n(0.999, EXACT_RUN + 1)],
            "a long one at once"
        );
    }

    /// The rebuilt scale is the stream's own recursion over the rows, at
    /// their raw weights and aged on the scale's clock row by row: `σ²` the
    /// EW mean of `e²` and its weight, and under the MAD a sketch fed the
    /// same residuals. Three rows -- learned, two ageings, learned -- worked
    /// by hand: `W` = 1 · 0.5 · 0.5 · 0.5 + 2 = 2.125, and `σ²` = (0.125 · 4
    /// + 2 · 1) / 2.125.
    #[test]
    fn the_rebuilt_scale_is_the_streams_recursion() {
        let mut b = RefitBuf::new();
        b.learn(&[1.0], 10.0, 1.0, 0.9, 0.5, 9);
        b.age(0.9, 0.5, 9);
        b.age(0.9, 0.5, 9);
        b.learn(&[2.0], 20.0, 2.0, 0.9, 0.5, 9);
        let (rows, ages) = (b.rows.clone(), b.ages.clone());
        let r = Rows::new(&rows, &ages, 1);
        let resid = |i: usize| if i == 0 { 2.0 } else { -1.0 };
        let rms = rebuild_scale(&r, HuberScale::Rms, resid);
        assert_eq!(rms.wsig, 2.125);
        assert_eq!(rms.sig2, (0.125 * 4.0 + 2.0 * 1.0) / 2.125);
        assert!(rms.mad.is_none());
        let mad = rebuild_scale(&r, HuberScale::Mad, resid);
        assert_eq!((mad.wsig, mad.sig2), (rms.wsig, rms.sig2));
        let q = mad.mad.expect("a sketch under the MAD");
        // |e| of 2 at weight 0.125 and 1 at weight 2: the median is 1.
        assert!((q.get(0).unwrap() - 1.0).abs() < 0.01, "{:?}", q.get(0));
        // A residual that is not finite leaves the scale as it was.
        let inf = rebuild_scale(&r, HuberScale::Rms, |i| {
            if i == 0 { 2.0 } else { f64::INFINITY }
        });
        assert_eq!((inf.wsig, inf.sig2), (0.125, 4.0));
    }

    fn two_rows() -> (Vec<f64>, Vec<f64>) {
        let mut b = RefitBuf::new();
        b.learn(&[0.0], 1.0, 1.0, 1.0, 1.0, 9);
        b.age(1.0, 1.0, 9);
        b.learn(&[1.0], 3.0, 1.0, 1.0, 1.0, 9);
        (b.rows, b.ages)
    }

    /// Weighted least squares on two rows through `y = 1 + 2x`: the line,
    /// with and without an intercept's slot; a penalty's mass divided by
    /// the total weight; and no fit without weight, whose means would be
    /// `0/0`.
    #[test]
    fn weighted_least_squares_takes_a_penalty_and_needs_weight() {
        let (rows, ages) = two_rows();
        let r = Rows::new(&rows, &ages, 1);
        let fit = wls(&r, &[1.0, 3.0], true, 0.0, &[0.0]).unwrap();
        assert!(
            (fit[0] - 1.0).abs() < 1e-12 && (fit[1] - 2.0).abs() < 1e-12,
            "{fit:?}"
        );
        let origin = wls(&r, &[1.0, 1.0], false, 0.0, &[0.0]).unwrap();
        assert!((origin[0] - 3.0).abs() < 1e-12, "{origin:?}");
        // Centred at equal weights the variance is 0.25 and the covariance
        // 0.5: a mass of 0.5 over a weight of 2 doubles the variance.
        let shrunk = wls(&r, &[1.0, 1.0], true, 0.0, &[0.5]).unwrap();
        assert!(
            (shrunk[1] - 1.0).abs() < 1e-12 && (shrunk[0] - 1.5).abs() < 1e-12,
            "{shrunk:?}"
        );
        assert_eq!(wls(&r, &[0.0, 0.0], true, 0.0, &[0.0]), None);
        assert_eq!(wls(&r, &[0.0, 0.0], false, 1e-6, &[0.0]), None);
        assert_eq!(wls(&r, &[f64::INFINITY, 1.0], true, 0.0, &[0.0]), None);
    }

    /// statsmodels' `RLM(y, [1, x], M=HuberT(1.345)).fit(scale_est="mad",
    /// conv="coefs", tol=1e-14)` on 31 fixed rows, its digits written here
    /// so the comparison runs where `cargo mutants` sees it: 3.1320549757807723
    /// and 2.3494099687612335, in 38 iterations. The rows: `x_i = (37 i mod
    /// 31)/31 − 0.5`, `y_i = 3 + 2 x_i + ((17 i mod 13) − 6)/10`, and 5 more
    /// on every row `i` with `i mod 7 = 3`. At unit weights, an odd count and
    /// no ridge the refit's iteration is statsmodels' loop.
    #[test]
    fn the_refit_is_statsmodels_converged_fit() {
        let mut b = RefitBuf::new();
        for i in 0..31u32 {
            let x = f64::from((i * 37) % 31) / 31.0 - 0.5;
            let noise = (f64::from((i * 17) % 13) - 6.0) / 10.0;
            let y = 3.0 + 2.0 * x + noise + if i % 7 == 3 { 5.0 } else { 0.0 };
            b.learn(&[x], y, 1.0, 1.0, 1.0, 99);
        }
        let (rows, ages) = b.take();
        let r = Rows::new(&rows, &ages, 1);
        let (fit, passes) = irls(&r, true, 1.345, HuberScale::Mad, 0.0, &[0.0]).unwrap();
        assert!((fit[0] - 3.132_054_975_780_772_3).abs() < 1e-9, "{fit:?}");
        assert!((fit[1] - 2.349_409_968_761_233_5).abs() < 1e-9, "{fit:?}");
        assert!(passes > 2 && passes < REFIT_PASSES as u32, "{passes}");
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
