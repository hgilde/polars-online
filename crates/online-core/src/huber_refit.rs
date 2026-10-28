//! The Huber refit: each target's first rows refit to convergence, once,
//! for `huber` ([`crate::Robust`]) and `sgd(loss = "huber")` ([`crate::Sgd`])
//! (docs/HUBER.md §3; docs/PLAN.md task 277). `quantile`'s refit, the
//! smoothed check loss's in place of IRLS, reads its buffers and rebuilds
//! its scale here too (`robust::quantile_refit`, task 278, D-Q2).
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
//! until no buffered row's fitted value `z_i'β` moves by more than
//! [`REFIT_TOL`] of the pass's scale `s` (past the rounding of the fitted
//! values, [`REFIT_ROUNDING`]), or [`REFIT_PASSES`] passes ([`converged`]
//! says why the rule is in those units). That is statsmodels' `RLM` loop with the
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

/// The most IRLS passes a refit takes. A refit that takes them all has run
/// to its cap ([`RefitStatus::ran_to_cap`]) and is counted in the model's
/// `solve_failures`, as `kmeans`' and `hmm`'s seeding counts a Lloyd run
/// stopped at its cap (docs/PLAN.md task 278). Measured over 3,000 streams
/// of 50-500 rows, `y = 1 + 2x + 0.5e` with a level added to up to 45% of
/// the rows with `x > 0` (the tests' `slow_refit_stream`), `sgd`'s IRLS
/// at a cut of 1.345 without decay: 50 passes ran out on 149 (none below
/// 20% of the rows), 200 on 9 (all past 39%, beyond the breakdown); every
/// one converged on its own, the slowest in 897 passes (`huber`'s own: 127,
/// 8 and 428). Under the stop relative to `max|β|` that [`converged`]
/// replaced it was 142, 6 and 880 (task 278's follow-ups, 2026-10-26). A pass is one weighted solve of the buffer, so the cap costs only
/// where a refit does not converge: a spec with fewer rows than
/// coefficients, whose refit is the ridge's fit and never meets the rule,
/// pays 200 passes once a target (8.7 s at 1,000 features and 200 rows,
/// where 50 took 1.9 s).
pub const REFIT_PASSES: usize = 200;

/// A refit has converged when no buffered row's fitted value moved by more
/// than this, relative to the pass's residual scale ([`converged`]).
pub const REFIT_TOL: f64 = 1e-10;

/// The move a converged pass may still make in a fitted value by rounding
/// alone: this times the fitted value's terms `Σ_j |z_ij β_j|` at its
/// largest ([`converged`]), a few hundred units in the last place of the
/// fitted values' level.
pub const REFIT_ROUNDING: f64 = 256.0 * f64::EPSILON;

/// The least of a spec's default `huber_refit_rows` ([`default_refit_rows`]).
/// Measured at 20% one-sided outliers over 40 seeds: 200 rows left 18 of 40
/// streams of 2,000 rows more than 0.05 in slope from statsmodels' `RLM`,
/// 500 left 10 and 1,000 left 3 (the user's choice, 2026-10-25: "500").
pub const DEFAULT_REFIT_ROWS: u32 = 500;

/// The most features a spec refits its first rows for by default: past it a
/// refit of `10 · features` rows runs for seconds a target, and of fewer
/// rows than coefficients is the ridge's fit, not the robust one.
pub const DEFAULT_REFIT_MAX_FEATURES: usize = 100;

/// A spec's default `huber_refit_rows` for `n_features` features:
/// `max(500, 10 · n_features)` up to [`DEFAULT_REFIT_MAX_FEATURES`], and 0
/// -- off -- past it (an explicit count is always taken).
pub fn default_refit_rows(n_features: usize) -> u32 {
    if n_features > DEFAULT_REFIT_MAX_FEATURES {
        return 0;
    }
    DEFAULT_REFIT_ROWS.max(10 * n_features as u32)
}

/// The rows a bin's refit wants per bin of its widest feature
/// ([`default_bin_refit_rows`]; `marginal(bin_stat = "huber")`,
/// docs/PLAN.md task 278). A bin whose rows at the refit are mostly wild
/// has its M-location among them, rightly for those rows, and the stream's
/// next rows, clipped at the cut, bring it back only slowly. Measured with
/// spikes of 20 scales on 20% of the rows, no decay, 50 seeds, statsmodels'
/// M-location per bin as the oracle: at 12.5 rows a bin (16 bins, 200
/// rows) 6 streams kept a bin more than half a scale off at 20,000 rows,
/// at 15.6 (32 bins, 500 rows) 1, and at 31 (16 bins at 500, 32 at 1,000)
/// none, the worst 0.07.
pub const BIN_REFIT_ROWS_PER_BIN: usize = 30;

/// A `marginal(bin_stat = "huber")` spec's default `bin_refit_rows` for
/// features of at most `bins` bins: `max(500, 30 · bins)`, 500 at 16 bins.
pub fn default_bin_refit_rows(bins: usize) -> u32 {
    let per = bins.saturating_mul(BIN_REFIT_ROWS_PER_BIN);
    DEFAULT_REFIT_ROWS.max(u32::try_from(per).unwrap_or(u32::MAX))
}

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
    /// The learned rows it holds while it waits. Once it ran, `huber`'s and
    /// `sgd`'s refits keep that count, and `marginal`'s bins report 0, their
    /// rows freed.
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
    /// Given up before it began, past its spec's `refit_budget`
    /// ([`refit_reservation`]).
    pub over_budget: bool,
}

impl RefitStatus {
    /// Whether the refit took every one of its [`REFIT_PASSES`] passes: the
    /// rule unmet before the last, as scikit-learn reads an iteration count
    /// at its `max_iter`. Counted in the model's `solve_failures`.
    pub fn ran_to_cap(&self) -> bool {
        self.passes as usize >= REFIT_PASSES
    }
}

/// The bytes one target's refit buffer can hold at most: `refit_rows`
/// learned rows of `n_features + 4` numbers, and [`REFIT_RUNS_PER_ROW`]
/// ageing runs of 4 numbers a learned row, 8 bytes each -- what a bank
/// reserves against a spec's `refit_budget` for it, its contents, not the
/// allocator's slack. 0 when the refit is off.
pub fn refit_reservation(n_features: usize, refit_rows: u32) -> usize {
    let n = refit_rows as usize;
    8 * n * (n_features + 4 + 4 * REFIT_RUNS_PER_ROW as usize)
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
    over_budget: bool,
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
            over_budget: false,
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
            over_budget: self.over_budget,
        }
    }

    /// Give the refit up before it begins, past the spec's `refit_budget`:
    /// done, empty, and marked. A buffer already done is left as it is.
    pub(crate) fn give_up_over_budget(&mut self) {
        if !self.done {
            self.done = true;
            self.over_budget = true;
            self.rows = Vec::new();
            self.ages = Vec::new();
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
        if self.rows.is_empty() {
            // Held at its full size from the first row, so it never grows
            // past what a bank reserves for it ([`refit_reservation`]).
            self.rows.reserve_exact(cap as usize * (x.len() + 4));
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
        rows_fit
            && ages_fit
            && !self.abandoned
            && !self.failed
            && !self.over_budget
            && self.passes == 0
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

/// Huber's M-location of `y` at weights `v` and a fixed cut: the root of
/// `g(m) = Σ v_i ψ_c(y_i − m)`, `ψ_c(e) = max(−c, min(c, e))`, found
/// exactly rather than iterated ([`crate::Marginal`]'s bins, docs/PLAN.md
/// task 278). `y` is sorted ascending. `g` is continuous, non-increasing and
/// linear between the breakpoints `y_i ± c`, so a binary search over the
/// merged breakpoints finds the segment where it crosses a level, and the
/// root is on that segment's line `c·(C₊ − C₋) + Σ_in v_i y_i − Σ_in v_i ·
/// m`, with `C₊`, `C₋` the weight above and below the band and the sums over
/// the rows inside it. `g` at each breakpoint and the segment's sums are
/// formed afresh, every term of `g` clipped to the cut, so rows far beyond
/// the band leave no rounding of their own size in them; and the values are
/// taken from the middle row, so a target at `10⁶` with a spread of `10⁻²`
/// keeps its digits.
///
/// The roots are where `g` crosses `+τ` and `−τ`, `τ` a rounding's worth
/// of `c·Σv`: the root itself, to that rounding, where `g` falls through 0,
/// and an interval where `g` is 0 along a segment -- a bin of two rows
/// further apart than the band, one of them wild, or two clusters. Every
/// point of that interval fits the rows alike, and the one nearest
/// `toward` is taken: for a bin, its target's own location, as the stream
/// judges a bin's first row against it. So a bin of a clean row and a wild
/// one sits a cut from the clean row, where the middle of the interval would
/// sit halfway to the wild one, and the stream's next rows, clipped at the
/// cut, move it back only slowly. A cut that is not finite is the weighted
/// mean. `None` without a row of positive weight.
pub(crate) fn huber_location(y: &[f64], v: &[f64], cut: f64, toward: f64) -> Option<f64> {
    let rows: Vec<usize> = (0..y.len()).filter(|&i| v[i] > 0.0).collect();
    let mid = *rows.get(rows.len() / 2)?;
    let at = y[mid];
    let d = |i: usize| y[i] - at;
    let total: f64 = rows.iter().map(|&i| v[i]).sum();
    if !cut.is_finite() {
        let s: f64 = rows.iter().map(|&i| v[i] * d(i)).sum();
        return Some(at + s / total);
    }
    // `m` with `g(m) = level` on the line of the segment whose rows inside
    // the band are those of `guess`, the sums formed afresh; `None` where
    // no row is inside, `g` flat there.
    let line = |guess: f64, level: f64| -> Option<f64> {
        let (mut a, mut b, mut net) = (0.0, 0.0, 0.0);
        for &i in &rows {
            let e = d(i) - guess;
            if e > cut {
                net += v[i];
            } else if e < -cut {
                net -= v[i];
            } else {
                a += v[i];
                b += v[i] * d(i);
            }
        }
        (a > 0.0).then(|| (cut * net + b - level) / a)
    };
    // The breakpoints, ascending: row `i` enters the band at `d_i − c` and
    // leaves it at `d_i + c`, both ascending in `i`.
    let n = rows.len();
    let mut points = Vec::with_capacity(2 * n);
    let (mut enter, mut leave) = (0usize, 0usize);
    while leave < n {
        let next_out = d(rows[leave]) + cut;
        match (enter < n).then(|| d(rows[enter]) - cut) {
            Some(e) if e <= next_out => {
                points.push(e);
                enter += 1;
            }
            _ => {
                points.push(next_out);
                leave += 1;
            }
        }
    }
    // `g` at a breakpoint, read afresh, each term clipped to the cut, so
    // its rounding is the cut's however far the rows lie: sums carried up
    // the breakpoints through the entries and exits of rows `1e16` away
    // rounded at a few units, where `g` is `c` a row, and read the root
    // `5e16` from a cluster whose band was `1e-3` wide (task 281).
    let g = |m: f64| -> f64 {
        rows.iter()
            .map(|&i| v[i] * (d(i) - m).clamp(-cut, cut))
            .sum()
    };
    // The least `m` with `g(m) <= level`: the first breakpoint at which `g`
    // is at most the level -- `g` does not rise, so a binary search -- and
    // on the segment before it the line through its rows. At the last
    // breakpoint `g` is `−c·Σv`, below any level here.
    let cross = |level: f64| -> f64 {
        let k = points.partition_point(|&x| g(x) > level);
        let Some(&x) = points.get(k) else {
            return points[points.len() - 1];
        };
        let prev = k.checked_sub(1).map_or(f64::NEG_INFINITY, |p| points[p]);
        let mid = if prev.is_finite() {
            0.5 * (prev + x)
        } else {
            x
        };
        // Flat along it, `g` above the level at its lower end and not at its
        // upper: a band narrower than the rounding of its row, `d_i ± c` one
        // double, where `g` steps down at an end -- at `prev` if the segment
        // is at most the level, at `x` if not.
        line(mid, level).map_or_else(
            || {
                if prev.is_finite() && g(mid) <= level {
                    prev
                } else {
                    x
                }
            },
            |m| m.clamp(prev, x),
        )
    };
    let tau = 64.0 * f64::EPSILON * cut * total;
    let (lo, hi) = (cross(tau), cross(-tau));
    let near = toward - at;
    Some(
        at + if near.is_nan() {
            0.5 * (lo + hi)
        } else {
            near.clamp(lo, hi.max(lo))
        },
    )
}

/// The target's own Huber M-location and MAD scale, refit jointly as
/// statsmodels' `RLM` with a constant column and `scale_est = "mad"` fits
/// them: from the weighted mean, each pass reads the scale
/// `s = wmedian(|y − m|; vs) / Φ⁻¹(3/4)` and moves `m` to the M-location at
/// the cut `δs` ([`huber_location`]), until `m` moves by no more than
/// [`REFIT_TOL`] of `s`, past the rounding of `m` ([`REFIT_ROUNDING`] of
/// `|m|`; [`converged`]'s rule for one coefficient), or [`REFIT_PASSES`]
/// passes. Each pass
/// solves the location exactly where `RLM` takes one reweighted step, so
/// the fixed point is `RLM`'s and is reached in fewer passes. `y` sorted
/// ascending; `v` the fit's weights and `vs` the scale's. Where a pass's
/// roots are an interval, the one nearest the last pass's location. The
/// location and the passes; `None` without a row of positive weight.
pub(crate) fn location_refit(y: &[f64], v: &[f64], vs: &[f64], delta: f64) -> Option<(f64, u32)> {
    let mut m = huber_location(y, v, f64::INFINITY, f64::NAN)?;
    let mut absr = vec![0.0; y.len()];
    let mut passes = 0u32;
    for _ in 0..REFIT_PASSES {
        passes += 1;
        for (a, &yi) in absr.iter_mut().zip(y) {
            *a = (yi - m).abs();
        }
        let Some(s) = scale_of(HuberScale::Mad, &absr, vs) else {
            // No spread to cut in: every weight 1, the mean, as the stream.
            break;
        };
        let next = huber_location(y, v, delta * s, m)?;
        let moved = (next - m).abs();
        m = next;
        if moved <= REFIT_TOL * s + REFIT_ROUNDING * m.abs() {
            break;
        }
    }
    Some((m, passes))
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

/// Whether a pass from `before` to `after` has converged: no buffered row's
/// fitted value moved by more than [`REFIT_TOL`] of the pass's residual
/// scale `s`, past the rounding of the fitted values,
///
/// ```text
/// max_i |z_i'(β' − β)|  ≤  REFIT_TOL · s  +  REFIT_ROUNDING · max_i Σ_j |z_ij β'_j|
/// ```
///
/// That is each coefficient against its own scale, in standardized units:
/// a slope moving by `Δβ_j` moves the fitted values by `Δβ_j` times its
/// feature's spread, and the rule holds that to a share of the residual
/// scale, the precision the rows give the fit; the intercept is held to the
/// same share of `s` directly. Nothing in it reads the target's level, so a
/// level added to every target stops the slopes where it would without it.
/// A rule relative to `max|β|` read the level through the intercept: at a
/// level of `1e3` a slope stopped once it moved by under `1e-7`, and its
/// error grew with the level. Nor does it read a coefficient that is
/// poorly determined: two near-collinear features trade weight along a
/// direction the fitted values do not see, and the rule waits only for the
/// fit. The second term is what rounding alone can leave moving: a fitted
/// value's terms are summed in floating point, and a few hundred units in
/// the last place of their size is a move no further pass removes. Without
/// a scale (`None`, every residual 0) the rule is that term alone; a pass
/// that moved nothing has converged. Every term scales with the target, so
/// a target scaled by a power of two takes the same passes and its fit is
/// the same fit scaled, to the bit.
pub(crate) fn converged(
    rows: &Rows<'_>,
    intercept: bool,
    before: Option<&[f64]>,
    after: &[f64],
    scale: Option<f64>,
) -> bool {
    let Some(before) = before else {
        return false;
    };
    let step: Vec<f64> = after.iter().zip(before).map(|(q, p)| q - p).collect();
    let (mut moved, mut size) = (0.0_f64, 0.0_f64);
    for i in 0..rows.len() {
        let x = rows.x(i);
        moved = moved.max(fitted(&step, x, intercept).abs());
        let (head, slopes) = if intercept {
            (after[0].abs(), &after[1..])
        } else {
            (0.0, after)
        };
        let terms = slopes
            .iter()
            .zip(x)
            .fold(head, |acc, (b, xi)| acc + (b * xi).abs());
        size = size.max(terms);
    }
    moved <= REFIT_TOL * scale.unwrap_or(0.0) + REFIT_ROUNDING * size
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
///
/// An anchored sketch is anchored at the residuals' own weighted median
/// size at the scale's weights, the MAD's numerator, not at the first
/// residual: a sketch's reads are its anchor times a bucket's
/// representative, so they carry the anchor's relative rounding, and a
/// residual is rounded at the target's level, not at its own size. A first
/// residual of `2.3e-4` at a level of `1e6` was `2e-7` off, and so was
/// every band and cut drawn from the sketch after the refit, where the
/// stream's rounding is `1e-10` (task 278, qequiv). The median is of the
/// residuals' size, so its rounding is theirs; it scales with the
/// residuals, to the bit by a power of two and to rounding by any other
/// factor, as the first residual does. Without a median above 0 (half the
/// weight on residuals of 0) the first residual anchors it, as on the
/// stream.
pub(crate) fn rebuild_scale(
    rows: &Rows<'_>,
    kind: HuberScale,
    anchored: bool,
    resid: impl Fn(usize) -> f64,
) -> Rebuilt {
    let mut mad = mad_sketches(kind, anchored, 1).pop();
    if let Some(q) = mad.as_mut().filter(|q| q.anchor().is_some()) {
        let absr: Vec<f64> = (0..rows.len()).map(|i| resid(i).abs()).collect();
        q.anchor_at(weighted_median(&absr, &rows.scale_weights()));
    }
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
        let scale = scale_of(kind, &absr, &vs);
        huber_weights(&absr, scale, delta, &mut u);
        for ((t, &vi), &ui) in wt.iter_mut().zip(&v).zip(&u) {
            *t = vi * ui;
        }
        let Some(next) = wls(rows, &wt, intercept, ridge, &penalty) else {
            break;
        };
        let done = converged(rows, intercept, Some(&beta), &next, scale);
        beta = next;
        if done {
            break;
        }
    }
    Some((beta, passes))
}

/// A stream whose refit converges slowly (task 278): `y = 1 + 2x + 0.5e`
/// with a level added to a share of the rows with `x > 0`, its length (50
/// to 500 rows), share (up to 45%) and level drawn from `trial`. Trial 687
/// is 71 rows at 39% whose IRLS needs 880 passes; the probe that found it
/// drew 3,000 trials.
#[cfg(test)]
pub(crate) fn slow_refit_stream(trial: u64) -> Vec<(f64, f64)> {
    let mut state: u64 = (trial + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 50 + (rnd() * 450.0) as usize;
    let frac = rnd() * 0.45;
    let level = (rnd() * 6.0 - 1.0).exp() * 3.0;
    (0..n)
        .map(|_| {
            let x = rnd() * 2.0 - 1.0;
            let e = (-2.0 * rnd().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * rnd()).cos();
            let out = if rnd() < frac && x > 0.0 { level } else { 0.0 };
            (x, 1.0 + 2.0 * x + 0.5 * e + out)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default count: 500 rows, ten a feature past 50 features, none
    /// above 100. And the reservation: the rows and their four runs each,
    /// `8 * rows * (features + 20)` bytes.
    #[test]
    fn the_default_count_scales_with_the_features_and_stops_past_100() {
        assert_eq!(default_refit_rows(0), 500);
        assert_eq!(default_refit_rows(1), 500);
        assert_eq!(default_refit_rows(50), 500);
        assert_eq!(default_refit_rows(51), 510);
        assert_eq!(default_refit_rows(99), 990);
        assert_eq!(default_refit_rows(100), 1000);
        assert_eq!(default_refit_rows(101), 0);
        assert_eq!(refit_reservation(20, 500), 160_000);
        assert_eq!(refit_reservation(1, 50), 8 * 50 * 21);
        assert_eq!(refit_reservation(3, 0), 0);
        // The bins': 30 rows a bin of the widest feature, at least 500.
        assert_eq!(default_bin_refit_rows(1), 500);
        assert_eq!(default_bin_refit_rows(16), 500);
        assert_eq!(default_bin_refit_rows(17), 510);
        assert_eq!(default_bin_refit_rows(64), 1920);
        assert_eq!(default_bin_refit_rows(usize::MAX), u32::MAX);
    }

    /// A buffer is held at its full size from its first row, so it never
    /// grows past the reservation; given up to the budget it is done,
    /// empty and marked, and one already done is left as it was; a refit
    /// that no pass solved is marked failed.
    #[test]
    fn a_buffer_is_sized_once_given_up_whole_and_marked_when_it_fails() {
        let mut b = RefitBuf::new();
        assert!(!b.learn(&[1.0, 2.0, 3.0], 4.0, 1.0, 1.0, 1.0, 50));
        assert!(b.rows.capacity() >= 50 * 7, "{}", b.rows.capacity());
        let mut one = RefitBuf::new();
        one.learn(&[1.0], 2.0, 1.0, 1.0, 1.0, 50);
        assert!(one.rows.capacity() >= 50 * 5, "{}", one.rows.capacity());
        b.age(0.5, 0.5, 50);
        assert!(b.held() > 0 && b.open());
        b.give_up_over_budget();
        let s = b.status();
        assert!(
            s.done && s.over_budget && !s.abandoned && !s.failed,
            "{s:?}"
        );
        assert_eq!((b.held(), b.rows.capacity(), b.ages.capacity()), (0, 0, 0));
        assert!(!b.open());
        assert!(!b.learn(&[1.0, 2.0, 3.0], 4.0, 1.0, 1.0, 1.0, 50));
        assert_eq!(b.status().learned, 1);

        let mut done = RefitBuf::new();
        assert!(done.learn(&[1.0], 2.0, 1.0, 1.0, 1.0, 1));
        done.done = true;
        done.give_up_over_budget();
        assert!(!done.status().over_budget);
        assert_eq!(done.held(), 5, "a buffer already done keeps what it held");

        let mut failed = RefitBuf::new();
        assert!(!failed.status().failed);
        failed.fail();
        assert!(failed.status().failed);
    }

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
        assert!(
            bad(&|c| c.over_budget = true),
            "over the budget and still open"
        );
        assert!(
            bad(&|c| c.rows[2] = f64::INFINITY),
            "a learned row of infinite weight"
        );
        // At most four runs a row of the count: eight at a count of 2.
        let mut runs = RefitBuf::new();
        runs.learn(&[1.0], 2.0, 1.0, 0.9, 0.8, 2);
        runs.ages = [0.5, 0.5, 1.0, 1.0].repeat(8);
        assert!(runs.fits(1, 2), "four runs a row of the count");
        runs.ages = [0.5, 0.5, 1.0, 1.0].repeat(9);
        assert!(!runs.fits(1, 2), "past four runs a row of the count");
        assert!(
            bad(&|c| {
                c.done = true;
                c.ages.clear();
            }),
            "done and still holding learned rows"
        );
        assert!(
            bad(&|c| {
                c.done = true;
                c.rows.clear();
            }),
            "done and still holding runs"
        );
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
        let rms = rebuild_scale(&r, HuberScale::Rms, false, resid);
        assert_eq!(rms.wsig, 2.125);
        assert_eq!(rms.sig2, (0.125 * 4.0 + 2.0 * 1.0) / 2.125);
        assert!(rms.mad.is_none());
        let mad = rebuild_scale(&r, HuberScale::Mad, false, resid);
        assert_eq!((mad.wsig, mad.sig2), (rms.wsig, rms.sig2));
        let q = mad.mad.expect("a sketch under the MAD");
        // |e| of 2 at weight 0.125 and 1 at weight 2: the median is 1.
        assert!((q.get(0).unwrap() - 1.0).abs() < 0.01, "{:?}", q.get(0));
        // A residual that is not finite leaves the scale as it was.
        let inf = rebuild_scale(&r, HuberScale::Rms, false, |i| {
            if i == 0 { 2.0 } else { f64::INFINITY }
        });
        assert_eq!((inf.wsig, inf.sig2), (0.125, 4.0));
    }

    /// A rebuilt anchored sketch reads to the rounding of the residuals'
    /// size, whatever the first residual's size (task 278, qequiv): each
    /// residual moved by `1e-12`, as a level of `1e4` rounds them, moves the
    /// read by about that over the median, `1e-12` -- where a sketch
    /// anchored at a first residual of `1e-6` moved by `1e-6`, a million
    /// times more, and every band drawn from it with it. A power of two
    /// still scales the read to the bit.
    #[test]
    fn a_rebuilt_sketch_reads_to_the_residuals_rounding() {
        let mut b = RefitBuf::new();
        let mut s = 7u64;
        let mut e = Vec::new();
        for i in 0..300 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            e.push(if i == 0 { 1e-6 } else { u });
            b.learn(&[0.0], 0.0, 1.0, 1.0, 1.0, 9);
        }
        let (rows, ages) = (b.rows.clone(), b.ages.clone());
        let r = Rows::new(&rows, &ages, 1);
        let read = |f: &dyn Fn(usize) -> f64| {
            let q = rebuild_scale(&r, HuberScale::Mad, true, f).mad.unwrap();
            (q.get(0).unwrap(), q.anchor().unwrap())
        };
        let (base, anchor) = read(&|i| e[i]);
        let (moved, _) = read(&|i| e[i] + if i % 2 == 0 { 1e-12 } else { -1e-12 });
        let rel = (moved - base).abs() / base;
        assert!(rel <= 1e-11, "{rel:e}");
        assert!(anchor > 0.1, "anchored at the median size: {anchor}");
        let (scaled, _) = read(&|i| e[i] * 2f64.powi(-30));
        assert_eq!(scaled.to_bits(), (base * 2f64.powi(-30)).to_bits());
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

    /// The 31 rows of [`the_refit_is_statsmodels_converged_fit`]'s noise
    /// and spikes, as a location sample, sorted: `((17 i mod 13) − 6)/10`,
    /// and 5 more on every row `i` with `i mod 7 = 3`.
    fn location_rows() -> Vec<f64> {
        let mut y: Vec<f64> = (0..31u32)
            .map(|i| (f64::from((i * 17) % 13) - 6.0) / 10.0 + if i % 7 == 3 { 5.0 } else { 0.0 })
            .collect();
        y.sort_by(f64::total_cmp);
        y
    }

    /// statsmodels' digits, written here so `cargo mutants` sees them:
    /// `RLM(y, 1, M=HuberT(1.345)).fit(start_scale=0.5, update_scale=False,
    /// conv="coefs", tol=1e-15)` gives 0.10302083333333342, the M-location at
    /// the fixed cut 1.345 · 0.5; and `.fit(scale_est="mad", ...)` gives
    /// 0.11402545645429082 at the scale 0.5722467145475555, the joint fixed
    /// point (27 iterations). An odd count, so the weighted median is
    /// numpy's median and the scale statsmodels' MAD.
    #[test]
    fn the_location_is_statsmodels_m_location() {
        let y = location_rows();
        let v = vec![1.0; y.len()];
        let m = huber_location(&y, &v, 1.345 * 0.5, f64::NAN).unwrap();
        assert!((m - 0.103_020_833_333_333_42).abs() < 1e-13, "{m}");
        let (joint, passes) = location_refit(&y, &v, &v, 1.345).unwrap();
        assert!((joint - 0.114_025_456_454_290_82).abs() < 1e-9, "{joint}");
        assert!(passes > 1 && passes < 27, "{passes}");
        // Each pass solves the location exactly: fewer passes than RLM's
        // steps. And the root is a root: g(m) = Σ ψ(y − m) is 0.
        let absr: Vec<f64> = y.iter().map(|yi| (yi - joint).abs()).collect();
        let s = scale_of(HuberScale::Mad, &absr, &v).unwrap();
        assert!((s - 0.572_246_714_547_555_5).abs() < 1e-9, "{s}");
    }

    /// The exact root against bisection on `g(m) = Σ v ψ_c(y − m)`, which
    /// shares no arithmetic with the sweep: over weighted samples with
    /// outliers, a far level, ties and rows of weight 0, the two agree to
    /// rounding of the level, and `g` at the root is 0 to rounding of the
    /// weight times the cut. A cut that is not finite is the weighted mean;
    /// no positive weight, no location.
    #[test]
    fn the_location_is_the_root_of_the_estimating_equation() {
        let g = |y: &[f64], v: &[f64], c: f64, m: f64| -> f64 {
            y.iter()
                .zip(v)
                .map(|(yi, vi)| vi * (yi - m).clamp(-c, c))
                .sum()
        };
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for trial in 0..300 {
            let n = 1 + trial % 40;
            let level = [0.0, 1e6, -3.0e3][trial % 3];
            let mut y: Vec<f64> = (0..n)
                .map(|_| {
                    let base = level + (next() - 0.5) * 2.0;
                    let r = next();
                    if r < 0.2 {
                        base + 50.0
                    } else if r < 0.3 {
                        level // a tie
                    } else {
                        base
                    }
                })
                .collect();
            y.sort_by(f64::total_cmp);
            let v: Vec<f64> = (0..n)
                .map(|_| if next() < 0.1 { 0.0 } else { 0.1 + next() })
                .collect();
            let c = [0.05, 0.7, 3.0, 1e3][trial % 4];
            let total: f64 = v.iter().sum();
            let toward = level + [0.0, 100.0, -100.0][trial % 3];
            let Some(m) = huber_location(&y, &v, c, toward) else {
                assert_eq!(total, 0.0, "trial {trial}");
                continue;
            };
            // Bisection over the sample's range.
            let (mut lo, mut hi) = (y[0] - 1.0, y[n - 1] + 1.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if g(&y, &v, c, mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let tol = 1e-12 * (1.0 + level.abs()) + 1e-12 * c;
            assert!(
                g(&y, &v, c, m).abs() <= 1e-10 * total * c.max(1.0) * (1.0 + level.abs() / 1e3),
                "trial {trial}: g({m}) = {}",
                g(&y, &v, c, m)
            );
            // Where g is flat at 0 over a segment every point is a root:
            // the bisection's lies in it, and so does ours.
            if (m - lo).abs() > tol + 1e-9 {
                assert!(g(&y, &v, c, lo).abs() <= 1e-9 * total * c, "trial {trial}");
            }
        }
        assert_eq!(huber_location(&[1.0, 2.0], &[0.0, 0.0], 1.0, 0.0), None);
        let mean = huber_location(&[1.0, 2.0, 10.0], &[1.0, 1.0, 2.0], f64::INFINITY, 0.0).unwrap();
        assert_eq!(mean, 23.0 / 4.0);
        // Two far clusters of equal weight: every point between their bands
        // is a root, and the one nearest `toward` is taken -- `toward`
        // itself inside the interval, its nearer end outside, the middle
        // without one; so for a bin of a clean row and a wild one.
        let y = [0.0, 0.0, 100.0, 100.0];
        assert_eq!(huber_location(&y, &[1.0; 4], 1.0, 30.0), Some(30.0));
        let low = huber_location(&y, &[1.0; 4], 1.0, -5.0).unwrap();
        assert!((low - 1.0).abs() < 1e-12, "{low}");
        assert_eq!(huber_location(&y, &[1.0; 4], 1.0, f64::NAN), Some(50.0));
        let two = huber_location(&[1.0, 51.0], &[1.0; 2], 3.1, 0.5).unwrap();
        assert!((two - 4.1).abs() < 1e-12, "{two}");
    }

    /// The stop reads each coefficient against the residual scale through
    /// the fitted values, not against `max|β|` (task 278's follow-ups): the
    /// same slope move is refused at a level of 0 and of `1e3` alike, where
    /// the old rule took `1e-8` at `1e3`; the intercept is held to the scale
    /// directly; a coefficient the rows do not see -- a feature that is 0 on
    /// every row -- may move freely; without a scale only rounding is left,
    /// and a level of `1e6` may move by a few hundred units in its last
    /// place.
    #[test]
    fn the_stop_reads_the_fit_against_the_scale_not_the_level() {
        let mut b = RefitBuf::new();
        for i in 0..21u32 {
            let x = f64::from(i) / 10.0 - 1.0;
            b.learn(&[x, 0.0], 0.0, 1.0, 1.0, 1.0, 99);
        }
        let (rows, ages) = b.take();
        let r = Rows::new(&rows, &ages, 2);
        let s = Some(0.5);
        for level in [0.0, 1e3, -1e3] {
            let before = [level + 1.0, 2.0, 0.0];
            let moved = |d: [f64; 3]| [before[0] + d[0], before[1] + d[1], before[2] + d[2]];
            // A slope move of 1e-8 is 2e-8 of the scale: refused at any level.
            assert!(
                !converged(&r, true, Some(&before), &moved([0.0, 1e-8, 0.0]), s),
                "{level}"
            );
            assert!(
                converged(&r, true, Some(&before), &moved([0.0, 1e-11, 0.0]), s),
                "{level}"
            );
            // The intercept against the scale: 1e-11 is 2e-11 of it.
            assert!(
                converged(&r, true, Some(&before), &moved([1e-11, 0.0, 0.0]), s),
                "{level}"
            );
            assert!(
                !converged(&r, true, Some(&before), &moved([1e-9, 0.0, 0.0]), s),
                "{level}"
            );
            // The second feature is 0 on every row: no fit sees its slope.
            assert!(
                converged(&r, true, Some(&before), &moved([0.0, 0.0, 5.0]), s),
                "{level}"
            );
        }
        // Without a scale a move must be rounding: 100 units in the last
        // place of a fit of 1e6 is, 1e-6 is not; nothing moved has converged.
        let far = [1e6, 2.0, 0.0];
        let ulps = |n: f64| [far[0] + n * 1e6 * f64::EPSILON, far[1], far[2]];
        assert!(converged(&r, true, Some(&far), &ulps(100.0), None));
        assert!(!converged(
            &r,
            true,
            Some(&far),
            &[far[0] + 1e-6, 2.0, 0.0],
            None
        ));
        assert!(converged(&r, true, Some(&far), &far, None));
        // The first pass has nothing to compare with; without an intercept
        // the slopes alone make the fit.
        assert!(!converged(&r, true, None, &far, s));
        assert!(!converged(
            &r,
            false,
            Some(&[2.0, 0.0]),
            &[2.0 + 1e-8, 0.0],
            s
        ));
        assert!(converged(
            &r,
            false,
            Some(&[2.0, 0.0]),
            &[2.0 + 1e-11, 0.0],
            s
        ));
        // The location's rule is the same for its one coefficient: at a
        // level of 1e3 the joint fixed point moves with the level, to
        // rounding, and takes the same passes.
        let y = location_rows();
        let v = vec![1.0; y.len()];
        let (m0, p0) = location_refit(&y, &v, &v, 1.345).unwrap();
        let shifted: Vec<f64> = y.iter().map(|yi| yi + 1e3).collect();
        let (m1, p1) = location_refit(&shifted, &v, &v, 1.345).unwrap();
        assert!((m1 - 1e3 - m0).abs() <= 1e-12, "{m0} against {m1}");
        assert_eq!(p0, p1);
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

    /// `g(m) = Σ v ψ_c(y − m)` evaluated afresh: the estimating equation the
    /// location solves, sharing no sum with the sweep.
    fn estimating(y: &[f64], v: &[f64], c: f64, m: f64) -> f64 {
        y.iter()
            .zip(v)
            .map(|(yi, vi)| vi * (yi - m).clamp(-c, c))
            .sum()
    }

    /// Bisection on `g` over `[lo, hi]`, 200 halvings: the root to the last
    /// bit where `g` falls through 0.
    fn bisect(y: &[f64], v: &[f64], c: f64, (mut lo, mut hi): (f64, f64)) -> f64 {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if estimating(y, v, c, mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The sweep carries its sums through additions and subtractions, and a
    /// far cluster that enters the band and leaves it below the root leaves
    /// a rounding of its own size behind: `1e9` a row is `1e-7` of residue.
    /// The segment's line is formed afresh (task 281, mutants), so the root
    /// is bisection's to rounding of the near rows, not of the far ones.
    #[test]
    fn a_far_cluster_leaves_no_rounding_in_the_location() {
        let mut y = vec![-1e9 + 0.1, -1e9 + 0.37, -1e9 + 0.71, -1e9 + 0.93];
        y.extend([0.05, 0.13, 0.29, 0.41, 0.47, 0.62, 0.83]);
        let v = [1.3, 0.7, 1.1, 0.9, 1.0, 0.6, 1.4, 0.8, 1.2, 1.0, 0.5];
        for c in [0.5, 1.0, 3.0] {
            let m = huber_location(&y, &v, c, f64::NAN).unwrap();
            let root = bisect(&y, &v, c, (-10.0, 10.0));
            assert!((m - root).abs() <= 1e-14, "cut {c}: {m} against {root}");
            assert!(estimating(&y, &v, c, m).abs() <= 1e-13, "cut {c}");
        }
    }

    /// The roots are where `g` crosses `±τ`, `τ = 64 ε c Σv`: with every row
    /// inside a cut of `1e12` the line is `g(m) = Σv (ȳ − m)`, so the least
    /// root is `ȳ − τ/Σv = ȳ − 64 ε c` and the greatest `ȳ + 64 ε c`, and a
    /// `toward` far below or above takes the one nearest it; at `Σv = 4`
    /// the interval is a weight's worth, not its square or its inverse.
    #[test]
    fn the_roots_are_where_g_crosses_a_rounding_of_its_scale() {
        let y = [0.25, 0.5, 1.0, 2.25];
        let v = [1.0; 4];
        let c = 1e12;
        let mean = 1.0;
        let half = 64.0 * f64::EPSILON * c;
        let low = huber_location(&y, &v, c, -1e6).unwrap();
        let high = huber_location(&y, &v, c, 1e6).unwrap();
        assert!((low - (mean - half)).abs() <= 1e-9 * half, "{low}");
        assert!((high - (mean + half)).abs() <= 1e-9 * half, "{high}");
        // The middle where `toward` is not a number, and `toward` itself
        // inside the interval.
        assert_eq!(
            huber_location(&y, &v, c, f64::NAN),
            Some(0.5 * (low + high))
        );
        assert_eq!(huber_location(&y, &v, c, 1.0 + 1e-3), Some(1.0 + 1e-3));
    }

    /// Rows `1e16` apart at a cut of `1e-3`, each band narrower than its
    /// row's rounding, `y_i ± c` one double, so `g` steps at each row: the
    /// root is the top row, which carries more than half the weight. Read
    /// from sums carried up the breakpoints it was `2.1e16` from it
    /// (task 281's search, which found this case), and an earlier form of
    /// the search read past its last breakpoint.
    #[test]
    fn bands_narrower_than_their_rows_rounding_step_g_at_each_row() {
        let y = [
            -1.822164633079547e16,
            -8166305640259041.0,
            2.439867804361081e16,
        ];
        let v = [0.835622537224729, 1.051623972756464, 2.280822064161749];
        let m = huber_location(&y, &v, 1e-3, f64::NAN).unwrap();
        assert!((m - y[2]).abs() <= 8.0, "{m}");
    }

    /// Rows `1e16` to `1e17` from a cluster whose band is `1e-3` wide: the
    /// sweep's sums, carried through those rows' entries and exits, round
    /// at a few units where `g` itself is `1e-3` a row, and the location
    /// they read sat `5e16` from the cluster (found by the search behind
    /// the test above, task 281). Each breakpoint's `g` is read afresh,
    /// each term clipped to the cut, so the root is bisection's.
    #[test]
    fn rows_far_beyond_the_cut_leave_the_root_in_the_cluster() {
        let mut y = vec![
            -1.0054330951384432e17,
            -3.0483978458765236e16,
            5.677624234228088e16,
        ];
        y.extend([
            0.00034, 0.00173, 0.00176, 0.00180, 0.00195, 0.00211, 0.00298, 0.00635, 0.00663,
            0.00856,
        ]);
        y.sort_by(f64::total_cmp);
        let v = [
            0.9, 1.1, 0.7, 1.3, 0.6, 1.0, 1.2, 0.8, 0.5, 1.4, 0.9, 1.1, 0.75,
        ];
        for c in [1e-3, 2e-3] {
            let m = huber_location(&y, &v, c, f64::NAN).unwrap();
            let root = bisect(&y, &v, c, (-1.0, 1.0));
            assert!((m - root).abs() <= 1e-12, "cut {c}: {m} against {root}");
        }
    }

    /// The joint location and scale scale with the rows, to the stop's
    /// tolerance of the scale: the rows at `1e-6`, at `1e6`, and moved so
    /// the fixed point sits at 0 (statsmodels' digits as in
    /// `the_location_is_statsmodels_m_location`), where the stop's rounding
    /// term reads `|m|` near 0 and must not loosen it.
    #[test]
    fn the_joint_location_scales_and_shifts_with_its_rows() {
        let at = 0.114_025_456_454_290_82;
        let y = location_rows();
        let v = vec![1.0; y.len()];
        for k in [1e-6, 1e6] {
            let scaled: Vec<f64> = y.iter().map(|yi| yi * k).collect();
            let (m, _) = location_refit(&scaled, &v, &v, 1.345).unwrap();
            assert!((m / k - at).abs() <= 1e-9, "{k}: {}", m / k);
        }
        let moved: Vec<f64> = y.iter().map(|yi| yi - at).collect();
        let (m, _) = location_refit(&moved, &v, &v, 1.345).unwrap();
        assert!(m.abs() <= 1e-9, "{m}");
    }

    /// Without an intercept the rounding term's size is the slopes' terms
    /// alone: a fit of `1e6` on a feature reaching `±1` may move by a few
    /// hundred units in its last place, and not by `1e-6`.
    #[test]
    fn the_stop_reads_the_slopes_terms_through_the_origin() {
        let mut b = RefitBuf::new();
        for i in 0..21u32 {
            b.learn(&[f64::from(i) / 10.0 - 1.0], 0.0, 1.0, 1.0, 1.0, 99);
        }
        let (rows, ages) = b.take();
        let r = Rows::new(&rows, &ages, 1);
        let far = [1e6];
        assert!(converged(
            &r,
            false,
            Some(&far),
            &[1e6 + 100.0 * 1e6 * f64::EPSILON],
            None
        ));
        assert!(!converged(&r, false, Some(&far), &[1e6 + 1e-6], None));
    }

    /// Weighted least squares at a level of `1e6` in the feature and twice
    /// that in the target, on rows of the line `y = 1 + 2x`: the centred
    /// sums cancel the level, so the slope is 2 to the rounding of the
    /// rows' values, where centring either side by its mean added rather
    /// than taken away leaves a term of the level times the rounding of
    /// `Σ w (x − x̄)`.
    #[test]
    fn weighted_least_squares_is_centred_at_a_level() {
        let mut b = RefitBuf::new();
        let xs = [1e6 + 0.1, 1e6 + 0.7, 1e6 + 0.3, 1e6 + 0.45];
        for &x in &xs {
            b.learn(&[x], 1.0 + 2.0 * x, 1.0, 1.0, 1.0, 9);
        }
        let (rows, ages) = b.take();
        let r = Rows::new(&rows, &ages, 1);
        let fit = wls(&r, &[1.0, 3.0, 0.7, 1.9], true, 0.0, &[0.0]).unwrap();
        assert!((fit[1] - 2.0).abs() <= 1e-8, "{fit:?}");
    }
}
