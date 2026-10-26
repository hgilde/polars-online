//! `quantile`'s refit of each target's first rows (docs/PLAN.md task 278,
//! D-Q2; [`RobustCfg::refit_rows`]), beside `huber`'s
//! ([`crate::huber_refit`]).
//!
//! **Why.** One pass takes each row as the fit before it sees it: as a
//! least-squares row in its warm-up (three rows a coefficient), and after
//! it as a row inside the band, a least-squares row in the sums, or outside
//! it, a nudge of the score alone. Without decay a row stays in the sums
//! as it was taken, at full weight, for good. So rows the first fits took
//! as least squares, or held inside a band drawn around a pulled fit --
//! wild targets in the first rows -- keep pulling, where the batch fit of
//! every row reads them as outliers, bounded by the band's width; they
//! leave only by dilution, as `1/n`. Measured over 50 seeds without decay,
//! three targets 1e4 off in the first rows left the fit 0.22 off
//! statsmodels' `QuantReg` on the same rows at 8,000 rows.
//!
//! **What.** On the row a target's learned rows reach the count, after the
//! row's own prediction and update, its buffered rows are refit to the
//! minimum of the smoothed check loss at the band the stream would draw:
//!
//! ```text
//! v_i  = w_i · Π_{l > i} λ_l                 the row's weight, decayed to now
//! r_i  = y_i − z_i'β                         against the pass's fit
//! s    = wmedian(|r|; v) / Φ⁻¹(3/4)          the MAD, as the stream's scale
//! h    = s · max(quantile_eps, (k/n)^(2/5))  n the rows counted one each, decayed
//! F(β) = Σ v_i ρ_h(y_i − z_i'β) + ½ · (W_h λ / 2h) · |Dβ|²
//! ρ_h(r) = r²/4h + (τ − ½) r + h/4   (|r| < h),   ρ_τ(r) elsewhere
//! ```
//!
//! `ρ_h` is the check loss convolved with a uniform kernel of half-width
//! `h`, the loss whose Newton step each row of the stream takes, and the
//! penalty is the one the model's solve puts on the band system (`W_h` the
//! band's weight, `D` the scales under `standardize`). A pass is that
//! Newton step, solved by the model's own solve on the sums the rows
//! rebuild at the pass's fit (`replay_rows`): a row inside the band a
//! least-squares row at target `y + 2h(τ − ½)`, a row outside a nudge
//! `w · 2h · ψ(r)`, unbounded. A full step can overshoot -- the band's rows
//! are a curvature measured at the pass's fit, not at the next -- so the
//! step is halved until `F` at the pass's `h` does not rise, which a
//! Newton direction of a convex `F` always allows. A band of fewer than
//! one row a coefficient has no curvature to step on; such a pass takes the
//! majorizer's step instead, every row a least-squares row at weight `v_i`
//! and target `z_i'β + 2h·ρ_h'(r_i)` (`ρ_h` has curvature at most `1/2h`),
//! which moves every prediction toward its row by up to the band's width.
//! The passes run until the full step moves no buffered row's fitted value
//! by more than [`crate::huber_refit::REFIT_TOL`] of the pass's scale `s`,
//! past the fitted values' rounding (`huber`'s rule,
//! [`crate::huber_refit::converged`]), or until no step along it lowers `F`
//! past `F`'s rounding (`Robust::damped`), or [`REFIT_PASSES`]; one that
//! takes them all is counted in the model's `solve_failures`, as `huber`'s
//! refit is.
//!
//! Then the target's sums are rebuilt at the converged fit -- each row as
//! the stream takes it, scored by that fit at its band -- and the scale
//! from its residuals, and the stream goes on online. A buffer the stream
//! still learns as least squares, its rows counted one each and decayed
//! short of the warm-up's three a coefficient (a half-life shorter than
//! the stream's warm-up), is left as the stream built it.

use super::{Robust, RowUpdate, WARM_ROWS};
use crate::huber_refit::{REFIT_PASSES, Rows, converged, fitted, rebuild_scale, scale_of};

/// What a pass of the refit does to the buffered rows.
#[derive(Debug, Clone, Copy)]
enum Pass {
    /// No fit yet, or no scale to draw a band in: every row least squares,
    /// as the stream takes it then.
    LeastSquares,
    /// The Newton step at the band of half-width `h`.
    Newton(f64),
    /// The majorizer's step at `h`, where the band holds fewer rows than
    /// coefficients.
    Majorize(f64),
}

#[cfg(test)]
thread_local! {
    /// The passes of each kind the refits on this thread took -- least
    /// squares, Newton, the majorizer's -- and the Newton steps halved: what
    /// a test reads to see a branch run.
    pub(super) static PASS_KINDS: std::cell::Cell<[u32; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// Count a pass of kind `i` ([`PASS_KINDS`]); nothing outside tests.
#[inline]
fn count_pass(_i: usize) {
    #[cfg(test)]
    PASS_KINDS.with(|c| {
        let mut v = c.get();
        v[_i] += 1;
        c.set(v);
    });
}

/// Halvings of a Newton step before the refit reads it as one that does
/// not lower `F` ([`Robust::damped`]): `2^-40` of a step no smaller than
/// the rounding of `F`.
const MAX_HALVINGS: u32 = 40;

/// `ρ_h(r)`, the check loss at level `tau` convolved with a uniform kernel
/// of half-width `h` (the module docs).
fn smoothed_check(r: f64, h: f64, tau: f64) -> f64 {
    if r.abs() < h {
        r * r / (4.0 * h) + (tau - 0.5) * r + 0.25 * h
    } else if r > 0.0 {
        tau * r
    } else {
        (tau - 1.0) * r
    }
}

/// `ψ(r) = τ − 1{r ≤ 0}`, as the stream reads a row outside its band.
fn psi(r: f64, tau: f64) -> f64 {
    if r > 0.0 { tau } else { tau - 1.0 }
}

impl Robust {
    /// The refit of target `j` (the module docs): the converged smoothed
    /// check-loss fit of its buffered rows, the sums rebuilt at it, the
    /// scale from its residuals, and the target's own slot solved from the
    /// rebuilt sums if a solve had fit it -- no other target's, and the
    /// cadence left as it stands. One that no solve has fit keeps waiting
    /// for its first, which then reads the rebuilt sums.
    #[inline(never)]
    pub(super) fn refit_quantile(&mut self, j: usize, tau: f64) {
        let nf = self.cfg.n_features;
        let (buf, ages) = self.refit[j].take();
        let rows = Rows::new(&buf, &ages, nf);
        let n = rows.len();
        let v = rows.decayed_weights();
        let vs = rows.scale_weights();
        let k = self.cfg.k_total();
        let kf = k as f64;
        // The rows counted one each, decayed: the stream's `nobs`.
        let count: f64 = (0..n).map(|i| v[i] / rows.w(i)).sum();
        if count.is_nan() || count < WARM_ROWS * kf {
            // Still the warm-up's least squares: the stream's sums are the
            // batch fit's, and the refit leaves them.
            return;
        }
        let width = self.cfg.quantile_eps.max((kf / count).powf(0.4));
        let intercept = self.cfg.fit_intercept;
        let fitted_before = self.fitted(j);
        let mut beta: Option<Vec<f64>> = if fitted_before {
            self.beta.as_ref().map(|b| b[j].clone())
        } else {
            None
        };
        let mut z = vec![0.0; k];
        let mut r = vec![0.0; n];
        // The passes' factor jitters are the refit's, not failures of the
        // fit, as in `huber`'s refit.
        let failures = self.solve_failures;
        let (mut passes, mut solved) = (0u32, false);
        for _ in 0..REFIT_PASSES {
            passes += 1;
            let mut scale = None;
            let pass = match &beta {
                None => Pass::LeastSquares,
                Some(b) => match self.band_at(&rows, b, &vs, width, &mut r) {
                    None => Pass::LeastSquares,
                    Some(h) => {
                        scale = Some(h / width);
                        let inside: f64 = (0..n)
                            .filter(|&i| r[i].abs() < h)
                            .map(|i| v[i] / rows.w(i))
                            .sum();
                        if inside < kf {
                            Pass::Majorize(h)
                        } else {
                            Pass::Newton(h)
                        }
                    }
                },
            };
            count_pass(match pass {
                Pass::LeastSquares => 0,
                Pass::Newton(_) => 1,
                Pass::Majorize(_) => 2,
            });
            match pass {
                Pass::LeastSquares => self.replay_rows(j, &rows, &mut z, |i| RowUpdate::Fit {
                    w: rows.w(i),
                    target: rows.y(i),
                }),
                Pass::Newton(h) => {
                    self.replay_rows(j, &rows, &mut z, |i| stream_row(&rows, &r, i, h, tau));
                }
                Pass::Majorize(h) => self.replay_rows(j, &rows, &mut z, |i| {
                    let step = if r[i].abs() < h {
                        r[i] + 2.0 * h * (tau - 0.5)
                    } else {
                        2.0 * h * psi(r[i], tau)
                    };
                    RowUpdate::Fit {
                        w: rows.w(i),
                        target: rows.y(i) - r[i] + step,
                    }
                }),
            }
            let next = if intercept {
                self.solve_centred(k, j, 0)
            } else {
                self.solve_through_origin(k, j, 0)
            };
            let Some((next, _)) = next else {
                break;
            };
            solved = true;
            let done = converged(&rows, intercept, beta.as_deref(), &next, scale);
            let (fit, stalled) = match (pass, beta.take()) {
                (Pass::Newton(h), Some(b)) if !done => {
                    match self.damped(j, &rows, &v, &r, &b, next, h, tau) {
                        Some(c) => (c, false),
                        None => (b, true),
                    }
                }
                (_, _) => (next, false),
            };
            beta = Some(fit);
            if done || stalled {
                break;
            }
        }
        self.solve_failures = failures;
        self.refit[j].ran(passes);
        let Some(b) = beta.filter(|_| solved) else {
            self.refit[j].fail();
            self.solve_failures += 1;
            return;
        };
        // A refit that took every pass is counted, as `huber`'s is: its fit
        // stands, the last pass's.
        if self.refit[j].status().ran_to_cap() {
            self.solve_failures += 1;
        }
        // The sums as the stream takes each row scored by the converged fit
        // at its band, and the scale from its residuals.
        match self.band_at(&rows, &b, &vs, width, &mut r) {
            Some(h) => self.replay_rows(j, &rows, &mut z, |i| stream_row(&rows, &r, i, h, tau)),
            None => self.replay_rows(j, &rows, &mut z, |i| RowUpdate::Fit {
                w: rows.w(i),
                target: rows.y(i),
            }),
        }
        let scale = rebuild_scale(&rows, self.cfg.huber_scale, self.cfg.mad_anchor, |i| {
            rows.y(i) - fitted(&b, rows.x(i), intercept)
        });
        self.sig2[j] = scale.sig2;
        self.wsig[j] = scale.wsig;
        if let Some(q) = scale.mad {
            self.mad[j] = q;
        }
        // The band system the passes kept is of the sums before the last
        // replay: the next solve or nudge builds it again.
        *self.systems.slot(self.cfg.n_targets, j) = None;
        if fitted_before && let Some(prev) = self.beta.take() {
            let mut out: Vec<Vec<f64>> = prev.to_vec();
            self.solve_target(j, Some(prev.as_slice()), &mut out);
            self.beta = Some(crate::model::Fit(out));
        }
    }

    /// The band's half-width at fit `b`, the residuals written into `r`:
    /// the MAD of the residuals (the root mean square under the RMS) at the
    /// scale's weights `vs` times `width`, `None` where there is no scale.
    fn band_at(
        &self,
        rows: &Rows<'_>,
        b: &[f64],
        vs: &[f64],
        width: f64,
        r: &mut [f64],
    ) -> Option<f64> {
        let intercept = self.cfg.fit_intercept;
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = rows.y(i) - fitted(b, rows.x(i), intercept);
        }
        let absr: Vec<f64> = r.iter().map(|v| v.abs()).collect();
        scale_of(self.cfg.huber_scale, &absr, vs).map(|s| s * width)
    }

    /// The Newton step from `b` to `next` at the band `h`, halved until the
    /// penalized smoothed loss `F` (the module docs) does not rise past its
    /// rounding: `next` itself where the full step does not. The sums are
    /// the Newton pass's, which give `F`'s penalty its band weight and
    /// scales; `r` holds the residuals at `b`.
    ///
    /// `F` is compared as its change from `b`, `ΔF = Σ v_i (ρ_h(r_i − d_i)
    /// − ρ_h(r_i))` plus the penalty's change, `d_i = z_i'(c − b)` the move
    /// of row `i`'s fitted value: every term is in the residuals' units,
    /// and the step's, so nothing in it reads the target's level. `F` itself
    /// does: each `y_i − z_i'c` is rounded at the level, `1e-10` a row at a
    /// level of `1e6`, so two values of `F` read a Newton step of `1e-5`
    /// that lowered it as one that raised it, ten thousand times past `F`'s
    /// slack, and the refit stopped there, a level apart from the fit it
    /// reached at 0. A row's change is written in its piece's closed form,
    /// a multiple of `d_i`, so its rounding is relative to the change and
    /// the slack is a few units in the last place of the change's terms.
    ///
    /// `None` where the full step raises `F` past that rounding and no
    /// shorter step along it lowers `F` at all: `b` is then `F`'s minimum to
    /// its rounding, the fit stands and the refit has converged. A Newton
    /// direction of a convex `F` lowers it near `b` unless its slope along
    /// it is lost in that rounding, which is where the solve's own rounding
    /// leaves a step of a few hundred units in the last place of the fitted
    /// values' level, pass after pass, at a level far from 0.
    #[allow(clippy::too_many_arguments)]
    fn damped(
        &self,
        j: usize,
        rows: &Rows<'_>,
        v: &[f64],
        r: &[f64],
        b: &[f64],
        next: Vec<f64>,
        h: f64,
        tau: f64,
    ) -> Option<Vec<f64>> {
        let intercept = self.cfg.fit_intercept;
        let (s, keep) = self.band_scales(j);
        let pen_weight = self.wj[j] * self.cfg.ridge / (4.0 * h);
        let off = usize::from(intercept);
        let mut step = vec![0.0; b.len()];
        // `ΔF` from `b` to `c`, and the size of its terms (its slack's unit).
        let mut change = |c: &[f64]| -> (f64, f64) {
            for ((d, &q), &p) in step.iter_mut().zip(c).zip(b) {
                *d = q - p;
            }
            let (mut f, mut size) = (0.0, 0.0);
            for (i, &vi) in v.iter().enumerate() {
                let (t, m) = check_change(r[i], fitted(&step, rows.x(i), intercept), h, tau);
                f += vi * t;
                size += vi * m;
            }
            // `Σ s² (c² − b²) = Σ s² (c − b)(c + b)`, in the slopes' units.
            let (mut pen, mut pen_size) = (0.0, 0.0);
            for &i in &keep {
                let t = s[i] * s[i] * step[i + off] * (c[i + off] + b[i + off]);
                pen += t;
                pen_size += t.abs();
            }
            (f + pen * pen_weight, size + pen_size * pen_weight)
        };
        let slack = |size: f64| 64.0 * f64::EPSILON * size;
        let (f, size) = change(&next);
        if f <= slack(size) {
            return Some(next);
        }
        count_pass(3);
        let mut t = 1.0;
        let mut out = next.clone();
        for _ in 0..MAX_HALVINGS {
            t *= 0.5;
            for ((o, &p), &q) in out.iter_mut().zip(b).zip(&next) {
                *o = p + t * (q - p);
            }
            let (f, size) = change(&out);
            if f <= slack(size) {
                return (f < 0.0).then_some(out);
            }
        }
        None
    }
}

/// `ρ_h(r − d) − ρ_h(r)`, the change of one row's smoothed check loss when
/// its fitted value moves by `d`, and the size of its terms, which bounds
/// its rounding. Where `r` and `r − d` share a piece the change is `d`
/// times that piece's mean slope, `−d · ((r + r')/4h + τ − ½)` inside the
/// band and `−d · τ` or `−d · (τ − 1)` outside it; across pieces it is the
/// difference of the two losses, each in the residuals' units.
fn check_change(r: f64, d: f64, h: f64, tau: f64) -> (f64, f64) {
    let after = r - d;
    let piece = |e: f64| {
        if e.abs() < h {
            0
        } else if e > 0.0 {
            1
        } else {
            2
        }
    };
    match (piece(r), piece(after)) {
        (0, 0) => {
            let mean = (r + after) / (4.0 * h);
            (
                -d * (mean + (tau - 0.5)),
                d.abs() * (mean.abs() + (tau - 0.5).abs()),
            )
        }
        (1, 1) => (-d * tau, (d * tau).abs()),
        (2, 2) => (-d * (tau - 1.0), (d * (tau - 1.0)).abs()),
        _ => {
            let (p, q) = (smoothed_check(after, h, tau), smoothed_check(r, h, tau));
            (p - q, p.abs() + q.abs())
        }
    }
}

/// Row `i` as the stream takes it against a fit whose residuals are `r`, at
/// the band `h`: inside it a least-squares row at target `y + 2h(τ − ½)`,
/// outside it a nudge `w · 2h · ψ(r)` (the module docs of `robust`).
fn stream_row(rows: &Rows<'_>, r: &[f64], i: usize, h: f64, tau: f64) -> RowUpdate {
    if r[i].abs() < h {
        RowUpdate::Fit {
            w: rows.w(i),
            target: rows.y(i) + 2.0 * h * (tau - 0.5),
        }
    } else {
        RowUpdate::Nudge {
            nudge: rows.w(i) * 2.0 * h * psi(r[i], tau),
        }
    }
}
