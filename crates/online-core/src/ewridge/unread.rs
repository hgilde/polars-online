//! A solve the schedule brings due that no row can read is not run
//! (docs/PLAN.md task 311, option C; the user's approval of 2026-11-02).
//!
//! Under a weight share `s` -- `solve_share`, or the default schedule's
//! [`crate::DEFAULT_SOLVE_SHARE`] in an explicit cadence's first period -- a
//! solve comes due once the weight learned since the last reaches `s` of the
//! weight the fit holds. While the weight is far enough below the lowest
//! floor any read of the fit uses, every row until the next solve comes due
//! is withheld, so the solve's fit is never read before the next solve
//! replaces it. Running it cost a factorization and the condition estimate
//! on every one of the warm-up's solves: at a thousand features, 270 of a
//! 2,000-row stream's 329 solves were of that kind, and a default fit there
//! ran 6.9 times faster without them (the speed record's figures).
//!
//! The rows and the reads, with `W` the model's weight after a row
//! ([`EwRidge::n_eff`]), `F` the floor ([`EwRidge::unread_floor`]), `u =
//! 2⁻⁵³` and `γ_n = n u / (1 − n u)`:
//!
//! 1. *Who reads the fit, and at what weight.* A row's prediction reads the
//!    fit where the weight before it, the `W` its previous row left, reaches
//!    `min_weight` ([`OnlineModel::predict`]); the residual variance's read
//!    on a row `min_weight` withholds is at the model's own floor,
//!    `min(min_weight, k_total)` or `min_weight`; a bank's ungated spread
//!    reads at the lowest of `k_total` and its `min_weight`s
//!    ([`OnlineModel::predict_with_floor`]), where `min_weight` is the
//!    lowest of them; the error inflation, the row's leverage and
//!    [`EwRidge::ill_conditioned`] answer for no fit below `min_weight`. A
//!    target's own first solve reads whether a solve has fit the target, and
//!    fires where the target's weight reaches its own threshold, capped at
//!    `k_total` under the output-only floor; the fresh model's first solve
//!    where `W` reaches the own floor. So with `F` the least of `k_total`,
//!    `min_weight` and every target's threshold, no read of the fit, and no
//!    trigger that reads whether there is one, happens while `W < F`: a
//!    target's weight is never above `W`, the same recursion over a subset
//!    of the rows, rounded the same way at each step. What reads the fit at
//!    any weight is the report of it: `coef` and its company
//!    ([`EwRidge::coefficients`], [`OnlineModel::support_coef`],
//!    [`OnlineModel::coef_variance`]), the state file and the solves'
//!    failure count. Those are the fields this changes, below the floor.
//! 2. *Between two due rows, `W` stays below `W_s / (1 − s)`.* Let row `s`
//!    be due, and each row `r` after it up to the next due row not. A row
//!    of weight 0 is never due and moves `W` only by its decay, never up.
//!    A row with weight that is not due has `Σ_r < s W_r`, `Σ_r` the weight
//!    learned since row `s`: under a weight share, or in the first period,
//!    which lasts until a solve its explicit cadence brings due, and a skip
//!    is never one ([`EwRidge::solve_is_unread`] requires `!explicit`).
//!    Every decay factor is in `[0, 1]` and `W_r = λ W + w`, so `W_r ≤ W_s +
//!    Σ_r` and `W_r < W_s / (1 − s)`.
//! 3. *In floating point.* `fl(λ W) ≤ W` (rounding is monotone and `λ ≤
//!    1`), so by induction the computed `W_r` is no larger than the
//!    recursive sum of `W_s` and the `n` weights since, `≤ (W_s + Σ_r)(1 +
//!    γ_n)` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
//!    ed., §4.2, all terms nonnegative); the computed `Σ̂_r` is at least
//!    `Σ_r (1 − γ_n)`, and not due says `Σ̂_r < fl(s W_r) ≤ s W_r (1 + u)`.
//!    Together `W_r < W_s (1 + γ_n) / (1 − c s)`, with
//!    `c = (1 + γ_n)(1 + u) / (1 − γ_n)`. The rows between two due rows are
//!    fewer than the row cap, a `u32` (a row past it is due), so `n < 2³²`,
//!    `γ_n < 2⁻²⁰` and `c < 1 + 2⁻¹⁹`; with `s ≤ 1/2`,
//!    `(c − 1) s / (1 − s) ≤ c − 1`, so `W_s ≤ F (1 − s)(1 − 2⁻¹⁸)` gives
//!    `W_r < F` on every row to the next due one. The skip asks for
//!    `W_s < F (1 − s)(1 − 2⁻¹⁶)`, its three roundings inside that, and is
//!    off for `s > 1/2`, where a share solves at least every time the weight
//!    doubles and there is little to skip.
//! 4. *So the next due row's prediction, and every one before it, is
//!    withheld; its trigger reads no fit.* The prediction at row `r + 1`
//!    reads `W_r < F`, down to the next due row's own, which reads the `W`
//!    of the row before it. A target's own first solve and the fresh
//!    model's fire only at a weight `≥ F`, so neither fires between, and a
//!    due row is due either way: the rows the schedule solves on are the
//!    same whether the skipped fit is there or not.
//! 5. *What the skip keeps.* A solve does two things to the state besides
//!    the fit: it brings a held Gram block up to date ([`crate::gaps::Acc`]'s
//!    `flush`), which is where the next block starts, and it restarts the
//!    schedule's clock, row count and weight. The skip does both, so the
//!    moments, the blocked Grams' merge bound and the schedule are the
//!    solve's to the bit. The first period's own count from the stream's
//!    start is kept around it as around a solve (`FirstPeriod`).
//! 6. *What a later solve reads of the fit before it.* Two things, and
//!    there it reads the last fit solved where it read the passed one, so
//!    these two are where a served number can move. A combo whose
//!    factorization fails at every rung of the jitter keeps the fit and the
//!    readiness shares before it: that takes a non-finite entry in the
//!    system -- a feature's square past the largest double -- or an
//!    indefiniteness past `1e-3` of its mean diagonal, which the moments'
//!    rounding, at most `γ_{2n+1}` of their trace over `n` rows of positive
//!    weight (the condition module's ridge shortcut, item 1), reaches only
//!    past about `4·10¹²/k` rows. And a target with no weight at all keeps
//!    the fit a solve gave it before, which takes its weight decayed to
//!    exactly 0 -- more than a thousand half-lives with no row of its own --
//!    while the stream stayed below the floor. Either must fall on the
//!    first solve run after a passed one, which then keeps an older fit, or
//!    none, where running every due solve kept the passed one's.
//! 7. *The rest of the model.* A zero-weight row (hard rule 9) never solves
//!    and never raises `W` (2). A clock gap only decays `W`. Groups are
//!    models of their own. `session_shrink`'s blend moves the moments only,
//!    never the weight, the fit or the schedule. A `restart_after_step_back`
//!    rebuilds the model. A saved state carries the schedule and the
//!    weight, and a bank loads it under the spec it was saved with, so a
//!    loaded model skips where the uninterrupted one does; a state written
//!    before the skip carries a fit solved below the floor, which goes on
//!    being reported, unread, until the next solve run. A `window_size` is
//!    left out: its weight is the difference of two sums, rounded at the
//!    size of the whole history rather than the window's, which (3) does
//!    not bound.
//!
//! A solve the row cap brings due runs, read or not: `max_rows_between_solves
//! = 1` is how a caller asks for a fit current at every row, and `coef`
//! reports it.
//!
//! The observable choice below the floor: `coef` is the last fit solved,
//! null before the first, as the schedule's latest solve is everywhere
//! else; a solve not run is not counted in `solve_failures`; and the state
//! file holds that fit beside a schedule restarted as after the due solve.

use super::EwRidge;

/// The skip's own margin on the floor, `2⁻¹⁶` (the module docs, item 3).
const MARGIN: f64 = 1.0 / 65_536.0;

/// The largest share the skip runs under (the module docs, item 3).
const MAX_SHARE: f64 = 0.5;

impl EwRidge {
    /// `F`: the least of the floors a read of the fit uses (the module docs,
    /// item 1) -- `k_total`, `min_weight` and each target's own threshold.
    pub(super) fn unread_floor(&self) -> f64 {
        let k = self.cfg.k_total() as f64;
        self.target_min_weight
            .iter()
            .copied()
            .fold(self.cfg.min_weight.min(k), f64::min)
    }

    /// Whether the solve due on this row, not one its explicit cadence or
    /// its row cap brings due, can have no reader before the next one comes
    /// due (the module docs): a weight share runs the schedule -- `first`,
    /// the explicit cadence's first period, runs the default one -- and the
    /// weight after the row is below `(1 − s)(1 − 2⁻¹⁶)` of the floor.
    pub(super) fn solve_is_unread(&self, first: bool) -> bool {
        #[cfg(test)]
        if !SKIP_UNREAD.with(std::cell::Cell::get) {
            return false;
        }
        let share = match self.cfg.solve_share {
            Some(share) => share,
            None if first => crate::DEFAULT_SOLVE_SHARE,
            None => return false,
        };
        // A row cap's solve runs: `max_rows_between_solves = 1` promises a
        // fit current at every row, read or not.
        // A share is finite and above 0 (`EwRidgeCfg::validate`).
        if share > MAX_SHARE
            || self.win.is_some()
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
        {
            return false;
        }
        self.n_eff() < self.unread_floor() * (1.0 - share) * (1.0 - MARGIN)
    }

    /// What a solve does to the state besides the fit (the module docs,
    /// item 5): the held Gram block merged, so the next block starts where
    /// it would, and the schedule restarted.
    pub(super) fn pass_unread_solve(&mut self) {
        #[cfg(test)]
        UNREAD.with(|c| c.set(c.get() + 1));
        self.acc.flush();
        self.since_solve.restart();
        self.rows_since_solve = 0;
        self.weight_since_solve = 0.0;
    }
}

#[cfg(test)]
thread_local! {
    /// The due solves this thread's models passed as unread: what an
    /// activation test counts.
    pub(crate) static UNREAD: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The solves this thread's models ran.
    pub(crate) static SOLVED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Whether a due solve no row can read is passed (`true`, the model's
    /// way) or run as every due solve once was: the reference the tests
    /// hold the served numbers to.
    pub(crate) static SKIP_UNREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
