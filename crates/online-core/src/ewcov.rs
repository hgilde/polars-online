//! `EwCov`: exponentially weighted first/second moments of a vector stream
//! (docs/PLAN.md §4.7). The shared accumulator behind EW-ridge, RLS and Kalman,
//! and exposed on its own as `online.ew_cov()`.
//!
//! All statistics are stored as weighted *means* (not sums), which keeps them
//! bounded under arbitrarily long runs (docs/PLAN.md §7), and the second
//! moments are kept **centered** (a weighted Welford update) rather than raw:
//!
//! ```text
//! W'     = lam * W + w
//! a      = lam * W / W'        b = w / W'        (a + b = 1)
//! delta  = x - m
//! m'     = m + b * delta
//! C'_ij  = a * C_ij + a * b * delta_i * delta_j
//! ```
//!
//! Centered updates matter because the earlier raw form derived the variance as
//! `E[x²] − m²`, and that subtraction loses precision when the mean is large
//! relative to the spread: a unit-variance feature around 1e8 has
//! `var/E[x²] ≈ 1e-16`, so the variance was destroyed entirely. With this form
//! the variance is accurate at any offset (see `docs/TESTING.md` T-E9 and the
//! river cross-check in `tests/test_river.py`). [`EwCov::raw`] reconstructs the
//! raw moment as `C_ij + m_i m_j` for the callers that genuinely need it (the
//! uncentered normal equations).

use serde::{Deserialize, Serialize};

/// Is a centered variance large enough to standardize by, given the raw second
/// moment it was computed from?
///
/// `var = E[x²] − m²` loses precision by cancellation when the mean is large
/// relative to the spread: the absolute error is on the order of
/// `f64::EPSILON * E[x²]`. The threshold is a small multiple of that noise
/// floor, so a genuine variance is kept even when it sits on a large offset
/// (a unit-variance feature around 1e6 has `var/raw ≈ 1e-12`, which is real),
/// while a variance that has actually been destroyed by cancellation is
/// rejected and the feature dropped from the solve.
///
/// A previous fixed `1e-10 * raw` threshold was ~450,000x the noise floor and
/// silently discarded usable features at ordinary financial scales.
///
/// Since [`EwCov`] switched to centered (Welford) updates the variance no
/// longer suffers cancellation, so the only question left is whether the
/// feature is genuinely constant — which gives *exactly* zero, because every
/// deviation from the running mean is exactly zero. The `raw_second_moment`
/// argument is kept for callers and for the doc trail, but is no longer used to
/// scale the threshold: doing so is what silently dropped usable features
/// sitting on a large offset.
#[inline]
pub fn variance_is_usable(var: f64, _raw_second_moment: f64) -> bool {
    var > 0.0 && var.is_finite()
}

/// A reusable row buffer that is not state: two accumulators with the same
/// moments are the same accumulator whatever is left in their scratch, and a
/// state file carries none of it.
#[derive(Debug, Clone, Default)]
struct Scratch(Vec<f64>);

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Rows waiting for a blocked Gram update.
///
/// `x` is the held rows, row-major `n * k`, with `lam[i]` and `w[i]` the
/// row's decay and weight, in arrival order. Flat rather than one `Vec` per
/// row so that the block is one allocation that keeps its capacity from one
/// flush to the next, and so that the product can read it in place as a
/// matrix. `w_open` is `w_sum` at the moment the block opened; the scalars
/// advance per row, so by flush time `w_sum` is already the post-block weight
/// and the pre-block one has to have been kept.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Pending {
    /// `0` means blocking is off, which is every model but `ewridge`.
    block_rows: usize,
    x: Vec<f64>,
    lam: Vec<f64>,
    w: Vec<f64>,
    w_open: f64,
}

impl Pending {
    fn is_off(&self) -> bool {
        self.block_rows == 0 && self.lam.is_empty()
    }

    fn len(&self) -> usize {
        self.lam.len()
    }

    /// Drop the held rows, keeping the capacity for the next block.
    fn clear_rows(&mut self) {
        self.x.clear();
        self.lam.clear();
        self.w.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwCov {
    k: usize,
    /// EW sum of weights (the `n_eff` count).
    w_sum: f64,
    /// EW sum of *squared* weights, `Q = Sum w^2` under `lam^2` decay, from
    /// which Kish's effective sample size `W^2 / Q` follows
    /// (docs/ENHANCEMENTS.md E45). `None` in a state written before task 38:
    /// the history behind `w_sum` cannot be replayed, and a `Q` accumulated
    /// from the resume point paired with a `W` from the whole stream reports
    /// an effective size that is wrong by the length of the history. Such a
    /// state keeps reporting `None`, which is the honest answer.
    ///
    /// Written even when `None` (as nil): the compact encoding is positional,
    /// and a field that skips anywhere but last slides the fields after it
    /// (`crates/online-core/tests/state_encoding.rs`). `pending` is the one
    /// that skips, and it is last.
    #[serde(default)]
    q_sum: Option<f64>,
    /// Product of all decay factors applied so far (a decaying prior's scale).
    prior_scale: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW **centered** co-moments, row-major `k*k`.
    #[serde(alias = "s")]
    c: Vec<f64>,
    /// Prior strength for the precision matrix `M⁻¹ = (C + s·prior·I)⁻¹`
    /// (docs/PLAN.md §4.7); `0` when none is wanted.
    ///
    /// The precision matrix is not stored. Schema-1 states tracked it by
    /// Sherman-Morrison, which has the same two failure modes as covariance
    /// form RLS (docs/IMPROVEMENTS.md C5): the division by `a` every row
    /// amplifies rounding asymmetry without bound, and a row that dominates
    /// the prior by `1/ulp` in some direction cancels the inverse to exactly
    /// zero there, which no later row can undo. [`Self::precision`] solves
    /// `M X = I` from the co-moments instead, which cannot drift, and is only
    /// paid for when partial correlations are read.
    #[serde(default, alias = "inv_prior")]
    precision_prior: f64,
    /// Row scratch: `x - m`, the deviations the co-moment update needs `k`
    /// times each (docs/ENHANCEMENTS.md E48). Not part of the state -- serde
    /// skips it and [`PartialEq`] ignores it, so a round-tripped accumulator
    /// still compares equal to the one that wrote it.
    #[serde(skip)]
    dev: Scratch,
    /// Decaying scale `s` on that prior. It must decay by the *same* factor
    /// `a` the co-moments do, not by `lam`, so that `M` stays a fixed multiple
    /// of what it would be with a constant prior:
    /// `M' = a·C + a·b·δδᵀ + a·s·prior·I = a·(M + b·δδᵀ)`. Like RLS's `P₀`,
    /// this makes the prior fade as data accumulates.
    #[serde(default, alias = "inv_scale")]
    precision_scale: f64,
    /// Rows buffered for a blocked Gram update, and `w_sum` as it stood when
    /// the block opened (the merge needs the pre-block weight, and the
    /// scalars have moved on by then).
    ///
    /// Empty and `0` unless [`Self::set_block_rows`] asked for blocking,
    /// which only `ewridge` does: blocking pays where the Gram is read on a
    /// schedule, and a model emitting per-row statistics would flush every
    /// row for nothing (docs/PLAN.md task 71).
    ///
    /// **Last, and it must stay last.** The compact msgpack encoding writes a
    /// struct as a bare array, so a `skip_serializing_if` field anywhere else
    /// slides everything after it: put here first, this decoded
    /// `precision_scale`'s float into `Pending`'s slot and broke six
    /// round-trip tests (`crates/online-core/tests/state_encoding.rs` states
    /// the rule).
    #[serde(default, skip_serializing_if = "Pending::is_off")]
    pending: Pending,
}

impl EwCov {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            q_sum: Some(0.0),
            prior_scale: 1.0,
            m: vec![0.0; k],
            c: vec![0.0; k * k],
            dev: Scratch(Vec::with_capacity(k)),
            precision_prior: 0.0,
            precision_scale: 1.0,
            pending: Pending::default(),
        }
    }

    /// Same, with a prior for the precision matrix `(C + s·prior·I)⁻¹`, so
    /// that [`Self::precision`] and partial correlations are available.
    ///
    /// `prior` regularizes the inverse and must be `> 0`: the centered
    /// co-moment matrix starts at zero and is singular until `k` independent
    /// rows have been seen, so there is nothing to invert without it.
    /// An accumulator standing at the given weight, means and **centred**
    /// co-moments, rather than at zero: the state a caller who already has
    /// the moments wants to filter with (`hmm`'s `means`/`covs`, a state
    /// lifted out of a fitted `ew_class`).
    ///
    /// The decaying prior scale starts at 1, as it does for a fresh
    /// accumulator, so the ridge behaves as if this were the first row.
    pub fn from_moments(
        k: usize,
        w_sum: f64,
        means: &[f64],
        comoments: &[f64],
        prior: f64,
    ) -> Result<Self, String> {
        if means.len() != k || comoments.len() != k * k {
            return Err(format!(
                "EwCov::from_moments: {k} columns need {k} means and {} co-moments, got {} and {}",
                k * k,
                means.len(),
                comoments.len()
            ));
        }
        if w_sum < 0.0 || !w_sum.is_finite() {
            return Err("EwCov::from_moments: w_sum must be finite and >= 0".into());
        }
        let mut ew = Self::with_precision_prior(k, prior)?;
        ew.w_sum = w_sum;
        ew.m.copy_from_slice(means);
        ew.c.copy_from_slice(comoments);
        Ok(ew)
    }

    pub fn with_precision_prior(k: usize, prior: f64) -> Result<Self, String> {
        if prior <= 0.0 || !prior.is_finite() {
            return Err("EwCov::with_precision_prior: prior must be finite and > 0".into());
        }
        let mut ew = Self::new(k);
        ew.precision_prior = prior;
        ew.precision_scale = 1.0;
        Ok(ew)
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn has_precision_prior(&self) -> bool {
        self.precision_prior > 0.0
    }

    /// Current decaying scale `s` on the precision prior (see the field).
    pub fn precision_scale(&self) -> f64 {
        self.precision_scale
    }

    /// The precision matrix `(C + s·prior·I)⁻¹`, row-major `k*k` and exactly
    /// symmetric, solved from the co-moments by Cholesky (`solve_spd`, so a
    /// numerically indefinite `M` gets the same jitter the regressions use).
    /// `None` when no prior was configured or the solve fails.
    pub fn precision(&self) -> Option<Vec<f64>> {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        if !self.has_precision_prior() {
            return None;
        }
        let k = self.k;
        let mut m = self.c.clone();
        let mut eye = vec![0.0; k * k];
        for i in 0..k {
            m[i * k + i] += self.precision_prior * self.precision_scale;
            eye[i * k + i] = 1.0;
        }
        // Columns of the solution are columns of the inverse; the identity is
        // symmetric, so its column-major layout is the same array.
        let (x, _) = crate::solve::solve_spd(&m, &eye, k, k)?;
        let mut p = vec![0.0; k * k];
        for i in 0..k {
            for j in i..k {
                let v = 0.5 * (x[j * k + i] + x[i * k + j]);
                p[i * k + j] = v;
                p[j * k + i] = v;
            }
        }
        p.iter().all(|v| v.is_finite()).then_some(p)
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Kish's effective sample size `W^2 / Q`: the number of *equally*
    /// weighted rows that carry the information these moments hold
    /// (docs/ENHANCEMENTS.md E45). `(1 + lam) / (1 - lam)` in the limit for
    /// unit weights on a constant clock.
    ///
    /// `None` before the first row, and for a state written before task 38
    /// (see the `q_sum` field). It is what a standard error from these
    /// moments must divide by: `n_eff` counts weight, not rows, so it is not
    /// a sample size.
    #[inline]
    pub fn n_kish(&self) -> Option<f64> {
        match self.q_sum {
            Some(q) if q > 0.0 => Some(self.w_sum * self.w_sum / q),
            _ => None,
        }
    }

    /// `Sum w^2` behind these moments, or `None` for a pre-task-38 state.
    #[inline]
    pub fn q_sum(&self) -> Option<f64> {
        self.q_sum
    }

    #[inline]
    pub fn prior_scale(&self) -> f64 {
        self.prior_scale
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        self.m[i]
    }

    /// Raw (uncentered) second moment `E_w[x_i x_j]`, reconstructed from the
    /// centered co-moment. Use [`Self::cov`] wherever a *centered* quantity is
    /// wanted: going through `raw` and subtracting the means again reintroduces
    /// exactly the cancellation this representation avoids.
    #[inline]
    pub fn raw(&self, i: usize, j: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        self.c[i * self.k + j] + self.m[i] * self.m[j]
    }

    /// Centered covariance, held directly.
    #[inline]
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        self.c[i * self.k + j]
    }

    /// Centered variance, floored at zero against rounding.
    #[inline]
    /// The full centered co-moment matrix, row-major `k*k` — the EW analogue
    /// of the centered `X'X / n`.
    ///
    /// This is the accumulator every solve reads, exposed so a caller can do
    /// something *other than* our solve with it: a custom penalty, an
    /// information criterion, `cond(G)`, a scree plot, forward stepwise or
    /// orthogonal matching pursuit — none of which needs a second pass over
    /// data that was never materialized (docs/ENHANCEMENTS.md E30).
    pub fn comoments(&self) -> &[f64] {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        &self.c
    }

    /// The EW mean vector, length `k`. Pairs with [`Self::comoments`]: the
    /// uncentered second moment is `c[i*k+j] + m[i]*m[j]`.
    pub fn means(&self) -> &[f64] {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        &self.m
    }

    pub fn var(&self, i: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first \
             (docs/PLAN.md task 71)"
        );
        self.cov(i, i).max(0.0)
    }

    /// The row's mixing factors `(a, b)` and the new weight, or `None` when
    /// the shipped update would return without changing anything: a negative
    /// weight, or no weight at all in the history and none on the row.
    ///
    /// Split out so the blocked path cannot drift from the rank-1 one. There
    /// is exactly one copy of this arithmetic and of [`Self::commit_scalars`],
    /// which is what makes `w_sum`, `q_sum`, `prior_scale` and
    /// `precision_scale` bit-identical whichever path ran (docs/PLAN.md task
    /// 71).
    fn step_factors(&self, lam: f64, w: f64) -> Option<(f64, f64, f64)> {
        if w < 0.0 {
            return None;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return None;
        }
        Some((lam * self.w_sum / w_new, w / w_new, w_new))
    }

    /// Everything the row does that is not the `k*k` matrix or the mean.
    /// `O(1)`, so the blocked path runs it per row and nothing emitted per
    /// row -- `n_eff`, `n_kish` -- can lag behind a pending block.
    fn commit_scalars(&mut self, lam: f64, w: f64, a: f64, w_new: f64) {
        // The precision prior decays with the co-moments. `a == 0` only on
        // the very first observation, when the whole history is discarded and
        // the centered co-moments are exactly zero: the prior starts over.
        self.precision_scale = if a <= 0.0 {
            1.0
        } else {
            self.precision_scale * a
        };
        self.w_sum = w_new;
        if let Some(q) = self.q_sum.as_mut() {
            // `W` decays by `lam`, so its square decays by `lam^2`; the row
            // adds `w^2`. Kish's `n` is then `W^2 / Q`.
            *q = lam * lam * *q + w * w;
        }
        self.prior_scale *= lam;
    }

    /// How many rows a blocked Gram update holds before merging. `0` is off,
    /// which is every model but `ewridge` (docs/PLAN.md task 71).
    pub fn set_block_rows(&mut self, block_rows: usize) {
        self.flush();
        self.pending.block_rows = block_rows;
    }

    pub fn block_rows(&self) -> usize {
        self.pending.block_rows
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.lam.is_empty()
    }

    /// Hold the row for a blocked merge, advancing the scalars now.
    ///
    /// A row the shipped update would ignore is not held: [`Self::step_factors`]
    /// returning `None` means nothing changes at all, not even the decay, so
    /// buffering it would put a row in the block that contributed no weight
    /// and no `lam`. Every held row therefore committed, which is what lets
    /// the suffix products in [`Self::flush`] be taken over the buffer as it
    /// stands.
    fn buffer(&mut self, x: &[f64], lam: f64, w: f64) {
        let Some((a, _b, w_new)) = self.step_factors(lam, w) else {
            return;
        };
        if self.pending.lam.is_empty() {
            self.pending.w_open = self.w_sum;
        }
        self.pending.x.extend_from_slice(x);
        self.pending.lam.push(lam);
        self.pending.w.push(w);
        self.commit_scalars(lam, w, a, w_new);
        if self.pending.len() >= self.pending.block_rows {
            self.flush();
        }
    }

    /// Merge the held rows into the mean and co-moments.
    ///
    /// With `Λᵢ` the decay from row `i` to the end of the block and
    /// `uᵢ = wᵢ·Λᵢ` the row's weight discounted to that point, two passes
    /// over the block -- it is in memory, so the second is free -- and one
    /// product:
    ///
    /// ```text
    /// m_B = Σ uᵢ xᵢ / U         (pass one, on rows shifted by one of them)
    /// dᵢ  = xᵢ − m_B            (pass two, in place)
    /// S   = Σ uᵢ dᵢ dᵢᵀ         (the rank-B product, half a GEMM)
    /// δ   = Σ uᵢ dᵢ / U         (the rounding residue of m_B; ≈ 0)
    /// Δ   = (m_B − m_A) + δ     (the two means' difference)
    /// C'  = (W_A·C_A + S − U·δδᵀ)/W' + (W_A·U/W'²)·ΔΔᵀ
    /// m'  = m_A + (U/W')·Δ
    /// ```
    ///
    /// Where the rows are centred is the whole of the numerical question.
    /// A rank-B product's natural form is the uncentred `XᵀX`, and
    /// subtracting `W·mmᵀ` afterwards is the cancellation that cost E67 its
    /// variance at an offset of `1e7` (docs/PLAN.md task 66). Centring on the
    /// *running* mean `m_A` looks like the fix and is not: it is exact when
    /// the history is gone only if `W_A = 0` is special-cased, and it is
    /// wrong by half the variance when the history is merely *nearly* gone
    /// -- a decay of `1e-30` after a long gap leaves `m_A` where the old data
    /// was, and `S − U·δδᵀ` is then a difference of two numbers of order
    /// `U·(m_B − m_A)²`: measured at 49% of the block's variance at an offset
    /// of `1e7`. Centring on a *row* of the block halves that and no more,
    /// because the row before the gap still carries a sliver of weight and
    /// can be the one chosen. Centring on the block's own weighted mean is
    /// the two-pass algorithm, and leaves nothing to cancel: `S` is the
    /// block's scatter about its own mean, `δ` is rounding, and the between
    /// term is formed from a difference of two means -- which is what the
    /// rank-1 recursion does too, one row at a time.
    ///
    /// The product is `faer`'s, sequential, on the lower triangle only, then
    /// mirrored: a symmetric rank-B update is half a GEMM. Sequential
    /// because the bank already runs its groups on a pool of its own, and
    /// because a parallel product's summation order is not a function of
    /// the row sequence. Measured against the shipped rank-1 loop, single
    /// threaded, 256-row block: 6.6× at `k = 1,000` and `k = 2,000`, 4.6× at
    /// `k = 256`; a 64-row block gives 4.9×, 4.5× and 2.9×. The design probe's
    /// 9.5×/11.2× was a multithreaded product against a single-threaded
    /// baseline, and is not the number (docs/PLAN.md task 71).
    ///
    /// `W'` is read from `w_sum`, which the per-row scalars already carried,
    /// so the merge cannot disagree with the count that was reported.
    pub fn flush(&mut self) {
        use faer::linalg::matmul::triangular::{BlockStructure, matmul};
        use faer::{Accum, MatMut, MatRef, Par};

        let n = self.pending.len();
        if n == 0 {
            return;
        }
        let k = self.k;
        debug_assert_eq!(self.pending.x.len(), n * k);
        debug_assert_eq!(self.pending.w.len(), n);
        if self.pending.x.len() != n * k || self.pending.w.len() != n {
            // Only a damaged state file can get here; the rows cannot be
            // trusted, and dropping them is the least wrong thing to do with
            // an accumulator whose scalars already counted them.
            self.pending.clear_rows();
            return;
        }

        // Suffix decays: suf[i] is the product of every lam after row i, so
        // u[i] is what the row is worth at the end of the block.
        let mut u = vec![1.0f64; n + 1];
        for i in (0..n).rev() {
            u[i] = u[i + 1] * self.pending.lam[i];
        }
        let lam_all = u[0];
        for i in 0..n {
            u[i] = self.pending.w[i] * u[i + 1];
        }
        u.truncate(n);
        let w_a = lam_all * self.pending.w_open;
        let w_new = self.w_sum;
        // `w_sum` is positive here, always: `step_factors` refuses any row
        // that would not leave it so, and no row reaches the buffer without
        // it. A capped gap (`lam = 0`) is handled there rather than here --
        // it does not empty the accumulator, it makes the row's own weight
        // the whole of it. Both mutators that could break the invariant
        // (`decay`, `set_moments`) flush first. Kept as an assertion because
        // the alternative is dividing by zero below.
        debug_assert!(
            w_new > 0.0,
            "EwCov::flush with no weight: step_factors should have refused every held row"
        );
        if w_new <= 0.0 || w_new.is_nan() {
            self.pending.clear_rows();
            return;
        }

        let u_sum: f64 = u.iter().sum();
        let Some(first) = u.iter().position(|&ui| ui > 0.0) else {
            // Held rows, no weight among them: the history only aged, and
            // `E_w` of a history that has merely aged is what it was.
            self.pending.clear_rows();
            return;
        };
        let x = &mut self.pending.x;

        // Pass one: the block's own mean, summed on rows shifted by one of
        // them so the sum is at the block's spread rather than its level.
        let shift: Vec<f64> = x[first * k..(first + 1) * k].to_vec();
        let mut m_b = vec![0.0f64; k];
        for (i, &ui) in u.iter().enumerate() {
            if ui > 0.0 {
                let row = &x[i * k..(i + 1) * k];
                for ((mb, &xi), &si) in m_b.iter_mut().zip(row).zip(&shift) {
                    *mb += ui * (xi - si);
                }
            }
        }
        for (mb, &si) in m_b.iter_mut().zip(&shift) {
            *mb = si + *mb / u_sum;
        }

        // Pass two, in place: each row becomes sqrt(u) * (x - m_B), so that
        // D^T D is S. A zero-weight row becomes a zero row (finite * 0). The
        // residue is taken before the scaling.
        let mut delta = vec![0.0f64; k];
        for (i, &ui) in u.iter().enumerate() {
            let row = &mut x[i * k..(i + 1) * k];
            for (xi, &mb) in row.iter_mut().zip(&m_b) {
                *xi -= mb;
            }
            if ui > 0.0 {
                for (di, &xi) in delta.iter_mut().zip(row.iter()) {
                    *di += ui * xi;
                }
            }
            // `sqrt` is one of the operations IEEE 754 rounds exactly, so
            // this is the same on every platform (unlike `powi` or `exp`,
            // which never reach the state).
            let root = ui.sqrt();
            for xi in row.iter_mut() {
                *xi *= root;
            }
        }
        for d in delta.iter_mut() {
            *d /= u_sum;
        }
        // Δ = m_B − m_A, with the residue folded in.
        let big: Vec<f64> = m_b
            .iter()
            .zip(self.m.iter())
            .zip(delta.iter())
            .map(|((mb, ma), d)| (mb - ma) + d)
            .collect();

        // C' on the lower triangle: scale the history, add the product,
        // then the two rank-1 corrections, then mirror.
        let scale = w_a / w_new;
        for i in 0..k {
            for j in 0..=i {
                self.c[i * k + j] *= scale;
            }
        }
        let d = MatRef::from_row_major_slice(x.as_slice(), n, k);
        let c = MatMut::from_row_major_slice_mut(self.c.as_mut_slice(), k, k);
        matmul(
            c,
            BlockStructure::TriangularLower,
            Accum::Add,
            d.transpose(),
            BlockStructure::Rectangular,
            d,
            BlockStructure::Rectangular,
            1.0 / w_new,
            Par::Seq,
        );
        let within = u_sum / w_new;
        let between = w_a * u_sum / (w_new * w_new);
        for i in 0..k {
            let (wi, bi) = (within * delta[i], between * big[i]);
            for j in 0..=i {
                self.c[i * k + j] += bi * big[j] - wi * delta[j];
            }
        }
        for i in 0..k {
            for j in 0..i {
                self.c[j * k + i] = self.c[i * k + j];
            }
        }
        let share = u_sum / w_new;
        for (mi, &bi) in self.m.iter_mut().zip(big.iter()) {
            *mi += share * bi;
        }
        self.pending.clear_rows();
    }

    /// One observation with decay factor `lam` (from [`crate::Decay::factor`])
    /// and row weight `w`. O(k^2) and allocation-free on the rank-1 path;
    /// with [`Self::set_block_rows`] on, the row is copied into the pending
    /// block and the `k^2` work is paid once per block by [`Self::flush`].
    ///
    /// `w` must be `>= 0`: these are weighted means, and a negative weight has
    /// no meaning here. Callers are expected to reject negative weights at the
    /// boundary (`online-polars` does, naming the offending row); a negative one
    /// reaching this far is a bug, so it is caught in debug builds and treated
    /// as a no-op in release rather than corrupting the accumulator.
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwCov::update requires a non-negative weight, got {w}"
        );
        if self.pending.block_rows > 0 {
            return self.buffer(x, lam, w);
        }
        let Some((a, b, w_new)) = self.step_factors(lam, w) else {
            return;
        };
        // Weighted Welford: co-moments are updated from the deviations against
        // the OLD mean, then the mean is advanced.
        //
        // The deviations are computed once into `dev` and the inner loop runs
        // over slices rather than indices (docs/ENHANCEMENTS.md E48). Both
        // matter and both are free: the old form recomputed `x[j] - m[j]`
        // once per *row* of the matrix -- `k^2` subtractions where `k` will
        // do -- and indexed `c`, `x` and `m` by `j`, whose bounds checks kept
        // the loop from vectorising. Same operations in the same order, so
        // every golden value is unchanged; 14% off at `k = 4`, 65% at
        // `k = 16`, 45% at `k = 64` and 24% from there up.
        let k = self.k;
        let d = &mut self.dev.0;
        d.clear();
        d.extend(x.iter().zip(self.m.iter()).map(|(xi, mi)| xi - mi));
        for i in 0..k {
            let ab_di = a * b * d[i];
            let row = &mut self.c[i * k..(i + 1) * k];
            for (cj, &dj) in row.iter_mut().zip(d.iter()) {
                *cj = a * *cj + ab_di * dj;
            }
        }
        // The precision prior decays with the co-moments. `a == 0` only on
        // the very first observation, when the whole history is discarded and
        // the centered co-moments are exactly zero: the prior starts over.
        for (mi, &di) in self.m.iter_mut().zip(d.iter()) {
            *mi += b * di;
        }
        self.commit_scalars(lam, w, a, w_new);
    }

    /// Overwrite the moments directly. Used when two accumulators are mixed
    /// (see `EwRidge::blend_toward_long_run`); the caller is responsible for
    /// the mixture being a valid set of weighted moments.
    pub fn set_moments(&mut self, mean: &[f64], centered: &[f64], w_sum: f64, q_sum: Option<f64>) {
        // Held rows are centred on the mean that was current when they were
        // held, and their merge reads `w_sum`. Replacing all three underneath
        // them would fold them into a state they were never measured against.
        // `EwRidge::blend_toward_long_run` is the caller that makes this
        // reachable the moment `ewridge` turns blocking on.
        self.flush();
        debug_assert_eq!(mean.len(), self.k);
        debug_assert_eq!(centered.len(), self.k * self.k);
        self.m.copy_from_slice(mean);
        self.c.copy_from_slice(centered);
        self.w_sum = w_sum;
        self.q_sum = q_sum;
    }

    /// Age the accumulator without adding data (pure decay: means unchanged,
    /// only the effective count shrinks).
    pub fn decay(&mut self, lam: f64) {
        // A held block snapshots `w_sum` as `w_open`; ageing `w_sum` under it
        // would leave the merge computing `W_A` from a weight that no longer
        // exists. Merge first, then age -- which is also the order the rows
        // arrived in.
        self.flush();
        self.w_sum *= lam;
        if let Some(q) = self.q_sum.as_mut() {
            *q *= lam * lam;
        }
        self.prior_scale *= lam;
    }

    /// Reference inverse of `C + prior·prior_scale·I` by Gauss-Jordan with
    /// partial pivoting, so the tests can check [`Self::precision`] against
    /// something that shares none of its code.
    #[cfg(test)]
    fn inverse_from_scratch(&self, prior: f64, prior_scale: f64) -> Option<Vec<f64>> {
        let k = self.k;
        let mut a = vec![0.0; k * 2 * k];
        for i in 0..k {
            for j in 0..k {
                a[i * 2 * k + j] = self.c[i * k + j];
            }
            a[i * 2 * k + i] += prior * prior_scale;
            a[i * 2 * k + k + i] = 1.0;
        }
        for col in 0..k {
            let piv = (col..k).max_by(|&r1, &r2| {
                a[r1 * 2 * k + col]
                    .abs()
                    .partial_cmp(&a[r2 * 2 * k + col].abs())
                    .unwrap()
            })?;
            if a[piv * 2 * k + col].abs() < 1e-300 {
                return None;
            }
            for j in 0..2 * k {
                a.swap(col * 2 * k + j, piv * 2 * k + j);
            }
            let d = a[col * 2 * k + col];
            for j in 0..2 * k {
                a[col * 2 * k + j] /= d;
            }
            for r in 0..k {
                if r != col {
                    let f = a[r * 2 * k + col];
                    for j in 0..2 * k {
                        a[r * 2 * k + j] -= f * a[col * 2 * k + j];
                    }
                }
            }
        }
        Some(
            (0..k)
                .flat_map(|i| (0..k).map(move |j| (i, j)))
                .map(|(i, j)| a[i * 2 * k + k + j])
                .collect(),
        )
    }
}

/// Per-target first and second moments, and `Sum w^2`, kept beside a model's
/// cross-moments (docs/ENHANCEMENTS.md E45).
///
/// The cross-moments alone are half of the sufficient statistic: without
/// `E[y]` and `Var[y]` no residual variance, `R^2`, information criterion or
/// standard error can be computed from a saved Gram. These are the other
/// half, in the same [`EwCov`] arithmetic and against the same per-target
/// weight `W_t`, so a target's variance here equals the variance an `ew_cov`
/// over that column would report, to the bit.
///
/// The moments are weighted *means*, `q` is a sum. Every field is per target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetMoments {
    /// EW mean of `y_t`.
    mean: Vec<f64>,
    /// EW **centered** variance of `y_t`.
    var: Vec<f64>,
    /// `Sum w^2` under `lam^2` decay, per target.
    q: Vec<f64>,
}

impl TargetMoments {
    pub fn new(n_targets: usize) -> Self {
        Self {
            mean: vec![0.0; n_targets],
            var: vec![0.0; n_targets],
            q: vec![0.0; n_targets],
        }
    }

    pub fn means(&self) -> &[f64] {
        &self.mean
    }

    pub fn vars(&self) -> &[f64] {
        &self.var
    }

    /// `Sum w^2` per target.
    pub fn q(&self) -> &[f64] {
        &self.q
    }

    /// Kish's effective sample size per target, `W_t^2 / Q_t`; `None` for a
    /// target that has not seen a weighted row yet.
    pub fn n_kish(&self, target_weights: &[f64]) -> Vec<Option<f64>> {
        self.q
            .iter()
            .zip(target_weights)
            .map(|(&q, &w)| (q > 0.0).then_some(w * w / q))
            .collect()
    }

    /// One present target, with the `a` and `b` its cross-moment update
    /// computed from the same `W_t` -- [`EwCov::update`]'s weighted Welford
    /// step, so the variance matches an `ew_cov` over the column to the bit.
    #[inline]
    pub fn learn(&mut self, t: usize, y: f64, a: f64, b: f64, lam: f64, w: f64) {
        let d = y - self.mean[t];
        self.var[t] = a * self.var[t] + a * b * d * d;
        self.mean[t] += b * d;
        self.q[t] = lam * lam * self.q[t] + w * w;
    }

    /// A row where this target is null: time passes for its weight, and
    /// `Q_t` decays with the square of it, as in [`EwCov::decay`].
    #[inline]
    pub fn age(&mut self, t: usize, lam: f64) {
        self.q[t] *= lam * lam;
    }

    /// Mix toward another set of moments with the same coefficients the
    /// caller mixes weights and co-moments by (`a + b == 1`); see
    /// `EwRidge::blend_toward_long_run`. Centered second moments are not
    /// additive across differing means, so the variance is mixed raw and
    /// re-centered on the mixed mean, exactly as the co-moments are.
    pub fn blend(&mut self, other: &Self, t: usize, a: f64, b: f64) {
        let mean = a * self.mean[t] + b * other.mean[t];
        let raw = a * (self.var[t] + self.mean[t] * self.mean[t])
            + b * (other.var[t] + other.mean[t] * other.mean[t]);
        self.var[t] = raw - mean * mean;
        self.mean[t] = mean;
        // `Q` is mixed by the same coefficients as the moments, not summed as
        // a union of two row sets would be: the twins see the *same* rows
        // under two halflives, so a union would count every row twice and
        // report a blend of a state with itself as more informative than the
        // state. With this form that blend is the identity, as it is for the
        // means, the co-moments and the weights.
        self.q[t] = a * self.q[t] + b * other.q[t];
    }
}

/// Partial correlation between columns `i` and `j` controlling for every other
/// column, read off a precision matrix `P` (row-major `k*k`, as
/// [`EwCov::precision`] returns it): `−P_ij / sqrt(P_ii · P_jj)`.
pub fn partial_corr(precision: &[f64], k: usize, i: usize, j: usize) -> f64 {
    let p = precision;
    let (pij, pii, pjj) = (p[i * k + j], p[i * k + i], p[j * k + j]);
    // `sqrt(pii) * sqrt(pjj)`, not `sqrt(pii * pjj)`: the product overflows
    // at 1e154 and underflows at 1e-154 where the factors are fine.
    let d = pii.sqrt() * pjj.sqrt();
    if d > 0.0 {
        (-pij / d).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Which statistics an [`EwCovModel`] emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EwCovStat {
    /// EW mean of each column.
    Mean,
    /// EW variance of each column (centered).
    Var,
    /// EW standard deviation of each column.
    Std,
    /// Centered covariance for each unordered pair `i < j`.
    Cov,
    /// Pearson correlation for each unordered pair `i < j`.
    Corr,
    /// Partial correlation for each unordered pair, controlling for every other
    /// column. Needs `precision_prior`.
    PartialCorr,
    /// Mahalanobis distance of the row from the decayed history,
    /// `sqrt((x − μ)ᵀ (C + s·prior·I)⁻¹ (x − μ))`, read before the row is
    /// learned (docs/ENHANCEMENTS.md E37). One slot. Needs `precision_prior`.
    Mahal,
    /// Lagged cross-correlation `C_ℓ[a,b] / √(C₀[a,a]·C₀[b,b])` for every
    /// lag and **ordered** pair, the auto terms included: `k²` slots a lag
    /// (docs/ENHANCEMENTS.md E56). The matrix is not symmetric for `ℓ > 0`
    /// -- `a` leading `b` is not `b` leading `a` -- which is why both
    /// orientations are emitted. Needs `lags`.
    LagCorr,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EwCovCfg {
    pub n_features: usize,
    pub decay: crate::Decay,
    /// Statistics to emit per row, in this order. Empty is legal and means
    /// **accumulate only** (docs/ENHANCEMENTS.md E43): the model learns the
    /// same moments, emits nothing but `n_eff`, and its value is its state
    /// — the Gram read back through [`EwCovModel::cov`], or the PCA and
    /// Mahalanobis slots below, which do not need a statistic in this list.
    pub stats: Vec<EwCovStat>,
    pub min_periods: f64,
    /// Prior for the precision matrix `(C + s·prior·I)⁻¹`. Required by
    /// [`EwCovStat::PartialCorr`] and [`EwCovStat::Mahal`], its consumers.
    #[serde(default)]
    pub precision_prior: Option<f64>,
    /// P² quantile levels of the past Mahalanobis scores, one slot each
    /// (`mahal_q<p>`): a threshold for [`EwCovStat::Mahal`] from its own
    /// history rather than from a χ² table. Needs `Mahal` in `stats`.
    #[serde(default)]
    pub mahal_quantiles: Vec<f64>,
    /// Number of principal components to track, `0` for none
    /// (docs/ENHANCEMENTS.md E38). Each component adds `k + 3` slots: its
    /// variance, its share of the total, its `k` loadings and the row's score.
    #[serde(default)]
    pub pca: usize,
    /// Learned rows between refreshes of the components; between refreshes
    /// the loadings are frozen, so a row's scores do not depend on how the
    /// stream was chunked. `1` refreshes on every row.
    #[serde(default)]
    pub pca_every: usize,
    /// Lags to accumulate cross-moments at, in output order
    /// (docs/ENHANCEMENTS.md E56): strictly increasing and `>= 1`. Empty for
    /// none, which is what every accumulator did before task 48 -- so a
    /// state written then still loads, with no ring.
    #[serde(default)]
    pub lags: Vec<usize>,
    /// Clock units of history the statistics are computed from, with a *hard*
    /// cutoff: a row older than this contributes nothing at all, where the
    /// exponential weight alone would leave `0.5^(age/halflife)` of it
    /// (docs/PLAN.md §13). Inside the window the weights are still
    /// exponential -- this is not a flat window. Absent for the ordinary
    /// accumulator, which is what every state written before task 63 has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<f64>,
    /// Learned rows between the snapshots the window is computed from; `1`
    /// (the default) snapshots every row and puts the boundary as close to
    /// `window` as the data allows. Larger divides the memory by the same
    /// factor and moves the boundary *inward*, so the effective window is in
    /// `[window - the snapshot spacing, window]` -- never longer than asked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_every: Option<usize>,
}

impl EwCovCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("ew_cov: at least one column is required".into());
        }
        let pairwise =
            |s: &EwCovStat| matches!(s, EwCovStat::Cov | EwCovStat::Corr | EwCovStat::PartialCorr);
        if self.n_features < 2 && self.stats.iter().any(pairwise) {
            return Err("ew_cov: cov/corr/partial_corr need at least two columns".into());
        }
        if self.stats.contains(&EwCovStat::PartialCorr) && self.precision_prior.is_none() {
            return Err(
                "ew_cov: partial_corr needs `precision_prior` (it is computed from the \
                 regularized precision matrix)"
                    .into(),
            );
        }
        if self.stats.contains(&EwCovStat::Mahal) && self.precision_prior.is_none() {
            return Err(
                "ew_cov: mahal needs `precision_prior` (the co-moments are singular until \
                 k independent rows have been seen)"
                    .into(),
            );
        }
        if !self.mahal_quantiles.is_empty() && !self.stats.contains(&EwCovStat::Mahal) {
            return Err("ew_cov: mahal_quantiles needs \"mahal\" in `stats`".into());
        }
        if let Some(w) = self.window {
            if !w.is_finite() || w <= 0.0 {
                return Err(format!(
                    "ew_cov: window must be finite and > 0 (got {w}); it is clock units of \
                     history to keep, and `None` is the ordinary accumulator that keeps all of it"
                ));
            }
            if self.window_every.is_some_and(|e| e == 0) {
                return Err("ew_cov: window_every must be >= 1".into());
            }
            if !self.lags.is_empty() {
                return Err(
                    "ew_cov: window and lags do not combine yet; the lagged co-moments are a \
                     second accumulator with its own ring, and truncating one and not the \
                     other would report a windowed corr beside an unwindowed lagcorr"
                        .into(),
                );
            }
            if !self.mahal_quantiles.is_empty() {
                return Err(
                    "ew_cov: window and mahal_quantiles do not combine; the P^2 quantiles are \
                     accumulated over every score the stream has produced, so thresholding a \
                     windowed distance against them compares two different histories"
                        .into(),
                );
            }
        } else if self.window_every.is_some() {
            return Err("ew_cov: window_every needs `window`".into());
        }
        if self.stats.contains(&EwCovStat::LagCorr) && self.lags.is_empty() {
            return Err(
                "ew_cov: lagcorr needs `lags` (which lags to accumulate, e.g. lags = [1, 2, 5])"
                    .into(),
            );
        }
        if !self.lags.is_empty() {
            // The list's own rules live with the accumulator, so the CLI, the
            // bank and the plugin all get one message.
            crate::EwLagCov::new(self.n_features, self.lags.clone())
                .map(|_| ())
                .map_err(|e| format!("ew_cov: {e}"))?;
        }
        for &q in &self.mahal_quantiles {
            if !(q > 0.0 && q < 1.0) {
                return Err(format!(
                    "ew_cov: mahal_quantiles must be strictly between 0 and 1, got {q}"
                ));
            }
        }
        if self.pca > self.n_features {
            return Err(format!(
                "ew_cov: pca asks for {} components of {} columns",
                self.pca, self.n_features
            ));
        }
        if self.pca > 0 && self.pca_every == 0 {
            return Err("ew_cov: pca_every must be >= 1".into());
        }
        Ok(())
    }

    /// Number of output slots, in emission order: the statistics, then the
    /// Mahalanobis quantiles, then `k + 3` per principal component.
    pub fn n_outputs(&self) -> usize {
        let k = self.n_features;
        let pairs = k * (k - 1) / 2;
        let stats: usize = self
            .stats
            .iter()
            .map(|s| match s {
                EwCovStat::Mean | EwCovStat::Var | EwCovStat::Std => k,
                EwCovStat::Cov | EwCovStat::Corr | EwCovStat::PartialCorr => pairs,
                EwCovStat::Mahal => 1,
                // Both orientations and the auto terms: the lagged matrix is
                // not symmetric.
                EwCovStat::LagCorr => self.lags.len() * k * k,
            })
            .sum();
        stats + self.mahal_quantiles.len() + self.pca * (k + 3)
    }
}

/// The frozen principal components between two refreshes
/// (docs/ENHANCEMENTS.md E38): the top `r` eigenpairs of the centred
/// co-moment matrix, largest first. An eigenvector's sign is arbitrary, so
/// each loading vector is signed for continuity with the previous refresh
/// (`v_new · v_old >= 0`); the first refresh makes its largest-magnitude
/// entry positive. Continuity is what keeps a score series from flipping
/// sign between two rows when two loadings of nearly equal size trade
/// places as the largest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Pca {
    /// Eigenvalues, descending: the variance along each component.
    pub eig: Vec<f64>,
    /// The trace of the co-moments, the total variance the shares are of.
    pub trace: f64,
    /// Loadings, row-major `r * k`: row `j` is component `j`'s unit vector.
    pub loadings: Vec<f64>,
}

impl Pca {
    /// The top `r` components of the symmetric matrix `c` (row-major `k*k`)
    /// by `faer`'s self-adjoint eigensolver, which is deterministic, signed
    /// for continuity with `prev` when it has the component. `None` when
    /// `c` has a non-finite entry or the decomposition fails.
    pub fn of(c: &[f64], k: usize, r: usize, prev: Option<&Pca>) -> Option<Self> {
        use faer::Side;
        use faer::prelude::*;
        if r == 0 || r > k || c.len() != k * k || c.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mat = Mat::from_fn(k, k, |i, j| c[i * k + j]);
        let evd = mat.self_adjoint_eigen(Side::Lower).ok()?;
        let (s, u) = (evd.S(), evd.U());
        // Eigenvalues come nondecreasing; take the last `r`, largest first.
        let mut eig = Vec::with_capacity(r);
        let mut loadings = Vec::with_capacity(r * k);
        for j in 0..r {
            let col = k - 1 - j;
            eig.push(s[col]);
            let v: Vec<f64> = (0..k).map(|i| u[(i, col)]).collect();
            let along_prev = prev
                .filter(|p| p.r() > j && p.loadings.len() == p.r() * k)
                .map(|p| {
                    p.loadings[j * k..(j + 1) * k]
                        .iter()
                        .zip(&v)
                        .fold(0.0, |acc, (a, b)| acc + a * b)
                });
            let sign = match along_prev {
                Some(d) if d != 0.0 => {
                    if d < 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                // No predecessor (or one exactly orthogonal): the
                // largest-magnitude entry, the first on a tie, is positive.
                _ => {
                    let mut lead = 0;
                    for (i, vi) in v.iter().enumerate() {
                        if vi.abs() > v[lead].abs() {
                            lead = i;
                        }
                    }
                    if v[lead] < 0.0 { -1.0 } else { 1.0 }
                }
            };
            loadings.extend(v.iter().map(|vi| sign * vi));
        }
        let trace = (0..k).map(|i| c[i * k + i]).sum();
        let out = Self {
            eig,
            trace,
            loadings,
        };
        (out.eig.iter().all(|v| v.is_finite()) && out.loadings.iter().all(|v| v.is_finite()))
            .then_some(out)
    }

    pub fn r(&self) -> usize {
        self.eig.len()
    }
}

/// [`EwCov`] exposed as a model in its own right (docs/PLAN.md §4.7), so EW
/// means, variances, covariances and correlations are available as outputs
/// without fitting a regression. Replaces pure-Polars pairwise EW correlations,
/// which cost O(k²) passes; this is one O(k²) update per row.
///
/// It has no targets: every column is a "feature", and the statistics are
/// reported from the state *before* each row, like every prediction here.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EwCovModel {
    cfg: EwCovCfg,
    cov: EwCov,
    /// P² estimators of the past Mahalanobis scores, one per level.
    #[serde(default)]
    mahal_q: Vec<crate::P2Quantile>,
    /// The components in force, refreshed every `pca_every` learned rows.
    #[serde(default)]
    pca: Option<Pca>,
    /// Learned rows since the last refresh.
    #[serde(default)]
    since_pca: usize,
    /// Lagged cross-moments, when the spec asks for lags (E56). Skipped when
    /// absent, so an accumulator without them writes the bytes it always
    /// did, and a state written before task 48 loads with `None` and starts
    /// a fresh ring under a spec that has since gained `lags`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lag: Option<crate::EwLagCov>,
    /// The hard-cutoff window, when the spec asks for one (docs/PLAN.md §13).
    /// Absent otherwise, so an ordinary accumulator writes the bytes it
    /// always did and a state written before task 63 loads unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
}

/// What a window needs beyond the accumulator: where the clock has got to,
/// and the snapshots to subtract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Windowed {
    /// Clock of the last learned row, which is the reference the statistics
    /// are read at (the state is pre-decay for the row being reported on).
    clock: f64,
    snaps: crate::Snapshots<crate::Moments>,
}

impl EwCovModel {
    pub fn new(cfg: EwCovCfg) -> Result<Self, String> {
        cfg.validate()?;
        let cfg_window = cfg.window;
        let every = cfg.window_every.unwrap_or(1);
        let cov = match cfg.precision_prior {
            Some(p) => EwCov::with_precision_prior(cfg.n_features, p)?,
            None => EwCov::new(cfg.n_features),
        };
        let mahal_q = cfg
            .mahal_quantiles
            .iter()
            .map(|&q| crate::P2Quantile::new(q))
            .collect::<Result<Vec<_>, _>>()?;
        let lag = if cfg.lags.is_empty() {
            None
        } else {
            Some(crate::EwLagCov::new(cfg.n_features, cfg.lags.clone())?)
        };
        Ok(Self {
            cfg,
            cov,
            mahal_q,
            pca: None,
            since_pca: 0,
            lag,
            win: match cfg_window {
                Some(w) => Some(Windowed {
                    clock: 0.0,
                    snaps: crate::Snapshots::new(w, every)?,
                }),
                None => None,
            },
        })
    }

    /// The accumulator the statistics are read from: the live one, or -- with
    /// a `window` -- the same accumulator with everything older than the
    /// window subtracted off.
    ///
    /// The subtraction is exact rather than approximate, because an EW sum
    /// contains its own past: everything at or before the boundary's clock
    /// `u` is `lam^(t-u)` times the accumulator as it stood then, so
    /// `A(t) - lam^(t-u)·A(u)` is precisely the rest. The boundary is the
    /// oldest snapshot still inside the window, so the rows dropped are a
    /// superset of the rows the window excludes -- the guarantee is honoured
    /// exactly and the shortfall is at most one snapshot's spacing.
    ///
    /// Two things this does *not* carry over from the live accumulator: the
    /// precision prior fades with the whole stream's decay rather than the
    /// window's, and the subtraction is a difference of positives, so its
    /// precision falls with the fraction discarded (negligible at
    /// `window = 3·halflife`, where the correction is an eighth).
    fn view(&self) -> std::borrow::Cow<'_, EwCov> {
        use std::borrow::Cow;
        let Some(win) = self.win.as_ref() else {
            return Cow::Borrowed(&self.cov);
        };
        let Some((u, old)) = win.snaps.boundary() else {
            return Cow::Borrowed(&self.cov);
        };
        if old.w == 0.0 {
            // The boundary carries no weight: nothing has aged out yet. Read
            // the accumulator itself rather than recomputing it from a
            // subtraction of zero, so a window nothing has reached reports
            // the untruncated numbers to the bit.
            return Cow::Borrowed(&self.cov);
        }
        let f = self.cfg.decay.factor(win.clock - u);
        match crate::truncated(&self.cov, old, f) {
            Some(cov) => Cow::Owned(cov),
            None => {
                // Nothing inside the window: a clock gap longer than it.
                // Report an empty state rather than dividing by it.
                let k = self.cfg.n_features;
                let mut out = self.cov.clone();
                out.set_moments(&vec![0.0; k], &vec![0.0; k * k], 0.0, old.q.map(|_| 0.0));
                Cow::Owned(out)
            }
        }
    }

    /// The components currently in force, if a refresh has happened.
    pub fn pca(&self) -> Option<&Pca> {
        self.pca.as_ref()
    }

    /// The Mahalanobis distance of `x` from the current moments, or NaN when
    /// no prior is configured or the solve fails. One Cholesky factorization
    /// and one triangular solve, O(k³).
    pub fn mahal(&self, x: &[f64]) -> f64 {
        if !self.cov.has_precision_prior() {
            return f64::NAN;
        }
        let k = self.cov.k();
        let delta: Vec<f64> = x
            .iter()
            .zip(self.cov.means())
            .map(|(xi, mi)| xi - mi)
            .collect();
        let mut m = self.cov.comoments().to_vec();
        let ridge = self.cov.precision_prior * self.cov.precision_scale;
        for i in 0..k {
            m[i * k + i] += ridge;
        }
        match crate::solve::solve_spd(&m, &delta, k, 1) {
            Some((sol, _)) => {
                let d2: f64 = delta.iter().zip(&sol).map(|(d, s)| d * s).sum();
                if d2.is_finite() {
                    d2.max(0.0).sqrt()
                } else {
                    f64::NAN
                }
            }
            None => f64::NAN,
        }
    }

    /// Recompute the components from the current co-moments. A failed
    /// decomposition keeps the previous ones.
    fn refresh_pca(&mut self) {
        if let Some(p) = Pca::of(
            self.cov.comoments(),
            self.cfg.n_features,
            self.cfg.pca,
            self.pca.as_ref(),
        ) {
            self.pca = Some(p);
        }
        self.since_pca = 0;
    }

    /// The accumulator itself, for callers that want the whole matrix rather
    /// than the pairwise statistics (docs/ENHANCEMENTS.md E30). At `k = 400`
    /// the pairwise form is 79,800 struct fields; this is one `k*k` array.
    pub fn cov(&self) -> &EwCov {
        &self.cov
    }

    /// The lagged cross-moments, when the spec asked for lags (E56).
    pub fn lag(&self) -> Option<&crate::EwLagCov> {
        self.lag.as_ref()
    }

    pub fn cfg(&self) -> &EwCovCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.cov.n_eff()
    }

    /// Output slot labels, in emission order (used for field names): the
    /// statistics, then `mahal_q<p>` per quantile level, then per component
    /// `j`: `pc<j>_var`, `pc<j>_share`, `pc<j>_<column>` for each column and
    /// `pc<j>_score`.
    pub fn labels(
        names: &[String],
        stats: &[EwCovStat],
        mahal_quantiles: &[f64],
        pca: usize,
        lags: &[usize],
    ) -> Vec<String> {
        let mut out = Vec::new();
        for stat in stats {
            match stat {
                EwCovStat::Mean => out.extend(names.iter().map(|n| format!("mean_{n}"))),
                EwCovStat::Var => out.extend(names.iter().map(|n| format!("var_{n}"))),
                EwCovStat::Std => out.extend(names.iter().map(|n| format!("std_{n}"))),
                EwCovStat::Cov => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("cov_{}_{}", names[i], names[j]));
                        }
                    }
                }
                EwCovStat::Corr => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("corr_{}_{}", names[i], names[j]));
                        }
                    }
                }
                EwCovStat::PartialCorr => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("pcorr_{}_{}", names[i], names[j]));
                        }
                    }
                }
                EwCovStat::Mahal => out.push("mahal".to_string()),
                EwCovStat::LagCorr => {
                    for l in lags {
                        for a in names {
                            for b in names {
                                out.push(format!("lagcorr_{a}_{b}_l{l}"));
                            }
                        }
                    }
                }
            }
        }
        for q in mahal_quantiles {
            out.push(format!("mahal_q{q}"));
        }
        for j in 0..pca {
            out.push(format!("pc{j}_var"));
            out.push(format!("pc{j}_share"));
            out.extend(names.iter().map(|n| format!("pc{j}_{n}")));
            out.push(format!("pc{j}_score"));
        }
        out
    }

    fn read(&self, cov: &EwCov, x: &[f64]) -> Vec<f64> {
        let k = self.cfg.n_features;
        let mut out = Vec::with_capacity(self.cfg.n_outputs());
        // One O(k³) solve per row, only when partial correlations are wanted.
        let precision = self
            .cfg
            .stats
            .contains(&EwCovStat::PartialCorr)
            .then(|| cov.precision())
            .flatten();
        for stat in &self.cfg.stats {
            match stat {
                EwCovStat::Mean => (0..k).for_each(|i| out.push(cov.mean(i))),
                EwCovStat::Var => (0..k).for_each(|i| out.push(cov.var(i))),
                EwCovStat::Std => (0..k).for_each(|i| out.push(cov.var(i).sqrt())),
                EwCovStat::Cov => {
                    for i in 0..k {
                        for j in (i + 1)..k {
                            out.push(cov.cov(i, j));
                        }
                    }
                }
                EwCovStat::Corr => {
                    // `k` square roots, not `k(k-1)`: at k = 20 the pairwise
                    // `sqrt` was 80% of `ew_cov`'s row (docs/PERFORMANCE.md
                    // §13). Same product of the same two roots, so the
                    // correlations are bit-identical.
                    let std: Vec<f64> = (0..k).map(|i| cov.var(i).sqrt()).collect();
                    for i in 0..k {
                        for j in (i + 1)..k {
                            let d = std[i] * std[j];
                            out.push(if d > 0.0 {
                                (cov.cov(i, j) / d).clamp(-1.0, 1.0)
                            } else {
                                f64::NAN
                            });
                        }
                    }
                }
                EwCovStat::PartialCorr => {
                    for i in 0..k {
                        for j in (i + 1)..k {
                            out.push(match &precision {
                                Some(p) => partial_corr(p, k, i, j),
                                None => f64::NAN,
                            });
                        }
                    }
                }
                EwCovStat::Mahal => out.push(self.mahal(x)),
                EwCovStat::LagCorr => {
                    // `C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`: the lagged
                    // covariance over the contemporaneous standard
                    // deviations, so a value of 1 means "a now is b `l` rows
                    // ago, scaled". Not clamped: unlike a contemporaneous
                    // correlation it is not bounded by 1 in finite samples,
                    // and clamping would hide that.
                    let std: Vec<f64> = (0..k).map(|i| cov.var(i).sqrt()).collect();
                    match &self.lag {
                        Some(lag) => {
                            for li in 0..lag.lags().len() {
                                for a in 0..k {
                                    for b in 0..k {
                                        let d = std[a] * std[b];
                                        out.push(if d > 0.0 {
                                            lag.get(li, a, b) / d
                                        } else {
                                            f64::NAN
                                        });
                                    }
                                }
                            }
                        }
                        None => {
                            out.extend(std::iter::repeat_n(f64::NAN, self.cfg.lags.len() * k * k))
                        }
                    }
                }
            }
        }
        for q in &self.mahal_q {
            out.push(q.get().unwrap_or(f64::NAN));
        }
        if self.cfg.pca > 0 {
            match &self.pca {
                Some(p) => {
                    for j in 0..p.r() {
                        let v = &p.loadings[j * k..(j + 1) * k];
                        out.push(p.eig[j]);
                        out.push(if p.trace > 0.0 {
                            p.eig[j] / p.trace
                        } else {
                            f64::NAN
                        });
                        out.extend_from_slice(v);
                        // The row's coordinate along the component, about the
                        // current mean: loadings frozen, centre live.
                        let score = v
                            .iter()
                            .zip(x)
                            .zip(cov.means())
                            .fold(0.0, |acc, ((vi, xi), mi)| acc + vi * (xi - mi));
                        out.push(score);
                    }
                }
                None => out.extend(std::iter::repeat_n(f64::NAN, self.cfg.pca * (k + 3))),
            }
        }
        out
    }
}

impl crate::OnlineModel for EwCovModel {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        // Statistics are read before this row is folded in, so an `ew_cov`
        // column is usable as a feature for the same row without leaking it.
        let out = self.predict(x, d_clock);
        if !self.mahal_q.is_empty() {
            // The row's own out-of-sample score joins the history it will be
            // thresholded against, as `resid_quantiles` does for |resid|.
            let slot = self
                .cfg
                .stats
                .iter()
                .take_while(|s| **s != EwCovStat::Mahal)
                .map(|s| match s {
                    EwCovStat::Mean | EwCovStat::Var | EwCovStat::Std => self.cfg.n_features,
                    _ => self.cfg.n_features * (self.cfg.n_features - 1) / 2,
                })
                .sum::<usize>();
            let score = out.pred[slot];
            if score.is_finite() {
                for q in &mut self.mahal_q {
                    q.update(score);
                }
            }
        }
        let lam = self.cfg.decay.factor(d_clock);
        // Before the accumulator moves: the lagged update reads the same
        // pre-row weight and mean `EwCov::update` is about to consume, which
        // is what makes `l = 0` its co-moments to the bit.
        if let Some(lag) = self.lag.as_mut() {
            lag.update(x, self.cov.means(), self.cov.n_eff(), lam, weight);
        }
        // The snapshot is the state *before* this row, decayed to this row's
        // clock, so subtracting it later retains this row and everything
        // after it. Taken before the update, and keyed by a clock the model
        // accumulates itself, so the cadence is counted in the stream and the
        // boundary cannot depend on how the data was chunked (hard rule 3).
        if let Some(win) = self.win.as_mut() {
            let t = win.clock + d_clock;
            let snap = crate::Moments::of(&self.cov, lam);
            win.snaps.offer(t, || snap);
            win.clock = t;
            win.snaps.trim(t);
        }
        self.cov.update(x, lam, weight);
        if self.cfg.pca > 0 {
            // A checkpoint after the update, counted in learned rows, so the
            // components a row is scored on never depend on the chunking and
            // `predict` sees the same frozen ones `step` does.
            self.since_pca += 1;
            if self.cov.n_eff() >= self.cfg.min_periods
                && (self.pca.is_none() || self.since_pca >= self.cfg.pca_every)
            {
                self.refresh_pca();
            }
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> crate::Step {
        // With a `window`, every statistic -- and `n_eff` with them -- comes
        // from the truncated accumulator, so `min_periods` gates on the
        // weight *inside* the window, which stops growing once the window is
        // full rather than rising for the life of the stream.
        let view = self.view();
        let n_eff = view.n_eff();
        let pred = if n_eff >= self.cfg.min_periods {
            self.read(&view, x)
        } else {
            vec![f64::NAN; self.cfg.n_outputs()]
        };
        crate::Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::EwCovModel(Box::new(self.clone())))
    }

    fn clear_lags(&mut self) {
        if let Some(lag) = self.lag.as_mut() {
            lag.clear();
        }
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::EwCovModel(m) => {
                let mut m = (**m).clone();
                // A state written before task 48, resumed under a spec that
                // has since gained `lags`: the ring cannot be recovered, so
                // it starts empty and the matrices from zero.
                if m.lag.is_none() && !m.cfg.lags.is_empty() {
                    m.lag = Some(
                        crate::EwLagCov::new(m.cfg.n_features, m.cfg.lags.clone())
                            .map_err(crate::StateError::Invalid)?,
                    );
                }
                Ok(m)
            }
            other => Err(crate::StateError::WrongModel {
                expected: "ew_cov",
                found: other.kind(),
            }),
        }
    }

    /// `ew_cov` has no targets; one nominal target carries all the slots.
    /// Zero: `ew_cov` regresses nothing, it reports statistics. The slot count
    /// callers want is [`EwCovCfg::n_outputs`], which is not `n_targets`-derived
    /// here as it is for every other model.
    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_outputs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct O(n^2) recomputation with explicit decayed weights.
    fn direct(xs: &[Vec<f64>], lams: &[f64], ws: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let n = xs.len();
        let k = xs[0].len();
        let mut wts = vec![0.0; n];
        for i in 0..n {
            let mut wi = ws[i];
            for lam in &lams[i + 1..] {
                wi *= lam;
            }
            wts[i] = wi;
        }
        let wsum: f64 = wts.iter().sum();
        let mut m = vec![0.0; k];
        let mut s = vec![0.0; k * k];
        for (x, &wi) in xs.iter().zip(&wts) {
            for i in 0..k {
                m[i] += wi * x[i] / wsum;
                for j in 0..k {
                    s[i * k + j] += wi * x[i] * x[j] / wsum;
                }
            }
        }
        (wsum, m, s)
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn model_cfg(k: usize, stats: Vec<EwCovStat>) -> EwCovCfg {
        EwCovCfg {
            n_features: k,
            decay: crate::Decay::Halflife(f64::INFINITY),
            stats,
            min_periods: 2.0,
            precision_prior: None,
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0,
            lags: Vec::new(),
            window: None,
            window_every: None,
        }
    }

    /// A windowed weighted mean and covariance, written from the definition:
    /// every row inside the window, weighted by `lam^age`, and nothing else.
    /// `t` are absolute clocks; the report at row `n` is referenced at the
    /// clock of row `n-1`, as every statistic in this library is.
    fn direct_window(
        xs: &[Vec<f64>],
        t: &[f64],
        halflife: f64,
        window: f64,
        upto: usize,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let k = xs[0].len();
        let now = t[upto - 1];
        let mut w_sum = 0.0;
        let mut m = vec![0.0; k];
        let mut raw = vec![0.0; k * k];
        for i in 0..upto {
            if now - t[i] >= window {
                continue;
            }
            let w = 0.5_f64.powf((now - t[i]) / halflife);
            w_sum += w;
            for a in 0..k {
                m[a] += w * xs[i][a];
                for b in 0..k {
                    raw[a * k + b] += w * xs[i][a] * xs[i][b];
                }
            }
        }
        for v in m.iter_mut() {
            *v /= w_sum;
        }
        let mut cen = vec![0.0; k * k];
        for a in 0..k {
            for b in 0..k {
                cen[a * k + b] = raw[a * k + b] / w_sum - m[a] * m[b];
            }
        }
        (w_sum, m, cen)
    }

    /// PLAN §13.4 (1): the truncated view against a direct windowed sum.
    #[test]
    fn a_window_reports_exactly_the_rows_inside_it() {
        let halflife = 40.0;
        for &window in &[15.0, 60.0, 150.0] {
            let mut cfg = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Cov]);
            cfg.decay = crate::Decay::Halflife(halflife);
            cfg.min_periods = 0.0;
            cfg.window = Some(window);
            let mut m = EwCovModel::new(cfg).unwrap();

            let (mut xs, mut t, mut clock) = (Vec::new(), Vec::new(), 0.0);
            let mut seed = 12345u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f64) / (u32::MAX as f64)
            };
            for i in 0..120 {
                let d = if i == 0 { 0.0 } else { 1.0 + 6.0 * rnd() };
                clock += d;
                let x = vec![rnd() * 4.0 - 2.0, rnd() * 10.0 + 100.0];
                if i > 0 {
                    // The report is referenced at the previous row's clock.
                    let got = crate::OnlineModel::predict(&m, &x, d);
                    let (w, mean, cen) = direct_window(&xs, &t, halflife, window, i);
                    assert!(
                        (got.n_eff - w).abs() < 1e-9 * w.max(1.0),
                        "window {window} row {i}: n_eff {} vs {w}",
                        got.n_eff
                    );
                    for a in 0..2 {
                        assert!(
                            (got.pred[a] - mean[a]).abs() < 1e-9,
                            "window {window} row {i} mean[{a}]: {} vs {}",
                            got.pred[a],
                            mean[a]
                        );
                        let var = cen[a * 2 + a];
                        assert!(
                            (got.pred[2 + a] - var).abs() < 1e-8 * var.abs().max(1.0),
                            "window {window} row {i} var[{a}]: {} vs {var}",
                            got.pred[2 + a]
                        );
                    }
                    assert!(
                        (got.pred[4] - cen[1]).abs() < 1e-8 * cen[1].abs().max(1.0),
                        "window {window} row {i} cov: {} vs {}",
                        got.pred[4],
                        cen[1]
                    );
                }
                crate::OnlineModel::step(&mut m, &x, &[], d, 1.0);
                xs.push(x);
                t.push(clock);
            }
        }
    }

    /// PLAN §13.4 (2): the guarantee itself. Rows older than the window are
    /// replaced with a number that would dominate any average they still
    /// touched.
    #[test]
    fn nothing_older_than_the_window_can_move_the_answer() {
        let run = |poison: bool| {
            let mut cfg = model_cfg(1, vec![EwCovStat::Mean, EwCovStat::Var]);
            cfg.decay = crate::Decay::Halflife(20.0);
            cfg.min_periods = 0.0;
            cfg.window = Some(60.0);
            let mut m = EwCovModel::new(cfg).unwrap();
            let mut last = crate::Step {
                pred: vec![],
                n_eff: 0.0,
                extra: None,
            };
            for i in 0..200 {
                // 200 rows one clock unit apart. The last report is read at
                // the clock of row 198 and its 60-unit window covers rows 139
                // and after, so rows before 130 are outside it with room to
                // spare. (Poisoning from 140 puts 1e6 *inside* the window,
                // which is what the first version of this test did.)
                let inside = i >= 130;
                let x = if poison && !inside {
                    vec![1e6]
                } else {
                    vec![(i as f64 * 0.37).sin()]
                };
                last =
                    crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            last
        };
        let clean = run(false);
        let dirty = run(true);
        assert!(
            (clean.pred[0] - dirty.pred[0]).abs() < 1e-8,
            "1e6 outside the window moved the mean: {} vs {}",
            clean.pred[0],
            dirty.pred[0]
        );
        assert!(
            (clean.n_eff - dirty.n_eff).abs() < 1e-9,
            "n_eff moved: {} vs {}",
            clean.n_eff,
            dirty.n_eff
        );
    }

    /// PLAN §13.4 (5): a window longer than the stream is the ordinary
    /// accumulator, slot for slot, so the feature cannot change what an
    /// existing spec reports.
    #[test]
    fn a_window_no_stream_reaches_is_the_untruncated_model() {
        let mk = |window: Option<f64>| {
            let mut cfg = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr]);
            cfg.decay = crate::Decay::Halflife(25.0);
            cfg.min_periods = 0.0;
            cfg.window = window;
            EwCovModel::new(cfg).unwrap()
        };
        let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
        for i in 0..80 {
            let x = vec![(i as f64 * 0.7).sin(), (i as f64 * 0.31).cos() * 3.0];
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = crate::OnlineModel::step(&mut plain, &x, &[], d, 1.0);
            let b = crate::OnlineModel::step(&mut windowed, &x, &[], d, 1.0);
            assert!(
                same_bits(&a.pred, &b.pred),
                "row {i}: {:?} vs {:?}",
                a.pred,
                b.pred
            );
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i} n_eff");
        }
    }

    /// `window_every` trades memory for a boundary that can only move
    /// *inward*: a coarser cadence never keeps a row the window excludes.
    #[test]
    fn a_coarse_cadence_discards_more_and_never_less() {
        let mk = |every: usize| {
            let mut cfg = model_cfg(1, vec![EwCovStat::Mean]);
            cfg.decay = crate::Decay::Halflife(30.0);
            cfg.min_periods = 0.0;
            cfg.window = Some(50.0);
            cfg.window_every = Some(every);
            EwCovModel::new(cfg).unwrap()
        };
        let (mut fine, mut coarse) = (mk(1), mk(10));
        for i in 0..300 {
            // A step function: old rows are 0, recent rows are 1. A window
            // that reaches further back reports a *smaller* mean.
            let x = vec![if i < 200 { 0.0 } else { 1.0 }];
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = crate::OnlineModel::step(&mut fine, &x, &[], d, 1.0);
            let b = crate::OnlineModel::step(&mut coarse, &x, &[], d, 1.0);
            if i > 210 {
                assert!(
                    b.pred[0] >= a.pred[0] - 1e-12,
                    "row {i}: the coarse window kept more of the old regime: {} < {}",
                    b.pred[0],
                    a.pred[0]
                );
                assert!(
                    b.n_eff <= a.n_eff + 1e-12,
                    "row {i}: coarse n_eff is larger"
                );
            }
        }
    }

    #[test]
    fn model_cfg_validation_rejects_each_bad_field() {
        use EwCovStat::*;
        let err = |c: EwCovCfg, want: &str| match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        };

        err(model_cfg(0, vec![Mean]), "at least one column");
        // No statistics is "accumulate only" (E43), not an error.
        model_cfg(2, vec![]).validate().unwrap();
        assert_eq!(model_cfg(2, vec![]).n_outputs(), 0);

        // Per-column stats are fine with a single column; pairwise ones are not.
        model_cfg(1, vec![Mean, Var, Std]).validate().unwrap();
        for s in [Cov, Corr] {
            err(model_cfg(1, vec![s]), "at least two columns");
        }
        let mut one_pcorr = model_cfg(1, vec![PartialCorr]);
        one_pcorr.precision_prior = Some(1e-6);
        err(one_pcorr, "at least two columns");

        // partial_corr is read off the tracked precision matrix, which only
        // exists when a prior is configured.
        err(model_cfg(3, vec![PartialCorr]), "precision_prior");
        let mut with_prior = model_cfg(3, vec![PartialCorr]);
        with_prior.precision_prior = Some(1e-6);
        with_prior.validate().unwrap();

        model_cfg(2, vec![Mean, Var, Std, Cov, Corr])
            .validate()
            .unwrap();

        // mahal solves against the regularized co-moments, so it needs the
        // prior too; its quantiles need the score they are quantiles of.
        err(model_cfg(3, vec![Mahal]), "precision_prior");
        let mut mahal = model_cfg(3, vec![Mahal]);
        mahal.precision_prior = Some(1e-6);
        mahal.validate().unwrap();
        let mut mahal1 = model_cfg(1, vec![Mahal]);
        mahal1.precision_prior = Some(1e-6);
        mahal1.validate().unwrap();
        let mut no_mahal = model_cfg(3, vec![Mean]);
        no_mahal.mahal_quantiles = vec![0.99];
        err(no_mahal, "mahal_quantiles needs");
        for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            let mut c = mahal.clone();
            c.mahal_quantiles = vec![0.5, bad];
            err(c, "strictly between 0 and 1");
        }
        let mut levels = mahal.clone();
        levels.mahal_quantiles = vec![0.5, 0.99];
        levels.validate().unwrap();

        // pca: at most k components, and a cadence of at least one row.
        let mut too_many = model_cfg(3, vec![Mean]);
        too_many.pca = 4;
        too_many.pca_every = 1;
        err(too_many, "4 components of 3 columns");
        let mut no_cadence = model_cfg(3, vec![Mean]);
        no_cadence.pca = 2;
        err(no_cadence, "pca_every must be >= 1");
        let mut ok = model_cfg(3, vec![Mean]);
        ok.pca = 3;
        ok.pca_every = 5;
        ok.validate().unwrap();
        let mut off = model_cfg(3, vec![Mean]);
        off.pca = 0;
        off.pca_every = 0;
        off.validate().unwrap();
    }

    fn mahal_cfg(k: usize, prior: f64) -> EwCovCfg {
        let mut c = model_cfg(k, vec![EwCovStat::Mahal]);
        c.precision_prior = Some(prior);
        c
    }

    fn step(m: &mut EwCovModel, x: &[f64], first: bool) -> Vec<f64> {
        crate::OnlineModel::step(m, x, &[], if first { 0.0 } else { 1.0 }, 1.0).pred
    }

    /// Bitwise equality, so NaN slots compare equal to NaN slots.
    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn mahal_is_the_standardized_distance_on_a_diagonal_covariance() {
        // Two independent columns with variances 4 and 1 (population form
        // under no decay): the distance of (m0 + 2a, m1 + b) is sqrt(a² + b²)
        // up to the prior's `1e-12` on the diagonal.
        let mut m = EwCovModel::new(mahal_cfg(2, 1e-12)).unwrap();
        let pts = [
            [2.0, 1.0],
            [-2.0, -1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
            [2.0, 1.0],
            [-2.0, -1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
        ];
        for (i, p) in pts.iter().enumerate() {
            step(&mut m, p, i == 0);
        }
        // Means 0, var0 = 4, var1 = 1, cov = 0.
        let cov = m.cov();
        assert!(cov.mean(0).abs() < 1e-12 && cov.mean(1).abs() < 1e-12);
        assert!((cov.var(0) - 4.0).abs() < 1e-12 && (cov.var(1) - 1.0).abs() < 1e-12);
        assert!(cov.cov(0, 1).abs() < 1e-12);
        let d = m.mahal(&[6.0, -4.0]); // a = 3, b = -4
        assert!((d - 5.0).abs() < 1e-9, "{d}");
        assert!(m.mahal(&[0.0, 0.0]).abs() < 1e-12);
        // Read off the model, before the row is learned, as slot 0.
        let out = crate::OnlineModel::predict(&m, &[6.0, -4.0], 1.0).pred;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], d);
    }

    #[test]
    fn mahal_matches_the_precision_matrix_quadratic_form() {
        // Against `precision()`, which the partial correlations already trust:
        // d² = δᵀ P δ, with P the regularized inverse.
        let mut m = EwCovModel::new(mahal_cfg(3, 0.01)).unwrap();
        let mut s = 7u64;
        for i in 0..50 {
            let a = lcg(&mut s);
            let x = [a, 0.5 * a + lcg(&mut s), lcg(&mut s) - 0.2 * a];
            step(&mut m, &x, i == 0);
        }
        let p = m.cov().precision().unwrap();
        let x = [0.7, -0.4, 1.1];
        let delta: Vec<f64> = (0..3).map(|i| x[i] - m.cov().mean(i)).collect();
        let mut d2 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                d2 += delta[i] * p[i * 3 + j] * delta[j];
            }
        }
        assert!((m.mahal(&x) - d2.sqrt()).abs() < 1e-9);
        // A single column is the |z| of the row.
        let mut one = EwCovModel::new(mahal_cfg(1, 1e-12)).unwrap();
        for i in 0..20 {
            step(&mut one, &[lcg(&mut s) * 3.0], i == 0);
        }
        let z = (2.5 - one.cov().mean(0)) / one.cov().var(0).sqrt();
        assert!((one.mahal(&[2.5]) - z.abs()).abs() < 1e-9);
    }

    #[test]
    fn mahal_quantiles_track_the_scores_seen_so_far() {
        // Each row's own out-of-sample score enters the P² estimators after
        // it is read, so the quantile field lags by one row and never
        // includes the row it is emitted with.
        let mut c = mahal_cfg(2, 1e-6);
        c.mahal_quantiles = vec![0.5];
        let mut m = EwCovModel::new(c).unwrap();
        let mut s = 11u64;
        let mut scores = Vec::new();
        let mut oracle = crate::P2Quantile::new(0.5).unwrap();
        for i in 0..2000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let out = step(&mut m, &x, i == 0);
            assert_eq!(out.len(), 2);
            // The field is the estimate *before* this row's score joins.
            assert_eq!(out[1].to_bits(), oracle.get().unwrap_or(f64::NAN).to_bits());
            if out[0].is_finite() {
                scores.push(out[0]);
                oracle.update(out[0]);
            }
            if scores.len() < 5 {
                assert!(out[1].is_nan(), "P² needs five observations");
            }
        }
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let q = crate::OnlineModel::predict(&m, &[0.0, 0.0], 1.0).pred[1];
        assert_eq!(q.to_bits(), oracle.get().unwrap().to_bits());
        assert!((q - median).abs() < 0.05, "P² median {q} vs {median}");
        // NaN scores (before min_periods) never enter the estimator.
        let mut fresh = EwCovModel::new(m.cfg().clone()).unwrap();
        let out = step(&mut fresh, &[1.0, 2.0], true);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(fresh.mahal_q[0].count(), 0);
    }

    #[test]
    fn pca_of_recovers_a_known_eigenbasis() {
        // [[2, 1], [1, 2]] has eigenpairs (3, (1,1)/√2) and (1, (1,−1)/√2).
        let p = Pca::of(&[2.0, 1.0, 1.0, 2.0], 2, 2, None).unwrap();
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((p.eig[0] - 3.0).abs() < 1e-12 && (p.eig[1] - 1.0).abs() < 1e-12);
        assert_eq!(p.trace, 4.0);
        assert!((p.loadings[0] - r).abs() < 1e-12 && (p.loadings[1] - r).abs() < 1e-12);
        // The tie (|v0| = |v1|) resolves to the first entry positive.
        assert!((p.loadings[2] - r).abs() < 1e-12 && (p.loadings[3] + r).abs() < 1e-12);
        // Only the top component when asked for one.
        let top = Pca::of(&[2.0, 1.0, 1.0, 2.0], 2, 1, None).unwrap();
        assert_eq!(top.r(), 1);
        assert_eq!(top.eig, vec![p.eig[0]]);
        assert_eq!(top.loadings, p.loadings[..2].to_vec());
        // Largest-magnitude entry positive, not the first entry.
        let q = Pca::of(&[1.0, 0.0, 0.0, 5.0], 2, 1, None).unwrap();
        assert!((q.eig[0] - 5.0).abs() < 1e-12);
        assert!(q.loadings[0].abs() < 1e-12 && (q.loadings[1] - 1.0).abs() < 1e-12);
        // Refused: no components, more than k, a non-finite entry, a wrong size.
        assert!(Pca::of(&[1.0, 0.0, 0.0, 1.0], 2, 0, None).is_none());
        assert!(Pca::of(&[1.0, 0.0, 0.0, 1.0], 2, 3, None).is_none());
        assert!(Pca::of(&[1.0, 0.0, 0.0, f64::NAN], 2, 1, None).is_none());
        assert!(Pca::of(&[1.0, 0.0, 0.0], 2, 1, None).is_none());
    }

    #[test]
    fn pca_signs_follow_the_previous_refresh() {
        // The top eigenvector of [[a, -b], [-b, a]] is (1, −1)/√2, and the
        // max-abs rule has a tie there that any perturbation decides. A
        // predecessor along (−1, 1) keeps the new vector on its side.
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let first = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 1, None).unwrap();
        assert!((first.loadings[0] - r).abs() < 1e-12 && (first.loadings[1] + r).abs() < 1e-12);
        let prev = Pca {
            eig: vec![3.0],
            trace: 4.0,
            loadings: vec![-r, r],
        };
        let next = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 1, Some(&prev)).unwrap();
        assert!((next.loadings[0] + r).abs() < 1e-12 && (next.loadings[1] - r).abs() < 1e-12);
        // Nearly tied magnitudes that trade the lead: max-abs would flip,
        // continuity does not.
        let mut a = Pca::of(&[2.0, -0.999, -0.999, 2.0], 2, 1, None).unwrap();
        for eps in [0.0005, -0.0005, 0.0005, -0.0005] {
            // Tilt the matrix so the lead entry alternates between the two.
            let m = [2.0 + eps, -0.999, -0.999, 2.0 - eps];
            let b = Pca::of(&m, 2, 1, Some(&a)).unwrap();
            let dot = a.loadings[0] * b.loadings[0] + a.loadings[1] * b.loadings[1];
            assert!(dot > 0.99, "dot {dot}");
            a = b;
        }
        // A predecessor with fewer components leaves the extra ones to the
        // max-abs rule.
        let wide = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 2, Some(&prev)).unwrap();
        assert!((wide.loadings[0] + r).abs() < 1e-12);
        assert!((wide.loadings[2] - r).abs() < 1e-12 && (wide.loadings[3] - r).abs() < 1e-12);
    }

    #[test]
    fn pca_loadings_are_orthonormal_and_ordered() {
        let k = 5;
        let mut s = 5u64;
        // A random SPD matrix: A = Bᵀ B + small ridge.
        let b: Vec<f64> = (0..k * k).map(|_| lcg(&mut s)).collect();
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|l| b[l * k + i] * b[l * k + j]).sum::<f64>()
                    + if i == j { 0.01 } else { 0.0 };
            }
        }
        let p = Pca::of(&a, k, k, None).unwrap();
        for j in 1..k {
            assert!(p.eig[j - 1] >= p.eig[j], "descending: {:?}", p.eig);
        }
        assert!((p.eig.iter().sum::<f64>() - p.trace).abs() < 1e-10);
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..k)
                    .map(|l| p.loadings[i * k + l] * p.loadings[j * k + l])
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10, "v{i}·v{j} = {dot}");
            }
            // A v = λ v.
            for l in 0..k {
                let av: f64 = (0..k).map(|c| a[l * k + c] * p.loadings[i * k + c]).sum();
                assert!((av - p.eig[i] * p.loadings[i * k + l]).abs() < 1e-9);
            }
            let lead = (0..k)
                .max_by(|&x, &y| {
                    p.loadings[i * k + x]
                        .abs()
                        .partial_cmp(&p.loadings[i * k + y].abs())
                        .unwrap()
                })
                .unwrap();
            assert!(p.loadings[i * k + lead] > 0.0);
        }
    }

    fn pca_cfg(k: usize, r: usize, every: usize) -> EwCovCfg {
        let mut c = model_cfg(k, vec![EwCovStat::Mean]);
        c.pca = r;
        c.pca_every = every;
        c
    }

    #[test]
    fn lagcorr_fields_and_values_follow_the_lags() {
        let k = 2;
        let mut cfg = model_cfg(k, vec![EwCovStat::Corr, EwCovStat::LagCorr]);
        cfg.decay = crate::Decay::Halflife(30.0);
        cfg.lags = vec![1, 3];
        let names = ["a".to_string(), "b".to_string()];
        let labels = EwCovModel::labels(&names, &cfg.stats, &[], 0, &cfg.lags);
        assert_eq!(
            labels,
            [
                "corr_a_b",
                "lagcorr_a_a_l1",
                "lagcorr_a_b_l1",
                "lagcorr_b_a_l1",
                "lagcorr_b_b_l1",
                "lagcorr_a_a_l3",
                "lagcorr_a_b_l3",
                "lagcorr_b_a_l3",
                "lagcorr_b_b_l3",
            ]
        );
        assert_eq!(cfg.n_outputs(), labels.len());

        // `b` is `a` delayed by one row, so `lagcorr_b_a_l1` -- b now against
        // a one row ago -- goes to 1 and `lagcorr_a_b_l1` does not.
        let mut m = EwCovModel::new(cfg).unwrap();
        let mut s = 4u64;
        let mut prev = 0.0;
        let mut last = Vec::new();
        for i in 0..400 {
            let a = lcg(&mut s);
            let x = [a, prev];
            prev = a;
            last =
                crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0).pred;
        }
        // Slots: corr_a_b, then the four at lag 1, then the four at lag 3.
        assert!(last[3] > 0.95, "lagcorr_b_a_l1 is {}", last[3]);
        assert!(last[2].abs() < 0.2, "lagcorr_a_b_l1 is {}", last[2]);
        // And the auto terms at lag 1 are the series' own autocorrelation,
        // which for white noise is near zero.
        assert!(last[1].abs() < 0.2 && last[4].abs() < 0.2);
    }

    #[test]
    fn clear_lags_empties_the_ring_and_leaves_the_moments() {
        let mut cfg = model_cfg(2, vec![EwCovStat::Corr]);
        cfg.lags = vec![2];
        let mut m = EwCovModel::new(cfg).unwrap();
        let mut s = 7u64;
        for i in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let moments = m.cov().clone();
        let lagged = m.lag().unwrap().comoments().to_vec();
        assert_eq!(m.lag().unwrap().depth(), 2);
        crate::OnlineModel::clear_lags(&mut m);
        assert_eq!(m.lag().unwrap().depth(), 0);
        assert_eq!(m.cov(), &moments, "clear_lags is not a reset");
        assert_eq!(m.lag().unwrap().comoments(), &lagged[..]);
    }

    #[test]
    fn lags_do_not_move_the_contemporaneous_moments() {
        let mut plain = EwCovModel::new(model_cfg(3, vec![EwCovStat::Corr])).unwrap();
        let mut cfg = model_cfg(3, vec![EwCovStat::Corr]);
        cfg.lags = vec![1, 2, 5];
        let mut lagged = EwCovModel::new(cfg).unwrap();
        let mut s = 21u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let (d, w) = (if i == 0 { 0.0 } else { 1.0 }, (i % 4) as f64 * 0.5);
            let a = crate::OnlineModel::step(&mut plain, &x, &[], d, w);
            let b = crate::OnlineModel::step(&mut lagged, &x, &[], d, w);
            assert_eq!(a.n_eff, b.n_eff);
            // NaN is not equal to itself, and both are NaN before warmup.
            assert!(
                a.pred
                    .iter()
                    .zip(&b.pred)
                    .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
                "{:?} vs {:?}",
                a.pred,
                b.pred
            );
        }
        assert_eq!(plain.cov(), lagged.cov(), "the accumulator is untouched");
    }

    #[test]
    fn a_state_without_lags_resumes_under_a_spec_that_has_them() {
        let mut m = EwCovModel::new(model_cfg(2, vec![EwCovStat::Corr])).unwrap();
        let mut s = 3u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // A pre-task-48 state: no `lag` field at all. Rewrite the config the
        // way a resumed spec would and restore.
        let mut state = crate::OnlineModel::state(&m);
        if let crate::ModelState::EwCovModel(inner) = &mut state.model {
            inner.cfg.lags = vec![1];
            inner.lag = None;
        }
        let restored = <EwCovModel as crate::OnlineModel>::restore(&state).unwrap();
        let lag = restored.lag().expect("a fresh ring");
        assert_eq!(lag.depth(), 0);
        assert!(lag.comoments().iter().all(|c| *c == 0.0));
        assert_eq!(restored.cov(), m.cov(), "the moments came through");
    }

    #[test]
    fn a_bad_lag_configuration_is_refused_by_name() {
        let mut cfg = model_cfg(2, vec![EwCovStat::LagCorr]);
        assert!(
            EwCovModel::new(cfg.clone())
                .unwrap_err()
                .contains("lagcorr needs `lags`")
        );
        cfg.stats = vec![EwCovStat::Corr];
        cfg.lags = vec![2, 1];
        assert!(
            EwCovModel::new(cfg)
                .unwrap_err()
                .contains("strictly increasing")
        );
    }

    #[test]
    fn pca_fields_follow_the_stats_and_are_frozen_between_refreshes() {
        // Outputs: mean_a, mean_b, then pc0_var, pc0_share, pc0_a, pc0_b,
        // pc0_score. With `pca_every = 3` the loadings change only every
        // third learned row, and a row's score uses the frozen loadings
        // about the live mean.
        let k = 2;
        let mut m = EwCovModel::new(pca_cfg(k, 1, 3)).unwrap();
        assert_eq!(m.cfg().n_outputs(), k + (k + 3));
        let mut s = 3u64;
        let mut last: Option<(Vec<f64>, usize)> = None;
        let mut changes = Vec::new();
        for i in 0..40 {
            let a = lcg(&mut s);
            let x = [a, 2.0 * a + 0.3 * lcg(&mut s)];
            let pre = crate::OnlineModel::predict(&m, &x, 1.0).pred;
            let out = step(&mut m, &x, i == 0);
            assert!(
                same_bits(&pre, &out),
                "predict and step read the same frozen state"
            );
            if !out[2].is_nan() {
                // pc0_score = v·(x − μ) with μ the mean *before* this row.
                let want = out[4] * (x[0] - out[0]) + out[5] * (x[1] - out[1]);
                assert!((out[6] - want).abs() < 1e-12, "row {i}");
                assert!((0.0..=1.0 + 1e-12).contains(&out[3]), "share {}", out[3]);
                let v = out[4..6].to_vec();
                if let Some((prev, at)) = &last {
                    if *prev != v {
                        changes.push(i - at);
                        last = Some((v, i));
                    }
                } else {
                    last = Some((v, i));
                }
            }
        }
        assert!(!changes.is_empty());
        assert!(changes.iter().all(|&c| c == 3), "refresh gaps {changes:?}");
        assert_eq!(m.pca().unwrap().r(), 1);
    }

    #[test]
    fn pca_refresh_counts_learned_rows_so_chunking_cannot_matter() {
        // The same rows through one model and through a clone that was
        // serialized and restored midway give identical outputs, including
        // the refresh cadence and the frozen loadings.
        let mut a = EwCovModel::new(pca_cfg(3, 2, 4)).unwrap();
        let mut s = 9u64;
        let rows: Vec<[f64; 3]> = (0..50)
            .map(|_| [lcg(&mut s), lcg(&mut s), lcg(&mut s)])
            .collect();
        let mut outs = Vec::new();
        for (i, x) in rows.iter().enumerate() {
            outs.push(step(&mut a, x, i == 0));
        }
        let mut b = EwCovModel::new(pca_cfg(3, 2, 4)).unwrap();
        for (i, x) in rows.iter().enumerate() {
            if i == 17 {
                let bytes = rmp_serde::to_vec(&b).unwrap();
                b = rmp_serde::from_slice(&bytes).unwrap();
            }
            let out = step(&mut b, x, i == 0);
            assert!(same_bits(&out, &outs[i]), "row {i}");
        }
        assert_eq!(a, b);
    }

    #[test]
    fn pca_with_every_component_explains_all_the_variance() {
        let k = 3;
        let mut m = EwCovModel::new(pca_cfg(k, k, 1)).unwrap();
        let mut s = 13u64;
        for i in 0..30 {
            step(&mut m, &[lcg(&mut s), lcg(&mut s), lcg(&mut s)], i == 0);
        }
        // Read after the last refresh, which followed the last update.
        let out = crate::OnlineModel::predict(&m, &[0.0; 3], 1.0).pred;
        let shares: f64 = (0..k).map(|j| out[k + j * (k + 3) + 1]).sum();
        assert!((shares - 1.0).abs() < 1e-10, "{shares}");
        let vars: f64 = (0..k).map(|j| out[k + j * (k + 3)]).sum();
        let trace: f64 = (0..k).map(|i| m.cov().cov(i, i)).sum();
        assert!((vars - trace).abs() < 1e-10);
    }

    #[test]
    fn pca_stays_nan_before_min_periods_and_on_a_degenerate_matrix() {
        let mut c = pca_cfg(2, 1, 1);
        c.min_periods = 5.0;
        let mut m = EwCovModel::new(c).unwrap();
        for i in 0..4 {
            let out = step(&mut m, &[i as f64, 1.0], i == 0);
            assert!(out[2..].iter().all(|v| v.is_nan()), "row {i}: {out:?}");
        }
        assert!(m.pca().is_none());
        let out = step(&mut m, &[4.0, 1.0], false);
        assert!(
            out[2..].iter().all(|v| v.is_nan()),
            "read before the first refresh"
        );
        assert!(m.pca().is_some(), "refreshed after the fifth learned row");
        let out = step(&mut m, &[5.0, 1.0], false);
        // Column 1 is constant: all the variance is on column 0.
        assert!((out[3] - 1.0).abs() < 1e-12, "share {}", out[3]);
        assert!((out[4] - 1.0).abs() < 1e-12 && out[5].abs() < 1e-12);
        // A constant stream: trace 0, so the share is NaN rather than 0/0.
        let mut flat = EwCovModel::new(pca_cfg(2, 1, 1)).unwrap();
        let mut out = Vec::new();
        for i in 0..6 {
            out = step(&mut flat, &[1.0, 1.0], i == 0);
        }
        assert_eq!(out[2], 0.0);
        assert!(out[3].is_nan());
    }

    #[test]
    fn n_outputs_and_labels_agree_slot_for_slot() {
        // `n_outputs` sizes the output buffer, `labels` names the fields and
        // `read` fills them: all three must walk the stats in the same order
        // and produce the same count, or a field ends up named after another
        // statistic's value.
        use EwCovStat::*;
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        for stats in [
            vec![Mean],
            vec![Var],
            vec![Std],
            vec![Cov],
            vec![Corr],
            vec![Mean, Var, Std, Cov, Corr],
            vec![Corr, Mean],
        ] {
            let mut c = model_cfg(3, stats.clone());
            c.precision_prior = Some(1e-6);
            let labels = EwCovModel::labels(&names, &stats, &[], 0, &[]);
            assert_eq!(c.n_outputs(), labels.len(), "{stats:?}");

            let mut m = EwCovModel::new(c).unwrap();
            let mut s = 31u64;
            for i in 0..40 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let vals = m.read(&m.cov, &[0.0; 3]);
            assert_eq!(vals.len(), labels.len(), "{stats:?}");

            // Spot-check that the label describes the value under it.
            for (label, v) in labels.iter().zip(&vals) {
                if label.starts_with("var_") || label.starts_with("std_") {
                    assert!(*v >= 0.0, "{label} = {v}");
                }
                if label.starts_with("corr_") {
                    assert!((-1.0..=1.0).contains(v), "{label} = {v}");
                }
            }
        }
    }

    #[test]
    fn labels_name_the_pair_in_column_order() {
        use EwCovStat::*;
        let names: Vec<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            EwCovModel::labels(&names, &[Mean], &[], 0, &[]),
            ["mean_x", "mean_y", "mean_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Var], &[], 0, &[]),
            ["var_x", "var_y", "var_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Std], &[], 0, &[]),
            ["std_x", "std_y", "std_z"]
        );
        // Upper triangle only, and never a self-pair.
        assert_eq!(
            EwCovModel::labels(&names, &[Cov], &[], 0, &[]),
            ["cov_x_y", "cov_x_z", "cov_y_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Corr], &[], 0, &[]),
            ["corr_x_y", "corr_x_z", "corr_y_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[PartialCorr], &[], 0, &[]),
            ["pcorr_x_y", "pcorr_x_z", "pcorr_y_z"]
        );
        // Stats concatenate in the order given, not a canonical order.
        assert_eq!(
            EwCovModel::labels(&names, &[Corr, Mean], &[], 0, &[])
                .first()
                .unwrap(),
            "corr_x_y"
        );
    }

    #[test]
    fn read_returns_the_statistic_each_slot_claims() {
        // `read` is the one place the statistics are actually computed rather
        // than accumulated: var vs std vs corr all come off the same moments,
        // so a slot filled from the wrong one is invisible to the accumulator
        // tests. Each is checked against the definition.
        use EwCovStat::*;
        let stats = vec![Mean, Var, Std, Cov, Corr];
        let mut m = EwCovModel::new(model_cfg(2, stats)).unwrap();
        let mut s = 37u64;
        let mut xs = Vec::new();
        for i in 0..80 {
            let a = lcg(&mut s);
            let x = [3.0 + a, 10.0 - 2.0 * a + 0.5 * lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            xs.push(x.to_vec());
        }
        let v = m.read(&m.cov, &[0.0; 2]);
        let (mean0, mean1) = (v[0], v[1]);
        let (var0, var1) = (v[2], v[3]);
        let (std0, std1) = (v[4], v[5]);
        let cov01 = v[6];
        let corr01 = v[7];

        // Against an unweighted recomputation (halflife is infinite here).
        let n = xs.len() as f64;
        let m0: f64 = xs.iter().map(|x| x[0]).sum::<f64>() / n;
        let m1: f64 = xs.iter().map(|x| x[1]).sum::<f64>() / n;
        let c01: f64 = xs.iter().map(|x| (x[0] - m0) * (x[1] - m1)).sum::<f64>() / n;
        let v0: f64 = xs.iter().map(|x| (x[0] - m0).powi(2)).sum::<f64>() / n;
        let v1: f64 = xs.iter().map(|x| (x[1] - m1).powi(2)).sum::<f64>() / n;

        assert!((mean0 - m0).abs() < 1e-10, "{mean0} vs {m0}");
        assert!((mean1 - m1).abs() < 1e-10, "{mean1} vs {m1}");
        assert!((var0 - v0).abs() < 1e-10, "{var0} vs {v0}");
        assert!((var1 - v1).abs() < 1e-10);
        assert!((std0 - v0.sqrt()).abs() < 1e-10, "std is sqrt(var)");
        assert!((std1 - v1.sqrt()).abs() < 1e-10);
        assert!((cov01 - c01).abs() < 1e-10, "{cov01} vs {c01}");
        assert!(
            (corr01 - c01 / (v0 * v1).sqrt()).abs() < 1e-10,
            "corr is cov / (std*std)"
        );
        // The relationship is strongly negative, so a sign error is visible.
        assert!(corr01 < -0.9, "{corr01}");
    }

    #[test]
    fn accumulate_only_learns_the_same_state_and_emits_nothing() {
        // `stats = []` (docs/ENHANCEMENTS.md E43): no slots, the same
        // accumulator. The two models must agree to the bit, because the
        // whole point is that the state is the product.
        use crate::OnlineModel;
        use EwCovStat::*;
        let mut full = EwCovModel::new(model_cfg(3, vec![Mean, Var, Corr])).unwrap();
        let mut bare = EwCovModel::new(model_cfg(3, vec![])).unwrap();
        assert_eq!(bare.n_outputs(), 0);
        assert!(
            EwCovModel::labels(&["a".into(), "b".into(), "c".into()], &[], &[], 0, &[]).is_empty()
        );
        let mut s = 11u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 5.0 * lcg(&mut s) + 1.0, lcg(&mut s) - 2.0];
            let (d, w) = (
                if i == 0 { 0.0 } else { 0.7 },
                if i % 7 == 0 { 0.0 } else { 1.5 },
            );
            let sf = full.step(&x, &[], d, w);
            let sb = bare.step(&x, &[], d, w);
            assert!(sb.pred.is_empty(), "no slots");
            assert_eq!(sb.n_eff.to_bits(), sf.n_eff.to_bits());
            assert_eq!(sf.pred.len(), 3 + 3 + 3);
            let (p, pf) = (bare.predict(&x, 0.0), full.predict(&x, 0.0));
            assert!(p.pred.is_empty() && p.n_eff.to_bits() == pf.n_eff.to_bits());
        }
        assert_eq!(bare.cov(), full.cov(), "identical accumulators");
        // The state round-trips and restores into the same accumulator.
        let again = EwCovModel::restore(&bare.state()).unwrap();
        assert_eq!(again.cov(), bare.cov());
        // PCA and Mahalanobis slots stand on their own, without a statistic.
        let mut cfg = model_cfg(3, vec![]);
        cfg.pca = 1;
        cfg.pca_every = 1;
        cfg.validate().unwrap();
        assert_eq!(cfg.n_outputs(), 3 + 3);
        let mut cfg = model_cfg(3, vec![]);
        cfg.mahal_quantiles = vec![0.9];
        assert!(cfg.validate().unwrap_err().contains("needs \"mahal\""));
    }

    #[test]
    fn corr_is_nan_not_infinite_when_a_column_is_constant() {
        use EwCovStat::*;
        let mut m = EwCovModel::new(model_cfg(2, vec![Corr, Var])).unwrap();
        let mut s = 41u64;
        for i in 0..40 {
            crate::OnlineModel::step(
                &mut m,
                &[lcg(&mut s), 5.0],
                &[],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let v = m.read(&m.cov, &[0.0; 2]);
        assert!(v[0].is_nan(), "corr against a constant column: {}", v[0]);
        assert_eq!(v[2], 0.0, "a constant column has exactly zero variance");
    }

    #[test]
    fn matches_direct_computation() {
        let xs: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let f = i as f64;
                vec![
                    f.sin() * 2.0,
                    f.cos() - 0.3,
                    (f * 0.7).sin() * (f * 0.1).cos(),
                ]
            })
            .collect();
        let lams: Vec<f64> = (0..40).map(|i| 0.9 + 0.0025 * (i % 3) as f64).collect();
        let ws: Vec<f64> = (0..40).map(|i| 0.5 + 0.03 * (i % 7) as f64).collect();

        let mut ew = EwCov::new(3);
        for i in 0..40 {
            ew.update(&xs[i], lams[i], ws[i]);
        }
        let (wsum, m, s) = direct(&xs, &lams, &ws);
        assert!((ew.n_eff() - wsum).abs() < 1e-10);
        for i in 0..3 {
            assert!((ew.mean(i) - m[i]).abs() < 1e-12, "mean {i}");
            for j in 0..3 {
                assert!((ew.raw(i, j) - s[i * 3 + j]).abs() < 1e-12, "raw {i},{j}");
            }
        }
    }

    #[test]
    fn cov_and_var_are_centered() {
        let mut ew = EwCov::new(2);
        for i in 0..1000 {
            let f = i as f64 * (std::f64::consts::TAU / 100.0); // 10 full periods
            ew.update(&[5.0 + f.sin(), -2.0 + f.cos()], 1.0, 1.0);
        }
        // full periods: means equal the offsets, var = 0.5, cov(sin,cos) = 0
        assert!((ew.mean(0) - 5.0).abs() < 0.02);
        assert!((ew.var(0) - 0.5).abs() < 0.02);
        assert!(ew.cov(0, 1).abs() < 0.02);
    }

    #[test]
    fn pure_decay_keeps_means() {
        let mut ew = EwCov::new(1);
        ew.update(&[3.0], 1.0, 1.0);
        ew.update(&[5.0], 1.0, 1.0);
        let m = ew.mean(0);
        let n = ew.n_eff();
        ew.decay(0.5);
        assert_eq!(ew.mean(0), m);
        assert!((ew.n_eff() - 0.5 * n).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "non-negative weight")]
    fn a_negative_weight_is_a_caller_error() {
        // The bank rejects negative weights upstream (T-E1), naming the row.
        // Reaching here means that check was bypassed, so this is a
        // `debug_assert`: loud in tests, and a silent no-op in release rather
        // than corrupted state. Only the assertion is testable -- the release
        // path's `if w < 0.0 { return }` is unreachable in a debug build.
        let mut ew = EwCov::new(2);
        ew.update(&[1.0, 2.0], 1.0, -0.5);
    }

    #[test]
    fn an_emptied_accumulator_does_not_divide_by_its_own_zero_weight() {
        // lam = 0 with weight 0 leaves no weight at all; the next update must
        // notice rather than produce NaN.
        let mut empty = EwCov::new(2);
        empty.update(&[1.0, 2.0], 0.0, 0.0);
        assert_eq!(empty.n_eff(), 0.0);
        assert!(empty.mean(0).is_finite() && empty.mean(1).is_finite());
        empty.update(&[3.0, 4.0], 1.0, 1.0);
        assert_eq!(empty.mean(0), 3.0, "the first real row seeds the mean");
        assert_eq!(empty.n_eff(), 1.0);
    }

    #[test]
    fn decay_shrinks_the_weight_and_the_prior_scale_but_not_the_means() {
        // `decay` is the pure-ageing path a skipped row takes. The prior scale
        // must ride along, or the ridge prior stops matching the co-moments it
        // is regularizing.
        let prior = 0.25;
        let mut ew = EwCov::with_precision_prior(2, prior).unwrap();
        let mut s = 109u64;
        for _ in 0..50 {
            ew.update(&[lcg(&mut s), 1.0 + lcg(&mut s)], 0.99, 1.0);
        }
        let (m0, m1) = (ew.mean(0), ew.mean(1));
        let (c01, w, ps) = (ew.cov(0, 1), ew.n_eff(), ew.prior_scale());

        ew.decay(0.5);
        assert_eq!(ew.mean(0), m0, "means are unchanged by pure decay");
        assert_eq!(ew.mean(1), m1);
        assert_eq!(ew.cov(0, 1), c01, "so are the centered co-moments");
        assert!((ew.n_eff() - w * 0.5).abs() < 1e-12, "the weight halves");
        assert!(
            (ew.prior_scale() - ps * 0.5).abs() < 1e-12,
            "so does the prior"
        );
    }

    #[test]
    fn k_reports_the_configured_width() {
        for k in [1usize, 2, 7] {
            assert_eq!(EwCov::new(k).k(), k);
            assert_eq!(EwCov::with_precision_prior(k, 1.0).unwrap().k(), k);
        }
    }

    #[test]
    fn the_model_withholds_statistics_until_min_periods() {
        use crate::OnlineModel;
        let mut c = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Corr]);
        c.min_periods = 5.0;
        let mut m = EwCovModel::new(c).unwrap();
        let mut s = 113u64;
        for i in 0..10 {
            let step = m.step(
                &[lcg(&mut s), lcg(&mut s)],
                &[],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            // n_eff is the weight before this row, so it reaches 5 on row 5.
            let want_ready = i >= 5;
            assert_eq!(
                step.pred.iter().all(|v| v.is_finite()),
                want_ready,
                "row {i}: n_eff = {}, min_periods = 5",
                step.n_eff
            );
            assert_eq!(m.n_eff(), step.n_eff + 1.0, "row {i}: n_eff accessor");
        }
    }

    #[test]
    fn the_from_scratch_inverse_really_inverts() {
        // `precision_matches_a_from_scratch_solve` uses this as its reference,
        // so it has to be checked against something else -- the definition.
        // A · A⁻¹ = I, where A = C + s·prior·I.
        let (prior, k) = (0.5, 3usize);
        let mut ew = EwCov::with_precision_prior(k, prior).unwrap();
        let mut s = 103u64;
        for _ in 0..80 {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s), 10.0 * lcg(&mut s)];
            ew.update(&x, 0.97, 0.5 + lcg(&mut s).abs());
        }
        let scale = ew.precision_scale();
        let inv = ew.inverse_from_scratch(prior, scale).unwrap();
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for m in 0..k {
                    let a_im = ew.cov(i, m) + if i == m { prior * scale } else { 0.0 };
                    acc += a_im * inv[m * k + j];
                }
                let want = f64::from(i == j);
                assert!(
                    (acc - want).abs() < 1e-9,
                    "(A A^-1)[{i}][{j}] = {acc}, want {want}"
                );
            }
        }
        // A singular matrix with no prior has no inverse to report.
        let mut flat = EwCov::new(2);
        for _ in 0..10 {
            flat.update(&[1.0, 2.0], 1.0, 1.0);
        }
        assert!(
            flat.inverse_from_scratch(0.0, 1.0).is_none(),
            "a rank-deficient matrix with no prior cannot be inverted"
        );
    }

    #[test]
    fn the_precision_prior_starts_at_i_over_prior_and_must_be_usable() {
        // Before any data the precision matrix is I/prior exactly.
        let prior = 4.0;
        let ew = EwCov::with_precision_prior(3, prior).unwrap();
        assert_eq!(ew.precision_scale(), 1.0);
        let p = ew.precision().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 / prior } else { 0.0 };
                assert_eq!(p[i * 3 + j], want, "({i},{j})");
            }
        }
        assert!(ew.has_precision_prior());
        assert!(!EwCov::new(3).has_precision_prior());
        assert!(EwCov::new(3).precision().is_none());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(EwCov::with_precision_prior(3, bad).is_err(), "prior {bad}");
        }
    }

    #[test]
    fn partial_corr_is_the_textbook_formula_on_the_precision_matrix() {
        // -P_ij / sqrt(P_ii P_jj), against a precision matrix obtained the
        // other way (Gauss-Jordan), so the formula and the solve cannot agree
        // by both being wrong.
        let (prior, k) = (1e-4, 4usize);
        let mut ew = EwCov::with_precision_prior(k, prior).unwrap();
        let mut s = 107u64;
        for _ in 0..300 {
            let a = lcg(&mut s);
            let b = lcg(&mut s);
            // A genuine conditional structure, so the values are not all ~0.
            let x = [a, b, a + b + 0.3 * lcg(&mut s), 0.5 * lcg(&mut s)];
            ew.update(&x, 0.99, 1.0);
        }
        let reference = ew
            .inverse_from_scratch(prior, ew.precision_scale())
            .unwrap();
        let p = ew.precision().unwrap();
        for i in 0..k {
            for j in 0..k {
                let want =
                    -reference[i * k + j] / (reference[i * k + i] * reference[j * k + j]).sqrt();
                let got = partial_corr(&p, k, i, j);
                assert!(
                    (got - want.clamp(-1.0, 1.0)).abs() < 1e-6,
                    "pcorr({i},{j}) = {got}, want {want}"
                );
            }
            // A column against itself is -1 by the formula's own definition;
            // callers only ask for i != j, but it must not be NaN.
            assert!(partial_corr(&p, k, i, i).is_finite());
        }
        // Exactly symmetric (the precision matrix is symmetrized), and always
        // a correlation.
        for i in 0..k {
            for j in 0..k {
                let (a, b) = (partial_corr(&p, k, i, j), partial_corr(&p, k, j, i));
                assert_eq!(a, b, "asymmetric at ({i},{j})");
                assert!((-1.0..=1.0).contains(&a), "({i},{j}) = {a}");
            }
        }
    }

    #[test]
    fn precision_matches_a_from_scratch_solve() {
        // The Cholesky solve must equal a Gauss-Jordan inversion of the same
        // matrix at every step, prior scale included.
        let prior = 0.5;
        let mut ew = EwCov::with_precision_prior(3, prior).unwrap();
        let mut state = 7u64;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for step in 0..60 {
            let x = [lcg(), lcg(), lcg() * 3.0];
            ew.update(&x, 0.97, 0.5 + lcg().abs());
            let want = ew
                .inverse_from_scratch(prior, ew.precision_scale())
                .expect("reference inverse should exist");
            let got = ew.precision().unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    let (g, w) = (got[i * 3 + j], want[i * 3 + j]);
                    assert!(
                        (g - w).abs() < 1e-9 * (1.0 + w.abs()),
                        "step {step}, ({i},{j}): cholesky {g}, gauss-jordan {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn partial_correlation_detects_a_spurious_link() {
        // x2 = x0 + x1 + noise. Marginally x0 and x2 correlate; controlling for
        // x1 they still do, but x0 and x1 are independent both ways. The
        // interesting case is the reverse: with a common driver, the marginal
        // correlation is high and the partial one is not.
        let mut ew = EwCov::with_precision_prior(3, 1e-6).unwrap();
        let mut state = 11u64;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for _ in 0..20000 {
            let driver = lcg();
            // two children of one driver: correlated with each other only
            // through it
            let a = driver + 0.1 * lcg();
            let b = driver + 0.1 * lcg();
            ew.update(&[driver, a, b], 1.0, 1.0);
        }
        let marginal = ew.cov(1, 2) / (ew.var(1) * ew.var(2)).sqrt();
        let partial = partial_corr(&ew.precision().unwrap(), 3, 1, 2);
        assert!(
            marginal > 0.9,
            "children should correlate marginally: {marginal}"
        );
        assert!(
            partial.abs() < 0.2,
            "controlling for the driver should remove it: {partial}"
        );
    }

    #[test]
    fn no_precision_prior_by_default() {
        let ew = EwCov::new(2);
        assert!(!ew.has_precision_prior());
        assert!(ew.precision().is_none());
    }

    /// E48 (task 41): the row scratch is a buffer, not state. A restored
    /// accumulator has an empty one and must produce the same bits as the
    /// accumulator that wrote it -- which it does, because the scratch is
    /// refilled from scratch on every row.
    #[test]
    fn the_row_scratch_is_not_state() {
        let mut a = EwCov::new(5);
        let mut s = 3u64;
        let row = |st: &mut u64| -> Vec<f64> {
            (0..5)
                .map(|_| {
                    *st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((*st >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
                })
                .collect()
        };
        for _ in 0..40 {
            a.update(&row(&mut s), 0.97, 1.5);
        }
        // A round trip leaves the scratch empty and the accumulator equal.
        let bytes = rmp_serde::to_vec_named(&a).unwrap();
        let mut b: EwCov = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(a, b, "the scratch must not make two equal states unequal");
        assert!(b.dev.0.is_empty(), "a state file carries no scratch");
        // And both continue to the bit from there.
        let mut s2 = s;
        for _ in 0..40 {
            a.update(&row(&mut s), 0.97, 1.5);
            b.update(&row(&mut s2), 0.97, 1.5);
        }
        assert_eq!(a.comoments(), b.comoments());
        assert_eq!(a.means(), b.means());
        assert_eq!(a.n_eff().to_bits(), b.n_eff().to_bits());
    }

    /// The co-moment matrix is symmetric to the eye but **not** to the bit:
    /// `c[i][j]` is `((a*b)*d_i)*d_j` and `c[j][i]` is `((a*b)*d_j)*d_i`,
    /// which round differently. E48 proposed mirroring one triangle onto the
    /// other and called it bit-identical; this is why it is not
    /// (docs/PERFORMANCE.md section 14). Kept as a test so the next person to
    /// reach for that shortcut finds the reason it was not taken.
    #[test]
    fn the_two_triangles_are_equal_but_not_bit_equal() {
        let mut ew = EwCov::new(6);
        let mut s = 11u64;
        for _ in 0..200 {
            let x: Vec<f64> = (0..6)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    100.0 + ((s >> 11) as f64) / ((1u64 << 53) as f64)
                })
                .collect();
            ew.update(&x, 0.99, 1.0);
        }
        let mut differing = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let (u, l) = (ew.cov(i, j), ew.cov(j, i));
                assert!(
                    (u - l).abs() < 1e-9 * u.abs().max(1.0),
                    "not symmetric at all"
                );
                if u.to_bits() != l.to_bits() {
                    differing += 1;
                }
            }
        }
        assert!(
            differing > 0,
            "if the triangles ever become bit-equal, E48's mirror is back on \
             the table -- but check docs/PERFORMANCE.md section 14 first: it \
             measured 49% to 107% *slower*, because the mirror store walks a \
             new cache line per element"
        );
    }

    #[test]
    fn serde_roundtrip() {
        let mut ew = EwCov::new(2);
        ew.update(&[1.0, 2.0], 0.95, 1.3);
        let bytes = rmp_serde::to_vec(&ew).unwrap();
        let back: EwCov = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(ew, back);
    }

    #[test]
    fn loads_a_schema_1_state() {
        // Schema 1 stored the Sherman-Morrison inverse under `inv` with its
        // prior as `inv_prior` / `inv_scale`. The inverse is discarded (it is
        // recomputed from the co-moments) and the prior carried over.
        let mut want = EwCov::with_precision_prior(2, 0.25).unwrap();
        want.update(&[1.0, 2.0], 0.95, 1.3);
        want.update(&[0.5, 2.5], 0.95, 1.0);
        let v1 = serde_json::json!({
            "k": 2,
            "w_sum": want.w_sum,
            "prior_scale": want.prior_scale,
            "m": want.m,
            "c": want.c,
            "inv": [1.0, 2.0, 3.0, 4.0],
            "inv_prior": 0.25,
            "inv_scale": want.precision_scale,
        });
        let got: EwCov = serde_json::from_value(v1).unwrap();
        // No `q_sum` in a schema-1 file, and it cannot be reconstructed: the
        // load reports `None` rather than a Kish size that would be wrong by
        // the length of the history (E45).
        assert_eq!(got.q_sum(), None);
        assert_eq!(got.n_kish(), None);
        want.q_sum = None;
        assert_eq!(got, want);
        assert!(got.has_precision_prior());
        assert_eq!(got.precision(), want.precision());
    }

    #[test]
    fn kish_size_of_a_unit_weight_window() {
        // Unit weights on a constant clock: `W -> 1/(1-lam)` and
        // `Q -> 1/(1-lam^2)`, so `W^2/Q -> (1+lam)/(1-lam)`.
        let lam = 0.98;
        let mut ew = EwCov::new(1);
        assert_eq!(ew.n_kish(), None, "no rows, no sample size");
        for i in 0..20_000 {
            ew.update(&[i as f64 % 3.0], lam, 1.0);
        }
        let want = (1.0 + lam) / (1.0 - lam);
        let got = ew.n_kish().unwrap();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        // And it is a *row* count, not a weight: doubling every weight leaves
        // it where it was, while `n_eff` doubles.
        let mut heavy = EwCov::new(1);
        for i in 0..20_000 {
            heavy.update(&[i as f64 % 3.0], lam, 2.0);
        }
        assert!((heavy.n_kish().unwrap() - got).abs() < 1e-9);
        assert!((heavy.n_eff() - 2.0 * ew.n_eff()).abs() < 1e-9);
    }

    #[test]
    fn kish_size_counts_the_rows_that_carry_the_weight() {
        // One heavy row among many light ones is worth about one row.
        let mut ew = EwCov::new(1);
        ew.update(&[1.0], 1.0, 1e6);
        for _ in 0..1_000 {
            ew.update(&[1.0], 1.0, 1.0);
        }
        let n = ew.n_kish().unwrap();
        assert!((1.0..1.01).contains(&n), "{n}");
    }

    #[test]
    fn a_pure_decay_ages_the_squared_weights_too() {
        let mut ew = EwCov::new(1);
        for _ in 0..50 {
            ew.update(&[1.0], 0.9, 1.0);
        }
        let before = ew.n_kish().unwrap();
        ew.decay(0.5);
        // `W` halves and `Q` quarters, so the Kish size is unchanged: aging
        // without data forgets weight, not rows.
        assert!((ew.n_kish().unwrap() - before).abs() < 1e-9);
    }

    #[test]
    fn a_zero_weight_row_is_still_a_decay() {
        let mut a = EwCov::new(1);
        let mut b = EwCov::new(1);
        for _ in 0..10 {
            a.update(&[1.5], 0.9, 1.0);
            b.update(&[1.5], 0.9, 1.0);
        }
        a.update(&[0.0], 0.9, 0.0);
        b.decay(0.9);
        assert_eq!(a.q_sum(), b.q_sum());
        assert_eq!(a.n_eff(), b.n_eff());
    }

    #[test]
    fn target_moments_match_an_ew_cov_over_the_column() {
        // `TargetMoments::learn` takes the `a`/`b` its caller computed from
        // the same per-target weight, so its variance is bit-identical to an
        // `EwCov` over that one column.
        let mut tm = TargetMoments::new(1);
        let mut ew = EwCov::new(1);
        let mut w_t = 0.0;
        for i in 0..500 {
            let y = 100.0 + (i as f64 * 0.37).sin();
            let (lam, w) = (0.97, 0.5 + (i % 4) as f64);
            let w_new = lam * w_t + w;
            let (a, b) = (lam * w_t / w_new, w / w_new);
            tm.learn(0, y, a, b, lam, w);
            w_t = w_new;
            ew.update(&[y], lam, w);
        }
        assert_eq!(tm.means()[0], ew.mean(0), "mean");
        assert_eq!(tm.vars()[0], ew.cov(0, 0), "variance");
        assert_eq!(tm.q()[0], ew.q_sum().unwrap(), "Sum w^2");
        assert_eq!(tm.n_kish(&[w_t])[0], ew.n_kish(), "Kish n");
    }

    #[test]
    fn blending_target_moments_with_a_copy_is_the_identity() {
        let mut tm = TargetMoments::new(1);
        let mut w_t = 0.0;
        for i in 0..100 {
            let y = (i as f64 * 0.11).cos();
            let (lam, w) = (0.95, 1.0);
            let w_new = lam * w_t + w;
            tm.learn(0, y, lam * w_t / w_new, w / w_new, lam, w);
            w_t = w_new;
        }
        let twin = tm.clone();
        let want = tm.clone();
        tm.blend(&twin, 0, 0.5, 0.5);
        assert!((tm.means()[0] - want.means()[0]).abs() < 1e-12);
        assert!((tm.vars()[0] - want.vars()[0]).abs() < 1e-12);
        // `Q` too: a union of the two row sets would report the blend as
        // twice as informative as the state it blended with itself.
        assert!((tm.q()[0] - want.q()[0]).abs() < 1e-12);
    }
}

/// The blocked Gram update (E51, docs/PLAN.md task 71).
///
/// Every test here runs the *same* rows through both paths and compares. The
/// point is not that the blocked path is self-consistent -- it is that the
/// existing recursion is the oracle, since that is the one every shipped
/// number has always come from.
#[cfg(test)]
mod block_tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    /// `(x, lam, w)` rows exercising everything a block has to survive:
    /// unequal weights with zeros, decay, and a feature offset far from zero
    /// so that a naive uncentred product would lose the variance.
    fn stream(
        n: usize,
        k: usize,
        seed: u64,
        offset: f64,
        lam: f64,
        wmode: u8,
    ) -> Vec<(Vec<f64>, f64, f64)> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let x: Vec<f64> = (0..k).map(|_| offset + lcg(&mut s)).collect();
                let w = match wmode {
                    0 => 1.0,
                    1 => 0.2 + 1.5 * (lcg(&mut s) + 1.0) / 2.0,
                    _ => {
                        if i % 7 == 0 {
                            0.0
                        } else {
                            1.0
                        }
                    }
                };
                (x, lam, w)
            })
            .collect()
    }

    fn run(rows: &[(Vec<f64>, f64, f64)], k: usize, block: usize) -> EwCov {
        let mut c = EwCov::new(k);
        if block > 0 {
            c.set_block_rows(block);
        }
        for (x, lam, w) in rows {
            c.update(x, *lam, *w);
        }
        c.flush();
        c
    }

    /// The mean and co-moments are a reassociation, so they agree closely
    /// rather than exactly. The co-moments are compared **against the
    /// matrix's own scale**, not entry by entry: an off-diagonal co-moment
    /// of two independent features is a small number that happens to be
    /// near zero, and a per-entry relative tolerance turns its rounding into
    /// a failure while saying nothing about the merge (at `k = 2,000` the
    /// worst per-entry ratio is `1.6e-5` on a merge that is right to `5e-13`
    /// of the largest variance).
    ///
    /// The tolerance is the **data's own resolution**: a feature at `1e7`
    /// is stored to `1e7·ε ≈ 2e-9`, so a centred value of order `σ` is only
    /// known to `1e7·ε/σ` and the two paths, which round it in different
    /// orders, legitimately differ by that much (`2.5e-9` of the variance at
    /// `k = 1`, block 2, offset `1e7`, where the floor of `1e-9` failed).
    /// The design measured `5e-13` of the largest variance at an offset of
    /// `1e3` and `7e-10` at `1e7`, so a wrong term -- of order `1` -- is a
    /// million times the bound at either offset.
    fn assert_moments_close(seq: &EwCov, blk: &EwCov, what: &str) {
        let k = seq.k;
        let scale = (0..k)
            .map(|i| seq.c[i * k + i].abs())
            .fold(0.0f64, f64::max)
            .max(1e-300);
        let magnitude = seq.m.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let resolution = magnitude * f64::EPSILON / scale.sqrt();
        let tol = 1e-9f64.max(64.0 * resolution);
        for i in 0..k {
            let (a, b) = (seq.m[i], blk.m[i]);
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "mean[{i}] {a} vs {b}, {what}"
            );
        }
        for i in 0..k * k {
            let (a, b) = (seq.c[i], blk.c[i]);
            assert!(
                (a - b).abs() <= tol * a.abs().max(scale),
                "c[{i}] {a} vs {b}: {:.2e} of the scale {scale:.3e}, tol {tol:.1e}, {what}",
                (a - b).abs() / scale
            );
        }
    }

    /// The four scalars must be **bit-identical**, not close.
    ///
    /// They are what `n_eff` and `n_kish` are read from, and those are
    /// emitted on every row, so anything less than equality would change
    /// shipped output the moment blocking was switched on. It holds because
    /// both paths run the one copy of `step_factors` / `commit_scalars`.
    fn assert_scalars_equal(seq: &EwCov, blk: &EwCov, what: &str) {
        assert_eq!(seq.w_sum, blk.w_sum, "w_sum, {what}");
        assert_eq!(seq.q_sum, blk.q_sum, "q_sum, {what}");
        assert_eq!(seq.prior_scale, blk.prior_scale, "prior_scale, {what}");
        assert_eq!(
            seq.precision_scale, blk.precision_scale,
            "precision_scale, {what}"
        );
    }

    #[test]
    fn the_scalars_are_bit_identical_whichever_path_ran() {
        for &k in &[1usize, 3, 8, 37] {
            for &lam in &[1.0f64, 0.995] {
                for wmode in 0u8..3 {
                    for &block in &[2usize, 7, 64] {
                        let rows = stream(200, k, 11, 1e3, lam, wmode);
                        let seq = run(&rows, k, 0);
                        let blk = run(&rows, k, block);
                        let what = format!("k={k} lam={lam} wmode={wmode} block={block}");
                        assert_scalars_equal(&seq, &blk, &what);
                    }
                }
            }
        }
    }

    #[test]
    fn the_moments_match_the_rank_one_recursion() {
        for &k in &[1usize, 3, 8, 37] {
            for &lam in &[1.0f64, 0.995] {
                for wmode in 0u8..3 {
                    for &block in &[2usize, 7, 64] {
                        for &offset in &[0.0f64, 1e3, 1e7] {
                            let rows = stream(200, k, 5, offset, lam, wmode);
                            let seq = run(&rows, k, 0);
                            let blk = run(&rows, k, block);
                            let what = format!(
                                "k={k} lam={lam} wmode={wmode} block={block} offset={offset}"
                            );
                            assert_moments_close(&seq, &blk, &what);
                        }
                    }
                }
            }
        }
    }

    /// The product is `faer`'s, and a GEMM has tile edges the tests above
    /// never reach at `k <= 37`: a block shorter than `k`, a block longer
    /// than `k`, and a `k` that is not a multiple of any vector width.
    #[test]
    fn a_wide_block_matches_too() {
        let k = 131;
        let rows = stream(300, k, 17, 1e3, 0.998, 1);
        let seq = run(&rows, k, 0);
        for &block in &[64usize, 256] {
            let blk = run(&rows, k, block);
            let what = format!("k={k} block={block}");
            assert_scalars_equal(&seq, &blk, &what);
            assert_moments_close(&seq, &blk, &what);
        }
    }

    /// The merge's numerical question, put as a test. A long uncapped gap
    /// leaves the history *nearly* gone -- `W_A` of `1e-30`, not zero -- with
    /// `m_A` still where the old data was. A merge centred on `m_A` then
    /// forms the block's scatter as a difference of two numbers of order
    /// `U·(m_B − m_A)²`, and at an offset of `1e7` that leaves an error of
    /// **17% of the variance** in this test (49% in the design probe, and
    /// 24% centred on the block's first weighted row, which here is the one
    /// *before* the gap). The rank-1 recursion is exact, because its mean
    /// jumps to the row. Centred on the block's own mean the merge measures
    /// `8.4e-10` of the variance, which is the data's own resolution at
    /// `1e7`, and `3e-14` at `1e3`. Verified to fail, at the 17%, with the
    /// centring switched to `m_A`.
    #[test]
    fn a_history_all_but_gone_leaves_nothing_to_cancel() {
        let k = 2;
        for &(gap_lam, offset) in &[(1e-200f64, 1e7f64), (1e-30, 1e7), (1e-30, 1e3), (1e-8, 1e3)] {
            let mut rows = stream(50, k, 3, 0.0, 1.0, 0);
            let far = stream(8, k, 4, offset, 1.0, 0);
            for (i, (x, _, w)) in far.into_iter().enumerate() {
                let lam = if i == 0 { gap_lam } else { 1.0 };
                rows.push((x, lam, w));
            }
            // A block of 8 over 58 rows puts two old rows and the gap in
            // one block, which is the arrangement that fooled the row origin.
            let seq = run(&rows, k, 0);
            let blk = run(&rows, k, 8);
            let what = format!("gap_lam={gap_lam:e} offset={offset:e}");
            assert_scalars_equal(&seq, &blk, &what);
            let scale = seq.c[0].max(seq.c[3]);
            for i in 0..k * k {
                let (a, b) = (seq.c[i], blk.c[i]);
                assert!(
                    (a - b).abs() <= 1e-8 * scale,
                    "c[{i}] {a} vs {b}: {:.2e} of the variance, {what}",
                    (a - b).abs() / scale
                );
            }
            for i in 0..k {
                assert!(
                    (seq.m[i] - blk.m[i]).abs() <= 4.0 * f64::EPSILON * seq.m[i].abs().max(1.0),
                    "mean[{i}], {what}"
                );
            }
        }
    }

    /// `EwCov` has no notion of a chunk: rows arrive one call at a time, and
    /// a block boundary is a function of the row sequence alone. So this
    /// cannot fail, and is here as a pin of that fact and a place to write
    /// down the rule it implies for the caller.
    ///
    /// **A flush forced at a chunk boundary would move the last digits** --
    /// a flush every row is the rank-1 association, a flush every sixteen
    /// is a merge -- which is exactly the chunk-dependence CLAUDE.md rule 3
    /// forbids. So the flush may only ever be triggered by things that are
    /// functions of the row sequence: the block filling, and a read on a
    /// learned-row schedule (`solve_every`). It may never be triggered by a
    /// chunk ending. The test with teeth is the bank's chunk-invariance
    /// test with `gram_block_rows` on, which straddles block boundaries
    /// with chunk boundaries (docs/PLAN.md task 71).
    #[test]
    fn handing_the_rows_over_differently_changes_nothing() {
        let k = 4;
        let rows = stream(150, k, 23, 1e3, 0.997, 1);
        let whole = run(&rows, k, 16);
        for pause in [1usize, 5, 16, 17, 31, 64] {
            let mut c = EwCov::new(k);
            c.set_block_rows(16);
            for chunk in rows.chunks(pause) {
                for (x, lam, w) in chunk {
                    c.update(x, *lam, *w);
                }
            }
            c.flush();
            assert_eq!(c.w_sum, whole.w_sum, "w_sum at pause={pause}");
            assert_eq!(
                c.c, whole.c,
                "c differs when the rows arrive {pause} at a time"
            );
        }
    }

    /// The block fills on the row that makes it `block_rows` long, and on
    /// no other: one short holds, the exact count merges.
    #[test]
    fn the_block_fills_on_exactly_the_last_row() {
        let k = 3;
        let rows = stream(40, k, 19, 1e3, 0.99, 0);
        let mut c = EwCov::new(k);
        c.set_block_rows(8);
        for (i, (x, lam, w)) in rows.iter().enumerate() {
            c.update(x, *lam, *w);
            let held = (i + 1) % 8 != 0;
            assert_eq!(
                c.has_pending(),
                held,
                "after row {i}: {} rows held",
                c.pending.len()
            );
            if held {
                assert_eq!(c.pending.len(), (i + 1) % 8);
            }
        }
    }

    /// Flushing early must be indistinguishable from not needing to: a read
    /// mid-block is allowed and does not change the sequence's answer.
    #[test]
    fn an_early_flush_is_not_a_different_stream() {
        let k = 3;
        let rows = stream(90, k, 31, 0.0, 0.99, 2);
        let never = run(&rows, k, 32);
        let mut every = EwCov::new(k);
        every.set_block_rows(32);
        for (x, lam, w) in &rows {
            every.update(x, *lam, *w);
            every.flush(); // degenerates to the rank-1 path
        }
        assert_scalars_equal(&never, &every, "flush every row");
        assert_moments_close(&never, &every, "flush every row");
    }

    /// A read with rows still held is the one mistake the `ewridge` wiring
    /// can make, and it would be silent -- stale moments, not a crash -- so
    /// every matrix read asserts in debug builds. This is the test that
    /// checks the assertion is there.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "block still pending")]
    fn reading_the_matrix_mid_block_is_caught() {
        let k = 2;
        let mut c = EwCov::new(k);
        c.set_block_rows(4);
        c.update(&[1.0, 2.0], 1.0, 1.0);
        let _ = c.cov(0, 0);
    }

    /// The scalar reads are the ones a per-row emitter needs, and they are
    /// advanced by the shipped path as each row is held, so they are never
    /// stale and are not guarded.
    #[test]
    fn the_scalars_are_readable_mid_block() {
        let k = 2;
        let rows = stream(5, k, 29, 1e3, 0.99, 1);
        let mut seq = EwCov::new(k);
        let mut blk = EwCov::new(k);
        blk.set_block_rows(8);
        for (x, lam, w) in &rows {
            seq.update(x, *lam, *w);
            blk.update(x, *lam, *w);
            assert!(blk.has_pending());
            assert_eq!(blk.n_eff(), seq.n_eff());
            assert_eq!(blk.q_sum(), seq.q_sum());
            assert_eq!(blk.prior_scale(), seq.prior_scale());
        }
    }

    /// A clock gap past `max_dclock` is capped there, and with the documented
    /// defaults that cap is a `lam` of `2^-10000` -- **exactly zero**
    /// (CLAUDE.md rule 9 and docs/PLAN.md task 67). It is reachable from any
    /// stream with one long pause, and it lands *inside* a block, so the
    /// merge has to survive it. None of the other tests here use a `lam`
    /// below 0.99.
    ///
    /// It does *not* empty the accumulator, which is what this test was
    /// called before the branch was checked for reachability: `lam = 0`
    /// makes the row's own weight the whole of `w_sum`, so `step_factors`
    /// accepts it and the merge divides by a positive number.
    #[test]
    fn a_capped_gap_inside_a_block_matches_the_sequential_path() {
        for &block in &[2usize, 8, 64] {
            for at in [0usize, 1, 5, 30] {
                let k = 3;
                let mut rows = stream(60, k, 3, 1e3, 1.0, 1);
                rows[at].1 = 0.0; // the capped gap
                let seq = run(&rows, k, 0);
                let blk = run(&rows, k, block);
                let what = format!("block={block} gap_at={at}");
                assert_scalars_equal(&seq, &blk, &what);
                assert_moments_close(&seq, &blk, &what);
            }
        }
    }

    /// Every row in a block carrying zero weight: the history only ages, and
    /// `E_w` of a history that has merely aged is what it was. This reaches
    /// the no-weight arm of the merge, which no other test does -- the
    /// `wmode = 2` streams never put two zeros next to each other.
    #[test]
    fn a_block_of_nothing_but_zero_weight_rows_only_ages() {
        let k = 3;
        let mut rows = stream(40, k, 9, 1e3, 0.99, 0);
        for r in rows.iter_mut().skip(8).take(6) {
            r.2 = 0.0; // six consecutive zero-weight rows, inside one block
        }
        let seq = run(&rows, k, 0);
        let blk = run(&rows, k, 4);
        assert_scalars_equal(&seq, &blk, "a zero-weight block");
        assert_moments_close(&seq, &blk, "a zero-weight block");
    }

    /// A zero-weight row is legal as the **first** row of a stream, where it
    /// means "advance the clock, learn nothing" (CLAUDE.md rule 9). It must
    /// not open a block, because the shipped path treats it as no event at
    /// all -- not even the decay.
    #[test]
    fn a_zero_weight_first_row_opens_no_block() {
        let k = 2;
        let mut c = EwCov::new(k);
        c.set_block_rows(4);
        c.update(&[1e3, 1e3], 0.5, 0.0);
        assert!(
            !c.has_pending(),
            "a row the shipped path ignores must not be held"
        );
        assert_eq!(c.w_sum, 0.0);
        assert_eq!(
            c.prior_scale, 1.0,
            "the shipped path does not age on that row either"
        );

        let rows = stream(30, k, 13, 1e3, 0.995, 0);
        let mut with_lead = vec![(vec![1e3; k], 0.5, 0.0)];
        with_lead.extend(rows.iter().cloned());
        let a = run(&with_lead, k, 4);
        let b = run(&rows, k, 4);
        assert_eq!(a.w_sum, b.w_sum, "the leading no-op changed the count");
        assert_eq!(a.c, b.c, "the leading no-op changed the moments");
    }

    /// The pending rows travel in the state, so a bank saved mid-block and
    /// resumed must continue as though it never stopped -- including the
    /// block boundary, which is a function of the row sequence. Both
    /// msgpack encodings: the bank writes the named one, and the compact one
    /// is where a `skip_serializing_if` field in the wrong place shows
    /// (`crates/online-core/tests/state_encoding.rs`).
    #[test]
    fn a_state_saved_mid_block_resumes_identically() {
        let k = 3;
        let rows = stream(80, k, 41, 1e3, 0.996, 1);
        let whole = run(&rows, k, 16);
        for cut in [1usize, 7, 16, 23, 40] {
            let mut c = EwCov::new(k);
            c.set_block_rows(16);
            for (x, lam, w) in &rows[..cut] {
                c.update(x, *lam, *w);
            }
            let named = rmp_serde::to_vec_named(&c).unwrap();
            let compact = rmp_serde::to_vec(&c).unwrap();
            for (bytes, enc) in [(named, "named"), (compact, "compact")] {
                let mut back: EwCov = rmp_serde::from_slice(&bytes)
                    .unwrap_or_else(|e| panic!("{enc} encoding at cut={cut}: {e}"));
                assert_eq!(back, c, "{enc} round-trip changed the state at cut={cut}");
                assert_eq!(
                    back.has_pending(),
                    c.has_pending(),
                    "pending lost at cut={cut} ({enc})"
                );
                for (x, lam, w) in &rows[cut..] {
                    back.update(x, *lam, *w);
                }
                back.flush();
                assert_eq!(
                    back.w_sum, whole.w_sum,
                    "w_sum after resume at {cut} ({enc})"
                );
                assert_eq!(back.c, whole.c, "moments after resume at {cut} ({enc})");
            }
        }
    }

    /// The JSON export (task 69) is for reading, so the held block must come
    /// out as named, finite fields a reader can navigate, and go back in.
    #[test]
    fn a_held_block_is_legible_in_json() {
        let k = 2;
        let rows = stream(3, k, 43, 1e3, 0.99, 1);
        let mut c = EwCov::new(k);
        c.set_block_rows(8);
        for (x, lam, w) in &rows {
            c.update(x, *lam, *w);
        }
        let v = serde_json::to_value(&c).unwrap();
        let p = &v["pending"];
        assert_eq!(p["block_rows"], 8);
        assert_eq!(p["lam"].as_array().unwrap().len(), 3);
        assert_eq!(p["w"].as_array().unwrap().len(), 3);
        assert_eq!(p["x"].as_array().unwrap().len(), 3 * k, "row-major n*k");
        assert!(p["w_open"].is_number());
        let back: EwCov = serde_json::from_value(v).unwrap();
        assert_eq!(back, c);

        // And blocking off writes what it always did: no `pending` at all.
        let plain = serde_json::to_value(EwCov::new(k)).unwrap();
        assert!(plain.get("pending").is_none(), "{plain}");
    }

    /// A block of one, and a block the stream never fills: the two ends of
    /// the parameter. Neither may lose a row.
    #[test]
    fn a_block_of_one_and_a_block_longer_than_the_stream() {
        let k = 3;
        let rows = stream(25, k, 55, 1e3, 0.995, 1);
        let seq = run(&rows, k, 0);
        for &block in &[1usize, 1000] {
            let blk = run(&rows, k, block);
            let what = format!("block={block}");
            assert_scalars_equal(&seq, &blk, &what);
            assert_moments_close(&seq, &blk, &what);
        }
    }

    /// Changing the block size mid-stream flushes what is held first, so no
    /// row is carried across a size change and none is dropped.
    #[test]
    fn changing_the_block_size_mid_stream_keeps_every_row() {
        let k = 3;
        let rows = stream(60, k, 67, 1e3, 0.995, 1);
        let seq = run(&rows, k, 0);
        let mut c = EwCov::new(k);
        c.set_block_rows(8);
        for (i, (x, lam, w)) in rows.iter().enumerate() {
            if i == 21 {
                c.set_block_rows(3);
            }
            if i == 44 {
                c.set_block_rows(0); // and off entirely
            }
            c.update(x, *lam, *w);
        }
        c.flush();
        assert_scalars_equal(&seq, &c, "resized mid-stream");
        assert_moments_close(&seq, &c, "resized mid-stream");
    }

    /// `decay` ages `w_sum`, which a held block has snapshotted as `w_open`.
    /// Ageing it underneath the block leaves the merge computing `W_A` from a
    /// weight that no longer exists, so `decay` merges first. Verified to
    /// fail without that flush.
    #[test]
    fn ageing_the_accumulator_mid_block_does_not_strand_the_held_rows() {
        let k = 3;
        let rows = stream(40, k, 71, 1e3, 1.0, 0);
        // sequential: the same interruption, with nothing held to strand
        let mut seq = EwCov::new(k);
        for (i, (x, lam, w)) in rows.iter().enumerate() {
            if i == 11 {
                seq.decay(0.5);
            }
            seq.update(x, *lam, *w);
        }
        let mut blk = EwCov::new(k);
        blk.set_block_rows(8);
        for (i, (x, lam, w)) in rows.iter().enumerate() {
            if i == 11 {
                blk.decay(0.5); // lands mid-block, with three rows held
            }
            blk.update(x, *lam, *w);
        }
        blk.flush();
        assert_scalars_equal(&seq, &blk, "a mid-block decay");
        assert_moments_close(&seq, &blk, "a mid-block decay");
    }

    /// `set_moments` replaces the mean, the co-moments and the weight at
    /// once. Rows still held at that point were centred on a state that is
    /// about to vanish, so they must not survive the call to be merged into
    /// the caller's state later; `set_moments` flushes them first. That the
    /// flush merges rather than drops is not observable here -- the caller's
    /// values overwrite the result either way -- which is the point: what
    /// protects a caller that read the moments *before* flushing is the
    /// assertion on the reads, not this. `EwRidge::blend_toward_long_run` is
    /// the caller that makes it reachable as soon as `ewridge` turns
    /// blocking on. Verified to fail without the flush.
    #[test]
    fn replacing_the_moments_mid_block_leaves_no_row_behind() {
        let k = 2;
        let rows = stream(24, k, 83, 1e3, 0.99, 1);
        let mut blk = EwCov::new(k);
        blk.set_block_rows(8);
        // The reference sees the same rows and the same replacement, with
        // nothing held: `set_moments` leaves `prior_scale` and
        // `precision_scale` alone, so the history before it still counts.
        let mut seq = EwCov::new(k);
        for (x, lam, w) in rows.iter().take(5) {
            seq.update(x, *lam, *w);
            blk.update(x, *lam, *w);
        }
        assert!(
            blk.has_pending(),
            "the test needs rows held to mean anything"
        );
        let (m, c) = ([7.0, 9.0], [1.0, 0.0, 0.0, 1.0]);
        seq.set_moments(&m, &c, 3.0, Some(3.0));
        blk.set_moments(&m, &c, 3.0, Some(3.0));
        assert!(!blk.has_pending(), "set_moments must not leave rows held");
        assert_eq!(blk.w_sum, 3.0, "and the caller's weight is what stands");
        assert_eq!(blk.means(), &m);
        assert_eq!(blk.comoments(), &c);
        assert_scalars_equal(&seq, &blk, "at set_moments");
        // ...and rows after it merge against the caller's state, not the old
        // one: the same as the rank-1 path from that state on.
        for (x, lam, w) in rows.iter().skip(5) {
            seq.update(x, *lam, *w);
            blk.update(x, *lam, *w);
        }
        blk.flush();
        assert_scalars_equal(&seq, &blk, "after set_moments");
        assert_moments_close(&seq, &blk, "after set_moments");
    }

    /// Blocking off must be the code that shipped, to the bit.
    #[test]
    fn block_rows_zero_is_the_untouched_path() {
        let k = 5;
        let rows = stream(120, k, 77, 1e3, 0.995, 1);
        let a = run(&rows, k, 0);
        let mut b = EwCov::new(k);
        b.set_block_rows(8);
        b.set_block_rows(0); // asked for, then turned off again
        for (x, lam, w) in &rows {
            b.update(x, *lam, *w);
        }
        assert_eq!(a.w_sum, b.w_sum);
        assert_eq!(a.m, b.m);
        assert_eq!(a.c, b.c, "turning blocking off must restore the exact path");
    }
}
