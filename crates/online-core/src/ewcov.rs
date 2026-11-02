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

use crate::Runs;
use crate::since::Since;
use serde::{Deserialize, Serialize};

mod held;
mod lower;
mod merge_bound;
mod par;

pub(crate) use held::HeldSteps;
pub use lower::LowerTriangle;

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

/// Below this many features, on x86-64, the serial rank-one step updates
/// whole rows of the storage, upper entries too ([`rank_one_whole_rows`]);
/// from it up, and on every other target, the lower triangle's rows alone
/// ([`rank_one_lower_rows`]). A lower row is `i + 1` entries long, so at a
/// few features each row is mostly the vector loop's prologue and tail.
/// The loop timed alone, lower rows' time over whole rows' (task 295, P10):
/// on x86-64 (the VM) 1.04-1.37 at 2-8 features (1.33-1.37 at 4),
/// 0.90-1.06 at 10-12, 0.85-0.90 at 16-20, 0.72 at 32; on arm64 (an
/// Apple M-series core) about 1 at 2 features, 0.69-0.95 at 3-7, 11-15
/// and 17-20, and above 1 only where whole rows fill the vector loop's
/// unrolled width: 1.41 at 8, 1.16 at 9, 1.04-1.07 at 10, 1.06-1.15 at 16,
/// about 1 at 24, 0.85-0.90 at 32. Whole rows at
/// 2-9 features on arm64 made `ewridge` at 5 features 3.4% slower, so
/// there the lower rows stay at every width. `None`: never whole rows.
#[cfg(target_arch = "x86_64")]
const WHOLE_ROWS_BELOW: Option<usize> = Some(10);
#[cfg(not(target_arch = "x86_64"))]
const WHOLE_ROWS_BELOW: Option<usize> = None;

/// Whether the serial rank-one step at `k` features takes whole rows
/// ([`WHOLE_ROWS_BELOW`]).
#[inline]
fn whole_rows(k: usize) -> bool {
    WHOLE_ROWS_BELOW.is_some_and(|below| k < below)
}

/// The co-moments' rank-one step on one thread, `c_ij = a·c_ij + (a·b·d_i)·d_j`
/// over the lower triangle, `j ≤ i` ([`LowerTriangle`]): [`EwCov::update`]'s
/// serial path, the same expression per lower entry as its threaded one,
/// by whole rows or by the lower rows ([`whole_rows`]). Each lower entry is
/// the same number either way, and only the lower triangle is read,
/// compared or saved, so the choice moves no bit of a prediction or a
/// state.
#[inline]
fn rank_one_rows(c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64) {
    if whole_rows(k) {
        rank_one_whole_rows(c, k, d, a, b);
    } else {
        rank_one_lower_rows(c, k, d, a, b);
    }
}

/// [`rank_one_rows`] row by row over the lower triangle. Out of line, so the
/// loop is compiled on its own rather than inside `update` beside the
/// threaded branch, where it ran 21% slower than 0.13.0's identical loop at
/// `k = 20` (task 250).
#[inline(never)]
fn rank_one_lower_rows(c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64) {
    for i in 0..k {
        let ab_di = a * b * d[i];
        let row = &mut c[i * k..=i * k + i];
        for (cj, &dj) in row.iter_mut().zip(d.iter()) {
            *cj = a * *cj + ab_di * dj;
        }
    }
}

/// [`rank_one_rows`] over whole rows of the storage: each lower entry gets
/// [`rank_one_lower_rows`]' expression, and each upper entry one that no
/// read uses ([`LowerTriangle`]). Out of line, as that one.
#[inline(never)]
fn rank_one_whole_rows(c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64) {
    // The rows as chunks and `d`'s entries beside them, so neither is
    // indexed and checked a row. A chunk of 0 panics; at `k = 0` there is
    // no row, and `max(1)` takes none from an empty slice.
    for (row, &di) in c[..k * k].chunks_exact_mut(k.max(1)).zip(&d[..k]) {
        let ab_di = a * b * di;
        for (cj, &dj) in row.iter_mut().zip(d.iter()) {
            *cj = a * *cj + ab_di * dj;
        }
    }
}

/// How [`EwCov::update_by`] takes the co-moments' rank-one step: at once
/// ([`EachRow`], every update but a replay's) or held for a blocked merge
/// ([`HeldSteps`]). A type parameter, so [`EwCov::update`] decides
/// nothing about it per row.
pub(crate) trait CoStep {
    fn rank_one(&mut self, c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64, threads: Threads);
}

/// The rank-one step taken as the row arrives: on one thread by
/// [`rank_one_rows`], or the same expression per entry with the rows
/// split across threads ([`EwCov::set_threads`]).
pub(crate) struct EachRow;

impl CoStep for EachRow {
    #[inline(always)]
    fn rank_one(&mut self, c: &mut [f64], k: usize, d: &[f64], a: f64, b: f64, threads: Threads) {
        let workers = if threads.0 > 1 && k * k >= par::PAR_MIN_ENTRIES {
            threads.workers()
        } else {
            1
        };
        if workers > 1 {
            par::rows(c, k, workers, |i, row| {
                let ab_di = a * b * d[i];
                for (cj, &dj) in row[..=i].iter_mut().zip(d.iter()) {
                    *cj = a * *cj + ab_di * dj;
                }
            });
        } else {
            rank_one_rows(c, k, d, a, b);
        }
    }
}

/// A reusable row buffer that is not state: two accumulators with the same
/// moments are the same accumulator whatever is left in their scratch, and a
/// state file carries none of it.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scratch(pub(crate) Vec<f64>);

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// How many threads the `k²` work runs on ([`EwCov::set_threads`]):
/// configuration, not state. A state file carries none of it, and two
/// accumulators with the same moments are the same accumulator whatever
/// threads each runs on, so [`PartialEq`] ignores it. One by default.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Threads(usize);

impl Default for Threads {
    fn default() -> Self {
        Self(1)
    }
}

impl Threads {
    /// The threads the `k²` work runs on here: the count set, capped at
    /// the current pool's size.
    fn workers(self) -> usize {
        self.0.min(rayon::current_num_threads()).max(1)
    }
}

impl PartialEq for Threads {
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

/// A held block as [`EwCov::flush`] has it before its product: each row's
/// deviation from the block's own weighted mean, `a = x − m_B` (row-major
/// `n × k`), each row's weight at the end of the block `u`, the block's mean
/// `m_B` and the residue `δ = Σ uᵢ aᵢ / U` the mean leaves. What an owner
/// that holds more per row than the features -- `ewridge`'s targets
/// ([`crate::gaps`], docs/PLAN.md task 257) -- folds into its own moments
/// from the same pass over the block.
pub(crate) struct Block<'a> {
    pub(crate) a: &'a [f64],
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) u: &'a [f64],
    pub(crate) m_b: &'a [f64],
    pub(crate) delta: &'a [f64],
    /// The threads the owner's own `k`-wide work may run on
    /// ([`EwCov::set_threads`], capped at the pool's).
    pub(crate) threads: usize,
}

/// A merge's between-term weight, `W_A·U/W'²`: the history's and the
/// block's shares of `W'`, multiplied. Formed as the one quotient where
/// `W'²` is a normal double, and as the product of the two shares where it
/// is not -- past `W' ≈ 1.34e154`, where the square overflows, and below
/// `1.5e-154`, where it leaves the normal range, both read `inf/inf` or
/// `0/0` as the quotient (docs/PLAN.md task 269, review round 8 A-2). Every
/// merge whose square is normal keeps the quotient's bits.
#[inline]
pub(crate) fn between_share(w_a: f64, u_sum: f64, w_new: f64) -> f64 {
    let sq = w_new * w_new;
    if sq.is_normal() {
        w_a * u_sum / sq
    } else {
        (w_a / w_new) * (u_sum / w_new)
    }
}

/// What an owner holds beside a blocked accumulator's rows: told of each
/// row the block holds, as it is held, and handed the block when it is
/// merged ([`EwCov::update_riding`], [`EwCov::skip_riding`],
/// [`EwCov::flush_riding`]). `fold(None)` is a block merged with nothing to
/// fold -- no weight among its rows, or rows a damaged state could not
/// trust -- whose held values are dropped.
///
/// The rider is a type parameter, not a trait object, so the accumulator
/// without one -- [`NoRider`], every caller but a blocked `ewridge`'s
/// targets -- compiles to the code it had before riders existed: no check
/// per row, no call, and the merge's two passes in one loop.
pub(crate) trait Rider {
    /// Whether the rider reads the block. `false` only for [`NoRider`], and
    /// a constant, so the branch on it is decided at compile time.
    const RIDES: bool = true;
    fn held(&mut self);
    fn fold(&mut self, block: Option<&Block<'_>>);
}

/// No rider: [`EwCov::update`], [`EwCov::skip`] and [`EwCov::flush`].
pub(crate) struct NoRider;

impl Rider for NoRider {
    const RIDES: bool = false;
    #[inline(always)]
    fn held(&mut self) {}
    #[inline(always)]
    fn fold(&mut self, _: Option<&Block<'_>>) {}
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
    /// Written even when `None` (as nil). A state file is a map read by name
    /// (`rmp_serde::to_vec_named`), where a skipped field moves nothing; the
    /// compact encoding some round-trip tests use is positional, and there a
    /// field that skips anywhere but last slides the fields after it
    /// (`crates/online-core/tests/state_encoding.rs`). `pending` is the one
    /// that skips, and it is last.
    #[serde(default)]
    q_sum: Option<f64>,
    /// Product of all decay factors applied so far (a decaying prior's scale).
    prior_scale: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW **centered** co-moments, row-major `k*k`, kept as their lower
    /// triangle: symmetric by construction, and written whole, mirrored, to
    /// the state file ([`LowerTriangle`]; docs/PLAN.md task 294).
    c: LowerTriangle,
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
    #[serde(default)]
    precision_prior: f64,
    /// Row scratch: `x - m`, the deviations the co-moment update needs `k`
    /// times each (docs/ENHANCEMENTS.md E48). Not part of the state -- serde
    /// skips it and [`PartialEq`] ignores it, so a round-tripped accumulator
    /// still compares equal to the one that wrote it.
    #[serde(skip)]
    dev: Scratch,
    /// A blocked Gram's merges' rounding, bounded as they happen
    /// ([`merge_bound::MergeBound`]; docs/PLAN.md task 308): what lets the
    /// ridge's shortcut read a blocked Gram's system. Not state: serde skips
    /// it, [`PartialEq`] ignores it, and a loaded accumulator rebuilds it at
    /// its next merge from the matrix itself.
    #[serde(skip)]
    merged: merge_bound::MergeBound,
    /// Decaying scale `s` on that prior. It must decay by the *same* factor
    /// `a` the co-moments do, not by `lam`, so that `M` stays a fixed multiple
    /// of what it would be with a constant prior:
    /// `M' = a·C + a·b·δδᵀ + a·s·prior·I = a·(M + b·δδᵀ)`. Like RLS's `P₀`,
    /// this makes the prior fade as data accumulates.
    #[serde(default)]
    precision_scale: f64,
    /// Threads the `k²` work runs on ([`Self::set_threads`]). Not state:
    /// serde skips it, a restored accumulator runs on one until its owner
    /// sets it again, and [`PartialEq`] ignores it.
    #[serde(skip)]
    threads: Threads,
    /// Per feature, the value it has held on every row learned since it last
    /// changed, and the learned row that started the run ([`crate::Runs`]):
    /// what lets a window say that a feature held one value over it, which
    /// its subtraction cannot (docs/PLAN.md task 94).
    /// Empty in a state written before schema 16: the next learned row starts
    /// them, and until one covers a window the window reads that feature's
    /// spread from the subtraction, as schema 15 did. Always written: only
    /// `pending` may skip, last.
    #[serde(default)]
    runs: Runs,
    /// Learned rows so far: what the runs' start rows count in, and what a
    /// window's snapshot records. Ahead of `pending`.
    #[serde(default)]
    rows_learned: u64,
    /// What each mean in `m` leaves out: the mean is `m[i] + m_lo[i]`, a pair
    /// no step is rounded off ([`crate::comp`]; docs/PLAN.md task 101). A
    /// plain mean given one value row after row stopped a few rounding steps
    /// short of it, and left the co-moments on that gap. One per mean, as
    /// `m` is ([`Self::has_shape`]).
    m_lo: Vec<f64>,
    /// Rows buffered for a blocked Gram update, and `w_sum` as it stood when
    /// the block opened (the merge needs the pre-block weight, and the
    /// scalars have moved on by then).
    ///
    /// Empty and `0` unless [`Self::set_block_rows`] asked for blocking,
    /// which only `ewridge` does: blocking pays where the Gram is read on a
    /// schedule, and a model emitting per-row statistics would flush every
    /// row for nothing (docs/PLAN.md task 71).
    ///
    /// **Last, and it must stay last for the compact encoding.** A state
    /// file is written with `rmp_serde::to_vec_named`, a map read back by
    /// name, so where a field sits does not move the file. The compact
    /// encoding some round-trip tests use, `rmp_serde::to_vec`, writes a
    /// struct as a bare array, so a `skip_serializing_if` field anywhere else
    /// slides everything after it there: put here first, this decoded
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
            c: LowerTriangle::zeros(k),
            dev: Scratch(Vec::with_capacity(k)),
            merged: merge_bound::MergeBound::default(),
            precision_prior: 0.0,
            precision_scale: 1.0,
            threads: Threads::default(),
            runs: Runs::new(k),
            rows_learned: 0,
            m_lo: vec![0.0; k],
            pending: Pending::default(),
        }
    }

    /// This accumulator with no runs kept ([`Runs::off`]): for an owner
    /// without a window, the one reader of them (docs/PLAN.md task 128).
    #[must_use]
    pub fn without_runs(mut self) -> Self {
        self.runs = Runs::off();
        self
    }

    /// Whether this accumulator keeps runs for a window to read.
    pub fn keeps_runs(&self) -> bool {
        !self.runs.is_off()
    }

    /// Keep no runs from here: for a restored owner without a window, whose
    /// state may have been written before the flag (review 2026-09-26, C4).
    pub fn set_runs_off(&mut self) {
        self.runs = Runs::off();
    }

    /// What mean `i`'s double leaves out ([`crate::comp`]).
    #[cfg(test)]
    pub(crate) fn mean_lo(&self, i: usize) -> f64 {
        self.m_lo[i]
    }

    /// What each mean's double leaves out, one per mean.
    pub(crate) fn means_lo(&self) -> &[f64] {
        &self.m_lo
    }

    /// `x`'s deviation from mean `i`, the pair's ([`crate::comp::dev`]): good to
    /// a rounding of itself where `x` is near the mean, as a deviation from
    /// the double [`Self::mean`] is not.
    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// The value feature `i` has held on every learned row from the `row`-th
    /// on; `None` where it moved since, has no learned row yet, or the runs
    /// are unknown (a state written before them). By row since the review
    /// of 2026-09-25 (`crate::Runs`).
    pub fn held_from(&self, i: usize, row: u64) -> Option<f64> {
        self.runs.started_by(self.k, i, row)
    }

    /// Learned rows so far: the count a window's snapshot records, in which
    /// the runs' start rows are indices.
    pub fn rows_learned(&self) -> u64 {
        self.rows_learned
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
    /// lifted out of a fitted `ew_class`). `comoments` is row-major `k*k`, and
    /// only its lower triangle is read ([`LowerTriangle`]).
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
        ew.c.copy_from(comoments);
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

    /// Whether the accumulator is over `k` slots with its vectors sized to
    /// match: what a restored state must hold to be updated (review
    /// 2026-09-18, B3).
    pub fn has_shape(&self, k: usize) -> bool {
        self.k == k && self.m.len() == k && self.c.len() == k * k && self.m_lo.len() == k
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
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        if !self.has_precision_prior() {
            return None;
        }
        let k = self.k;
        let mut m = self.c.to_full();
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
    /// (docs/ENHANCEMENTS.md E45). `(1 + lam^d) / (1 - lam^d)` in the limit
    /// for unit weights `d` clock units apart.
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

    /// The factor that makes these moments' covariance unbiased,
    /// `W² / (W² − Q)` with `Q = Σw²` beside `W = Σw`: Polars' correction
    /// for `ewm_var(bias=False)`, in its arithmetic (`polars-compute`'s
    /// `ewm/cov.rs`, `EwmCovState::update_iter`), and the reliability
    /// weights' `V₁² / (V₁² − V₂)`. It is `n / (n − 1)` at Kish's size
    /// `n = W² / Q` ([`Self::n_kish`]), Bessel's correction at the number of
    /// equally weighted rows the moments are worth, so `n` unit rows with
    /// no decay give `n / (n − 1)`, `ddof = 1`.
    ///
    /// NaN where it is not defined: `W² − Q` is `2 Σᵢ<ⱼ wᵢwⱼ`, zero with at
    /// most one row of weight -- before any, on the first, and where every
    /// other row weighed 0 -- and a single value has no spread to correct,
    /// as Polars gives null there. NaN too where `Q` is not known or is 0
    /// beside a weight (a window whose sum of squares kept no digit,
    /// [`crate::truncated`]), as [`Self::n_kish`] gives `None`. Read, never
    /// stored: the moments themselves stay the weighted means.
    pub fn unbiased_factor(&self) -> f64 {
        let w2 = self.w_sum * self.w_sum;
        match self.q_sum {
            Some(q) if q > 0.0 && w2 - q > 0.0 && w2.is_finite() => w2 / (w2 - q),
            _ => f64::NAN,
        }
    }

    #[inline]
    pub fn prior_scale(&self) -> f64 {
        self.prior_scale
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first"
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
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        self.c.get(self.k, i, j) + self.m[i] * self.m[j]
    }

    /// Centered covariance, held directly, read from the lower triangle
    /// whichever way round `(i, j)` is asked for.
    #[inline]
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        self.c.get(self.k, i, j)
    }

    /// The centered co-moment matrix, `k*k` — the EW analogue of the
    /// centered `X'X / n` -- kept as its lower triangle: read an entry with
    /// [`LowerTriangle::get`], hand a factorization that reads the lower
    /// triangle [`LowerTriangle::lower`], and take a whole symmetric matrix
    /// with [`LowerTriangle::to_full`].
    ///
    /// This is the accumulator every solve reads, exposed so a caller can do
    /// something *other than* our solve with it: a custom penalty, an
    /// information criterion, `cond(G)`, a scree plot, forward stepwise or
    /// orthogonal matching pursuit — none of which needs a second pass over
    /// data that was never materialized (docs/ENHANCEMENTS.md E30).
    #[inline]
    pub fn comoments(&self) -> &LowerTriangle {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        &self.c
    }

    /// The EW mean vector, length `k`. Pairs with [`Self::comoments`]: the
    /// uncentered second moment is `c_ij + m[i]*m[j]` ([`Self::raw`]).
    pub fn means(&self) -> &[f64] {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        &self.m
    }

    /// Centered variance, floored at zero against rounding; a NaN stays NaN,
    /// where the floor read it as no spread at all (task 182).
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        debug_assert!(
            !self.has_pending(),
            "EwCov: read the moments with a block still pending; call `flush()` first"
        );
        crate::solve::clamp_rounding(self.cov(i, i))
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
        // The moments may have moved by rank-one steps the bound never saw;
        // the next merge rebuilds it from the matrix.
        self.merged.forget();
    }

    /// Per feature, a bound on the rounding a blocked accumulator's merges
    /// left in its co-moments: `C = C* + E`, `C*` semidefinite, `|E| ≤ M`
    /// entrywise for a nonnegative semidefinite `M` whose diagonal this is
    /// (the condition module's docs, *A blocked Gram's bound*). `None`
    /// unless blocking is on, nothing is pending, and the bound is known --
    /// it is not after a load, a blend or a switch of blocking, until the
    /// next merge.
    pub(crate) fn merge_rounding(&self) -> Option<&[f64]> {
        if self.pending.block_rows == 0 || self.has_pending() {
            return None;
        }
        self.merged.get(self.k)
    }

    pub fn block_rows(&self) -> usize {
        self.pending.block_rows
    }

    /// Run the `k²` work -- the per-row rank-1 update, a block's merge in
    /// [`Self::flush`] -- on up to `threads` threads of the current rayon
    /// pool (`0` is taken as `1`), and never on more than the pool has:
    /// [`usize::MAX`] is every thread of whichever pool runs it. The moments
    /// are the same bits at every count: the rank-1 update is one
    /// expression per entry, and a merge's product is cut into pieces fixed
    /// by `k` alone, each summed by one sequential call (the `par` module's
    /// doc). Configuration, not state: a restored accumulator runs on one
    /// thread until it is set again.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = Threads(threads.max(1));
    }

    /// The thread count [`Self::set_threads`] set; `1` unless it did.
    pub fn threads(&self) -> usize {
        self.threads.0
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.lam.is_empty()
    }

    /// The accumulator with any held block merged in, for a reader that
    /// must not move the model's block boundary: the bank's `gram()` and a
    /// closed row read mid-block, and where they fall in the stream is the
    /// user's business, not the update schedule's. Borrows when nothing is
    /// pending, which is every model but a blocked `ewridge`.
    pub fn flushed(&self) -> std::borrow::Cow<'_, Self> {
        if self.has_pending() {
            let mut merged = self.clone();
            merged.flush();
            std::borrow::Cow::Owned(merged)
        } else {
            std::borrow::Cow::Borrowed(self)
        }
    }

    /// Hold the row for a blocked merge, advancing the scalars now.
    ///
    /// A row the shipped update would ignore is not held: [`Self::step_factors`]
    /// returning `None` means nothing changes at all, not even the decay, so
    /// buffering it would put a row in the block that contributed no weight
    /// and no `lam`. Every held row therefore committed, which is what lets
    /// the suffix products in [`Self::flush`] be taken over the buffer as it
    /// stands.
    fn buffer<R: Rider>(&mut self, x: &[f64], lam: f64, w: f64, rider: &mut R) {
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
        rider.held();
        if self.pending.len() >= self.pending.block_rows {
            self.flush_riding(rider);
        }
    }

    /// Rows held for the next merge.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
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
    /// The product is `faer`'s, on the lower triangle only, then mirrored: a
    /// symmetric rank-B update is half a GEMM. Measured against the shipped
    /// rank-1 loop, single threaded, 256-row block: 6.6× at `k = 1,000` and
    /// `k = 2,000`, 4.6× at `k = 256`; a 64-row block gives 4.9×, 4.5× and
    /// 2.9×. The design probe's 9.5×/11.2× was a multithreaded product
    /// against a single-threaded baseline, and is not the number
    /// (docs/PLAN.md task 71). It runs on [`Self::set_threads`] threads, cut
    /// into pieces fixed by `k` alone, each summed by one sequential call, so
    /// the bits are the same at every count: a parallel product whose split
    /// followed the thread count would sum an entry in another order on
    /// another count (the `par` module, docs/PLAN.md task 225).
    ///
    /// `W'` is read from `w_sum`, which the per-row scalars already carried,
    /// so the merge cannot disagree with the count that was reported.
    pub fn flush(&mut self) {
        self.flush_riding(&mut NoRider);
    }

    /// [`Self::flush`], handing the block to `rider` between its two passes
    /// and its product ([`Block`]): the deviations from the block's mean
    /// and each row's weight, which the rider folds into moments of its own
    /// (`ewridge`'s targets, docs/PLAN.md task 257). The merge itself is the
    /// same operations on the same numbers with a rider or without; with one,
    /// the second pass scales the rows after the rider has read them,
    /// rather than in the same loop.
    pub(crate) fn flush_riding<R: Rider>(&mut self, rider: &mut R) {
        let n = self.pending.len();
        if n == 0 {
            return;
        }
        let drop_rows = |rider: &mut R| rider.fold(None);
        let k = self.k;
        debug_assert_eq!(self.pending.x.len(), n * k);
        debug_assert_eq!(self.pending.w.len(), n);
        if self.pending.x.len() != n * k || self.pending.w.len() != n {
            // Only a damaged state file can get here; the rows cannot be
            // trusted, and dropping them is the least wrong thing to do with
            // an accumulator whose scalars already counted them.
            drop_rows(rider);
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
            drop_rows(rider);
            self.pending.clear_rows();
            return;
        }

        let u_sum: f64 = u.iter().sum();
        let Some(first) = u.iter().position(|&ui| ui > 0.0) else {
            // Held rows, no weight among them: the history only aged, and
            // `E_w` of a history that has merely aged is what it was.
            drop_rows(rider);
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
        let centre = |row: &mut [f64], delta: &mut [f64], ui: f64| {
            for (xi, &mb) in row.iter_mut().zip(&m_b) {
                *xi -= mb;
            }
            if ui > 0.0 {
                for (di, &xi) in delta.iter_mut().zip(row.iter()) {
                    *di += ui * xi;
                }
            }
        };
        // `sqrt` is one of the operations IEEE 754 rounds exactly, so this
        // is the same on every platform (unlike `powi` or `exp`, which never
        // reach the state). Where `1/W'` is not a double -- a `W'` below
        // `5.6e-309`, in the subnormal range -- the rows are weighted by
        // their shares `uᵢ/W'` and the product taken whole, where `1/W'`
        // times it read `inf · 0` (docs/PLAN.md task 269); every other merge
        // keeps its bits.
        let inv_w = 1.0 / w_new;
        let shares = !inv_w.is_finite();
        let scale = |row: &mut [f64], ui: f64| {
            let root = if shares {
                (ui / w_new).sqrt()
            } else {
                ui.sqrt()
            };
            for xi in row.iter_mut() {
                *xi *= root;
            }
        };
        if R::RIDES {
            for (i, &ui) in u.iter().enumerate() {
                centre(&mut x[i * k..(i + 1) * k], &mut delta, ui);
            }
            for d in delta.iter_mut() {
                *d /= u_sum;
            }
            rider.fold(Some(&Block {
                a: x.as_slice(),
                n,
                k,
                u: &u,
                m_b: &m_b,
                delta: &delta,
                threads: self.threads.workers(),
            }));
            for (i, &ui) in u.iter().enumerate() {
                scale(&mut x[i * k..(i + 1) * k], ui);
            }
        } else {
            for (i, &ui) in u.iter().enumerate() {
                let row = &mut x[i * k..(i + 1) * k];
                centre(row, &mut delta, ui);
                scale(row, ui);
            }
            for d in delta.iter_mut() {
                *d /= u_sum;
            }
        }
        // Δ = m_B − m_A, with the residue folded in; m_A is the pair. The
        // covariance takes Δ as one double. The mean takes its two pieces
        // in turn: for a first block, or one far from the accumulator's
        // level, `(m_B − m_A) + δ` rounds δ away, and the pair would start
        // a rounding step off the block's mean.
        let d_hi: Vec<f64> = (0..k)
            .map(|i| crate::comp::dev(m_b[i], self.m[i], self.m_lo[i]))
            .collect();
        let big: Vec<f64> = d_hi.iter().zip(&delta).map(|(h, d)| h + d).collect();

        // C' on the lower triangle, which is all that is kept
        // ([`LowerTriangle`]): scale the history, add the product, then the
        // two rank-1 corrections -- each entry in that order, one piece of
        // the triangle at a time (`par::merge_lower`).
        let within = u_sum / w_new;
        let between = between_share(w_a, u_sum, w_new);
        let wi: Vec<f64> = delta.iter().map(|&di| within * di).collect();
        let bi: Vec<f64> = big.iter().map(|&gi| between * gi).collect();
        // The bound, rebuilt from the matrix where it is not known, before
        // the merge moves it.
        self.merged.open(&self.c, k);
        let merge = par::Merge {
            scale: w_a / w_new,
            alpha: if shares { 1.0 } else { inv_w },
            bi: &bi,
            wi: &wi,
            big: &big,
            delta: &delta,
        };
        par::merge_lower(
            self.c.lower_mut(),
            k,
            x.as_slice(),
            n,
            &merge,
            self.threads.workers(),
        );
        self.merged.merged(
            &self.c,
            k,
            &merge_bound::MergeStep {
                n,
                scale: merge.scale,
                alpha: merge.alpha,
                between,
                within,
                u_sum,
                big: &big,
                delta: &delta,
                shares,
            },
        );
        let share = u_sum / w_new;
        for ((i, mi), lo) in self.m.iter_mut().enumerate().zip(self.m_lo.iter_mut()) {
            crate::comp::add(mi, lo, share * d_hi[i]);
            // A block of one row, or of one value, has no residue, and a
            // step of 0 could round the pair afresh (`crate::comp::add`).
            if delta[i] != 0.0 {
                crate::comp::add(mi, lo, share * delta[i]);
            }
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
        self.update_riding(x, lam, w, &mut NoRider);
    }

    /// [`Self::update`], with `rider` told of the row if the block holds it
    /// and handed any block merged on the way ([`Rider`]): the one merge
    /// a full block brings, or the held rows a decay that takes the whole
    /// history merges before it.
    pub(crate) fn update_riding<R: Rider>(&mut self, x: &[f64], lam: f64, w: f64, rider: &mut R) {
        self.update_by(x, lam, w, rider, &mut EachRow);
    }

    /// [`Self::update`] with the co-moments' rank-one step held in `held`
    /// rather than taken ([`HeldSteps`]): every other number the row moves --
    /// the means, the weights, the prior's scale -- moves now, and the
    /// co-moments are behind until [`Self::merge_held`]. For an owner that
    /// replays many rows and reads the co-moments only after the last (the
    /// robust refit's passes); nothing may read them in between. On an
    /// accumulator that blocks its rows ([`Self::set_block_rows`]), whose
    /// merge reads the co-moments, the held steps are merged first and the
    /// row is an ordinary [`Self::update`].
    pub(crate) fn update_held(&mut self, x: &[f64], lam: f64, w: f64, held: &mut HeldSteps) {
        if self.pending.block_rows > 0 {
            self.merge_held(held);
            return self.update(x, lam, w);
        }
        self.update_by(x, lam, w, &mut NoRider, held);
    }

    /// The co-moments brought up to date with every step [`Self::update_held`]
    /// held, each entry's in the rows' order, to the bit of the rows taken
    /// one by one ([`HeldSteps`]).
    pub(crate) fn merge_held(&mut self, held: &mut HeldSteps) {
        if held.is_open() {
            self.merged.forget();
        }
        held.merge_into(self.c.lower_mut(), self.k);
    }

    /// [`Self::update_riding`], the co-moments' rank-one step taken by
    /// `step` ([`CoStep`]): at once ([`EachRow`]) or held ([`HeldSteps`]).
    #[inline(always)]
    fn update_by<R: Rider, S: CoStep>(
        &mut self,
        x: &[f64],
        lam: f64,
        w: f64,
        rider: &mut R,
        step: &mut S,
    ) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwCov::update requires a non-negative weight, got {w}"
        );
        // Every row with a weight counts for the runs, however light next to
        // the live weight: the window alone says what is nothing, by its own
        // weight (`crate::Runs`). A row below `EMPTY_FRACTION` of the live
        // weight counted for none, so a window of such rows, together above
        // the fraction, read each feature as held at the last heavy row's
        // value (review 2026-10-05, CE3).
        if w > 0.0 {
            self.rows_learned += 1;
            self.runs.track(x, self.rows_learned);
        }
        // A zero-weight row with no weight to come out of the update --
        // `lam·W` is 0: the decay took the whole history, from 1075
        // half-lives on, or there was none, a zero-weight head row -- is the
        // decay alone: the weight stays 0, and the prior's scale ages by
        // `lam`, as a row advances the clock (hard rule 9) and as `rls` ages
        // its factor. The update below is `0/0` there; refusing it kept the
        // history whole (task 115 (c), PLAN §12) but, at `W == 0`, left the
        // prior's scale at 1 across the row, 6.8e-2 from the definition
        // after two such rows (task 159, R1).
        if w == 0.0 && lam * self.w_sum == 0.0 {
            self.flush_riding(rider);
            return self.decay(lam);
        }
        if self.pending.block_rows > 0 {
            return self.buffer(x, lam, w, rider);
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
        d.extend(
            x.iter()
                .zip(&self.m)
                .zip(&self.m_lo)
                .map(|((&xi, &mi), &li)| crate::comp::dev(xi, mi, li)),
        );
        step.rank_one(self.c.lower_mut(), k, d, a, b, self.threads);
        // The precision prior decays with the co-moments. `a == 0` only on
        // the very first observation, when the whole history is discarded and
        // the centered co-moments are exactly zero: the prior starts over.
        // Each mean takes its step as a pair, so no part of it is rounded off
        // (docs/PLAN.md task 101); a row of weight 0 takes none
        // (`crate::comp::add` says why).
        if b > 0.0 {
            for ((mi, lo), &di) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(d.iter()) {
                crate::comp::add(mi, lo, b * di);
            }
        }
        self.commit_scalars(lam, w, a, w_new);
    }

    /// Overwrite the moments directly. Used when two accumulators are mixed
    /// (see `EwRidge::blend_toward_long_run`); the caller is responsible for
    /// the mixture being a valid set of weighted moments. `centered` is a
    /// whole row-major `k*k` matrix of which only the lower triangle is read
    /// ([`LowerTriangle`]): a caller may fill that alone.
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
        self.m_lo = vec![0.0; self.k];
        self.c.copy_from(centered);
        self.merged.forget();
        self.w_sum = w_sum;
        self.q_sum = q_sum;
        // New moments stand for rows the runs never saw (a blend with the
        // slow twin, say), so no run is known to cover any of them: each
        // starts again from the next row. That can leave a window reading a
        // held feature from its subtraction, never the reverse.
        self.runs.forget();
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

    /// Age the accumulator over a row it does not learn: [`Self::decay`],
    /// except that a blocked accumulator holds the row as a zero-weight one
    /// instead of flushing its block, so a Gram that skips rows often -- a
    /// target's, under `target_gaps = "own_rows"` (docs/PLAN.md task 81) --
    /// keeps the block's saving. The two are the same numbers: a zero-weight
    /// row moves neither the means nor the co-moments, and ages the weights
    /// by `lam` as `decay` does, by the same products. With no weight to age,
    /// or a `lam` that takes all of it, there is nothing to hold and `decay`
    /// is the answer.
    pub fn skip(&mut self, x: &[f64], lam: f64) {
        self.skip_riding(x, lam, &mut NoRider);
    }

    /// [`Self::skip`] with a [`Rider`], as [`Self::update_riding`].
    pub(crate) fn skip_riding<R: Rider>(&mut self, x: &[f64], lam: f64, rider: &mut R) {
        if self.pending.block_rows > 0 && lam * self.w_sum > 0.0 {
            self.buffer(x, lam, 0.0, rider);
        } else {
            self.flush_riding(rider);
            self.decay(lam);
        }
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
    /// What each mean leaves out: the mean is `mean[t] + mean_lo[t]`
    /// ([`crate::comp`]; docs/PLAN.md task 101). One per target, as `mean`
    /// is.
    mean_lo: Vec<f64>,
}

impl TargetMoments {
    pub fn new(n_targets: usize) -> Self {
        Self {
            mean: vec![0.0; n_targets],
            var: vec![0.0; n_targets],
            q: vec![0.0; n_targets],
            mean_lo: vec![0.0; n_targets],
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
    ///
    /// Always inlined: `ewridge`'s accumulator calls it from its per-row and
    /// its blocked path, and at two call sites `#[inline]` left it out of
    /// line, a call per target on every row (task 257's per-row cost).
    #[inline(always)]
    pub fn learn(&mut self, t: usize, y: f64, a: f64, b: f64, lam: f64, w: f64) {
        let d = crate::comp::dev(y, self.mean[t], self.mean_lo[t]);
        self.var[t] = a * self.var[t] + a * b * d * d;
        // A row of weight 0 takes no step (`crate::comp::add` says why).
        if b > 0.0 {
            crate::comp::add(&mut self.mean[t], &mut self.mean_lo[t], b * d);
        }
        self.q[t] = lam * lam * self.q[t] + w * w;
    }

    /// The moments as a window's snapshot holds them (docs/PLAN.md task 136):
    /// decayed by `lam` to the row the snapshot precedes, so `Q_t` by `lam²`
    /// while a mean and a variance do not decay. The means' low parts come
    /// too, since the truncation reads the two means' difference as the
    /// pairs': from the doubles alone it was off by a rounding step of the
    /// level, which at 1e8 left a held target a variance of 2.5e-9 (review
    /// 2026-10-05, CB2).
    pub(crate) fn decayed(&self, lam: f64) -> Self {
        Self {
            mean: self.mean.clone(),
            var: self.var.clone(),
            q: self.q.iter().map(|q| q * lam * lam).collect(),
            mean_lo: self.mean_lo.clone(),
        }
    }

    /// What each mean leaves out ([`crate::comp`]), one per target. For a
    /// snapshot's footprint.
    pub(crate) fn means_lo(&self) -> &[f64] {
        &self.mean_lo
    }

    /// The moments over the rows inside a window: these, less the snapshot
    /// `old` taken `f` of decay ago, per target by the pooling identity the
    /// Grams are truncated by ([`crate::truncated`]). `per[t]` is the
    /// target's `(ratio, g) = (W_u/W_R, W/W_R)` from its weights, or `None`
    /// when no row of it is left in the window, where the moments are NaN
    /// and `Q_t` 0:
    ///
    /// ```text
    /// m_R = m − ratio·d,   v_R = g·v − ratio·v_u − ratio·g·d²,   Q_R = Q − f²·Q_u,
    /// d = m_u − m          (the two means as pairs, `hi + lo`)
    /// ```
    ///
    /// With `crate::truncated`'s floor: a variance no larger than the
    /// rounding of the terms it is formed from, `64 ε (g·v + ratio·v_u)`,
    /// carries no digit of the window's spread and is 0, so a target held at
    /// one value over the window reads none (review 2026-10-05, CB2), and
    /// below the normal range 64 steps of `2⁻¹⁰⁷⁴` instead
    /// (`crate::window::rounding_floor`; docs/PLAN.md task 273). The
    /// Grams' other guard, a run that says a slot held one value since a
    /// row inside the window, has no target-side twin: the targets keep no
    /// runs. A `Q_R` within `64 ε` of `Q` keeps no digit and is 0, so no
    /// Kish size is read from it (review 2026-10-05, CE6b).
    pub(crate) fn truncated(&self, old: &Self, f: f64, per: &[Option<(f64, f64)>]) -> Self {
        let n = self.mean.len();
        let mut out = Self {
            mean: vec![f64::NAN; n],
            var: vec![f64::NAN; n],
            q: vec![0.0; n],
            mean_lo: vec![0.0; n],
        };
        for (t, p) in per.iter().enumerate() {
            let Some((ratio, g)) = *p else { continue };
            let (lo, lo_old) = (self.mean_lo[t], old.mean_lo[t]);
            let d = (old.mean[t] - self.mean[t]) + (lo_old - lo);
            out.mean[t] = self.mean[t] + (lo - ratio * d);
            // Rounding below zero is zero, and a NaN stays one: `max(0.0)`
            // read it as a target with no spread (review round 4, CB7).
            let var = crate::solve::clamp_rounding(
                g * self.var[t] - ratio * old.var[t] - ratio * g * d * d,
            );
            let terms = g * self.var[t] + ratio * old.var[t];
            out.var[t] = if var <= crate::window::rounding_floor(terms) {
                0.0
            } else {
                var
            };
            // And a Kish sum the subtraction leaves no digit of is 0, so
            // `Self::n_kish` reports no size (`crate::truncated`; review
            // 2026-10-05, CE6b).
            let q = self.q[t] - f * f * old.q[t];
            out.q[t] = if q > 64.0 * f64::EPSILON * self.q[t] {
                q
            } else {
                0.0
            };
        }
        out
    }

    /// A row where this target is null: time passes for its weight, and
    /// `Q_t` decays with the square of it, as in [`EwCov::decay`].
    #[inline]
    pub fn age(&mut self, t: usize, lam: f64) {
        self.q[t] *= lam * lam;
    }

    /// Mix toward another set of moments with the shares the caller mixes
    /// co-moments by (`a + b == 1`), leaving `Q` as it is; see
    /// `EwRidge::blend_toward_long_run`. Centred second moments are not
    /// additive across differing means; the mixture is the centred identity
    /// `var = a·var + b·var' + a·b·(m - m')²`, as the co-moments' is, and needs
    /// nothing the size of `m²`. It went through the raw moment `var + m²` and
    /// re-centred, which at a level of `1e8` leaves a unit variance at 0, 2 or
    /// 4 -- whatever the two rounded terms happened to differ by (review
    /// 2026-09-12, C16).
    pub fn blend(&mut self, other: &Self, t: usize, a: f64, b: f64) {
        let d = self.mean[t] - other.mean[t];
        self.var[t] = a * self.var[t] + b * other.var[t] + a * b * d * d;
        self.mean[t] = a * self.mean[t] + b * other.mean[t];
        self.mean_lo[t] = 0.0;
        // `Q` stays this side's, as the weight does: the blend mixes what the
        // moments say, not how much evidence stands behind them (docs/PLAN.md
        // task 145). Summed as a union of two row sets it would count every
        // row twice, since the twins see the *same* rows; kept, a blend of a
        // state with itself is the identity, as it is for the means and the
        // co-moments.
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
    pub min_weight: f64,
    /// Polars' `bias` (its `ewm_var`, `ewm_std`), with its default:
    /// `false` multiplies every second moment the model reports by
    /// [`EwCov::unbiased_factor`], `c = W² / (W² − Σw²)`, the unbiased form
    /// under unequal weights; `true` reports the weighted means as kept.
    /// The co-moments and the ridge on them are scaled together, `c·(C +
    /// s·prior·I)`, so `var`, `std`, `cov` and the components' variances
    /// move with `c`, `mahal` with `1/√c`, and the correlations --
    /// `corr`, `partial_corr`, `lag_corr`, the components' shares --
    /// not at all. The state is the same either way.
    #[serde(default)]
    pub bias: bool,
    /// Prior for the precision matrix `(C + s·prior·I)⁻¹`. Required by
    /// [`EwCovStat::PartialCorr`] and [`EwCovStat::Mahal`], its consumers.
    #[serde(default)]
    pub precision_prior: Option<f64>,
    /// Quantile levels of the past Mahalanobis scores, one slot each
    /// (`mahal_q<p>`): a threshold for [`EwCovStat::Mahal`] from its own
    /// history rather than from a χ² table, exponentially weighted at the
    /// model's decay ([`crate::EwQuantile`], docs/PLAN.md task 146). Needs
    /// `Mahal` in `stats`.
    #[serde(default)]
    pub mahal_quantiles: Vec<f64>,
    /// Number of principal components to track, `0` for none
    /// (docs/ENHANCEMENTS.md E38). Each component adds `k + 3` slots: its
    /// variance, its share of the total, its `k` loadings and the row's score.
    #[serde(default)]
    pub pca: usize,
    /// Clock units between refreshes of the components, as `solve_every`
    /// is the regressions' (docs/PLAN.md task 161): `0` refreshes on every
    /// row, and `inf` never by the clock, leaving `max_rows_between_pca`
    /// the cadence. The clock is the one the model is stepped on, so a gap
    /// capped at `gap_cap` counts as the cap, a row of weight zero advances
    /// it, and without a clock column it is the row index; it is measured
    /// from the last refresh on the rows' stamps, the decayed clock held
    /// exactly, where the caller hands them (task 180). Between
    /// refreshes the loadings are frozen, so a row's scores do not depend
    /// on how the stream was chunked. Infinite under a row cap alone, which
    /// a JSON export writes as a tag (`crate::humanfloat`), where serde_json
    /// would write null and not read it back.
    #[serde(default, with = "crate::humanfloat::f64_or_tag")]
    pub pca_every: f64,
    /// At most this many rows between refreshes, as `max_rows_between_solves`
    /// is the regressions': counted on every row the model is stepped with,
    /// rows of weight zero included -- the cadence rule the window's
    /// snapshots keep (review 2026-09-26, C7; review 2026-10-05, CB3).
    /// `u32::MAX` for none; `0` or `1` refreshes on every row. Whichever of
    /// the two comes first refreshes.
    #[serde(default = "no_row_cap")]
    pub max_rows_between_pca: u32,
    /// Lags to accumulate cross-moments at, in output order
    /// (docs/ENHANCEMENTS.md E56): strictly increasing and `>= 1`. Empty for
    /// none, which is what every accumulator did before task 48 -- so a
    /// state written then still loads, with no ring.
    #[serde(default)]
    pub lags: Vec<usize>,
    /// Clock units of history the statistics are computed from, with a *hard*
    /// cutoff: a row older than this contributes nothing at all, where the
    /// exponential weight alone would leave `0.5^(age/half_life)` of it
    /// (docs/PLAN.md §13). Inside the window the weights are still
    /// exponential -- this is not a flat window. Absent for the ordinary
    /// accumulator, which is what every state written before task 63 has.
    ///
    /// **Last, with `window_every` and `max_rows_between_snapshots`**: a
    /// state file is a map read by name, but the compact encoding some
    /// round-trip tests use is positional, so only the row cap, the last,
    /// skips when absent, and these two are written as nil.
    #[serde(default)]
    pub window: Option<f64>,
    /// Clock units between the snapshots the window is computed from, as
    /// `solve_every` is between a regression's solves (docs/PLAN.md task
    /// 162): `0` snapshots every row. With `max_rows_between_snapshots`
    /// too, whichever comes first; with neither, every row, which puts the
    /// boundary as close to `window` as the data allows. A coarser cadence
    /// divides the memory and moves the boundary *inward*: under a clock
    /// spacing the effective window is in `[window - window_every, window]`
    /// in clock units, exactly -- never longer than asked. The clock is the
    /// one the model is stepped on, so a gap capped at `gap_cap` counts as
    /// the cap.
    #[serde(default)]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots, counted on
    /// every row the model is stepped with, rows of weight zero included;
    /// `1` is every row, and `0` is refused (docs/PLAN.md task 196).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows_between_snapshots: Option<usize>,
}

/// No row cap on the components' refreshes (`max_rows_between_pca`).
fn no_row_cap() -> u32 {
    u32::MAX
}

impl EwCovCfg {
    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("ew_cov: {e}"))?;
        if self.n_features == 0 {
            return Err("ew_cov: at least one column is required".into());
        }
        // A NaN `min_weight` is never reached, and a negative one is no
        // floor; `inf` never reports, which is legal (review round 5, C6).
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err(format!(
                "ew_cov: min_weight must be >= 0, got {}",
                self.min_weight
            ));
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
                    "ew_cov: window_size must be finite and > 0 (got {w}); it is clock units of \
                     history to keep, and `None` is the ordinary accumulator that keeps all of it"
                ));
            }
            if !self.lags.is_empty() {
                return Err(
                    "ew_cov: window_size and lags do not combine yet; the lagged co-moments are a \
                     second accumulator with its own ring, and truncating one and not the \
                     other would report a windowed corr beside an unwindowed lag_corr"
                        .into(),
                );
            }
            if !self.mahal_quantiles.is_empty() {
                return Err(
                    "ew_cov: window_size and mahal_quantiles do not combine; the quantiles decay \
                     over every score the stream has produced, with no window, so \
                     thresholding a windowed distance against them compares two different \
                     histories"
                        .into(),
                );
            }
        }
        crate::window::check_cadence(
            "ew_cov",
            self.window,
            self.window_every,
            self.max_rows_between_snapshots,
        )?;
        if self.stats.contains(&EwCovStat::LagCorr) && self.lags.is_empty() {
            return Err(
                "ew_cov: lag_corr needs `lags` (which lags to accumulate, e.g. lags = [1, 2, 5])"
                    .into(),
            );
        }
        if !self.lags.is_empty() {
            // The list's own rules live with the accumulator, so the CLI and
            // the bank get one message; checked without building it, whose
            // ring reserves `max(lags)` rows (a `restore` runs this).
            crate::EwLagCov::check_lags(&self.lags).map_err(|e| format!("ew_cov: {e}"))?;
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
        // With `pca` or without: a cadence below 0 is no cadence, and the
        // spec refuses one either way.
        if self.pca_every.is_nan() || self.pca_every < 0.0 {
            return Err(format!(
                "ew_cov: pca_every must be >= 0 clock units (0 refreshes on every row), got {}",
                self.pca_every
            ));
        }
        Ok(())
    }

    /// Number of output slots, in emission order: the statistics, then the
    /// Mahalanobis quantiles, then `k + 3` per principal component.
    pub fn n_outputs(&self) -> usize {
        let k = self.n_features;
        let stats: usize = self.stats.iter().map(|s| self.width(s)).sum();
        stats + self.mahal_quantiles.len() + self.pca * (k + 3)
    }

    /// The slots one statistic occupies. One home for the arithmetic, read by
    /// `n_outputs` and by the step that finds the `mahal` slot for the
    /// quantile estimators; that step once had a copy of its own which counted
    /// `lag_corr` as `k(k-1)/2` slots, so beside a `lag_corr` the quantiles were
    /// fed a lagged correlation instead of the distance (review 2026-09-12,
    /// C15).
    pub fn width(&self, stat: &EwCovStat) -> usize {
        let k = self.n_features;
        match stat {
            EwCovStat::Mean | EwCovStat::Var | EwCovStat::Std => k,
            EwCovStat::Cov | EwCovStat::Corr | EwCovStat::PartialCorr => k * (k - 1) / 2,
            EwCovStat::Mahal => 1,
            // Both orientations and the auto terms: the lagged matrix is
            // not symmetric.
            EwCovStat::LagCorr => self.lags.len() * k * k,
        }
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
        use faer::prelude::*;
        if r == 0 || r > k || c.len() != k * k || c.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mat = Mat::from_fn(k, k, |i, j| c[i * k + j]);
        // At a fixed degree: faer's own call follows the pool's size in its last
        // bits (`crate::solve::self_adjoint_eigen`, docs/PLAN.md task 231).
        let (s, u) = crate::solve::self_adjoint_eigen(mat.as_ref())?;
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

    /// The components of `scale` times the matrix: each variance and the
    /// trace multiplied by it, and the loadings, unit vectors, as they are.
    #[must_use]
    pub fn scaled(mut self, scale: f64) -> Self {
        for e in &mut self.eig {
            *e *= scale;
        }
        self.trace *= scale;
        self
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
    /// The sketch of the past Mahalanobis scores, every level from one;
    /// `None` without levels.
    mahal_q: Option<crate::EwQuantile>,
    /// The components in force, refreshed every `pca_every` clock units or
    /// every `max_rows_between_pca` rows, whichever comes first.
    #[serde(default)]
    pca: Option<Pca>,
    /// Rows stepped since the last refresh, rows of weight zero included.
    #[serde(default)]
    rows_since_pca: usize,
    /// Where `pca_every`'s clock stands (docs/PLAN.md task 161): the stamp
    /// of the last refresh, the decayed clock held exactly (task 180).
    since_pca: Since,
    /// Lagged cross-moments, when the spec asks for lags (E56). Written as
    /// `nil` when absent. A state file is a map read by name; the compact
    /// encoding some round-trip tests use is positional, so there at most one
    /// field may be skipped and it must be the last, which is `win`. With
    /// both skipped, a windowed model without lags decoded its window into
    /// this slot in that encoding (review 2026-09-18, B6; the byte change
    /// rides on `SCHEMA_VERSION` 11).
    #[serde(default)]
    lag: Option<crate::EwLagCov>,
    /// The hard-cutoff window, when the spec asks for one (docs/PLAN.md §13).
    /// Absent otherwise, so an ordinary accumulator writes the bytes it
    /// always did and a state written before task 63 loads unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
    /// The features' standard deviations a row's correlations divide by,
    /// kept so a step allocates nothing (task 270, M-4): not state, and
    /// written nowhere, so neither encoding changes.
    #[serde(skip)]
    std: Scratch,
}

/// What a window needs beyond the accumulator: where the clock has got to,
/// and the snapshots to subtract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Windowed {
    /// Clock of the last learned row, summed from the steps, which is the
    /// reference the statistics are read at (the state is pre-decay for the
    /// row being reported on): what decays a snapshot forward. The edge is
    /// the snapshots' stamps' ([`crate::Stamp`], task 175).
    clock: f64,
    snaps: crate::Snapshots<crate::Moments>,
}

/// The next `n` slots of the row's output, for the caller to write whole:
/// `rest` keeps the slots after them. The output is sized once and every
/// slot written by index, so no slot is filled twice: a zero-fill of each
/// statistic's slots before the writes, where the writes are in a function
/// the compiler does not see through (`LowerTriangle::upper_rows_map`), was
/// a `memset` on every row: 5-12% of `ew_cov`'s sampled cycles on arm64,
/// and 6% of its time at 20 features (task 295).
fn slots<'a>(rest: &mut &'a mut [f64], n: usize) -> &'a mut [f64] {
    let (head, tail) = std::mem::take(rest).split_at_mut(n);
    *rest = tail;
    head
}

impl EwCovModel {
    pub fn new(cfg: EwCovCfg) -> Result<Self, String> {
        cfg.validate()?;
        let cfg_window = cfg.window;
        let cadence = crate::Cadence::of(cfg.window_every, cfg.max_rows_between_snapshots);
        let cov = match cfg.precision_prior {
            Some(p) => EwCov::with_precision_prior(cfg.n_features, p)?,
            None => EwCov::new(cfg.n_features),
        };
        // Runs are a window's (docs/PLAN.md task 128).
        let cov = if cfg.window.is_some() {
            cov
        } else {
            cov.without_runs()
        };
        let mahal_q = if cfg.mahal_quantiles.is_empty() {
            None
        } else {
            Some(crate::EwQuantile::new(&cfg.mahal_quantiles)?)
        };
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
            rows_since_pca: 0,
            since_pca: Since::default(),
            lag,
            win: match cfg_window {
                Some(w) => Some(Windowed {
                    clock: 0.0,
                    snaps: crate::Snapshots::with_cadence(w, cadence)?,
                }),
                None => None,
            },
            std: Scratch::default(),
        })
    }

    /// The accumulator the statistics are read from: the live one, or -- with
    /// a `window` -- the same accumulator with everything older than the
    /// window subtracted off.
    ///
    /// The subtraction is exact rather than approximate, because an EW sum
    /// contains its own past: everything before the row at the boundary's
    /// clock `u` is `lam^(t-u)` times the accumulator as it stood then, so
    /// `A(t) - lam^(t-u)·A(u)` is precisely the rest. The boundary is the
    /// oldest snapshot still inside the window, so the rows dropped are a
    /// superset of the rows the window excludes -- the guarantee is honoured
    /// exactly and the shortfall is at most one snapshot's spacing.
    ///
    /// Two things this does *not* carry over from the live accumulator: the
    /// precision prior fades with the whole stream's decay rather than the
    /// window's, and the subtraction is a difference of positives, so its
    /// precision falls with the fraction discarded (negligible at
    /// `window = 3·half_life`, where the correction is an eighth).
    fn view(&self) -> std::borrow::Cow<'_, EwCov> {
        use std::borrow::Cow;
        let Some(win) = self.win.as_ref() else {
            return Cow::Borrowed(&self.cov);
        };
        let Some((f, old)) = win.snaps.cut(win.clock, self.cfg.decay, 0) else {
            return Cow::Borrowed(&self.cov);
        };
        if old.w == 0.0 {
            // The boundary carries no weight: nothing has aged out yet. Read
            // the accumulator itself rather than recomputing it from a
            // subtraction of zero, so a window nothing has reached reports
            // the untruncated numbers to the bit.
            return Cow::Borrowed(&self.cov);
        }
        match crate::truncated(&self.cov, old, f) {
            Some(cov) => Cow::Owned(cov),
            None => {
                // Nothing inside the window: every row since the boundary
                // carried no weight, as rows of weight 0 after a gap longer
                // than the window do. A gap alone does not empty it: the row
                // after a gap reads the window before its own decay (hard
                // rule 8), as the previous row left it, and is inside it
                // from then on. Report an empty state rather than dividing
                // by it.
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
    ///
    /// Under a `window`, from the window's moments, as every other statistic
    /// is: this read the live accumulator, so `mahal` was the distance from
    /// the whole history's mean in the whole history's metric, emitted beside
    /// a windowed mean and variance (review 2026-09-12, C14).
    pub fn mahal(&self, x: &[f64]) -> f64 {
        let view = self.view();
        Self::mahal_of(&view, x, self.second_moment_scale(&view))
    }

    /// What the second moments read from `cov` are multiplied by
    /// ([`EwCovCfg::bias`]): 1 under `bias`, and otherwise
    /// [`EwCov::unbiased_factor`], NaN while the read holds fewer than two
    /// rows of weight. One weighted row leaves the co-moments exactly zero
    /// and its spread undefined, and `W² − Q` alone cannot always say so:
    /// once a row of weight 0 has aged that one row, `W²` and `Q` are two
    /// roundings of the same number, and their difference can land a hair
    /// above zero, where the factor would turn a variance of 0/0 into 0.
    fn second_moment_scale(&self, cov: &EwCov) -> f64 {
        if self.cfg.bias {
            1.0
        } else if self.rows_read() < 2 {
            f64::NAN
        } else {
            cov.unbiased_factor()
        }
    }

    /// The rows of positive weight the statistics are read from: every one
    /// the accumulator has learned, or under a window those learned since
    /// its boundary. A row before the boundary carries none of the weight
    /// read, the boundary's own weight being what the window subtracts, or
    /// nothing when [`Self::view`] reads the live accumulator because that
    /// weight is 0.
    fn rows_read(&self) -> u64 {
        let live = self.cov.rows_learned();
        match self.win.as_ref().and_then(|win| win.snaps.boundary()) {
            Some((_, old)) => old.rows.map_or(live, |r| live.saturating_sub(r)),
            None => live,
        }
    }

    /// The Mahalanobis distance of `x` from `cov`'s moments in the metric of
    /// `scale·(C + s·prior·I)`: the co-moments and their ridge scaled
    /// together, so the squared distance is divided by `scale`.
    /// [`Self::mahal_of`] from the precision matrix `p` of `cov`
    /// ([`EwCov::precision`]): `d² = δᵀPδ / scale`. The same quantity by
    /// the same factorization, its inverse formed and read where `mahal_of`
    /// solves against the row; the two agree to rounding.
    fn mahal_by(cov: &EwCov, x: &[f64], p: &[f64], scale: f64) -> f64 {
        let k = cov.k();
        let delta: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| cov.deviation(i, xi))
            .collect();
        let quad: f64 = delta
            .iter()
            .enumerate()
            .map(|(i, di)| {
                di * p[i * k..(i + 1) * k]
                    .iter()
                    .zip(&delta)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            })
            .sum();
        let d2 = quad / scale;
        if d2.is_finite() {
            d2.max(0.0).sqrt()
        } else {
            f64::NAN
        }
    }

    fn mahal_of(cov: &EwCov, x: &[f64], scale: f64) -> f64 {
        if !cov.has_precision_prior() {
            return f64::NAN;
        }
        let k = cov.k();
        let delta: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| cov.deviation(i, xi))
            .collect();
        // The lower triangle, which `solve_spd` reads.
        let mut m = cov.comoments().lower().to_vec();
        let ridge = cov.precision_prior * cov.precision_scale;
        for i in 0..k {
            m[i * k + i] += ridge;
        }
        match crate::solve::solve_spd(&m, &delta, k, 1) {
            Some((sol, _)) => {
                let d2 = delta.iter().zip(&sol).map(|(d, s)| d * s).sum::<f64>() / scale;
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
        // From the moments the scores are read against: under a `window`, the
        // window's. This read the live co-moments, so `pc<j>_var`, the share
        // and the loadings were the whole history's while `pc<j>_score` was
        // centred on the window's mean -- a loading from one history applied
        // about the centre of another (review 2026-09-12, C14).
        // The variances along the components are the scaled co-moments'
        // (`EwCovCfg::bias`), as of this refresh; a common factor moves no
        // loading.
        let fresh = {
            let view = self.view();
            let scale = self.second_moment_scale(&view);
            Pca::of(
                &view.comoments().to_full(),
                self.cfg.n_features,
                self.cfg.pca,
                self.pca.as_ref(),
            )
            .map(|p| p.scaled(scale))
        };
        if let Some(p) = fresh {
            self.pca = Some(p);
        }
        self.rows_since_pca = 0;
        self.since_pca.restart();
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

    /// The accumulated weight the statistics are read from: under a `window`,
    /// the weight inside it -- the number `predict` emits and gates
    /// `min_weight` on. This returned the live weight while the emitted field
    /// was windowed, so the bank's readers of the accessor (`Bank::coef`, a
    /// closed row) disagreed with the struct (review 2026-09-12, S15).
    pub fn n_eff(&self) -> f64 {
        self.window_n_eff()
    }

    /// The window's weight alone, as [`Self::view`]'s accumulator has it --
    /// one subtraction, with `crate::truncated`'s rule for an empty window --
    /// without the O(k²) copy the view is (review 2026-09-12, P3).
    fn window_n_eff(&self) -> f64 {
        let live = self.cov.n_eff();
        let Some(win) = self.win.as_ref() else {
            return live;
        };
        let Some((f, old)) = win.snaps.cut(win.clock, self.cfg.decay, 0) else {
            return live;
        };
        if old.w == 0.0 {
            return live;
        }
        let w = live - f * old.w;
        if w <= crate::window::EMPTY_FRACTION * live || !w.is_finite() {
            0.0
        } else {
            w
        }
    }

    /// The accumulator the statistics are read from: under a `window`, the
    /// truncated one (empty when nothing is inside it), otherwise the live one.
    /// What `Bank::gram` and a closed row report (review 2026-09-12, S19).
    pub fn windowed_cov(&self) -> std::borrow::Cow<'_, EwCov> {
        self.view()
    }

    /// Output slot labels, in emission order (used for field names): the
    /// statistics, then `mahal_q<p>` per quantile level, then per component
    /// `j`: `pc<j>_var`, `pc<j>_share`, `pc<j>_loading_<column>` for each
    /// column and `pc<j>_score`.
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
                            out.push(format!("partial_corr_{}_{}", names[i], names[j]));
                        }
                    }
                }
                EwCovStat::Mahal => out.push("mahal".to_string()),
                EwCovStat::LagCorr => {
                    for l in lags {
                        for a in names {
                            for b in names {
                                out.push(format!("lag_corr_l{l}_{a}_{b}"));
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
            out.extend(names.iter().map(|n| format!("pc{j}_loading_{n}")));
            out.push(format!("pc{j}_score"));
        }
        out
    }

    #[cfg(test)]
    fn read(&self, cov: &EwCov, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.read_into(cov, x, &mut out, &mut Vec::new());
        out
    }

    /// The row's statistics, into `out`, whatever it held; `std` is scratch
    /// for the standard deviations the correlations divide by.
    fn read_into(&self, cov: &EwCov, x: &[f64], out: &mut Vec<f64>, std: &mut Vec<f64>) {
        let k = self.cfg.n_features;
        // Sized to the row's slots, which a stream keeps from row to row, so
        // from the second row on this moves nothing; every slot is then
        // written below, whatever `out` held.
        let n = self.cfg.n_outputs();
        if out.len() != n {
            out.resize(n, 0.0);
        }
        let mut rest = out.as_mut_slice();
        // `bias`'s factor on every second moment, 1 under `bias`, where the
        // multiplication is exact. The correlations read the co-moments as
        // kept: the factor cancels from them.
        let scale = self.second_moment_scale(cov);
        // One O(k³) solve per row, only when partial correlations are wanted.
        let precision = self
            .cfg
            .stats
            .contains(&EwCovStat::PartialCorr)
            .then(|| cov.precision())
            .flatten();
        for stat in &self.cfg.stats {
            match stat {
                EwCovStat::Mean => {
                    for (i, o) in slots(&mut rest, k).iter_mut().enumerate() {
                        *o = cov.mean(i);
                    }
                }
                EwCovStat::Var => {
                    for (i, o) in slots(&mut rest, k).iter_mut().enumerate() {
                        *o = scale * cov.var(i);
                    }
                }
                EwCovStat::Std => {
                    for (i, o) in slots(&mut rest, k).iter_mut().enumerate() {
                        *o = (scale * cov.var(i)).sqrt();
                    }
                }
                EwCovStat::Cov => {
                    // Pair `(i, j)`, `j > i`, is the lower entry `(j, i)`.
                    let dst = slots(&mut rest, k * (k - 1) / 2);
                    cov.comoments().upper_rows_map(k, dst, |_, _, v| scale * v);
                }
                EwCovStat::Corr => {
                    // `k` square roots, not `k(k-1)`: at k = 20 the pairwise
                    // `sqrt` was 80% of `ew_cov`'s row (docs/PERFORMANCE.md
                    // §13). Same product of the same two roots, so the
                    // correlations are bit-identical.
                    // Each pair written into its slot (a `push` per pair
                    // checked the capacity and went through a stack slot,
                    // 36 instructions a pair), from the lower entry
                    // `(j, i)`, as `Cov`. The same quotient of the same
                    // numbers, in the same order.
                    std.clear();
                    std.extend((0..k).map(|i| cov.var(i).sqrt()));
                    let dst = slots(&mut rest, k * (k - 1) / 2);
                    let sd = &*std;
                    cov.comoments().upper_rows_map(k, dst, |i, j, cij| {
                        let d = sd[i] * sd[j];
                        if d > 0.0 {
                            (cij / d).clamp(-1.0, 1.0)
                        } else {
                            f64::NAN
                        }
                    });
                }
                EwCovStat::PartialCorr => {
                    let mut dst = slots(&mut rest, k * (k - 1) / 2).iter_mut();
                    for i in 0..k {
                        for (j, o) in ((i + 1)..k).zip(dst.by_ref()) {
                            *o = match &precision {
                                Some(p) => partial_corr(p, k, i, j),
                                None => f64::NAN,
                            };
                        }
                    }
                }
                // From the precision in hand where the partial correlations
                // asked for it: the same matrix factorized again was a
                // second `k³` a row (task 270, A-2).
                EwCovStat::Mahal => {
                    slots(&mut rest, 1)[0] = match &precision {
                        Some(p) => Self::mahal_by(cov, x, p, scale),
                        None => Self::mahal_of(cov, x, scale),
                    }
                }
                EwCovStat::LagCorr => {
                    // `C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`: the lagged
                    // covariance over the contemporaneous standard
                    // deviations, so a value of 1 means "a now is b `l` rows
                    // ago, scaled". Not clamped: unlike a contemporaneous
                    // correlation it is not bounded by 1 in finite samples,
                    // and clamping would hide that.
                    std.clear();
                    std.extend((0..k).map(|i| cov.var(i).sqrt()));
                    let dst = slots(&mut rest, self.cfg.lags.len() * k * k);
                    match &self.lag {
                        Some(lag) => {
                            // `restore` holds the lags to the cfg's.
                            let mut dst = dst.iter_mut();
                            for li in 0..lag.lags().len() {
                                for a in 0..k {
                                    for (b, o) in (0..k).zip(dst.by_ref()) {
                                        let d = std[a] * std[b];
                                        *o = if d > 0.0 {
                                            lag.get(li, a, b) / d
                                        } else {
                                            f64::NAN
                                        };
                                    }
                                }
                            }
                        }
                        None => dst.fill(f64::NAN),
                    }
                }
            }
        }
        if let Some(q) = &self.mahal_q {
            // `restore` holds the sketch's levels to the cfg's.
            let dst = slots(&mut rest, q.levels().len());
            for (l, o) in dst.iter_mut().enumerate() {
                *o = q.get(l).unwrap_or(f64::NAN);
            }
        }
        if self.cfg.pca > 0 {
            let dst = slots(&mut rest, self.cfg.pca * (k + 3));
            match &self.pca {
                Some(p) => {
                    // A component per `k + 3` slots. The model refreshes all
                    // `pca` of them at once; a loaded state may hold fewer,
                    // and their slots are NaN, where the row was short.
                    let mut comps = dst.chunks_exact_mut(k + 3);
                    for (j, o) in (0..p.r()).zip(comps.by_ref()) {
                        let v = &p.loadings[j * k..(j + 1) * k];
                        let (head, tail) = o.split_at_mut(2);
                        head[0] = p.eig[j];
                        head[1] = if p.trace > 0.0 {
                            p.eig[j] / p.trace
                        } else {
                            f64::NAN
                        };
                        let (load, score_at) = tail.split_at_mut(k);
                        load.copy_from_slice(v);
                        // The row's coordinate along the component, about the
                        // current mean: loadings frozen, centre live.
                        let score = v
                            .iter()
                            .zip(x)
                            .enumerate()
                            .fold(0.0, |acc, (i, (vi, xi))| acc + vi * cov.deviation(i, *xi));
                        score_at[0] = score;
                    }
                    comps.for_each(|o| o.fill(f64::NAN));
                }
                None => dst.fill(f64::NAN),
            }
        }
        debug_assert!(rest.is_empty(), "every slot is written");
    }

    /// [`crate::OnlineModel::predict`], into the caller's buffer: what
    /// `step` reports, read before the row (task 270, M-4).
    fn predict_into(&self, x: &[f64], d_clock: f64, out: &mut crate::Step, std: &mut Vec<f64>) {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            *out = refused;
            return;
        }
        // With a `window`, every statistic -- and `n_eff` with them -- comes
        // from the truncated accumulator, so `min_weight` gates on the
        // weight *inside* the window, which stops growing once the window is
        // full rather than rising for the life of the stream.
        // The weight gates, and the view is built only when there is a
        // statistic to read from it: an `ew_cov` that emits none (E43's
        // accumulate-only use) built the O(k²) copy every row for its
        // `n_eff` (review 2026-09-12, P3).
        let n_eff = self.window_n_eff();
        if self.cfg.n_outputs() == 0 {
            out.pred.clear();
        } else if n_eff >= self.cfg.min_weight {
            self.read_into(&self.view(), x, &mut out.pred, std);
        } else {
            out.pred.clear();
            out.pred.resize(self.cfg.n_outputs(), f64::NAN);
        }
        out.n_eff = n_eff;
        out.extra = None;
    }
}

impl crate::OnlineModel for EwCovModel {
    /// The window keys the row's snapshot by its stamp (task 175), and the
    /// components' cadence measures its clock by it (task 180), only when
    /// there are components: without, the cadence never moves.
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.stamp_next(stamp);
        }
        if self.cfg.pca > 0 {
            self.since_pca.stamp_next(stamp);
        }
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    /// The co-moments' per-row update; the lagged cross-moments stay on
    /// one thread.
    fn set_gram_threads(&mut self, threads: usize) {
        self.cov.set_threads(threads);
    }

    fn set_window_closed(&mut self, closed: crate::WindowClosed) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_closed(closed);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        self.win.as_ref().and_then(|win| win.snaps.over_budget())
    }

    /// The ring's shadow, its snapshots formed as the step forms them
    /// (docs/PLAN.md task 115 (d)).
    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        let win = self.win.as_ref()?;
        Some(crate::WindowShadow::new(win.clock, &win.snaps, || {
            crate::Moments::of(&self.cov, 1.0)
        }))
    }

    /// The step, into the caller's buffer (task 270, M-4).
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut crate::Step,
    ) {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature that is not a number reached the
        // co-moments and the lag matrices through the zero-weight update's
        // `0 · d`, and with a weight the ring (docs/PLAN.md task 183). A row
        // of weight 0 takes no ring slot (`EwLagCov`), so the rows either
        // side of a refused row are adjacent, as they are when the plumbing
        // skips one.
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        // Statistics are read before this row is folded in, so an `ew_cov`
        // column is usable as a feature for the same row without leaking it.
        let mut std = std::mem::take(&mut self.std.0);
        self.predict_into(x, d_clock, out, &mut std);
        self.std.0 = std;
        let lam = self.cfg.decay.factor(d_clock);
        if let Some(q) = self.mahal_q.as_mut() {
            // The row's own out-of-sample score joins the history it will be
            // thresholded against, as `resid_quantiles` does for |resid|.
            let slot = self
                .cfg
                .stats
                .iter()
                .take_while(|s| **s != EwCovStat::Mahal)
                .map(|s| self.cfg.width(s))
                .sum::<usize>();
            let score = out.pred[slot];
            // A zero-weight row is scored but not seen: its distance stays out
            // of the history the quantiles threshold against, as the stream
            // keeps a zero-weight residual out of `resid_quantiles` (review
            // 2026-09-12, S28). The clock reaches the history on every row,
            // scored or not.
            q.age(lam);
            if score.is_finite() && weight > 0.0 {
                q.add(score, weight);
            }
        }
        // Before the accumulator moves: the lagged update reads the same
        // pre-row weight and mean `EwCov::update` is about to consume, which
        // is what makes `l = 0` its co-moments to the bit.
        if let Some(lag) = self.lag.as_mut() {
            lag.update(
                x,
                (self.cov.means(), self.cov.means_lo()),
                self.cov.n_eff(),
                lam,
                weight,
            );
        }
        // The snapshot is the state *before* this row, decayed to this row's
        // clock, so subtracting it later retains this row and everything
        // after it. Taken before the update, and keyed by the row's stamp
        // (task 175) beside the clock the model accumulates itself, so the
        // cadence is counted in the stream and the boundary cannot depend on
        // how the data was chunked (hard rule 3).
        if let Some(win) = self.win.as_mut() {
            // Built inside the closure so the O(k²) snapshot is only formed on
            // the rows `offer` keeps, not on every row (review 2026-09-18, P1).
            win.snaps
                .learn_decayed(&mut win.clock, d_clock, &[lam], || {
                    crate::Moments::of(&self.cov, lam)
                });
        }
        self.cov.update(x, lam, weight);
        if self.cfg.pca > 0 {
            // A checkpoint after the update, on the clock or the rows,
            // whichever comes first, as the regressions' solves (task 161).
            // Both count every row the model is stepped with, rows of weight
            // zero included -- the rule the window's snapshots keep (review
            // 2026-09-26, C7; review 2026-10-05, CB3) -- so the components a
            // row is scored on never depend on the chunking and `predict`
            // sees the same frozen ones `step` does. The clock is measured on
            // the row's stamp (task 180).
            self.rows_since_pca += 1;
            self.since_pca.step(d_clock);
            let by_clock = self.cfg.pca_every <= 0.0 || self.since_pca.reached(self.cfg.pca_every);
            let by_rows = self.rows_since_pca >= self.cfg.max_rows_between_pca as usize;
            if self.n_eff() >= self.cfg.min_weight && (self.pca.is_none() || by_clock || by_rows) {
                self.refresh_pca();
            }
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        let mut out = crate::Step::default();
        self.predict_into(x, d_clock, &mut out, &mut Vec::new());
        out
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
                crate::model::check_cfg("ew_cov", m.cfg.validate())?;
                // Levels need the sketch they are read from.
                if m.mahal_q.is_none() && !m.cfg.mahal_quantiles.is_empty() {
                    return Err(crate::StateError::Invalid(
                        "ew_cov: the state keeps no sketch for its mahal_quantiles".into(),
                    ));
                }
                // A state written before task 48, resumed under a spec that
                // has since gained `lags`: the ring cannot be recovered, so
                // it starts empty and the matrices from zero.
                if m.lag.is_none() && !m.cfg.lags.is_empty() {
                    m.lag = Some(
                        crate::EwLagCov::new(m.cfg.n_features, m.cfg.lags.clone())
                            .map_err(crate::StateError::Invalid)?,
                    );
                }
                // What the cfg asks for, the state carries, at the cfg's
                // width, and nothing else; `restore` checked none of it
                // (review 2026-09-18, B3).
                let k = m.cfg.n_features;
                let lag_ok = match &m.lag {
                    Some(l) => l.has_shape(k, &m.cfg.lags),
                    None => m.cfg.lags.is_empty(),
                };
                let pca_ok = m.pca.as_ref().is_none_or(|p| {
                    p.eig.len() <= m.cfg.pca && p.loadings.len() == p.eig.len() * k
                });
                // And inside the sketch and the window's snapshots, which
                // a row indexes (review 2026-10-06, CB2).
                let parts = m.mahal_q.as_ref().is_none_or(crate::EwQuantile::has_shape)
                    && m.win
                        .as_ref()
                        .is_none_or(|w| w.snaps.iter().all(|s| s.has_shape(k)));
                if !m.cov.has_shape(k)
                    || !parts
                    || m.mahal_q.as_ref().map_or(&[][..], |q| q.levels())
                        != m.cfg.mahal_quantiles.as_slice()
                    || !lag_ok
                    || !pca_ok
                    || m.win.is_some() != m.cfg.window.is_some()
                {
                    return Err(crate::StateError::Invalid(
                        "ew_cov: the state has the wrong shape".into(),
                    ));
                }
                // The runs follow the window, not the file: a state written
                // before the flag tracks them, which a model without a
                // window never reads (task 128); and a windowed model whose
                // runs are off would read every held slot as moving, so it
                // is refused (review 2026-09-26, C4).
                match m.cfg.window {
                    None => m.cov.set_runs_off(),
                    Some(_) if !m.cov.keeps_runs() => {
                        return Err(crate::StateError::Invalid(
                            "ew_cov: a windowed state whose runs are off".into(),
                        ));
                    }
                    Some(_) => {}
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
mod bias_tests;

#[cfg(test)]
pub(crate) mod rounding;

#[cfg(test)]
mod damaged_state_tests;

#[cfg(test)]
mod tests;

/// The blocked Gram update (E51, docs/PLAN.md task 71).
///
/// Every test here runs the *same* rows through both paths and compares. The
/// point is not that the blocked path is self-consistent -- it is that the
/// existing recursion is the oracle, since that is the one every shipped
/// number has always come from.
#[cfg(test)]
mod block_tests;

#[cfg(test)]
mod refusal_tests;

#[cfg(test)]
mod path_tests;

#[cfg(test)]
mod symmetry_tests;
