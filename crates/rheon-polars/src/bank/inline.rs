//! Running a call on the calling thread rather than on the bank's pool, and
//! the parallel sections that follow it.

use super::*;

/// `items.into_par_iter().map(f).collect()` when `par`, the same on this
/// thread otherwise.
pub(super) fn map_vec_maybe_par<T, R, C>(
    items: Vec<T>,
    par: bool,
    f: impl Fn(T) -> R + Sync + Send,
) -> C
where
    T: Send,
    R: Send,
    C: FromIterator<R> + FromParallelIterator<R>,
{
    if par {
        items.into_par_iter().map(f).collect()
    } else {
        items.into_iter().map(f).collect()
    }
}

/// `(0..n).into_par_iter().map(f).collect()` when `par`, the same on this
/// thread otherwise.
pub(super) fn map_range_maybe_par<R, C>(
    n: usize,
    par: bool,
    f: impl Fn(usize) -> R + Sync + Send,
) -> C
where
    R: Send,
    C: FromIterator<R> + FromParallelIterator<R>,
{
    if par {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// `items.par_iter().map(f)` when `par`, the same on this thread otherwise.
pub(super) fn map_maybe_par<T, R, C>(items: &[T], par: bool, f: impl Fn(&T) -> R + Sync + Send) -> C
where
    T: Sync,
    R: Send,
    C: FromIterator<R> + FromParallelIterator<R>,
{
    if par {
        items.par_iter().map(f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

/// `rayon::join(a, b)` when `par`, `a` then `b` on this thread otherwise.
pub(super) fn join_maybe_par<A, B>(
    par: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    if par { rayon::join(a, b) } else { (a(), b()) }
}

/// Whether the parallel sections of this call run on the pool: not when the
/// call runs inline on the calling thread ([`crate::pool::Inline`]).
pub(super) fn par() -> bool {
    !crate::pool::inline()
}

/// The work, `rows * sum over specs of instances * outputs * (features + 1)^2`,
/// up to which a call runs on the calling thread (see [`Bank::runs_inline`]).
pub(super) const INLINE_MAX_WORK: usize = 1500;

/// The widest spec that runs inline: below it, no Gram update or
/// factorization splits over a pool.
pub(super) const INLINE_MAX_FEATURES: usize = 64;

impl Bank {
    /// Whether a call over a chunk of `rows` runs on the calling thread
    /// ([`crate::pool::Inline`]) rather than being handed to the bank's pool.
    ///
    /// Handing a call over and waking a parked worker costs about 13 us on
    /// the Mac: a one-row call falls from 40 us to 28 on 4 and 14 threads
    /// (34 to 28 on one), and a 3-spec bank's 70-90 us to 54-62. The pool pays
    /// once there are tasks to spread: a 5-feature `ewridge` on 10 groups
    /// ran faster inline to 50 rows and slower from 100 (1.2x on 4 threads,
    /// 1.0x on 14), a 3-spec bank to 30 rows and slower from 100 (1.2x),
    /// on 1, 4 and 14 threads, 1 to 10,000 rows and 1 to 1,000 groups. The
    /// cut is therefore on the work of the call, in units of `rows` times
    /// each spec's `(features + 1)^2` times its instances and outputs: up
    /// to [`INLINE_MAX_WORK`], about 40 rows of that `ewridge`.
    ///
    /// A bank of one ungrouped, single-instance spec has one task whatever
    /// the chunk, so a second thread has nothing to take: it runs inline at
    /// every size below [`PAR_MIN_ROWS`] (0.84 to 0.97 of the pool's time
    /// from 1 to 3,000 rows, 1.03 at 10,000, which keeps the pool).
    ///
    /// Not inline: a spec that sends work to rayon on its own account
    /// ([`Spec::threads_inside`]), one wider than [`INLINE_MAX_FEATURES`],
    /// where a Gram update or a factorization splits over the pool, and a
    /// caller already on a rayon thread, which is on a pool and stays there.
    /// The arithmetic of a task is the same on any thread, so the output is
    /// the same to the bit.
    pub(super) fn runs_inline(&self, rows: usize) -> bool {
        if rows >= PAR_MIN_ROWS
            || rayon::current_thread_index().is_some()
            || self
                .specs
                .iter()
                .any(|s| s.threads_inside() || s.features.len() > INLINE_MAX_FEATURES)
        {
            return false;
        }
        // One task and one instance: nothing for a second thread to take.
        if let [spec] = self.specs.as_slice()
            && spec.group.is_none()
            && self.derived[0].n_models <= 1
        {
            return true;
        }
        let per_row: usize = self
            .specs
            .iter()
            .zip(&self.derived)
            .map(|(s, d)| d.n_models.max(1) * d.per_model.max(1) * (s.features.len() + 1).pow(2))
            .sum();
        rows.saturating_mul(per_row) <= INLINE_MAX_WORK
    }
}
