//! A stream's output buffers for one run of rows ([`ChunkOut`]), moved out
//! of `stream.rs`, which had passed the size a source file is held to
//! (task 270, G-1).

use super::*;
use std::mem::size_of;

/// Flat output buffers for one (stream, chunk) task (docs/PERFORMANCE.md P1).
///
/// One allocation per output *column* for the whole chunk, rather than the
/// ~11 `Vec`s per row the previous `RowOut` needed. Every numeric buffer is
/// `n_slots * n_rows`, slot-major, so a slot's values are contiguous and the
/// scatter into the final column is a straight walk. NaN is null.
///
/// `processed` distinguishes "this row was skipped" from "this row produced a
/// NaN", which matters only for `drift` (a bool, with no NaN to spare) and for
/// `n_eff`, which is otherwise always finite.
pub struct ChunkOut {
    /// Absolute row indices this task wrote, in order.
    pub rows: Vec<usize>,
    /// Per row: false = skipped, every output is null.
    pub processed: Vec<bool>,
    /// `n_models * n_slots * n_rows` unless noted; NaN = null.
    pub pred: Vec<f64>,
    pub resid: Vec<f64>,
    pub sigma: Vec<f64>,
    pub zscore: Vec<f64>,
    pub autocorr: Vec<f64>,
    /// `(ic, r2, hit_rate)`, model-major: `n_models * 3 * n_slots * n_rows`.
    pub metrics: Vec<f64>,
    /// `(pred_lo, pred_hi, coverage)`, laid out like `metrics`.
    pub conformal: Vec<f64>,
    /// Task 221's values, `n_checks` per slot, laid out like `metrics`:
    /// `n_models * n_checks * n_slots * n_rows` ([`crate::checks`]).
    pub checks: Vec<f64>,
    /// Feature health under `emit_feature_health`, per instance and
    /// feature, not per slot: `n_models * 2 * k * n_rows`, the spread
    /// ratios' block and then the mean shifts' (task 221 (g)).
    pub health: Vec<f64>,
    /// Model-major: `n_models * n_levels * n_slots * n_rows`.
    pub resid_q: Vec<f64>,
    pub drift: Vec<bool>,
    /// `n_models * n_rows`.
    pub n_eff: Vec<f64>,
    /// `n_models * n_targets * n_rows`: lasso's selected penalty, or
    /// `huber`'s residual scale under `emit_huber_scale`; empty otherwise.
    pub lam_selected: Vec<f64>,
    /// Emitted on a cadence rather than every row, so it stays boxed:
    /// `[model][row]`.
    pub coef: Vec<Vec<Option<Vec<f64>>>>,
    /// `n_models * n_rows`: how settled the instance was before the row
    /// (docs/WARMUP-AND-CONVERGENCE.md §3); NaN = null, where nothing decays.
    pub settled: Vec<f64>,
    /// `n_models * n_rows`: why the row's predictions were withheld, as an
    /// index into [`WITHHELD_REASONS`] plus one; 0 = none (null).
    pub reason: Vec<u8>,
    /// `n_models * n_slots * n_rows` under `emit_error_inflation`, else
    /// empty: the row's own error inflation per slot.
    pub inflation: Vec<f64>,
    /// Each coefficient's data share, on `coef`'s cadence: `[model][row]`.
    pub support_coef: Vec<Vec<Option<Vec<f64>>>>,
    /// Each coefficient's standard error under `emit_se_coef`, on `coef`'s
    /// cadence: `[model][row]` (docs/PLAN.md task 116, F).
    pub se_coef: Vec<Vec<Option<Vec<f64>>>>,
    /// The robust standard errors under `emit_robust_se`, HC0's and Newey
    /// and West's, laid out like `se_coef` (docs/PLAN.md task 221 (c)).
    pub se_coef_hc0: Vec<Vec<Option<Vec<f64>>>>,
    pub se_coef_hac: Vec<Vec<Option<Vec<f64>>>>,
    /// `n_rows` each under `emit_clocks`, else empty: the row's own clock,
    /// and the newest learned row's when it was scored (task 152).
    pub scored_clock: Vec<Option<ClockValue>>,
    pub learned_clock: Vec<Option<ClockValue>>,
    /// Slot counts this layout was built for.
    pub n_models: usize,
    pub n_slots: usize,
    pub n_levels: usize,
    pub n_checks: usize,
}

/// Which optional buffers a spec fills: read by `ChunkOut::new` to size them
/// and by `ChunkOut::run_rows` to count the values a row writes.
pub(super) struct Buffers {
    pub(super) residuals: bool,
    pub(super) extras: bool,
    n_levels: usize,
    /// One value per target and row from the model's step extras: lasso's
    /// selected penalty, or `huber`'s scale under `emit_huber_scale`.
    per_target: bool,
}

impl Buffers {
    pub(super) fn of(spec: &Spec) -> Self {
        Self {
            // A model with no target to predict has no residual field, so its
            // `resid` buffer would be filled, written once per slot per row
            // and never read: for a 230-statistic `ew_cov` that was a fifth
            // of the row (docs/PERFORMANCE.md §13).
            residuals: !spec.model.predicts_no_target(),
            // `sigma` is also the loss that `emit_selected` and
            // `emit_averaged` rank slots by (E13/E14 reuse E12's tracked
            // error), so it has to be materialized for them even when it is
            // not itself an output field.
            extras: spec.emit_sigma || spec.emit_zscore || spec.emit_selected || spec.emit_averaged,
            n_levels: spec.resid_quantiles.as_ref().map_or(0, Vec::len),
            per_target: matches!(
                spec.model,
                crate::ModelKind::Lasso { .. }
                    | crate::ModelKind::Huber {
                        emit_huber_scale: true,
                        ..
                    }
            ),
        }
    }

    /// `f64` values one row writes across every buffer, for `n_models`
    /// instances of `n_slots` slots.
    fn per_row(&self, spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        let per = n_models * n_slots;
        let on = |flag: bool| if flag { per } else { 0 };
        per + on(self.residuals)
            + 2 * on(self.extras)
            + on(spec.emit_autocorr)
            + 3 * on(spec.emit_metrics)
            + 3 * on(spec.conformal.is_some())
            + crate::checks::value_names(spec).len() * per
            + if spec.emit_feature_health {
                n_models * 2 * spec.k()
            } else {
                0
            }
            + self.n_levels * per
            + n_models
            + n_models
            + on(spec.emit_error_inflation)
            + if self.per_target {
                n_models * spec.m()
            } else {
                0
            }
    }
}

impl ChunkOut {
    /// Buffers for `n_rows` rows of one stream, all null until written.
    pub fn new(spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) -> Self {
        let mut out = Self::empty();
        out.renew(spec, n_models, n_slots, n_rows);
        out
    }

    /// No buffers at all, for [`Self::renew`] to size.
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            processed: Vec::new(),
            pred: Vec::new(),
            resid: Vec::new(),
            sigma: Vec::new(),
            zscore: Vec::new(),
            autocorr: Vec::new(),
            metrics: Vec::new(),
            conformal: Vec::new(),
            checks: Vec::new(),
            health: Vec::new(),
            resid_q: Vec::new(),
            drift: Vec::new(),
            n_eff: Vec::new(),
            lam_selected: Vec::new(),
            coef: Vec::new(),
            settled: Vec::new(),
            reason: Vec::new(),
            inflation: Vec::new(),
            support_coef: Vec::new(),
            se_coef: Vec::new(),
            se_coef_hc0: Vec::new(),
            se_coef_hac: Vec::new(),
            scored_clock: Vec::new(),
            learned_clock: Vec::new(),
            n_models: 0,
            n_slots: 0,
            n_levels: 0,
            n_checks: 0,
        }
    }

    /// [`Self::new`]'s buffers, in the memory these already hold: every
    /// buffer cleared and filled as `new` fills it, allocating only where
    /// one has less room than it needs. `new` is this on empty buffers, so
    /// the two cannot differ. A bank of many small groups makes a run's
    /// buffers for every group of every chunk, and allocating and freeing
    /// them was the largest share of such a task (task 270, G-1).
    pub fn renew(&mut self, spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) {
        fn fill<T: Clone>(v: &mut Vec<T>, n: usize, x: T) {
            v.clear();
            v.resize(n, x);
        }
        fn lists(v: &mut Vec<Vec<Option<Vec<f64>>>>, n_models: usize, n_rows: usize) {
            v.truncate(n_models);
            for inner in v.iter_mut() {
                fill(inner, n_rows, None);
            }
            v.resize_with(n_models, || vec![None; n_rows]);
        }
        let per = n_models * n_slots * n_rows;
        let on = |flag: bool| if flag { per } else { 0 };
        let Buffers {
            residuals,
            extras,
            n_levels,
            per_target,
        } = Buffers::of(spec);
        let n_checks = crate::checks::value_names(spec).len();
        self.rows.clear();
        self.rows.reserve(n_rows);
        fill(&mut self.processed, n_rows, false);
        fill(&mut self.pred, per, f64::NAN);
        fill(&mut self.resid, on(residuals), f64::NAN);
        fill(&mut self.sigma, on(extras), f64::NAN);
        fill(&mut self.zscore, on(extras), f64::NAN);
        fill(&mut self.autocorr, on(spec.emit_autocorr), f64::NAN);
        fill(&mut self.metrics, 3 * on(spec.emit_metrics), f64::NAN);
        fill(
            &mut self.conformal,
            3 * on(spec.conformal.is_some()),
            f64::NAN,
        );
        fill(&mut self.checks, n_checks * per, f64::NAN);
        let health = if spec.emit_feature_health {
            n_models * 2 * spec.k() * n_rows
        } else {
            0
        };
        fill(&mut self.health, health, f64::NAN);
        fill(&mut self.resid_q, n_levels * per, f64::NAN);
        fill(&mut self.drift, on(spec.emit_drift), false);
        fill(&mut self.n_eff, n_models * n_rows, f64::NAN);
        let lam = if per_target {
            n_models * spec.m() * n_rows
        } else {
            0
        };
        fill(&mut self.lam_selected, lam, f64::NAN);
        lists(&mut self.coef, n_models, n_rows);
        fill(&mut self.settled, n_models * n_rows, f64::NAN);
        fill(&mut self.reason, n_models * n_rows, 0);
        fill(&mut self.inflation, on(spec.emit_error_inflation), f64::NAN);
        lists(&mut self.support_coef, n_models, n_rows);
        lists(&mut self.se_coef, n_models, n_rows);
        lists(&mut self.se_coef_hc0, n_models, n_rows);
        lists(&mut self.se_coef_hac, n_models, n_rows);
        let clocks = if spec.emit_clocks { n_rows } else { 0 };
        fill(&mut self.scored_clock, clocks, None);
        fill(&mut self.learned_clock, clocks, None);
        self.n_models = n_models;
        self.n_slots = n_slots;
        self.n_levels = n_levels;
        self.n_checks = n_checks;
    }

    /// The bytes these buffers hold, the struct itself included: what a
    /// bank keeps when it keeps them for a later run ([`Self::renew`]).
    pub fn heap_bytes(&self) -> usize {
        self.flat_bytes() + self.list_bytes()
    }

    /// The bytes these buffers hold, if they are at most `limit`. The flat
    /// buffers are summed first, from their headers alone, so a long run,
    /// whose flat buffers pass a small `limit` by themselves, is answered
    /// without walking its `coef`-like lists, one `Option` per model and row
    /// each (task 270, R2-1).
    pub fn heap_bytes_within(&self, limit: usize) -> Option<usize> {
        let flat = self.flat_bytes();
        if flat > limit {
            return None;
        }
        Some(flat + self.list_bytes()).filter(|&bytes| bytes <= limit)
    }

    /// The bytes held by the struct and its flat buffers: every buffer but
    /// the `coef`-like lists, read from the headers alone.
    fn flat_bytes(&self) -> usize {
        size_of::<Self>()
            + cap(&self.rows)
            + cap(&self.processed)
            + [
                &self.pred,
                &self.resid,
                &self.sigma,
                &self.zscore,
                &self.autocorr,
                &self.metrics,
                &self.conformal,
                &self.checks,
                &self.health,
                &self.resid_q,
                &self.n_eff,
                &self.lam_selected,
                &self.settled,
                &self.inflation,
            ]
            .into_iter()
            .map(cap)
            .sum::<usize>()
            + cap(&self.drift)
            + cap(&self.reason)
            + cap(&self.scored_clock)
            + cap(&self.learned_clock)
    }

    /// The bytes held by the `coef`-like lists: each list's header and
    /// every vector it holds.
    fn list_bytes(&self) -> usize {
        fn lists(v: &Vec<Vec<Option<Vec<f64>>>>) -> usize {
            cap(v)
                + v.iter()
                    .map(|inner| cap(inner) + inner.iter().flatten().map(cap).sum::<usize>())
                    .sum::<usize>()
        }
        [
            &self.coef,
            &self.support_coef,
            &self.se_coef,
            &self.se_coef_hc0,
            &self.se_coef_hac,
        ]
        .into_iter()
        .map(lists)
        .sum::<usize>()
    }

    /// How many rows one learning run covers (docs/PERFORMANCE.md §13).
    ///
    /// The buffers are slot-major, so a row is one store per slot, each a
    /// slot's stride apart. Two things about that stride decide how fast a
    /// wide model runs, and both are set here rather than by whoever chose
    /// the chunk size:
    ///
    /// - The run's buffers stay within `BUDGET` bytes, so the lines a wide
    ///   model's row touches are still in cache when the next rows complete
    ///   them; over a 400k-row chunk, a 230-statistic `ew_cov` spent a third
    ///   of its time on stores to lines that had already been evicted. The
    ///   run's feature rows count against it too.
    /// - A slot's stride is an odd number of 128-byte lines, so no two slots
    ///   of a row share a cache set. A power-of-two chunk (65 536 rows, the
    ///   natural streaming batch) put every slot of that `ew_cov` in the
    ///   same set and ran it 2-3x slower than a chunk 16 rows longer.
    ///
    /// A narrow model writes a few values per row and gets runs of a hundred
    /// thousand rows: a chunk is one or two, and nothing changes for it. A
    /// model too wide for the budget gets the floor, five lines. Chunk
    /// invariance is what makes the split invisible.
    pub fn run_rows(spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        const BUDGET: usize = 2 << 20;
        /// `f64`s in a 128-byte line, the longest of the two OSes' lines.
        const LINE: usize = 16;
        /// The row's schedule, written in pass 1 and read in pass 2, in
        /// `f64`s: 128 bytes since the exact stamps joined it, against 56
        /// before. Left out of the budget, a narrow model's run of a hundred
        /// thousand rows held 13 to 16 MB of it, past the cache, and a `seqtest`
        /// row spent a large share storing and reloading it (task 250).
        const PLAN: usize = std::mem::size_of::<RowPlan>().div_ceil(8);
        // The run's features as rows, which the bank copies from the
        // chunk's columns just before the run and the passes read back: a
        // `f64` a feature (docs/PERFORMANCE.md §42). Inside the budget they
        // are read from cache; the whole chunk in rows was written to
        // memory and read back, 160 MB at 20,000 rows of 1,000 features.
        let width = Buffers::of(spec).per_row(spec, n_models, n_slots).max(1) + PLAN + spec.k();
        let lines = BUDGET / (8 * width * LINE);
        let lines = if lines.is_multiple_of(2) {
            lines.saturating_sub(1)
        } else {
            lines
        };
        lines.max(5) * LINE
    }

    /// Offset of `(model, slot)` at row `ri`, in any `n_models * n_slots *
    /// n_rows` buffer. The single place the layout is spelled out: each
    /// instance's block is written by `run_instance` at `slot * n_rows + ri`,
    /// and the sink in `bank/assembly.rs` reads the whole buffer through this.
    #[inline]
    pub fn at(n_slots: usize, n_rows: usize, mi: usize, slot: usize, ri: usize) -> usize {
        (mi * n_slots + slot) * n_rows + ri
    }

    /// The first row when this task's rows are one unbroken run `base..base +
    /// n`: a single group, or a group that arrives in blocks. `rows` is
    /// strictly increasing by construction (`group_indices` walks the chunk
    /// in order and each row belongs to one group), so the two ends decide.
    pub fn contiguous(&self) -> Option<usize> {
        let (&first, &last) = (self.rows.first()?, self.rows.last()?);
        debug_assert!(
            self.rows.windows(2).all(|w| w[0] < w[1]),
            "rows out of order"
        );
        (last - first + 1 == self.rows.len()).then_some(first)
    }
}

/// The bytes a vector's buffer holds: its capacity, not its length.
fn cap<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}
