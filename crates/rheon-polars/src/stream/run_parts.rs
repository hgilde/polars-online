//! What a stream's run is made of beyond the models: the instances split
//! out of the stream's state ([`build_instances`]), the plans' list a thread
//! keeps between runs, and the slots' lists joined into one. Moved out of
//! `stream.rs`, which is held under the size a source file is.

use super::*;

/// Split per-instance state and output into disjoint pieces, so instances
/// can run concurrently. Every `ChunkOut` buffer is laid out model-major,
/// which is what makes an instance's region one contiguous slice; the
/// state vectors are already `[mi]`-indexed.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_instances<'a>(
    spec: &'a Spec,
    models: impl Iterator<Item = ModelRef<'a>>,
    mut rings: impl Iterator<Item = Option<SpreadRef<'a>>>,
    decays: &[Decay],
    check_cfgs: &'a [CheckCfg],
    diag: Diagnostics<'a>,
    out: &'a mut ChunkOut,
    n_rows: usize,
    rows_before: u64,
) -> Instances<'a> {
    let n = decays.len();
    let block = out.n_slots * n_rows;
    let n_targets = if n == 0 || n_rows == 0 || out.lam_selected.is_empty() {
        0
    } else {
        out.lam_selected.len() / (n * n_rows)
    };

    let mut drift = diag.drift.iter_mut();
    let mut resid_q = diag.resid_q.iter_mut();
    let mut autocorr = diag.autocorr.iter_mut();
    let mut metrics = diag.metrics.iter_mut();
    let mut conformal = diag.conformal.iter_mut();
    let mut checks = diag.checks.iter_mut();
    let mut o_pred = out.pred.chunks_mut(block.max(1));
    let mut o_resid = out.resid.chunks_mut(block.max(1));
    let mut o_sigma = out.sigma.chunks_mut(block.max(1));
    let mut o_resid_z = out.zscore.chunks_mut(block.max(1));
    let mut o_autocorr = out.autocorr.chunks_mut(block.max(1));
    let mut o_metrics = out.metrics.chunks_mut((3 * block).max(1));
    let mut o_conformal = out.conformal.chunks_mut((3 * block).max(1));
    let mut o_checks = out.checks.chunks_mut((out.n_checks * block).max(1));
    let health_block = if spec.emit_feature_health {
        2 * spec.k() * n_rows
    } else {
        0
    };
    let mut o_health = out.health.chunks_mut(health_block.max(1));
    let mut o_resid_q = out.resid_q.chunks_mut((out.n_levels * block).max(1));
    let mut o_drift = out.drift.chunks_mut(block.max(1));
    let mut o_n_eff = out.n_eff.chunks_mut(n_rows.max(1));
    let mut o_lam = out.lam_selected.chunks_mut((n_targets * n_rows).max(1));
    let mut o_coef = out.coef.iter_mut();
    let mut o_settled = out.settled.chunks_mut(n_rows.max(1));
    let mut o_reason = out.reason.chunks_mut(n_rows.max(1));
    let mut o_inflation = out.inflation.chunks_mut(block.max(1));
    let mut o_support_coef = out.support_coef.iter_mut();
    let mut o_se_coef = out.se_coef.iter_mut();
    let mut o_se_hc0 = out.se_coef_hc0.iter_mut();
    let mut o_se_hac = out.se_coef_hac.iter_mut();

    let n_slots = out.n_slots;
    let mut decays = decays.iter();
    let mut check_cfgs = check_cfgs.iter();
    let mut resid_var = diag.resid_var.iter_mut();
    let mut resid_w = diag.resid_w.iter_mut();
    let mut scratch = diag.scratch.iter_mut();
    let mut score_pred = diag.score_pred.iter_mut();
    let mut decay_time = diag.decay_time.iter_mut();
    let mut restarted = diag.restarted.iter_mut();
    let mut pending_clock = diag.pending_clock.iter_mut();
    let mut notified = diag.notified.iter_mut();

    // Pulled in lockstep: each iterator yields disjoint `&mut`s, so every
    // Instance owns its own piece of everything.
    let mut built = models.map(|model| {
        let decay = *decays.next().expect("one per instance");
        Instance {
            steps: rows_before,
            spec,
            shards: marginal_shards(spec, model.get()),
            model,
            resid_win: rings.next().expect("one per instance"),
            decay,
            check_cfg: check_cfgs.next().expect("one per instance"),
            residuals: !spec.model.predicts_no_target(),
            resid_var: resid_var.next().expect("one per instance"),
            resid_w: resid_w.next().expect("one per instance"),
            drift: drift.next(),
            resid_q: resid_q.next(),
            autocorr: autocorr.next(),
            metrics: metrics.next(),
            conformal: conformal.next(),
            checks: checks.next(),
            scratch: scratch.next().expect("one per instance"),
            score_pred: score_pred.next().expect("one per instance"),
            n_slots,
            n_rows,
            o_pred: o_pred.next().unwrap_or_default(),
            o_resid: o_resid.next().unwrap_or_default(),
            o_sigma: o_sigma.next().unwrap_or_default(),
            o_resid_z: o_resid_z.next().unwrap_or_default(),
            o_autocorr: o_autocorr.next().unwrap_or_default(),
            o_metrics: o_metrics.next().unwrap_or_default(),
            o_conformal: o_conformal.next().unwrap_or_default(),
            o_checks: o_checks.next().unwrap_or_default(),
            o_health: o_health.next().unwrap_or_default(),
            o_resid_q: o_resid_q.next().unwrap_or_default(),
            o_drift: o_drift.next().unwrap_or_default(),
            o_n_eff: o_n_eff.next().unwrap_or_default(),
            o_lam: o_lam.next().unwrap_or_default(),
            o_coef: o_coef.next().expect("one per instance"),
            o_settled: o_settled.next().unwrap_or_default(),
            o_reason: o_reason.next().unwrap_or_default(),
            o_inflation: o_inflation.next().unwrap_or_default(),
            o_support_coef: o_support_coef.next().expect("one per instance"),
            o_se_coef: o_se_coef.next().expect("one per instance"),
            o_se_hc0: o_se_hc0.next().expect("one per instance"),
            o_se_hac: o_se_hac.next().expect("one per instance"),
            decay_time: decay_time.next().expect("one per instance"),
            restarted: restarted.next().expect("one per instance"),
            notified: notified.next().expect("one per instance"),
            refit_over_budget: diag.refit_over_budget,
            ungated_spread: diag.ungated_spread,
            weighted_autocorr: diag.weighted_autocorr,
            ungated_drift: diag.ungated_drift,
            // None without an embargo, which keeps no held rows' clock.
            pending_clock: pending_clock.next(),
        }
    });
    // One instance, the commonest, is held where it is made: a list of one
    // was an allocation of its own for every group of every chunk, five
    // hundred bytes each, and the call that frees it.
    let Some(first) = built.next() else {
        return Instances::Many(Vec::new());
    };
    let Some(second) = built.next() else {
        return Instances::One(first);
    };
    let mut all = Vec::with_capacity(n.max(2));
    all.push(first);
    all.push(second);
    all.extend(built);
    Instances::Many(all)
}

/// A stream's instances for one run: the one it has, or the list of them.
// The size difference is the point: boxing the one would be the allocation
// it is held here to avoid.
#[allow(clippy::large_enum_variant)]
pub(super) enum Instances<'a> {
    One(Instance<'a>),
    Many(Vec<Instance<'a>>),
}

impl<'a> Instances<'a> {
    pub(super) fn as_mut_slice(&mut self) -> &mut [Instance<'a>] {
        match self {
            Self::One(inst) => std::slice::from_mut(inst),
            Self::Many(all) => all,
        }
    }
}

thread_local! {
    /// The plans' buffer of the last run this thread scheduled, empty, for
    /// the next ([`plan_buffer`]).
    static PLANS: std::cell::Cell<Vec<RowPlan>> = const { std::cell::Cell::new(Vec::new()) };
}

/// The most plans a thread keeps room for between runs: a run of a group's
/// few rows is the case that made them (a hundred and twenty-eight bytes
/// each, and a call to allocate them for every group of every chunk); a
/// long run's are freed as before.
const PLANS_KEPT: usize = 4096;

/// An empty list for a run's plans: the one this thread kept, if any.
pub(super) fn plan_buffer() -> Vec<RowPlan> {
    PLANS.take()
}

/// Keep a run's plans' room for the next run on this thread, where it is
/// at most [`PLANS_KEPT`] plans.
pub(super) fn keep_plans(mut plans: Vec<RowPlan>) {
    if plans.capacity() <= PLANS_KEPT {
        plans.clear();
        PLANS.set(plans);
    }
}

/// The slots' lists as one list, slot after slot: what `into_iter().flatten()
/// .collect()` gives, without the list it was collected into when the model
/// has one slot (the one a target's coefficients are), and with its size
/// known where it has several.
pub(super) fn joined(mut slots: Vec<Vec<f64>>) -> Vec<f64> {
    match slots.len() {
        0 => Vec::new(),
        1 => slots.pop().expect("one slot"),
        _ => {
            let mut all = Vec::with_capacity(slots.iter().map(Vec::len).sum());
            for slot in slots {
                all.extend(slot);
            }
            all
        }
    }
}
