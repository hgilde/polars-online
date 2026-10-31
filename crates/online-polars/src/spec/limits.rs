//! Values a spec refuses because they mean nothing, each with the reason:
//! accepted until task 278 (the user's decision of 2026-10-25), and each did
//! something no one could want. Refused here, in the spec layer, rather than
//! in the core's configuration checks, so a state saved with one still
//! loads: a state carries its configuration, and the core checks it again
//! when it is restored.

use super::{ModelKind, Spec};

impl Spec {
    /// The refusals, in the order a spec meets them: `min_weight` first, as
    /// a common parameter, and then the model's own.
    pub(super) fn refuse_values_that_mean_nothing(&self) -> Result<(), String> {
        // A floor at `inf` never opens: every row withheld for good while the
        // model learns from each, said only by a notice a half-life after
        // the stream settles.
        if self
            .min_periods_per_target()
            .iter()
            .any(|v| v.is_infinite())
        {
            return Err(format!(
                "spec {:?}: min_weight must be finite, got inf: a floor no weight reaches \
                 withholds every row for good while the model learns from each; give the \
                 weight a model must have seen before it reports, or leave min_weight out \
                 for the model's default",
                self.name
            ));
        }
        // A refit's buffer is held at its full size from a target's first
        // learned row, so its count sizes an allocation before any row is
        // seen, and `refit_budget = inf` bounds nothing: `2^32 − 1` rows
        // reserved 172 GB at one feature and aborted the process
        // (`online_core::MAX_REFIT_ROWS`).
        let refit_rows = match &self.model {
            ModelKind::Huber {
                huber_refit_rows, ..
            }
            | ModelKind::Sgd {
                huber_refit_rows, ..
            } => Some(("huber_refit_rows", *huber_refit_rows)),
            ModelKind::Quantile {
                quantile_refit_rows,
                ..
            } => Some(("quantile_refit_rows", *quantile_refit_rows)),
            _ => None,
        };
        if let Some((key, Some(rows))) = refit_rows
            && rows > online_core::MAX_REFIT_ROWS
        {
            return Err(format!(
                "spec {:?}: {key} must be at most {} (2^20), got {rows}: the refit's buffer \
                 is held at its full size, (features + 4) * 8 bytes a row, from a target's \
                 first learned row, so the count sizes an allocation before any row is seen",
                self.name,
                online_core::MAX_REFIT_ROWS
            ));
        }
        match &self.model {
            // The dead rule halves the source cluster and gives the freed
            // centre the other half, so with the source at an equal share
            // `weight_sum / k` both start at `weight_sum / (2k)`: the floor
            // itself at 0.5, and each check then re-places one of them on
            // the far rows (task 278, batch 33: over 50 seeds, no cascade at
            // 0.45, 3 at 0.5, 29 at 0.6, all 50 at 0.9). At 1 or above the
            // lightest centre, never above the mean share, is "dead" at
            // every check whatever the clusters. A value that is not finite
            // is the next check's.
            ModelKind::KMeans { dead_frac, .. } => {
                if let Some(v) = dead_frac.filter(|v| v.is_finite() && *v >= 0.5) {
                    return Err(format!(
                        "spec {:?}: dead_frac must be below 0.5, got {v}: a re-placed centre \
                         starts at half its source's weight, so at 0.5 or above it can be born \
                         at the floor that marks a centre dead, dead_frac * weight_sum / \
                         n_clusters, and the next split-merge check re-places it again",
                        self.name
                    ));
                }
            }
            // An empty list is one open bin: no response curve and no cut to
            // report. One feature left without edges beside others that have
            // them is a request that runs (that feature reports its single
            // bin), but when every list is empty the binned view holds
            // nothing at all, which no caller asks for on purpose.
            ModelKind::Marginal { bin_edges, .. } => {
                if bin_edges
                    .as_ref()
                    .is_some_and(|e| !e.is_empty() && e.iter().all(Vec::is_empty))
                {
                    return Err(format!(
                        "spec {:?}: marginal bin_edges gives no feature an edge: each would be \
                         one open bin, with no response curve and no cut to report; give at \
                         least one feature an edge, or leave bin_edges out for no bins",
                        self.name
                    ));
                }
            }
            // The Bartlett estimate weights lag `l` by `1 − l/γ` for `l < γ`,
            // over a span of `window_rows` rows whose last lag is
            // `window_rows − 1`. At `γ = window_rows` every lag the span has
            // is weighted; past it no lag is added, and the weights only
            // flatten toward 1, where the centred moments' lags sum to 0 and
            // the long-run variance with them: a statistic that grows with
            // the bandwidth, not the data. `"window"` takes no bandwidth,
            // which the model's own check says.
            ModelKind::CorrChange {
                kind,
                bandwidth: Some(b),
                window_rows: Some(m),
                ..
            } if kind.as_deref() != Some("window") && b > m => {
                return Err(format!(
                    "spec {:?}: corrchange bandwidth must be at most window_rows ({m}), got {b}: \
                     the Bartlett kernel weights lag l by 1 - l/bandwidth, and a span of \
                     window_rows rows has no lag past window_rows - 1, so a larger bandwidth \
                     adds no lag; it only flattens the weights toward 1, where the centred \
                     moments' lags sum to 0 and the long-run variance falls toward 0 with them",
                    self.name
                ));
            }
            _ => {}
        }
        Ok(())
    }
}
