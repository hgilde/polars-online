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
        match &self.model {
            // The lightest of `k` centres never carries more than the mean
            // share, `weight_sum / k`, so at 1 or above every check finds a
            // centre "dead" whatever the clusters, and re-places it on any
            // far row. A value that is not finite is the next check's.
            ModelKind::KMeans { dead_frac, .. } => {
                if let Some(v) = dead_frac.filter(|v| v.is_finite() && *v >= 1.0) {
                    return Err(format!(
                        "spec {:?}: dead_frac must be below 1, got {v}: the lightest centre never \
                         carries more than the mean share, weight_sum / n_clusters, so a \
                         dead_frac of 1 or more calls a centre dead at every split-merge check \
                         and re-places it on any far row, whatever the clusters",
                        self.name
                    ));
                }
            }
            // An empty list is one open bin: no response curve and no cut to
            // report, so the binned view of that feature is nothing. A learned
            // feature that is constant gets one, but a list a caller gives
            // says what to bin, and saying nothing is a mistake.
            ModelKind::Marginal { bin_edges, .. } => {
                if let Some(j) = bin_edges
                    .as_ref()
                    .and_then(|e| e.iter().position(Vec::is_empty))
                {
                    let feature = self.features.get(j).map_or("", String::as_str);
                    return Err(format!(
                        "spec {:?}: marginal bin_edges gives feature {feature:?} no edges: that is \
                         one open bin, which has no response curve and no cut to report; give \
                         every feature at least one edge",
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
