//! What each model reads of the readiness statistics
//! (docs/WARMUP-AND-CONVERGENCE.md; docs/PLAN.md task 116): which models the
//! noise gate, its per-row field, the data shares and the coefficients'
//! standard errors exist for, and why the others have none. The settings
//! that read them are `Spec`'s fields; their refusals are `Spec::validate`'s.

use super::{ModelKind, Spec};

impl Spec {
    /// Whether this spec's model reads the noise gate's statistic
    /// ([`online_core::OnlineModel::error_inflation_into`]): the fits
    /// linear in the past targets whose estimation variance is known --
    /// `ewridge`, `rls` and `kalman` -- and `lasso`, from its active count
    /// (docs/PLAN.md task 116). The others are left to `min_weight`, and
    /// `max_error_inflation` is refused for them, with the reason
    /// [`Self::no_noise_statistic`] gives.
    pub fn has_error_inflation(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. }
                | ModelKind::Rls { .. }
                | ModelKind::Kalman { .. }
                | ModelKind::Lasso { .. }
        )
    }

    /// Whether this spec's model reads the statistic for one row's
    /// features ([`online_core::OnlineModel::row_error_inflation_into`]),
    /// the field `emit_error_inflation` writes: `ewridge` and `rls` from
    /// the factor their fit keeps, `kalman` from its `P`. Not `lasso`, which
    /// keeps no factor.
    pub fn has_row_error_inflation(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. } | ModelKind::Rls { .. } | ModelKind::Kalman { .. }
        )
    }

    /// Why this spec's model has no noise statistic, for the refusal of a
    /// setting that reads one (docs/PLAN.md task 116).
    pub(super) fn no_noise_statistic(&self) -> &'static str {
        match self.model {
            ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                "a gradient fit keeps no second moment of the features to read an estimation \
                 variance from"
            }
            ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                "a robust fit is not linear in the targets, so the ridge family's variance does \
                 not hold for it"
            }
            ModelKind::EwCov { .. }
            | ModelKind::Marginal { .. }
            | ModelKind::Deco { .. }
            | ModelKind::Rcov { .. }
            | ModelKind::CorrChange { .. } => {
                "its statistics are EW moments, and 1 / n_kish is a mean's variance, not a \
                 standard deviation's or a correlation's; its min_weight of k + 1 already holds \
                 the gate's point"
            }
            _ => "it keeps no estimation variance of a prediction",
        }
    }

    /// Whether this spec's model reports its coefficients' sampling
    /// variances ([`online_core::OnlineModel::coef_variance`]), the field
    /// `emit_se_coef` writes: `ewridge`, `rls` and `kalman` (docs/PLAN.md
    /// task 116, F).
    pub fn has_se_coef(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. } | ModelKind::Rls { .. } | ModelKind::Kalman { .. }
        )
    }

    /// Whether this spec's model reports `support_coef`
    /// ([`online_core::OnlineModel::support_coef`]): the fits whose solve
    /// inverts a Gram with a mean-form ridge on its diagonal, `ewridge`,
    /// `huber` and `quantile` (docs/WARMUP-AND-CONVERGENCE.md §2.2). Not
    /// `lasso`, an L1 penalty having no shrinkage matrix -- every active
    /// coefficient would read 1 at the default -- nor `ew_cov`, which has no
    /// coefficients.
    pub fn has_support_coef(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. } | ModelKind::Huber { .. } | ModelKind::Quantile { .. }
        )
    }
}
