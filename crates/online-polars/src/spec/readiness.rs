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

    /// Why this spec's model does not track how uncertain each prediction
    /// is, for the refusal of `max_error_inflation` (docs/PLAN.md task 116).
    pub(super) fn no_noise_statistic(&self) -> &'static str {
        match self.model {
            ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                "a gradient fit keeps no second moments of the features to read that \
                 uncertainty from"
            }
            ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                "a robust fit is not linear in its targets, so a least-squares fit's \
                 uncertainty does not hold for it"
            }
            ModelKind::EwCov { .. }
            | ModelKind::Marginal { .. }
            | ModelKind::Deco { .. }
            | ModelKind::Rcov { .. }
            | ModelKind::CorrChange { .. } => {
                "its statistics are weighted means and covariances of the columns, not \
                 predictions of a target"
            }
            _ => "it keeps no measure of a prediction's uncertainty",
        }
    }

    /// Why this spec's model cannot read one row's leverage, for the
    /// refusals of `emit_influence` and `emit_error_inflation`.
    pub(super) fn no_leverage(&self) -> &'static str {
        match self.model {
            ModelKind::Lasso { .. } => {
                "it keeps no inverse of its Gram to read one row's leverage from"
            }
            ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                "a robust fit is not linear in its targets, so a least-squares leverage does \
                 not hold for it"
            }
            ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                "a gradient fit keeps no second moments of the features to read one row's \
                 leverage from"
            }
            _ => "it fits no coefficients of features to read one row's leverage from",
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
