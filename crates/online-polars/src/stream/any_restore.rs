//! `AnyModel`'s restore from a saved state.

use online_core::{ModelState, State, StateError};

use super::*;

impl AnyModel {
    pub fn restore(s: &State) -> Result<Self, StateError> {
        Self::restore_owned(s.clone())
    }

    /// [`Self::restore`] taking the state: `ewridge` and `ew_cov` move their
    /// model out of it, where the others copy it as `restore` does.
    pub fn restore_owned(s: State) -> Result<Self, StateError> {
        match &s.model {
            ModelState::EwRidge(_) => Ok(AnyModel::EwRidge(Box::new(EwRidge::restore_owned(s)?))),
            ModelState::EwCovModel(_) => {
                Ok(AnyModel::EwCov(Box::new(EwCovModel::restore_owned(s)?)))
            }
            _ => Self::restore_cloned(&s),
        }
    }

    fn restore_cloned(s: &State) -> Result<Self, StateError> {
        match &s.model {
            ModelState::EwRidge(_) => Ok(AnyModel::EwRidge(Box::new(EwRidge::restore(s)?))),
            ModelState::Rls(_) => Ok(AnyModel::Rls(Box::new(Rls::restore(s)?))),
            ModelState::Lasso(_) => Ok(AnyModel::Lasso(Box::new(Lasso::restore(s)?))),
            ModelState::Kalman(_) => Ok(AnyModel::Kalman(Box::new(Kalman::restore(s)?))),
            ModelState::Robust(_) => Ok(AnyModel::Robust(Box::new(Robust::restore(s)?))),
            ModelState::Ftrl(_) => Ok(AnyModel::Ftrl(Box::new(Ftrl::restore(s)?))),
            ModelState::EwCovModel(_) => Ok(AnyModel::EwCov(Box::new(EwCovModel::restore(s)?))),
            ModelState::Sgd(_) => Ok(AnyModel::Sgd(Box::new(Sgd::restore(s)?))),
            ModelState::Pa(_) => Ok(AnyModel::Pa(Box::new(Pa::restore(s)?))),
            ModelState::Holt(_) => Ok(AnyModel::Holt(Box::new(Holt::restore(s)?))),
            ModelState::KMeans(_) => Ok(AnyModel::KMeans(Box::new(KMeans::restore(s)?))),
            ModelState::Micro(_) => Ok(AnyModel::Micro(Box::new(Micro::restore(s)?))),
            ModelState::EwClass(_) => Ok(AnyModel::EwClass(Box::new(EwClass::restore(s)?))),
            ModelState::SeqTest(_) => Ok(AnyModel::SeqTest(Box::new(SeqTest::restore(s)?))),
            ModelState::Marginal(_) => Ok(AnyModel::Marginal(Box::new(Marginal::restore(s)?))),
            ModelState::Deco(_) => Ok(AnyModel::Deco(Box::new(Deco::restore(s)?))),
            ModelState::Rcov(_) => Ok(AnyModel::Rcov(Box::new(Rcov::restore(s)?))),
            ModelState::Hmm(_) => Ok(AnyModel::Hmm(Box::new(Hmm::restore(s)?))),
            ModelState::CorrChange(_) => {
                Ok(AnyModel::CorrChange(Box::new(CorrChange::restore(s)?)))
            }
            ModelState::Bocpd(_) => Ok(AnyModel::Bocpd(Box::new(Bocpd::restore(s)?))),
            ModelState::Audit(_) => Ok(AnyModel::Audit(Box::new(online_core::Audit::restore(s)?))),
            // A kind only online-core builds, such as `ew_cov`'s bare running
            // sums: its name is the code's, so the message does not print it.
            _ => Err(StateError::Invalid(format!(
                "it is damaged, since it holds a model of a kind no spec builds. {}",
                online_core::DAMAGED
            ))),
        }
    }
}
