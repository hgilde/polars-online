//! XGBoost's and LightGBM's training inputs, built from a stream of frames:
//! the component behind `po.gbdt` (docs/GBDT-ADAPTER.md §6, docs/PLAN.md
//! task 242).
//!
//! Both libraries train in two stages. Binning places each feature's bins and
//! quantizes every value into one, and boosting grows trees from per-bin sums
//! of gradients without reading a raw value again. This module does the
//! first stage from frames that arrive in order and may not fit in memory,
//! and hands the binned object back as a native handle, which the Python
//! layer wraps in the library's own class (`xgboost.QuantileDMatrix`,
//! `lightgbm.Dataset`), so that `xgb.train` and `lgb.train` run unchanged.
//!
//! **Nothing is linked.** Each library is loaded at run time from the shared
//! library its Python package ships ([`XgbLib::load`], [`LgbLib::load`]),
//! through `libloading`, by the path the Python package loaded it from: the
//! operating system hands back the image already in the process, so a handle
//! made here is one the Python package can free (CLAUDE.md, rule 12).
//!
//! - **XGBoost** ([`xgboost_matrix`]): each frame's columns go to
//!   `XGQuantileDMatrixCreateFromCallback` as array interfaces on their own
//!   Arrow buffers, the validity bitmap as the mask, with no copy. XGBoost
//!   reads the stream four times: it counts, sketches each feature's
//!   quantiles over every row, indexes and builds its column matrix. With
//!   cuts from a sample ([`Sampler`]) handed over as a `ref`, the sketch pass
//!   goes, and with it half the time before the first round.
//! - **LightGBM** ([`lightgbm_dataset`]): LightGBM's own sample rows
//!   (`LGBM_SampleIndices`) gathered in a first pass place its bins
//!   (`LGBM_DatasetCreateFromSampledColumn`), and a second pass pushes every
//!   frame as float32 rows built in parallel (`LGBM_DatasetPushRows`); or,
//!   when the data fits, the whole stream goes in as Arrow
//!   (`LGBM_DatasetCreateFromArrowStream`), which LightGBM pushes column by
//!   column.
//!
//! The frames carry the features in order, then the label, then the weight
//! when there is one ([`Layout`]); the Python layer selects and casts them
//! (a `pl.Enum` to its codes, a Boolean to `UInt8`, the label and weight to
//! `Float64`), so every feature arrives as a primitive numeric column.

mod frames;
mod ipc;
mod lgb;
mod libs;
mod sample;
mod xgb;

use std::fmt;

pub use frames::{Frame, Frames, PlanFrames, VecFrames};
pub use ipc::IpcFrames;
pub use lgb::{LgbBuilt, LgbOptions, lightgbm_dataset};
pub use libs::{LgbLib, XgbLib};
pub use sample::{Planted, Sampler};
pub use xgb::{XgbBuilt, XgbOptions, xgboost_matrix};

/// Why a build could not be made. `Source` keeps an error the frames' source
/// raised as it is, so that a binding can raise the caller's own exception
/// again rather than a message about it.
#[derive(Debug)]
pub enum GbdtError {
    /// A refusal or a library's error, as a message.
    Msg(String),
    /// The source of the frames failed.
    Source(Box<dyn std::error::Error + Send + Sync>),
}

impl GbdtError {
    pub(crate) fn msg(m: impl Into<String>) -> Self {
        GbdtError::Msg(m.into())
    }
}

impl fmt::Display for GbdtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GbdtError::Msg(m) => f.write_str(m),
            GbdtError::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GbdtError {}

impl From<polars::prelude::PolarsError> for GbdtError {
    fn from(e: polars::prelude::PolarsError) -> Self {
        GbdtError::Source(Box::new(e))
    }
}

/// What a feature column holds, as the libraries see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A floating-point column.
    Float,
    /// An integer column.
    Int,
    /// A Boolean column, arriving as 0 and 1.
    Bool,
    /// A categorical column, arriving as codes `0..levels` (a `pl.Enum`'s
    /// physical values).
    Category { levels: u32 },
}

impl Kind {
    /// XGBoost's feature type: what its Arrow route records for the column.
    pub(crate) fn xgb_type(self) -> &'static str {
        match self {
            Kind::Float => "float",
            Kind::Int => "int",
            Kind::Bool => "i",
            Kind::Category { .. } => "c",
        }
    }
}

/// The columns of every frame, by position: the features in this order,
/// then the label, then the weight when `weight` is set.
#[derive(Debug, Clone)]
pub struct Layout {
    pub names: Vec<String>,
    pub kinds: Vec<Kind>,
    pub label: String,
    pub weight: Option<String>,
}

impl Layout {
    /// Columns a frame carries.
    pub(crate) fn width(&self) -> usize {
        self.names.len() + 1 + usize::from(self.weight.is_some())
    }

    pub(crate) fn check(&self) -> Result<(), GbdtError> {
        if self.names.is_empty() {
            return Err(GbdtError::msg("no feature columns"));
        }
        if self.names.len() != self.kinds.len() {
            return Err(GbdtError::msg(format!(
                "{} feature names for {} kinds",
                self.names.len(),
                self.kinds.len()
            )));
        }
        Ok(())
    }
}

/// A rayon pool of `threads` threads (all the machine's when 0), for the
/// conversions a build runs beside the library.
pub fn conversion_pool(threads: usize) -> Result<rayon::ThreadPool, GbdtError> {
    let mut b = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("gbdt-{i}"));
    if threads > 0 {
        b = b.num_threads(threads);
    }
    b.build()
        .map_err(|e| GbdtError::msg(format!("a thread pool for the conversions: {e}")))
}

#[cfg(test)]
mod tests;
