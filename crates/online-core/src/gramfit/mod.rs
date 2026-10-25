//! Fits read off an exported Gram, offline: what `polars_online.gram` runs
//! in Rust (docs/PLAN.md tasks 226 and 227). Nothing here is a model or
//! touches a state; it takes the arrays `ModelBank.gram()` hands back.
//!
//! - [`Design`], [`Response`], [`Correlation`]: a Gram's regression of each
//!   target on a set of its columns, centred at that target's means where
//!   the Gram has an intercept, and its correlation form;
//! - [`lars_lasso`]: the lasso path by least angle regression, stopped early;
//! - [`cd_path`]: the elastic-net path by the `lasso` model's coordinate
//!   descent, over a grid of penalties, by its stopping rule
//!   ([`sweep_tol`]);
//! - [`merge`]: Grams of disjoint row sets pooled, on a subset of columns;
//!   [`merge_within`]: pooled within the parts, each about its own means
//!   (block fixed effects), with a sign per part on the target
//!   ([`within_cross`]);
//! - [`ridge_fits`]: a ridge fit of each target with its standard errors.

mod cd;
mod colstats;
#[cfg(test)]
mod colstats_tests;
mod lars;
#[cfg(test)]
mod lars_tie_tests;
mod merge;
#[cfg(test)]
mod merge_tests;
mod ridge;
mod system;
#[cfg(test)]
mod tests;
mod trisolve;
#[cfg(test)]
mod trisolve_tests;
#[cfg(test)]
mod within_tests;

pub(crate) use cd::settled;
pub use cd::{CdPath, cd_path, sweep_tol, target_spread};
pub use lars::{
    CorrRows, DenseRows, LarsLimits, LarsPath, LarsStop, lars_lasso, lars_lasso_rows,
    lars_lasso_weighted,
};
pub use merge::{OwnedGram, merge, merge_within, within_cross};
pub use ridge::{RidgeFit, ridge_fits};
pub use system::{Comoments, Correlation, Design, GramArrays, GramRows, Response, Scaling};
