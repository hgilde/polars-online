//! Fits read off an exported Gram, offline: what `polars_online.gram` runs
//! in Rust (docs/PLAN.md tasks 226 and 227). Nothing here is a model or
//! touches a state; it takes the arrays `ModelBank.gram()` hands back.
//!
//! - [`Design`], [`Response`], [`Correlation`]: a Gram's regression of each
//!   target on a set of its columns, centred at that target's means where
//!   the Gram has an intercept, and its correlation form;
//! - [`lars_lasso`]: the lasso path by least angle regression, stopped early;
//! - [`cd_path`]: the elastic-net path by the `lasso` model's coordinate
//!   descent, over a grid of penalties.

mod cd;
mod lars;
mod system;
#[cfg(test)]
mod tests;

pub use cd::cd_path;
pub use lars::{
    CorrRows, DenseRows, LarsLimits, LarsPath, LarsStop, lars_lasso, lars_lasso_rows,
    lars_lasso_weighted,
};
pub use system::{Correlation, Design, GramArrays, GramRows, Response, Scaling};
