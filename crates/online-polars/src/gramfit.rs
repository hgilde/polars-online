//! The offline Gram fits of [`online_core::gramfit`], many at once on the
//! bank's pool (docs/PLAN.md task 226): what `polars_online.gram`'s
//! `lars_path`, `lars_paths` and `lasso_path` run.
//!
//! A path reads only its active columns' rows of the correlation matrix,
//! formed from the Gram's co-moments as each column enters; a Gram's
//! scaling is formed once and its targets' paths run beside each other on
//! it, and the Grams run beside each other too. Each path is the same
//! arithmetic on one thread whatever the pool's size, so the numbers do
//! not depend on it.

use online_core::gramfit::{
    Correlation, Design, GramArrays, GramRows, LarsLimits, LarsStop, Response, cd_path,
    lars_lasso_weighted,
};
use polars::prelude::*;
use rayon::prelude::*;

/// The columns a fit regresses on, by position in the Gram's `columns`,
/// and the intercept's position, if the Gram has one.
#[derive(Clone, Debug)]
pub struct FitColumns {
    pub slots: Vec<usize>,
    pub icept: Option<usize>,
}

/// One lasso path, in the Gram's own terms.
#[derive(Clone, Debug, PartialEq)]
pub struct GramPath {
    /// `λ` at each knot, falling.
    pub penalties: Vec<f64>,
    /// The coefficients at each knot over the Gram's `k` columns, in the
    /// features' original units with the intercept recovered, `n × k`.
    pub coefs: Vec<f64>,
    /// The active columns at each knot, by position in the Gram's
    /// `columns`, in the order they entered.
    pub active: Vec<Vec<usize>>,
    pub stop: LarsStop,
}

/// Each target's lasso path on each Gram, `targets[g]` naming Gram `g`'s.
/// `weights` scales the penalty per slot (empty: 1 for every one).
///
/// # Errors
///
/// The pool's, when `POLARS_ONLINE_MAX_THREADS` is set wrong, and a
/// weight that is not finite and above 0 on a column that varies.
pub fn lars_paths(
    grams: &[GramArrays<'_>],
    targets: &[Vec<usize>],
    cols: &FitColumns,
    weights: &[f64],
    limits: LarsLimits,
) -> PolarsResult<Vec<Vec<GramPath>>> {
    let n = cols.slots.len();
    let weights: Vec<f64> = if weights.is_empty() {
        vec![1.0; n]
    } else {
        weights.to_vec()
    };
    if weights.len() != n {
        polars_bail!(ComputeError: "{} penalty weights for {} features", weights.len(), n);
    }
    crate::pool()?.install(|| {
        grams
            .par_iter()
            .zip(targets)
            .map(|(g, ts)| {
                let rows = GramRows::new(*g, &cols.slots, cols.icept);
                ts.par_iter()
                    .map(|&t| path_of(g, t, &rows, cols, &weights, limits))
                    .collect::<PolarsResult<Vec<GramPath>>>()
            })
            .collect()
    })
}

/// Target `t`'s path on `g`, whose design's correlation form `rows` reads.
fn path_of(
    g: &GramArrays<'_>,
    t: usize,
    rows: &GramRows<'_>,
    cols: &FitColumns,
    weights: &[f64],
    limits: LarsLimits,
) -> PolarsResult<GramPath> {
    let resp = Response::of(g, t, &cols.slots, cols.icept);
    let corr = &rows.scaling;
    let d = corr.d(&resp);
    let path = lars_lasso_weighted(rows, &d, &corr.live, weights, limits)
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let n = cols.slots.len();
    let mut coefs = Vec::with_capacity(path.len() * g.k);
    for i in 0..path.len() {
        let b = &path.coefs[i * n..(i + 1) * n];
        coefs.extend(resp.coef(corr, b, g.k, &cols.slots, cols.icept));
    }
    Ok(GramPath {
        penalties: path.penalties,
        coefs,
        active: path
            .active
            .into_iter()
            .map(|a| a.into_iter().map(|p| cols.slots[p]).collect())
            .collect(),
        stop: path.stop,
    })
}

/// Target `t`'s elastic-net path on `g` over `penalties` by the `lasso`
/// model's coordinate descent, `penalties.len() × k` over the Gram's
/// columns in original units with the intercept recovered. `weights`
/// scales the penalty per slot.
#[allow(clippy::too_many_arguments)]
pub fn cd_path_of(
    g: &GramArrays<'_>,
    t: usize,
    cols: &FitColumns,
    penalties: &[f64],
    l1_ratio: f64,
    weights: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let corr = Correlation::of(&Design::of(g, &cols.slots, cols.icept));
    let resp = Response::of(g, t, &cols.slots, cols.icept);
    let d = corr.scaling.d(&resp);
    let path = cd_path(
        &corr.r,
        &d,
        &corr.scaling.live,
        penalties,
        l1_ratio,
        weights,
        max_iter,
        tol,
    );
    let n = cols.slots.len();
    let mut out = Vec::with_capacity(penalties.len() * g.k);
    for i in 0..penalties.len() {
        let b = &path[i * n..(i + 1) * n];
        out.extend(resp.coef(&corr.scaling, b, g.k, &cols.slots, cols.icept));
    }
    out
}
