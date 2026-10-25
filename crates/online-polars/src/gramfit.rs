//! The offline Gram fits of [`online_core::gramfit`], many at once on the
//! bank's pool or the one `threads` names (docs/PLAN.md tasks 226, 227
//! and 253): what `polars_online.gram`'s
//! `lars_path`, `lars_paths`, `lasso_path` and `solve_subsets` run.
//!
//! A path reads only its active columns' rows of the correlation matrix,
//! formed from the Gram's co-moments as each column enters; a Gram's
//! scaling is formed once and its targets' paths run beside each other on
//! it, and the Grams run beside each other too. Each path is the same
//! arithmetic on one thread whatever the pool's size, so the numbers do
//! not depend on it.

use online_core::gramfit::{
    Correlation, Design, GramArrays, GramRows, LarsLimits, LarsStop, Response, RidgeFit, cd_path,
    lars_lasso_weighted, merge, merge_within, ridge_fits, target_spread, within_cross,
};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

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
    /// `columns`, in the order they entered, those entering at one knot (a
    /// tie) in `slots` order.
    pub active: Vec<Vec<usize>>,
    pub stop: LarsStop,
}

/// Each target's lasso path on each Gram, `targets[g]` naming Gram `g`'s.
/// `weights` scales the penalty per slot (empty: 1 for every one).
///
/// # Errors
///
/// The pool's, when `POLARS_ONLINE_MAX_THREADS` is set wrong and
/// `threads` is the process default, and a weight that is not finite and above 0 on a column that varies.
pub fn lars_paths(
    grams: &[GramArrays<'_>],
    targets: &[Vec<usize>],
    cols: &FitColumns,
    weights: &[f64],
    limits: LarsLimits,
    threads: &crate::Threads,
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
    threads.install(|| {
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
/// columns in original units with the intercept recovered, and per penalty
/// whether its descent ran out of sweeps. `weights` scales the penalty per
/// slot. The stopping rule reads the target's spread as the model's does
/// (`online_core::gramfit::sweep_tol`): its standard deviation with an
/// intercept, its root mean square through the origin, as `ridge_fits`
/// reads the target's second moment.
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
) -> (Vec<f64>, Vec<bool>) {
    let corr = Correlation::of(&Design::of(g, &cols.slots, cols.icept));
    let resp = Response::of(g, t, &cols.slots, cols.icept);
    let d = corr.scaling.d(&resp);
    // NaN where the Gram carries no target moments: `target_spread` reads
    // it as 0, which leaves `sweep_tol`'s floor.
    let at = |v: &[f64]| v.get(t).copied().unwrap_or(f64::NAN);
    let spread = target_spread(at(g.target_vars), at(g.target_means), cols.icept.is_some());
    let path = cd_path(
        &corr.r,
        &d,
        &corr.scaling.live,
        penalties,
        l1_ratio,
        weights,
        max_iter,
        tol,
        spread,
    );
    let n = cols.slots.len();
    let mut out = Vec::with_capacity(penalties.len() * g.k);
    for i in 0..penalties.len() {
        let b = &path.coefs[i * n..(i + 1) * n];
        out.extend(resp.coef(&corr.scaling, b, g.k, &cols.slots, cols.icept));
    }
    (out, path.unconverged)
}

/// What [`fit_subsets`] fits on each merged Gram.
#[derive(Clone, Debug)]
pub enum SubsetFit {
    /// A ridge of this penalty, standardized or not, with its statistics.
    Ridge { ridge: f64, standardize: bool },
    /// The lasso path, weighted per slot (empty: 1 for every one), stopped
    /// at `limits`.
    Path {
        weights: Vec<f64>,
        limits: LarsLimits,
    },
}

/// How [`fit_subsets`] pools each subset's Grams.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Pooling {
    /// Into the Gram of their union (`online_core::gramfit::merge`).
    #[default]
    Union,
    /// Within each part, about its own means (`merge_within`): a fit with
    /// an intercept per part, which recovers none (docs/PLAN.md task 240).
    /// Each of `signs` is a draw, a sign per Gram (in `grams`' order) on its
    /// targets' centred cross-moments (`within_cross`, task 241), and every
    /// subset is fitted once per draw; no draws is one, every sign +1.
    Within { signs: Vec<Vec<f64>> },
}

/// One target's fit on one subset's merged Gram.
#[derive(Clone, Debug, PartialEq)]
pub enum SubsetOut {
    Ridge(RidgeFit),
    Path(GramPath),
}

/// Each subset of `grams` merged as `pooling` says (on the columns `cols`
/// reads) and fitted for each of `targets`, the subsets beside each other
/// on the bank's pool: one list per draw of signs (one for a union, or
/// within without signs), in it one list per subset, one fit per target. A
/// subset is a list of positions in `grams`, which share their columns and
/// targets. It is [`fit_jobs`] with one column set and a job per draw,
/// subset and target, so its numbers are those jobs' to the bit.
///
/// # Errors
///
/// As [`fit_jobs`].
pub fn fit_subsets(
    grams: &[GramArrays<'_>],
    subsets: &[Vec<usize>],
    targets: &[usize],
    cols: &FitColumns,
    fit: &SubsetFit,
    pooling: &Pooling,
    threads: &crate::Threads,
) -> PolarsResult<Vec<Vec<Vec<SubsetOut>>>> {
    let (job_fit, weights) = match fit {
        SubsetFit::Ridge { ridge, standardize } => (
            JobFit::Ridge {
                ridge: *ridge,
                standardize: *standardize,
            },
            Vec::new(),
        ),
        SubsetFit::Path { weights, limits } => (JobFit::Path { limits: *limits }, weights.clone()),
    };
    let sets = [ColumnSet {
        slots: cols.slots.clone(),
        weights,
    }];
    let n_draws = draws_of(pooling).len();
    let jobs: Vec<SubsetJob> = (0..n_draws)
        .flat_map(|draw| {
            (0..subsets.len()).flat_map(move |subset| {
                targets.iter().map(move |&target| SubsetJob {
                    subset,
                    target,
                    draw,
                    set: 0,
                })
            })
        })
        .collect();
    let mut fits = fit_jobs(
        grams, subsets, &sets, &jobs, cols.icept, &job_fit, pooling, threads,
    )?
    .into_iter();
    Ok((0..n_draws)
        .map(|_| {
            (0..subsets.len())
                .map(|_| fits.by_ref().take(targets.len()).collect())
                .collect()
        })
        .collect())
}

/// What [`fit_jobs`] fits for each job: [`SubsetFit`] without the
/// penalty weights, which each [`ColumnSet`] carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JobFit {
    /// A ridge of this penalty, standardized or not, with its statistics.
    Ridge { ridge: f64, standardize: bool },
    /// The lasso path, stopped at `limits`.
    Path { limits: LarsLimits },
}

/// The columns a job regresses on, by position in the Gram's `columns`,
/// the intercept never among them, and its path's penalty weight per slot
/// (empty: 1 for every one; a ridge reads none).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColumnSet {
    pub slots: Vec<usize>,
    pub weights: Vec<f64>,
}

/// One fit of [`fit_jobs`]: target `target` of subset `subset`'s merged
/// Gram, under draw `draw` of the signs (0 for a union, or within without
/// signs), on the columns of set `set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubsetJob {
    pub subset: usize,
    pub target: usize,
    pub draw: usize,
    pub set: usize,
}

/// The draws `pooling` runs each subset under: its signs, or one draw of
/// none.
fn draws_of(pooling: &Pooling) -> Vec<&[f64]> {
    match pooling {
        Pooling::Within { signs } if !signs.is_empty() => signs.iter().map(Vec::as_slice).collect(),
        _ => vec![&[]],
    }
}

/// Each job's fit, in `jobs`' order: its subset of `grams` merged as
/// `pooling` says, its draw's signs on the merge's cross-moments, its
/// target fitted on its column set. Every job reads the columns of its own
/// set, so a screen per subset, per target and per draw runs in one call.
///
/// A subset is merged once, on the intercept (at `icept` among the Grams'
/// columns, or none) and every column its jobs read, and dropped when its
/// jobs are done; a draw of signs re-pools only its targets'
/// cross-moments, once per subset and draw; jobs of one subset, draw and
/// set share one design (a ridge's factorization, a path's scaling). The
/// subsets run beside each other on the pool, and their draws, sets and
/// targets inside them. The merge is entry for entry, so a job's numbers
/// are those of its set's columns merged alone, to the bit, at any pool
/// size. The Grams themselves are only read.
///
/// # Errors
///
/// `ComputeError` for a job naming a subset, set, draw or target out of
/// range, a subset naming no Gram or one out of range, a set naming a
/// column out of range or the intercept, or its weights of another length
/// than its slots; the pool's, when `POLARS_ONLINE_MAX_THREADS` is set
/// wrong and `threads` is the process default; and a path's weight that is
/// not finite and above 0 on a column that varies.
#[allow(clippy::too_many_arguments)]
pub fn fit_jobs(
    grams: &[GramArrays<'_>],
    subsets: &[Vec<usize>],
    sets: &[ColumnSet],
    jobs: &[SubsetJob],
    icept: Option<usize>,
    fit: &JobFit,
    pooling: &Pooling,
    threads: &crate::Threads,
) -> PolarsResult<Vec<SubsetOut>> {
    let draws = draws_of(pooling);
    check_jobs(grams, subsets, sets, jobs, icept, &draws)?;
    let mut by_subset: Vec<Vec<usize>> = vec![Vec::new(); subsets.len()];
    for (j, job) in jobs.iter().enumerate() {
        by_subset[job.subset].push(j);
    }
    let ctx = JobContext {
        grams,
        sets,
        jobs,
        icept,
        fit,
        pooling,
        draws: &draws,
    };
    let fitted: Vec<Vec<(usize, SubsetOut)>> = threads.install(|| {
        by_subset
            .par_iter()
            .zip(subsets)
            .filter(|(js, _)| !js.is_empty())
            .map(|(js, subset)| ctx.subset(subset, js))
            .collect::<PolarsResult<_>>()
    })?;
    let mut out: Vec<Option<SubsetOut>> = vec![None; jobs.len()];
    for (j, o) in fitted.into_iter().flatten() {
        out[j] = Some(o);
    }
    // Every job is in exactly one subset's list, so every slot is filled.
    Ok(out.into_iter().flatten().collect())
}

/// Why a column named twice in one fit is refused: what follows "names
/// column `c` twice; " wherever a fit's columns are checked.
pub const TWICE: &str = "a column is one regressor, and a fit given it twice solves a singular \
    system and writes one of its two coefficients over the other: name it once";

/// The first column `slots` names a second time, if any (task 269, D-3).
/// A selection may repeat a column; a fit may not ([`TWICE`]).
pub fn repeated_slot(slots: &[usize]) -> Option<usize> {
    let mut seen = std::collections::HashSet::with_capacity(slots.len());
    slots.iter().copied().find(|&c| !seen.insert(c))
}

/// Every index [`fit_jobs`] reads, in range, and no column set naming a
/// column twice.
fn check_jobs(
    grams: &[GramArrays<'_>],
    subsets: &[Vec<usize>],
    sets: &[ColumnSet],
    jobs: &[SubsetJob],
    icept: Option<usize>,
    draws: &[&[f64]],
) -> PolarsResult<()> {
    let (k, m) = grams.first().map_or((0, 0), |g| (g.k, g.targets()));
    if let Some(g) = grams.iter().find(|g| g.k != k || g.targets() != m) {
        polars_bail!(ComputeError: "every Gram must have the same columns and targets: \
            {k} and {m}, then {} and {}", g.k, g.targets());
    }
    for (s, subset) in subsets.iter().enumerate() {
        if subset.is_empty() {
            polars_bail!(ComputeError: "subset {s} names no Gram");
        }
        if let Some(&i) = subset.iter().find(|&&i| i >= grams.len()) {
            polars_bail!(ComputeError: "subset {s} names Gram {i} of {}", grams.len());
        }
    }
    if let Some(d) = draws
        .iter()
        .find(|d| !d.is_empty() && d.len() != grams.len())
    {
        polars_bail!(ComputeError: "a draw has {} signs for {} Grams", d.len(), grams.len());
    }
    if let Some(&c) = icept.iter().find(|&&c| c >= k) {
        polars_bail!(ComputeError: "intercept column {c} out of range for {k} columns");
    }
    for (i, set) in sets.iter().enumerate() {
        if let Some(&c) = set.slots.iter().find(|&&c| c >= k || Some(c) == icept) {
            polars_bail!(ComputeError: "column set {i} names column {c}, out of range for {k} \
                columns or the intercept");
        }
        if let Some(c) = repeated_slot(&set.slots) {
            polars_bail!(ComputeError: "column set {i} names column {c} twice; {TWICE}");
        }
        if !set.weights.is_empty() && set.weights.len() != set.slots.len() {
            polars_bail!(ComputeError: "column set {i}: {} penalty weights for {} features",
                set.weights.len(), set.slots.len());
        }
    }
    for (j, job) in jobs.iter().enumerate() {
        let bad = [
            ("subset", job.subset, subsets.len()),
            ("column set", job.set, sets.len()),
            ("draw", job.draw, draws.len()),
            ("target", job.target, m),
        ]
        .into_iter()
        .find(|&(_, at, n)| at >= n);
        if let Some((what, at, n)) = bad {
            polars_bail!(ComputeError: "job {j} names {what} {at} of {n}");
        }
    }
    Ok(())
}

/// One draw's jobs of a subset: per column set, its jobs' positions.
type BySet = Vec<(usize, Vec<usize>)>;

/// What every subset's jobs read in [`fit_jobs`].
struct JobContext<'a, 'g> {
    grams: &'a [GramArrays<'g>],
    sets: &'a [ColumnSet],
    jobs: &'a [SubsetJob],
    icept: Option<usize>,
    fit: &'a JobFit,
    pooling: &'a Pooling,
    draws: &'a [&'a [f64]],
}

impl JobContext<'_, '_> {
    /// The jobs `js` of one subset, each with its position in `jobs`.
    fn subset(&self, subset: &[usize], js: &[usize]) -> PolarsResult<Vec<(usize, SubsetOut)>> {
        let parts: Vec<GramArrays<'_>> = subset.iter().map(|&i| self.grams[i]).collect();
        let k = parts[0].k;
        // The merged Gram's columns: the intercept first, then every column
        // a job reads, in the Grams' order; `at` places each in it.
        let mut used = vec![false; k];
        for &j in js {
            for &c in &self.sets[self.jobs[j].set].slots {
                used[c] = true;
            }
        }
        let all: Vec<usize> = self
            .icept
            .into_iter()
            .chain((0..k).filter(|&c| used[c]))
            .collect();
        let mut at = vec![0; k];
        for (r, &c) in all.iter().enumerate() {
            at[c] = r;
        }
        // The jobs by draw, then by column set, each list in `jobs`' order.
        let mut groups: BTreeMap<usize, BTreeMap<usize, Vec<usize>>> = BTreeMap::new();
        for &j in js {
            let job = self.jobs[j];
            groups
                .entry(job.draw)
                .or_default()
                .entry(job.set)
                .or_default()
                .push(j);
        }
        let groups: Vec<(usize, BySet)> = groups
            .into_iter()
            .map(|(d, by_set)| (d, by_set.into_iter().collect()))
            .collect();
        let merged = match self.pooling {
            Pooling::Union => merge(&parts, &all, self.icept),
            Pooling::Within { .. } => merge_within(&parts, &all, self.icept),
        };
        let per_draw = groups
            .par_iter()
            .map(|(d, by_set)| {
                let signs = self.draws[*d];
                let cc = if signs.is_empty() {
                    Vec::new()
                } else {
                    let own: Vec<f64> = subset.iter().map(|&i| signs[i]).collect();
                    within_cross(&parts, &all, &own)
                };
                let g = if signs.is_empty() {
                    merged.arrays()
                } else {
                    GramArrays {
                        cross_moments: &cc,
                        cross_centred: &cc,
                        ..merged.arrays()
                    }
                };
                let per_set = by_set
                    .par_iter()
                    .map(|(set, js)| self.group(&g, &self.sets[*set], js, &all, &at))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(per_set.into_iter().flatten().collect::<Vec<_>>())
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(per_draw.into_iter().flatten().collect())
    }

    /// The jobs `js`, which share a draw and column set `set`, fitted on
    /// `g`, the merged Gram on the columns `all`, which `at` places the
    /// Grams' columns in.
    fn group(
        &self,
        g: &GramArrays<'_>,
        set: &ColumnSet,
        js: &[usize],
        all: &[usize],
        at: &[usize],
    ) -> PolarsResult<Vec<(usize, SubsetOut)>> {
        let k = at.len();
        let expand = |v: &[f64], fill: f64| -> Vec<f64> {
            let mut out = vec![fill; k];
            for (r, &i) in all.iter().enumerate() {
                out[i] = v[r];
            }
            out
        };
        let cols = FitColumns {
            slots: set.slots.iter().map(|&c| at[c]).collect(),
            icept: self.icept.map(|_| 0),
        };
        let targets: Vec<usize> = js.iter().map(|&j| self.jobs[j].target).collect();
        let fits: Vec<SubsetOut> = match self.fit {
            JobFit::Ridge { ridge, standardize } => {
                ridge_fits(g, &targets, &cols.slots, cols.icept, *ridge, *standardize)
                    .into_iter()
                    .map(|f| {
                        SubsetOut::Ridge(RidgeFit {
                            coef: expand(&f.coef, 0.0),
                            se: expand(&f.se, f64::NAN),
                            t: expand(&f.t, f64::NAN),
                            ..f
                        })
                    })
                    .collect()
            }
            JobFit::Path { limits } => {
                let weights = if set.weights.is_empty() {
                    vec![1.0; cols.slots.len()]
                } else {
                    set.weights.clone()
                };
                let rows = GramRows::new(*g, &cols.slots, cols.icept);
                targets
                    .par_iter()
                    .map(|&t| {
                        let p = path_of(g, t, &rows, &cols, &weights, *limits)?;
                        let n = g.k;
                        let coefs = (0..p.penalties.len())
                            .flat_map(|r| expand(&p.coefs[r * n..(r + 1) * n], 0.0))
                            .collect();
                        let active = p
                            .active
                            .iter()
                            .map(|a| a.iter().map(|&r| all[r]).collect())
                            .collect();
                        Ok(SubsetOut::Path(GramPath { coefs, active, ..p }))
                    })
                    .collect::<PolarsResult<_>>()?
            }
        };
        Ok(js.iter().copied().zip(fits).collect())
    }
}
