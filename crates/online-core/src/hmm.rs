//! `hmm`: a Gaussian hidden Markov model, filtered online
//! (docs/ENHANCEMENTS.md E60).
//!
//! `ew_class` classifies a row against labelled Gaussians. An `hmm` does the
//! same arithmetic with **no labels**: the state is hidden, and what carries
//! information from one row to the next is a transition matrix. That is the
//! difference between "which regime does this row look like" and "which
//! regime are we in", and the second is usually the question.
//!
//! # The filter
//!
//! Hamilton's, one row at a time. Before the row, from the filtered `p` the
//! previous row left (uniform `1/K` before the first):
//!
//! ```text
//! p̃ₗ = Σₖ pₖ·Πₖₗ                                   the predicted state
//! fₗ = N(x | μₗ, Σₗ + rₗI)                          the state's density
//! loglik = ln Σₗ p̃ₗ·fₗ                              the row's surprise
//! pₗ ← p̃ₗ·fₗ / Σ                                    the filtered state
//! ```
//!
//! `p`, `p̃`, `argmax p̃` and `loglik` are all read from the state *before*
//! the row is learned from, so they are safe as features for that same row.
//! The densities go through the same `quad_forms_logdet` and softmax path
//! `ew_class` uses, with the same decaying `precision_prior` ridge -- which
//! is **required** here as it is there: a state's centred co-moments start
//! at zero, and a zero matrix has no density.
//!
//! # Learning
//!
//! Each state's accumulator takes the row at weight `w·pₗ`. The
//! responsibilities sum to `w`, so the model's `n_eff` is the shared
//! recursion untouched.
//!
//! The transition matrix is learned from the **filtered joint of
//! consecutive states**:
//!
//! ```text
//! ξₖₗ = pₖ(t−1)·Πₖₗ·fₗ / Σ_{k'l'} p_{k'}(t−1)·Π_{k'l'}·f_{l'}
//! Aₖₗ ← decay·Aₖₗ + w·ξₖₗ
//! Πₖₗ = (Aₖₗ + τ) / Σₗ(Aₖₗ + τ)
//! ```
//!
//! with `τ` a Dirichlet pseudo-count per cell, which is what keeps a
//! never-visited row of `Π` a distribution. A transition is one row: `Π`
//! applies once per row whatever the clock between rows, so a weekend is one
//! step, and `A` decays on the clock but grows by `w` per row, so the
//! staying probability rises with the rows' density (docs/PLAN.md task 146). **Not** the EW mean of
//! `pₖ(t−1)·pₗ(t)/pₖ(t−1)`: that ratio is `pₗ(t)`, whose mean does not
//! depend on `k` and cannot identify a transition matrix at all.
//!
//! With `tvtp` the matrix is a function of an exogenous column instead,
//! `Πₖₗ(t) = softmaxₗ(Aₖₗ + Bₖₗ·zₜ)` from fixed coefficients, and the
//! count-based learning is off.

use serde::{Deserialize, Serialize};

use crate::cluster::kmeans::seed_centres;
use crate::solve::{SpdFactor, quad_forms_logdet};
use crate::{Covariance, Decay, EwCov, SeedRule};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HmmCfg {
    pub n_features: usize,
    /// Hidden states, `>= 2`.
    pub k: usize,
    pub decay: Decay,
    pub covariance: Covariance,
    /// Ridge on every state covariance, finite and `> 0`; decays with the
    /// state's own data, as `ew_class`'s does. Required: a state's centred
    /// co-moments start at zero.
    pub precision_prior: f64,
    pub min_weight: f64,
    /// Update the states and the transition counts. `false` filters with
    /// what it was given and learns nothing.
    pub learn: bool,
    /// Dirichlet pseudo-count per cell of the transition matrix.
    pub transition_prior: f64,
    /// A `K*K` row-stochastic matrix `Π₀`, the Dirichlet prior's mean: each
    /// cell's pseudo-count is `τ·K·Π₀[r][c]` in place of `τ`, and the learned
    /// counts start at zero (see `prior`); `None` is uniform.
    pub transition: Option<Vec<f64>>,
    /// State means, `K*d` row-major: given, there is no warm-up. The pair
    /// enters the accumulators at **weight 1** -- one row's worth -- so the
    /// given states are a starting point that the stream washes out under
    /// `learn = true` -- within about one half-life under a finite one, since
    /// the pair weighs one row -- and are held exactly under `learn = false`.
    pub means: Option<Vec<f64>>,
    /// State covariances, `K*d*d` row-major, beside `means`.
    pub covs: Option<Vec<f64>>,
    /// Learned rows buffered before the states are seeded from them.
    pub warm_rows: usize,
    pub seed_rule: SeedRule,
    pub seed: u64,
    /// `(A, B)`, each `K*K` row-major: with them, `Π(t) =
    /// softmaxₗ(Aₖₗ + Bₖₗ·zₜ)` from an exogenous column and the counts are
    /// not learned.
    pub tvtp: Option<(Vec<f64>, Vec<f64>)>,
}

impl HmmCfg {
    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("hmm: {e}"))?;
        if self.n_features == 0 {
            return Err("hmm: n_features must be >= 1".into());
        }
        if self.k < 2 {
            return Err("hmm: k must be >= 2 (one state is not a Markov chain)".into());
        }
        if !(self.precision_prior.is_finite() && self.precision_prior > 0.0) {
            return Err(
                "hmm: precision_prior must be finite and > 0; a state's centred co-moments start \
                 at zero, and a zero matrix has no density"
                    .into(),
            );
        }
        if !(self.transition_prior.is_finite() && self.transition_prior >= 0.0) {
            return Err("hmm: transition_prior must be finite and >= 0".into());
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err("hmm: min_weight must be >= 0".into());
        }
        let (k, d) = (self.k, self.n_features);
        if let Some(p) = &self.transition {
            if p.len() != k * k {
                return Err(format!("hmm: transition must be {k}x{k}, got {}", p.len()));
            }
            for r in 0..k {
                let row = &p[r * k..(r + 1) * k];
                if row.iter().any(|v| *v < 0.0 || !v.is_finite())
                    || (row.iter().sum::<f64>() - 1.0).abs() > 1e-9
                {
                    return Err(format!("hmm: transition row {r} is not a distribution"));
                }
            }
        }
        match (&self.means, &self.covs) {
            (Some(m), Some(c)) => {
                if m.len() != k * d {
                    return Err(format!("hmm: means must be {k}x{d}, got {}", m.len()));
                }
                if c.len() != k * d * d {
                    return Err(format!(
                        "hmm: covs must be {k} matrices of {d}x{d}, got {}",
                        c.len()
                    ));
                }
                if m.iter().chain(c.iter()).any(|v| !v.is_finite()) {
                    return Err("hmm: means and covs must be finite".into());
                }
                // A state covariance that will not factorize has no density,
                // so every row would be a solve failure and every output a
                // null, with nothing said (docs/REVIEW-E54-E64.md H4).
                for s in 0..k {
                    let block = &c[s * d * d..(s + 1) * d * d];
                    for i in 0..d {
                        for j in (i + 1)..d {
                            let (a, b) = (block[i * d + j], block[j * d + i]);
                            if (a - b).abs() > 1e-12 * (1.0 + a.abs().max(b.abs())) {
                                return Err(format!(
                                    "hmm: covs[{s}] must be symmetric; [{i}][{j}] is {a} and \
                                     [{j}][{i}] is {b}"
                                ));
                            }
                        }
                    }
                    if !matches!(SpdFactor::of(block, d), Some(f) if f.attempts() == 0) {
                        return Err(format!(
                            "hmm: covs[{s}] must be positive definite; a state with no density \
                             takes no responsibility for any row"
                        ));
                    }
                }
            }
            (None, None) => {
                if !self.learn {
                    return Err(
                        "hmm: learn = false with no means/covs has nothing to filter with; give \
                         the states, or let it seed them"
                            .into(),
                    );
                }
                if self.warm_rows < k {
                    return Err(format!(
                        "hmm: warm_rows must be at least k ({k}) to seed that many states"
                    ));
                }
            }
            _ => return Err("hmm: means and covs go together".into()),
        }
        if let Some((a, b)) = &self.tvtp
            && (a.len() != k * k || b.len() != k * k)
        {
            return Err(format!("hmm: tvtp_coef A and B must each be {k}x{k}"));
        }
        Ok(())
    }
}

/// What [`Hmm::read`] hands the update: the filtered posterior after the
/// row, and the per-state log densities the transition counts need.
type Update = (Vec<f64>, Vec<f64>);

/// Per-state Cholesky factors for the `full` shape: built on demand and
/// dropped whenever a state changes. Derived state -- a pure function of the
/// accumulators -- so it is neither serialized (rebuilt after a load) nor
/// compared. The same cache `ew_class` keeps (docs/PERFORMANCE.md §13); the
/// win is the `learn = false` scorer, which never changes a state and so
/// factorizes each state once rather than on every row (review 2026-09-18).
#[derive(Debug, Clone, Default)]
struct Factors(Vec<Option<SpdFactor>>);

impl PartialEq for Factors {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Factors {
    /// The ready factor for state `s`, or `None` if it must be built.
    fn peek(&self, s: usize) -> Option<&SpdFactor> {
        self.0.get(s).and_then(Option::as_ref)
    }

    /// Store state `s`'s factor, sizing the cache to `k` states first.
    fn set(&mut self, s: usize, k: usize, f: Option<SpdFactor>) {
        if self.0.len() != k {
            self.0 = vec![None; k];
        }
        self.0[s] = f;
    }

    /// Drop every factor; the next `ensure_factors` rebuilds, so a stale
    /// factor is never read.
    fn clear(&mut self) {
        self.0.clear();
    }
}

/// A Gaussian hidden Markov model; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hmm {
    cfg: HmmCfg,
    /// One accumulator per state.
    states: Vec<EwCov>,
    /// The filtered posterior left by the last learned row.
    p: Vec<f64>,
    /// Decayed counts of the filtered joint, `K*K` row-major.
    a: Vec<f64>,
    /// EW weight of every accepted row.
    n_eff: f64,
    /// Rows waiting to seed the states, with their weights.
    buffer: Vec<(Vec<f64>, f64)>,
    seeded: bool,
    pub solve_failures: u64,
    /// The `full` shape's per-state factors between rows; see [`Factors`].
    #[serde(skip)]
    factors: Factors,
}

impl Hmm {
    pub fn new(cfg: HmmCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (k, d) = (cfg.k, cfg.n_features);
        // No window reads a run here, so none is kept (docs/PLAN.md task
        // 128; review 2026-09-26, C3).
        let mut states = (0..k)
            .map(|_| EwCov::with_precision_prior(d, cfg.precision_prior).map(EwCov::without_runs))
            .collect::<Result<Vec<_>, _>>()?;
        let seeded = cfg.means.is_some();
        if let (Some(m), Some(c)) = (&cfg.means, &cfg.covs) {
            for (s, i) in states.iter_mut().zip(0..k) {
                *s = EwCov::from_moments(
                    d,
                    1.0,
                    &m[i * d..(i + 1) * d],
                    &c[i * d * d..(i + 1) * d * d],
                    cfg.precision_prior,
                )?
                .without_runs();
            }
        }
        Ok(Self {
            p: vec![1.0 / k as f64; k],
            a: vec![0.0; k * k],
            n_eff: 0.0,
            buffer: Vec::new(),
            seeded,
            solve_failures: 0,
            factors: Factors::default(),
            states,
            cfg,
        })
    }

    /// Build the `full` shape's per-state factors from the current (pre-row)
    /// accumulators so `read` reuses them instead of factorizing every row.
    /// A no-op for the other shapes and before seeding; the state-updating
    /// paths `clear` the cache, so under `learn = false` it is built once.
    fn ensure_factors(&mut self) {
        if !self.seeded || !matches!(self.cfg.covariance, Covariance::Full) {
            return;
        }
        let (k, d) = (self.cfg.k, self.cfg.n_features);
        for s in 0..k {
            if self.factors.peek(s).is_none() {
                let m = self.state_matrix(s);
                self.factors.set(s, k, SpdFactor::of(&m, d));
            }
        }
    }

    pub fn cfg(&self) -> &HmmCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// The filtered posterior over states, as the next row will read it.
    pub fn filtered(&self) -> &[f64] {
        &self.p
    }

    /// The accumulator of state `s`.
    pub fn state_cov(&self, s: usize) -> &EwCov {
        &self.states[s]
    }

    /// The Dirichlet pseudo-count of cell `(r, c)`.
    ///
    /// A flat `τ` when no `transition` was given, and `τ·K·Π₀[r][c]` when
    /// one was: the row's prior mass is `τ·K` either way, and with no
    /// counts yet `Π` is exactly `Π₀`. (`docs/PLAN.md` §11a said the
    /// *counts* start at `τ·K·Π₀`; that gives `(K·Π₀ + 1)/(2K)`, not `Π₀`,
    /// and a cell of `Π₀` below `1/K` would need a negative count. Putting
    /// the shape in the prior is the same idea with the arithmetic right.)
    fn prior(&self, r: usize, c: usize) -> f64 {
        let (k, tau) = (self.cfg.k, self.cfg.transition_prior);
        match &self.cfg.transition {
            Some(p) => tau * k as f64 * p[r * k + c],
            None => tau,
        }
    }

    /// The transition matrix in force, `K*K` row-major: the counts plus the
    /// Dirichlet prior, normalised. Under `tvtp` this is the count-based
    /// one, which that mode does not use.
    pub fn transition(&self) -> Vec<f64> {
        let k = self.cfg.k;
        let mut out = vec![0.0; k * k];
        for r in 0..k {
            let row: Vec<f64> = (0..k)
                .map(|c| self.a[r * k + c] + self.prior(r, c))
                .collect();
            let z: f64 = row.iter().sum();
            for (c, v) in row.iter().enumerate() {
                out[r * k + c] = if z > 0.0 {
                    v / z
                } else {
                    // No counts and no prior mass (`τ = 0`): the prior's
                    // mean, `Π₀` or uniform, which is what the row is at
                    // every `τ > 0` before a count (task 158).
                    match &self.cfg.transition {
                        Some(p) => p[r * k + c],
                        None => 1.0 / k as f64,
                    }
                };
            }
        }
        out
    }

    /// The exogenous value rides in `y[0]` when `tvtp` is configured: the
    /// plumbing declares it like `weight`, and the model reads one number.
    fn exog_of(&self, y: &[Option<f64>]) -> Option<f64> {
        self.cfg
            .tvtp
            .as_ref()
            .and_then(|_| y.first().copied().flatten())
    }

    /// `Π(t)` from the exogenous value, when `tvtp` is configured.
    fn transition_at(&self, exog: Option<f64>) -> Vec<f64> {
        let k = self.cfg.k;
        let Some((a, b)) = &self.cfg.tvtp else {
            return self.transition();
        };
        let z = exog.filter(|v| v.is_finite()).unwrap_or(0.0);
        let mut out = vec![0.0; k * k];
        for r in 0..k {
            let ell: Vec<f64> = (0..k).map(|c| a[r * k + c] + b[r * k + c] * z).collect();
            let top = ell.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = ell.iter().map(|v| (v - top).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (c, e) in exps.iter().enumerate() {
                out[r * k + c] = e / sum;
            }
        }
        out
    }

    fn ridge(&self, s: usize) -> f64 {
        self.cfg.precision_prior * self.states[s].precision_scale()
    }

    fn state_matrix(&self, s: usize) -> Vec<f64> {
        let d = self.cfg.n_features;
        let mut m = self.states[s].comoments().to_vec();
        let r = self.ridge(s);
        for i in 0..d {
            m[i * d + i] += r;
        }
        m
    }

    /// `ln fₗ` per state, or `None` when a factorization failed.
    ///
    /// The Gaussian constant `−(d/2)·ln 2π` is included, so `loglik` is a
    /// density and not a density up to a constant.
    fn log_densities(&self, x: &[f64]) -> Option<Vec<f64>> {
        let (k, d) = (self.cfg.k, self.cfg.n_features);
        let base = -0.5 * d as f64 * std::f64::consts::TAU.ln();
        let mut out = vec![f64::NEG_INFINITY; k];
        match self.cfg.covariance {
            Covariance::Diagonal => {
                for (s, o) in out.iter_mut().enumerate() {
                    let cov = &self.states[s];
                    let r = self.ridge(s);
                    let (mut log_det, mut q) = (0.0, 0.0);
                    for (i, xi) in x.iter().enumerate() {
                        let v = cov.var(i) + r;
                        let dv = cov.deviation(i, *xi);
                        log_det += v.ln();
                        q += dv * dv / v;
                    }
                    *o = base - 0.5 * log_det - 0.5 * q;
                }
            }
            Covariance::Shared => {
                // One matrix, weighted by the states' own weights, and every
                // state's quadratic form against it from one factorization.
                let weights: Vec<f64> = self.states.iter().map(EwCov::n_eff).collect();
                let total: f64 = weights.iter().sum();
                let mut m = vec![0.0; d * d];
                for (s, &w) in weights.iter().enumerate() {
                    let pi = if total > 0.0 {
                        w / total
                    } else {
                        1.0 / k as f64
                    };
                    for (mij, cij) in m.iter_mut().zip(self.states[s].comoments()) {
                        *mij += pi * cij;
                    }
                    for i in 0..d {
                        m[i * d + i] += pi * self.ridge(s);
                    }
                }
                let mut deltas = vec![0.0; d * k];
                for (s, state) in self.states.iter().enumerate() {
                    for (i, &xi) in x.iter().enumerate() {
                        deltas[s * d + i] = state.deviation(i, xi);
                    }
                }
                let (q, log_det, _) = quad_forms_logdet(&m, &deltas, d, k)?;
                for (s, o) in out.iter_mut().enumerate() {
                    *o = base - 0.5 * log_det - 0.5 * q[s];
                }
            }
            Covariance::Full => {
                let mut delta = vec![0.0; d];
                for (s, o) in out.iter_mut().enumerate() {
                    let cov = &self.states[s];
                    for (i, (dv, &xi)) in delta.iter_mut().zip(x).enumerate() {
                        *dv = cov.deviation(i, xi);
                    }
                    // The cached factor when `ensure_factors` has built it
                    // (the same `SpdFactor::of` on the same matrix, so the
                    // result is bit-identical); otherwise built here, which is
                    // the `&self` predict path and the failure case.
                    let built;
                    let f = match self.factors.peek(s) {
                        Some(f) => f,
                        None => {
                            built = SpdFactor::of(&self.state_matrix(s), d)?;
                            &built
                        }
                    };
                    *o = base - 0.5 * f.log_det() - 0.5 * f.quad_forms(&delta, d, 1)[0];
                }
            }
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// Output slot labels in emission order: `p_<k>`, `p1_<k>`, `state`,
    /// `loglik`.
    pub fn labels(k: usize) -> Vec<String> {
        let mut out: Vec<String> = (0..k).map(|s| format!("filtered_{s}")).collect();
        out.extend((0..k).map(|s| format!("predicted_{s}")));
        out.push("state".into());
        out.push("loglik".into());
        out
    }

    pub fn n_outputs_for(k: usize) -> usize {
        2 * k + 2
    }

    /// The row's outputs, and the pieces the update needs: the filtered
    /// posterior after this row and the per-state log densities.
    fn read(&self, x: &[f64], exog: Option<f64>) -> (Vec<f64>, Option<Update>) {
        let k = self.cfg.k;
        let nan = vec![f64::NAN; Self::n_outputs_for(k)];
        if !self.seeded {
            return (nan, None);
        }
        let pi = self.transition_at(exog);
        // `p̃ₗ = Σₖ pₖ Πₖₗ`.
        let mut pred = vec![0.0; k];
        for (kk, &pk) in self.p.iter().enumerate() {
            for (l, o) in pred.iter_mut().enumerate() {
                *o += pk * pi[kk * k + l];
            }
        }
        let Some(logf) = self.log_densities(x) else {
            return (nan, None);
        };
        // `ln Σ p̃ f` about the maximum, so a very small density does not
        // underflow to `-inf` before the sum.
        let ell: Vec<f64> = (0..k)
            .map(|l| {
                if pred[l] > 0.0 {
                    pred[l].ln() + logf[l]
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let top = ell.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return (nan, None);
        }
        let exps: Vec<f64> = ell.iter().map(|v| (v - top).exp()).collect();
        let z: f64 = exps.iter().sum();
        let loglik = top + z.ln();
        let post: Vec<f64> = exps.iter().map(|e| e / z).collect();
        let mut best = 0;
        for l in 1..k {
            if pred[l] > pred[best] {
                best = l;
            }
        }
        let mut out = Vec::with_capacity(Self::n_outputs_for(k));
        out.extend_from_slice(&self.p);
        out.extend_from_slice(&pred);
        out.push(best as f64);
        out.push(loglik);
        // `min_weight` gates what is *reported*, never what is learned: a
        // gated row still moves the filter, as it does in every other model
        // here. Returning `None` instead withheld the row from the update
        // and counted it as a solve failure, so under decay a filter whose
        // `n_eff` plateaued below `min_weight` never learned at all
        // (docs/REVIEW-E54-E64.md H1).
        if self.n_eff < self.cfg.min_weight {
            return (nan, Some((post, logf)));
        }
        (out, Some((post, logf)))
    }

    /// Seed the states from the buffer, `kmeans`' recipe: choose centres
    /// under `seed_rule`, then replay the buffer through them as hard
    /// assignments so each state starts with real moments.
    fn seed(&mut self) {
        let (k, d) = (self.cfg.k, self.cfg.n_features);
        let rows: Vec<Vec<f64>> = self.buffer.iter().map(|(x, _)| x.clone()).collect();
        let w: Vec<f64> = self.buffer.iter().map(|(_, w)| *w).collect();
        let mw = vec![1.0; d];
        let Some(centres) =
            seed_centres(&rows, &w, k, self.cfg.seed_rule, self.cfg.seed, true, &mw)
        else {
            return;
        };
        for (x, wi) in &self.buffer {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (s, c) in centres.iter().enumerate() {
                let dist: f64 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                if dist < best_d {
                    best_d = dist;
                    best = s;
                }
            }
            self.states[best].update(x, 1.0, *wi);
        }
        self.buffer.clear();
        self.seeded = true;
        self.factors.clear();
    }
}

impl crate::OnlineModel for Hmm {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        self.ensure_factors();
        let (pred, extra) = self.read(x, self.exog_of(_y));
        let out = crate::Step {
            pred,
            n_eff: self.n_eff,
            extra: None,
        };
        let lam = self.cfg.decay.factor(d_clock);
        // The warm-up rows age as `n_eff` does, so the states the replay seeds
        // weigh what `n_eff` does; replayed at their raw weights, a state
        // began heavier than its rows' age warranted and its ridge weaker
        // (review 2026-09-12, V25). `kmeans` ages its buffer the same way. The
        // buffer is empty once the states are seeded.
        for (_, w) in &mut self.buffer {
            *w *= lam;
        }
        if weight <= 0.0 {
            // Advance the clock and learn nothing: the counts age with the
            // accumulators, `n_eff` decays with them -- hard rule 8, the
            // same recursion in every model, which is what makes
            // `min_weight` mean the same number of rows across a bank
            // (docs/REVIEW-E54-E64.md H3) -- and `p` does not move.
            self.n_eff *= lam;
            if self.cfg.learn {
                self.a.iter_mut().for_each(|v| *v *= lam);
                for s in self.states.iter_mut() {
                    s.update(x, lam, 0.0);
                }
                self.factors.clear();
            }
            return out;
        }
        let before = self.n_eff;
        self.n_eff = lam * before + weight;
        if !self.seeded {
            self.buffer.push((x.to_vec(), weight));
            if self.buffer.len() >= self.cfg.warm_rows {
                self.seed();
            }
            return out;
        }
        let Some((post, logf)) = extra else {
            // The densities were all non-finite, so the row cannot be scored
            // or learned. It still happened, so it ages the clock like a
            // zero-weight row -- `n_eff`, the counts and the states all decay
            // by `lam`, nothing is added -- rather than `n_eff` advancing on
            // its own (review 2026-09-18).
            self.solve_failures += 1;
            self.n_eff = lam * before;
            if self.cfg.learn {
                self.a.iter_mut().for_each(|v| *v *= lam);
                for s in self.states.iter_mut() {
                    s.update(x, lam, 0.0);
                }
                self.factors.clear();
            }
            return out;
        };
        if self.cfg.learn {
            let k = self.cfg.k;
            // The filtered joint of consecutive states, from the *pre-row*
            // `p` and `Π`; the counts decay on the clock like every
            // accumulator here.
            if self.cfg.tvtp.is_none() {
                let pi = self.transition();
                // About the largest density, not `logf[0]`: a state whose
                // density is astronomically small makes `exp(logf[l] −
                // logf[0])` overflow the other way, and the counts go to
                // NaN. The reference cancels in the normalisation.
                let top = logf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let scaled: Vec<f64> = logf.iter().map(|v| (v - top).exp()).collect();
                let mut joint = vec![0.0; k * k];
                let mut z = 0.0;
                for kk in 0..k {
                    for l in 0..k {
                        let v = self.p[kk] * pi[kk * k + l] * scaled[l];
                        joint[kk * k + l] = v;
                        z += v;
                    }
                }
                for (a, j) in self.a.iter_mut().zip(&joint) {
                    *a = lam * *a + if z > 0.0 { weight * j / z } else { 0.0 };
                }
            }
            for (s, cov) in self.states.iter_mut().enumerate() {
                cov.update(x, lam, weight * post[s]);
            }
            self.factors.clear();
        }
        self.p = post;
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> crate::Step {
        let exog = self.cfg.tvtp.as_ref().map(|_| 0.0);
        crate::Step {
            pred: self.read(x, exog).0,
            n_eff: self.n_eff,
            extra: None,
        }
    }

    /// The exogenous value rides in `y[0]` under `tvtp`, and `Π(t)` is a
    /// function of it, so the answer depends on it exactly as the step's
    /// does (docs/REVIEW-E54-E64.md C1/H2).
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], _d_clock: f64) -> crate::Step {
        crate::Step {
            pred: self.read(x, self.exog_of(y)).0,
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::Hmm(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Hmm(m) => {
                let mut m = (**m).clone();
                // `k` states at the cfg's width, `k` marginals, a `k×k`
                // transition matrix and buffered rows `d` long (review
                // 2026-09-18, B3).
                let (k, d) = (m.cfg.k, m.cfg.n_features);
                if m.states.len() != k
                    || m.states.iter().any(|s| !s.has_shape(d))
                    || m.p.len() != k
                    || m.a.len() != k * k
                    || m.buffer.iter().any(|(x, _)| x.len() != d)
                {
                    return Err(crate::StateError::Invalid(
                        "hmm: the state has the wrong shape".into(),
                    ));
                }
                // No window reads a run here (review 2026-09-26, C3 and C4).
                m.states.iter_mut().for_each(EwCov::set_runs_off);
                Ok(m)
            }
            other => Err(crate::StateError::WrongModel {
                expected: "hmm",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        Self::n_outputs_for(self.cfg.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Hmm::new(cfg(2, 2)).unwrap();
        let mut s = m.state();
        let ModelState::Hmm(inner) = &mut s.model else {
            unreachable!()
        };
        inner.p.pop();
        match Hmm::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::{EwClass, EwClassCfg, OnlineModel};

    /// A row whose densities are all non-finite cannot be scored or learned,
    /// but it still happened: it must age `n_eff`, the counts and the states
    /// together, as a zero-weight row does -- not advance `n_eff` alone
    /// (review 2026-09-18). At `half_life = inf` (lam = 1) that means `n_eff`
    /// does not move on the failed row. A huge but finite feature overflows
    /// the quadratic form and forces the failure.
    #[test]
    fn a_row_that_fails_to_score_does_not_advance_n_eff() {
        let mut m = Hmm::new(cfg(2, 2)).unwrap();
        for (x, _) in &stream(200, 11, 40) {
            crate::OnlineModel::step(&mut m, x, &[], 1.0, 1.0);
        }
        let n_eff_before = m.n_eff;
        let sum_before: f64 = (0..2).map(|s| m.state_cov(s).n_eff()).sum();
        let fails = m.solve_failures;
        crate::OnlineModel::step(&mut m, &[1e300, 1e300], &[], 1.0, 1.0);
        assert_eq!(
            m.solve_failures,
            fails + 1,
            "the huge row did not fail to score"
        );
        assert_eq!(m.n_eff, n_eff_before, "a failed row advanced n_eff alone");
        let sum_after: f64 = (0..2).map(|s| m.state_cov(s).n_eff()).sum();
        assert_eq!(
            sum_after, sum_before,
            "a failed row moved the state weights"
        );
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(d: usize, k: usize) -> HmmCfg {
        HmmCfg {
            n_features: d,
            k,
            decay: Decay::Halflife(f64::INFINITY),
            covariance: Covariance::Full,
            precision_prior: 1e-3,
            min_weight: 0.0,
            learn: true,
            transition_prior: 1.0,
            transition: None,
            means: None,
            covs: None,
            warm_rows: 20,
            seed_rule: SeedRule::First,
            seed: 7,
            tvtp: None,
        }
    }

    /// Two well-separated Gaussian blobs, visited in runs.
    fn stream(n: usize, seed: u64, run: usize) -> Vec<(Vec<f64>, usize)> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let g = (i / run) % 2;
                let c = if g == 0 { -3.0 } else { 3.0 };
                (vec![c + lcg(&mut s), c + lcg(&mut s)], g)
            })
            .collect()
    }

    /// The warm-up replay gives each state its rows' weights as the clock has
    /// aged them, as `n_eff` has: at seeding, the states' weights sum to
    /// `n_eff`. They took the raw weights, replayed at `lam = 1`, so a state
    /// began heavier than its rows' age warranted, and its `precision_prior`
    /// ridge weaker (review 2026-09-12, V25).
    #[test]
    fn the_seeded_states_weigh_what_n_eff_does() {
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(10.0);
        let mut m = Hmm::new(c).unwrap();
        let mut s = 5u64;
        for (i, (x, _)) in stream(20, 17, 5).iter().enumerate() {
            let w = 0.5 + lcg(&mut s).abs();
            m.step(x, &[], if i == 0 { 0.0 } else { 1.0 }, w);
        }
        assert!(m.seeded, "twenty rows seed the states");
        let total: f64 = (0..2).map(|j| m.state_cov(j).n_eff()).sum();
        assert!(
            (total - m.n_eff()).abs() <= 1e-12 * m.n_eff(),
            "{total} in the states, {} in n_eff",
            m.n_eff()
        );
    }

    /// With a uniform `Π` and `learn = false`, the filter is the classifier:
    /// `p̃` is flat, so `p ∝ f` -- exactly what `ew_class`'s posterior is
    /// when its classes carry equal weight, at the same ridge. The
    /// classifier's ridge is its `precision_prior` times each class's decayed
    /// prior scale, 1/100 after 100 rows a class, and `from_moments` starts
    /// the hmm's scale at 1, so the hmm is built at the classifier's
    /// `precision_prior` times that scale.
    ///
    /// On classes that overlap (blobs at ±0.5 with noise of ±1), whose
    /// posteriors are numbers between 0 and 1, so the ridge and the prior
    /// both show in them. Blobs at ±3 decided every posterior to 1e-32, and
    /// the test passed with the hmm's ridge 100 times the classifier's
    /// (review 2026-10-05, CD3): that ridge is the control here, and must
    /// show.
    ///
    /// Not bit-identical: `ew_class` adds `ln π_c` to each log density
    /// before the softmax and this does not, and adding then subtracting a
    /// constant inside a softmax is not the identity in f64. `1e-12` is what
    /// the two orders of operations agree to.
    #[test]
    fn a_uniform_chain_is_the_classifier() {
        use crate::OnlineModel;
        let d = 2;
        let mut cls = EwClass::new(EwClassCfg {
            n_features: d,
            n_classes: 2,
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            covariance: Covariance::Full,
            precision_prior: 1e-3,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut s = 3u64;
        let rows: Vec<(Vec<f64>, usize)> = (0..200)
            .map(|i| {
                let c = if i % 2 == 0 { -0.5 } else { 0.5 };
                (vec![c + lcg(&mut s), c + lcg(&mut s)], i % 2)
            })
            .collect();
        for (x, g) in &rows {
            cls.step(x, &[Some(*g as f64)], 1.0, 1.0);
        }
        // Exactly balanced classes: `ln π_c` is then the same f64 for both,
        // so the only difference left is the softmax's own arithmetic; and
        // the two classes' prior scales are the same double.
        assert_eq!(cls.class_weights()[0], cls.class_weights()[1]);
        let scale = cls.class_cov(0).precision_scale();
        assert_eq!(scale, cls.class_cov(1).precision_scale());
        assert!((scale - 0.01).abs() < 1e-12, "100 rows a class: {scale}");
        let means: Vec<f64> = (0..2)
            .flat_map(|c| cls.class_cov(c).means().to_vec())
            .collect();
        let covs: Vec<f64> = (0..2)
            .flat_map(|c| cls.class_cov(c).comoments().to_vec())
            .collect();
        let build = |precision_prior: f64| {
            Hmm::new(HmmCfg {
                learn: false,
                means: Some(means.clone()),
                covs: Some(covs.clone()),
                precision_prior,
                ..cfg(d, 2)
            })
            .unwrap()
        };
        let mut hmm = build(1e-3 * scale);
        let mut control = build(1e-3);
        let (mut open, mut off) = (0, 0.0f64);
        // The classifier frozen: each row scored by both from fixed moments.
        for (x, _) in &rows {
            let a = cls.predict(x, 1.0);
            hmm.step(x, &[], 1.0, 1.0);
            control.step(x, &[], 1.0, 1.0);
            // `ew_class`: [class, p_0, p_1]. The hmm's filtered posterior
            // after the row is `p̃ f` normalised, which with a flat `p̃` is
            // the classifier's posterior for the row.
            for (c, &got) in hmm.filtered().iter().enumerate() {
                let want = a.pred[1 + c];
                assert!((got - want).abs() < 1e-12, "state {c}: {got} vs {want}");
                if (0.05..0.95).contains(&want) {
                    open += 1;
                }
                off = off.max((control.filtered()[c] - want).abs());
            }
        }
        assert!(open > 100, "the posteriors are not decided: {open}");
        assert!(
            off > 1e-6,
            "a ridge 100 times the classifier's shows: {off:e}"
        );
    }

    /// A longhand Hamilton filter at fixed parameters.
    /// Each state's log density is the Gaussian of its own moments, under
    /// every covariance kind, written out with explicit indices for two
    /// features and two states: the deviation from the pair, the ridge added
    /// to each variance, and the quadratic form through a 2x2 inverse.
    #[test]
    fn log_densities_are_the_gaussians_of_each_state() {
        let (d, k) = (2usize, 2usize);
        let x = [0.3, -0.2];
        for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
            let m = Hmm::new(HmmCfg {
                learn: false,
                covariance,
                means: Some(vec![-3.0, -2.0, 3.0, 2.5]),
                covs: Some(vec![1.0, 0.3, 0.3, 2.0, 1.5, -0.2, -0.2, 0.8]),
                ..cfg(d, k)
            })
            .unwrap();
            let got = m.log_densities(&x).unwrap();
            let base = -0.5 * d as f64 * std::f64::consts::TAU.ln();
            let quad = |mat: &[f64], dv: &[f64]| {
                let det = mat[0] * mat[3] - mat[1] * mat[2];
                let q = (mat[3] * dv[0] * dv[0] - (mat[1] + mat[2]) * dv[0] * dv[1]
                    + mat[0] * dv[1] * dv[1])
                    / det;
                (det.ln(), q)
            };
            let total: f64 = m.states.iter().map(EwCov::n_eff).sum();
            let mut shared = vec![0.0; 4];
            for (s, state) in m.states.iter().enumerate() {
                let pi = if total > 0.0 {
                    state.n_eff() / total
                } else {
                    0.5
                };
                for (j, c) in state.comoments().iter().enumerate() {
                    shared[j] += pi * c;
                }
                shared[0] += pi * m.ridge(s);
                shared[3] += pi * m.ridge(s);
            }
            for (s, state) in m.states.iter().enumerate() {
                let dv = [state.deviation(0, x[0]), state.deviation(1, x[1])];
                let r = m.ridge(s);
                let want = match covariance {
                    Covariance::Diagonal => {
                        let (v0, v1) = (state.var(0) + r, state.var(1) + r);
                        base - 0.5 * (v0.ln() + v1.ln())
                            - 0.5 * (dv[0] * dv[0] / v0 + dv[1] * dv[1] / v1)
                    }
                    Covariance::Shared => {
                        let (log_det, q) = quad(&shared, &dv);
                        base - 0.5 * log_det - 0.5 * q
                    }
                    Covariance::Full => {
                        let mut mat = state.comoments().to_vec();
                        mat[0] += r;
                        mat[3] += r;
                        let (log_det, q) = quad(&mat, &dv);
                        base - 0.5 * log_det - 0.5 * q
                    }
                };
                assert!(
                    (got[s] - want).abs() <= 1e-9 * want.abs() + 1e-12,
                    "{covariance:?}, state {s}: {} against {want}",
                    got[s]
                );
            }
        }
    }

    #[test]
    fn the_filter_is_the_longhand_recursion() {
        let (d, k) = (2usize, 2usize);
        let means = vec![-3.0, -3.0, 3.0, 3.0];
        let covs = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let pi = vec![0.9, 0.1, 0.2, 0.8];
        let mut m = Hmm::new(HmmCfg {
            learn: false,
            means: Some(means.clone()),
            covs: Some(covs.clone()),
            transition: Some(pi.clone()),
            transition_prior: 1.0,
            ..cfg(d, k)
        })
        .unwrap();
        assert!(
            m.transition()
                .iter()
                .zip(&pi)
                .all(|(a, b)| (a - b).abs() < 1e-12),
            "a given matrix is the prior mean: {:?}",
            m.transition()
        );
        let ridge = 1e-3;
        let mut p = vec![0.5, 0.5];
        for (x, _) in stream(120, 5, 3) {
            let step = m.step(&x, &[], 1.0, 1.0);
            // Longhand: predict, weight by the density, normalise.
            let pred: Vec<f64> = (0..k)
                .map(|l| (0..k).map(|kk| p[kk] * pi[kk * k + l]).sum())
                .collect();
            let f: Vec<f64> = (0..k)
                .map(|l| {
                    let (mut q, mut ld) = (0.0, 0.0);
                    for i in 0..d {
                        let v = covs[l * d * d + i * d + i] + ridge;
                        let dv = x[i] - means[l * d + i];
                        q += dv * dv / v;
                        ld += v.ln();
                    }
                    (-0.5 * (d as f64 * std::f64::consts::TAU.ln() + ld + q)).exp()
                })
                .collect();
            let z: f64 = (0..k).map(|l| pred[l] * f[l]).sum();
            for c in 0..k {
                assert!((step.pred[c] - p[c]).abs() < 1e-12, "p_{c}");
                assert!((step.pred[k + c] - pred[c]).abs() < 1e-12, "p1_{c}");
            }
            assert!((step.pred[2 * k + 1] - z.ln()).abs() < 1e-9, "loglik");
            p = (0..k).map(|l| pred[l] * f[l] / z).collect();
        }
    }

    /// The chain is learned: a stream that stays in a state gives a
    /// transition matrix with a heavy diagonal.
    #[test]
    fn the_transition_matrix_is_learned_from_the_filtered_joint() {
        let mut m = Hmm::new(HmmCfg {
            warm_rows: 40,
            seed_rule: SeedRule::Lloyd,
            ..cfg(2, 2)
        })
        .unwrap();
        for (x, _) in stream(4000, 11, 100) {
            m.step(&x, &[], 1.0, 1.0);
        }
        let pi = m.transition();
        assert!(pi[0] > 0.9 && pi[3] > 0.9, "{pi:?}");
        assert!(pi[1] < 0.1 && pi[2] < 0.1, "{pi:?}");
        // And the states found the blobs.
        let mut centres: Vec<f64> = (0..2).map(|s| m.state_cov(s).mean(0)).collect();
        centres.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (centres[0] + 3.0).abs() < 0.3 && (centres[1] - 3.0).abs() < 0.3,
            "{centres:?}"
        );
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut m = Hmm::new(cfg(2, 2)).unwrap();
        for (x, _) in stream(200, 13, 5) {
            let want = m.predict(&x, 1.0);
            let before = m.clone();
            let got = m.step(&x, &[], 1.0, 1.0);
            assert_eq!(want.n_eff, got.n_eff);
            assert!(
                want.pred
                    .iter()
                    .zip(&got.pred)
                    .all(|(a, b)| a == b || (a.is_nan() && b.is_nan())),
                "{:?} vs {:?}",
                want.pred,
                got.pred
            );
            let _ = before;
        }
    }

    #[test]
    fn a_zero_weight_first_row_is_legal_and_teaches_nothing() {
        let mut m = Hmm::new(cfg(2, 2)).unwrap();
        let step = m.step(&[1e6, -1e6], &[], 1.0, 0.0);
        assert_eq!(step.n_eff, 0.0);
        assert!(step.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.filtered(), &[0.5, 0.5]);
        assert!(m.states.iter().all(|s| s.n_eff() == 0.0));
        // And it goes on learning.
        for (x, _) in stream(200, 17, 4) {
            m.step(&x, &[], 1.0, 1.0);
        }
        assert!(m.filtered().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_zero_weight_row_mid_stream_moves_nothing_but_the_clock() {
        let mut m = Hmm::new(cfg(2, 2)).unwrap();
        for (x, _) in stream(200, 19, 4) {
            m.step(&x, &[], 1.0, 1.0);
        }
        let p = m.filtered().to_vec();
        let a = m.a.clone();
        m.step(&[1e6, -1e6], &[], 1.0, 0.0);
        assert_eq!(m.filtered(), &p[..]);
        // No decay in this configuration, so the counts do not move either.
        assert_eq!(m.a, a);
    }

    #[test]
    fn nothing_is_reported_before_the_states_are_seeded() {
        let mut m = Hmm::new(HmmCfg {
            warm_rows: 30,
            ..cfg(2, 2)
        })
        .unwrap();
        for (i, (x, _)) in stream(60, 23, 3).into_iter().enumerate() {
            let step = m.step(&x, &[], 1.0, 1.0);
            if i < 30 {
                assert!(step.pred.iter().all(|v| v.is_nan()), "row {i}");
            }
        }
        assert!(m.seeded);
        assert!(
            m.states.iter().all(|s| s.n_eff() > 0.0),
            "seeded from the buffer"
        );
    }

    #[test]
    fn tvtp_reads_the_exogenous_column() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 5.0, 0.0, 0.0];
        let m = Hmm::new(HmmCfg {
            learn: false,
            means: Some(vec![-3.0, -3.0, 3.0, 3.0]),
            covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            tvtp: Some((a, b)),
            ..cfg(2, 2)
        })
        .unwrap();
        // z = 0: both rows uniform. z = 1: row 0 leans hard to state 1.
        let flat = m.transition_at(Some(0.0));
        assert!((flat[0] - 0.5).abs() < 1e-12 && (flat[1] - 0.5).abs() < 1e-12);
        let leaning = m.transition_at(Some(1.0));
        assert!(leaning[1] > 0.99, "{leaning:?}");
        assert!((leaning[2] - 0.5).abs() < 1e-12, "row 1 does not move");
    }

    #[test]
    fn the_labels_and_slot_count_follow_k() {
        assert_eq!(
            Hmm::labels(2),
            [
                "filtered_0",
                "filtered_1",
                "predicted_0",
                "predicted_1",
                "state",
                "loglik"
            ]
        );
        assert_eq!(Hmm::n_outputs_for(2), 6);
        assert_eq!(Hmm::n_outputs_for(4), 10);
    }

    #[test]
    fn a_bad_configuration_is_refused_by_name() {
        let bad = |c: HmmCfg, msg: &str| {
            let e = Hmm::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(HmmCfg { k: 1, ..cfg(2, 1) }, "k must be >= 2");
        bad(
            HmmCfg {
                precision_prior: 0.0,
                ..cfg(2, 2)
            },
            "precision_prior must be finite and > 0",
        );
        bad(
            HmmCfg {
                learn: false,
                ..cfg(2, 2)
            },
            "nothing to filter with",
        );
        bad(
            HmmCfg {
                means: Some(vec![0.0; 4]),
                ..cfg(2, 2)
            },
            "means and covs go together",
        );
        bad(
            HmmCfg {
                transition: Some(vec![0.5, 0.4, 0.5, 0.5]),
                ..cfg(2, 2)
            },
            "not a distribution",
        );
        bad(
            HmmCfg {
                warm_rows: 1,
                ..cfg(2, 2)
            },
            "warm_rows must be at least k",
        );
        // docs/REVIEW-E54-E64.md H4: given states that cannot be filtered
        // with. Two 2x2 blocks, the first one wrong in each case.
        let given = |c: Vec<f64>| HmmCfg {
            means: Some(vec![-1.0, -1.0, 1.0, 1.0]),
            covs: Some(c),
            ..cfg(2, 2)
        };
        bad(
            given(vec![1.0, 0.5, 0.4, 1.0, 1.0, 0.0, 0.0, 1.0]),
            "covs[0] must be symmetric",
        );
        bad(
            given(vec![1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            "covs[0] must be positive definite",
        );
        bad(
            HmmCfg {
                means: Some(vec![f64::NAN, 0.0, 1.0, 1.0]),
                ..given(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            },
            "means and covs must be finite",
        );
    }

    /// An `hmm` has no window, so its states' accumulators keep no runs, as
    /// every other owner without a window keeps none (docs/PLAN.md task 128;
    /// review 2026-09-26, C3).
    #[test]
    fn an_hmm_keeps_no_runs() {
        let m = Hmm::new(cfg(2, 2)).unwrap();
        assert!(m.states.iter().all(|s| !s.keeps_runs()));
    }

    // The mutation survivors of the weekly pass (docs/PLAN.md task 158).

    /// Two states given at `±3` in two columns, unit covariances.
    fn given(c: HmmCfg) -> HmmCfg {
        HmmCfg {
            means: Some(vec![-3.0, -3.0, 3.0, 3.0]),
            covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            ..c
        }
    }

    /// The configuration's edges, each where the docs put it, and the shapes
    /// read at `k = 3`, where `k·k`, `k·d` and `k + k` part.
    #[test]
    fn the_configuration_edges_are_where_the_docs_put_them() {
        let bad = |c: HmmCfg, msg: &str| {
            let e = Hmm::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        for tau in [-1.0, f64::INFINITY] {
            bad(
                HmmCfg {
                    transition_prior: tau,
                    ..cfg(2, 2)
                },
                "transition_prior must be finite and >= 0",
            );
        }
        bad(
            HmmCfg {
                min_weight: -1.0,
                ..cfg(2, 2)
            },
            "min_weight must be >= 0",
        );
        // A row that sums to one with a negative cell is not a distribution.
        bad(
            HmmCfg {
                transition: Some(vec![1.5, -0.5, 0.5, 0.5]),
                ..cfg(2, 2)
            },
            "transition row 0 is not a distribution",
        );
        bad(
            HmmCfg {
                tvtp: Some((vec![0.0; 3], vec![0.0; 4])),
                ..cfg(2, 2)
            },
            "tvtp_coef A and B must each be 2x2",
        );
        bad(
            HmmCfg {
                tvtp: Some((vec![0.0; 4], vec![0.0; 3])),
                ..cfg(2, 2)
            },
            "tvtp_coef A and B must each be 2x2",
        );
        // Three states in two columns, every shape right, a cell of zero.
        let eye = [1.0, 0.0, 0.0, 1.0];
        let m = Hmm::new(HmmCfg {
            learn: false,
            transition: Some(vec![1.0, 0.0, 0.0, 0.2, 0.5, 0.3, 0.0, 0.5, 0.5]),
            means: Some(vec![-3.0, 0.0, 0.0, 0.0, 3.0, 0.0]),
            covs: Some(eye.repeat(3)),
            tvtp: Some((vec![0.1; 9], vec![0.2; 9])),
            ..cfg(2, 3)
        })
        .unwrap();
        assert_eq!(m.transition().len(), 9);
        // `warm_rows` of `k` exactly seeds `k` states.
        let mut m = Hmm::new(HmmCfg {
            warm_rows: 3,
            ..cfg(2, 3)
        })
        .unwrap();
        for x in [[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]] {
            m.step(&x, &[], 1.0, 1.0);
        }
        assert!(m.seeded);
    }

    /// A state covariance is held to symmetry cell by cell, each upper entry
    /// against its lower one, to `1e-12` of the larger plus `1e-12`. Of
    /// three columns, an asymmetry in `[1][2]` alone is found and named; one
    /// at the tolerance exactly is rounding and is taken, as is one under it
    /// at a large scale or a tiny one.
    #[test]
    fn a_state_covariance_is_symmetric_to_a_tolerance() {
        let with = |first: Vec<f64>| {
            let d = if first.len() == 9 { 3 } else { 2 };
            let mut covs = first;
            covs.extend((0..d * d).map(|ij| f64::from(ij / d == ij % d)));
            HmmCfg {
                means: Some((0..2 * d).map(|i| i as f64).collect()),
                covs: Some(covs),
                ..cfg(d, 2)
            }
        };
        let e = Hmm::new(with(vec![4.0, 1.0, 0.5, 1.0, 4.0, 0.9, 0.5, 0.5, 4.0])).unwrap_err();
        assert!(
            e.contains("covs[0] must be symmetric; [1][2] is 0.9 and [2][1] is 0.5"),
            "{e}"
        );
        let mut c = 1e-12f64;
        while 1e-12 * (1.0 + c) != c {
            c = 1e-12 * (1.0 + c);
        }
        assert_eq!((0.0 - c).abs(), 1e-12 * (1.0 + 0.0f64.max(c)));
        Hmm::new(with(vec![1.0, 0.0, c, 1.0])).unwrap();
        Hmm::new(with(vec![1e7, 1e6, 1e6 * (1.0 + 1e-13), 1e7])).unwrap();
        Hmm::new(with(vec![1.0, 0.0, 5e-13, 1.0])).unwrap();
    }

    /// With no prior mass (`transition_prior = 0`), no counts yet and no
    /// `Π₀` given, a row of `Π` has nothing to normalise and is uniform, the
    /// flat prior's mean, and the filter runs.
    #[test]
    fn a_transition_row_with_no_mass_is_uniform() {
        for k in [2, 3] {
            let mut m = Hmm::new(HmmCfg {
                learn: false,
                transition_prior: 0.0,
                means: Some((0..2 * k).map(|i| i as f64).collect()),
                covs: Some([1.0, 0.0, 0.0, 1.0].repeat(k)),
                ..cfg(2, k)
            })
            .unwrap();
            let u = 1.0 / k as f64;
            assert_eq!(m.transition(), vec![u; k * k]);
            let out = m.step(&[0.5, 0.5], &[], 1.0, 1.0);
            assert!(out.pred.iter().all(|v| v.is_finite()), "{out:?}");
        }
    }

    /// With no prior mass (`transition_prior = 0`) and a `Π₀` given, a row
    /// with no counts is `Π₀`: the row `(counts + τ·K·Π₀)/(its sum)` is `Π₀`
    /// at every `τ > 0` before a count, and so its limit at 0. Under `learn =
    /// false` the filter then runs on the matrix it was given, where it ran
    /// on a uniform one and never used `Π₀` (docs/PLAN.md task 158).
    #[test]
    fn a_row_with_no_mass_is_the_given_matrix() {
        let pi0 = vec![0.9, 0.1, 0.2, 0.8];
        let mut m = Hmm::new(given(HmmCfg {
            learn: false,
            transition_prior: 0.0,
            transition: Some(pi0.clone()),
            ..cfg(2, 2)
        }))
        .unwrap();
        assert_eq!(m.transition(), pi0);
        // From the uniform start, p̃ = [1/2, 1/2]·Π₀ = [0.55, 0.45]; a uniform
        // chain would give [0.5, 0.5].
        let out = m.step(&[0.0, 0.0], &[], 1.0, 1.0);
        assert!(
            (out.pred[2] - 0.55).abs() < 1e-12,
            "p̃ {:?}",
            &out.pred[2..4]
        );
        assert!(
            (out.pred[3] - 0.45).abs() < 1e-12,
            "p̃ {:?}",
            &out.pred[2..4]
        );
    }

    /// Under `tvtp` the filter is Hamilton's with `Π(t) = softmaxₗ(Aₖₗ +
    /// Bₖₗ·zₜ)`, `zₜ` the row's exogenous value from `y[0]`, and 0 where it is
    /// missing or not finite: the longhand at fixed states, every row, from
    /// `step` and from `predict_with`.
    #[test]
    fn tvtp_filters_with_the_matrix_of_each_rows_exogenous_value() {
        let (d, k) = (2usize, 2usize);
        let (a, b) = (vec![0.5, -0.3, 0.1, 0.8], vec![0.2, 1.5, -0.7, 0.4]);
        let mut m = Hmm::new(given(HmmCfg {
            learn: false,
            tvtp: Some((a.clone(), b.clone())),
            ..cfg(d, k)
        }))
        .unwrap();
        let ridge = 1e-3;
        let zs = [
            Some(0.3),
            Some(-2.0),
            None,
            Some(1.7),
            Some(f64::NAN),
            Some(0.6),
        ];
        let mut p = vec![0.5, 0.5];
        for (t, (x, _)) in stream(60, 7, 4).into_iter().enumerate() {
            let y = [zs[t % zs.len()]];
            let z = y[0].filter(|v| v.is_finite()).unwrap_or(0.0);
            let pi: Vec<f64> = (0..k * k)
                .map(|rc| {
                    let r = rc / k;
                    let num = (a[rc] + b[rc] * z).exp();
                    let den: f64 = (0..k)
                        .map(|c| (a[r * k + c] + b[r * k + c] * z).exp())
                        .sum();
                    num / den
                })
                .collect();
            let pred: Vec<f64> = (0..k)
                .map(|l| (0..k).map(|kk| p[kk] * pi[kk * k + l]).sum())
                .collect();
            let f: Vec<f64> = (0..k)
                .map(|l| {
                    let c = if l == 0 { -3.0 } else { 3.0 };
                    let q: f64 = x.iter().map(|xi| (xi - c).powi(2) / (1.0 + ridge)).sum();
                    (-0.5
                        * (d as f64 * std::f64::consts::TAU.ln()
                            + d as f64 * (1.0 + ridge).ln()
                            + q))
                        .exp()
                })
                .collect();
            let zsum: f64 = (0..k).map(|l| pred[l] * f[l]).sum();
            let shown = m.predict_with(&x, &y, 1.0).pred;
            let step = m.step(&x, &y, 1.0, 1.0).pred;
            assert_eq!(shown, step, "row {t}");
            for c in 0..k {
                assert!((step[k + c] - pred[c]).abs() < 1e-12, "row {t}: p1_{c}");
            }
            assert!(
                (step[2 * k + 1] - zsum.ln()).abs() < 1e-9,
                "row {t}: loglik"
            );
            p = (0..k).map(|l| pred[l] * f[l] / zsum).collect();
        }
    }

    /// The softmax is taken about the largest logit, so logits in the
    /// hundreds give the distribution they define rather than `inf / inf`.
    #[test]
    fn tvtp_logits_in_the_hundreds_are_a_distribution() {
        let m = Hmm::new(given(HmmCfg {
            learn: false,
            tvtp: Some((vec![1000.0, 999.0, 0.0, 0.0], vec![0.0; 4])),
            ..cfg(2, 2)
        }))
        .unwrap();
        let pi = m.transition_at(Some(0.0));
        let e = (-1.0f64).exp();
        assert!((pi[0] - 1.0 / (1.0 + e)).abs() < 1e-15, "{pi:?}");
        assert!((pi[1] - e / (1.0 + e)).abs() < 1e-15, "{pi:?}");
    }

    /// `shared` pools the states' covariances by their weights, `πₛ = nₛ/Σn`
    /// -- and by `1/K` when no state has any weight, here after a gap the
    /// decay took the whole history in -- and scores every state against the
    /// pool: the Gaussian from the test oracle's eigendecomposition
    /// (`crate::oracle`), nothing shared with the model's Cholesky, from the
    /// model's own states.
    #[test]
    fn shared_pools_the_states_by_their_weights_or_evenly_with_none() {
        let mut m = Hmm::new(given(HmmCfg {
            covariance: Covariance::Shared,
            decay: Decay::Halflife(1.0),
            ..cfg(2, 2)
        }))
        .unwrap();
        // Unequal weights: most rows near the second state.
        for (i, (x, _)) in stream(40, 3, 10).into_iter().enumerate() {
            let w = if i % 10 < 3 { 0.2 } else { 1.0 };
            m.step(&x, &[], 0.1, w);
        }
        let x = [0.4, -0.7];
        let want = |m: &Hmm| -> Vec<f64> {
            let weights: Vec<f64> = m.states.iter().map(EwCov::n_eff).collect();
            let total: f64 = weights.iter().sum();
            let mut pool = vec![0.0; 4];
            for (s, state) in m.states.iter().enumerate() {
                let pi = if total > 0.0 { weights[s] / total } else { 0.5 };
                for (p, c) in pool.iter_mut().zip(state.comoments()) {
                    *p += pi * c;
                }
                pool[0] += pi * m.ridge(s);
                pool[3] += pi * m.ridge(s);
            }
            m.states
                .iter()
                .map(|s| {
                    let delta = [s.deviation(0, x[0]), s.deviation(1, x[1])];
                    crate::oracle::gaussian_log_density(&pool, &delta)
                })
                .collect()
        };
        let w: Vec<f64> = m.states.iter().map(EwCov::n_eff).collect();
        assert!(
            (w[0] - w[1]).abs() > 0.1 * (w[0] + w[1]),
            "the weights are unequal: {w:?}"
        );
        let (got, expect) = (m.log_densities(&x).unwrap(), want(&m));
        for s in 0..2 {
            assert!(
                (got[s] - expect[s]).abs() < 1e-9 * expect[s].abs(),
                "{got:?} vs {expect:?}"
            );
        }
        // A gap of two thousand half-lives: every state's weight is 0.
        m.step(&x, &[], 2000.0, 0.0);
        assert!(m.states.iter().all(|s| s.n_eff() == 0.0));
        let (got, expect) = (m.log_densities(&x).unwrap(), want(&m));
        for s in 0..2 {
            assert!(
                (got[s] - expect[s]).abs() < 1e-9 * expect[s].abs(),
                "{got:?} vs {expect:?}"
            );
        }
        let fails = m.solve_failures;
        let out = m.step(&x, &[], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| v.is_finite()), "{out:?}");
        assert_eq!(m.solve_failures, fails);
    }

    /// The predicted state is the first maximum of `p̃`: on the first row,
    /// with a uniform chain, `p̃` is flat and the state is 0.
    #[test]
    fn a_tie_in_the_predicted_state_reports_the_first() {
        let mut m = Hmm::new(given(HmmCfg {
            learn: false,
            ..cfg(2, 2)
        }))
        .unwrap();
        let out = m.step(&[2.0, 2.0], &[], 1.0, 1.0);
        assert_eq!(out.pred[2], out.pred[3], "p̃ is flat");
        assert_eq!(out.pred[4], 0.0);
    }

    /// The warm-up replay assigns each buffered row to its nearest centre,
    /// the first on a tie: a row midway between the two first rows goes to
    /// the first state.
    #[test]
    fn a_seeding_row_midway_goes_to_the_first_centre() {
        let mut m = Hmm::new(HmmCfg {
            warm_rows: 3,
            seed_rule: SeedRule::First,
            ..cfg(2, 2)
        })
        .unwrap();
        for x in [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]] {
            m.step(&x, &[], 1.0, 1.0);
        }
        assert!(m.seeded);
        assert_eq!(m.state_cov(0).n_eff(), 2.0);
        assert_eq!(m.state_cov(1).n_eff(), 1.0);
    }

    /// A row of no weight, and a row that fails to score, age the transition
    /// counts by the row's decay: one half-life halves them.
    #[test]
    fn a_row_that_learns_nothing_ages_the_counts() {
        for (x, w) in [([0.5, 0.5], 0.0), ([1e300, 1e300], 1.0)] {
            let mut m = Hmm::new(HmmCfg {
                decay: Decay::Halflife(10.0),
                ..cfg(2, 2)
            })
            .unwrap();
            for (row, _) in stream(200, 11, 40) {
                m.step(&row, &[], 1.0, 1.0);
            }
            let (a, fails) = (m.a.clone(), m.solve_failures);
            assert!(a.iter().all(|v| *v > 0.0));
            m.step(&x, &[], 10.0, w);
            let halved: Vec<f64> = a.iter().map(|v| 0.5 * v).collect();
            assert_eq!(m.a, halved, "weight {w}");
            assert_eq!(m.solve_failures, fails + u64::from(w > 0.0), "weight {w}");
        }
    }

    /// The filtered joint `ξ` sums to one over its cells, so a learned row
    /// adds its weight to the counts -- also a row far from every state,
    /// whose densities are each too small to be a double; the reference the
    /// joint is scaled about cancels.
    #[test]
    fn a_row_far_from_every_state_still_adds_its_weight_to_the_counts() {
        let mut m = Hmm::new(given(cfg(2, 2))).unwrap();
        let x = [40.0, 40.0];
        let logf = m.log_densities(&x).unwrap();
        assert!(logf.iter().all(|l| l.exp() == 0.0), "{logf:?}");
        let before: f64 = m.a.iter().sum();
        let out = m.step(&x, &[], 1.0, 0.7);
        assert!(out.pred.iter().all(|v| v.is_finite()), "{out:?}");
        let after: f64 = m.a.iter().sum();
        assert!(
            (after - (before + 0.7)).abs() < 1e-12,
            "{before} -> {after}"
        );
    }

    /// A row whose most likely state the chain gives no way to reach (no
    /// prior mass, and counts that never left the first state) has a
    /// filtered joint of zero everywhere: it teaches the counts nothing,
    /// rather than `0/0`, and the chain goes on.
    #[test]
    fn a_row_the_chain_cannot_reach_teaches_the_counts_nothing() {
        let mut m = Hmm::new(given(HmmCfg {
            transition_prior: 0.0,
            ..cfg(2, 2)
        }))
        .unwrap();
        // Far on the first state's side, lightly: the filter lands on it
        // exactly, and the counts learn only "from either state to it".
        m.step(&[-80.0, -80.0], &[], 1.0, 1e-6);
        assert_eq!(m.filtered(), &[1.0, 0.0]);
        assert_eq!(m.transition(), vec![1.0, 0.0, 1.0, 0.0]);
        // Far on the second state's side: its density is the larger by more
        // than a double spans, and `p̃` gives it nothing.
        let x = [80.0, 80.0];
        let logf = m.log_densities(&x).unwrap();
        assert!((logf[0] - logf[1]).exp() == 0.0, "{logf:?}");
        let a = m.a.clone();
        let out = m.step(&x, &[], 1.0, 1.0);
        assert_eq!(&out.pred[2..4], &[1.0, 0.0], "p̃");
        assert_eq!(m.a, a);
        assert!(m.transition().iter().all(|v| v.is_finite()));
        let next = m.step(&[-3.0, -3.0], &[], 1.0, 1.0);
        assert!(next.pred.iter().all(|v| v.is_finite()), "{next:?}");
    }

    /// Each shape check refuses alone: too few states, a short count
    /// matrix; and the shapes that are right restore, at `k = 3`, and with
    /// rows still buffered for the warm-up. The factor cache is derived, so
    /// a model with its factors built equals itself read back without them.
    #[test]
    fn each_shape_check_refuses_alone_and_the_right_shapes_restore() {
        use crate::{ModelState, StateError};
        let refuse = |edit: &dyn Fn(&mut Hmm)| {
            let mut s = Hmm::new(cfg(2, 2)).unwrap().state();
            let ModelState::Hmm(inner) = &mut s.model else {
                unreachable!()
            };
            edit(inner);
            match Hmm::restore(&s) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
                other => panic!("{other:?}"),
            }
        };
        refuse(&|m| {
            m.states.pop();
        });
        refuse(&|m| {
            m.a.pop();
        });
        let three = Hmm::new(cfg(2, 3)).unwrap();
        assert_eq!(Hmm::restore(&three.state()).unwrap(), three);
        let mut warming = Hmm::new(cfg(2, 2)).unwrap();
        for (x, _) in stream(5, 3, 2) {
            warming.step(&x, &[], 1.0, 1.0);
        }
        assert_eq!(warming.buffer.len(), 5);
        assert_eq!(Hmm::restore(&warming.state()).unwrap(), warming);
        let mut scorer = Hmm::new(given(HmmCfg {
            learn: false,
            ..cfg(2, 2)
        }))
        .unwrap();
        scorer.step(&[0.1, 0.2], &[], 1.0, 1.0);
        assert!(scorer.factors.peek(0).is_some(), "the cache is built");
        let bytes = rmp_serde::to_vec(&scorer.state()).unwrap();
        let back = Hmm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert!(back.factors.peek(0).is_none(), "and not written");
        assert_eq!(back, scorer);
    }

    /// The factor cache is the `full` shape's, built once the states are
    /// seeded (`ensure_factors`): a `diagonal` model builds none, nor does a
    /// `full` one still warming up.
    #[test]
    fn only_a_seeded_full_model_builds_factors() {
        let mut diag = Hmm::new(given(HmmCfg {
            learn: false,
            covariance: Covariance::Diagonal,
            ..cfg(2, 2)
        }))
        .unwrap();
        diag.step(&[0.1, 0.2], &[], 1.0, 1.0);
        assert!(diag.factors.peek(0).is_none(), "diagonal");
        let mut warming = Hmm::new(cfg(2, 2)).unwrap();
        warming.step(&[0.1, 0.2], &[], 1.0, 1.0);
        assert!(!warming.seeded);
        assert!(warming.factors.peek(0).is_none(), "warming up");
        let mut full = Hmm::new(given(HmmCfg {
            learn: false,
            ..cfg(2, 2)
        }))
        .unwrap();
        full.step(&[0.1, 0.2], &[], 1.0, 1.0);
        assert!(full.factors.peek(0).is_some() && full.factors.peek(1).is_some());
    }
}
