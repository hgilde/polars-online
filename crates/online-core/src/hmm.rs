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
//! never-visited row of `Π` a distribution. **Not** the EW mean of
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
    pub min_periods: f64,
    /// Update the states and the transition counts. `false` filters with
    /// what it was given and learns nothing.
    pub learn: bool,
    /// Dirichlet pseudo-count per cell of the transition matrix.
    pub transition_prior: f64,
    /// A `K*K` row-stochastic matrix to seed the counts at `τ·K·Π₀`, so it
    /// is the prior mean; `None` is uniform.
    pub transition: Option<Vec<f64>>,
    /// State means, `K*d` row-major: given, there is no warm-up. The pair
    /// enters the accumulators at **weight 1** -- one row's worth -- so the
    /// given states are a starting point that the stream washes out under
    /// `learn = true`, and are held exactly under `learn = false`.
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
        if self.min_periods.is_nan() || self.min_periods < 0.0 {
            return Err("hmm: min_periods must be >= 0".into());
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
        if let Some((a, b)) = &self.tvtp {
            if a.len() != k * k || b.len() != k * k {
                return Err(format!("hmm: tvtp_coef A and B must each be {k}x{k}"));
            }
        }
        Ok(())
    }
}

/// What [`Hmm::read`] hands the update: the filtered posterior after the
/// row, and the per-state log densities the transition counts need.
type Update = (Vec<f64>, Vec<f64>);

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
}

impl Hmm {
    pub fn new(cfg: HmmCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (k, d) = (cfg.k, cfg.n_features);
        let mut states = (0..k)
            .map(|_| EwCov::with_precision_prior(d, cfg.precision_prior))
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
                )?;
            }
        }
        Ok(Self {
            p: vec![1.0 / k as f64; k],
            a: vec![0.0; k * k],
            n_eff: 0.0,
            buffer: Vec::new(),
            seeded,
            solve_failures: 0,
            states,
            cfg,
        })
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
                out[r * k + c] = if z > 0.0 { v / z } else { 1.0 / k as f64 };
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
                        let dv = xi - cov.mean(i);
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
                    for (i, (xi, mi)) in x.iter().zip(state.means()).enumerate() {
                        deltas[s * d + i] = xi - mi;
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
                    for (dv, (xi, mi)) in delta.iter_mut().zip(x.iter().zip(cov.means())) {
                        *dv = xi - mi;
                    }
                    let f = SpdFactor::of(&self.state_matrix(s), d)?;
                    *o = base - 0.5 * f.log_det() - 0.5 * f.quad_forms(&delta, d, 1)[0];
                }
            }
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// Output slot labels in emission order: `p_<k>`, `p1_<k>`, `state`,
    /// `loglik`.
    pub fn labels(k: usize) -> Vec<String> {
        let mut out: Vec<String> = (0..k).map(|s| format!("p_{s}")).collect();
        out.extend((0..k).map(|s| format!("p1_{s}")));
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
        // `min_periods` gates what is *reported*, never what is learned: a
        // gated row still moves the filter, as it does in every other model
        // here. Returning `None` instead withheld the row from the update
        // and counted it as a solve failure, so under decay a filter whose
        // `n_eff` plateaued below `min_periods` never learned at all
        // (docs/REVIEW-E54-E64.md H1).
        if self.n_eff < self.cfg.min_periods {
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
    }
}

impl crate::OnlineModel for Hmm {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        let (pred, extra) = self.read(x, self.exog_of(_y));
        let out = crate::Step {
            pred,
            n_eff: self.n_eff,
            extra: None,
        };
        let lam = self.cfg.decay.factor(d_clock);
        if weight <= 0.0 {
            // Advance the clock and learn nothing: the counts age with the
            // accumulators, `n_eff` decays with them -- hard rule 8, the
            // same recursion in every model, which is what makes
            // `min_periods` mean the same number of rows across a bank
            // (docs/REVIEW-E54-E64.md H3) -- and `p` does not move.
            self.n_eff *= lam;
            if self.cfg.learn {
                self.a.iter_mut().for_each(|v| *v *= lam);
                for s in self.states.iter_mut() {
                    s.update(x, lam, 0.0);
                }
            }
            return out;
        }
        self.n_eff = lam * self.n_eff + weight;
        if !self.seeded {
            self.buffer.push((x.to_vec(), weight));
            if self.buffer.len() >= self.cfg.warm_rows {
                self.seed();
            }
            return out;
        }
        let Some((post, logf)) = extra else {
            self.solve_failures += 1;
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
            crate::ModelState::Hmm(m) => Ok((**m).clone()),
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
    use crate::{EwClass, EwClassCfg, OnlineModel};

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
            min_periods: 0.0,
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

    /// With a uniform `Π` and `learn = false`, the filter is the classifier:
    /// `p̃` is flat, so `p ∝ f` -- exactly what `ew_class`'s posterior is
    /// when its classes carry equal weight.
    ///
    /// Not bit-identical: `ew_class` adds `ln π_c` to each log density
    /// before the softmax and this does not, and adding then subtracting a
    /// constant inside a softmax is not the identity in f64. `1e-12` is what
    /// the two orders of operations agree to.
    #[test]
    fn a_uniform_chain_is_the_classifier() {
        let d = 2;
        let mut cls = EwClass::new(EwClassCfg {
            n_features: d,
            n_classes: 2,
            decay: Decay::Halflife(f64::INFINITY),
            min_periods: 0.0,
            covariance: Covariance::Full,
            precision_prior: 1e-3,
        })
        .unwrap();
        let rows = stream(200, 3, 1);
        for (x, g) in &rows {
            cls.step(x, &[Some(*g as f64)], 1.0, 1.0);
        }
        // Exactly balanced classes: `ln π_c` is then the same f64 for both,
        // so the only difference left is the softmax's own arithmetic.
        assert_eq!(cls.class_weights()[0], cls.class_weights()[1]);
        let means: Vec<f64> = (0..2)
            .flat_map(|c| cls.class_cov(c).means().to_vec())
            .collect();
        let covs: Vec<f64> = (0..2)
            .flat_map(|c| cls.class_cov(c).comoments().to_vec())
            .collect();
        let mut hmm = Hmm::new(HmmCfg {
            learn: false,
            means: Some(means),
            covs: Some(covs),
            ..cfg(d, 2)
        })
        .unwrap();
        let mut checked = 0;
        for (x, g) in &rows {
            let a = cls.step(x, &[Some(*g as f64)], 1.0, 1.0);
            let b = hmm.step(x, &[], 1.0, 1.0);
            // `ew_class`: [class, p_0, p_1]. `hmm`: [p_0, p_1, p1_0, p1_1,
            // state, loglik], and its `p` is the posterior *before* the row,
            // so compare against the classifier's on the next row -- or,
            // more directly, against `p̃ f` normalised, which is what `p1`
            // and the densities give. The filtered posterior after the row
            // is what the next row reports.
            let _ = b;
            for (c, &got) in hmm.filtered().iter().enumerate() {
                assert!(
                    (got - a.pred[1 + c]).abs() < 1e-12,
                    "state {c}: {got} vs {}",
                    a.pred[1 + c]
                );
            }
            checked += 1;
        }
        assert!(checked > 190);
    }

    /// A longhand Hamilton filter at fixed parameters.
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
            ["p_0", "p_1", "p1_0", "p1_1", "state", "loglik"]
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
}
