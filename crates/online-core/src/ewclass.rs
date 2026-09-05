//! Class-conditional Gaussian classifier on exponentially weighted class
//! moments (docs/ENHANCEMENTS.md E39; docs/PLAN.md §11a, task 27). One
//! [`EwCov`] accumulator per class holds the EW mean and centered co-moments
//! of the rows labelled with it, and a row is scored — before it is learned —
//! by the posterior of every class under a Gaussian with those moments, in
//! one of three covariance shapes: `full` (quadratic discriminant analysis),
//! `shared` (linear discriminant analysis, the classes pooled) or `diagonal`
//! (Gaussian naive Bayes).
//!
//! ```text
//! class c     μ_c, C_c = EW mean and centered co-moments of the rows labelled c
//!             n_c = their EW weight,  π_c = n_c / Σ_c' n_c'
//! ridge       r_c = precision_prior · s_c    (s_c: the class's decaying prior scale)
//! full        M_c = C_c + r_c I
//! shared      M   = Σ_c π_c (C_c + r_c I)    (one factorization scores every class)
//! diagonal    M_c = diag(v_ci + r_c),  v_ci = C_c[i, i]
//! score       ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)    n_c > 0
//!             ℓ_c = −∞                                                          n_c = 0
//! outputs     p_c = softmax(ℓ)_c,   class = argmax_c p_c    (first maximum wins)
//! learn       C_y ← update(x, lam, w);  every other class decays by lam
//!             n_eff ← lam · n_eff + w    on every accepted row, labelled or not
//!             lam = 0.5^(d / halflife)
//! ```
//!
//! Every output is read *before* the row is learned, so `class` and the
//! posteriors are out of sample (CLAUDE.md rule 2), and `n_eff` is the EW
//! weight before the row and before its own decay (rule 8), counting
//! unlabelled rows as `ew_ridge` counts rows without a target: they advance
//! the clock and the feature history, so `min_periods` means the same number
//! of rows here as everywhere else. The class weights `n_c` count only the
//! labelled rows and are what the priors `π_c` are read from; a class no row
//! has carried yet has posterior exactly zero, and until some row has, every
//! output is null.
//!
//! The precision prior is required: a class's centered co-moments start at
//! zero and stay singular until `k` independent rows have been seen, so
//! there is nothing to invert without it. As in `ew_cov`, its scale `s_c`
//! decays with the class's own co-moments, so the prior fades as the class
//! accumulates data and a class seen once is still scorable. The `shared`
//! shape pools the ridges with the co-moments, `Σ_c π_c r_c`, so it is the
//! same regularization the `full` shape would apply, averaged.

use serde::{Deserialize, Serialize};

use crate::clock::Decay;
use crate::ewcov::EwCov;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{SpdFactor, quad_forms_logdet};

/// The shape of the class covariance matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Covariance {
    /// One full matrix per class: quadratic discriminant analysis.
    #[default]
    Full,
    /// One matrix pooled over the classes by their priors: linear
    /// discriminant analysis. The decision boundaries are hyperplanes.
    Shared,
    /// Per-class variances only, the features taken as conditionally
    /// independent: Gaussian naive Bayes.
    Diagonal,
}

impl Covariance {
    pub fn name(self) -> &'static str {
        match self {
            Covariance::Full => "full",
            Covariance::Shared => "shared",
            Covariance::Diagonal => "diagonal",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "full" => Ok(Covariance::Full),
            "shared" => Ok(Covariance::Shared),
            "diagonal" => Ok(Covariance::Diagonal),
            other => Err(format!(
                "unknown ew_class covariance {other:?} (expected full, shared or diagonal)"
            )),
        }
    }
}

/// Configuration for [`EwClass`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwClassCfg {
    pub n_features: usize,
    /// Number of classes, `>= 2`; a label is an index below it.
    pub n_classes: usize,
    pub decay: Decay,
    /// Outputs are null while `n_eff < min_periods`.
    pub min_periods: f64,
    pub covariance: Covariance,
    /// Ridge on every class covariance, finite and `> 0`; decays as the
    /// class accumulates data (see the [module docs](self)).
    pub precision_prior: f64,
}

impl EwClassCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("ew_class: n_features must be >= 1".into());
        }
        if self.n_classes < 2 {
            return Err("ew_class: n_classes must be >= 2".into());
        }
        if !(self.precision_prior.is_finite() && self.precision_prior > 0.0) {
            return Err("ew_class: precision_prior must be finite and > 0".into());
        }
        if self.min_periods.is_nan() || self.min_periods < 0.0 {
            return Err("ew_class: min_periods must be >= 0".into());
        }
        Ok(())
    }
}

/// The Cholesky factor of one class's `C_c + r_c I` for the `full` shape,
/// kept between rows. A class that did not learn the row has the same
/// matrix bit for bit -- [`EwCov::decay`] moves neither the co-moments nor
/// the precision scale -- so its factor is reused and only the class the row
/// belongs to is refactorized: one O(k³) factorization per learned row
/// instead of one per class (docs/PERFORMANCE.md §13). `Failed` is kept
/// too, so a matrix no jitter could factorize is not retried on every row.
#[derive(Debug, Clone)]
enum Cached {
    Stale,
    Failed,
    Ready(SpdFactor),
}

/// Per-class [`Cached`] factors. Derived state: a pure function of the
/// class accumulators, so it is neither serialized (rebuilt on demand after
/// a load) nor compared.
#[derive(Debug, Clone, Default)]
struct Factors(Vec<Cached>);

impl PartialEq for Factors {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Factors {
    /// The factor for class `c`, computing and keeping it when stale.
    fn get(
        &mut self,
        c: usize,
        n_classes: usize,
        matrix: impl FnOnce() -> Vec<f64>,
        k: usize,
    ) -> Option<&SpdFactor> {
        if self.0.len() != n_classes {
            self.0 = vec![Cached::Stale; n_classes];
        }
        if matches!(self.0[c], Cached::Stale) {
            self.0[c] = match SpdFactor::of(&matrix(), k) {
                Some(f) => Cached::Ready(f),
                None => Cached::Failed,
            };
        }
        match &self.0[c] {
            Cached::Ready(f) => Some(f),
            _ => None,
        }
    }

    /// The factor for class `c` if it is ready; `None` when stale or failed,
    /// without computing anything.
    fn peek(&self, c: usize) -> Option<&SpdFactor> {
        match self.0.get(c) {
            Some(Cached::Ready(f)) => Some(f),
            _ => None,
        }
    }

    fn invalidate(&mut self, c: usize) {
        if let Some(slot) = self.0.get_mut(c) {
            *slot = Cached::Stale;
        }
    }
}

/// Class-conditional Gaussian classifier; see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwClass {
    cfg: EwClassCfg,
    /// One accumulator per class, in label order.
    classes: Vec<EwCov>,
    /// EW weight of every accepted row, labelled or not.
    n_eff: f64,
    /// Rows whose covariance could not be factorized even with jitter; the
    /// row's outputs are null.
    pub solve_failures: u64,
    /// The `full` shape's per-class factors between rows; see [`Factors`].
    #[serde(skip)]
    factors: Factors,
}

impl EwClass {
    pub fn new(cfg: EwClassCfg) -> Result<Self, String> {
        cfg.validate()?;
        let classes = (0..cfg.n_classes)
            .map(|_| EwCov::with_precision_prior(cfg.n_features, cfg.precision_prior))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            cfg,
            classes,
            n_eff: 0.0,
            solve_failures: 0,
            factors: Factors::default(),
        })
    }

    pub fn cfg(&self) -> &EwClassCfg {
        &self.cfg
    }

    /// EW weight of every accepted row so far: the `n_eff` the next row reports.
    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// EW weight of the rows labelled with each class, in label order.
    pub fn class_weights(&self) -> Vec<f64> {
        self.classes.iter().map(EwCov::n_eff).collect()
    }

    /// The accumulator of class `c`.
    pub fn class_cov(&self, c: usize) -> &EwCov {
        &self.classes[c]
    }

    /// The class means, one row of `n_features` per class in label order —
    /// what `coef` holds in the stream. A class no row has carried is NaN.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.classes
            .iter()
            .map(|c| {
                if c.n_eff() > 0.0 {
                    c.means().to_vec()
                } else {
                    vec![f64::NAN; self.cfg.n_features]
                }
            })
            .collect()
    }

    /// Output names: `class`, then `p_<name>` per class.
    pub fn labels(classes: &[String]) -> Vec<String> {
        let mut out = Vec::with_capacity(1 + classes.len());
        out.push("class".to_string());
        out.extend(classes.iter().map(|c| format!("p_{c}")));
        out
    }

    /// The class a target value names, if it is a valid label: a finite,
    /// non-negative integer below `n_classes`. Anything else is "no label".
    fn label(&self, y: &[Option<f64>]) -> Option<usize> {
        let v = y.first().copied().flatten()?;
        if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < self.cfg.n_classes as f64 {
            Some(v as usize)
        } else {
            None
        }
    }

    /// The ridge in force on class `c`'s covariance.
    fn ridge(&self, c: usize) -> f64 {
        self.cfg.precision_prior * self.classes[c].precision_scale()
    }

    /// `C_c + r_c I`, the matrix the `full` shape factorizes for class `c`.
    fn class_matrix(&self, c: usize) -> Vec<f64> {
        let k = self.cfg.n_features;
        let mut m = self.classes[c].comoments().to_vec();
        let ridge = self.ridge(c);
        for i in 0..k {
            m[i * k + i] += ridge;
        }
        m
    }

    /// `[class, p_0, .., p_{C-1}]` for `x` from the current state; NaN
    /// throughout before `min_periods`, before any labelled row, on a
    /// non-finite `x`, and when a covariance cannot be factorized.
    ///
    /// `factors` is the `full` shape's cache of class factors: `step` hands
    /// its own in and gets the stale ones rebuilt; `predict` has only `&self`
    /// and reads whatever is ready, factorizing the rest without keeping it.
    fn score(
        &self,
        x: &[f64],
        valid: bool,
        solve_failed: &mut bool,
        mut factors: Option<&mut Factors>,
    ) -> Vec<f64> {
        let nc = self.cfg.n_classes;
        let k = self.cfg.n_features;
        let nan = || vec![f64::NAN; 1 + nc];
        if !valid || self.n_eff < self.cfg.min_periods {
            return nan();
        }
        let weights = self.class_weights();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 || total.is_nan() {
            return nan();
        }
        // ln π_c − ½ ln det M_c − ½ q_c per class; −∞ where n_c = 0.
        let mut ell = vec![f64::NEG_INFINITY; nc];
        match self.cfg.covariance {
            Covariance::Full => {
                let mut delta = vec![0.0; k];
                for c in 0..nc {
                    if weights[c] <= 0.0 {
                        continue;
                    }
                    let cov = &self.classes[c];
                    for (d, (xi, mi)) in delta.iter_mut().zip(x.iter().zip(cov.means())) {
                        *d = xi - mi;
                    }
                    let fresh;
                    let factor = match factors.as_deref_mut() {
                        Some(cache) => cache.get(c, nc, || self.class_matrix(c), k),
                        None => match self.factors.peek(c) {
                            Some(f) => Some(f),
                            None => {
                                fresh = SpdFactor::of(&self.class_matrix(c), k);
                                fresh.as_ref()
                            }
                        },
                    };
                    match factor {
                        Some(f) => {
                            let q = f.quad_forms(&delta, k, 1);
                            ell[c] = (weights[c] / total).ln() - 0.5 * f.log_det() - 0.5 * q[0];
                        }
                        None => {
                            *solve_failed = true;
                            return nan();
                        }
                    }
                }
            }
            Covariance::Shared => {
                // M = Σ_c π_c (C_c + r_c I), then every class's quadratic
                // form against it from one factorization.
                let mut m = vec![0.0; k * k];
                let seen: Vec<usize> = (0..nc).filter(|&c| weights[c] > 0.0).collect();
                for &c in &seen {
                    let pi = weights[c] / total;
                    let ridge = self.ridge(c);
                    for (mij, cij) in m.iter_mut().zip(self.classes[c].comoments()) {
                        *mij += pi * cij;
                    }
                    for i in 0..k {
                        m[i * k + i] += pi * ridge;
                    }
                }
                let mut deltas = vec![0.0; k * seen.len()];
                for (j, &c) in seen.iter().enumerate() {
                    for (i, (xi, mi)) in x.iter().zip(self.classes[c].means()).enumerate() {
                        deltas[j * k + i] = xi - mi;
                    }
                }
                match quad_forms_logdet(&m, &deltas, k, seen.len()) {
                    Some((q, log_det, _)) => {
                        for (j, &c) in seen.iter().enumerate() {
                            ell[c] = (weights[c] / total).ln() - 0.5 * log_det - 0.5 * q[j];
                        }
                    }
                    None => {
                        *solve_failed = true;
                        return nan();
                    }
                }
            }
            Covariance::Diagonal => {
                for c in 0..nc {
                    if weights[c] <= 0.0 {
                        continue;
                    }
                    let cov = &self.classes[c];
                    let ridge = self.ridge(c);
                    let mut log_det = 0.0;
                    let mut q = 0.0;
                    for (i, xi) in x.iter().enumerate() {
                        let v = cov.var(i) + ridge;
                        let d = xi - cov.mean(i);
                        log_det += v.ln();
                        q += d * d / v;
                    }
                    ell[c] = (weights[c] / total).ln() - 0.5 * log_det - 0.5 * q;
                }
            }
        }
        // Softmax about the maximum; the first maximum is the class.
        let mut best = 0;
        for c in 1..nc {
            if ell[c] > ell[best] {
                best = c;
            }
        }
        if !ell[best].is_finite() {
            return nan();
        }
        let top = ell[best];
        let mut out = Vec::with_capacity(1 + nc);
        out.push(best as f64);
        let mut z = 0.0;
        for &l in &ell {
            z += (l - top).exp();
        }
        out.extend(ell.iter().map(|&l| (l - top).exp() / z));
        out
    }
}

impl OnlineModel for EwClass {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let lam = self.cfg.decay.factor(d_clock);
        let n_before = self.n_eff;
        let valid = x.iter().all(|v| v.is_finite());
        let mut failed = false;
        let mut factors = std::mem::take(&mut self.factors);
        let pred = self.score(x, valid, &mut failed, Some(&mut factors));
        self.factors = factors;
        if failed {
            self.solve_failures += 1;
        }
        let learn = valid && weight > 0.0 && weight.is_finite();
        let label = if learn { self.label(y) } else { None };
        for (c, cov) in self.classes.iter_mut().enumerate() {
            if label == Some(c) {
                cov.update(x, lam, weight);
                // The one matrix this row moved.
                self.factors.invalidate(c);
            } else {
                cov.decay(lam);
            }
        }
        self.n_eff = lam * self.n_eff + if learn { weight } else { 0.0 };
        Step {
            pred,
            n_eff: n_before,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        let valid = x.iter().all(|v| v.is_finite());
        let mut failed = false;
        Step {
            pred: self.score(x, valid, &mut failed, None),
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::EwClass(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::EwClass(m) => Ok((**m).clone()),
            other => Err(StateError::WrongModel {
                expected: "ew_class",
                found: other.kind(),
            }),
        }
    }

    /// Zero: the label column is learned from, not predicted as a number.
    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// `class`, then one posterior per class.
    fn n_outputs(&self) -> usize {
        1 + self.cfg.n_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(k: usize, nc: usize, covariance: Covariance) -> EwClassCfg {
        EwClassCfg {
            n_features: k,
            n_classes: nc,
            decay: Decay::Halflife(30.0),
            min_periods: 0.0,
            covariance,
            precision_prior: 0.1,
        }
    }

    struct Row {
        x: Vec<f64>,
        label: Option<usize>,
        d: f64,
        w: f64,
    }

    /// A random stream of `n` rows over `nc` classes with distinct means,
    /// with unlabelled rows, zero weights and uneven clock gaps mixed in.
    fn stream(n: usize, k: usize, nc: usize, seed: u64) -> Vec<Row> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let c = ((lcg(&mut s) + 1.0) * 0.5 * nc as f64).floor() as usize % nc;
                let x: Vec<f64> = (0..k)
                    .map(|j| 2.0 * (c as f64) * (j as f64 + 1.0) + lcg(&mut s))
                    .collect();
                let u = lcg(&mut s);
                let label = if u > 0.6 { None } else { Some(c) };
                let w = if u < -0.8 {
                    0.0
                } else {
                    0.5 + 0.5 * (lcg(&mut s) + 1.0)
                };
                let d = if i == 0 {
                    0.0
                } else {
                    0.5 + (lcg(&mut s) + 1.0)
                };
                Row { x, label, d, w }
            })
            .collect()
    }

    /// Gauss-Jordan inverse and determinant of a small dense matrix.
    fn inverse_det(m: &[f64], k: usize) -> (Vec<f64>, f64) {
        let mut a: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let mut r: Vec<f64> = m[i * k..(i + 1) * k].to_vec();
                r.extend((0..k).map(|j| if i == j { 1.0 } else { 0.0 }));
                r
            })
            .collect();
        let mut det = 1.0;
        for col in 0..k {
            let piv = (col..k)
                .max_by(|&p, &q| a[p][col].abs().partial_cmp(&a[q][col].abs()).unwrap())
                .unwrap();
            if piv != col {
                a.swap(piv, col);
                det = -det;
            }
            let p = a[col][col];
            det *= p;
            for v in a[col].iter_mut() {
                *v /= p;
            }
            for r in 0..k {
                if r != col {
                    let f = a[r][col];
                    let pivot_row = a[col].clone();
                    for (v, pv) in a[r].iter_mut().zip(&pivot_row) {
                        *v -= f * pv;
                    }
                }
            }
        }
        let inv = (0..k).flat_map(|i| a[i][k..].to_vec()).collect();
        (inv, det)
    }

    /// The posteriors at `x` from scratch: explicit decayed weights per row,
    /// the per-class prior scale by its own recursion, and the log-density
    /// through a Gauss-Jordan inverse. Shares no code with the model.
    fn oracle(rows: &[Row], upto: usize, cfg: &EwClassCfg, x: &[f64]) -> Option<Vec<f64>> {
        let k = cfg.n_features;
        let nc = cfg.n_classes;
        let lams: Vec<f64> = rows[..upto].iter().map(|r| cfg.decay.factor(r.d)).collect();
        // n_eff over every accepted row (weight 0 rows learn nothing).
        let mut n_eff = 0.0;
        for (r, &lam) in rows[..upto].iter().zip(&lams) {
            n_eff = lam * n_eff + if r.w > 0.0 { r.w } else { 0.0 };
        }
        if n_eff < cfg.min_periods {
            return None;
        }
        // Per-class moments from the explicit weights.
        let mut n = vec![0.0; nc];
        let mut mu = vec![vec![0.0; k]; nc];
        let mut cov = vec![vec![0.0; k * k]; nc];
        let mut scale = vec![1.0; nc];
        let mut n_run = vec![0.0; nc];
        for (i, r) in rows[..upto].iter().enumerate() {
            let lam = lams[i];
            for c in 0..nc {
                let is_row = r.label == Some(c) && r.w > 0.0;
                let w_new = lam * n_run[c] + if is_row { r.w } else { 0.0 };
                if is_row && w_new > 0.0 {
                    let a = lam * n_run[c] / w_new;
                    scale[c] = if a <= 0.0 { 1.0 } else { scale[c] * a };
                }
                n_run[c] = w_new;
            }
            if let Some(c) = r.label {
                if r.w > 0.0 {
                    let wt: f64 = r.w * lams[i + 1..].iter().product::<f64>();
                    n[c] += wt;
                    for (m, xj) in mu[c].iter_mut().zip(&r.x) {
                        *m += wt * xj;
                    }
                }
            }
        }
        for c in 0..nc {
            if n[c] > 0.0 {
                for m in mu[c].iter_mut() {
                    *m /= n[c];
                }
            }
        }
        for (i, r) in rows[..upto].iter().enumerate() {
            if let Some(c) = r.label {
                if r.w > 0.0 {
                    let wt: f64 = r.w * lams[i + 1..].iter().product::<f64>();
                    for a in 0..k {
                        for b in 0..k {
                            cov[c][a * k + b] += wt * (r.x[a] - mu[c][a]) * (r.x[b] - mu[c][b]);
                        }
                    }
                }
            }
        }
        for c in 0..nc {
            if n[c] > 0.0 {
                for v in cov[c].iter_mut() {
                    *v /= n[c];
                }
            }
        }
        let total: f64 = n.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let pooled: Vec<f64> = (0..k * k)
            .map(|ij| {
                (0..nc)
                    .filter(|&c| n[c] > 0.0)
                    .map(|c| {
                        n[c] / total
                            * (cov[c][ij]
                                + if ij / k == ij % k {
                                    cfg.precision_prior * scale[c]
                                } else {
                                    0.0
                                })
                    })
                    .sum()
            })
            .collect();
        let mut ell = vec![f64::NEG_INFINITY; nc];
        for c in 0..nc {
            if n[c] <= 0.0 {
                continue;
            }
            let ridge = cfg.precision_prior * scale[c];
            let m: Vec<f64> = match cfg.covariance {
                Covariance::Full => (0..k * k)
                    .map(|ij| cov[c][ij] + if ij / k == ij % k { ridge } else { 0.0 })
                    .collect(),
                Covariance::Shared => pooled.clone(),
                Covariance::Diagonal => (0..k * k)
                    .map(|ij| {
                        if ij / k == ij % k {
                            cov[c][ij] + ridge
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            };
            let (inv, det) = inverse_det(&m, k);
            let d: Vec<f64> = (0..k).map(|j| x[j] - mu[c][j]).collect();
            let mut q = 0.0;
            for a in 0..k {
                for b in 0..k {
                    q += d[a] * inv[a * k + b] * d[b];
                }
            }
            ell[c] = (n[c] / total).ln() - 0.5 * det.ln() - 0.5 * q;
        }
        let top = ell.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = ell.iter().map(|l| (l - top).exp()).sum();
        let best = (0..nc).find(|&c| ell[c] == top).unwrap();
        let mut out = vec![best as f64];
        out.extend(ell.iter().map(|l| (l - top).exp() / z));
        Some(out)
    }

    /// Bitwise equality, so NaN slots compare equal to NaN slots.
    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(p, q)| (p.is_nan() && q.is_nan()) || (p - q).abs() <= tol * (1.0 + q.abs()))
    }

    #[test]
    fn every_shape_matches_the_from_scratch_oracle() {
        for shape in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
            let cfg = cfg(3, 3, shape);
            let rows = stream(400, 3, 3, 7);
            let mut m = EwClass::new(cfg.clone()).unwrap();
            for (i, r) in rows.iter().enumerate() {
                let y = r.label.map(|c| c as f64);
                let step = m.step(&r.x, &[y], r.d, r.w);
                let want = oracle(&rows, i, &cfg, &r.x).unwrap_or(vec![f64::NAN; 4]);
                assert!(
                    close(&step.pred, &want, 1e-9),
                    "{shape:?} row {i}: {:?} vs oracle {:?}",
                    step.pred,
                    want
                );
                assert_eq!(step.pred.len(), 4);
                if step.pred[0].is_finite() {
                    let s: f64 = step.pred[1..].iter().sum();
                    assert!((s - 1.0).abs() < 1e-12, "{shape:?} row {i}: sum {s}");
                }
            }
            assert_eq!(m.solve_failures, 0);
        }
    }

    #[test]
    fn class_weights_and_means_follow_the_labelled_rows_only() {
        let cfg = cfg(2, 2, Covariance::Full);
        let rows = stream(200, 2, 2, 11);
        let mut m = EwClass::new(cfg.clone()).unwrap();
        let mut n = [0.0f64; 2];
        let mut n_eff = 0.0;
        for r in &rows {
            let lam = cfg.decay.factor(r.d);
            assert!((m.n_eff() - n_eff).abs() < 1e-12);
            let cw = m.class_weights();
            assert!((cw[0] - n[0]).abs() < 1e-12 && (cw[1] - n[1]).abs() < 1e-12);
            m.step(&r.x, &[r.label.map(|c| c as f64)], r.d, r.w);
            for (c, nc) in n.iter_mut().enumerate() {
                *nc = lam * *nc + if r.label == Some(c) { r.w } else { 0.0 };
            }
            n_eff = lam * n_eff + r.w;
        }
        // Unlabelled rows counted in n_eff but in no class.
        assert!(n_eff > n[0] + n[1]);
        let coef = m.coefficients();
        assert_eq!(coef.len(), 2);
        assert_eq!(coef[0], m.class_cov(0).means());
        assert_eq!(coef[1], m.class_cov(1).means());
    }

    #[test]
    fn shared_equals_full_when_the_classes_share_their_moments() {
        // Class 1 sees every class-0 row shifted by a constant, one tick
        // later, so the two centered co-moments and prior scales coincide
        // exactly and QDA is LDA.
        let mut full = EwClass::new(cfg(2, 2, Covariance::Full)).unwrap();
        let mut shared = EwClass::new(cfg(2, 2, Covariance::Shared)).unwrap();
        let mut s = 3u64;
        for i in 0..300 {
            let x = [lcg(&mut s), 0.5 * lcg(&mut s) + 0.3 * lcg(&mut s)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            full.step(&x, &[Some(0.0)], d, 1.0);
            shared.step(&x, &[Some(0.0)], d, 1.0);
            let x1 = [x[0] + 1.5, x[1] - 0.7];
            full.step(&x1, &[Some(1.0)], 1.0, 1.0);
            shared.step(&x1, &[Some(1.0)], 1.0, 1.0);
        }
        assert!(close(
            full.class_cov(0).comoments(),
            full.class_cov(1).comoments(),
            1e-12
        ));
        for _ in 0..50 {
            let x = [2.0 * lcg(&mut s), 2.0 * lcg(&mut s)];
            let a = full.predict(&x, 1.0).pred;
            let b = shared.predict(&x, 1.0).pred;
            assert!(close(&a, &b, 1e-9), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn diagonal_equals_full_when_the_features_are_uncorrelated() {
        // Symmetric rows with no decay: the centered co-moments are exactly
        // diagonal, so naive Bayes is QDA.
        let mut c = cfg(2, 2, Covariance::Full);
        c.decay = Decay::Halflife(f64::INFINITY);
        let mut full = EwClass::new(c.clone()).unwrap();
        c.covariance = Covariance::Diagonal;
        let mut diag = EwClass::new(c).unwrap();
        let pattern = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        for (label, shift) in [(0.0, 0.0), (1.0, 2.0)] {
            for (i, p) in pattern.iter().enumerate() {
                let x = [p[0] + shift, 3.0 * p[1] + shift];
                let d = if i == 0 && label == 0.0 { 0.0 } else { 1.0 };
                full.step(&x, &[Some(label)], d, 1.0);
                diag.step(&x, &[Some(label)], d, 1.0);
            }
        }
        for c in 0..2 {
            assert_eq!(full.class_cov(c).cov(0, 1), 0.0);
        }
        let mut s = 5u64;
        for _ in 0..50 {
            let x = [3.0 * lcg(&mut s), 3.0 * lcg(&mut s)];
            let a = full.predict(&x, 1.0).pred;
            let b = diag.predict(&x, 1.0).pred;
            assert!(close(&a, &b, 1e-12), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn a_class_no_row_has_carried_has_posterior_zero_and_none_is_null() {
        let mut m = EwClass::new(cfg(2, 3, Covariance::Full)).unwrap();
        // Nothing learned: null.
        let p = m.step(&[0.0, 0.0], &[Some(0.0)], 0.0, 1.0).pred;
        assert!(p.iter().all(|v| v.is_nan()));
        // One class seen: it takes everything.
        let p = m.step(&[0.1, 0.1], &[Some(2.0)], 1.0, 1.0).pred;
        assert_eq!(p, vec![0.0, 1.0, 0.0, 0.0]);
        // Two classes seen; class 1 has never been labelled.
        let p = m.step(&[5.0, 5.0], &[None], 1.0, 1.0).pred;
        assert_eq!(p[2], 0.0);
        assert!(p[1] > 0.0 && p[3] > 0.0 && ((p[1] + p[3]) - 1.0).abs() < 1e-15);
        assert_eq!(p[0], if p[1] >= p[3] { 0.0 } else { 2.0 });
        assert_eq!(m.class_weights()[1], 0.0);
        assert!(m.coefficients()[1].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn unlabelled_rows_score_and_decay_but_do_not_learn() {
        let cfg = cfg(2, 2, Covariance::Full);
        let mut a = EwClass::new(cfg.clone()).unwrap();
        let mut b = EwClass::new(cfg.clone()).unwrap();
        for i in 0..20 {
            let x = [i as f64, -(i as f64)];
            let y = Some((i % 2) as f64);
            a.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            b.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let lam = cfg.decay.factor(1.0);
        let before = a.class_weights();
        // The unlabelled forms: no target, a null target, a non-integer, a
        // negative one, one past the last class, and non-finite ones.
        let unlabelled: Vec<Vec<Option<f64>>> = vec![
            vec![],
            vec![None],
            vec![Some(0.5)],
            vec![Some(-1.0)],
            vec![Some(2.0)],
            vec![Some(f64::NAN)],
            vec![Some(f64::INFINITY)],
        ];
        let scored = a.predict(&[3.0, -3.0], 1.0).pred;
        let mut ticks = 0;
        for y in &unlabelled {
            let s = a.step(&[3.0, -3.0], y, 1.0, 1.0);
            if ticks == 0 {
                assert_eq!(s.pred, scored);
            }
            ticks += 1;
            let w = a.class_weights();
            assert!((w[0] - before[0] * lam.powi(ticks)).abs() < 1e-12);
            assert!((w[1] - before[1] * lam.powi(ticks)).abs() < 1e-12);
            assert_eq!(a.class_cov(0).means(), b.class_cov(0).means());
            assert_eq!(a.class_cov(0).comoments(), b.class_cov(0).comoments());
        }
        // n_eff counted every one of them; a zero-weight row only ticks.
        let n = a.n_eff();
        let s = a.step(&[3.0, -3.0], &[Some(1.0)], 1.0, 0.0);
        assert_eq!(s.n_eff, n);
        assert_eq!(a.n_eff(), lam * n);
        assert!((a.class_weights()[1] - before[1] * lam.powi(ticks + 1)).abs() < 1e-12);
        assert_eq!(a.class_cov(1).means(), b.class_cov(1).means());
    }

    #[test]
    fn a_zero_weight_first_row_is_legal() {
        let mut m = EwClass::new(cfg(2, 2, Covariance::Full)).unwrap();
        let s = m.step(&[1.0, 2.0], &[Some(0.0)], 0.0, 0.0);
        assert_eq!(s.n_eff, 0.0);
        assert!(s.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.n_eff(), 0.0);
        assert_eq!(m.class_weights(), vec![0.0, 0.0]);
        // The next rows learn as if it never happened.
        m.step(&[1.0, 2.0], &[Some(0.0)], 1.0, 1.0);
        m.step(&[-1.0, -2.0], &[Some(1.0)], 1.0, 1.0);
        let p = m.predict(&[1.0, 2.0], 1.0).pred;
        assert_eq!(p[0], 0.0);
        assert!(p[1] > p[2] && p.iter().all(|v| v.is_finite()));
        assert_eq!(m.class_cov(0).means(), &[1.0, 2.0]);
    }

    #[test]
    fn a_non_finite_feature_row_ticks_the_clock_and_learns_nothing() {
        let cfg = cfg(2, 2, Covariance::Diagonal);
        let mut m = EwClass::new(cfg.clone()).unwrap();
        for i in 0..10 {
            let x = [i as f64, 1.0];
            m.step(
                &x,
                &[Some((i % 2) as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.clone();
        let s = m.step(&[f64::NAN, 1.0], &[Some(0.0)], 1.0, 1.0);
        assert!(s.pred.iter().all(|v| v.is_nan()));
        let lam = cfg.decay.factor(1.0);
        assert_eq!(m.n_eff(), lam * before.n_eff());
        assert_eq!(m.class_weights()[0], lam * before.class_weights()[0]);
        assert_eq!(m.class_cov(0).means(), before.class_cov(0).means());
    }

    #[test]
    fn min_periods_holds_every_output_back() {
        let mut c = cfg(2, 2, Covariance::Full);
        c.min_periods = 3.0;
        let mut m = EwClass::new(c).unwrap();
        let mut seen_nan = 0;
        for i in 0..6 {
            let s = m.step(
                &[i as f64, 1.0],
                &[Some((i % 2) as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            if s.n_eff < 3.0 {
                assert!(s.pred.iter().all(|v| v.is_nan()), "row {i}");
                seen_nan += 1;
            } else {
                assert!(s.pred.iter().all(|v| v.is_finite()), "row {i}");
            }
        }
        // n_eff before rows 0..3 is 0, 1, 1 + lam, 1 + lam + lam² < 3.
        assert_eq!(seen_nan, 4);
    }

    #[test]
    fn predict_is_the_step_and_state_round_trips() {
        for shape in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
            let rows = stream(150, 2, 3, 19);
            let mut m = EwClass::new(cfg(2, 3, shape)).unwrap();
            for (i, r) in rows.iter().enumerate() {
                let y = [r.label.map(|c| c as f64)];
                let p = m.predict(&r.x, r.d).pred;
                let s = m.step(&r.x, &y, r.d, r.w);
                assert!(
                    same_bits(&p, &s.pred),
                    "{shape:?} row {i}: {p:?} vs {:?}",
                    s.pred
                );
                if i == 75 {
                    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                    let back = EwClass::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                    assert_eq!(back, m);
                }
            }
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let mut back = EwClass::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            let more = stream(20, 2, 3, 23);
            for r in &more {
                let y = [r.label.map(|c| c as f64)];
                let (a, b) = (m.step(&r.x, &y, 1.0, r.w), back.step(&r.x, &y, 1.0, r.w));
                assert!(same_bits(&a.pred, &b.pred) && a.n_eff == b.n_eff);
            }
        }
    }

    #[test]
    fn restoring_the_wrong_model_names_both() {
        let m = EwClass::new(cfg(2, 2, Covariance::Full)).unwrap();
        let s = m.state();
        match crate::EwCovModel::restore(&s) {
            Err(StateError::WrongModel { expected, found }) => {
                assert_eq!(expected, "ew_cov");
                assert_eq!(found, "ew_class");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn well_separated_classes_are_recovered() {
        for shape in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
            let mut m = EwClass::new(cfg(2, 3, shape)).unwrap();
            let mut s = 29u64;
            let mut right = 0;
            let mut total = 0;
            for i in 0..3000 {
                let c = (i * 7 % 3) as usize;
                let x = [
                    6.0 * c as f64 + lcg(&mut s),
                    -3.0 * c as f64 + 0.5 * lcg(&mut s),
                ];
                let st = m.step(&x, &[Some(c as f64)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                if i >= 100 {
                    total += 1;
                    if st.pred[0] == c as f64 {
                        right += 1;
                    }
                    assert!(st.pred[1 + c] > 0.9, "{shape:?} row {i}: {:?}", st.pred);
                }
            }
            assert_eq!(right, total, "{shape:?}");
        }
    }

    #[test]
    fn a_shifted_class_is_relearned_within_a_few_halflives() {
        let mut m = EwClass::new(cfg(1, 2, Covariance::Full)).unwrap();
        let mut s = 31u64;
        let mut row = |m: &mut EwClass, c: usize, mean: f64, d: f64| {
            let x = [mean + 0.3 * lcg(&mut s)];
            m.step(&x, &[Some(c as f64)], d, 1.0).pred
        };
        for i in 0..600 {
            let c = i % 2;
            row(
                &mut m,
                c,
                if c == 0 { 0.0 } else { 4.0 },
                if i == 0 { 0.0 } else { 1.0 },
            );
        }
        // Class 0 jumps to 8: the far side of class 1.
        let mut first_right = None;
        for i in 0..600 {
            let c = i % 2;
            let p = row(&mut m, c, if c == 0 { 8.0 } else { 4.0 }, 1.0);
            if c == 0 && p[0] == 0.0 && first_right.is_none() {
                first_right = Some(i);
            }
        }
        // 30-row halflife: the old mean is outweighed well inside 5 halflives.
        let at = first_right.expect("class 0 never relearned");
        assert!(at < 150, "relearned at row {at}");
        let p = m.predict(&[8.0], 1.0).pred;
        assert!(p[0] == 0.0 && p[1] > 0.99, "{p:?}");
    }

    #[test]
    fn configuration_is_validated() {
        let ok = cfg(2, 2, Covariance::Full);
        for (bad, msg) in [
            (
                EwClassCfg {
                    n_features: 0,
                    ..ok.clone()
                },
                "n_features",
            ),
            (
                EwClassCfg {
                    n_classes: 1,
                    ..ok.clone()
                },
                "n_classes",
            ),
            (
                EwClassCfg {
                    precision_prior: 0.0,
                    ..ok.clone()
                },
                "precision_prior",
            ),
            (
                EwClassCfg {
                    precision_prior: f64::NAN,
                    ..ok.clone()
                },
                "precision_prior",
            ),
            (
                EwClassCfg {
                    min_periods: -1.0,
                    ..ok.clone()
                },
                "min_periods",
            ),
        ] {
            let err = EwClass::new(bad).unwrap_err();
            assert!(err.contains(msg), "{err}");
        }
        assert_eq!(Covariance::parse("shared").unwrap(), Covariance::Shared);
        assert!(
            Covariance::parse("spherical")
                .unwrap_err()
                .contains("spherical")
        );
        assert_eq!(Covariance::Diagonal.name(), "diagonal");
        assert_eq!(
            EwClass::labels(&["a".into(), "b".into()]),
            vec!["class", "p_a", "p_b"]
        );
    }
}
