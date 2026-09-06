//! `deco`: dynamic equicorrelation (Engle & Kelly 2012), `O(m)` a row.
//!
//! A correlation matrix of `n` series has `n(n−1)/2` free entries, and a
//! stream cannot keep them all moving without `O(n²)` a row. DECO replaces
//! them with **one** number, the average pairwise correlation, and estimates
//! it from a scalar recursion. `blocks` generalises that to one number per
//! block and one per pair of blocks — still `O(n + K²)` a row.
//!
//! # The recursion
//!
//! The row is standardised against the *pre-row* moments of an [`EwDiag`],
//! `r_i = (x_i − m_i)/√v_i`, and the diag is then updated with it. From
//! `S₁ = Σᵢ rᵢ` and `S₂ = Σᵢ rᵢ²` over the `n` features, the row's
//! equicorrelation estimate is their Lemma 2.3:
//!
//! ```text
//! u = (S₁² − S₂) / ((n − 1)·S₂)         = mean_{i≠j} rᵢrⱼ / mean_i rᵢ²
//! ```
//!
//! in `(−1/(n−1), 1)`. With blocks it is the same ratio, restricted:
//!
//! ```text
//! u_A  = (S₁ₐ² − S₂ₐ) / ((n_A − 1)·S₂ₐ)                    (within block A)
//! u_AB = S₁ₐ·S₁_B / √(n_A·n_B·S₂ₐ·S₂_B)                    (between A and B)
//! ```
//!
//! The level `ρ` then follows one of two dynamics, on the model's clock with
//! decay factor `λ` and row weight `w`:
//!
//! ```text
//! "ew":      W' = λW + w,  b = w/W'
//!            ρ' = ρ + b·(u − ρ)                  (ρ = u at W = 0)
//! "linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ     (ρ̄ the "ew" recursion)
//! ```
//!
//! Two departures from the paper, deliberate (docs/PLAN.md §11a). Their
//! eq. 21 has a free intercept `ω`, and they apply correlation targeting to
//! the DECO-DCC `Q` recursion rather than to the linear one; writing the
//! intercept as `(1 − α − β)·ρ̄` is our reparameterisation, chosen because it
//! removes a free parameter that a streaming model has no sample to fit.
//! And the paper lets `α + β` sit slightly above 1 under numerical bounds,
//! where `α + β < 1` here is the stricter, stationary choice.
//!
//! The paper also notes that `uₜ` is a *downward biased* estimate of the
//! equicorrelation, and gives an alternative `u^var = 1 − (1/(n−1))·Σ(rᵢ −
//! r̄)²`. This model offers the Lemma 2.3 form only.
//!
//! # The log-likelihood
//!
//! `loglik` is the row's Gaussian log-density in standardised coordinates
//! under the **pre-row** `ρ`, `−½[n·ln 2π + ln det R + r'R⁻¹r]`. With one
//! block that is the closed form
//!
//! ```text
//! det R   = (1 − ρ)^{n−1}·(1 + (n−1)ρ)
//! r'R⁻¹r  = (S₂ − ρ·S₁²/(1 + (n−1)ρ)) / (1 − ρ)
//! ```
//!
//! and with `K` blocks it is the same thing through Woodbury and the matrix
//! determinant lemma on `R = D + U P U'` — `D = diag(1 − ρ_{A(i)A(i)})`, `U`
//! the `n × K` block indicator, `P` the `K × K` matrix of block correlations
//! — which is a `K × K` factorization rather than an `n × n` one:
//!
//! ```text
//! d_A     = 1 − ρ_AA        N_A = n_A / d_A       Q = N^{1/2} P N^{1/2}
//! S       = I + Q                                 y_A = S₁ₐ / √(n_A·d_A)
//! ln det R = Σ_A n_A·ln d_A + ln det S
//! r'R⁻¹r   = Σ_A S₂ₐ/d_A − y'y + y'S⁻¹y
//! ```
//!
//! (The one-block case of that *is* the closed form above; a unit test says
//! so.) `ρ` is clamped into `(−1/(n−1) + 1e-9, 1 − 1e-9)` **for the density
//! only** — the emitted `ρ` is never clamped.

use serde::{Deserialize, Serialize};

use crate::{Decay, EwDiag, SpdFactor};

/// How the equicorrelation level moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoDynamics {
    /// `ρ' = ρ + b·(u − ρ)`, the exponentially weighted mean of the row
    /// estimates, on the model's own clock.
    Ew,
    /// `ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ` (Engle–Kelly eq. 21 with
    /// correlation targeting).
    Linear,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoCfg {
    pub n_features: usize,
    pub decay: Decay,
    pub dynamics: DecoDynamics,
    /// Weight on the row estimate under [`DecoDynamics::Linear`].
    pub alpha: Option<f64>,
    /// Weight on the previous level under [`DecoDynamics::Linear`].
    pub beta: Option<f64>,
    /// Feature indices per block, in emission order. Empty means one block
    /// holding every feature, which is the unblocked model.
    pub blocks: Vec<Vec<usize>>,
    pub min_periods: f64,
}

impl DecoCfg {
    /// The blocks as the model uses them: the configured ones, or one block
    /// of every feature.
    fn resolved_blocks(&self) -> Vec<Vec<usize>> {
        if self.blocks.is_empty() {
            vec![(0..self.n_features).collect()]
        } else {
            self.blocks.clone()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features < 2 {
            return Err(format!(
                "deco: at least two columns are required (got {}); an equicorrelation is an \
                 average over pairs",
                self.n_features
            ));
        }
        let blocks = self.resolved_blocks();
        let mut seen = vec![false; self.n_features];
        for (bi, b) in blocks.iter().enumerate() {
            if b.len() < 2 {
                return Err(format!(
                    "deco: block {bi} has {} column(s); a block has a within-block correlation, \
                     which needs at least two",
                    b.len()
                ));
            }
            for &i in b {
                if i >= self.n_features {
                    return Err(format!(
                        "deco: block {bi} names column {i}, and there are {}",
                        self.n_features
                    ));
                }
                if std::mem::replace(&mut seen[i], true) {
                    return Err(format!("deco: column {i} is in more than one block"));
                }
            }
        }
        if let Some(i) = seen.iter().position(|s| !s) {
            return Err(format!(
                "deco: column {i} is in no block; every column must be in exactly one"
            ));
        }
        match self.dynamics {
            DecoDynamics::Ew => {
                if self.alpha.is_some() || self.beta.is_some() {
                    return Err(
                        "deco: alpha/beta belong to dynamics = \"linear\"; the \"ew\" recursion \
                         takes its weights from the decay"
                            .into(),
                    );
                }
            }
            DecoDynamics::Linear => {
                let (Some(a), Some(b)) = (self.alpha, self.beta) else {
                    return Err("deco: dynamics = \"linear\" needs both alpha and beta".into());
                };
                if !(a >= 0.0 && b >= 0.0) {
                    return Err("deco: alpha and beta must be >= 0".into());
                }
                // NaN fails the `>= 0` check above, so this is a plain
                // comparison by the time it runs.
                if a + b >= 1.0 {
                    return Err(format!(
                        "deco: alpha + beta must be < 1 (got {}); the intercept is \
                         (1 - alpha - beta) times the targeted level",
                        a + b
                    ));
                }
            }
        }
        if self.min_periods < 0.0 || self.min_periods.is_nan() {
            return Err("deco: min_periods must be >= 0".into());
        }
        Ok(())
    }
}

/// Engle–Kelly dynamic equicorrelation; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deco {
    cfg: DecoCfg,
    /// The standardiser: pre-row means and variances of the features.
    diag: EwDiag,
    /// Blocks as used, with the unblocked case resolved to one block.
    blocks: Vec<Vec<usize>>,
    /// The level per value, in emission order (within-block, then pairs);
    /// NaN before the first update.
    rho: Vec<f64>,
    /// The `"ew"` recursion run alongside as `"linear"`'s target; equal to
    /// `rho` under `"ew"`, and kept there too so the state is one shape.
    rho_bar: Vec<f64>,
    /// Accumulated weight behind `rho_bar`. Not the diag's: a row whose `u`
    /// is not finite teaches the level nothing, and must not decay it.
    rho_w: f64,
}

impl Deco {
    pub fn new(cfg: DecoCfg) -> Result<Self, String> {
        cfg.validate()?;
        let blocks = cfg.resolved_blocks();
        let m = n_values(blocks.len());
        let diag = EwDiag::new(cfg.n_features);
        Ok(Self {
            cfg,
            diag,
            blocks,
            rho: vec![f64::NAN; m],
            rho_bar: vec![f64::NAN; m],
            rho_w: 0.0,
        })
    }

    pub fn cfg(&self) -> &DecoCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.diag.n_eff()
    }

    /// The correlation values in emission order: one per block, then one per
    /// ordered pair of blocks. NaN before the first update.
    pub fn rho(&self) -> &[f64] {
        &self.rho
    }

    /// Output slot labels, in emission order. With no named blocks it is
    /// `u`, `rho`, `loglik`; with `K` named blocks -- one included, which is
    /// the unblocked model under a name -- it is `u_<A>` per block then
    /// `u_<A>_<B>` per pair, the same again for `rho`, then one `loglik`.
    pub fn labels(block_names: &[String]) -> Vec<String> {
        if block_names.is_empty() {
            return vec!["u".into(), "rho".into(), "loglik".into()];
        }
        let mut out = Vec::new();
        for prefix in ["u", "rho"] {
            for a in block_names {
                out.push(format!("{prefix}_{a}"));
            }
            for i in 0..block_names.len() {
                for j in (i + 1)..block_names.len() {
                    out.push(format!("{prefix}_{}_{}", block_names[i], block_names[j]));
                }
            }
        }
        out.push("loglik".into());
        out
    }

    /// Slots for `K` blocks: `u` and `rho` per correlation value plus one
    /// `loglik`. One block gives the three scalars, named or not.
    pub fn n_outputs_for(n_blocks: usize) -> usize {
        2 * n_values(n_blocks) + 1
    }

    /// `(S₁, S₂)` per block from the standardised row, and whether every
    /// entry was finite.
    fn block_sums(&self, r: &[f64]) -> (Vec<f64>, Vec<f64>, bool) {
        let mut s1 = vec![0.0; self.blocks.len()];
        let mut s2 = vec![0.0; self.blocks.len()];
        let mut ok = true;
        for (bi, b) in self.blocks.iter().enumerate() {
            for &i in b {
                let ri = r[i];
                ok &= ri.is_finite();
                s1[bi] += ri;
                s2[bi] += ri * ri;
            }
        }
        (s1, s2, ok)
    }

    /// The row's equicorrelation estimates, in emission order. NaN entries
    /// where the row says nothing (a zero-variance feature, an all-zero
    /// block).
    fn row_u(&self, s1: &[f64], s2: &[f64]) -> Vec<f64> {
        let k = self.blocks.len();
        let mut out = Vec::with_capacity(n_values(k));
        for (bi, b) in self.blocks.iter().enumerate() {
            let n = b.len() as f64;
            out.push(if s2[bi] > 0.0 {
                (s1[bi] * s1[bi] - s2[bi]) / ((n - 1.0) * s2[bi])
            } else {
                f64::NAN
            });
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let d = (self.blocks[i].len() as f64 * self.blocks[j].len() as f64 * s2[i] * s2[j])
                    .sqrt();
                out.push(if d > 0.0 { s1[i] * s1[j] / d } else { f64::NAN });
            }
        }
        out
    }

    /// `−½[n·ln 2π + ln det R + r'R⁻¹r]` for the row, under `rho`. See the
    /// module docs for the block algebra; `NaN` when `rho` is, or when the
    /// `K × K` factorization fails.
    fn loglik(&self, rho: &[f64], s1: &[f64], s2: &[f64]) -> f64 {
        let k = self.blocks.len();
        if rho.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        // Clamped for the density only: outside these bounds `R` is not a
        // correlation matrix and the density has no value to report.
        let clamp = |value: f64, n: f64| {
            let lo = -1.0 / (n - 1.0) + 1e-9;
            value.clamp(lo, 1.0 - 1e-9)
        };
        let n_total: f64 = self.blocks.iter().map(|b| b.len() as f64).sum();
        let mut log_det = 0.0;
        let mut quad = 0.0;
        let mut y = vec![0.0; k];
        let mut sqrt_n = vec![0.0; k];
        for (bi, b) in self.blocks.iter().enumerate() {
            let n = b.len() as f64;
            let d = 1.0 - clamp(rho[bi], n);
            log_det += n * d.ln();
            quad += s2[bi] / d;
            y[bi] = s1[bi] / (n * d).sqrt();
            sqrt_n[bi] = (n / d).sqrt();
        }
        // `S = I + N^{1/2} P N^{1/2}`, symmetric and positive definite when
        // `R` is; one `K x K` Cholesky where a dense form would need `n x n`.
        let mut s = vec![0.0; k * k];
        for i in 0..k {
            s[i * k + i] = 1.0 + sqrt_n[i] * clamp(rho[i], self.blocks[i].len() as f64) * sqrt_n[i];
        }
        let mut p = k;
        for i in 0..k {
            for j in (i + 1)..k {
                // Between-block values follow the within-block ones.
                let v = clamp(rho[p], 2.0);
                p += 1;
                s[i * k + j] = sqrt_n[i] * v * sqrt_n[j];
                s[j * k + i] = s[i * k + j];
            }
        }
        let Some(f) = SpdFactor::of(&s, k) else {
            return f64::NAN;
        };
        let yy: f64 = y.iter().map(|v| v * v).sum();
        let sinv = f.quad_forms(&y, k, 1)[0];
        log_det += f.log_det();
        quad += sinv - yy;
        // TAU is 2 pi, so `TAU.ln()` is the `ln 2 pi` of the density.
        -0.5 * (n_total * std::f64::consts::TAU.ln() + log_det + quad)
    }

    /// The row standardised against the pre-row moments.
    fn standardise(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        for (i, xi) in x.iter().enumerate() {
            let v = self.diag.var(i);
            out.push(if v > 0.0 {
                (xi - self.diag.mean(i)) / v.sqrt()
            } else {
                f64::NAN
            });
        }
    }

    /// The row's outputs, from the state as it stands.
    fn read(&self, x: &[f64]) -> Vec<f64> {
        let mut r = Vec::with_capacity(x.len());
        self.standardise(x, &mut r);
        let (s1, s2, ok) = self.block_sums(&r);
        let m = self.rho.len();
        let u = if ok {
            self.row_u(&s1, &s2)
        } else {
            vec![f64::NAN; m]
        };
        let loglik = if ok {
            self.loglik(&self.rho, &s1, &s2)
        } else {
            f64::NAN
        };
        let mut out = Vec::with_capacity(Self::n_outputs_for(self.blocks.len()));
        out.extend_from_slice(&u);
        out.extend_from_slice(&self.rho);
        out.push(loglik);
        out
    }
}

/// Correlation values for `k` blocks: one per block, one per pair.
fn n_values(k: usize) -> usize {
    k + k * (k - 1) / 2
}

impl crate::OnlineModel for Deco {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        // Read before the update: `u`, `rho` and `loglik` are all as of the
        // state before this row, which is what makes them usable as features
        // for the same row.
        let out = self.predict(x, d_clock);
        let lam = self.cfg.decay.factor(d_clock);
        let mut r = Vec::with_capacity(x.len());
        self.standardise(x, &mut r);
        let (s1, s2, ok) = self.block_sums(&r);
        if ok && weight >= 0.0 {
            let u = self.row_u(&s1, &s2);
            if u.iter().all(|v| v.is_finite()) {
                self.advance(&u, lam, weight);
            }
        }
        self.diag.update(x, lam, weight);
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> crate::Step {
        let n_eff = self.diag.n_eff();
        let pred = if n_eff >= self.cfg.min_periods {
            self.read(x)
        } else {
            vec![f64::NAN; Self::n_outputs_for(self.blocks.len())]
        };
        crate::Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::Deco(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::Deco(m) => Ok((**m).clone()),
            other => Err(crate::StateError::WrongModel {
                expected: "deco",
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
        Self::n_outputs_for(self.blocks.len())
    }
}

impl Deco {
    /// Move the level by one row's estimate.
    ///
    /// Written as `EwCov::update` writes its mean -- `ρ + b·(u − ρ)`, not the
    /// algebraically equal `a·ρ + b·u` -- so that `rho` under `"ew"` is
    /// **bit-identical** to an `ew_cov(stats = ["mean"])` over the same `u`
    /// sequence at the same halflife. The two forms differ in the last bit,
    /// and a test pins this one.
    fn advance(&mut self, u: &[f64], lam: f64, w: f64) {
        let w_new = lam * self.rho_w + w;
        if w_new <= 0.0 {
            // A zero-weight row at zero weight: advance the clock, learn
            // nothing, and do not divide 0/0 (hard rule 9).
            return;
        }
        let b = w / w_new;
        for (m, &um) in u.iter().enumerate() {
            let bar = &mut self.rho_bar[m];
            *bar = if bar.is_finite() {
                *bar + b * (um - *bar)
            } else {
                um
            };
            self.rho[m] = match self.cfg.dynamics {
                DecoDynamics::Ew => *bar,
                DecoDynamics::Linear => {
                    let (alpha, beta) = (
                        self.cfg.alpha.expect("validated"),
                        self.cfg.beta.expect("validated"),
                    );
                    if self.rho[m].is_finite() {
                        (1.0 - alpha - beta) * *bar + alpha * um + beta * self.rho[m]
                    } else {
                        um
                    }
                }
            };
        }
        self.rho_w = w_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnlineModel;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(n: usize) -> DecoCfg {
        DecoCfg {
            n_features: n,
            decay: Decay::Halflife(20.0),
            dynamics: DecoDynamics::Ew,
            alpha: None,
            beta: None,
            blocks: Vec::new(),
            min_periods: 0.0,
        }
    }

    /// A correlated stream: one common factor plus idiosyncratic noise, so
    /// the true equicorrelation is `rho`.
    fn stream(n: usize, k: usize, rho: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let a = rho.sqrt();
        let b = (1.0 - rho).sqrt();
        (0..n)
            .map(|_| {
                let f = lcg(&mut s);
                (0..k).map(|_| a * f + b * lcg(&mut s)).collect()
            })
            .collect()
    }

    /// NaN is not equal to itself, and these slots are NaN whenever the row
    /// says nothing; two outputs are "the same" when every slot is equal or
    /// both NaN.
    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    /// The longhand pair sum: `u` is the mean off-diagonal product over the
    /// mean squared entry, which is what Lemma 2.3's closed form is.
    fn u_longhand(r: &[f64]) -> f64 {
        let n = r.len();
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += r[i] * r[j];
                }
            }
        }
        let s2: f64 = r.iter().map(|v| v * v).sum();
        off / ((n as f64 - 1.0) * s2)
    }

    #[test]
    fn u_is_the_longhand_pair_sum() {
        let mut m = Deco::new(cfg(5)).unwrap();
        let rows = stream(200, 5, 0.4, 7);
        for x in &rows {
            let before = m.clone();
            let step = m.step(x, &[], 1.0, 1.0);
            // The standardisation is the pre-row diag's, so redo it here.
            let r: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, xi)| (xi - before.diag.mean(i)) / before.diag.var(i).sqrt())
                .collect();
            if r.iter().all(|v| v.is_finite()) {
                assert!(
                    (step.pred[0] - u_longhand(&r)).abs() < 1e-12,
                    "{} vs {}",
                    step.pred[0],
                    u_longhand(&r)
                );
            }
        }
    }

    /// `u` stays inside `(−1/(n−1), 1)` and orders with the truth. It does
    /// **not** converge to the true equicorrelation: `E[u]` is `0.20` for a
    /// true `0.30` at `k = 6` and `0.60` for a true `0.80`, because `u` is a
    /// ratio of two averages and `E[A/B] != E[A]/E[B]`. The paper says as
    /// much (`uₜ` is downward biased), so the test holds the estimator to
    /// its own expectation -- measured by Monte Carlo, and reproduced here
    /// to 0.02 -- rather than to a number it does not estimate.
    #[test]
    fn u_is_inside_its_bounds_and_orders_with_the_truth() {
        let k = 6;
        let mut levels = Vec::new();
        for (rho, want) in [(0.0, 0.0), (0.3, 0.20), (0.8, 0.60)] {
            let mut m = Deco::new(DecoCfg {
                decay: Decay::Halflife(400.0),
                ..cfg(k)
            })
            .unwrap();
            let mut last = f64::NAN;
            for x in stream(6000, k, rho, 11) {
                let step = m.step(&x, &[], 1.0, 1.0);
                let u = step.pred[0];
                if u.is_finite() {
                    assert!(u > -1.0 / (k as f64 - 1.0) - 1e-12 && u < 1.0, "{u}");
                }
                last = step.pred[1];
            }
            assert!(
                (last - want).abs() < 0.02,
                "rho {last} for a true {rho} (E[u] is about {want})"
            );
            levels.push(last);
        }
        assert!(levels[0] < levels[1] && levels[1] < levels[2], "{levels:?}");
    }

    /// One block of everything is the unblocked model, and the block
    /// log-likelihood path must reproduce the closed form exactly.
    #[test]
    fn the_block_loglik_reduces_to_the_closed_form() {
        let k = 5;
        let m = Deco::new(cfg(k)).unwrap();
        let s1 = 1.7;
        let s2 = 4.25;
        for rho in [-0.1, 0.0, 0.25, 0.9] {
            let got = m.loglik(&[rho], &[s1], &[s2]);
            let n = k as f64;
            let det = (n - 1.0) * (1.0 - rho).ln() + (1.0 + (n - 1.0) * rho).ln();
            let quad = (s2 - rho * s1 * s1 / (1.0 + (n - 1.0) * rho)) / (1.0 - rho);
            let want = -0.5 * (n * std::f64::consts::TAU.ln() + det + quad);
            assert!((got - want).abs() < 1e-10, "rho {rho}: {got} vs {want}");
        }
    }

    /// The Woodbury path against a dense `n x n` density, built from the
    /// block correlations by hand.
    #[test]
    fn the_block_loglik_is_the_dense_gaussian_density() {
        let blocks = vec![vec![0usize, 1, 2], vec![3, 4]];
        let m = Deco::new(DecoCfg {
            blocks: blocks.clone(),
            ..cfg(5)
        })
        .unwrap();
        let rho = [0.5, 0.2, 0.35];
        let r = [0.4, -1.1, 0.7, 0.25, -0.6];
        let (s1, s2, ok) = m.block_sums(&r);
        assert!(ok);
        let got = m.loglik(&rho, &s1, &s2);

        let n = 5;
        let of = |i: usize| if blocks[0].contains(&i) { 0 } else { 1 };
        let mut dense = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dense[i * n + j] = if i == j {
                    1.0
                } else if of(i) == of(j) {
                    rho[of(i)]
                } else {
                    rho[2]
                };
            }
        }
        let f = SpdFactor::of(&dense, n).unwrap();
        let quad = f.quad_forms(&r, n, 1)[0];
        let want = -0.5 * (n as f64 * std::f64::consts::TAU.ln() + f.log_det() + quad);
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }

    /// One block holding every feature reproduces the unblocked `u`.
    #[test]
    fn one_block_of_everything_is_the_unblocked_model() {
        let rows = stream(300, 4, 0.5, 3);
        let mut plain = Deco::new(cfg(4)).unwrap();
        let mut blocked = Deco::new(DecoCfg {
            blocks: vec![vec![0, 1, 2, 3]],
            ..cfg(4)
        })
        .unwrap();
        for x in &rows {
            let a = plain.step(x, &[], 1.0, 1.0);
            let b = blocked.step(x, &[], 1.0, 1.0);
            assert!(same(&a.pred, &b.pred), "{:?} vs {:?}", a.pred, b.pred);
            assert_eq!(a.n_eff, b.n_eff);
        }
    }

    /// Hard rule 9: a zero-weight first row advances the clock and learns
    /// nothing, and nothing divides 0/0.
    #[test]
    fn a_zero_weight_first_row_is_legal() {
        let mut m = Deco::new(cfg(3)).unwrap();
        let step = m.step(&[1.0, 2.0, 3.0], &[], 1.0, 0.0);
        assert_eq!(step.n_eff, 0.0);
        assert!(step.pred.iter().all(|v| v.is_nan()));
        assert_eq!(m.n_eff(), 0.0);
        assert!(m.rho.iter().all(|v| v.is_nan()));
        // And it keeps learning afterwards.
        for x in stream(100, 3, 0.3, 5) {
            m.step(&x, &[], 1.0, 1.0);
        }
        assert!(m.rho[0].is_finite());
    }

    #[test]
    fn a_zero_weight_row_mid_stream_moves_nothing_but_the_clock() {
        let mut m = Deco::new(cfg(3)).unwrap();
        for x in stream(50, 3, 0.3, 5) {
            m.step(&x, &[], 1.0, 1.0);
        }
        let before = m.rho.clone();
        m.step(&[9.0, -9.0, 9.0], &[], 1.0, 0.0);
        assert_eq!(m.rho, before);
    }

    #[test]
    fn the_linear_dynamics_is_its_recursion() {
        let (alpha, beta) = (0.05, 0.9);
        let mut m = Deco::new(DecoCfg {
            dynamics: DecoDynamics::Linear,
            alpha: Some(alpha),
            beta: Some(beta),
            ..cfg(4)
        })
        .unwrap();
        let mut bar = f64::NAN;
        let mut rho = f64::NAN;
        let mut w = 0.0;
        let lam = Decay::Halflife(20.0).factor(1.0);
        for x in stream(200, 4, 0.4, 13) {
            let before = m.clone();
            let step = m.step(&x, &[], 1.0, 1.0);
            assert!(step.pred[1].is_nan() == rho.is_nan());
            if step.pred[1].is_finite() {
                assert!((step.pred[1] - rho).abs() < 1e-12);
            }
            let u = step.pred[0];
            if u.is_finite() {
                let w_new = lam * w + 1.0;
                let b = 1.0 / w_new;
                bar = if bar.is_finite() {
                    bar + b * (u - bar)
                } else {
                    u
                };
                rho = if rho.is_finite() {
                    (1.0 - alpha - beta) * bar + alpha * u + beta * rho
                } else {
                    u
                };
                w = w_new;
            }
            let _ = before;
        }
        assert!(rho.is_finite());
    }

    #[test]
    fn predict_is_the_step_without_the_step() {
        let mut m = Deco::new(cfg(4)).unwrap();
        for x in stream(120, 4, 0.4, 17) {
            let want = m.predict(&x, 1.0);
            let before = m.clone();
            let got = m.step(&x, &[], 1.0, 1.0);
            assert_eq!(want.n_eff, got.n_eff);
            assert!(same(&want.pred, &got.pred));
            assert_ne!(before, m);
        }
    }

    #[test]
    fn a_bad_configuration_is_refused_by_name() {
        let bad = |c: DecoCfg, msg: &str| {
            let e = Deco::new(c).unwrap_err();
            assert!(e.contains(msg), "{e}");
        };
        bad(cfg(1), "at least two columns");
        bad(
            DecoCfg {
                blocks: vec![vec![0], vec![1, 2]],
                ..cfg(3)
            },
            "needs at least two",
        );
        bad(
            DecoCfg {
                blocks: vec![vec![0, 1]],
                ..cfg(3)
            },
            "column 2 is in no block",
        );
        bad(
            DecoCfg {
                blocks: vec![vec![0, 1], vec![1, 2]],
                ..cfg(3)
            },
            "in more than one block",
        );
        bad(
            DecoCfg {
                dynamics: DecoDynamics::Linear,
                ..cfg(3)
            },
            "needs both alpha and beta",
        );
        bad(
            DecoCfg {
                dynamics: DecoDynamics::Linear,
                alpha: Some(0.5),
                beta: Some(0.6),
                ..cfg(3)
            },
            "alpha + beta must be < 1",
        );
        bad(
            DecoCfg {
                alpha: Some(0.5),
                ..cfg(3)
            },
            "alpha/beta belong to",
        );
    }

    #[test]
    fn the_labels_follow_the_blocks() {
        assert_eq!(Deco::labels(&[]), ["u", "rho", "loglik"]);
        // One named block is the unblocked model under a name: the same
        // three slots, carrying the name the caller gave.
        assert_eq!(
            Deco::labels(&["all".to_string()]),
            ["u_all", "rho_all", "loglik"]
        );
        let named = ["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(
            Deco::labels(&named),
            [
                "u_a", "u_b", "u_c", "u_a_b", "u_a_c", "u_b_c", "rho_a", "rho_b", "rho_c",
                "rho_a_b", "rho_a_c", "rho_b_c", "loglik",
            ]
        );
        assert_eq!(Deco::n_outputs_for(1), 3);
        assert_eq!(Deco::n_outputs_for(3), 13);
    }
}
