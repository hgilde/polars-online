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
//! # A column with no spread
//!
//! A column whose variance is exactly zero -- one constant from its first
//! row, the only way an EW variance is exactly zero -- has no standardised
//! value, and is left out of its block's sums, so `n_A` above counts the
//! columns that have one (docs/PLAN.md task 115 (h)). Its block reads the
//! correlation among its other columns, and so does a pair it is in; a
//! block left with fewer than two has no `u_A` on the row, and a pair with
//! an empty side no `u_AB`. Each correlation value keeps its own weight
//! `W`: a value with no estimate on a row learns nothing and does not
//! decay, while the others learn. So while a column is flat, its block reads
//! what the model without that column reads, to the bit. `loglik` needs
//! every column, and is NaN on such a row. Until the decision, a row with
//! any such column taught no value anything.
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
//! only** — the emitted `ρ` is never clamped. Near that edge `R⁻¹` is large,
//! and a row far past the standardiser's spread can put the density below
//! the double's range: `loglik` is then NaN, no reading, as `hmm`'s is.

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
    /// Accumulated weight behind each `rho_bar`, one per value. Not the
    /// diag's: a row whose `u` for a value is not finite teaches that value
    /// nothing, and must not decay it. One weight served every value before
    /// task 115 (h); a state written then holds a number, which
    /// `weight_or_weights` reads as a list of one and `restore` widens.
    #[serde(deserialize_with = "weight_or_weights")]
    rho_w: Vec<f64>,
    /// What each `rho_bar` leaves out: the level is a pair no step is
    /// rounded off, as an `ew_cov`'s mean is ([`crate::comp`]; docs/PLAN.md
    /// task 101), so the identity `Deco::advance` states holds. Empty in a
    /// state written before it.
    #[serde(default)]
    rho_bar_lo: Vec<f64>,
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
            rho_w: vec![0.0; m],
            rho_bar_lo: vec![0.0; m],
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

    /// `(S₁, S₂, n)` per block from the standardised row, over the entries
    /// that are finite, and whether every entry was: a column with no spread
    /// is left out of its block (see the module docs).
    fn block_sums(&self, r: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
        let k = self.blocks.len();
        let (mut s1, mut s2, mut n) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        let mut ok = true;
        for (bi, b) in self.blocks.iter().enumerate() {
            for &i in b {
                let ri = r[i];
                if ri.is_finite() {
                    s1[bi] += ri;
                    s2[bi] += ri * ri;
                    n[bi] += 1.0;
                } else {
                    ok = false;
                }
            }
        }
        (s1, s2, n, ok)
    }

    /// The row's equicorrelation estimates, in emission order, from each
    /// block's sums over the `n` columns that have a standardised value. NaN
    /// where the row says nothing: a block with fewer than two such columns,
    /// a pair with an empty side, an all-zero block.
    fn row_u(&self, s1: &[f64], s2: &[f64], n: &[f64]) -> Vec<f64> {
        let k = self.blocks.len();
        let mut out = Vec::with_capacity(n_values(k));
        for bi in 0..k {
            out.push(if n[bi] >= 2.0 && s2[bi] > 0.0 {
                (s1[bi] * s1[bi] - s2[bi]) / ((n[bi] - 1.0) * s2[bi])
            } else {
                f64::NAN
            });
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let d = (n[i] * n[j] * s2[i] * s2[j]).sqrt();
                out.push(if d > 0.0 { s1[i] * s1[j] / d } else { f64::NAN });
            }
        }
        out
    }

    /// `−½[n·ln 2π + ln det R + r'R⁻¹r]` for the row, under `rho`. See the
    /// module docs for the block algebra; `NaN` when `rho` is, when the
    /// `K × K` factorization fails, or when the density is past the
    /// double's range.
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
        let ll = -0.5 * (n_total * std::f64::consts::TAU.ln() + log_det + quad);
        // A density below the double's range is no reading, as `hmm`'s is: a
        // row far past the spread under a `ρ` clamped to the edge of
        // singular overflowed the quadratic form and said `-inf` (review
        // 2026-09-27, G4).
        if ll.is_finite() { ll } else { f64::NAN }
    }

    /// The row standardised against the pre-row moments.
    fn standardise(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        for (i, xi) in x.iter().enumerate() {
            let v = self.diag.var(i);
            out.push(if v > 0.0 {
                self.diag.deviation(i, *xi) / v.sqrt()
            } else {
                f64::NAN
            });
        }
    }

    /// The row's outputs, from the state as it stands.
    fn read(&self, x: &[f64]) -> Vec<f64> {
        let mut r = Vec::with_capacity(x.len());
        self.standardise(x, &mut r);
        let (s1, s2, n, ok) = self.block_sums(&r);
        let u = self.row_u(&s1, &s2, &n);
        // The density is over every column, so a row with one that has no
        // standardised value has none (task 115 (h)).
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
        let (s1, s2, n, _) = self.block_sums(&r);
        if weight >= 0.0 {
            // Each value learns from its own estimate, where the row has one
            // (task 115 (h)).
            self.advance(&self.row_u(&s1, &s2, &n), lam, weight);
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
            crate::ModelState::Deco(m) => {
                let mut m = (**m).clone();
                // `blocks` is read from the state and indexes the features
                // in `block_sums`, so it must be the cfg's own; the two
                // correlation vectors are one per block pair (review
                // 2026-09-18, B3).
                let values = n_values(m.blocks.len());
                // One weight served every value before task 115 (h).
                if m.rho_w.len() == 1 && values > 1 {
                    m.rho_w = vec![m.rho_w[0]; values];
                }
                if m.diag.k() != m.cfg.n_features
                    || m.blocks != m.cfg.resolved_blocks()
                    || m.rho.len() != values
                    || m.rho_bar.len() != values
                    || m.rho_w.len() != values
                {
                    return Err(crate::StateError::Invalid(
                        "deco: the state has the wrong shape".into(),
                    ));
                }
                Ok(m)
            }
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
    /// Move each level by the row's estimate for it. A value whose estimate
    /// is not finite learns nothing and keeps its weight undecayed, as the
    /// whole row did before each value had its own (task 115 (h)).
    ///
    /// Written as `EwCov::update` writes its mean -- `ρ + b·(u − ρ)` as a
    /// pair ([`crate::comp`]), not the algebraically equal `a·ρ + b·u` -- so
    /// that `rho` under `"ew"` is
    /// **bit-identical** to an `ew_cov(stats = ["mean"])` over the same `u`
    /// sequence at the same halflife. The two forms differ in the last bit,
    /// and a test pins this one.
    fn advance(&mut self, u: &[f64], lam: f64, w: f64) {
        let n = self.rho_bar.len();
        for (m, &um) in u.iter().enumerate() {
            if !um.is_finite() {
                continue;
            }
            let w_old = self.rho_w[m];
            let w_new = lam * w_old + w;
            if w_new <= 0.0 && w_old <= 0.0 {
                // A zero-weight row at zero weight: advance the clock, learn
                // nothing, and do not divide 0/0 (hard rule 9).
                continue;
            }
            // A zero-weight row after a decay that took the whole history --
            // `lam·W` is 0 from 1075 halflives on -- is the row one halflife
            // short of it: `b` is 0 and the weight goes to 0, where the
            // history used to be kept whole (task 115 (c), PLAN §12).
            let b = if w_new > 0.0 { w / w_new } else { 0.0 };
            let bar = &mut self.rho_bar[m];
            let lo = crate::comp::lo_slot(&mut self.rho_bar_lo, n, m);
            if bar.is_finite() {
                // A row of weight 0 takes no step (`crate::comp::add` says
                // why).
                if b > 0.0 {
                    let d = crate::comp::dev(um, *bar, *lo);
                    crate::comp::add(bar, lo, b * d);
                }
            } else {
                (*bar, *lo) = (um, 0.0);
            }
            self.rho[m] = match self.cfg.dynamics {
                DecoDynamics::Ew => *bar,
                DecoDynamics::Linear => {
                    let (alpha, beta) = (
                        self.cfg.alpha.expect("validated"),
                        self.cfg.beta.expect("validated"),
                    );
                    // A row of weight 0 advances the clock and learns
                    // nothing (hard rule 9): the paper's recursion has no
                    // row weights, and took its `u` at full strength in the
                    // `α·u` term (review 2026-09-28). A positive weight
                    // reaches `rho` through `rho_bar` alone.
                    if w <= 0.0 {
                        self.rho[m]
                    } else if self.rho[m].is_finite() {
                        (1.0 - alpha - beta) * *bar + alpha * um + beta * self.rho[m]
                    } else {
                        um
                    }
                }
            };
            self.rho_w[m] = w_new;
        }
    }
}

/// `rho_w` was one weight for every correlation value before task 115 (h);
/// a state written then holds a number, read here as a list of one, which
/// [`Deco`]'s `restore` widens to every value.
fn weight_or_weights<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Weights {
        One(f64),
        Each(Vec<f64>),
    }
    Ok(match Weights::deserialize(d)? {
        Weights::One(w) => vec![w],
        Weights::Each(v) => v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Deco::new(cfg(3)).unwrap();
        let mut s = m.state();
        let ModelState::Deco(inner) = &mut s.model else {
            unreachable!()
        };
        inner.rho.pop();
        match Deco::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::OnlineModel;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// A feature without variance yet -- the first row, or one constant so
    /// far -- has no standardized value: NaN, not a division by a zero
    /// variance (which would be infinite for any row off the mean).
    #[test]
    fn a_feature_without_variance_has_no_standardized_value() {
        let m = Deco::new(cfg(2)).unwrap();
        let mut out = Vec::new();
        m.standardise(&[3.0, -2.0], &mut out);
        assert!(out.iter().all(|v| v.is_nan()), "{out:?}");
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

    /// The level is an `ew_cov`'s mean of `u`, to the bit, through a value
    /// `u` repeats: both take their steps as pairs ([`crate::comp`];
    /// docs/PLAN.md task 101), so neither stops short of the value, and the
    /// level reaches it. Rows of weight 0 among them move nothing.
    #[test]
    fn the_level_is_ew_covs_mean_through_a_value_u_repeats() {
        let mut d = Deco::new(cfg(2)).unwrap();
        let mut cov = crate::EwCov::new(1);
        let lam = Decay::Halflife(20.0).factor(1.0);
        let mut s = 5u64;
        for row in 0..3000 {
            let u = if row < 300 { 0.3 * lcg(&mut s) } else { 0.8 };
            let w = if row % 7 == 3 { 0.0 } else { 1.0 };
            d.advance(&[u], lam, w);
            cov.update(&[u], lam, w);
            assert_eq!(d.rho_bar[0].to_bits(), cov.mean(0).to_bits(), "row {row}");
        }
        assert_eq!(d.rho_bar[0], 0.8, "the level reaches the value u holds");
    }

    /// A row of weight 0 takes no step in the level, to the bit: the rows
    /// 0.7 and 5.292162135665459 leave its pair with a low part of a whole
    /// rounding step, where a zero step would round the double up
    /// (`crate::comp::add`).
    #[test]
    fn a_row_of_no_weight_leaves_the_level_as_it_was() {
        let mut d = Deco::new(cfg(2)).unwrap();
        d.advance(&[0.7], 1.0, 1.0);
        d.advance(&[5.292162135665459], 1.0, 1.0);
        assert_eq!(d.rho_bar[0], 2.996081067832729, "the fixture");
        assert_eq!(d.rho_bar_lo[0], 4.440892098500626e-16);
        d.advance(&[1.0], 1.0, 0.0);
        assert_eq!(d.rho_bar[0], 2.996081067832729);
        assert_eq!(d.rho_bar_lo[0], 4.440892098500626e-16);
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
        let (s1, s2, _, ok) = m.block_sums(&r);
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

    /// Under either dynamics: the linear one took the row's `u` at full
    /// strength in its `α·u` term whatever the row's weight, so a zero-weight
    /// row moved `rho` (review 2026-09-28; the user: "Hold rho still on a
    /// zero-weight row"). This test ran `"ew"` alone.
    #[test]
    fn a_zero_weight_row_mid_stream_moves_nothing_but_the_clock() {
        let linear = DecoCfg {
            dynamics: DecoDynamics::Linear,
            alpha: Some(0.05),
            beta: Some(0.9),
            ..cfg(3)
        };
        for c in [cfg(3), linear] {
            let mut m = Deco::new(c.clone()).unwrap();
            for x in stream(50, 3, 0.3, 5) {
                m.step(&x, &[], 1.0, 1.0);
            }
            let before = m.rho.clone();
            m.step(&[9.0, -9.0, 9.0], &[], 1.0, 0.0);
            assert_eq!(m.rho, before, "{:?}", c.dynamics);
        }
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

    /// A row far past the standardiser's spread, under a `ρ` at the edge of
    /// what a correlation matrix allows, has a log-density below the
    /// double's range (review 2026-09-27, G4, from the contract's proptest):
    /// weights of `0.01`, `0.01` and `1e100` leave `x0` a spread near
    /// `5e-52`, the next row at `1e100` standardises to about `2e151`, `ρ` is
    /// `-1`, the density clamps it to `1e-9` of singular, and the quadratic
    /// form reaches `1e311`. `loglik` was `-inf`; it is NaN, no reading, as
    /// `hmm`'s is when its densities leave the range.
    #[test]
    fn a_log_density_past_the_range_is_no_reading() {
        use crate::OnlineModel;
        let mut m = Deco::new(DecoCfg {
            min_periods: 3.0,
            ..cfg(2)
        })
        .unwrap();
        m.step(&[0.5, 0.0], &[], 0.0, 0.01);
        m.step(&[0.0, 1e100], &[], 1.0, 0.01);
        m.step(&[0.5, 0.0], &[], 1.0, 1e100);
        let out = m.step(&[1e100, 0.0], &[], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| !v.is_infinite()), "{:?}", out.pred);
        assert!(out.pred[2].is_nan(), "{:?}", out.pred);
        assert_eq!(out.pred[1], -1.0, "the emitted rho is never clamped");
        // An ordinary row still has its density.
        let near = m.predict(&[0.5, 0.0], 1.0);
        assert!(near.pred[2].is_finite(), "{:?}", near.pred);
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

    /// A column with no spread is left out of its block's sums (docs/PLAN.md
    /// task 115 (h)): while it is constant, the model reads what the model
    /// without it reads, `u` and `rho` to the bit, and `loglik`, which needs
    /// every column, is NaN. No value learned on such a row before.
    #[test]
    fn a_column_without_spread_is_left_out_of_its_block() {
        let mut four = Deco::new(cfg(4)).unwrap();
        let mut three = Deco::new(cfg(3)).unwrap();
        for (i, x) in stream(300, 3, 0.4, 7).iter().enumerate() {
            let flat = [x[0], x[1], x[2], 2.0];
            let got = four.step(&flat, &[], 1.0, 1.0).pred;
            let want = three.step(x, &[], 1.0, 1.0).pred;
            assert!(same(&got[..2], &want[..2]), "row {i}: {got:?} vs {want:?}");
            assert!(got[2].is_nan(), "row {i}: no density without every column");
        }
        assert!(four.rho[0].is_finite(), "and it learned");
    }

    /// Each correlation value keeps its own weight (task 115 (h)). With a
    /// flat column in block B, B's value has no estimate and stays NaN, while
    /// A's learns -- as the one-block model over A's columns, to the bit, on
    /// every row -- and the pair's learns from B's other column. Once the
    /// column moves, every value learns.
    #[test]
    fn each_value_keeps_its_own_weight() {
        let mut m = Deco::new(DecoCfg {
            blocks: vec![vec![0, 1], vec![2, 3]],
            ..cfg(4)
        })
        .unwrap();
        let mut a = Deco::new(cfg(2)).unwrap();
        for (i, row) in stream(600, 4, 0.4, 11).iter().enumerate() {
            let mut x = row.clone();
            if i < 300 {
                x[3] = 2.0;
            }
            m.step(&x, &[], 1.0, 1.0);
            a.step(&x[..2], &[], 1.0, 1.0);
            assert!(
                same(&m.rho[..1], &a.rho),
                "row {i}: block A is its own model"
            );
            if i < 300 {
                assert!(m.rho[1].is_nan(), "row {i}: B has one column with a spread");
                assert!(
                    i < 2 || m.rho[2].is_finite(),
                    "row {i}: the pair learns from x2"
                );
            }
        }
        assert!(m.rho.iter().all(|v| v.is_finite()), "{:?}", m.rho);
    }

    /// A state written before task 115 (h) has one weight for every value:
    /// it loads with that weight on each value, from the state itself (its
    /// JSON, `rho_w` a number) and from the format a bank file holds.
    #[test]
    fn a_state_with_one_weight_for_every_value_loads() {
        let mut m = Deco::new(DecoCfg {
            blocks: vec![vec![0, 1], vec![2, 3]],
            ..cfg(4)
        })
        .unwrap();
        for x in stream(50, 4, 0.3, 3) {
            m.step(&x, &[], 1.0, 1.0);
        }
        let mut v = serde_json::to_value(crate::OnlineModel::state(&m)).unwrap();
        crate::window::json_edit(&mut v, "rho_w", &mut |x| *x = serde_json::json!(0.75));
        let back =
            <Deco as crate::OnlineModel>::restore(&serde_json::from_value(v).unwrap()).unwrap();
        assert_eq!(back.rho_w, vec![0.75; 3]);
        #[derive(Serialize)]
        struct Written {
            rho_w: f64,
        }
        #[derive(Deserialize)]
        struct Read {
            #[serde(deserialize_with = "super::weight_or_weights")]
            rho_w: Vec<f64>,
        }
        let bytes = rmp_serde::to_vec_named(&Written { rho_w: 2.5 }).unwrap();
        let read: Read = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(read.rho_w, vec![2.5]);
    }
}
