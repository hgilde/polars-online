//! Recursive least squares (docs/PLAN.md §4.2), in square-root information
//! (QR) form.
//!
//! The model is decayed ridge least squares solved exactly every row -- the
//! `ridge_decay` mode of [`crate::EwRidge`], which the agreement test exploits:
//!
//! ```text
//! A   <- lam A   + w z z^T          A_0 = ridge I
//! b_j <- lam b_j + w y_j z          b_0 = ridge coef0_j
//! beta_j = A^-1 b_j
//! ```
//!
//! Neither `A` nor its inverse `P` is stored. The state is the Cholesky factor
//! `R` of `A` (upper triangular, `A = R^T R`) and `u_j = R^-T b_j`; a row is
//! folded in by the `k` Givens rotations that re-triangularize the stacked
//! matrix, and the coefficients come from one back-substitution:
//!
//! ```text
//! R <- sqrt(lam) R,   u_j <- sqrt(lam) u_j                (decay)
//! [ R        u    ]       [ R'  u' ]
//! [ sw z^T   sw y ]  =  Q [ 0   e  ]      sw = sqrt(w)     (rotations)
//! beta_j = R'^-1 u'_j                                     (back-substitution)
//! ```
//!
//! This is O(k^2) per row, like the textbook gain/covariance recursion
//! `P <- (P - g z^T P) / lam`, which is what this model used to be. That
//! recursion is unfit for unbounded streams (docs/IMPROVEMENTS.md C5): the
//! one-ulp asymmetry between `g_i (Pz)_j` and `g_j (Pz)_i` is never touched
//! by the rank-1 downdate and is multiplied by `1/lam` every row, so `P` is
//! garbage after ~60 halflives on any data; and a row whose information
//! exceeds the prior's by `1/ulp` in some direction (a feature ~1e8 times its
//! usual scale) cancels that direction of `P` to zero or to rounding noise,
//! and a zero never regrows because the only growth `P` has is
//! multiplicative -- the coefficient is frozen for good. Rotations are
//! orthogonal: nothing cancels, an outlier's information decays with
//! `sqrt(lam)` per row like everything else, and every later row is kept.
//!
//! `ridge` is the classic RLS prior strength (`P_0 = I / ridge`, i.e.
//! `A_0 = ridge I`), and the intercept is penalized too.
//!
//! Null policy deviation, documented: a row with ANY null target is predict-only
//! for all targets, because `R` is shared across targets and a per-target
//! update would desynchronize it.

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::dot_aug;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlsCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Prior strength: `P0 = I / ridge`.
    pub ridge: f64,
    /// Initial coefficients per target (length `k_total`), default zeros.
    pub coef0: Option<Vec<Vec<f64>>>,
    pub min_periods: f64,
}

impl RlsCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.ridge <= 0.0 || self.ridge.is_nan() {
            return Err("rls: ridge must be > 0 (it sets P0 = I / ridge)".into());
        }
        if let Some(c) = &self.coef0 {
            if c.len() != self.n_targets || c.iter().any(|v| v.len() != self.k_total()) {
                return Err("rls: coef0 must be n_targets x k_total".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RlsWire")]
pub struct Rls {
    cfg: RlsCfg,
    /// Cholesky factor of the decayed information matrix, `A = R^T R`: upper
    /// triangular, row-major `k*k`, lower triangle zero, diagonal `>= 0`.
    r: Vec<f64>,
    /// `u_j = R^-T b_j` per target, each `k_total`: the rotated right-hand side.
    u: Vec<Vec<f64>>,
    /// Coefficients per target, `R^-1 u_j`, refreshed after every update.
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    seen: bool,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    ybuf: Vec<f64>,
}

/// The layouts `Rls` loads. Schema-1 files held the covariance `P` (see the
/// module docs for why it is gone); they are converted on load.
///
/// Newtype variants, not struct variants: an untagged struct variant only
/// deserializes from a map, and the compact msgpack encoding writes structs
/// as arrays.
#[derive(Deserialize)]
#[serde(untagged)]
enum RlsWire {
    Current(RlsV2),
    Schema1(RlsV1),
}

#[derive(Deserialize)]
struct RlsV2 {
    cfg: RlsCfg,
    r: Vec<f64>,
    u: Vec<Vec<f64>>,
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    seen: bool,
}

#[derive(Deserialize)]
struct RlsV1 {
    cfg: RlsCfg,
    p: Vec<f64>,
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    seen: bool,
}

impl TryFrom<RlsWire> for Rls {
    type Error = String;

    fn try_from(w: RlsWire) -> Result<Self, String> {
        match w {
            RlsWire::Current(RlsV2 {
                cfg,
                r,
                u,
                beta,
                w_sum,
                seen,
            }) => Ok(Self {
                cfg,
                r,
                u,
                beta,
                w_sum,
                seen,
                zbuf: vec![],
                ybuf: vec![],
            }),
            RlsWire::Schema1(RlsV1 {
                cfg,
                p,
                beta,
                w_sum,
                seen,
            }) => {
                let k = cfg.k_total();
                if p.len() != k * k || beta.len() != cfg.n_targets {
                    return Err("rls: schema-1 state has the wrong shape".into());
                }
                let r = factor_of_inverse(&p, k).ok_or_else(|| {
                    "rls: schema-1 state's covariance is not positive definite".to_string()
                })?;
                // b = A beta = R^T R beta  =>  u = R^-T b = R beta.
                let u = beta
                    .iter()
                    .map(|b| {
                        (0..k)
                            .map(|i| (i..k).map(|j| r[i * k + j] * b[j]).sum())
                            .collect()
                    })
                    .collect();
                Ok(Self {
                    cfg,
                    r,
                    u,
                    beta,
                    w_sum,
                    seen,
                    zbuf: vec![],
                    ybuf: vec![],
                })
            }
        }
    }
}

/// Upper-triangular `R` with `R^T R = P^-1`, for a symmetric positive definite
/// `P` (row-major `k*k`); `None` if `P` is not positive definite.
///
/// With `J` the row-reversal permutation and `J P J = L L^T` (Cholesky),
/// `P^-1 = (J L^-1 J)^T (J L^-1 J)`, and `J L^-1 J` is upper triangular.
fn factor_of_inverse(p: &[f64], k: usize) -> Option<Vec<f64>> {
    let flip = |i: usize| k - 1 - i;
    // L: Cholesky of the reversed P, lower triangular.
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut acc = p[flip(i) * k + flip(j)];
            for t in 0..j {
                acc -= l[i * k + t] * l[j * k + t];
            }
            if i == j {
                // Not positive definite (or already broken): nothing to convert.
                if acc <= 0.0 || !acc.is_finite() {
                    return None;
                }
                l[i * k + i] = acc.sqrt();
            } else {
                l[i * k + j] = acc / l[j * k + j];
            }
        }
    }
    // M = L^-1 by forward substitution, one column at a time.
    let mut m = vec![0.0; k * k];
    for col in 0..k {
        for i in col..k {
            let mut acc = if i == col { 1.0 } else { 0.0 };
            for t in col..i {
                acc -= l[i * k + t] * m[t * k + col];
            }
            m[i * k + col] = acc / l[i * k + i];
        }
    }
    let mut r = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            r[flip(i) * k + flip(j)] = m[i * k + j];
        }
    }
    Some(r)
}

impl Rls {
    pub fn new(cfg: RlsCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let root = cfg.ridge.sqrt();
        let mut r = vec![0.0; k * k];
        for i in 0..k {
            r[i * k + i] = root;
        }
        let beta = cfg
            .coef0
            .clone()
            .unwrap_or_else(|| vec![vec![0.0; k]; cfg.n_targets]);
        // A_0 = ridge I, b_0 = ridge coef0  =>  u_0 = R_0^-T b_0 = sqrt(ridge) coef0.
        let u = beta
            .iter()
            .map(|b| b.iter().map(|v| root * v).collect())
            .collect();
        Ok(Self {
            r,
            u,
            beta,
            w_sum: 0.0,
            seen: false,
            zbuf: vec![0.0; k],
            ybuf: vec![0.0; cfg.n_targets],
            cfg,
        })
    }

    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.beta
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
        }
        if self.ybuf.len() != self.cfg.n_targets {
            self.ybuf = vec![0.0; self.cfg.n_targets];
        }
    }

    /// `beta_j = R^-1 u_j` by back-substitution. A zero pivot is a direction
    /// no row has ever excited after the prior has decayed away entirely
    /// (`sqrt(ridge) lam_acc^(1/2)` underflows after ~2000 halflives without
    /// data): the coefficient there is set to zero rather than to `0/0`.
    fn solve(&mut self) {
        let k = self.cfg.k_total();
        for (beta, u) in self.beta.iter_mut().zip(&self.u) {
            for i in (0..k).rev() {
                let row = i * k;
                let mut acc = u[i];
                for (rij, bj) in self.r[row + i + 1..row + k].iter().zip(&beta[i + 1..]) {
                    acc -= rij * bj;
                }
                let d = self.r[row + i];
                let b = if d > 0.0 { acc / d } else { 0.0 };
                beta[i] = if b.is_finite() { b } else { 0.0 };
            }
        }
    }
}

impl OnlineModel for Rls {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict (state before the update) ----
        let out = self.predict(x, d_clock);

        // ---- decay ----
        // A <- lam A, b <- lam b  =>  R <- sqrt(lam) R, u <- sqrt(lam) u. The
        // coefficients are invariant to it. Skipped when lam == 1 (no-op).
        if lam != 1.0 {
            let s = lam.sqrt();
            for i in 0..k {
                for v in &mut self.r[i * k + i..(i + 1) * k] {
                    *v *= s;
                }
            }
            for u in &mut self.u {
                for v in u.iter_mut() {
                    *v *= s;
                }
            }
        }
        self.w_sum = lam * self.w_sum + weight;

        // ---- update (only when every target is present) ----
        if weight > 0.0 && y.iter().all(Option::is_some) {
            // The working row is sqrt(w) [z, y]; each rotation zeroes one of
            // its entries against the matching row of R and moves what was
            // there down into the rest of the working row.
            let sw = weight.sqrt();
            for z in self.zbuf.iter_mut() {
                *z *= sw;
            }
            for (yb, yj) in self.ybuf.iter_mut().zip(y) {
                *yb = sw * yj.unwrap();
            }
            for i in 0..k {
                let zi = self.zbuf[i];
                if zi == 0.0 {
                    continue;
                }
                let row = i * k;
                let rii = self.r[row + i];
                let mut rho = (rii * rii + zi * zi).sqrt();
                if !(rho > 0.0 && rho.is_finite()) {
                    // Squares over- or underflowed (inputs beyond 1e154 or
                    // below 1e-154); `hypot` scales internally.
                    rho = rii.hypot(zi);
                    if !(rho > 0.0 && rho.is_finite()) {
                        continue;
                    }
                }
                let (c, s) = (rii / rho, zi / rho);
                self.r[row + i] = rho;
                self.zbuf[i] = 0.0;
                for j in i + 1..k {
                    let rij = self.r[row + j];
                    let zj = self.zbuf[j];
                    self.r[row + j] = c * rij + s * zj;
                    self.zbuf[j] = c * zj - s * rij;
                }
                for (u, yb) in self.u.iter_mut().zip(self.ybuf.iter_mut()) {
                    let ui = u[i];
                    u[i] = c * ui + s * *yb;
                    *yb = c * *yb - s * ui;
                }
            }
            self.solve();
            self.seen = true;
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        let n_eff = self.w_sum;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        if n_eff >= self.cfg.min_periods && self.seen {
            for (p, beta) in pred.iter_mut().zip(&self.beta) {
                *p = dot_aug(beta, x, self.cfg.add_intercept);
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Rls(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Rls(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "rls",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EwRidge, EwRidgeCfg};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn rls_cfg(k: usize, m: usize, hl: f64, ridge: f64) -> RlsCfg {
        RlsCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(hl),
            ridge,
            coef0: None,
            min_periods: 0.0,
        }
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let bad = |f: &dyn Fn(&mut RlsCfg), want: &str| {
            let mut c = rls_cfg(2, 1, 100.0, 1.0);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");
        // ridge sets P0 = I/ridge, so zero would be an infinite prior variance.
        bad(&|c| c.ridge = 0.0, "ridge must be > 0");
        bad(&|c| c.ridge = -1.0, "ridge must be > 0");
        bad(&|c| c.ridge = f64::NAN, "ridge must be > 0");
        // coef0 is one vector per target, each of length k_total (2 + intercept).
        bad(
            &|c| c.coef0 = Some(vec![vec![0.0; 3], vec![0.0; 3]]),
            "n_targets x k_total",
        );
        bad(
            &|c| c.coef0 = Some(vec![vec![0.0; 2]]),
            "n_targets x k_total",
        );
        let mut ok = rls_cfg(2, 1, 100.0, 1.0);
        ok.coef0 = Some(vec![vec![1.0, 2.0, 3.0]]);
        ok.validate().unwrap();
        rls_cfg(2, 1, 100.0, 1.0).validate().unwrap();
    }

    #[test]
    fn a_zero_weight_row_is_pure_decay() {
        let mut m = Rls::new(rls_cfg(2, 1, 10.0, 1.0)).unwrap();
        let mut s = 97u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.beta.clone();
        let w = m.w_sum;
        m.step(&[0.4, -0.2], &[Some(-500.0)], 1.0, 0.0);
        assert_eq!(m.beta, beta, "weight 0 must not move the fit");
        let lam = 0.5f64.powf(1.0 / 10.0);
        assert!((m.w_sum - w * lam).abs() < 1e-12);
    }

    /// docs/PLAN.md §9 class 1: RLS and EW-ridge with `solve_every` = 1 row and
    /// the matching decaying prior must agree to float precision.
    #[test]
    fn agrees_with_ewridge_solved_every_row() {
        let (k, hl, ridge) = (3usize, 40.0, 0.7);
        let mut rls = Rls::new(rls_cfg(k, 2, hl, ridge)).unwrap();
        let mut ew = EwRidge::new(EwRidgeCfg {
            n_features: k,
            n_targets: 2,
            add_intercept: true,
            decay: Decay::Halflife(hl),
            ridge: vec![ridge],
            feature_sets: vec![],
            standardize: false,
            ridge_decay: true,
            session_shrink: None,
            long_halflife: None,
            coef0: None,
            min_periods: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
        })
        .unwrap();

        let mut s = 99u64;
        let mut max_diff: f64 = 0.0;
        for i in 0..400 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y0 = 1.5 * x[0] - x[1] + 0.3 + 0.05 * lcg(&mut s);
            let y1 = 0.2 * x[2] + 0.01 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 0.5 + lcg(&mut s).abs() };
            let w = 0.5 + lcg(&mut s).abs();
            let a = rls.step(&x, &[Some(y0), Some(y1)], d, w);
            let b = ew.step(&x, &[Some(y0), Some(y1)], d, w);
            if i > 5 {
                for j in 0..2 {
                    assert!(a.pred[j].is_finite() && b.pred[j].is_finite());
                    max_diff = max_diff.max((a.pred[j] - b.pred[j]).abs());
                }
            }
        }
        assert!(max_diff < 1e-9, "max pred difference {max_diff}");
    }

    #[test]
    fn recovers_static_beta() {
        let mut m = Rls::new(rls_cfg(2, 1, f64::INFINITY, 1e-6)).unwrap();
        let mut s = 21u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - 0.5 * x[1] + 1.0;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!((b[0] - 1.0).abs() < 1e-6);
        assert!((b[1] - 2.0).abs() < 1e-6);
        assert!((b[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Rls::new(rls_cfg(1, 2, 100.0, 1.0)).unwrap();
        let mut s = 31u64;
        for i in 0..40 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.beta.clone();
        let (r_before, u_before, w_before) = (m.r.clone(), m.u.clone(), m.w_sum);
        let lam = 0.5f64.powf(1.0 / 100.0);
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred.iter().all(|p| p.is_finite()));
        assert_eq!(
            m.beta, before,
            "a null target must not update any coefficient"
        );
        // RLS shares one information factor across targets, so it cannot
        // update some and not others -- but the row is not ignored: the weight
        // advances and the forgetting factor still rescales R and u.
        assert!((m.w_sum - (w_before * lam + 1.0)).abs() < 1e-12);
        let root = lam.sqrt();
        for (a, b) in m.r.iter().zip(&r_before) {
            assert!(
                (a - b * root).abs() < 1e-9 * (1.0 + b.abs()),
                "R must still be rescaled by the decay"
            );
        }
        for (a, b) in m.u.iter().flatten().zip(u_before.iter().flatten()) {
            assert!(
                (a - b * root).abs() < 1e-9 * (1.0 + b.abs()),
                "u must still be rescaled by the decay"
            );
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Rls::new(rls_cfg(2, 1, 50.0, 0.5)).unwrap();
        let mut s = 41u64;
        let rows: Vec<([f64; 2], f64)> = (0..80)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..40].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Rls::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[40..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    /// A schema-1 state carried the covariance `P = (R^T R)^-1` and the
    /// coefficients; loading one must reproduce the same model.
    #[test]
    fn loads_a_schema_1_state() {
        use crate::{MIN_SCHEMA_VERSION, SCHEMA_VERSION};
        const { assert!(MIN_SCHEMA_VERSION == 1 && SCHEMA_VERSION >= 2) };

        // The schema-1 layout, as `rmp_serde::to_vec_named` wrote it.
        #[derive(Serialize)]
        struct RlsV1 {
            cfg: RlsCfg,
            p: Vec<f64>,
            beta: Vec<Vec<f64>>,
            w_sum: f64,
            seen: bool,
        }
        #[derive(Serialize)]
        enum ModelStateV1 {
            Rls(RlsV1),
        }
        #[derive(Serialize)]
        struct StateV1 {
            schema_version: u32,
            model: ModelStateV1,
        }

        let k = 3;
        let mut m1 = Rls::new(rls_cfg(2, 2, 50.0, 0.5)).unwrap();
        let mut s = 43u64;
        let rows: Vec<([f64; 2], f64)> = (0..120)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - x[1] + 0.2)
            })
            .collect();
        for (i, (x, y)) in rows[..60].iter().enumerate() {
            m1.step(
                x,
                &[Some(*y), Some(-y)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        // P = (R^T R)^-1, the way a schema-1 model would have held it.
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|t| m1.r[t * k + i] * m1.r[t * k + j]).sum();
            }
        }
        let mut eye = vec![0.0; k * k];
        for i in 0..k {
            eye[i * k + i] = 1.0;
        }
        let (p_cols, _) = crate::solve::solve_spd(&a, &eye, k, k).unwrap();
        let p: Vec<f64> = (0..k * k).map(|i| p_cols[(i % k) * k + i / k]).collect();
        let v1 = StateV1 {
            schema_version: 1,
            model: ModelStateV1::Rls(RlsV1 {
                cfg: m1.cfg.clone(),
                p,
                beta: m1.beta.clone(),
                w_sum: m1.w_sum,
                seen: m1.seen,
            }),
        };
        // Bank files are map-encoded; the compact array encoding must load too.
        for bytes in [
            rmp_serde::to_vec_named(&v1).unwrap(),
            rmp_serde::to_vec(&v1).unwrap(),
        ] {
            let st: State = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(st.schema_version, 1);
            let mut m2 = Rls::restore(&st).unwrap();
            assert_eq!(m2.beta, m1.beta);
            assert_eq!(m2.w_sum, m1.w_sum);
            let mut m1 = m1.clone();
            for (x, y) in &rows[60..] {
                let a = m1.step(x, &[Some(*y), Some(-y)], 1.0, 1.0).pred;
                let b = m2.step(x, &[Some(*y), Some(-y)], 1.0, 1.0).pred;
                for (a, b) in a.iter().zip(&b) {
                    assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
                }
            }
        }
    }
}
