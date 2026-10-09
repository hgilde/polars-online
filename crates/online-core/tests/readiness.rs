//! The noise gate's statistic is exact by theory
//! (docs/WARMUP-AND-CONVERGENCE.md §2.1): `h(x) = x' Σ̂⁻¹ x / n_kish` is the
//! estimation variance of a prediction over the noise, for a design whose
//! covariance is constant over the memory window, and `edf / n_kish` is its
//! average. These are the identities the implementation must satisfy, each
//! of which fails loudly if the squared-weight sum, the weights or the ridge
//! are wired wrongly (§7.4). They verify the wiring; nothing here is tuned.

use online_core::{
    CoefVariance, Decay, EwRidge, EwRidgeCfg, Lasso, LassoCfg, OnlineModel, Rls, RlsCfg, TargetGaps,
};

fn cfg(k: usize, half_life: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(half_life),
        ridge: vec![1e-8],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// A model that keeps its factors, so the per-row leverage can be read.
fn model(c: EwRidgeCfg) -> EwRidge {
    let mut m = EwRidge::new(c).unwrap();
    m.set_keep_factor(true);
    m
}

/// A deterministic standard-normal-ish draw (twelve uniforms, centred).
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
    fn row(&mut self, k: usize) -> Vec<f64> {
        (0..k).map(|_| self.normal()).collect()
    }
}

/// `h(x)` for this row, from the state before it: `inflation² − 1`.
fn h_of(m: &EwRidge, x: &[f64]) -> f64 {
    let mut out = Vec::new();
    assert!(m.row_error_inflation_into(x, 1.0, &mut out));
    out[0] * out[0] - 1.0
}

/// The stream-level `edf / n_kish`, the same way.
fn h_mean_of(m: &EwRidge) -> f64 {
    let mut out = Vec::new();
    assert!(m.error_inflation_into(&mut out));
    out[0] * out[0] - 1.0
}

fn steady_n_kish(half_life: f64) -> f64 {
    let lam = (-(1.0 / half_life)).exp2();
    (1.0 + lam) / (1.0 - lam)
}

/// (i) On a stationary design the mean of the per-row `h` is `edf / n_kish`:
/// the per-row statistic and the stream-level one are the same quantity
/// seen two ways, and `n_kish` reaches `(1 + λ) / (1 − λ)`.
#[test]
fn the_mean_of_h_over_a_stationary_design_is_edf_over_n_kish() {
    let (k, half_life) = (4, 20.0);
    let mut m = model(cfg(k, half_life));
    let mut rng = Rng(11);
    let mut hs = Vec::new();
    for i in 0..3000 {
        let x = rng.row(k);
        let y = x.iter().sum::<f64>() + rng.normal();
        if i >= 2000 {
            hs.push(h_of(&m, &x));
        }
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let mean_h = hs.iter().sum::<f64>() / hs.len() as f64;
    let want = h_mean_of(&m);
    // Conservative, and within a quarter: the per-row form reads the design
    // through `(s₂/s₁)·G` where the exact variance has `G₂`, and the two
    // share their sampled rows, so at `n_kish / k_eff ≈ 12` the per-row
    // mean sits 10% above the average (measured 2026-09-21; the test pins
    // the direction and the order, docs/WARMUP-AND-CONVERGENCE.md §5.8).
    assert!(
        mean_h >= 0.95 * want && mean_h <= 1.25 * want,
        "mean h {mean_h} vs edf / n_kish {want}"
    );
    // And that average is `(k + 1) / n_kish` at steady state with a tiny ridge.
    let n_kish = steady_n_kish(half_life);
    let expect = (k + 1) as f64 / n_kish;
    assert!(
        (want - expect).abs() < 0.05 * expect,
        "edf / n_kish {want} vs (k + 1) / n_kish {expect}"
    );
}

/// (ii) The observed out-of-sample error is the noise floor inflated by
/// the gate's `sqrt(1 + edf / n_kish)`: the statistic predicts the error it
/// says it predicts, and the per-row form errs on the safe side.
#[test]
fn observed_error_inflation_matches_the_prediction() {
    let (k, half_life, sigma) = (4, 4.0, 1.0);
    let mut m = model(cfg(k, half_life));
    let mut rng = Rng(23);
    let (mut se, mut hs) = (0.0, Vec::new());
    let n = 6000;
    for i in 0..n {
        let x = rng.row(k);
        let y = 1.0 + x.iter().sum::<f64>() + sigma * rng.normal();
        if i >= 200 {
            hs.push(h_of(&m, &x));
        }
        let step = m.step(&x, &[Some(y)], 1.0, 1.0);
        if i >= 200 {
            let e = y - step.pred[0];
            se += e * e;
        }
    }
    let rms = (se / hs.len() as f64).sqrt();
    let mean_h = hs.iter().sum::<f64>() / hs.len() as f64;
    let observed = rms / sigma;
    // The gate's statistic -- the stream-level `sqrt(1 + edf / n_kish)` --
    // is what a user's predictions are held to, and it tracks the error
    // within a few percent even at `n_kish ≈ 2.3 k_eff` (1.197 predicted
    // against 1.214 observed, 2026-09-21).
    let gate = (1.0 + h_mean_of(&m)).sqrt();
    assert!(
        (observed - gate).abs() < 0.05 * gate,
        "observed inflation {observed} vs the gate's {gate}"
    );
    // The per-row form is conservative here: it overstates the excess,
    // by half at this half-life (0.73 against 0.47 observed), never
    // understates it. Exactness per row is what `G₂` would buy (§2.1).
    let excess = observed * observed - 1.0;
    assert!(
        mean_h >= excess && mean_h <= 2.0 * excess,
        "per-row mean h {mean_h} vs observed excess {excess}"
    );
    // Well away from 1: at half-life 4 the fit is visibly noisy.
    assert!(gate > 1.1, "{gate}");
}

/// (iii) Weights enter through Kish's `n` exactly: scaling every weight by
/// a constant changes nothing, and splitting every row into two halves at
/// the same clock leaves the Gram and `n_eff` where they were while `n_kish`
/// doubles, so `h` halves.
#[test]
fn weights_enter_through_kish_n_exactly() {
    let (k, half_life) = (3, 30.0);
    let mut whole = model(cfg(k, half_life));
    let mut scaled = model(cfg(k, half_life));
    let mut split = model(cfg(k, half_life));
    let mut rng = Rng(5);
    let probe = rng.row(k);
    for _ in 0..400 {
        let x = rng.row(k);
        let y = x[0] - x[1] + rng.normal();
        whole.step(&x, &[Some(y)], 1.0, 1.0);
        scaled.step(&x, &[Some(y)], 1.0, 2.5);
        split.step(&x, &[Some(y)], 1.0, 0.5);
        split.step(&x, &[Some(y)], 0.0, 0.5);
    }
    let (hw, hs, hh) = (
        h_of(&whole, &probe),
        h_of(&scaled, &probe),
        h_of(&split, &probe),
    );
    assert!((hs - hw).abs() < 1e-9 * hw, "scaled {hs} vs {hw}");
    assert!(
        (hh - hw / 2.0).abs() < 1e-9 * hw,
        "split {hh} vs half of {hw}"
    );
    let (mw, ms, mh) = (h_mean_of(&whole), h_mean_of(&scaled), h_mean_of(&split));
    assert!((ms - mw).abs() < 1e-9 * mw);
    assert!((mh - mw / 2.0).abs() < 1e-9 * mw);
    assert!(
        (split.n_eff() - whole.n_eff()).abs() < 1e-9 * whole.n_eff(),
        "n_eff is the weight, unchanged by the split"
    );
}

/// (iv) A row whose `x` breaks a collinearity the fit relied on reads a large
/// `h`, while the rows that respect it read a small one -- the §5.7 hazard
/// caught per prediction.
#[test]
fn a_row_that_breaks_a_collinearity_reads_a_large_h() {
    let (k, half_life) = (3, 50.0);
    let mut m = model(cfg(k, half_life));
    let mut rng = Rng(7);
    let mut in_sample = Vec::new();
    for i in 0..600 {
        let mut x = rng.row(k);
        x[2] = x[0];
        let y = x[0] + 2.0 * x[1] + rng.normal();
        if i >= 300 {
            in_sample.push(h_of(&m, &x));
        }
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let typical = in_sample.iter().sum::<f64>() / in_sample.len() as f64;
    let mut probe = rng.row(k);
    probe[2] = probe[0] + 1.0;
    let broken = h_of(&m, &probe);
    assert!(
        broken > 100.0 * typical,
        "break {broken} vs in-sample {typical}"
    );
    let mut respects = rng.row(k);
    respects[2] = respects[0];
    let fine = h_of(&m, &respects);
    assert!(
        fine < 10.0 * typical,
        "respects {fine} vs in-sample {typical}"
    );
}

/// (v) `support_coef = 1 − λ (Σ̂⁻¹)_jj`: a duplicated pair reads 0.5 each, a
/// clean design reads 1, the intercept is not a share (null), and the sum
/// over the slopes is the effective degrees of freedom the gate reads.
#[test]
fn support_coef_reads_half_on_a_duplicate_and_one_on_a_clean_design() {
    let k = 3;
    let mut dup = model(cfg(k, f64::INFINITY));
    let mut clean = model(cfg(k, f64::INFINITY));
    let mut rng = Rng(3);
    for _ in 0..300 {
        let x = rng.row(k);
        let y = x[0] + x[1] + rng.normal();
        let mut xd = x.clone();
        xd[2] = xd[0];
        dup.step(&xd, &[Some(y)], 1.0, 1.0);
        clean.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let s = dup.support_coef().unwrap();
    assert_eq!(s.len(), 1);
    assert!(s[0][0].is_nan(), "the intercept is not a share: {:?}", s[0]);
    assert!((s[0][1] - 0.5).abs() < 1e-3, "{:?}", s[0]);
    assert!((s[0][2] - 1.0).abs() < 1e-3, "{:?}", s[0]);
    assert!((s[0][3] - 0.5).abs() < 1e-3, "{:?}", s[0]);
    let c = clean.support_coef().unwrap();
    assert!(
        c[0][1..].iter().all(|v| (v - 1.0).abs() < 1e-3),
        "{:?}",
        c[0]
    );
    // edf = intercept + Σ support: 1 + 2 for the duplicate, 1 + 3 clean.
    let mut out = Vec::new();
    assert!(dup.error_inflation_into(&mut out));
    let n_kish: f64 = 300.0;
    let want = (1.0 + 3.0 / n_kish).sqrt();
    assert!((out[0] - want).abs() < 1e-3, "{} vs {want}", out[0]);
    assert!(clean.error_inflation_into(&mut out));
    let want = (1.0 + 4.0 / n_kish).sqrt();
    assert!((out[0] - want).abs() < 1e-3, "{} vs {want}", out[0]);
}

/// A heavy ridge reads low everywhere -- the truth about that spec, not a
/// blind spot -- and the through-origin and standardized solves carry the
/// same identity in their own spaces.
#[test]
fn a_heavy_ridge_reads_low_and_every_solve_path_carries_the_identity() {
    let k = 2;
    let mut rng = Rng(9);
    let rows: Vec<(Vec<f64>, f64)> = (0..400)
        .map(|_| {
            let x = rng.row(k);
            let y = 3.0 + x[0] - x[1] + rng.normal();
            (x, y)
        })
        .collect();
    let run = |c: EwRidgeCfg| {
        let mut m = model(c);
        for (x, y) in &rows {
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        m
    };
    let mut heavy = cfg(k, f64::INFINITY);
    heavy.ridge = vec![5.0];
    let s = run(heavy).support_coef().unwrap();
    assert!(s[0][1..].iter().all(|v| *v < 0.3), "{:?}", s[0]);

    let mut origin = cfg(k, f64::INFINITY);
    origin.fit_intercept = false;
    let s = run(origin).support_coef().unwrap();
    assert_eq!(s[0].len(), k);
    assert!(s[0].iter().all(|v| (v - 1.0).abs() < 1e-3), "{:?}", s[0]);

    let mut std = cfg(k, f64::INFINITY);
    std.standardize = true;
    std.ridge = vec![1.0];
    // In correlation form each column has unit variance, so with `ridge = 1`
    // an orthogonal design reads exactly 1 / (1 + 1) on every slope.
    let s = run(std).support_coef().unwrap();
    assert!(
        s[0][1..].iter().all(|v| (v - 0.5).abs() < 0.05),
        "{:?}",
        s[0]
    );
}

/// Before the first solve there is nothing to read: the gate is infinite
/// (withheld), the support absent. After it, `predict` reads the same
/// numbers as the step would, without moving anything.
#[test]
fn before_the_first_solve_the_gate_is_infinite() {
    let k = 2;
    let mut c = cfg(k, 10.0);
    c.min_weight = (k + 1) as f64;
    let m = model(c);
    let mut out = Vec::new();
    assert!(m.error_inflation_into(&mut out));
    assert!(out[0].is_infinite());
    assert!(m.support_coef().is_none());
    assert!(m.row_error_inflation_into(&[1.0, 2.0], 1.0, &mut out));
    assert!(out[0].is_infinite());
}

// ---- `rls` (docs/PLAN.md task 116): the §7.4 identities, in sum form ----

fn rls(k: usize, half_life: f64, delta: f64) -> Rls {
    Rls::new(RlsCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(half_life),
        delta,
        coef_prior: None,
        min_weight: 0.0,
    })
    .unwrap()
}

/// `h(x)` for this row from `rls`'s per-row field, and the gate's bound.
fn rls_h(m: &Rls, x: &[f64]) -> f64 {
    let mut out = Vec::new();
    assert!(m.row_error_inflation_into(x, 1.0, &mut out));
    out[0] * out[0] - 1.0
}

fn rls_gate_h(m: &Rls) -> f64 {
    let mut out = Vec::new();
    assert!(m.error_inflation_into(&mut out));
    out[0] * out[0] - 1.0
}

/// (i) On a stationary design the mean of the per-row `h` is the gate's
/// `k_total / n_kish` once the prior has faded, as `ewridge`'s is its
/// `edf / n_kish`: the per-row form conservative by the same sampling
/// correlation (§2.1), at `delta` small and not.
#[test]
fn rls_the_mean_of_h_over_a_stationary_design_is_k_over_n_kish() {
    for delta in [1e-6, 10.0] {
        let (k, half_life) = (4, 20.0);
        let mut m = rls(k, half_life, delta);
        let mut rng = Rng(11);
        let mut hs = Vec::new();
        for i in 0..3000 {
            let x = rng.row(k);
            let y = x.iter().sum::<f64>() + rng.normal();
            if i >= 2000 {
                hs.push(rls_h(&m, &x));
            }
            m.step(&x, &[Some(y)], 1.0, 1.0);
        }
        let mean_h = hs.iter().sum::<f64>() / hs.len() as f64;
        let gate = rls_gate_h(&m);
        assert!(
            mean_h >= 0.95 * gate && mean_h <= 1.25 * gate,
            "delta {delta}: mean h {mean_h} vs k / n_kish {gate}"
        );
        let expect = (k + 1) as f64 / steady_n_kish(half_life);
        assert!((gate - expect).abs() < 0.01 * expect, "{gate} vs {expect}");
    }
}

/// (ii) The observed out-of-sample error is the noise floor inflated by
/// the gate's `sqrt(1 + k_total / n_kish)`, within 5% at a half-life of 4
/// rows, where the fit is visibly noisy; the per-row form errs on the safe
/// side.
#[test]
fn rls_observed_error_inflation_matches_the_gate() {
    let (k, half_life, sigma) = (4, 4.0, 1.0);
    let mut m = rls(k, half_life, 1e-6);
    let mut rng = Rng(23);
    let (mut se, mut hs) = (0.0, Vec::new());
    for i in 0..6000 {
        let x = rng.row(k);
        let y = 1.0 + x.iter().sum::<f64>() + sigma * rng.normal();
        if i >= 200 {
            hs.push(rls_h(&m, &x));
        }
        let step = m.step(&x, &[Some(y)], 1.0, 1.0);
        if i >= 200 {
            let e = y - step.pred[0];
            se += e * e;
        }
    }
    let observed = (se / hs.len() as f64).sqrt() / sigma;
    let gate = (1.0 + rls_gate_h(&m)).sqrt();
    assert!(
        (observed - gate).abs() < 0.05 * gate,
        "observed {observed} vs the gate's {gate}"
    );
    let mean_h = hs.iter().sum::<f64>() / hs.len() as f64;
    assert!(mean_h >= observed * observed - 1.0, "{mean_h}");
    assert!(gate > 1.1, "{gate}");
}

/// (iii) Weights enter through Kish's `n`: scaling every weight changes
/// nothing (to the prior's share, here 1e-9 of the information), and
/// splitting every row into two halves at the same clock keeps `A` and the
/// weight while `s₂` halves, so `h` halves.
#[test]
fn rls_weights_enter_through_kish_n_exactly() {
    let (k, half_life) = (3, 30.0);
    let (mut whole, mut scaled, mut split) = (
        rls(k, half_life, 1e-9),
        rls(k, half_life, 1e-9),
        rls(k, half_life, 1e-9),
    );
    let mut rng = Rng(5);
    let probe = rng.row(k);
    for _ in 0..400 {
        let x = rng.row(k);
        let y = x[0] - x[1] + rng.normal();
        whole.step(&x, &[Some(y)], 1.0, 1.0);
        scaled.step(&x, &[Some(y)], 1.0, 2.5);
        split.step(&x, &[Some(y)], 1.0, 0.5);
        split.step(&x, &[Some(y)], 0.0, 0.5);
    }
    let (hw, hs, hh) = (
        rls_h(&whole, &probe),
        rls_h(&scaled, &probe),
        rls_h(&split, &probe),
    );
    assert!((hs - hw).abs() < 1e-6 * hw, "scaled {hs} vs {hw}");
    assert!(
        (hh - hw / 2.0).abs() < 1e-9 * hw,
        "split {hh} vs half of {hw}"
    );
    let (mw, mh) = (rls_gate_h(&whole), rls_gate_h(&split));
    assert!((mh - mw / 2.0).abs() < 1e-9 * mw);
}

/// (iv) A row that breaks a collinearity the fit relied on reads an `h`
/// more than 100 times the rows that respect it.
#[test]
fn rls_a_row_that_breaks_a_collinearity_reads_a_large_h() {
    let (k, half_life) = (3, 50.0);
    let mut m = rls(k, half_life, 1e-6);
    let mut rng = Rng(7);
    let mut in_sample = Vec::new();
    for i in 0..600 {
        let mut x = rng.row(k);
        x[2] = x[0];
        let y = x[0] + 2.0 * x[1] + rng.normal();
        if i >= 300 {
            in_sample.push(rls_h(&m, &x));
        }
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let typical = in_sample.iter().sum::<f64>() / in_sample.len() as f64;
    let mut probe = rng.row(k);
    probe[2] = probe[0] + 1.0;
    assert!(rls_h(&m, &probe) > 100.0 * typical, "{typical}");
}

/// (v) The prior only adds information, so a stronger `delta` reads no
/// larger `h` on any row, while the gate's bound, `k_total / n_kish`, is
/// the same for both: conservative under the fading prior, by exactly what
/// the prior holds.
#[test]
fn rls_a_stronger_prior_reads_no_larger_h_and_the_same_bound() {
    let k = 2;
    let (mut weak, mut strong) = (rls(k, 25.0, 1e-6), rls(k, 25.0, 50.0));
    let mut rng = Rng(13);
    for i in 0..300 {
        let x = rng.row(k);
        let y = 1.0 + x[0] - x[1] + rng.normal();
        if i > 0 {
            let (hw, hs) = (rls_h(&weak, &x), rls_h(&strong, &x));
            assert!(hs <= hw * (1.0 + 1e-12), "row {i}: {hs} > {hw}");
            assert_eq!(rls_gate_h(&weak).to_bits(), rls_gate_h(&strong).to_bits());
        }
        weak.step(&x, &[Some(y)], 1.0, 1.0);
        strong.step(&x, &[Some(y)], 1.0, 1.0);
    }
}

// ---- `lasso` (docs/PLAN.md task 116): the active count ----

/// The gate reads `sqrt(1 + df / n_kish)` per path point, `df` the active
/// coefficients plus the intercept: held against the fit's own non-zeros
/// and Kish's size of unit rows at a literal `λ`, `(Σ λ^i)² / Σ λ^(2i)`.
#[test]
fn lasso_reads_the_active_count_over_kish_n() {
    let (k, n, lam) = (5, 400, 0.98);
    let mut m = Lasso::new(LassoCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Lam(lam),
        lasso_path: vec![0.5, 0.05, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 0.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        target_gaps: TargetGaps::OwnRows,
        max_iter: 500,
        tol: 1e-12,
    })
    .unwrap();
    let mut rng = Rng(19);
    let mut out = Vec::new();
    let mut seen_sparse = false;
    for i in 0..n {
        let x = rng.row(k);
        let y = 1.0 + 2.0 * x[0] - 1.5 * x[1] + 0.5 * rng.normal();
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        let rows = i + 1;
        let s1: f64 = (0..rows).map(|a| lam.powi(a)).sum();
        let s2: f64 = (0..rows).map(|a| lam.powi(2 * a)).sum();
        let n_kish = s1 * s1 / s2;
        assert!(m.error_inflation_into(&mut out));
        let path = &m.coefficients().unwrap()[0];
        for (li, b) in path.iter().enumerate() {
            if b.iter().any(|v| v.is_nan()) {
                assert!(out[li].is_infinite());
                continue;
            }
            let active = b[1..].iter().filter(|v| **v != 0.0).count();
            seen_sparse |= li == 0 && active == 2 && i > 100;
            let want = (1.0 + (active + 1) as f64 / n_kish).sqrt();
            assert!(
                (out[li] - want).abs() <= 1e-12 * want,
                "row {i}, point {li}"
            );
        }
    }
    assert!(
        seen_sparse,
        "the heavy penalty kept the two real features alone"
    );
    // The unpenalized point keeps every feature: df = k + 1.
    let b0 = &m.coefficients().unwrap()[0][2];
    assert!(b0.iter().all(|v| *v != 0.0));
    // And the gate's own read is the same statistic.
    let mut gate = Vec::new();
    assert!(m.error_inflation_gate_into(&[0.0; 5], 1.0, &mut gate, 1.1));
    assert_eq!(gate, out);
}

/// A `lasso` of `n_targets` targets over two features, solved every row,
/// at a literal `λ`.
fn lasso(n_targets: usize, fit_intercept: bool, path: Vec<f64>, lam: f64) -> Lasso {
    Lasso::new(LassoCfg {
        n_features: 2,
        n_targets,
        fit_intercept,
        decay: Decay::Lam(lam),
        lasso_path: path,
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 0.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        target_gaps: TargetGaps::OwnRows,
        max_iter: 500,
        tol: 1e-12,
    })
    .unwrap()
}

/// Each (target, path point) reads its own `sqrt(1 + df / n_kish)` in its
/// own slot, target-major: two targets over three path points, the
/// second target's fit unlike the first's, each slot held to its own
/// fit's active count over Kish's size of unit rows (docs/PLAN.md task
/// 220). With one target, as above, a slot read as another target's could
/// not show.
#[test]
fn lasso_reads_each_target_and_path_point_in_its_own_slot() {
    let lam = 0.97;
    let mut m = lasso(2, true, vec![0.8, 0.1, 0.0], lam);
    let mut rng = Rng(23);
    let (mut out, mut differ) = (Vec::new(), 0);
    for i in 0..300 {
        let x = rng.row(2);
        let y0 = 1.0 + 2.0 * x[0] + 0.5 * rng.normal();
        let y1 = -0.5 * x[1] + 0.5 * rng.normal();
        m.step(
            &x,
            &[Some(y0), Some(y1)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        let s1: f64 = (0..=i).map(|a| lam.powi(a)).sum();
        let s2: f64 = (0..=i).map(|a| lam.powi(2 * a)).sum();
        let n_kish = s1 * s1 / s2;
        assert!(m.error_inflation_into(&mut out));
        assert_eq!(out.len(), 6, "row {i}");
        let fits = m.coefficients().unwrap();
        for (j, path) in fits.iter().enumerate() {
            for (li, b) in path.iter().enumerate() {
                let active = b[1..].iter().filter(|v| **v != 0.0).count();
                let want = (1.0 + (active + 1) as f64 / n_kish).sqrt();
                let got = out[j * 3 + li];
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "row {i}, target {j}, point {li}: {got} against {want}"
                );
            }
        }
        differ += usize::from(out[..3] != out[3..]);
    }
    assert!(differ > 100, "the targets' slots differed on {differ} rows");
}

/// Where the Gram has no weight, every path point reads infinite (the
/// module doc), the one whose fit has no column through the origin
/// included, whose `df` is 0 (docs/PLAN.md task 220). Across 5,000 rows of
/// weight 0 at `λ = 0.9` both of Kish's sums fade until each holds only
/// the smallest doubles, where `λ` above one half keeps them: `W` some
/// `2.5e-323`, whose square is 0, and `Σ w²` one `2^-1074`, so `n_kish`
/// is 0 -- and `0 / 0` is no statistic.
#[test]
fn lasso_reads_infinite_where_the_gram_has_no_weight() {
    let mut m = lasso(1, false, vec![1e6, 0.0], 0.9);
    let mut rng = Rng(29);
    let mut out = Vec::new();
    for i in 0..5050 {
        let x = rng.row(2);
        let y = 2.0 * x[0] + 0.5 * rng.normal();
        let w = if i < 50 { 1.0 } else { 0.0 };
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
        assert!(m.error_inflation_into(&mut out));
        if i == 49 {
            assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        }
    }
    let fit = &m.coefficients().unwrap()[0];
    assert!(
        fit[0] == [0.0, 0.0] && fit[1].iter().all(|v| *v != 0.0),
        "{fit:?}"
    );
    assert!(
        m.n_eff() > 0.0 && m.n_eff() * m.n_eff() == 0.0,
        "{:e}",
        m.n_eff()
    );
    assert!(out.iter().all(|v| *v == f64::INFINITY), "{out:?}");
}

// ---- coefficient standard errors (docs/PLAN.md task 116, F) ----

/// The coefficient variances `ewridge` reports are `M = Σ̂⁻¹ / n_kish`
/// mapped to `coef`'s units over the noise: with no decay, unit weights and
/// a vanishing ridge, `(X'X)⁻¹`'s diagonal, the intercept's included, which
/// the plain and the standardized solve read alike. `X'X` over `[1, x]` is
/// summed here and inverted by a Gauss-Jordan elimination written here,
/// which no model runs.
#[test]
fn ewridge_coef_variance_is_the_least_squares_covariance() {
    let k = 2;
    let mut rng = Rng(31);
    let rows: Vec<(Vec<f64>, f64)> = (0..200)
        .map(|_| {
            let x = vec![3.0 * rng.normal() + 10.0, 0.5 * rng.normal() - 2.0];
            let y = 1.0 + 0.5 * x[0] - 2.0 * x[1] + rng.normal();
            (x, y)
        })
        .collect();
    let mut xtx = [[0.0f64; 3]; 3];
    for (x, _) in &rows {
        let z = [1.0, x[0], x[1]];
        for (r, zr) in z.iter().enumerate() {
            for (c, zc) in z.iter().enumerate() {
                xtx[r][c] += zr * zc;
            }
        }
    }
    let inv = invert3(xtx);
    for standardize in [false, true] {
        let mut c = cfg(k, f64::INFINITY);
        c.ridge = vec![1e-10];
        c.standardize = standardize;
        let mut m = model(c);
        for (x, y) in &rows {
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        let Some(CoefVariance::PerNoise(v)) = m.coef_variance() else {
            panic!("ewridge's variances are over the noise");
        };
        for (j, row) in inv.iter().enumerate() {
            let want = row[j];
            assert!(
                (v[0][j] - want).abs() <= 1e-6 * want,
                "standardize {standardize}, {j}: {} vs {want}",
                v[0][j]
            );
        }
    }
    // Without the kept systems there is nothing to read them from.
    let mut plain = EwRidge::new(cfg(k, f64::INFINITY)).unwrap();
    for (x, y) in &rows {
        plain.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(plain.coef_variance().is_none());
}

/// Each output slot reads its own target's Gram and its own feature set
/// (docs/PLAN.md task 218): two targets, the second missing one row in
/// three, so its Gram is over its own rows, and two feature sets, the first
/// feature alone and both. With no decay, unit weights and a vanishing
/// ridge, slot `(j, set)`'s variances are `(X'X)⁻¹`'s diagonal over target
/// `j`'s rows and the set's columns `[1, x_set]`, and NaN for a feature
/// outside the set; the plain and the standardized solve read alike.
#[test]
fn ewridge_coef_variance_is_per_target_and_per_feature_set() {
    let mut rng = Rng(37);
    let rows: Vec<(Vec<f64>, [Option<f64>; 2])> = (0..240)
        .map(|i| {
            let x = vec![3.0 * rng.normal() + 10.0, 0.5 * rng.normal() - 2.0];
            let y0 = 1.0 + 0.5 * x[0] - 2.0 * x[1] + rng.normal();
            let y1 = (i % 3 != 0).then(|| -x[0] + x[1] + rng.normal());
            (x, [Some(y0), y1])
        })
        .collect();
    // `(X'X)⁻¹`'s diagonal over target `j`'s rows: with the second feature
    // when `both`, else with that column dropped.
    let diag = |j: usize, both: bool| -> [f64; 3] {
        let mut xtx = [[0.0f64; 3]; 3];
        for (x, y) in &rows {
            if y[j].is_none() {
                continue;
            }
            let z = [1.0, x[0], x[1]];
            for (r, zr) in z.iter().enumerate() {
                for (c, zc) in z.iter().enumerate() {
                    xtx[r][c] += zr * zc;
                }
            }
        }
        if both {
            let inv = invert3(xtx);
            [inv[0][0], inv[1][1], inv[2][2]]
        } else {
            // The 2×2 block over `[1, x0]`, inverted by its adjugate.
            let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
            [xtx[1][1] / det, xtx[0][0] / det, f64::NAN]
        }
    };
    for standardize in [false, true] {
        let mut c = cfg(2, f64::INFINITY);
        c.n_targets = 2;
        c.ridge = vec![1e-10];
        c.feature_sets = vec![("one".into(), vec![0]), ("both".into(), vec![0, 1])];
        c.standardize = standardize;
        let mut m = model(c);
        for (x, y) in &rows {
            m.step(x, y, 1.0, 1.0);
        }
        let Some(CoefVariance::PerNoise(v)) = m.coef_variance() else {
            panic!("ewridge's variances are over the noise");
        };
        assert_eq!(v.len(), 4, "a slot per target and feature set");
        for j in 0..2 {
            for (set, both) in [false, true].into_iter().enumerate() {
                let want = diag(j, both);
                let got = &v[j * 2 + set];
                for i in 0..3 {
                    assert!(
                        (want[i].is_nan() && got[i].is_nan())
                            || (got[i] - want[i]).abs() <= 1e-6 * want[i],
                        "standardize {standardize}, target {j}, set {set}, slot {i}: {} vs {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
        // The two targets' rows differ, and so do their variances.
        assert!((v[0][1] - v[2][1]).abs() > 1e-3 * v[0][1], "{v:?}");
    }
}

/// `A⁻¹` of a 3×3 by Gauss-Jordan with partial pivoting.
fn invert3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 6]; 3];
    for r in 0..3 {
        m[r][..3].copy_from_slice(&a[r]);
        m[r][3 + r] = 1.0;
    }
    for col in 0..3 {
        let piv = (col..3)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap();
        m.swap(col, piv);
        let d = m[col][col];
        m[col].iter_mut().for_each(|v| *v /= d);
        for r in 0..3 {
            if r != col {
                let f = m[r][col];
                let pivot_row = m[col];
                for (dst, p) in m[r].iter_mut().zip(pivot_row) {
                    *dst -= f * p;
                }
            }
        }
    }
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        out[r].copy_from_slice(&m[r][3..]);
    }
    out
}
