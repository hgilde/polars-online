//! The noise gate's statistic is exact by theory
//! (docs/WARMUP-AND-CONVERGENCE.md §2.1): `h(x) = x' Σ̂⁻¹ x / n_kish` is the
//! estimation variance of a prediction over the noise, for a design whose
//! covariance is constant over the memory window, and `edf / n_kish` is its
//! average. These are the identities the implementation must satisfy, each
//! of which fails loudly if the squared-weight sum, the weights or the ridge
//! are wired wrongly (§7.4). They verify the wiring; nothing here is tuned.

use online_core::{Decay, EwRidge, EwRidgeCfg, OnlineModel, TargetGaps};

fn cfg(k: usize, halflife: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(halflife),
        ridge: vec![1e-8],
        feature_sets: vec![],
        standardize: false,
        ridge_decay: false,
        coef_prior: None,
        session_shrink: None,
        long_halflife: None,
        min_periods: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
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
    assert!(m.row_error_inflation_into(x, &mut out));
    out[0] * out[0] - 1.0
}

/// The stream-level `edf / n_kish`, the same way.
fn h_mean_of(m: &EwRidge) -> f64 {
    let mut out = Vec::new();
    assert!(m.error_inflation_into(&mut out));
    out[0] * out[0] - 1.0
}

fn steady_n_kish(halflife: f64) -> f64 {
    let lam = (-(1.0 / halflife)).exp2();
    (1.0 + lam) / (1.0 - lam)
}

/// (i) On a stationary design the mean of the per-row `h` is `edf / n_kish`:
/// the per-row statistic and the stream-level one are the same quantity
/// seen two ways, and `n_kish` reaches `(1 + λ) / (1 − λ)`.
#[test]
fn the_mean_of_h_over_a_stationary_design_is_edf_over_n_kish() {
    let (k, halflife) = (4, 20.0);
    let mut m = model(cfg(k, halflife));
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
    let n_kish = steady_n_kish(halflife);
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
    let (k, halflife, sigma) = (4, 4.0, 1.0);
    let mut m = model(cfg(k, halflife));
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
    // by half at this halflife (0.73 against 0.47 observed), never
    // understates it. Exactness per row is what `G₂` would buy (§2.1).
    let excess = observed * observed - 1.0;
    assert!(
        mean_h >= excess && mean_h <= 2.0 * excess,
        "per-row mean h {mean_h} vs observed excess {excess}"
    );
    // Well away from 1: at halflife 4 the fit is visibly noisy.
    assert!(gate > 1.1, "{gate}");
}

/// (iii) Weights enter through Kish's `n` exactly: scaling every weight by
/// a constant changes nothing, and splitting every row into two halves at
/// the same clock leaves the Gram and `n_eff` where they were while `n_kish`
/// doubles, so `h` halves.
#[test]
fn weights_enter_through_kish_n_exactly() {
    let (k, halflife) = (3, 30.0);
    let mut whole = model(cfg(k, halflife));
    let mut scaled = model(cfg(k, halflife));
    let mut split = model(cfg(k, halflife));
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
    let (k, halflife) = (3, 50.0);
    let mut m = model(cfg(k, halflife));
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
    origin.add_intercept = false;
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
    c.min_periods = (k + 1) as f64;
    let m = model(c);
    let mut out = Vec::new();
    assert!(m.error_inflation_into(&mut out));
    assert!(out[0].is_infinite());
    assert!(m.support_coef().is_none());
    assert!(m.row_error_inflation_into(&[1.0, 2.0], &mut out));
    assert!(out[0].is_infinite());
}
