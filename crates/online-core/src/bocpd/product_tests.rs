//! The tests of [`Bocpd`]'s diagonal predictive taking the features' logarithms
//! as the logs of products (`ln_sum_prod`), against the single-log form it
//! replaced, and of the length terms it holds by weight. A file of their own
//! because `bocpd.rs` is at the size a source file may be.

use super::tests::{Normals, cfg, learn_reference, step_bits};
use super::*;
use crate::OnlineModel as _;

/// The diagonal predictive as it was before the features' logarithms were
/// taken as products: per feature, `ln(νₙπ var)` and `ln(1 + z²/νₙ)`, one
/// logarithm each, summed in feature order. The reference for what the
/// product form may move, and for what it may not (one feature).
fn single_log_predictive(
    m: &Bocpd,
    run: &Run,
    x: &[f64],
    mu0: &[f64],
    psi0: &[f64],
) -> Option<(f64, f64)> {
    let d = m.cfg.n_features;
    let (k0, nu0) = (m.cfg.prior_kappa, m.cfg.nu0());
    let (kn, nun) = (k0 + run.n, nu0 + run.n);
    let gammas = gamma_diff(nun, 1.0);
    let (mut lp, mut mode) = (0.0, 0.0);
    for i in 0..d {
        let dm = run.mean[i] - mu0[i];
        let psin = psi0[i * d + i] + run.m2[i] + k0 * run.n / kn * dm * dm;
        let var = psin * (kn + 1.0) / (nun * kn);
        if var.is_nan() || var <= 0.0 || nun <= 0.0 {
            return None;
        }
        let z = (x[i] - m.run_mean_at(run, mu0, i)) / var.sqrt();
        let base = gammas - 0.5 * (nun * std::f64::consts::PI * var).ln();
        lp += base - 0.5 * (nun + 1.0) * (1.0 + z * z / nun).ln();
        mode += base;
    }
    Some((lp, mode))
}

/// A model of `d` features a few hundred rows in, over a stream whose
/// features differ in scale by `spread` (`10^spread` between the first
/// and the last), and its runs.
fn model_with_runs(d: usize, spread: f64, seed: u64) -> (Bocpd, Normals) {
    let mut m = Bocpd::new(BocpdCfg {
        max_run: 40,
        prune_below: 1e-6,
        ..cfg(d)
    })
    .unwrap();
    let mut n = Normals::new(seed);
    let scale = |i: usize| 10f64.powf(spread * i as f64 / d.max(2).saturating_sub(1) as f64);
    for t in 0..300 {
        let level = if t % 120 < 60 { 0.0 } else { 3.0 };
        let x: Vec<f64> = (0..d).map(|i| scale(i) * (level + n.normal())).collect();
        m.step(&x, &[], 1.0, 1.0);
    }
    (m, n)
}

/// One feature keeps the single logarithm of the original form: its
/// per-run predictive, with the held length terms and without them, is
/// the original expression to the bit.
#[test]
fn one_feature_is_the_single_log_predictive_to_the_bit() {
    let (m, mut n) = model_with_runs(1, 0.0, 17);
    assert!(m.runs.len() > 20);
    let (mu0, psi0) = (m.mu0(), m.psi0());
    let mut held = LengthTerms::default();
    let mut checked = 0;
    for _ in 0..50 {
        let x = [3.0 * n.normal()];
        held.begin(m.runs.len(), 1.0, false);
        for run in &m.runs {
            let want = single_log_predictive(&m, run, &x, &mu0, &psi0).unwrap();
            for cache in [None, Some(&mut held)] {
                let got = m
                    .log_predictive(run, &x, &mu0, &psi0, cache, &mut RunWork::default())
                    .unwrap();
                assert_eq!(
                    (got.0.to_bits(), got.1.to_bits()),
                    (want.0.to_bits(), want.1.to_bits())
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 2_000 && held.hits > 0, "{checked} {}", held.hits);
    // And the whole recursion: a one-feature model's steps are those of
    // the reference, which has no held terms at all.
    let mut fast = Bocpd::new(BocpdCfg {
        max_run: 40,
        prune_below: 1e-6,
        ..cfg(1)
    })
    .unwrap();
    let mut slow = fast.clone();
    let mut n = Normals::new(5);
    for t in 0..400 {
        let x = [if t % 100 < 50 { 0.0 } else { 4.0 } + n.normal()];
        let a = fast.step(&x, &[], 1.0, 1.0);
        let b = learn_reference(&mut slow, &x, &[], 1.0, 1.0);
        assert_eq!(step_bits(&a), step_bits(&b), "row {t}");
    }
}

/// Two features or more -- both sides of the guard, and each size around
/// the block of 8 the products take (7, 8 and 9; 16 and 17) -- the
/// product form is the single-log form within a few last bits of the
/// terms' sum: `|Δ| <= 8 · 2⁻⁵² · (1 + Σ|terms|)`, with the model's
/// features spread over `10^spread` in scale (the products' ranges), and
/// the held terms give the values of the ones made on the spot.
#[test]
fn the_product_form_is_the_single_log_form_within_the_sum_of_its_terms() {
    let mut worst = 0.0f64;
    for d in [2usize, 3, 7, 8, 9, 15, 16, 17, 24, 25] {
        for spread in [0.0, 6.0, 40.0] {
            let (m, mut n) = model_with_runs(d, spread, d as u64);
            let (mu0, psi0) = (m.mu0(), m.psi0());
            let mut held = LengthTerms::default();
            for row in 0..20 {
                let level = if row % 7 == 0 { 8.0 } else { 0.5 };
                let x: Vec<f64> = (0..d)
                    .map(|i| 10f64.powf(spread * i as f64 / (d - 1) as f64) * level * n.normal())
                    .collect();
                held.begin(m.runs.len(), 1.0, true);
                for run in &m.runs {
                    let want = single_log_predictive(&m, run, &x, &mu0, &psi0);
                    let alone =
                        m.log_predictive(run, &x, &mu0, &psi0, None, &mut RunWork::default());
                    let with = m.log_predictive(
                        run,
                        &x,
                        &mu0,
                        &psi0,
                        Some(&mut held),
                        &mut RunWork::default(),
                    );
                    let (Some(want), Some(alone), Some(with)) = (want, alone, with) else {
                        panic!("d {d}, spread {spread}: a predictive refused");
                    };
                    // A hit is the value a miss computes.
                    assert_eq!(
                        (alone.0.to_bits(), alone.1.to_bits()),
                        (with.0.to_bits(), with.1.to_bits()),
                        "d {d}"
                    );
                    // The sum of the terms the two forms add up.
                    let nun = m.cfg.nu0() + run.n;
                    let terms: f64 = want.0.abs() + want.1.abs() + 0.5 * (nun + 1.0) * d as f64;
                    for (got, want, what) in [(alone.0, want.0, "lp"), (alone.1, want.1, "mode")] {
                        let bound = 8.0 * f64::EPSILON * (1.0 + terms);
                        let off = (got - want).abs();
                        worst = worst.max(off / bound);
                        assert!(
                            off <= bound,
                            "d {d}, spread {spread}, {what}: {got} against {want} (bound {bound})"
                        );
                    }
                }
            }
        }
    }
    eprintln!("the product form's worst: {worst:.3} of its bound");
}

/// A block whose product would leave the normal range takes its logs one
/// at a time, so a feature of huge or tiny scale costs no `inf`, 0 or
/// subnormal in a product: the sum is the single-log sum's, and a block
/// of mixed ranges is the sum of its two kinds of block.
#[test]
fn the_products_guard_their_range_and_fall_back_to_single_logs() {
    let single = |v: &[f64]| v.iter().map(|x| x.ln()).sum::<f64>();
    // Every block outside the range: the same logs, the same order within
    // a block, so equal to the bit when the blocks sum to the same.
    for v in [1e100f64, 1e-100, 1e300, 1e-300] {
        let all = vec![v; 8];
        assert_eq!(ln_sum_prod(&all).to_bits(), single(&all).to_bits(), "{v}");
    }
    // Past the overflow a product has, and past what a subnormal keeps.
    let huge: Vec<f64> = (0..20).map(|i| 1e150 * (1.0 + i as f64)).collect();
    let tiny: Vec<f64> = (0..20).map(|i| 1e-150 * (1.0 + i as f64)).collect();
    for v in [&huge, &tiny] {
        let got = ln_sum_prod(v);
        let want = single(v);
        assert!(got.is_finite());
        assert!((got - want).abs() <= 1e-12 * want.abs(), "{got} {want}");
    }
    // Mixed: a block of normal values beside a block that overflows.
    let mut mixed = vec![2.0; 8];
    mixed.extend(vec![1e200; 8]);
    mixed.extend(vec![3.0; 3]);
    let (got, want) = (ln_sum_prod(&mixed), single(&mixed));
    assert!((got - want).abs() <= 1e-13 * want.abs(), "{got} {want}");
    // A NaN anywhere is a NaN, and an empty sum is zero.
    assert!(ln_sum_prod(&[1.0, f64::NAN, 2.0]).is_nan());
    assert_eq!(ln_sum_prod(&[]).to_bits(), 0f64.to_bits());
    // Every length around the blocks: the sum of the logs, within the
    // bound the model test above states.
    for len in (1..=34).chain([63, 64, 65]) {
        let v: Vec<f64> = (0..len)
            .map(|i| 0.3 + 1.7 * ((i * 7) % 11) as f64)
            .collect();
        let (got, want) = (ln_sum_prod(&v), single(&v));
        let terms: f64 = v.iter().map(|x| x.ln().abs()).sum();
        assert!(
            (got - want).abs() <= 8.0 * f64::EPSILON * (1.0 + terms),
            "{len}: {got} {want}"
        );
    }
}

/// The diagonal form's `ln(π(κₙ+1)/κₙ)` is held by the run's weight, not
/// by `νₙ`: two weights so small beside `ν₀` that they round to one `νₙ`
/// are two `κₙ` when `κ₀` is small, so each keeps its own, and a model
/// that holds nothing (restored) takes the same steps to the bit as one
/// that holds everything.
#[test]
fn the_held_ln_c_is_a_function_of_the_weight_and_not_of_nu() {
    let (k0, nu0) = (1e-8, 2.0);
    let (n1, n2) = (1e-17, 3e-17);
    let nu = |n: f64| nu0 + n;
    let kn = |n: f64| k0 + n;
    assert_eq!(nu(n1).to_bits(), nu(n2).to_bits(), "one nu");
    assert_ne!(kn(n1).to_bits(), kn(n2).to_bits(), "two kappa");
    let ln_c = |n: f64| (std::f64::consts::PI * (kn(n) + 1.0) / kn(n)).ln();
    assert_ne!(ln_c(n1).to_bits(), ln_c(n2).to_bits(), "two ln c");
    let mut t = LengthTerms::default();
    for pass in 0..3 {
        t.begin(4, 1.0, true);
        for n in [n1, n2, n1, n2, 0.0, 1.0] {
            let (g, l) = t.gamma_and_ln_c(n, nu(n), kn(n));
            assert_eq!(g.to_bits(), gamma_diff(nu(n), 1.0).to_bits(), "{pass} {n}");
            assert_eq!(l.to_bits(), ln_c(n).to_bits(), "{pass} {n}");
        }
    }
    assert_eq!((t.misses, t.hits), (4, 14));
    // A NaN weight is computed and never stored.
    let before = (t.misses, t.hits, t.filled);
    let (g, l) = t.gamma_and_ln_c(f64::NAN, f64::NAN, f64::NAN);
    assert!(g.is_nan() && l.is_nan());
    assert_eq!((t.misses, t.hits, t.filled), before);
    // The model: tiny weights, so many runs share a `νₙ`.
    let c = BocpdCfg {
        prior_kappa: k0,
        max_run: 30,
        prune_below: 1e-9,
        ..cfg(3)
    };
    let mut fast = Bocpd::new(c).unwrap();
    let mut slow = fast.clone();
    let mut n = Normals::new(3);
    for row in 0..300 {
        let w = [1e-17, 3e-17, 2e-17, 5e-17][row % 4];
        let x = [n.normal(), n.normal(), n.normal()];
        let a = fast.step(&x, &[], 1.0, w);
        slow.gammas = LengthTerms::default();
        let b = slow.step(&x, &[], 1.0, w);
        assert_eq!(step_bits(&a), step_bits(&b), "row {row}");
    }
    assert!(fast.gammas.hits > 0);
    assert_eq!(
        rmp_serde::to_vec(&fast).unwrap(),
        rmp_serde::to_vec(&slow).unwrap()
    );
}
