//! `min_weight` withholds `ewridge`'s output alone
//! ([`EwRidge::set_min_weight_output_only`]; docs/PLAN.md task 278, the
//! user's decision of 2026-10-26): the Rust side of
//! `tests/test_readiness_sweep.py`'s
//! `test_ewridge_min_weight_moves_neither_its_noise_nor_its_first_solve`,
//! where `cargo mutants` sees it.

use super::*;

/// A seeded generator in `[-1, 1)`.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 2;
const HALF_LIFE: f64 = 20.0;

/// Two targets over two features, the second target's first rows null, so
/// its own first solve falls after the model's; the solves on a clock
/// cadence or a row cap, where the first solve's own rule decides the fit
/// until the cadence's next.
fn cfg(min_weight: f64, solve_every: f64, rows: u32) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(HALF_LIFE),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: true,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight,
        solve_every,
        max_rows_between_solves: rows,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

struct Row {
    x: [f64; K],
    y: [Option<f64>; 2],
    d: f64,
    w: f64,
}

/// 300 rows: uneven weights with zeros, a clock gap of 40 every 100 rows
/// that takes the weight back under every floor, and the second target null
/// on its first 30 rows and on one row in seven.
fn rows() -> Vec<Row> {
    let mut s = 278u64;
    (0..300)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let noise = 0.3 * lcg(&mut s);
            let y0 = 0.5 + 1.5 * x[0] - 0.75 * x[1] + noise;
            let y1 = -1.0 + 0.5 * x[0] + 2.0 * x[1] - noise;
            let d = match i {
                0 => 0.0,
                _ if i % 100 == 50 => 40.0,
                _ => 1.0,
            };
            let w = match i % 13 {
                5 => 0.0,
                _ => 1.0 + 0.5 * lcg(&mut s),
            };
            Row {
                x,
                y: [Some(y0), (i >= 30 && i % 7 != 3).then_some(y1)],
                d,
                w,
            }
        })
        .collect()
}

/// One row's prediction and the coefficients after it, as bits.
struct Seen {
    pred: Vec<f64>,
    coef: Vec<u64>,
}

/// Each row's step and the model after the stream.
fn run(mut m: EwRidge, rows: &[Row]) -> (Vec<Seen>, EwRidge) {
    let steps = rows
        .iter()
        .map(|r| {
            let pred = m.step(&r.x, &r.y, r.d, r.w).pred;
            let coef = m
                .coefficients()
                .map(|c| c.iter().flatten().map(|v| v.to_bits()).collect())
                .unwrap_or_default();
            Seen { pred, coef }
        })
        .collect();
    (steps, m)
}

/// The model a spec with `min_weight` builds: the flag, and each target's
/// own threshold.
fn gated(c: EwRidgeCfg) -> EwRidge {
    let floor = c.min_weight;
    let mut m = EwRidge::new(c).unwrap();
    m.set_target_min_weight(vec![floor; 2]).unwrap();
    m.set_min_weight_output_only(true);
    m
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// With the flag, at every `min_weight` at or above a row per unknown, the
/// residual variance, its weight and every coefficient are the model's at
/// its own floor, on every row, under every cadence; and a prediction it
/// reports is that model's. Without the flag -- a state written before
/// schema 64 -- `min_weight` holds both back: the variance and, under a
/// cadence, the coefficients move.
#[test]
fn min_weight_moves_neither_the_noise_nor_the_first_solve() {
    let rows = rows();
    let own = (K + 1) as f64;
    for (solve_every, cap) in [(0.0, 1), (25.0, u32::MAX), (1e9, 40)] {
        let (free, free_m) = run(EwRidge::new(cfg(own, solve_every, cap)).unwrap(), &rows);
        for mw in [own, 8.0, 15.0, 30.0] {
            let what = format!("solve_every {solve_every}, cap {cap}, min_weight {mw}");
            let (out, m) = run(gated(cfg(mw, solve_every, cap)), &rows);
            assert_eq!(bits(&m.sig2), bits(&free_m.sig2), "{what}: sig2");
            assert_eq!(bits(&m.wsig), bits(&free_m.wsig), "{what}: wsig");
            assert_eq!(m.beta, free_m.beta, "{what}: the last fit");
            let mut withheld = 0;
            for (i, (a, b)) in out.iter().zip(&free).enumerate() {
                assert_eq!(a.coef, b.coef, "{what}: row {i}");
                for (p, q) in a.pred.iter().zip(&b.pred) {
                    if p.is_finite() {
                        assert_eq!(p.to_bits(), q.to_bits(), "{what}: row {i}");
                    } else if q.is_finite() {
                        withheld += 1;
                    }
                }
            }
            if mw > own {
                assert!(
                    withheld > 10,
                    "{what}: withheld {withheld}, so it proves nothing"
                );
            }
            // The old rule moves the variance wherever the gate withheld a
            // prediction with a target.
            let mut old = EwRidge::new(cfg(mw, solve_every, cap)).unwrap();
            old.set_target_min_weight(vec![mw; 2]).unwrap();
            let (old_out, old_m) = run(old, &rows);
            if mw > own {
                assert_ne!(
                    bits(&old_m.sig2),
                    bits(&free_m.sig2),
                    "{what}: the old rule"
                );
                if solve_every > 0.0 {
                    let moved = old_out
                        .iter()
                        .zip(&free)
                        .filter(|(a, b)| a.coef != b.coef)
                        .count();
                    let largest = old_out
                        .iter()
                        .zip(&free)
                        .flat_map(|(a, b)| a.coef.iter().zip(&b.coef))
                        .map(|(p, q)| (f64::from_bits(*p) - f64::from_bits(*q)).abs())
                        .filter(|d| d.is_finite())
                        .fold(0.0, f64::max);
                    println!(
                        "{what}: the old rule moved coef on {moved} rows, by at most {largest:.3e}"
                    );
                    assert!(moved > 0, "{what}: the old rule moved no coefficient");
                }
            }
        }
    }
}

/// The variance against its definition: with `v_t = w_t Π_{s>t} λ_s`,
/// `wsig = Σ v_t` and `sig2 = Σ v_t r_t² / Σ v_t` over the rows with a target,
/// a positive weight and a prediction from the model at its own floor --
/// here the model with no `min_weight` above it, whose predictions are its
/// steps'. The gated model's variance is that, within rounding.
#[test]
fn the_noise_is_the_ew_mean_square_of_the_floors_residuals() {
    let rows = rows();
    let own = (K + 1) as f64;
    let (free, _) = run(EwRidge::new(cfg(own, 0.0, 1)).unwrap(), &rows);
    let (_, m) = run(gated(cfg(30.0, 0.0, 1)), &rows);
    for j in 0..2 {
        let (mut w_sum, mut sq) = (0.0f64, 0.0f64);
        for (r, step) in rows.iter().zip(&free) {
            let lam = Decay::Halflife(HALF_LIFE).factor(r.d);
            w_sum *= lam;
            sq *= lam;
            if let (Some(y), true) = (r.y[j], r.w > 0.0 && step.pred[j].is_finite()) {
                w_sum += r.w;
                sq += r.w * (y - step.pred[j]).powi(2);
            }
        }
        let want = sq / w_sum;
        assert!(
            (m.wsig[j] - w_sum).abs() <= 1e-12 * w_sum,
            "{j}: {} vs {w_sum}",
            m.wsig[j]
        );
        assert!(
            (m.sig2[j] - want).abs() <= 1e-12 * want,
            "{j}: {} vs {want}",
            m.sig2[j]
        );
    }
}
