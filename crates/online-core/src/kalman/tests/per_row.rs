//! The per-row paths task 250 made faster, each held to the form it
//! replaced, to the bit, and shown to run: the gap noise's kept factors
//! and the map from the anchor written across the slots. A file of its own
//! because `kalman.rs` stands near the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

const K: usize = 6;

/// `n` rows of `K` features -- feature 2 always 0, so its scale is never
/// usable, with or without an intercept, feature 4 far from 0 against a small spread -- a target, and the
/// clock step before each row: 1 on a regular clock, else irregular.
#[allow(clippy::type_complexity)]
fn stream(n: usize, regular: bool, seed: u64) -> Vec<([f64; K], f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let mut x = [0.0; K];
            for (j, v) in x.iter_mut().enumerate() {
                *v = match j {
                    2 => 0.0,
                    4 => 1e4 + 0.01 * lcg(&mut s),
                    _ => lcg(&mut s) * (1.0 + j as f64),
                };
            }
            let y = x[0] - 0.5 * x[1] + 2e-3 * (x[4] - 1e4) + 0.1 * lcg(&mut s);
            let d = if i == 0 {
                0.0
            } else if regular {
                1.0
            } else {
                0.25 + lcg(&mut s).abs() * 3.0
            };
            (x, y, d)
        })
        .collect()
}

fn cfg_with(revert: Vec<f64>, fit_intercept: bool, standardize: bool) -> KalmanCfg {
    KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: K,
        n_targets: 1,
        fit_intercept,
        decay: Decay::Halflife(200.0),
        half_life: vec![50.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 5.0,
        revert_half_life: revert,
        standardize,
    }
}

/// The reversions the gap factors are taken under: the random walk, one
/// half-life for every slot, and one per slot with a walk among them.
fn reversions() -> Vec<Vec<f64>> {
    let per_slot: Vec<f64> = (0..=K)
        .map(|i| {
            if i == 3 {
                f64::INFINITY
            } else {
                40.0 * (i + 1) as f64
            }
        })
        .collect();
    vec![vec![f64::INFINITY], vec![30.0], per_slot]
}

fn bits(m: &Kalman) -> (Vec<u64>, Vec<u64>) {
    (
        m.p.iter().flatten().map(|v| v.to_bits()).collect(),
        m.beta.iter().flatten().map(|v| v.to_bits()).collect(),
    )
}

/// What the cache hands back is the call's value for each slot, to the
/// bit, on a hit as on a miss, and a hit is taken only on the same bits.
#[test]
fn the_gap_factors_are_gap_noise_to_the_bit() {
    for revert in reversions() {
        let cfg = cfg_with(revert.clone(), true, true);
        let mut f = GapFactors::default();
        let gaps = [1.0, 1.0, 0.7, 0.7, 1.0, 1e-9, 1e4, 1e4, -0.0, 0.0];
        for &d in &gaps {
            let got = f.of(&cfg, d).to_vec();
            assert_eq!(got.len(), cfg.k_total());
            for (i, g) in got.iter().enumerate() {
                assert_eq!(
                    g.to_bits(),
                    cfg.gap_noise(i, d).to_bits(),
                    "{revert:?} d={d} slot {i}"
                );
            }
        }
        // 1, 0.7 and 1e4 repeated; -0.0 and 0.0 are two keys.
        assert_eq!(f.hits, 3, "{revert:?}");
    }
}

/// A filter on a regular clock charges the same gap on every row after
/// the first, and takes the kept factors on each; on an irregular clock it
/// never does. Either way its `P` and `b` are, to the bit, those of a twin
/// whose cache is emptied before every row, so every row computes.
#[test]
fn a_regular_clock_takes_the_kept_gap_factors_and_moves_no_bit() {
    for revert in reversions() {
        for regular in [true, false] {
            let n = 300;
            let rows = stream(n, regular, 5);
            let mut kept = Kalman::new(cfg_with(revert.clone(), true, true)).unwrap();
            let mut fresh = kept.clone();
            for (i, (x, y, d)) in rows.iter().enumerate() {
                fresh.gap_factors = GapFactors::default();
                let a = kept.step(x, &[Some(*y)], *d, 1.0).pred[0];
                let b = fresh.step(x, &[Some(*y)], *d, 1.0).pred[0];
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{revert:?} regular={regular} row {i}"
                );
                assert_eq!(
                    bits(&kept),
                    bits(&fresh),
                    "{revert:?} regular={regular} row {i}"
                );
            }
            // Row 0 has no gap to charge; row 1 computes the factors of a
            // gap of 1, every later row takes them.
            let want = if regular { n as u64 - 2 } else { 0 };
            assert_eq!(kept.gap_factors.hits, want, "{revert:?} regular={regular}");
            assert_eq!(fresh.gap_factors.hits, 0);
        }
    }
}

/// A gap that spans rows that observed nothing is charged whole on the
/// row that observes: its own bits, a miss, and the same numbers as a
/// filter that kept nothing.
#[test]
fn a_gap_across_unobserved_rows_misses_and_stays_exact() {
    let rows = stream(200, true, 9);
    let mut kept = Kalman::new(cfg_with(vec![30.0], true, true)).unwrap();
    let mut fresh = kept.clone();
    for (i, (x, y, d)) in rows.iter().enumerate() {
        let y = (i % 5 != 3 && i % 5 != 4).then_some(*y);
        fresh.gap_factors = GapFactors::default();
        kept.step(x, &[y], *d, 1.0);
        fresh.step(x, &[y], *d, 1.0);
        assert_eq!(bits(&kept), bits(&fresh), "row {i}");
    }
    // Per five rows: gaps of 1, 1, 3 (after two nulls); each change of gap
    // is a miss, so hits are the rows whose gap repeats the last one's.
    assert!(kept.gap_factors.hits > 0 && kept.gap_factors.hits < 200);
}

/// A restored filter starts with nothing kept (the cache is not state),
/// and continues to the bit as the filter it was saved from, which kept
/// its factors.
#[test]
fn a_restored_filter_starts_with_no_gap_factors() {
    let rows = stream(160, true, 13);
    let mut m1 = Kalman::new(cfg_with(vec![30.0], true, true)).unwrap();
    for (x, y, d) in &rows[..80] {
        m1.step(x, &[Some(*y)], *d, 1.0);
    }
    assert!(m1.gap_factors.bits.is_some());
    let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
    let mut m2 = Kalman::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(m2.gap_factors.bits.is_none() && m2.gap_factors.hits == 0);
    assert!(
        m1.gap_factors == m2.gap_factors,
        "a cache is no part of equality"
    );
    for (i, (x, y, d)) in rows[80..].iter().enumerate() {
        let a = m1.step(x, &[Some(*y)], *d, 1.0).pred[0];
        let b = m2.step(x, &[Some(*y)], *d, 1.0).pred[0];
        assert_eq!(a.to_bits(), b.to_bits(), "row {}", 80 + i);
        assert_eq!(bits(&m1), bits(&m2), "row {}", 80 + i);
    }
    assert_eq!(m2.gap_factors.hits, 79, "the restored filter keeps its own");
}

/// The map a slot at a time, as `map_into` was written before it ran
/// across the slots: the reference it is held to.
fn map_by_slot(m: &Kalman, live: &[bool], a: &mut [f64], c: &mut [f64]) -> f64 {
    let off = usize::from(m.cfg.fit_intercept);
    let mut drift = 0.0f64;
    for i in 0..a.len() {
        if i < off || !m.cfg.standardize {
            a[i] = 1.0;
            c[i] = 0.0;
            continue;
        }
        let sa = m.anchor_scale[i];
        a[i] = m.current_scale(i).0 / sa;
        c[i] = if off == 1 {
            let (hi, lo) = m.stats.mean_pair(i);
            ((hi - m.anchor_hi[i]) + (lo - m.anchor_lo[i])) / sa
        } else {
            0.0
        };
        if live[i] {
            drift = drift.max(a[i] - 1.0).max(1.0 / a[i] - 1.0).max(c[i].abs());
        }
    }
    drift
}

/// `map_into` across the slots is the map a slot at a time, to the bit --
/// `a`, `c` and the drift -- after every row of a stream, with and without
/// an intercept and `standardize`, over the slots that hold anything, all
/// of them and none; through a slot whose scale is never usable (a constant
/// feature, both branches of the scale's select taken on one row) and one
/// whose mean is far from 0.
#[test]
fn the_map_across_the_slots_is_the_map_a_slot_at_a_time() {
    for (fit_intercept, standardize) in [(true, true), (false, true), (true, false), (false, false)]
    {
        let mut m = Kalman::new(cfg_with(vec![30.0], fit_intercept, standardize)).unwrap();
        let k = m.cfg.k_total();
        let off = usize::from(fit_intercept);
        let mut moved = false;
        let mut both = false;
        for (i, (x, y, d)) in stream(250, false, 21).iter().enumerate() {
            m.step(x, &[Some(*y)], *d, 1.0);
            m.mark_live();
            let usable: Vec<bool> = (off..k).map(|j| m.current_scale(j).1).collect();
            both |= usable.iter().any(|u| *u) && usable.iter().any(|u| !*u);
            for live in [m.live.clone(), vec![true; k], vec![false; k]] {
                let (mut a1, mut c1) = (vec![f64::NAN; k], vec![f64::NAN; k]);
                let (mut a2, mut c2) = (vec![f64::NAN; k], vec![f64::NAN; k]);
                let d1 = m.map_into(&live, &mut a1, &mut c1);
                let d2 = map_by_slot(&m, &live, &mut a2, &mut c2);
                let b = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let what = format!("intercept={fit_intercept} standardize={standardize} row {i}");
                assert_eq!(b(&a1), b(&a2), "{what}: a");
                assert_eq!(b(&c1), b(&c2), "{what}: c");
                assert_eq!(d1.to_bits(), d2.to_bits(), "{what}: drift");
                moved |= standardize && a1.iter().any(|a| *a != 1.0) && d1 > 0.0;
            }
        }
        if standardize {
            assert!(
                moved,
                "intercept={fit_intercept}: the map moved off the identity"
            );
            assert!(
                both,
                "intercept={fit_intercept}: usable and unusable scales on one row"
            );
        }
    }
}
