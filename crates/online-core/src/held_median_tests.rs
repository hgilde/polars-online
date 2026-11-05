//! The held median (task 323's follow-up): the tree's read is the
//! definition's, its weights are the live ones after a rebuild, and it
//! agrees with the backward-decayed weights `Rows::scale_weights` gives
//! wherever their crossing of half is clear of rounding.

use super::*;
use crate::huber_refit::{RefitBuf, Rows, counted_weights, weighted_median};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn fallbacks() -> usize {
    FALLBACKS.with(std::cell::Cell::get)
}

/// The definition, from the weights held: in ascending `|e|`, ties in
/// arrival order, the first at which the running sum reaches half the
/// sum, both in that order (`weighted_median_sorted`).
fn definition(h: &HeldMedian, t: usize) -> Option<f64> {
    let (e, w): (Vec<f64>, Vec<f64>) = h.weights(t).into_iter().unzip();
    let m = crate::huber_refit::weighted_median_sorted(&e, &w);
    (!m.is_nan()).then_some(m)
}

#[test]
fn frexp_is_exact_on_normals_and_subnormals() {
    let mut s = 3u64;
    for _ in 0..2000 {
        let x = lcg(&mut s) * 2f64.powi((lcg(&mut s) * 2000.0) as i32 - 1000);
        let (m, e) = frexp(x);
        assert!((0.5..1.0).contains(&m), "{x}: {m}");
        assert_eq!(m * 2f64.powi(e as i32), x, "{x}");
    }
    for x in [
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 3.0,
        5e-324,
        1.0,
        0.5,
        f64::MAX,
    ] {
        let (m, e) = frexp(x);
        assert!((0.5..1.0).contains(&m), "{x}: {m}");
        // Rebuilt in two exact steps, the subnormals through 2^64.
        let back = m * 2f64.powi((e + 64).min(1023) as i32)
            / 2f64.powi(64.min((e + 64).min(1023) - e) as i32);
        assert_eq!(back, x, "{x}");
    }
}

#[test]
fn a_run_by_squaring_neither_underflows_nor_moves_a_power_of_two() {
    let half = Split::pow(0.5, 1_000_000);
    assert_eq!((half.m, half.k), (0.5, 1 - 1_000_000));
    assert_eq!(Split::pow(0.37, 1), Split::of(0.37));
    assert_eq!(Split::pow(0.9, 0), Split::ONE);
    let p = Split::pow(0.9, 300);
    let want = 0.9f64.powi(300);
    let got = p.m * 2f64.powi(p.k as i32);
    assert!((got - want).abs() <= 1e-13 * want, "{got} {want}");
}

/// Random streams -- decays near 1 and a gap now and then, runs of
/// ageings, untaken rows, zero `|e|`, ties, uneven weights, a half-life of
/// a row that rescales the weights hundreds of times -- read at every row:
/// the tree's read is the definition's, to the bit, and the definition's
/// walk is taken on few reads.
#[test]
fn the_trees_read_is_the_definitions_on_every_row() {
    let (mut reads, before) = (0usize, fallbacks());
    for seed in 0..40u64 {
        let mut s = seed * 7919 + 1;
        let fast = seed % 4 == 0;
        let mut h = HeldMedian::new(2);
        for i in 0..800 {
            if lcg(&mut s) < 0.1 {
                h.begin_run();
                let lam = if fast { 0.5 } else { 0.99 };
                h.run(lam, 1 + (lcg(&mut s) * 2000.0) as u64);
            }
            h.decay(if fast { 0.5 } else { 1.0 - 0.01 * lcg(&mut s) });
            for t in 0..2 {
                let e = match (lcg(&mut s) * 20.0) as u32 {
                    0 => f64::NAN,
                    1 => 0.0,
                    2 => 1.0,
                    _ => -lcg(&mut s).ln(),
                };
                let w = if t == 0 { 1.0 } else { 0.25 + lcg(&mut s) };
                h.insert(t, e, w);
                assert_eq!(h.median(t), definition(&h, t), "seed {seed} row {i} t {t}");
                reads += 1;
            }
        }
    }
    let walked = fallbacks() - before;
    println!("{walked} of {reads} reads took the definition's walk");
    assert!(walked * 100 < reads, "{walked} of {reads}");
}

/// No decay and weights of 1: every sum exact, and no read walks.
#[test]
fn exact_weights_never_walk() {
    let before = fallbacks();
    let mut h = HeldMedian::new(1);
    let mut s = 9u64;
    for _ in 0..500 {
        h.decay(1.0);
        h.insert(0, (lcg(&mut s) * 8.0).floor(), 1.0);
        assert_eq!(h.median(0), definition(&h, 0));
    }
    assert_eq!(fallbacks(), before);
}

/// A buffer saved and restored mid-warm-up rebuilds the tree from its rows
/// and runs to the same weights, to the bit, and reads the same scale on
/// every row after; and against `weighted_median` at the backward-decayed
/// weights `Rows::scale_weights` gives the rebuild, the read is the same
/// wherever their crossing of half is clear of rounding: where the two
/// differ they are neighbours in `|e|`, and the weight below the boundary
/// between them is within `4 (n + 256) u` of half, by a compensated sum.
#[test]
fn a_rebuilt_buffer_reads_the_live_one_and_the_backward_weights_agree() {
    let (mut reads, mut differ, mut worst) = (0usize, 0usize, 0.0f64);
    for seed in 0..30u64 {
        let mut s = seed * 104_729 + 5;
        let mut b = RefitBuf::keeping_resid();
        let cap = 600;
        let lam = if seed % 3 == 0 { 0.7 } else { 0.995 };
        for _ in 0..cap - 1 {
            if lcg(&mut s) < 0.15 {
                let n = 1 + (lcg(&mut s) * 30.0) as u32;
                for _ in 0..n {
                    b.age(lam, lam, cap);
                }
            }
            let e = if lcg(&mut s) < 0.05 {
                f64::NAN
            } else {
                -lcg(&mut s).ln()
            };
            b.learn_scored(&[0.0], 0.0, e, 0.5 + lcg(&mut s), lam, lam, cap);
            let live = b.warm_scale().expect("the buffer reads its |e|");
            let mut back = b.clone();
            back.rebuild_median(1);
            assert_eq!(back.warm_scale(), Some(live), "seed {seed}");
            let held = b.clone().take();
            let rows = Rows::new(&held.rows, &held.ages, 1);
            let e = held.resid.unwrap();
            let v = counted_weights(&rows, &e);
            let backward = weighted_median(&e, &v) / crate::robust::NORMAL_Q3;
            let backward = (backward > 0.0 && backward.is_finite()).then_some(backward);
            reads += 1;
            if live != backward {
                differ += 1;
                // The boundary's share of the weight, compensated.
                let mut idx: Vec<usize> = (0..e.len()).filter(|&i| v[i] > 0.0).collect();
                idx.sort_by(|&p, &q| e[p].total_cmp(&e[q]));
                let total: f64 = idx.iter().map(|&i| v[i]).sum();
                let (a, z) = (live.unwrap(), backward.unwrap());
                let lo = a.min(z) * crate::robust::NORMAL_Q3;
                let (mut sum, mut c) = (0.0f64, 0.0f64);
                for &i in &idx {
                    if e[i] > lo {
                        break;
                    }
                    let y = v[i] - c;
                    let t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                let gap = (sum / total - 0.5).abs();
                worst = worst.max(gap);
                assert!(
                    gap <= 4.0 * (idx.len() + 256) as f64 * f64::EPSILON,
                    "{gap:e}"
                );
            }
        }
    }
    println!("{differ} of {reads} reads differ from the backward weights', worst share {worst:e}");
}
