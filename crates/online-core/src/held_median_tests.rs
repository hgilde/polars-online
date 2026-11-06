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

/// A buffer written and read back as a state holds it: the tree is not
/// state, and reading the buffer builds it from the rows.
fn read_back(b: &RefitBuf) -> RefitBuf {
    rmp_serde::from_slice(&rmp_serde::to_vec_named(b).unwrap()).unwrap()
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
            let back = read_back(&b);
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

/// Each node's depth in target `t`'s tree, the root's 1: the steps a read
/// and an insert walk.
fn depths(h: &HeldMedian, t: usize) -> Vec<usize> {
    let tree = &h.trees[t];
    let mut out = Vec::with_capacity(tree.nodes.len());
    let mut stack = vec![(tree.root, 1usize)];
    while let Some((at, d)) = stack.pop() {
        if at == NIL {
            continue;
        }
        out.push(d);
        let n = &tree.nodes[at as usize];
        stack.push((n.l, d + 1));
        stack.push((n.r, d + 1));
    }
    out
}

/// The read is `O(log n)` a row (task 329, the priority hash's mutants): a
/// treap whose priorities are a hash of each row's place keeps its depth
/// logarithmic whatever order the `|e|` come in. Rows arriving sorted --
/// a run of rising or falling residuals, or one value repeated, which the
/// ties' order makes rising -- are the order a plain search tree, or a heap
/// whose priorities follow the arrival order, degrades into a list, a read
/// `O(n)`. Measured on this build (4,096 rows): the deepest node 32 and
/// the mean depth 16.5 at most, against `log2 n = 12`; a list is 4,096 deep.
#[test]
fn the_tree_stays_logarithmic_in_any_arrival_order() {
    let n = 4096usize;
    let log2 = n.ilog2() as f64;
    let mut s = 17u64;
    let orders: [(&str, Vec<f64>); 4] = [
        ("rising", (0..n).map(|i| i as f64).collect()),
        ("falling", (0..n).map(|i| (n - i) as f64).collect()),
        ("repeated", vec![1.0; n]),
        ("random", (0..n).map(|_| lcg(&mut s)).collect()),
    ];
    for (name, e) in &orders {
        let mut h = HeldMedian::new(1);
        for &v in e {
            h.decay(1.0);
            h.insert(0, v, 1.0);
        }
        let d = depths(&h, 0);
        assert_eq!(d.len(), n, "{name}");
        let deepest = *d.iter().max().unwrap();
        let mean = d.iter().sum::<usize>() as f64 / n as f64;
        println!("{name}: deepest {deepest}, mean {mean:.1}");
        assert!(deepest as f64 <= 3.0 * log2, "{name}: deepest {deepest}");
        assert!(mean <= 1.5 * log2, "{name}: mean depth {mean}");
        assert_eq!(h.median(0), definition(&h, 0), "{name}");
    }
}

/// Two rows held at weights `1 ∓ d/2` and `1 ± d/2` (`d` in units of
/// `u = 2^-52`), the first the smaller `|e|`: the tree holds them at twice
/// that (a decay of 1 is `m = 1/2`), every sum exact (the total 4, half
/// 2), and the crossing is the second row's, `d` under half below it.
/// Neither weight is a multiple of `2^-20`, so the read takes the margin.
fn two_rows(d: f64) -> HeldMedian {
    let mut h = HeldMedian::new(1);
    let u = f64::EPSILON;
    h.insert(0, 1.0, 1.0 - 0.5 * d * u);
    h.insert(0, 2.0, 1.0 + 0.5 * d * u);
    assert_eq!(h.weights(0), [(1.0, 2.0 - d * u), (2.0, 2.0 + d * u)]);
    h
}

/// Whether target `t`'s read walked the definition, and the read.
fn read_walks(h: &HeldMedian, t: usize) -> (bool, Option<f64>) {
    let before = fallbacks();
    let m = h.median(t);
    (fallbacks() > before, m)
}

/// The margin is `4 (n + 256) u` of the total (task 329, the margin's
/// mutants): a crossing the sums below and through clear by more than it
/// is read from the tree, one inside it walks the definition. With two
/// rows and a total of 4 the margin is `4 · 258 · 4 = 4,128 u`: the sum
/// below at `2 − 2,000 u` walks, at `2 − 6,000 u` does not. A third row
/// moves the crossing off the last row, so the sum through it, `2 + 2,000
/// u` with the margin at `4,144 u`, is the one inside it. Every read is the
/// definition's.
#[test]
fn a_crossing_inside_the_margin_walks_and_one_clear_of_it_does_not() {
    let near = two_rows(2000.0);
    assert!(!near.trees[0].exact);
    assert_eq!(read_walks(&near, 0), (true, Some(2.0)), "inside the margin");
    assert_eq!(near.median(0), definition(&near, 0));
    let clear = two_rows(6000.0);
    assert_eq!(read_walks(&clear, 0), (false, Some(2.0)), "clear of it");
    // The sum through the crossing inside the margin: weights 1/2, `1/2 +
    // 1,000 u` and `1 − 1,000 u`, held at 1, `1 + 2,000 u`, `2 − 2,000 u`.
    let u = f64::EPSILON;
    let mut through = HeldMedian::new(1);
    through.insert(0, 1.0, 0.5);
    through.insert(0, 2.0, 0.5 + 1000.0 * u);
    through.insert(0, 3.0, 1.0 - 1000.0 * u);
    assert_eq!(
        through.weights(0),
        [(1.0, 1.0), (2.0, 1.0 + 2000.0 * u), (3.0, 2.0 - 2000.0 * u)]
    );
    assert_eq!(
        read_walks(&through, 0),
        (true, Some(2.0)),
        "the sum through"
    );
    assert_eq!(through.median(0), definition(&through, 0));
}

/// Every weight a multiple of `2^-20` and the sum under `2^33` -- no margin
/// needed -- or not, and the read takes the margin (task 329, `dyadic`'s
/// and the flag's mutants). A weight off that grid by `1,000 u` (lost in
/// `w + 2^20`, so only `w · 2^20` sees it) and weights on it whose sum
/// reaches `2^33` each take the margin; the same crossing on the grid
/// under `2^33` reads from the tree.
#[test]
fn only_weights_whose_every_sum_is_exact_skip_the_margin() {
    // Off the grid: the two rows above.
    assert!(read_walks(&two_rows(2000.0), 0).0, "off the grid");
    // On the grid, each held weight under 2^33 and their sum past it:
    // `2^32 − 3g` and `2^32 − g` (`g = 2^-20`), held at `2^33 − 6g` and
    // `2^33 − 2g`, their sum `2^34 − 8g` exact; the sum below the
    // crossing `2g` under half, inside the margin (`4 · 258 · 2^34 u`,
    // about `2^-8`).
    let mut big = HeldMedian::new(1);
    let g = 2f64.powi(-20);
    big.insert(0, 1.0, 2f64.powi(32) - 3.0 * g);
    big.insert(0, 2.0, 2f64.powi(32) - g);
    let held = big.weights(0);
    assert_eq!(
        held,
        [
            (1.0, 2f64.powi(33) - 6.0 * g),
            (2.0, 2f64.powi(33) - 2.0 * g)
        ]
    );
    assert!(
        held.iter().all(|&(_, w)| dyadic(w)),
        "each held weight on the grid"
    );
    assert!(!big.trees[0].exact, "a sum past 2^33");
    assert_eq!(read_walks(&big, 0), (true, Some(2.0)), "the sum past 2^33");
    // On the grid with an exact sum: the same shape at `1 ∓ 2^-20`.
    let mut grid = HeldMedian::new(1);
    grid.insert(0, 1.0, 1.0 - g);
    grid.insert(0, 2.0, 1.0 + g);
    assert!(grid.trees[0].exact);
    assert_eq!(read_walks(&grid, 0), (false, Some(2.0)), "exact sums");
    // One weight off the grid takes every later read off it, though the
    // weights after it are on it.
    let mut once = two_rows(2000.0);
    once.insert(0, 0.5, 0.5);
    once.insert(0, 3.0, 0.5);
    assert!(!once.trees[0].exact);
}

/// Every weight held is rescaled by `2^-512` once the decay's exponent has
/// fallen 512 behind (task 329, `rescale`'s mutant): a tree off the grid
/// stays off it after the rescale, so a crossing inside the margin still
/// walks. Target 1 holds the two rows of [`two_rows`]; 600 halvings, and a
/// row of target 0 rescales both trees.
#[test]
fn a_rescale_keeps_a_tree_off_the_grid_on_the_margin() {
    let u = f64::EPSILON;
    let mut h = HeldMedian::new(2);
    h.insert(1, 1.0, 1.0 - 1000.0 * u);
    h.insert(1, 2.0, 1.0 + 1000.0 * u);
    assert_eq!(read_walks(&h, 1), (true, Some(2.0)));
    h.begin_run();
    h.run(0.5, 600);
    h.insert(0, 5.0, 1.0);
    // Rescaled exactly, the reference lowered 512.
    assert_eq!(
        h.weights(1),
        [
            (1.0, (2.0 - 2000.0 * u) * RESCALE),
            (2.0, (2.0 + 2000.0 * u) * RESCALE)
        ]
    );
    assert_eq!(read_walks(&h, 1), (true, Some(2.0)), "after the rescale");
    assert_eq!(h.median(1), definition(&h, 1));
    assert_eq!(h.median(0), Some(5.0));
}

/// A decay of 0 leaves every row held weighing nothing (task 329): the
/// read is the rows' after it alone, and the scale starts afresh.
#[test]
fn a_decay_of_zero_leaves_the_rows_held_weighing_nothing() {
    let mut h = HeldMedian::new(1);
    for e in [1.0, 2.0, 3.0] {
        h.decay(0.9);
        h.insert(0, e, 1.0);
    }
    assert_eq!(h.median(0), Some(2.0));
    h.decay(0.0);
    assert_eq!(h.median(0), None);
    assert!(h.weights(0).iter().all(|&(_, w)| w == 0.0));
    for e in [7.0, 8.0, 9.0] {
        h.decay(1.0);
        h.insert(0, e, 1.0);
    }
    assert_eq!(h.median(0), Some(8.0));
    assert_eq!(h.median(0), definition(&h, 0));
}

/// A row weighing nothing counts nowhere (task 329, the definition's
/// filter): a row a decay of 0 left at 0, and a subnormal total half of
/// which rounds to 0. Half the weight is then 0, which every running sum
/// reaches, the zero row's included; the read is the row of weight.
#[test]
fn a_row_weighing_nothing_is_never_the_median() {
    let mut h = HeldMedian::new(1);
    h.insert(0, 1.0, 1.0);
    h.decay(0.0);
    // `D = 3/4`: the smallest subnormal over it rounds back to itself.
    h.decay(0.75);
    h.insert(0, 2.0, 5e-324);
    assert_eq!(h.weights(0), [(1.0, 0.0), (2.0, 5e-324)]);
    assert_eq!(0.5 * 5e-324, 0.0);
    assert_eq!(h.median(0), Some(2.0));
    assert_eq!(h.median(0), definition(&h, 0));
}

/// A row is held only at a finite `|e|` at least 0 and a weight above 0
/// (task 329): a weight of 0 holds nothing, as `NOT_TAKEN`, a NaN and an
/// infinite `|e|` or weight do.
#[test]
fn only_a_finite_e_at_a_positive_weight_is_held() {
    let mut h = HeldMedian::new(1);
    h.insert(0, 1.0, 2.0);
    for (e, w) in [
        (3.0, 0.0),
        (crate::huber_refit::NOT_TAKEN, 1.0),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (3.0, f64::INFINITY),
        (3.0, f64::NAN),
        (3.0, -1.0),
    ] {
        h.insert(0, e, w);
        assert_eq!(h.weights(0).len(), 1, "|e| {e} at {w}");
    }
    assert_eq!(h.seq, 1);
}

/// A run of decays begins where the last ended (task 329, `begin_run`'s
/// mutant): two runs of different factors after one learned row, `1/2`
/// three times then `1/4` twice, decay the rows before them by `2^-7`, so
/// a row after them weighs `2^7` times its raw weight against them. Held
/// in the weights, and in the read of a buffer that ages so.
#[test]
fn a_run_begins_where_the_last_one_ended() {
    let mut h = HeldMedian::new(1);
    h.decay(1.0);
    h.insert(0, 1.0, 1.0);
    h.begin_run();
    h.run(0.5, 3);
    h.begin_run();
    h.run(0.25, 2);
    h.decay(1.0);
    h.insert(0, 2.0, 1.0);
    let w = h.weights(0);
    assert_eq!(w[1].1 / w[0].1, 128.0, "{w:?}");
    // The buffer: rows at `|e|` 1 and 2 of weight 1, the two runs, a row
    // at 3 of weight 1/16: weights 1, 1, 8, so the median is 3; with the
    // first run lost, 1, 1, 1 and the median 2.
    let mut b = RefitBuf::keeping_resid();
    b.learn_scored(&[0.0], 0.0, 1.0, 1.0, 1.0, 1.0, 10);
    b.learn_scored(&[0.0], 0.0, 2.0, 1.0, 1.0, 1.0, 10);
    for _ in 0..3 {
        b.age(0.5, 0.5, 10);
    }
    for _ in 0..2 {
        b.age(0.25, 0.25, 10);
    }
    b.learn_scored(&[0.0], 0.0, 3.0, 1.0 / 16.0, 1.0, 1.0, 10);
    let want = 3.0 / crate::robust::NORMAL_Q3;
    assert_eq!(b.warm_scale(), Some(Some(want)));
    let back = read_back(&b);
    assert_eq!(back.warm_scale(), Some(Some(want)), "rebuilt");
}

/// Room for the rows, allocated once (task 329, `reserve`'s mutant).
#[test]
fn reserve_makes_room_in_every_tree() {
    let mut h = HeldMedian::new(3);
    h.reserve_to(3, 100);
    for t in &h.trees {
        assert!(t.nodes.capacity() >= 100, "{}", t.nodes.capacity());
    }
    // Room in all for a tree that holds rows already, and trees made for
    // targets it had none for.
    let mut h = HeldMedian::new(1);
    for i in 0..10 {
        h.insert(0, f64::from(i), 1.0);
    }
    h.reserve_to(2, 100);
    assert_eq!(h.trees.len(), 2);
    for t in &h.trees {
        assert_eq!(t.nodes.capacity(), 100, "{}", t.nodes.len());
    }
}

/// A tree is made by its target's first row, and an absent tree reads as
/// an empty one (task 335, F9): the default holds no row and refuses none,
/// and reads what [`HeldMedian::new`] of the targets reads, to the bit.
#[test]
fn a_targets_tree_is_made_by_its_first_row() {
    let mut s = 11u64;
    let mut lazy = HeldMedian::default();
    let mut eager = HeldMedian::new(3);
    assert_eq!(lazy.median(2), None);
    for i in 0..400 {
        let lam = if i % 50 == 49 { 0.0 } else { 0.97 };
        for h in [&mut lazy, &mut eager] {
            h.decay(lam);
        }
        // Target 2 first takes a row at 120, target 1 at 7.
        for t in 0..3 {
            if (t == 2 && i < 120) || (t == 1 && i < 7) {
                continue;
            }
            let (e, w) = (lcg(&mut s), 0.5 + lcg(&mut s));
            lazy.insert(t, e, w);
            eager.insert(t, e, w);
        }
        for t in 0..3 {
            assert_eq!(
                lazy.median(t).map(f64::to_bits),
                eager.median(t).map(f64::to_bits),
                "row {i}, target {t}"
            );
        }
    }
}

/// Cleared, a hold reads as a new one and keeps its room (task 335, F5).
#[test]
fn a_cleared_hold_is_a_new_one_with_its_room() {
    let mut s = 3u64;
    let mut h = HeldMedian::new(2);
    h.reserve_to(2, 64);
    for _ in 0..40 {
        h.decay(0.9);
        h.insert(0, lcg(&mut s), 1.0);
        h.insert(1, lcg(&mut s), 1.0);
    }
    h.clear(2);
    let mut fresh = HeldMedian::new(2);
    for t in &h.trees {
        assert!(t.nodes.is_empty() && t.nodes.capacity() == 64);
    }
    for _ in 0..30 {
        let (e, w, lam) = (lcg(&mut s), lcg(&mut s), 0.8 + 0.2 * lcg(&mut s));
        for m in [&mut h, &mut fresh] {
            m.decay(lam);
            m.insert(1, e, w);
        }
        assert_eq!(
            h.median(1).map(f64::to_bits),
            fresh.median(1).map(f64::to_bits)
        );
        assert_eq!(h.weights(1), fresh.weights(1));
    }
}
