//! An anchored [`EwQuantile`] (task 278, the user's decision of
//! 2026-10-26; task 323): its values bucketed in units of an anchor its
//! caller sets from values it holds, so a quantile read scales with the
//! values by any factor to rounding, where the fixed buckets moved it by up
//! to a bucket at any factor but a power of two; until then the grid fixed
//! at 1, the fixed buckets to the bit. A sketch saved before schema 65
//! keeps the first-value anchor of its build.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The exponentially weighted quantile by its definition: the smallest
/// value whose cumulative weight, ascending, reaches `p` of the total.
fn ew_quantile(seen: &[(f64, f64)], p: f64) -> f64 {
    let mut v = seen.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut cum = 0.0;
    for (x, w) in &v {
        cum += w;
        if cum >= p * total {
            return *x;
        }
    }
    v.last().unwrap().0
}

/// A stream of `(λ, v, w)`: irregular decays at a half-life of 40 rows,
/// heavy-tailed values (an exponential's), a zero now and then, and
/// uneven weights. Its decays and values are drawn once, so every sketch
/// fed it sees the same rows.
fn stream(seed: u64, n: usize) -> Vec<(f64, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let lam = (-(2.0 * lcg(&mut s) / 40.0)).exp2();
            let v = if i % 41 == 7 {
                0.0
            } else {
                -0.37 * lcg(&mut s).ln()
            };
            (lam, v, 0.5 + lcg(&mut s))
        })
        .collect()
}

/// The median after each row of `rows`, each value times `c`.
fn medians(q: &mut EwQuantile, rows: &[(f64, f64, f64)], c: f64) -> Vec<f64> {
    rows.iter()
        .map(|&(lam, v, w)| {
            q.age(lam);
            q.add(v * c, w);
            q.get(0).unwrap()
        })
        .collect()
}

fn anchored() -> EwQuantile {
    EwQuantile::anchored(&[0.5], 8).unwrap()
}

/// An anchored sketch its caller anchors at `a` before any value.
fn anchored_at(a: f64) -> EwQuantile {
    let mut q = anchored();
    q.anchor_at(a);
    assert_eq!(q.anchor(), Some(a));
    q
}

/// An anchored sketch as a state of schema 64 or earlier restores it while
/// it awaits its first value: `Some(0.0)`, marked to keep that rule.
fn awaiting() -> EwQuantile {
    let mut q = anchored();
    q.anchor = Some(0.0);
    q.keep_first_value_anchor();
    assert!(q.has_shape() && q.anchors_by_first_value());
    q
}

/// The first value of a stream at or above the smallest normal float: a
/// value scaled with the stream, as a refit's median is.
fn first_value(rows: &[(f64, f64, f64)]) -> f64 {
    rows.iter()
        .map(|r| r.1)
        .find(|v| *v >= f64::MIN_POSITIVE)
        .unwrap()
}

/// The read is within `α` of the definition at every level, row by row,
/// as an unanchored sketch's is: the bound is relative, and the anchor a
/// factor.
#[test]
fn an_anchored_read_is_within_alpha_of_the_definition() {
    let levels = [0.1, 0.5, 0.9];
    let mut q = EwQuantile::anchored(&levels, 8).unwrap();
    q.anchor_at(0.37);
    let mut seen: Vec<(f64, f64)> = Vec::new();
    for (i, (lam, v, w)) in stream(3, 3000).into_iter().enumerate() {
        q.age(lam);
        seen.iter_mut().for_each(|e| e.1 *= lam);
        q.add(v, w);
        seen.push((v, w));
        for (l, &p) in levels.iter().enumerate() {
            let (got, want) = (q.get(l).unwrap(), ew_quantile(&seen, p));
            assert!(
                (got - want).abs() <= EW_QUANTILE_ALPHA * want * (1.0 + 1e-9),
                "row {i}, p = {p}: {got} against {want}"
            );
        }
    }
    assert_eq!(q.anchor(), Some(0.37));
}

/// A decimal scale moves a median anchored at a value scaled with the
/// values by rounding on every row of five streams -- a few ulps of the
/// division and the read's product -- where it moves the unanchored one,
/// and one on the grid fixed at 1, by up to a bucket, which is printed and
/// held above half a percent as evidence that the edges were crossed, and
/// within `2α / (1 − α)`, each read being within `α` of the quantile.
#[test]
fn a_decimal_scale_moves_an_anchored_median_by_rounding() {
    let mut worst_fixed: f64 = 0.0;
    let mut worst_anchored: f64 = 0.0;
    for seed in 0..5 {
        let rows = stream(seed, 2000);
        let a = first_value(&rows);
        let fixed = medians(&mut EwQuantile::bounded(&[0.5], 8).unwrap(), &rows, 1.0);
        let grid = medians(&mut anchored(), &rows, 1.0);
        let same: Vec<u64> = fixed.iter().map(|v| v.to_bits()).collect();
        let on_grid: Vec<u64> = grid.iter().map(|v| v.to_bits()).collect();
        assert_eq!(same, on_grid, "the grid fixed at 1 is the fixed buckets");
        let base = medians(&mut anchored_at(a), &rows, 1.0);
        for c in [1e-6, 1e-3, 1e3, 1e6] {
            let moved = medians(&mut EwQuantile::bounded(&[0.5], 8).unwrap(), &rows, c);
            let scaled = medians(&mut anchored_at(a * c), &rows, c);
            for i in 0..rows.len() {
                worst_fixed = worst_fixed.max((moved[i] / c - fixed[i]).abs() / fixed[i]);
                let r = (scaled[i] - c * base[i]).abs() / (c * base[i]);
                worst_anchored = worst_anchored.max(r);
                assert!(
                    r <= 8.0 * f64::EPSILON,
                    "seed {seed}, c {c}, row {i}: {r:e}"
                );
            }
        }
    }
    println!("relative move: fixed {worst_fixed:e}, anchored {worst_anchored:e}");
    assert!(
        worst_fixed > 5e-3,
        "the fixed buckets moved {worst_fixed:e}"
    );
    let a = EW_QUANTILE_ALPHA;
    assert!(worst_fixed <= 2.0 * a / (1.0 - a), "{worst_fixed:e}");
}

/// A power of two divides out exactly: the read is bitwise `c` times the
/// unscaled one, as it is unanchored.
#[test]
fn a_power_of_two_scales_an_anchored_median_bitwise() {
    let rows = stream(11, 2000);
    let a = first_value(&rows);
    for (base, at) in [
        (medians(&mut anchored(), &rows, 1.0), None),
        (medians(&mut anchored_at(a), &rows, 1.0), Some(a)),
    ] {
        for k in [-30, -7, 1, 20] {
            let c = 2f64.powi(k);
            let mut q = at.map_or_else(anchored, |a| anchored_at(a * c));
            let scaled = medians(&mut q, &rows, c);
            for i in 0..rows.len() {
                assert_eq!(
                    scaled[i].to_bits(),
                    (c * base[i]).to_bits(),
                    "{at:?}, 2^{k}, row {i}"
                );
            }
        }
    }
}

/// An anchored sketch keeps the grid fixed at 1 through every value and a
/// reset: nothing it takes sets an anchor. One saved awaiting its first
/// value by a build of schema 64 or earlier keeps that rule: the first
/// value that counts (hard rule 9) sets it -- a weight of 0, a value of
/// exactly 0, one below the smallest normal float, and one that is not
/// finite set nothing; a later one does not move it; and a reset waits for
/// the next first value.
#[test]
fn only_a_sketch_saved_awaiting_its_first_value_is_anchored_by_one() {
    let mut g = anchored();
    assert_eq!(g.anchor(), Some(1.0));
    g.add(0.75, 2.0);
    g.add(3.0, 1.0);
    assert_eq!(g.anchor(), Some(1.0), "no value anchors the grid");
    g.reset();
    assert_eq!((g.anchor(), g.get(0), g.band()), (Some(1.0), None, Some(8)));
    assert!(!g.anchors_by_first_value());
    let mut q = awaiting();
    q.add(5.0, 0.0);
    q.add(0.0, 1.0);
    q.add(f64::MIN_POSITIVE / 2.0, 1.0);
    q.add(f64::NAN, 1.0);
    q.add(f64::INFINITY, 1.0);
    q.add(-3.0, 1.0);
    assert_eq!(q.anchor(), Some(0.0), "nothing counted has set it");
    assert_eq!(
        q.get(0),
        Some(0.0),
        "two zeros at weight 1: the median is 0"
    );
    q.add(0.75, 2.0);
    assert_eq!(q.anchor(), Some(0.75));
    q.add(3.0, 1.0);
    assert_eq!(q.anchor(), Some(0.75), "a later value keeps it");
    // The zero bucket is read as 0, not as a multiple of the anchor.
    let mut z = awaiting();
    z.add(0.0, 3.0);
    z.add(2.0, 1.0);
    assert_eq!(z.get(0), Some(0.0));
    q.reset();
    assert_eq!((q.anchor(), q.get(0), q.band()), (Some(0.0), None, Some(8)));
    q.add(6.0, 1.0);
    assert_eq!(q.anchor(), Some(6.0), "the next first value");
    assert!(q.anchors_by_first_value(), "the rule kept through a reset");
    // An unanchored sketch stays one through a reset, and takes no mark.
    let mut b = EwQuantile::bounded(&[0.5], 8).unwrap();
    b.add(1.0, 1.0);
    b.keep_first_value_anchor();
    b.reset();
    assert_eq!((b.anchor(), b.anchors_by_first_value()), (None, false));
}

/// The definition: an anchored sketch is an unanchored bounded one fed
/// each value over the anchor, its read times the anchor -- to the bit, on
/// a stream that also reaches past the floats either way: a ratio past the
/// largest float counts in the top bucket, one under the smallest normal as
/// 0, and the band folds both as it folds any value.
#[test]
fn an_anchored_sketch_is_a_sketch_of_the_ratios_times_the_anchor() {
    let mut rows = stream(9, 600);
    rows[0].1 = 1e-300;
    rows[200].1 = 1e300;
    rows[201].1 = 1e300;
    rows[400].1 = 1e-310;
    let a = rows[0].1;
    let mut q = anchored_at(a);
    let mut ratios = EwQuantile::bounded(&[0.5], 8).unwrap();
    for (i, &(lam, v, w)) in rows.iter().enumerate() {
        q.age(lam);
        ratios.age(lam);
        q.add(v, w);
        let r = if v >= f64::MIN_POSITIVE {
            (v / a).min(f64::MAX)
        } else {
            v
        };
        ratios.add(r, w);
        let want = ratios.get(0).unwrap() * a;
        assert_eq!(q.get(0).unwrap().to_bits(), want.to_bits(), "row {i}");
        assert!(q.has_shape());
    }
    assert_eq!(q.anchor(), Some(a));
}

/// Both encodings round-trip each form -- the grid fixed at 1, awaiting
/// under the first-value rule, set, unanchored with and without a band --
/// and an unanchored sketch writes no anchor, so its bytes are what they
/// were before the field; only a sketch under the first-value rule writes
/// that rule.
#[test]
fn every_form_round_trips_in_both_encodings() {
    let mut set = anchored_at(0.3);
    medians(&mut set, &stream(5, 50), 1.0);
    let mut grid = anchored();
    medians(&mut grid, &stream(5, 50), 1.0);
    let mut first = awaiting();
    medians(&mut first, &stream(5, 50), 1.0);
    let mut fixed = EwQuantile::bounded(&[0.5], 8).unwrap();
    medians(&mut fixed, &stream(5, 50), 1.0);
    let mut open = EwQuantile::new(&[0.5]).unwrap();
    medians(&mut open, &stream(5, 50), 1.0);
    for q in [
        anchored(),
        grid.clone(),
        awaiting(),
        first,
        set,
        fixed.clone(),
        open,
    ] {
        for bytes in [
            rmp_serde::to_vec(&q).unwrap(),
            rmp_serde::to_vec_named(&q).unwrap(),
        ] {
            let back: EwQuantile = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(back, q);
            assert!(back.has_shape());
        }
    }
    let named = rmp_serde::to_vec_named(&fixed).unwrap();
    assert!(
        !named.windows(6).any(|w| w == b"anchor"),
        "an unanchored sketch writes no anchor"
    );
    let rule = b"anchor_by_first_value";
    let has = |q: &EwQuantile| {
        rmp_serde::to_vec_named(q)
            .unwrap()
            .windows(rule.len())
            .any(|w| w == rule)
    };
    assert!(!has(&grid) && has(&awaiting()));
}

/// A restored anchor is 0 (awaiting) or a positive normal float, and only
/// on a bounded sketch: an anchor the sketch could not have set is refused.
#[test]
fn a_restored_anchor_must_be_one_the_sketch_could_set() {
    let mut q = anchored();
    q.add(1.0, 1.0);
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE / 2.0] {
        let mut b = q.clone();
        b.anchor = Some(bad);
        assert!(!b.has_shape(), "{bad}");
    }
    let mut unbounded = EwQuantile::new(&[0.5]).unwrap();
    unbounded.anchor = Some(0.0);
    assert!(!unbounded.has_shape(), "an anchor without a band");
}

/// Task 281: a sketch still awaiting its anchor has taken no positive
/// normal value, so it holds no bucket: a state that says otherwise read
/// its buckets in raw units and bucketed the next value in the anchor's,
/// and is refused. Its median is then the zero bucket's 0, whatever it is
/// multiplied by.
#[test]
fn a_restored_sketch_awaiting_its_anchor_holds_no_bucket() {
    let mut z = awaiting();
    z.add(0.0, 2.0);
    assert!(z.has_shape());
    assert_eq!(z.get(0), Some(0.0));
    let mut set = anchored();
    set.add(3.0, 1.0);
    set.add(5.0, 1.0);
    assert!(set.has_shape());
    set.anchor = Some(0.0);
    assert!(!set.has_shape(), "awaiting, with buckets");
}

/// Task 281: `anchor_at` anchors only an anchored sketch that holds no
/// weight yet, and only at a positive normal float; it leaves an
/// unanchored sketch unanchored and the anchor of one holding weight where
/// it is.
#[test]
fn anchor_at_sets_only_an_empty_anchored_sketch_to_a_positive_normal() {
    for bad in [0.0, -1.0, f64::MIN_POSITIVE / 2.0, f64::INFINITY, f64::NAN] {
        let mut q = anchored();
        q.anchor_at(bad);
        assert_eq!(q.anchor(), Some(1.0), "{bad}");
    }
    let mut q = anchored();
    q.anchor_at(f64::MIN_POSITIVE);
    assert_eq!(q.anchor(), Some(f64::MIN_POSITIVE));
    q.add(1.0, 1.0);
    q.anchor_at(2.0);
    assert_eq!(
        q.anchor(),
        Some(f64::MIN_POSITIVE),
        "a sketch with weight keeps it"
    );
    let mut zero = anchored();
    zero.add(0.0, 1.0);
    zero.anchor_at(2.0);
    assert_eq!(zero.anchor(), Some(1.0), "weight at 0 is weight");
    let mut taken = awaiting();
    taken.add(0.5, 1.0);
    taken.anchor_at(2.0);
    assert_eq!(taken.anchor(), Some(0.5), "the first value's anchor stays");
    let mut early = awaiting();
    early.anchor_at(2.0);
    assert_eq!(early.anchor(), Some(2.0), "an awaited anchor is set");
    let mut plain = EwQuantile::bounded(&[0.5], 8).unwrap();
    plain.anchor_at(2.0);
    assert_eq!(plain.anchor(), None, "an unanchored sketch stays one");
}
