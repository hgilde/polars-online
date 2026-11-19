//! A bounded [`EwQuantile`] against an unbounded one fed the same rows
//! (docs/PLAN.md task 271 (b)): the band folds the far buckets into its end
//! buckets, and an answer moves only where folded weight lies on the other
//! side of it from where it arrived.

use std::collections::BTreeMap;

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A draw of `|Z|`, `Z` standard normal, by Box-Muller.
fn abs_normal(s: &mut u64) -> f64 {
    let (u, v) = (lcg(s), lcg(s));
    ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()).abs()
}

/// The bucket the unbounded sketch's walk reaches at level `q`: the first,
/// ascending from the zero bucket, whose cumulative weight reaches `q` of
/// the total. Below 0 it is the zero bucket, past 1 the last bucket.
fn bucket_at(u: &EwQuantile, q: f64) -> i64 {
    let target = q * u.total;
    let mut cum = u.zero;
    if q <= 0.0 || (u.zero > 0.0 && cum >= target) {
        return i64::MIN;
    }
    for (i, b) in u.buckets.iter().enumerate() {
        cum += b;
        if cum >= target {
            return i64::from(u.lo) + i as i64;
        }
    }
    i64::from(u.end())
}

/// What a side-by-side run saw.
#[derive(Debug, Default)]
struct Seen {
    /// Rows on which some weight had been folded.
    folded_rows: usize,
    /// Checks at which folded weight lay across the unbounded answer.
    crossed: usize,
    /// Checks at which the two answers differed.
    differed: usize,
    /// The most buckets the bounded sketch held, and the unbounded.
    most: usize,
    most_unbounded: usize,
}

/// Feed `rows` (value, weight, decay before it) to a bounded sketch and an
/// unbounded one at `levels`, and hold the bounded one's answers to the
/// unbounded one's at every row:
///
/// - to the bit, wherever no folded weight lies across the unbounded
///   answer: none counted on one side of its bucket that arrived on the
///   other;
/// - otherwise within the folded weight's share of the level: its bucket
///   between the unbounded sketch's at `p - φ` and at `p + φ`, `φ` all the
///   folded weight over the total.
///
/// A value's weight is tracked from its arrival: unfolded while its bucket
/// is inside the bounded sketch's, and folded, into the end bucket it was
/// cut off by, once it is not.
fn side_by_side(levels: &[f64], octaves: u32, rows: &[(f64, f64, f64)]) -> Seen {
    let mut b = EwQuantile::bounded(levels, octaves).unwrap();
    let mut u = EwQuantile::new(levels).unwrap();
    // Weight per original bucket, not yet folded; folded weight per
    // (original bucket, end bucket it now counts in).
    let mut unfolded: BTreeMap<i32, f64> = BTreeMap::new();
    let mut folded: BTreeMap<(i32, i32), f64> = BTreeMap::new();
    let mut seen = Seen::default();
    let width = 2 * 64 * octaves as usize + 1;
    for (row, &(v, w, lam)) in rows.iter().enumerate() {
        b.age(lam);
        u.age(lam);
        unfolded.values_mut().for_each(|x| *x *= lam);
        folded.values_mut().for_each(|x| *x *= lam);
        b.add(v, w);
        u.add(v, w);
        if v >= f64::MIN_POSITIVE && w > 0.0 {
            *unfolded.entry(EwQuantile::index(v)).or_default() += w;
        }
        if !b.buckets.is_empty() {
            let (lo, end) = (b.lo, b.end());
            let cut: Vec<i32> = unfolded
                .range(..lo)
                .chain(unfolded.range(end + 1..))
                .map(|(k, _)| *k)
                .collect();
            for k in cut {
                let x = unfolded.remove(&k).unwrap();
                *folded.entry((k, k.clamp(lo, end))).or_default() += x;
            }
        }
        seen.most = seen.most.max(b.buckets.len());
        seen.most_unbounded = seen.most_unbounded.max(u.buckets.len());
        // One level: the buckets stay within the band, with one more for
        // each level's spread.
        if levels.len() == 1 {
            assert!(b.buckets.len() <= width, "row {row}: {}", b.buckets.len());
        }
        assert!(b.has_shape(), "row {row}");
        let all_folded: f64 = folded.values().sum();
        seen.folded_rows += usize::from(all_folded > 0.0);
        for (l, &p) in levels.iter().enumerate() {
            let j = EwQuantile::rank(u.at[l]);
            let across: f64 = folded
                .iter()
                .filter(|((i, f), _)| {
                    let (i, f) = (i64::from(*i), i64::from(*f));
                    (i < j && j <= f) || (f < j && j <= i)
                })
                .map(|(_, x)| x)
                .sum();
            let (got, want) = (b.get(l), u.get(l));
            let same = got.map(f64::to_bits) == want.map(f64::to_bits);
            seen.differed += usize::from(!same);
            if across == 0.0 {
                assert!(
                    same,
                    "row {row}, p = {p}: {got:?} against {want:?} with no folded weight across"
                );
                continue;
            }
            seen.crossed += 1;
            let phi = all_folded / (u.total * u.scale) * (1.0 + 1e-9) + 1e-12;
            let (low, high) = (bucket_at(&u, p - phi), bucket_at(&u, p + phi));
            let at = EwQuantile::rank(b.at[l]);
            assert!(
                low <= at && at <= high,
                "row {row}, p = {p}: bucket {at} outside {low}..={high} (φ = {phi})"
            );
        }
    }
    seen
}

/// Rows of `|e|`, `e` normal at `sigma`, after `start` rows at `level`;
/// weights in `[0.5, 1.5)` with every 13th 0, every 37th value 0, and the
/// values times `2^20` from row `shift` on.
fn rows(
    n: usize,
    lam: f64,
    start: usize,
    level: f64,
    shift: usize,
    seed: u64,
) -> Vec<(f64, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let e = 0.5 * abs_normal(&mut s);
            let w = if i % 13 == 0 { 0.0 } else { 0.5 + lcg(&mut s) };
            let v = if i % 37 == 0 {
                0.0
            } else if i < start {
                level + e
            } else if i >= shift {
                e * 2f64.powi(20)
            } else {
                e
            };
            (v, w, lam)
        })
        .collect()
}

/// The start-up fall a MAD's sketch sees -- residuals at a target's level
/// of 1,000 for the first rows, then at the noise's -- spans more octaves
/// than an eight-octave band's sixteen with the noise's low tail, so the
/// band folds, decayed and not, and every answer is the unbounded one's to
/// the bit: the median falls by a factor of 2^12, inside the band, so no
/// weight it reads was ever folded.
#[test]
fn a_bounded_sketch_answers_to_the_bit_through_a_start_up_fall() {
    for (what, lam) in [("undecayed", 1.0), ("decayed", (-1.0f64 / 300.0).exp2())] {
        for levels in [&[0.5][..], &[0.5, 0.99][..]] {
            let seen = side_by_side(levels, 8, &rows(12_000, lam, 30, 1000.0, usize::MAX, 61));
            assert!(
                seen.folded_rows > 1000,
                "{what} {levels:?}: the band never folded: {seen:?}"
            );
            assert_eq!(seen.crossed, 0, "{what} {levels:?}: {seen:?}");
            assert_eq!(seen.differed, 0, "{what} {levels:?}: {seen:?}");
            assert!(
                seen.most_unbounded > seen.most,
                "{what} {levels:?}: {seen:?}"
            );
        }
    }
}

/// A level shift of `2^20` takes the values past the band before the
/// median follows: the new values fold into the upper end bucket, the
/// median reads it, and until that weight no longer counts across it the
/// answer is off by at most the folded weight's share of the level; where
/// none lies across it, decayed or not and at 0.5 or 0.99, it is the
/// unbounded answer to the bit.
#[test]
fn a_bounded_sketch_moves_by_at_most_its_folded_share_across_a_level_shift() {
    for (what, lam) in [("undecayed", 1.0), ("decayed", (-1.0f64 / 500.0).exp2())] {
        for (levels, octaves) in [(&[0.5][..], 8), (&[0.99][..], 4), (&[0.5, 0.99][..], 6)] {
            let seen = side_by_side(levels, octaves, &rows(12_000, lam, 0, 0.0, 4_000, 67));
            assert!(
                seen.crossed > 0,
                "{what} {levels:?}: nothing crossed: {seen:?}"
            );
            assert!(
                seen.differed > 0,
                "{what} {levels:?}: nothing differed: {seen:?}"
            );
        }
    }
}

/// The case the band changes an answer by more than a bucket: more than
/// the level's share of the weight past the band. Unit weights, no decay:
/// 100 values of 1, then values of `2^20`, twenty octaves up, past an
/// eight-octave band's sixteen (bucket 1,024). They fold into that end
/// bucket, and from the 101st the median reads it: `2^16` and a bit, under
/// the true `2^20` by a factor of 16 -- an end bucket's value is a bound on
/// the answer, not within `α` of it. The band then moves up with the
/// median and the values of `2^20` land in their own bucket, but the 101
/// folded below it still count there, so the answer stays at bucket 1,024
/// until the new bucket alone holds half the weight: past 100 + 101 values,
/// the 202nd. From there on it is the unbounded answer to the bit.
#[test]
fn past_the_band_the_end_bucket_answers() {
    let mut b = EwQuantile::bounded(&[0.5], 8).unwrap();
    let mut u = EwQuantile::new(&[0.5]).unwrap();
    for _ in 0..100 {
        b.add(1.0, 1.0);
        u.add(1.0, 1.0);
    }
    let top = 2f64.powi(20);
    let end = EwQuantile::representative(1024);
    assert!(end > 2f64.powi(16) && end < 2f64.powi(16) * (1.0 + 1.0 / 64.0));
    for m in 1..=1000 {
        b.add(top, 1.0);
        u.add(top, 1.0);
        let (got, want) = (b.get(0).unwrap(), u.get(0).unwrap());
        if m <= 100 {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{m}: the ones hold the median"
            );
        } else if m <= 100 + 202 {
            assert_eq!(got, end, "{m}: the end bucket answers");
            assert!((want - top).abs() <= EW_QUANTILE_ALPHA * top, "{m}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "{m}");
        }
    }
    // The ones were folded up into bucket 256 when the band moved, and the
    // first 101 values of 2^20 still count at 1,024.
    assert_eq!(b.lo, 256);
    assert_eq!(b.buckets[0], 100.0);
    assert_eq!(b.buckets[1024 - 256], 101.0);
    assert_eq!(b.buckets.len(), 1280 - 256 + 1);
}

/// A wild value no longer stretches the buckets to it: one `|e|` of
/// `1e200`, 664 octaves up, and one of `1e-200` took an unbounded sketch to
/// 85,000 buckets, about 680 KB, for as long as either counted -- for
/// good, without a decay. A bounded one keeps its band, and reads the same
/// median.
#[test]
fn a_wild_value_folds_instead_of_stretching_the_buckets() {
    let mut s = 71u64;
    let mut rows: Vec<(f64, f64, f64)> =
        (0..3000).map(|_| (abs_normal(&mut s), 1.0, 1.0)).collect();
    rows[1000].0 = 1e200;
    rows[2000].0 = 1e-200;
    let seen = side_by_side(&[0.5], 8, &rows);
    assert!(seen.most_unbounded > 80_000, "{seen:?}");
    assert!(seen.most <= 2 * 64 * 8 + 1, "{seen:?}");
    assert_eq!(seen.differed, 0, "{seen:?}");
}

/// A saved bounded sketch, its ends folded, goes on as the live one to the
/// bit through both encodings, its band with it; and an unbounded sketch
/// writes exactly the bytes it wrote before the band existed, named and as
/// an array, so every state saved without it loads unbounded and saves its
/// bytes again.
#[test]
fn a_saved_bounded_sketch_goes_on_and_an_unbounded_one_writes_its_old_bytes() {
    let mut live = EwQuantile::bounded(&[0.5, 0.9], 4).unwrap();
    let mut s = 73u64;
    for i in 0..2000 {
        live.age(0.995);
        let v = abs_normal(&mut s) * if i % 100 == 0 { 1e6 } else { 1.0 };
        live.add(v, 1.0);
    }
    assert!(
        live.buckets.len() <= 2 * 64 * 4 + 1 + 64 * 2,
        "{}",
        live.buckets.len()
    );
    for bytes in [
        rmp_serde::to_vec(&live).unwrap(),
        rmp_serde::to_vec_named(&live).unwrap(),
    ] {
        let mut back: EwQuantile = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.band(), Some(4));
        let mut l = live.clone();
        for _ in 0..2000 {
            let v = abs_normal(&mut s) * 30.0;
            l.age(0.995);
            back.age(0.995);
            l.add(v, 1.0);
            back.add(v, 1.0);
        }
        assert_eq!(l, back);
    }

    /// `EwQuantile` as it was before the band.
    #[derive(Serialize)]
    struct Before<'a> {
        levels: &'a [f64],
        buckets: &'a [f64],
        lo: i32,
        zero: f64,
        total: f64,
        scale: f64,
        at: &'a [Option<i32>],
        below: &'a [f64],
    }
    let mut q = EwQuantile::new(&[0.5, 0.99]).unwrap();
    for i in 0..300 {
        q.age(0.99);
        q.add(f64::from(i % 17), 1.0);
    }
    let before = Before {
        levels: &q.levels,
        buckets: &q.buckets,
        lo: q.lo,
        zero: q.zero,
        total: q.total,
        scale: q.scale,
        at: &q.at,
        below: &q.below,
    };
    assert_eq!(
        rmp_serde::to_vec(&q).unwrap(),
        rmp_serde::to_vec(&before).unwrap()
    );
    assert_eq!(
        rmp_serde::to_vec_named(&q).unwrap(),
        rmp_serde::to_vec_named(&before).unwrap()
    );
    let back: EwQuantile = rmp_serde::from_slice(&rmp_serde::to_vec(&before).unwrap()).unwrap();
    assert_eq!(back, q);
    assert_eq!(back.band(), None);
}

/// A band of 0 octaves, or past the floats' range, is refused at
/// construction and on a restored sketch; `reset` keeps the band.
#[test]
fn a_band_out_of_range_is_refused_and_reset_keeps_it() {
    for bad in [0, MAX_BAND_OCTAVES + 1] {
        assert!(EwQuantile::bounded(&[0.5], bad).is_err(), "{bad}");
        let mut q = EwQuantile::bounded(&[0.5], 8).unwrap();
        q.band = Some(bad);
        assert!(!q.has_shape(), "{bad}");
    }
    assert!(
        EwQuantile::bounded(&[0.5], MAX_BAND_OCTAVES)
            .unwrap()
            .has_shape()
    );
    assert!(EwQuantile::bounded(&[1.0], 8).is_err());
    let mut q = EwQuantile::bounded(&[0.5], 8).unwrap();
    q.add(3.0, 1.0);
    q.reset();
    assert_eq!(q, EwQuantile::bounded(&[0.5], 8).unwrap());
}

/// The weight each bucket of a sketch holds, by bucket index, the empty
/// ones left out.
fn held(q: &EwQuantile) -> Vec<(i32, f64)> {
    (q.lo..)
        .zip(q.buckets.iter().copied())
        .filter(|(_, b)| *b != 0.0)
        .collect()
}

/// A fold at both ends, written out: at a band of one octave the median's
/// bucket keeps 128 buckets beside it. Ten rows at 1 (bucket 0) hold the
/// median there; one in bucket 121, near `2^1.89`, spans less than the band and
/// folds nothing; one at `1/2` (bucket −64) then spans 185, more than the
/// band, so the arm below keeps its 64 and the one above is cut to 64:
/// the row at bucket 121 now counts in bucket 64, the end bucket on its
/// side, and no weight is lost (docs/PLAN.md task 272 (m)).
#[test]
fn a_fold_moves_the_weight_past_the_band_into_its_end_bucket() {
    let mut q = EwQuantile::bounded(&[0.5], 1).unwrap();
    q.add(1.0, 10.0);
    let high = EwQuantile::representative(121);
    assert_eq!(EwQuantile::index(high), 121);
    q.add(high, 1.0);
    assert_eq!(held(&q), vec![(0, 10.0), (121, 1.0)], "no fold yet");
    q.add(0.5, 1.0);
    assert_eq!((q.lo, q.buckets.len()), (-64, 129));
    assert_eq!(held(&q), vec![(-64, 1.0), (0, 10.0), (64, 1.0)]);
    assert_eq!(q.total, 12.0);
    assert_eq!(q.at[0], Some(0), "the median stays in the bucket of 1");
}

/// While half the weight or more is at exactly 0 the median is 0, and the
/// band is kept from the lowest value up: half the weight at 0 and two
/// values at `1e-300` and `1e300`, 1,993 octaves apart, spanned 127,563
/// buckets, a megabyte, where the band holds `2·64·8 + 1 = 1,025`. The
/// value past the band counts in its end bucket, and no weight is lost
/// (docs/PLAN.md task 272 (m2)).
#[test]
fn a_median_at_zero_keeps_the_band_from_the_lowest_value() {
    let mut q = EwQuantile::bounded(&[0.5], 8).unwrap();
    q.add(0.0, 2.0);
    q.add(1e-300, 1.0);
    q.add(1e300, 1.0);
    assert_eq!(q.at[0], None, "the median is 0");
    assert_eq!(q.get(0), Some(0.0));
    let most = EwQuantile::one_level_buckets(8);
    let (len, cap) = q.buckets_held();
    assert!(len <= most && cap <= most, "{len} spanned, {cap} held");
    let low = EwQuantile::index(1e-300);
    assert_eq!(held(&q), vec![(low, 1.0), (low + 1024, 1.0)]);
    assert_eq!((q.zero, q.total), (2.0, 4.0));
    // Unbounded, the same rows span the octaves between them.
    let mut wide = EwQuantile::new(&[0.5]).unwrap();
    for (v, w) in [(0.0, 2.0), (1e-300, 1.0), (1e300, 1.0)] {
        wide.add(v, w);
    }
    assert_eq!(wide.buckets_held().0, 127_563);
}

/// The median leaves 0 for the bucket an unbounded sketch reads, to the
/// bit, when the values sit within the band above the lowest of them: the
/// zeros of the first rows decay away under a stream spanning 12 octaves,
/// and from the row the median steps off 0 the two read alike.
#[test]
fn a_median_leaving_zero_is_the_unbounded_sketchs() {
    let mut bounded = EwQuantile::bounded(&[0.5], 8).unwrap();
    let mut wide = EwQuantile::new(&[0.5]).unwrap();
    let mut s = 3u64;
    let mut left = None;
    for i in 0..400 {
        let (v, w) = if i < 5 {
            (0.0, 4.0)
        } else {
            ((12.0 * lcg(&mut s)).exp2(), 1.0)
        };
        for q in [&mut bounded, &mut wide] {
            q.age(0.97);
            q.add(v, w);
        }
        assert_eq!(bounded.get(0), wide.get(0), "row {i}");
        if left.is_none() && bounded.get(0) != Some(0.0) {
            left = Some(i);
        }
    }
    assert!(left.is_some_and(|i| i > 5), "{left:?}");
    assert!(
        bounded.zero > 0.0,
        "the zeros are still held, at a small share"
    );
}

/// Where the values sit past the band above the lowest of them, the median
/// leaving 0 reads the band's end bucket, a bound on the quantile, as past
/// any band: one row at 1 (bucket 0) and the rest at `2^20` (bucket 1,280),
/// folded into bucket 1,024 while the median was 0, so it steps from 0 to
/// `2^16`. The rows still arriving at `2^20` then fold the band about the
/// median, and once their weight passes the folded weight's the median
/// reads the unbounded sketch's bucket again.
#[test]
fn a_median_leaving_zero_past_the_band_reads_its_end_bucket() {
    let mut bounded = EwQuantile::bounded(&[0.5], 8).unwrap();
    let mut wide = EwQuantile::new(&[0.5]).unwrap();
    let rows = std::iter::once((0.0, 20.0))
        .chain(std::iter::once((1.0, 1.0)))
        .chain(std::iter::repeat_n((2f64.powi(20), 1.0), 200));
    let (mut at_end, mut agreed) = (0, None);
    for (i, (v, w)) in rows.enumerate() {
        for q in [&mut bounded, &mut wide] {
            q.age(0.97);
            q.add(v, w);
        }
        let (b, u) = (bounded.at[0], wide.at[0]);
        if b == Some(1024) {
            at_end += 1;
            assert_eq!(u, Some(1280), "row {i}");
        }
        if b.is_some() && b == u && agreed.is_none() {
            agreed = Some(i);
        }
        assert!(bounded.buckets_held().1 <= EwQuantile::one_level_buckets(8));
        assert!((bounded.total - wide.total).abs() <= 1e-12 * wide.total);
    }
    assert!(at_end > 0, "the median read the end bucket");
    assert!(agreed.is_some(), "and then the unbounded sketch's bucket");
}

/// A sketch wider than its band, as a state written while a median at 0
/// kept none, folds on its next value, even one inside its buckets.
#[test]
fn a_sketch_wider_than_its_band_folds_on_its_next_value() {
    let mut q = EwQuantile::new(&[0.5]).unwrap();
    for (v, w) in [(0.0, 2.0), (1e-300, 1.0), (1e300, 1.0)] {
        q.add(v, w);
    }
    q.band = Some(8);
    assert!(q.has_shape());
    assert_eq!(q.buckets_held().0, 127_563);
    q.add(1.0, 1.0);
    assert_eq!(q.buckets_held().0, EwQuantile::one_level_buckets(8));
    assert_eq!(q.total, 5.0);
    assert_eq!(q.buckets.iter().sum::<f64>(), 3.0);
}

/// A median at 0 and a value arriving more than the band below the
/// buckets: the band is kept from that value up, so every bucket lies past
/// it and folds into its end bucket, one the buckets did not reach yet.
/// The fold took the end bucket's place in the buckets as `hi − lo + 1`,
/// negative here, and drained from a start near `usize::MAX` (docs/PLAN.md
/// task 272 (p): `huber` panicked on four rows). The weight is kept, the
/// buckets stay within the band, the same rows in the other order leave
/// the same sketch, and the median reads it as a walk would.
#[test]
fn a_median_at_zero_and_a_value_far_below_the_buckets_folds_them_all() {
    let mut q = EwQuantile::bounded(&[0.5], 8).unwrap();
    q.add(0.0, 2.0);
    q.add(1e300, 1.0);
    q.add(1e-300, 1.0);
    assert_eq!(q.at[0], None, "the median is 0");
    let low = EwQuantile::index(1e-300);
    assert_eq!(held(&q), vec![(low, 1.0), (low + 1024, 1.0)]);
    assert_eq!((q.zero, q.total), (2.0, 4.0));
    let most = EwQuantile::one_level_buckets(8);
    let (len, cap) = q.buckets_held();
    assert!(len <= most && cap <= most, "{len} spanned, {cap} held");
    let mut ascending = EwQuantile::bounded(&[0.5], 8).unwrap();
    for (v, w) in [(0.0, 2.0), (1e-300, 1.0), (1e300, 1.0)] {
        ascending.add(v, w);
    }
    assert_eq!(q, ascending);
    // The median then leaves 0 as it would have: for the lowest value's
    // bucket, then for the band's end bucket, a bound on the values folded
    // into it; each pointer and the weight below it are a walk's.
    let walked = |q: &EwQuantile| {
        let mut r = q.clone();
        r.rebuild();
        (r.at, r.below, r.total)
    };
    q.add(1e-300, 3.0);
    assert_eq!(q.at[0], Some(low));
    assert_eq!(q.get(0), Some(EwQuantile::representative(low)));
    assert_eq!((q.at.clone(), q.below.clone(), q.total), walked(&q));
    q.add(1e300, 10.0);
    assert_eq!(q.at[0], Some(low + 1024));
    assert_eq!(held(&q), vec![(low, 4.0), (low + 1024, 11.0)]);
    assert_eq!((q.at.clone(), q.below.clone(), q.total), walked(&q));
    let (len, cap) = q.buckets_held();
    assert!(len <= most && cap <= most, "{len} spanned, {cap} held");

    // Just past the band: every bucket folds into the bucket right below
    // them, where `hi − lo + 1` was 0 and the fold indexed `usize::MAX`.
    let mut q = EwQuantile::bounded(&[0.5], 1).unwrap();
    q.add(0.0, 2.0);
    q.add(1.0, 1.0);
    q.add(EwQuantile::representative(-129), 1.0);
    assert_eq!(held(&q), vec![(-129, 1.0), (-1, 1.0)]);
    assert_eq!((q.lo, q.buckets.len(), q.total), (-129, 129, 4.0));
}

/// Where the band's top end is the first bucket itself, the fold is the
/// one it always was: the buckets past it summed, then added to it. Summed
/// with it from the start, `1 + 2^-53 + 2^-53` rounds to 1, where `1 +
/// (2^-53 + 2^-53)` is `1 + 2^-52`, so the grouping shows in the bits.
#[test]
fn a_band_ending_at_the_first_bucket_folds_into_it_as_before() {
    let mut q = EwQuantile::bounded(&[0.5], 1).unwrap();
    q.add(0.0, 10.0);
    let tiny = f64::EPSILON / 2.0;
    q.add(1.0, 1.0);
    for k in [1, 2] {
        q.add(EwQuantile::representative(k), tiny);
    }
    assert_eq!(held(&q), vec![(0, 1.0), (1, tiny), (2, tiny)]);
    q.add(EwQuantile::representative(-128), 1.0);
    assert_eq!(q.at[0], None, "the median is 0");
    assert_eq!(held(&q), vec![(-128, 1.0), (0, 1.0 + f64::EPSILON)]);
    assert_eq!((q.lo, q.buckets.len()), (-128, 129));
}
