//! The held targets' state under a blocked Gram (docs/PLAN.md task 257):
//! what it holds on each row, when two of them are the same state, which
//! shapes a restored one may take, and that a reader between blocks reads
//! the accumulators themselves.

use super::fold_tests::Damage;
use super::*;

/// Eight rows of two features and two targets: target 1 absent on row 2,
/// row 3 of weight 0.
fn rows() -> Vec<(Vec<f64>, Vec<Option<f64>>, f64)> {
    (0..8)
        .map(|i| {
            let x = i as f64;
            let z = vec![1.0, 0.5 * x - 0.05 * x * x];
            let y = vec![Some(0.3 * x + 1.0), (i != 2).then_some(2.0 - 0.1 * x)];
            (z, y, if i == 3 { 0.0 } else { 1.0 })
        })
        .collect()
}

/// The accumulators after the first `n` rows, under a block of 16.
fn held_after(n: usize, gaps: TargetGaps) -> Acc {
    let mut acc = Acc::new(2, 2, 16, false);
    for (z, y, w) in &rows()[..n] {
        acc.learn(z, y, 1.0, *w, gaps);
    }
    acc
}

/// Each target holds its value on each of its Gram's held rows where it
/// learns from the row, and NaN where it does not: absent, or on a row of
/// weight 0, whose value is no part of any moment.
#[test]
fn a_held_value_is_nan_where_the_target_learns_nothing() {
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let acc = held_after(8, gaps);
        for (j, held) in acc.held.y.iter().enumerate() {
            let want: Vec<f64> = rows()
                .iter()
                .map(|(_, y, w)| match y[j] {
                    Some(v) if *w > 0.0 => v,
                    _ => f64::NAN,
                })
                .collect();
            // A target's values are the last rows its Gram holds.
            let tail = &want[want.len() - held.len()..];
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(held), bits(tail), "{gaps:?}, target {j}");
            assert!(
                held.len() >= 5,
                "{gaps:?}, target {j}: the rows after a split"
            );
        }
    }
}

/// Two held states are the same state only where every target holds the
/// same values, history weight, weight after the block and mean pair, to
/// the bit; a target that holds nothing holds no history either, whatever
/// was last written there.
#[test]
fn held_states_are_equal_exactly_where_every_part_is() {
    let good = held_after(6, TargetGaps::Pairwise).held;
    assert!(good == good.clone());
    assert!(good.y.iter().all(|v| !v.is_empty()));
    let parts: [Damage<'_, Held>; 6] = [
        ("a value", &|h| h.y[1][0] += 1.0),
        ("a value's NaN", &|h| h.y[0][3] = 0.0),
        ("the history", &|h| h.hist[1] += 1.0),
        ("the weight after", &|h| h.w_end[0] *= 2.0),
        ("the mean", &|h| h.my[1] += 1.0),
        ("the mean's low part", &|h| h.my_lo[0] = 1e-20),
    ];
    for (what, change) in parts {
        let mut other = good.clone();
        change(&mut other);
        assert!(other != good, "{what}");
        assert!(good != other, "{what}, the other way");
    }
    // A target holding nothing: equal whatever its history says, and to a
    // state that never held anything.
    let mut quiet = good.clone();
    quiet.y[1].clear();
    let mut stale = quiet.clone();
    stale.hist[1] = 7.0;
    stale.my[1] = -3.0;
    assert!(stale == quiet);
    let mut none = good.clone();
    none.y.iter_mut().for_each(Vec::clear);
    assert!(none == Held::default());
    assert!(Held::default() == none);
    assert!(none != good);
}

/// One corruption per condition the held state's shape is checked on, each
/// refused alone by `Acc::has_shape`, what a restored state must pass: with
/// nothing held, any history, weight or mean left beside it; with values
/// held, a target short in any of the vectors, a target too many, and more
/// values than the Gram holds rows.
#[test]
fn each_part_of_the_held_shape_is_checked_alone() {
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let empty = Acc::new(2, 2, 16, false);
        let holding = held_after(6, gaps);
        assert!(empty.has_shape(2, 2) && holding.has_shape(2, 2));
        let parts: [(&Acc, Damage<'_, Held>); 10] = [
            (&empty, ("nothing held, a history", &|h| h.hist = vec![1.0])),
            (&empty, ("nothing held, a weight", &|h| h.w_end = vec![1.0])),
            (&empty, ("nothing held, a mean", &|h| h.my = vec![1.0])),
            (
                &empty,
                ("nothing held, a low part", &|h| h.my_lo = vec![1.0]),
            ),
            (&holding, ("a history short", &|h| h.hist.truncate(1))),
            (&holding, ("a weight short", &|h| h.w_end.truncate(1))),
            (&holding, ("a mean short", &|h| h.my.truncate(1))),
            (&holding, ("a low part short", &|h| h.my_lo.truncate(1))),
            (&holding, ("a target too many", &|h| h.y.push(vec![1.0]))),
            (
                &holding,
                ("more values than rows", &|h| h.y[0].extend([1.0; 16])),
            ),
        ];
        for (base, (what, corrupt)) in parts {
            let mut a = base.clone();
            corrupt(&mut a.held);
            assert!(!a.has_shape(2, 2), "{gaps:?}, {what}");
        }
    }
}

/// A reader between blocks reads the accumulators themselves, borrowed;
/// mid-block, a copy with the block merged, the accumulators left holding
/// it.
#[test]
fn a_reader_borrows_between_blocks_and_merges_a_copy_mid_block() {
    use std::borrow::Cow;
    let mut acc = held_after(6, TargetGaps::Pairwise);
    assert!(matches!(acc.flushed(), Cow::Owned(_)), "mid-block");
    assert!(
        !acc.held.is_empty(),
        "the copy merged, not the accumulators"
    );
    acc.flush();
    assert!(matches!(acc.flushed(), Cow::Borrowed(_)), "between blocks");
    let unblocked = {
        let mut a = Acc::new(2, 2, 0, false);
        for (z, y, w) in &rows() {
            a.learn(z, y, 1.0, *w, TargetGaps::Pairwise);
        }
        a
    };
    assert!(matches!(unblocked.flushed(), Cow::Borrowed(_)), "unblocked");
}
