//! The queue's memos and its turn-over: what a memo answers is the function's
//! own bits, the cut's table keeps the last distance of each of four classes,
//! and a turn-over moves every row into room made for them at once.
use super::held_mass::HELD_MASS_FROM;
use super::*;

/// A discount or a mass taken through a queue's memo is the one taken
/// without it, to the bit, whatever the distances run through it: a
/// repeat, a different one between two repeats, 0, a step back, `NaN`,
/// an infinity, and a half-life of infinity, which has no decay.
#[test]
fn a_memoed_discount_and_mass_are_the_unmemoed_ones() {
    let distances = [
        0.0,
        1.0,
        1.0,
        0.25,
        1.0,
        f64::NAN,
        f64::NAN,
        -0.0,
        -3.0,
        3.0,
        3.0,
        f64::INFINITY,
        f64::INFINITY,
        1e-300,
        1e-300,
        1e300,
        0.1 + 0.2,
        0.1 + 0.2,
        0.3,
    ];
    for half_life in [0.5, 5.0, 1e12, f64::INFINITY] {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life,
            window_size: Some(5.0),
            window_ns: None,
            closed: Closed::Right,
        };
        let mut memo = DecayMemo::default();
        let mut fwd = FwdMemo::default();
        for pass in 0..2 {
            for &d in &distances {
                assert_eq!(
                    k.discount_in(d, &mut memo).to_bits(),
                    k.discount(d).to_bits(),
                    "discount of {d} at {half_life}, pass {pass}"
                );
                assert_eq!(
                    k.mass_in(d, &mut memo).to_bits(),
                    k.mass(d).to_bits(),
                    "mass of {d} at {half_life}, pass {pass}"
                );
                assert_eq!(
                    k.mass_far_in(d, &mut memo).to_bits(),
                    k.mass(d).to_bits(),
                    "far mass of {d} at {half_life}, pass {pass}"
                );
                assert_eq!(
                    k.mass_cut_in(d, &mut memo).to_bits(),
                    k.mass(d).to_bits(),
                    "cut mass of {d} at {half_life}, pass {pass}"
                );
                // The held mass is the cut's from the lower cut on (a
                // hundredth of a half-life), and `c (1 - d)` of the
                // discount from it up.
                let held = if d > 0.0 && half_life.is_finite() && d / half_life >= HELD_MASS_FROM {
                    (half_life / std::f64::consts::LN_2) * (1.0 - k.discount(d))
                } else {
                    k.mass(d)
                };
                assert_eq!(
                    k.mass_held_in(d, &mut memo).to_bits(),
                    held.to_bits(),
                    "held mass of {d} at {half_life}, pass {pass}"
                );
                // A forward kernel's close takes each of these on a slot of
                // its own (task 441).
                for slot in [&mut fwd.mass_open, &mut fwd.mass_span] {
                    assert_eq!(
                        k.mass_slot(d, slot).to_bits(),
                        k.mass(d).to_bits(),
                        "slot mass of {d} at {half_life}, pass {pass}"
                    );
                }
                for slot in [&mut fwd.disc_seg, &mut fwd.disc_open] {
                    assert_eq!(
                        k.discount_slot(d, slot).to_bits(),
                        k.discount(d).to_bits(),
                        "slot discount of {d} at {half_life}, pass {pass}"
                    );
                }
            }
        }
    }
}

/// The memo is used: a distance it holds is answered from it, and one
/// it does not is taken and kept; 0 and an infinite half-life are
/// answered before it, and leave it alone.
#[test]
fn a_queues_memo_answers_the_distance_it_holds_and_keeps_the_last() {
    let kernel = |half_life: f64| KernelDef {
        direction: Direction::Backward,
        half_life,
        window_size: Some(5.0),
        window_ns: None,
        closed: Closed::Right,
    };
    let k = kernel(5.0);
    let mut memo = DecayMemo {
        discount: (2.0f64.to_bits(), 123.0),
        mass: (2.0f64.to_bits(), 456.0),
        held: (2.0f64.to_bits(), 321.0),
        mass_far: (2.0f64.to_bits(), 789.0),
        fwd: None,
        cut: None,
    };
    assert_eq!(k.discount_in(2.0, &mut memo), 123.0);
    assert_eq!(k.mass_in(2.0, &mut memo), 456.0);
    assert_eq!(k.mass_far_in(2.0, &mut memo), 789.0);
    assert_eq!(k.mass_held_in(2.0, &mut memo), 321.0);
    // Another distance is taken, and replaces the one held.
    assert_eq!(k.discount_in(3.0, &mut memo), k.discount(3.0));
    assert_eq!(k.mass_in(3.0, &mut memo), k.mass(3.0));
    assert_eq!(k.mass_far_in(3.0, &mut memo), k.mass(3.0));
    // The held mass of 0.6 half-lives is the discount's partner, `c (1 - d)`.
    let held_mass = (5.0 / std::f64::consts::LN_2) * (1.0 - k.discount(3.0));
    assert_eq!(k.mass_held_in(3.0, &mut memo), held_mass);
    assert_eq!(memo.held, (3.0f64.to_bits(), held_mass));
    assert_eq!(memo.discount, (3.0f64.to_bits(), k.discount(3.0)));
    assert_eq!(memo.mass, (3.0f64.to_bits(), k.mass(3.0)));
    assert_eq!(memo.mass_far, (3.0f64.to_bits(), k.mass(3.0)));
    // No distance, and no decay, are answered before the memo.
    let held = memo.clone();
    assert_eq!(k.discount_in(0.0, &mut memo), 1.0);
    assert_eq!(k.mass_in(0.0, &mut memo), 0.0);
    assert_eq!(k.mass_far_in(-1.0, &mut memo), 0.0);
    assert_eq!(k.mass_held_in(0.0, &mut memo), 0.0);
    assert_eq!(kernel(f64::INFINITY).discount_in(4.0, &mut memo), 1.0);
    assert_eq!(kernel(f64::INFINITY).mass_in(4.0, &mut memo), 4.0);
    assert_eq!(kernel(f64::INFINITY).mass_far_in(4.0, &mut memo), 4.0);
    assert_eq!(kernel(f64::INFINITY).mass_held_in(4.0, &mut memo), 4.0);
    assert_eq!(
        (memo.discount, memo.mass, memo.mass_far, memo.held),
        (held.discount, held.mass, held.mass_far, held.held)
    );
}

/// The cut's mass keeps the last distance of each of four low-bit
/// classes, so the few values `tau - (tau - w)` takes are answered from
/// the table, and a distance that is not held is the function's.
#[test]
fn the_cut_s_mass_holds_the_last_distance_of_each_of_four_classes() {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 5.0,
        window_size: Some(60.0),
        window_ns: None,
        closed: Closed::Right,
    };
    let mut memo = DecayMemo::default();
    // Four neighbours of 60: one of each class.
    let near: Vec<f64> = (0..4)
        .map(|i| f64::from_bits(60.0f64.to_bits() + i))
        .collect();
    for &d in &near {
        assert_eq!(k.mass_cut_in(d, &mut memo).to_bits(), k.mass(d).to_bits());
    }
    // Planted values come back for the held distances: the table, not the function.
    let table = memo.cut.as_mut().expect("made at the first ask");
    for slot in table.iter_mut() {
        slot.1 = -1.0;
    }
    for &d in &near {
        assert_eq!(k.mass_cut_in(d, &mut memo), -1.0);
    }
    // A distance of a held class replaces that class's, and only it.
    let other = f64::from_bits(60.0f64.to_bits() + 4);
    assert_eq!(
        k.mass_cut_in(other, &mut memo).to_bits(),
        k.mass(other).to_bits()
    );
    assert_eq!(
        k.mass_cut_in(near[0], &mut memo).to_bits(),
        k.mass(near[0]).to_bits()
    );
    for &d in &near[1..] {
        assert_eq!(k.mass_cut_in(d, &mut memo), -1.0);
    }
    // No distance and no decay are answered before the table.
    assert_eq!(k.mass_cut_in(0.0, &mut memo), 0.0);
    let inf = KernelDef {
        half_life: f64::INFINITY,
        ..k
    };
    assert_eq!(inf.mass_cut_in(7.0, &mut memo), 7.0);
}

/// A turn-over moves every row of `back` to `front`, the oldest on top,
/// into room made for them at once, and leaves `back` empty.
#[test]
fn a_turn_over_moves_every_row_into_room_made_for_them() {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 5.0,
        window_size: Some(100.0),
        window_ns: None,
        closed: Closed::Right,
    };
    let mut q = Queue::new(vec![false, false], &[false, false]);
    let mut scratch = vec![0.0; ROW * 2];
    let rows = 50usize;
    for i in 0..rows {
        let t = i as f64 * 0.5;
        let mut row = [0.0; ROW * 2];
        for j in 0..2 {
            let x = (i * 2 + j) as f64;
            row[j * ROW..j * ROW + 5].copy_from_slice(&[t, x, x, 1.0, 1.0]);
        }
        q.push_row(&k, i as u64, t, i as i64, t, &row, &mut scratch);
    }
    assert_eq!((q.front.len(), q.back.len()), (0, rows));
    q.settle::<false>(&k, &mut scratch);
    assert_eq!((q.front.len(), q.back.len()), (rows, 0));
    assert!(q.back.vals.is_empty() && q.back.sums.is_empty());
    // The oldest row is on top: the newest at the bottom.
    assert_eq!(
        (q.front.seq[0], q.front.seq[rows - 1]),
        (rows as u64 - 1, 0)
    );
    // Room for the whole stack in every arena, made at once (a row-by-row
    // growth would have doubled past it, to 64 for 50 rows).
    let n = rows;
    for (cap, need) in [
        (q.front.seq.capacity(), n),
        (q.front.tau.capacity(), n),
        (q.front.off.capacity(), n),
        (q.front.end.capacity(), n),
        (q.front.at.capacity(), n),
        (q.front.vals.capacity(), n * ROW * 2),
        (q.front.sums.capacity(), n * SUM * 2),
    ] {
        assert!(need <= cap && cap < need + 8, "{cap} for {need}");
    }
    // A second turn-over with nothing in `back` changes nothing.
    q.settle::<false>(&k, &mut scratch);
    assert_eq!((q.front.len(), q.back.len()), (rows, 0));
}

/// A stack's `append` is `extend_from_slice` at every length, the written-out
/// ones and the others, onto an empty vector and onto one with values.
#[test]
fn append_copies_every_length_as_extend_from_slice_does() {
    for len in 0..=40usize {
        let s: Vec<f64> = (0..len)
            .map(|i| f64::from_bits(0x7ff0_0000_0000_0000 ^ (i as u64 * 0x9e37)))
            .collect();
        for before in [0usize, 1, 7] {
            let mut got: Vec<f64> = (0..before).map(|i| i as f64).collect();
            let mut want = got.clone();
            append(&mut got, &s);
            want.extend_from_slice(&s);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{len} after {before}");
        }
    }
}

/// The held mass is `exp_m1`'s below a hundredth of a half-life and the
/// discount's `exp2` form from it up: the last distance below the edge and
/// the first at it are each their form's, to the bit, and the two forms
/// agree to `2 / (1 - d)` ulps wherever they are both taken.
#[test]
fn the_held_mass_changes_form_at_a_hundredth_of_a_half_life() {
    for half_life in [0.25, 5.0, 3600.0, 1e9] {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
        };
        let new_form = |d: f64| (half_life / std::f64::consts::LN_2) * (1.0 - k.discount(d));
        // The first distance at the edge, and the one before it.
        let mut at = HELD_MASS_FROM * half_life;
        while at / half_life >= HELD_MASS_FROM {
            at = f64::from_bits(at.to_bits() - 1);
        }
        while at / half_life < HELD_MASS_FROM {
            at = f64::from_bits(at.to_bits() + 1);
        }
        let below = f64::from_bits(at.to_bits() - 1);
        assert!(below / half_life < HELD_MASS_FROM && at / half_life >= HELD_MASS_FROM);
        let mut memo = DecayMemo::default();
        assert_eq!(
            k.mass_held_in(below, &mut memo).to_bits(),
            k.mass(below).to_bits(),
            "just below the edge at {half_life}"
        );
        assert_eq!(
            k.mass_held_in(at, &mut memo).to_bits(),
            new_form(at).to_bits(),
            "at the edge at {half_life}"
        );
        // From just under the edge to far above it, the forms agree to
        // rounding: the mass's `2^-53 / (1 - d)` and the discount's own.
        let mut d = below;
        while d < 200.0 * half_life {
            let (held, old) = (k.mass_held_in(d, &mut memo), k.mass(d));
            let bound = 2.0 * f64::EPSILON / (1.0 - k.discount(d));
            assert!(
                (held - old).abs() <= bound * old,
                "held mass {held} against {old} at {d} of half-life {half_life}"
            );
            d *= 1.37;
        }
    }
}
