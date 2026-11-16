//! `ewridge`'s solves form their lists in a kept scratch (task 415): the
//! combos, readers and feature indices, the system and its right-hand
//! sides and the solution, which the thread keeps, and the fit's storage
//! and the pending solves' `Arc`s, which the model does, each written over
//! by the next solve. Held here to a model whose scratch is emptied before
//! every row, which forms each list afresh as the solves did before: every prediction, every state
//! byte and every row's inflation read against the kept systems are the
//! same to the bit.

use super::*;

/// A seeded generator in `[-1, 1)`.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A configuration of `k` features and three targets, a solve on every row.
fn cfg(k: usize, shape: (bool, bool, bool), sets: bool, window: Option<f64>) -> EwRidgeCfg {
    let (fit_intercept, standardize, ridge_scale) = shape;
    EwRidgeCfg {
        n_features: k,
        n_targets: 3,
        fit_intercept,
        decay: Decay::Halflife(40.0),
        ridge: if sets { vec![1e-3, 1.0] } else { vec![1e-3] },
        feature_sets: if sets {
            vec![
                ("a".to_string(), vec![0, 1]),
                ("b".to_string(), (1..k).collect()),
            ]
        } else {
            vec![]
        },
        standardize,
        ridge_scale,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// What a model keeps besides its state: emptied, the solves form what they
/// work in afresh.
fn empty_the_scratch(m: &mut EwRidge) {
    drop(SOLVE_BUFS.take());
    m.spare_fit = crate::window::Kept::default();
    m.ready.spare = crate::window::Kept::default();
    m.factors.work = None;
}

#[test]
fn solves_in_a_kept_scratch_are_the_ones_in_a_fresh_scratch() {
    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
    for (k, rows) in [(3usize, 300), (70, 90)] {
        for shape in [
            (true, false, false),
            (true, true, false),
            (true, false, true),
            (false, false, false),
            (false, true, false),
        ] {
            for keep_factor in [false, true] {
                // The sum scale takes neither a window nor a grid (`validate`),
                // and feature sets and windows are tried where they are cheap.
                for (sets, window) in [(false, None), (true, None), (false, Some(30.0))] {
                    if (shape.2 && (sets || window.is_some()))
                        || (k > 3 && (sets || window.is_some()))
                    {
                        continue;
                    }
                    let case = format!(
                        "k {k}, shape {shape:?}, keep_factor {keep_factor}, sets {sets}, \
                         window {window:?}"
                    );
                    let mut kept = EwRidge::new(cfg(k, shape, sets, window)).unwrap();
                    kept.set_keep_factor(keep_factor);
                    let mut fresh = kept.clone();
                    let mut s = 41u64;
                    for i in 0..rows {
                        let x: Vec<f64> = (0..k)
                            .map(|j| {
                                // A column with no spread, which the standardiser drops.
                                if j == 2 && k == 3 {
                                    1.0
                                } else {
                                    2.0 * lcg(&mut s) + j as f64
                                }
                            })
                            .collect();
                        let y: Vec<Option<f64>> = (0..3)
                            .map(|t| {
                                let v = 1.0
                                    + x.iter()
                                        .enumerate()
                                        .map(|(j, xj)| (j + t + 1) as f64 * 0.3 * xj)
                                        .sum::<f64>()
                                    + 0.2 * lcg(&mut s);
                                (t == 0 || i % 3 != t).then_some(v)
                            })
                            .collect();
                        let w = 0.5 + (lcg(&mut s) + 1.0);
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        // The kept model reads the lists the fresh one's last
                        // solve left on the thread, another model's; the fresh
                        // one forms its own.
                        let a = kept.step(&x, &y, d, w);
                        empty_the_scratch(&mut fresh);
                        let b = fresh.step(&x, &y, d, w);
                        assert_eq!(bits(&a.pred), bits(&b.pred), "{case}, row {i}");
                        let (mut ra, mut rb) = (Vec::new(), Vec::new());
                        kept.row_error_inflation_into(&x, 1.0, &mut ra);
                        fresh.row_error_inflation_into(&x, 1.0, &mut rb);
                        assert_eq!(bits(&ra), bits(&rb), "{case}, row {i}: inflation");
                        if i % 53 == 52 {
                            kept.settle_readiness();
                            fresh.settle_readiness();
                            // The pending solves it kept are hollow: no factor.
                            assert!(
                                kept.ready
                                    .spare
                                    .get()
                                    .0
                                    .iter()
                                    .all(|p| p.factor.order() == 0),
                                "{case}: a spare holds a factor"
                            );
                        }
                    }
                    assert_eq!(
                        format!("{:?}", kept.state()),
                        format!("{:?}", fresh.state()),
                        "{case}"
                    );
                    assert_eq!(
                        rmp_serde::to_vec_named(&kept.state()).unwrap(),
                        rmp_serde::to_vec_named(&fresh.state()).unwrap(),
                        "{case}: the state's bytes"
                    );
                    // A copy and a debug print show none of the scratch.
                    assert!(!format!("{kept:?}").contains("SolveBufs"), "{case}");
                    assert!(kept.clone().ready.spare.get().0.is_empty(), "{case}");
                }
            }
        }
    }
}

/// What the solves keep is bounded: no more pending solves than
/// [`Spare::KEPT`], and no list past [`KEEP_ENTRIES`] -- a model of order
/// 70 forms its system afresh, as it did.
#[test]
fn the_scratch_is_bounded() {
    let mut m = EwRidge::new(cfg(70, (true, false, false), false, None)).unwrap();
    let mut s = 3u64;
    for i in 0..90 {
        let x: Vec<f64> = (0..70).map(|_| lcg(&mut s)).collect();
        let y = [Some(x[0]), Some(x[1]), Some(x[2])];
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m.settle_readiness();
    assert!(m.ready.spare.get().0.len() <= Spare::KEPT);
    let bufs = SOLVE_BUFS.take().expect("the thread keeps its lists");
    let parts = &bufs.parts;
    for (what, cap) in [("a", parts.a.capacity()), ("out", parts.out.capacity())] {
        assert!(cap <= KEEP_ENTRIES, "{what}: {cap} entries kept");
    }
}

// What the pool keeps between solves, held to the capacities and the counts
// it keeps (task 422): a kept list or space changes no number a solve
// gives, so these read the pool itself. The thresholds are written as
// literals (`KEPT_ORDER` is 64, so `KEEP_ENTRIES` is 4096): a test that read
// the constant would move with a mutation of it.

/// Row-by-row `steps` of a model of `n_targets` targets over `k` features,
/// a solve on every row; each row's features seeded and spread.
fn run_model(mut cfg: EwRidgeCfg, rows: usize) -> EwRidge {
    let (k, m) = (cfg.n_features, cfg.n_targets);
    cfg.n_targets = m;
    let mut model = EwRidge::new(cfg).unwrap();
    let mut s = 17u64;
    for i in 0..rows {
        let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let y: Vec<Option<f64>> = (0..m).map(|t| Some(x[t % k] + 0.1 * lcg(&mut s))).collect();
        model.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    model
}

#[test]
fn the_previous_fit_is_kept_up_to_4096_entries() {
    // The fit's storage is `n_targets` lists of `k_total` entries, kept
    // for the next solve's while `n_targets * k_total <= 4096`.
    for (k, m, kept) in [(3usize, 3usize, true), (63, 64, true), (63, 65, false)] {
        let mut c = cfg(k, (true, false, false), false, None);
        c.n_targets = m;
        let mut model = run_model(c, 6);
        let held = model.spare_fit.get().len();
        assert_eq!(held, if kept { m } else { 0 }, "k {k}, m {m}");
    }
}

/// The shapes a solve keeps a working matrix for: one each, 16 at most,
/// the 17th emptying the list first.
#[test]
fn a_working_matrix_is_made_once_a_shape_and_sixteen_are_kept() {
    let mut quads = Quads::new();
    for i in 1..=16 {
        quad_for(&mut quads, i, 1);
        quad_for(&mut quads, i, 1);
        assert_eq!(quads.len(), i);
    }
    // A shape held is found, not made again, at any position.
    let first: *const QuadWork = quad_for(&mut quads, 1, 1);
    assert_eq!(first, &quads[0].2 as *const QuadWork);
    assert_eq!(quads.len(), 16);
    // The same order at another width is another shape.
    quad_for(&mut quads, 1, 2);
    assert_eq!(quads.len(), 1);
    assert_eq!((quads[0].0, quads[0].1), (1, 2));
    quad_for(&mut quads, 1, 1);
    assert_eq!(quads.len(), 2);
}

#[test]
fn a_solve_keeps_a_working_matrix_up_to_order_64() {
    // `solve_into` and its kept matrix to the order 64, `solve` past it.
    for (k, kept) in [(3usize, true), (64, true), (65, false)] {
        drop(SOLVE_BUFS.take());
        let mut c = cfg(k, (true, false, false), false, None);
        c.n_targets = 1;
        run_model(c, k + 4);
        let bufs = SOLVE_BUFS.take().expect("the thread keeps its lists");
        let shapes: Vec<(usize, usize)> = bufs.parts.quads.iter().map(|q| (q.0, q.1)).collect();
        assert_eq!(shapes, if kept { vec![(k, 1)] } else { vec![] }, "k {k}");
    }
}

fn parts_with(back: usize) -> Solved {
    Solved {
        sol: Vec::with_capacity(back),
        factor: None,
        a: Vec::with_capacity(back),
        keep: Vec::with_capacity(back),
        s: Vec::with_capacity(back),
        mean: Vec::with_capacity(back),
        centred: false,
        shift: 0.0,
        scale: 1.0,
        ill: false,
    }
}

#[test]
fn reclaim_keeps_the_larger_of_the_list_held_and_the_one_back() {
    // The storage comes back to a pool that holds none.
    let mut parts = Parts::default();
    parts.reclaim(parts_with(10));
    for (what, cap) in [
        ("out", parts.out.capacity()),
        ("a", parts.a.capacity()),
        ("pos", parts.pos.capacity()),
        ("kept_s", parts.kept_s.capacity()),
        ("mean", parts.mean.capacity()),
    ] {
        assert!(cap >= 10, "{what}: {cap}");
    }
    // A list the reader took the storage of comes back empty, and does not
    // replace the one held.
    parts.reclaim(parts_with(0));
    for (what, cap) in [
        ("out", parts.out.capacity()),
        ("a", parts.a.capacity()),
        ("pos", parts.pos.capacity()),
        ("kept_s", parts.kept_s.capacity()),
        ("mean", parts.mean.capacity()),
    ] {
        assert!(cap >= 10, "{what}: {cap} after an empty one came back");
    }
}

#[test]
fn release_drops_a_list_past_4096_entries_and_keeps_one_at_it() {
    let sized = |n: usize| {
        let (f, u) = (Vec::<f64>::with_capacity(n), Vec::<usize>::with_capacity(n));
        assert!(f.capacity() == n && u.capacity() == n);
        (f, u)
    };
    for (n, kept) in [(200usize, true), (4096, true), (4097, false)] {
        let mut parts = Parts::default();
        let (f, u) = sized(n);
        let (f2, u2) = sized(n);
        parts.a = f;
        parts.b = f2;
        parts.cols = u;
        parts.pos = u2;
        parts.release();
        let caps = [
            parts.a.capacity(),
            parts.b.capacity(),
            parts.cols.capacity(),
            parts.pos.capacity(),
        ];
        for cap in caps {
            assert_eq!(cap == n, kept, "{n} entries: {caps:?}");
            assert!(kept || cap == 0, "{n} entries: {caps:?}");
        }
    }
}

#[test]
fn a_spare_pending_solve_is_kept_up_to_64() {
    let mut spare = Spare::default();
    for i in 1..=70 {
        let f = spare.take_factor(Arc::new(Pending::hollow()));
        assert!(f.is_some());
        assert_eq!(spare.0.len(), i.min(64));
    }
    // A solve another slot holds is left to it, and kept by nobody.
    let held = Arc::new(Pending::hollow());
    let other = Arc::clone(&held);
    let mut spare = Spare::default();
    assert!(spare.take_factor(held).is_none());
    assert!(spare.0.is_empty());
    drop(other);
}

#[test]
fn a_standardized_fit_through_the_origin_of_all_zero_columns_is_zero_for_every_target() {
    // No raw second moment is positive, so no column is solved: every
    // target's `kc` coefficients are 0, `kc * m` entries in all.
    let mut c = cfg(3, (false, true, false), false, None);
    c.n_targets = 3;
    let mut model = EwRidge::new(c).unwrap();
    for i in 0..8 {
        let out = model.step(
            &[0.0; 3],
            &[Some(1.0), Some(2.0), Some(3.0)],
            f64::from(i > 0),
            1.0,
        );
        assert_eq!(out.pred.len(), 3);
    }
    let beta = model.beta.as_ref().expect("a fit");
    assert_eq!(beta.0.len(), 3);
    for b in &beta.0 {
        assert_eq!(b.len(), 3);
        assert!(b.iter().all(|&v| v == 0.0), "{b:?}");
    }
}

#[test]
fn a_system_with_no_kept_column_is_kept_empty_beside_one_that_was_solved() {
    // Two feature sets over one pool: the first system's matrix is left in
    // the scratch, and the second, whose only feature has no spread, keeps
    // no column, so the system it keeps for the leverage is empty.
    let mut c = cfg(2, (true, true, false), true, None);
    c.n_targets = 1;
    c.feature_sets = vec![("a".to_string(), vec![0]), ("b".to_string(), vec![1])];
    c.ridge = vec![1e-3];
    let mut model = EwRidge::new(c).unwrap();
    model.set_keep_factor(true);
    let mut s = 5u64;
    for i in 0..20 {
        let x = [lcg(&mut s), 1.0];
        model.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let systems: Vec<&System> = model.ready.systems.iter().flatten().collect();
    assert_eq!(systems.len(), 2);
    assert_eq!(systems[0].z.len(), 1);
    assert!(systems[1].z.is_empty());
    assert!(systems[1].a.is_empty(), "{:?}", systems[1].a);
}
