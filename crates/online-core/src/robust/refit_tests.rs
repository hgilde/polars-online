//! `huber`'s refit (`RobustCfg::refit_rows`, `crate::huber_refit`;
//! docs/PLAN.md task 277): the replay rebuilds the sums the stream built,
//! the refit fires on a count of learned rows and moves its own target
//! alone, off the solve cadence, holds through a save and a load
//! mid-buffer, and ignores zero-weight rows.

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base(delta: f64, decay: Decay, refit_rows: u32) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay,
        loss: RobustLoss::Huber { delta },
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows,
        mad_anchor: false,
        min_weight_holds_learning: false,
    }
}

/// One row: the features, the two targets and the weight.
type Row = ([f64; 2], [Option<f64>; 2], f64);

/// A two-target stream at a level of 1,000 with one-sided outliers, a null
/// target every 17 rows and, where asked, a zero-weight row every 9.
fn rows(n: usize, zeros: bool) -> Vec<Row> {
    let mut s = 5u64;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let out = if x[0] > 0.0 && lcg(&mut s) > 0.6 {
                50.0
            } else {
                0.0
            };
            let y0 = (i % 17 != 3).then(|| 1000.0 + 2.0 * x[0] - x[1] + 0.5 * lcg(&mut s) + out);
            let y1 = Some(-3.0 + x[1] + 0.5 * lcg(&mut s));
            let w = if zeros && i % 9 == 4 { 0.0 } else { 1.0 };
            (x, [y0, y1], w)
        })
        .collect()
}

fn run(m: &mut Robust, data: &[Row]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|(x, y, w)| m.step(x, y, 1.0, *w).pred)
        .collect()
}

fn bits(p: &[Vec<f64>]) -> Vec<Vec<u64>> {
    p.iter()
        .map(|r| r.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// At a cut of `1e300` every Huber weight is 1 (an infinite one is least
/// squares, which takes no refit), so the refit's replay
/// must rebuild the sums the stream built, and with a solve on every row
/// nothing downstream can tell: every prediction to the bit, decayed or not.
#[test]
fn the_replay_rebuilds_the_streams_sums_to_the_bit() {
    for decay in [Decay::Lam(1.0), Decay::Lam(0.995)] {
        let data = rows(400, true);
        let plain = run(&mut Robust::new(base(1e300, decay, 0)).unwrap(), &data);
        let warm = run(&mut Robust::new(base(1e300, decay, 60)).unwrap(), &data);
        assert_eq!(bits(&plain), bits(&warm), "{decay:?}");
    }
}

/// The refit moves the fit where the Huber cut binds: after it the
/// predictions part from the plain stream's, and not before.
#[test]
fn the_refit_fires_on_the_row_the_count_is_reached() {
    let data = rows(300, false);
    let plain = run(
        &mut Robust::new(base(1.345, Decay::Lam(1.0), 0)).unwrap(),
        &data,
    );
    let mut m = Robust::new(base(1.345, Decay::Lam(1.0), 100)).unwrap();
    let warm = run(&mut m, &data);
    // Target 0 is null on rows 3, 20, 37, 54, 71, 88, 105, ...: its 100th
    // learned row is row 106 (seven nulls before it), so row 107 is the
    // first its refit can move.
    let first = (0..300).find(|&i| plain[i][0].to_bits() != warm[i][0].to_bits());
    assert_eq!(first, Some(107));
    assert!(m.refit.iter().all(|b| !b.open() && b.held() == 0));
}

/// Saved and loaded mid-buffer, the model goes on as one that never
/// stopped, to the bit: the buffer is state.
#[test]
fn a_save_and_load_mid_buffer_goes_on_to_the_bit() {
    let data = rows(300, true);
    let cfg = base(1.345, Decay::Lam(0.999), 120);
    let whole = run(&mut Robust::new(cfg.clone()).unwrap(), &data);
    let mut m = Robust::new(cfg).unwrap();
    let mut parts = run(&mut m, &data[..70]);
    assert!(m.refit.iter().all(|b| b.open() && b.held() > 0));
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    parts.extend(run(&mut back, &data[70..]));
    assert_eq!(bits(&whole), bits(&parts));
}

/// A zero-weight row is clock alone (hard rule 9): it is not buffered and
/// does not count, so without decay a stream with such rows predicts every
/// other row as the stream without them does, to the bit -- and a first
/// row of weight 0 poisons nothing.
#[test]
fn zero_weight_rows_neither_count_nor_move_the_refit() {
    let with = rows(300, true);
    let without: Vec<_> = with.iter().filter(|r| r.2 > 0.0).cloned().collect();
    let cfg = base(1.345, Decay::Lam(1.0), 80);
    let mut lead = vec![([0.3, -0.2], [Some(1000.0), Some(0.0)], 0.0)];
    lead.extend(with.iter().cloned());
    let a = run(&mut Robust::new(cfg.clone()).unwrap(), &lead);
    let b = run(&mut Robust::new(cfg).unwrap(), &without);
    let kept: Vec<Vec<f64>> = lead
        .iter()
        .zip(a)
        .filter(|(r, _)| r.2 > 0.0)
        .map(|(_, p)| p)
        .collect();
    assert_eq!(bits(&kept), bits(&b));
}

/// The setting is refused under an infinite cut and beside the sum-scale
/// prior; a quantile fit takes it from the warm-up's three rows a
/// coefficient up (`quantile_refit_tests`, task 278, D-Q2).
#[test]
fn the_refit_is_refused_where_it_does_not_apply() {
    assert!(Robust::new(base(f64::INFINITY, Decay::Lam(1.0), 10)).is_err());
    assert!(Robust::new(base(f64::INFINITY, Decay::Lam(1.0), 0)).is_ok());
    let mut q = base(1.345, Decay::Lam(1.0), 10);
    q.loss = RobustLoss::Quantile { tau: 0.5 };
    q.huber_scale = HuberScale::Rms;
    assert!(Robust::new(q.clone()).is_ok());
    q.refit_rows = 8;
    assert!(Robust::new(q).is_err());
    let mut s = base(1.345, Decay::Lam(1.0), 10);
    s.ridge_scale = true;
    assert!(Robust::new(s).is_err());
}

/// On the row a target's buffer fills only that target's fit moves: the
/// other target's predictions are the run without a refit's to the bit
/// until its own buffer fills, and the solve cadence -- rows and clock since
/// the last solve -- is that run's on every row.
#[test]
fn the_refit_moves_its_own_target_alone_and_not_the_cadence() {
    let data = rows(300, false);
    let mut c = base(1.345, Decay::Lam(0.999), 0);
    c.max_rows_between_solves = 7;
    c.solve_every = f64::INFINITY;
    let mut off = Robust::new(c.clone()).unwrap();
    c.refit_rows = 60;
    let mut on = Robust::new(c).unwrap();
    // Target 1 is present on every row and fills on row 59; target 0 is
    // null every 17 rows and fills on row 63.
    let mut fills = [None, None];
    for (i, (x, y, w)) in data.iter().enumerate() {
        let (a, b) = (on.step(x, y, 1.0, *w).pred, off.step(x, y, 1.0, *w).pred);
        assert_eq!(on.rows_since_solve, off.rows_since_solve, "row {i}");
        assert_eq!(on.since_solve, off.since_solve, "row {i}");
        for (j, fill) in fills.iter_mut().enumerate() {
            if fill.is_none() && !on.refit[j].open() {
                *fill = Some(i);
            }
        }
        if fills[0].is_none() {
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "row {i}: target 0 moved");
        }
    }
    assert_eq!(fills, [Some(63), Some(59)]);
}

/// Under feature sets the refit reads the largest set's residuals, target by
/// target: a largest set of every feature is the plain model's fit, refit
/// included, to the bit, for each of two targets.
#[test]
fn the_largest_feature_set_is_the_plain_refit_for_every_target() {
    let data = rows(200, true);
    let mut sets = base(1.345, Decay::Lam(0.999), 60);
    sets.feature_sets = vec![("one".into(), vec![1]), ("all".into(), vec![0, 1])];
    let plain = base(1.345, Decay::Lam(0.999), 60);
    let a = run(&mut Robust::new(sets).unwrap(), &data);
    let b = run(&mut Robust::new(plain).unwrap(), &data);
    // Slots are target-major, then set: the largest set is slot 1 and 3.
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        for (slot, j) in [(1, 0), (3, 1)] {
            assert_eq!(p[slot].to_bits(), q[j].to_bits(), "row {i}, target {j}");
        }
    }
}

/// A restored state's buffers are the ones its configuration writes: one a
/// target while the refit is on, each of whole rows, and none while it is
/// off. A state short of a target's buffer, or with one past its count, or
/// with buffers under a refit that is off, is refused.
#[test]
fn a_state_whose_buffers_do_not_fit_is_refused() {
    let data = rows(30, false);
    let mut m = Robust::new(base(1.345, Decay::Lam(1.0), 60)).unwrap();
    run(&mut m, &data);
    assert!(Robust::restore(&m.state()).is_ok());
    let mut short = m.clone();
    short.refit.pop();
    assert!(
        Robust::restore(&short.state()).is_err(),
        "a target's buffer missing"
    );
    let mut full = m.clone();
    full.cfg.refit_rows = 10;
    assert!(
        Robust::restore(&full.state()).is_err(),
        "more rows learned than the count"
    );
    let mut off = m.clone();
    off.cfg.refit_rows = 0;
    assert!(
        Robust::restore(&off.state()).is_err(),
        "buffers under a refit that is off"
    );
}

/// A sparse target under an irregular clock gives its refit up past four
/// ageing runs a learned row of its count, its buffer freed; under a
/// regular clock its ageings are one run a gap, and it refits. The run of
/// ageings is replayed decay by decay, so the regular case is the stream's
/// own sums at a cut of `1e300` (every Huber weight 1), to the bit.
#[test]
fn a_sparse_target_refits_on_a_regular_clock_and_gives_up_on_an_irregular_one() {
    let mut s = 9u64;
    let data: Vec<([f64; 2], [Option<f64>; 2], f64)> = (0..600)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = (i % 10 == 0).then(|| 1.0 + x[0] - x[1]);
            (x, [y0, Some(x[1] + 0.1 * lcg(&mut s))], lcg(&mut s).abs())
        })
        .collect();
    let mut irregular = Robust::new(base(1.345, Decay::Halflife(80.0), 20)).unwrap();
    for (x, y, d) in &data {
        irregular.step(x, y, 1.0 + d, 1.0);
    }
    let st = irregular.refit_status();
    assert!(st[0].abandoned && st[0].done, "{st:?}");
    assert!(!st[1].abandoned && st[1].passes > 0, "{st:?}");
    // A regular clock: the gaps are runs of nine.
    let regular = |rows: u32| {
        let mut m = Robust::new(base(1e300, Decay::Halflife(80.0), rows)).unwrap();
        let p: Vec<Vec<f64>> = data
            .iter()
            .map(|(x, y, _)| m.step(x, y, 1.0, 1.0).pred)
            .collect();
        (m, p)
    };
    let (m, on) = regular(20);
    let st = m.refit_status();
    assert!(st[0].done && !st[0].abandoned && st[0].passes > 0, "{st:?}");
    let (_, off) = regular(0);
    assert_eq!(bits(&on), bits(&off));
}

/// The refit's passes do not count as solve failures: on a design whose
/// every solve takes a jitter (two equal features, no ridge), the refit adds
/// to the count only its own target's one solve after it, where each pass
/// added its jitters.
#[test]
fn the_refit_passes_are_not_solve_failures() {
    let mut s = 4u64;
    let data: Vec<([f64; 2], [Option<f64>; 2], f64)> = (0..80)
        .map(|_| {
            let a = lcg(&mut s);
            ([a, a], [Some(2.0 * a + 0.1 * lcg(&mut s)), Some(a)], 1.0)
        })
        .collect();
    let mut c = base(1.345, Decay::Lam(1.0), 0);
    c.ridge = 0.0;
    let mut off = Robust::new(c.clone()).unwrap();
    c.refit_rows = 40;
    let mut on = Robust::new(c).unwrap();
    for (x, y, w) in &data[..40] {
        on.step(x, y, 1.0, *w);
        off.step(x, y, 1.0, *w);
    }
    assert!(on.refit_status().iter().all(|r| r.done && r.passes > 0));
    assert!(off.solve_failures > 0, "the design jitters");
    // Two targets refit on row 39, each solving its own slots once more.
    assert!(
        on.solve_failures <= off.solve_failures + 2,
        "{} vs {}",
        on.solve_failures,
        off.solve_failures
    );
}

/// Task 278: a refit that takes every one of its IRLS passes is counted in
/// `solve_failures`, as a seeding's Lloyd run stopped at its cap is. The
/// stream (`slow_refit_stream(687)`: 71 rows, a level added to 39% of the
/// rows with `x > 0`) needs 428 passes to meet the rule, so it runs to the
/// cap of 200; the stream of trial 0 converges well inside it and counts
/// nothing.
#[test]
fn a_refit_that_runs_to_its_cap_is_counted() {
    for (trial, capped) in [(687, true), (0, false)] {
        let rows = crate::huber_refit::slow_refit_stream(trial);
        let mut c = base(1.345, Decay::Halflife(f64::INFINITY), rows.len() as u32);
        c.n_features = 1;
        c.n_targets = 1;
        let mut m = Robust::new(c).unwrap();
        for (x, y) in &rows {
            m.step(&[*x], &[Some(*y)], 1.0, 1.0);
        }
        let st = m.refit_status()[0];
        assert!(st.done && !st.failed, "trial {trial}: {st:?}");
        assert_eq!(st.ran_to_cap(), capped, "trial {trial}: {st:?}");
        assert_eq!(
            st.passes == REFIT_PASSES as u32,
            capped,
            "trial {trial}: {st:?}"
        );
        assert_eq!(m.solve_failures, u64::from(capped), "trial {trial}");
    }
}

/// A bank gives a group's refits up past their budget: every target's
/// buffer is let go before it begins, marked over budget, and the rows that
/// would have filled it pass without a refit. The reservation is the
/// targets' buffers at their full size, `n_targets · refit_reservation`, 0
/// with the refit off (task 281, mutants).
#[test]
fn a_refit_given_up_and_its_reservation() {
    let mut m = Robust::new(base(1.345, Decay::Lam(1.0), 60)).unwrap();
    assert_eq!(m.refit_reservation(), 2 * crate::refit_reservation(2, 60));
    assert!(m.refit_reservation() > crate::refit_reservation(2, 60));
    run(&mut m, &rows(10, false));
    m.refit_give_up();
    run(&mut m, &rows(100, false));
    for st in m.refit_status() {
        assert!(st.done && st.over_budget && st.passes == 0, "{st:?}");
    }
    let off = Robust::new(base(1.345, Decay::Lam(1.0), 0)).unwrap();
    assert_eq!(off.refit_reservation(), 0);
}
