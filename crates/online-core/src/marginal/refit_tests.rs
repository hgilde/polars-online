//! The bins' Huber refit of the first rows (docs/PLAN.md task 278): when it
//! runs, what it leaves, and what it must not move.

use super::*;
use crate::OnlineModel;

/// A row: its features, its targets and its weight.
type Row = (Vec<f64>, Vec<Option<f64>>, f64);

/// Two targets, three features; the second target missing on every fifth
/// row; spikes of 40 on one row in six of the first 60, then on one in
/// thirteen; a weight of 0 on every seventh row.
fn rows(n: usize) -> Vec<Row> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..n)
        .map(|i| {
            let x: Vec<f64> = (0..3).map(|_| 2.0 * u()).collect();
            let spike = if (i < 60 && i % 6 == 1) || i % 13 == 5 {
                40.0
            } else {
                0.0
            };
            let y0 = (3.0 * x[0]).sin() + 0.3 * u() + spike;
            let y1 = (i % 5 != 2).then(|| 100.0 + x[1] + 0.2 * u() - spike);
            let w = if i % 7 == 3 {
                0.0
            } else {
                1.0 + 0.5 * (i % 3) as f64
            };
            (x, vec![Some(y0), y1], w)
        })
        .collect()
}

fn model(edges: Option<Vec<Vec<f64>>>, warm: usize, refit: u32, h: f64) -> Marginal {
    let mut c = MarginalCfg {
        n_features: 3,
        n_targets: 2,
        decay: Decay::Halflife(h),
        min_weight: vec![0.0; 2],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    c.bins = Some(Box::new(crate::BinCfg {
        n_bins: 4,
        edges,
        rule: crate::BinRule::Quantile,
        warm_rows: warm,
        stat: crate::BinStat::Huber { delta: 1.345 },
        refit_rows: refit,
        budget_mib: None,
        mad_anchor: true,
    }));
    Marginal::new(c).unwrap()
}

fn given() -> Option<Vec<Vec<f64>>> {
    Some(vec![vec![-0.5, 0.0, 0.5]; 3])
}

/// Learned rows: those of positive weight, as the warm-up counts them.
fn learned_before(rows: &[Row], n: usize) -> usize {
    rows[..n].iter().filter(|r| r.2 > 0.0).count()
}

/// With given edges the refit waits, holding every learned row and no
/// other, and runs on the row its count is reached, after that row's own
/// update; then it holds nothing, and each target's location took passes.
/// The rows' weights -- every cell's `W`, `bin_n`, and every pair moment --
/// are those of the model without the refit, to the bit (CLAUDE.md rule 8),
/// and the locations are not.
#[test]
fn the_refit_runs_at_its_count_and_moves_no_weight() {
    let data = rows(400);
    let mut on = model(given(), 2, 60, 80.0);
    let mut off = model(given(), 2, 0, 80.0);
    let mut ran_at = None;
    for (i, (x, y, w)) in data.iter().enumerate() {
        on.step(x, y, 1.0, *w);
        off.step(x, y, 1.0, *w);
        let st = on.refit_status();
        assert_eq!(st.len(), 2);
        if ran_at.is_none() && st[0].done {
            ran_at = Some(i + 1);
            assert!(st.iter().all(|s| s.passes > 0 && !s.over_budget), "{st:?}");
        } else if ran_at.is_none() {
            let held = on.bins.as_ref().unwrap().held.len();
            assert_eq!(held, learned_before(&data, i + 1), "row {i}");
            assert_eq!(st[0].learned as usize, held);
        }
    }
    let at = ran_at.expect("the refit ran");
    assert_eq!(learned_before(&data, at), 60, "on the 60th learned row");
    assert!(on.bins.as_ref().unwrap().held.is_empty());
    assert!(off.refit_status().is_empty(), "off: no status");
    let mut moved = 0;
    for t in 0..2 {
        for j in 0..3 {
            let (a, b) = (on.pair(t, j), off.pair(t, j));
            assert_eq!(a.bin_n, b.bin_n);
            assert_eq!((a.cov, a.corr, a.n_eff), (b.cov, b.corr, b.n_eff));
            moved += a
                .bin_mean_y
                .iter()
                .zip(&b.bin_mean_y)
                .filter(|(p, q)| p != q)
                .count();
        }
    }
    assert!(moved > 0, "the refit moved no location");
}

/// The refit's cells are Huber's M-locations at the cut the rebuilt scale
/// gives: on the refit's row, with no decay, each cell's location is a root
/// of `Σ w_i ψ_c(y_i − m)` over its rows at the raw weights, the cut read
/// from the target's judge, and its `Ω` the rows' Huber weights. Held
/// against the estimating equation itself, which shares nothing with the
/// sweep that solves it.
#[test]
fn each_cell_is_the_m_location_of_its_rows_at_the_refit() {
    let data = rows(80);
    let mut m = model(given(), 2, 60, f64::INFINITY);
    let mut seen = Vec::new();
    for (x, y, w) in &data {
        m.step(x, y, 1.0, *w);
        if *w > 0.0 {
            seen.push((x.clone(), y.clone(), *w));
        }
        if m.refit_status()[0].done {
            break;
        }
    }
    assert_eq!(seen.len(), 60);
    let hist = m.bins.as_ref().unwrap().hist.as_ref().unwrap();
    for t in 0..2 {
        for j in 0..3 {
            for b in 0..4 {
                let (w_cell, loc, om_cell, cut) = hist.cell_and_cut(t, j, b);
                assert!(cut.is_finite() && cut > 0.0, "{cut}");
                let here: Vec<(f64, f64)> = seen
                    .iter()
                    .filter(|(x, _, _)| crate::MarginalBins::bin_of(hist.edges(j), x[j]) == Some(b))
                    .filter_map(|(_, y, w)| y[t].map(|v| (v, *w)))
                    .collect();
                if here.is_empty() {
                    continue;
                }
                let g: f64 = here
                    .iter()
                    .map(|(y, w)| w * (y - loc).clamp(-cut, cut))
                    .sum();
                let total: f64 = here.iter().map(|(_, w)| w).sum();
                assert!(g.abs() <= 1e-9 * total * cut, "t{t} j{j} b{b}: g = {g}");
                let om: f64 = here
                    .iter()
                    .map(|(y, w)| w * (cut / (y - loc).abs()).min(1.0))
                    .sum();
                assert!((om_cell - om).abs() <= 1e-12 * total, "t{t} j{j} b{b}");
                assert_eq!(w_cell, total, "the cell's weight is its rows'");
            }
        }
    }
}

/// With learned edges and a count within the warm-up, the refit runs as the
/// edges are fixed, over every held row, and holds nothing of its own (its
/// reservation is 0); a count past the warm-up keeps the rows held after
/// the edges are fixed, until it, and reserves them. Given edges reserve
/// the count's rows.
#[test]
fn learned_edges_refit_at_the_freeze_or_hold_on() {
    let data = rows(200);
    let mut at_freeze = model(None, 40, 30, 50.0);
    assert_eq!(at_freeze.refit_reservation(), 0);
    let mut later = model(None, 40, 90, 50.0);
    assert_eq!(
        later.refit_reservation(),
        crate::margbins::hold_bytes(90, 3, 2)
    );
    assert_eq!(
        model(given(), 2, 90, 50.0).refit_reservation(),
        crate::margbins::hold_bytes(90, 3, 2)
    );
    for (i, (x, y, w)) in data.iter().enumerate() {
        at_freeze.step(x, y, 1.0, *w);
        later.step(x, y, 1.0, *w);
        let n = learned_before(&data, i + 1);
        let frozen = at_freeze.bins.as_ref().unwrap().hist.is_some();
        assert_eq!(frozen, n >= 40, "row {i}");
        assert_eq!(at_freeze.refit_status()[0].done, n >= 40, "row {i}");
        assert_eq!(later.refit_status()[0].done, n >= 90, "row {i}");
        // The warm-up's rows are its own hold, which the refit's count
        // does not read until the edges are fixed (task 281).
        if n < 40 {
            assert_eq!(at_freeze.refit_status()[0].learned, 0, "row {i}");
            assert_eq!(later.refit_status()[0].learned, 0, "row {i}");
        }
        if (40..90).contains(&n) {
            assert_eq!(
                later.bins.as_ref().unwrap().held.len(),
                n,
                "held on, row {i}"
            );
        }
    }
    assert_eq!(at_freeze.refit_reservation(), 0);
    assert_eq!(later.refit_reservation(), 0, "done: nothing left to hold");
}

/// Given up past the budget before its first row, the refit never holds a
/// row, and the numbers are the refit's off form's to the bit; the status
/// says so. Given up mid-warm-up, the warm-up's rows stay for the edges.
#[test]
fn a_refit_given_up_holds_nothing_and_is_the_one_pass() {
    let data = rows(150);
    let mut up = model(given(), 2, 60, 80.0);
    up.refit_give_up();
    assert_eq!(up.refit_reservation(), 0);
    let mut off = model(given(), 2, 0, 80.0);
    for (x, y, w) in &data {
        up.step(x, y, 1.0, *w);
        off.step(x, y, 1.0, *w);
        assert!(up.bins.as_ref().unwrap().held.is_empty());
    }
    let st = up.refit_status();
    assert!(
        st.iter().all(|s| s.done && s.over_budget && s.passes == 0),
        "{st:?}"
    );
    assert_eq!(
        up.bins.as_ref().unwrap().hist,
        off.bins.as_ref().unwrap().hist,
        "the one pass, to the bit"
    );
    let mut warm = model(None, 40, 30, 80.0);
    for (x, y, w) in &data[..20] {
        warm.step(x, y, 1.0, *w);
    }
    let held = warm.bins.as_ref().unwrap().held.len();
    warm.refit_give_up();
    assert_eq!(
        warm.bins.as_ref().unwrap().held.len(),
        held,
        "the warm-up keeps its rows"
    );
    for (x, y, w) in &data[20..] {
        warm.step(x, y, 1.0, *w);
    }
    assert!(warm.bins.as_ref().unwrap().hist.is_some());
    assert!(warm.refit_status()[0].over_budget);
}

/// A state is restored only as the refit writes it: rows held beside fixed
/// edges only while the refit waits, and fewer than its count; passes for
/// every target or none. A state saved mid-hold and after the refit goes
/// on to the bit.
#[test]
fn a_restored_refit_is_the_one_it_wrote() {
    use crate::{ModelState, StateError};
    let data = rows(120);
    let mut m = model(given(), 2, 60, 80.0);
    for (x, y, w) in &data[..30] {
        m.step(x, y, 1.0, *w);
    }
    let s = m.state();
    let mut back = Marginal::restore(&s).unwrap();
    let mut whole = m.clone();
    for (x, y, w) in &data[30..] {
        back.step(x, y, 1.0, *w);
        whole.step(x, y, 1.0, *w);
    }
    assert_eq!(back, whole, "resumed mid-hold, to the bit");
    assert!(back.refit_status()[0].done);
    let refused = |edit: &dyn Fn(&mut Marginal)| {
        let mut s = s.clone();
        let ModelState::Marginal(inner) = &mut s.model else {
            unreachable!()
        };
        edit(inner);
        matches!(Marginal::restore(&s), Err(StateError::Invalid(_)))
    };
    assert!(
        refused(&|m| m.bins.as_mut().unwrap().refit.done = true),
        "done and holding"
    );
    assert!(
        refused(&|m| m.cfg.bins.as_mut().unwrap().refit_rows = 10),
        "as many as the count"
    );
    assert!(
        refused(&|m| m.bins.as_mut().unwrap().refit.passes = vec![3]),
        "passes of one target"
    );
    assert!(!refused(&|_| {}));
}

/// The refit's rebuilt scale reads to the rounding of the deviations' size
/// (task 278, qequiv): a target whose first row sits within `1e-4` of its
/// location (the rest symmetric about 0) has a first deviation that a
/// level of `1e6` rounds by `1e-10`, a relative `1e-6`, and a sketch
/// anchored at it read every scale `1e-6` off where the deviations
/// themselves are `1e-10` off. Anchored at the deviations' median size, it
/// moves by rounding.
#[test]
fn a_level_moves_the_refits_scale_by_rounding() {
    let n = 61;
    let rows_at = |level: f64| -> Vec<Row> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..n)
            .map(|i| {
                let x: Vec<f64> = (0..3).map(|_| 2.0 * u()).collect();
                // 1e-4, then +a, -a, +a', -a', ...: symmetric about 0.
                let a = 0.1 + 0.037 * ((i + 1) / 2) as f64;
                let y = match i {
                    0 => 1e-4,
                    _ if i % 2 == 1 => a,
                    _ => -a,
                };
                (x, vec![Some(y + level), Some(-y)], 1.0)
            })
            .collect()
    };
    let cut = |level: f64| {
        let mut m = model(given(), 2, n as u32, f64::INFINITY);
        for (x, y, w) in &rows_at(level) {
            m.step(x, y, 1.0, *w);
        }
        assert!(m.refit_status()[0].done);
        let hist = m.bins.as_ref().unwrap().hist.as_ref().unwrap();
        hist.cell_and_cut(0, 0, 0).3
    };
    let at0 = cut(0.0);
    assert!(at0.is_finite() && at0 > 0.1, "{at0}");
    for level in [1e6, -1e9] {
        let moved = (cut(level) - at0).abs() / at0;
        println!("{level}: {moved:e}");
        assert!(moved <= 1e-8, "{level}: {moved:e}");
    }
}

/// Task 281: under decay, the refit's rows weigh what the stream has left
/// of them -- each its raw weight times every step's decay since,
/// `v_i = w_i·λ^(n − i)`, a row of weight 0 carrying its step's decay to
/// the next -- and in those weights each cell's location is a root of
/// `Σ v_i ψ_c(y_i − m)`, its `Ω` and `M2` the rows' Huber weights and
/// clipped squares, and the target's own robust weight the Huber weights
/// of all its rows about its location. Held against the definitions.
#[test]
fn under_decay_the_refit_weighs_the_rows_as_the_stream_left_them() {
    let data = rows(80);
    let h = 20.0;
    let lam = Decay::Halflife(h).factor(1.0);
    let mut m = model(given(), 2, 60, h);
    let mut at = None;
    for (i, (x, y, w)) in data.iter().enumerate() {
        m.step(x, y, 1.0, *w);
        if m.refit_status()[0].done {
            at = Some(i);
            break;
        }
    }
    let at = at.expect("the refit ran");
    let seen: Vec<Row> = data[..=at]
        .iter()
        .enumerate()
        .filter(|(_, r)| r.2 > 0.0)
        .map(|(i, (x, y, w))| (x.clone(), y.clone(), w * lam.powi((at - i) as i32)))
        .collect();
    assert_eq!(seen.len(), 60);
    let near = |a: f64, b: f64, scale: f64, what: &str| {
        assert!((a - b).abs() <= 1e-9 * scale, "{what}: {a} against {b}");
    };
    let hist = m.bins.as_ref().unwrap().hist.as_ref().unwrap();
    let mut clipped = 0;
    for t in 0..2 {
        let cut = hist.cell_and_cut(t, 0, 0).3;
        assert!(cut.is_finite() && cut > 0.0, "{cut}");
        let weigh = |e: f64| (cut / e.abs()).min(1.0);
        let (_, tw, loc_t) = hist.m2_and_target(t, 0, 0);
        let mine: Vec<(f64, f64)> = seen
            .iter()
            .filter_map(|(_, y, v)| y[t].map(|yv| (yv, *v)))
            .collect();
        let total_t: f64 = mine.iter().map(|(_, v)| v).sum();
        let want_tw: f64 = mine.iter().map(|(y, v)| v * weigh(y - loc_t)).sum();
        near(
            tw,
            want_tw,
            total_t,
            &format!("t{t}: the target's robust weight"),
        );
        clipped += mine.iter().filter(|(y, _)| (y - loc_t).abs() > cut).count();
        for j in 0..3 {
            for b in 0..4 {
                let (w_cell, loc, om_cell, _) = hist.cell_and_cut(t, j, b);
                let (m2_cell, _, _) = hist.m2_and_target(t, j, b);
                let here: Vec<(f64, f64)> = seen
                    .iter()
                    .filter(|(x, _, _)| crate::MarginalBins::bin_of(hist.edges(j), x[j]) == Some(b))
                    .filter_map(|(_, y, v)| y[t].map(|yv| (yv, *v)))
                    .collect();
                if here.is_empty() {
                    continue;
                }
                let what = format!("t{t} j{j} b{b}");
                let total: f64 = here.iter().map(|(_, v)| v).sum();
                let g: f64 = here
                    .iter()
                    .map(|(y, v)| v * (y - loc).clamp(-cut, cut))
                    .sum();
                assert!(g.abs() <= 1e-9 * total * cut, "{what}: g = {g}");
                near(w_cell, total, total, &format!("{what}: W"));
                let om: f64 = here.iter().map(|(y, v)| v * weigh(y - loc)).sum();
                near(om_cell, om, total, &format!("{what}: Ω"));
                let m2: f64 = here
                    .iter()
                    .map(|(y, v)| v * (y - loc).abs().min(cut).powi(2))
                    .sum();
                near(m2_cell, m2, total * cut * cut, &format!("{what}: M2"));
            }
        }
    }
    assert!(clipped > 0, "no row was clipped");
}

/// Task 281: a held row whose weight has decayed to nothing by the refit
/// -- a half-life of a hundredth of a step, so the first of twelve rows
/// weighs `2^-1100`, which no double holds -- takes no part in it: the
/// target's location and cut are the same to the bit whatever its value,
/// here a target of `1e12`, a million times the rest.
#[test]
fn a_row_decayed_to_nothing_takes_no_part_in_the_refit() {
    let refit = |first: f64| {
        let mut m = model(given(), 2, 12, 0.01);
        for (i, (x, y, _)) in rows(12).into_iter().enumerate() {
            let y = if i == 0 { vec![Some(first), y[1]] } else { y };
            m.step(&x, &y, 1.0, 1.0);
        }
        assert!(m.refit_status()[0].done);
        let hist = m.bins.as_ref().unwrap().hist.as_ref().unwrap();
        (hist.cell_and_cut(0, 0, 0).3, hist.m2_and_target(0, 0, 0).2)
    };
    let (cut, loc) = refit(0.3);
    assert!(cut.is_finite() && cut > 0.0, "{cut}");
    let (far_cut, far_loc) = refit(1e12);
    assert_eq!(far_cut.to_bits(), cut.to_bits(), "{far_cut} against {cut}");
    assert_eq!(far_loc.to_bits(), loc.to_bits());
}

/// Task 281: the model's `solve_failures` counts the targets whose refit
/// ran to its pass cap, and no other.
#[test]
fn solve_failures_count_the_targets_whose_refit_ran_to_the_cap() {
    use crate::huber_refit::REFIT_PASSES;
    let cap = REFIT_PASSES as u32;
    let mut m = model(given(), 2, 60, 80.0);
    assert_eq!(m.solve_failures(), 0, "before the refit");
    for (passes, want) in [
        (vec![3, 7], 0),
        (vec![cap, 7], 1),
        (vec![3, cap], 1),
        (vec![cap, cap], 2),
    ] {
        let b = m.bins.as_mut().unwrap();
        b.refit.done = true;
        b.refit.passes = passes.clone();
        assert_eq!(m.solve_failures(), want, "{passes:?}");
    }
}

/// Task 281: rows held beside fixed edges number fewer than the refit's
/// count, since the refit runs on the row that reaches it: a state holding
/// exactly the count is refused, and one fewer loads.
#[test]
fn a_restored_hold_of_exactly_the_count_is_refused() {
    use crate::{ModelState, StateError};
    let data = rows(40);
    let mut m = model(given(), 2, 60, 80.0);
    for (x, y, w) in &data {
        m.step(x, y, 1.0, *w);
    }
    let held = m.bins.as_ref().unwrap().held.len();
    assert!(held > 0 && !m.refit_status()[0].done);
    let with_count = |count: usize| {
        let mut s = m.state();
        let ModelState::Marginal(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cfg.bins.as_mut().unwrap().refit_rows = count as u32;
        Marginal::restore(&s)
    };
    assert!(matches!(with_count(held), Err(StateError::Invalid(_))));
    assert!(with_count(held + 1).is_ok());
}

/// Task 281: the refit's hold grows by doubling up to its count, as the
/// warm-up's does: a short hold holds about its rows, never more than
/// twice, a long one reallocates a handful of times, and the full hold is
/// its count exactly, which is what its reservation counts.
#[test]
fn the_refits_hold_grows_by_doubling_to_its_count() {
    let count = 37;
    let mut m = model(given(), 2, count as u32, 80.0);
    let (mut caps, mut held) = (vec![0usize], 0);
    for (x, y, w) in rows(200) {
        if w > 0.0 && held == count - 1 {
            break;
        }
        m.step(&x, &y, 1.0, w);
        let hold = &m.bins.as_ref().unwrap().held;
        held = hold.len();
        assert!(
            hold.capacity() <= (2 * held).max(4),
            "{held}: {}",
            hold.capacity()
        );
        if caps.last() != Some(&hold.capacity()) {
            caps.push(hold.capacity());
        }
    }
    assert_eq!(held, count - 1);
    assert_eq!(caps.last(), Some(&count), "{caps:?}");
    assert!(
        caps.len() <= 6,
        "reallocated {} times: {caps:?}",
        caps.len() - 1
    );
}
